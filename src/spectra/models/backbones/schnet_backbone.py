# === src/spectra/models/backbones/schnet.py ===

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter
import numpy as np


class GaussianSmearing(nn.Module):
    """高斯径向基函数"""
    
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        n_gaussians: int = 50,
        trainable: bool = False
    ):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        
        if trainable:
            self.register_parameter('offset', nn.Parameter(offset))
        else:
            self.register_buffer('offset', offset)
    
    def forward(self, dist):
        """
        Args:
            dist: [n_edges] 距离
        Returns:
            [n_edges, n_gaussians] RBF 特征
        """
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CFConv(MessagePassing):
    """连续滤波卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_filters: int,
        n_gaussians: int,
        cutoff: float,
        edge_attr_dim: int = None,
    ):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.edge_attr_dim = int(edge_attr_dim) if edge_attr_dim is not None else int(n_gaussians)
        
        # 滤波器网络
        self.filter_network = nn.Sequential(
            nn.Linear(self.edge_attr_dim, n_filters),
            nn.Softplus(),
            nn.Linear(n_filters, n_filters)
        )
        
        # 特征变换
        self.feature_net = nn.Sequential(
            nn.Linear(in_channels, n_filters),
            nn.Softplus(),
            nn.Linear(n_filters, out_channels)
        )
        
        self.rbf = GaussianSmearing(0.0, cutoff, n_gaussians)
    
    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Args:
            x: [n_nodes, in_channels] 节点特征
            pos: [n_nodes, 3] 3D坐标
            edge_index: [2, n_edges] 边索引
            edge_attr: [n_edges, edge_attr_dim] (optional)
        
        Returns:
            [n_nodes, out_channels] 更新后的节点特征
        """
        if edge_attr is None:
            row, col = edge_index
            dist = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = self.rbf(dist)
        else:
            if edge_attr.size(-1) != self.edge_attr_dim:
                raise ValueError(f"edge_attr_dim mismatch: got {edge_attr.size(-1)} expected {self.edge_attr_dim}")
        
        W = self.filter_network(edge_attr)  # [n_edges, n_filters]
        
        # 消息传递
        return self.propagate(edge_index, x=x, W=W)
    
    def message(self, x_j, W):
        """
        Args:
            x_j: [n_edges, in_channels] 邻居节点特征
            W: [n_edges, n_filters] 滤波器权重
        """
        # 应用滤波器
        x_j = self.feature_net(x_j)  # [n_edges, out_channels]
        return x_j * W[:, :self.out_channels]


class SchNetInteraction(nn.Module):
    """SchNet 交互块"""
    
    def __init__(
        self,
        n_atom_basis: int,
        n_filters: int,
        n_gaussians: int,
        cutoff: float,
        edge_attr_dim: int = None,
    ):
        super().__init__()
        
        self.cfconv = CFConv(
            n_atom_basis,
            n_atom_basis,
            n_filters,
            n_gaussians,
            cutoff,
            edge_attr_dim=edge_attr_dim,
        )
        
        # 原子级更新
        self.atom_update = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.Softplus(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
    
    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Args:
            x: [n_nodes, n_atom_basis]
            pos: [n_nodes, 3]
            edge_index: [2, n_edges]
            edge_attr: [n_edges, edge_attr_dim] (optional)
        """
        # 连续滤波卷积
        v = self.cfconv(x, pos, edge_index, edge_attr=edge_attr)
        
        # 原子级更新
        v = self.atom_update(v)
        
        # 残差连接
        x = x + v
        return x


class SchNet(nn.Module):
    """
    SchNet: 用于分子性质预测的连续滤波卷积神经网络
    
    Reference:
        Schütt et al. "SchNet: A continuous-filter convolutional 
        neural network for modeling quantum interactions."
        NeurIPS 2017.
    """
    
    def __init__(
        self,
        n_atom_basis: int = 128,
        n_filters: int = 128,
        n_interactions: int = 6,
        n_gaussians: int = 25,
        cutoff: float = 5.0,
        max_z: int = 100,
        cutoff_network: str = "cosine",
        edge_attr_dim: int = None,
    ):
        super().__init__()
        
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        
        # 原子嵌入层
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        
        # 交互块
        self.interactions = nn.ModuleList([
            SchNetInteraction(
                n_atom_basis,
                n_filters,
                n_gaussians,
                cutoff,
                edge_attr_dim=edge_attr_dim,
            )
            for _ in range(n_interactions)
        ])
    
    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        z_s: torch.Tensor = None,
        node_batch: torch.Tensor = None,
        gbm_conditioner: nn.Module = None,
        deep_film: bool = False,
        film_every: int = 1,
        film_layers: list = None,
        film_beta_only: bool = False,
        film_scale: float = 1.0,
        edge_attr: torch.Tensor = None,
    ):
        """
        Args:
            Z: [n_atoms] 原子序数
            pos: [n_atoms, 3] 3D坐标
            batch: [n_atoms] 批次索引
        Returns:
            [n_atoms, n_atom_basis] 节点嵌入
        """
        if node_batch is None:
            node_batch = batch

        # 初始嵌入
        x = self.embedding(Z)  # [n_atoms, n_atom_basis]
        
        # 构建边（基于距离截断）
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        # 计算注入层集合（1-based）
        L = len(self.interactions)
        if film_layers is not None:
            inject_set = {int(i) for i in film_layers if 1 <= int(i) <= L}
        elif deep_film:
            inject_set = {i for i in range(1, L + 1) if (i % max(int(film_every), 1) == 0)}
        else:
            inject_set = set()

        # 节点级 FiLM 注入工具
        def _apply_node_film_local(h: torch.Tensor) -> torch.Tensor:
            if (gbm_conditioner is None) or (z_s is None):
                return h
            gamma, beta = gbm_conditioner.scalar_film(z_s)  # [B, C] or [N, C]
            
            if film_beta_only:
                gamma = torch.ones_like(gamma)
            if film_scale != 1.0:
                gamma = 1.0 + film_scale * (gamma - 1.0)
                beta = film_scale * beta
            
            # 自动识别是 Graph-level 还是 Node-level
            # 如果 z_s 是 [N, D] 且 N > 1，则 gamma/beta 是 [N, C] -> Node-level
            is_node_level = (gamma.size(0) == h.size(0)) and (h.size(0) > 1)
            
            if is_node_level:
                g = gamma
                b = beta
            else:
                g = gamma[node_batch]
                b = beta[node_batch]
                
            return g * h + b

        # 交互层
        for i, interaction in enumerate(self.interactions, 1):
            x = interaction(x, pos, edge_index, edge_attr=edge_attr)  # 残差已在模块内部完成
            if i in inject_set:
                x = _apply_node_film_local(x)
        
        return x

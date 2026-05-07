import math
from math import pi
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from spectra.models.backbones.nonlocal_performer import PerformerNonLocal
from spectra.models.ehc.edge_geom_cache import EdgeGeomCache

# 可选半径图后端
try:
    from torch_cluster import radius_graph as _clust_radius_graph
except Exception:
    _clust_radius_graph = None
try:
    from torch_geometric.nn import radius_graph as _pyg_radius_graph
except Exception:
    _pyg_radius_graph = None


def _radius_graph(pos: torch.Tensor, r: float, batch: torch.Tensor, max_num_neighbors: int = 64) -> torch.Tensor:
    """Prefer torch_cluster → torch_geometric → O(N^2) fallback."""
    if _clust_radius_graph is not None:
        return _clust_radius_graph(pos, r=r, batch=batch, loop=False, max_num_neighbors=max_num_neighbors)
    if _pyg_radius_graph is not None:
        return _pyg_radius_graph(pos, r=r, batch=batch, loop=False)
    # Brute-force per-graph fallback
    device = pos.device
    row_list, col_list = [], []
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    for b in range(B):
        idx = (batch == b).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        xb = pos[idx]
        d = torch.cdist(xb, xb, p=2.0)  # [nb, nb]
        mask = (d <= r) & (~torch.eye(idx.numel(), dtype=torch.bool, device=device))
        src, dst = mask.nonzero(as_tuple=True)
        row_list.append(idx[src])
        col_list.append(idx[dst])
    if len(row_list) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return torch.stack([row, col], dim=0)


def nan_to_num(x: torch.Tensor, num: float = 0.0) -> torch.Tensor:
    """In-place NaN to constant."""
    idx = torch.isnan(x)
    if idx.any():
        x[idx] = num
    return x


def _normalize(vec: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safe normalize with stronger clamp."""
    n = torch.norm(vec, dim=dim, keepdim=True).clamp(min=1e-12)
    return nan_to_num(vec / n)


class RBFEmb(nn.Module):
    """Soft-cutoff RBF on exp(-d), fp32 计算，前置限幅."""
    def __init__(self, num_rbf: int, soft_cutoff_upper: float):
        super().__init__()
        self.soft_cutoff_upper = float(soft_cutoff_upper)
        self.soft_cutoff_lower = 0.0
        self.num_rbf = int(num_rbf)
        means, betas = self._initial_params()
        self.register_buffer("means", means)  # fp32
        self.register_buffer("betas", betas)  # fp32

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper, dtype=torch.float32))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower, dtype=torch.float32))
        means = torch.linspace(start_value, end_value, self.num_rbf, dtype=torch.float32)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf, dtype=torch.float32)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist in fp32; 前置限幅避免极端输入
        dist = dist.clamp_min(0.0).clamp_max(self.soft_cutoff_upper * 4)
        dist = dist.unsqueeze(-1)  # [E, 1]
        soft_cutoff = 0.5 * (torch.cos(dist * pi / self.soft_cutoff_upper) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.soft_cutoff_upper).float()
        return soft_cutoff * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    """聚合邻居嵌入，按 RBF 权缩放."""
    def __init__(self, hid_dim: int):
        super().__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.embedding(z)  # [N, H]
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)  # [N, H]
        return s + s_neighbors

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class SVector(MessagePassing):
    """从标量通道与边方向构建初始等变向量特征."""
    def __init__(self, hid_dim: int):
        super().__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SiLU())

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v  # [E, 3, H]
        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)  # [E, 1, H]
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    """LEFTNet 等变消息传递（标量+向量）."""
    def __init__(self, hidden_channels: int, num_radial: int):
        super().__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_radial = num_radial

        self.inv_proj = nn.Sequential(
            nn.Linear(3 * hidden_channels + num_radial, hidden_channels * 3),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)

        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, edge_rbf, weight, edge_vector):
        xh = self.x_proj(x)
        rbfh = self.rbf_proj(edge_rbf)
        weight = self.inv_proj(weight)
        rbfh = rbfh * weight
        dx, dvec = self.propagate(edge_index, xh=xh, vec=vec, rbfh_ij=rbfh, r_ij=edge_vector, size=None)
        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h
        return x, vec

    def aggregate(self, features, index: torch.Tensor, ptr: Optional[torch.Tensor], dim_size: Optional[int]):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return x, vec

    def update(self, inputs):
        return inputs


class FTE(nn.Module):
    """Frame Transition Encoding 在节点坐标系中融合."""
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.equi_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xequi_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.equi_proj.weight)
        nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, node_frame):
        vec = self.equi_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)

        scalrization = torch.sum(vec1.unsqueeze(2) * node_frame.unsqueeze(-1), dim=1)
        scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())
        scalar = torch.norm(vec1, dim=-2)

        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        x_vec_h = self.xequi_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = (xvec1 + xvec2 + vec_dot) * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(1) * vec2
        return dx, dvec


class AggregatePos(MessagePassing):
    """按边聚合邻域位置，默认 mean。"""
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)
    def forward(self, vector, edge_index):
        return self.propagate(edge_index, x=vector)
    def message(self, x_j):
        return x_j


class LEFTNetBackbone(nn.Module):
    """
    LEFTNet encoder-only backbone: 返回节点嵌入 H:[N, hidden]
    - 内部构建半径图、按图居中坐标
    - 几何分支强制 fp32 + 限幅，提高数值稳定
    - 可选 FiLM 钩子（默认不注入）
    """
    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 4,
        num_radial: int = 32,
        cutoff: float = 5.0,
        use_lse: bool = True,
        use_fte: bool = True,
        use_vector_features: bool = True,
        dropout: float = 0.1,
        use_geom_gate: bool = False,
        geom_n_rbf: int = None,
        geom_use_moments: bool = True,
        geom_use_global: bool = True,
        geom_gate_scale: float = 0.1,
        geom_use_ln: bool = True,
        use_nonlocal: bool = False,
        nonlocal_heads: int = 4,
        nonlocal_features: int = 64,
        nonlocal_dropout: float = 0.0,
        nonlocal_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        
        self.use_lse = use_lse
        self.use_fte = use_fte
        self.use_vector_features = use_vector_features
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.use_geom_gate = bool(use_geom_gate)
        self.geom_gate_scale = float(geom_gate_scale)
        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)

        self.z_emb = nn.Embedding(95, self.hidden_channels)

        # RBF 分支模块以 fp32 初始化
        self.radial_emb = RBFEmb(self.num_radial, self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(self.num_radial, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        self.neighbor_emb = NeighborEmb(self.hidden_channels)
        
        if self.use_lse:
            self.S_vector = SVector(self.hidden_channels)
            self.lin = nn.Sequential(
                nn.Linear(3, self.hidden_channels // 4),
                nn.SiLU(inplace=True),
                nn.Linear(self.hidden_channels // 4, 1),
            )

        self.message_layers = nn.ModuleList(
            [EquiMessagePassing(self.hidden_channels, self.num_radial) for _ in range(self.num_layers)]
        )
        
        if self.use_fte:
            self.FTEs = nn.ModuleList([FTE(self.hidden_channels) for _ in range(self.num_layers)])

        self.mean_neighbor_pos = AggregatePos(aggr='mean')

        self.geom_gate_cache = None
        self.geom_gate_norm = None
        self.geom_gate_proj = None
        if self.use_geom_gate:
            gate_rbf = int(geom_n_rbf or self.num_radial)
            geom_dim = gate_rbf
            if geom_use_moments:
                geom_dim += 5
            if geom_use_global:
                geom_dim += 2
            self.geom_gate_cache = EdgeGeomCache(
                cutoff=self.cutoff,
                n_rbf=gate_rbf,
                tanh_clip=True,
                use_moments=geom_use_moments,
                use_global=geom_use_global,
            )
            if geom_use_ln:
                self.geom_gate_norm = nn.LayerNorm(geom_dim)
            self.geom_gate_proj = nn.Sequential(
                nn.Linear(geom_dim, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, 1),
                nn.Tanh(),
            )

        self.nonlocal_block = None
        self.nonlocal_pre_norm = None
        self.nonlocal_post_norm = None
        if self.use_nonlocal:
            self.nonlocal_block = PerformerNonLocal(
                hidden_dim=self.hidden_channels,
                num_heads=nonlocal_heads,
                num_features=nonlocal_features,
                dropout=nonlocal_dropout,
            )
            self.nonlocal_pre_norm = nn.LayerNorm(self.hidden_channels)
            self.nonlocal_post_norm = nn.LayerNorm(self.hidden_channels)

    @staticmethod
    def _compute_centered(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        center = scatter(pos, batch, dim=0, reduce='mean')  # [B, 3]
        return pos - center[batch]

    def forward(
        self,
        Z: torch.Tensor,               # [N] atomic numbers
        pos: torch.Tensor,             # [N, 3]
        batch: torch.Tensor,           # [N]
        z_s: torch.Tensor = None,      # [B, z_dim] (for FiLM, 可不传)
        node_batch: torch.Tensor = None,
        gbm_conditioner: nn.Module = None,
        deep_film: bool = False,
        film_every: int = 1,
        film_layers: List[int] = None,
        film_beta_only: bool = False,
        film_scale: float = 1.0,
    ) -> torch.Tensor:
        if node_batch is None:
            node_batch = batch

        # 按图居中
        posc = self._compute_centered(pos, batch)

        # 半径图（按 posc）
        edge_index = _radius_graph(posc, r=self.cutoff, batch=batch, max_num_neighbors=64)
        i, j = edge_index
        if i.numel() == 0:
            return self.z_emb(Z)

        # 几何分支强制 fp32
        pos32 = posc.float()
        rij32 = pos32[i] - pos32[j]
        dist32 = rij32.norm(dim=-1).clamp_min(0.0)
        soft_c32 = 0.5 * (torch.cos(dist32 * math.pi / self.cutoff) + 1.0)
        soft_c32 = soft_c32 * (dist32 < self.cutoff).float()

        rbf32 = self.radial_emb(dist32)              # [E, R] fp32
        radial_hidden32 = self.radial_lin(rbf32)     # [E, H] fp32
        radial_hidden32 = soft_c32.unsqueeze(-1) * radial_hidden32

        edge_diff32 = _normalize(rij32)
        edge_cross32 = _normalize(torch.cross(pos32[i], pos32[j], dim=-1))
        edge_vert32 = torch.cross(edge_diff32, edge_cross32, dim=-1)
        edge_frame32 = torch.cat(
            (edge_diff32.unsqueeze(-1), edge_cross32.unsqueeze(-1), edge_vert32.unsqueeze(-1)), dim=-1
        )

        mean_nb32 = self.mean_neighbor_pos(pos32, edge_index)
        node_diff32 = _normalize(pos32 - mean_nb32)
        node_cross32 = _normalize(torch.cross(pos32, mean_nb32, dim=-1))
        node_vert32 = torch.cross(node_diff32, node_cross32, dim=-1)
        node_frame32 = torch.cat(
            (node_diff32.unsqueeze(-1), node_cross32.unsqueeze(-1), node_vert32.unsqueeze(-1)), dim=-1
        )

        # cast 回模型 dtype
        mdl_dtype = self.z_emb.weight.dtype  # 通常是 fp32/bf16
        rbf = rbf32.to(mdl_dtype)
        radial_hidden = radial_hidden32.to(mdl_dtype)
        soft_cutoff = soft_c32.to(mdl_dtype)
        edge_diff = edge_diff32.to(mdl_dtype)
        edge_frame = edge_frame32.to(mdl_dtype)
        node_frame = node_frame32.to(mdl_dtype)

        # 初始标量/向量通道
        s0 = self.z_emb(Z)
        s = self.neighbor_emb(Z, s0, edge_index, radial_hidden)  # [N, H]
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device, dtype=s.dtype)

        # LSE (Local Scalar Encoding)
        if self.use_lse:
            S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)  # [N, 3, H]
            scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)  # [E, 3, H]
            scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)  # [E, 3, H]
            scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
            scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

            # 标量摘要（fp32 计算更稳），再回写 dtype
            sc3_32 = (self.lin(torch.permute(scalrization1.float(), (0, 2, 1))) +
                      torch.permute(scalrization1.float(), (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1)
            sc4_32 = (self.lin(torch.permute(scalrization2.float(), (0, 2, 1))) +
                      torch.permute(scalrization2.float(), (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1)
            scalar3 = sc3_32.to(mdl_dtype)
            scalar4 = sc4_32.to(mdl_dtype)

            A_i_j = torch.cat((scalar3, scalar4), dim=-1) * soft_cutoff.unsqueeze(-1)  # [E, 2H]
            A_i_j = torch.cat((A_i_j, radial_hidden, rbf), dim=-1)                     # [E, 3H+R]
        else:
            # Fallback: just use radial info
            # To keep dims consistent for message_layers, we might need padding?
            # EquiMessagePassing.inv_proj takes: 3*hidden + num_radial
            # LSE produces: 2*hidden scalars.
            # So A_i_j must be [E, 2H + H + R] = [E, 3H + R]
            # If LSE is off, we pad the first 2H dims with zeros.
            pad_scalars = torch.zeros(rbf.size(0), 2 * self.hidden_channels, device=rbf.device, dtype=mdl_dtype)
            A_i_j = torch.cat((pad_scalars, radial_hidden, rbf), dim=-1)

        A_i_j = torch.tanh(A_i_j)  # 饱和，避免偶发极端放大
        if self.use_geom_gate and self.geom_gate_cache is not None and self.geom_gate_proj is not None:
            geom_attr = self.geom_gate_cache(posc, edge_index, batch=batch)
            if self.geom_gate_norm is not None:
                geom_attr = self.geom_gate_norm(geom_attr)
            geom_gate = self.geom_gate_proj(geom_attr).to(dtype=A_i_j.dtype)
            A_i_j = A_i_j * (1.0 + self.geom_gate_scale * geom_gate)

        # 决定 FiLM 注入层（默认不注入）
        L = self.num_layers
        if film_layers is not None:
            inject_set = {int(x) for x in film_layers if 1 <= int(x) <= L}
        elif deep_film:
            fe = max(int(film_every), 1)
            inject_set = {li for li in range(1, L + 1) if (li % fe == 0)}
        else:
            inject_set = set()

        def _apply_node_film_local(h: torch.Tensor) -> torch.Tensor:
            if (gbm_conditioner is None) or (z_s is None):
                return h
            gamma, beta = gbm_conditioner.scalar_film(z_s)  # [B/N, H]
            if film_beta_only:
                gamma = torch.ones_like(gamma)
            if film_scale != 1.0:
                gamma = 1.0 + film_scale * (gamma - 1.0)
                beta = film_scale * beta
            
            # Auto-detect node-level
            is_node_level = (gamma.size(0) == h.size(0)) and (h.size(0) > 1)
            
            if is_node_level:
                return gamma * h + beta
            else:
                return gamma[node_batch] * h + beta[node_batch]

        # 堆叠
        for li in range(self.num_layers):
            # Pass LSE/RBF feats
            # If use_vector_features is False, we force vec to be 0 always?
            # Or we let EquiMessagePassing run but ensure input vec is 0?
            # EquiMessagePassing updates vec. If we want "No Vector", we should not add dvec.
            
            ds, dvec = self.message_layers[li](s, vec, edge_index, rbf, A_i_j, edge_diff)
            s = s + ds
            s = self.dropout(s)
            if self.use_vector_features:
                vec = vec + dvec

            # FTE (Frame Transition)
            if self.use_fte:
                ds, dvec = self.FTEs[li](s, vec, node_frame)
                s = s + ds
                s = self.dropout(s)
                if self.use_vector_features:
                    vec = vec + dvec

            if self.use_nonlocal and self.nonlocal_block is not None:
                if self.nonlocal_pre_norm is not None:
                    s = self.nonlocal_pre_norm(s)
                s = s + self.nonlocal_scale * self.nonlocal_block(s, node_batch)
                if self.nonlocal_post_norm is not None:
                    s = self.nonlocal_post_norm(s)

            if (li + 1) in inject_set:
                s = _apply_node_film_local(s)

        return s  # [N, hidden]

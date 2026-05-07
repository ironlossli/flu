import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.data import Batch, Data, DataLoader
from torch_geometric.nn import radius, radius_graph

# 尝试使用 deepqmc.torchext；若不可用，则提供本地兜底实现
try:
    from deepqmc.torchext import SSP as _SSP, get_log_dnn as _get_log_dnn  # type: ignore
    SSP = _SSP
    get_log_dnn = _get_log_dnn
except Exception:
    class SSP(nn.Module):
        """稳定版 Softplus 的简化替代实现"""
        def __init__(self, beta: float = 1.0, threshold: float = 20.0):
            super().__init__()
            self.act = nn.Softplus(beta=beta, threshold=threshold)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x)

    def get_log_dnn(
        in_dim: int,
        out_dim: int,
        activation_factory: type[nn.Module] = SSP,
        n_layers: int = 3,
    ) -> nn.Module:
        """
        简化替代：返回多层感知机（宽度固定为 in_dim），激活用 activation_factory。
        原 deepqmc 版本用于构造对数域网络；此处只保证形状与数值稳定。
        """
        layers: list[nn.Module] = []
        hid = in_dim
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(hid, hid), activation_factory()]
        layers += [nn.Linear(hid, out_dim)]
        return nn.Sequential(*layers)


class CosineCutoff(nn.Module):
    """
    余弦截断：r <= cutoff 时 0.5*(cos(pi*r/cutoff)+1)，否则 0
    接受 [E] 或 [E,1]，返回同形状张量
    """
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = distances * (math.pi / self.cutoff)
        cut = 0.5 * (torch.cos(x) + 1.0)
        mask = (distances < self.cutoff).to(cut.dtype)
        return cut * mask


class BesselBasis(nn.Module):
    """
    径向 Bessel 基（简化版 DimeNet 风格）
    phi_n(r) = sin(n*pi*r/cutoff) / r, r->0 用极限 n*pi/cutoff
    输入可为向量差 [E,3] 或距离 [E] / [E,1]；输出 [E, n_rbf]
    """
    def __init__(self, cutoff: float = 5.0, n_rbf: Optional[int] = None):
        super().__init__()
        if not n_rbf or n_rbf <= 0:
            raise ValueError("n_rbf must be a positive integer")
        self.cutoff = float(cutoff)
        freqs = torch.arange(1, n_rbf + 1, dtype=torch.float32) * (math.pi / self.cutoff)
        self.register_buffer("freqs", freqs)  # [n_rbf]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 支持位移向量或距离
        if inputs.dim() == 2 and inputs.size(-1) in (3,):
            r = torch.norm(inputs, p=2, dim=1)           # [E]
        elif inputs.dim() == 1:
            r = inputs                                    # [E]
        elif inputs.dim() == 2 and inputs.size(-1) == 1:
            r = inputs.squeeze(-1)                        # [E]
        else:
            raise ValueError(f"Unexpected inputs shape: {tuple(inputs.shape)}")

        ax = torch.outer(r, self.freqs)                   # [E, n_rbf]
        sinax = torch.sin(ax)

        denom = torch.where(r > 0, r, torch.ones_like(r)) # 避免除零
        y = sinax / denom.unsqueeze(-1)                   # [E, n_rbf]

        if torch.any(r == 0):
            # 极限值：sin(n*pi*0)/0 -> n*pi/cutoff = freqs
            y = torch.where(r.eq(0).unsqueeze(-1), self.freqs.expand_as(y), y)

        # cutoff 之外置 0（可与 CosineCutoff 叠加）
        mask = (r <= self.cutoff).to(y.dtype).unsqueeze(-1)
        return y * mask


class Jastrow(nn.Module):
    """
    简化版 Jastrow 因子：对嵌入做（求和→MLP）或（MLP→求和）
    """
    def __init__(self, embedding_dim: int, activation_factory=SSP, *, n_layers: int = 3, sum_first: bool = True):
        super().__init__()
        self.net = get_log_dnn(embedding_dim, 1, activation_factory, n_layers=n_layers)
        self.sum_first = sum_first

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        if self.sum_first:
            xs = self.net(xs.sum(dim=-2))
        else:
            xs = self.net(xs).sum(dim=-2)
        return xs.squeeze(dim=-1)


class Bipartite(Data):
    """
    用于 nucleus-electron 二部图的 Data 容器
    """
    def __init__(self, edge_index, coord_elec, coord_nuc, s_nuc, v_nuc, num_nodes):
        super().__init__()
        self.edge_index = edge_index
        self.coord_elec = coord_elec
        self.coord_nuc = coord_nuc
        self.s_nuc = s_nuc
        self.v_nuc = v_nuc
        self.num_nodes = num_nodes

    def __inc__(self, key, value):
        if key == "edge_index":
            return torch.tensor([[self.coord_nuc.size(0)], [self.coord_elec.size(0)]])
        return super().__inc__(key, value)


class BatchGraphNuc(nn.Module):
    """
    基于半径阈值在 nucleus/electron 之间连边，返回拼接后的 (s_nuc, v_nuc, edge_index, edge_attr)
    """
    def __init__(self, cut_off: float):
        super().__init__()
        self.cut_off = float(cut_off)

    def forward(self, s_nuc: torch.Tensor, v_nuc: torch.Tensor, coord_elec: torch.Tensor, coord_nuc: torch.Tensor):
        # coord_elec: [B, Ne, 3], coord_nuc: [Nn, 3] 或 [B, Nn, 3]
        batch_dim, n_elec = coord_elec.shape[:2]
        if coord_nuc.dim() == 2:
            coord_nuc = coord_nuc.unsqueeze(0).repeat(batch_dim, 1, 1)  # [B, Nn, 3]

        data_list = [
            Bipartite(
                radius(e, n, self.cut_off),
                e, n, sn, vn, n_elec
            )
            for e, n, sn, vn in zip(coord_elec, coord_nuc, s_nuc, v_nuc)
        ]

        loader = DataLoader(data_list, batch_size=batch_dim)
        batch = next(iter(loader))

        row, col = batch.edge_index
        edge_attr = batch.coord_elec[col] - batch.coord_nuc[row]  # [E, 3]

        return (batch.s_nuc, batch.v_nuc, batch.edge_index, edge_attr)


class BatchGraphElec(nn.Module):
    """
    在电子间按半径阈值连边，返回拼接后的 (x, v, edge_index, edge_attr)
    """
    def __init__(self, cut_off: float = 5.0):
        super().__init__()
        self.cut_off = float(cut_off)

    def forward(self, s: torch.Tensor, v: torch.Tensor, rs: torch.Tensor):
        # s: [B, Ne, F], v: [B, Ne, F, 3], rs: [B, Ne, 3]
        data = Batch.from_data_list([Data(x=ss, v=vv, r=rr) for ss, vv, rr in zip(s, v, rs)])
        edge_index = radius_graph(data.r, r=self.cut_off, batch=data.batch, loop=False)
        row, col = edge_index
        edge_attr = data.r[row] - data.r[col]  # [E, 3]
        return data.x, data.v, edge_index, edge_attr


class BackflowPaiNN(nn.Module):
    """
    简化 Backflow 网络：两层线性将 embedding_dim -> 1，并在前向中 squeeze
    """
    def __init__(self, embedding_dim: int, n_backflows: int, num_electrons: int):
        super().__init__()
        self.net = nn.Sequential(Linear(embedding_dim, embedding_dim), Linear(embedding_dim, 1))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.net(xs))

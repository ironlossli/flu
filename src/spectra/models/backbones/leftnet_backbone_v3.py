import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from spectra.models.backbones.nonlocal_performer import PerformerNonLocal
from spectra.models.ehc.edge_geom_cache import EdgeGeomCache

# Optional radius graph backends
try:
    from torch_cluster import radius_graph as _clust_radius_graph
except Exception:
    _clust_radius_graph = None
try:
    from torch_geometric.nn import radius_graph as _pyg_radius_graph
except Exception:
    _pyg_radius_graph = None


def _radius_graph(pos: torch.Tensor, r: float, batch: torch.Tensor, max_num_neighbors: int = 64) -> torch.Tensor:
    if _clust_radius_graph is not None:
        return _clust_radius_graph(pos, r=r, batch=batch, max_num_neighbors=max_num_neighbors)
    if _pyg_radius_graph is not None:
        return _pyg_radius_graph(pos, r=r, batch=batch, max_num_neighbors=max_num_neighbors)

    # Brute force fallback (slow)
    row, col = [], []
    for b in batch.unique().tolist():
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        p = pos[idx]
        dist = torch.cdist(p, p)
        mask = (dist < r) & (~torch.eye(len(idx), device=pos.device, dtype=torch.bool))
        r_idx, c_idx = mask.nonzero(as_tuple=True)
        row.append(idx[r_idx])
        col.append(idx[c_idx])

    if len(row) == 0:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)

    return torch.stack([torch.cat(row), torch.cat(col)], dim=0)


def nan_to_num_(x: torch.Tensor, num: float = 0.0) -> torch.Tensor:
    idx = torch.isnan(x)
    if idx.any():
        x[idx] = num
    return x


def _normalize(vec: torch.Tensor, dim: int = -1) -> torch.Tensor:
    n = torch.norm(vec, dim=dim, keepdim=True).clamp(min=1e-12)
    return nan_to_num_(vec / n)


def _stable_frame_from_dir(dir_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Deterministic, translation-invariant orthonormal frame from a unit direction."""
    device, dtype = dir_hat.device, dir_hat.dtype
    ref1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(dir_hat)
    ref2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(dir_hat)
    aligned = (dir_hat * ref1).sum(dim=-1).abs() > 0.9
    ref = torch.where(aligned.unsqueeze(-1), ref2, ref1)

    e1 = dir_hat
    e2 = torch.cross(e1, ref, dim=-1)
    e2 = e2 / (e2.norm(dim=-1, keepdim=True).clamp(min=eps))
    e3 = torch.cross(e1, e2, dim=-1)

    return torch.stack([e1, e2, e3], dim=-1)


def _project_to_frame(vec: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """returns frame^T @ vec"""
    return torch.einsum('...ji,...jh->...ih', frame, vec)


class RBFEmb(nn.Module):
    """Soft-cutoff RBF on exp(-d), computed in fp32."""
    def __init__(self, num_rbf: int, soft_cutoff_upper: float):
        super().__init__()
        self.soft_cutoff_upper = float(soft_cutoff_upper)
        self.soft_cutoff_lower = 0.0
        self.num_rbf = int(num_rbf)
        means, betas = self._initial_params()
        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

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
        dist = dist.clamp(min=0.0, max=self.soft_cutoff_upper)
        x = torch.exp(-dist).unsqueeze(-1)
        return torch.exp(-self.betas * (x - self.means) ** 2)


class NeighborEmb(nn.Module):
    def __init__(self, num_atom_types: int, hidden_channels: int):
        super().__init__()
        self.emb = nn.Embedding(num_atom_types, hidden_channels)
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.lin[0].weight)
        self.lin[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.lin[2].weight)
        self.lin[2].bias.data.zero_()

    def forward(self, Z, s0, edge_index, radial_hidden):
        i, j = edge_index
        z_j = self.emb(Z)[j]
        m = self.lin(z_j) * radial_hidden
        agg = scatter(m, i, dim=0, dim_size=s0.size(0), reduce="sum")
        return s0 + agg


class SVector(MessagePassing):
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
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class UVector(nn.Module):
    def __init__(self, hidden_channels: int, num_radial: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.zero_()

    def forward(self, edge_index, edge_dir: torch.Tensor, edge_rbf: torch.Tensor, soft_cutoff: torch.Tensor, num_nodes: int):
        i, _ = edge_index
        w32 = self.mlp(edge_rbf.float())  # [E,H]
        c32 = soft_cutoff.float()
        if c32.dim() == 2:
            c32 = c32.squeeze(-1)
        w32 = w32 * c32.unsqueeze(-1)
        u_msg = edge_dir.float().unsqueeze(-1) * w32.unsqueeze(1)  # [E,3,H]
        return scatter(u_msg, i, dim=0, dim_size=num_nodes, reduce="sum")  # [N,3,H] fp32


class EquiMessagePassingCond(MessagePassing):
    def __init__(self, hidden_channels: int, num_radial: int, cond_dim: int):
        super().__init__(aggr="add", node_dim=0)
        self.hidden_channels = int(hidden_channels)
        self.num_radial = int(num_radial)
        self.cond_dim = int(cond_dim)

        self.inv_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_channels * 3),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3),
        )
        self.x_proj = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(self.num_radial, self.hidden_channels * 3)

        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(self.hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.zero_()

        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.zero_()

        nn.init.xavier_uniform_(self.inv_proj[0].weight)
        self.inv_proj[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.inv_proj[2].weight)
        self.inv_proj[2].bias.data.zero_()

    def forward(self, x, vec, edge_index, edge_rbf, cond_ij, edge_vector):
        xh = self.x_proj(x)
        rbfh = self.rbf_proj(edge_rbf)
        weight = self.inv_proj(cond_ij)
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


class FTEv2(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.equi_proj = nn.Linear(self.hidden_channels, self.hidden_channels * 2, bias=False)

        mid = max(4, self.hidden_channels // 4)
        self.dir_lin = nn.Sequential(
            nn.Linear(3, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 1),
        )
        self.xequi_proj = nn.Sequential(
            nn.Linear(self.hidden_channels * 3, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(self.hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.equi_proj.weight)

        nn.init.xavier_uniform_(self.dir_lin[0].weight)
        self.dir_lin[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.dir_lin[2].weight)
        self.dir_lin[2].bias.data.zero_()

        nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.zero_()

    def forward(self, x: torch.Tensor, vec: torch.Tensor, node_frame: torch.Tensor):
        vec = self.equi_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)

        proj = _project_to_frame(vec1.float(), node_frame.float())
        proj_sq = torch.log1p(proj * proj)
        proj_sq = torch.permute(proj_sq, (0, 2, 1))
        dir_stat32 = self.dir_lin(proj_sq).squeeze(-1)

        scalar32 = torch.norm(vec1.float(), dim=-2).clamp(min=0.0)
        vec_dot32 = (vec1.float() * vec2.float()).sum(dim=1) * self.inv_sqrt_h

        mdl_dtype = x.dtype
        dir_stat = dir_stat32.to(mdl_dtype)
        scalar = scalar32.to(mdl_dtype)
        vec_dot = vec_dot32.to(mdl_dtype)

        x_vec_h = self.xequi_proj(torch.cat([x, scalar, dir_stat], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = (xvec1 + xvec2 + vec_dot) * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(1) * vec2
        return dx, dvec


class AggregatePos(MessagePassing):
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        return self.propagate(edge_index, x=vector)

    def message(self, x_j):
        return x_j


class LEFTNetBackboneV3Cond(nn.Module):
    def __init__(
        self,
        num_atom_types: int = 100,
        hidden_channels: int = 256,
        num_layers: int = 4,
        cutoff: float = 5.0,
        num_radial: int = 32,
        max_num_neighbors: int = 48,
        cond_dim: int = 512,
        use_lse: bool = True,
        use_fte: bool = True,
        use_vector_features: bool = True,
        use_uvec: bool = True,
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
        self.num_atom_types = int(num_atom_types)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.num_radial = int(num_radial)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cond_dim = int(cond_dim)

        self.use_lse = bool(use_lse)
        self.use_fte = bool(use_fte)
        self.use_vector_features = bool(use_vector_features)
        self.use_uvec = bool(use_uvec)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.use_geom_gate = bool(use_geom_gate)
        self.geom_gate_scale = float(geom_gate_scale)
        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)

        self.z_emb = nn.Embedding(self.num_atom_types, self.hidden_channels)

        self.radial_emb = RBFEmb(num_rbf=self.num_radial, soft_cutoff_upper=4.0 * self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(self.num_radial, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.neighbor_emb = NeighborEmb(self.num_atom_types, self.hidden_channels)

        if self.use_lse:
            self.S_vector = SVector(self.hidden_channels)
            mid = max(4, self.hidden_channels // 4)
            self.lin_lse = nn.Sequential(
                nn.Linear(3, mid),
                nn.SiLU(inplace=True),
                nn.Linear(mid, 1),
            )
        else:
            self.S_vector = None
            self.lin_lse = None

        if self.use_uvec:
            self.uvec = UVector(self.hidden_channels, self.num_radial)
            mid2 = max(8, self.hidden_channels // 8)
            self.lin_angle = nn.Sequential(
                nn.Linear(4, mid2),
                nn.SiLU(inplace=True),
                nn.Linear(mid2, 1),
            )
        else:
            self.uvec = None
            self.lin_angle = None

        # compress A_full (3H+R) -> cond_dim (keep only cond_ij for the stack)
        a_dim = 3 * self.hidden_channels + self.num_radial
        self.cond_proj = nn.Sequential(
            nn.Linear(a_dim, self.cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.message_layers = nn.ModuleList(
            [EquiMessagePassingCond(self.hidden_channels, self.num_radial, self.cond_dim) for _ in range(self.num_layers)]
        )
        self.FTEs = nn.ModuleList([FTEv2(self.hidden_channels) for _ in range(self.num_layers)]) if self.use_fte else None

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
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.z_emb.weight)

        self.radial_emb.reset_parameters()
        nn.init.xavier_uniform_(self.radial_lin[0].weight)
        self.radial_lin[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.radial_lin[2].weight)
        self.radial_lin[2].bias.data.zero_()

        self.neighbor_emb.reset_parameters()

        if self.use_lse:
            nn.init.xavier_uniform_(self.lin_lse[0].weight)
            self.lin_lse[0].bias.data.zero_()
            nn.init.xavier_uniform_(self.lin_lse[2].weight)
            self.lin_lse[2].bias.data.zero_()

        if self.use_uvec:
            self.uvec.reset_parameters()
            nn.init.xavier_uniform_(self.lin_angle[0].weight)
            self.lin_angle[0].bias.data.zero_()
            nn.init.xavier_uniform_(self.lin_angle[2].weight)
            self.lin_angle[2].bias.data.zero_()

        nn.init.xavier_uniform_(self.cond_proj[0].weight)
        self.cond_proj[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.cond_proj[2].weight)
        self.cond_proj[2].bias.data.zero_()

        for m in self.message_layers:
            m.reset_parameters()
        if self.use_fte:
            for f in self.FTEs:
                f.reset_parameters()

    @staticmethod
    def _compute_centered(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pos_sum = scatter(pos, batch, dim=0, reduce="sum")
        cnt = scatter(torch.ones_like(pos[:, :1]), batch, dim=0, reduce="sum").clamp(min=1.0)
        mean = pos_sum / cnt
        return pos - mean[batch]

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        node_batch: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,
        gbm_conditioner: Optional[nn.Module] = None,
        deep_film: bool = False,
        film_every: int = 1,
        film_layers: List[int] = None,
        film_beta_only: bool = False,
        film_scale: float = 1.0,
    ) -> torch.Tensor:
        if node_batch is None:
            node_batch = batch

        posc = self._compute_centered(pos, batch)

        if edge_index is None:
            edge_index = _radius_graph(posc, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        i, j = edge_index
        if i.numel() == 0:
            return self.z_emb(Z)

        # Geometry in fp32
        pos32 = posc.float()
        rij32 = pos32[i] - pos32[j]
        dist32 = rij32.norm(dim=-1).clamp(min=0.0)

        soft_c32 = 0.5 * (torch.cos(dist32 * math.pi / self.cutoff) + 1.0)
        soft_c32 = soft_c32 * (dist32 < self.cutoff).float()

        rbf32 = self.radial_emb(dist32)
        radial_hidden32 = self.radial_lin(rbf32)
        radial_hidden32 = soft_c32.unsqueeze(-1) * radial_hidden32

        edge_dir32 = _normalize(rij32)
        edge_frame32 = _stable_frame_from_dir(edge_dir32)

        mean_nb32 = self.mean_neighbor_pos(pos32, edge_index)
        node_dir32 = pos32 - mean_nb32
        node_dir_norm = node_dir32.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=pos32.device, dtype=pos32.dtype).view(1, 3)
        node_dir32 = torch.where(
            node_dir_norm > 1e-9,
            node_dir32 / node_dir_norm.clamp(min=1e-12),
            fallback.expand_as(node_dir32),
        )
        node_frame32 = _stable_frame_from_dir(node_dir32)

        mdl_dtype = self.z_emb.weight.dtype
        rbf = rbf32.to(mdl_dtype)
        radial_hidden = radial_hidden32.to(mdl_dtype)
        soft_cutoff = soft_c32.to(mdl_dtype)
        edge_dir = edge_dir32.to(mdl_dtype)

        s0 = self.z_emb(Z)
        s = self.neighbor_emb(Z, s0, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device, dtype=s.dtype)

        u32 = self.uvec(edge_index, edge_dir, rbf, soft_cutoff, num_nodes=s.size(0)) if self.use_uvec else None

        if self.use_lse:
            S = self.S_vector(s, edge_dir.unsqueeze(-1), edge_index, radial_hidden)

            proj_i = _project_to_frame(S[i].float(), edge_frame32)
            proj_j = _project_to_frame(S[j].float(), edge_frame32)
            feat_i = torch.log1p(proj_i * proj_i)
            feat_j = torch.log1p(proj_j * proj_j)
            feat_i = torch.permute(feat_i, (0, 2, 1))
            feat_j = torch.permute(feat_j, (0, 2, 1))
            sc3_32 = self.lin_lse(feat_i).squeeze(-1) + feat_i[:, :, 0]
            sc4_32 = self.lin_lse(feat_j).squeeze(-1) + feat_j[:, :, 0]

            if self.use_uvec and (u32 is not None):
                ui = u32[i]
                uj = u32[j]
                r = edge_dir32.unsqueeze(-1)

                nui = torch.norm(ui, dim=1)
                nuj = torch.norm(uj, dim=1)
                ai = (ui * r).sum(dim=1)
                aj = (uj * r).sum(dim=1)
                cos_i = ai / (nui + 1e-8)
                cos_j = aj / (nuj + 1e-8)
                uu = (ui * uj).sum(dim=1)
                cos_uu = uu / (nui * nuj + 1e-8)

                ui_eh3 = torch.permute(ui, (0, 2, 1))
                uj_eh3 = torch.permute(uj, (0, 2, 1))
                r_e13 = edge_dir32.unsqueeze(1)
                cross_i = torch.cross(ui_eh3, r_e13, dim=-1).norm(dim=-1)
                cross_j = torch.cross(uj_eh3, r_e13, dim=-1).norm(dim=-1)
                sin_i = cross_i / (nui + 1e-8)
                sin_j = cross_j / (nuj + 1e-8)

                fi = torch.stack([cos_i, cos_uu, torch.log1p(nui * nui), torch.log1p(sin_i * sin_i)], dim=-1)
                fj = torch.stack([cos_j, cos_uu, torch.log1p(nuj * nuj), torch.log1p(sin_j * sin_j)], dim=-1)
                add_i = self.lin_angle(fi.reshape(-1, 4)).reshape(fi.size(0), fi.size(1))
                add_j = self.lin_angle(fj.reshape(-1, 4)).reshape(fj.size(0), fj.size(1))
                sc3_32 = sc3_32 + add_i
                sc4_32 = sc4_32 + add_j

            scalar3 = sc3_32.to(mdl_dtype)
            scalar4 = sc4_32.to(mdl_dtype)
            a_full = torch.cat((scalar3, scalar4), dim=-1) * soft_cutoff.unsqueeze(-1)
            a_full = torch.cat((a_full, radial_hidden, rbf), dim=-1)
        else:
            pad_scalars = torch.zeros(rbf.size(0), 2 * self.hidden_channels, device=rbf.device, dtype=mdl_dtype)
            a_full = torch.cat((pad_scalars, radial_hidden, rbf), dim=-1)

        a_full = torch.tanh(a_full)
        cond_ij = torch.tanh(self.cond_proj(a_full))
        if self.use_geom_gate and self.geom_gate_cache is not None and self.geom_gate_proj is not None:
            geom_attr = self.geom_gate_cache(posc, edge_index, batch=batch)
            if self.geom_gate_norm is not None:
                geom_attr = self.geom_gate_norm(geom_attr)
            geom_gate = self.geom_gate_proj(geom_attr).to(dtype=cond_ij.dtype)
            cond_ij = cond_ij * (1.0 + self.geom_gate_scale * geom_gate)
        del a_full

        # Optional node FiLM injection (same as existing semantics)
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
            gamma, beta = gbm_conditioner.scalar_film(z_s)
            if film_beta_only:
                gamma = torch.ones_like(gamma)
            if film_scale != 1.0:
                gamma = 1.0 + film_scale * (gamma - 1.0)
                beta = film_scale * beta
            
            # Handle both Node-level (N, C) and Graph-level (B, C) FiLM
            if gamma.size(0) == h.size(0):
                return gamma * h + beta
            else:
                return gamma[node_batch] * h + beta[node_batch]

        node_frame = node_frame32.to(mdl_dtype)
        for li in range(self.num_layers):
            ds, dvec = self.message_layers[li](s, vec, edge_index, rbf, cond_ij, edge_dir)
            s = s + ds
            s = self.dropout(s)
            if self.use_vector_features:
                vec = vec + dvec

            if self.use_fte:
                ds2, dvec2 = self.FTEs[li](s, vec, node_frame)
                s = s + ds2
                s = self.dropout(s)
                if self.use_vector_features:
                    vec = vec + dvec2

            if self.use_nonlocal and self.nonlocal_block is not None:
                if self.nonlocal_pre_norm is not None:
                    s = self.nonlocal_pre_norm(s)
                s = s + self.nonlocal_scale * self.nonlocal_block(s, node_batch)
                if self.nonlocal_post_norm is not None:
                    s = self.nonlocal_post_norm(s)

            if (li + 1) in inject_set:
                s = _apply_node_film_local(s)

        return s

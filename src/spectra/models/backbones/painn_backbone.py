import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

# Try fast neighbor search; fall back if unavailable.
try:
    from torch_cluster import radius_graph as clust_radius_graph
except Exception:
    clust_radius_graph = None
try:
    from torch_geometric.nn import radius_graph as pyg_radius_graph
except Exception:
    pyg_radius_graph = None

from ..schnetpack import properties
from .painn import PaiNN
from ..fusion import get_fusion_layer
from ..heads import PhysicsHead


class _GaussianRBF(nn.Module):
    """Minimal Gaussian RBF expansion with fixed centers within [0, cutoff]."""

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, self.n_rbf)
        # Use constant width based on center spacing.
        delta = (self.cutoff / max(self.n_rbf - 1, 1)) if self.n_rbf > 1 else self.cutoff
        gamma = 1.0 / (max(delta, 1e-6) ** 2)
        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("gamma", torch.as_tensor(gamma), persistent=False)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [E, 1] -> [E, n_rbf]
        diff = d - self.centers.view(1, -1)
        return torch.exp(-self.gamma * diff.pow(2))


class _CosineCutoff(nn.Module):
    """Cosine cutoff; returns 0 outside the radius."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [E, 1]
        x = (math.pi * d) / self.cutoff
        val = 0.5 * (torch.cos(x).clamp(min=-1.0, max=1.0) + 1.0)
        mask = (d <= self.cutoff).to(val.dtype)
        return val * mask


def _build_edges(pos: torch.Tensor, batch: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Build radius graph edges. Falls back to O(N^2) per-graph if unavailable."""
    if clust_radius_graph is not None:
        return clust_radius_graph(pos, r=cutoff, batch=batch, loop=False)
    if pyg_radius_graph is not None:
        return pyg_radius_graph(pos, r=cutoff, batch=batch, loop=False)

    device = pos.device
    row_list, col_list = [], []
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    for b in range(B):
        idx = (batch == b).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        xb = pos[idx]  # [n, 3]
        d2 = torch.cdist(xb, xb, p=2.0)  # [n, n]
        mask = (d2 <= cutoff) & (~torch.eye(idx.numel(), dtype=torch.bool, device=device))
        src, dst = mask.nonzero(as_tuple=True)
        row_list.append(idx[src])
        col_list.append(idx[dst])
    if len(row_list) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return torch.stack([row, col], dim=0)


class _PaiNNSoluteEncoder(nn.Module):
    """Wrap PaiNN to work with (z, pos, batch) and internal radius graph."""

    def __init__(
        self,
        hidden: int,
        num_interactions: int,
        n_rbf: int,
        cutoff: float,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        max_z: int = 100,
        use_virtual_node: bool = False,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        use_geom_gate: bool = False,
        geom_n_rbf: int = 20,
        geom_use_moments: bool = True,
        geom_use_global: bool = True,
        geom_edge_scale: float = 0.1,
    ):
        super().__init__()
        self.cutoff = float(cutoff)

        # Minimal, dependency-free radial basis & cutoff (shape-compatible with PaiNN).
        radial_basis = _GaussianRBF(n_rbf=n_rbf, cutoff=self.cutoff)
        cutoff_fn = _CosineCutoff(cutoff=self.cutoff)

        self.painn = PaiNN(
            n_atom_basis=int(hidden),
            n_interactions=int(num_interactions),
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
            activation=F.silu,
            shared_interactions=bool(shared_interactions),
            shared_filters=bool(shared_filters),
            epsilon=float(epsilon),
            nuclear_embedding=nn.Embedding(int(max_z), int(hidden)),
            electronic_embeddings=None,
            use_virtual_node=use_virtual_node,
            dropout=dropout,
            residual_scale=float(residual_scale),
            use_geom_gate=use_geom_gate,
            geom_n_rbf=geom_n_rbf,
            geom_use_moments=geom_use_moments,
            geom_use_global=geom_use_global,
            geom_edge_scale=geom_edge_scale,
        )

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
    ) -> torch.Tensor:
        # Build neighbor list within cutoff and relative vectors Rij.
        edge_index = _build_edges(pos, batch, cutoff=self.cutoff)  # [2, E]
        idx_i, idx_j = edge_index[0], edge_index[1]
        r_ij = pos.index_select(0, idx_i) - pos.index_select(0, idx_j)  # [E, 3]

        if node_batch is None:
            node_batch = batch

        # Determine injection layers (1-based) if underlying model exposes interactions.
        L = len(getattr(self.painn, "interactions", []))
        if film_layers is not None:
            inject_set = {int(i) for i in film_layers if 1 <= int(i) <= L}
        elif deep_film:
            inject_set = {i for i in range(1, L + 1) if (i % max(int(film_every), 1) == 0)}
        else:
            inject_set = set()

        # Local FiLM utilities (node scalar and optional vector gating).
        def _apply_node_film_local(h: torch.Tensor) -> torch.Tensor:
            if (gbm_conditioner is None) or (z_s is None) or h is None:
                return h
            gamma, beta = gbm_conditioner.scalar_film(z_s)  # [B, C] or [N, C]
            if film_beta_only:
                gamma = torch.ones_like(gamma)
            if film_scale != 1.0:
                gamma = 1.0 + film_scale * (gamma - 1.0)
                beta = film_scale * beta
            
            # Auto-detect node-level
            is_node_level = (gamma.size(0) == h.size(0)) and (h.size(0) > 1)
            
            if is_node_level:
                g = gamma
                b = beta
            else:
                g = gamma[node_batch]
                b = beta[node_batch]
                
            return g * h + b

        def _apply_vec_gate_local(v: torch.Tensor) -> torch.Tensor:
            if (gbm_conditioner is None) or (z_s is None) or (v is None):
                return v
            # Support both [N,C,3] and [N,3,C] vector layouts.
            if v.dim() == 3 and v.size(-1) == 3:
                return gbm_conditioner.vector_gate_apply(v, z_s, node_batch)
            if v.dim() == 3 and v.size(1) == 3:
                v_t = v.transpose(1, 2)                     # [N, C, 3]
                v_t = gbm_conditioner.vector_gate_apply(v_t, z_s, node_batch)
                return v_t.transpose(1, 2)                  # restore shape
            return v

        # Prepare SchNetPack-style inputs for PaiNN.
        inputs = {
            properties.Z: Z,          # [N]
            properties.Rij: r_ij,     # [E, 3]
            properties.idx_i: idx_i,  # [E]
            properties.idx_j: idx_j,  # [E]
            properties.R: pos,        # [N, 3]
            # Pass conditioning context for potential deep injection inside PaiNN (if supported).
            "_conditioning": {
                "gbm_conditioner": gbm_conditioner,
                "z_s": z_s,
                "node_batch": node_batch,
                "deep_film": bool(deep_film),
                "film_every": int(film_every),
                "film_layers": list(film_layers) if film_layers is not None else None,
                "film_beta_only": bool(film_beta_only),
                "film_scale": float(film_scale),
                "inject_set": inject_set,
            },
        }

        outputs = self.painn(inputs)
        h = outputs["scalar_representation"]  # [N, H]

        # If deep_film requested but underlying PaiNN does not apply layer-wise injection,
        # apply a safe scalar FiLM at output as a minimal fallback.
        if deep_film and not inject_set:
            h = _apply_node_film_local(h)

        return h

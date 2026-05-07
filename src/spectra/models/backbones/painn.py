from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schnetpack import properties
from ..schnetpack import nn as snn


__all__ = ["PaiNN", "PaiNNInteraction", "PaiNNMixing"]


class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        dropout: float = 0.1,
        res_scale: float = 1.0,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.res_scale = float(res_scale)

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        Wij: torch.Tensor,
        dir_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
    ):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.interatomic_context_net(q)
        x = self.dropout(x)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij.unsqueeze(1) * xj  # [E,1,3H] * [E,1,3H] -> [E,1,3H]

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = snn.scatter_add(dmu, idx_i, dim_size=n_atoms)

        q = q + self.res_scale * dq
        mu = mu + self.res_scale * dmu

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        epsilon: float = 1e-8,
        dropout: float = 0.1,
        res_scale: float = 1.0,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = snn.Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.res_scale = float(res_scale)

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        # intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)
        x = self.dropout(x)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + self.res_scale * (dq_intra + dqmu_intra)
        mu = mu + self.res_scale * dmu_intra
        return q, mu


# painn.py
# PaiNN model with optional deep FiLM conditioning and vector gating hooks.
# Changes:
# - Robust handling of optional cutoff_fn (self.cutoff may be None).
# - Layer-wise FiLM/gating injection via _conditioning context.
# - Returns "_applied_deep_film" flag to signal whether deep FiLM ran.
# - film_layers accepts 0- or 1-based indices (auto-normalized to 1-based).

from typing import Optional, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
from ..schnetpack import properties
from spectra.models.ehc.edge_geom_cache import EdgeGeomCache


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:
    Schütt, Unke, Gastegger:
    Equivariant message passing for the prediction of tensorial properties and molecular spectra.
    ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = F.silu,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        dropout: float = 0.1,
        epsilon: float = 1e-8,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
        use_virtual_node: bool = False,
        residual_scale: float = 1.0,
        use_geom_gate: bool = False,
        geom_n_rbf: int = 20,
        geom_use_moments: bool = True,
        geom_use_global: bool = True,
        geom_edge_scale: float = 0.1,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set.
                          Must expose attribute `n_rbf`.
            cutoff_fn: cutoff function; if provided, expected to have attribute `cutoff`.
            activation: activation function for interaction/mixing blocks.
            shared_interactions: if True, share the weights across interaction blocks.
            shared_filters: if True, share the weights across filter-generating networks.
            epsilon: numerical stability parameter (used in mixing).
            nuclear_embedding: custom nuclear embedding (nn.Embedding-like).
            electronic_embeddings: list of electronic embedding modules.
            use_virtual_node: if True, enable global virtual node communication.
            residual_scale: scale factor applied to residual updates in interaction/mixing blocks.
        """
        super().__init__()

        self.n_atom_basis = int(n_atom_basis)
        self.n_interactions = int(n_interactions)
        self.radial_basis = radial_basis
        self.use_virtual_node = use_virtual_node

        # Optional cutoff; tolerate None and missing attribute.
        self.cutoff_fn = cutoff_fn
        self.cutoff = getattr(cutoff_fn, "cutoff", None)

        # Initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, self.n_atom_basis)
        self.embedding = nuclear_embedding

        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings)

        # Global Virtual Node MLP
        if self.use_virtual_node:
            self.global_norm = nn.LayerNorm(self.n_atom_basis)
            self.global_mlp = nn.Sequential(
                snn.Dense(self.n_atom_basis, self.n_atom_basis, activation=activation),
                snn.Dense(self.n_atom_basis, self.n_atom_basis, activation=torch.tanh) # Bound output
            )

        # Initialize filter layers
        self.share_filters = bool(shared_filters)
        self.residual_scale = float(residual_scale)
        if self.share_filters:
            # Shared filter net across interactions
            self.filter_net = snn.Dense(self.radial_basis.n_rbf, 3 * self.n_atom_basis, activation=None)
        else:
            # Separate filters for each interaction
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * self.n_atom_basis * 3,
                activation=None,
            )

        self.use_geom_gate = bool(use_geom_gate)
        self.geom_edge_scale = float(geom_edge_scale)
        if self.use_geom_gate:
            geom_edge_dim = int(geom_n_rbf)
            if geom_use_moments:
                geom_edge_dim += 5
            if geom_use_global:
                geom_edge_dim += 2
            self.geom_edge_cache = EdgeGeomCache(
                cutoff=self.cutoff,
                n_rbf=geom_n_rbf,
                tanh_clip=True,
                use_moments=geom_use_moments,
                use_global=geom_use_global,
            )
            gate_dim = 3 * self.n_atom_basis if self.share_filters else 3 * self.n_atom_basis * self.n_interactions
            self.geom_gate_proj = nn.Sequential(
                nn.Linear(geom_edge_dim, self.n_atom_basis),
                nn.SiLU(),
                nn.Linear(self.n_atom_basis, gate_dim),
                nn.Tanh(),
            )
        else:
            self.geom_edge_cache = None
            self.geom_gate_proj = None

        # Initialize interaction/mixing blocks
        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis,
                activation=activation,
                dropout=dropout,
                res_scale=self.residual_scale,
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.n_atom_basis,
                activation=activation,
                epsilon=epsilon,
                dropout=dropout,
                res_scale=self.residual_scale,
            ),
            self.n_interactions,
            shared_interactions,
        )

    def apply_geom_gate(
        self,
        filters: torch.Tensor,
        pos: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        node_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_geom_gate:
            return filters
        if self.geom_edge_cache is None or self.geom_gate_proj is None:
            raise RuntimeError("Geom gate enabled but modules are missing.")
        if pos is None or edge_index is None:
            raise RuntimeError("Geom gate requires positions and edge_index.")
        edge_attr_cached = self.geom_edge_cache(pos, edge_index, batch=node_batch)
        geom_gate = self.geom_gate_proj(edge_attr_cached)
        geom_gate = torch.nan_to_num(geom_gate)
        filters = filters * (1.0 + self.geom_edge_scale * geom_gate)
        return torch.nan_to_num(filters)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute atom-wise scalar and vector representations.

        Expects keys:
          - properties.Z: [N] long
          - properties.Rij: [E, 3] float (convention: Rij = pos[i] - pos[j])
          - properties.idx_i: [E] long
          - properties.idx_j: [E] long
        Optional conditioning context in inputs["_conditioning"].

        Returns:
          inputs dict extended with:
            - "scalar_representation": [N, H]
            - "vector_representation": [N, 3, H]
            - "_applied_deep_film": bool
        """
        # Conditioning context (optional)
        cond = inputs.get("_conditioning", None)
        gbm_conditioner = cond.get("gbm_conditioner") if isinstance(cond, dict) else None
        z_s = cond.get("z_s") if isinstance(cond, dict) else None
        node_batch = cond.get("node_batch") if isinstance(cond, dict) else None
        deep_film = bool(cond.get("deep_film", False)) if isinstance(cond, dict) else False
        film_layers = cond.get("film_layers") if isinstance(cond, dict) else None
        film_every = int(cond.get("film_every", 1)) if isinstance(cond, dict) else 1
        film_beta_only = bool(cond.get("film_beta_only", False)) if isinstance(cond, dict) else False
        film_scale = float(cond.get("film_scale", 1.0)) if isinstance(cond, dict) else 1.0
        inject_set_ctx = cond.get("inject_set") if isinstance(cond, dict) else None

        # Input tensors
        atomic_numbers = inputs[properties.Z]           # [N]
        r_ij = inputs[properties.Rij]                   # [E, 3]
        idx_i = inputs[properties.idx_i]                # [E]
        idx_j = inputs[properties.idx_j]                # [E]
        n_atoms = int(atomic_numbers.shape[0])

        # Pairwise features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)    # [E, 1]
        dir_ij = F.normalize(r_ij, p=2, dim=1, eps=1e-9)  # [E, 3]
        phi_ij = self.radial_basis(d_ij)                # [E, n_rbf]

        filters = self.filter_net(phi_ij)               # [E, 3H] or [E, L*3H]
        if self.cutoff_fn is not None:
            fcut = self.cutoff_fn(d_ij).view(-1, 1)     # [E, 1]
            filters = filters * fcut
        if self.use_geom_gate:
            edge_index = torch.stack([idx_i, idx_j], dim=0)
            pos = inputs.get(properties.R)
            filters = self.apply_geom_gate(filters, pos, edge_index, node_batch)
        filters = torch.nan_to_num(filters)

        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # Initial embeddings
        q = self.embedding(atomic_numbers)              # [N, H]
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)                              # [N, 1, H]

        # Vector channel (equivariant)
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)  # [N, 3, H]

        # Determine layers for injection (1-based)
        L = self.n_interactions
        if isinstance(inject_set_ctx, (set, list, tuple)) and len(inject_set_ctx) > 0:
            inject_set = {int(i) for i in inject_set_ctx if 1 <= int(i) <= L}
        elif deep_film:
            if film_layers is not None:
                vals = [int(i) for i in film_layers]
                # Accept 0-based layer ids: if any id is 0 and none negative, convert to 1-based.
                if any(i == 0 for i in vals) and all(i >= 0 for i in vals):
                    vals = [i + 1 for i in vals]
                inject_set = {i for i in vals if 1 <= i <= L}
            else:
                step = max(int(film_every), 1)
                inject_set = {i for i in range(1, L + 1) if (i % step) == 0}
        else:
            inject_set = set()

        # Local FiLM utilities; track whether any deep FiLM/gate actually applied
        applied_deep_film = False

        def _apply_scalar_film_local(q_in: torch.Tensor) -> torch.Tensor:
            nonlocal applied_deep_film
            if q_in is None or gbm_conditioner is None or z_s is None or node_batch is None:
                return q_in
            gamma, beta = gbm_conditioner.scalar_film(z_s)  # [B/N, H]
            if film_beta_only:
                gamma = torch.ones_like(gamma)
            if film_scale != 1.0:
                gamma = 1.0 + film_scale * (gamma - 1.0)
                beta = film_scale * beta
            
            # Auto-detect node-level
            # q_in is [N, 1, H]
            is_node_level = (gamma.size(0) == q_in.size(0)) and (q_in.size(0) > 1)
            
            if is_node_level:
                g = gamma
                b = beta
            else:
                g = gamma[node_batch]  # [N, H]
                b = beta[node_batch]   # [N, H]
                
            applied_deep_film = True
            return g.unsqueeze(1) * q_in + b.unsqueeze(1)  # [N, 1, H]

        def _apply_vector_gate_local(mu_in: torch.Tensor) -> torch.Tensor:
            nonlocal applied_deep_film
            if mu_in is None or gbm_conditioner is None or z_s is None or node_batch is None:
                return mu_in
            # mu: [N, 3, H] -> [N, H, 3], gate across H (per-channel scalar)
            mu_t = mu_in.transpose(1, 2)  # [N, H, 3]
            mu_t = gbm_conditioner.vector_gate_apply(mu_t, z_s, node_batch)  # [N, H, 3]
            applied_deep_film = True
            return mu_t.transpose(1, 2)  # back to [N, 3, H]

        # Init global state (hidden)
        global_state = None 

        # Interaction + mixing blocks
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing), 1):
            q, mu = interaction(q, mu, filter_list[i - 1], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

            # --- Virtual Node Injection ---
            if self.use_virtual_node and node_batch is not None:
                # q is [N, 1, H] -> squeeze to [N, H] for aggregation
                q_flat = q.squeeze(1)
                
                # 1. Aggregate
                num_graphs = int(node_batch.max().item()) + 1
                # Use snn.scatter_mean if available or implement manual
                # Using pure torch scatter logic for safety as snn.scatter_add is available
                # But snn.scatter_add might not handle mean. Let's do manual sum/count
                # or assume global_mean_pool from torch_geometric is cleaner but we don't have it imported here?
                # Actually, I can use snn.scatter_add and divide by counts?
                # snn is imported. Let's check snn.scatter_add signature? 
                # Defined in PaiNNInteraction: dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)
                # It just sums.
                
                # Global Mean Pooling
                # We need to sum q_flat by batch
                # Assuming batch is [N]
                
                # Init global_state if needed
                if global_state is None:
                    global_state = torch.zeros(num_graphs, self.n_atom_basis, device=q.device, dtype=q.dtype)

                batch_sum = torch.zeros_like(global_state)
                batch_sum.index_add_(0, node_batch, q_flat)
                
                batch_count = torch.zeros(num_graphs, device=q.device, dtype=q.dtype)
                batch_count.index_add_(0, node_batch, torch.ones_like(node_batch, dtype=q.dtype))
                batch_mean = batch_sum / batch_count.clamp(min=1).unsqueeze(-1)
                
                # 2. Update Global State
                # G_new = MLP(Norm(G_old + Mean))
                # Residual connection for stability
                update_in = self.global_norm(global_state + batch_mean)
                global_delta = self.global_mlp(update_in)
                global_state = global_state + global_delta
                
                # 3. Broadcast back
                # q = q + 0.1 * G[batch] (Scaled residual)
                q_update = global_state[node_batch].unsqueeze(1) # [N, 1, H]
                q = q + 0.1 * q_update

            if i in inject_set:
                q = _apply_scalar_film_local(q)
                mu = _apply_vector_gate_local(mu)

        q = q.squeeze(1)  # [N, H]

        # Collect results
        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu
        inputs["_applied_deep_film"] = applied_deep_film

        return inputs

import warnings
from typing import Optional, List, Set, Tuple, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from torch_geometric.utils import to_dense_batch
except ImportError:
    to_dense_batch = None

# Backbones
from spectra.models.backbones.egnn_backbone import EGNNBackbone
from spectra.models.backbones.schnet_backbone import SchNet
from spectra.models.backbones.painn_backbone import _PaiNNSoluteEncoder
from spectra.models.backbones.egnn_layers import unsorted_segment_mean
from spectra.models.backbones.nonlocal_performer import PerformerNonLocal
from spectra.models.backbones.leftnet_backbone import LEFTNetBackbone
from spectra.models.backbones.leftnet_backbone_v3 import LEFTNetBackboneV3Cond
from spectra.models.backbones.equiformer.graph_attention_transformer import GraphAttentionTransformer
from spectra.models.backbones.equiformer.dp_attention_transformer import DotProductAttentionTransformer

# Conditioning & Protocol
from .gbm_conditioning import GBMConditioner, compute_inject_set
from .backbone_protocol import BackboneProtocol, BackboneOutput, ConditionContext, FiLMStrategy
from .edge_geom_cache import EdgeGeomCache
from .geometry_enhancer import GeometryEnhancer, stable_frame_from_dir

# Optional deps
try:
    from e3nn import o3
except ImportError:
    o3 = None

# Equiformer V2 imports
from spectra.models.backbones.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from spectra.models.backbones.equiformer_v2.so3 import SO3_Embedding
from spectra.models.backbones.equiformer_v2.edge_rot_mat import init_edge_rot_mat

# CGNN Import
from spectra.models.backbones.cgnn.cgnn import CGNN

class SelectiveInteractionBlock(nn.Module):
    """
    轻量级 Cross-Attention 模块，实现“溶质原子主动选择溶剂环境”。
    Query: 溶质原子特征 h_i
    Key/Value: 溶剂图的所有节点特征 {s_j}
    Output: 为每个溶质原子生成的特异性溶剂上下文 z_s_node[i]
    """
    def __init__(self, node_dim: int, solvent_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=node_dim, 
            num_heads=num_heads, 
            kdim=solvent_dim, 
            vdim=solvent_dim, 
            batch_first=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(node_dim)
        self.out_proj = nn.Linear(node_dim, solvent_dim)

    def forward(
        self, 
        h: torch.Tensor, 
        node_batch: torch.Tensor,
        solvent_h: torch.Tensor, 
        solvent_batch: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        h: [N_solute, D_node]
        solvent_h: [N_solvent, D_solv]
        """
        if to_dense_batch is None:
            return torch.zeros_like(h), None

        h_dense, h_mask = to_dense_batch(h, node_batch)
        s_dense, s_mask = to_dense_batch(solvent_h, solvent_batch, max_num_nodes=None)
        
        B = h_dense.size(0)
        if s_dense.size(0) < B:
            pad_s = s_dense.new_zeros(B - s_dense.size(0), *s_dense.shape[1:])
            pad_m = s_mask.new_zeros(B - s_mask.size(0), s_mask.shape[1], dtype=torch.bool)
            s_dense = torch.cat([s_dense, pad_s], dim=0)
            s_mask = torch.cat([s_mask, pad_m], dim=0)
        elif s_dense.size(0) > B:
             s_dense = s_dense[:B]
             s_mask = s_mask[:B]

        attn_out, attn_weights = self.attn(
            query=h_dense, 
            key=s_dense, 
            value=s_dense, 
            key_padding_mask=~s_mask
        )
        
        out_sparse = attn_out[h_mask] # [N_solute, D_node]
        z_s_node = self.out_proj(self.norm(out_sparse)) # [N_solute, D_solv]
        
        return z_s_node, attn_weights


# ---------------- Utilities ----------------

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
        return _clust_radius_graph(pos, r=r, batch=batch, loop=False, max_num_neighbors=max_num_neighbors)
    if _pyg_radius_graph is not None:
        return _pyg_radius_graph(pos, r=r, batch=batch, loop=False)
    
    raise ImportError("Standard radius_graph implementation not found.")




ELEMENTS12_Z = torch.tensor([6, 7, 8, 16, 9, 17, 35, 1, 15, 5, 14, 53], dtype=torch.long)

def decode_Z_or_idx(node_feats: torch.Tensor, want: str, max_num_elements: int = 12) -> torch.Tensor:
    device = node_feats.device
    
    if node_feats.dim() == 2 and node_feats.size(-1) == 12:
        idx = node_feats.argmax(dim=-1)
        if want == 'idx':
            return idx.clamp_(0, max_num_elements - 1)
        return ELEMENTS12_Z.to(device)[idx]

    if node_feats.dim() == 1:
        z = node_feats.long()
    else:
        z = node_feats[:, 0].long()

    if want == 'idx':
        z_map = torch.full((100,), 0, device=device, dtype=torch.long)
        z_map[ELEMENTS12_Z.to(device)] = torch.arange(12, device=device)
        idx = z_map[z.clamp(0, 99)]
        return idx.clamp(0, max_num_elements - 1)
        
    return z


def parse_class_map(class_map) -> Optional[list[int]]:
    if class_map is None:
        return None
    if isinstance(class_map, str):
        key = class_map.strip().lower()
        if key in {"12", "elements12"}:
            return [int(x) for x in ELEMENTS12_Z.tolist()]
        if key in {"qm9"}:
            return [1, 6, 7, 8, 9]
    if isinstance(class_map, int):
        if class_map == 12:
            return [int(x) for x in ELEMENTS12_Z.tolist()]
        raise ValueError(f"Unsupported class_map integer: {class_map}")
    if isinstance(class_map, (list, tuple)):
        return [int(x) for x in class_map]
    raise ValueError(f"Unsupported class_map type: {type(class_map)}")


# ---------------- Mixin & Base Adapter ----------------

class _FiLMMixin:
    """
    适配器通用 Mixin：负责管理 FiLM 注入策略和执行具体的注入操作。
    """
    def __init__(self):
        self._film_strategy: Optional[FiLMStrategy] = None
        self._inject_set: Optional[Set[int]] = None
        self._film_injected_layers: List[int] = []

    def set_film_strategy(self, film: Optional[FiLMStrategy], num_layers: int) -> None:
        self._film_strategy = film
        self._inject_set = compute_inject_set(num_layers, film) if film is not None else None
        self._film_injected_layers = [] 

    def _should_inject(self, layer_idx_1based: int) -> bool:
        if self._inject_set is not None:
            return layer_idx_1based in self._inject_set
        return False
    
    def _mark_injected(self, layer_idx_1based: int) -> None:
        self._film_injected_layers.append(int(layer_idx_1based))

    def _apply_node_film(self, h: torch.Tensor, z_s: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
        if (not hasattr(self, "gbm")) or (self.gbm is None): return h
        if (not hasattr(self.gbm, "scalar_film")) or (not getattr(self.gbm, "use_scalar_film", True)): pass

        gamma, beta = self.gbm.scalar_film(z_s)  # [B/N, C]

        film = self._film_strategy
        beta_only = film.beta_only if film is not None else False
        scale = film.scale if film is not None else 1.0

        if beta_only:
            gamma = torch.ones_like(gamma)
        if scale != 1.0:
            gamma = 1.0 + scale * (gamma - 1.0)
            beta = scale * beta

        is_node_level = (gamma.size(0) == h.size(0)) and (h.size(0) > 1)
        
        if is_node_level:
            g, b = gamma, beta
        else:
            g, b = gamma[node_batch], beta[node_batch]
            
        return g * h + b

    def _apply_edge_film(self, e: torch.Tensor, z_s: torch.Tensor, edge_batch: torch.Tensor) -> torch.Tensor:
        if (not hasattr(self, "gbm") or self.gbm is None or not hasattr(self.gbm, "edge_film") 
            or not getattr(self.gbm, "use_edge_film", True)):
            return e

        gamma, beta = self.gbm.edge_film(z_s)
        film = self._film_strategy
        beta_only = film.beta_only if film is not None else False
        scale = film.scale if film is not None else 1.0

        if beta_only: gamma = torch.ones_like(gamma)
        if scale != 1.0:
            gamma = 1.0 + scale * (gamma - 1.0)
            beta = scale * beta

        return gamma[edge_batch] * e + beta[edge_batch]


class BaseSolventAdapter(_FiLMMixin, nn.Module):
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        edge_dim: int = 0,
        use_scalar_film: bool = True,
        use_edge_film: bool = False,
        use_vector_gate: bool = False
    ):
        nn.Module.__init__(self)
        _FiLMMixin.__init__(self)
        
        self.hidden_dim = hidden_dim
        
        self.gbm = GBMConditioner(
            z_dim=z_dim,
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            use_scalar_film=use_scalar_film,
            use_edge_film=use_edge_film,
            use_vector_gate=use_vector_gate,
        )
        
        self.selective_interaction = SelectiveInteractionBlock(
            node_dim=hidden_dim,
            solvent_dim=z_dim
        )
        
    def _get_solvent_context(self, h_query: torch.Tensor, node_batch: torch.Tensor, condition: ConditionContext) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_weights = None
        if (condition.solvent_x is not None) and (condition.solvent_batch is not None):
            z_s_node, attn_weights = self.selective_interaction(
                h=h_query, 
                node_batch=node_batch, 
                solvent_h=condition.solvent_x, 
                solvent_batch=condition.solvent_batch
            )
        else:
            z_s_node = condition.z_s_node if condition.z_s_node is not None else condition.z_s
            
        return z_s_node, attn_weights

    def _setup_runtime(self, condition: ConditionContext, num_layers: int) -> None:
        if condition.gbm is not None:
            self.gbm = condition.gbm # Allow override
        if condition.film is not None:
            self.set_film_strategy(condition.film, num_layers=num_layers)


# ---------------- EGNN (With Late Querying) ----------------

class EGNN_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        node_in_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        num_layers: int,
        z_dim: int,
        use_scalar_film: bool = True,
        use_edge_film: bool = True,
        use_vector_gate: bool = False,
        query_layer_index: int = 1,
        use_nonlocal: bool = False,
        nonlocal_heads: int = 4,
        nonlocal_features: int = 64,
        nonlocal_dropout: float = 0.0,
        nonlocal_scale: float = 1.0,
        edge_attr_mode: Literal["invariant_only", "legacy"] = "legacy",
        use_fcvm: bool = False,
        chirality_invariant: bool = False,
        fcvm_update_frames: Literal["once", "per_layer"] = "once",
        clamp: bool = False,
        coords_weight: float = 1.0,
        use_rbf: bool = False,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        log_fcvm: bool = False,
        log_config: Optional[dict] = None,
        use_vector_features: bool = False,
        use_moments: bool = True,
        use_global: bool = True,
        use_virtual_node: bool = True,
        use_advanced_geometry: bool = True,
        use_vector_mixing: bool = False,
        vector_mix_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        # External Cache Setup
        # n_rbf + 5 (moments) + 2 (global anchors)
        cached_edge_dim = int(n_rbf)
        if use_moments: cached_edge_dim += 5
        if use_global: cached_edge_dim += 2
        
        super().__init__(
            z_dim=z_dim, hidden_dim=hidden_dim, edge_dim=cached_edge_dim,
            use_scalar_film=use_scalar_film, use_edge_film=use_edge_film, use_vector_gate=use_vector_gate
        )
        
        # New Embedding & Cache
        self.atom_emb = nn.Embedding(95, hidden_dim) # Support up to 95 elements
        self.geom_enhancer = GeometryEnhancer(
            cutoff=cutoff,
            n_rbf=n_rbf,
            use_moments=use_moments,
            use_global=use_global,
            use_edge_frame=False,
            tanh_clip=True,
        )
        self.edge_cache = self.geom_enhancer.edge_cache
        self.nei_rbf_proj = nn.Linear(n_rbf, hidden_dim, bias=False)
        
        self.backbone = EGNNBackbone(
            node_in_dim=hidden_dim, # Expects already embedded H
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            edge_attr_dim=cached_edge_dim, 
            attention=True,
            edge_attr_mode="legacy", # Using raw inputs from cache
            use_fcvm=False,          # Disabled internal calculation
            chirality_invariant=chirality_invariant,
            fcvm_update_frames="once",
            clamp=clamp, 
            coords_weight=0.0,       # Force freeze
            use_rbf=False,           # Disabled internal RBF
            n_rbf=n_rbf, 
            cutoff=cutoff,
            log_fcvm=log_fcvm, 
            log_config=log_config or {},
            skip_node_embedding=True, # Bypass backbone embedding
            use_vector_features=use_vector_features,
            use_virtual_node=use_virtual_node,
            use_advanced_geometry=use_advanced_geometry,
            use_vector_mixing=use_vector_mixing,
            vector_mix_scale=vector_mix_scale,
            dropout=dropout,
        )
        self.layers = self.backbone.layers
        self.pool_head = self.backbone.pool
        self._use_edge_film = bool(use_edge_film)
        self.query_layer_index = int(query_layer_index)
        self.edge_attr_mode = edge_attr_mode
        self.use_fcvm = use_fcvm
        self.chirality_invariant = chirality_invariant
        self.fcvm_update_frames = fcvm_update_frames
        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)
        self.nonlocal_block = None
        self.nonlocal_norm = None
        if self.use_nonlocal:
            self.nonlocal_block = PerformerNonLocal(
                hidden_dim=hidden_dim,
                num_heads=nonlocal_heads,
                num_features=nonlocal_features,
                dropout=nonlocal_dropout,
            )
            self.nonlocal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        if edge_index is None: raise ValueError("EGNN requires edge_index")
        
        # 1. Runtime Setup
        self._setup_runtime(condition, len(self.layers))
        z_s = condition.z_s

        # 2. Input Embedding (External)
        Z_in = decode_Z_or_idx(Z, want='Z').long()
        h0 = self.atom_emb(Z_in) # [N, H]
        
        # 3. External Edge Cache (RBF + Moments + Global)
        # Note: We ignore incoming edge_attr and compute from scratch based on pos
        geom_out = self.geom_enhancer(pos, edge_index, batch=node_batch)
        edge_attr_cached = geom_out["edge_attr"]  # [E, n_rbf+5+2]
        
        # 4. Neighbor Embedding Lite
        # Aggregate neighbor features weighted by projected RBF
        row, col = edge_index
        rbf = edge_attr_cached[:, :self.edge_cache.n_rbf] # [E, n_rbf]
        w = self.nei_rbf_proj(rbf) # [E, H]
        msg = w * h0[col]
        agg = torch.zeros_like(h0)
        agg.index_add_(0, row, msg)
        h0 = h0 + agg
        
        h = h0

        # Initialize context with fallback (global z_s)
        z_s_node = condition.z_s
        attn_weights = None
        
        # Check if Cross-Attn is enabled
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        # Early Querying (Layer 0)
        if self.query_layer_index == 0 and cross_attn_enabled:
             z_s_node, attn_weights = self._get_solvent_context(h, node_batch, condition)

        # Use cached edges
        x = pos
        eattr = edge_attr_cached
        
        row = edge_index[0]
        edge_batch = node_batch[row] if row.numel() > 0 else row
        
        # Vector Features State
        v = None
        global_h = None

        # 5. Layers
        for li, layer in enumerate(self.layers, 1):
            inject_here = self._should_inject(li)
            
            # Edge FiLM uses global z_s
            if self._use_edge_film and (eattr is not None) and inject_here and eattr.numel() > 0:
                eattr_mod = self._apply_edge_film(eattr, z_s, edge_batch)
            else:
                eattr_mod = eattr

            # --- Backbone Layer Execution ---
            # use_fcvm is False in backbone, so it behaves as legacy GNN taking full edge_attr
            # V-EGNN: Pass v and receive v_new
            # Update: Pass global_h and batch for Virtual Node support
            h, x, _, v, global_h = layer(
                h, edge_index, x, edge_attr=eattr_mod, fcvm_frames=None, v=v, 
                global_h=global_h, batch=node_batch
            )

            if self.use_nonlocal and self.nonlocal_block is not None:
                h = h + self.nonlocal_scale * self.nonlocal_block(h, node_batch)
                if self.nonlocal_norm is not None:
                    h = self.nonlocal_norm(h)
            
            # --- Late Querying Logic ---
            # If we just finished Layer K, update z_s_node using the geometry-aware h
            if li == self.query_layer_index and cross_attn_enabled:
                z_s_node, attn_weights = self._get_solvent_context(h, node_batch, condition)

            # Node FiLM uses potentially updated z_s_node
            if inject_here:
                h = self._apply_node_film(h, z_s_node, node_batch)
                self._mark_injected(li)

            if self.gbm.use_vector_gate and v is not None and inject_here:
                v = self.gbm.vector_gate_apply(v, z_s_node, node_batch)

        # 6. Output
        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = self.pool_head(unsorted_segment_mean(h, node_batch, num_segments=num_graphs))

        return BackboneOutput(
            node_scalar=h, coords=x, graph_emb=graph_emb,
            aux={"film_layers": self._film_injected_layers, "attn_weights": attn_weights, 
                 "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


# ---------------- SchNet (With Late Querying) ----------------

class SchNet_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        n_atom_basis: int = 128,
        n_filters: int = 128,
        n_interactions: int = 6,
        n_gaussians: int = 100,
        cutoff: float = 5.0,
        use_scalar_film: bool = True,
        query_layer_index: int = 1,
        use_geom: bool = False,
        geom_n_rbf: int = 20,
        geom_use_moments: bool = False,
        geom_use_global: bool = False,
        use_nonlocal: bool = False,
        nonlocal_heads: int = 4,
        nonlocal_features: int = 64,
        nonlocal_dropout: float = 0.0,
        nonlocal_scale: float = 0.1,
    ):
        super().__init__(
            z_dim=z_dim, hidden_dim=n_atom_basis, use_scalar_film=use_scalar_film
        )
        self.use_geom = bool(use_geom)
        self.geom_enhancer = None
        edge_attr_dim = None
        if self.use_geom:
            edge_attr_dim = int(geom_n_rbf)
            if geom_use_moments:
                edge_attr_dim += 5
            if geom_use_global:
                edge_attr_dim += 2
            self.geom_enhancer = GeometryEnhancer(
                cutoff=cutoff,
                n_rbf=geom_n_rbf,
                use_moments=geom_use_moments,
                use_global=geom_use_global,
                use_edge_frame=False,
                tanh_clip=True,
            )
        self.schnet = SchNet(
            n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=n_interactions,
            n_gaussians=n_gaussians, cutoff=cutoff, max_z=100, cutoff_network="cosine",
            edge_attr_dim=edge_attr_dim,
        )
        self.query_layer_index = int(query_layer_index)
        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)
        self.nonlocal_block = None
        self.nonlocal_pre_norm = None
        self.nonlocal_post_norm = None
        if self.use_nonlocal:
            self.nonlocal_block = PerformerNonLocal(
                hidden_dim=n_atom_basis,
                num_heads=nonlocal_heads,
                num_features=nonlocal_features,
                dropout=nonlocal_dropout,
            )
            self.nonlocal_pre_norm = nn.LayerNorm(n_atom_basis)
            self.nonlocal_post_norm = nn.LayerNorm(n_atom_basis)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        Z_in = decode_Z_or_idx(Z, want='Z')
        self._setup_runtime(condition, self.schnet.n_interactions)
        
        # 1. Initial Embedding
        h = self.schnet.embedding(Z_in)
        
        # 2. Build Edges (Standard SchNet behavior)
        if edge_index is None:
            from torch_geometric.nn import radius_graph as _rg
            edge_index = _rg(pos, r=self.schnet.cutoff, batch=node_batch)

        edge_attr = None
        if self.use_geom and self.geom_enhancer is not None:
            geom_out = self.geom_enhancer(pos, edge_index, batch=node_batch)
            edge_attr = geom_out["edge_attr"]

        # 3. Solvent Context Initialization
        z_s_node = condition.z_s
        attn_weights = None
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        if self.query_layer_index == 0 and cross_attn_enabled:
            z_s_node, attn_weights = self._get_solvent_context(h, node_batch, condition)

        # 4. Interaction Blocks (Unrolled)
        for i, interaction in enumerate(self.schnet.interactions, 1):
            h = interaction(h, pos, edge_index, edge_attr=edge_attr)
            if self.use_nonlocal and self.nonlocal_block is not None:
                h_nl = h
                if self.nonlocal_pre_norm is not None:
                    h_nl = self.nonlocal_pre_norm(h_nl)
                h = h + self.nonlocal_scale * self.nonlocal_block(h_nl, node_batch)
                if self.nonlocal_post_norm is not None:
                    h = self.nonlocal_post_norm(h)
            
            # Late Querying update
            if i == self.query_layer_index and cross_attn_enabled:
                z_s_node, attn_weights = self._get_solvent_context(h, node_batch, condition)
            
            # Deep FiLM Injection
            if self._should_inject(i):
                h = self._apply_node_film(h, z_s_node, node_batch)
                self._mark_injected(i)

        # 5. Global FiLM fallback
        if not self._film_injected_layers:
            h = self._apply_node_film(h, z_s_node, node_batch)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(h, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=h, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": self._film_injected_layers, "attn_weights": attn_weights, 
                 "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


# ---------------- PaiNN (With Late Querying) ----------------

class PaiNN_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        hidden: int = 128,
        num_interactions: int = 6,
        n_rbf: int = 64,
        cutoff: float = 5.0,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        use_scalar_film: bool = True,
        use_vector_gate: bool = False,
        query_layer_index: int = 1,
        use_virtual_node: bool = True,
        use_nonlocal: bool = False,
        nonlocal_heads: int = 4,
        nonlocal_features: int = 64,
        nonlocal_dropout: float = 0.0,
        nonlocal_scale: float = 1.0,
        use_geom_gate: bool = False,
        geom_n_rbf: int = 20,
        geom_use_moments: bool = True,
        geom_use_global: bool = True,
        geom_edge_scale: float = 0.1,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ):
        super().__init__(
            z_dim=z_dim, hidden_dim=hidden, use_scalar_film=use_scalar_film, use_vector_gate=use_vector_gate
        )
        self.encoder = _PaiNNSoluteEncoder(
            hidden=hidden, num_interactions=num_interactions, n_rbf=n_rbf, cutoff=cutoff,
            shared_interactions=shared_interactions, shared_filters=shared_filters,
            epsilon=1e-8, max_z=100,
            use_virtual_node=use_virtual_node,
            dropout=dropout,
            residual_scale=residual_scale,
            use_geom_gate=use_geom_gate,
            geom_n_rbf=geom_n_rbf,
            geom_use_moments=geom_use_moments,
            geom_use_global=geom_use_global,
            geom_edge_scale=geom_edge_scale,
        )
        self.painn_share_filters = bool(shared_filters)
        self.num_layers = num_interactions
        self.query_layer_index = int(query_layer_index)
        self.virtual_node_scale = 0.1
        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)
        self.nonlocal_block = None
        self.nonlocal_pre_norm = None
        self.nonlocal_post_norm = None
        if self.use_nonlocal:
            self.nonlocal_block = PerformerNonLocal(
                hidden_dim=hidden,
                num_heads=nonlocal_heads,
                num_features=nonlocal_features,
                dropout=nonlocal_dropout,
            )
            self.nonlocal_pre_norm = nn.LayerNorm(hidden)
            self.nonlocal_post_norm = nn.LayerNorm(hidden)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        Z_in = decode_Z_or_idx(Z, want='Z')
        self._setup_runtime(condition, self.num_layers)
        
        # 1. PaiNN Input Preprocessing (Edges & Directions)
        from spectra.models.backbones.painn_backbone import _build_edges
        edge_index = _build_edges(pos, node_batch, cutoff=self.encoder.cutoff)
        idx_i, idx_j = edge_index[0], edge_index[1]
        r_ij = pos.index_select(0, idx_i) - pos.index_select(0, idx_j)
        dist = r_ij.norm(dim=-1, keepdim=True)
        unit_ij = r_ij / (dist + 1e-8)

        rbf = self.encoder.painn.radial_basis(dist)
        
        # Generation of filter list (mirroring PaiNN.forward)
        filters = self.encoder.painn.filter_net(rbf)
        if self.encoder.painn.cutoff_fn is not None:
            fcut = self.encoder.painn.cutoff_fn(dist).view(-1, 1)
            filters = filters * fcut
        filters = self.encoder.painn.apply_geom_gate(filters, pos, edge_index, node_batch)
        filters = torch.nan_to_num(filters)

        if self.encoder.painn.share_filters:
            filter_list = [filters] * self.num_layers
        else:
            filter_list = torch.split(filters, 3 * self.encoder.painn.n_atom_basis, dim=-1)
        
        # 2. Embedding
        s = self.encoder.painn.embedding(Z_in)
        v = torch.zeros(s.size(0), 3, s.size(1), device=s.device, dtype=s.dtype)

        # 3. Context Initialization
        z_s_node = condition.z_s
        attn_weights = None
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        if self.query_layer_index == 0 and cross_attn_enabled:
            z_s_node, attn_weights = self._get_solvent_context(s, node_batch, condition)

        # 4. Interactions (Unrolled)
        # PaiNN uses interactions and mixing blocks in pairs
        n_atoms = int(s.size(0))
        q = s.unsqueeze(1) # [N, 1, H] for PaiNN's internal format
        global_state = None
        
        for i, (interaction, mixing) in enumerate(zip(self.encoder.painn.interactions, self.encoder.painn.mixing), 1):
            q, v = interaction(q, v, filter_list[i - 1], unit_ij, idx_i, idx_j, n_atoms)
            q, v = mixing(q, v)

            if self.use_nonlocal and self.nonlocal_block is not None:
                q_flat = q.squeeze(1)
                if self.nonlocal_pre_norm is not None:
                    q_flat = self.nonlocal_pre_norm(q_flat)
                q_flat = q_flat + self.nonlocal_scale * self.nonlocal_block(q_flat, node_batch)
                if self.nonlocal_post_norm is not None:
                    q_flat = self.nonlocal_post_norm(q_flat)
                q = q_flat.unsqueeze(1)

            if self.encoder.painn.use_virtual_node and node_batch is not None:
                q_flat = q.squeeze(1)
                num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
                if global_state is None:
                    global_state = torch.zeros(
                        num_graphs,
                        self.encoder.painn.n_atom_basis,
                        device=q.device,
                        dtype=q.dtype,
                    )

                batch_sum = torch.zeros_like(global_state)
                batch_sum.index_add_(0, node_batch, q_flat)

                batch_count = torch.zeros(num_graphs, device=q.device, dtype=q.dtype)
                batch_count.index_add_(0, node_batch, torch.ones_like(node_batch, dtype=q.dtype))
                batch_mean = batch_sum / batch_count.clamp(min=1).unsqueeze(-1)

                update_in = self.encoder.painn.global_norm(global_state + batch_mean)
                global_delta = self.encoder.painn.global_mlp(update_in)
                global_state = global_state + global_delta

                q_update = global_state[node_batch].unsqueeze(1)
                q = q + self.virtual_node_scale * q_update
            
            if i == self.query_layer_index and cross_attn_enabled:
                z_s_node, attn_weights = self._get_solvent_context(q.squeeze(1), node_batch, condition)
            
            if self._should_inject(i):
                # Apply scalar FiLM
                q = q.squeeze(1) # [N, H]
                q = self._apply_node_film(q, z_s_node, node_batch)
                q = q.unsqueeze(1) # Restore [N, 1, H]
                
                # Vector gate if enabled
                if self.gbm.use_vector_gate:
                    v = self.gbm.vector_gate_apply(v.transpose(1, 2), z_s_node, node_batch).transpose(1, 2)
                self._mark_injected(i)

        # 5. Output
        s = q.squeeze(1)
        if not self._film_injected_layers:
            s = self._apply_node_film(s, z_s_node, node_batch)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(s, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=s, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": self._film_injected_layers, "attn_weights": attn_weights, 
                 "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


# ---------------- LEFTNet (Standard) ----------------

class LEFTNet_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        hidden: int = 256,
        num_layers: int = 4,
        num_radial: int = 32,
        cutoff: float = 5.0,
        use_scalar_film: bool = True,
        query_layer_index: int = 1,
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
        super().__init__(
            z_dim=z_dim, hidden_dim=hidden, use_scalar_film=use_scalar_film
        )
        self.encoder = LEFTNetBackbone(
            hidden_channels=hidden, num_layers=num_layers, num_radial=num_radial, cutoff=cutoff,
            use_lse=use_lse, use_fte=use_fte, use_vector_features=use_vector_features, dropout=dropout,
            use_geom_gate=use_geom_gate,
            geom_n_rbf=geom_n_rbf,
            geom_use_moments=geom_use_moments,
            geom_use_global=geom_use_global,
            geom_gate_scale=geom_gate_scale,
            geom_use_ln=geom_use_ln,
            use_nonlocal=use_nonlocal,
            nonlocal_heads=nonlocal_heads,
            nonlocal_features=nonlocal_features,
            nonlocal_dropout=nonlocal_dropout,
            nonlocal_scale=nonlocal_scale,
        )
        self.query_layer_index = int(query_layer_index)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        Z_in = decode_Z_or_idx(Z, want='Z').long()
        self._setup_runtime(condition, getattr(self.encoder, "num_layers", 4))
        
        # 1. Initial Embedding
        h = self.encoder.z_emb(Z_in)
        
        # 2. Context Calculation (Late Querying Lite)
        # Even if we don't fully unroll, we can at least query based on h_in (layer 0)
        # or find a way to update it.
        z_s_node = condition.z_s
        attn_weights = None
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        if cross_attn_enabled:
            # We use h (embedding) as query
            z_s_node, attn_weights = self._get_solvent_context(h, node_batch, condition)

        deep_film_flag = False
        if condition.film:
            deep_film_flag = condition.film.deep or (condition.film.layers is not None)

        H = self.encoder(
            Z_in, pos, node_batch,
            z_s=z_s_node, 
            gbm_conditioner=self.gbm,
            deep_film=deep_film_flag,
            film_every=condition.film.every if condition.film else 1,
            film_layers=condition.film.layers if condition.film else None,
            film_beta_only=condition.film.beta_only if condition.film else False,
            film_scale=condition.film.scale if condition.film else 1.0,
        )

        if not deep_film_flag:
            H = self._apply_node_film(H, z_s_node, node_batch)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(H, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=H, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": getattr(self.encoder, "_film_injected_layers", []), 
                 "attn_weights": attn_weights, "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


# ---------------- Equiformer V1 ----------------

class Equiformer_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        variant: str = "gat",
        max_radius: float = 5.0,
        max_neighbors: int = 1000,
        num_layers: int = 6,
        number_of_basis: int = 128,
        irreps_node_embedding: str = "128x0e+64x1e+32x2e",
        irreps_feature: str = "512x0e",
        irreps_head: str = "32x0e+16x1o+8x2e",
        norm_layer: str = "layer",
        num_heads: int = 4,
        basis_type: str = "gaussian",
        fc_neurons: Optional[list[int]] = None,
        irreps_mlp_mid: str = "128x0e+64x1e+32x2e",
        class_map: Optional[object] = "12",
        max_atom_type: Optional[int] = None,
        query_layer_index: int = 1,
        use_scalar_film: bool = True,
        use_geom_gate: bool = False,
        geom_n_rbf: int = 20,
        geom_use_moments: bool = True,
        geom_use_global: bool = True,
        geom_gate_scale: float = 0.1,
        geom_use_ln: bool = True,
        use_nonlocal: bool = False,
        nonlocal_heads: int = 4,
        nonlocal_features: int = 64,
        nonlocal_dropout: float = 0.0,
        nonlocal_scale: float = 0.1,
        **kwargs
    ):
        if o3 is None:
            raise ImportError("e3nn is required for Equiformer v1 adapter.")

        class_map_list = parse_class_map(class_map) or [int(x) for x in ELEMENTS12_Z.tolist()]
        self.class_map = class_map_list
        self.max_atom_type = int(max_atom_type or len(class_map_list))

        def _build_l0_indices(irreps_obj: o3.Irreps) -> list[int]:
            idxs: list[int] = []
            offset = 0
            for mul, ir in irreps_obj:
                dim = ir.dim
                if ir.l == 0:
                    for mi in range(mul):
                        idxs.append(offset + mi * dim)
                offset += mul * dim
            return idxs

        irreps_embed = o3.Irreps(irreps_node_embedding)
        irreps_feat = o3.Irreps(irreps_feature)
        l0_embed = _build_l0_indices(irreps_embed)
        l0_feat = _build_l0_indices(irreps_feat)
        if not l0_feat:
            raise ValueError("Equiformer v1 requires at least one l=0 channel in irreps_feature.")

        hidden = len(l0_feat)
        self._query_dim_embed = len(l0_embed)

        super().__init__(
            z_dim=z_dim, hidden_dim=hidden, use_scalar_film=use_scalar_film
        )

        model_cls = GraphAttentionTransformer if variant == "gat" else DotProductAttentionTransformer
        self.model = model_cls(
            irreps_in=f"{self.max_atom_type}x0e",
            irreps_node_embedding=irreps_node_embedding,
            num_layers=num_layers,
            irreps_node_attr="1x0e",
            irreps_sh=None,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            basis_type=basis_type,
            fc_neurons=fc_neurons or [64, 64],
            irreps_feature=irreps_feature,
            irreps_head=irreps_head,
            num_heads=num_heads,
            irreps_pre_attn=None,
            rescale_degree=False,
            nonlinear_message=False,
            irreps_mlp_mid=irreps_mlp_mid,
            norm_layer=norm_layer,
            alpha_drop=kwargs.get("alpha_drop", 0.2),
            proj_drop=kwargs.get("proj_drop", 0.0),
            out_drop=kwargs.get("out_drop", 0.0),
            drop_path_rate=kwargs.get("drop_path_rate", 0.0),
            max_atom_type=self.max_atom_type,
            class_map=None,
            node_atom_is_index=True,
        )
        self.max_radius = float(max_radius)
        self.max_neighbors = int(max_neighbors)
        self.num_layers = int(num_layers)
        self.query_layer_index = int(query_layer_index)
        self.use_geom_gate = bool(use_geom_gate)
        self.geom_gate_scale = float(geom_gate_scale)
        self.geom_gate_cache = None
        self.geom_gate_norm = None
        self.geom_gate_proj = None
        if self.use_geom_gate:
            geom_dim = int(geom_n_rbf)
            if geom_use_moments:
                geom_dim += 5
            if geom_use_global:
                geom_dim += 2
            self.geom_gate_cache = EdgeGeomCache(
                cutoff=self.max_radius,
                n_rbf=geom_n_rbf,
                tanh_clip=True,
                use_moments=geom_use_moments,
                use_global=geom_use_global,
            )
            if geom_use_ln:
                self.geom_gate_norm = nn.LayerNorm(geom_dim)
            self.geom_gate_proj = nn.Sequential(
                nn.Linear(geom_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )

        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)
        self.nonlocal_blocks = nn.ModuleDict()
        self.nonlocal_pre_norms = nn.ModuleDict()
        self.nonlocal_post_norms = nn.ModuleDict()
        if self.use_nonlocal:
            for dim in sorted({len(l0_embed), len(l0_feat)}):
                if dim <= 0:
                    continue
                key = str(dim)
                self.nonlocal_blocks[key] = PerformerNonLocal(
                    hidden_dim=dim,
                    num_heads=nonlocal_heads,
                    num_features=nonlocal_features,
                    dropout=nonlocal_dropout,
                )
                self.nonlocal_pre_norms[key] = nn.LayerNorm(dim)
                self.nonlocal_post_norms[key] = nn.LayerNorm(dim)
        self.register_buffer("_l0_indices_embed", torch.as_tensor(l0_embed, dtype=torch.long))
        self.register_buffer("_l0_indices_feat", torch.as_tensor(l0_feat, dtype=torch.long))
        self._embed_dim = irreps_embed.dim
        self._feat_dim = irreps_feat.dim
        self.query_proj = None
        if self._query_dim_embed and self._query_dim_embed != hidden:
            self.query_proj = nn.Linear(self._query_dim_embed, hidden)

        max_z = max(self.class_map) if self.class_map else 0
        z_to_idx = torch.full((max_z + 1,), -1, dtype=torch.long)
        for i, z in enumerate(self.class_map):
            z_to_idx[z] = i
        self.register_buffer("_z_to_idx", z_to_idx)

    def _map_z_to_idx(self, Z: torch.Tensor) -> torch.Tensor:
        Z_in = Z if Z.dim() == 1 else Z[:, 0]
        Z_in = Z_in.long()
        if Z_in.numel() == 0:
            return Z_in
        if Z_in.max() >= self._z_to_idx.numel():
            raise ValueError("solute Z contains elements outside class_map range.")
        idx = self._z_to_idx[Z_in]
        if (idx < 0).any():
            raise ValueError("solute Z contains elements not in class_map.")
        return idx

    def _select_l0_indices(self, node_features: torch.Tensor) -> torch.Tensor:
        dim = int(node_features.size(1))
        if dim == self._feat_dim:
            return self._l0_indices_feat
        if dim == self._embed_dim:
            return self._l0_indices_embed
        if int(self._l0_indices_feat.max().item()) < dim:
            return self._l0_indices_feat
        if int(self._l0_indices_embed.max().item()) < dim:
            return self._l0_indices_embed
        # Fallback: clamp to available range
        return self._l0_indices_embed[self._l0_indices_embed < dim]

    def _project_query(self, h_query: torch.Tensor) -> torch.Tensor:
        if h_query.size(1) == self.hidden_dim:
            return h_query
        if self.query_proj is None:
            raise ValueError("Equiformer v1 query dim mismatch; no projection available.")
        if h_query.size(1) != self._query_dim_embed:
            raise ValueError("Equiformer v1 query dim mismatch for projection.")
        return self.query_proj(h_query)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        self._setup_runtime(condition, self.num_layers)

        idx = self._map_z_to_idx(Z)

        if edge_index is None:
            edge_index = _radius_graph(
                pos, r=self.max_radius, batch=node_batch, max_num_neighbors=self.max_neighbors
            )
        edge_src, edge_dst = edge_index
        if edge_src.numel() > 0:
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
            edge_sh = o3.spherical_harmonics(
                l=self.model.irreps_edge_attr, x=edge_vec, normalize=True, normalization="component"
            )
            edge_len = edge_vec.norm(dim=1)
        else:
            edge_vec = pos.new_zeros((0, 3))
            edge_sh = pos.new_zeros((0, self.model.irreps_edge_attr.dim))
            edge_len = pos.new_zeros((0,))

        edge_length_embedding = self.model.rbf(edge_len)
        if self.use_geom_gate and self.geom_gate_cache is not None and self.geom_gate_proj is not None:
            geom_attr = self.geom_gate_cache(pos, edge_index, batch=node_batch)
            if self.geom_gate_norm is not None:
                geom_attr = self.geom_gate_norm(geom_attr)
            geom_gate = self.geom_gate_proj(geom_attr).to(dtype=edge_length_embedding.dtype)
            edge_length_embedding = edge_length_embedding * (1.0 + self.geom_gate_scale * geom_gate)
        atom_embedding, atom_attr, atom_onehot = self.model.atom_embed(idx)
        edge_degree_embedding = self.model.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, node_batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        z_s_node = condition.z_s
        attn_weights = None
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        if self.query_layer_index == 0 and cross_attn_enabled:
            l0_idx = self._select_l0_indices(node_features)
            h_query = node_features.index_select(1, l0_idx)
            h_query = self._project_query(h_query)
            z_s_node, attn_weights = self._get_solvent_context(h_query, node_batch, condition)

        for li, blk in enumerate(self.model.blocks, 1):
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=node_batch,
            )

            if self.use_nonlocal and self.nonlocal_blocks:
                l0_idx = self._select_l0_indices(node_features)
                l0 = node_features.index_select(1, l0_idx)
                key = str(l0.size(1))
                if key in self.nonlocal_blocks:
                    l0 = self.nonlocal_pre_norms[key](l0)
                    l0 = l0 + self.nonlocal_scale * self.nonlocal_blocks[key](l0, node_batch)
                    l0 = self.nonlocal_post_norms[key](l0)
                    node_features[:, l0_idx] = l0

            if li == self.query_layer_index and cross_attn_enabled:
                l0_idx = self._select_l0_indices(node_features)
                h_query = node_features.index_select(1, l0_idx)
                h_query = self._project_query(h_query)
                z_s_node, attn_weights = self._get_solvent_context(h_query, node_batch, condition)

            if self._should_inject(li):
                l0_idx = self._select_l0_indices(node_features)
                if l0_idx.numel() == self.hidden_dim:
                    l0 = node_features.index_select(1, l0_idx)
                    l0_mod = self._apply_node_film(l0, z_s_node, node_batch)
                    node_features[:, l0_idx] = l0_mod
                self._mark_injected(li)

        node_features = self.model.norm(node_features, batch=node_batch)
        if self.model.out_dropout is not None:
            node_features = self.model.out_dropout(node_features)

        l0_idx = self._select_l0_indices(node_features)
        H = node_features.index_select(1, l0_idx)
        if not self._film_injected_layers:
            if H.size(1) == self.hidden_dim:
                H = self._apply_node_film(H, z_s_node, node_batch)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(H, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=H, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": self._film_injected_layers, "attn_weights": attn_weights, "z_s_node": z_s_node}
        )


# ---------------- Equiformer V2 (Standard) ----------------

class EquiformerV2_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        cutoff: float = 5.0,
        max_num_elements: int = 12,
        num_layers: int = 12,
        sphere_channels: int = 128,
        use_scalar_film: bool = True,
        query_layer_index: int = 1,
        use_nonlocal: bool = False,
        nonlocal_heads: int = 4,
        nonlocal_features: int = 64,
        nonlocal_dropout: float = 0.0,
        nonlocal_scale: float = 0.1,
        **kwargs
    ):
        lmax_list = kwargs.get('lmax_list', [4]) 
        num_resolutions = len(lmax_list)
        hidden = int(sphere_channels) * num_resolutions
        
        super().__init__(
            z_dim=z_dim, hidden_dim=hidden, use_scalar_film=use_scalar_film
        )
        
        if EquiformerV2_OC20 is None: raise ImportError("EquiformerV2 modules not found.")
        
        self.model = EquiformerV2_OC20(
            num_layers=num_layers, max_radius=cutoff, max_num_elements=max_num_elements,
            sphere_channels=sphere_channels, **kwargs
        )
        self.cutoff = float(cutoff)
        self.max_num_elements = int(max_num_elements)
        self.num_resolutions = num_resolutions
        self.sphere_channels = int(sphere_channels)
        self.num_layers = num_layers
        self.query_layer_index = int(query_layer_index)
        self.use_nonlocal = bool(use_nonlocal)
        self.nonlocal_scale = float(nonlocal_scale)
        self.nonlocal_block = None
        self.nonlocal_pre_norm = None
        self.nonlocal_post_norm = None
        if self.use_nonlocal:
            hidden_dim = self.sphere_channels * self.num_resolutions
            self.nonlocal_block = PerformerNonLocal(
                hidden_dim=hidden_dim,
                num_heads=nonlocal_heads,
                num_features=nonlocal_features,
                dropout=nonlocal_dropout,
            )
            self.nonlocal_pre_norm = nn.LayerNorm(hidden_dim)
            self.nonlocal_post_norm = nn.LayerNorm(hidden_dim)

        self._l0_indices = []
        offset_res = 0
        for lmax in self.model.lmax_list:
            self._l0_indices.append(offset_res)
            offset_res += int((lmax + 1) ** 2)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        self._setup_runtime(condition, self.num_layers)
        
        idx = decode_Z_or_idx(Z, want='idx', max_num_elements=self.max_num_elements)
        
        # Initial Query feature (Input Embedding)
        h_query = self.model.sphere_embedding(idx) 
        if self.num_resolutions > 1:
            h_query = h_query.repeat(1, self.num_resolutions) 

        # Context Initialization
        z_s_node = condition.z_s
        attn_weights = None
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        if self.query_layer_index == 0 and cross_attn_enabled:
            z_s_node, attn_weights = self._get_solvent_context(h_query, node_batch, condition)

        device = pos.device
        dtype = pos.dtype
        N = pos.size(0)

        if edge_index is None:
            edge_index = _radius_graph(pos, r=self.cutoff, batch=node_batch, max_num_neighbors=self.model.max_neighbors)
        
        row, col = edge_index
        if row.numel() > 0:
            edge_vec = pos[row] - pos[col]
            edge_len = torch.linalg.norm(edge_vec, dim=1)
            mask = edge_len > 1e-5
            if not mask.all():
                row, col = row[mask], col[mask]
                edge_vec, edge_len = edge_vec[mask], edge_len[mask]
                edge_index = torch.stack([row, col])
        else:
             edge_vec = torch.zeros(0, 3, device=device, dtype=dtype)
             edge_len = torch.zeros(0, device=device, dtype=dtype)

        edge_rot_mat = init_edge_rot_mat(edge_vec)
        for i in range(self.num_resolutions):
            self.model.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = SO3_Embedding(N, self.model.lmax_list, self.model.sphere_channels, device=device, dtype=dtype)
        
        offset_res = 0
        if self.num_resolutions == 1:
            x.embedding[:, offset_res, :] = self.model.sphere_embedding(idx)
        else:
            offset = 0
            for i, lmax in enumerate(self.model.lmax_list):
                x.embedding[:, offset_res, :] = self.model.sphere_embedding(idx)[:, offset: offset + self.model.sphere_channels]
                offset += self.model.sphere_channels
                offset_res += int((lmax + 1) ** 2)

        edge_scalar = self.model.distance_expansion(edge_len)
        if self.model.share_atom_edge_embedding and self.model.use_atom_edge_embedding:
            src_embed = self.model.source_embedding(idx[row])
            dst_embed = self.model.target_embedding(idx[col])
            edge_scalar = torch.cat([edge_scalar, src_embed, dst_embed], dim=1)

        edge_degree = self.model.edge_degree_embedding(idx, edge_scalar, edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        for li, blk in enumerate(self.model.blocks, 1):
            x = blk(x, idx, edge_scalar, edge_index, batch=node_batch)
            if self.use_nonlocal and self.nonlocal_block is not None:
                l0 = x.embedding[:, self._l0_indices, :]
                l0_flat = l0.reshape(N, -1)
                if self.nonlocal_pre_norm is not None:
                    l0_flat = self.nonlocal_pre_norm(l0_flat)
                l0_flat = l0_flat + self.nonlocal_scale * self.nonlocal_block(l0_flat, node_batch)
                if self.nonlocal_post_norm is not None:
                    l0_flat = self.nonlocal_post_norm(l0_flat)
                x.embedding[:, self._l0_indices, :] = l0_flat.view(
                    N, self.num_resolutions, self.sphere_channels
                )
            
            # Late Querying update
            if li == self.query_layer_index and cross_attn_enabled:
                # Extract L=0 features for querying
                l0 = x.embedding[:, self._l0_indices, :] # [N, res, C]
                h_l0 = l0.reshape(N, -1)
                z_s_node, attn_weights = self._get_solvent_context(h_l0, node_batch, condition)

            if self._should_inject(li):
                l0 = x.embedding[:, self._l0_indices, :]
                l0_flat = l0.reshape(N, -1)
                l0_mod = self._apply_node_film(l0_flat, z_s_node, node_batch)
                x.embedding[:, self._l0_indices, :] = l0_mod.view(N, self.num_resolutions, self.sphere_channels)
                self._mark_injected(li)

        x.embedding = self.model.norm(x.embedding)
        l0 = x.embedding[:, self._l0_indices, :]
        H = l0.reshape(N, -1)
        
        if not self._film_injected_layers:
             H = self._apply_node_film(H, z_s_node, node_batch)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(H, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=H, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": self._film_injected_layers, "attn_weights": attn_weights, 
                 "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


        return BackboneOutput(
            node_scalar=H, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": self._film_injected_layers, "attn_weights": attn_weights, 
                 "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


# ---------------- C-GNN (Clifford GNN) ----------------

class CGNN_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        hidden: int = 128,
        num_layers: int = 4,
        use_scalar_film: bool = True,
        query_layer_index: int = 1,
    ):
        super().__init__(
            z_dim=z_dim, hidden_dim=hidden, use_scalar_film=use_scalar_film
        )
        self.model = CGNN(hidden_dim=hidden, num_layers=num_layers)
        self.query_layer_index = int(query_layer_index)

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        self._setup_runtime(condition, len(self.model.layers))
        
        Z_in = decode_Z_or_idx(Z, want='Z').long()
        
        node_feats = self.model(Z_in, pos, node_batch, edge_index)
        
        # CGNN exposes only final invariant features, so apply FiLM post-hoc.
        
        z_s_node = condition.z_s
        attn_weights = None
        if condition.solvent_x is not None:
             h_query_s = node_feats[:, :self.hidden_dim]
             z_s_node, attn_weights = self._get_solvent_context(h_query_s, node_batch, condition)
        
        gamma, beta = self.gbm.scalar_film(z_s_node) # [N, H]
        gamma = gamma.repeat(1, 4)
        beta = beta.repeat(1, 4)
        
        if self._film_strategy:
             if self._film_strategy.beta_only: gamma = torch.ones_like(gamma)
             scale = self._film_strategy.scale
             if scale != 1.0:
                 gamma = 1.0 + scale * (gamma - 1.0)
                 beta = scale * beta
                 
        node_feats = gamma * node_feats + beta
        
        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(node_feats, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=node_feats, coords=pos, graph_emb=graph_emb,
            aux={"attn_weights": attn_weights, "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


# ---------------- LEFTNet V3 ----------------

class LEFTNetV3_GBM_Adapter(BaseSolventAdapter):
    def __init__(
        self,
        z_dim: int,
        hidden: int = 256,
        num_layers: int = 4,
        num_radial: int = 50,
        cutoff: float = 5.0,
        use_scalar_film: bool = True,
        query_layer_index: int = 1,
        use_lse: bool = True,
        use_fte: bool = True,
        use_vector_features: bool = True,
        use_uvec: bool = True,
        max_num_neighbors: int = 48,
        cond_dim: int = None,
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
        super().__init__(
            z_dim=z_dim, hidden_dim=hidden, use_scalar_film=use_scalar_film
        )
        self.encoder = LEFTNetBackboneV3Cond(
            hidden_channels=hidden, num_layers=num_layers, num_radial=num_radial, cutoff=cutoff,
            use_lse=use_lse, use_fte=use_fte, use_vector_features=use_vector_features, use_uvec=use_uvec,
            max_num_neighbors=max_num_neighbors, dropout=dropout,
            use_geom_gate=use_geom_gate,
            geom_n_rbf=geom_n_rbf,
            geom_use_moments=geom_use_moments,
            geom_use_global=geom_use_global,
            geom_gate_scale=geom_gate_scale,
            geom_use_ln=geom_use_ln,
            use_nonlocal=use_nonlocal,
            nonlocal_heads=nonlocal_heads,
            nonlocal_features=nonlocal_features,
            nonlocal_dropout=nonlocal_dropout,
            nonlocal_scale=nonlocal_scale,
        )
        self.query_layer_index = int(query_layer_index)
        
        # Projection for conditioning dimension if requested
        self.cond_dim = cond_dim if cond_dim is not None else z_dim
        self.cond_proj = None
        if self.cond_dim != z_dim:
            self.cond_proj = nn.Linear(z_dim, self.cond_dim)
            # Re-initialize GBM with new z_dim
            # We keep other GBM settings from BaseSolventAdapter (use_scalar_film etc)
            self.gbm = GBMConditioner(
                z_dim=self.cond_dim,
                node_dim=hidden,
                edge_dim=0, # adapter doesn't use edge film currently for LeftNet
                use_scalar_film=use_scalar_film,
                use_edge_film=False,
                use_vector_gate=False, # LeftNet adapter didn't enable vector gate in Base init
            )

    def forward(self, Z, pos, node_batch, condition: ConditionContext, edge_index=None, edge_attr=None) -> BackboneOutput:
        Z_in = decode_Z_or_idx(Z, want='Z').long()
        self._setup_runtime(condition, getattr(self.encoder, "num_layers", 4))
        
        # 1. Initial Embedding
        h = self.encoder.z_emb(Z_in)
        
        # 2. Context Calculation
        # Selective Interaction uses original z_dim (matches solvent_x)
        z_s_node = condition.z_s
        attn_weights = None
        cross_attn_enabled = (condition.solvent_x is not None) and (condition.solvent_batch is not None)

        if cross_attn_enabled:
            z_s_node, attn_weights = self._get_solvent_context(h, node_batch, condition)
            
        # Apply projection if needed (z_dim -> cond_dim)
        if self.cond_proj is not None:
            z_s_node = self.cond_proj(z_s_node)

        deep_film_flag = False
        if condition.film:
            deep_film_flag = condition.film.deep or (condition.film.layers is not None)

        H = self.encoder(
            Z_in, pos, node_batch,
            z_s=z_s_node, 
            gbm_conditioner=self.gbm,
            deep_film=deep_film_flag,
            film_every=condition.film.every if condition.film else 1,
            film_layers=condition.film.layers if condition.film else None,
            film_beta_only=condition.film.beta_only if condition.film else False,
            film_scale=condition.film.scale if condition.film else 1.0,
        )

        if not deep_film_flag:
            H = self._apply_node_film(H, z_s_node, node_batch)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        graph_emb = unsorted_segment_mean(H, node_batch, num_segments=num_graphs)

        return BackboneOutput(
            node_scalar=H, coords=pos, graph_emb=graph_emb,
            aux={"film_layers": getattr(self.encoder, "_film_injected_layers", []), 
                 "attn_weights": attn_weights, "z_s_node": z_s_node, "solvent_h": condition.solvent_x}
        )


BACKBONE_REGISTRY = {
    "egnn": EGNN_GBM_Adapter,
    "schnet": SchNet_GBM_Adapter,
    "painn": PaiNN_GBM_Adapter,
    "leftnet": LEFTNet_GBM_Adapter,
    "leftnet_v3": LEFTNetV3_GBM_Adapter,
    "equiformer": Equiformer_GBM_Adapter,
    "equiformer_v2": EquiformerV2_GBM_Adapter,     
    "cgnn": CGNN_GBM_Adapter,
}

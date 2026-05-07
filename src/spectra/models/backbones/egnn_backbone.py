import torch
import torch.nn as nn
from typing import Literal
from .egnn_layers import E_GCL, unsorted_segment_mean

class EGNNBackbone(nn.Module):
    """EGNN encoder for node features and 3D coordinates."""
    
    def __init__(
        self, 
        node_in_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        edge_attr_dim: int = 1,
        coords_weight: float = 1.0,
        attention: bool = True,
        edge_attr_mode: Literal["invariant_only", "legacy"] = "legacy",
        use_fcvm: bool = False,
        chirality_invariant: bool = False,
        fcvm_update_frames: Literal["once", "per_layer"] = "once",
        clamp: bool = False,
        use_rbf: bool = False,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        log_fcvm: bool = False,
        log_config: dict = None,
        skip_node_embedding: bool = False,
        use_vector_features: bool = False,
        use_virtual_node: bool = False,
        use_advanced_geometry: bool = True,
        use_vector_mixing: bool = False,
        vector_mix_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_fcvm = use_fcvm
        self.use_vector_features = use_vector_features
        self.use_virtual_node = use_virtual_node
        self.use_advanced_geometry = use_advanced_geometry
        self.use_vector_mixing = bool(use_vector_mixing)
        self.vector_mix_scale = float(vector_mix_scale)
        self.dropout = float(dropout)
        # Kept for config compatibility; not used.
        
        if skip_node_embedding:
            self.node_embedding = nn.Identity()
        else:
            self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                E_GCL(
                    input_nf=hidden_dim,
                    output_nf=hidden_dim,
                    hidden_nf=hidden_dim,
                    edges_in_d=edge_attr_dim,
                    act_fn=nn.SiLU(),
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention,
                    edge_attr_mode=edge_attr_mode,
                    use_fcvm=use_fcvm,
                    chirality_invariant=chirality_invariant,
                    clamp=clamp,
                    use_rbf=use_rbf,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    log_fcvm=log_fcvm,
                    log_config=log_config,
                    use_vector_features=use_vector_features,
                    use_virtual_node=use_virtual_node,
                    use_advanced_geometry=use_advanced_geometry,
                    use_vector_mixing=self.use_vector_mixing,
                    vector_mix_scale=self.vector_mix_scale,
                    dropout=self.dropout,
                )
            )
        
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_feats, coords, edge_index, edge_attr=None, batch=None):
        h = self.node_embedding(node_feats)  # [N, hidden_dim]
        v = None
        global_h = None
        for layer in self.layers:
            h, coords, _, v, global_h = layer(
                h, edge_index, coords, edge_attr=edge_attr, fcvm_frames=None, v=v, 
                global_h=global_h, batch=batch
            )
        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            graph_emb = unsorted_segment_mean(h, batch, num_segments=num_graphs) # [B, hidden_dim]
        else:
            graph_emb = torch.mean(h, dim=0, keepdim=True)  # [1, hidden_dim]
            
        graph_emb = self.pool(graph_emb)  # [B, hidden_dim]
        
        return graph_emb, h, coords

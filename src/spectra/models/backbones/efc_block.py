import torch
import torch.nn as nn

from spectra.models.backbones.egnn_layers import unsorted_segment_sum


class EdgeFrameCoupledBlock(nn.Module):
    """Edge-frame coupled interaction block (scalar+vector update, O(E))."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        use_vector_features: bool = False,
        use_advanced_geometry: bool = True,
        use_vector_mixing: bool = False,
        vector_mix_scale: float = 1.0,
        use_edge_frame: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_vector_features = bool(use_vector_features)
        self.use_advanced_geometry = bool(use_advanced_geometry)
        self.use_vector_mixing = bool(use_vector_mixing)
        self.vector_mix_scale = float(vector_mix_scale)
        self.use_edge_frame = bool(use_edge_frame)

        vector_readout_dim = 0
        if self.use_vector_features:
            vector_readout_dim = 4
            if self.use_advanced_geometry:
                vector_readout_dim += 2

        in_dim = 2 * self.hidden_dim + int(edge_feat_dim) + vector_readout_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        vec_gate_out = self.hidden_dim * 3 if self.use_edge_frame else self.hidden_dim
        if self.use_vector_features:
            self.vec_gate_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, vec_gate_out),
            )
            self.vec_lin = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        else:
            self.vec_gate_mlp = None
            self.vec_lin = None

        if self.use_vector_features and self.use_vector_mixing:
            self.vec_mix_lin = nn.Linear(self.hidden_dim, 2 * self.hidden_dim, bias=False)
            self.mix_mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3 * self.hidden_dim),
            )
        else:
            self.vec_mix_lin = None
            self.mix_mlp = None

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.edge_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.node_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_frame: torch.Tensor = None,
    ):
        row, col = edge_index
        h_i = h[row]
        h_j = h[col]

        out_list = [h_i, h_j, edge_attr]

        vector_weights = None
        if self.use_vector_features and v is not None:
            dist = torch.norm(coord_diff, dim=-1, keepdim=True) + 1e-8
            u_ij = coord_diff / dist
            v_i = v[row]
            v_j = v[col]

            v_i_proj = (v_i * u_ij.unsqueeze(1)).sum(dim=-1)
            v_j_proj = (v_j * u_ij.unsqueeze(1)).sum(dim=-1)
            v_i_norm = torch.sqrt((v_i ** 2).sum(dim=-1) + 1e-8)
            v_j_norm = torch.sqrt((v_j ** 2).sum(dim=-1) + 1e-8)

            out_list.extend(
                [
                    v_i_proj.mean(dim=-1, keepdim=True),
                    v_j_proj.mean(dim=-1, keepdim=True),
                    v_i_norm.mean(dim=-1, keepdim=True),
                    v_j_norm.mean(dim=-1, keepdim=True),
                ]
            )

            if self.use_advanced_geometry:
                cross_ij = torch.cross(v_i, v_j, dim=-1)
                cross_norm = torch.sqrt((cross_ij ** 2).sum(dim=-1) + 1e-8)
                triple = (cross_ij * u_ij.unsqueeze(1)).sum(dim=-1)
                out_list.extend(
                    [
                        cross_norm.mean(dim=-1, keepdim=True),
                        triple.mean(dim=-1, keepdim=True),
                    ]
                )

        edge_feat = torch.cat(out_list, dim=-1)
        edge_feat = self.edge_dropout(self.edge_mlp(edge_feat))

        if self.use_vector_features and v is not None:
            vector_weights = self.vec_gate_mlp(edge_feat)

        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        h_update = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_update = self.node_dropout(h_update)

        h_out = h + h_update

        v_out = v
        if self.use_vector_features and v is not None and vector_weights is not None:
            dist = torch.norm(coord_diff, dim=-1, keepdim=True) + 1e-8
            u_ij = coord_diff / dist
            if self.use_edge_frame and edge_frame is not None:
                weights = vector_weights.view(vector_weights.size(0), self.hidden_dim, 3)
                edge_v = torch.einsum("ehk,ekd->ehd", weights, edge_frame)
            else:
                weights = vector_weights
                edge_v = weights.unsqueeze(-1) * u_ij.unsqueeze(1)

            delta_v = torch.zeros_like(v)
            delta_v.index_add_(0, row, edge_v)
            v_out = self.vec_lin(v.transpose(1, 2)).transpose(1, 2) + delta_v

        if self.use_vector_features and self.use_vector_mixing and v_out is not None:
            h_out, v_out = self._painn_mixing(h_out, v_out)

        return h_out, v_out

    def _painn_mixing(self, h: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # PaiNN-style scalar/vector coupling on per-node features.
        v_mix = self.vec_mix_lin(v.transpose(1, 2)).transpose(1, 2)
        v_v, v_w = torch.split(v_mix, self.hidden_dim, dim=1)
        v_vn = torch.sqrt((v_v ** 2).sum(dim=-1) + 1e-8)
        ctx = torch.cat([h, v_vn], dim=-1)
        x = self.mix_mlp(ctx)
        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.hidden_dim, dim=-1)
        dmu_intra = dmu_intra.unsqueeze(-1) * v_w
        dqmu_intra = dqmu_intra * (v_v * v_w).sum(dim=-1)
        h = h + self.vector_mix_scale * (dq_intra + dqmu_intra)
        v = v + self.vector_mix_scale * dmu_intra
        return h, v

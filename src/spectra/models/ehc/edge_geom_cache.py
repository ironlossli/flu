import math
import torch
import torch.nn as nn
from spectra.models.backbones.egnn_layers import compute_moment_invariants, unsorted_segment_mean

class LeftStyleRBFEmb(nn.Module):
    """
    LEFTNet-like RBF with cosine soft cutoff.
    Input: dist [E] or [E,1] (float)
    Output: rbf [E, num_rbf]
    """
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)

        # same init logic as LEFTNet: means/betas in exp(-d) space
        start = torch.exp(torch.tensor(-self.cutoff, dtype=torch.float32))
        end = torch.exp(torch.tensor(0.0, dtype=torch.float32))
        means = torch.linspace(start, end, self.num_rbf, dtype=torch.float32)
        betas = torch.tensor([(2 / self.num_rbf * (end - start)) ** -2] * self.num_rbf, dtype=torch.float32)

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.dim() == 2 and dist.size(-1) == 1:
            dist = dist.squeeze(-1)
        dist = dist.float().clamp_min(0.0).clamp_max(self.cutoff * 4.0)  # fp32 clamp

        # cosine soft cutoff
        d = dist.unsqueeze(-1)  # [E,1]
        soft = 0.5 * (torch.cos(d * math.pi / self.cutoff) + 1.0)
        soft = soft * (d < self.cutoff).float()

        # rbf in exp(-d) space
        x = torch.exp(-d)
        rbf = soft * torch.exp(-self.betas * (x - self.means).pow(2))
        return rbf


class EdgeGeomCache(nn.Module):
    """
    Compute once per forward:
      edge_attr_cached = concat([rbf(dist), moment_invariants, global_feats(2)])
    Output dtype matches coords dtype.
    """
    def __init__(self, cutoff: float = 5.0, n_rbf: int = 20, tanh_clip: bool = True, use_moments: bool = True, use_global: bool = True):
        super().__init__()
        self.cutoff = float(cutoff)
        self.n_rbf = int(n_rbf)
        self.rbf = LeftStyleRBFEmb(num_rbf=n_rbf, cutoff=cutoff)
        self.tanh_clip = bool(tanh_clip)
        self.use_moments = bool(use_moments)
        self.use_global = bool(use_global)

    @torch.no_grad()
    def _debug_check(self, edge_attr: torch.Tensor):
        # optional: you can add nan/inf checks here if needed
        return

    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        # compute in fp32 for stability
        coord32 = coords.float()
        row, col = edge_index
        diff = coord32[row] - coord32[col]
        dist = torch.norm(diff, dim=-1)  # [E]

        rbf = self.rbf(dist)  # [E, n_rbf]
        
        feats_list = [rbf]

        if self.use_moments:
            moments, _ = compute_moment_invariants(coord32, edge_index, cutoff=self.cutoff)  # [E,5]
            feats_list.append(moments)
        
        # --- Global Anchored Features ---
        if self.use_global and (batch is not None):
            # 1. Compute Batch Centers (Strict Batch Isolation)
            batch = batch.long()
            num_graphs = int(batch.max().item()) + 1
            centers = unsorted_segment_mean(coord32, batch, num_segments=num_graphs) # [B, 3]
            
            # 2. Global Vectors
            # vec_i_global: Vector from Graph Center to Atom i
            vec_global = coord32 - centers[batch] # [N, 3]
            vec_i_global = vec_global[row] # [E, 3]
            
            # 3. Robust Normalization
            dist_global = torch.norm(vec_i_global, dim=-1, keepdim=True)
            # Safe normalize: if dist is 0 (atom at center), dir is 0.
            vec_i_global_dir = vec_i_global / (dist_global + 1e-6)
            
            # 4. Edge Unit Vector
            u_ij = diff / (dist.unsqueeze(-1) + 1e-6) # [E, 3]
            
            # 5. Feature Extraction
            # F1: Radial Alignment (Dot)
            f_radial = torch.sum(u_ij * vec_i_global_dir, dim=-1, keepdim=True) # [E, 1]
            
            # F2: Radial Scale (Distance to centroid)
            f_radius = dist_global # [E, 1]
            
            feats_list.extend([f_radial, f_radius])

        # Concat: [RBF(20) + Moments(5)? + Global(2)?]
        edge_attr = torch.cat(feats_list, dim=-1)

        if self.tanh_clip:
            edge_attr = torch.tanh(edge_attr)

        edge_attr = edge_attr.to(dtype=coords.dtype)
        return edge_attr

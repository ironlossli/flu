import math
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from .algebra import CliffordAlgebra
from .layers import CliffordLinear, CliffordGatedActivation, CliffordInteraction, CliffordNorm

class GaussianRBF(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # use (num_gaussians-1) step to reduce edge bias
        step = (stop - start) / max(1, (num_gaussians - 1))
        self.coeff = -0.5 / (step ** 2 + 1e-12)
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [E] or [E,1]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist.pow(2))


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, dist):
        # dist: [E,1] float32
        x = dist / self.cutoff
        out = 0.5 * (torch.cos(math.pi * x).clamp(min=-1.0, max=1.0) + 1.0)
        return out * (x <= 1.0).to(dist.dtype)


class CGNN(nn.Module):
    """
    Stabilized CGNN for molecules.

    Key changes:
      1) smooth cosine cutoff
      2) fp32 geometry (dist/rbf/cutoff) for AMP safety
      3) bounded edge_scale to prevent geometric-product blow-up
      4) max_num_neighbors cap to control degree long-tail
      5) symmetric degree normalization + 1/sqrt(C) scaling in interaction
      6) optional per-graph position centering
    """
    def __init__(
        self,
        num_atom_types=100,
        hidden_dim=128,
        num_layers=4,
        cutoff=5.0,
        max_num_neighbors=64,
        rbf_num=50,
        edge_scale_max=2.0,
        center_pos=True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)
        self.edge_scale_max = float(edge_scale_max)
        self.center_pos = bool(center_pos)

        self.algebra = CliffordAlgebra(device=None)

        self.embedding = nn.Embedding(num_atom_types, self.hidden_dim)

        self.rbf = GaussianRBF(0.0, self.cutoff, rbf_num)
        self.cutoff_fn = CosineCutoff(self.cutoff)

        self.rbf_mlp = nn.Sequential(
            nn.Linear(rbf_num, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.layers = nn.ModuleList([
            CliffordInteraction(self.hidden_dim) for _ in range(self.num_layers)
        ])

        self.mixings = nn.ModuleList()
        for _ in range(self.num_layers):
            self.mixings.append(nn.ModuleList([
                CliffordNorm(self.hidden_dim),
                CliffordLinear(self.hidden_dim, self.hidden_dim),
                CliffordGatedActivation(self.hidden_dim),
                CliffordLinear(self.hidden_dim, self.hidden_dim),
            ]))

    def _center_positions(self, pos, batch):
        # subtract per-graph centroid
        sum_pos = scatter_add(pos, batch, dim=0)
        cnt = scatter_add(torch.ones_like(pos[:, :1]), batch, dim=0).clamp(min=1.0)
        mean = sum_pos / cnt
        return pos - mean[batch]

    def forward(self, Z, pos, batch, edge_index=None):
        if self.algebra.device != Z.device:
            self.algebra.to(Z.device)

        if self.center_pos:
            pos = self._center_positions(pos, batch)

        h_s = self.embedding(Z)  # [N, C]
        x = torch.zeros(Z.size(0), self.hidden_dim, 8, device=Z.device, dtype=h_s.dtype)
        x[..., 0] = h_s

        if edge_index is None:
            from torch_geometric.nn import radius_graph
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )

        row, col = edge_index

        # Geometry in fp32 for AMP stability
        pos32 = pos.float()
        diff32 = pos32[row] - pos32[col]  # [E,3]
        dist32 = torch.sqrt(torch.sum(diff32 * diff32, dim=-1, keepdim=True) + 1e-12)
        dist32 = dist32.clamp(min=1e-6, max=self.cutoff)

        dir32 = diff32 / dist32
        dir32 = torch.nan_to_num(dir32, nan=0.0, posinf=0.0, neginf=0.0)

        # Symmetric degree normalization
        ones = torch.ones(row.size(0), 1, device=row.device, dtype=torch.float32)
        deg = scatter_add(ones, row, dim=0, dim_size=x.size(0))  # [N,1]
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5).view(-1, 1, 1).to(x.dtype)

        # Smooth cutoff + RBF (fp32)
        rbf_feat32 = self.rbf(dist32.squeeze(-1))  # [E, rbf_num]
        edge_scale32 = self.rbf_mlp(rbf_feat32)    # [E, C]

        # Bound edge scale, then apply cutoff
        a = self.edge_scale_max
        edge_scale32 = a * torch.tanh(edge_scale32 / max(1e-6, a))  # [E,C]
        cutoff32 = self.cutoff_fn(dist32)  # [E,1]
        edge_scale32 = edge_scale32 * cutoff32
        edge_scale32 = torch.nan_to_num(edge_scale32, nan=0.0, posinf=0.0, neginf=0.0)

        # Edge vector: [E,C,3]
        edge_vector = (dir32.unsqueeze(1) * edge_scale32.unsqueeze(-1)).to(dtype=x.dtype)

        for layer, block in zip(self.layers, self.mixings):
            delta = layer(x, edge_index, edge_vector, self.algebra, deg_inv_sqrt=deg_inv_sqrt)
            x = x + delta

            h = x
            for module in block:
                h = module(h, self.algebra)
            x = x + h

        # Readout invariants
        s, v, b, t = self.algebra.split_grades(x)
        inv_s = s.squeeze(-1)
        inv_t = torch.sqrt(t.squeeze(-1) ** 2 + 1e-6)
        inv_v = torch.sqrt(torch.sum(v * v, dim=-1) + 1e-6)
        inv_b = torch.sqrt(torch.sum(b * b, dim=-1) + 1e-6)

        return torch.cat([inv_s, inv_v, inv_b, inv_t], dim=-1)

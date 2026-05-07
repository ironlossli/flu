import torch
import torch.nn as nn

from .edge_geom_cache import EdgeGeomCache


def stable_frame_from_dir(dir_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Deterministic orthonormal frame for each edge direction."""
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


class GeometryEnhancer(nn.Module):
    """Shared geometry features (rbf + moments + global anchors) with optional edge frames."""

    def __init__(
        self,
        cutoff: float = 5.0,
        n_rbf: int = 20,
        use_moments: bool = True,
        use_global: bool = True,
        use_edge_frame: bool = False,
        tanh_clip: bool = True,
    ):
        super().__init__()
        self.edge_cache = EdgeGeomCache(
            cutoff=cutoff,
            n_rbf=n_rbf,
            tanh_clip=tanh_clip,
            use_moments=use_moments,
            use_global=use_global,
        )
        self.use_edge_frame = bool(use_edge_frame)

    @property
    def n_rbf(self) -> int:
        return int(self.edge_cache.n_rbf)

    def forward(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        edge_attr = self.edge_cache(coords, edge_index, batch=batch)

        coord32 = coords.float()
        row, col = edge_index
        diff = coord32[row] - coord32[col]
        dist = torch.norm(diff, dim=-1, keepdim=True)
        unit = diff / (dist + 1e-6)

        edge_frame = None
        if self.use_edge_frame:
            edge_frame = stable_frame_from_dir(unit)

        return {
            "edge_attr": edge_attr,
            "coord_diff": diff.to(coords.dtype),
            "unit": unit.to(coords.dtype),
            "edge_frame": edge_frame,
        }

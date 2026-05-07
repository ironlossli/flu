import math
import torch
import torch.nn as nn


def _gaussian_orthogonal_random_matrix(num_rows: int, num_cols: int, device, dtype):
    block_list = []
    num_full_blocks = num_rows // num_cols
    for _ in range(num_full_blocks):
        q = torch.randn(num_cols, num_cols, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(q, mode="reduced")
        block_list.append(q.t())
    remaining_rows = num_rows - num_full_blocks * num_cols
    if remaining_rows > 0:
        q = torch.randn(num_cols, num_cols, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(q, mode="reduced")
        block_list.append(q.t()[:remaining_rows])
    return torch.cat(block_list, dim=0)


class PerformerNonLocal(nn.Module):
    """Linearized nonlocal self-attention (Performer/FAVOR+ style) for scalar features."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_features: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_features = int(num_features)
        self.eps = float(eps)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = self.hidden_dim // self.num_heads

        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        proj = _gaussian_orthogonal_random_matrix(
            self.num_features, self.head_dim, device=torch.device("cpu"), dtype=torch.float32
        )
        proj = proj.t()  # [head_dim, num_features]
        self.register_buffer(
            "proj",
            proj[None, ...].repeat(self.num_heads, 1, 1),
            persistent=False,
        )

        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def _softmax_kernel(self, data: torch.Tensor, is_query: bool) -> torch.Tensor:
        data = data.float() / math.sqrt(math.sqrt(self.head_dim))
        proj = self.proj.to(device=data.device, dtype=data.dtype)
        data_dash = torch.einsum("nhd,hdm->nhm", data, proj)
        diag = (data ** 2).sum(dim=-1, keepdim=True) * 0.5
        data_dash = data_dash - diag
        data_dash = data_dash - data_dash.max(dim=-1, keepdim=True).values
        data_dash = torch.clamp(data_dash, min=-15.0, max=15.0)
        data_dash = torch.exp(data_dash) + self.eps
        return data_dash

    def forward(self, x: torch.Tensor, node_batch: torch.Tensor = None) -> torch.Tensor:
        if x.numel() == 0:
            return x

        if node_batch is None:
            node_batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        orig_dtype = x.dtype
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        q_feat = self._softmax_kernel(q, is_query=True)
        k_feat = self._softmax_kernel(k, is_query=False)

        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        kv = torch.zeros(
            num_graphs,
            self.num_heads,
            self.num_features,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        k_sum = torch.zeros(
            num_graphs,
            self.num_heads,
            self.num_features,
            device=x.device,
            dtype=x.dtype,
        )

        kv.index_add_(0, node_batch, k_feat.unsqueeze(-1) * v.unsqueeze(2))
        k_sum.index_add_(0, node_batch, k_feat)

        kv_g = kv[node_batch]
        k_sum_g = k_sum[node_batch]

        out = torch.einsum("nhm,nhmd->nhd", q_feat, kv_g)
        denom = torch.einsum("nhm,nhm->nh", q_feat, k_sum_g).unsqueeze(-1)
        out = out / (denom + self.eps)

        out = out.reshape(-1, self.hidden_dim)
        out = self.to_out(out)
        out = self.dropout(out)
        return out.to(dtype=orig_dtype)

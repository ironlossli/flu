import math
import torch
import torch.nn as nn

class CliffordLinear(nn.Module):
    """
    Grade-wise linear transform that preserves O(3) structure:
    scalar / vector / bivector / pseudoscalar each has its own channel mixing.
    Input/Output: [N, C, 8]
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_s = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_b = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_t = nn.Linear(in_channels, out_channels, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, algebra):
        s, v, b, t = algebra.split_grades(x)

        # scalar / pseudoscalar: [N, in, 1] -> [N, out, 1]
        s_out = self.lin_s(s.squeeze(-1)).unsqueeze(-1)
        t_out = self.lin_t(t.squeeze(-1)).unsqueeze(-1)

        # vector / bivector: apply same channel-mixing to each of 3 components
        # v: [N, in, 3] -> [N, out, 3]
        v_in = v.permute(0, 2, 1).reshape(-1, self.in_channels)      # [N*3, in]
        v_out = self.lin_v(v_in).reshape(x.size(0), 3, self.out_channels).permute(0, 2, 1)

        b_in = b.permute(0, 2, 1).reshape(-1, self.in_channels)
        b_out = self.lin_b(b_in).reshape(x.size(0), 3, self.out_channels).permute(0, 2, 1)

        if self.bias is not None:
            s_out = s_out + self.bias.view(1, -1, 1)

        return algebra.cat_grades(s_out, v_out, b_out, t_out)


class CliffordNorm(nn.Module):
    """
    RMSNorm over the 8 basis components (per node, per channel).
    Keeps equivariance because scaling is invariant.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x, algebra):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class CliffordGatedActivation(nn.Module):
    """
    Invariant gating:
      - compute magnitudes per grade
      - a small MLP produces gates in (0,1)
      - apply gates to each grade; use SiLU for scalar & pseudoscalar pre-nonlinearity
    """
    def __init__(self, channels, hidden_factor=2, eps=1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps

        hidden = hidden_factor * 4 * channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * channels),
        )
        # Avoid early sigmoid saturation: initialize last layer to near-zero
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        self.act = nn.SiLU()

    def forward(self, x, algebra):
        s, v, b, t = algebra.split_grades(x)

        m_s = s.squeeze(-1)
        m_v = torch.sqrt(torch.sum(v * v, dim=-1) + self.eps)
        m_b = torch.sqrt(torch.sum(b * b, dim=-1) + self.eps)
        m_t = torch.sqrt(torch.sum(t * t, dim=-1) + self.eps)

        m_all = torch.cat([m_s, m_v, m_b, m_t], dim=-1)    # [N, 4C]
        gates = torch.sigmoid(self.gate_mlp(m_all))

        g_s, g_v, g_b, g_t = torch.split(gates, self.channels, dim=-1)

        s = self.act(s) * g_s.unsqueeze(-1)
        t = self.act(t) * g_t.unsqueeze(-1)
        v = v * g_v.unsqueeze(-1)
        b = b * g_b.unsqueeze(-1)

        return algebra.cat_grades(s, v, b, t)


class CliffordInteraction(nn.Module):
    """
    Message passing with Clifford geometric products.
    Stabilized with:
      - symmetric degree normalization
      - 1/sqrt(C) scaling to control variance
      - nan_to_num safety
    """
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.lin_a = CliffordLinear(channels, channels, bias=False)
        self.lin_b = CliffordLinear(channels, channels, bias=False)
        self.out_lin = CliffordLinear(channels, channels, bias=False)

        self.inv_sqrt_c = 1.0 / math.sqrt(max(1, channels))

    def forward(self, x, edge_index, edge_vector, algebra, deg_inv_sqrt=None):
        """
        x: [N, C, 8]
        edge_index: [2, E]
        edge_vector: [E, C, 3]
        deg_inv_sqrt: [N, 1, 1] with values 1/sqrt(deg)
        """
        row, col = edge_index

        x_j = x[col]  # source node features

        a = self.lin_a(x_j, algebra)
        b = self.lin_b(x_j, algebra)

        # M*E + E*M
        prod_1 = algebra.geometric_product_vector_edge(a, edge_vector)
        prod_2 = algebra.geometric_product_vector_edge_left(edge_vector, b)
        msg = (prod_1 + prod_2) * self.inv_sqrt_c

        if deg_inv_sqrt is not None:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E,1,1]
            msg = msg * norm

        msg = torch.nan_to_num(msg, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.zeros_like(x)
        out.index_add_(0, row, msg)

        out = self.out_lin(out, algebra)
        return out

# src/spectra/models/ehc/gbm_conditioning.py
import torch
import torch.nn as nn
from typing import Optional, Set

from .backbone_protocol import FiLMStrategy


class FiLMGenerator(nn.Module):
    """z_s -> (gamma, beta)"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
        init_std: float = 1e-2,
    ):
        super().__init__()
        self.g_net = nn.Linear(in_dim, out_dim)
        self.b_net = nn.Linear(in_dim, out_dim)

        nn.init.normal_(self.g_net.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.g_net.bias, init_gamma)
        nn.init.normal_(self.b_net.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.b_net.bias, init_beta)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, in_dim]
        return: gamma, beta [B, out_dim]
        """
        gamma = self.g_net(z)
        beta = self.b_net(z)
        return gamma, beta


class GBMConditioner(nn.Module):
    """
    将溶剂 embedding z_s 映射为：
      - scalar_film: γ_s, β_s  → Node scalar channels
      - edge_film:   γ_e, β_e  → Edge scalar features
      - vector_gate: gate_v    → Vector channels 的幅度门控

    只负责 “z_s → 参数”，**不负责** 决定在哪一层注入（交给 FiLMStrategy + compute_inject_set）。
    """

    def __init__(
        self,
        z_dim: int,
        node_dim: int,
        edge_dim: int = 0,
        use_scalar_film: bool = True,
        use_edge_film: bool = True,
        use_vector_gate: bool = False,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)

        # 与 _FiLMMixin 对齐的 flag 名
        self.use_scalar_film: bool = bool(use_scalar_film)
        self.use_edge_film: bool = bool(use_edge_film) and self.edge_dim > 0
        self.use_vector_gate: bool = bool(use_vector_gate)

        if self.use_scalar_film:
            self.scalar_film = FiLMGenerator(self.z_dim, self.node_dim)

        if self.use_edge_film:
            self.edge_film = FiLMGenerator(self.z_dim, self.edge_dim)

        if self.use_vector_gate:
            self.v_gate = nn.Sequential(
                nn.Linear(self.z_dim, self.node_dim),
                nn.SiLU(),
                nn.Linear(self.node_dim, self.node_dim),
                nn.Sigmoid(),
            )

    # --------- Node scalar film ---------

    def node_film(
        self,
        h: torch.Tensor,
        z_s: torch.Tensor,
        node_batch: torch.Tensor,
        film: Optional[FiLMStrategy] = None,
    ) -> torch.Tensor:
        """
        兼容旧接口的 fallback：当 _FiLMMixin 无法使用 scalar_film 时调用。
        h: [N, C], z_s: [B, z_dim] or [N, z_dim], node_batch: [N]
        """
        if (not self.use_scalar_film) or (h is None) or (z_s is None):
            return h

        gamma, beta = self.scalar_film(z_s)  # [B/N, C]

        if film is not None:
            if film.beta_only:
                gamma = torch.ones_like(gamma)
            if film.scale != 1.0:
                gamma = 1.0 + film.scale * (gamma - 1.0)
                beta = film.scale * beta

        # Auto-detect node-level
        is_node_level = (gamma.size(0) == h.size(0)) and (h.size(0) > 1)
        
        if is_node_level:
            g = gamma
            b = beta
        else:
            g = gamma[node_batch]  # [N, C]
            b = beta[node_batch]
            
        return g * h + b

    # --------- Edge scalar film ---------

    def edge_film_apply(
        self,
        edge_attr: torch.Tensor,
        z_s: torch.Tensor,
        edge_batch: torch.Tensor,
        film: Optional[FiLMStrategy] = None,
    ) -> torch.Tensor:
        """
        方便直接在骨干内部调用的边 FiLM 接口。
        edge_attr: [E, Ce], z_s: [B, z_dim], edge_batch: [E]
        """
        if (not self.use_edge_film) or (edge_attr is None) or (z_s is None):
            return edge_attr

        gamma, beta = self.edge_film(z_s)  # [B, Ce]

        if film is not None:
            if film.beta_only:
                gamma = torch.ones_like(gamma)
            if film.scale != 1.0:
                gamma = 1.0 + film.scale * (gamma - 1.0)
                beta = film.scale * beta

        g = gamma[edge_batch]
        b = beta[edge_batch]
        return g * edge_attr + b

    # --------- Vector gate ---------

    def vector_gate_apply(
        self,
        v: torch.Tensor,
        z_s: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        v: [N, C] 或 [N, C, 3] 的幅度门控（只改幅度，不改方向）。

        如果是 [N, C, 3]，相当于对每个通道整体做缩放：gate ∈ (0,1)。
        """
        if (not self.use_vector_gate) or (v is None) or (z_s is None):
            return v

        gate_raw = self.v_gate(z_s) # [B, C] or [N, C]
        
        # Auto-detect node-level
        # v size(0) is N. gate_raw size(0) is B or N.
        is_node_level = (gate_raw.size(0) == v.size(0)) and (v.size(0) > 1)
        
        if is_node_level:
            gate = gate_raw
        else:
            gate = gate_raw[node_batch]  # [N, C]
            
        if v.dim() == 2:
            return gate * v
        elif v.dim() == 3:
            return gate.unsqueeze(-1) * v
        return v


# === PATCH: 统一 FiLM 注入层决策 ===
from typing import Set

def compute_inject_set(num_layers: int, film: FiLMStrategy) -> Set[int]:
    """
    根据 FiLMStrategy 决定在哪些层注入 FiLM。
    层号使用 1-based（第 1 层 == 1），返回一个层号集合。

    规则：
      - 如果 film.layers 非空：优先使用 film.layers，并裁剪到 [1, num_layers]
      - 否则，如果 film.deep 为 True：
          * film.every > 0: 在 {every, 2*every, ...} ∩ [1, num_layers] 注入
          * film.every <= 0: 默认每一层都注入 {1..num_layers}
      - 如果 film.deep 为 False 且 film.layers 为空：不注入（返回空集）
    """
    if num_layers <= 0:
        return set()

    inject_layers: Set[int] = set()

    # 1. 优先使用 film.layers
    if film.layers is not None and len(film.layers) > 0:
        for l in film.layers:
            if 1 <= l <= num_layers:
                inject_layers.add(int(l))
        return inject_layers

    # 2. 其次使用 deep_film / film_every
    if film.deep:
        if film.every is not None and film.every > 0:
            l = film.every
            while l <= num_layers:
                inject_layers.add(int(l))
                l += film.every
        else:
            # every <= 0: 默认每层注入
            inject_layers = set(range(1, num_layers + 1))

    # 3. deep=False 且 layers 为空 → 空集
    return inject_layers

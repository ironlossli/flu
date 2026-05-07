# src/spectra/models/ehc/solvent_encoder.py
import torch
import torch.nn as nn
from typing import Optional

from spectra.models.backbones.egnn_layers import (
    unsorted_segment_sum,
    unsorted_segment_mean,
)


class SolventMPNN(nn.Module):
    """
    轻量 2D MPNN，用于对 solvent 图编码，输出样本级 embedding z_s。

    输入:
      x:            [Ns, node_in]
      edge_index:   [2, Es] (row, col)
      edge_attr:    [Es, edge_in] 或 None
      batch:        [Ns] solvent 图索引

    输出:
      z_s:          [B, hidden]，B = batch 中图的个数
    """

    def __init__(
        self,
        node_in: int = 14,
        edge_in: int = 7,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_in = node_in
        self.edge_in = edge_in
        self.hidden = hidden

        self.node_emb = nn.Linear(node_in, hidden)

        self.msg_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden + edge_in, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )

        self.upd_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden + hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:          [Ns, node_in]
        edge_index: [2, Es]
        edge_attr:  [Es, edge_in] 或 None
        batch:      [Ns]

        返回:
          z_s: [B, hidden]，按 batch mean-pool。
          h:   [Ns, hidden], 节点级特征
        """
        if x.numel() == 0:
            # 空 batch 情况
            return x.new_zeros(1, self.hidden), x.new_zeros(0, self.hidden)

        h = self.node_emb(x)  # [Ns, H]
        row, col = edge_index  # [Es]

        for msg_mlp, upd_mlp, ln in zip(self.msg_mlps, self.upd_mlps, self.norms):
            if edge_attr is None:
                # 无边特征时，用全 0 占位
                zeros = h.new_zeros(col.shape[0], self.edge_in)
                m_ij = msg_mlp(torch.cat([h[col], zeros], dim=-1))
            else:
                m_ij = msg_mlp(torch.cat([h[col], edge_attr], dim=-1))  # [Es, H]

            m_i = unsorted_segment_sum(m_ij, row, num_segments=h.size(0))  # [Ns, H]
            h_new = upd_mlp(torch.cat([h, m_i], dim=-1))
            h = ln(h + self.dropout(h_new))

        # 图级 mean-pool
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        z_s = unsorted_segment_mean(h, batch, num_segments=B)  # [B, H]
        
        return z_s, h

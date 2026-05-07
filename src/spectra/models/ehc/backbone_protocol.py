# src/spectra/models/ehc/backbone_protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol

import torch


@dataclass
class FiLMStrategy:
    """
    统一描述 FiLM 策略：
      - deep: 是否在多层注入 (deep_film)
      - every: 每隔多少层注入一次（当 layers 为空且 deep=True 时）
      - layers: 显式指定在哪几层注入（1-based），优先级高于 every
      - beta_only: 只用 beta（gamma=1）
      - scale: 对 gamma,beta 的缩放系数
    """
    deep: bool = False
    every: int = 1
    layers: Optional[list[int]] = None
    beta_only: bool = False
    scale: float = 1.0


@dataclass
class ConditionContext:
    """
    溶剂条件化的统一上下文。所有 backbone / adapter 理论上只看这个结构体：
      - z_s:           [B, z_dim]  溶剂 embedding (Graph-level)
      - z_s_node:      [N, z_dim]  溶剂 embedding (Node-level, optional)
      - solvent_x:     [M, D]      溶剂节点特征 (未池化)
      - solvent_batch: [M]         溶剂节点 Batch 索引
      - node_batch:    [N]         节点 -> 图 索引
      - gbm:           GBMConditioner（或兼容接口）
      - film:          FiLMStrategy
    """
    z_s: Optional[torch.Tensor] = None
    z_s_node: Optional[torch.Tensor] = None
    solvent_x: Optional[torch.Tensor] = None
    solvent_batch: Optional[torch.Tensor] = None
    node_batch: Optional[torch.Tensor] = None
    gbm: Optional[torch.nn.Module] = None
    film: Optional[FiLMStrategy] = None


@dataclass
class BackboneOutput:
    """
    EHC 视角下统一的 backbone 输出：
      - node_scalar:   [N, H_s]    标量节点特征（必需）
      - node_vector:   [N, 3, H_v] 或 None，等变向量特征（可选）
      - coords:        [N, 3]      当前坐标（可以是更新后的）
      - graph_emb:     [B, H_s]    图级 embedding（可选；否则用 pooling(node_scalar)）
      - aux:           任意附加信息（per-layer 特征、attention map、FiLM 记录等）
    """
    node_scalar: torch.Tensor
    coords: torch.Tensor
    node_vector: Optional[torch.Tensor] = None
    graph_emb: Optional[torch.Tensor] = None
    aux: Optional[Dict[str, Any]] = None


class BackboneProtocol(Protocol):
    """
    所有 EHC 中使用的 backbone / wrapper 应当实现的接口。
    注意：目前 Phase 1 可以先只在类型层面依赖，不强制所有已有 adapter 立刻实现。
    """

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        condition: Optional[ConditionContext] = None,
        **kwargs: Any,
    ) -> BackboneOutput:
        """
        统一的 forward 接口：

        参数:
          Z:      [N]  原子序数 / 类型索引
          pos:    [N, 3] 坐标
          batch:  [N]  图 ID
          condition: ConditionContext，描述溶剂条件化信息（可为 None）

        返回:
          BackboneOutput
        """
        ...

# === src/spectra/models/ehc/ehc_model.py ===
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .solvent_encoder import SolventMPNN
from .readout import SpectralFocusReadout
from spectra.models.backbones.egnn_layers import unsorted_segment_mean
from .backbone_protocol import ConditionContext, FiLMStrategy, BackboneOutput
from .backbone_adapter import BACKBONE_REGISTRY


@dataclass
class SolventConfig:
    node_in: int
    edge_in: int
    z_dim: int
    layers: int
    dropout: float = 0.1


@dataclass
class BackboneConfig:
    name: str
    adapter_kwargs: dict
    hidden_dim: int


@dataclass
class FiLMConfig:
    deep_film: bool
    film_every: int
    film_layers: Optional[list[int]]
    film_beta_only: bool
    film_scale: float


@dataclass
class FBConfig:
    enabled: bool = False  # Deprecated
    roles: int = 1
    d_k: int = 8
    d_v: int = 8
    fb_cutoff: Optional[float] = None
    use_updated_coords_for_fb: bool = False
    use_geom: bool = False


@dataclass
class HeadConfig:
    hidden_dim: int
    energy_space: bool
    use_backbone_readout: bool
    readout_type: str = "mean"  # "mean", "sum", "spectral_focus"


class PhysicsHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1, energy_space: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.energy_space = energy_space
        if energy_space:
            self.register_buffer("hc", torch.tensor(1239.84))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(h)
        if self.energy_space:
            E = torch.abs(y) + 0.1
            return self.hc / E
        return y


class EHCModel(nn.Module):
    """
    通用 EHC 顶层：Solvent z_s → Backbone(+GBM) → Decoder Head → y_pred
    """

    def __init__(
        self,
        solvent_cfg: SolventConfig,
        backbone_cfg: BackboneConfig,
        film_cfg: FiLMConfig,
        fb_cfg: FBConfig, # Kept for config compatibility but unused
        head_cfg: HeadConfig,
        use_concat_baseline: bool = False,
    ):
        super().__init__()

        self.solvent_cfg = solvent_cfg
        self.backbone_cfg = backbone_cfg
        self.film_cfg = film_cfg
        self.head_cfg = head_cfg
        self.use_concat_baseline = use_concat_baseline

        self.backbone_name = backbone_cfg.name.lower()
        self.use_backbone_readout = bool(head_cfg.use_backbone_readout)

        # 1) 溶剂编码器
        self.solvent_encoder = SolventMPNN(
            node_in=solvent_cfg.node_in,
            edge_in=solvent_cfg.edge_in,
            hidden=solvent_cfg.z_dim,
            layers=solvent_cfg.layers,
        )

        # 2) 主干 + GBM 适配器
        if self.backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone_name: {self.backbone_name}")
        backbone_cls = BACKBONE_REGISTRY[self.backbone_name]
        self.backbone = backbone_cls(**backbone_cfg.adapter_kwargs)
        self.hidden = int(backbone_cfg.hidden_dim)

        # 3) 预测头
        # 如果启用 Concat Baseline，GBM Head 的输入维度需要加上溶剂维度
        gbm_in_dim = self.hidden
        if self.use_concat_baseline:
            gbm_in_dim += solvent_cfg.z_dim

        self.head = PhysicsHead(
            in_dim=gbm_in_dim,
            hidden=head_cfg.hidden_dim,
            energy_space=head_cfg.energy_space,
        )

        # 4) Readout Module
        self.readout_type = getattr(head_cfg, "readout_type", "mean")
        self.readout_module = None
        if self.readout_type == "spectral_focus":
            self.readout_module = SpectralFocusReadout(in_channels=self.hidden)

    # === 统一前向 ===
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # --- 兼容 GraphBatch 和 dict ---
        def _get(obj, name: str):
            if hasattr(obj, name):
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            raise AttributeError(f"batch 缺少字段 '{name}'")

        # 1) 溶剂编码
        solvent_x = _get(batch, "solvent_x")
        solvent_edge_index = _get(batch, "solvent_edge_index")
        solvent_edge_attr = _get(batch, "solvent_edge_attr")
        solvent_batch = _get(batch, "solvent_batch")

        z_s, solvent_h = self.solvent_encoder(
            solvent_x,
            solvent_edge_index,
            solvent_edge_attr,
            solvent_batch,
        )  # z_s: [B, z_dim], solvent_h: [Ns, z_dim]

        # 2) 溶质图特征
        solute_x = _get(batch, "solute_x")
        solute_pos = _get(batch, "solute_pos")
        solute_edge_index = _get(batch, "solute_edge_index")
        solute_edge_attr = _get(batch, "solute_edge_attr")
        solute_batch = _get(batch, "solute_batch")

        # 3) 构造 ConditionContext
        # [Baseline Logic] 如果启用双塔基线，切断溶剂信息进入 Backbone 的路径
        if self.use_concat_baseline:
            z_s_for_backbone = torch.zeros_like(z_s)
            solvent_h_for_backbone = None
        else:
            z_s_for_backbone = z_s
            solvent_h_for_backbone = solvent_h

        film = FiLMStrategy(
            deep=False,
            every=self.film_cfg.film_every,
            layers=None,
            beta_only=self.film_cfg.film_beta_only,
            scale=self.film_cfg.film_scale,
        )
        condition = ConditionContext(
            z_s=z_s_for_backbone,
            z_s_node=None, # Leave empty, adapters will compute it via Cross-Attention
            solvent_x=solvent_h_for_backbone,
            solvent_batch=solvent_batch,
            node_batch=solute_batch,
            gbm=getattr(self.backbone, "gbm", None),
            film=film,
        )

        # 4) 调用 backbone
        backbone_out: BackboneOutput = self.backbone(
            Z=solute_x,
            pos=solute_pos,
            node_batch=solute_batch,
            condition=condition,
            edge_index=solute_edge_index,
            edge_attr=solute_edge_attr,
        )

        node_scalar = backbone_out.node_scalar
        coords = backbone_out.coords
        graph_emb = backbone_out.graph_emb
        backbone_aux = backbone_out.aux or {}

        # 补齐 graph_emb (防御性)
        if solute_batch.numel() > 0:
            num_graphs = int(solute_batch.max().item()) + 1
        else:
            num_graphs = 1

        if graph_emb is None:
            if self.readout_type == "spectral_focus" and self.readout_module is not None:
                graph_emb, attn_weights = self.readout_module(node_scalar, solute_batch, return_weights=True)
                backbone_aux["readout_attn"] = attn_weights
            else:
                graph_emb = unsorted_segment_mean(node_scalar, solute_batch, num_segments=num_graphs)

        # 5) 预测
        # 如果是 Late Fusion 基线模式，先拼接溶剂特征
        if self.use_concat_baseline:
            gbm_input = torch.cat([graph_emb, z_s], dim=-1)
            y_pred = self.head(gbm_input)
        else:
            y_pred = self.head(graph_emb)  # [B, 1]

        # 6) 打包输出
        out: Dict[str, Any] = {
            "pred": y_pred,
            "z_s": z_s,
            "graph_emb": graph_emb,
            "node_scalar": node_scalar,
            "coords": coords,
            "backbone_aux": backbone_aux,
            "edge_index": solute_edge_index,
        }

        return out

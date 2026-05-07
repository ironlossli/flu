# datamodule_extended.py
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from preprocessing import DatasetPreprocessor

logger = logging.getLogger(__name__)


def resolve_model_cutoff(model_config: Optional[dict]) -> float:
    if not model_config:
        raise ValueError("model_config is required to resolve cutoff.")
    candidates = (
        "cutoff",
        "egnn_cutoff",
        "painn_cutoff",
        "schnet_cutoff",
        "eqf_radius",
        "eq2_cutoff",
        "leftnet_cutoff",
        "leftnet_v3_cutoff",
        "cgnn_cutoff",
    )
    for key in candidates:
        if key in model_config and model_config[key] is not None:
            return float(model_config[key])
    raise ValueError(
        "Missing cutoff in model_config. Set one of: "
        + ", ".join(candidates)
    )


def _load_indices(idx_json: str) -> List[int]:
    """Read split indices from json; supports list or dict."""
    with open(idx_json, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [int(i) for i in data]
    if isinstance(data, dict):
        return [int(v) for v in data.values()]
    raise ValueError(f"Unsupported split json format: {idx_json}")


class GraphData:
    """Lightweight sample container with .to(device)."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


class GraphDataset(Dataset):
    """
    3D solute (XYZ+cutoff) + 2D solvent (RDKit) dataset.
    Constructed by DatasetPreprocessor, emits sample dicts wrapped by GraphData.
    """
    def __init__(self, parquet_path: str, idx_json: str, cache_dir: Optional[str] = None, cutoff_3d: Optional[float] = None):
        indices = _load_indices(idx_json)
        pre = DatasetPreprocessor(parquet_path, cutoff_3d=cutoff_3d)
        samples: List[Dict[str, Any]] = pre.preprocess_split(indices, cache_dir)
        self.samples = [GraphData(**s) for s in samples]
        logger.info(f"Loaded {len(self.samples)} samples for {Path(idx_json).name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class GraphBatch:
    """
    Collate B samples into one batch. Keeps solute and solvent graphs independent:
      - solute_* concatenated with node offsetting; builds solute_batch index
      - solvent_* concatenated with node offsetting; builds solvent_batch index
    y is selected from lambda_abs or lambda_em via `target`.
    """
    def __init__(self, data_list: List[GraphData], target: str):
        self.batch_size = len(data_list)
        self._target = target

        def pick(d: GraphData, names: List[str], required: bool = True) -> Optional[torch.Tensor]:
            for n in names:
                if hasattr(d, n):
                    return getattr(d, n)
            if required:
                raise AttributeError(f"Missing required fields: {names}")
            return None

        # ---- Solute concat ----
        solute_node_offset = 0
        solute_x_list, solute_pos_list, solute_eidx_list, solute_eattr_list, solute_batch_idx = [], [], [], [], []
        for bi, d in enumerate(data_list):
            x = pick(d, ["solute_x"])                               # [N, F=12]
            pos = pick(d, ["solute_pos"])                            # [N, 3]
            eidx = pick(d, ["solute_edge_index"])                    # [2, E]
            eattr = pick(d, ["solute_edge_attr"])                    # [E, 4]
            n = int(x.size(0))
            solute_x_list.append(x)
            solute_pos_list.append(pos)
            solute_eidx_list.append(eidx + solute_node_offset)
            solute_eattr_list.append(eattr)
            solute_batch_idx.extend([bi] * n)
            solute_node_offset += n

        if solute_x_list:
            self.solute_x = torch.cat(solute_x_list, 0)                      # [N, 12]
            self.solute_pos = torch.cat(solute_pos_list, 0)                  # [N, 3]
            self.solute_edge_index = torch.cat(solute_eidx_list, 1)          # [2, E]
            self.solute_edge_attr = torch.cat(solute_eattr_list, 0)          # [E, 4]
            self.solute_batch = torch.as_tensor(solute_batch_idx, dtype=torch.long)  # [N]
        else:
            # empty batch safety (unlikely)
            self.solute_x = torch.zeros((0, 12), dtype=torch.float32)
            self.solute_pos = torch.zeros((0, 3), dtype=torch.float32)
            self.solute_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.solute_edge_attr = torch.zeros((0, 4), dtype=torch.float32)
            self.solute_batch = torch.zeros((0,), dtype=torch.long)

        # ---- Solvent concat ----
        solvent_node_offset = 0
        solv_x_list, solv_eidx_list, solv_eattr_list, solv_batch_idx = [], [], [], []
        for bi, d in enumerate(data_list):
            x = pick(d, ["solvent_x"])                               # [Ns, 14]
            eidx = pick(d, ["solvent_edge_index"])                   # [2, Es]
            eattr = pick(d, ["solvent_edge_attr"])                   # [Es, 7]
            ns = int(x.size(0))
            solv_x_list.append(x)
            solv_eidx_list.append(eidx + solvent_node_offset)
            solv_eattr_list.append(eattr)
            solv_batch_idx.extend([bi] * ns)
            solvent_node_offset += ns

        if solv_x_list:
            self.solvent_x = torch.cat(solv_x_list, 0)                       # [Ns, 14]
            self.solvent_edge_index = torch.cat(solv_eidx_list, 1)           # [2, Es]
            self.solvent_edge_attr = torch.cat(solv_eattr_list, 0)           # [Es, 7]
            self.solvent_batch = torch.as_tensor(solv_batch_idx, dtype=torch.long)  # [Ns]
        else:
            self.solvent_x = torch.zeros((0, 14), dtype=torch.float32)
            self.solvent_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.solvent_edge_attr = torch.zeros((0, 7), dtype=torch.float32)
            self.solvent_batch = torch.zeros((0,), dtype=torch.long)

        # ---- Targets ----
        target_field = f"lambda_{target}"
        if not hasattr(data_list[0], target_field):
            raise AttributeError(f"样本缺少目标字段: '{target_field}'，请确认预处理是否包含该列")
        self.y = torch.as_tensor(
            [getattr(d, target_field) for d in data_list],
            dtype=torch.float32
        )  # [B]

        # ---- Metadata (optional) ----
        self.xyz_path = [getattr(d, "xyz_path", None) for d in data_list]
        self.solute_smiles = [getattr(d, "solute_smiles", None) for d in data_list]
        self.solvent_smiles = [getattr(d, "solvent_smiles", None) for d in data_list]

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def make_collate_graph(target: str) -> Callable[[List[GraphData]], GraphBatch]:
    def _collate(data_list: List[GraphData]) -> GraphBatch:
        return GraphBatch(data_list, target=target)
    return _collate


# --------- Path/split resolution helpers (flexible layouts) ----------
def _resolve_paths(data_config: dict, target: str, split_name: Optional[str]) -> Tuple[str, Path, Path]:
    """
    返回 parquet_path, splits_root, cache_base
    - parquet 路径优先 processed.{target}_parquet / processed.parquet
    - splits_root 固定为 <splits_dir>/<target>
    - split_name 允许为 None（上层默认成 'random'）
    """
    dc = data_config or {}
    proc = dc.get("processed", {})

    parquet = (
        proc.get("parquet")
        or proc.get(f"{target}_parquet")
        or dc.get("parquet")
        or dc.get("data", {}).get("parquet")
        or proc.get("base_parquet")
    )
    if not parquet:
        raise ValueError("data_config 缺少 parquet 路径: 期望 processed.parquet 或 processed.{target}_parquet")
    parquet_path = str(parquet)

    base_splits = (
        proc.get("splits_dir")
        or dc.get("splits_dir")
        or dc.get("data", {}).get("splits_dir")
    )
    if not base_splits:
        raise ValueError("data_config 缺少 splits_dir: 期望 processed.splits_dir")
    splits_root = Path(base_splits) / target  # new layout: <splits_dir>/<target>

    cache_base = Path(dc.get("cache_dir", str(splits_root / "cache")))
    return parquet_path, splits_root, cache_base


def _resolve_split_jsons(splits_root: Path, split_name: Optional[str], target: str) -> Dict[str, Path]:
    """
    新规范优先: <splits_root>/{split_name}_{target}_{train,valid,test}.json
    兼容旧规范: <splits_root>/{split_name}_{train,valid,test}.json
    split_name 缺省为 'random'
    """
    sn = split_name or "random"
    # new naming
    cand_new = {
        "train": splits_root / f"{sn}_{target}_train.json",
        "valid": splits_root / f"{sn}_{target}_valid.json",
        "test":  splits_root / f"{sn}_{target}_test.json",
    }
    if all(p.exists() for p in cand_new.values()):
        return cand_new

    # legacy naming (without target in filename)
    cand_old = {
        "train": splits_root / f"{sn}_train.json",
        "valid": splits_root / f"{sn}_valid.json",
        "test":  splits_root / f"{sn}_test.json",
    }
    if all(p.exists() for p in cand_old.values()):
        logger.warning("使用旧版分割文件命名（未包含 target）: %s", splits_root)
        return cand_old

    raise FileNotFoundError(
        f"未找到分割文件于 {splits_root}，尝试了: "
        f"{cand_new['train'].name}/{cand_new['valid'].name}/{cand_new['test'].name} 和 "
        f"{cand_old['train'].name}/{cand_old['valid'].name}/{cand_old['test'].name}"
    )


# --------- Dataloaders ----------
def create_dataloaders(
    data_config: dict,
    train_config: dict,
    split_name: Optional[str],
    target: str,
    model_config: dict = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    创建 train/valid/test 三个 DataLoader。
    - 训练集：启用加权采样 (WeightedRandomSampler) 以平衡长尾分布（仅单卡模式）。
    - 分布式：使用 DistributedSampler (暂不支持同时使用 WeightedSampler，需自定义)。
    """
    parquet_path, splits_root, cache_base = _resolve_paths(
        data_config, target=target, split_name=split_name
    )
    split_jsons = _resolve_split_jsons(splits_root, split_name, target)

    # loader config with fallbacks
    loader_cfg = model_config.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 32))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", True))

    # cutoff: prefer explicit per-backbone cutoff keys
    cutoff = resolve_model_cutoff(model_config)

    loaders: Dict[str, DataLoader] = {}
    for split, json_path in split_jsons.items():  # 'train', 'valid', 'test'
        cache_dir = cache_base / split
        ds = GraphDataset(
            parquet_path=parquet_path,
            idx_json=str(json_path),
            cache_dir=str(cache_dir),
            cutoff_3d=cutoff,
        )

        # ---- Sampler & Shuffle 逻辑 ----
        sampler = None
        shuffle = (split == "train")
        
        if split == "train" and distributed:
            # 分布式训练：必须使用 DistributedSampler
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            shuffle = False 

        dl_kwargs: Dict[str, Any] = {}
        if num_workers > 0:
            if "prefetch_factor" in loader_cfg:
                dl_kwargs["prefetch_factor"] = int(loader_cfg["prefetch_factor"])
            if "persistent_workers" in loader_cfg:
                dl_kwargs["persistent_workers"] = bool(loader_cfg["persistent_workers"])

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=make_collate_graph(target),
            drop_last=bool(loader_cfg.get("drop_last", False)),
            **dl_kwargs,
        )

    logger.info(
        "✓ DataLoaders created | target=%s | parquet=%s | splits_root=%s | split_name=%s | batch=%d | workers=%d | distributed=%s",
        target,
        parquet_path,
        str(splits_root),
        split_name or "random",
        batch_size,
        num_workers,
        distributed,
    )
    return loaders["train"], loaders["valid"], loaders["test"]

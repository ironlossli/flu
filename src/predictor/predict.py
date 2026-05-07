# predict.py — minimal offline inference runner
# Usage examples:
#   python -m predictor.predict \
#       --ckpt_dir checkpoints/ehc_egnn_abs_default_20251110_013753 \
#       --input   src/predictor/samples/new_data.csv \
#       --target  abs \
#       --out     predictions_abs.csv
#
# Notes:
# - Accepts CSV or Parquet as input. CSV must include columns: xyz_path, solvent_smiles.
# - Will reuse the saved configs (data.json, model.json, train.json) under the checkpoint dir
#   to rebuild the model, then load weights from best.pt (or an explicit --ckpt path).
# - New data are preprocessed with the same 3D/2D featurizers (cutoff taken from model config).
# - If the input has true labels (lambda_abs / lambda_em), --eval will compute MAE/RMSE/R2.

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import pandas as pd

# -----------------------------------------------------------------------------
# Wire up import paths so we can reuse the existing engine modules
# The engine modules were designed for script-level imports, so we add that folder
# directly to sys.path (similar to engine/train.py logic).
CUR = Path(__file__).resolve()
ENGINE_DIR = (CUR.parent.parent / "spectra" / "engine").resolve()
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from model_registry import build_model_from_config  # type: ignore
from preprocessing import DatasetPreprocessor       # type: ignore
from datamodule import GraphData, make_collate_graph  # type: ignore


# ----------------------------- small utils -----------------------------------

def device_from_pref(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def extract_pred(output: Any) -> torch.Tensor:
    """Mirror runner.extract_pred: unify user model outputs to [B] tensor."""
    if isinstance(output, torch.Tensor):
        y = output
    elif isinstance(output, dict):
        for k in ("pred", "y_pred", "output", "out", "logits"):
            v = output.get(k)
            if isinstance(v, torch.Tensor):
                y = v
                break
        else:
            raise ValueError("Model dict output missing known prediction keys")
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        y = output[0]
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y


def load_configs(ckpt_dir: Path) -> Dict[str, dict]:
    cfg_dir = ckpt_dir / "configs"
    out = {}
    for name in ("data", "model", "train"):
        p = cfg_dir / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing config: {p}")
        with open(p, "r") as f:
            out[name] = json.load(f)
    return out


def locate_ckpt(ckpt_dir: Path, ckpt_path: Optional[Path]) -> Path:
    if ckpt_path is not None:
        if ckpt_path.is_dir():
            # allow user to pass a dir that contains best.pt
            p = ckpt_path / "best.pt"
            if p.exists():
                return p
            raise FileNotFoundError(f"--ckpt points to a folder that has no best.pt: {ckpt_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")
        return ckpt_path
    # default: <ckpt_dir>/best.pt
    p = ckpt_dir / "best.pt"
    if not p.exists():
        # fallback: last.pt if best missing
        q = ckpt_dir / "last.pt"
        if q.exists():
            return q
        raise FileNotFoundError(f"No best.pt/last.pt under {ckpt_dir}")
    return p


def to_parquet_if_csv(input_path: Path, tmp_dir: Path) -> Path:
    if input_path.suffix.lower() == ".parquet":
        return input_path
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        # Ensure required columns exist
        for col in ("xyz_path", "solvent_smiles"):
            if col not in df.columns:
                raise ValueError(f"Input CSV missing required column: {col}")
        # Targets can be absent for prediction; add zeros so collate works
        for col in ("lambda_abs", "lambda_em"):
            if col not in df.columns:
                df[col] = 0.0
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pq_path = tmp_dir / (input_path.stem + ".parquet")
        df.to_parquet(pq_path, index=False)
        return pq_path
    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def build_predict_loader(parquet_path: Path, batch_size: int, num_workers: int, target: str, cutoff: float):
    pre = DatasetPreprocessor(str(parquet_path), cutoff_3d=float(cutoff))
    # predict on all rows of this parquet
    import pyarrow.parquet as pq  # fast row count
    n_rows = pq.ParquetFile(str(parquet_path)).metadata.num_rows
    indices = list(range(int(n_rows)))
    samples = pre.preprocess_split(indices, cache_dir=None)  # return list[dict]
    data_list = [GraphData(**s) for s in samples]

    from torch.utils.data import DataLoader, Dataset

    class _ListDS(Dataset):
        def __len__(self):
            return len(data_list)
        def __getitem__(self, i):
            return data_list[i]

    ds = _ListDS()
    collate = make_collate_graph(target)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate)
    return dl


@torch.no_grad()
def run_predict(args: argparse.Namespace) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt dir not found: {ckpt_dir}")

    cfgs = load_configs(ckpt_dir)
    data_cfg, model_cfg, train_cfg = cfgs["data"], cfgs["model"], cfgs["train"]

    # In case user provides a different input file for prediction, override the parquet in data config
    input_path = Path(args.input).resolve()
    tmp_dir = ckpt_dir / "_predict_tmp"
    pq_path = to_parquet_if_csv(input_path, tmp_dir)

    # Model/data knobs
    target = (args.target or train_cfg.get("target") or "abs").lower()
    loader_cfg = (model_cfg.get("loader") or {})
    batch_size = int(args.batch_size or loader_cfg.get("batch_size", 32))
    num_workers = int(args.num_workers or loader_cfg.get("num_workers", 0))
    cutoff = float(model_cfg.get("cutoff", 5.0))

    # Build model from saved model config
    device = device_from_pref(args.device)
    model = build_model_from_config(model_cfg, data_cfg, train_cfg)
    model.to(device)
    model.eval()

    # Load weights
    ckpt_path = locate_ckpt(ckpt_dir, Path(args.ckpt) if args.ckpt else None)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    # Build loader and run
    loader = build_predict_loader(pq_path, batch_size=batch_size, num_workers=num_workers, target=target, cutoff=cutoff)

    preds: List[float] = []
    xyzs: List[str] = []
    ssolv: List[str] = []
    # Optional: if true labels exist in the file and --eval requested, collect them
    ys_true: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = extract_pred(out).detach().float().cpu()
        preds.extend(y.tolist())
        xyzs.extend(batch.xyz_path)
        ssolv.extend(batch.solvent_smiles)
        if args.eval:
            target_field = f"lambda_{target}"
            if hasattr(batch, target_field):
                yt = getattr(batch, target_field).detach().float().cpu().tolist()
                ys_true.extend(yt)

    import numpy as np
    import csv

    out_path = Path(args.out).resolve() if args.out else (ckpt_dir / f"predictions_{target}.csv").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["idx", "xyz_path", "solvent_smiles", f"pred_lambda_{target}"]
        if args.eval and len(ys_true) == len(preds):
            header += [f"true_lambda_{target}"]
        w.writerow(header)
        for i, (xp, sv, yp) in enumerate(zip(xyzs, ssolv, preds)):
            row = [i, xp, sv, yp]
            if args.eval and len(ys_true) == len(preds):
                row += [ys_true[i]]
            w.writerow(row)

    # Optional metrics
    if args.eval and len(ys_true) == len(preds) and len(preds) > 0:
        a = np.asarray(ys_true, dtype=float)
        b = np.asarray(preds, dtype=float)
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        # Guard against degenerate variance
        r2 = 1.0 - float(((a - b) ** 2).sum() / max(1e-12, ((a - a.mean()) ** 2).sum()))
        logging.info("Eval: MAE=%.4f | RMSE=%.4f | R2=%.4f (n=%d)", mae, rmse, r2, len(a))

    logging.info("Saved predictions -> %s", out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Predict on new data with a trained checkpoint")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Path to a single run dir containing configs/ and best.pt")
    p.add_argument("--ckpt", type=str, default=None, help="Optional explicit checkpoint path (.pt). Overrides --ckpt_dir/best.pt")
    p.add_argument("--input", type=str, required=True, help="Input CSV/Parquet with columns: xyz_path, solvent_smiles")
    p.add_argument("--target", type=str, default=None, choices=["abs", "em"], help="Which head to use. Defaults to training cfg target if unset")
    p.add_argument("--out", type=str, default=None, help="Output CSV path. Default: <ckpt_dir>/predictions_<target>.csv")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size. Default: model.cfg.loader.batch_size")
    p.add_argument("--num_workers", type=int, default=None, help="Override num_workers. Default: model.cfg.loader.num_workers")
    p.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    p.add_argument("--eval", action="store_true", help="If true labels present, compute MAE/RMSE/R2")
    args = p.parse_args()

    run_predict(args)


if __name__ == "__main__":
    main()

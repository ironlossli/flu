#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch

try:
    import yaml
except Exception as exc:
    raise RuntimeError(f"yaml import failed: {exc}") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectra.models.ehc import build_ehc_equiformer_v2  # noqa: E402


def load_cfg(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return yaml.safe_load(path.read_text())


def make_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)


def make_batch(device: torch.device):
    solute_z = torch.tensor([6, 7, 8, 16, 9], device=device, dtype=torch.long)
    solute_pos = torch.randn(5, 3, device=device, dtype=torch.float32)
    solute_edge_index = make_edge_index(5, device)
    solute_edge_attr = torch.zeros(solute_edge_index.size(1), 4, device=device, dtype=torch.float32)
    solute_batch = torch.zeros(5, device=device, dtype=torch.long)

    solvent_x = torch.randn(3, 14, device=device, dtype=torch.float32)
    solvent_edge_index = make_edge_index(3, device)
    solvent_edge_attr = torch.zeros(solvent_edge_index.size(1), 7, device=device, dtype=torch.float32)
    solvent_batch = torch.zeros(3, device=device, dtype=torch.long)

    return {
        "solute_x": solute_z,
        "solute_pos": solute_pos,
        "solute_edge_index": solute_edge_index,
        "solute_edge_attr": solute_edge_attr,
        "solute_batch": solute_batch,
        "solvent_x": solvent_x,
        "solvent_edge_index": solvent_edge_index,
        "solvent_edge_attr": solvent_edge_attr,
        "solvent_batch": solvent_batch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/model/ehc_equiformer_v2.yaml"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    device_type = args.device
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    if device_type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("CPU mode; BF16 support depends on CPU + PyTorch ops.")

    cfg = load_cfg(Path(args.config))
    cfg.setdefault("eq2_num_atoms", 0)
    cfg.setdefault("eq2_bond_feat_dim", 0)
    cfg.setdefault("eq2_num_targets", 1)

    print("Building Equiformer v2...")
    try:
        model = build_ehc_equiformer_v2(cfg)
    except Exception as exc:
        print(f"[BUILD_FAIL] {exc}")
        return

    model.eval().to(device)
    batch = make_batch(device)

    ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )
    try:
        with ctx:
            out = model(batch)
        pred = out.get("pred")
        print("[OK] pred dtype:", pred.dtype)
    except Exception as exc:
        print(f"[RUN_FAIL] {exc}")
        print("TRACE:", " | ".join(traceback.format_exc().splitlines()[-6:]))


if __name__ == "__main__":
    main()

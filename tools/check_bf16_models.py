#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

try:
    import yaml
except Exception as exc:
    raise RuntimeError(f"yaml import failed: {exc}") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectra.models.ehc import (  # noqa: E402
    build_ehc_leftnet,
    build_ehc_equiformer,
    build_ehc_equiformer_v2,
)


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
    # Solute (Z values from ELEMENTS12_Z)
    solute_z = torch.tensor([6, 7, 8, 16, 9], device=device, dtype=torch.long)
    solute_pos = torch.randn(5, 3, device=device, dtype=torch.float32)
    solute_edge_index = make_edge_index(5, device)
    solute_edge_attr = torch.zeros(solute_edge_index.size(1), 4, device=device, dtype=torch.float32)
    solute_batch = torch.zeros(5, device=device, dtype=torch.long)

    # Solvent
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


def run_case(name, build_fn, cfg_path, device, device_type):
    if cfg_path is None or not cfg_path.exists():
        print(f"[SKIP] {name}: config not found")
        return

    print(f"\n== {name} ==")
    try:
        cfg = load_cfg(cfg_path)
        model = build_fn(cfg)
    except Exception as exc:
        print(f"[BUILD_FAIL] {name}: {exc}")
        return

    model.eval().to(device)
    batch = make_batch(device)

    try:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(batch)
        pred = out.get("pred")
        print(f"[BF16_OK] pred dtype={pred.dtype}")
    except Exception as exc:
        print(f"[BF16_FAIL] {name}: {exc}")
        print("TRACE:", " | ".join(traceback.format_exc().splitlines()[-6:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--only", default="", help="Comma list: leftnet,equiformer,equiformer_v2")
    parser.add_argument(
        "--leftnet-config",
        default=str(
            ROOT
            / "checkpoints/quick_tests/leftnet_geom_nonlocal/ehc_leftnet_abs_default_20260113_014428/configs/model.json"
        ),
    )
    parser.add_argument("--equiformer-config", default=str(ROOT / "configs/model/ehc_equiformer.yaml"))
    parser.add_argument(
        "--equiformer-v2-config",
        default=str(ROOT / "configs/model/ehc_equiformer_v2.yaml"),
    )
    args = parser.parse_args()

    if args.device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = args.device
    device = torch.device(device_type)

    if device_type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("CPU mode; BF16 support depends on CPU + PyTorch ops.")

    only = {x.strip() for x in args.only.split(",") if x.strip()}

    def want(name):
        return (not only) or (name in only)

    if want("leftnet"):
        run_case("leftnet", build_ehc_leftnet, Path(args.leftnet_config), device, device_type)
    if want("equiformer"):
        run_case("equiformer", build_ehc_equiformer, Path(args.equiformer_config), device, device_type)
    if want("equiformer_v2"):
        run_case("equiformer_v2", build_ehc_equiformer_v2, Path(args.equiformer_v2_config), device, device_type)


if __name__ == "__main__":
    main()

# train.py
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from runner import ExperimentRunner, default_run_dir

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # 指向 .../flu/src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def apply_overrides(cfg: Dict[str, Any], overrides: str):
    """
    Apply dot-notation overrides to a config dict.
    Format: "key1.subkey=value1,key2=value2"
    """
    if not overrides:
        return
    for item in overrides.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Simple type inference
        try:
            if value.lower() == "true": value = True
            elif value.lower() == "false": value = False
            elif value.lower() == "none": value = None
            elif "." in value: value = float(value)
            else: value = int(value)
        except ValueError:
            pass # Keep as string

        # Apply nested update
        parts = key.split(".")
        current = cfg
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
        print(f"Override: Set {key} = {value} ({type(value).__name__})")

import warnings
# Suppress torch.load warning
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# Suppress AMP deprecation warning
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*")

def main():
    ap = argparse.ArgumentParser(description="Unified training entrypoint")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--model", type=str, required=True, help="Path to model.yaml")
    ap.add_argument("--train", type=str, required=True, help="Path to train.yaml only target")
    ap.add_argument("--target", type=str, default=None, help="Override target (abs/em); else from train.yaml")
    ap.add_argument("--split", type=str, default=None, help="Override split name; else derived from data.yaml")
    ap.add_argument("--run_dir", type=str, default=None, help="Override run directory")
    ap.add_argument("--config_overrides", type=str, default=None, help="Override model config values (e.g. 'trainer.epochs=10')")
    args = ap.parse_args()

    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)

    if args.config_overrides:
        apply_overrides(model_cfg, args.config_overrides)

    target = args.target if args.target else train_cfg.get("target")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(
        base_dir=(model_cfg.get("trainer", {}) or {}).get("output_dir"),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        target=target,
        split=args.split,
        )

    from runner import setup_logging
    setup_logging(run_dir, level=logging.INFO)

    runner = ExperimentRunner(run_dir)
    result = runner.run(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        target_override=target,
        split_override=args.split,
    )
    # Also save a compact summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

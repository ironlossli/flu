#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def infer_target(name: str) -> Optional[str]:
    if "_abs_" in name:
        return "abs"
    if "_em_" in name:
        return "em"
    return None


def infer_model_name(run_dir: Path) -> str:
    model_cfg = run_dir / "configs" / "model.json"
    if model_cfg.exists():
        data = load_json(model_cfg)
        if isinstance(data, dict) and data.get("name"):
            return str(data["name"])
    name = run_dir.name
    if name.startswith("ablation_"):
        parts = name.split("_")
        if len(parts) >= 4:
            return "_".join(parts[:4])
    return name


def load_reference(ref_path: Optional[str]) -> Dict[str, Dict]:
    if not ref_path:
        return {}
    ref = Path(ref_path)
    if ref.is_dir():
        ref_summary = ref / "summary.json"
    else:
        ref_summary = ref
    data = load_json(ref_summary)
    test = data.get("test") or {}
    target = infer_target(ref_summary.parent.name) or infer_target(ref_summary.name)
    if target:
        return {target: test}
    return {"all": test}


def fmt(val: Optional[float]) -> str:
    if val is None:
        return "-"
    return f"{val:.4f}"


def build_rows(root: Path) -> List[Dict]:
    rows = []
    for summary in root.rglob("summary.json"):
        run_dir = summary.parent
        data = load_json(summary)
        test = data.get("test") or {}
        model = infer_model_name(run_dir)
        target = infer_target(run_dir.name) or infer_target(summary.name)
        model_cfg = load_json(run_dir / "configs" / "model.json")
        batch = None
        if isinstance(model_cfg, dict):
            batch = (model_cfg.get("loader") or {}).get("batch_size")
        rows.append(
            {
                "run": run_dir.name,
                "path": str(run_dir),
                "model": model,
                "target": target,
                "batch": batch,
                "mae": test.get("mae"),
                "rmse": test.get("rmse"),
                "r2": test.get("r2"),
            }
        )
    return rows


def pick_best(rows: List[Dict]) -> List[Dict]:
    best: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (r["model"], r["target"])
        if r["mae"] is None:
            continue
        if key not in best or r["mae"] < best[key]["mae"]:
            best[key] = r
    return [best[k] for k in sorted(best)]


def add_notes(row: Dict) -> str:
    notes = []
    r2 = row.get("r2")
    rmse = row.get("rmse")
    mae = row.get("mae")
    if r2 is not None and r2 < 0:
        notes.append("r2<0")
    if rmse is not None and rmse > 100:
        notes.append("rmse>100")
    if mae is not None and mae > 200:
        notes.append("mae>200")
    return ",".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation summaries.")
    parser.add_argument("--root", default="checkpoints/ablation", help="Root folder to scan for summary.json")
    parser.add_argument("--reference", default=None, help="Reference run dir or summary.json for delta")
    parser.add_argument("--best-only", action="store_true", help="Only show best per model/target")
    parser.add_argument("--show-paths", action="store_true", help="Show run dir paths")
    args = parser.parse_args()

    root = Path(args.root)
    rows = build_rows(root)
    if args.best_only:
        rows = pick_best(rows)

    rows.sort(key=lambda r: (r["target"] or "", r["model"], r["mae"] if r["mae"] is not None else 1e9))
    ref = load_reference(args.reference)

    header = ["target", "model", "batch", "mae", "rmse", "r2", "delta_mae", "notes"]
    if args.show_paths:
        header.append("path")
    print(" | ".join(header))
    print("-" * 120)

    for r in rows:
        target = r["target"] or "-"
        delta_mae = None
        if ref:
            ref_key = target if target in ref else "all"
            base = ref.get(ref_key) or {}
            if base.get("mae") is not None and r.get("mae") is not None:
                delta_mae = r["mae"] - base["mae"]
        notes = add_notes(r)
        row = [
            target,
            r["model"],
            str(r["batch"]) if r["batch"] is not None else "-",
            fmt(r["mae"]),
            fmt(r["rmse"]),
            fmt(r["r2"]),
            fmt(delta_mae),
            notes or "-",
        ]
        if args.show_paths:
            row.append(r["path"])
        print(" | ".join(row))


if __name__ == "__main__":
    main()

import math
import os
from pathlib import Path
import json


def parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"true", "false"}:
            return low == "true"
    return val


def values_equal(got, expect):
    if expect is False and got is None:
        return True
    if isinstance(expect, bool):
        return got is expect
    if isinstance(expect, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(got), float(expect), rel_tol=1e-9, abs_tol=1e-12)
    return got == expect


def matches(model, expect):
    for key, val in expect.items():
        got = model.get(key)
        if not values_equal(got, val):
            return False
    return True


def iter_runs(root: Path):
    if not root.exists():
        return []
    runs = []
    for path in root.rglob("model.json"):
        if path.parent.name != "configs":
            continue
        run_dir = path.parent.parent
        try:
            model = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        runs.append((run_dir, model))
    runs.sort(key=lambda item: item[0].stat().st_mtime if item[0].exists() else 0, reverse=True)
    return runs


def find_match(runs, expect):
    for run_dir, model in runs:
        if matches(model, expect):
            return run_dir, model
    return None, None


def render_kv(model, keys):
    return {k: model.get(k) for k in keys}


def print_section(title):
    print("")
    print(f"=== {title} ===")


def should_run(model_list, name):
    if not model_list:
        return True
    return name in {item.strip() for item in model_list.split(",") if item.strip()}


def main():
    run_models = os.environ.get("RUN_MODELS", "egnn,painn,schnet,eqf,eqf2")

    specs = []

    if should_run(run_models, "egnn"):
        root = Path(os.environ.get("EGNN_RUN_OUT", "checkpoints/ablation_vegnn_current"))
        specs.append((
            "EGNN",
            root,
            [
                ("baseline_full", {
                    "name": "benchmark_vegnn",
                    "use_moments": True,
                    "use_global": True,
                    "use_virtual_node": True,
                    "mgil_use_nonlocal": False,
                }),
                ("no_moments", {
                    "name": "benchmark_vegnn",
                    "use_moments": False,
                    "use_global": True,
                    "use_virtual_node": True,
                    "mgil_use_nonlocal": False,
                }),
                ("no_global", {
                    "name": "benchmark_vegnn",
                    "use_moments": True,
                    "use_global": False,
                    "use_virtual_node": True,
                    "mgil_use_nonlocal": False,
                }),
                ("no_virtual_node", {
                    "name": "benchmark_vegnn",
                    "use_moments": True,
                    "use_global": True,
                    "use_virtual_node": False,
                    "mgil_use_nonlocal": False,
                }),
                ("all_off", {
                    "name": "benchmark_vegnn",
                    "use_moments": False,
                    "use_global": False,
                    "use_virtual_node": False,
                    "mgil_use_nonlocal": False,
                }),
            ],
            ["use_moments", "use_global", "use_virtual_node", "mgil_use_nonlocal"],
        ))

    if should_run(run_models, "painn"):
        root = Path(os.environ.get("PAINN_RUN_OUT", "checkpoints/painn_geom_ablation"))
        specs.append((
            "PaiNN",
            root,
            [
                ("baseline_full", {
                    "name": "ehc_painn",
                    "painn_use_geom_gate": True,
                    "painn_geom_use_moments": True,
                    "painn_geom_use_global": True,
                    "use_virtual_node": True,
                }),
                ("geom_rbf_on", {
                    "name": "ehc_painn",
                    "painn_use_geom_gate": True,
                    "painn_geom_use_moments": False,
                    "painn_geom_use_global": False,
                    "use_virtual_node": False,
                }),
                ("moments_on", {
                    "name": "ehc_painn",
                    "painn_use_geom_gate": True,
                    "painn_geom_use_moments": True,
                    "painn_geom_use_global": False,
                    "use_virtual_node": False,
                }),
                ("global_on", {
                    "name": "ehc_painn",
                    "painn_use_geom_gate": True,
                    "painn_geom_use_moments": False,
                    "painn_geom_use_global": True,
                    "use_virtual_node": False,
                }),
                ("virtual_node_on", {
                    "name": "ehc_painn",
                    "painn_use_geom_gate": False,
                    "painn_geom_use_moments": False,
                    "painn_geom_use_global": False,
                    "use_virtual_node": True,
                }),
                ("all_off", {
                    "name": "ehc_painn",
                    "painn_use_geom_gate": False,
                    "painn_geom_use_moments": False,
                    "painn_geom_use_global": False,
                    "use_virtual_node": False,
                }),
            ],
            ["painn_use_geom_gate", "painn_geom_use_moments", "painn_geom_use_global", "use_virtual_node"],
        ))

    if should_run(run_models, "schnet"):
        root = Path(os.environ.get("SCHNET_RUN_OUT", "checkpoints/schnet_geom_ablation"))
        specs.append((
            "SchNet",
            root,
            [
                ("geom_full", {
                    "name": "ehc_schnet",
                    "schnet_use_geom": True,
                    "schnet_use_moments": True,
                    "schnet_use_global": True,
                }),
                ("no_moments", {
                    "name": "ehc_schnet",
                    "schnet_use_geom": True,
                    "schnet_use_moments": False,
                    "schnet_use_global": True,
                }),
                ("no_global", {
                    "name": "ehc_schnet",
                    "schnet_use_geom": True,
                    "schnet_use_moments": True,
                    "schnet_use_global": False,
                }),
                ("geom_off", {
                    "name": "ehc_schnet",
                    "schnet_use_geom": False,
                    "schnet_use_moments": False,
                    "schnet_use_global": False,
                }),
            ],
            ["schnet_use_geom", "schnet_use_moments", "schnet_use_global"],
        ))

    if should_run(run_models, "eqf"):
        root = Path(os.environ.get("EQF_RUN_OUT", "checkpoints/equiformer_geom_ablation"))
        specs.append((
            "Equiformer v1",
            root,
            [
                ("geom_full", {
                    "name": "ehc_equiformer",
                    "eqf_use_geom_gate": True,
                    "eqf_geom_use_moments": True,
                    "eqf_geom_use_global": True,
                    "eqf_geom_use_ln": True,
                    "mgil_use_nonlocal": True,
                }),
                ("no_moments", {
                    "name": "ehc_equiformer",
                    "eqf_use_geom_gate": True,
                    "eqf_geom_use_moments": False,
                    "eqf_geom_use_global": True,
                    "eqf_geom_use_ln": True,
                    "mgil_use_nonlocal": True,
                }),
                ("no_global", {
                    "name": "ehc_equiformer",
                    "eqf_use_geom_gate": True,
                    "eqf_geom_use_moments": True,
                    "eqf_geom_use_global": False,
                    "eqf_geom_use_ln": True,
                    "mgil_use_nonlocal": True,
                }),
                ("geom_off", {
                    "name": "ehc_equiformer",
                    "eqf_use_geom_gate": False,
                    "eqf_geom_use_moments": False,
                    "eqf_geom_use_global": False,
                    "eqf_geom_use_ln": False,
                    "mgil_use_nonlocal": False,
                }),
            ],
            ["eqf_use_geom_gate", "eqf_geom_use_moments", "eqf_geom_use_global", "eqf_geom_use_ln", "mgil_use_nonlocal"],
        ))

    if should_run(run_models, "eqf2"):
        root = Path(os.environ.get("EQF2_RUN_OUT", "checkpoints/equiformer_v2_geom_ablation"))
        specs.append((
            "Equiformer v2",
            root,
            [
                ("nonlocal_on", {
                    "name": "ehc_equiformer_v2",
                    "mgil_use_nonlocal": True,
                }),
                ("nonlocal_off", {
                    "name": "ehc_equiformer_v2",
                    "mgil_use_nonlocal": False,
                }),
            ],
            ["mgil_use_nonlocal"],
        ))

    for title, root, ablations, keys in specs:
        print_section(f"{title} (root={root})")
        runs = iter_runs(root)
        if not runs:
            print("no runs found")
            continue
        for name, expect in ablations:
            run_dir, model = find_match(runs, expect)
            if run_dir is None:
                print(f"{name}: MISSING (no run with expected overrides)")
                continue
            actual = render_kv(model, keys)
            print(f"{name}: OK {run_dir}")
            print(f"  actual: {actual}")


if __name__ == "__main__":
    main()

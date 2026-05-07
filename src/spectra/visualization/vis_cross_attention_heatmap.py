import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from matplotlib.patches import Rectangle
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Adjust path to find sibling modules
current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


def load_predictor(checkpoint_path: str, device: str = "cuda"):
    import torch
    core_path = src_root / "predictor" / "core.py"
    spec = importlib.util.spec_from_file_location("predictor_core", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load predictor core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SpectraPredictor = module.SpectraPredictor

    if not torch.cuda.is_available() and device == "cuda":
        print("Warning: CUDA not available, switching to CPU.")
        device = "cpu"

    predictor = SpectraPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return predictor


def compute_cross_attention(predictor, xyz_path, solvent_smiles):
    """
    Run a forward pass and extract cross-attention weights.
    Returns a [N_solute, N_solvent] numpy array.
    """
    predictor.model.eval()

    record = {
        "xyz_path": xyz_path,
        "solvent_smiles": solvent_smiles,
        "lambda_abs": 0.0,
        "lambda_em": 0.0,
    }
    sample = predictor.featurizer.featurize_record(record)
    if sample is None:
        raise ValueError("Featurization failed.")

    from datamodule import GraphBatch, GraphData
    batch = GraphBatch([GraphData(**sample)], target=predictor.target)
    batch = batch.to(predictor.device)

    with torch.no_grad():
        output = predictor.model(batch)

    aux = output.get("backbone_aux", {}) if isinstance(output, dict) else {}
    attn_weights = aux.get("attn_weights")
    if attn_weights is None:
        raise ValueError("No cross-attention weights found. Check if cross-attn is enabled.")

    if isinstance(attn_weights, (list, tuple)):
        if not attn_weights:
            raise ValueError("Empty attn_weights list in backbone_aux.")
        attn_weights = attn_weights[-1]

    attn = attn_weights.detach().cpu()
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    if attn.dim() == 3:
        attn = attn[0]
    if attn.dim() != 2:
        raise ValueError(f"Unexpected attn_weights shape: {tuple(attn_weights.shape)}")

    n_solute = int((batch.solute_batch == 0).sum().item()) if hasattr(batch, "solute_batch") else attn.shape[0]
    n_solvent = int((batch.solvent_batch == 0).sum().item()) if hasattr(batch, "solvent_batch") else attn.shape[1]
    n_solute = min(n_solute, attn.shape[0])
    n_solvent = min(n_solvent, attn.shape[1])

    heatmap = attn[:n_solute, :n_solvent].numpy()
    return heatmap


def plot_heatmap(
    heatmap,
    xyz_path,
    solvent_smiles,
    output_path,
    title=None,
    cmap="viridis",
    x_labels=None,
    highlight_col=None,
    hide_ylabel=False,
    hide_yticks=False,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap=cmap)

    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Solvent Atoms")
    else:
        ax.set_xlabel(f"Solvent Atoms ({solvent_smiles})")

    if not hide_ylabel:
        ax.set_ylabel("Solute Atoms (Index)")
    if hide_yticks:
        ax.set_yticks([])

    if title is None:
        solute_name = Path(xyz_path).name
        title = f"Solute-Solvent Cross-Attention\nSolute: {solute_name}, Solvent: {solvent_smiles}"
    if title:
        ax.set_title(title)

    if highlight_col is not None:
        col_idx = int(highlight_col) - 1
        if 0 <= col_idx < heatmap.shape[1]:
            rect = Rectangle(
                (col_idx - 0.5, -0.5),
                1.0,
                heatmap.shape[0],
                fill=False,
                edgecolor="white",
                linewidth=2.0,
                linestyle="--",
            )
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize cross-attention heatmap.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--xyz", type=str, required=True, help="Path to input XYZ file (solute)")
    parser.add_argument("--solvent", type=str, required=True, help="Solvent SMILES")
    parser.add_argument("--output", type=str, default="cross_attention_heatmap.png", help="Output image path")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--no-title", action="store_true", help="Disable plot title")
    parser.add_argument("--x-labels", type=str, default=None, help="Comma-separated solvent atom labels")
    parser.add_argument("--highlight-col", type=int, default=None, help="1-based column to highlight")
    parser.add_argument("--hide-ylabel", action="store_true", help="Hide Y axis label")
    parser.add_argument("--hide-yticks", action="store_true", help="Hide Y axis ticks")
    args = parser.parse_args()

    predictor = load_predictor(args.ckpt)
    heatmap = compute_cross_attention(predictor, args.xyz, args.solvent)
    title = None if args.no_title else args.title
    x_labels = None
    if args.x_labels:
        x_labels = [lab.strip() for lab in args.x_labels.split(",") if lab.strip()]
    plot_heatmap(
        heatmap,
        args.xyz,
        args.solvent,
        args.output,
        title=title,
        cmap=args.cmap,
        x_labels=x_labels,
        highlight_col=args.highlight_col,
        hide_ylabel=args.hide_ylabel,
        hide_yticks=args.hide_yticks,
    )
    print(f"Saved cross-attention heatmap to {args.output}")


if __name__ == "__main__":
    main()

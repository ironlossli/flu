# Reproducing Paper Results

This document describes how to reproduce the main experimental results from the paper.

## Prerequisites

1. Data prepared via `configs/data.yaml` (see README)
2. Environment set up with all dependencies
3. GPU with sufficient memory (16GB+ recommended for Equiformer)

## Table 1: Main Results Across Backbones

Evaluates MGIL + SFM on six backbones for both Absorption (Abs) and Emission (EM) tasks.

```bash
TARGETS="abs em"
MODELS="ehc_egnn_cross_attn ehc_schnet ehc_painn ehc_leftnet ehc_equiformer ehc_equiformer_v2"

for target in $TARGETS; do
    for model in $MODELS; do
        MODEL_CFG="configs/model/${model}.yaml" \
        TARGET=$target BATCH_SIZE=64 SEED=42 \
        bash scripts/train_absorption.sh
    done
done
```

**Metrics reported:** MAE (nm), RMSE (nm), mean ± std over 3 seeds (41, 42, 43).

The baseline (without MGIL/SFM) results are obtained by training backbones with their original configurations.

## Table 2: Incremental Impact of MGIL and SFM

Ablates MGIL and SFM on EGNN and PaiNN, isolating geometry vs. environment contributions.

```bash
# EGNN baseline (no MGIL, no SFM)
MODEL_CFG=configs/model/ehc_egnn_baseline.yaml TARGET=abs bash scripts/train_absorption.sh

# EGNN + MGIL only (SFM disabled via concatenation baseline)
MODEL_CFG=configs/model/ehc_egnn_baseline.yaml TARGET=abs bash scripts/train_absorption.sh

# EGNN + MGIL + SFM (full model)
MODEL_CFG=configs/model/ehc_egnn_cross_attn.yaml TARGET=abs bash scripts/train_absorption.sh

# Repeat for PaiNN
MODEL_CFG=configs/model/ehc_painn.yaml TARGET=abs bash scripts/train_absorption.sh
```

**Metrics reported:** Abs RMSE (nm), mean ± std.

## Table 3: EGNN Component Ablation

Ablates individual MGIL components (Anchors, Moments, Virtual Node) under a fixed EGNN backbone.

```bash
TARGETS="abs em"
MODELS="benchmark_vegnn ablation_v1_no_vector ablation_v2_no_global ablation_v3_no_moments ablation_v4_no_vnode"

for target in $TARGETS; do
    for model in $MODELS; do
        MODEL_CFG="configs/model/${model}.yaml" TARGET=$target bash scripts/train_absorption.sh
    done
done
```

The model configs correspond to:
- `benchmark_vegnn` — Full model (all MGIL components enabled)
- `ablation_v1_no_vector` — w/o vector features
- `ablation_v2_no_global` — w/o global anchors
- `ablation_v3_no_moments` — w/o moment invariants
- `ablation_v4_no_vnode` — w/o virtual node
- Baseline (off) — EGNN without MGIL

**Metrics reported:** Abs RMSE (nm), EM RMSE (nm), mean ± std.

## Table 4: Solvent Conditioning Comparison

Compares tag/concatenation baseline with cross-attention-based SFM on top of fixed MGIL.

```bash
# Concatenation baseline
MODEL_CFG=configs/model/ehc_egnn_baseline.yaml TARGET=abs bash scripts/train_absorption.sh

# Cross-attention SFM
MODEL_CFG=configs/model/ehc_egnn_cross_attn.yaml TARGET=abs bash scripts/train_absorption.sh
```

**Metrics reported:** Abs RMSE (nm), EM RMSE (nm), mean ± std.

## Pareto Chart

The accuracy-efficiency trade-off (R² vs training time) figure compares all backbones with and without MGIL+SFM. After training each model, collect training time from the log and R² from the summary, then plot using the data format in `tools/summarize_ablation.py`.

## Collecting Results

After all runs complete, use:

```bash
python tools/summarize_ablation.py
```

This scans `checkpoints/` for `summary.json` files and prints a formatted table with all metrics.

## Hyperparameter Reference

| Model | Batch Size | Initial LR | Hidden Dim | Layers | Epochs | Dropout |
|-------|-----------|------------|------------|--------|--------|---------|
| EGNN | 64 | 5×10⁻⁴ | 128 | 6 | 200 | 0.1 |
| PaiNN | 64 | 3×10⁻⁴ | 128 | 6 | 200 | 0.1 |
| SchNet | 64 | 5×10⁻⁴ | 128 | 6 | 200 | 0.1 |
| Equiformer V1 | 16 | 5×10⁻⁴ | 128 | 6 | 200 | 0.1 |
| Equiformer V2 | 16 | 5×10⁻⁴ | 128 | 6 | 200 | 0.1 |
| LEFTNet | 64 | 1×10⁻⁴ | 256 | 4 | 200 | 0.1 |

Common settings: AdamW (β₁=0.9, β₂=0.999, ε=10⁻⁸), weight decay 1×10⁻⁵, cosine annealing, min LR 5×10⁻⁵, MSE loss, seeds {41, 42, 43}.

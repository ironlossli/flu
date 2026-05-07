#!/bin/bash
# Reproduce key paper results by running the full ablation suite
# NOTE: This requires prepared data and significant compute.
# Usage: bash scripts/reproduce_tables.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_CFG="configs/data.yaml"
TRAIN_CFG="configs/train.yaml"
SEED="${SEED:-42}"

echo "============================================"
echo "Reproducing Paper Results"
echo "============================================"
echo ""
echo "This script reproduces the key experiments from:"
echo "  Table 1: Main Results Across Backbones"
echo "  Table 2: Incremental Impact of MGIL and SFM"
echo "  Table 3: EGNN Ablation of Anchors, Moments, and VNode"
echo "  Table 4: Solvent Conditioning Comparison"
echo ""
echo "See docs/reproduce_results.md for per-table commands."
echo ""

# Table 2: MGIL + SFM incremental impact on EGNN and PaiNN
echo "=== Table 2: MGIL Component Gains ==="
for target in abs em; do
    for model in ehc_egnn_cross_attn ehc_egnn_baseline ehc_painn; do
        echo ">> $model | $target"
        $PYTHON src/spectra/engine/train.py \
            --data "$DATA_CFG" \
            --train "$TRAIN_CFG" \
            --model "configs/model/${model}.yaml" \
            --target "$target" \
            --config_overrides "trainer.seed=${SEED}"
    done
done

# Table 3: EGNN component ablation (anchors, moments, VNode)
echo "=== Table 3: EGNN Component Ablation ==="
for target in abs em; do
    for model in benchmark_vegnn ablation_v1_no_vector ablation_v2_no_global ablation_v3_no_moments ablation_v4_no_vnode; do
        echo ">> $model | $target"
        $PYTHON src/spectra/engine/train.py \
            --data "$DATA_CFG" \
            --train "$TRAIN_CFG" \
            --model "configs/model/${model}.yaml" \
            --target "$target" \
            --config_overrides "trainer.seed=${SEED}"
    done
done

echo ""
echo "All experiments complete."
echo "Use tools/summarize_ablation.py to collect results from checkpoints/."

#!/bin/bash
# Evaluate a trained model checkpoint
# Usage: bash scripts/evaluate.sh --checkpoint checkpoints/train/best.pt --target abs
set -euo pipefail

PYTHON="${PYTHON:-python}"

CKPT=""
TARGET="abs"
SPLIT="test"
DATA_CFG="configs/data.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --data) DATA_CFG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "Usage: bash scripts/evaluate.sh --checkpoint <path> [--target abs|em] [--split test|valid]"
    exit 1
fi

echo "=== Evaluating model ==="
echo "  Checkpoint: ${CKPT}"
echo "  Target:     ${TARGET}"
echo "  Split:      ${SPLIT}"

$PYTHON src/spectra/engine/eval.py \
    --data "$DATA_CFG" \
    --checkpoint "$CKPT" \
    --target "$TARGET" \
    --split "$SPLIT"

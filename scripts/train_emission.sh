#!/bin/bash
# Train emission (EM) prediction model with MGIL + SFM
# Usage: bash scripts/train_emission.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_CFG="${DATA_CFG:-configs/data.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train.yaml}"
MODEL_CFG="${MODEL_CFG:-configs/model/benchmark_vegnn.yaml}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-checkpoints/train}"
EPOCHS="${EPOCHS:-200}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

overrides="loader.batch_size=${BATCH_SIZE},trainer.seed=${SEED},trainer.output_dir=${OUT_DIR},trainer.epochs=${EPOCHS}"
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
    overrides+=",${EXTRA_OVERRIDES}"
fi

echo "=== Training Emission Model ==="
echo "  Model:  ${MODEL_CFG}"
echo "  Target: em"
echo "  Batch:  ${BATCH_SIZE} | Seed: ${SEED} | Epochs: ${EPOCHS}"
echo "  Out:    ${OUT_DIR}"

$PYTHON src/spectra/engine/train.py \
    --data "$DATA_CFG" \
    --model "$MODEL_CFG" \
    --train "$TRAIN_CFG" \
    --target em \
    --config_overrides "$overrides"

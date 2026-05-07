#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-abs}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-80}"
EQF_BATCH_SIZE="${EQF_BATCH_SIZE:-16}"
EQF2_BATCH_SIZE="${EQF2_BATCH_SIZE:-16}"

EQF_MODEL_CFG="${EQF_MODEL_CFG:-configs/model/ehc_equiformer.yaml}"
EQF2_MODEL_CFG="${EQF2_MODEL_CFG:-configs/model/ehc_equiformer_v2.yaml}"

OUT_DIR="${OUT_DIR:-checkpoints/equiformer_nonlocal_ablation}"

run_one() {
    local name="$1"
    local model_cfg="$2"
    local batch_size="$3"
    local seed="$4"
    local overrides="$5"
    echo "=== ${name} (seed=${seed}) ==="
    TARGET="${TARGET}" SEED="${seed}" BATCH_SIZE="${batch_size}" EPOCHS="${EPOCHS}" \
        OUT_DIR="${OUT_DIR}/${name}" MODEL_CFG="${model_cfg}" \
        EXTRA_OVERRIDES="${overrides}" \
        bash run_train.sh
}

for SEED in ${SEEDS//,/ }; do
    # Equiformer v1: Full vs Full-without-nonlocal (geom gate on)
    run_one "eqf_full" "${EQF_MODEL_CFG}" "${EQF_BATCH_SIZE}" "${SEED}" \
        "eqf_use_geom_gate=true,eqf_geom_use_moments=true,eqf_geom_use_global=true,eqf_geom_use_ln=true,mgil_use_nonlocal=true"
    run_one "eqf_full_wo_nonlocal" "${EQF_MODEL_CFG}" "${EQF_BATCH_SIZE}" "${SEED}" \
        "eqf_use_geom_gate=true,eqf_geom_use_moments=true,eqf_geom_use_global=true,eqf_geom_use_ln=true,mgil_use_nonlocal=false"

    # Equiformer v2: nonlocal on/off
    run_one "eqf2_full" "${EQF2_MODEL_CFG}" "${EQF2_BATCH_SIZE}" "${SEED}" \
        "mgil_use_nonlocal=true"
    run_one "eqf2_full_wo_nonlocal" "${EQF2_MODEL_CFG}" "${EQF2_BATCH_SIZE}" "${SEED}" \
        "mgil_use_nonlocal=false"
done

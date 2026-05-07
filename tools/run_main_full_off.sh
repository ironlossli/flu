#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-abs}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-200}"
OUT_ROOT="${OUT_ROOT:-checkpoints/main_full_off}"

EGNN_BATCH_SIZE="${EGNN_BATCH_SIZE:-64}"
PAINN_BATCH_SIZE="${PAINN_BATCH_SIZE:-64}"
SCHNET_BATCH_SIZE="${SCHNET_BATCH_SIZE:-64}"
EQF_BATCH_SIZE="${EQF_BATCH_SIZE:-16}"
EQF2_BATCH_SIZE="${EQF2_BATCH_SIZE:-16}"
LEFTNET_BATCH_SIZE="${LEFTNET_BATCH_SIZE:-64}"

EGNN_MODEL_CFG="${EGNN_MODEL_CFG:-configs/model/benchmark_vegnn.yaml}"
PAINN_MODEL_CFG="${PAINN_MODEL_CFG:-configs/model/ehc_painn.yaml}"
SCHNET_MODEL_CFG="${SCHNET_MODEL_CFG:-configs/model/ehc_schnet.yaml}"
EQF_MODEL_CFG="${EQF_MODEL_CFG:-configs/model/ehc_equiformer.yaml}"
EQF2_MODEL_CFG="${EQF2_MODEL_CFG:-configs/model/ehc_equiformer_v2.yaml}"
LEFTNET_MODEL_CFG="${LEFTNET_MODEL_CFG:-configs/model/ehc_leftnet.yaml}"

run_one() {
    local name="$1"
    local model_cfg="$2"
    local batch_size="$3"
    local seed="$4"
    local out_dir="$5"
    local overrides="$6"
    echo "=== ${name} (seed=${seed}) ==="
    TARGET="${TARGET}" SEED="${seed}" BATCH_SIZE="${batch_size}" EPOCHS="${EPOCHS}" \
        OUT_DIR="${out_dir}" MODEL_CFG="${model_cfg}" \
        EXTRA_OVERRIDES="${overrides}" \
        bash run_train.sh
}

for SEED in ${SEEDS//,/ }; do
    # EGNN Full/Off
    run_one "egnn_full" "${EGNN_MODEL_CFG}" "${EGNN_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/egnn/full" \
        "use_backbone_dropout=false,use_moments=true,use_global=true,use_virtual_node=true,mgil_use_nonlocal=true"
    run_one "egnn_off" "${EGNN_MODEL_CFG}" "${EGNN_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/egnn/off" \
        "use_backbone_dropout=false,use_moments=false,use_global=false,use_virtual_node=false,mgil_use_nonlocal=false"

    # PaiNN Full/Off (geom gate)
    run_one "painn_full" "${PAINN_MODEL_CFG}" "${PAINN_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/painn/full" \
        "painn_use_geom_gate=true,painn_geom_use_moments=true,painn_geom_use_global=true,use_virtual_node=true,mgil_use_nonlocal=true"
    run_one "painn_off" "${PAINN_MODEL_CFG}" "${PAINN_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/painn/off" \
        "painn_use_geom_gate=false,painn_geom_use_moments=false,painn_geom_use_global=false,use_virtual_node=false,mgil_use_nonlocal=false"

    # SchNet Full/Off
    run_one "schnet_full" "${SCHNET_MODEL_CFG}" "${SCHNET_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/schnet/full" \
        "schnet_use_geom=true,schnet_use_moments=true,schnet_use_global=true"
    run_one "schnet_off" "${SCHNET_MODEL_CFG}" "${SCHNET_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/schnet/off" \
        "schnet_use_geom=false,schnet_use_moments=false,schnet_use_global=false"

    # Equiformer v1 Full/Off
    run_one "eqf_full" "${EQF_MODEL_CFG}" "${EQF_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/equiformer/full" \
        "eqf_use_geom_gate=true,eqf_geom_use_moments=true,eqf_geom_use_global=true,eqf_geom_use_ln=true,mgil_use_nonlocal=true"
    run_one "eqf_off" "${EQF_MODEL_CFG}" "${EQF_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/equiformer/off" \
        "eqf_use_geom_gate=false,eqf_geom_use_moments=false,eqf_geom_use_global=false,eqf_geom_use_ln=false,mgil_use_nonlocal=false"

    # Equiformer v2 Full/Off (nonlocal on/off)
    run_one "eqf2_full" "${EQF2_MODEL_CFG}" "${EQF2_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/equiformer_v2/full" \
        "mgil_use_nonlocal=true"
    run_one "eqf2_off" "${EQF2_MODEL_CFG}" "${EQF2_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/equiformer_v2/off" \
        "mgil_use_nonlocal=false"

    # LEFTNet Full/Off
    run_one "leftnet_full" "${LEFTNET_MODEL_CFG}" "${LEFTNET_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/leftnet/full" \
        "leftnet_use_geom_gate=true,leftnet_geom_use_moments=true,leftnet_geom_use_global=true,leftnet_geom_use_ln=true,mgil_use_nonlocal=true"
    run_one "leftnet_off" "${LEFTNET_MODEL_CFG}" "${LEFTNET_BATCH_SIZE}" "${SEED}" \
        "${OUT_ROOT}/leftnet/off" \
        "leftnet_use_geom_gate=false,leftnet_geom_use_moments=false,leftnet_geom_use_global=false,leftnet_geom_use_ln=false,mgil_use_nonlocal=false"
done

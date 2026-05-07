#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-abs}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-200}"

EGNN_BATCH_SIZE="${EGNN_BATCH_SIZE:-64}"
PAINN_BATCH_SIZE="${PAINN_BATCH_SIZE:-64}"

EGNN_MODEL_CFG="${EGNN_MODEL_CFG:-configs/model/benchmark_vegnn.yaml}"
PAINN_MODEL_CFG="${PAINN_MODEL_CFG:-configs/model/ehc_painn.yaml}"

OUT_DIR="${OUT_DIR:-checkpoints/vnode_cross_backbone}"

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
    # EGNN: MGIL full with/without VNode
    run_one "egnn_full" "${EGNN_MODEL_CFG}" "${EGNN_BATCH_SIZE}" "${SEED}" \
        "use_backbone_dropout=false,use_moments=true,use_global=true,use_virtual_node=true,mgil_use_nonlocal=true"
    run_one "egnn_no_vnode" "${EGNN_MODEL_CFG}" "${EGNN_BATCH_SIZE}" "${SEED}" \
        "use_backbone_dropout=false,use_moments=true,use_global=true,use_virtual_node=false,mgil_use_nonlocal=true"

    # PaiNN: geom gate on with/without VNode
    run_one "painn_full" "${PAINN_MODEL_CFG}" "${PAINN_BATCH_SIZE}" "${SEED}" \
        "painn_use_geom_gate=true,painn_geom_use_moments=true,painn_geom_use_global=true,use_virtual_node=true,mgil_use_nonlocal=true"
    run_one "painn_no_vnode" "${PAINN_MODEL_CFG}" "${PAINN_BATCH_SIZE}" "${SEED}" \
        "painn_use_geom_gate=true,painn_geom_use_moments=true,painn_geom_use_global=true,use_virtual_node=false,mgil_use_nonlocal=true"
done

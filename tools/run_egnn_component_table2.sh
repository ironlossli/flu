#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-abs}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_CFG="${MODEL_CFG:-configs/model/benchmark_vegnn.yaml}"
OUT_DIR="${OUT_DIR:-checkpoints/egnn_component_table2}"

# Component ablation keeps nonlocal off by request.
COMMON_OVERRIDES="use_backbone_dropout=false,mgil_use_nonlocal=false"

run_one() {
    local name="$1"
    local seed="$2"
    local overrides="$3"
    echo "=== ${name} (seed=${seed}) ==="
    TARGET="${TARGET}" SEED="${seed}" BATCH_SIZE="${BATCH_SIZE}" EPOCHS="${EPOCHS}" \
        OUT_DIR="${OUT_DIR}/${name}" MODEL_CFG="${MODEL_CFG}" \
        EXTRA_OVERRIDES="${COMMON_OVERRIDES},${overrides}" \
        bash run_train.sh
}

for SEED in ${SEEDS//,/ }; do
    run_one "off" "${SEED}" "use_moments=false,use_global=false,use_virtual_node=false"
    run_one "anchors_on" "${SEED}" "use_moments=false,use_global=true,use_virtual_node=false"
    run_one "moments_on" "${SEED}" "use_moments=true,use_global=true,use_virtual_node=false"
    run_one "vnode_on" "${SEED}" "use_moments=true,use_global=true,use_virtual_node=true"
    run_one "full" "${SEED}" "use_moments=true,use_global=true,use_virtual_node=true"
done

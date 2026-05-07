#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-abs}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_CFG="${MODEL_CFG:-configs/model/benchmark_vegnn.yaml}"
OUT_DIR="${OUT_DIR:-checkpoints/conditioning_ablations}"

# MGIL Full fixed.
BASE_OVERRIDES="use_backbone_dropout=false,use_moments=true,use_global=true,use_virtual_node=true,mgil_use_nonlocal=true"

run_one() {
    local name="$1"
    local seed="$2"
    local overrides="$3"
    echo "=== ${name} (seed=${seed}) ==="
    TARGET="${TARGET}" SEED="${seed}" BATCH_SIZE="${BATCH_SIZE}" EPOCHS="${EPOCHS}" \
        OUT_DIR="${OUT_DIR}/${name}" MODEL_CFG="${MODEL_CFG}" \
        EXTRA_OVERRIDES="${BASE_OVERRIDES},${overrides}" \
        bash run_train.sh
}

for SEED in ${SEEDS//,/ }; do
    # Concat baseline (tag conditioning)
    run_one "concat_baseline" "${SEED}" "use_concat_baseline=true,film_scale=0.3,egnn_query_layer_index=1,egnn_use_edge_film=true"

    # Scalar FiLM (bulk field modulation; no cross-attn, no edge film)
    run_one "scalar_film" "${SEED}" "use_concat_baseline=false,film_scale=0.3,egnn_query_layer_index=999,egnn_use_edge_film=false"

    # Cross-attn (site-specific modulation; no edge film)
    run_one "cross_attn" "${SEED}" "use_concat_baseline=false,film_scale=0.3,egnn_query_layer_index=1,egnn_use_edge_film=false"

    # EHC strategy (node + edge FiLM with cross-attn)
    run_one "ehc_strategy" "${SEED}" "use_concat_baseline=false,film_scale=0.3,egnn_query_layer_index=1,egnn_use_edge_film=true"
done

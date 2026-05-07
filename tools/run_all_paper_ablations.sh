#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-abs}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-200}"
EQF_EPOCHS="${EQF_EPOCHS:-80}"

echo "=== 1) Across backbones Full/Off ==="
TARGET="${TARGET}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS}" \
    bash tools/ablation_suites/run_main_full_off.sh

echo "=== 2) EGNN component ablations (nonlocal off) ==="
TARGET="${TARGET}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS}" \
    bash tools/ablation_suites/run_egnn_component_table2.sh

echo "=== 3) VNode cross-backbone (EGNN + PaiNN) ==="
TARGET="${TARGET}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS}" \
    bash tools/ablation_suites/run_vnode_cross_backbone.sh

echo "=== 4) Conditioning ablations (MGIL Full fixed) ==="
TARGET="${TARGET}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS}" \
    bash tools/ablation_suites/run_conditioning_ablations.sh

echo "=== 5) Nonlocal contribution (Equiformer v1/v2) ==="
TARGET="${TARGET}" SEEDS="${SEEDS}" EPOCHS="${EQF_EPOCHS}" \
    bash tools/ablation_suites/run_equiformer_nonlocal_ablation.sh

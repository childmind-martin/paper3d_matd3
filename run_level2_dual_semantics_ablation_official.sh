#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"

# Three-run MATD3 dual-semantics ablation:
#   1) full dual replay/critic/target semantics
#   2) collapsed replay that stores only corrected executed actions
#   3) dual replay with corrected target reconstruction disabled
export RUN_TAG="${RUN_TAG:-level2_dual_semantics_ablation}"
export LABELS_OVERRIDE="${LABELS_OVERRIDE:-matd3_full_dual_semantic matd3_collapsed_replay matd3_no_corrected_target_reconstruction}"

exec bash "$ROOT_DIR/run_level2_multiseed_all_algos_official.sh" "$@"

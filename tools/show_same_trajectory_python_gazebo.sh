#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-python_original}"
PYTHON_BIN="${PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/results/gazebo_apf_validation/same_trajectory_replay}"
OPEN_HTML="${OPEN_HTML:-0}"
LAUNCH_GAZEBO="${LAUNCH_GAZEBO:-1}"

SCENARIO_JSON="$ROOT_DIR/results/gazebo_apf_validation/single_laggard_matrix_20260625_193939/E1_adaptive_goal_floor/gazebo_live_ep001/scenario.json"
BASE_WORLD_SDF="$ROOT_DIR/results/gazebo_apf_validation/single_laggard_matrix_20260625_193939/E1_adaptive_goal_floor/gazebo_live_ep001/world.sdf"

case "$MODE" in
  python_original)
    RESULTS_JSON="$ROOT_DIR/results/gazebo_apf_validation/python_vs_gazebo_python_20260625_202857_python_only/evaluation_results.json"
    LABEL="python_original_same_trajectory"
    OUTPUT_DIR="$OUTPUT_ROOT/python_original_seed88"
    ;;
  gazebo_authoritative)
    RESULTS_JSON="$ROOT_DIR/results/gazebo_apf_validation/single_laggard_matrix_20260625_193939/E1_adaptive_goal_floor/evaluation_results.json"
    LABEL="gazebo_authoritative_same_trajectory"
    OUTPUT_DIR="$OUTPUT_ROOT/gazebo_authoritative_seed88"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [python_original|gazebo_authoritative]" >&2
    exit 2
    ;;
esac

set +u
source /opt/ros/jazzy/setup.bash
set -u
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matd3_mpl_config}"
export HOME="${HOME:-/tmp/matd3_gz_home_same_trajectory}"
mkdir -p "$HOME" "$MPLCONFIGDIR" "$OUTPUT_DIR"

ARGS=(
  "$ROOT_DIR/tools/show_same_trajectory_in_python_and_gazebo.py"
  --results-json "$RESULTS_JSON"
  --scenario-json "$SCENARIO_JSON"
  --base-world-sdf "$BASE_WORLD_SDF"
  --output-dir "$OUTPUT_DIR"
  --label "$LABEL"
  --render-engine ogre
)

if [[ "$OPEN_HTML" == "1" || "$OPEN_HTML" == "true" || "$OPEN_HTML" == "yes" ]]; then
  ARGS+=(--open-html)
fi

if [[ "$LAUNCH_GAZEBO" == "1" || "$LAUNCH_GAZEBO" == "true" || "$LAUNCH_GAZEBO" == "yes" ]]; then
  ARGS+=(--launch-gazebo)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"

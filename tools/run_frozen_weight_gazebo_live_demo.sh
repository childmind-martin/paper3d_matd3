#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/matd3_full_dual_semantic__seed909__batch_groupB_seed909_20260514_143240_20260516_101844/best_by_team_sr}"
POSITIONS_FILE="${POSITIONS_FILE:-$ROOT_DIR/saved_positions/strict_ablation_seed88_groupB.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/results/gazebo_apf_validation/frozen_weight_live_demo_${RUN_TAG}}"
LIVE_EXPORT_DIR="${GAZEBO_LIVE_EXPORT_DIR:-$RUN_ROOT/gazebo_live_ep001}"
VALIDATION_DIR="${GAZEBO_LIVE_VALIDATION_DIR:-$RUN_ROOT/gazebo_live_validation}"
SAVE_VIZ_DIR="${SAVE_VIZ_DIR:-$RUN_ROOT/eval_outputs}"
RUN_LOG="$RUN_ROOT/run.log"
WORLD_SDF="$LIVE_EXPORT_DIR/world_live_weight_test.sdf"
MODEL_PARENT_DIR="$LIVE_EXPORT_DIR/models"
GUI_CONFIG="${GUI_CONFIG:-$ROOT_DIR/tools/gazebo_minimal_ogre_gui.config}"
LAUNCH_GUI="${LAUNCH_GUI:-1}"
GUI_WAIT_SECONDS="${GUI_WAIT_SECONDS:-4}"
TAIL_LOG="${TAIL_LOG:-1}"

mkdir -p "$RUN_ROOT" "$LIVE_EXPORT_DIR" "$VALIDATION_DIR" "$SAVE_VIZ_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model dir not found: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$POSITIONS_FILE" ]]; then
  echo "Positions file not found: $POSITIONS_FILE" >&2
  exit 2
fi

export MODEL_PATH="$MODEL_DIR"
export EVAL_BACKEND=gazebo_live
export APF_BACKEND=gazebo_apf
export GAZEBO_LIVE_EXPORT_DIR="$LIVE_EXPORT_DIR"
export GAZEBO_LIVE_VALIDATION_DIR="$VALIDATION_DIR"
export GAZEBO_APF_VALIDATION_DIR="$RUN_ROOT/gazebo_apf"

export TERRAIN_SEED_SEQUENCE="${TERRAIN_SEED_SEQUENCE:-88}"
export TERRAIN_VARIANT_SEED_SEQUENCE="${TERRAIN_VARIANT_SEED_SEQUENCE:-387032}"
export OBSTACLE_SEED_SEQUENCE="${OBSTACLE_SEED_SEQUENCE:-12088}"
export SCENARIO_SEED="${SCENARIO_SEED:-88}"
export TERRAIN_BASE_SEED="${TERRAIN_BASE_SEED:-88}"

export GAZEBO_LIVE_SYNC=1
export GAZEBO_LIVE_AUTOLAUNCH=1
export GAZEBO_LIVE_AUTOLAUNCH_START=1
export GAZEBO_LIVE_AUTOLAUNCH_RUN=1
export GAZEBO_LIVE_AUTOLAUNCH_GUI=0
export GAZEBO_LIVE_REQUIRED=1
export GAZEBO_LIVE_AUTOLAUNCH_REQUIRED=1
export GAZEBO_LIVE_STATE_FEEDBACK_REQUIRED=1
export GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK_REQUIRED=1
export GAZEBO_LIVE_SCENE_CHECK_REQUIRED=1
export GAZEBO_LIVE_CONSISTENCY_MODE=gazebo_authoritative
export GAZEBO_LIVE_SEMANTIC_MODE=transfer_equivalence
export GAZEBO_LIVE_CONTROL_MODE=velocity
export GAZEBO_LIVE_STATE_FEEDBACK=1
export GAZEBO_LIVE_AUTHORITATIVE_FEEDBACK=1
export GAZEBO_LIVE_FEEDBACK_VELOCITY_MODE=clamp
export GAZEBO_LIVE_FEEDBACK_ACCELERATION_MODE=estimate
export GAZEBO_LIVE_CONTACT_FEEDBACK=1
export GAZEBO_LIVE_CONTACT_AUTHORITATIVE=1
export GAZEBO_LIVE_CONTACT_MARKS_COLLISION=1
export GAZEBO_LIVE_CONTACT_TERMINATES=1

export GAZEBO_LIVE_WAIT_ACK="${GAZEBO_LIVE_WAIT_ACK:-1}"
export GAZEBO_LIVE_PRE_STEP_SLEEP_MS="${GAZEBO_LIVE_PRE_STEP_SLEEP_MS:-20}"
export GAZEBO_LIVE_POST_STEP_SLEEP_MS="${GAZEBO_LIVE_POST_STEP_SLEEP_MS:-20}"
export GAZEBO_LIVE_WALL_TIME_STEP_MS="${GAZEBO_LIVE_WALL_TIME_STEP_MS:-80}"
export GAZEBO_LIVE_COMMAND_SLEEP="${GAZEBO_LIVE_COMMAND_SLEEP:-0.0}"

export GAZEBO_LIVE_OBSTACLE_SAFETY_MODE="${GAZEBO_LIVE_OBSTACLE_SAFETY_MODE:-velocity_filter_goal_projection_recovery}"
export GAZEBO_LIVE_ADAPTIVE_GOAL_FLOOR_BY_REMAINING_TIME="${GAZEBO_LIVE_ADAPTIVE_GOAL_FLOOR_BY_REMAINING_TIME:-1}"
export GAZEBO_LIVE_MIN_GOAL_PROGRESS_FLOOR="${GAZEBO_LIVE_MIN_GOAL_PROGRESS_FLOOR:-0.5}"
export GAZEBO_LIVE_MAX_GOAL_PROGRESS_FLOOR="${GAZEBO_LIVE_MAX_GOAL_PROGRESS_FLOOR:-2.0}"
export GAZEBO_LIVE_GOAL_PROGRESS_FINISH_MARGIN="${GAZEBO_LIVE_GOAL_PROGRESS_FINISH_MARGIN:-0.3}"

echo "Frozen-weight Gazebo-authoritative demo"
echo "  model:      $MODEL_DIR"
echo "  output:     $RUN_ROOT"
echo "  live world: $WORLD_SDF"
echo "  log:        $RUN_LOG"

"$PYTHON_BIN" evaluate_optimized.py \
  --load-model-path "$MODEL_DIR" \
  --eval-backend gazebo_live \
  --validation-output-dir "$VALIDATION_DIR" \
  --eval-episodes 1 \
  --eval-episode-parallelism 1 \
  --eval-env-step-threads 1 \
  --save-viz-path "$SAVE_VIZ_DIR" \
  --scenario-name paper3d_terrain_energy \
  --episode-length 2800 \
  --algorithm matd3 \
  --apf-backend gazebo_apf \
  --terrain-seed 88 \
  --terrain-complexity-level 3 \
  --use-fixed-positions \
  --positions-file "$POSITIONS_FILE" \
  --disable-gif \
  >"$RUN_LOG" 2>&1 &

EVAL_PID=$!
TAIL_PID=""

if [[ "$TAIL_LOG" == "1" || "$TAIL_LOG" == "true" || "$TAIL_LOG" == "yes" ]]; then
  tail -n +1 -f "$RUN_LOG" &
  TAIL_PID=$!
fi

for _ in $(seq 1 180); do
  if [[ -f "$WORLD_SDF" ]]; then
    break
  fi
  if ! kill -0 "$EVAL_PID" 2>/dev/null; then
    [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
    echo "Evaluation exited before Gazebo world was exported. See: $RUN_LOG" >&2
    wait "$EVAL_PID"
    exit 1
  fi
  sleep 1
done

if [[ ! -f "$WORLD_SDF" ]]; then
  [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
  echo "Timed out waiting for Gazebo live world: $WORLD_SDF" >&2
  exit 1
fi

if [[ "$LAUNCH_GUI" == "1" || "$LAUNCH_GUI" == "true" || "$LAUNCH_GUI" == "yes" ]]; then
  sleep "$GUI_WAIT_SECONDS"
  mkdir -p /tmp/matd3_gz_home_frozen_demo
  (
    unset LD_LIBRARY_PATH PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
    unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH OPENCV_QT_PLUGIN_PATH
    unset QT_XCB_GL_INTEGRATION QT_OPENGL QT_QUICK_BACKEND LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
    export DISPLAY="${DISPLAY:-:0}"
    unset WAYLAND_DISPLAY
    export QT_QPA_PLATFORM=xcb
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
    export HOME=/tmp/matd3_gz_home_frozen_demo
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    export GZ_SIM_RESOURCE_PATH="$MODEL_PARENT_DIR:${GZ_SIM_RESOURCE_PATH:-}"
    exec gz sim -g -v 4 \
      --gui-config "$GUI_CONFIG" \
      --render-engine-gui ogre \
      "$WORLD_SDF"
  ) >"$RUN_ROOT/gz_gui.stdout.log" 2>"$RUN_ROOT/gz_gui.stderr.log" &
  echo "Gazebo GUI started. Logs:"
  echo "  $RUN_ROOT/gz_gui.stdout.log"
  echo "  $RUN_ROOT/gz_gui.stderr.log"
fi

set +e
wait "$EVAL_PID"
EVAL_RC=$?
[[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
set -e

echo
echo "Demo finished with exit code $EVAL_RC"
echo "Results:"
echo "  $RUN_ROOT"
echo "  $VALIDATION_DIR/summary.json"
echo "  $VALIDATION_DIR/episode_metrics.csv"
exit "$EVAL_RC"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-level2_dual_semantics_ablation}"
RUN_TAG="${RUN_TAG:-level2_dual_semantics_eval10x30}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TRAIN_SEEDS=(${TRAIN_SEEDS:-101 202 936487})
EVAL_SEEDS=(${EVAL_SEEDS:-30088 30188 30288 30388 30488 30588 30688 30788 30888 30988})
LABELS=(${LABELS_OVERRIDE:-matd3_full_dual_semantic matd3_collapsed_replay matd3_no_corrected_target_reconstruction})
MODEL_VARIANT="${MODEL_VARIANT:-best_by_team_sr}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-fixed}"
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EVAL_FR_MODE="${EVAL_FR_MODE:-checkpoint}"
FORCE_EVAL_ACTION_FORCE_RATIO_VALUE="${FORCE_EVAL_ACTION_FORCE_RATIO_VALUE:-0.50}"
SUMMARY_FILENAME_TAG="${SUMMARY_FILENAME_TAG:-}"
if [ -z "$SUMMARY_FILENAME_TAG" ]; then
  case "${EVAL_FR_MODE,,}" in
    checkpoint|checkpoint_fr|model|model_fr|corresponding|corresponding_fr)
      SUMMARY_FILENAME_TAG="model_fr"
      ;;
  esac
fi
PYTHON_BIN="${PYTHON_BIN:-${EVAL_PYTHON_BIN:-${TRAIN_PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}}}"
CONDA_SH="${CONDA_SH:-/home/tang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-maddpg_env}"

BASE_SPEC="${BASE_SPEC:-/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/seed_batches/batch_groupB_seed101_20260331_220752/results/post_eval_shared_spec.json}"
SPEC_ROOT="${SPEC_ROOT:-/home/tang/matd3/ablation_experiments/${RUN_TAG}_specs}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-/home/tang/matd3/parallel_logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_LOG_ROOT" "$SPEC_ROOT"

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

model_dirs_newest_first() {
  local label="$1"
  local train_seed="$2"
  find "$ROOT_DIR/models" -maxdepth 1 -type d -name "${TRAIN_RUN_TAG}_${label}_seed${train_seed}_*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk '{ $1=""; sub(/^ /, ""); print }'
}

model_completed() {
  local model_dir="$1"
  local train_seed="$2"
  "$PYTHON_BIN" - "$model_dir" 1000 "$train_seed" <<'PY'
import sys
from pathlib import Path

from experiment_runtime_config import training_unit_completion_errors

model_dir = Path(sys.argv[1]).resolve()
errors = training_unit_completion_errors(
    model_dir,
    int(sys.argv[2]),
    repo_root=model_dir.parent.parent,
    expected_agents=3,
    expected_seed=int(sys.argv[3]),
)
raise SystemExit(1 if errors else 0)
PY
}

latest_completed_model_dir() {
  local label="$1"
  local train_seed="$2"
  local candidate
  while IFS= read -r candidate; do
    if [ -n "$candidate" ] && model_completed "$candidate" "$train_seed"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(model_dirs_newest_first "$label" "$train_seed")
  return 1
}

model_has_variant() {
  local model_dir="$1"
  local variant="$2"
  find "$model_dir/$variant" -maxdepth 1 -type f -name 'actor_*.weights.h5' -print -quit 2>/dev/null | grep -q .
}

eval_completed() {
  local eval_dir="$1"
  local model_dir="$2"
  "$PYTHON_BIN" - "$eval_dir/evaluation_results.json" "$EVAL_EPISODES" "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
model_dir = Path(sys.argv[3])
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
summary = data.get("summary", {}) if isinstance(data, dict) else {}
try:
    episodes = int(summary.get("episodes", data.get("episodes")) or 0)
except Exception:
    raise SystemExit(1)
has_selection = isinstance(data.get("model_selection"), dict)
setup = data.get("evaluation_setup", {}) if isinstance(data, dict) else {}
if not isinstance(setup, dict):
    setup = {}
fr_source = str(setup.get("action_force_ratio_source", "") or "").strip()
fr_source_ok = bool(fr_source) and fr_source != "forced_override"

def to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None

def to_float(value):
    try:
        return float(value)
    except Exception:
        return None

def load_training_results() -> dict:
    candidates = [model_dir / "results.json"]
    log_dir = Path("/home/tang/matd3/logs") / model_dir.name
    try:
        candidates.extend(sorted(log_dir.glob("**/results.json")))
    except Exception:
        pass
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}

physics_ok = True
model_data = load_training_results()
training_args = model_data.get("args", {}) if isinstance(model_data.get("args"), dict) else model_data
for key in (
    "gravity",
    "control_accel_gain",
    "agent_max_speed",
    "agent_accel",
    "damping",
    "simulation_dt",
    "quadrotor_attitude_response_time",
    "quadrotor_psi_cmd",
    "action_range_x",
    "action_range_y",
    "action_range_z",
):
    expected = to_float(training_args.get(key))
    actual = to_float(setup.get(key))
    if expected is not None and (actual is None or abs(actual - expected) > 1e-6):
        physics_ok = False
expected_quad = to_bool(training_args.get("use_quadrotor_dynamics"))
actual_quad = to_bool(setup.get("use_quadrotor_dynamics"))
if expected_quad is not None and actual_quad != expected_quad:
    physics_ok = False
if not training_args or expected_quad is None:
    physics_ok = False

raise SystemExit(0 if episodes == target and has_selection and fr_source_ok and physics_ok else 1)
PY
}

prepare_specs() {
  "$PYTHON_BIN" "$ROOT_DIR/prepare_level2_eval_seed_specs.py" \
    --base-spec "$BASE_SPEC" \
    --output-root "$SPEC_ROOT" \
    --episodes "$EVAL_EPISODES" \
    --eval-seeds "${EVAL_SEEDS[@]}"
}

run_eval_job() {
  local label="$1"
  local train_seed="$2"
  local eval_seed="$3"
  local model_dir="$4"
  local spec="$SPEC_ROOT/eval_seed_${eval_seed}/post_eval_shared_spec.json"
  local eval_dir="$ROOT_DIR/logs/${RUN_TAG}_${label}_trainseed${train_seed}_testseed${eval_seed}/evaluation_official"
  local log_file="$RUN_LOG_ROOT/${label}_trainseed${train_seed}_testseed${eval_seed}.log"
  local force_args=()
  local eval_env_cmd=(env)

  if truthy "$FORCE_RERUN"; then
    force_args+=(--force-rerun)
  elif eval_completed "$eval_dir" "$model_dir"; then
    echo "[skip-eval] $label trainseed${train_seed} testseed${eval_seed}: complete"
    return 0
  else
    force_args+=(--force-rerun)
  fi

  (
    if [ -f "$CONDA_SH" ]; then
      # Activate the same runtime environment used by the original training/eval terminals.
      # Calling the env Python directly is not enough for TensorFlow GPU libraries on WSL.
      # shellcheck source=/home/tang/miniconda3/etc/profile.d/conda.sh
      source "$CONDA_SH"
      conda activate "$CONDA_ENV_NAME"
      PYTHON_BIN="$(command -v python3)"
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    fi
    export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
    case "${EVAL_FR_MODE,,}" in
      checkpoint|checkpoint_fr|model|model_fr|corresponding|corresponding_fr)
        eval_env_cmd+=( -u FORCE_EVAL_ACTION_FORCE_RATIO -u ACTION_FORCE_RATIO )
        ;;
      forced|fixed|fixed_fr)
        eval_env_cmd+=( "FORCE_EVAL_ACTION_FORCE_RATIO=${FORCE_EVAL_ACTION_FORCE_RATIO_VALUE}" )
        ;;
      *)
        echo "[eval-error] unknown EVAL_FR_MODE=$EVAL_FR_MODE (expected forced or checkpoint)" >&2
        exit 2
        ;;
    esac
    "${eval_env_cmd[@]}" \
      OFFICIAL_EVAL_QUIET=1 \
      QUIET_OUTPUT=1 \
      TQDM_DISABLE=1 \
      EVAL_ARTIFACT_FILENAME_TAG="$SUMMARY_FILENAME_TAG" \
      "$PYTHON_BIN" "$ROOT_DIR/official_eval_with_matched_validation.py" \
        --experiment-root "$model_dir" \
        --official-spec "$spec" \
        --output-dir "$eval_dir" \
        --model-variant "$MODEL_VARIANT" \
        --selection-protocol "$SELECTION_PROTOCOL" \
        --quiet-output 1 \
        --python-bin "$PYTHON_BIN" \
        "${force_args[@]}"
  ) 2>&1 | sed -u "s/^/[${label} trainseed${train_seed} testseed${eval_seed}] /" | tee "$log_file" &

  echo "[launch-eval] $label trainseed${train_seed} testseed${eval_seed} -> $log_file"
}

preflight_checks() {
  local errors=0
  local label
  local train_seed
  local eval_seed
  local model_dir
  local spec

  is_positive_int "$EVAL_EPISODES" || { echo "[preflight-error] EVAL_EPISODES must be positive: $EVAL_EPISODES" >&2; errors=$((errors + 1)); }
  is_positive_int "$MAX_PARALLEL" || { echo "[preflight-error] MAX_PARALLEL must be positive: $MAX_PARALLEL" >&2; errors=$((errors + 1)); }
  [ -f "$BASE_SPEC" ] || { echo "[preflight-error] BASE_SPEC missing: $BASE_SPEC" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/official_eval_with_matched_validation.py" ] || { echo "[preflight-error] official_eval_with_matched_validation.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/prepare_level2_eval_seed_specs.py" ] || { echo "[preflight-error] prepare_level2_eval_seed_specs.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/summarize_level2_dual_semantics_eval_sweep.py" ] || { echo "[preflight-error] summarize_level2_dual_semantics_eval_sweep.py missing" >&2; errors=$((errors + 1)); }
  [ -x "$PYTHON_BIN" ] || { echo "[preflight-error] PYTHON_BIN not executable: $PYTHON_BIN" >&2; errors=$((errors + 1)); }
  case "${EVAL_FR_MODE,,}" in
    checkpoint|checkpoint_fr|model|model_fr|corresponding|corresponding_fr|forced|fixed|fixed_fr) ;;
    *) echo "[preflight-error] unknown EVAL_FR_MODE=$EVAL_FR_MODE (expected forced or checkpoint)" >&2; errors=$((errors + 1)); ;;
  esac

  for eval_seed in "${EVAL_SEEDS[@]}"; do
    spec="$SPEC_ROOT/eval_seed_${eval_seed}/post_eval_shared_spec.json"
    [ -f "$spec" ] || { echo "[preflight-error] eval spec missing after preparation: $spec" >&2; errors=$((errors + 1)); }
  done

  for label in "${LABELS[@]}"; do
    for train_seed in "${TRAIN_SEEDS[@]}"; do
      model_dir="$(latest_completed_model_dir "$label" "$train_seed" || true)"
      if [ -z "$model_dir" ]; then
        echo "[preflight-error] completed model missing: $label trainseed${train_seed}" >&2
        errors=$((errors + 1))
        continue
      fi
      if ! model_has_variant "$model_dir" "$MODEL_VARIANT"; then
        echo "[preflight-error] model variant missing: $MODEL_VARIANT | $model_dir" >&2
        errors=$((errors + 1))
      fi
    done
  done

  if [ "$errors" -ne 0 ]; then
    echo "[preflight] failed with $errors error(s)." >&2
    return 1
  fi
  echo "[preflight] ok: specs and completed $MODEL_VARIANT models are available."
  if [[ "${EVAL_FR_MODE,,}" == checkpoint* || "${EVAL_FR_MODE,,}" == model* || "${EVAL_FR_MODE,,}" == corresponding* ]]; then
    echo "[preflight] FR mode: checkpoint-consistent, FORCE_EVAL_ACTION_FORCE_RATIO will be unset."
  else
    echo "[preflight] FR mode: forced, FORCE_EVAL_ACTION_FORCE_RATIO=${FORCE_EVAL_ACTION_FORCE_RATIO_VALUE}."
  fi
}

prepare_specs
preflight_checks
if truthy "$PREFLIGHT_ONLY"; then
  echo "[preflight] PREFLIGHT_ONLY=1, exiting before launching eval jobs."
  exit 0
fi

for label in "${LABELS[@]}"; do
  for train_seed in "${TRAIN_SEEDS[@]}"; do
    model_dir="$(latest_completed_model_dir "$label" "$train_seed")"
    for eval_seed in "${EVAL_SEEDS[@]}"; do
      while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
      done
      run_eval_job "$label" "$train_seed" "$eval_seed" "$model_dir"
    done
  done
done

wait
echo "10-seed x 30-episode eval sweep complete. Logs: $RUN_LOG_ROOT"

SUMMARY_FLAGS=()
if truthy "$STRICT_SUMMARY"; then
  SUMMARY_FLAGS+=(--strict)
fi
if [ -n "$SUMMARY_FILENAME_TAG" ]; then
  SUMMARY_FLAGS+=(--filename-tag "$SUMMARY_FILENAME_TAG")
fi

MPLCONFIGDIR=/tmp/mplconfig "$PYTHON_BIN" "$ROOT_DIR/summarize_level2_dual_semantics_eval_sweep.py" \
  --run-tag "$RUN_TAG" \
  --labels "${LABELS[@]}" \
  --train-seeds "${TRAIN_SEEDS[@]}" \
  --eval-seeds "${EVAL_SEEDS[@]}" \
  --expected-episodes "$EVAL_EPISODES" \
  "${SUMMARY_FLAGS[@]}"

echo "10-seed x 30-episode eval sweep summary complete."

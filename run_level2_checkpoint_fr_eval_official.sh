#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-level2_ms_official}"
RUN_TAG="${RUN_TAG:-level2_ms_checkpoint_fr_eval}"
TRAIN_EPISODES="${TRAIN_EPISODES:-1000}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SEEDS=(${SEEDS:-101 202 936487})
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SUMMARY_FILENAME_TAG="${SUMMARY_FILENAME_TAG:-model_fr}"
MODEL_VARIANT="${MODEL_VARIANT:-best_by_team_sr}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-matched_validation}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-10}"
VALIDATION_CANDIDATES="${VALIDATION_CANDIDATES:-best_by_team_sr,best,checkpoint,final,latest_ep}"
PYTHON_BIN="${PYTHON_BIN:-${EVAL_PYTHON_BIN:-${TRAIN_PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}}}"
CONDA_SH="${CONDA_SH:-/home/tang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-maddpg_env}"

SPEC_ROOT="${SPEC_ROOT:-/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/seed_batches}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-/home/tang/matd3/parallel_logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_LOG_ROOT"

if [ -n "${LABELS_OVERRIDE:-}" ]; then
  read -r -a LABELS <<< "$LABELS_OVERRIDE"
else
  LABELS=(
    matd3_single_q
    matd3_dual_q
    matd3_separated_gradient
    matd3_separated_hybrid_actor
    matd3_separated_hybrid_actor_alpha20
    maddpg_baseline
    maddpg_dual_q
    maddpg_separated_gradient
    mappo_baseline
    mappo_fusion_only
    mappo_separated_gradient
  )
fi

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
  local seed="$2"
  find "$ROOT_DIR/models" -maxdepth 1 -type d -name "${TRAIN_RUN_TAG}_${label}_seed${seed}_*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk '{ $1=""; sub(/^ /, ""); print }'
}

model_completed() {
  local model_dir="$1"
  python3 - "$model_dir" "$TRAIN_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
target = int(sys.argv[2])
episodes = []

results_path = model_dir / "results.json"
if results_path.exists():
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        episodes.append(int(data.get("episodes", 0) or 0))
    except Exception:
        pass

for state_path in model_dir.glob("*/checkpoint_state.json"):
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    candidates = [int(state.get("episode", 0) or 0)]
    for key in ("episode_rewards", "episode_force_ratios", "team_success_flags", "success_flags"):
        value = state.get(key)
        if isinstance(value, list):
            candidates.append(len(value))
    episodes.append(max(candidates))

raise SystemExit(0 if episodes and max(episodes) >= target else 1)
PY
}

latest_completed_model_dir() {
  local label="$1"
  local seed="$2"
  local candidate
  while IFS= read -r candidate; do
    if [ -n "$candidate" ] && model_completed "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(model_dirs_newest_first "$label" "$seed")
  return 1
}

model_has_eval_candidate() {
  local model_dir="$1"
  find "$model_dir" -mindepth 2 -maxdepth 2 -type f -name 'actor_*.weights.h5' -print -quit 2>/dev/null | grep -q .
}

eval_completed() {
  local eval_dir="$1"
  python3 - "$eval_dir/evaluation_results.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
selection = data.get("model_selection")
setup = data.get("evaluation_setup", {})
source = setup.get("action_force_ratio_source") if isinstance(setup, dict) else None
raise SystemExit(0 if isinstance(selection, dict) and source != "forced_override" else 1)
PY
}

eval_dir_for() {
  local label="$1"
  local seed="$2"
  local model_dir="$3"
  local stamp
  stamp="$(basename "$model_dir")"
  stamp="${stamp#${TRAIN_RUN_TAG}_${label}_seed${seed}_}"
  printf '%s/logs/%s_%s_seed%s_%s/evaluation_official\n' "$ROOT_DIR" "$RUN_TAG" "$label" "$seed" "$stamp"
}

run_eval_job() {
  local label="$1"
  local seed="$2"
  local model_dir="$3"
  local spec="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
  local eval_dir
  local log_file
  local force_args=()
  eval_dir="$(eval_dir_for "$label" "$seed" "$model_dir")"
  log_file="${RUN_LOG_ROOT}/${label}_seed${seed}_checkpoint_fr_eval.log"

  if truthy "$FORCE_RERUN"; then
    force_args+=(--force-rerun)
  elif eval_completed "$eval_dir"; then
    echo "[skip-eval] $label seed${seed}: checkpoint-FR eval complete"
    return 0
  fi

  (
    if [ -f "$CONDA_SH" ]; then
      # shellcheck source=/home/tang/miniconda3/etc/profile.d/conda.sh
      source "$CONDA_SH"
      conda activate "$CONDA_ENV_NAME"
      PYTHON_BIN="$(command -v python3)"
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    fi
    export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
    env -u FORCE_EVAL_ACTION_FORCE_RATIO -u ACTION_FORCE_RATIO \
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
        --validation-episodes "$VALIDATION_EPISODES" \
        --validation-candidates "$VALIDATION_CANDIDATES" \
        --quiet-output 1 \
        --python-bin "$PYTHON_BIN" \
        "${force_args[@]}"
  ) 2>&1 | sed -u "s/^/[${label} seed${seed} checkpoint-fr] /" | tee "$log_file" &

  echo "[launch-eval] $label seed${seed} -> $log_file"
}

preflight_checks() {
  local errors=0
  local label
  local seed
  local model_dir
  local spec

  is_positive_int "$TRAIN_EPISODES" || { echo "[preflight-error] TRAIN_EPISODES must be positive: $TRAIN_EPISODES" >&2; errors=$((errors + 1)); }
  is_positive_int "$MAX_PARALLEL" || { echo "[preflight-error] MAX_PARALLEL must be positive: $MAX_PARALLEL" >&2; errors=$((errors + 1)); }
  is_positive_int "$VALIDATION_EPISODES" || { echo "[preflight-error] VALIDATION_EPISODES must be positive: $VALIDATION_EPISODES" >&2; errors=$((errors + 1)); }
  [ -x "$PYTHON_BIN" ] || { echo "[preflight-error] PYTHON_BIN not executable: $PYTHON_BIN" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/official_eval_with_matched_validation.py" ] || { echo "[preflight-error] official_eval_with_matched_validation.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/run_evaluation.sh" ] || { echo "[preflight-error] run_evaluation.sh missing" >&2; errors=$((errors + 1)); }

  for seed in "${SEEDS[@]}"; do
    spec="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
    [ -f "$spec" ] || { echo "[preflight-error] official spec missing: $spec" >&2; errors=$((errors + 1)); }
  done

  for label in "${LABELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      model_dir="$(latest_completed_model_dir "$label" "$seed" || true)"
      if [ -z "$model_dir" ]; then
        echo "[preflight-error] completed model missing: $label seed${seed}" >&2
        errors=$((errors + 1))
        continue
      fi
      if ! model_has_eval_candidate "$model_dir"; then
        echo "[preflight-error] no actor weights under model dir: $model_dir" >&2
        errors=$((errors + 1))
      fi
    done
  done

  if [ "$errors" -ne 0 ]; then
    echo "[preflight] failed with $errors error(s)." >&2
    return 1
  fi

  echo "[preflight] ok: completed models and official specs are available."
  echo "[preflight] checkpoint-FR mode: FORCE_EVAL_ACTION_FORCE_RATIO is unset during evaluation."
}

preflight_checks
if truthy "$PREFLIGHT_ONLY"; then
  echo "[preflight] PREFLIGHT_ONLY=1, exiting before launching eval jobs."
  exit 0
fi

for label in "${LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    model_dir="$(latest_completed_model_dir "$label" "$seed")"
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n
    done
    run_eval_job "$label" "$seed" "$model_dir"
  done
done

wait
echo "Checkpoint-FR official eval complete. Logs: $RUN_LOG_ROOT"

SUMMARY_FLAGS=()
if truthy "$STRICT_SUMMARY"; then
  SUMMARY_FLAGS+=(--strict)
fi

for seed in "${SEEDS[@]}"; do
  MPLCONFIGDIR=/tmp/mplconfig python3 "$ROOT_DIR/summarize_level2_official_eval.py" \
    --run-tag "$RUN_TAG" \
    --filename-tag "$SUMMARY_FILENAME_TAG" \
    --seed "$seed" \
    --labels "${LABELS[@]}" \
    "${SUMMARY_FLAGS[@]}"
done

MPLCONFIGDIR=/tmp/mplconfig python3 "$ROOT_DIR/summarize_level2_official_eval_multiseed.py" \
  --run-tag "$RUN_TAG" \
  --filename-tag "$SUMMARY_FILENAME_TAG" \
  --seeds "${SEEDS[@]}" \
  --labels "${LABELS[@]}" \
  "${SUMMARY_FLAGS[@]}"

echo "Checkpoint-FR official eval summaries complete."

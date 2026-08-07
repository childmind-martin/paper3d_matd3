#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-selector_checkpoint_audit_seed101_v11}"
VALIDATION_SEED="${VALIDATION_SEED:-114817}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-30}"
EVAL_MAX_PARALLEL="${EVAL_MAX_PARALLEL:-4}"
EVAL_PROCESS_SHARDS="${EVAL_PROCESS_SHARDS:-3}"
EVAL_PROCESS_WORKERS="${EVAL_PROCESS_WORKERS:-3}"
EVAL_EPISODE_PARALLELISM="${EVAL_EPISODE_PARALLELISM:-4}"
EVAL_ENV_STEP_THREADS="${EVAL_ENV_STEP_THREADS:-4}"
EVAL_SHARD_EPISODE_PARALLELISM="${EVAL_SHARD_EPISODE_PARALLELISM:-4}"
EVAL_SHARD_ENV_STEP_THREADS="${EVAL_SHARD_ENV_STEP_THREADS:-4}"
EVAL_NOISE_SEED="${EVAL_NOISE_SEED:-101}"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[STOP] RUN_ID 只允许字母、数字、点、下划线和连字符: $RUN_ID" >&2
  exit 2
fi
for value_name in VALIDATION_SEED VALIDATION_EPISODES EVAL_MAX_PARALLEL EVAL_PROCESS_SHARDS EVAL_PROCESS_WORKERS EVAL_EPISODE_PARALLELISM EVAL_ENV_STEP_THREADS EVAL_SHARD_EPISODE_PARALLELISM EVAL_SHARD_ENV_STEP_THREADS EVAL_NOISE_SEED; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[STOP] $value_name 必须是正整数: $value" >&2
    exit 2
  fi
done
if [ "$EVAL_PROCESS_WORKERS" -gt "$EVAL_PROCESS_SHARDS" ]; then
  echo "[STOP] EVAL_PROCESS_WORKERS 不能大于 EVAL_PROCESS_SHARDS" >&2
  exit 2
fi
if [[ "$FORCE_RERUN" != "0" && "$FORCE_RERUN" != "1" ]]; then
  echo "[STOP] FORCE_RERUN 只能是 0 或 1: $FORCE_RERUN" >&2
  exit 2
fi
if [ ! -x "$PY" ]; then
  echo "[STOP] Python 不可执行: $PY" >&2
  exit 2
fi

readonly SEED_BATCH_DIR="$ROOT_DIR/ablation_experiments/multi_seed_groupB_20260727_122936/seed_batches/batch_groupB_seed101_20260727_122936"
readonly OFFICIAL_SPEC="$SEED_BATCH_DIR/results/post_eval_shared_spec.json"
readonly OUT_ROOT="$ROOT_DIR/evaluation_results_${RUN_ID}"
readonly STATE_DIR="$ROOT_DIR/selector_experiment_runs/$RUN_ID"
readonly LOG_FILE="$STATE_DIR/checkpoint_audit.log"
readonly STATUS_FILE="$STATE_DIR/status.txt"

mkdir -p "$STATE_DIR" "$OUT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

if [ ! -f "$OFFICIAL_SPEC" ]; then
  echo "[STOP] 缺少正式测试 spec: $OFFICIAL_SPEC" >&2
  exit 3
fi

MODEL_IDS=(M0 M1 M2 M3)
MODEL_ROOTS=(
  "$ROOT_DIR/models/matd3_cross_agent_ref_behavior_label_agent_quality_gate__seed101__batch_groupB_seed101_20260727_122936_20260727_122938"
  "$ROOT_DIR/models/matd3_cross_agent_ref_aqual_split_teacher__seed101__batch_groupB_seed101_20260727_122936_20260727_122940"
  "$ROOT_DIR/models/matd3_cross_agent_ref_adaptive_twin_advantage__seed101__batch_groupB_seed101_20260727_122936_20260727_122942"
  "$ROOT_DIR/models/matd3_cross_agent_ref_shared_twin_advantage_selector__seed101__batch_groupB_seed101_20260727_122936_20260727_122944"
)
MODEL_CANDIDATES=(
  "best_by_team_sr,best,final"
  "best,final"
  "best,final"
  "best,final"
)

write_status() {
  local state="$1"
  local detail="$2"
  local tmp="$STATUS_FILE.tmp.$$"
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'state=%s\n' "$state"
    printf 'detail=%s\n' "$detail"
    printf 'pid=%s\n' "$$"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'output=%s\n' "$OUT_ROOT"
    printf 'log=%s\n' "$LOG_FILE"
  } > "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

finish() {
  local code=$?
  trap - EXIT
  if [ "$code" -eq 0 ]; then
    write_status completed "4/4 models completed held-out selection and deterministic official test"
  else
    write_status failed "checkpoint audit exited with code $code"
  fi
  exit "$code"
}
trap finish EXIT
write_status running "preflight"

PY_ENV_ROOT="$(cd "$(dirname "$PY")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${PY_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MATD3_REQUIRE_GPU=1
export SUPPRESS_MA_PROMPT=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-matd3-checkpoint-audit}"
export EVAL_PYTHON_BIN="$PY"
export EVAL_PROCESS_SHARDS
export EVAL_PROCESS_WORKERS
export EVAL_EPISODE_PARALLELISM
export EVAL_ENV_STEP_THREADS
export EVAL_SHARD_EPISODE_PARALLELISM
export EVAL_SHARD_ENV_STEP_THREADS
export EVAL_NOISE_SEED
export OFFICIAL_EVAL_QUIET="${OFFICIAL_EVAL_QUIET:-1}"
export EVAL_LIGHT_MODE=1
export FAST_ARTIFACTS=1
export SAVE_INTERACTIVE_TRAJ=0
export SAVE_EVAL_ALL_EPISODES=0
export SAVE_BEST_TRAJ=0
export SAVE_TEAM_SUCCESS_HTML=0
export SAVE_EVAL_TRAJECTORY_JSON=0
export SAVE_EVAL_TRAJECTORY_PNG=0
export SAVE_EVAL_ACTOR_SEQUENCE=0
export SAVE_EVAL_CONTROL_DIAGNOSTICS=0
export DISABLE_GIF=1

echo "============================================================"
echo "[Checkpoint Audit] run_id=$RUN_ID"
echo "[Checkpoint Audit] held-out seed=$VALIDATION_SEED episodes=$VALIDATION_EPISODES"
echo "[Checkpoint Audit] official seed=10088 episodes=30"
echo "[Checkpoint Audit] paired evaluator random stream=$EVAL_NOISE_SEED"
echo "[Checkpoint Audit] candidates=9 unique checkpoints"
echo "[Checkpoint Audit] concurrency=$EVAL_MAX_PARALLEL models × $EVAL_PROCESS_SHARDS process shards/model × $EVAL_SHARD_EPISODE_PARALLELISM episode envs/shard"
echo "[Checkpoint Audit] maximum concurrent episode envs=$((EVAL_MAX_PARALLEL * EVAL_PROCESS_SHARDS * EVAL_SHARD_EPISODE_PARALLELISM))"
echo "[Checkpoint Audit] output=$OUT_ROOT"
echo "[Checkpoint Audit] log=$LOG_FILE"
echo "============================================================"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
"$PY" - <<'PY'
import tensorflow as tf
physical = tf.config.list_physical_devices("GPU")
logical = tf.config.list_logical_devices("GPU")
if not physical or not logical:
    raise SystemExit("[STOP] TensorFlow 未检测到可用 GPU")
print("[GPU PASS]", physical, logical)
PY

for idx in "${!MODEL_IDS[@]}"; do
  model_id="${MODEL_IDS[$idx]}"
  model_root="${MODEL_ROOTS[$idx]}"
  candidates="${MODEL_CANDIDATES[$idx]}"
  if [ ! -f "$model_root/results.json" ]; then
    echo "[STOP] $model_id 缺少 results.json: $model_root" >&2
    exit 3
  fi
  IFS=',' read -r -a aliases <<< "$candidates"
  for alias in "${aliases[@]}"; do
    if [ ! -f "$model_root/$alias/actor_0.weights.h5" ]; then
      echo "[STOP] $model_id 缺少候选权重: $model_root/$alias" >&2
      exit 3
    fi
  done
done

run_one() {
  local model_id="$1"
  local model_root="$2"
  local candidates="$3"
  local output_dir="$OUT_ROOT/$model_id/official_deterministic"
  local cmd=(
    "$PY"
    "$ROOT_DIR/official_eval_with_matched_validation.py"
    --experiment-root "$model_root"
    --official-spec "$OFFICIAL_SPEC"
    --output-dir "$output_dir"
    --model-variant auto
    --selection-protocol matched_validation
    --validation-episodes "$VALIDATION_EPISODES"
    --validation-seed "$VALIDATION_SEED"
    --validation-candidates "$candidates"
    --python-bin "$PY"
    --quiet-output "$OFFICIAL_EVAL_QUIET"
  )
  if [ "$FORCE_RERUN" = "1" ]; then
    cmd+=(--force-rerun)
  fi
  echo "[$model_id] START candidates=$candidates"
  "${cmd[@]}" 2>&1 | sed -u "s/^/[$model_id] /"
  echo "[$model_id] COMPLETE"
}

write_status running "held-out selection and deterministic official tests"
active_pids=()
active_ids=()
for idx in "${!MODEL_IDS[@]}"; do
  run_one "${MODEL_IDS[$idx]}" "${MODEL_ROOTS[$idx]}" "${MODEL_CANDIDATES[$idx]}" &
  active_pids+=("$!")
  active_ids+=("${MODEL_IDS[$idx]}")
  if [ "${#active_pids[@]}" -ge "$EVAL_MAX_PARALLEL" ]; then
    for job_idx in "${!active_pids[@]}"; do
      if ! wait "${active_pids[$job_idx]}"; then
        echo "[STOP] ${active_ids[$job_idx]} checkpoint audit failed" >&2
        exit 4
      fi
    done
    active_pids=()
    active_ids=()
  fi
done
for job_idx in "${!active_pids[@]}"; do
  if ! wait "${active_pids[$job_idx]}"; then
    echo "[STOP] ${active_ids[$job_idx]} checkpoint audit failed" >&2
    exit 4
  fi
done

echo "============================================================"
echo "[Checkpoint Audit COMPLETE] 4 个算法均完成 held-out 选模和正式确定性测试"
echo "[Checkpoint Audit COMPLETE] output=$OUT_ROOT"
echo "[Checkpoint Audit COMPLETE] log=$LOG_FILE"
echo "============================================================"

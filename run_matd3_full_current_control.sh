#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-matd3_full_control_seed101_preselector_v1}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[STOP] RUN_ID 只允许字母、数字、点、下划线和连字符: $RUN_ID" >&2
  exit 2
fi
if [ ! -x "$PY" ]; then
  echo "[STOP] Python 不可执行: $PY" >&2
  exit 2
fi

readonly TRAIN_SEED=101
readonly TRAIN_EPISODES=1000
readonly TRAIN_NUM_ENVS=4
readonly TRAIN_BATCH_SIZE=1024
readonly VALIDATION_SEED=114817
readonly EVAL_EPISODES=30
readonly EVAL_NOISE_SEED=101
readonly FULL_LABEL="matd3_full_dual_semantic"
readonly M0_MANIFEST="$ROOT_DIR/ablation_experiments/multi_seed_groupB_20260727_122936/seed_batches/batch_groupB_seed101_20260727_122936/manifests/matd3_cross_agent_ref_behavior_label_agent_quality_gate_resolved_manifest.json"
readonly REFERENCE_RUN_SPEC="$ROOT_DIR/evaluation_results_selector_m0_m3_env4_seed101_v10_formal_gpu_v10/M0/deterministic/run_spec.json"
readonly OFFICIAL_SPEC="$ROOT_DIR/ablation_experiments/multi_seed_groupB_20260727_122936/seed_batches/batch_groupB_seed101_20260727_122936/results/post_eval_shared_spec.json"

STATE_DIR="$ROOT_DIR/selector_experiment_runs/$RUN_ID"
LOG_FILE="$STATE_DIR/full_control.log"
STATUS_FILE="$STATE_DIR/status.txt"
SOURCE_SNAPSHOT="$STATE_DIR/source_and_config_preflight.json"
TRAINING_AUDIT="$STATE_DIR/completed_training_audit.json"
MODEL_ROOT_FILE="$STATE_DIR/model_root.txt"
EVAL_ROOT="$ROOT_DIR/evaluation_results_${RUN_ID}"
EVAL_AUDIT="$EVAL_ROOT/full_control_checkpoint_audit.json"
EVAL_REPORT="$EVAL_ROOT/full_control_checkpoint_audit.md"
mkdir -p "$STATE_DIR"

CURRENT_PHASE="initializing"
CURRENT_DETAIL="pipeline startup"
PIPELINE_STARTED_AT="$(date --iso-8601=seconds)"

write_status() {
  local state="$1"
  local temporary="$STATUS_FILE.tmp.$$"
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'state=%s\n' "$state"
    printf 'phase=%s\n' "$CURRENT_PHASE"
    printf 'detail=%s\n' "$CURRENT_DETAIL"
    printf 'pid=%s\n' "$$"
    printf 'started_at=%s\n' "$PIPELINE_STARTED_AT"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'log=%s\n' "$LOG_FILE"
    printf 'evaluation_root=%s\n' "$EVAL_ROOT"
  } > "$temporary"
  mv "$temporary" "$STATUS_FILE"
}

mark_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DETAIL="$2"
  write_status running
  echo
  echo "[$CURRENT_PHASE] $CURRENT_DETAIL"
}

finish() {
  local code=$?
  trap - EXIT
  if [ "$code" -eq 0 ]; then
    CURRENT_PHASE="complete"
    CURRENT_DETAIL="full control training, checkpoint selection, and official deterministic evaluation completed"
    write_status completed
  else
    CURRENT_DETAIL="pipeline exited with code $code during: $CURRENT_DETAIL"
    write_status failed
  fi
  exit "$code"
}

write_status running
trap finish EXIT
exec > >(tee -a "$LOG_FILE") 2>&1

PY_ENV_ROOT="$(cd "$(dirname "$PY")/.." && pwd)"
export TRAIN_PYTHON_BIN="$PY"
export PYTHON_BIN="$PY"
export EVAL_PYTHON_BIN="$PY"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${PY_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MATD3_REQUIRE_GPU=1
export SUPPRESS_MA_PROMPT=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-matd3-full-control}"
export SAVE_INTERVAL=0

echo "============================================================"
echo "[MATD3 Full Control] run_id=$RUN_ID"
echo "[MATD3 Full Control] train=1 algorithm × $TRAIN_EPISODES synchronous iterations × $TRAIN_NUM_ENVS envs"
echo "[MATD3 Full Control] resume=保留完整 algorithm×seed；不完整单元从 episode 0 重训"
echo "[MATD3 Full Control] selection=3 checkpoints × $EVAL_EPISODES held-out episodes"
echo "[MATD3 Full Control] official=selected checkpoint × $EVAL_EPISODES deterministic episodes"
echo "[MATD3 Full Control] log=$LOG_FILE"
echo "============================================================"

mark_phase "Phase 0" "GPU、源码身份和 Full-vs-M0 配置隔离预检"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
"$PY" - <<'PY'
import tensorflow as tf
physical = tf.config.list_physical_devices("GPU")
logical = tf.config.list_logical_devices("GPU")
if not physical or not logical:
    raise SystemExit("[STOP] TensorFlow 未检测到物理和逻辑 GPU")
print("[GPU PASS]", physical, logical)
PY
"$PY" -m py_compile tools/audit_matd3_full_control.py
bash -n run_matd3_full_current_control.sh
"$PY" tools/audit_matd3_full_control.py preflight \
  --reference-run-spec "$REFERENCE_RUN_SPEC" \
  --output "$SOURCE_SNAPSHOT"

read_or_create_stamp() {
  local stamp_file="$1"
  local stamp=""
  if [ -f "$stamp_file" ]; then
    IFS= read -r stamp < "$stamp_file"
  else
    stamp="$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "$stamp" > "$stamp_file"
  fi
  if [[ ! "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    echo "[STOP] 非法父批次时间标识: $stamp_file -> $stamp" >&2
    return 2
  fi
  printf '%s' "$stamp"
}

PARENT_STAMP="$(read_or_create_stamp "$STATE_DIR/parent_stamp.txt")"
PARENT_BATCH="$ROOT_DIR/ablation_experiments/multi_seed_groupB_$PARENT_STAMP"

mark_phase "Phase 1" "同版 MATD3 full seed101、4环境、1000迭代 GPU 训练"
TRAIN_COMMAND=(
  "$PY"
  ablation_dual_q_separated_gradient.py
  --multi-seed
  --seeds "$TRAIN_SEED"
  --episodes "$TRAIN_EPISODES"
  --parent-run-stamp "$PARENT_STAMP"
  --batch-size "$TRAIN_BATCH_SIZE"
  --num-envs "$TRAIN_NUM_ENVS"
  --max-parallel 1
  --experiment-max-parallel 1
  --worker-launch-stagger-seconds 8
  --experiment-group B
  --config-mode strict_ablation
  --env-isolation strict
  --scenario-seed 88
  --use-weighted-reward 1
  --action-force-ratio 0.50
  --action-force-ratio-schedule-pct
  '0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32'
  --disable-post-eval
  --skip-local-plots
  --experiments "$FULL_LABEL"
)
if [ -f "$PARENT_BATCH/config.json" ]; then
  echo "[RESUME] 发现既有父批次；仅复用通过完整性校验的算法×seed 单元"
  TRAIN_COMMAND+=(--resume-parent-batch-dir "$PARENT_BATCH" --reuse)
elif [ -e "$PARENT_BATCH" ]; then
  echo "[STOP] 父批次目录存在但缺少 config.json: $PARENT_BATCH" >&2
  exit 3
else
  echo "[NEW] parent_batch=$PARENT_BATCH"
fi
printf '[TRAIN COMMAND] '
printf '%q ' "${TRAIN_COMMAND[@]}"
printf '\n'
"${TRAIN_COMMAND[@]}"

mark_phase "Phase 2" "完整训练产物、GPU/4环境证据及关闭 cross-reference 数值隔离终审"
"$PY" tools/audit_matd3_full_control.py completed \
  --parent-batch-dir "$PARENT_BATCH" \
  --m0-manifest "$M0_MANIFEST" \
  --source-snapshot "$SOURCE_SNAPSHOT" \
  --output "$TRAINING_AUDIT" \
  --model-root-file "$MODEL_ROOT_FILE"
if [ ! -s "$MODEL_ROOT_FILE" ]; then
  echo "[STOP] 完整训练审计未输出模型根目录: $MODEL_ROOT_FILE" >&2
  exit 4
fi
IFS= read -r MODEL_ROOT < "$MODEL_ROOT_FILE"
if [ ! -f "$MODEL_ROOT/results.json" ]; then
  echo "[STOP] 模型根目录非法: $MODEL_ROOT" >&2
  exit 4
fi

mark_phase "Phase 3" "best_by_team_sr/best/final 的 30 回合 held-out 选模和正式确定性测试"
export EVAL_PROCESS_SHARDS=3
export EVAL_PROCESS_WORKERS=3
export EVAL_EPISODE_PARALLELISM=4
export EVAL_ENV_STEP_THREADS=4
export EVAL_SHARD_EPISODE_PARALLELISM=4
export EVAL_SHARD_ENV_STEP_THREADS=4
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
"$PY" official_eval_with_matched_validation.py \
  --experiment-root "$MODEL_ROOT" \
  --official-spec "$OFFICIAL_SPEC" \
  --output-dir "$EVAL_ROOT/official_deterministic" \
  --model-variant auto \
  --selection-protocol matched_validation \
  --validation-episodes "$EVAL_EPISODES" \
  --validation-seed "$VALIDATION_SEED" \
  --validation-candidates best_by_team_sr,best,final \
  --python-bin "$PY" \
  --quiet-output "$OFFICIAL_EVAL_QUIET"

mark_phase "Phase 4" "控制组选模、随机流、checkpoint FR 与 30 回合结果独立校验"
"$PY" tools/audit_matd3_full_control.py evaluation \
  --training-audit "$TRAINING_AUDIT" \
  --eval-root "$EVAL_ROOT" \
  --output "$EVAL_AUDIT" \
  --report "$EVAL_REPORT"

printf '%s\n' "$(date --iso-8601=seconds)" > "$STATE_DIR/completed_at.txt"
echo
echo "============================================================"
echo "[MATD3 Full Control COMPLETE]"
echo "parent_batch=$PARENT_BATCH"
echo "model_root=$MODEL_ROOT"
echo "training_audit=$TRAINING_AUDIT"
echo "evaluation_audit=$EVAL_AUDIT"
echo "report=$EVAL_REPORT"
echo "log=$LOG_FILE"
echo "============================================================"

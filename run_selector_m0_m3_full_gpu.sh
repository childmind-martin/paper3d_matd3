#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/home/tang/miniconda3/envs/maddpg_env/bin/python}"
RUN_ID="${RUN_ID:-selector_m0_m3_env4_seed101_v10}"
RUN_PILOT="${RUN_PILOT:-1}"
GPU_ID="${GPU_ID:-0}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[STOP] RUN_ID 只允许字母、数字、点、下划线和连字符: $RUN_ID" >&2
  exit 2
fi
if [[ "$RUN_PILOT" != "0" && "$RUN_PILOT" != "1" ]]; then
  echo "[STOP] RUN_PILOT 只能是 0 或 1" >&2
  exit 2
fi
if [ ! -x "$PY" ]; then
  echo "[STOP] Python 不可执行: $PY" >&2
  exit 2
fi

readonly TRAIN_NUM_ENVS=4
readonly TRAIN_MODEL_MAX_PARALLEL=2
readonly TRAIN_WORKER_LAUNCH_STAGGER_SECONDS=8
readonly PILOT_TRAIN_SEED=9101
readonly PILOT_TRAIN_EPISODES=100
readonly FORMAL_TRAIN_SEED=101
readonly FORMAL_TRAIN_EPISODES=1000
readonly FORMAL_EVAL_EPISODES=30

STATE_DIR="$REPO_ROOT/selector_experiment_runs/$RUN_ID"
mkdir -p "$STATE_DIR"
LOG_FILE="$STATE_DIR/pipeline.log"
STATUS_FILE="$STATE_DIR/status.txt"
PIPELINE_STARTED_AT="$(date --iso-8601=seconds)"
CURRENT_PHASE="initializing"
CURRENT_DETAIL="pipeline startup"

write_status() {
  local state="$1"
  local status_tmp="$STATUS_FILE.tmp.$$"
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'state=%s\n' "$state"
    printf 'phase=%s\n' "$CURRENT_PHASE"
    printf 'detail=%s\n' "$CURRENT_DETAIL"
    printf 'pipeline_pid=%s\n' "$$"
    printf 'started_at=%s\n' "$PIPELINE_STARTED_AT"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'log=%s\n' "$LOG_FILE"
  } > "$status_tmp"
  mv "$status_tmp" "$STATUS_FILE"
}

mark_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DETAIL="$2"
  write_status "running"
  echo "[$CURRENT_PHASE] $CURRENT_DETAIL"
}

finish_status() {
  local exit_code=$?
  trap - EXIT
  if [ "$exit_code" -eq 0 ]; then
    CURRENT_PHASE="complete"
    CURRENT_DETAIL="all requested training and evaluation units completed"
    write_status "completed"
  else
    CURRENT_DETAIL="pipeline exited with code $exit_code during: $CURRENT_DETAIL"
    write_status "failed"
  fi
  exit "$exit_code"
}

write_status "running"
trap finish_status EXIT
exec > >(tee -a "$LOG_FILE") 2>&1

echo
echo "============================================================"
echo "[Selector Pipeline] run_id=$RUN_ID"
echo "[Selector Pipeline] state=$STATE_DIR"
echo "[Selector Pipeline] log=$LOG_FILE"
echo "[Selector Pipeline] train=4 models × ${FORMAL_TRAIN_EPISODES} iterations × ${TRAIN_NUM_ENVS} envs = 16000 trajectories"
echo "[Selector Pipeline] train_parallelism=${TRAIN_MODEL_MAX_PARALLEL} model workers × ${TRAIN_NUM_ENVS} envs/model = $((TRAIN_MODEL_MAX_PARALLEL * TRAIN_NUM_ENVS)) concurrent env trajectories"
echo "[Selector Pipeline] formal_eval=4 models × 4 modes × ${FORMAL_EVAL_EPISODES} episodes = 480 episodes"
echo "[Selector Pipeline] eval_parallelism=3 process shards × 4 episode envs = 12 episodes/cell"
echo "[Selector Pipeline] resume=keep complete algorithm×seed/cell; restart incomplete unit from episode 0"
echo "[Selector Pipeline] status=$STATUS_FILE"
echo "[Selector Pipeline] 查看一次: RUN_ID=$RUN_ID FOLLOW=0 bash watch_selector_m0_m3.sh"
echo "[Selector Pipeline] 持续查看: RUN_ID=$RUN_ID bash watch_selector_m0_m3.sh"
echo "============================================================"

PY_ENV_ROOT="$(cd "$(dirname "$PY")/.." && pwd)"
export TRAIN_PYTHON_BIN="$PY"
export PYTHON_BIN="$PY"
export EVAL_PYTHON_BIN="$PY"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${PY_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MATD3_REQUIRE_GPU=1
export SUPPRESS_MA_PROMPT=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-matd3-selector}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

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
    echo "[STOP] 非法批次时间标识: $stamp_file -> $stamp" >&2
    return 2
  fi
  printf '%s' "$stamp"
}

SELECTOR_EXPERIMENTS=(
  matd3_cross_agent_ref_behavior_label_agent_quality_gate
  matd3_cross_agent_ref_aqual_split_teacher
  matd3_cross_agent_ref_adaptive_twin_advantage
  matd3_cross_agent_ref_shared_twin_advantage_selector
)

COMMON_TRAIN_ARGS=(
  --batch-size 1024
  --num-envs "$TRAIN_NUM_ENVS"
  --max-parallel "$TRAIN_MODEL_MAX_PARALLEL"
  --experiment-max-parallel "$TRAIN_MODEL_MAX_PARALLEL"
  --worker-launch-stagger-seconds "$TRAIN_WORKER_LAUNCH_STAGGER_SECONDS"
  --experiment-group B
  --config-mode strict_ablation
  --env-isolation strict
  --scenario-seed 88
  --use-weighted-reward 1
  --action-force-ratio 0.50
  --action-force-ratio-schedule-pct
  '0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32'
  --post-eval-episodes 30
  --post-eval-episode-length-multiplier 1.1
  --post-eval-seed 10088
  --post-eval-mode shared_match_train_env
  --post-eval-selection-protocol fixed
  --post-eval-model-variant final
  --allow-post-eval-without-train-success
  --skip-local-plots
)

run_training_batch() {
  local phase_name="$1"
  local train_seed="$2"
  local train_episodes="$3"
  local parent_stamp="$4"
  local parent_dir="$5"

  local command=(
    "$PY"
    ablation_dual_q_separated_gradient.py
    --multi-seed
    --seeds "$train_seed"
    --episodes "$train_episodes"
    --parent-run-stamp "$parent_stamp"
    "${COMMON_TRAIN_ARGS[@]}"
    --experiments "${SELECTOR_EXPERIMENTS[@]}"
  )
  if [ -f "$parent_dir/config.json" ]; then
    echo "[$phase_name] 恢复父批次: $parent_dir"
    command+=(--resume-parent-batch-dir "$parent_dir" --reuse)
  elif [ -e "$parent_dir" ]; then
    echo "[STOP] 父批次目录存在但缺少 config.json: $parent_dir" >&2
    return 3
  else
    echo "[$phase_name] 新建父批次: $parent_dir"
  fi
  "${command[@]}"
  if [ ! -f "$parent_dir/config.json" ]; then
    echo "[STOP] $phase_name 完成后仍缺少父批次 config.json: $parent_dir" >&2
    return 3
  fi
}

mark_phase "Phase 0" "GPU、代码、单元测试与4环境真实动作路径检查"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" - <<'PY'
import tensorflow as tf
physical = tf.config.list_physical_devices("GPU")
logical = tf.config.list_logical_devices("GPU")
if not physical or not logical:
    raise SystemExit("[STOP] TensorFlow 未检测到物理和逻辑 GPU")
print("[GPU PASS] physical=", physical, "logical=", logical)
PY

"$PY" -m py_compile \
  cross_agent_reference_state.py \
  cross_agent_reference_selector.py \
  selector_experiment_protocol.py \
  experiment_runtime_config.py \
  paper3d_train_optimized.py \
  ablation_dual_q_separated_gradient.py \
  tools/build_selector_protocol_batch_spec.py \
  tools/preflight_selector_num_env_manifests.py \
  tools/smoke_selector_update.py \
  tools/smoke_parallel_env_audit.py \
  analyze_noise_dependency_batch.py

bash -n run_optimized.sh
bash -n run_noise_dependency_check_resume_fast.sh
bash -n run_selector_m0_m3_full_gpu.sh

"$PY" -m unittest -v \
  test_cross_agent_reference_selector.py \
  test_experiment_integrity.py
"$PY" tools/preflight_selector_num_env_manifests.py
"$PY" tools/smoke_selector_update.py
"$PY" tools/smoke_parallel_env_audit.py

if [ "$RUN_PILOT" = "1" ]; then
  mark_phase "Phase 1" "4 模型（2 模型并发）× 100 同步迭代 × 每模型4环境 pilot"
  PILOT_STAMP="$(read_or_create_stamp "$STATE_DIR/pilot_parent_stamp.txt")"
  PILOT_PARENT="$REPO_ROOT/ablation_experiments/multi_seed_groupB_${PILOT_STAMP}"
  PILOT_OUT="$REPO_ROOT/evaluation_results_${RUN_ID}_pilot"
  PILOT_SPEC="$PILOT_PARENT/results/selector_pilot_batch_spec_v10.json"
  run_training_batch \
    "Pilot" \
    "$PILOT_TRAIN_SEED" \
    "$PILOT_TRAIN_EPISODES" \
    "$PILOT_STAMP" \
    "$PILOT_PARENT"
  "$PY" tools/build_selector_protocol_batch_spec.py \
    --parent-batch-dir "$PILOT_PARENT" \
    --train-seed "$PILOT_TRAIN_SEED" \
    --train-episodes "$PILOT_TRAIN_EPISODES" \
    --train-num-envs "$TRAIN_NUM_ENVS" \
    --out-root "$PILOT_OUT" \
    --output "$PILOT_SPEC"
  "$PY" tools/build_selector_protocol_batch_spec.py \
    --validate \
    --output "$PILOT_SPEC"
else
  mark_phase "Phase 1" "RUN_PILOT=0，按显式请求跳过 pilot"
fi

mark_phase "Phase 2" "4 模型（2 模型并发）× 1000 同步迭代 × 每模型4环境正式 GPU 训练"
FORMAL_STAMP="$(read_or_create_stamp "$STATE_DIR/formal_parent_stamp.txt")"
FORMAL_PARENT="$REPO_ROOT/ablation_experiments/multi_seed_groupB_${FORMAL_STAMP}"
run_training_batch \
  "Formal Train" \
  "$FORMAL_TRAIN_SEED" \
  "$FORMAL_TRAIN_EPISODES" \
  "$FORMAL_STAMP" \
  "$FORMAL_PARENT"

mark_phase "Phase 3" "冻结并核验 4 × 4 × 30 正式评估规范"
FORMAL_OUT="$REPO_ROOT/evaluation_results_${RUN_ID}_formal_gpu_v10"
FORMAL_SPEC="$FORMAL_PARENT/results/selector_formal_batch_spec_v10.json"
"$PY" tools/build_selector_protocol_batch_spec.py \
  --parent-batch-dir "$FORMAL_PARENT" \
  --train-seed "$FORMAL_TRAIN_SEED" \
  --train-episodes "$FORMAL_TRAIN_EPISODES" \
  --train-num-envs "$TRAIN_NUM_ENVS" \
  --eval-noise-seed 101 \
  --eval-process-shards 3 \
  --eval-process-workers 3 \
  --eval-shard-episode-parallelism 4 \
  --eval-shard-env-step-threads 4 \
  --out-root "$FORMAL_OUT" \
  --output "$FORMAL_SPEC"
"$PY" tools/build_selector_protocol_batch_spec.py \
  --validate \
  --output "$FORMAL_SPEC"

mark_phase "Phase 4 preflight" "16 个正式评估单元 preflight"
BATCH_SPEC_JSON="$FORMAL_SPEC" \
PREFLIGHT_ONLY=1 \
PY="$PY" \
bash run_noise_dependency_check_resume_fast.sh

mark_phase "Phase 4 eval" "4 模型 × 4 模式 × 30 episode 正式 GPU 评估"
BATCH_SPEC_JSON="$FORMAL_SPEC" \
PY="$PY" \
bash run_noise_dependency_check_resume_fast.sh

mark_phase "Phase 5" "独立复核 16/16 单元与 480/480 episode"
"$PY" analyze_noise_dependency_batch.py "$FORMAL_OUT"

printf '%s\n' "$(date --iso-8601=seconds)" > "$STATE_DIR/completed_at.txt"
echo
echo "============================================================"
echo "[COMPLETE] 正式训练与 4 × 4 × 30 GPU 实验全部完成"
echo "[COMPLETE] parent=$FORMAL_PARENT"
echo "[COMPLETE] spec=$FORMAL_SPEC"
echo "[COMPLETE] results=$FORMAL_OUT"
echo "[COMPLETE] report=$FORMAL_OUT/formal_batch_report.md"
echo "[COMPLETE] log=$LOG_FILE"
echo "============================================================"

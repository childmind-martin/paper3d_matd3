#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SEEDS="${SEEDS:-101,202,936487}"
SEEDS_CSV="${SEEDS// /,}"
PYTHON_BIN="${PYTHON_BIN:-${TRAIN_PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}}"
POST_EVAL_EPISODES="${POST_EVAL_EPISODES:-30}"
POST_EVAL_MODEL_VARIANT="${POST_EVAL_MODEL_VARIANT:-best_by_team_sr}"
POST_EVAL_SELECTION_PROTOCOL="${POST_EVAL_SELECTION_PROTOCOL:-matched_validation}"
POST_EVAL_VALIDATION_EPISODES="${POST_EVAL_VALIDATION_EPISODES:-10}"
POST_EVAL_VALIDATION_SEED="${POST_EVAL_VALIDATION_SEED:-}"
POST_EVAL_VALIDATION_CANDIDATES="${POST_EVAL_VALIDATION_CANDIDATES:-best_by_team_sr,best,checkpoint,final,latest_ep}"
WORKER_LAUNCH_STAGGER_SECONDS="${WORKER_LAUNCH_STAGGER_SECONDS:-8}"
XLA_COMPILE_PARALLELISM="${XLA_COMPILE_PARALLELISM:-1}"
RESUME_PARENT_BATCH_DIR="${RESUME_PARENT_BATCH_DIR:-}"
REUSE="${REUSE:-0}"
REUSE_ONLY="${REUSE_ONLY:-0}"
FORCE_POST_EVAL_RERUN="${FORCE_POST_EVAL_RERUN:-0}"
FORCE_POST_EVAL_TESTSET_REGEN="${FORCE_POST_EVAL_TESTSET_REGEN:-0}"
SKIP_LOCAL_PLOTS="${SKIP_LOCAL_PLOTS:-0}"
AGENT_SIZE="${AGENT_SIZE:-0.5}"
export AGENT_SIZE
read -r -a EXPERIMENT_LABELS <<< "${LABELS_OVERRIDE:-matd3_full_dual_semantic matd3_collapsed_replay matd3_no_corrected_target_reconstruction}"

export SAVE_INTERVAL

append_optional_arg() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"
  if [ -n "$value" ]; then
    cmd+=("$flag" "$value")
  fi
}

cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/ablation_dual_q_separated_gradient.py"
  --multi-seed
  --seeds "$SEEDS_CSV"
  --episodes "$EPISODES"
  --batch-size "$BATCH_SIZE"
  --experiment-group B
  --experiments
    "${EXPERIMENT_LABELS[@]}"
  --max-parallel "$MAX_PARALLEL"
  --worker-launch-stagger-seconds "$WORKER_LAUNCH_STAGGER_SECONDS"
  --xla-compile-parallelism "$XLA_COMPILE_PARALLELISM"
  --post-eval-episodes "$POST_EVAL_EPISODES"
  --post-eval-model-variant "$POST_EVAL_MODEL_VARIANT"
  --post-eval-selection-protocol "$POST_EVAL_SELECTION_PROTOCOL"
  --post-eval-validation-episodes "$POST_EVAL_VALIDATION_EPISODES"
  --no-force-eval-action-force-ratio
  --allow-post-eval-without-train-success
  --post-eval-light-mode 1
  --post-eval-save-interactive-html 0
  --post-eval-save-all-episodes 0
  --post-eval-save-best-reward-html 0
  --post-eval-save-team-success-html 0
  --post-eval-save-trajectory-json 0
  --post-eval-save-trajectory-png 0
  --post-eval-save-actor-sequence 0
  --post-eval-save-control-diagnostics 0
  --post-eval-enable-overlay 0
  --post-eval-disable-gif 1
)
if [ -n "$POST_EVAL_VALIDATION_SEED" ]; then
  cmd+=(--post-eval-validation-seed "$POST_EVAL_VALIDATION_SEED")
fi
cmd+=(--post-eval-validation-candidates "$POST_EVAL_VALIDATION_CANDIDATES")

append_optional_arg GROUP_B_PEAK_JITTER_RANGE --group-b-peak-jitter-range
append_optional_arg GROUP_B_PEAK_CENTER_JITTER_RANGE --group-b-peak-center-jitter-range
append_optional_arg GROUP_B_PEAK_HEIGHT_JITTER_RATIO_MIN --group-b-peak-height-jitter-ratio-min
append_optional_arg GROUP_B_PEAK_HEIGHT_JITTER_RATIO_MAX --group-b-peak-height-jitter-ratio-max
append_optional_arg GROUP_B_PEAK_HEIGHT_MAX_SCALE --group-b-peak-height-max-scale
append_optional_arg GROUP_B_TERRAIN_VARIANT_NOISE_RATIO --group-b-terrain-variant-noise-ratio
append_optional_arg GROUP_B_SEMI_RANDOM_HOLD_MODE --group-b-semi-random-hold-mode
append_optional_arg GROUP_B_SEMI_RANDOM_HOLD_EPISODES --group-b-semi-random-hold-episodes
append_optional_arg GROUP_B_SEMI_RANDOM_HOLD_MIN_EPISODES --group-b-semi-random-hold-min-episodes
append_optional_arg GROUP_B_SEMI_RANDOM_HOLD_MAX_EPISODES --group-b-semi-random-hold-max-episodes
append_optional_arg ACTION_FORCE_RATIO --action-force-ratio
append_optional_arg ACTION_FORCE_RATIO_SCHEDULE_PCT --action-force-ratio-schedule-pct

if [ -n "${XLA_GLOBAL:-}" ]; then
  cmd+=(--xla-global "$XLA_GLOBAL")
fi
if [ -n "${JIT_COMPILE:-}" ]; then
  cmd+=(--jit-compile "$JIT_COMPILE")
fi
if [ -n "${CUDA_LAUNCH_BLOCKING:-}" ]; then
  cmd+=(--cuda-launch-blocking "$CUDA_LAUNCH_BLOCKING")
fi
if [ -n "${TF_SYNC_ON_FINISH:-}" ]; then
  cmd+=(--tf-sync-on-finish "$TF_SYNC_ON_FINISH")
fi
case "${FORCE_OUTER_JIT_COMPILE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    cmd+=(--force-outer-jit-compile)
    ;;
esac

case "${SKIP_LOCAL_PLOTS:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    cmd+=(--skip-local-plots)
    ;;
esac

if [ -n "$RESUME_PARENT_BATCH_DIR" ]; then
  cmd+=(--resume-parent-batch-dir "$RESUME_PARENT_BATCH_DIR")
fi
if [ "$REUSE" = "1" ]; then
  cmd+=(--reuse)
fi
if [ "$REUSE_ONLY" = "1" ]; then
  cmd+=(--reuse-only)
fi
if [ "$FORCE_POST_EVAL_RERUN" = "1" ]; then
  cmd+=(--force-post-eval-rerun)
fi
if [ "$FORCE_POST_EVAL_TESTSET_REGEN" = "1" ]; then
  cmd+=(--force-post-eval-testset-regen)
fi

printf '[level3-semantic-train] '
printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"

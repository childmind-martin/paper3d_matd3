#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

TRAIN_SEEDS="${TRAIN_SEEDS:-101 202 936487 303 404 505 606 707 808 909}"
TRAIN_SEEDS_CSV="${TRAIN_SEEDS// /,}"
SEED_COUNT="$(wc -w <<< "$TRAIN_SEEDS" | tr -d ' ')"
PHASES="${PHASES:-level2 semantic level3 level1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
REFRESH_LEVEL2_ARTIFACTS="${REFRESH_LEVEL2_ARTIFACTS:-1}"
SKIP_LOCAL_PLOTS="${SKIP_LOCAL_PLOTS:-0}"
AUTO_RESUME_PARENT_BATCHES="${AUTO_RESUME_PARENT_BATCHES:-1}"

EPISODES="${EPISODES:-1000}"
LEVEL1_EPISODES="${LEVEL1_EPISODES:-400}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TRAIN_MAX_PARALLEL="${TRAIN_MAX_PARALLEL:-$MAX_PARALLEL}"
EVAL_MAX_PARALLEL="${EVAL_MAX_PARALLEL:-$MAX_PARALLEL}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"
RESTART_INCOMPLETE="${RESTART_INCOMPLETE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_EVAL_RERUN="${FORCE_EVAL_RERUN:-$FORCE_RERUN}"
FORCE_POST_EVAL_RERUN="${FORCE_POST_EVAL_RERUN:-$FORCE_RERUN}"
FORCE_POST_EVAL_TESTSET_REGEN="${FORCE_POST_EVAL_TESTSET_REGEN:-0}"
LEVEL1_RESUME_PARENT_BATCH_DIR="${LEVEL1_RESUME_PARENT_BATCH_DIR:-${RESUME_LEVEL1_PARENT_BATCH_DIR:-}}"
LEVEL3_RESUME_PARENT_BATCH_DIR="${LEVEL3_RESUME_PARENT_BATCH_DIR:-${RESUME_LEVEL3_PARENT_BATCH_DIR:-}}"

LEVEL3_SCOPE="${LEVEL3_SCOPE:-semantic}"
LEVEL3_POST_EVAL_EPISODES="${LEVEL3_POST_EVAL_EPISODES:-30}"
RUN_LEVEL2_SEMANTIC_LONG_EVAL="${RUN_LEVEL2_SEMANTIC_LONG_EVAL:-1}"
LEVEL2_SEMANTIC_LONG_EVAL_EPISODES="${LEVEL2_SEMANTIC_LONG_EVAL_EPISODES:-30}"
LEVEL2_SEMANTIC_LONG_RUN_TAG="${LEVEL2_SEMANTIC_LONG_RUN_TAG:-level2_dual_semantics_eval10x30_model_fr}"
LEVEL2_SEMANTIC_LONG_SELECTION_PROTOCOL="${LEVEL2_SEMANTIC_LONG_SELECTION_PROTOCOL:-fixed}"
LEVEL1_POST_EVAL_EPISODES="${LEVEL1_POST_EVAL_EPISODES:-20}"

OFFICIAL_LABELS=(
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

SEMANTIC_LABELS=(
  matd3_full_dual_semantic
  matd3_collapsed_replay
  matd3_no_corrected_target_reconstruction
)

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

phase_enabled() {
  local phase="$1"
  [[ " $PHASES " == *" $phase "* ]]
}

run_cmd() {
  printf '\n[ordered-seed-expansion] '
  printf '%q ' "$@"
  printf '\n'
  if truthy "$DRY_RUN"; then
    return 0
  fi
  "$@"
}

resolve_matching_parent_batch() {
  local group="$1"
  local episodes="$2"
  local completion_mode="$3"
  shift 3
  "$PYTHON_BIN" - "$ROOT_DIR" "$group" "$episodes" "$TRAIN_SEEDS_CSV" "$completion_mode" "$@" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
group = str(sys.argv[2]).strip().upper()
try:
    expected_episodes = int(sys.argv[3])
except Exception:
    expected_episodes = None
seed_tokens = [token for token in str(sys.argv[4]).replace(" ", ",").split(",") if token]
try:
    expected_seeds = [int(token) for token in seed_tokens]
except Exception:
    raise SystemExit(1)
completion_mode = str(sys.argv[5]).strip().lower()
expected_labels = list(sys.argv[6:])
if not expected_seeds or not expected_labels:
    raise SystemExit(1)

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def seed_has_artifact(batch_dir, seed, label):
    seed_batches = batch_dir / "seed_batches"
    for child in seed_batches.glob(f"batch_*_seed{seed}_*"):
        if (child / "results" / "experiment_artifacts" / f"{label}.json").is_file():
            return True
    return False

def batch_complete(batch_dir):
    return all(
        seed_has_artifact(batch_dir, seed, label)
        for seed in expected_seeds
        for label in expected_labels
    )

candidates = []
for batch_dir in (root / "ablation_experiments").glob(f"multi_seed_group{group}_*"):
    config = load_json(batch_dir / "config.json")
    if not isinstance(config, dict):
        continue
    if str(config.get("experiment_group", "")).strip().upper() != group:
        continue
    if expected_episodes is not None and int(config.get("episodes", -1) or -1) != expected_episodes:
        continue
    config_seeds = [int(seed) for seed in (config.get("seeds") or [])]
    if config_seeds != expected_seeds:
        continue
    config_labels = list(config.get("experiments") or [])
    if len(config_labels) != len(expected_labels) or set(config_labels) != set(expected_labels):
        continue
    complete = batch_complete(batch_dir)
    if completion_mode == "complete" and not complete:
        continue
    if completion_mode == "incomplete" and complete:
        continue
    candidates.append((batch_dir.stat().st_mtime, complete, batch_dir))

if not candidates:
    raise SystemExit(1)

# Prefer complete batches when several matching candidates exist, then newest mtime.
candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
print(candidates[0][2])
PY
}

prepare_level2_specs() {
  read -r -a seed_array <<< "$TRAIN_SEEDS"
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/prepare_level2_official_train_seed_specs.py" \
    --train-seeds "${seed_array[@]}"
}

run_level2_main() {
  if truthy "$RUN_PREFLIGHT"; then
    run_cmd env \
      TRAIN_PYTHON_BIN="$PYTHON_BIN" \
      EVAL_PYTHON_BIN="$PYTHON_BIN" \
      SEEDS="$TRAIN_SEEDS" \
      EPISODES="$EPISODES" \
      BATCH_SIZE="$BATCH_SIZE" \
      MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
      TRAIN_MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
      EVAL_MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
      STRICT_SUMMARY="$STRICT_SUMMARY" \
      RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
      FORCE_RERUN="$FORCE_RERUN" \
      FORCE_EVAL_RERUN="$FORCE_EVAL_RERUN" \
      PREFLIGHT_ONLY=1 \
      bash "$ROOT_DIR/run_level2_multiseed_all_algos_official.sh"
  fi

  run_cmd env \
    TRAIN_PYTHON_BIN="$PYTHON_BIN" \
    EVAL_PYTHON_BIN="$PYTHON_BIN" \
    SEEDS="$TRAIN_SEEDS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
    TRAIN_MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
    EVAL_MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
    EPISODES="$EPISODES" \
    BATCH_SIZE="$BATCH_SIZE" \
    STRICT_SUMMARY="$STRICT_SUMMARY" \
    RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
    FORCE_RERUN="$FORCE_RERUN" \
    FORCE_EVAL_RERUN="$FORCE_EVAL_RERUN" \
    bash "$ROOT_DIR/run_level2_multiseed_all_algos_official.sh"
}

refresh_level2_main_artifacts() {
  read -r -a seed_array <<< "$TRAIN_SEEDS"
  run_cmd env MPLCONFIGDIR=/tmp/mplconfig \
    "$PYTHON_BIN" "$ROOT_DIR/summarize_level2_official_eval_multiseed.py" \
      --seeds "${seed_array[@]}" \
      --labels "${OFFICIAL_LABELS[@]}" \
      --run-tag level2_ms_official \
      --strict \
      --output-dir "$ROOT_DIR/diagnostics/level2_ms_official_${SEED_COUNT}seed"
  run_cmd env MPLCONFIGDIR=/tmp/mplconfig \
    "$PYTHON_BIN" "$ROOT_DIR/plot_level2_training_curves_multiseed.py" \
      --seeds "${seed_array[@]}" \
      --labels "${OFFICIAL_LABELS[@]}" \
      --run-tag level2_ms_official \
      --expected-episodes "$EPISODES" \
      --strict \
      --output-dir "$ROOT_DIR/diagnostics/level2_ms_official_${SEED_COUNT}seed/training_curves" \
      --timestamp "${SEED_COUNT}seed"
}

run_level2_semantic() {
  if truthy "$RUN_PREFLIGHT"; then
    run_cmd env \
      TRAIN_PYTHON_BIN="$PYTHON_BIN" \
      EVAL_PYTHON_BIN="$PYTHON_BIN" \
      SEEDS="$TRAIN_SEEDS" \
      EPISODES="$EPISODES" \
      BATCH_SIZE="$BATCH_SIZE" \
      MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
      TRAIN_MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
      EVAL_MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
      STRICT_SUMMARY="$STRICT_SUMMARY" \
      RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
      FORCE_RERUN="$FORCE_RERUN" \
      FORCE_EVAL_RERUN="$FORCE_EVAL_RERUN" \
      PREFLIGHT_ONLY=1 \
      bash "$ROOT_DIR/run_level2_dual_semantics_ablation_official.sh"
  fi

  run_cmd env \
    TRAIN_PYTHON_BIN="$PYTHON_BIN" \
    EVAL_PYTHON_BIN="$PYTHON_BIN" \
    SEEDS="$TRAIN_SEEDS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
    TRAIN_MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
    EVAL_MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
    EPISODES="$EPISODES" \
    BATCH_SIZE="$BATCH_SIZE" \
    STRICT_SUMMARY="$STRICT_SUMMARY" \
    RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
    FORCE_RERUN="$FORCE_RERUN" \
    FORCE_EVAL_RERUN="$FORCE_EVAL_RERUN" \
    bash "$ROOT_DIR/run_level2_dual_semantics_ablation_official.sh"
}

refresh_level2_semantic_artifacts() {
  read -r -a seed_array <<< "$TRAIN_SEEDS"
  run_cmd env MPLCONFIGDIR=/tmp/mplconfig \
    "$PYTHON_BIN" "$ROOT_DIR/summarize_level2_official_eval_multiseed.py" \
      --seeds "${seed_array[@]}" \
      --labels "${SEMANTIC_LABELS[@]}" \
      --run-tag level2_dual_semantics_ablation \
      --strict \
      --output-dir "$ROOT_DIR/diagnostics/level2_dual_semantics_ablation_${SEED_COUNT}seed"
  run_cmd env MPLCONFIGDIR=/tmp/mplconfig \
    "$PYTHON_BIN" "$ROOT_DIR/plot_level2_training_curves_multiseed.py" \
      --seeds "${seed_array[@]}" \
      --labels "${SEMANTIC_LABELS[@]}" \
      --run-tag level2_dual_semantics_ablation \
      --expected-episodes "$EPISODES" \
      --strict \
      --output-dir "$ROOT_DIR/diagnostics/level2_dual_semantics_ablation_${SEED_COUNT}seed/training_curves" \
      --timestamp "${SEED_COUNT}seed"
}

run_level2_semantic_long_eval() {
  run_cmd env \
    TRAIN_SEEDS="$TRAIN_SEEDS" \
    RUN_TAG="$LEVEL2_SEMANTIC_LONG_RUN_TAG" \
    EVAL_FR_MODE=checkpoint \
    SUMMARY_FILENAME_TAG=model_fr \
    EVAL_EPISODES="$LEVEL2_SEMANTIC_LONG_EVAL_EPISODES" \
    MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
    MODEL_VARIANT=best_by_team_sr \
    SELECTION_PROTOCOL="$LEVEL2_SEMANTIC_LONG_SELECTION_PROTOCOL" \
    STRICT_SUMMARY="$STRICT_SUMMARY" \
    FORCE_RERUN="$FORCE_EVAL_RERUN" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT_DIR/run_level2_dual_semantics_eval10x30_official.sh"
}

run_level3_semantic() {
  local resume_parent="$LEVEL3_RESUME_PARENT_BATCH_DIR"
  if [ -z "$resume_parent" ] && truthy "$AUTO_RESUME_PARENT_BATCHES" && ! truthy "$FORCE_RERUN" && ! truthy "$FORCE_POST_EVAL_RERUN"; then
    resume_parent="$(resolve_matching_parent_batch B "$EPISODES" complete "${SEMANTIC_LABELS[@]}" 2>/dev/null || true)"
  fi
  if [ -n "$resume_parent" ]; then
    echo "[ordered-seed-expansion] Level-3 semantic resume parent: $resume_parent"
  fi
  run_cmd env \
    TRAIN_SEEDS="$TRAIN_SEEDS" \
    SEEDS="$TRAIN_SEEDS_CSV" \
    RESUME_PARENT_BATCH_DIR="$resume_parent" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
    TRAIN_MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
    EVAL_MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
    EPISODES="$EPISODES" \
    BATCH_SIZE="$BATCH_SIZE" \
    POST_EVAL_EPISODES="$LEVEL3_POST_EVAL_EPISODES" \
    EVAL_EPISODES="$LEVEL3_POST_EVAL_EPISODES" \
    EVAL_FR_MODE=checkpoint \
    STRICT_SUMMARY="$STRICT_SUMMARY" \
    REUSE=1 \
    FORCE_RERUN="$FORCE_EVAL_RERUN" \
    FORCE_POST_EVAL_RERUN="$FORCE_POST_EVAL_RERUN" \
    FORCE_POST_EVAL_TESTSET_REGEN="$FORCE_POST_EVAL_TESTSET_REGEN" \
    SKIP_LOCAL_PLOTS="$SKIP_LOCAL_PLOTS" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT_DIR/run_level3_dual_semantics_train_eval_official.sh"
}

run_level3_all_algos() {
  local plot_args=()
  local post_eval_args=()
  if truthy "$SKIP_LOCAL_PLOTS"; then
    plot_args+=(--skip-local-plots)
  fi
  if truthy "$FORCE_POST_EVAL_RERUN"; then
    post_eval_args+=(--force-post-eval-rerun)
  fi
  run_cmd env \
    MPLCONFIGDIR=/tmp/mplconfig \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    "$PYTHON_BIN" "$ROOT_DIR/ablation_dual_q_separated_gradient.py" \
      --multi-seed \
      --seeds "$TRAIN_SEEDS_CSV" \
      --episodes "$EPISODES" \
      --batch-size "$BATCH_SIZE" \
      --experiment-group B \
      --experiments "${OFFICIAL_LABELS[@]}" \
      --max-parallel "$TRAIN_MAX_PARALLEL" \
      --reuse \
      --worker-launch-stagger-seconds 8 \
      --xla-compile-parallelism 1 \
      --post-eval-episodes "$LEVEL3_POST_EVAL_EPISODES" \
      --post-eval-model-variant best_by_team_sr \
      --post-eval-selection-protocol matched_validation \
      --post-eval-validation-episodes 10 \
      --post-eval-validation-candidates best_by_team_sr,best,checkpoint,final,latest_ep \
      --no-force-eval-action-force-ratio \
      --allow-post-eval-without-train-success \
      "${post_eval_args[@]}" \
      --post-eval-light-mode 1 \
      --post-eval-save-interactive-html 0 \
      --post-eval-save-all-episodes 0 \
      --post-eval-save-best-reward-html 0 \
      --post-eval-save-team-success-html 0 \
      --post-eval-save-trajectory-json 0 \
      --post-eval-save-trajectory-png 0 \
      --post-eval-save-actor-sequence 0 \
      --post-eval-save-control-diagnostics 0 \
      --post-eval-enable-overlay 0 \
      --post-eval-disable-gif 1 \
      "${plot_args[@]}"
}

run_level3() {
  case "${LEVEL3_SCOPE,,}" in
    semantic)
      run_level3_semantic
      ;;
    all|all_algos|official)
      run_level3_all_algos
      ;;
    *)
      echo "[ordered-seed-expansion] unknown LEVEL3_SCOPE=$LEVEL3_SCOPE (expected semantic or all_algos)" >&2
      return 2
      ;;
  esac
}

run_level1() {
  local plot_args=()
  local post_eval_args=()
  local resume_args=()
  local resume_parent="$LEVEL1_RESUME_PARENT_BATCH_DIR"
  if truthy "$SKIP_LOCAL_PLOTS"; then
    plot_args+=(--skip-local-plots)
  fi
  if truthy "$FORCE_POST_EVAL_RERUN"; then
    post_eval_args+=(--force-post-eval-rerun)
  fi
  if [ -z "$resume_parent" ] && truthy "$AUTO_RESUME_PARENT_BATCHES" && ! truthy "$FORCE_RERUN"; then
    resume_parent="$(resolve_matching_parent_batch A "$LEVEL1_EPISODES" any "${OFFICIAL_LABELS[@]}" 2>/dev/null || true)"
  fi
  if [ -n "$resume_parent" ]; then
    echo "[ordered-seed-expansion] Level-1 resume parent: $resume_parent"
    resume_args+=(--resume-parent-batch-dir "$resume_parent")
  fi
  run_cmd env \
    MPLCONFIGDIR=/tmp/mplconfig \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    "$PYTHON_BIN" "$ROOT_DIR/ablation_dual_q_separated_gradient.py" \
      --multi-seed \
      "${resume_args[@]}" \
      --seeds "$TRAIN_SEEDS_CSV" \
      --episodes "$LEVEL1_EPISODES" \
      --batch-size "$BATCH_SIZE" \
      --experiment-group A \
      --experiments "${OFFICIAL_LABELS[@]}" \
      --max-parallel "$TRAIN_MAX_PARALLEL" \
      --reuse \
      --worker-launch-stagger-seconds 8 \
      --xla-compile-parallelism 1 \
      --post-eval-episodes "$LEVEL1_POST_EVAL_EPISODES" \
      --post-eval-model-variant best_by_team_sr \
      --post-eval-selection-protocol matched_validation \
      --post-eval-validation-episodes 10 \
      --post-eval-validation-candidates best_by_team_sr,best,checkpoint,final,latest_ep \
      --no-force-eval-action-force-ratio \
      "${post_eval_args[@]}" \
      --post-eval-light-mode 1 \
      --post-eval-save-interactive-html 0 \
      --post-eval-save-all-episodes 0 \
      --post-eval-save-best-reward-html 0 \
      --post-eval-save-team-success-html 0 \
      --post-eval-save-trajectory-json 0 \
      --post-eval-save-trajectory-png 0 \
      --post-eval-save-actor-sequence 0 \
      --post-eval-save-control-diagnostics 0 \
      --post-eval-enable-overlay 0 \
      --post-eval-disable-gif 1 \
      "${plot_args[@]}"
}

echo "[ordered-seed-expansion] phases=$PHASES"
echo "[ordered-seed-expansion] train_seeds=$TRAIN_SEEDS"
echo "[ordered-seed-expansion] max_parallel=$MAX_PARALLEL train_max_parallel=$TRAIN_MAX_PARALLEL eval_max_parallel=$EVAL_MAX_PARALLEL episodes=$EPISODES batch_size=$BATCH_SIZE"
echo "[ordered-seed-expansion] level3_scope=$LEVEL3_SCOPE"
echo "[ordered-seed-expansion] force_rerun=$FORCE_RERUN force_eval_rerun=$FORCE_EVAL_RERUN force_post_eval_rerun=$FORCE_POST_EVAL_RERUN force_post_eval_testset_regen=$FORCE_POST_EVAL_TESTSET_REGEN"

if phase_enabled level2 || phase_enabled semantic; then
  echo "[ordered-seed-expansion] Preparing Level-2 official specs"
  prepare_level2_specs
fi

if phase_enabled level2; then
  echo "[ordered-seed-expansion] Phase 1: Level-2 held-out evaluation"
  run_level2_main
  if truthy "$REFRESH_LEVEL2_ARTIFACTS"; then
    echo "[ordered-seed-expansion] Refreshing Level-2 official old+new seed tables and training curves"
    refresh_level2_main_artifacts
  fi
fi

if phase_enabled semantic; then
  echo "[ordered-seed-expansion] Phase 2: Level-2 semantic ablation"
  run_level2_semantic
  if truthy "$RUN_LEVEL2_SEMANTIC_LONG_EVAL"; then
    echo "[ordered-seed-expansion] Phase 2b: Level-2 semantic 10x30 checkpoint-FR evaluation"
    run_level2_semantic_long_eval
  fi
  if truthy "$REFRESH_LEVEL2_ARTIFACTS"; then
    echo "[ordered-seed-expansion] Refreshing Level-2 semantic old+new seed tables and training curves"
    refresh_level2_semantic_artifacts
  fi
fi

if phase_enabled level3; then
  echo "[ordered-seed-expansion] Phase 3: Level-3"
  run_level3
fi

if phase_enabled level1; then
  echo "[ordered-seed-expansion] Phase 4: Level-1"
  run_level1
fi

echo "[ordered-seed-expansion] Done."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

TRAIN_SCRIPT="$ROOT_DIR/run_level3_dual_semantics_ablation_official.sh"
EVAL_SCRIPT="$ROOT_DIR/run_level3_dual_semantics_eval10x30_official.sh"

EPISODES="${EPISODES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TRAIN_MAX_PARALLEL="${TRAIN_MAX_PARALLEL:-$MAX_PARALLEL}"
EVAL_MAX_PARALLEL="${EVAL_MAX_PARALLEL:-$MAX_PARALLEL}"
TRAIN_SEEDS="${TRAIN_SEEDS:-101 202 936487}"
SEEDS="${SEEDS:-${TRAIN_SEEDS// /,}}"
EVAL_SEEDS="${EVAL_SEEDS:-30088 30188 30288 30388 30488 30588 30688 30788 30888 30988}"
POST_EVAL_EPISODES="${POST_EVAL_EPISODES:-30}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
RUN_TAG="${RUN_TAG:-level3_dual_semantics_eval10x30}"
EVAL_FR_MODE="${EVAL_FR_MODE:-checkpoint}"
SUMMARY_FILENAME_TAG="${SUMMARY_FILENAME_TAG:-model_fr}"
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}"
SKIP_LOCAL_PLOTS="${SKIP_LOCAL_PLOTS:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_POST_EVAL_RERUN="${FORCE_POST_EVAL_RERUN:-0}"
FORCE_POST_EVAL_TESTSET_REGEN="${FORCE_POST_EVAL_TESTSET_REGEN:-0}"
RESUME_PARENT_BATCH_DIR="${RESUME_PARENT_BATCH_DIR:-}"

SEMANTIC_LABELS=(
  matd3_full_dual_semantic
  matd3_collapsed_replay
  matd3_no_corrected_target_reconstruction
)

resolve_latest_semantic_parent_batch() {
  "$PYTHON_BIN" - "$ROOT_DIR" "${SEMANTIC_LABELS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = list(sys.argv[2:])
base = root / "ablation_experiments"
matches = []
for path in base.glob("multi_seed_groupB_*"):
    if not path.is_dir():
        continue
    config_path = path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    experiments = list(config.get("experiments") or [])
    if len(experiments) != len(labels) or set(experiments) != set(labels):
        continue
    seeds = [int(seed) for seed in (config.get("seeds") or [])]
    complete = True
    for seed in seeds:
        for label in labels:
            found = False
            for child in (path / "seed_batches").glob(f"batch_*_seed{seed}_*"):
                if (child / "results" / "experiment_artifacts" / f"{label}.json").is_file():
                    found = True
                    break
            if not found:
                complete = False
                break
        if not complete:
            break
    matches.append((complete, path.stat().st_mtime, path))
if not matches:
    raise SystemExit(1)
# Prefer complete parent batches, then newest mtime.
print(max(matches, key=lambda item: (item[0], item[1]))[2])
PY
}

resolve_base_spec_from_parent() {
  local parent_dir="$1"
  local first_seed
  local spec
  read -r first_seed _ <<< "$TRAIN_SEEDS"
  spec="$(find "$parent_dir/seed_batches" -path "*/batch_groupB_seed${first_seed}_*/results/post_eval_shared_spec.json" -print -quit 2>/dev/null || true)"
  if [ -z "$spec" ]; then
    spec="$(find "$parent_dir/seed_batches" -path "*/results/post_eval_shared_spec.json" -print -quit 2>/dev/null || true)"
  fi
  if [ -z "$spec" ] || [ ! -f "$spec" ]; then
    echo "[level3-semantic-onekey] cannot find Level3 post_eval_shared_spec.json under: $parent_dir" >&2
    return 1
  fi
  printf '%s\n' "$spec"
}

echo "[level3-semantic-onekey] Step 1/2: training + official post-eval"
EPISODES="$EPISODES" \
BATCH_SIZE="$BATCH_SIZE" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
MAX_PARALLEL="$TRAIN_MAX_PARALLEL" \
SEEDS="$SEEDS" \
RESUME_PARENT_BATCH_DIR="$RESUME_PARENT_BATCH_DIR" \
REUSE=1 \
POST_EVAL_EPISODES="$POST_EVAL_EPISODES" \
FORCE_POST_EVAL_RERUN="$FORCE_POST_EVAL_RERUN" \
FORCE_POST_EVAL_TESTSET_REGEN="$FORCE_POST_EVAL_TESTSET_REGEN" \
SKIP_LOCAL_PLOTS="$SKIP_LOCAL_PLOTS" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$TRAIN_SCRIPT"

if [ -n "${RESUME_PARENT_BATCH_DIR:-}" ]; then
  PARENT_BATCH_DIR="$RESUME_PARENT_BATCH_DIR"
else
  PARENT_BATCH_DIR="$(resolve_latest_semantic_parent_batch)"
fi
BASE_SPEC="${BASE_SPEC:-$(resolve_base_spec_from_parent "$PARENT_BATCH_DIR")}"

echo "[level3-semantic-onekey] Parent batch: $PARENT_BATCH_DIR"
echo "[level3-semantic-onekey] Level3 base spec: $BASE_SPEC"
echo "[level3-semantic-onekey] Step 2/2: 10x30 official eval using model/checkpoint FR"

BASE_SPEC="$BASE_SPEC" \
RUN_TAG="$RUN_TAG" \
EVAL_EPISODES="$EVAL_EPISODES" \
TRAIN_EPISODES="$EPISODES" \
MAX_PARALLEL="$EVAL_MAX_PARALLEL" \
TRAIN_SEEDS="$TRAIN_SEEDS" \
EVAL_SEEDS="$EVAL_SEEDS" \
EVAL_FR_MODE="$EVAL_FR_MODE" \
SUMMARY_FILENAME_TAG="$SUMMARY_FILENAME_TAG" \
STRICT_SUMMARY="$STRICT_SUMMARY" \
FORCE_RERUN="$FORCE_RERUN" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$EVAL_SCRIPT"

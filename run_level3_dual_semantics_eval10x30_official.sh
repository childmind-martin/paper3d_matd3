#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-level3_dual_semantics_eval10x30}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
TRAIN_EPISODES="${TRAIN_EPISODES:-1000}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SELECTION_MAX_PARALLEL="${SELECTION_MAX_PARALLEL:-1}"
TRAIN_SEEDS=(${TRAIN_SEEDS:-101 202 936487})
EVAL_SEEDS=(${EVAL_SEEDS:-30088 30188 30288 30388 30488 30588 30688 30788 30888 30988})
LABELS=(${LABELS_OVERRIDE:-matd3_full_dual_semantic matd3_collapsed_replay matd3_no_corrected_target_reconstruction})
MODEL_VARIANT="${MODEL_VARIANT:-best_by_team_sr}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-fixed}"
SELECTION_VALIDATION_SEEDS=(${SELECTION_VALIDATION_SEEDS:-41088 41188 41288 41388 41488})
SELECTION_VALIDATION_EPISODES="${SELECTION_VALIDATION_EPISODES:-20}"
SELECTION_VALIDATION_CANDIDATES="${SELECTION_VALIDATION_CANDIDATES:-best_by_team_sr,best,checkpoint,final,latest_ep}"
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EVAL_FR_MODE="${EVAL_FR_MODE:-checkpoint}"
FORCE_EVAL_ACTION_FORCE_RATIO_VALUE="${FORCE_EVAL_ACTION_FORCE_RATIO_VALUE:-0.50}"
SUMMARY_FILENAME_TAG="${SUMMARY_FILENAME_TAG:-model_fr}"
EVAL_EPISODE_PARALLELISM="${EVAL_EPISODE_PARALLELISM:-1}"
EVAL_ENV_STEP_THREADS="${EVAL_ENV_STEP_THREADS:-1}"
SELECTION_EVAL_EPISODE_PARALLELISM="${SELECTION_EVAL_EPISODE_PARALLELISM:-3}"
SELECTION_EVAL_ENV_STEP_THREADS="${SELECTION_EVAL_ENV_STEP_THREADS:-$SELECTION_EVAL_EPISODE_PARALLELISM}"
EVAL_DEBUG_ACTION_STEPS="${EVAL_DEBUG_ACTION_STEPS:-0}"
FAST_ARTIFACTS="${FAST_ARTIFACTS:-1}"
EVAL_LIGHT_MODE="${EVAL_LIGHT_MODE:-1}"
SUPPRESS_TERRAIN_OUTPUT="${SUPPRESS_TERRAIN_OUTPUT:-1}"
SUPPRESS_REWARD_CONFIG_OUTPUT="${SUPPRESS_REWARD_CONFIG_OUTPUT:-1}"
PYTHON_BIN="${PYTHON_BIN:-${EVAL_PYTHON_BIN:-${TRAIN_PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}}}"
CONDA_SH="${CONDA_SH:-/home/tang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-maddpg_env}"
REWARD_VERSION="${REWARD_VERSION:-v1}"
REWARD_TERMINAL_ORDER_FIX="${REWARD_TERMINAL_ORDER_FIX:-1}"
AGENT_SIZE="${AGENT_SIZE:-0.5}"

BASE_SPEC="${BASE_SPEC:-}"
SPEC_ROOT="${SPEC_ROOT:-/home/tang/matd3/ablation_experiments/${RUN_TAG}_specs}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-/home/tang/matd3/parallel_logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
SHARED_SELECTION_ROOT="${SHARED_SELECTION_ROOT:-/home/tang/matd3/logs/${RUN_TAG}_shared_checkpoint_selection}"
mkdir -p "$RUN_LOG_ROOT" "$SPEC_ROOT"

prepend_path_once() {
  local path="$1"
  [ -n "$path" ] && [ -d "$path" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${path}:"*) ;;
    *) export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

configure_eval_gpu_env() {
  local python_for_lib="${1:-$PYTHON_BIN}"
  local python_path=""
  local env_dir=""

  if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    env_dir="$CONDA_PREFIX"
  elif [ -x "$python_for_lib" ]; then
    python_path="$python_for_lib"
  elif command -v "$python_for_lib" >/dev/null 2>&1; then
    python_path="$(command -v "$python_for_lib")"
  fi

  if [ -z "$env_dir" ] && [ -n "$python_path" ]; then
    if command -v readlink >/dev/null 2>&1; then
      python_path="$(readlink -f "$python_path" 2>/dev/null || printf '%s' "$python_path")"
    fi
    env_dir="$(cd "$(dirname "$python_path")/.." 2>/dev/null && pwd -P || true)"
  fi

  if [ -n "$env_dir" ] && [ -d "$env_dir/lib" ]; then
    prepend_path_once "$env_dir/lib"
    if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$env_dir/conda-meta" ]; then
      export CONDA_PREFIX="$env_dir"
    fi
    export MATD3_EVAL_CONDA_LIB_DIR="$env_dir/lib"
  fi
  prepend_path_once "/usr/lib/wsl/lib"

  if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
  fi
  export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
}

configure_eval_gpu_env "$PYTHON_BIN"

declare -A SHARED_SELECTED_MODEL_PATHS=()
declare -A SHARED_SELECTION_SUMMARIES=()

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

use_shared_selection() {
  case "${SELECTION_PROTOCOL,,}" in
    shared|shared_validation|shared_matched_validation) return 0 ;;
    *) return 1 ;;
  esac
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

active_job_count() {
  local count=0
  local _job
  while IFS= read -r _job; do
    [ -n "$_job" ] || continue
    count=$((count + 1))
  done < <(jobs -pr || true)
  printf '%s\n' "$count"
}

model_dirs_newest_first() {
  local label="$1"
  local train_seed="$2"
  find "$ROOT_DIR/models" -maxdepth 1 -type d -name "${label}__seed${train_seed}__batch_groupB_seed${train_seed}_*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk '{ $1=""; sub(/^ /, ""); print }'
}

model_completed() {
  local model_dir="$1"
  local train_seed="$2"
  "$PYTHON_BIN" - "$model_dir" "$TRAIN_EPISODES" "$train_seed" <<'PY'
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
  "$PYTHON_BIN" - "$eval_dir/evaluation_results.json" "$EVAL_EPISODES" "$EVAL_FR_MODE" "$model_dir" "$MODEL_VARIANT" "$SELECTION_PROTOCOL" "$REWARD_VERSION" "$REWARD_TERMINAL_ORDER_FIX" "$AGENT_SIZE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
fr_mode = str(sys.argv[3]).lower()
model_dir = Path(sys.argv[4])
model_variant = str(sys.argv[5]).strip() or "best_by_team_sr"
selection_protocol = str(sys.argv[6]).strip().lower() or "fixed"
expected_reward_version = str(sys.argv[7]).strip() or "v1"
expected_terminal_order_raw = str(sys.argv[8]).strip()
expected_agent_size_raw = str(sys.argv[9]).strip()
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
summary = data.get("summary", {}) if isinstance(data, dict) else {}
setup = data.get("evaluation_setup", {}) if isinstance(data, dict) else {}
selection = data.get("model_selection", {}) if isinstance(data.get("model_selection"), dict) else {}
if not isinstance(setup, dict):
    setup = {}
try:
    episodes = int(summary.get("episodes", data.get("episodes")) or 0)
except Exception:
    raise SystemExit(1)
has_selection = bool(selection)
fr_source = str(setup.get("action_force_ratio_source", "") or "").strip()
fr_source_ok = bool(fr_source)
if fr_mode in {"checkpoint", "checkpoint_fr", "model", "model_fr", "corresponding", "corresponding_fr"}:
    fr_source_ok = fr_source_ok and fr_source != "forced_override"

def model_signature(model_path: Path):
    try:
        weight_files = sorted(model_path.glob("actor_*.weights.h5"))
        if not weight_files:
            return None
        hasher = hashlib.sha1()
        for weight_path in weight_files:
            hasher.update(weight_path.name.encode("utf-8", errors="ignore"))
            with open(weight_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

selected_path_raw = str(selection.get("selected_model_path", "") or "")
if not selected_path_raw:
    raise SystemExit(1)
try:
    selected_path = Path(selected_path_raw).resolve()
    model_root = model_dir.resolve()
except Exception:
    raise SystemExit(1)
if fr_mode in {"checkpoint", "checkpoint_fr", "model", "model_fr", "corresponding", "corresponding_fr"}:
    selected_leaf = selected_path.name.strip().lower()
    if selected_leaf.startswith("ep") and selected_leaf[2:].isdigit():
        fr_source_ok = fr_source_ok and fr_source == "指定回合"

selection_ok = True
if selection_protocol == "fixed":
    expected_path = (model_root / model_variant).resolve()
    selection_ok = selected_path == expected_path
else:
    try:
        selection_ok = selected_path == model_root or selected_path.is_relative_to(model_root)
    except Exception:
        selection_ok = str(selected_path).startswith(str(model_root) + "/")

recorded_signature = str(selection.get("selected_model_signature", "") or "").strip()
current_signature = model_signature(selected_path)
signature_ok = bool(recorded_signature) and bool(current_signature) and recorded_signature == current_signature

recorded_reward_version = str(setup.get("reward_version", "") or "").strip()
if expected_terminal_order_raw:
    expected_terminal_order_fix = expected_terminal_order_raw.lower() in {"1", "true", "yes", "on"}
else:
    expected_terminal_order_fix = True
recorded_terminal_order_fix = setup.get("reward_terminal_order_fix", None)
if isinstance(recorded_terminal_order_fix, bool):
    terminal_order_ok = recorded_terminal_order_fix == expected_terminal_order_fix
else:
    terminal_order_ok = str(recorded_terminal_order_fix).strip().lower() in {"1", "true", "yes", "on"} if recorded_terminal_order_fix is not None else False
    terminal_order_ok = terminal_order_ok == expected_terminal_order_fix
reward_setup_ok = recorded_reward_version == expected_reward_version and terminal_order_ok
agent_size_ok = True
if expected_agent_size_raw:
    try:
        expected_agent_size = float(expected_agent_size_raw)
        actual_agent_size = float(setup.get("agent_size"))
        agent_size_ok = abs(actual_agent_size - expected_agent_size) <= 1e-6
    except Exception:
        agent_size_ok = False

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

raise SystemExit(
    0
    if episodes == target
    and has_selection
    and selection_ok
    and signature_ok
    and reward_setup_ok
    and agent_size_ok
    and fr_source_ok
    and physics_ok
    else 1
)
PY
}

eval_dir_for() {
  local label="$1"
  local train_seed="$2"
  local eval_seed="$3"
  printf '%s/logs/%s_%s_trainseed%s_testseed%s/evaluation_official\n' "$ROOT_DIR" "$RUN_TAG" "$label" "$train_seed" "$eval_seed"
}

selection_dir_for() {
  local label="$1"
  local train_seed="$2"
  printf '%s/%s_trainseed%s\n' "$SHARED_SELECTION_ROOT" "$label" "$train_seed"
}

resolve_base_spec() {
  if [ -n "$BASE_SPEC" ]; then
    printf '%s\n' "$BASE_SPEC"
    return 0
  fi
  local first_seed="${TRAIN_SEEDS[0]}"
  local model_dir
  model_dir="$(latest_completed_model_dir "${LABELS[0]}" "$first_seed" || true)"
  if [ -z "$model_dir" ]; then
    return 1
  fi
  local base_name
  local child_tag
  local candidate
  base_name="$(basename "$model_dir")"
  child_tag="${base_name#*__seed${first_seed}__}"
  candidate="$(find "$ROOT_DIR/ablation_experiments" -path "*/seed_batches/${child_tag}/results/post_eval_shared_spec.json" -print -quit)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  # run_optimized.sh appends its own YYYYMMDD_HHMMSS suffix to the
  # ablation-provided experiment name.  The child batch directory itself does
  # not contain that final training-launch suffix.
  if [[ "$child_tag" =~ ^(.+)_[0-9]{8}_[0-9]{6}$ ]]; then
    child_tag="${BASH_REMATCH[1]}"
    find "$ROOT_DIR/ablation_experiments" -path "*/seed_batches/${child_tag}/results/post_eval_shared_spec.json" -print -quit
  fi
}

prepare_specs() {
  local base_spec
  base_spec="$(resolve_base_spec)"
  if [ -z "$base_spec" ] || [ ! -f "$base_spec" ]; then
    echo "[preflight-error] BASE_SPEC not found. Set BASE_SPEC=/path/to/level3/post_eval_shared_spec.json" >&2
    return 1
  fi
  echo "[preflight] Level3 base spec: $base_spec"
  "$PYTHON_BIN" "$ROOT_DIR/prepare_level2_eval_seed_specs.py" \
    --base-spec "$base_spec" \
    --output-root "$SPEC_ROOT" \
    --episodes "$EVAL_EPISODES" \
    --testset-role "level3_dual_semantics_eval10x30" \
    --eval-seeds "${EVAL_SEEDS[@]}"
}

load_selection_result() {
  local result_path="$1"
  local base_spec="$2"
  "$PYTHON_BIN" - "$result_path" "$base_spec" "$MODEL_VARIANT" "$SELECTION_VALIDATION_EPISODES" "${SELECTION_VALIDATION_SEEDS[*]}" "$SELECTION_VALIDATION_CANDIDATES" "$EVAL_FR_MODE" "$REWARD_VERSION" "$REWARD_TERMINAL_ORDER_FIX" "$AGENT_SIZE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from selection_scoring import (
    SELECTION_RESULT_SCHEMA_VERSION,
    SELECTION_SCORE_SCHEMA_VERSION,
)

VOLATILE_SPEC_HASH_KEYS = {"spec_path"}
ALLOWED_CANDIDATES = {"auto", "final", "best", "best_by_team_sr", "latest_ep", "checkpoint"}

def stable_json_hash(payload):
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def spec_for_hash(spec):
    if not isinstance(spec, dict):
        return {}
    return {k: v for k, v in spec.items() if k not in VOLATILE_SPEC_HASH_KEYS}

def normalize_candidates(raw_value):
    tokens = [
        token.strip().lower()
        for token in str(raw_value or "").replace(";", ",").split(",")
    ]
    ordered = []
    for token in tokens:
        is_explicit_ep = token.startswith("ep") and token[2:].isdigit()
        if not token or (token not in ALLOWED_CANDIDATES and not is_explicit_ep) or token in ordered:
            continue
        ordered.append(token)
    if not ordered:
        ordered = ["best_by_team_sr", "best", "checkpoint", "final", "latest_ep"]
    return ordered

def parse_seeds(raw_value):
    seeds = []
    for token in str(raw_value or "").replace(",", " ").split():
        try:
            seed = int(token)
        except Exception:
            continue
        if seed not in seeds:
            seeds.append(seed)
    return seeds

def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def model_signature(model_dir):
    try:
        weight_files = sorted(Path(model_dir).glob("actor_*.weights.h5"))
        if not weight_files:
            return None
        hasher = hashlib.sha1()
        for weight_path in weight_files:
            hasher.update(weight_path.name.encode("utf-8", errors="ignore"))
            with weight_path.open("rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

path = Path(sys.argv[1])
base_spec = Path(sys.argv[2]).resolve()
expected_model_variant = str(sys.argv[3]).strip().lower() or "best_by_team_sr"
expected_validation_episodes = int(sys.argv[4])
expected_validation_seeds = parse_seeds(sys.argv[5])
expected_candidates = normalize_candidates(sys.argv[6])
if expected_model_variant != "auto" and expected_model_variant not in expected_candidates:
    expected_candidates = [expected_model_variant, *expected_candidates]
expected_eval_fr_mode = str(sys.argv[7]).strip().lower() or "checkpoint"
expected_reward_version = str(sys.argv[8]).strip() or "v1"
expected_terminal_order_fix = to_bool(sys.argv[9]) if str(sys.argv[9]).strip() else True
expected_agent_size = str(sys.argv[10]).strip() or "0.5"
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
if int(data.get("schema_version", 0) or 0) != SELECTION_RESULT_SCHEMA_VERSION:
    raise SystemExit(1)
selection = data.get("model_selection", {})
if not isinstance(selection, dict):
    raise SystemExit(1)
if int(selection.get("schema_version", 0) or 0) != SELECTION_RESULT_SCHEMA_VERSION:
    raise SystemExit(1)
if int(selection.get("selection_score_schema_version", 0) or 0) != SELECTION_SCORE_SCHEMA_VERSION:
    raise SystemExit(1)
context = selection.get("selection_context", {})
if not isinstance(context, dict):
    context = data.get("selection_context", {}) if isinstance(data.get("selection_context"), dict) else {}
if int(context.get("schema_version", 0) or 0) != SELECTION_RESULT_SCHEMA_VERSION:
    raise SystemExit(1)
if int(context.get("selection_score_schema_version", 0) or 0) != SELECTION_SCORE_SCHEMA_VERSION:
    raise SystemExit(1)
if not base_spec.exists():
    raise SystemExit(1)
base_spec_payload = json.loads(base_spec.read_text(encoding="utf-8"))
expected_spec_hash = stable_json_hash(spec_for_hash(base_spec_payload))
if str(context.get("official_spec_sha1", "") or "") != expected_spec_hash:
    raise SystemExit(1)
if str(context.get("official_spec_path", "") or "") and Path(str(context["official_spec_path"])).resolve() != base_spec:
    raise SystemExit(1)
if str(context.get("requested_model_variant", "") or "").strip().lower() != expected_model_variant:
    raise SystemExit(1)
if int(context.get("validation_episodes", 0) or 0) != expected_validation_episodes:
    raise SystemExit(1)
if [int(v) for v in context.get("validation_seeds", [])] != expected_validation_seeds:
    raise SystemExit(1)
if [str(v) for v in context.get("validation_candidates", [])] != expected_candidates:
    raise SystemExit(1)
if str(context.get("eval_fr_mode", "") or "").strip().lower() != expected_eval_fr_mode:
    raise SystemExit(1)
if str(context.get("reward_version", "") or "").strip() != expected_reward_version:
    raise SystemExit(1)
if bool(context.get("reward_terminal_order_fix", False)) != expected_terminal_order_fix:
    raise SystemExit(1)
try:
    if abs(float(context.get("agent_size")) - float(expected_agent_size)) > 1e-6:
        raise SystemExit(1)
except SystemExit:
    raise
except Exception:
    raise SystemExit(1)
model_path = str(selection.get("selected_model_path", "") or "").strip()
summary_path = str(selection.get("validation_selection_summary_path", "") or "").strip()
if not model_path or not Path(model_path).exists():
    raise SystemExit(1)
if not summary_path or not Path(summary_path).exists():
    raise SystemExit(1)
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
if int(summary.get("schema_version", 0) or 0) != SELECTION_RESULT_SCHEMA_VERSION:
    raise SystemExit(1)
if int(summary.get("selection_score_schema_version", 0) or 0) != SELECTION_SCORE_SCHEMA_VERSION:
    raise SystemExit(1)
selected_summary = summary.get("selected", {})
if not isinstance(selected_summary, dict):
    raise SystemExit(1)
try:
    resolved_model_path = Path(model_path).resolve()
    summary_model_path = Path(str(selected_summary.get("model_path", "") or "")).resolve()
except Exception:
    raise SystemExit(1)
if summary_model_path != resolved_model_path:
    raise SystemExit(1)
current_signature = model_signature(resolved_model_path)
selection_signature = str(selection.get("selected_model_signature", "") or "").strip()
summary_signature = str(selected_summary.get("model_signature", "") or "").strip()
if not current_signature or selection_signature != current_signature or summary_signature != current_signature:
    raise SystemExit(1)
print(model_path)
print(summary_path)
PY
}

select_shared_checkpoint() {
  local label="$1"
  local train_seed="$2"
  local model_dir="$3"
  run_shared_selection_job "$label" "$train_seed" "$model_dir"
  load_shared_selection_into_arrays "$label" "$train_seed"
}

run_shared_selection_job() {
  local label="$1"
  local train_seed="$2"
  local model_dir="$3"
  local selection_dir
  local selection_result
  local selection_log
  local selection_base_spec
  local force_args=()

  selection_dir="$(selection_dir_for "$label" "$train_seed")"
  selection_result="$selection_dir/selection_result.json"
  selection_log="$RUN_LOG_ROOT/${label}_trainseed${train_seed}_shared_selection.log"
  mkdir -p "$selection_dir"
  selection_base_spec="$(resolve_base_spec)"
  if [ -z "$selection_base_spec" ] || [ ! -f "$selection_base_spec" ]; then
    echo "[shared-selection-error] BASE_SPEC not found for $label trainseed${train_seed}" >&2
    return 1
  fi

  if ! truthy "$FORCE_RERUN" && load_selection_result "$selection_result" "$selection_base_spec" >/dev/null 2>/dev/null; then
    echo "[shared-selection] reuse valid selection_result for $label trainseed${train_seed}"
    return 0
  fi

  if truthy "$FORCE_RERUN"; then
    force_args+=(--force-rerun)
  fi
  echo "[shared-selection] selecting checkpoint for $label trainseed${train_seed}"
  echo "[shared-selection] validation seeds: ${SELECTION_VALIDATION_SEEDS[*]} | episodes/seed=$SELECTION_VALIDATION_EPISODES"
  (
    REWARD_VERSION="$REWARD_VERSION" \
    REWARD_TERMINAL_ORDER_FIX="$REWARD_TERMINAL_ORDER_FIX" \
    AGENT_SIZE="$AGENT_SIZE" \
    EVAL_FR_MODE="$EVAL_FR_MODE" \
    EVAL_EPISODE_PARALLELISM="$SELECTION_EVAL_EPISODE_PARALLELISM" \
    EVAL_ENV_STEP_THREADS="$SELECTION_EVAL_ENV_STEP_THREADS" \
    EVAL_DEBUG_ACTION_STEPS="$EVAL_DEBUG_ACTION_STEPS" \
    FAST_ARTIFACTS="$FAST_ARTIFACTS" \
    EVAL_LIGHT_MODE="$EVAL_LIGHT_MODE" \
    SUPPRESS_TERRAIN_OUTPUT="$SUPPRESS_TERRAIN_OUTPUT" \
    SUPPRESS_REWARD_CONFIG_OUTPUT="$SUPPRESS_REWARD_CONFIG_OUTPUT" \
    OFFICIAL_EVAL_QUIET=1 \
    QUIET_OUTPUT=1 \
    TQDM_DISABLE=1 \
    EVAL_ARTIFACT_FILENAME_TAG="$SUMMARY_FILENAME_TAG" \
    "$PYTHON_BIN" "$ROOT_DIR/official_eval_with_matched_validation.py" \
      --experiment-root "$model_dir" \
      --official-spec "$selection_base_spec" \
      --output-dir "$selection_dir" \
      --model-variant "$MODEL_VARIANT" \
      --selection-protocol matched_validation \
      --validation-episodes "$SELECTION_VALIDATION_EPISODES" \
      --validation-seeds "${SELECTION_VALIDATION_SEEDS[@]}" \
      --validation-candidates "$SELECTION_VALIDATION_CANDIDATES" \
      --selection-only \
      --quiet-output 1 \
      --python-bin "$PYTHON_BIN" \
      "${force_args[@]}"
  ) 2>&1 | sed -u "s/^/[shared-selection ${label} trainseed${train_seed}] /" | tee "$selection_log"

  load_selection_result "$selection_result" "$selection_base_spec" >/dev/null
}

load_shared_selection_into_arrays() {
  local label="$1"
  local train_seed="$2"
  local selection_dir
  local selection_result
  local selection_base_spec
  local selected_model_path
  local selection_summary_path
  local key="${label}|${train_seed}"
  local tmp_file
  local status=0

  selection_dir="$(selection_dir_for "$label" "$train_seed")"
  selection_result="$selection_dir/selection_result.json"
  selection_base_spec="$(resolve_base_spec)"
  if [ -z "$selection_base_spec" ] || [ ! -f "$selection_base_spec" ]; then
    echo "[shared-selection-error] BASE_SPEC not found for $label trainseed${train_seed}" >&2
    return 1
  fi

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/level3_shared_selection_result.XXXXXX")"
  if ! load_selection_result "$selection_result" "$selection_base_spec" >"$tmp_file"; then
    status=1
  fi
  selected_model_path="$(sed -n '1p' "$tmp_file" 2>/dev/null || true)"
  selection_summary_path="$(sed -n '2p' "$tmp_file" 2>/dev/null || true)"
  rm -f "$tmp_file"

  if [ "$status" -ne 0 ]; then
    echo "[shared-selection-error] invalid selection_result for $label trainseed${train_seed}: $selection_result" >&2
    return 1
  fi
  if [ -z "$selected_model_path" ]; then
    echo "[shared-selection-error] selected checkpoint missing for $label trainseed${train_seed}" >&2
    return 1
  fi
  SHARED_SELECTED_MODEL_PATHS["$key"]="$selected_model_path"
  SHARED_SELECTION_SUMMARIES["$key"]="$selection_summary_path"
  echo "[shared-selection] selected $label trainseed${train_seed}: $selected_model_path"
}

run_shared_selection_all() {
  local label
  local train_seed
  local model_dir
  local selection_failed=0
  if [ "$SELECTION_MAX_PARALLEL" -le 1 ]; then
    for label in "${LABELS[@]}"; do
      for train_seed in "${TRAIN_SEEDS[@]}"; do
        model_dir="$(latest_completed_model_dir "$label" "$train_seed")"
        select_shared_checkpoint "$label" "$train_seed" "$model_dir"
      done
    done
    return 0
  fi

  echo "[shared-selection] launching shared selection with max_parallel=$SELECTION_MAX_PARALLEL"
  for label in "${LABELS[@]}"; do
    for train_seed in "${TRAIN_SEEDS[@]}"; do
      model_dir="$(latest_completed_model_dir "$label" "$train_seed")"
      while [ "$(active_job_count)" -ge "$SELECTION_MAX_PARALLEL" ]; do
        if ! wait -n; then
          selection_failed=1
        fi
      done
      (
        run_shared_selection_job "$label" "$train_seed" "$model_dir"
      ) &
    done
  done

  while [ "$(active_job_count)" -gt 0 ]; do
    if ! wait -n; then
      selection_failed=1
    fi
  done

  if [ "$selection_failed" -ne 0 ]; then
    echo "[shared-selection-error] at least one shared selection job failed." >&2
    return 1
  fi

  for label in "${LABELS[@]}"; do
    for train_seed in "${TRAIN_SEEDS[@]}"; do
      load_shared_selection_into_arrays "$label" "$train_seed"
    done
  done
}

run_eval_job() {
  local label="$1"
  local train_seed="$2"
  local eval_seed="$3"
  local model_dir="$4"
  local spec="$SPEC_ROOT/eval_seed_${eval_seed}/post_eval_shared_spec.json"
  local eval_dir
  local log_file
  local force_args=()
  local preselected_args=()
  local helper_selection_protocol="$SELECTION_PROTOCOL"
  local key="${label}|${train_seed}"
  local eval_env_cmd=(env)
  eval_dir="$(eval_dir_for "$label" "$train_seed" "$eval_seed")"
  log_file="$RUN_LOG_ROOT/${label}_trainseed${train_seed}_testseed${eval_seed}.log"

  if truthy "$FORCE_RERUN"; then
    force_args+=(--force-rerun)
  elif eval_completed "$eval_dir" "$model_dir"; then
    echo "[skip-eval] $label trainseed${train_seed} testseed${eval_seed}: complete"
    return 0
  elif [ -f "$eval_dir/evaluation_results.json" ]; then
    force_args+=(--force-rerun)
  fi

  if use_shared_selection; then
    if [ -z "${SHARED_SELECTED_MODEL_PATHS[$key]:-}" ]; then
      echo "[eval-error] missing shared selected checkpoint for $label trainseed${train_seed}" >&2
      return 2
    fi
    helper_selection_protocol="fixed"
    preselected_args+=(
      --preselected-model-path "${SHARED_SELECTED_MODEL_PATHS[$key]}"
    )
    if [ -n "${SHARED_SELECTION_SUMMARIES[$key]:-}" ]; then
      preselected_args+=(
        --preselected-selection-summary "${SHARED_SELECTION_SUMMARIES[$key]}"
      )
    fi
  fi

  (
    if [ -f "$CONDA_SH" ]; then
      # shellcheck source=/home/tang/miniconda3/etc/profile.d/conda.sh
      source "$CONDA_SH"
      conda activate "$CONDA_ENV_NAME"
      PYTHON_BIN="$(command -v python3)"
      configure_eval_gpu_env "$PYTHON_BIN"
    fi
    configure_eval_gpu_env "$PYTHON_BIN"
    case "${EVAL_FR_MODE,,}" in
      checkpoint|checkpoint_fr|model|model_fr|corresponding|corresponding_fr)
        eval_env_cmd+=( -u FORCE_EVAL_ACTION_FORCE_RATIO -u ACTION_FORCE_RATIO )
        ;;
      forced|fixed|fixed_fr)
        eval_env_cmd+=( "FORCE_EVAL_ACTION_FORCE_RATIO=${FORCE_EVAL_ACTION_FORCE_RATIO_VALUE}" )
        ;;
      *)
        echo "[eval-error] unknown EVAL_FR_MODE=$EVAL_FR_MODE (expected checkpoint or forced)" >&2
        exit 2
        ;;
    esac
    "${eval_env_cmd[@]}" \
      REWARD_VERSION="$REWARD_VERSION" \
      REWARD_TERMINAL_ORDER_FIX="$REWARD_TERMINAL_ORDER_FIX" \
      AGENT_SIZE="$AGENT_SIZE" \
      EVAL_FR_MODE="$EVAL_FR_MODE" \
      EVAL_EPISODE_PARALLELISM="$EVAL_EPISODE_PARALLELISM" \
      EVAL_ENV_STEP_THREADS="$EVAL_ENV_STEP_THREADS" \
      EVAL_DEBUG_ACTION_STEPS="$EVAL_DEBUG_ACTION_STEPS" \
      FAST_ARTIFACTS="$FAST_ARTIFACTS" \
      EVAL_LIGHT_MODE="$EVAL_LIGHT_MODE" \
      SUPPRESS_TERRAIN_OUTPUT="$SUPPRESS_TERRAIN_OUTPUT" \
      SUPPRESS_REWARD_CONFIG_OUTPUT="$SUPPRESS_REWARD_CONFIG_OUTPUT" \
      OFFICIAL_EVAL_QUIET=1 \
      QUIET_OUTPUT=1 \
      TQDM_DISABLE=1 \
      EVAL_ARTIFACT_FILENAME_TAG="$SUMMARY_FILENAME_TAG" \
      "$PYTHON_BIN" "$ROOT_DIR/official_eval_with_matched_validation.py" \
        --experiment-root "$model_dir" \
        --official-spec "$spec" \
        --output-dir "$eval_dir" \
        --model-variant "$MODEL_VARIANT" \
        --selection-protocol "$helper_selection_protocol" \
        --quiet-output 1 \
        --python-bin "$PYTHON_BIN" \
        "${preselected_args[@]}" \
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

  is_positive_int "$TRAIN_EPISODES" || { echo "[preflight-error] TRAIN_EPISODES must be positive: $TRAIN_EPISODES" >&2; errors=$((errors + 1)); }
  is_positive_int "$EVAL_EPISODES" || { echo "[preflight-error] EVAL_EPISODES must be positive: $EVAL_EPISODES" >&2; errors=$((errors + 1)); }
  is_positive_int "$MAX_PARALLEL" || { echo "[preflight-error] MAX_PARALLEL must be positive: $MAX_PARALLEL" >&2; errors=$((errors + 1)); }
  if use_shared_selection; then
    is_positive_int "$SELECTION_MAX_PARALLEL" || { echo "[preflight-error] SELECTION_MAX_PARALLEL must be positive: $SELECTION_MAX_PARALLEL" >&2; errors=$((errors + 1)); }
    is_positive_int "$SELECTION_VALIDATION_EPISODES" || { echo "[preflight-error] SELECTION_VALIDATION_EPISODES must be positive: $SELECTION_VALIDATION_EPISODES" >&2; errors=$((errors + 1)); }
    if [ "${#SELECTION_VALIDATION_SEEDS[@]}" -eq 0 ]; then
      echo "[preflight-error] SELECTION_VALIDATION_SEEDS must not be empty for shared selection" >&2
      errors=$((errors + 1))
    fi
  fi
  [ -f "$ROOT_DIR/official_eval_with_matched_validation.py" ] || { echo "[preflight-error] official_eval_with_matched_validation.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/selection_scoring.py" ] || { echo "[preflight-error] selection_scoring.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/prepare_level2_eval_seed_specs.py" ] || { echo "[preflight-error] prepare_level2_eval_seed_specs.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/summarize_level2_dual_semantics_eval_sweep.py" ] || { echo "[preflight-error] summarize_level2_dual_semantics_eval_sweep.py missing" >&2; errors=$((errors + 1)); }
  [ -x "$PYTHON_BIN" ] || { echo "[preflight-error] PYTHON_BIN not executable: $PYTHON_BIN" >&2; errors=$((errors + 1)); }
  if [ -x "$PYTHON_BIN" ] && ! "$PYTHON_BIN" -c 'from selection_scoring import SELECTION_RESULT_SCHEMA_VERSION, SELECTION_SCORE_SCHEMA_VERSION' >/dev/null 2>&1; then
    echo "[preflight-error] shared selection schema module cannot be imported" >&2
    errors=$((errors + 1))
  fi
  echo "[preflight] reward version: $REWARD_VERSION | terminal_order_fix=$REWARD_TERMINAL_ORDER_FIX"
  case "${EVAL_FR_MODE,,}" in
    checkpoint|checkpoint_fr|model|model_fr|corresponding|corresponding_fr|forced|fixed|fixed_fr) ;;
    *) echo "[preflight-error] unknown EVAL_FR_MODE=$EVAL_FR_MODE (expected checkpoint or forced)" >&2; errors=$((errors + 1)); ;;
  esac

  for eval_seed in "${EVAL_SEEDS[@]}"; do
    spec="$SPEC_ROOT/eval_seed_${eval_seed}/post_eval_shared_spec.json"
    [ -f "$spec" ] || { echo "[preflight-error] eval spec missing after preparation: $spec" >&2; errors=$((errors + 1)); }
  done

  for label in "${LABELS[@]}"; do
    for train_seed in "${TRAIN_SEEDS[@]}"; do
      model_dir="$(latest_completed_model_dir "$label" "$train_seed" || true)"
      if [ -z "$model_dir" ]; then
        echo "[preflight-error] completed Level3 model missing: $label trainseed${train_seed}" >&2
        errors=$((errors + 1))
        continue
      fi
      if ! use_shared_selection && ! model_has_variant "$model_dir" "$MODEL_VARIANT"; then
        echo "[preflight-error] model variant missing: $MODEL_VARIANT | $model_dir" >&2
        errors=$((errors + 1))
      fi
    done
  done

  if [ "$errors" -ne 0 ]; then
    echo "[preflight] failed with $errors error(s)." >&2
    return 1
  fi
  echo "[preflight] ok: Level3 specs and completed models are available."
  echo "[preflight] FR mode: $EVAL_FR_MODE"
  echo "[preflight] official eval speed: episode_parallelism=$EVAL_EPISODE_PARALLELISM | env_step_threads=$EVAL_ENV_STEP_THREADS | fast_artifacts=$FAST_ARTIFACTS | light_mode=$EVAL_LIGHT_MODE | debug_action_steps=$EVAL_DEBUG_ACTION_STEPS"
  echo "[preflight] selection eval speed: max_parallel=$SELECTION_MAX_PARALLEL | episode_parallelism=$SELECTION_EVAL_EPISODE_PARALLELISM | env_step_threads=$SELECTION_EVAL_ENV_STEP_THREADS"
  echo "[preflight] selection protocol: $SELECTION_PROTOCOL | model variant: $MODEL_VARIANT"
  if use_shared_selection; then
    echo "[preflight] shared validation seeds: ${SELECTION_VALIDATION_SEEDS[*]} | episodes/seed=$SELECTION_VALIDATION_EPISODES | candidates=$SELECTION_VALIDATION_CANDIDATES"
  fi
}

prepare_specs
preflight_checks
if truthy "$PREFLIGHT_ONLY"; then
  echo "[preflight] PREFLIGHT_ONLY=1, exiting before launching eval jobs."
  exit 0
fi

if use_shared_selection; then
  run_shared_selection_all
fi

for label in "${LABELS[@]}"; do
  for train_seed in "${TRAIN_SEEDS[@]}"; do
    model_dir="$(latest_completed_model_dir "$label" "$train_seed")"
    for eval_seed in "${EVAL_SEEDS[@]}"; do
      while [ "$(active_job_count)" -ge "$MAX_PARALLEL" ]; do
        wait -n
      done
      run_eval_job "$label" "$train_seed" "$eval_seed" "$model_dir"
    done
  done
done

wait
echo "Level3 dual-semantics 10x30 official eval complete. Logs: $RUN_LOG_ROOT"

SUMMARY_FLAGS=()
if truthy "$STRICT_SUMMARY"; then
  SUMMARY_FLAGS+=(--strict)
fi
case "${EVAL_FR_MODE,,}" in
  forced|fixed|fixed_fr)
    SUMMARY_FLAGS+=(--allow-forced-fr)
    ;;
esac

MPLCONFIGDIR=/tmp/mplconfig "$PYTHON_BIN" "$ROOT_DIR/summarize_level2_dual_semantics_eval_sweep.py" \
  --run-tag "$RUN_TAG" \
  --labels "${LABELS[@]}" \
  --train-seeds "${TRAIN_SEEDS[@]}" \
  --eval-seeds "${EVAL_SEEDS[@]}" \
  --expected-episodes "$EVAL_EPISODES" \
  --filename-tag "$SUMMARY_FILENAME_TAG" \
  "${SUMMARY_FLAGS[@]}"

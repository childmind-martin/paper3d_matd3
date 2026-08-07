#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TRAIN_MAX_PARALLEL="${TRAIN_MAX_PARALLEL:-$MAX_PARALLEL}"
EVAL_MAX_PARALLEL="${EVAL_MAX_PARALLEL:-$MAX_PARALLEL}"
SEEDS=(${SEEDS:-101 202 936487})
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FORCE_EVAL_RERUN="${FORCE_EVAL_RERUN:-${FORCE_RERUN:-0}}"
UTILITY_PYTHON_BIN="${UTILITY_PYTHON_BIN:-${EVAL_PYTHON_BIN:-${TRAIN_PYTHON_BIN:-python3}}}"

SPEC_ROOT="${SPEC_ROOT:-/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260331_220752_testset2_20260409/seed_batches}"
RUN_TAG="${RUN_TAG:-level2_ms_official}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-/home/tang/matd3/parallel_logs/${RUN_TAG}_ep${EPISODES}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_LOG_ROOT"

COMMON_ENV=(
  USE_FIXED_POSITIONS=1
  POSITIONS_FILE=/home/tang/matd3/saved_positions/strict_ablation_seed88_groupB.json
  USE_SCENARIO_SEED=1
  SCENARIO_SEED=88
  TRAIN_ENV_SEQUENCE_SEED=88
  TERRAIN_BASE_SEED=88
  TRAIN_OBSTACLE_SEQUENCE_MODE=post_eval_family
  TRAIN_OBSTACLE_SEQUENCE_NAMESPACE=train_obstacle
  USE_DYNAMIC_OBSTACLES=1
  RANDOM_TERRAIN=0
  SEMI_RANDOM_TERRAIN=0
  DETERMINISTIC_TRAIN_ENV_SEQUENCE=0
  TERRAIN_COMPLEXITY_LEVEL=3
  MAP_SIZE=200
  MOUNTAIN_MIN_DISTANCE=55
  PEAK_JITTER_RANGE=0
  PEAK_CENTER_JITTER_RANGE=0
  PEAK_HEIGHT_JITTER_RATIO_MIN=0
  PEAK_HEIGHT_JITTER_RATIO_MAX=0
  PEAK_HEIGHT_MAX_SCALE=1
  TERRAIN_VARIANT_NOISE_RATIO=0
  ENABLE_ROLE_SHUFFLE=1
  ACTION_FORCE_RATIO=0.50
  ACTION_FORCE_RATIO_SCHEDULE_PCT=0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32
  FORCE_EVAL_ACTION_FORCE_RATIO=
  AUTO_EVAL=1
  AUTO_EVAL_MODE=official_spec
  OFFICIAL_POST_EVAL_MODEL_VARIANT=best_by_team_sr
  OFFICIAL_POST_EVAL_SELECTION_PROTOCOL=matched_validation
  OFFICIAL_POST_EVAL_VALIDATION_EPISODES=10
  OFFICIAL_POST_EVAL_VALIDATION_CANDIDATES=best_by_team_sr,best,checkpoint,final,latest_ep
  OFFICIAL_EVAL_QUIET=1
  EVAL_ARTIFACT_FILENAME_TAG=model_fr
  SAVE_INTERVAL=0
  SAVE_TRAINING_RESUME_STATE=0
  BATCH_RESUME_POLICY=completed_units_only_restart_incomplete
)

if [ -n "${LABELS_OVERRIDE:-}" ]; then
  # Space-separated labels, used by narrow wrapper scripts while preserving this script's default behavior.
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

common_env_value() {
  local key="$1"
  local default_value="${2:-}"
  local item
  for item in "${COMMON_ENV[@]}"; do
    if [[ "$item" == "$key="* ]]; then
      printf '%s\n' "${item#*=}"
      return 0
    fi
  done
  printf '%s\n' "$default_value"
}

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

check_python_bin() {
  local bin="$1"
  local label="$2"
  if [[ "$bin" == */* ]]; then
    if [ ! -x "$bin" ]; then
      echo "[preflight-error] $label is not executable: $bin" >&2
      return 1
    fi
  elif ! command -v "$bin" >/dev/null 2>&1; then
    echo "[preflight-error] $label not found in PATH: $bin" >&2
    return 1
  fi
  return 0
}

label_supported() {
  case "$1" in
    matd3_single_q|matd3_dual_q|matd3_separated_gradient|matd3_separated_hybrid_actor|matd3_separated_hybrid_actor_alpha20)
      return 0
      ;;
    matd3_full_dual_semantic|matd3_collapsed_replay|matd3_no_corrected_target_reconstruction|matd3_full_dual_semantic_cross_agent_ref|matd3_cross_agent_ref_agent_quality|matd3_cross_agent_ref_soft_advantage|matd3_cross_agent_ref_selector_mix|matd3_cross_agent_ref_reward_to_success_selector|matd3_cross_agent_ref_reward_to_success_selector_tail0|matd3_cross_agent_ref_reward_to_success_selector_tail01|matd3_cross_agent_ref_reward_to_success_selector_tail10|matd3_cross_agent_ref_no_quality_gate|matd3_cross_agent_ref_behavior_label)
      return 0
      ;;
    ds_matd3_original|ds_matd3_uniform|ds_matd3_legacy_per)
      return 0
      ;;
    maddpg_baseline|maddpg_dual_q|maddpg_separated_gradient)
      return 0
      ;;
    mappo_baseline|mappo_fusion_only|mappo_separated_gradient)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

model_dirs_newest_first() {
  local label="$1"
  local seed="$2"
  find "$ROOT_DIR/models" -mindepth 1 -maxdepth 1 -type d \
    -name "${RUN_TAG}_${label}_seed${seed}_*" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
}

latest_model_dir() {
  local label="$1"
  local seed="$2"
  model_dirs_newest_first "$label" "$seed" | head -1
}

latest_completed_model_dir() {
  local label="$1"
  local seed="$2"
  local candidate
  while IFS= read -r candidate; do
    if [ -n "$candidate" ] && model_completed "$candidate" "$seed"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(model_dirs_newest_first "$label" "$seed")
  return 1
}

model_completed() {
  local model_dir="$1"
  local seed="$2"
  "$UTILITY_PYTHON_BIN" - "$model_dir" "$EPISODES" "$seed" <<'PY'
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

official_eval_completed() {
  local model_dir="$1"
  local seed="$2"
  local result_path="$ROOT_DIR/logs/$(basename "$model_dir")/evaluation_official/evaluation_results.json"
  local spec_path="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
  local expected_protocol
  local expected_validation_episodes
  expected_protocol="$(common_env_value OFFICIAL_POST_EVAL_SELECTION_PROTOCOL matched_validation)"
  expected_validation_episodes="$(common_env_value OFFICIAL_POST_EVAL_VALIDATION_EPISODES 10)"
  "$UTILITY_PYTHON_BIN" - "$result_path" "$spec_path" "$expected_protocol" "$expected_validation_episodes" "$model_dir/results.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec_path = Path(sys.argv[2])
expected_protocol = str(sys.argv[3] or "").strip()
model_results_path = Path(sys.argv[5])
try:
    expected_validation_episodes = int(sys.argv[4])
except Exception:
    expected_validation_episodes = 0
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
try:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    target_episodes = int(spec.get("episodes", 0) or 0)
except Exception:
    target_episodes = 0
summary = data.get("summary", {}) if isinstance(data, dict) else {}
episodes = summary.get("episodes", data.get("episodes") if isinstance(data, dict) else None)
try:
    episodes = int(episodes)
except Exception:
    raise SystemExit(1)
has_selection = isinstance(data.get("model_selection"), dict) if isinstance(data, dict) else False
selection = data.get("model_selection", {}) if isinstance(data, dict) else {}
setup = data.get("evaluation_setup", {}) if isinstance(data, dict) else {}
if not isinstance(setup, dict):
    setup = {}
fr_source = str(setup.get("action_force_ratio_source", "") or "").strip()
fr_source_ok = bool(fr_source) and fr_source != "forced_override"
protocol_ok = True
validation_ok = True
if expected_protocol:
    protocol_ok = str(selection.get("selection_protocol", "")).strip() == expected_protocol
if expected_protocol == "matched_validation":
    summary_path = Path(str(selection.get("validation_selection_summary_path", "")).strip())
    if not summary_path.exists():
        validation_ok = False
    else:
        try:
            validation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            validation_ok = False
        else:
            validation_ok = (
                str(validation_summary.get("selection_protocol", "")).strip() == "matched_validation"
                and (
                    expected_validation_episodes <= 0
                    or int(validation_summary.get("validation_episodes", 0) or 0) == expected_validation_episodes
                )
            )
has_expected_episodes = target_episodes <= 0 or episodes == target_episodes

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

physics_ok = True
def load_training_results(path: Path) -> dict:
    candidates = []
    if path.exists():
        candidates.append(path)
    try:
        log_root = Path(sys.argv[1]).parents[1]
        candidates.extend(sorted(log_root.glob("**/results.json")))
    except Exception:
        pass
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}

model_data = load_training_results(model_results_path)
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

raise SystemExit(0 if has_selection and protocol_ok and validation_ok and has_expected_episodes and fr_source_ok and physics_ok else 1)
PY
}

model_has_eval_candidate() {
  local model_dir="$1"
  local variant
  for variant in best_by_team_sr best checkpoint final; do
    if compgen -G "$model_dir/$variant/actor_*.weights.h5" >/dev/null; then
      return 0
    fi
  done
  if find "$model_dir" -mindepth 2 -maxdepth 2 -type f -path "$model_dir/ep*/actor_*.weights.h5" -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

delete_incomplete_artifacts() {
  local model_dir="$1"
  local base_name
  local log_dir
  base_name="$(basename "$model_dir")"
  log_dir="$ROOT_DIR/logs/$base_name"
  case "$model_dir" in
    "$ROOT_DIR/models/${RUN_TAG}_"*|"$ROOT_DIR/models/${RUN_TAG}"_*)
      ;;
    *)
      echo "[refuse-delete] unexpected model dir: $model_dir" >&2
      return 1
      ;;
  esac
  echo "[delete-incomplete] model: $model_dir"
  rm -rf "$model_dir"
  if [ -d "$log_dir" ]; then
    echo "[delete-incomplete] logs : $log_dir"
    rm -rf "$log_dir"
  fi
}

delete_incomplete_artifacts_for_label_seed() {
  local label="$1"
  local seed="$2"
  local candidate
  local deleted=0

  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    if model_completed "$candidate" "$seed"; then
      continue
    fi
    delete_incomplete_artifacts "$candidate"
    deleted=$((deleted + 1))
  done < <(model_dirs_newest_first "$label" "$seed")

  echo "[delete-incomplete] $label seed${seed}: deleted $deleted incomplete run(s)"
}

run_official_eval_job() {
  local label="$1"
  local seed="$2"
  local model_dir="$3"
  local spec="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
  local eval_dir="$ROOT_DIR/logs/$(basename "$model_dir")/evaluation_official"
  local log_file="${RUN_LOG_ROOT}/${label}_seed${seed}_official_eval.log"
  local eval_python="${EVAL_PYTHON_BIN:-${TRAIN_PYTHON_BIN:-python3}}"
  local model_variant
  local selection_protocol
  local validation_episodes
  local validation_candidates
  local quiet_output
  model_variant="$(common_env_value OFFICIAL_POST_EVAL_MODEL_VARIANT best_by_team_sr)"
  selection_protocol="$(common_env_value OFFICIAL_POST_EVAL_SELECTION_PROTOCOL matched_validation)"
  validation_episodes="$(common_env_value OFFICIAL_POST_EVAL_VALIDATION_EPISODES 10)"
  validation_candidates="$(common_env_value OFFICIAL_POST_EVAL_VALIDATION_CANDIDATES best_by_team_sr,best,checkpoint,final,latest_ep)"
  quiet_output="$(common_env_value OFFICIAL_EVAL_QUIET 1)"

  (
    env -u FORCE_EVAL_ACTION_FORCE_RATIO "${COMMON_ENV[@]}" \
      SEED="$seed" \
      OFFICIAL_POST_EVAL_SPEC="$spec" \
      "$eval_python" "$ROOT_DIR/official_eval_with_matched_validation.py" \
        --experiment-root "$model_dir" \
        --official-spec "$spec" \
        --output-dir "$eval_dir" \
        --model-variant "$model_variant" \
        --selection-protocol "$selection_protocol" \
        --validation-episodes "$validation_episodes" \
        --validation-candidates "$validation_candidates" \
        --quiet-output "$quiet_output" \
        --python-bin "$eval_python" \
        --force-rerun
  ) 2>&1 | sed -u "s/^/[${label} seed${seed} official-eval] /" | tee "$log_file" &

  echo "[launch-eval] $label seed${seed} -> $log_file"
}

run_level2_job() {
  local label="$1"
  local seed="$2"
  local resume_model="${3:-}"
  local spec="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
  local exp="${RUN_TAG}_${label}_seed${seed}"
  local log_file="${RUN_LOG_ROOT}/${label}_seed${seed}.log"
  local resume_env=()
  local r2s_tail_weight="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}"
  if [ -n "$resume_model" ]; then
    echo "[refuse-resume] batch policy forbids episode/checkpoint continuation: $label seed${seed}" >&2
    return 2
  fi
  case "$label" in
    matd3_cross_agent_ref_reward_to_success_selector_tail0)
      r2s_tail_weight="0.0"
      ;;
    matd3_cross_agent_ref_reward_to_success_selector_tail01)
      r2s_tail_weight="0.1"
      ;;
    matd3_cross_agent_ref_reward_to_success_selector_tail10)
      r2s_tail_weight="1.0"
      ;;
  esac

  (
    case "$label" in
      matd3_single_q)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=false \
          MATD3_USE_SEPARATED_GRADIENT=false \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_full_dual_semantic)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=hard \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_full_dual_semantic_cross_agent_ref)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=1 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=1 \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=hard \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_cross_agent_ref_agent_quality)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=1 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=1 \
          CROSS_AGENT_REFERENCE_GATE_MODE=agent_quality \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=hard \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_cross_agent_ref_soft_advantage)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=1 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=1 \
          CROSS_AGENT_REFERENCE_GATE_MODE=agent_quality \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=soft_advantage \
          CROSS_AGENT_REFERENCE_SELECTOR_ALPHA="${CROSS_AGENT_REFERENCE_SELECTOR_ALPHA:-0.7}" \
          CROSS_AGENT_REFERENCE_SELECTOR_Q_TAU="${CROSS_AGENT_REFERENCE_SELECTOR_Q_TAU:-500.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP="${CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP:-5.0}" \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_cross_agent_ref_selector_mix)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=1 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=1 \
          CROSS_AGENT_REFERENCE_GATE_MODE=agent_quality \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=1 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=selector_mix \
          CROSS_AGENT_REFERENCE_SELECTOR_ALPHA="${CROSS_AGENT_REFERENCE_SELECTOR_ALPHA:-0.7}" \
          CROSS_AGENT_REFERENCE_SELECTOR_Q_TAU="${CROSS_AGENT_REFERENCE_SELECTOR_Q_TAU:-500.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_LR="${CROSS_AGENT_REFERENCE_SELECTOR_LR:-1e-4}" \
          CROSS_AGENT_REFERENCE_SELECTOR_HIDDEN="${CROSS_AGENT_REFERENCE_SELECTOR_HIDDEN:-128,64}" \
          CROSS_AGENT_REFERENCE_SELECTOR_INIT_LOGIT="${CROSS_AGENT_REFERENCE_SELECTOR_INIT_LOGIT:--2.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP="${CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP:-5.0}" \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_cross_agent_ref_reward_to_success_selector|matd3_cross_agent_ref_reward_to_success_selector_tail0|matd3_cross_agent_ref_reward_to_success_selector_tail01|matd3_cross_agent_ref_reward_to_success_selector_tail10)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="$r2s_tail_weight" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=1 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=1 \
          CROSS_AGENT_REFERENCE_GATE_MODE=agent_quality \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=1 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=reward_to_success_priority \
          CROSS_AGENT_REFERENCE_SELECTOR_ALPHA="${CROSS_AGENT_REFERENCE_SELECTOR_ALPHA:-0.7}" \
          CROSS_AGENT_REFERENCE_SELECTOR_Q_TAU="${CROSS_AGENT_REFERENCE_SELECTOR_Q_TAU:-500.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_LR="${CROSS_AGENT_REFERENCE_SELECTOR_LR:-1e-4}" \
          CROSS_AGENT_REFERENCE_SELECTOR_HIDDEN="${CROSS_AGENT_REFERENCE_SELECTOR_HIDDEN:-128,64}" \
          CROSS_AGENT_REFERENCE_SELECTOR_INIT_LOGIT="${CROSS_AGENT_REFERENCE_SELECTOR_INIT_LOGIT:--2.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP="${CROSS_AGENT_REFERENCE_SELECTOR_ADV_CLIP:-5.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_REWARD_TAU="${CROSS_AGENT_REFERENCE_SELECTOR_REWARD_TAU:-500.0}" \
          CROSS_AGENT_REFERENCE_SELECTOR_REWARD_TIEBREAK="${CROSS_AGENT_REFERENCE_SELECTOR_REWARD_TIEBREAK:-0.05}" \
          CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_WINDOW="${CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_WINDOW:-100}" \
          CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_DELTA="${CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_DELTA:-0.02}" \
          CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_MIN_RATE="${CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_MIN_RATE:-0.05}" \
          CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_MIN_EPISODES="${CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_STABLE_MIN_EPISODES:-200}" \
          CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_RAMP_EPISODES="${CROSS_AGENT_REFERENCE_SELECTOR_SUCCESS_RAMP_EPISODES:-50}" \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_cross_agent_ref_no_quality_gate)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=1 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=hard \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_cross_agent_ref_behavior_label)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=1 \
          CROSS_AGENT_REFERENCE_COEF="${CROSS_AGENT_REFERENCE_COEF:-0.03}" \
          CROSS_AGENT_REFERENCE_START_EPISODE="${CROSS_AGENT_REFERENCE_START_EPISODE:-50}" \
          CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD="${CROSS_AGENT_REFERENCE_PROGRESS_THRESHOLD:-0.0005}" \
          CROSS_AGENT_REFERENCE_MARGIN="${CROSS_AGENT_REFERENCE_MARGIN:-0.0}" \
          CROSS_AGENT_REFERENCE_HEAD_WEIGHT="${CROSS_AGENT_REFERENCE_HEAD_WEIGHT:-1.0}" \
          CROSS_AGENT_REFERENCE_TAIL_WEIGHT="${CROSS_AGENT_REFERENCE_TAIL_WEIGHT:-0.3}" \
          CROSS_AGENT_REFERENCE_USE_CLEAN_LABEL=0 \
          CROSS_AGENT_REFERENCE_EXCLUDE_RANDOM=1 \
          CROSS_AGENT_REFERENCE_QUALITY_GATE=1 \
          CROSS_AGENT_REFERENCE_SELECTOR_ENABLED=0 \
          CROSS_AGENT_REFERENCE_SELECTOR_MODE=hard \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      ds_matd3_uniform)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          ALGORITHM_NAME=DS_MATD3_UNIFORM \
          PER_ENABLED=0 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      ds_matd3_legacy_per|ds_matd3_original)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          ALGORITHM_NAME=DS_MATD3_LEGACY_PER \
          PER_ENABLED=1 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_collapsed_replay)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=collapsed_replay \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=true \
          CROSS_AGENT_REFERENCE_ENABLED=0 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_no_corrected_target_reconstruction)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          MATD3_ACTION_SEMANTICS_MODE=dual \
          MATD3_RECONSTRUCT_CORRECTED_TARGET=false \
          CROSS_AGENT_REFERENCE_ENABLED=0 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_dual_q)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=false \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_separated_gradient)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=false \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_separated_hybrid_actor)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=true \
          MATD3_HYBRID_ACTOR_ALPHA=0.80 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      matd3_separated_hybrid_actor_alpha20)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MATD3_USE_DUAL_Q=true \
          MATD3_USE_SEPARATED_GRADIENT=true \
          MATD3_USE_HYBRID_ACTOR_OBJECTIVE=true \
          MATD3_HYBRID_ACTOR_ALPHA=0.20 \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 matd3
        ;;
      maddpg_baseline)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MADDPG_USE_DUAL_Q=false \
          MADDPG_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 maddpg
        ;;
      maddpg_dual_q)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MADDPG_USE_DUAL_Q=true \
          MADDPG_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 maddpg
        ;;
      maddpg_separated_gradient)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MADDPG_USE_DUAL_Q=true \
          MADDPG_USE_SEPARATED_GRADIENT=true \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 maddpg
        ;;
      mappo_baseline)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=0 \
          MAPPO_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 mappo
        ;;
      mappo_fusion_only)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MAPPO_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 mappo
        ;;
      mappo_separated_gradient)
        env "${COMMON_ENV[@]}" "${resume_env[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MAPPO_USE_SEPARATED_GRADIENT=true \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 mappo
        ;;
      *)
        echo "Unknown label: $label" >&2
        exit 1
        ;;
    esac
  ) 2>&1 | sed -u "s/^/[${label} seed${seed}] /" | tee "$log_file" &

  echo "[launch] $label seed${seed} -> $log_file"
}

preflight_checks() {
  local errors=0
  local seed
  local label
  local path
  local model_dir
  local train_python="${TRAIN_PYTHON_BIN:-python3}"
  local eval_python="${EVAL_PYTHON_BIN:-${TRAIN_PYTHON_BIN:-python3}}"
  local utility_python="$UTILITY_PYTHON_BIN"
  local positions_file

  is_positive_int "$EPISODES" || { echo "[preflight-error] EPISODES must be a positive integer: $EPISODES" >&2; errors=$((errors + 1)); }
  is_positive_int "$BATCH_SIZE" || { echo "[preflight-error] BATCH_SIZE must be a positive integer: $BATCH_SIZE" >&2; errors=$((errors + 1)); }
  is_positive_int "$MAX_PARALLEL" || { echo "[preflight-error] MAX_PARALLEL must be a positive integer: $MAX_PARALLEL" >&2; errors=$((errors + 1)); }
  is_positive_int "$TRAIN_MAX_PARALLEL" || { echo "[preflight-error] TRAIN_MAX_PARALLEL must be a positive integer: $TRAIN_MAX_PARALLEL" >&2; errors=$((errors + 1)); }
  is_positive_int "$EVAL_MAX_PARALLEL" || { echo "[preflight-error] EVAL_MAX_PARALLEL must be a positive integer: $EVAL_MAX_PARALLEL" >&2; errors=$((errors + 1)); }

  [ -d "$ROOT_DIR" ] || { echo "[preflight-error] ROOT_DIR missing: $ROOT_DIR" >&2; errors=$((errors + 1)); }
  [ -d "$ROOT_DIR/models" ] || mkdir -p "$ROOT_DIR/models"
  [ -d "$ROOT_DIR/logs" ] || mkdir -p "$ROOT_DIR/logs"

  [ -x "$ROOT_DIR/run_optimized.sh" ] || { echo "[preflight-error] run_optimized.sh is missing or not executable" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/run_evaluation.sh" ] || { echo "[preflight-error] run_evaluation.sh missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/official_eval_with_matched_validation.py" ] || { echo "[preflight-error] official_eval_with_matched_validation.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/summarize_level2_official_eval.py" ] || { echo "[preflight-error] summarize_level2_official_eval.py missing" >&2; errors=$((errors + 1)); }
  [ -f "$ROOT_DIR/summarize_level2_official_eval_multiseed.py" ] || { echo "[preflight-error] summarize_level2_official_eval_multiseed.py missing" >&2; errors=$((errors + 1)); }

  for label in "${LABELS[@]}"; do
    label_supported "$label" || { echo "[preflight-error] unsupported label: $label" >&2; errors=$((errors + 1)); }
  done

  check_python_bin "$train_python" TRAIN_PYTHON_BIN || errors=$((errors + 1))
  check_python_bin "$eval_python" EVAL_PYTHON_BIN || errors=$((errors + 1))
  check_python_bin "$utility_python" UTILITY_PYTHON_BIN || errors=$((errors + 1))

  positions_file="$(common_env_value POSITIONS_FILE "")"
  [ -n "$positions_file" ] && [ -f "$positions_file" ] || {
    echo "[preflight-error] POSITIONS_FILE missing: ${positions_file:-<empty>}" >&2
    errors=$((errors + 1))
  }

  for seed in "${SEEDS[@]}"; do
    path="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
    [ -f "$path" ] || { echo "[preflight-error] official spec missing for seed${seed}: $path" >&2; errors=$((errors + 1)); }
  done

  for seed in "${SEEDS[@]}"; do
    for label in "${LABELS[@]}"; do
      model_dir="$(latest_completed_model_dir "$label" "$seed" || true)"
      if [ -n "$model_dir" ] && ! model_has_eval_candidate "$model_dir"; then
        echo "[preflight-error] completed model has no evaluable actor weights: $model_dir" >&2
        errors=$((errors + 1))
      fi
    done
  done

  if [ "$errors" -ne 0 ]; then
    echo "[preflight] failed with $errors error(s)." >&2
    return 1
  fi
  echo "[preflight] ok: configuration, specs, scripts, python bins, and completed model weights look usable."
}

preflight_checks
if truthy "$PREFLIGHT_ONLY"; then
  echo "[preflight] PREFLIGHT_ONLY=1, exiting before launching jobs."
  exit 0
fi

for seed in "${SEEDS[@]}"; do
  for label in "${LABELS[@]}"; do
    model_dir="$(latest_completed_model_dir "$label" "$seed" || true)"
    if [ -n "$model_dir" ] && model_completed "$model_dir" "$seed"; then
      if official_eval_completed "$model_dir" "$seed"; then
        if ! truthy "$FORCE_EVAL_RERUN"; then
          echo "[skip] $label seed${seed}: training and official eval already complete ($model_dir)"
          continue
        fi
        echo "[force-eval] $label seed${seed}: training and official eval complete, FORCE_EVAL_RERUN=1 -> rerun $model_dir"
      else
        echo "[resume-eval] $label seed${seed}: training complete, official eval missing/stale -> $model_dir"
      fi
      while [ "$(jobs -pr | wc -l)" -ge "$EVAL_MAX_PARALLEL" ]; do
        wait -n
      done
      run_official_eval_job "$label" "$seed" "$model_dir"
      continue
    fi

    model_dir="$(latest_model_dir "$label" "$seed")"
    if [ -n "$model_dir" ]; then
      echo "[restart-train] $label seed${seed}: incomplete previous run(s) will be deleted; restart at episode 0"
      delete_incomplete_artifacts_for_label_seed "$label" "$seed"
    else
      echo "[new-train] $label seed${seed}: no previous model directory found"
    fi

    while [ "$(jobs -pr | wc -l)" -ge "$TRAIN_MAX_PARALLEL" ]; do
      wait -n
    done
    run_level2_job "$label" "$seed"
  done
done

wait
echo "训练与官方测试完成。日志目录: $RUN_LOG_ROOT"

SUMMARY_FLAGS=()
if [ "$STRICT_SUMMARY" = "1" ]; then
  SUMMARY_FLAGS+=(--strict)
fi

for seed in "${SEEDS[@]}"; do
  MPLCONFIGDIR=/tmp/mplconfig "$UTILITY_PYTHON_BIN" /home/tang/matd3/summarize_level2_official_eval.py --seed "$seed" --labels "${LABELS[@]}" "${SUMMARY_FLAGS[@]}"
done

MPLCONFIGDIR=/tmp/mplconfig "$UTILITY_PYTHON_BIN" /home/tang/matd3/summarize_level2_official_eval_multiseed.py --seeds "${SEEDS[@]}" --labels "${LABELS[@]}" "${SUMMARY_FLAGS[@]}"

echo "单 seed 与多 seed 汇总完成。"

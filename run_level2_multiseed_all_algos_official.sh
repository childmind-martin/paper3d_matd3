#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEEDS=(${SEEDS:-101 202 936487})
STRICT_SUMMARY="${STRICT_SUMMARY:-1}"

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
  AUTO_EVAL=1
  AUTO_EVAL_MODE=official_spec
  OFFICIAL_POST_EVAL_MODEL_VARIANT=best_by_team_sr
  OFFICIAL_POST_EVAL_SELECTION_PROTOCOL=matched_validation
  OFFICIAL_POST_EVAL_VALIDATION_EPISODES=10
  OFFICIAL_POST_EVAL_VALIDATION_CANDIDATES=best_by_team_sr,best,checkpoint,final,latest_ep
  OFFICIAL_EVAL_QUIET=1
  FORCE_EVAL_ACTION_FORCE_RATIO=0.50
)

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

run_level2_job() {
  local label="$1"
  local seed="$2"
  local spec="${SPEC_ROOT}/batch_groupB_seed${seed}_20260331_220752/results/post_eval_shared_spec.json"
  local exp="${RUN_TAG}_${label}_seed${seed}"
  local log_file="${RUN_LOG_ROOT}/${label}_seed${seed}.log"

  (
    case "$label" in
      matd3_single_q)
        env "${COMMON_ENV[@]}" \
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
      matd3_dual_q)
        env "${COMMON_ENV[@]}" \
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
        env "${COMMON_ENV[@]}" \
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
        env "${COMMON_ENV[@]}" \
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
        env "${COMMON_ENV[@]}" \
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
        env "${COMMON_ENV[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MADDPG_USE_DUAL_Q=false \
          MADDPG_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 maddpg
        ;;
      maddpg_dual_q)
        env "${COMMON_ENV[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MADDPG_USE_DUAL_Q=true \
          MADDPG_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 maddpg
        ;;
      maddpg_separated_gradient)
        env "${COMMON_ENV[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MADDPG_USE_DUAL_Q=true \
          MADDPG_USE_SEPARATED_GRADIENT=true \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 maddpg
        ;;
      mappo_baseline)
        env "${COMMON_ENV[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=0 \
          MAPPO_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 mappo
        ;;
      mappo_fusion_only)
        env "${COMMON_ENV[@]}" \
          SEED="$seed" \
          OFFICIAL_POST_EVAL_SPEC="$spec" \
          USE_FR_FEATURE=1 \
          USE_PF_FEATURE=1 \
          MAPPO_USE_SEPARATED_GRADIENT=false \
          ./run_optimized.sh "$EPISODES" "$BATCH_SIZE" "$exp" 1 mappo
        ;;
      mappo_separated_gradient)
        env "${COMMON_ENV[@]}" \
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

for seed in "${SEEDS[@]}"; do
  for label in "${LABELS[@]}"; do
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
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
  MPLCONFIGDIR=/tmp/mplconfig python3 /home/tang/matd3/summarize_level2_official_eval.py --seed "$seed" "${SUMMARY_FLAGS[@]}"
done

MPLCONFIGDIR=/tmp/mplconfig python3 /home/tang/matd3/summarize_level2_official_eval_multiseed.py --seeds "${SEEDS[@]}" "${SUMMARY_FLAGS[@]}"

echo "单 seed 与多 seed 汇总完成。"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/tang/matd3}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models/matd3_cross_agent_ref_reward_to_success_head_tail_selector__seed101__batch_groupB_seed101_20260618_151436_20260618_151443}"
BASE_SPEC="${BASE_SPEC:-$ROOT_DIR/ablation_experiments/multi_seed_groupB_20260618_151436/seed_batches/batch_groupB_seed101_20260618_151436/results/post_eval_shared_spec.json}"

CANDIDATES="${CANDIDATES:-best_by_team_sr,best,ep100,ep200,ep300,ep400,ep500,ep600,ep700,ep800,ep900,ep1000,checkpoint,final,latest_ep}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-114817}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-10}"
TRAINSEQ_MODEL_VARIANT="${TRAINSEQ_MODEL_VARIANT:-best_by_team_sr}"
TRAINSEQ_EPISODES="${TRAINSEQ_EPISODES:-1000}"

RUN_TAG="${RUN_TAG:-verify_head_tail_ep_sweep_trainseq_$(date +%Y%m%d_%H%M%S)}"
MATCH_OUT="${MATCH_OUT:-$ROOT_DIR/logs/${RUN_TAG}_matchedval_ep100_ep1000}"
TRAINSEQ_OUT="${TRAINSEQ_OUT:-$ROOT_DIR/logs/${RUN_TAG}_trainseq_no_noise_${TRAINSEQ_MODEL_VARIANT}_${TRAINSEQ_EPISODES}ep}"
TRAINSEQ_SPEC="${TRAINSEQ_SPEC:-$TRAINSEQ_OUT/post_eval_trainseq_${TRAINSEQ_EPISODES}_spec.json}"

echo "[配置] MODEL_ROOT=$MODEL_ROOT"
echo "[配置] BASE_SPEC=$BASE_SPEC"
echo "[配置] CANDIDATES=$CANDIDATES"
echo "[配置] MATCH_OUT=$MATCH_OUT"
echo "[配置] TRAINSEQ_OUT=$TRAINSEQ_OUT"

if [[ "${RUN_MATCHED_VALIDATION:-1}" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/official_eval_with_matched_validation.py" \
    --experiment-root "$MODEL_ROOT" \
    --official-spec "$BASE_SPEC" \
    --output-dir "$MATCH_OUT" \
    --model-variant auto \
    --selection-protocol matched_validation \
    --validation-seeds $VALIDATION_SEEDS \
    --validation-episodes "$VALIDATION_EPISODES" \
    --validation-candidates "$CANDIDATES" \
    --selection-only \
    --python-bin "$PYTHON_BIN" \
    --force-rerun

  MATCH_OUT="$MATCH_OUT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["MATCH_OUT"] + "_matched_validation/selection_summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))
print(f"[matched-validation] summary={summary_path}")
print(f"[matched-validation] candidates={data.get('validation_candidates')}")
selected = data.get("selected", {})
print(
    "[matched-validation] selected="
    f"{selected.get('candidate_alias')} -> {selected.get('resolved_variant')} "
    f"score={selected.get('score')}"
)
for item in data.get("candidates", []):
    s = item.get("summary", {})
    print(
        f"  {item.get('candidate_alias'):>15s} -> {item.get('resolved_variant'):<8s} "
        f"team={s.get('success_episode_count')}/{s.get('episodes')} "
        f"sr={s.get('team_success_rate')} "
        f"agent={s.get('agent_success_rates')} "
        f"dist={s.get('avg_team_final_goal_distance')} "
        f"coll={s.get('avg_collision_count')}"
    )
PY
fi

if [[ "${RUN_TRAINSEQ_REPLAY:-1}" == "1" ]]; then
  ROOT_DIR="$ROOT_DIR" BASE_SPEC="$BASE_SPEC" TRAINSEQ_SPEC="$TRAINSEQ_SPEC" TRAINSEQ_EPISODES="$TRAINSEQ_EPISODES" "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
base_spec = Path(os.environ["BASE_SPEC"])
out_path = Path(os.environ["TRAINSEQ_SPEC"])
episodes = int(os.environ["TRAINSEQ_EPISODES"])

spec = json.loads(base_spec.read_text(encoding="utf-8"))
terrain_base_seed = 88
sequence_seed = 88
complexity = int(spec.get("terrain_complexity", 3))
map_size = int(round(float(spec.get("map_size", 200))))
hold_mode = str(spec.get("semi_random_hold_mode", "range")).strip().lower()
hold_episodes = max(1, int(spec.get("semi_random_hold_episodes", 18) or 18))
hold_min = max(1, int(spec.get("semi_random_hold_min_episodes", 15) or 15))
hold_max = max(hold_min, int(spec.get("semi_random_hold_max_episodes", 20) or 20))

def deterministic_seed(namespace: str, episode_idx: int, env_id: int = 0) -> int:
    payload = (
        f"{namespace}|seq={sequence_seed}|terrain={terrain_base_seed}|"
        f"episode={int(episode_idx)}|env={int(env_id)}|"
        f"complexity={complexity}|map={map_size}"
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % 2147483647
    return int(seed if seed > 0 else 1)

def hold_length(block_idx: int, env_id: int = 0) -> int:
    if hold_mode != "range":
        return hold_episodes
    span = hold_max - hold_min + 1
    if span <= 1:
        return hold_min
    payload = (
        f"semi_random_hold|seq={sequence_seed}|terrain={terrain_base_seed}|"
        f"block={int(block_idx)}|env={int(env_id)}|"
        f"complexity={complexity}|map={map_size}"
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int(hold_min + (int.from_bytes(digest, "little") % span))

def block_for_episode(episode_idx: int) -> int:
    if hold_mode == "fixed":
        return int(episode_idx) // hold_episodes
    if hold_mode != "range":
        return int(episode_idx)
    block_idx = 0
    block_start = 0
    while True:
        block_end = block_start + hold_length(block_idx)
        if episode_idx < block_end:
            return block_idx
        block_idx += 1
        block_start = block_end

position_file = root / "saved_positions/strict_ablation_seed88_groupB.json"
artifact_policy = dict(spec.get("artifact_policy", {}) or {})
artifact_policy.update({
    "light_mode": True,
    "save_interactive_html": False,
    "save_all_episodes": False,
    "save_best_reward_html": False,
    "save_team_success_html": False,
    "save_trajectory_json": False,
    "save_trajectory_png": False,
    "save_actor_sequence": False,
    "save_control_diagnostics": False,
    "enable_overlay": False,
    "disable_gif": True,
})

spec.update({
    "mode": "train_sequence_no_noise",
    "episodes": episodes,
    "episode_length_multiplier": 1.0,
    "seed": sequence_seed,
    "scenario_seed": terrain_base_seed,
    "terrain_seed": terrain_base_seed,
    "terrain_base_seed": terrain_base_seed,
    "terrain_family": "train_sequence",
    "position_family": "train_match",
    "random_terrain": True,
    "semi_random_terrain": True,
    "use_dynamic_obstacles": True,
    "use_fixed_positions": True,
    "agent_size": float(spec.get("agent_size") or 0.5),
    "reference_positions_file": str(position_file),
    "shared_positions_file": str(position_file),
    "default_positions_file": str(position_file),
    "episode_positions_dir": "",
    "terrain_seed_sequence": [terrain_base_seed for _ in range(episodes)],
    "terrain_variant_seed_sequence": [
        deterministic_seed("terrain_variant", block_for_episode(ep)) for ep in range(episodes)
    ],
    "obstacle_seed_sequence": [
        deterministic_seed("obstacle", ep) for ep in range(episodes)
    ],
    "artifact_policy": artifact_policy,
})

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[trainseq-spec] wrote {out_path}")
print(f"[trainseq-spec] obstacle_first3={spec['obstacle_seed_sequence'][:3]}")
print(f"[trainseq-spec] obstacle_last3={spec['obstacle_seed_sequence'][-3:]}")
print(f"[trainseq-spec] variant_first10={spec['terrain_variant_seed_sequence'][:10]}")
PY

  "$PYTHON_BIN" "$ROOT_DIR/official_eval_with_matched_validation.py" \
    --experiment-root "$MODEL_ROOT" \
    --official-spec "$TRAINSEQ_SPEC" \
    --output-dir "$TRAINSEQ_OUT" \
    --model-variant "$TRAINSEQ_MODEL_VARIANT" \
    --selection-protocol fixed \
    --python-bin "$PYTHON_BIN" \
    --force-rerun

  TRAINSEQ_OUT="$TRAINSEQ_OUT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

results_path = Path(os.environ["TRAINSEQ_OUT"]) / "evaluation_results.json"
data = json.loads(results_path.read_text(encoding="utf-8"))
summary = data.get("summary", {})
print(f"[trainseq-replay] results={results_path}")
for key in (
    "episodes",
    "success_episode_count",
    "team_success_rate",
    "agent_success_rates",
    "avg_reward",
    "avg_collision_count",
    "collision_free_rate",
    "avg_team_final_goal_distance",
    "avg_team_min_goal_distance",
):
    print(f"[trainseq-replay] {key}={summary.get(key)}")
PY
fi

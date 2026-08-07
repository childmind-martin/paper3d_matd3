#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/home/tang/miniconda3/envs/maddpg_env/bin/python}"
EVAL="${EVAL:-/home/tang/matd3/evaluate_optimized.py}"
EVAL_WRAPPER="${EVAL_WRAPPER:-/home/tang/matd3/run_evaluation.sh}"
BATCH_SPEC_JSON="${BATCH_SPEC_JSON:-}"
if [ -z "$BATCH_SPEC_JSON" ]; then
  echo "[STOP] 必须设置 BATCH_SPEC_JSON。先用 tools/build_selector_protocol_batch_spec.py 生成冻结批次规范。"
  exit 2
fi
BATCH_SPEC_JSON="$(realpath "$BATCH_SPEC_JSON")"
if [ ! -f "$BATCH_SPEC_JSON" ]; then
  echo "[STOP] BATCH_SPEC_JSON 不存在: $BATCH_SPEC_JSON"
  exit 2
fi

eval "$("$PY" - "$BATCH_SPEC_JSON" <<'PY'
import json
import shlex
import sys
from pathlib import Path

from tools.build_selector_protocol_batch_spec import validate_batch_spec

spec_path = Path(sys.argv[1]).resolve()
payload = json.loads(spec_path.read_text(encoding="utf-8"))
errors = validate_batch_spec(payload)
if errors:
    raise SystemExit("[STOP] 冻结批次规范无效: " + "; ".join(errors))

def export(name, value):
    print(f"export {name}={shlex.quote(str(value))}")

export("OUT_ROOT", payload["out_root"])
export("POS_FILE", payload["positions_file"])
export("SEQUENCE_SOURCE_JSON", payload["sequence_source_json"])
environment = payload["environment"]
export("ENV_SEQUENCE_SEED", environment["scenario_seed"])
export("EVAL_NOISE_SEED", payload["eval_noise_seed"])
export("PROTOCOL_VERSION", payload["protocol_version"])
export("EVAL_EPISODES", payload["episodes"])
export("REQUIRE_GPU", 1 if payload["require_gpu"] else 0)
export(
    "EVAL_EPISODE_LENGTH_MULTIPLIER",
    payload["episode_length_multiplier"],
)
export("EVAL_PROCESS_SHARDS", payload["eval_process_shards"])
export("EVAL_PROCESS_WORKERS", payload["eval_process_workers"])
export(
    "EVAL_SHARD_EPISODE_PARALLELISM",
    payload["eval_shard_episode_parallelism"],
)
export(
    "EVAL_SHARD_ENV_STEP_THREADS",
    payload["eval_shard_env_step_threads"],
)
export(
    "EVAL_EPISODE_PARALLELISM",
    int(payload["eval_process_shards"])
    * int(payload["eval_shard_episode_parallelism"]),
)
export(
    "EVAL_ENV_STEP_THREADS",
    int(payload["eval_process_shards"])
    * int(payload["eval_shard_env_step_threads"]),
)
export(
    "SEMI_RANDOM_TERRAIN",
    1 if environment["semi_random_terrain"] else 0,
)
export("TERRAIN_BASE_SEED", environment["terrain_base_seed"])
export("SCENARIO_SEED", environment["scenario_seed"])
export(
    "USE_DYNAMIC_OBSTACLES",
    1 if environment["use_dynamic_obstacles"] else 0,
)
export("POST_EVAL_MODE", environment["post_eval_mode"])
export(
    "POST_EVAL_TERRAIN_FAMILY",
    environment["post_eval_terrain_family"],
)
export(
    "POST_EVAL_POSITION_FAMILY",
    environment["post_eval_position_family"],
)
export("BATCH_SPEC_SHA256", payload["content_sha256"])
print(
    "MODELS=("
    + " ".join(
        shlex.quote(f"{item['id']}|{item['label']}|{item['model_path']}")
        for item in payload["models"]
    )
    + ")"
)
print(
    "MODES=("
    + " ".join(
        shlex.quote(
            f"{item['id']}|{item['eval_noise_scale']}|"
            f"{item['eval_random_action_prob']}"
        )
        for item in payload["modes"]
    )
    + ")"
)
PY
)"

export OUT_ROOT
export BATCH_SPEC_JSON
export BATCH_SPEC_SHA256
export SEQUENCE_SOURCE_JSON
export ENV_SEQUENCE_SEED
export EVAL_NOISE_SEED
export PROTOCOL_VERSION
export EVAL_EPISODES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-matd3}"

mkdir -p "$OUT_ROOT"
if [ -f "$OUT_ROOT/batch_spec.json" ] && ! cmp -s "$BATCH_SPEC_JSON" "$OUT_ROOT/batch_spec.json"; then
  echo "[STOP] OUT_ROOT 已包含不同的 batch_spec.json，不能混写协议。"
  exit 3
fi
cp "$BATCH_SPEC_JSON" "$OUT_ROOT/batch_spec.json"
BATCH_SPEC_JSON="$OUT_ROOT/batch_spec.json"
export BATCH_SPEC_JSON

if ps -eo pid,cmd | grep -F "$EVAL" | grep -F "$OUT_ROOT" | grep -v grep >/dev/null; then
  echo "[STOP] 检测到仍有 evaluate_optimized.py 正在写当前 OUT_ROOT："
  ps -eo pid,etimes,pcpu,pmem,cmd | grep -F "$EVAL" | grep -F "$OUT_ROOT" | grep -v grep
  echo "[STOP] 请先等待它结束，或手动停止后再运行本脚本，避免同一子项并发写入。"
  exit 3
fi

grep -q -- "--eval-process-shards" "$EVAL"
grep -q -- "--eval-noise-scale" "$EVAL"
grep -q -- "EVAL_NOISE_SCALE" "$EVAL_WRAPPER"

export EVAL_LIGHT_MODE=1
export SAVE_EVAL_TRAJECTORY_PNG=0
export SAVE_TEAM_SUCCESS_HTML=0
export SAVE_EVAL_ACTOR_SEQUENCE=0
export SAVE_EVAL_CONTROL_DIAGNOSTICS=0
export SAVE_GAZEBO_REPLAY=0
export SAVE_GAZEBO_DYNAMIC_REPLAY=0
export SAVE_TRAJECTORY_SNAPSHOT=0
export FAST_ARTIFACTS=1
export MATD3_REQUIRE_GPU="$REQUIRE_GPU"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export QUIET_OUTPUT=1

# 纠正版继续使用该实验定义的单文件固定位置模式；旧目录仅作为地形序列来源，
# 不再把旧的混合并行协议结果当作可复用缓存。
unset EPISODE_POSITIONS_DIR
unset EVAL_REQUIRE_EPISODE_POSITIONS

eval "$("$PY" - <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

source = Path(os.environ["BATCH_SPEC_JSON"])
data = json.loads(source.read_text(encoding="utf-8"))
episodes = int(data.get("episodes", 0) or 0)
sequences = data.get("sequences") or {}
if episodes != 30:
    raise SystemExit(f"[STOP] 正式协议必须是30个episode: {source}, got={episodes}")

def emit_sequence(env_name, field_name):
    values = sequences.get(field_name)
    if not isinstance(values, list) or len(values) != episodes:
        raise SystemExit(
            f"[STOP] {source} 的 sequences.{field_name} 长度错误"
        )
    encoded = ",".join(str(int(value)) for value in values)
    print(f"export {env_name}={shlex.quote(encoded)}")

emit_sequence("TERRAIN_COMPLEXITY_LEVEL_SEQUENCE", "terrain_complexity_level")
emit_sequence("TERRAIN_SEED_SEQUENCE", "terrain_seed")
emit_sequence("TERRAIN_VARIANT_SEED_SEQUENCE", "terrain_variant_seed")
emit_sequence("OBSTACLE_SEED_SEQUENCE", "obstacle_seed")
PY
)"

# 基础动力学、奖励与动作修正参数由 run_evaluation.sh 从每个模型自己的
# results.json 严格回读；这样 safety-reward 模型不会被公共硬编码覆盖。
export EVAL_PROCESS_SHARDS
export EVAL_PROCESS_WORKERS
export EVAL_SHARD_EPISODE_PARALLELISM
export EVAL_SHARD_ENV_STEP_THREADS
export EVAL_EPISODE_PARALLELISM
export EVAL_ENV_STEP_THREADS
export EVAL_EPISODE_LENGTH_MULTIPLIER
export EVAL_RESPECT_INPUT_POSITIONS=1
export EVAL_PYTHON_BIN="$PY"
export STRICT_EVAL_MATCH=1

cache_protocol() {
  local action="$1"
  local model_id="$2"
  local experiment_label="$3"
  local model="$4"
  local mode="$5"
  local noise="$6"
  local random_prob="$7"
  local out_dir="$8"
  "$PY" - "$action" "$model_id" "$experiment_label" "$model" "$mode" "$noise" "$random_prob" "$out_dir" "$POS_FILE" \
    "$EVAL_PROCESS_SHARDS" "$EVAL_PROCESS_WORKERS" "$EVAL_NOISE_SEED" "$PROTOCOL_VERSION" <<'PY'
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

from experiment_runtime_config import (
    find_training_runtime_manifest,
    load_training_runtime_manifest,
    runtime_environment_from_manifest,
)

(
    action, model_id, experiment_label, model_raw, mode, noise_raw, random_raw,
    out_raw, positions_raw,
    shards_raw, workers_raw, noise_seed_raw, protocol_raw,
) = sys.argv[1:]
model = Path(model_raw).resolve()
out_dir = Path(out_raw)
result_path = out_dir / "evaluation_results.json"
spec_path = out_dir / "run_spec.json"

def sequence(name):
    raw = os.environ.get(name, "").strip()
    return [int(item) for item in raw.split(",") if item.strip()]

def actor_signature(model_dir):
    actors = sorted(model_dir.glob("actor_*.weights.h5"))
    if not actors:
        raise SystemExit(f"[STOP] 模型缺少 actor 权重: {model_dir}")
    digest = hashlib.sha1()
    for actor in actors:
        digest.update(actor.name.encode("utf-8"))
        with actor.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()

def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"[STOP] 协议依赖文件不存在: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

batch_spec_path = Path(os.environ["BATCH_SPEC_JSON"]).resolve()
batch_spec = json.loads(batch_spec_path.read_text(encoding="utf-8"))
positions_path = Path(positions_raw).resolve()
sequence_source_path = Path(
    os.environ["SEQUENCE_SOURCE_JSON"]
).resolve()
if positions_path != Path(str(batch_spec["positions_file"])).resolve():
    raise SystemExit(
        "[STOP] 运行位置文件路径与冻结批次规范不一致: "
        f"got={positions_path}, expected={batch_spec['positions_file']}"
    )
if sequence_source_path != Path(
    str(batch_spec["sequence_source_json"])
).resolve():
    raise SystemExit(
        "[STOP] 地形序列来源路径与冻结批次规范不一致: "
        f"got={sequence_source_path}, expected={batch_spec['sequence_source_json']}"
    )
positions_signature = file_sha256(positions_path)
sequence_source_signature = file_sha256(sequence_source_path)
if positions_signature != batch_spec["positions_file_sha256"]:
    raise SystemExit("[STOP] 固定位置文件内容已偏离冻结批次规范")
if sequence_source_signature != batch_spec["sequence_source_sha256"]:
    raise SystemExit("[STOP] 地形序列来源文件内容已偏离冻结批次规范")

training_result = model.parent / "results.json"
if not training_result.exists():
    raise SystemExit(f"[STOP] 缺少模型训练配置: {training_result}")
training_data = json.loads(training_result.read_text(encoding="utf-8"))
training_args = training_data.get("args", {})
training_env = training_data.get("training_environment", {})
repo_root = Path.cwd().resolve()
protocol_sources = (
    repo_root / "run_noise_dependency_check_resume_fast.sh",
    repo_root / "run_evaluation.sh",
    repo_root / "evaluate_optimized.py",
    repo_root / "paper3d_train_optimized.py",
    repo_root / "multiagent/scenarios/paper3d_terrain_energy.py",
    repo_root / "multiagent/scenarios/obstacle_observation.py",
    repo_root / "experiment_runtime_config.py",
    repo_root / "cross_agent_reference_state.py",
    repo_root / "cross_agent_reference_selector.py",
    repo_root / "selector_experiment_protocol.py",
    repo_root / "tools/build_selector_protocol_batch_spec.py",
)

exp_name = model.parent.name
runtime_manifest_path = find_training_runtime_manifest(
    repo_root,
    exp_name,
    training_data.get("training_manifest_path"),
)
if runtime_manifest_path is None:
    raise SystemExit(f"[STOP] 无法唯一定位训练 resolved manifest: {exp_name}")
runtime_manifest = load_training_runtime_manifest(
    runtime_manifest_path,
    exp_name=exp_name,
    expected_content_sha256=training_data.get("training_manifest_sha256"),
)

def train_value(key, default=None):
    if isinstance(training_env, dict) and training_env.get(key) is not None:
        return training_env[key]
    if isinstance(training_args, dict) and training_args.get(key) is not None:
        return training_args[key]
    return default

runtime_setup = runtime_environment_from_manifest(runtime_manifest)

reward_fields = (
    "reward_pos_scale", "reward_neg_scale", "distance_weight", "exploration_weight",
    "stationary_weight", "direction_weight", "deviation_weight", "start_area_weight",
    "approach_weight", "energy_weight", "height_weight", "height_reward_enabled",
    "height_ideal_min", "height_ideal_max", "lateral_weight", "clearance_weight",
    "clearance_d_max", "success_weight", "collision_weight", "collision_reduction_weight",
    "global_weight", "shaping_weight", "max_reward", "min_reward",
    "success_reward_value", "no_collision_reward_value", "success_distance_threshold",
    "collision_penalty_value", "collision_distance_threshold", "global_reward_mode",
    "shaping_gamma", "reward_version", "reward_terminal_order_fix",
    "goal_ring_individual_scale", "goal_ring_team_gated", "goal_ring_require_agent_safe",
    "progress_distance_state_scale", "progress_reward_scale", "team_progress_bottleneck_only",
    "team_progress_non_bottleneck_scale", "team_progress_bottleneck_eps",
    "team_success_bonus", "unsafe_arrival_penalty", "non_success_terminal_guard_enabled",
    "non_success_terminal_penalty_base", "non_success_terminal_penalty_per_meter",
    "non_success_terminal_penalty_max", "terminal_failure_penalty_base",
    "terminal_failure_penalty_per_meter", "terminal_failure_penalty_max",
    "clearance_quality_bonus_weight", "efficiency_bonus_weight", "team_sync_reward_enabled",
    "team_goal_occupancy_scale", "team_bottleneck_progress_scale", "team_waiting_scale",
    "team_bottleneck_delta_clip", "clearance_dense_positive_scale",
    "height_dense_positive_scale",
)
reward_setup = {
    key: train_value(key)
    for key in reward_fields
    if train_value(key) is not None
}

episodes = int(os.environ["EVAL_EPISODES"])
training_episode_length = int(train_value("episode_length", 0) or 0)
episode_length_multiplier = float(
    os.environ["EVAL_EPISODE_LENGTH_MULTIPLIER"]
)
if training_episode_length != int(batch_spec["training_episode_length"]):
    raise SystemExit(
        "[STOP] 模型训练 episode_length 与冻结批次规范不一致: "
        f"got={training_episode_length}, "
        f"expected={batch_spec['training_episode_length']}"
    )
resolved_episode_length = int(
    training_episode_length * episode_length_multiplier + 0.5
)
if resolved_episode_length != int(batch_spec["episode_length"]):
    raise SystemExit(
        "[STOP] 评估 episode_length 与冻结批次规范不一致: "
        f"got={resolved_episode_length}, expected={batch_spec['episode_length']}"
    )
expected = {
    "protocol_version": int(protocol_raw),
    "model_id": model_id,
    "experiment_label": experiment_label,
    "mode": mode,
    "model_path": str(model),
    "model_signature": actor_signature(model),
    "training_results_path": str(training_result.resolve()),
    "training_results_signature": file_sha256(training_result),
    "training_runtime_manifest_path": str(runtime_manifest_path),
    "training_runtime_manifest_signature": file_sha256(runtime_manifest_path),
    "positions_file_signature": positions_signature,
    "sequence_source_signature": sequence_source_signature,
    "batch_spec_path": str(batch_spec_path),
    "batch_spec_signature": file_sha256(
        Path(os.environ["BATCH_SPEC_JSON"]).resolve()
    ),
    "batch_spec_content_sha256": os.environ["BATCH_SPEC_SHA256"],
    "protocol_source_signatures": {
        str(path.relative_to(repo_root)): file_sha256(path)
        for path in protocol_sources
    },
    "episodes": episodes,
    "require_gpu": bool(int(os.environ["REQUIRE_GPU"])),
    "episode_length": resolved_episode_length,
    "episode_length_multiplier": episode_length_multiplier,
    "positions_file": str(positions_path),
    "eval_noise_scale": float(noise_raw),
    "eval_random_action_prob": float(random_raw),
    "eval_noise_seed": int(noise_seed_raw),
    "eval_process_shards": int(shards_raw),
    "eval_process_workers": int(workers_raw),
    "terrain_complexity_level_sequence": sequence("TERRAIN_COMPLEXITY_LEVEL_SEQUENCE"),
    "terrain_seed_sequence": sequence("TERRAIN_SEED_SEQUENCE"),
    "terrain_variant_seed_sequence": sequence("TERRAIN_VARIANT_SEED_SEQUENCE"),
    "obstacle_seed_sequence": sequence("OBSTACLE_SEED_SEQUENCE"),
    "training_reward_setup": reward_setup,
    "training_runtime_setup": runtime_setup,
}
for key in (
    "terrain_complexity_level_sequence", "terrain_seed_sequence",
    "terrain_variant_seed_sequence", "obstacle_seed_sequence",
):
    if len(expected[key]) != episodes:
        raise SystemExit(f"[STOP] {key} 长度不是 {episodes}: {len(expected[key])}")

if action == "reset":
    root = Path(os.environ["OUT_ROOT"]).resolve()
    expected_out_dir = (root / model_id / mode).resolve()
    if out_dir.resolve() != expected_out_dir:
        raise SystemExit(
            "[STOP] 拒绝清理越界的评估 cell: "
            f"got={out_dir.resolve()}, expected={expected_out_dir}"
        )
    if out_dir.is_symlink():
        raise SystemExit(f"[STOP] 拒绝清理符号链接 cell: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[RESET] {model_id}/{mode} 不完整或协议失配，已清理该 cell 并从 episode 0 重跑")
    raise SystemExit(0)

if action == "check":
    print(
        f"[PREFLIGHT] {model_id}/{mode}: model_signature={expected['model_signature'][:12]} "
        f"episodes={episodes} shards={expected['eval_process_shards']} "
        f"noise={expected['eval_noise_scale']} random={expected['eval_random_action_prob']}"
    )
    raise SystemExit(0)

if action == "write":
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0)

errors = []
if not spec_path.exists() or not result_path.exists():
    errors.append("缺少 run_spec.json 或 evaluation_results.json")
else:
    try:
        recorded_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if recorded_spec != expected:
            errors.append("run_spec 与当前模型/协议不一致")
    except Exception as exc:
        errors.append(f"run_spec 无法读取: {exc}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result = {}
        errors.append(f"evaluation_results 无法读取: {exc}")

    details = result.get("episode_details", []) if isinstance(result, dict) else []
    try:
        episode_ids = sorted(int(item.get("episode")) for item in details)
    except Exception:
        episode_ids = []
    if int(result.get("episodes", 0) or 0) != episodes or episode_ids != list(range(episodes)):
        errors.append("episode 数量或编号不完整")
    try:
        if Path(str(result.get("model_path", ""))).resolve() != model:
            errors.append("model_path 不匹配")
    except Exception:
        errors.append("model_path 无法解析")

    setup = result.get("evaluation_setup", {}) if isinstance(result, dict) else {}
    exact_setup = {
        "eval_episode_parallelism_mode": "process_shards",
        "eval_backend": "python_only",
        "eval_process_shards": int(shards_raw),
        "eval_process_workers": int(workers_raw),
        "episode_length": expected["episode_length"],
        "use_fixed_positions": True,
        "positions_file": expected["positions_file"],
        "semi_random_terrain": True,
        "use_dynamic_obstacles": True,
        "training_runtime_manifest_path": expected["training_runtime_manifest_path"],
        "eval_noise_type": "gaussian",
        "eval_noise_stream_mode": "per_episode_seedsequence_pcg64_v1",
    }
    exact_setup.update(expected["training_runtime_setup"])
    for key, wanted in exact_setup.items():
        actual = setup.get(key)
        if key == "positions_file" and actual:
            actual = str(Path(str(actual)).resolve())
        if actual != wanted:
            errors.append(f"evaluation_setup.{key}={actual!r}, expected={wanted!r}")

    device = setup.get("eval_device")
    if expected["require_gpu"]:
        if not isinstance(device, dict):
            errors.append("evaluation_setup.eval_device 缺失")
        else:
            if device.get("require_gpu") is not True:
                errors.append(
                    "evaluation_setup.eval_device.require_gpu 不是 true"
                )
            try:
                physical_gpus = int(device.get("physical_gpus", 0) or 0)
                logical_gpus = int(device.get("logical_gpus", 0) or 0)
            except (TypeError, ValueError):
                physical_gpus = 0
                logical_gpus = 0
            if physical_gpus < 1 or logical_gpus < 1:
                errors.append(
                    "evaluation_setup.eval_device 未记录物理和逻辑 GPU"
                )

    float_setup = {
        "requested_episode_length_multiplier": episode_length_multiplier,
        "eval_noise_scale": float(noise_raw),
        "eval_random_action_prob": float(random_raw),
        "eval_noise_seed_base": float(noise_seed_raw),
    }
    for key, wanted in float_setup.items():
        try:
            actual = float(setup.get(key))
        except (TypeError, ValueError):
            errors.append(f"evaluation_setup.{key} 缺失")
            continue
        if not math.isfinite(actual) or abs(actual - wanted) > 1e-12:
            errors.append(f"evaluation_setup.{key}={actual}, expected={wanted}")

    shard_specs = setup.get("eval_process_shard_specs", [])
    if not isinstance(shard_specs, list) or len(shard_specs) != int(shards_raw):
        errors.append("eval_process_shard_specs 不完整")
    else:
        for shard in shard_specs:
            wanted_seed = int(noise_seed_raw)
            if shard.get("eval_noise_seed") != wanted_seed:
                errors.append(f"shard {shard.get('index')} 噪声种子不匹配")

    sequence_fields = (
        "terrain_complexity_level_sequence", "terrain_seed_sequence",
        "terrain_variant_seed_sequence", "obstacle_seed_sequence",
    )
    episode_field = {
        "terrain_complexity_level_sequence": "terrain_complexity_level",
        "terrain_seed_sequence": "terrain_seed",
        "terrain_variant_seed_sequence": "terrain_variant_seed",
        "obstacle_seed_sequence": "obstacle_seed",
    }
    for key in sequence_fields:
        if result.get(key) != expected[key]:
            errors.append(f"{key} 与协议不一致")
        actual_detail_sequence = [item.get(episode_field[key]) for item in details]
        if actual_detail_sequence != expected[key]:
            errors.append(f"episode_details.{episode_field[key]} 与协议不一致")

    for key, wanted in reward_setup.items():
        actual = setup.get(key)
        if isinstance(wanted, bool):
            if bool(actual) != wanted:
                errors.append(f"reward setup {key} 不匹配")
        elif isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            try:
                # 部分奖励系数进入向量化环境后以 float32 记录；按 IEEE-754
                # float32 的表示精度比较，而不是用比其 ULP 更小的 1e-9
                # 绝对阈值制造假不一致。该容差仍远小于任何实验参数改动。
                matches = math.isfinite(float(actual)) and math.isclose(
                    float(actual),
                    float(wanted),
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(f"reward setup {key}={actual!r}, expected={wanted!r}")
        elif str(actual) != str(wanted):
            errors.append(f"reward setup {key}={actual!r}, expected={wanted!r}")

if errors:
    print(f"[RERUN] {model_id}/{mode}: " + "; ".join(errors[:8]))
    raise SystemExit(1)
print(f"[SKIP] {model_id}/{mode} 已通过模型、环境、噪声、分片与完整性校验")
PY
}

preflight_eval_wrapper() {
  local label="$1"
  local model="$2"
  local resolved_output=""
  local preflight_save="/tmp/matd3_noise_eval_config_${label}_$$"
  if ! resolved_output=$(
    EVAL_CONFIG_RESOLVE_ONLY=1 \
    EVAL_NOISE_SCALE=0.0 \
    EVAL_RANDOM_ACTION_PROB=0.0 \
    EVAL_NOISE_SEED="$EVAL_NOISE_SEED" \
      bash "$EVAL_WRAPPER" "$model" "$EVAL_EPISODES" "$preflight_save" "$POS_FILE" 1 false 2>&1
  ); then
    echo "$resolved_output"
    echo "[STOP] $label 的 run_evaluation.sh 配置预检失败"
    return 1
  fi
  RESOLVED_WRAPPER_OUTPUT="$resolved_output" "$PY" - "$label" "$model" "$POS_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

from experiment_runtime_config import (
    find_training_runtime_manifest,
    load_training_runtime_manifest,
    runtime_environment_from_manifest,
)

label, model_raw, positions_raw = sys.argv[1:]
marker = "RESOLVED_EVAL_CONFIG_JSON="
resolved_line = None
for line in os.environ.get("RESOLVED_WRAPPER_OUTPUT", "").splitlines():
    if line.startswith(marker):
        resolved_line = line[len(marker):].strip()
if not resolved_line:
    raise SystemExit(f"[STOP] {label} 的包装器未输出解析JSON")
payload = json.loads(resolved_line)
args = payload.get("args", {})
model = Path(model_raw).resolve()
repo_root = Path.cwd().resolve()
training_data = json.loads((model.parent / "results.json").read_text(encoding="utf-8"))
manifest_path = find_training_runtime_manifest(
    repo_root,
    model.parent.name,
    training_data.get("training_manifest_path"),
)
manifest = load_training_runtime_manifest(
    manifest_path,
    exp_name=model.parent.name,
    expected_content_sha256=training_data.get("training_manifest_sha256"),
)
expected_runtime = runtime_environment_from_manifest(manifest)
errors = []
for key, expected in expected_runtime.items():
    if args.get(key) != expected:
        errors.append(f"{key}={args.get(key)!r}, expected={expected!r}")
if Path(str(args.get("positions_file", ""))).resolve() != Path(positions_raw).resolve():
    errors.append("positions_file 未保留噪声实验指定的公共位置")
if args.get("use_fixed_positions") is not True:
    errors.append("use_fixed_positions 不是 true")
training_episode_length = int(
    (training_data.get("args") or {}).get("episode_length", 0) or 0
)
episode_length_multiplier = float(
    os.environ["EVAL_EPISODE_LENGTH_MULTIPLIER"]
)
expected_episode_length = int(
    training_episode_length * episode_length_multiplier + 0.5
)
if int(args.get("episode_length", 0) or 0) != expected_episode_length:
    errors.append(
        f"episode_length={args.get('episode_length')!r}, "
        f"expected={expected_episode_length}"
    )
expected_shards = int(os.environ["EVAL_PROCESS_SHARDS"])
if int(args.get("eval_process_shards", 0) or 0) != expected_shards:
    errors.append(
        f"eval_process_shards={args.get('eval_process_shards')!r}, "
        f"expected={expected_shards}"
    )
if Path(str(payload.get("training_runtime_manifest_path", ""))).resolve() != manifest_path:
    errors.append("Python评估器采用的 training_runtime_manifest_path 不一致")
if errors:
    raise SystemExit(f"[STOP] {label} 包装器配置不一致: " + "; ".join(errors))
print(f"[PREFLIGHT-WRAPPER] {label}: runtime/positions/episode_length/shards 均已对齐")
PY
}

run_eval() {
  local model_id="$1"
  local experiment_label="$2"
  local model="$3"
  local mode="$4"
  local noise="$5"
  local random_prob="$6"
  local out_dir="$OUT_ROOT/$model_id/$mode"

  case "${PREFLIGHT_ONLY:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      cache_protocol check "$model_id" "$experiment_label" "$model" "$mode" "$noise" "$random_prob" "$out_dir"
      return 0
      ;;
  esac

  if cache_protocol validate "$model_id" "$experiment_label" "$model" "$mode" "$noise" "$random_prob" "$out_dir"; then
    return 0
  fi

  cache_protocol reset "$model_id" "$experiment_label" "$model" "$mode" "$noise" "$random_prob" "$out_dir"
  cache_protocol write "$model_id" "$experiment_label" "$model" "$mode" "$noise" "$random_prob" "$out_dir"
  echo
  echo "========== RUN $model_id ($experiment_label) / $mode =========="
  echo "out=$out_dir"
  EVAL_NOISE_SCALE="$noise" \
  EVAL_RANDOM_ACTION_PROB="$random_prob" \
  EVAL_NOISE_SEED="$EVAL_NOISE_SEED" \
    bash "$EVAL_WRAPPER" "$model" "$EVAL_EPISODES" "$out_dir" "$POS_FILE" 1 false
  cache_protocol validate "$model_id" "$experiment_label" "$model" "$mode" "$noise" "$random_prob" "$out_dir"
}

case "${PREFLIGHT_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    for model_entry in "${MODELS[@]}"; do
      IFS='|' read -r model_id experiment_label model_path <<< "$model_entry"
      preflight_eval_wrapper "$model_id" "$model_path"
    done
    ;;
esac

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r model_id experiment_label model_path <<< "$model_entry"
  for mode_entry in "${MODES[@]}"; do
    IFS='|' read -r mode noise random_prob <<< "$mode_entry"
    run_eval "$model_id" "$experiment_label" "$model_path" "$mode" "$noise" "$random_prob"
  done
done

case "${PREFLIGHT_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    subitem_count=$((${#MODELS[@]} * ${#MODES[@]}))
    echo "[PREFLIGHT] ${#MODELS[@]}×${#MODES[@]}=${subitem_count} 个子项的模型、训练配置和共享环境序列均可解析；未启动评估。"
    exit 0
    ;;
esac

"$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
batch_spec = json.loads(
    Path(os.environ["BATCH_SPEC_JSON"]).read_text(encoding="utf-8")
)
labels = [item["id"] for item in batch_spec["models"]]
modes = [item["id"] for item in batch_spec["modes"]]
print("\n========== SUMMARY ==========")
missing = []
for label in labels:
    for mode in modes:
        path = root / label / mode / "evaluation_results.json"
        if not path.exists():
            missing.append(f"{label}/{mode}")
            print(f"[MISSING] {label}/{mode}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        setup = data.get("evaluation_setup", {})
        print(
            f"[DONE] {label}/{mode} "
            f"episodes={data.get('episodes')} "
            f"sr={summary.get('team_success_rate')} "
            f"coll={summary.get('avg_collision_count')} "
            f"mode={setup.get('eval_episode_parallelism_mode')}"
        )
if missing:
    raise SystemExit(f"[STOP] 仍有未完成子项: {missing}")
print(f"DONE: {root}")
PY

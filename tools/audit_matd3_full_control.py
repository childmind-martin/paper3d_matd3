#!/usr/bin/env python3
"""Preflight and validate the seed-101 MATD3 full control experiment.

The control is only scientifically useful when it is the same training stack
as M0 with cross-agent reference learning disabled.  This tool proves that
contract before launch and re-checks the completed training unit afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ablation_dual_q_separated_gradient as ablation
from experiment_runtime_config import training_unit_completion_errors
from selection_scoring import (
    comparison_score,
    score_summary,
    selection_summary_errors,
)


FULL_LABEL = "matd3_full_dual_semantic"
M0_LABEL = "matd3_cross_agent_ref_behavior_label_agent_quality_gate"
TRAIN_SEED = 101
TRAIN_EPISODES = 1000
TRAIN_NUM_ENVS = 4
TRAIN_BATCH_SIZE = 1024
SCENARIO_SEED = 88
FR_SCHEDULE = (
    "0%:0.50,25%:0.48,50%:0.45,70%:0.40,85%:0.35,100%:0.32"
)
VALIDATION_SEED = 114817
EVAL_NOISE_SEED = 101
EVAL_EPISODES = 30
EVAL_CANDIDATES = ("best_by_team_sr", "best", "final")
ZERO_DIAGNOSTIC_KEYS = (
    "cross_ref_active",
    "cross_ref_loss",
    "cross_ref_valid_ratio",
    "cross_ref_selector_loss",
    "cross_ref_selector_gradient_norm",
    "cross_ref_selector_update_count",
)
ADDITIONAL_TRAINING_SOURCES = (
    "ablation_dual_q_separated_gradient.py",
    "run_optimized.sh",
    "experiment_runtime_config.py",
    "selection_scoring.py",
)


class ControlAuditError(RuntimeError):
    pass


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ControlAuditError(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ControlAuditError(f"JSON 顶层不是对象: {path}")
    return payload


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ControlAuditError(f"文件不存在: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_exact_args(labels: Sequence[str]):
    original_argv = list(sys.argv)
    try:
        sys.argv = [
            "audit_matd3_full_control.py",
            "--episodes",
            str(TRAIN_EPISODES),
            "--batch-size",
            str(TRAIN_BATCH_SIZE),
            "--num-envs",
            str(TRAIN_NUM_ENVS),
            "--batch-seed",
            str(TRAIN_SEED),
            "--experiment-group",
            "B",
            "--config-mode",
            "strict_ablation",
            "--env-isolation",
            "strict",
            "--scenario-seed",
            str(SCENARIO_SEED),
            "--use-weighted-reward",
            "1",
            "--action-force-ratio",
            "0.50",
            "--action-force-ratio-schedule-pct",
            FR_SCHEDULE,
            "--disable-post-eval",
            "--skip-local-plots",
            "--experiments",
            *labels,
        ]
        args = ablation.parse_args()
    finally:
        sys.argv = original_argv
    args.parsed_seeds = [TRAIN_SEED]
    args.resolved_scenario_seed = ablation._resolve_scenario_seed(
        args.config_mode,
        args.scenario_seed,
    )
    (
        args.resolved_unlock_env_on_success,
        args.resolved_unlock_env_on_plateau,
    ) = ablation._resolve_unlock_thresholds(
        config_mode=args.config_mode,
        unlock_env_on_success=args.unlock_env_on_success,
        unlock_env_on_plateau=args.unlock_env_on_plateau,
    )
    ablation._apply_experiment_group_overrides(args)
    return args


def _resolve_preview_manifests() -> Dict[str, Dict[str, Any]]:
    labels = (FULL_LABEL, M0_LABEL)
    args = _build_exact_args(labels)
    configs = ablation._select_experiment_configs(args)
    if [str(item["label"]) for item in configs] != list(labels):
        raise ControlAuditError("full/M0 配置解析顺序不符合请求")
    old_prompt = os.environ.get("SUPPRESS_MA_PROMPT")
    os.environ["SUPPRESS_MA_PROMPT"] = "1"
    try:
        with tempfile.TemporaryDirectory(
            prefix="matd3_full_control_manifest_"
        ) as temporary:
            root = Path(temporary)
            manifests = {
                str(config["label"]): ablation._resolve_experiment_manifest(
                    config,
                    root / "positions.json",
                    args,
                    root / "manifests",
                )[0]
                for config in configs
            }
    finally:
        if old_prompt is None:
            os.environ.pop("SUPPRESS_MA_PROMPT", None)
        else:
            os.environ["SUPPRESS_MA_PROMPT"] = old_prompt
    return manifests


def _manifest_parity(
    full_manifest: Mapping[str, Any],
    m0_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    diff = ablation._build_manifest_diff(
        dict(full_manifest),
        dict(m0_manifest),
    )
    if not bool(diff.get("compatible")):
        unexpected = {
            key: diff.get(key)
            for key in (
                "unexpected_argv_only_in_ref",
                "unexpected_argv_only_in_cur",
                "unexpected_argv_changed",
                "unexpected_env_only_in_ref",
                "unexpected_env_only_in_cur",
                "unexpected_env_changed",
            )
            if diff.get(key)
        }
        raise ControlAuditError(
            "MATD3 full 与 M0 存在 cross-reference 白名单之外的配置差异: "
            + json.dumps(unexpected, ensure_ascii=False, sort_keys=True)
        )

    full_env = dict(full_manifest.get("audit_env", {}) or {})
    m0_env = dict(m0_manifest.get("audit_env", {}) or {})
    required_common = {
        "ALGORITHM": "matd3",
        "MATD3_USE_DUAL_Q": "1",
        "MATD3_USE_SEPARATED_GRADIENT": "1",
        "MATD3_USE_HYBRID_ACTOR_OBJECTIVE": "0",
        "MATD3_ACTION_SEMANTICS_MODE": "dual",
        "MATD3_RECONSTRUCT_CORRECTED_TARGET": "1",
        "MATD3_REQUIRE_GPU": "1",
        "USE_TF_POTENTIAL_FIELD": "1",
        "USE_FR_FEATURE": "1",
        "USE_PF_FEATURE": "1",
        "SELECTOR_PROTOCOL_LOCK": "1",
    }
    for key, expected in required_common.items():
        for label, env in ((FULL_LABEL, full_env), (M0_LABEL, m0_env)):
            actual = str(env.get(key, ""))
            if actual != expected:
                raise ControlAuditError(
                    f"{label} {key}={actual!r}, expected={expected!r}"
                )
    if _as_bool(full_env.get("CROSS_AGENT_REFERENCE_ENABLED")):
        raise ControlAuditError("MATD3 full 未关闭 CROSS_AGENT_REFERENCE_ENABLED")
    if not _as_bool(m0_env.get("CROSS_AGENT_REFERENCE_ENABLED")):
        raise ControlAuditError("M0 未开启 CROSS_AGENT_REFERENCE_ENABLED")

    return {
        "compatible": True,
        "reference_label": FULL_LABEL,
        "current_label": M0_LABEL,
        "scientific_difference_scope": "cross_agent_reference_only",
        "argv_changed": diff.get("argv_changed", {}),
        "argv_only_in_ref": diff.get("argv_only_in_ref", []),
        "argv_only_in_cur": diff.get("argv_only_in_cur", []),
        "env_changed": diff.get("env_changed", {}),
        "env_only_in_ref": diff.get("env_only_in_ref", []),
        "env_only_in_cur": diff.get("env_only_in_cur", []),
    }


def _source_snapshot(reference_run_spec: Path) -> Dict[str, Any]:
    reference = _load_json(reference_run_spec)
    signatures = reference.get("protocol_source_signatures")
    if not isinstance(signatures, dict) or not signatures:
        raise ControlAuditError(
            f"参考 run_spec 缺少 protocol_source_signatures: "
            f"{reference_run_spec}"
        )
    verified: Dict[str, str] = {}
    for relative, recorded in sorted(signatures.items()):
        source = (REPO_ROOT / str(relative)).resolve()
        actual = _sha256(source)
        if actual != str(recorded):
            raise ControlAuditError(
                f"训练前源码已偏离 M0 正式批次: {relative}: "
                f"got={actual}, expected={recorded}"
            )
        verified[str(relative)] = actual
    additional = {
        relative: _sha256(REPO_ROOT / relative)
        for relative in ADDITIONAL_TRAINING_SOURCES
    }
    return {
        "reference_run_spec": str(reference_run_spec.resolve()),
        "reference_run_spec_sha256": _sha256(reference_run_spec),
        "protocol_source_signatures": verified,
        "additional_training_source_signatures": additional,
    }


def _verify_snapshot_sources(snapshot: Mapping[str, Any]) -> None:
    for section in (
        "protocol_source_signatures",
        "additional_training_source_signatures",
    ):
        signatures = snapshot.get(section)
        if not isinstance(signatures, dict) or not signatures:
            raise ControlAuditError(f"source snapshot 缺少 {section}")
        for relative, expected in sorted(signatures.items()):
            actual = _sha256(REPO_ROOT / str(relative))
            if actual != str(expected):
                raise ControlAuditError(
                    f"控制组训练期间源码发生漂移: {relative}: "
                    f"got={actual}, expected={expected}"
                )


def run_preflight(
    *,
    reference_run_spec: Path,
    output: Path,
) -> Dict[str, Any]:
    source = _source_snapshot(reference_run_spec)
    manifests = _resolve_preview_manifests()
    parity = _manifest_parity(
        manifests[FULL_LABEL],
        manifests[M0_LABEL],
    )
    payload = {
        "schema_version": 1,
        "status": "pass",
        "phase": "preflight",
        "train_seed": TRAIN_SEED,
        "train_episodes": TRAIN_EPISODES,
        "train_num_envs": TRAIN_NUM_ENVS,
        "batch_size": TRAIN_BATCH_SIZE,
        "scenario_seed": SCENARIO_SEED,
        "action_force_ratio_schedule_pct": FR_SCHEDULE,
        "source_snapshot": source,
        "manifest_parity": parity,
    }
    if output.exists():
        previous = _load_json(output)
        previous_source = previous.get("source_snapshot")
        if previous_source != source:
            raise ControlAuditError(
                f"已有 source snapshot 与当前请求不同，拒绝覆盖: {output}"
            )
    _write_json_atomic(output, payload)
    return payload


def _discover_seed_batch(parent_batch_dir: Path) -> Path:
    candidates = sorted(
        path.resolve()
        for path in (parent_batch_dir / "seed_batches").glob(
            f"batch_groupB_seed{TRAIN_SEED}_*"
        )
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise ControlAuditError(
            f"无法唯一定位 seed 子批次: parent={parent_batch_dir}, "
            f"candidates={[str(path) for path in candidates]}"
        )
    return candidates[0]


def _loss_history_path(results: Mapping[str, Any]) -> Path:
    args = results.get("args")
    if not isinstance(args, dict):
        raise ControlAuditError("results.json 缺少 args")
    run_dir = str(args.get("_run_dir", "") or "").strip()
    if not run_dir:
        raise ControlAuditError("results.args 缺少 _run_dir")
    path = Path(run_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    loss_path = path.resolve() / "loss_history.json"
    if not loss_path.is_file():
        raise ControlAuditError(f"缺少 loss_history.json: {loss_path}")
    return loss_path


def _validate_disabled_diagnostics(loss_path: Path) -> Dict[str, Any]:
    try:
        rows = json.loads(loss_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ControlAuditError(
            f"无法读取 loss history: {loss_path}: {exc}"
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise ControlAuditError(f"loss history 为空或格式错误: {loss_path}")
    counts = {key: 0 for key in ZERO_DIAGNOSTIC_KEYS}
    max_abs = {key: 0.0 for key in ZERO_DIAGNOSTIC_KEYS}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ControlAuditError(
                f"loss history 第 {row_index} 项不是对象"
            )
        for key in ZERO_DIAGNOSTIC_KEYS:
            if key not in row or row.get(key) is None:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ControlAuditError(
                    f"{loss_path} {key} 含非法值: {row[key]!r}"
                ) from exc
            if not math.isfinite(value):
                raise ControlAuditError(
                    f"{loss_path} {key} 含非有限值: {value!r}"
                )
            counts[key] += 1
            max_abs[key] = max(max_abs[key], abs(value))
    for key in ZERO_DIAGNOSTIC_KEYS:
        if counts[key] <= 0:
            raise ControlAuditError(f"loss history 未记录 {key}")
        if max_abs[key] > 1e-8:
            raise ControlAuditError(
                f"关闭 cross-reference 后 {key} 非零: "
                f"max_abs={max_abs[key]}"
            )
    return {
        "loss_history_path": str(loss_path),
        "rows": len(rows),
        "observed_counts": counts,
        "max_abs": max_abs,
    }


def run_completed(
    *,
    parent_batch_dir: Path,
    m0_manifest_path: Path,
    source_snapshot_path: Path,
    output: Path,
    model_root_file: Path | None,
) -> Dict[str, Any]:
    snapshot_payload = _load_json(source_snapshot_path)
    source_snapshot = snapshot_payload.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ControlAuditError(
            f"source snapshot 文件格式错误: {source_snapshot_path}"
        )
    _verify_snapshot_sources(source_snapshot)

    seed_batch = _discover_seed_batch(parent_batch_dir.resolve())
    manifest_path = (
        seed_batch
        / "manifests"
        / f"{FULL_LABEL}_resolved_manifest.json"
    )
    full_manifest = ablation._load_manifest(manifest_path)
    m0_manifest = ablation._load_manifest(m0_manifest_path.resolve())
    parity = _manifest_parity(full_manifest, m0_manifest)

    meta = (
        full_manifest.get("meta", {})
        if isinstance(full_manifest.get("meta"), dict)
        else {}
    )
    experiment_name = str(meta.get("exp_name_with_timestamp", "") or "")
    if not experiment_name:
        raise ControlAuditError(f"manifest 缺少 exp_name: {manifest_path}")
    model_root = (REPO_ROOT / "models" / experiment_name).resolve()
    positions_file = Path(
        str(
            (full_manifest.get("audit_env") or {}).get(
                "POSITIONS_FILE",
                "",
            )
        )
    ).resolve()
    expected_agents = ablation._expected_agent_count_from_positions_file(
        positions_file
    )
    completion_errors = training_unit_completion_errors(
        model_root,
        TRAIN_EPISODES,
        repo_root=REPO_ROOT,
        expected_agents=expected_agents,
        expected_seed=TRAIN_SEED,
        expected_num_envs=TRAIN_NUM_ENVS,
        require_gpu=True,
    )
    if completion_errors:
        raise ControlAuditError(
            "MATD3 full 完整训练单元校验失败: "
            + " | ".join(completion_errors)
        )

    results_path = model_root / "results.json"
    results = _load_json(results_path)
    args = results.get("args")
    if not isinstance(args, dict):
        raise ControlAuditError(f"results.args 缺失: {results_path}")
    required_args = {
        "algo": "matd3",
        "seed": TRAIN_SEED,
        "train_episodes": TRAIN_EPISODES,
        "num_envs": TRAIN_NUM_ENVS,
        "matd3_use_dual_q": True,
        "matd3_use_separated_gradient": True,
        "matd3_use_hybrid_actor_objective": False,
        "matd3_action_semantics_mode": "dual",
        "matd3_reconstruct_corrected_target": True,
        "use_tf_potential_field": True,
        "use_fr_feature": True,
        "use_pf_feature": True,
        "cross_agent_reference_enabled": False,
        "cross_agent_reference_selector_enabled": False,
    }
    for key, expected in required_args.items():
        actual = args.get(key)
        if isinstance(expected, bool):
            matches = _as_bool(actual) is expected
        elif isinstance(expected, int):
            try:
                matches = int(actual) == expected
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(actual or "").strip().lower() == str(
                expected
            ).strip().lower()
        if not matches:
            raise ControlAuditError(
                f"results.args.{key}={actual!r}, expected={expected!r}"
            )

    forbidden_selector_files = sorted(
        path.name
        for path in (model_root / "final").glob(
            "*reference_selector*.weights.h5"
        )
    )
    if forbidden_selector_files:
        raise ControlAuditError(
            "关闭 selector 的控制组仍保存了 selector 权重: "
            + ", ".join(forbidden_selector_files)
        )
    diagnostics = _validate_disabled_diagnostics(
        _loss_history_path(results)
    )

    payload = {
        "schema_version": 1,
        "status": "pass",
        "phase": "completed_training",
        "parent_batch_dir": str(parent_batch_dir.resolve()),
        "seed_batch_dir": str(seed_batch),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "model_root": str(model_root),
        "results_path": str(results_path),
        "results_sha256": _sha256(results_path),
        "training_device": results.get("training_device"),
        "training_parallelism": results.get("training_parallelism"),
        "manifest_parity": parity,
        "disabled_cross_reference_diagnostics": diagnostics,
        "source_snapshot_path": str(source_snapshot_path.resolve()),
        "source_snapshot_sha256": _sha256(source_snapshot_path),
    }
    _write_json_atomic(output, payload)
    if model_root_file is not None:
        model_root_file.parent.mkdir(parents=True, exist_ok=True)
        model_root_file.write_text(str(model_root) + "\n", encoding="utf-8")
    return payload


def _validate_evaluation_payload(
    path: Path,
    *,
    expected_model_path: Path,
) -> Dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ControlAuditError(f"评估缺少 summary: {path}")
    errors = selection_summary_errors(summary)
    if errors:
        raise ControlAuditError(
            f"评估 summary 不完整: {path}: {'; '.join(errors)}"
        )
    if int(summary.get("episodes", 0) or 0) != EVAL_EPISODES:
        raise ControlAuditError(
            f"评估回合数错误: {path}: "
            f"{summary.get('episodes')} != {EVAL_EPISODES}"
        )
    details = payload.get("episode_details")
    if not isinstance(details, list) or len(details) != EVAL_EPISODES:
        raise ControlAuditError(
            f"episode_details 数量错误: {path}: "
            f"{len(details) if isinstance(details, list) else 'invalid'}"
        )
    episode_ids = [int(item.get("episode", -1)) for item in details]
    if episode_ids != list(range(EVAL_EPISODES)):
        raise ControlAuditError(f"episode 编号不连续: {path}")
    actual_model_path = Path(str(payload.get("model_path", ""))).resolve()
    if actual_model_path != expected_model_path.resolve():
        raise ControlAuditError(
            f"评估模型路径错误: {path}: "
            f"{actual_model_path} != {expected_model_path}"
        )
    setup = payload.get("evaluation_setup")
    if not isinstance(setup, dict):
        raise ControlAuditError(f"评估缺少 evaluation_setup: {path}")
    if int(setup.get("eval_noise_seed", -1) or -1) != EVAL_NOISE_SEED:
        raise ControlAuditError(
            f"eval_noise_seed 错误: {path}: "
            f"{setup.get('eval_noise_seed')} != {EVAL_NOISE_SEED}"
        )
    if float(setup.get("eval_noise_scale", 0.0) or 0.0) != 0.0:
        raise ControlAuditError(f"控制组 checkpoint 评估含高斯噪声: {path}")
    if float(setup.get("eval_random_action_prob", 0.0) or 0.0) != 0.0:
        raise ControlAuditError(f"控制组 checkpoint 评估含随机动作: {path}")
    fr_source = str(setup.get("action_force_ratio_source", "") or "")
    if fr_source in ("", "forced_override"):
        raise ControlAuditError(f"checkpoint FR 来源非法: {path}: {fr_source!r}")
    return payload


def _write_eval_report(path: Path, payload: Mapping[str, Any]) -> None:
    selected = payload["selected"]
    summary = payload["official_summary"]
    candidates = payload["validation_candidates"]
    lines = [
        "# MATD3 Full Control Checkpoint Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- selected checkpoint: `{selected['alias']}`",
        f"- selected model: `{selected['model_path']}`",
        f"- validation seed / episodes: `{VALIDATION_SEED}` / `{EVAL_EPISODES}`",
        f"- official eval noise seed / episodes: `{EVAL_NOISE_SEED}` / `{EVAL_EPISODES}`",
        "",
        "## Held-out candidates",
        "",
        "| checkpoint | Team SR | mean agent SR | collision-free | avg collisions | final team distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        candidate_summary = item["summary"]
        agent_rates = candidate_summary.get("agent_success_rates") or []
        mean_agent_rate = (
            sum(float(value) for value in agent_rates) / len(agent_rates)
            if agent_rates
            else 0.0
        )
        lines.append(
            "| {alias} | {team:.1%} | {agent:.1%} | {cf:.1%} | "
            "{coll:.3f} | {dist:.3f} |".format(
                alias=item["alias"],
                team=float(candidate_summary["team_success_rate"]),
                agent=mean_agent_rate,
                cf=float(candidate_summary["collision_free_rate"]),
                coll=float(candidate_summary["avg_collision_count"]),
                dist=float(candidate_summary["avg_team_final_goal_distance"]),
            )
        )
    official_agents = summary.get("agent_success_rates") or []
    official_agent_mean = (
        sum(float(value) for value in official_agents)
        / len(official_agents)
        if official_agents
        else 0.0
    )
    lines.extend(
        [
            "",
            "## Official deterministic result",
            "",
            f"- Team SR: `{float(summary['team_success_rate']):.1%}`",
            f"- Mean agent SR: `{official_agent_mean:.1%}`",
            f"- Collision-free rate: `{float(summary['collision_free_rate']):.1%}`",
            f"- Average collisions: `{float(summary['avg_collision_count']):.3f}`",
            f"- Average team final goal distance: `{float(summary['avg_team_final_goal_distance']):.3f}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation_audit(
    *,
    training_audit_path: Path,
    eval_root: Path,
    output: Path,
    report: Path | None,
) -> Dict[str, Any]:
    training_audit = _load_json(training_audit_path)
    if training_audit.get("status") != "pass":
        raise ControlAuditError(
            f"训练审计未通过: {training_audit_path}"
        )
    model_root = Path(str(training_audit.get("model_root", ""))).resolve()
    selection_path = (
        eval_root
        / "official_deterministic_matched_validation"
        / "selection_summary.json"
    )
    selection = _load_json(selection_path)
    if selection.get("selection_protocol") != "matched_validation":
        raise ControlAuditError("控制组选模协议不是 matched_validation")
    if int(selection.get("validation_episodes", 0) or 0) != EVAL_EPISODES:
        raise ControlAuditError("控制组选模 validation_episodes 错误")
    if selection.get("validation_seeds") != [VALIDATION_SEED]:
        raise ControlAuditError("控制组选模 validation seed 错误")
    if tuple(selection.get("validation_candidates", [])) != EVAL_CANDIDATES:
        raise ControlAuditError(
            "控制组选模候选集错误: "
            f"{selection.get('validation_candidates')!r}"
        )
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(
        EVAL_CANDIDATES
    ):
        raise ControlAuditError("控制组选模没有得到 3 个有效候选")
    if tuple(str(item.get("candidate_alias")) for item in candidates) != (
        EVAL_CANDIDATES
    ):
        raise ControlAuditError("控制组选模候选顺序发生漂移")

    candidate_rows = []
    for candidate in candidates:
        model_path = Path(str(candidate.get("model_path", ""))).resolve()
        if model_path.parent != model_root:
            raise ControlAuditError(
                f"候选模型不属于控制组: {model_path}"
            )
        runs = candidate.get("validation_runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise ControlAuditError(
                f"{candidate.get('candidate_alias')} validation_runs 错误"
            )
        run_path = Path(str(runs[0].get("results_path", ""))).resolve()
        run_payload = _validate_evaluation_payload(
            run_path,
            expected_model_path=model_path,
        )
        candidate_summary = run_payload["summary"]
        if score_summary(candidate_summary) != score_summary(
            candidate.get("summary", {})
        ):
            raise ControlAuditError(
                f"{candidate.get('candidate_alias')} 汇总与原始结果不一致"
            )
        candidate_rows.append(
            {
                "alias": str(candidate.get("candidate_alias")),
                "resolved_variant": str(candidate.get("resolved_variant")),
                "model_path": str(model_path),
                "model_signature": str(candidate.get("model_signature")),
                "comparison_score": list(
                    comparison_score(score_summary(candidate_summary))
                ),
                "summary": candidate_summary,
                "results_path": str(run_path),
            }
        )

    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ControlAuditError("selection_summary 缺少 selected")
    selected_alias = str(selected.get("candidate_alias", ""))
    selected_model_path = Path(str(selected.get("model_path", ""))).resolve()
    if selected_alias not in EVAL_CANDIDATES:
        raise ControlAuditError(f"选中未知 checkpoint: {selected_alias}")

    official_path = (
        eval_root / "official_deterministic" / "evaluation_results.json"
    )
    official = _validate_evaluation_payload(
        official_path,
        expected_model_path=selected_model_path,
    )
    payload = {
        "schema_version": 1,
        "status": "pass",
        "phase": "checkpoint_selection_and_official_eval",
        "training_audit_path": str(training_audit_path.resolve()),
        "eval_root": str(eval_root.resolve()),
        "selection_summary_path": str(selection_path.resolve()),
        "validation_seed": VALIDATION_SEED,
        "eval_noise_seed": EVAL_NOISE_SEED,
        "episodes": EVAL_EPISODES,
        "validation_candidates": candidate_rows,
        "selected": {
            "alias": selected_alias,
            "model_path": str(selected_model_path),
            "comparison_score": list(
                comparison_score(score_summary(selected.get("summary", {})))
            ),
        },
        "official_results_path": str(official_path.resolve()),
        "official_summary": official["summary"],
    }
    _write_json_atomic(output, payload)
    if report is not None:
        _write_eval_report(report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--reference-run-spec", required=True)
    preflight.add_argument("--output", required=True)

    completed = subparsers.add_parser("completed")
    completed.add_argument("--parent-batch-dir", required=True)
    completed.add_argument("--m0-manifest", required=True)
    completed.add_argument("--source-snapshot", required=True)
    completed.add_argument("--output", required=True)
    completed.add_argument("--model-root-file")

    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("--training-audit", required=True)
    evaluation.add_argument("--eval-root", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        payload = run_preflight(
            reference_run_spec=Path(args.reference_run_spec).resolve(),
            output=Path(args.output).resolve(),
        )
        print(
            "[FULL CONTROL PREFLIGHT PASS] "
            f"source_files="
            f"{len(payload['source_snapshot']['protocol_source_signatures'])} "
            "manifest_difference=cross_agent_reference_only"
        )
        return 0
    if args.command == "completed":
        payload = run_completed(
            parent_batch_dir=Path(args.parent_batch_dir).resolve(),
            m0_manifest_path=Path(args.m0_manifest).resolve(),
            source_snapshot_path=Path(args.source_snapshot).resolve(),
            output=Path(args.output).resolve(),
            model_root_file=(
                Path(args.model_root_file).resolve()
                if args.model_root_file
                else None
            ),
        )
        print(
            "[FULL CONTROL COMPLETION PASS] "
            f"model_root={payload['model_root']} "
            "cross_reference_diagnostics=all_zero"
        )
        return 0
    if args.command == "evaluation":
        payload = run_evaluation_audit(
            training_audit_path=Path(args.training_audit).resolve(),
            eval_root=Path(args.eval_root).resolve(),
            output=Path(args.output).resolve(),
            report=Path(args.report).resolve() if args.report else None,
        )
        print(
            "[FULL CONTROL EVALUATION PASS] "
            f"selected={payload['selected']['alias']} "
            f"team_sr="
            f"{float(payload['official_summary']['team_success_rate']):.6f}"
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

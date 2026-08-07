#!/usr/bin/env python3
"""Validate and summarize the selector checkpoint degradation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selection_scoring import (
    SELECTION_SCORE_FIELDS,
    comparison_score,
    score_summary,
    selection_summary_errors,
)


MODEL_IDS = ("M0", "M1", "M2", "M3")
EXPECTED_CANDIDATES = {
    "M0": ("best_by_team_sr", "best", "final"),
    "M1": ("best", "final"),
    "M2": ("best", "final"),
    "M3": ("best", "final"),
}
EXPECTED_VALIDATION_SEED = 114817
EXPECTED_EVAL_NOISE_SEED = 101
EXPECTED_EPISODES = 30


class AuditError(RuntimeError):
    pass


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"JSON 顶层不是对象: {path}")
    return payload


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditError(f"{field} 不是数值: {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{field} 不是有限值: {value!r}")
    return result


def _episode_map(payload: Mapping[str, Any], *, source: Path) -> Dict[int, Dict[str, Any]]:
    details = payload.get("episode_details")
    if not isinstance(details, list) or len(details) != EXPECTED_EPISODES:
        raise AuditError(
            f"{source} episode_details 数量错误: "
            f"{len(details) if isinstance(details, list) else 'invalid'}"
        )
    result: Dict[int, Dict[str, Any]] = {}
    for item in details:
        if not isinstance(item, dict):
            raise AuditError(f"{source} 含非对象 episode_details")
        episode = int(item.get("episode", -1))
        if episode in result:
            raise AuditError(f"{source} episode={episode} 重复")
        result[episode] = item
    expected = set(range(EXPECTED_EPISODES))
    if set(result) != expected:
        raise AuditError(f"{source} episode 编号不完整")
    return result


def _directory_content_digest(paths: Iterable[Path]) -> Tuple[str, int]:
    hasher = hashlib.sha256()
    count = 0
    for path in sorted(paths, key=lambda item: item.name):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
        count += 1
    return hasher.hexdigest(), count


def _evaluation_position_digest(
    payload: Mapping[str, Any],
    *,
    source: Path,
) -> str:
    setup = payload.get("evaluation_setup")
    if not isinstance(setup, dict):
        raise AuditError(f"{source} 缺少 evaluation_setup")
    raw_path = str(setup.get("positions_file", "")).strip()
    if not raw_path:
        raise AuditError(f"{source} 缺少 evaluation_setup.positions_file")
    first_position_path = Path(raw_path).resolve()
    if not first_position_path.is_file():
        raise AuditError(f"{source} 位置文件不存在: {first_position_path}")
    digest, count = _directory_content_digest(
        first_position_path.parent.glob("episode_*.json")
    )
    if count != EXPECTED_EPISODES:
        raise AuditError(
            f"{source} 正式测试位置文件数量错误: {count} "
            f"(expected={EXPECTED_EPISODES})"
        )
    return digest


def _bootstrap_ci(values: Sequence[float], *, seed: int, samples: int = 20000) -> Tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise AuditError("无法对空数据计算 bootstrap CI")
    rng = np.random.default_rng(int(seed))
    chunk_size = 1000
    means: List[np.ndarray] = []
    remaining = int(samples)
    while remaining > 0:
        take = min(chunk_size, remaining)
        indices = rng.integers(0, data.size, size=(take, data.size))
        means.append(np.mean(data[indices], axis=1))
        remaining -= take
    distribution = np.concatenate(means)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return float(low), float(high)


def _paired_metrics(
    selected_payload: Mapping[str, Any],
    final_payload: Mapping[str, Any],
    *,
    model_index: int,
    selected_source: Path,
    final_source: Path,
) -> Dict[str, Any]:
    selected = _episode_map(selected_payload, source=selected_source)
    final = _episode_map(final_payload, source=final_source)
    deltas: Dict[str, List[float]] = {
        "team_success": [],
        "mean_agent_success": [],
        "collision_free": [],
        "collision_count": [],
        "reward": [],
        "final_goal_distance": [],
    }
    for episode in range(EXPECTED_EPISODES):
        left = selected[episode]
        right = final[episode]
        for seed_key in ("terrain_seed", "terrain_variant_seed", "obstacle_seed"):
            if left.get(seed_key) != right.get(seed_key):
                raise AuditError(
                    f"episode={episode} {seed_key} 不配对: "
                    f"{left.get(seed_key)!r} != {right.get(seed_key)!r}"
                )
        left_agents = left.get("agent_success_flags")
        right_agents = right.get("agent_success_flags")
        if not isinstance(left_agents, list) or not isinstance(right_agents, list):
            raise AuditError(f"episode={episode} 缺少 agent_success_flags")
        deltas["team_success"].append(
            float(int(left.get("team_success", 0) or 0) - int(right.get("team_success", 0) or 0))
        )
        deltas["mean_agent_success"].append(
            float(np.mean(left_agents) - np.mean(right_agents))
        )
        deltas["collision_free"].append(
            float(
                int(float(left.get("collision_count", 0) or 0) == 0.0)
                - int(float(right.get("collision_count", 0) or 0) == 0.0)
            )
        )
        deltas["collision_count"].append(
            _finite(left.get("collision_count"), field="selected collision_count")
            - _finite(right.get("collision_count"), field="final collision_count")
        )
        deltas["reward"].append(
            _finite(left.get("reward"), field="selected reward")
            - _finite(right.get("reward"), field="final reward")
        )
        deltas["final_goal_distance"].append(
            _finite(left.get("final_goal_distance"), field="selected final_goal_distance")
            - _finite(right.get("final_goal_distance"), field="final final_goal_distance")
        )
    result: Dict[str, Any] = {}
    for metric_index, (key, values) in enumerate(deltas.items()):
        ci = _bootstrap_ci(
            values,
            seed=20260729 + model_index * 100 + metric_index,
        )
        result[key] = {
            "mean_delta_selected_minus_final": float(np.mean(values)),
            "bootstrap_ci_95": [ci[0], ci[1]],
        }
    result["team_success_gains"] = int(sum(value > 0.0 for value in deltas["team_success"]))
    result["team_success_losses"] = int(sum(value < 0.0 for value in deltas["team_success"]))
    return result


def _validate_eval(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_model_path: Path | None = None,
) -> Dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise AuditError(f"{path} 缺少 summary")
    errors = selection_summary_errors(summary)
    if errors:
        raise AuditError(f"{path} summary 无法用于选模: {'; '.join(errors)}")
    if int(summary.get("episodes", 0) or 0) != EXPECTED_EPISODES:
        raise AuditError(f"{path} summary.episodes != {EXPECTED_EPISODES}")
    _episode_map(payload, source=path)
    setup = payload.get("evaluation_setup")
    if not isinstance(setup, dict):
        raise AuditError(f"{path} 缺少 evaluation_setup")
    if int(setup.get("eval_noise_seed", -1) or -1) != EXPECTED_EVAL_NOISE_SEED:
        raise AuditError(f"{path} eval_noise_seed 不是 {EXPECTED_EVAL_NOISE_SEED}")
    if _finite(setup.get("eval_noise_scale", 0.0), field="eval_noise_scale") != 0.0:
        raise AuditError(f"{path} 不是确定性评估")
    if _finite(setup.get("eval_random_action_prob", 0.0), field="eval_random_action_prob") != 0.0:
        raise AuditError(f"{path} 含随机动作")
    if expected_model_path is not None:
        actual_model_path = Path(str(payload.get("model_path", ""))).resolve()
        if actual_model_path != expected_model_path.resolve():
            raise AuditError(
                f"{path} model_path 不匹配: {actual_model_path} != {expected_model_path}"
            )
    return summary


def analyze(output_root: Path, formal_root: Path) -> Dict[str, Any]:
    output_root = output_root.resolve()
    formal_root = formal_root.resolve()
    if not output_root.is_dir():
        raise AuditError(f"输出目录不存在: {output_root}")
    if not formal_root.is_dir():
        raise AuditError(f"历史正式目录不存在: {formal_root}")

    validation_position_digests: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    total_candidates = 0
    for model_index, model_id in enumerate(MODEL_IDS):
        model_root = output_root / model_id
        selection_path = (
            model_root
            / "official_deterministic_matched_validation"
            / "selection_summary.json"
        )
        selection = _load_json(selection_path)
        if int(selection.get("validation_episodes", 0) or 0) != EXPECTED_EPISODES:
            raise AuditError(f"{model_id} validation_episodes 不匹配")
        if selection.get("validation_seeds") != [EXPECTED_VALIDATION_SEED]:
            raise AuditError(f"{model_id} validation_seeds 不匹配")
        requested_candidates = tuple(selection.get("validation_candidates", []))
        if requested_candidates != EXPECTED_CANDIDATES[model_id]:
            raise AuditError(
                f"{model_id} 候选集不匹配: {requested_candidates!r} "
                f"!= {EXPECTED_CANDIDATES[model_id]!r}"
            )
        candidates = selection.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(EXPECTED_CANDIDATES[model_id]):
            raise AuditError(f"{model_id} 有效候选数量错误")
        total_candidates += len(candidates)
        candidate_aliases = tuple(str(item.get("candidate_alias")) for item in candidates)
        if candidate_aliases != EXPECTED_CANDIDATES[model_id]:
            raise AuditError(f"{model_id} 实际候选顺序不匹配")
        for candidate in candidates:
            runs = candidate.get("validation_runs")
            if not isinstance(runs, list) or len(runs) != 1:
                raise AuditError(f"{model_id}/{candidate.get('candidate_alias')} validation_runs 错误")
            run = runs[0]
            run_path = Path(str(run.get("results_path", ""))).resolve()
            run_payload = _load_json(run_path)
            _validate_eval(
                run_payload,
                path=run_path,
                expected_model_path=Path(str(candidate.get("model_path", ""))),
            )
            setup = run_payload["evaluation_setup"]
            if str(setup.get("action_force_ratio_source", "")).strip() in ("", "forced_override"):
                raise AuditError(
                    f"{model_id}/{candidate.get('candidate_alias')} checkpoint FR 来源非法"
                )

        position_files = list(
            (
                model_root
                / "official_deterministic_matched_validation"
                / "testset"
            ).glob("*/episode_positions/episode_*.json")
        )
        digest, count = _directory_content_digest(position_files)
        if count != EXPECTED_EPISODES:
            raise AuditError(f"{model_id} held-out 位置文件数量错误: {count}")
        validation_position_digests[model_id] = digest

        selected = selection.get("selected")
        if not isinstance(selected, dict):
            raise AuditError(f"{model_id} selection_summary 缺少 selected")
        selected_alias = str(selected.get("candidate_alias", ""))
        selected_model_path = Path(str(selected.get("model_path", ""))).resolve()
        candidates_by_alias = {
            str(item.get("candidate_alias", "")): item
            for item in candidates
        }
        final_candidate = candidates_by_alias.get("final")
        if not isinstance(final_candidate, dict):
            raise AuditError(f"{model_id} 候选集中缺少 final")
        final_model_path = Path(str(final_candidate.get("model_path", ""))).resolve()

        official_path = model_root / "official_deterministic" / "evaluation_results.json"
        official_payload = _load_json(official_path)
        selected_summary = _validate_eval(
            official_payload,
            path=official_path,
            expected_model_path=selected_model_path,
        )
        final_path = formal_root / model_id / "deterministic" / "evaluation_results.json"
        final_payload = _load_json(final_path)
        final_summary = _validate_eval(
            final_payload,
            path=final_path,
            expected_model_path=final_model_path,
        )
        selected_position_digest = _evaluation_position_digest(
            official_payload,
            source=official_path,
        )
        final_position_digest = _evaluation_position_digest(
            final_payload,
            source=final_path,
        )
        if selected_position_digest != final_position_digest:
            raise AuditError(
                f"{model_id} selected/final 正式测试位置内容不一致: "
                f"{selected_position_digest} != {final_position_digest}"
            )

        selected_score = comparison_score(score_summary(selected_summary))
        final_score = comparison_score(score_summary(final_summary))
        if selected_alias != "final" and selected_score > final_score:
            degradation_status = "confirmed"
        elif selected_alias != "final":
            degradation_status = "inconclusive"
        else:
            degradation_status = "not_detected"

        rows.append(
            {
                "model_id": model_id,
                "validation_candidates": [
                    {
                        "alias": str(item.get("candidate_alias")),
                        "resolved_variant": str(item.get("resolved_variant")),
                        "model_path": str(item.get("model_path")),
                        "model_signature": str(item.get("model_signature")),
                        "score": list(item.get("comparison_score", [])),
                        "summary": item.get("summary", {}),
                    }
                    for item in candidates
                ],
                "selected_alias": selected_alias,
                "selected_model_path": str(selected_model_path),
                "validation_selected_score": list(selected.get("comparison_score", [])),
                "official_selected_score": list(selected_score),
                "official_final_score": list(final_score),
                "official_selected_summary": selected_summary,
                "official_final_summary": final_summary,
                "formal_positions_sha256": selected_position_digest,
                "paired_selected_minus_final": _paired_metrics(
                    official_payload,
                    final_payload,
                    model_index=model_index,
                    selected_source=official_path,
                    final_source=final_path,
                ),
                "final_degradation_status": degradation_status,
            }
        )

    if total_candidates != 9:
        raise AuditError(f"唯一候选总数不是 9: {total_candidates}")
    if len(set(validation_position_digests.values())) != 1:
        raise AuditError(
            f"四个模型的 held-out 位置内容不一致: {validation_position_digests}"
        )

    return {
        "schema_version": 1,
        "status": "pass",
        "output_root": str(output_root),
        "formal_reference_root": str(formal_root),
        "validation_seed": EXPECTED_VALIDATION_SEED,
        "eval_noise_seed": EXPECTED_EVAL_NOISE_SEED,
        "episodes_per_candidate": EXPECTED_EPISODES,
        "unique_checkpoint_count": total_candidates,
        "validation_positions_sha256": next(iter(validation_position_digests.values())),
        "selection_score_fields": list(SELECTION_SCORE_FIELDS),
        "models": rows,
    }


def _format_rate(value: Any) -> str:
    return f"{100.0 * _finite(value, field='rate'):.1f}%"


def write_report(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Selector checkpoint degradation audit",
        "",
        f"Validation: **{str(payload['status']).upper()}**; "
        f"{payload['unique_checkpoint_count']} unique checkpoints; "
        f"{payload['episodes_per_candidate']} held-out episodes per candidate.",
        "",
        f"Held-out seed: `{payload['validation_seed']}`; paired evaluator seed: "
        f"`{payload['eval_noise_seed']}`; validation position digest: "
        f"`{payload['validation_positions_sha256']}`.",
        "",
        "## Selection and independent formal test",
        "",
        "| Model | Held-out candidates | Selected | Held-out team SR | "
        "Official selected team SR | Official final team SR | "
        "Selected final distance | Final final distance | Degradation |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["models"]:
        selected_candidate = next(
            item
            for item in row["validation_candidates"]
            if item["alias"] == row["selected_alias"]
        )
        heldout = selected_candidate["summary"]
        selected_summary = row["official_selected_summary"]
        final_summary = row["official_final_summary"]
        lines.append(
            f"| {row['model_id']} | "
            f"{', '.join(item['alias'] for item in row['validation_candidates'])} | "
            f"{row['selected_alias']} | "
            f"{_format_rate(heldout['team_success_rate'])} | "
            f"{_format_rate(selected_summary['team_success_rate'])} | "
            f"{_format_rate(final_summary['team_success_rate'])} | "
            f"{_finite(selected_summary['avg_team_final_goal_distance'], field='distance'):.2f} | "
            f"{_finite(final_summary['avg_team_final_goal_distance'], field='distance'):.2f} | "
            f"{row['final_degradation_status']} |"
        )

    lines.extend(
        [
            "",
            "## Paired selected-minus-final deltas",
            "",
            "Positive success/reward deltas are better; negative collision/distance deltas are better.",
            "",
            "| Model | Team success Δ | Gains/losses | Collision count Δ [95% CI] | "
            "Reward Δ [95% CI] | Final distance Δ [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["models"]:
        paired = row["paired_selected_minus_final"]

        def cell(key: str) -> str:
            value = paired[key]
            low, high = value["bootstrap_ci_95"]
            return (
                f"{value['mean_delta_selected_minus_final']:.3f} "
                f"[{low:.3f}, {high:.3f}]"
            )

        lines.append(
            f"| {row['model_id']} | {cell('team_success')} | "
            f"{paired['team_success_gains']}/{paired['team_success_losses']} | "
            f"{cell('collision_count')} | {cell('reward')} | "
            f"{cell('final_goal_distance')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--formal-reference-root",
        type=Path,
        default=Path(
            "evaluation_results_selector_m0_m3_env4_seed101_v10_formal_gpu_v10"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze(args.output_root, args.formal_reference_root)
    output_root = args.output_root.resolve()
    json_path = output_root / "checkpoint_degradation_audit.json"
    report_path = output_root / "checkpoint_degradation_report.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(payload, report_path)
    print(f"[PASS] checkpoint audit: {json_path}")
    print(f"[PASS] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

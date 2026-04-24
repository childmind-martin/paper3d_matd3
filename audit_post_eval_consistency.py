#!/usr/bin/env python3
"""Audit multi-seed post-evaluation consistency for strict ablation batches."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EVAL_SETUP_FIELDS = (
    "terrain_family",
    "position_family",
    "terrain_complexity",
    "map_size",
    "mountain_min_distance",
    "random_terrain",
    "semi_random_terrain",
    "terrain_seed",
    "terrain_base_seed",
    "peak_jitter_range",
    "peak_center_jitter_range",
    "peak_height_jitter_ratio_min",
    "peak_height_jitter_ratio_max",
    "peak_height_max_scale",
    "terrain_variant_noise_ratio",
    "use_dynamic_obstacles",
    "use_quadrotor_dynamics",
    "gravity",
    "control_accel_gain",
    "damping",
    "agent_max_speed",
    "agent_accel",
    "action_range_x",
    "action_range_y",
    "action_range_z",
    "simulation_dt",
    "z_action_bias",
    "quadrotor_attitude_response_time",
    "quadrotor_psi_cmd",
    "episode_length",
    "requested_episode_length_multiplier",
    "action_force_ratio",
    "forced_action_force_ratio",
)

SPEC_FIELDS = (
    "version",
    "mode",
    "episodes",
    "episode_length_multiplier",
    "seed",
    "model_variant",
    "selection_protocol",
    "requested_model_variant",
    "validation_episodes",
    "validation_seed",
    "scenario_seed",
    "terrain_complexity",
    "map_size",
    "mountain_min_distance",
    "terrain_family",
    "random_terrain",
    "semi_random_terrain",
    "terrain_seed",
    "terrain_base_seed",
    "peak_jitter_range",
    "peak_center_jitter_range",
    "peak_height_jitter_ratio_min",
    "peak_height_jitter_ratio_max",
    "peak_height_max_scale",
    "terrain_variant_noise_ratio",
    "position_family",
    "start_center_jitter",
    "agent_local_jitter",
    "goal_region_radius",
    "use_dynamic_obstacles",
    "use_fixed_positions",
    "semi_random_hold_mode",
    "semi_random_hold_episodes",
    "semi_random_hold_min_episodes",
    "semi_random_hold_max_episodes",
    "terrain_seed_sequence",
    "terrain_variant_seed_sequence",
    "obstacle_seed_sequence",
)

TOP_LEVEL_EVAL_FIELDS = (
    "episodes",
    "terrain_seed_sequence",
    "terrain_variant_seed_sequence",
)

POST_EVAL_INHERITED_RUNTIME_ENV_KEYS = (
    "USE_QUADROTOR_DYNAMICS",
    "SIMULATION_DT",
    "Z_ACTION_BIAS",
    "QUADROTOR_ATTITUDE_RESPONSE_TIME",
    "QUADROTOR_PSI_CMD",
    "GRAVITY",
    "CONTROL_ACCEL_GAIN",
    "DAMPING",
    "AGENT_MAX_SPEED",
    "AGENT_ACCEL",
    "ACTION_RANGE_X",
    "ACTION_RANGE_Y",
    "ACTION_RANGE_Z",
)

POST_EVAL_LAUNCH_ENV_KEYS = (
    "TRAIN_PYTHON_BIN",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "GPU_ID",
)

RUNTIME_BOOL_KEYS = {"USE_QUADROTOR_DYNAMICS"}
RUNTIME_FLOAT_KEYS = {
    "SIMULATION_DT",
    "Z_ACTION_BIAS",
    "QUADROTOR_ATTITUDE_RESPONSE_TIME",
    "QUADROTOR_PSI_CMD",
    "GRAVITY",
    "CONTROL_ACCEL_GAIN",
    "DAMPING",
    "AGENT_MAX_SPEED",
    "AGENT_ACCEL",
    "ACTION_RANGE_X",
    "ACTION_RANGE_Y",
    "ACTION_RANGE_Z",
}

RESULTS_ARG_RUNTIME_ENV_MAPPING = {
    "use_quadrotor_dynamics": ("USE_QUADROTOR_DYNAMICS", lambda value: "1" if _to_bool(value) else "0"),
    "simulation_dt": ("SIMULATION_DT", lambda value: str(float(value))),
    "z_action_bias": ("Z_ACTION_BIAS", lambda value: str(float(value))),
    "quadrotor_attitude_response_time": ("QUADROTOR_ATTITUDE_RESPONSE_TIME", lambda value: str(float(value))),
    "quadrotor_psi_cmd": ("QUADROTOR_PSI_CMD", lambda value: str(float(value))),
    "gravity": ("GRAVITY", lambda value: str(float(value))),
    "control_accel_gain": ("CONTROL_ACCEL_GAIN", lambda value: str(float(value))),
    "damping": ("DAMPING", lambda value: str(float(value))),
    "agent_max_speed": ("AGENT_MAX_SPEED", lambda value: str(float(value))),
    "agent_accel": ("AGENT_ACCEL", lambda value: str(float(value))),
    "action_range_x": ("ACTION_RANGE_X", lambda value: str(float(value))),
    "action_range_y": ("ACTION_RANGE_Y", lambda value: str(float(value))),
    "action_range_z": ("ACTION_RANGE_Z", lambda value: str(float(value))),
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_runtime_value(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key in RUNTIME_BOOL_KEYS:
        return "1" if _to_bool(value) else "0"
    if key in RUNTIME_FLOAT_KEYS:
        try:
            return f"{float(value):.12g}"
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _stable_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or None


def _parse_seed_csv(value: Optional[str]) -> Optional[List[int]]:
    items = _parse_csv(value)
    if not items:
        return None
    return [int(item) for item in items]


def _parse_key_value(items: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_item in items:
        raw = str(raw_item).strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty KEY in: {raw}")
        parsed[key] = value.strip()
    return parsed


def _summary_path(batch_dir: Path) -> Path:
    return batch_dir / "plots" / "latest_summary.json"


def _infer_expected_from_summary(batch_dir: Path) -> Tuple[List[str], List[int]]:
    path = _summary_path(batch_dir)
    if not path.exists():
        return [], []
    data = _load_json(path)
    audit_report = data.get("audit_report") if isinstance(data.get("audit_report"), dict) else {}
    labels = audit_report.get("expected_labels") if isinstance(audit_report, dict) else None
    if not isinstance(labels, list) or not labels:
        labels = [item.get("label") for item in data.get("aggregated_experiments", []) if isinstance(item, dict)]
    seeds = data.get("seeds")
    if not isinstance(seeds, list):
        seeds = audit_report.get("expected_seeds") if isinstance(audit_report, dict) else []
    return [str(label) for label in labels if str(label).strip()], [int(seed) for seed in seeds]


def _iter_seed_batch_dirs(batch_dir: Path) -> Iterable[Path]:
    root = batch_dir / "seed_batches"
    if not root.exists():
        return []
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith("batch_") and "_seed" in child.name
    )


def _collect_artifacts(batch_dir: Path) -> Dict[Tuple[int, str], Path]:
    artifacts: Dict[Tuple[int, str], Path] = {}
    for child in _iter_seed_batch_dirs(batch_dir):
        artifact_dir = child / "results" / "experiment_artifacts"
        if not artifact_dir.exists():
            continue
        for artifact_path in sorted(artifact_dir.glob("*.json")):
            try:
                payload = _load_json(artifact_path)
            except Exception:
                continue
            seed = payload.get("seed")
            exp = payload.get("experiment") if isinstance(payload.get("experiment"), dict) else {}
            label = exp.get("label") or artifact_path.stem
            if seed is None or not label:
                continue
            artifacts[(int(seed), str(label))] = artifact_path
    return artifacts


def _seed_batch_dir_from_artifact(artifact_path: Path) -> Path:
    for parent in artifact_path.parents:
        if parent.name.startswith("batch_") and (parent / "results").exists():
            return parent
    return artifact_path.parents[2]


def _resolve_manifest_path_for_artifact(
    artifact_path: Path,
    label: str,
    exp: Dict[str, Any],
) -> Tuple[Optional[Path], str, List[str]]:
    issues: List[str] = []
    raw_manifest_path = str(exp.get("manifest_path") or "").strip()
    seed_batch_dir = _seed_batch_dir_from_artifact(artifact_path)
    fallback_manifest_path = seed_batch_dir / "manifests" / f"{label}_resolved_manifest.json"

    if raw_manifest_path:
        manifest_path = Path(raw_manifest_path)
        if manifest_path.exists() and manifest_path.is_file():
            return manifest_path, "artifact_manifest_path", issues
        issues.append(f"artifact manifest_path is missing on disk: {manifest_path}")
        if fallback_manifest_path.exists() and fallback_manifest_path.is_file():
            return fallback_manifest_path, "seed_batch_manifest_fallback_after_missing_artifact_path", issues
        return manifest_path, "artifact_manifest_path_missing", issues

    if fallback_manifest_path.exists() and fallback_manifest_path.is_file():
        return fallback_manifest_path, "seed_batch_manifest_fallback", issues
    issues.append(f"missing fallback manifest: {fallback_manifest_path}")
    return None, "missing_manifest", issues


def _load_results_args(log_dir: str) -> Optional[Dict[str, Any]]:
    raw_log_dir = str(log_dir or "").strip()
    if not raw_log_dir:
        return None
    results_path = Path(raw_log_dir) / "results.json"
    if not results_path.exists() or not results_path.is_file():
        return None
    try:
        data = _load_json(results_path)
    except Exception:
        return None
    args_obj = data.get("args")
    return args_obj if isinstance(args_obj, dict) else None


def _load_runtime_env_for_preflight(manifest_path: Optional[Path], log_dir: str) -> Dict[str, str]:
    runtime_env: Dict[str, str] = {}
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            manifest = {}
        for section_name in ("exec_env", "audit_env"):
            section = manifest.get(section_name)
            if not isinstance(section, dict) or not section:
                continue
            for key in POST_EVAL_INHERITED_RUNTIME_ENV_KEYS + POST_EVAL_LAUNCH_ENV_KEYS:
                value = section.get(key)
                if value is None:
                    continue
                value_str = str(value).strip()
                if value_str:
                    runtime_env.setdefault(key, value_str)

    run_args = _load_results_args(log_dir)
    if isinstance(run_args, dict):
        for arg_key, (env_key, formatter) in RESULTS_ARG_RUNTIME_ENV_MAPPING.items():
            if env_key in runtime_env:
                continue
            value = run_args.get(arg_key)
            if value is None:
                continue
            try:
                runtime_env[env_key] = str(formatter(value)).strip()
            except Exception:
                continue
    return runtime_env


def _existing_post_eval_dirs(batch_dir: Path) -> List[str]:
    names = {"post_eval", "post_eval_validation", "post_eval_testset"}
    existing: List[str] = []
    for child in _iter_seed_batch_dirs(batch_dir):
        results_dir = child / "results"
        for name in sorted(names):
            candidate = results_dir / name
            if candidate.exists():
                existing.append(str(candidate))
    return existing


def _record_from_artifact(seed: int, label: str, artifact_path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "seed": seed,
        "label": label,
        "artifact_path": str(artifact_path),
        "status": "artifact_loaded",
        "issues": [],
    }
    try:
        artifact = _load_json(artifact_path)
    except Exception as exc:
        record["status"] = "artifact_load_failed"
        record["issues"].append(str(exc))
        return record

    exp = artifact.get("experiment") if isinstance(artifact.get("experiment"), dict) else {}
    post_eval = exp.get("post_eval") if isinstance(exp.get("post_eval"), dict) else {}
    record["post_eval_status"] = post_eval.get("status")
    record["post_eval_skip_reason"] = post_eval.get("skip_reason")
    record["results_path"] = post_eval.get("results_path")
    record["spec_path"] = post_eval.get("spec_path")
    record["eval_dir"] = post_eval.get("eval_dir")
    record["selected_model_candidate"] = post_eval.get("selected_model_candidate")
    record["selected_model_variant"] = post_eval.get("selected_model_variant")

    if post_eval.get("status") != "completed":
        record["issues"].append(f"post_eval status is {post_eval.get('status')!r}")

    raw_results_path = str(post_eval.get("results_path") or "").strip()
    if not raw_results_path:
        record["status"] = "missing_results"
        record["issues"].append("missing results_path in post_eval artifact")
        return record
    results_path = Path(raw_results_path)
    if not results_path.exists() or not results_path.is_file():
        record["status"] = "missing_results"
        record["issues"].append(f"missing evaluation_results.json: {results_path}")
        return record

    try:
        eval_data = _load_json(results_path)
    except Exception as exc:
        record["status"] = "results_load_failed"
        record["issues"].append(str(exc))
        return record

    spec_data: Dict[str, Any] = {}
    spec_path = Path(str(post_eval.get("spec_path") or ""))
    if spec_path.exists():
        try:
            spec_data = _load_json(spec_path)
        except Exception as exc:
            record["issues"].append(f"spec load failed: {exc}")

    eval_setup = eval_data.get("evaluation_setup")
    if not isinstance(eval_setup, dict):
        eval_setup = {}
        record["issues"].append("missing evaluation_setup")

    record["status"] = "ok"
    record["eval_fields"] = {
        f"eval.{key}": eval_setup.get(key)
        for key in EVAL_SETUP_FIELDS
    }
    record["eval_fields"].update(
        {
            f"eval_top.{key}": eval_data.get(key)
            for key in TOP_LEVEL_EVAL_FIELDS
        }
    )
    record["spec_fields"] = {
        f"spec.{key}": spec_data.get(key)
        for key in SPEC_FIELDS
    }
    record["summary_metrics"] = eval_data.get("summary") if isinstance(eval_data.get("summary"), dict) else {}
    return record


def _compare_records(
    records: Sequence[Dict[str, Any]],
    field_prefix: str,
    ignored_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    ignored_fields = ignored_fields or set()
    buckets_by_field: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for record in records:
        fields = record.get(field_prefix)
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            if key in ignored_fields:
                continue
            buckets_by_field.setdefault(key, defaultdict(list))[_stable_value(value)].append(
                {"seed": record["seed"], "label": record["label"]}
            )

    diffs: Dict[str, Any] = {}
    for key, buckets in sorted(buckets_by_field.items()):
        if len(buckets) <= 1:
            continue
        diffs[key] = [
            {"value": value, "count": len(items), "items": items}
            for value, items in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
    return diffs


def audit_batch(
    batch_dir: Path,
    expected_labels: Optional[List[str]] = None,
    expected_seeds: Optional[List[int]] = None,
    ignored_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    summary_labels, summary_seeds = _infer_expected_from_summary(batch_dir)
    labels = expected_labels or summary_labels
    seeds = expected_seeds or summary_seeds

    artifacts = _collect_artifacts(batch_dir)
    if not labels:
        labels = sorted({label for _, label in artifacts.keys()})
    if not seeds:
        seeds = sorted({seed for seed, _ in artifacts.keys()})

    records: List[Dict[str, Any]] = []
    issues: List[str] = []
    for seed in seeds:
        for label in labels:
            artifact_path = artifacts.get((int(seed), str(label)))
            if artifact_path is None:
                records.append(
                    {
                        "seed": int(seed),
                        "label": str(label),
                        "status": "missing_artifact",
                        "issues": ["missing experiment artifact"],
                    }
                )
                issues.append(f"missing artifact: seed={seed}, label={label}")
                continue
            record = _record_from_artifact(int(seed), str(label), artifact_path)
            records.append(record)
            for issue in record.get("issues", []):
                issues.append(f"seed={seed}, label={label}: {issue}")

    ignored_fields = ignored_fields or set()
    eval_diffs = _compare_records(records, "eval_fields", ignored_fields=ignored_fields)
    spec_diffs = _compare_records(records, "spec_fields", ignored_fields=ignored_fields)
    for key in eval_diffs:
        issues.append(f"inconsistent evaluation field: {key}")
    for key in spec_diffs:
        issues.append(f"inconsistent post-eval spec field: {key}")

    completed = sum(1 for record in records if record.get("status") == "ok")
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(batch_dir),
        "summary_path": str(_summary_path(batch_dir)),
        "expected_labels": labels,
        "expected_seeds": seeds,
        "expected_record_count": len(labels) * len(seeds),
        "completed_record_count": completed,
        "ignored_fields": sorted(ignored_fields),
        "passed": not issues,
        "issues": issues,
        "eval_field_diffs": eval_diffs,
        "spec_field_diffs": spec_diffs,
        "records": records,
    }


def _preflight_runtime_diffs(
    records: Sequence[Dict[str, Any]],
    ignored_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    ignored_fields = ignored_fields or set()
    buckets_by_field: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for record in records:
        runtime_env = record.get("runtime_env")
        if not isinstance(runtime_env, dict):
            continue
        for key in POST_EVAL_INHERITED_RUNTIME_ENV_KEYS:
            field = f"runtime.{key}"
            if field in ignored_fields:
                continue
            value = runtime_env.get(key)
            buckets_by_field.setdefault(field, defaultdict(list))[
                _normalize_runtime_value(key, value)
            ].append({"seed": record["seed"], "label": record["label"]})

    diffs: Dict[str, Any] = {}
    for key, buckets in sorted(buckets_by_field.items()):
        non_empty_buckets = {value: items for value, items in buckets.items() if value != ""}
        if len(non_empty_buckets) <= 1 and "" not in buckets:
            continue
        if len(non_empty_buckets) <= 1 and len(buckets) <= 1:
            continue
        diffs[key] = [
            {"value": value, "count": len(items), "items": items}
            for value, items in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
    return diffs


def preflight_batch(
    batch_dir: Path,
    expected_labels: Optional[List[str]] = None,
    expected_seeds: Optional[List[int]] = None,
    expected_runtime_env: Optional[Dict[str, str]] = None,
    require_clean_post_eval: bool = True,
    ignored_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    summary_labels, summary_seeds = _infer_expected_from_summary(batch_dir)
    labels = expected_labels or summary_labels
    seeds = expected_seeds or summary_seeds

    artifacts = _collect_artifacts(batch_dir)
    if not labels:
        labels = sorted({label for _, label in artifacts.keys()})
    if not seeds:
        seeds = sorted({seed for seed, _ in artifacts.keys()})

    expected_runtime_env = expected_runtime_env or {}
    ignored_fields = ignored_fields or set()
    records: List[Dict[str, Any]] = []
    issues: List[str] = []

    existing_eval_dirs = _existing_post_eval_dirs(batch_dir)
    if require_clean_post_eval and existing_eval_dirs:
        issues.append(
            "existing post-eval directories found before rerun; clean them or pass --allow-existing-post-eval"
        )

    for seed in seeds:
        for label in labels:
            artifact_path = artifacts.get((int(seed), str(label)))
            record: Dict[str, Any] = {
                "seed": int(seed),
                "label": str(label),
                "status": "pending",
                "issues": [],
            }
            if artifact_path is None:
                record["status"] = "missing_artifact"
                record["issues"].append("missing experiment artifact")
                records.append(record)
                issues.append(f"missing artifact: seed={seed}, label={label}")
                continue

            record["artifact_path"] = str(artifact_path)
            try:
                artifact = _load_json(artifact_path)
            except Exception as exc:
                record["status"] = "artifact_load_failed"
                record["issues"].append(str(exc))
                records.append(record)
                issues.append(f"seed={seed}, label={label}: artifact load failed: {exc}")
                continue

            exp = artifact.get("experiment") if isinstance(artifact.get("experiment"), dict) else {}
            log_dir = str(exp.get("log_dir") or "").strip()
            record["log_dir"] = log_dir
            if not log_dir:
                record["issues"].append("missing training log_dir in artifact")
            elif not Path(log_dir).exists():
                record["issues"].append(f"training log_dir missing on disk: {log_dir}")
            elif not (Path(log_dir) / "results.json").exists():
                record["issues"].append(f"training results.json missing: {Path(log_dir) / 'results.json'}")

            post_eval = exp.get("post_eval") if isinstance(exp.get("post_eval"), dict) else {}
            record["artifact_post_eval_status"] = post_eval.get("status")
            if require_clean_post_eval:
                for field_name in ("results_path", "spec_path", "eval_dir", "log_path"):
                    raw_value = str(post_eval.get(field_name) or "").strip()
                    if raw_value:
                        record["issues"].append(
                            f"artifact still contains stale post_eval.{field_name}: {raw_value}"
                        )

            manifest_path, manifest_source, manifest_issues = _resolve_manifest_path_for_artifact(
                artifact_path,
                str(label),
                exp,
            )
            record["manifest_path"] = str(manifest_path) if manifest_path is not None else ""
            record["manifest_source"] = manifest_source
            record["issues"].extend(manifest_issues)
            if manifest_path is None or not manifest_path.exists():
                record["issues"].append("post-eval runtime manifest cannot be resolved")

            runtime_env = _load_runtime_env_for_preflight(manifest_path, log_dir)
            normalized_runtime_env = {
                key: _normalize_runtime_value(key, runtime_env.get(key))
                for key in POST_EVAL_INHERITED_RUNTIME_ENV_KEYS
                if key in runtime_env
            }
            record["runtime_env"] = normalized_runtime_env
            record["launch_env"] = {
                key: runtime_env.get(key)
                for key in POST_EVAL_LAUNCH_ENV_KEYS
                if runtime_env.get(key)
            }

            for key, expected_value in expected_runtime_env.items():
                actual_value = normalized_runtime_env.get(key)
                expected_norm = _normalize_runtime_value(key, expected_value)
                if actual_value is None:
                    record["issues"].append(f"missing expected runtime env {key}={expected_norm}")
                elif _normalize_runtime_value(key, actual_value) != expected_norm:
                    record["issues"].append(
                        f"runtime env {key} mismatch: got={actual_value}, expected={expected_norm}"
                    )

            record["status"] = "ok" if not record["issues"] else "failed"
            records.append(record)
            for issue in record["issues"]:
                issues.append(f"seed={seed}, label={label}: {issue}")

    runtime_diffs = _preflight_runtime_diffs(records, ignored_fields=ignored_fields)
    for key in runtime_diffs:
        issues.append(f"inconsistent inherited runtime env: {key}")

    ready = sum(1 for record in records if record.get("status") == "ok")
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "preflight",
        "batch_dir": str(batch_dir),
        "summary_path": str(_summary_path(batch_dir)),
        "expected_labels": labels,
        "expected_seeds": seeds,
        "expected_record_count": len(labels) * len(seeds),
        "ready_record_count": ready,
        "expected_runtime_env": expected_runtime_env,
        "require_clean_post_eval": require_clean_post_eval,
        "existing_post_eval_dirs": existing_eval_dirs,
        "ignored_fields": sorted(ignored_fields),
        "passed": not issues,
        "issues": issues,
        "runtime_env_diffs": runtime_diffs,
        "records": records,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print("=" * 72)
    if report.get("mode") == "preflight":
        print("Post-eval preflight audit")
    else:
        print("Post-eval consistency audit")
    print(f"Batch: {report['batch_dir']}")
    print(f"Expected labels: {len(report['expected_labels'])} -> {report['expected_labels']}")
    print(f"Expected seeds: {report['expected_seeds']}")
    if report.get("mode") == "preflight":
        print(
            "Records: "
            f"{report['ready_record_count']}/{report['expected_record_count']} ready"
        )
        if report.get("expected_runtime_env"):
            print(f"Expected runtime env: {report['expected_runtime_env']}")
    else:
        print(
            "Records: "
            f"{report['completed_record_count']}/{report['expected_record_count']} ok"
        )
    print(f"Passed: {report['passed']}")
    if report["issues"]:
        print("\nIssues:")
        for issue in report["issues"][:80]:
            print(f"  - {issue}")
        if len(report["issues"]) > 80:
            print(f"  ... {len(report['issues']) - 80} more")

    for title, key in (("Evaluation field diffs", "eval_field_diffs"), ("Spec field diffs", "spec_field_diffs")):
        diffs = report.get(key) or {}
        if not diffs:
            continue
        print(f"\n{title}:")
        for field, buckets in list(diffs.items())[:40]:
            print(f"  {field}:")
            for bucket in buckets:
                sample = bucket["items"][:8]
            print(f"    value={bucket['value']} count={bucket['count']} sample={sample}")
        if len(diffs) > 40:
            print(f"  ... {len(diffs) - 40} more fields")
    runtime_diffs = report.get("runtime_env_diffs") or {}
    if runtime_diffs:
        print("\nRuntime env diffs:")
        for field, buckets in list(runtime_diffs.items())[:40]:
            print(f"  {field}:")
            for bucket in buckets:
                sample = bucket["items"][:8]
                print(f"    value={bucket['value']} count={bucket['count']} sample={sample}")
        if len(runtime_diffs) > 40:
            print(f"  ... {len(runtime_diffs) - 40} more fields")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="multi-seed parent batch directory")
    parser.add_argument("--expected-labels", type=str, default=None, help="comma-separated labels; defaults to latest_summary audit labels")
    parser.add_argument("--expected-seeds", type=str, default=None, help="comma-separated seeds; defaults to latest_summary seeds")
    parser.add_argument("--output", type=Path, default=None, help="audit JSON path; defaults to batch_dir/results/post_eval_consistency_audit_<ts>.json")
    parser.add_argument("--preflight", action="store_true", help="audit artifacts/manifests/runtime inheritance before running post-eval")
    parser.add_argument(
        "--expect-runtime-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="preflight expectation for inherited runtime env, e.g. USE_QUADROTOR_DYNAMICS=1; repeatable",
    )
    parser.add_argument(
        "--allow-existing-post-eval",
        action="store_true",
        help="preflight: do not fail when post_eval directories or stale artifact paths already exist",
    )
    parser.add_argument(
        "--ignore-field",
        action="append",
        default=[],
        help="fully-qualified field to ignore, e.g. eval.action_force_ratio; repeatable",
    )
    parser.add_argument("--strict", action="store_true", help="return non-zero when inconsistencies are found")
    args = parser.parse_args(argv)

    batch_dir = args.batch_dir.resolve()
    ignored_fields = {str(item).strip() for item in args.ignore_field if str(item).strip()}
    if args.preflight:
        report = preflight_batch(
            batch_dir,
            expected_labels=_parse_csv(args.expected_labels),
            expected_seeds=_parse_seed_csv(args.expected_seeds),
            expected_runtime_env=_parse_key_value(args.expect_runtime_env),
            require_clean_post_eval=not bool(args.allow_existing_post_eval),
            ignored_fields=ignored_fields,
        )
    else:
        report = audit_batch(
            batch_dir,
            expected_labels=_parse_csv(args.expected_labels),
            expected_seeds=_parse_seed_csv(args.expected_seeds),
            ignored_fields=ignored_fields,
        )
    _print_report(report)

    output_path = args.output
    if output_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = "post_eval_preflight_audit" if args.preflight else "post_eval_consistency_audit"
        output_path = batch_dir / "results" / f"{stem}_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Audit JSON: {output_path}")
    return 1 if args.strict and not report["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())

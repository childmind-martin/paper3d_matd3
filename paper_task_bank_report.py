#!/usr/bin/env python3
"""Create paper-facing, task-level characterization of an immutable task bank."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from paper_experiment_schema import (
    PaperSchemaError,
    atomic_write_json,
    content_sha256,
    fingerprint_artifact,
)
from paper_task_bank import (
    load_task_bank,
    validate_dynamic_feasibility_audit,
)
from paper_task_layout import task_difficulty_summary


REPORT_SCHEMA_VERSION = "paper_task_bank_characterization_v1"
_STRUCTURAL_METRICS = (
    "minimum_horizontal_distance_m",
    "maximum_horizontal_distance_m",
    "mean_reference_path_length_m",
    "maximum_reference_path_length_m",
    "maximum_detour_ratio",
    "maximum_cumulative_climb_m",
    "minimum_boundary_clearance_m",
    "minimum_obstacle_clearance_m",
)
_WITNESS_METRICS = (
    "deadline_usage_fraction",
    "episode_minimum_clearance_m",
)


def _numeric_summary(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise PaperSchemaError("numeric characterization requires finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "sample_std": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
    }


def _optional_numeric_summary(values: Iterable[Any]) -> Dict[str, Any] | None:
    finite = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number):
            raise PaperSchemaError("task characterization contains non-finite data")
        finite.append(number)
    return _numeric_summary(finite) if finite else None


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    if path.exists():
        raise PaperSchemaError(f"refusing to overwrite task report CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def characterize_task_bank(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    if manifest_file.is_dir():
        manifest_file = manifest_file / "task_bank_manifest.json"
    bank = load_task_bank(manifest_file)
    feasibility_errors = validate_dynamic_feasibility_audit(bank)
    if feasibility_errors:
        raise PaperSchemaError(
            "task characterization requires the formal dynamic audit: "
            + "; ".join(feasibility_errors)
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise PaperSchemaError(
            f"refusing to overwrite non-empty task report directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    metadata = dict(bank.get("metadata") or {})
    generation_audit = dict(metadata.get("generation_acceptance_audit") or {})
    dynamic_audit = dict(metadata["dynamic_feasibility_audit"])
    dynamic_splits = dict(dynamic_audit.get("splits") or {})
    task_rows: list[Dict[str, Any]] = []
    split_summaries: Dict[str, Any] = {}

    for split_name, split_cfg_raw in sorted(dict(bank["splits"]).items()):
        split_cfg = dict(split_cfg_raw)
        layouts = []
        bank_root = manifest_file.parent
        for entry in list(split_cfg.get("tasks") or []):
            layout_path = (bank_root / str(entry["task_layout_file"])).resolve()
            if not layout_path.is_relative_to(bank_root):
                raise PaperSchemaError(
                    f"task layout escapes immutable bank: {layout_path}"
                )
            with layout_path.open("r", encoding="utf-8") as stream:
                layouts.append(json.load(stream))
        witness_episodes = list(
            dict(dynamic_splits.get(split_name) or {}).get("episodes") or []
        )
        if len(layouts) != int(split_cfg["task_count"]) or len(
            witness_episodes
        ) != len(layouts):
            raise PaperSchemaError(
                f"task/witness count mismatch in split={split_name}"
            )
        if [str(item.get("task_id", "")) for item in witness_episodes] != [
            str(layout["task_id"]) for layout in layouts
        ]:
            raise PaperSchemaError(
                f"task/witness order mismatch in split={split_name}"
            )

        summaries = [task_difficulty_summary(layout) for layout in layouts]
        runtime = dict(split_cfg.get("runtime_config") or {})
        difficulty_values = {str(layout["difficulty_bin"]) for layout in layouts}
        if len(difficulty_values) != 1:
            raise PaperSchemaError(f"split mixes difficulty bins: {split_name}")
        difficulty = next(iter(difficulty_values))
        acceptance = dict(generation_audit.get(split_name) or {})
        if int(acceptance.get("accepted", -1)) != len(layouts):
            raise PaperSchemaError(
                f"generation acceptance audit mismatch in split={split_name}"
            )

        structural = {
            metric: _optional_numeric_summary(
                summary.get(metric) for summary in summaries
            )
            for metric in _STRUCTURAL_METRICS
        }
        witness = {
            metric: _numeric_summary(
                float(episode[metric]) for episode in witness_episodes
            )
            for metric in _WITNESS_METRICS
        }
        deadlines_steps = [
            int((layout.get("contract") or {})["deadline"]["deadline_steps"])
            for layout in layouts
        ]
        deadlines_seconds = [
            float((layout.get("contract") or {})["deadline"]["deadline_seconds"])
            for layout in layouts
        ]
        split_summaries[split_name] = {
            "role": str(split_cfg["role"]),
            "difficulty_bin": difficulty,
            "task_count": len(layouts),
            "terrain_seed": int(runtime["terrain_seed"]),
            "terrain_mode": {
                "semi_random": bool(runtime.get("semi_random_terrain", False)),
                "per_episode": bool(runtime.get("per_episode_terrain", False)),
            },
            "obstacle_count": int(runtime["paper_task_num_obstacles"]),
            "generation_acceptance": acceptance,
            "structural_metrics": structural,
            "deadline_steps": _numeric_summary(deadlines_steps),
            "deadline_seconds": _numeric_summary(deadlines_seconds),
            "oracle_dynamic_feasibility_witness": {
                "role": dynamic_audit["audit_role"],
                "team_successes": len(witness_episodes),
                "team_success_rate": 1.0,
                "metrics": witness,
            },
        }

        for layout, structural_row, episode in zip(
            layouts, summaries, witness_episodes
        ):
            task_rows.append(
                {
                    "split": split_name,
                    "role": str(split_cfg["role"]),
                    "difficulty_bin": difficulty,
                    "task_id": str(layout["task_id"]),
                    "terrain_seed": int(layout["seeds"]["terrain"]),
                    "terrain_variant_seed": int(
                        layout["seeds"]["terrain_variant"]
                    ),
                    "obstacle_seed": int(layout["seeds"]["obstacle"]),
                    "position_seed": int(layout["seeds"]["position"]),
                    "obstacle_count": int(structural_row["obstacle_count"]),
                    **{
                        metric: structural_row.get(metric)
                        for metric in _STRUCTURAL_METRICS
                    },
                    "deadline_steps": int(
                        layout["contract"]["deadline"]["deadline_steps"]
                    ),
                    "deadline_seconds": float(
                        layout["contract"]["deadline"]["deadline_seconds"]
                    ),
                    "oracle_executed_steps": int(episode["executed_steps"]),
                    "oracle_deadline_usage_fraction": float(
                        episode["deadline_usage_fraction"]
                    ),
                    "oracle_episode_minimum_clearance_m": float(
                        episode["episode_minimum_clearance_m"]
                    ),
                    "oracle_team_success": int(bool(episode["team_success"])),
                }
            )

    manifest_artifact = fingerprint_artifact(manifest_file.parent)
    manifest_artifact.update(
        {
            "manifest_path": str(manifest_file),
            "bank_id": str(bank["bank_id"]),
            "bank_content_sha256": str(bank["content_sha256"]),
        }
    )
    csv_path = destination / "paper_task_bank_tasks.csv"
    _write_csv(csv_path, task_rows)
    repo_root = Path(__file__).resolve().parent
    source_artifacts = []
    for source_path in (
        Path(__file__).resolve(),
        repo_root / "paper_task_bank.py",
        repo_root / "paper_task_layout.py",
    ):
        artifact = fingerprint_artifact(source_path)
        artifact["logical_name"] = str(source_path.relative_to(repo_root))
        source_artifacts.append(artifact)
    payload: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_bank": manifest_artifact,
        "task_level_csv": fingerprint_artifact(csv_path),
        "report_source_artifacts": source_artifacts,
        "task_count": len(task_rows),
        "split_count": len(split_summaries),
        "split_summaries": split_summaries,
        "physics_contract": dict(dynamic_audit["controller"]["physics_contract"]),
        "task_semantics_contract": dict(
            dynamic_audit["controller"]["task_semantics"]
        ),
        "oracle_witness_is_comparison_baseline": False,
    }
    payload["content_sha256"] = content_sha256(payload)
    json_path = destination / "paper_task_bank_characterization.json"
    atomic_write_json(json_path, payload)
    print(f"task bank characterization written: {json_path}")
    print(f"task-level characterization written: {csv_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-bank-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    characterize_task_bank(args.task_bank_manifest, args.output_dir)


if __name__ == "__main__":
    main()

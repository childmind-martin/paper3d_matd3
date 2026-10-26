#!/usr/bin/env python3
"""Generate immutable C0-C3 and held-out G1 task banks from the real scenario."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List

from audit_paper_task_feasibility import AUDIT_SCHEMA_VERSION, audit_split
from multiagent.scenarios.paper3d_terrain_energy import Scenario
from paper_curriculum import build_paper_curriculum, build_paper_generalization_suite
from paper_experiment_schema import content_sha256, fingerprint_artifact
from paper_task_bank import validate_dynamic_feasibility_audit, write_task_bank
from paper_task_layout import (
    TaskLayoutError,
    atomic_write_json,
    validate_task_difficulty_acceptance,
)


PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION = "paper_task_bank_build_cache_v5"


def _audit_split_worker(manifest: str, split_name: str) -> Dict[str, Any]:
    """Process-pool entry point for one immutable split audit."""
    return audit_split(Path(manifest), split_name)


def _validated_cache_payload(path: Path, *, expected_kind: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"task-bank cache schema mismatch: {path}")
    if payload.get("cache_kind") != expected_kind:
        raise RuntimeError(f"task-bank cache kind mismatch: {path}")
    expected_hash = content_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    if payload.get("content_sha256") != expected_hash:
        raise RuntimeError(f"task-bank cache content hash mismatch: {path}")
    return payload


def _write_cache_payload(path: Path, payload: Dict[str, Any]) -> None:
    normalized = dict(payload)
    normalized["schema_version"] = PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION
    normalized["content_sha256"] = content_sha256(normalized)
    atomic_write_json(path, normalized)


def _build_generation_spec(
    *,
    bank_id: str,
    base_terrain_seed: int,
    held_out_terrain_seeds: Iterable[int],
    train_count: int,
    validation_count: int,
    test_count: int,
    generalization_count_per_seed: int,
    curriculum: List[Dict[str, Any]],
    generalization: List[Dict[str, Any]],
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    sources = []
    for source_path in (
        Path(__file__).resolve(),
        repo_root / "audit_paper_task_feasibility.py",
        repo_root / "paper_curriculum.py",
        repo_root / "paper_task_bank.py",
        repo_root / "paper_task_layout.py",
        repo_root / "multiagent" / "core.py",
        repo_root / "multiagent" / "environment.py",
        repo_root / "multiagent" / "scenarios" / "paper3d_terrain_energy.py",
    ):
        artifact = fingerprint_artifact(source_path)
        artifact["logical_name"] = str(source_path.relative_to(repo_root))
        sources.append(artifact)
    payload: Dict[str, Any] = {
        "schema_version": PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION,
        "cache_kind": "generation_spec",
        "bank_id": str(bank_id),
        "base_terrain_seed": int(base_terrain_seed),
        "held_out_terrain_seeds": [
            int(seed) for seed in held_out_terrain_seeds
        ],
        "task_counts": {
            "train_per_difficulty": int(train_count),
            "validation_per_difficulty": int(validation_count),
            "test_per_difficulty": int(test_count),
            "generalization_per_terrain_seed": int(
                generalization_count_per_seed
            ),
        },
        "curriculum": curriculum,
        "generalization": generalization,
        "source_artifacts": sources,
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def _prepare_work_dir(work_dir: Path, spec: Dict[str, Any]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / "generation_spec.json"
    if spec_path.exists():
        persisted = _validated_cache_payload(
            spec_path, expected_kind="generation_spec"
        )
        if persisted != spec:
            raise RuntimeError(
                f"task-bank work directory belongs to a different generation spec: {work_dir}"
            )
    else:
        unexpected = [path for path in work_dir.iterdir()]
        if unexpected:
            raise RuntimeError(
                f"task-bank work directory lacks generation spec but is non-empty: {work_dir}"
            )
        atomic_write_json(spec_path, spec)


def _cached_or_generate_split(
    *,
    work_dir: Path | None,
    generation_spec_sha256: str,
    config: Dict[str, Any],
    count: int,
    bank_id: str,
    split_name: str,
    role: str,
) -> Dict[str, Any]:
    cache_path = (
        work_dir / "generated_splits" / f"{split_name}.json"
        if work_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        cached = _validated_cache_payload(cache_path, expected_kind="split")
        if cached.get("generation_spec_sha256") != generation_spec_sha256:
            raise RuntimeError(f"cached split generation spec mismatch: {split_name}")
        if cached.get("split") != split_name or cached.get("role") != role:
            raise RuntimeError(f"cached split identity mismatch: {split_name}")
        layouts = list(cached.get("layouts") or [])
        if len(layouts) != int(count):
            raise RuntimeError(f"cached split task count mismatch: {split_name}")
        generation_audit = cached.get("generation_audit")
        if not isinstance(generation_audit, dict):
            raise RuntimeError(
                f"cached split lacks generation audit: {split_name}"
            )
        if int(generation_audit.get("accepted", -1)) != len(layouts):
            raise RuntimeError(
                f"cached split generation audit mismatch: {split_name}"
            )
        print(f"[task-bank] reused generated split={split_name} tasks={len(layouts)}")
        return {"layouts": layouts, "generation_audit": generation_audit}
    print(f"[task-bank] generating split={split_name} tasks={count}")
    layouts, generation_audit = _generate_layouts(
        config, count=count, bank_id=bank_id, split_name=split_name
    )
    if cache_path is not None:
        _write_cache_payload(
            cache_path,
            {
                "cache_kind": "split",
                "generation_spec_sha256": generation_spec_sha256,
                "split": split_name,
                "role": role,
                "task_count": len(layouts),
                "generation_audit": generation_audit,
                "layouts": layouts,
            },
        )
    print(f"[task-bank] completed split={split_name} tasks={len(layouts)}")
    return {"layouts": layouts, "generation_audit": generation_audit}


def _namespace_seed(bank_id: str, split_name: str) -> int:
    digest = hashlib.sha256(f"{bank_id}|{split_name}".encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % 2_000_000_000


def _scenario_kwargs(config: Dict[str, Any], sequence_seed: int) -> Dict[str, Any]:
    keys = (
        "terrain_seed",
        "terrain_base_seed",
        "terrain_complexity_level",
        "use_dynamic_obstacles",
        "use_fixed_positions",
        "dynamic_first_time",
        "random_terrain",
        "semi_random_terrain",
        "per_episode_terrain",
        "per_env_terrain",
        "min_start_goal_dist",
        "max_start_goal_dist",
        "min_agent_start_separation",
        "paper_goal_formation_radius",
        "paper_goal_boundary_clearance",
        "paper_task_reference_surface_clearance",
        "success_distance_threshold",
        "collision_distance_threshold",
        "paper_task_deadline_policy",
        "paper_task_contract_enabled",
        "paper_task_contract_strict",
        "paper_task_require_reference_feasible",
        "hard_deadline_terminal",
        "paper_terminal_on_collision",
        "paper_reward_profile",
        "paper_progress_reward_scale",
        "paper_individual_reach_bonus",
        "paper_team_success_bonus",
        "paper_collision_penalty",
        "paper_out_of_bounds_penalty",
        "paper_deadline_penalty",
        "paper_step_penalty",
        "paper_control_effort_penalty",
        "paper_clearance_margin",
        "paper_clearance_penalty",
        "include_time_to_go_observation",
        "paper_task_num_obstacles",
    )
    kwargs = {key: config[key] for key in keys if key in config}
    kwargs["seed"] = int(config["terrain_seed"])
    kwargs["task_difficulty_bin"] = str(config["difficulty_bin"])
    kwargs["deterministic_train_env_sequence"] = True
    kwargs["training_env_sequence_seed"] = int(sequence_seed)
    kwargs["paper_task_reference_grid_resolution"] = 4.0
    kwargs["paper_task_obstacle_inflation"] = 1.5
    return kwargs


def _generate_layouts(
    config: Dict[str, Any], *, count: int, bank_id: str, split_name: str
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if count <= 0:
        raise ValueError(f"task count for {split_name} must be positive")
    sequence_seed = _namespace_seed(bank_id, split_name)
    scenario = Scenario(**_scenario_kwargs(config, sequence_seed))
    world = scenario.make_world()
    world.env_id = 0
    layouts: List[Dict[str, Any]] = []
    acceptance = config.get("difficulty_acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"split={split_name} lacks difficulty_acceptance")
    rejection_counts: Counter[str] = Counter()
    candidate_index = 0
    maximum_candidates = max(100, int(count) * 100)
    while len(layouts) < count and candidate_index < maximum_candidates:
        world.episode_index = int(candidate_index)
        try:
            scenario.reset_world(world)
            layout = world.current_task_layout
            if not isinstance(layout, dict):
                raise RuntimeError(
                    f"scenario did not produce task layout for "
                    f"{split_name}/{candidate_index}"
                )
            acceptance_errors = validate_task_difficulty_acceptance(
                layout, acceptance
            )
            if acceptance_errors:
                for error in acceptance_errors:
                    rejection_counts[str(error).split("=", 1)[0]] += 1
            else:
                layouts.append(layout)
        except TaskLayoutError as exc:
            rejection_counts[str(exc).split(":", 1)[0]] += 1
        candidate_index += 1
    if len(layouts) != count:
        raise RuntimeError(
            f"unable to generate accepted tasks for split={split_name}: "
            f"accepted={len(layouts)}/{count}, candidates={candidate_index}, "
            f"rejections={dict(rejection_counts)}"
        )
    return layouts, {
        "accepted": len(layouts),
        "candidates_examined": int(candidate_index),
        "rejected": int(candidate_index - len(layouts)),
        "acceptance_rate": float(len(layouts) / candidate_index),
        "rejection_reason_incidence": dict(sorted(rejection_counts.items())),
    }


def _generate_split_worker(
    config: Dict[str, Any],
    count: int,
    bank_id: str,
    split_name: str,
) -> Dict[str, Any]:
    """Process-pool entry point for one deterministic split generation."""
    layouts, generation_audit = _generate_layouts(
        config,
        count=count,
        bank_id=bank_id,
        split_name=split_name,
    )
    return {"layouts": layouts, "generation_audit": generation_audit}


def generate_bank(
    output_dir: str | Path,
    *,
    bank_id: str,
    base_terrain_seed: int,
    held_out_terrain_seeds: Iterable[int],
    train_count: int,
    validation_count: int,
    test_count: int,
    generalization_count_per_seed: int,
    run_dynamic_feasibility_audit: bool = True,
    generation_workers: int = 1,
    dynamic_audit_workers: int = 1,
    work_dir: str | Path | None = None,
) -> Dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty task-bank output: {destination}"
        )
    held_out_terrain_seeds = [int(seed) for seed in held_out_terrain_seeds]
    generation_workers = int(generation_workers)
    dynamic_audit_workers = int(dynamic_audit_workers)
    if generation_workers <= 0:
        raise ValueError("generation_workers must be positive")
    if dynamic_audit_workers <= 0:
        raise ValueError("dynamic_audit_workers must be positive")
    curriculum = build_paper_curriculum(base_terrain_seed)
    generalization = build_paper_generalization_suite(held_out_terrain_seeds)
    generation_spec = _build_generation_spec(
        bank_id=bank_id,
        base_terrain_seed=base_terrain_seed,
        held_out_terrain_seeds=held_out_terrain_seeds,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        generalization_count_per_seed=generalization_count_per_seed,
        curriculum=curriculum,
        generalization=generalization,
    )
    resolved_work_dir = (
        Path(work_dir).expanduser().resolve() if work_dir is not None else None
    )
    if resolved_work_dir is not None:
        _prepare_work_dir(resolved_work_dir, generation_spec)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    roles: Dict[str, str] = {}
    runtime_configs: Dict[str, Dict[str, Any]] = {}
    generation_audit: Dict[str, Dict[str, int]] = {}
    generation_requests: List[Dict[str, Any]] = []
    for stage in curriculum:
        difficulty = stage["difficulty_bin"]
        for prefix, role, count in (
            ("train", "train", train_count),
            ("validation", "validation", validation_count),
            ("test", "test_in_distribution", test_count),
        ):
            split_name = f"{prefix}_{difficulty}"
            generation_requests.append(
                {
                    "split_name": split_name,
                    "role": role,
                    "count": int(count),
                    "config": dict(stage),
                }
            )
    for suite in generalization:
        split_name = f"test_{suite['suite_id']}"
        generation_requests.append(
            {
                "split_name": split_name,
                "role": "test_generalization",
                "count": int(generalization_count_per_seed),
                "config": dict(suite),
            }
        )

    generated_results: Dict[str, Dict[str, Any]] = {}
    missing_generations: List[Dict[str, Any]] = []
    for request in generation_requests:
        split_name = request["split_name"]
        cache_path = (
            resolved_work_dir / "generated_splits" / f"{split_name}.json"
            if resolved_work_dir is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            generated_results[split_name] = _cached_or_generate_split(
                work_dir=resolved_work_dir,
                generation_spec_sha256=generation_spec["content_sha256"],
                config=request["config"],
                count=request["count"],
                bank_id=bank_id,
                split_name=split_name,
                role=request["role"],
            )
        else:
            missing_generations.append(request)

    def persist_generation(
        request: Dict[str, Any], split_result: Dict[str, Any]
    ) -> None:
        split_name = request["split_name"]
        layouts = list(split_result["layouts"])
        split_generation_audit = dict(split_result["generation_audit"])
        if len(layouts) != request["count"]:
            raise RuntimeError(
                f"generated split task count mismatch: {split_name}"
            )
        if resolved_work_dir is not None:
            _write_cache_payload(
                resolved_work_dir
                / "generated_splits"
                / f"{split_name}.json",
                {
                    "cache_kind": "split",
                    "generation_spec_sha256": generation_spec[
                        "content_sha256"
                    ],
                    "split": split_name,
                    "role": request["role"],
                    "task_count": len(layouts),
                    "generation_audit": split_generation_audit,
                    "layouts": layouts,
                },
            )
        generated_results[split_name] = {
            "layouts": layouts,
            "generation_audit": split_generation_audit,
        }
        print(
            f"[task-bank] completed split={split_name} tasks={len(layouts)}"
        )

    if missing_generations and generation_workers == 1:
        for request in missing_generations:
            split_name = request["split_name"]
            print(
                f"[task-bank] generating split={split_name} "
                f"tasks={request['count']}"
            )
            persist_generation(
                request,
                _generate_split_worker(
                    request["config"],
                    request["count"],
                    bank_id,
                    split_name,
                ),
            )
    elif missing_generations:
        worker_count = min(generation_workers, len(missing_generations))
        print(
            f"[task-bank] parallel split generation splits="
            f"{len(missing_generations)} workers={worker_count}"
        )
        worker_errors = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _generate_split_worker,
                    request["config"],
                    request["count"],
                    bank_id,
                    request["split_name"],
                ): request
                for request in missing_generations
            }
            for future in as_completed(futures):
                request = futures[future]
                split_name = request["split_name"]
                try:
                    split_result = future.result()
                except Exception as exc:
                    worker_errors.append((split_name, repr(exc)))
                    print(
                        f"[task-bank] split generation worker failed "
                        f"split={split_name}: {exc!r}"
                    )
                    continue
                persist_generation(request, split_result)
        if worker_errors:
            raise RuntimeError(
                f"task-bank split generation worker failures: {worker_errors}"
            )

    for request in generation_requests:
        split_name = request["split_name"]
        split_result = generated_results[split_name]
        splits[split_name] = list(split_result["layouts"])
        roles[split_name] = request["role"]
        runtime_configs[split_name] = dict(request["config"])
        generation_audit[split_name] = dict(
            split_result["generation_audit"]
        )
    dynamic_feasibility_audit: Dict[str, Any]
    if run_dynamic_feasibility_audit:
        with tempfile.TemporaryDirectory(
            prefix="matd3_paper_bank_audit_"
        ) as staging_dir:
            staging_manifest = write_task_bank(
                staging_dir,
                bank_id=f"{bank_id}__dynamic_audit_staging",
                splits=splits,
                split_roles=roles,
                split_runtime_configs=runtime_configs,
                metadata={"staging_only": True},
            )
            resolved_audits: Dict[str, Dict[str, Any]] = {}
            missing_audits: List[str] = []
            for split_name in sorted(splits):
                audit_cache_path = (
                    resolved_work_dir
                    / "dynamic_audits"
                    / f"{split_name}.json"
                    if resolved_work_dir is not None
                    else None
                )
                if audit_cache_path is not None and audit_cache_path.exists():
                    cached_audit = _validated_cache_payload(
                        audit_cache_path,
                        expected_kind="dynamic_audit_split",
                    )
                    cached_payload = dict(cached_audit.get("audit") or {})
                    expected_task_ids = [
                        str(layout["task_id"]) for layout in splits[split_name]
                    ]
                    cached_task_ids = [
                        str(episode.get("task_id", ""))
                        for episode in cached_payload.get("episodes") or []
                    ]
                    if (
                        cached_audit.get("generation_spec_sha256")
                        != generation_spec["content_sha256"]
                        or cached_audit.get("staging_task_bank_content_sha256")
                        != staging_manifest["content_sha256"]
                        or cached_payload.get("split") != split_name
                        or cached_payload.get("split_role") != roles[split_name]
                        or cached_task_ids != expected_task_ids
                    ):
                        raise RuntimeError(
                            f"cached dynamic audit identity mismatch: {split_name}"
                        )
                    resolved_audits[split_name] = cached_payload
                    print(
                        f"[task-bank] reused dynamic audit split={split_name} "
                        f"tasks={len(expected_task_ids)}"
                    )
                else:
                    missing_audits.append(split_name)

            staging_manifest_path = str(
                Path(staging_dir) / "task_bank_manifest.json"
            )

            def persist_audit(split_name: str, audit: Dict[str, Any]) -> None:
                audit_cache_path = (
                    resolved_work_dir
                    / "dynamic_audits"
                    / f"{split_name}.json"
                    if resolved_work_dir is not None
                    else None
                )
                if audit_cache_path is not None:
                    _write_cache_payload(
                        audit_cache_path,
                        {
                            "cache_kind": "dynamic_audit_split",
                            "generation_spec_sha256": generation_spec[
                                "content_sha256"
                            ],
                            "staging_task_bank_content_sha256": (
                                staging_manifest["content_sha256"]
                            ),
                            "split": split_name,
                            "audit": audit,
                        },
                    )
                resolved_audits[split_name] = audit
                print(
                    f"[task-bank] completed dynamic audit split={split_name} "
                    f"tasks={len(splits[split_name])} "
                    f"successes={audit['team_successes']}"
                )

            if missing_audits and dynamic_audit_workers == 1:
                for split_name in missing_audits:
                    print(
                        f"[task-bank] auditing dynamic feasibility split={split_name} "
                        f"tasks={len(splits[split_name])}"
                    )
                    audit = audit_split(
                        staging_manifest_path,
                        split_name,
                    )
                    persist_audit(split_name, audit)
            elif missing_audits:
                worker_count = min(
                    dynamic_audit_workers, len(missing_audits)
                )
                print(
                    f"[task-bank] parallel dynamic audits splits="
                    f"{len(missing_audits)} workers={worker_count}"
                )
                worker_errors = []
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=context,
                ) as executor:
                    futures = {
                        executor.submit(
                            _audit_split_worker,
                            staging_manifest_path,
                            split_name,
                        ): split_name
                        for split_name in missing_audits
                    }
                    for future in as_completed(futures):
                        split_name = futures[future]
                        try:
                            audit = future.result()
                        except Exception as exc:
                            worker_errors.append((split_name, repr(exc)))
                            print(
                                f"[task-bank] dynamic audit worker failed "
                                f"split={split_name}: {exc!r}"
                            )
                            continue
                        persist_audit(split_name, audit)
                if worker_errors:
                    raise RuntimeError(
                        f"dynamic feasibility audit worker failures: "
                        f"{worker_errors}"
                    )

            split_audits = {}
            frozen_controller = None
            for split_name in sorted(splits):
                audit = resolved_audits[split_name]
                if not bool(audit["all_tasks_dynamically_feasible"]):
                    failed_ids = [
                        episode["task_id"]
                        for episode in audit["episodes"]
                        if not episode["team_success"]
                    ]
                    raise RuntimeError(
                        f"dynamic feasibility audit failed for split={split_name}: "
                        f"task_ids={failed_ids}"
                    )
                if frozen_controller is None:
                    frozen_controller = dict(audit["controller"])
                elif dict(audit["controller"]) != frozen_controller:
                    raise RuntimeError(
                        "dynamic feasibility controller changed across task splits"
                    )
                split_audits[split_name] = {
                    "split_role": str(audit["split_role"]),
                    "tasks_audited": int(audit["tasks_audited"]),
                    "team_successes": int(audit["team_successes"]),
                    "all_tasks_dynamically_feasible": True,
                    "episodes": audit["episodes"],
                }
            dynamic_feasibility_audit = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "audit_role": (
                    "oracle_dynamic_feasibility_witness_not_comparison_baseline"
                ),
                "performed": True,
                "all_tasks_dynamically_feasible": True,
                "controller": frozen_controller,
                "staging_task_bank_content_sha256": staging_manifest[
                    "content_sha256"
                ],
                "splits": split_audits,
            }
            audit_validation_manifest = {
                "splits": {
                    split_name: {
                        "role": roles[split_name],
                        "runtime_config": runtime_configs[split_name],
                        "tasks": [
                            {
                                "task_id": layout["task_id"],
                                "difficulty_bin": layout["difficulty_bin"],
                                "deadline": layout["contract"]["deadline"],
                            }
                            for layout in layouts
                        ],
                    }
                    for split_name, layouts in splits.items()
                },
                "metadata": {
                    "dynamic_feasibility_audit": dynamic_feasibility_audit
                },
            }
            audit_errors = validate_dynamic_feasibility_audit(
                audit_validation_manifest
            )
            if audit_errors:
                raise RuntimeError(
                    "combined dynamic feasibility audit is invalid: "
                    + "; ".join(audit_errors)
                )
    else:
        dynamic_feasibility_audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "performed": False,
            "all_tasks_dynamically_feasible": False,
            "splits": {},
        }

    return write_task_bank(
        output_dir,
        bank_id=bank_id,
        splits=splits,
        split_roles=roles,
        split_runtime_configs=runtime_configs,
        metadata={
            "generator": "build_paper_task_bank.py",
            "generation_spec": generation_spec,
            "generation_spec_content_sha256": generation_spec[
                "content_sha256"
            ],
            "generation_source_artifacts": generation_spec["source_artifacts"],
            "base_terrain_seed": int(base_terrain_seed),
            "held_out_terrain_seeds": [int(seed) for seed in held_out_terrain_seeds],
            "task_counts": {
                "train_per_difficulty": int(train_count),
                "validation_per_difficulty": int(validation_count),
                "test_per_difficulty": int(test_count),
                "generalization_per_terrain_seed": int(generalization_count_per_seed),
            },
            "role_shuffle_enabled": False,
            "generation_acceptance_audit": generation_audit,
            "dynamic_feasibility_audit": dynamic_feasibility_audit,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--base-terrain-seed", type=int, default=67)
    parser.add_argument("--held-out-terrain-seeds", default="701,719,743,787")
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--generalization-count-per-seed", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        default="",
        help=(
            "可恢复生成缓存目录；默认使用 OUTPUT_DIR.work。缓存与生成参数及源码哈希绑定"
        ),
    )
    parser.add_argument(
        "--skip-dynamic-feasibility-audit",
        action="store_true",
        help="development-only; resulting bank is rejected by formal training",
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) // 2)),
        help="并行确定性 split 生成进程数；输出仍按固定 split 顺序组装",
    )
    parser.add_argument(
        "--dynamic-audit-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) // 2)),
        help="并行动态可行性 split 审计进程数；缓存仍按 split 独立保存",
    )
    args = parser.parse_args()
    held_out = [int(token.strip()) for token in args.held_out_terrain_seeds.split(",") if token.strip()]
    os.environ["QUIET_OUTPUT"] = "1"
    os.environ["SUPPRESS_TERRAIN_OUTPUT"] = "1"
    os.environ["ENABLE_ROLE_SHUFFLE"] = "0"
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if str(args.work_dir).strip()
        else Path(str(Path(args.output_dir).expanduser().resolve()) + ".work")
    )
    manifest = generate_bank(
        args.output_dir,
        bank_id=args.bank_id,
        base_terrain_seed=args.base_terrain_seed,
        held_out_terrain_seeds=held_out,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        generalization_count_per_seed=args.generalization_count_per_seed,
        run_dynamic_feasibility_audit=not args.skip_dynamic_feasibility_audit,
        generation_workers=args.generation_workers,
        dynamic_audit_workers=args.dynamic_audit_workers,
        work_dir=work_dir,
    )
    print(
        f"task bank written: {Path(args.output_dir).resolve()} "
        f"hash={manifest['content_sha256']} splits={len(manifest['splits'])}"
    )
    print(f"resumable generation cache retained: {work_dir}")


if __name__ == "__main__":
    main()

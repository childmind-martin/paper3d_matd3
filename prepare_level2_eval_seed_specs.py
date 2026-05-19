#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from official_eval_with_matched_validation import (
    _build_post_eval_sequence_fields,
    _ensure_episode_positions,
    _generate_post_eval_testset_tag,
)


DEFAULT_BASE_SPEC = (
    "/home/tang/matd3/ablation_experiments/"
    "multi_seed_groupB_20260331_220752_testset2_20260409/"
    "seed_batches/batch_groupB_seed101_20260331_220752/results/post_eval_shared_spec.json"
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Spec JSON must be an object: {path}")
    return payload


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _light_artifact_policy() -> Dict[str, bool]:
    return {
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
    }


def _build_eval_spec(
    *,
    base_spec: Dict[str, Any],
    output_root: Path,
    eval_seed: int,
    episodes: int,
    force_regenerate: bool,
    testset_role: str,
) -> Path:
    spec = dict(base_spec)
    try:
        spec["version"] = int(spec.get("version", 10)) + 1
    except Exception:
        spec["version"] = 11
    spec["enabled"] = True
    spec["episodes"] = int(episodes)
    spec["seed"] = int(eval_seed)
    spec["model_variant"] = "fixed_best_by_team_sr"
    spec["selection_protocol"] = "fixed"
    spec["requested_model_variant"] = "best_by_team_sr"
    spec["validation_episodes"] = 0
    spec["validation_candidates"] = ["best_by_team_sr"]
    spec["artifact_policy"] = _light_artifact_policy()
    spec["force_regenerate_testset"] = bool(force_regenerate)
    spec["testset_role"] = str(testset_role)

    spec.update(_build_post_eval_sequence_fields(spec))

    eval_root = output_root / f"eval_seed_{int(eval_seed)}"
    testset_dir = eval_root / "post_eval_testset" / _generate_post_eval_testset_tag(spec) / "episode_positions"
    spec["episode_positions_dir"] = str(testset_dir)
    spec = _ensure_episode_positions(spec, force_regenerate=force_regenerate)

    spec_path = eval_root / "post_eval_shared_spec.json"
    spec["spec_path"] = str(spec_path)
    _save_json(spec_path, spec)
    return spec_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare repeated eval specs from an existing official post-eval base spec."
    )
    parser.add_argument("--base-spec", default=DEFAULT_BASE_SPEC)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--testset-role", default="level2_dual_semantics_eval10x30")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_spec_path = Path(args.base_spec).resolve()
    output_root = Path(args.output_root).resolve()
    if not base_spec_path.exists():
        raise RuntimeError(f"Base spec missing: {base_spec_path}")
    if int(args.episodes) <= 0:
        raise RuntimeError(f"episodes must be positive: {args.episodes}")

    base_spec = _load_json(base_spec_path)
    spec_paths = []
    for seed in args.eval_seeds:
        spec_paths.append(
            _build_eval_spec(
                base_spec=base_spec,
                output_root=output_root,
                eval_seed=int(seed),
                episodes=int(args.episodes),
                force_regenerate=bool(args.force_regenerate),
                testset_role=str(args.testset_role),
            )
        )

    print(f"Prepared {len(spec_paths)} eval spec(s) under {output_root}")
    for path in spec_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

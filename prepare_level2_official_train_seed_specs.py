#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
DEFAULT_OUTPUT_ROOT = (
    "/home/tang/matd3/ablation_experiments/"
    "multi_seed_groupB_20260331_220752_testset2_20260409/seed_batches"
)
DEFAULT_BATCH_STAMP = "20260331_220752"


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


def _target_spec_path(output_root: Path, train_seed: int, batch_stamp: str) -> Path:
    return output_root / f"batch_groupB_seed{int(train_seed)}_{batch_stamp}" / "results" / "post_eval_shared_spec.json"


def _build_spec_for_train_seed(
    *,
    base_spec: Dict[str, Any],
    output_root: Path,
    train_seed: int,
    batch_stamp: str,
    episodes: Optional[int],
    official_eval_seed: Optional[int],
    force_regenerate_testset: bool,
) -> Dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    spec["enabled"] = True
    if episodes is not None:
        spec["episodes"] = int(episodes)
    if official_eval_seed is not None:
        spec["seed"] = int(official_eval_seed)

    spec_path = _target_spec_path(output_root, train_seed, batch_stamp)
    testset_dir = spec_path.parent / "post_eval_testset" / _generate_post_eval_testset_tag(spec) / "episode_positions"
    spec["spec_path"] = str(spec_path)
    spec["episode_positions_dir"] = str(testset_dir)
    spec["force_regenerate_testset"] = bool(force_regenerate_testset)
    spec["train_seed"] = int(train_seed)

    spec.update(_build_post_eval_sequence_fields(spec))
    spec = _ensure_episode_positions(spec, force_regenerate=force_regenerate_testset)
    spec["testset_prepared"] = True
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Level-2 official post-eval specs in the exact directory layout "
            "expected by run_level2_multiseed_all_algos_official.sh."
        )
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--base-spec", default=DEFAULT_BASE_SPEC)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-stamp", default=DEFAULT_BATCH_STAMP)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override official test episodes. Omit this to keep the base spec value.",
    )
    parser.add_argument(
        "--official-eval-seed",
        type=int,
        default=None,
        help="Override the official testset seed. Omit this to keep the base spec value.",
    )
    parser.add_argument("--force-regenerate-testset", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_spec_path = Path(args.base_spec).resolve()
    output_root = Path(args.output_root).resolve()
    if not base_spec_path.exists():
        raise RuntimeError(f"Base spec missing: {base_spec_path}")
    if args.episodes is not None and int(args.episodes) <= 0:
        raise RuntimeError(f"episodes must be positive: {args.episodes}")

    base_spec = _load_json(base_spec_path)
    prepared: List[Path] = []
    skipped: List[Path] = []

    for train_seed in args.train_seeds:
        spec_path = _target_spec_path(output_root, int(train_seed), str(args.batch_stamp))
        if spec_path.exists() and not args.overwrite_existing:
            skipped.append(spec_path)
            continue
        if args.dry_run:
            prepared.append(spec_path)
            continue

        spec = _build_spec_for_train_seed(
            base_spec=base_spec,
            output_root=output_root,
            train_seed=int(train_seed),
            batch_stamp=str(args.batch_stamp),
            episodes=args.episodes,
            official_eval_seed=args.official_eval_seed,
            force_regenerate_testset=bool(args.force_regenerate_testset),
        )
        _save_json(spec_path, spec)
        prepared.append(spec_path)

    if prepared:
        action = "Would prepare" if args.dry_run else "Prepared"
        print(f"{action} {len(prepared)} spec(s):")
        for path in prepared:
            print(path)
    if skipped:
        print(f"Skipped {len(skipped)} existing spec(s) without --overwrite-existing:")
        for path in skipped:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

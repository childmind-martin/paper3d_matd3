#!/usr/bin/env python3
"""Resolve all M0-M3 manifests and prove num_envs propagation without training."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ablation_dual_q_separated_gradient as ablation
from selector_experiment_protocol import SELECTOR_PROTOCOL_EXPERIMENT_LABELS


def _args():
    original_argv = list(sys.argv)
    try:
        sys.argv = [
            "preflight_selector_num_env_manifests.py",
            "--episodes",
            "1",
            "--batch-size",
            "1024",
            "--num-envs",
            "4",
            "--batch-seed",
            "101",
            "--experiment-group",
            "B",
            "--config-mode",
            "strict_ablation",
            "--env-isolation",
            "strict",
            "--scenario-seed",
            "88",
            "--experiments",
            *SELECTOR_PROTOCOL_EXPERIMENT_LABELS,
        ]
        args = ablation.parse_args()
    finally:
        sys.argv = original_argv
    args.parsed_seeds = [101]
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


def main() -> int:
    args = _args()
    configs = ablation._select_experiment_configs(args)
    if [cfg["label"] for cfg in configs] != list(
        SELECTOR_PROTOCOL_EXPERIMENT_LABELS
    ):
        raise RuntimeError("selector configuration order is not M0-M3")

    rows = []
    with tempfile.TemporaryDirectory(
        prefix="matd3_selector_num_env_preflight_"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        manifest_dir = temp_root / "manifests"
        positions_file = temp_root / "positions.json"
        for config in configs:
            manifest, manifest_path = ablation._resolve_experiment_manifest(
                config,
                positions_file,
                args,
                manifest_dir,
            )
            exec_env = manifest.get("exec_env", {})
            meta = manifest.get("meta", {})
            argv = list(manifest.get("argv", []) or [])
            try:
                num_env_flag_index = argv.index("--num-envs")
                argv_num_envs = int(argv[num_env_flag_index + 1])
            except (ValueError, IndexError, TypeError):
                argv_num_envs = None
            observed = {
                "meta": int(meta.get("num_envs", 0) or 0),
                "exec_env": int(exec_env.get("NUM_ENVS", 0) or 0),
                "argv": argv_num_envs,
            }
            if set(observed.values()) != {4}:
                raise RuntimeError(
                    f"{config['label']} num_envs propagation failed: "
                    f"{observed}"
                )
            ablation._validate_resolved_manifest_identity(
                manifest,
                manifest_path,
                label=config["label"],
                exp_name_base=meta["exp_name_base"],
                seed=101,
                episodes=1,
                batch_size=1024,
                num_envs=4,
            )
            rows.append(
                {
                    "label": config["label"],
                    "num_envs": 4,
                    "manifest_sha256": meta[
                        ablation.RESOLVED_TRAINING_MANIFEST_HASH_KEY
                    ],
                }
            )

    print(
        json.dumps(
            {
                "status": "PASS",
                "models": rows,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

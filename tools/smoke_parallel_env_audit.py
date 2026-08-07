#!/usr/bin/env python3
"""Exercise the production ParallelEnv IPC and episode-audit path."""

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")
os.environ.setdefault("QUIET_OUTPUT", "1")
os.environ.setdefault("DEBUG_EPISODE_SUMMARY", "0")
os.environ.setdefault("DEBUG_COLLISION_SUMMARY", "0")
os.environ.setdefault("ENABLE_BEST_TRAJECTORY", "0")

import paper3d_train_optimized as training


def _production_args():
    original_argv = list(sys.argv)
    try:
        sys.argv = [
            "smoke_parallel_env_audit.py",
            "--algo",
            "matd3",
            "--scenario",
            "paper3d_terrain_vectorized",
            "--train-episodes",
            "1",
            "--episode-length",
            "2",
            "--num-envs",
            "4",
            "--actor-hidden",
            "16,16",
            "--critic-hidden",
            "16,16",
            "--use-tf-potential-field",
            "true",
            "--use-fr-feature",
            "true",
            "--use-pf-feature",
            "true",
            "--action-force-ratio",
            "0.5",
            "--jit-compile",
            "false",
            "--xla-global",
            "false",
            "--seed",
            "101",
            "--terrain-seed",
            "88",
            "--terrain-base-seed",
            "88",
            "--training-env-sequence-seed",
            "88",
            "--deterministic-train-env-sequence",
            "true",
            "--semi-random-terrain",
            "true",
            "--use-dynamic-obstacles",
            "true",
        ]
        return training.parse_args()
    finally:
        sys.argv = original_argv


def main() -> int:
    args = _production_args()
    args_dict = vars(args)
    probe_env = training.make_env_init(0, args_dict)
    try:
        agent_count = int(probe_env.n)
        observation_shapes = [
            int(space.shape[0]) for space in probe_env.observation_space
        ]
        environment_action_dims = [
            int(space.shape[0]) for space in probe_env.action_space
        ]
    finally:
        probe_env.close()

    args.eval_actor_only = True
    # Production training intentionally uses a 7-D policy action: xyz control
    # plus four PF parameters.  The environment's native action space only
    # describes the xyz execution head, so do not derive this value from it.
    action_dims = [7] * agent_count
    if any(dim != 3 for dim in environment_action_dims):
        raise RuntimeError(
            "unexpected native environment action dimensions: "
            f"{environment_action_dims}"
        )
    learner = training.OptimizedMATD3(
        n_agents=agent_count,
        obs_shapes=observation_shapes,
        action_dims=action_dims,
        args=args,
    )
    scenario = training.load_scenario_module(args.scenario, args)
    scenario.make_world()
    learner.scenario = scenario
    learner.update_terrain_cache(scenario)
    logical_gpu_count = len(training.tf.config.list_logical_devices("GPU"))

    env_fns = [
        functools.partial(training.make_env_init, env_id, args_dict)
        for env_id in range(int(args.num_envs))
    ]
    parent_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    parallel_env = training.ParallelEnv(env_fns)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != parent_cuda_visible_devices:
        parallel_env.close()
        raise RuntimeError(
            "ParallelEnv construction changed the parent "
            "CUDA_VISIBLE_DEVICES value"
        )
    try:
        observations = parallel_env.reset()
        if observations.shape[0] != int(args.num_envs):
            raise RuntimeError(
                f"reset batch={observations.shape}, expected env axis "
                f"{int(args.num_envs)}"
            )
        processed_observations = (
            learner.obs_processor.batch_process_observations_vectorized(
                observations
            )
        )
        training._validate_training_observation_batch(
            processed_observations,
            expected_envs=args.num_envs,
            expected_agents=agent_count,
            expected_obs_dim=max(observation_shapes),
            context="parallel action smoke",
        )
        action_outputs = learner.batch_select_actions_vectorized(
            processed_observations,
            add_noise=True,
        )
        expected_output_shapes = (
            (int(args.num_envs), agent_count, max(action_dims)),
            (int(args.num_envs), agent_count, max(action_dims)),
            (int(args.num_envs), agent_count, 3),
            (int(args.num_envs), agent_count, max(action_dims)),
            (int(args.num_envs), agent_count, 3),
            (int(args.num_envs), agent_count),
        )
        actual_output_shapes = tuple(
            tuple(int(dim) for dim in output.shape)
            for output in action_outputs
        )
        if actual_output_shapes != expected_output_shapes:
            raise RuntimeError(
                "production action path shape mismatch: "
                f"actual={actual_output_shapes}, "
                f"expected={expected_output_shapes}"
            )
        actions = np.ascontiguousarray(
            action_outputs[1].numpy(),
            dtype=np.float32,
        )
        _, _, _, infos = parallel_env.step(actions)
        if len(infos) != int(args.num_envs):
            raise RuntimeError(
                f"info batch={len(infos)}, expected={int(args.num_envs)}"
            )
        for env_index, env_info in enumerate(infos):
            rows = env_info.get("n", []) if isinstance(env_info, dict) else []
            if len(rows) != agent_count:
                raise RuntimeError(
                    f"Env{env_index} info agent rows={len(rows)}, "
                    f"expected={agent_count}"
                )
            if any(
                "episode_audit_d_min_current" not in row
                for row in rows
                if isinstance(row, dict)
            ):
                raise RuntimeError(
                    f"Env{env_index} did not expose per-step D_min audit data"
                )
        snapshots = training._validate_episode_audit_snapshots(
            parallel_env.get_episode_audit_snapshots(),
            expected_envs=int(args.num_envs),
            expected_agents=agent_count,
        )
    finally:
        parallel_env.close()

    print(
        json.dumps(
            {
                "status": "PASS",
                "num_envs": int(args.num_envs),
                "agent_count": agent_count,
                "observation_batch_shape": list(
                    processed_observations.shape
                ),
                "action_batch_shape": list(actions.shape),
                "logical_gpu_count": logical_gpu_count,
                "ou_resource_device": str(
                    learner.vectorized_ou_noise._resource_device
                ),
                "snapshot_env_ids": [
                    int(snapshot["env_id"]) for snapshot in snapshots
                ],
                "snapshot_schema_version": int(
                    training.EPISODE_AUDIT_SNAPSHOT_SCHEMA_VERSION
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

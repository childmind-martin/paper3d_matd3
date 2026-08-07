#!/usr/bin/env python3
"""Regression tests for experiment provenance, claims, and checkpoint selection."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ablation_dual_q_separated_gradient as ablation
import evaluate_optimized as evaluation
import experiment_runtime_config
import official_eval_with_matched_validation as official
import paper3d_train_optimized as training
import selection_scoring
from tools import build_selector_protocol_batch_spec as selector_batch_spec
from tools import recover_completed_training_postamble as postamble_recovery
from cross_agent_reference_state import (
    MODE_ADAPTIVE_TWIN_HEAD_TAIL,
    MODE_SHARED_TWIN_HEAD_TAIL,
    selector_state_payload,
)
from selector_experiment_protocol import (
    SELECTOR_PROTOCOL_CONFIG_BY_LABEL,
    SELECTOR_PROTOCOL_EXPERIMENT_LABELS,
)
from multiagent.scenarios.obstacle_observation import normalize_obstacle_observation_mode


class SelectorFormalBatchSpecTests(unittest.TestCase):
    @staticmethod
    def _payload():
        payload = {
            "schema_version": selector_batch_spec.BATCH_SPEC_SCHEMA_VERSION,
            "protocol_version": selector_batch_spec.BATCH_PROTOCOL_VERSION,
            "selector_protocol_schema_version": 1,
            "train_seed": 101,
            "train_episodes": 1000,
            "train_num_envs": 4,
            "train_environment_trajectories": 4000,
            "training_episode_length": 2800,
            "episodes": selector_batch_spec.FORMAL_EVAL_EPISODES,
            "episode_length": 3080,
            "episode_length_multiplier": 1.1,
            "eval_noise_seed": 101,
            "require_gpu": True,
            "eval_process_shards": 3,
            "eval_process_workers": 3,
            "eval_shard_episode_parallelism": 4,
            "eval_shard_env_step_threads": 4,
            "positions_file_sha256": "1" * 64,
            "sequence_source_sha256": "2" * 64,
            "environment": {
                "semi_random_terrain": True,
                "use_dynamic_obstacles": True,
                "scenario_seed": 88,
                "terrain_base_seed": 88,
                "post_eval_mode": "shared_match_train_env",
                "post_eval_terrain_family": "train_match",
                "post_eval_position_family": "train_match",
                "position_protocol": "single_fixed_positions_file",
            },
            "models": [
                {
                    "id": f"M{index}",
                    "label": label,
                    "model_variant": "final",
                    "actor_signature_sha1": format(index + 1, "x") * 40,
                    "preliminary_post_eval": {
                        "status": "completed",
                        "mode": "shared_match_train_env",
                        "episodes": 30,
                        "episode_length_multiplier": 1.1,
                        "seed": 10088,
                        "selection_protocol": "fixed",
                        "requested_model_variant": "final",
                        "resolved_model_variant": "final",
                        "selected_model_signature_sha1": (
                            format(index + 1, "x") * 40
                        ),
                        "gpu_required": True,
                        "physical_gpus": 1,
                        "logical_gpus": 1,
                        "results_sha256": "3" * 64,
                        "spec_sha256": "4" * 64,
                    },
                    "training_device": {
                        "require_gpu": True,
                        "physical_gpus": 1,
                        "logical_gpus": 1,
                    },
                    "training_parallelism": {
                        "num_envs": 4,
                        "synchronous_iterations": 1000,
                        "environment_trajectories": 4000,
                        **selector_batch_spec.TRAINING_PARALLELISM_CONTRACT,
                    },
                }
                for index, label in enumerate(
                    SELECTOR_PROTOCOL_EXPERIMENT_LABELS
                )
            ],
            "modes": list(selector_batch_spec.FORMAL_MODES),
            "sequences": {
                key: [value] * selector_batch_spec.FORMAL_EVAL_EPISODES
                for key, value in (
                    ("terrain_complexity_level", 3),
                    ("terrain_seed", 88),
                    ("terrain_variant_seed", 1001),
                    ("obstacle_seed", 2001),
                )
            },
        }
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        return payload

    def test_frozen_matrix_is_exactly_four_by_four_by_thirty(self):
        payload = self._payload()
        self.assertEqual(
            selector_batch_spec.validate_batch_spec(
                payload,
                require_paths=False,
            ),
            [],
        )
        self.assertEqual(len(payload["models"]), 4)
        self.assertEqual(len(payload["modes"]), 4)
        self.assertEqual(payload["episodes"], 30)

    def test_environment_or_content_drift_invalidates_batch_spec(self):
        payload = self._payload()
        payload["environment"]["use_dynamic_obstacles"] = False
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertTrue(
            any("use_dynamic_obstacles" in error for error in errors)
        )
        self.assertIn("content_sha256 mismatch", errors)

    def test_episode_length_must_match_frozen_training_length(self):
        payload = self._payload()
        payload["episode_length"] = 3079
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertTrue(
            any("training_episode_length" in error for error in errors)
        )

    def test_input_content_hashes_are_required(self):
        payload = self._payload()
        payload["positions_file_sha256"] = ""
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertIn(
            "positions_file_sha256 is not a SHA-256 digest",
            errors,
        )

    def test_formal_models_must_record_training_gpu(self):
        payload = self._payload()
        payload["models"][3]["training_device"]["logical_gpus"] = 0
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertTrue(
            any("no recorded training GPU" in error for error in errors)
        )

    def test_formal_models_must_prove_four_training_environments(self):
        payload = self._payload()
        payload["models"][1]["training_parallelism"]["num_envs"] = 1
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertTrue(
            any(
                "training_parallelism.num_envs" in error
                for error in errors
            )
        )

    def test_training_trajectory_count_is_iterations_times_environments(self):
        payload = self._payload()
        payload["train_environment_trajectories"] = 1000
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertIn(
            "train_environment_trajectories must equal "
            "train_episodes * train_num_envs",
            errors,
        )

    def test_formal_models_must_have_distinct_actor_signatures(self):
        payload = self._payload()
        payload["models"][3]["actor_signature_sha1"] = payload["models"][2][
            "actor_signature_sha1"
        ]
        payload["models"][3]["preliminary_post_eval"][
            "selected_model_signature_sha1"
        ] = payload["models"][3]["actor_signature_sha1"]
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertTrue(
            any("four distinct actor signatures" in error for error in errors)
        )

    def test_effective_training_args_must_match_selector_protocol(self):
        expected_env = SELECTOR_PROTOCOL_CONFIG_BY_LABEL[
            SELECTOR_PROTOCOL_EXPERIMENT_LABELS[3]
        ]["env"]
        run_args = {}
        for env_key, raw_value in expected_env.items():
            if env_key in selector_batch_spec._PROTOCOL_MANIFEST_ONLY_ENV_KEYS:
                continue
            arg_key = "algo" if env_key == "ALGORITHM" else env_key.lower()
            if env_key in selector_batch_spec._PROTOCOL_RESULT_BOOL_ENV_KEYS:
                run_args[arg_key] = selector_batch_spec._strict_bool(raw_value)
            elif env_key in selector_batch_spec._PROTOCOL_RESULT_INT_ENV_KEYS:
                run_args[arg_key] = int(raw_value)
            elif env_key in selector_batch_spec._PROTOCOL_RESULT_STRING_ENV_KEYS:
                run_args[arg_key] = str(raw_value)
            else:
                run_args[arg_key] = float(raw_value)
        self.assertEqual(
            selector_batch_spec._selector_result_arg_errors(
                run_args,
                expected_env,
            ),
            [],
        )
        run_args["cross_agent_reference_target_semantics"] = "legacy"
        errors = selector_batch_spec._selector_result_arg_errors(
            run_args,
            expected_env,
        )
        self.assertTrue(
            any(
                "cross_agent_reference_target_semantics" in error
                for error in errors
            )
        )

    def test_cross_reference_activity_accepts_explicit_and_legacy_evidence(self):
        explicit = {
            "episode": 50,
            "cross_ref_active": 1.0,
            "cross_ref_valid_ratio": 0.25,
            "cross_ref_loss": 0.1,
        }
        legacy = {
            "episode": 50,
            "cross_ref_valid_ratio": 0.25,
            "cross_ref_loss": 0.1,
            "cross_ref_actor_weight": 1.0,
        }
        active, eligible = selector_batch_spec._cross_reference_activity_rows(
            [explicit, legacy],
            start_episode=50,
        )
        self.assertEqual(active, [explicit, legacy])
        self.assertEqual(eligible, [explicit, legacy])

    def test_cross_reference_activity_never_overrides_explicit_inactive(self):
        explicit_inactive = {
            "episode": 50,
            "cross_ref_active": 0.0,
            "cross_ref_valid_ratio": 0.25,
            "cross_ref_loss": 0.1,
            "cross_ref_actor_weight": 1.0,
        }
        legacy_before_start = {
            "episode": 49,
            "cross_ref_valid_ratio": 0.25,
            "cross_ref_loss": 0.1,
            "cross_ref_actor_weight": 1.0,
        }
        legacy_zero_weight = {
            "episode": 50,
            "cross_ref_valid_ratio": 0.25,
            "cross_ref_loss": 0.1,
            "cross_ref_actor_weight": 0.0,
        }
        active, eligible = selector_batch_spec._cross_reference_activity_rows(
            [
                explicit_inactive,
                legacy_before_start,
                legacy_zero_weight,
            ],
            start_episode=50,
        )
        self.assertEqual(active, [])
        self.assertEqual(eligible, [])

    def test_formal_models_require_completed_gpu_preliminary_post_eval(self):
        payload = self._payload()
        preliminary = payload["models"][2]["preliminary_post_eval"]
        preliminary["status"] = "skipped_no_train_success"
        preliminary["gpu_required"] = False
        payload["content_sha256"] = selector_batch_spec._canonical_sha256(
            payload
        )
        errors = selector_batch_spec.validate_batch_spec(
            payload,
            require_paths=False,
        )
        self.assertTrue(
            any("preliminary_post_eval.status" in error for error in errors)
        )
        self.assertTrue(
            any("preliminary post-eval did not require GPU" in error for error in errors)
        )

    def test_preliminary_shared_spec_is_fixed_final(self):
        shared_spec = {
            "enabled": True,
            "mode": "shared_match_train_env",
            "episodes": 30,
            "episode_length_multiplier": 1.1,
            "seed": 10088,
            "model_variant": "final",
            "selection_protocol": "fixed",
            "requested_model_variant": "final",
            "position_family": "train_match",
            "semi_random_terrain": True,
            "use_dynamic_obstacles": True,
            "use_fixed_positions": True,
            "terrain_seed_sequence": [88] * 30,
            "terrain_variant_seed_sequence": [1001] * 30,
            "obstacle_seed_sequence": [2001] * 30,
        }
        selector_batch_spec._validate_preliminary_shared_spec(shared_spec)
        shared_spec["selection_protocol"] = "matched_validation"
        with self.assertRaisesRegex(ValueError, "selection_protocol"):
            selector_batch_spec._validate_preliminary_shared_spec(shared_spec)

    def test_preliminary_episode_details_must_match_frozen_sequences(self):
        embedded_spec = {
            "terrain_seed_sequence": [88] * 30,
            "terrain_variant_seed_sequence": list(range(1001, 1031)),
            "obstacle_seed_sequence": list(range(2001, 2031)),
        }
        details = [
            {
                "episode": episode,
                "terrain_seed": embedded_spec["terrain_seed_sequence"][
                    episode
                ],
                "terrain_variant_seed": embedded_spec[
                    "terrain_variant_seed_sequence"
                ][episode],
                "obstacle_seed": embedded_spec["obstacle_seed_sequence"][
                    episode
                ],
            }
            for episode in range(30)
        ]
        self.assertEqual(
            selector_batch_spec._preliminary_episode_detail_errors(
                details,
                embedded_spec,
            ),
            [],
        )
        details[7]["obstacle_seed"] = -1
        errors = selector_batch_spec._preliminary_episode_detail_errors(
            details,
            embedded_spec,
        )
        self.assertTrue(
            any("obstacle_seed" in error for error in errors)
        )

    def test_preliminary_post_eval_inherits_and_validates_gpu_requirement(self):
        self.assertIn(
            "MATD3_REQUIRE_GPU",
            ablation.POST_EVAL_LAUNCH_ENV_KEYS,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "evaluation_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "episodes": 1,
                        "summary": {
                            "team_success_rate": 0.0,
                            "avg_collision_count": 0.0,
                            "avg_team_total_path_length": 0.0,
                        },
                        "evaluation_setup": {},
                    }
                ),
                encoding="utf-8",
            )
            errors = ablation._validate_post_eval_results(
                result_path,
                {
                    "episodes": 1,
                    "artifact_policy": {},
                },
                expected_runtime_env={"MATD3_REQUIRE_GPU": "1"},
            )
        self.assertTrue(
            any("eval_device" in error for error in errors)
        )


class SelectionScoringTests(unittest.TestCase):
    @staticmethod
    def _summary(agent_success_rates):
        return {
            "team_success_rate": 0.0,
            "agent_success_rates": list(agent_success_rates),
            "all_reached_without_safe_team_success_rate": 0.0,
            "collision_free_rate": 1.0,
            "avg_collision_count": 0.0,
            "avg_team_final_goal_distance": 10.0,
            "avg_team_total_path_length": 20.0,
        }

    def test_quantized_tie_uses_next_real_metric(self):
        best_score = selection_scoring.score_summary(self._summary([0.0, 0.25, 0.05]))
        checkpoint_score = selection_scoring.score_summary(self._summary([0.1, 0.2, 0.0]))
        self.assertLess(best_score[1], checkpoint_score[1])
        selected = selection_scoring.select_best_candidate(
            [
                {"candidate_alias": "best", "order": 0, "score": best_score},
                {"candidate_alias": "checkpoint", "order": 1, "score": checkpoint_score},
            ]
        )
        self.assertEqual(selected["candidate_alias"], "best")
        self.assertEqual(
            selection_scoring.comparison_score(best_score)[1],
            selection_scoring.comparison_score(checkpoint_score)[1],
        )

    def test_both_selectors_use_identical_schema_and_score(self):
        summary = self._summary([0.0, 0.25, 0.05])
        self.assertEqual(official._score_summary(summary), ablation._score_post_eval_summary(summary))
        self.assertEqual(
            official._selection_score_schema(),
            ablation._post_eval_selection_score_schema(),
        )

    def test_non_finite_metric_is_penalized(self):
        score = selection_scoring.score_summary(
            {
                **self._summary([0.0, 0.0, 0.0]),
                "team_success_rate": float("nan"),
                "avg_collision_count": float("inf"),
            }
        )
        self.assertEqual(score[0], -1.0)
        self.assertEqual(score[7], -1e12)

    def test_checkpoint_selection_requires_every_ordering_metric(self):
        summary = self._summary([0.1, 0.2, 0.3])
        self.assertEqual(selection_scoring.selection_summary_errors(summary), [])
        summary.pop("all_reached_without_safe_team_success_rate")
        errors = selection_scoring.selection_summary_errors(summary)
        self.assertTrue(any("all_reached_without" in item for item in errors))


class ClaimsTests(unittest.TestCase):
    def test_custom_group_core_claims_are_not_applicable(self):
        report = ablation._evaluate_claims([], {"custom_group_b_label"})
        self.assertIsNone(report["required_pass"])
        self.assertEqual(report["required_status"], "not_applicable")
        self.assertEqual(report["required_applicable_count"], 0)

    def test_partial_core_comparison_is_incomplete(self):
        series = [
            {"label": "matd3_dual_q", "metrics": {"episode_rewards": [1.0, 2.0]}},
            {"label": "matd3_separated_gradient", "metrics": {"episode_rewards": [2.0, 3.0]}},
        ]
        report = ablation._evaluate_claims(
            series,
            {"matd3_dual_q", "matd3_separated_gradient"},
        )
        self.assertIs(report["required_pass"], False)
        self.assertEqual(report["required_status"], "incomplete")
        self.assertEqual(report["required_skipped_count"], 1)


class ManifestIntegrityTests(unittest.TestCase):
    @staticmethod
    def _manifest():
        return {
            "version": 1,
            "argv": [
                "--noise-scale", "0.30",
                "--noise-decay", "0.98",
                "--noise-decay-steps", "30000",
                "--noise-staircase", "1",
                "--noise-decay-enabled", "1",
                "--noise-min", "0.10",
                "--random-action-prob", "0.01",
                "--learning-rate-actor", "0.0003",
                "--learning-rate-critic", "0.0005",
            ],
            "exec_env": {"SEED": "101", "NUM_ENVS": "4"},
            "meta": {
                "label": "example",
                "exp_name_base": "example__seed101",
                "exp_name_with_timestamp": "example__seed101_20260721_120000",
                "seed": "101",
                "episodes": 1000,
                "batch_size": 1024,
                "num_envs": 4,
            },
        }

    def test_manifest_fingerprint_detects_tampering(self):
        manifest = ablation._stamp_resolved_manifest(self._manifest())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            ablation._save_resolved_manifest_exclusive(path, manifest)
            loaded = ablation._load_manifest(path)
            self.assertEqual(
                loaded["meta"][ablation.RESOLVED_TRAINING_MANIFEST_HASH_KEY],
                manifest["meta"][ablation.RESOLVED_TRAINING_MANIFEST_HASH_KEY],
            )
            tampered = copy.deepcopy(loaded)
            tampered["argv"][1] = "0.99"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "内容指纹不匹配"):
                ablation._load_manifest(path)

    def test_result_manifest_provenance_rejects_exp_and_noise_mismatch(self):
        manifest = ablation._stamp_resolved_manifest(self._manifest())
        payload = {
            "training_manifest_sha256": manifest["meta"][ablation.RESOLVED_TRAINING_MANIFEST_HASH_KEY],
            "training_hyperparameters": {"noise_decay": 0.9995, "noise_min": 0.003},
        }
        run_args = {
            "exp_name": "example__seed101_20260721_130000",
            "noise_scale": 0.30,
            "noise_decay_steps": 30000,
            "noise_staircase": True,
            "noise_decay_enabled": True,
            "random_action_prob": 0.01,
            "learning_rate_actor": 0.0003,
            "learning_rate_critic": 0.0005,
        }
        errors = ablation._validate_result_manifest_provenance(run_args, payload, manifest)
        self.assertTrue(any("exp_name" in item for item in errors))
        self.assertTrue(any("noise_decay" in item for item in errors))
        self.assertTrue(any("noise_min" in item for item in errors))

    def test_legacy_mutated_runtime_noise_scale_is_not_treated_as_initial_value(self):
        manifest = ablation._stamp_resolved_manifest(self._manifest())
        payload = {
            "training_manifest_sha256": manifest["meta"][ablation.RESOLVED_TRAINING_MANIFEST_HASH_KEY],
        }
        run_args = {
            "exp_name": manifest["meta"]["exp_name_with_timestamp"],
            "noise_scale": 0.11359936702251434,
            "noise_decay": 0.98,
            "noise_decay_steps": 30000,
            "noise_staircase": True,
            "noise_decay_enabled": True,
            "noise_min": 0.10,
            "random_action_prob": 0.01,
            "learning_rate_actor": 0.0003,
            "learning_rate_critic": 0.0005,
        }
        errors = ablation._validate_result_manifest_provenance(run_args, payload, manifest)
        self.assertEqual(errors, [])

    def test_environment_schema_boundary(self):
        schema1_keys = ablation._required_training_environment_keys({"schema_version": 1})
        schema2_keys = ablation._required_training_environment_keys({"schema_version": 2})
        self.assertNotIn("obstacle_observation_mode", schema1_keys)
        self.assertIn("obstacle_observation_mode", schema2_keys)

    def test_manifest_identity_rejects_parallel_environment_drift(self):
        manifest = ablation._stamp_resolved_manifest(self._manifest())
        with self.assertRaisesRegex(RuntimeError, "num_envs"):
            ablation._validate_resolved_manifest_identity(
                manifest,
                Path("/tmp/manifest.json"),
                label="example",
                exp_name_base="example__seed101",
                seed=101,
                episodes=1000,
                batch_size=1024,
                num_envs=1,
            )


class ParallelTrainingAuditTests(unittest.TestCase):
    @staticmethod
    def _fake_env(env_id=0):
        agents = []
        for agent_index in range(3):
            goal = SimpleNamespace(
                state=SimpleNamespace(
                    p_pos=np.asarray(
                        [10.0 + agent_index, 0.0, 5.0],
                        dtype=np.float32,
                    )
                )
            )
            agents.append(
                SimpleNamespace(
                    goal_a=goal,
                    state=SimpleNamespace(
                        p_pos=np.asarray(
                            [9.0 + agent_index, 0.0, 5.0],
                            dtype=np.float32,
                        )
                    ),
                    debug_info={
                        "total_penetration_count": agent_index,
                        "terrain_penetration_count": agent_index,
                        "obstacle_collision_count": 0,
                        "d_min_current": 2.5 - agent_index,
                    },
                    last_min_distance=2.5 - agent_index,
                )
            )
        world = SimpleNamespace(
            env_id=env_id,
            policy_agents=agents,
            _episode_agent_reach_flags=[1, 1, 0],
            _episode_agent_safe_flags=[1, 0, 1],
            _episode_agent_success_flags=[1, 0, 0],
            _episode_team_success_flag=0,
            _episode_success_thr_snapshot=2.0,
            _episode_done_reason=None,
        )
        return SimpleNamespace(world=world, agents=agents, scenario=None)

    def test_episode_audit_snapshot_preserves_authoritative_worker_state(self):
        snapshot0 = training._build_episode_audit_snapshot(
            self._fake_env(env_id=0)
        )
        snapshot1 = training._build_episode_audit_snapshot(
            self._fake_env(env_id=1)
        )
        snapshots = training._validate_episode_audit_snapshots(
            [snapshot0, snapshot1],
            expected_envs=2,
            expected_agents=3,
        )
        self.assertEqual(snapshots[0]["agent_success_flags"], [1, 0, 0])
        self.assertEqual(snapshots[0]["agent_collision_counts"], [0, 1, 2])
        self.assertEqual(snapshots[1]["env_id"], 1)

    def test_episode_audit_snapshot_rejects_inconsistent_success(self):
        env = self._fake_env(env_id=0)
        env.world._episode_agent_success_flags = [1, 1, 0]
        with self.assertRaisesRegex(RuntimeError, "reach AND safe"):
            training._build_episode_audit_snapshot(env)

    def test_ablation_runtime_override_propagates_num_envs(self):
        args = SimpleNamespace(num_envs=4)
        env = ablation._apply_runtime_env_overrides({}, args)
        self.assertEqual(env["NUM_ENVS"], "4")
        command = []
        ablation._append_runtime_override_args(command, args)
        self.assertEqual(command[:2], ["--num-envs", "4"])

    def test_vectorized_observation_processor_preserves_all_four_envs(self):
        processor = training.VectorizedObservationProcessor(
            n_agents=3,
            obs_dim=5,
            use_vectorization=True,
        )
        observations = np.zeros((4, 3, 7), dtype=np.float32)
        for env_index in range(4):
            observations[env_index, :, :] = np.float32(env_index + 1)

        processed = processor.batch_process_observations_vectorized(
            observations
        )

        self.assertEqual(processed.shape, (4, 3, 5))
        np.testing.assert_array_equal(
            processed[:, 0, 0],
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

    def test_nonvectorized_observation_processor_preserves_all_four_envs(self):
        processor = training.ObservationProcessor(n_agents=3, obs_dim=5)
        observations = np.arange(
            4 * 3 * 4,
            dtype=np.float32,
        ).reshape(4, 3, 4)

        processed = processor.batch_process_observations_parallel(
            observations
        )

        self.assertEqual(processed.shape, (4, 3, 5))
        np.testing.assert_array_equal(processed[:, :, :4], observations)
        np.testing.assert_array_equal(
            processed[:, :, 4],
            np.zeros((4, 3), dtype=np.float32),
        )

    def test_training_observation_contract_rejects_lost_environment_axis(self):
        with self.assertRaisesRegex(RuntimeError, "禁止丢弃或广播并行环境轴"):
            training._validate_training_observation_batch(
                np.zeros((1, 3, 81), dtype=np.float32),
                expected_envs=4,
                expected_agents=3,
                expected_obs_dim=81,
                context="regression",
            )

    def test_training_device_provenance_survives_train_main_boundary(self):
        args = SimpleNamespace()
        captured = {
            "python": "/opt/conda/bin/python",
            "cuda_visible_devices": "0",
            "physical_gpus": 1,
            "logical_gpus": 1,
            "physical_gpu_names": ["/physical_device:GPU:0"],
            "logical_gpu_names": ["/device:GPU:0"],
            "configure_gpu": "ok",
            "require_gpu": True,
        }

        recorded = training._record_training_device_info(args, captured)
        recovered = training._require_recorded_training_device_info(args)

        self.assertEqual(recovered, captured)
        self.assertEqual(recorded, captured)
        self.assertIsNot(recovered, args._training_device_info)

    def test_training_device_provenance_is_not_redetected_after_training(self):
        with self.assertRaisesRegex(RuntimeError, "没有传出本次训练"):
            training._require_recorded_training_device_info(SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "禁止生成完成结果"):
            training._record_training_device_info(
                SimpleNamespace(),
                {
                    "python": "/opt/conda/bin/python",
                    "cuda_visible_devices": "0",
                    "physical_gpus": 1,
                    "logical_gpus": 0,
                    "configure_gpu": "fallback_cpu",
                    "require_gpu": True,
                },
            )

    def test_postamble_recovery_binds_device_to_exact_completed_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs" / "experiment" / "20260724_010203"
            log_dir.mkdir(parents=True)
            launcher = root / "launcher.log"
            launcher.write_text(
                "\n".join(
                    [
                        "[Train Device] python=/old/python | "
                        "CUDA_VISIBLE_DEVICES=7 | physical_gpus=1 | "
                        "logical_gpus=1 | require_gpu=True",
                        "训练进度 [###] 100/100",
                        "可视化输出目录: logs/old/20260723_000000",
                        postamble_recovery.KNOWN_POSTAMBLE_ERROR,
                        "[Train Device] python=/new/python | "
                        "CUDA_VISIBLE_DEVICES=0 | physical_gpus=1 | "
                        "logical_gpus=1 | require_gpu=True",
                        "[MATD3网络] 智能体0 - Actor输入: 81, "
                        "Critic全局状态: 243",
                        "[MATD3网络] 智能体1 - Actor输入: 81, "
                        "Critic全局状态: 243",
                        "[MATD3网络] 智能体2 - Actor输入: 81, "
                        "Critic全局状态: 243",
                        "  - 观察维度: 81",
                        "  - 动作维度: 7",
                        "训练进度 [##################################################] "
                        "100/100",
                        "可视化输出目录: "
                        "logs/experiment/20260724_010203",
                        "训练出错: name 'training_device_info' is not defined",
                        postamble_recovery.KNOWN_POSTAMBLE_ERROR,
                    ]
                ),
                encoding="utf-8",
            )

            device, evidence = (
                postamble_recovery._parse_training_device_evidence(
                    launcher.read_text(encoding="utf-8"),
                    launcher_path=launcher,
                    log_dir=log_dir,
                    repo_root=root,
                    expected_episodes=100,
                )
            )

            self.assertEqual(device["python"], "/new/python")
            self.assertEqual(device["cuda_visible_devices"], "0")
            self.assertEqual(device["physical_gpus"], 1)
            self.assertEqual(
                evidence["base_obs_shapes"],
                [81, 81, 81],
            )
            self.assertEqual(
                evidence["base_action_dims"],
                [7, 7, 7],
            )
            self.assertEqual(
                evidence["completed_run_dir"],
                str(log_dir.resolve()),
            )

    def test_postamble_recovery_rejects_run_without_known_terminal_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs" / "experiment" / "20260724_010203"
            log_dir.mkdir(parents=True)
            launcher = root / "launcher.log"
            launcher.write_text(
                "\n".join(
                    [
                        "[Train Device] python=/new/python | "
                        "CUDA_VISIBLE_DEVICES=0 | physical_gpus=1 | "
                        "logical_gpus=1 | require_gpu=True",
                        "训练进度 [###] 100/100",
                        "可视化输出目录: "
                        "logs/experiment/20260724_010203",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "不是已知"):
                postamble_recovery._parse_training_device_evidence(
                    launcher.read_text(encoding="utf-8"),
                    launcher_path=launcher,
                    log_dir=log_dir,
                    repo_root=root,
                    expected_episodes=100,
                )


class ObstacleObservationIntegrityTests(unittest.TestCase):
    def test_alias_is_canonical_and_invalid_mode_fails(self):
        self.assertEqual(normalize_obstacle_observation_mode("risk-lite"), "risk_lite_v2")
        self.assertEqual(normalize_obstacle_observation_mode("nearest3"), "nearest_surface")
        with self.assertRaises(ValueError):
            normalize_obstacle_observation_mode("unknown_mode")

    def test_explicit_zero_risk_weights_survive_ablation_resolution(self):
        args = SimpleNamespace(
            resolved_scenario_seed=88,
            training_env_sequence_seed=88,
            experiment_group="A",
            use_dynamic_obstacles=False,
            obstacle_observation_mode="risk",
            obstacle_risk_velocity_forward_weight=0.0,
            obstacle_risk_goal_along_weight=0.0,
        )
        setup = ablation._resolve_training_environment_setup(args)
        self.assertEqual(setup["obstacle_observation_mode"], "risk_lite_v2")
        self.assertEqual(setup["obstacle_risk_velocity_forward_weight"], 0.0)
        self.assertEqual(setup["obstacle_risk_goal_along_weight"], 0.0)

    def test_structured_training_environment_wins_over_stale_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir) / "experiment"
            model_variant = model_root / "final"
            model_variant.mkdir(parents=True)
            payload = {
                "args": {
                    "scenario": "paper3d_terrain_vectorized",
                    "algo": "matd3",
                    "terrain_seed": 67,
                    "obstacle_observation_mode": "nearest_surface",
                    "obstacle_risk_velocity_forward_weight": 4.0,
                    "obstacle_risk_goal_along_weight": 3.0,
                },
                "training_environment": {
                    "schema_version": 2,
                    "terrain_seed": 88,
                    "obstacle_observation_mode": "risk",
                    "obstacle_risk_velocity_forward_weight": 0.0,
                    "obstacle_risk_goal_along_weight": 0.0,
                },
            }
            (model_root / "results.json").write_text(json.dumps(payload), encoding="utf-8")
            snapshot = evaluation._load_training_alignment_snapshot(str(model_variant))
            self.assertEqual(snapshot["terrain_seed"], 88)
            self.assertEqual(snapshot["obstacle_observation_mode"], "risk_lite_v2")
            self.assertEqual(snapshot["obstacle_risk_velocity_forward_weight"], 0.0)
            self.assertEqual(snapshot["obstacle_risk_goal_along_weight"], 0.0)

    def test_training_runtime_reward_fields_reach_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir) / "experiment"
            model_variant = model_root / "final"
            model_variant.mkdir(parents=True)
            payload = {
                "args": {
                    "scenario": "paper3d_terrain_vectorized",
                    "algo": "matd3",
                    "unsafe_arrival_penalty": 6500.0,
                    "terminal_failure_penalty_base": 80.0,
                    "team_progress_bottleneck_only": True,
                    "reward_terminal_order_fix": False,
                    "agent_size": 0.75,
                    "use_quadrotor_dynamics": True,
                    "start_altitude_offset": 12.0,
                    "goal_altitude": 25.0,
                }
            }
            (model_root / "results.json").write_text(json.dumps(payload), encoding="utf-8")
            snapshot = evaluation._load_training_alignment_snapshot(str(model_variant))
            args = SimpleNamespace(
                obstacle_risk_velocity_forward_weight=0.0,
                obstacle_risk_goal_along_weight=0.0,
            )
            evaluation._apply_training_alignment_to_args(args, snapshot, quiet=True)
            with mock.patch.dict(os.environ, {}, clear=True):
                evaluation._apply_runtime_env_overrides_from_args(args)
                self.assertEqual(os.environ["UNSAFE_ARRIVAL_PENALTY"], "6500.0")
                self.assertEqual(os.environ["TERMINAL_FAILURE_PENALTY_BASE"], "80.0")
                self.assertEqual(os.environ["TEAM_PROGRESS_BOTTLENECK_ONLY"], "1")
                self.assertEqual(os.environ["REWARD_TERMINAL_ORDER_FIX"], "0")
                self.assertEqual(os.environ["AGENT_SIZE"], "0.75")
                self.assertEqual(os.environ["USE_QUADROTOR_DYNAMICS"], "1")
                self.assertEqual(os.environ["START_ALTITUDE_OFFSET"], "12.0")
                self.assertEqual(os.environ["GOAL_ALTITUDE"], "25.0")
                self.assertEqual(os.environ["MOUNTAIN_MIN_DISTANCE"], "55")
                self.assertEqual(os.environ["MOUNTAIN_MARGIN"], "20")
                self.assertEqual(os.environ["OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT"], "0.0")
                self.assertEqual(os.environ["OBSTACLE_RISK_GOAL_ALONG_WEIGHT"], "0.0")


class RuntimeEnvironmentProvenanceTests(unittest.TestCase):
    def _write_manifest(self, root, exp_name, tree_name="run_a"):
        label, batch_dir = experiment_runtime_config.infer_training_manifest_identity(exp_name)
        manifest_path = (
            Path(root)
            / "ablation_experiments"
            / tree_name
            / "seed_batches"
            / batch_dir
            / "manifests"
            / f"{label}_resolved_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True)
        payload = {
            "version": 1,
            "exec_env": {
                "START_ALTITUDE_OFFSET": "12.0",
                "GOAL_ALTITUDE": "25.0",
                "USE_QUADROTOR_DYNAMICS": "1",
                "MIN_START_GOAL_DIST": "75.0",
                "MAX_START_GOAL_DIST": "120.0",
            },
            "meta": {
                "label": label,
                "exp_name_with_timestamp": exp_name,
            },
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def test_legacy_manifest_reconstructs_actual_values_and_code_defaults(self):
        exp_name = "runtime_case__seed101__batch_groupB_seed101_20260705_163631_20260705_163644"
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = self._write_manifest(temp_dir, exp_name)
            resolved_path = experiment_runtime_config.find_training_runtime_manifest(
                temp_dir, exp_name
            )
            self.assertEqual(resolved_path, expected_path.resolve())
            manifest = experiment_runtime_config.load_training_runtime_manifest(
                resolved_path, exp_name=exp_name
            )
            runtime = experiment_runtime_config.runtime_environment_from_manifest(manifest)
            self.assertEqual(runtime["start_altitude_offset"], 12.0)
            self.assertEqual(runtime["goal_altitude"], 25.0)
            self.assertIs(runtime["use_quadrotor_dynamics"], True)
            self.assertEqual(runtime["min_start_goal_dist"], 75.0)
            self.assertEqual(runtime["max_start_goal_dist"], 120.0)
            self.assertEqual(runtime["mountain_min_distance"], 55)
            self.assertIsInstance(runtime["mountain_min_distance"], int)
            self.assertEqual(runtime["mountain_margin"], 20)
            # This field is absent from the old manifest, so reconstruction must
            # use the scenario's canonical training-time default.
            self.assertEqual(runtime["init_vel_jitter_max"], 0.3)

    def test_manifest_lookup_rejects_ambiguous_identity(self):
        exp_name = "runtime_case__seed101__batch_groupB_seed101_20260705_163631_20260705_163644"
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_manifest(temp_dir, exp_name, "run_a")
            self._write_manifest(temp_dir, exp_name, "run_b")
            with self.assertRaisesRegex(RuntimeError, "匹配不唯一"):
                experiment_runtime_config.find_training_runtime_manifest(temp_dir, exp_name)

    def test_manifest_loader_rejects_wrong_experiment(self):
        exp_name = "runtime_case__seed101__batch_groupB_seed101_20260705_163631_20260705_163644"
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self._write_manifest(temp_dir, exp_name)
            with self.assertRaisesRegex(RuntimeError, "身份不一致"):
                experiment_runtime_config.load_training_runtime_manifest(
                    manifest_path,
                    exp_name="different_case__seed101__batch_groupB_seed101_20260705_163631_20260705_163644",
                )

    def test_batch_reset_records_actual_obstacle_seed(self):
        scenario = SimpleNamespace(current_episode_obstacle_seed=None)

        class FakeEnv:
            agents = []

            def reset(self):
                scenario.current_episode_obstacle_seed = 123456
                return []

        evaluator = evaluation.ModelEvaluator.__new__(evaluation.ModelEvaluator)
        evaluator.args = SimpleNamespace(
            success_distance_threshold=2.0,
            episode_length=8,
            disable_visualization=True,
            disable_gif=True,
        )
        evaluator._process_observations_for_eval = lambda observations: observations
        context = SimpleNamespace(
            episode_idx=0,
            env=FakeEnv(),
            world=SimpleNamespace(agents=[], dt=0.08),
            scenario=scenario,
            terrain_info={"terrain_seed": 88, "terrain_variant_seed": 99, "obstacle_seed": None},
        )
        evaluator._init_batched_episode_state(context)
        self.assertEqual(context.terrain_info["obstacle_seed"], 123456)

    def test_runtime_reapply_preserves_explicit_episode_terrain_identity(self):
        scenario = SimpleNamespace(
            seed=89,
            terrain_seed=89,
            current_terrain_seed=89,
            terrain_variant_seed=189,
            current_terrain_variant_seed=189,
            terrain_base_seed=88,
            rng=np.random.RandomState(189),
        )
        world = SimpleNamespace(terrain_seed=88)
        args = SimpleNamespace(
            terrain_seed=88,
            terrain_variant_seed=188,
            terrain_base_seed=88,
        )
        rng_state_before = scenario.rng.get_state()

        evaluation._apply_terrain_runtime_params_to_scenario(
            scenario,
            world,
            args,
            preserve_episode_terrain=True,
        )

        self.assertEqual(scenario.seed, 89)
        self.assertEqual(scenario.terrain_seed, 89)
        self.assertEqual(scenario.current_terrain_seed, 89)
        self.assertEqual(scenario.terrain_variant_seed, 189)
        self.assertEqual(scenario.current_terrain_variant_seed, 189)
        self.assertEqual(world.terrain_seed, 89)
        rng_state_after = scenario.rng.get_state()
        self.assertEqual(rng_state_before[0], rng_state_after[0])
        np.testing.assert_array_equal(rng_state_before[1], rng_state_after[1])
        self.assertEqual(rng_state_before[2:], rng_state_after[2:])

    def test_serial_episode_rebuild_requests_terrain_preservation(self):
        scenario = SimpleNamespace(
            seed=88,
            terrain_seed=88,
            current_terrain_seed=88,
            terrain_variant_seed=None,
            current_terrain_variant_seed=None,
            use_semi_random_terrain=True,
            random_terrain=False,
            current_episode_obstacle_seed_override=None,
        )

        def regenerate_terrain(new_seed=None, variant_seed=None):
            scenario.seed = int(new_seed)
            scenario.terrain_seed = int(new_seed)
            scenario.current_terrain_seed = int(new_seed)
            scenario.terrain_variant_seed = variant_seed
            scenario.current_terrain_variant_seed = variant_seed

        scenario.regenerate_terrain = regenerate_terrain
        evaluator = evaluation.ModelEvaluator.__new__(evaluation.ModelEvaluator)
        evaluator.args = SimpleNamespace(random_terrain=False)
        evaluator.scenario = scenario
        evaluator.env = SimpleNamespace(scenario=scenario)
        evaluator._load_episode_positions = mock.Mock()
        evaluator._rebuild_environment = mock.Mock()

        terrain_info = evaluator._prepare_episode_terrain(
            1,
            terrain_seed_sequence=[88, 89],
            terrain_variant_seed_sequence=[188, 189],
            obstacle_seed_sequence=[10088, 10089],
        )

        evaluator._rebuild_environment.assert_called_once_with(
            preserve_episode_terrain=True,
        )
        self.assertEqual(terrain_info["terrain_seed"], 89)
        self.assertEqual(terrain_info["terrain_variant_seed"], 189)
        self.assertEqual(terrain_info["obstacle_seed"], 10089)

    def test_batch_episode_runtime_apply_requests_terrain_preservation(self):
        scenario = SimpleNamespace(
            seed=88,
            terrain_seed=88,
            current_terrain_seed=88,
            terrain_variant_seed=None,
            current_terrain_variant_seed=None,
            use_semi_random_terrain=True,
            random_terrain=False,
            reset_world=lambda world: None,
            reward=lambda agent, world: 0.0,
            observation=lambda agent, world: np.zeros(1),
        )

        def regenerate_terrain(new_seed=None, variant_seed=None):
            scenario.seed = int(new_seed)
            scenario.terrain_seed = int(new_seed)
            scenario.current_terrain_seed = int(new_seed)
            scenario.terrain_variant_seed = variant_seed
            scenario.current_terrain_variant_seed = variant_seed

        scenario.regenerate_terrain = regenerate_terrain
        scenario.make_world = lambda: SimpleNamespace(agents=[])

        evaluator = evaluation.ModelEvaluator.__new__(evaluation.ModelEvaluator)
        evaluator.args = SimpleNamespace(
            scenario_name="paper3d_terrain_vectorized",
            random_terrain=False,
            terrain_seed=88,
            terrain_base_seed=88,
        )
        evaluator._apply_eval_context_runtime_params = mock.Mock()
        evaluator._load_episode_positions_into_scenario = mock.Mock()

        with (
            mock.patch.object(evaluation, "load_scenario_module", return_value=scenario),
            mock.patch.object(evaluation, "_apply_runtime_env_overrides_from_args"),
            mock.patch.object(evaluation, "MultiAgentEnv", side_effect=lambda *a, **k: SimpleNamespace(scenario=scenario)),
        ):
            context = evaluator._build_episode_eval_context(
                1,
                3,
                terrain_seed_sequence=[88, 89],
                terrain_variant_seed_sequence=[188, 189],
                obstacle_seed_sequence=[10088, 10089],
            )

        evaluator._apply_eval_context_runtime_params.assert_called_once_with(
            scenario,
            context.world,
            context.env,
            preserve_episode_terrain=True,
        )
        self.assertEqual(context.terrain_info["terrain_seed"], 89)
        self.assertEqual(context.terrain_info["terrain_variant_seed"], 189)
        self.assertEqual(context.terrain_info["obstacle_seed"], 10089)

    def test_episode_complexity_refreshes_derived_scenario_parameters(self):
        class FakeScenario:
            terrain_complexity_level = 3
            num_mountains = 7
            num_obstacles = 12

            def _setup_complexity_parameters(self):
                by_level = {
                    1: (5, 4),
                    2: (6, 8),
                    3: (7, 12),
                    4: (8, 16),
                }
                self.num_mountains, self.num_obstacles = by_level[
                    self.terrain_complexity_level
                ]

        scenario = FakeScenario()
        applied = evaluation._set_scenario_terrain_complexity(scenario, 4)
        self.assertEqual(applied, 4)
        self.assertEqual(scenario.terrain_complexity_level, 4)
        self.assertEqual(scenario.num_mountains, 8)
        self.assertEqual(scenario.num_obstacles, 16)

    def test_variant_seed_is_reported_only_for_semi_random_terrain(self):
        scenario = SimpleNamespace(
            use_semi_random_terrain=False,
            terrain_variant_seed=88,
            current_terrain_variant_seed=88,
        )
        self.assertIsNone(evaluation._scenario_terrain_variant_seed(scenario))
        scenario.use_semi_random_terrain = True
        self.assertEqual(evaluation._scenario_terrain_variant_seed(scenario), 88)


class SelectorProtocolRegistryTests(unittest.TestCase):
    def test_official_selector_runner_uses_two_models_by_four_envs(self):
        runner = (
            Path(__file__).resolve().parent
            / "run_selector_m0_m3_full_gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("readonly TRAIN_NUM_ENVS=4", runner)
        self.assertIn("readonly TRAIN_MODEL_MAX_PARALLEL=2", runner)
        self.assertIn(
            '--max-parallel "$TRAIN_MODEL_MAX_PARALLEL"',
            runner,
        )
        self.assertIn(
            '--experiment-max-parallel "$TRAIN_MODEL_MAX_PARALLEL"',
            runner,
        )
        self.assertIn(
            '--worker-launch-stagger-seconds '
            '"$TRAIN_WORKER_LAUNCH_STAGGER_SECONDS"',
            runner,
        )

    def test_active_registry_contains_exact_m0_m3_protocol(self):
        active_labels = [
            config["label"]
            for config in ablation.EXPERIMENT_CONFIGS
            if config["label"] in SELECTOR_PROTOCOL_EXPERIMENT_LABELS
        ]
        self.assertEqual(active_labels, list(SELECTOR_PROTOCOL_EXPERIMENT_LABELS))
        registry_labels = {
            config["label"] for config in ablation.EXPERIMENT_CONFIGS
        }
        self.assertTrue(
            registry_labels.isdisjoint(
                ablation.RETIRED_CROSS_AGENT_REFERENCE_LABELS
            )
        )
        self.assertEqual(
            set(ablation.CROSS_AGENT_REFERENCE_LABELS),
            set(SELECTOR_PROTOCOL_EXPERIMENT_LABELS),
        )

    def test_m0_m3_configs_differ_only_by_declared_mechanism(self):
        configs = [
            SELECTOR_PROTOCOL_CONFIG_BY_LABEL[label]["env"]
            for label in SELECTOR_PROTOCOL_EXPERIMENT_LABELS
        ]
        mechanism_keys = {
            "CROSS_AGENT_REFERENCE_TARGET_SEMANTICS",
            "CROSS_AGENT_REFERENCE_SELECTOR_MODE",
            "CROSS_AGENT_REFERENCE_SELECTOR_ENABLED",
        }
        common_projections = [
            {
                key: value
                for key, value in config.items()
                if key not in mechanism_keys
            }
            for config in configs
        ]
        self.assertTrue(
            all(
                projection == common_projections[0]
                for projection in common_projections[1:]
            )
        )
        self.assertEqual(
            [
                (
                    config["CROSS_AGENT_REFERENCE_TARGET_SEMANTICS"],
                    config["CROSS_AGENT_REFERENCE_SELECTOR_MODE"],
                    config["CROSS_AGENT_REFERENCE_SELECTOR_ENABLED"],
                )
                for config in configs
            ],
            [
                ("legacy", "hard", "0"),
                ("split_raw_head_corrected_tail", "hard", "0"),
                (
                    "split_raw_head_corrected_tail",
                    "adaptive_twin_advantage_head_tail",
                    "0",
                ),
                (
                    "split_raw_head_corrected_tail",
                    "shared_twin_advantage_head_tail",
                    "1",
                ),
            ],
        )
        for config in configs:
            self.assertEqual(config["MATD3_REQUIRE_GPU"], "1")
            self.assertEqual(
                config["CROSS_AGENT_REFERENCE_START_EPISODE"],
                "50",
            )
            self.assertEqual(
                config["CROSS_AGENT_REFERENCE_ACTOR_START_EPISODE"],
                "50",
            )
            self.assertEqual(
                config["CROSS_AGENT_REFERENCE_ACTOR_RAMP_EPISODES"],
                "0",
            )
            self.assertEqual(
                config["CROSS_AGENT_REFERENCE_UPDATE_INTERVAL"],
                "1",
            )
            self.assertEqual(
                config["CROSS_AGENT_REFERENCE_PAIRS_PER_AGENT"],
                "0",
            )


class ResumeIntegrityTests(unittest.TestCase):
    @staticmethod
    def _write_complete_matd3_unit(root: Path, name: str = "unit", episodes: int = 4) -> Path:
        model_root = root / "models" / name
        final_dir = model_root / "final"
        final_dir.mkdir(parents=True)
        for agent_index in range(3):
            for filename in (
                f"actor_{agent_index}.weights.h5",
                f"critic1_{agent_index}.weights.h5",
                f"critic2_{agent_index}.weights.h5",
            ):
                (final_dir / filename).write_bytes(b"weights")
        payload = {
            "episodes": episodes,
            "rewards": [float(index) for index in range(episodes)],
            "training_environment": {"schema_version": 2, "terrain_seed": 88},
            "args": {
                "algo": "matd3",
                "exp_name": name,
                "train_episodes": episodes,
                "seed": 101,
            },
        }
        (model_root / "results.json").write_text(json.dumps(payload), encoding="utf-8")
        return model_root

    def test_whole_unit_completion_requires_final_results_and_final_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_matd3_unit(root)
            self.assertEqual(
                experiment_runtime_config.training_unit_completion_errors(
                    complete,
                    4,
                    repo_root=root,
                    expected_agents=3,
                    expected_seed=101,
                ),
                [],
            )

            checkpoint_only = root / "models" / "checkpoint_only"
            checkpoint_dir = checkpoint_only / "checkpoint"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "checkpoint_state.json").write_text(
                json.dumps({"episode": 4, "episode_rewards": [1.0] * 4}),
                encoding="utf-8",
            )
            errors = experiment_runtime_config.training_unit_completion_errors(
                checkpoint_only,
                4,
                repo_root=root,
                expected_agents=3,
            )
            self.assertTrue(any("final results.json" in item for item in errors))
            self.assertTrue(any("final model directory" in item for item in errors))

    def test_whole_unit_completion_rejects_wrong_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_matd3_unit(root)
            errors = experiment_runtime_config.training_unit_completion_errors(
                complete,
                4,
                repo_root=root,
                expected_agents=3,
                expected_seed=202,
            )
            self.assertTrue(any("results.args.seed=101, expected=202" in item for item in errors))

    def test_whole_unit_completion_proves_parallel_environment_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_matd3_unit(root)
            result_path = complete / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["args"]["num_envs"] = 4
            result["training_parallelism"] = {
                "num_envs": 4,
                "synchronous_iterations": 4,
                "environment_trajectories": 16,
                "reward_aggregation": "equal_mean_across_environments",
                "success_aggregation": "equal_mean_across_environments",
                "worker_seed_derivation": "base_seed_plus_env_id_times_100003",
                "episode_audit_snapshot_schema_version": 1,
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(
                experiment_runtime_config.training_unit_completion_errors(
                    complete,
                    4,
                    repo_root=root,
                    expected_agents=3,
                    expected_seed=101,
                    expected_num_envs=4,
                ),
                [],
            )

            result["training_parallelism"]["environment_trajectories"] = 4
            result_path.write_text(json.dumps(result), encoding="utf-8")
            errors = experiment_runtime_config.training_unit_completion_errors(
                complete,
                4,
                repo_root=root,
                expected_agents=3,
                expected_seed=101,
                expected_num_envs=4,
            )
            self.assertTrue(
                any(
                    "training_parallelism.environment_trajectories=4, expected=16"
                    in item
                    for item in errors
                )
            )

    def test_gpu_required_unit_needs_recorded_training_device(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_matd3_unit(root)
            errors = (
                experiment_runtime_config.training_unit_completion_errors(
                    complete,
                    4,
                    repo_root=root,
                    expected_agents=3,
                    expected_seed=101,
                    require_gpu=True,
                )
            )
            self.assertTrue(
                any("training_device is missing" in item for item in errors)
            )

            result_path = complete / "results.json"
            result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            result["training_device"] = {
                "require_gpu": True,
                "physical_gpus": 1,
                "logical_gpus": 1,
            }
            result_path.write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            self.assertEqual(
                experiment_runtime_config.training_unit_completion_errors(
                    complete,
                    4,
                    repo_root=root,
                    expected_agents=3,
                    expected_seed=101,
                    require_gpu=True,
                ),
                [],
            )

    def test_adaptive_unit_completion_requires_full_selector_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = self._write_complete_matd3_unit(
                root,
                name="adaptive_unit",
            )
            result_path = model_root / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["args"].update(
                {
                    "cross_agent_reference_selector_mode": (
                        MODE_ADAPTIVE_TWIN_HEAD_TAIL
                    ),
                    "cross_agent_reference_enabled": True,
                    "matd3_use_dual_q": True,
                    "matd3_use_separated_gradient": True,
                    "matd3_use_hybrid_actor_objective": False,
                    "matd3_action_semantics_mode": "dual",
                    "matd3_reconstruct_corrected_target": True,
                    "cross_agent_reference_use_clean_label": False,
                    "cross_agent_reference_target_semantics": (
                        "split_raw_head_corrected_tail"
                    ),
                    "cross_agent_reference_exclude_random": True,
                    "cross_agent_reference_quality_gate": True,
                    "cross_agent_reference_gate_mode": "agent_quality",
                    "cross_agent_reference_selector_enabled": False,
                    "cross_agent_reference_advantage_ema_decay": 0.99,
                    "cross_agent_reference_advantage_epsilon": 1e-6,
                    "cross_agent_reference_selector_adv_clip": 5.0,
                }
            )
            result_path.write_text(json.dumps(result), encoding="utf-8")
            final_dir = model_root / "final"
            errors = experiment_runtime_config.training_unit_completion_errors(
                model_root,
                4,
                repo_root=root,
                expected_seed=101,
            )
            self.assertTrue(
                any("selector state missing/unreadable" in item for item in errors)
            )

            state = selector_state_payload(
                mode=MODE_ADAPTIVE_TWIN_HEAD_TAIL,
                head_scale=2.0,
                tail_scale=3.0,
                head_initialized=True,
                tail_initialized=True,
                update_count=4,
                ema_decay=0.99,
                epsilon=1e-6,
                advantage_clip=5.0,
                input_dim=None,
            )
            (final_dir / "cross_agent_reference_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
            self.assertEqual(
                experiment_runtime_config.training_unit_completion_errors(
                    model_root,
                    4,
                    repo_root=root,
                    expected_seed=101,
                ),
                [],
            )

    def test_shared_selector_unit_has_one_shared_weight_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = self._write_complete_matd3_unit(
                root,
                name="shared_selector_unit",
            )
            result_path = model_root / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["args"].update(
                {
                    "cross_agent_reference_selector_mode": (
                        MODE_SHARED_TWIN_HEAD_TAIL
                    ),
                    "cross_agent_reference_enabled": True,
                    "matd3_use_dual_q": True,
                    "matd3_use_separated_gradient": True,
                    "matd3_use_hybrid_actor_objective": False,
                    "matd3_action_semantics_mode": "dual",
                    "matd3_reconstruct_corrected_target": True,
                    "cross_agent_reference_use_clean_label": False,
                    "cross_agent_reference_target_semantics": (
                        "split_raw_head_corrected_tail"
                    ),
                    "cross_agent_reference_exclude_random": True,
                    "cross_agent_reference_quality_gate": True,
                    "cross_agent_reference_gate_mode": "agent_quality",
                    "cross_agent_reference_selector_enabled": True,
                    "cross_agent_reference_advantage_ema_decay": 0.99,
                    "cross_agent_reference_advantage_epsilon": 1e-6,
                    "cross_agent_reference_selector_adv_clip": 5.0,
                    "base_obs_shapes": [81, 81, 81],
                    "base_action_dims": [7, 7, 7],
                    "use_pf_feature": True,
                    "pf_feature_dim": 3,
                }
            )
            result_path.write_text(json.dumps(result), encoding="utf-8")
            final_dir = model_root / "final"
            (final_dir / "reference_selector_shared.weights.h5").write_bytes(
                b"shared"
            )
            state = selector_state_payload(
                mode=MODE_SHARED_TWIN_HEAD_TAIL,
                head_scale=2.0,
                tail_scale=3.0,
                head_initialized=True,
                tail_initialized=True,
                update_count=4,
                ema_decay=0.99,
                epsilon=1e-6,
                advantage_clip=5.0,
                input_dim=106,
            )
            (final_dir / "cross_agent_reference_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
            self.assertEqual(
                experiment_runtime_config.training_unit_completion_errors(
                    model_root,
                    4,
                    repo_root=root,
                    expected_seed=101,
                ),
                [],
            )
            (final_dir / "reference_selector_0.weights.h5").write_bytes(
                b"retired"
            )
            errors = experiment_runtime_config.training_unit_completion_errors(
                model_root,
                4,
                repo_root=root,
                expected_seed=101,
            )
            self.assertTrue(
                any("retired per-agent selector" in item for item in errors)
            )

    def test_mappo_log_result_can_prove_completion_without_model_mirror(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            name = "mappo_unit"
            model_root = root / "models" / name
            final_dir = model_root / "final"
            final_dir.mkdir(parents=True)
            for agent_index in range(3):
                (final_dir / f"actor_{agent_index}.weights.h5").write_bytes(b"actor")
            for filename in ("value_critic.weights.h5", "actor_log_std.npy", "mappo_meta.json"):
                (final_dir / filename).write_bytes(b"mappo")
            log_root = root / "logs" / name
            log_root.mkdir(parents=True)
            (log_root / "results.json").write_text(
                json.dumps(
                    {
                        "episodes": 3,
                        "rewards": [1.0, 2.0, 3.0],
                        "training_environment": {"schema_version": 2},
                        "args": {
                            "algo": "mappo",
                            "exp_name": name,
                            "train_episodes": 3,
                            "seed": 202,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                experiment_runtime_config.training_unit_completion_errors(
                    model_root,
                    3,
                    repo_root=root,
                    expected_agents=3,
                    expected_seed=202,
                ),
                [],
            )

    def test_batch_environment_disables_episode_resume_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "RESUME_MODEL_ENV": "/tmp/partial",
                    "CHECKPOINT_MODEL": "/tmp/partial/checkpoint",
                    "SAVE_INTERVAL": "100",
                },
                clear=False,
            ):
                env = ablation.setup_base_env_vars(
                    Path(temp_dir) / "positions.json",
                    env_isolation="strict",
                    config_mode="strict_ablation",
                    scenario_seed=88,
                )
            self.assertEqual(env["SAVE_TRAINING_RESUME_STATE"], "0")
            self.assertEqual(env["SAVE_INTERVAL"], "0")
            self.assertEqual(env["BATCH_RESUME_POLICY"], ablation.BATCH_RESUME_POLICY)
            self.assertNotIn("RESUME_MODEL_ENV", env)
            self.assertNotIn("CHECKPOINT_MODEL", env)

    def test_post_eval_failure_never_falls_back_to_retraining_complete_unit(self):
        cfg = {"label": "unit", "name": "Unit", "description": "test"}
        args = SimpleNamespace(
            script="/home/tang/matd3/run_optimized.sh",
            reuse=True,
            reuse_only=False,
            logs_root="logs",
            manifest_dir="/tmp/manifests",
            episodes=4,
            num_envs=4,
            resolved_scenario_seed=88,
            batch_seed=101,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "complete_log"
            candidate.mkdir()
            with (
                mock.patch.object(ablation, "_candidate_log_dirs_by_exp_name_base", return_value=[candidate]),
                mock.patch.object(ablation, "find_latest_log_dir", return_value=None),
                mock.patch.object(ablation, "load_metrics", return_value={"episode_rewards": [1.0] * 4}),
                mock.patch.object(ablation, "_validate_loaded_result", return_value=[]),
                mock.patch.object(ablation, "_completed_training_model_errors", return_value=[]),
                mock.patch.object(
                    ablation,
                    "_run_post_training_evaluation",
                    side_effect=RuntimeError("post-eval failed"),
                ),
                mock.patch.object(ablation.subprocess, "run") as run_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "post-eval failed"):
                    ablation.run_experiment(
                        cfg,
                        Path(temp_dir) / "positions.json",
                        args,
                        {},
                        Path(temp_dir) / "batch",
                        {},
                    )
            run_mock.assert_not_called()

    def test_subset_resume_preserves_existing_experiment_inventory(self):
        existing = {
            "episodes": 1000,
            "batch_size": 1024,
            "num_envs": 4,
            "seed": 101,
            "scenario_seed": 88,
            "config_mode": "strict_ablation",
            "experiment_group": "B",
            "positions_file": "/tmp/positions.json",
            "experiments": ["a", "b"],
        }
        requested = dict(existing)
        requested["experiments"] = ["b"]
        merged = ablation._merge_resumed_child_batch_config(
            existing,
            requested,
            ["b"],
            config_path=Path("/tmp/config.json"),
        )
        self.assertEqual(merged["experiments"], ["a", "b"])

    def test_resume_rejects_immutable_seed_change(self):
        existing = {"seed": 101, "experiments": ["a"]}
        requested = {"seed": 202, "experiments": ["a"]}
        with self.assertRaisesRegex(RuntimeError, "seed"):
            ablation._merge_resumed_child_batch_config(
                existing,
                requested,
                ["a"],
                config_path=Path("/tmp/config.json"),
            )

    def test_resume_rejects_immutable_num_envs_change(self):
        existing = {
            "num_envs": 4,
            "seed": 101,
            "experiments": ["a"],
        }
        requested = {
            "num_envs": 1,
            "seed": 101,
            "experiments": ["a"],
        }
        with self.assertRaisesRegex(RuntimeError, "num_envs"):
            ablation._merge_resumed_child_batch_config(
                existing,
                requested,
                ["a"],
                config_path=Path("/tmp/config.json"),
            )

    def test_parent_resume_restores_exact_fixed_final_post_eval_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = Path(temp_dir)
            (batch_dir / "config.json").write_text(
                json.dumps(
                    {
                        "batch_mode": "multi_seed_parent",
                        "episodes": 1000,
                        "batch_size": 1024,
                        "num_envs": 4,
                        "scenario_seed": 88,
                        "seeds": [101],
                        "post_eval_enabled": True,
                        "allow_post_eval_without_train_success": True,
                        "post_eval_mode": "shared_match_train_env",
                        "post_eval_episodes": 30,
                        "post_eval_episode_length_multiplier": 1.1,
                        "post_eval_seed": 10088,
                        "post_eval_selection_protocol": "fixed",
                        "post_eval_model_variant": "final",
                        "post_eval_requested_model_variant": "final",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                ablation.sys,
                "argv",
                ["ablation_dual_q_separated_gradient.py"],
            ):
                args = ablation.parse_args()
            args.parsed_seeds = []
            args.resolved_scenario_seed = 88
            args.resolved_post_eval_seed = ablation._resolve_post_eval_seed(args)
            args.cli_disable_post_eval_specified = False
            args.cli_post_eval_mode_specified = False
            args.cli_post_eval_model_variant_specified = False
            args.cli_post_eval_selection_protocol_specified = False
            args.cli_post_eval_validation_episodes_specified = False
            args.cli_post_eval_validation_seed_specified = False
            args.cli_post_eval_validation_candidates_specified = False
            args.cli_post_eval_episode_length_multiplier_specified = False
            args.cli_max_parallel_specified = False
            args.cli_num_envs_specified = False

            ablation._restore_args_from_parent_batch(args, batch_dir)

            self.assertFalse(args.disable_post_eval)
            self.assertTrue(args.allow_post_eval_without_train_success)
            self.assertEqual(args.post_eval_selection_protocol, "fixed")
            self.assertEqual(args.post_eval_model_variant, "final")
            self.assertEqual(args.post_eval_episodes, 30)
            self.assertEqual(args.post_eval_episode_length_multiplier, 1.1)
            self.assertEqual(args.resolved_post_eval_seed, 10088)
            self.assertEqual(args.num_envs, 4)

    @staticmethod
    def _checkpoint_state(episode=2):
        state = {
            "checkpoint_state_schema_version": training.CHECKPOINT_STATE_SCHEMA_VERSION,
            "episode": episode,
        }
        for field in training._CHECKPOINT_PER_EPISODE_FIELDS:
            state[field] = [0] * episode
        return state

    def test_checkpoint_histories_must_match_completed_episode_count(self):
        state = self._checkpoint_state(episode=2)
        state["team_success_flags"].append(0)
        with self.assertRaisesRegex(ValueError, "history length mismatch"):
            training._resolve_checkpoint_completed_episodes(state)

    def test_legacy_best_checkpoint_off_by_one_is_normalized_only_when_consistent(self):
        state = {"episode": 1}
        for field in training._CHECKPOINT_PER_EPISODE_FIELDS:
            state[field] = [0, 0]
        completed, normalized = training._resolve_checkpoint_completed_episodes(state)
        self.assertEqual(completed, 2)
        self.assertTrue(normalized)
        state["success_flags"].append(0)
        with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
            training._resolve_checkpoint_completed_episodes(state)

    def test_resume_contract_ignores_only_operational_arguments(self):
        base = SimpleNamespace(
            exp_name="run_a",
            train_episodes=100,
            checkpoint=None,
            resume=False,
            save_model=True,
            save_interval=10,
            profiling=False,
            mem_debug=False,
            seed=101,
            agent_max_speed=25.0,
            agent_accel=8.5,
            batch_size=64,
            use_fixed_positions=False,
            training_environment_config={"schema_version": 2, "terrain_seed": 88},
            training_hyperparameters_config={"learning_rate_actor": 3e-4},
        )
        changed_operational = copy.deepcopy(base)
        changed_operational.exp_name = "run_b"
        changed_operational.train_episodes = 200
        changed_operational.resume = True
        changed_operational.checkpoint = "/tmp/checkpoint"
        self.assertEqual(
            training._capture_checkpoint_resume_config(base),
            training._capture_checkpoint_resume_config(changed_operational),
        )

        changed_scientific = copy.deepcopy(base)
        changed_scientific.seed = 202
        saved = training._capture_checkpoint_resume_config(base)
        current = training._capture_checkpoint_resume_config(changed_scientific)
        state = self._checkpoint_state(episode=0)
        state["resume_config"] = saved
        with self.assertRaisesRegex(ValueError, "arguments.seed"):
            training._validate_checkpoint_resume_contract(state, current)

        changed_dynamics = copy.deepcopy(base)
        changed_dynamics.agent_max_speed = 42.5
        current = training._capture_checkpoint_resume_config(changed_dynamics)
        with self.assertRaisesRegex(ValueError, "arguments.agent_max_speed"):
            training._validate_checkpoint_resume_contract(state, current)

    def test_resume_contract_hashes_fixed_positions_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            positions = Path(temp_dir) / "positions.json"
            positions.write_text('{"value": 1}', encoding="utf-8")
            args = SimpleNamespace(
                exp_name="run",
                train_episodes=1,
                checkpoint=None,
                resume=False,
                save_model=True,
                save_interval=1,
                seed=101,
                use_fixed_positions=True,
                positions_file=str(positions),
                training_environment_config={"schema_version": 2},
                training_hyperparameters_config={},
            )
            before = training._capture_checkpoint_resume_config(args)
            positions.write_text('{"value": 2}', encoding="utf-8")
            after = training._capture_checkpoint_resume_config(args)
            differences = training._checkpoint_config_differences(before, after)
            self.assertTrue(any("sha256" in item for item in differences))


class RuntimeScheduleIntegrityTests(unittest.TestCase):
    def test_noise_restart_default_matches_recorded_training_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOISE_RESTART_INTERVAL", None)
            self.assertEqual(
                training._resolve_noise_restart_interval(),
                training.DEFAULT_NOISE_RESTART_INTERVAL,
            )
            self.assertEqual(training.DEFAULT_NOISE_RESTART_INTERVAL, 25)
            os.environ["NOISE_RESTART_INTERVAL"] = "9"
            self.assertEqual(training._resolve_noise_restart_interval(), 9)

    def test_traced_schedule_observes_runtime_base_update(self):
        schedule = training.ClippedExponentialDecaySchedule(
            initial_value=0.1,
            decay_steps=10,
            decay_rate=0.5,
            min_value=0.001,
            staircase=True,
            name="test_lr",
        )

        @training.tf.function
        def read_schedule(step):
            return schedule(step)

        self.assertAlmostEqual(float(read_schedule(training.tf.constant(0)).numpy()), 0.1, places=6)
        schedule.set_base_value(0.04)
        self.assertAlmostEqual(float(read_schedule(training.tf.constant(0)).numpy()), 0.04, places=6)

    def test_zero_noise_scale_is_not_reenabled_by_positive_minimum(self):
        args = SimpleNamespace(
            noise_scale=0.0,
            noise_min=0.003,
            noise_decay=0.9995,
            noise_decay_enabled=True,
            noise_decay_steps=10,
            noise_staircase=True,
        )
        schedule, config = training._build_noise_scale_schedule(args, 10, True)
        self.assertTrue(config["disabled_by_zero_scale"])
        self.assertEqual(config["effective_min_noise_scale"], 0.0)
        self.assertEqual(float(schedule(training.tf.constant(100)).numpy()), 0.0)

    def test_adaptive_noise_updates_once_for_global_controller(self):
        args = SimpleNamespace(noise_scale=0.3, noise_min=0.003)
        schedule, _ = training._build_noise_scale_schedule(
            SimpleNamespace(
                noise_scale=0.3,
                noise_min=0.003,
                noise_decay=0.9995,
                noise_decay_enabled=True,
                noise_decay_steps=100,
                noise_staircase=True,
            ),
            100,
            True,
        )
        controller = SimpleNamespace(
            noise_scale_schedule=schedule,
            total_steps_var=training.tf.Variable(50, dtype=training.tf.int64),
            noise_scale_var=training.tf.Variable(0.3, dtype=training.tf.float32),
            current_ou_noise_std_var=training.tf.Variable(0.3, dtype=training.tf.float32),
            noise_min_var=training.tf.Variable(0.003, dtype=training.tf.float32),
            noise_disabled=False,
            vectorized_ou_noise=None,
            ou_noises=None,
            adaptive_learning={"noise_boost_factor": 1.2, "max_noise_scale": 2.0},
            args=args,
        )
        with mock.patch.dict(
            os.environ,
            {"ADAPTIVE_NOISE_MAX": "0.6", "ADAPTIVE_NOISE_SMOOTH": "0.3"},
            clear=False,
        ):
            old, new, target, cap, beta = training._adaptive_behavior_noise_update(controller, 0.6)
        self.assertAlmostEqual(old, 0.3, places=6)
        self.assertAlmostEqual(target, 0.36, places=6)
        self.assertAlmostEqual(new, 0.318, places=6)
        self.assertAlmostEqual(cap, 0.6, places=6)
        self.assertAlmostEqual(beta, 0.3, places=6)
        self.assertEqual(int(schedule.anchor_step_var.numpy()), 50)

    def test_invalid_noise_decay_fails_before_training(self):
        args = SimpleNamespace(
            noise_scale=0.3,
            noise_min=0.003,
            noise_decay=1.01,
            noise_decay_enabled=True,
            noise_decay_steps=10,
            noise_staircase=True,
        )
        with self.assertRaisesRegex(ValueError, "noise_decay"):
            training._build_noise_scale_schedule(args, 10, True)

    def test_learning_rate_runtime_state_restores_progress_without_adam_counters(self):
        def schedule(name, value):
            return training.ClippedExponentialDecaySchedule(
                initial_value=value,
                decay_steps=2,
                decay_rate=0.9,
                min_value=1e-6,
                staircase=True,
                name=name,
            )

        source_actor_schedule = schedule("source_actor", 0.04)
        source_critic_schedule = schedule("source_critic", 0.08)
        source_agent = {
            "actor_lr_var": training.tf.Variable(0.04, dtype=training.tf.float32),
            "critic_lr_var": training.tf.Variable(0.08, dtype=training.tf.float32),
            "actor_lr_schedule": source_actor_schedule,
            "critic_lr_schedule": source_critic_schedule,
            "actor_optimizer": training.tf.keras.optimizers.Adam(source_actor_schedule),
            "critic1_optimizer": training.tf.keras.optimizers.Adam(source_critic_schedule),
            "critic2_optimizer": training.tf.keras.optimizers.Adam(source_critic_schedule),
        }
        source_agent["actor_optimizer"].iterations.assign(3)
        source_agent["critic1_optimizer"].iterations.assign(7)
        source_agent["critic2_optimizer"].iterations.assign(7)
        source_actor_effective = float(
            source_actor_schedule(source_agent["actor_optimizer"].iterations).numpy()
        )
        source_critic_effective = float(
            source_critic_schedule(source_agent["critic1_optimizer"].iterations).numpy()
        )
        saved = training._capture_learning_rate_runtime_state(
            SimpleNamespace(agents=[source_agent])
        )

        target_actor_schedule = schedule("target_actor", 0.5)
        target_critic_schedule = schedule("target_critic", 0.6)
        target_agent = {
            "actor_lr_var": training.tf.Variable(0.5, dtype=training.tf.float32),
            "critic_lr_var": training.tf.Variable(0.6, dtype=training.tf.float32),
            "actor_lr_schedule": target_actor_schedule,
            "critic_lr_schedule": target_critic_schedule,
            "actor_optimizer": training.tf.keras.optimizers.Adam(target_actor_schedule),
            "critic1_optimizer": training.tf.keras.optimizers.Adam(target_critic_schedule),
            "critic2_optimizer": training.tf.keras.optimizers.Adam(target_critic_schedule),
        }
        training._restore_learning_rate_runtime_state(
            SimpleNamespace(agents=[target_agent]),
            saved,
        )
        self.assertAlmostEqual(float(target_agent["actor_lr_var"].numpy()), 0.04, places=7)
        self.assertAlmostEqual(float(target_actor_schedule.initial_value_var.numpy()), 0.04, places=7)
        self.assertAlmostEqual(float(target_agent["critic_lr_var"].numpy()), 0.08, places=7)
        self.assertAlmostEqual(float(target_critic_schedule.initial_value_var.numpy()), 0.08, places=7)
        self.assertEqual(int(target_agent["actor_optimizer"].iterations.numpy()), 0)
        self.assertEqual(int(target_agent["critic1_optimizer"].iterations.numpy()), 0)
        self.assertEqual(int(target_agent["critic2_optimizer"].iterations.numpy()), 0)
        self.assertAlmostEqual(
            float(target_actor_schedule(target_agent["actor_optimizer"].iterations).numpy()),
            source_actor_effective,
            places=7,
        )
        self.assertAlmostEqual(
            float(target_critic_schedule(target_agent["critic1_optimizer"].iterations).numpy()),
            source_critic_effective,
            places=7,
        )


class ModelLoadingIntegrityTests(unittest.TestCase):
    class _FakeModel:
        def __init__(self):
            self.loaded = []

        def load_weights(self, path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            self.loaded.append(str(path))

        def get_weights(self):
            return []

        def set_weights(self, weights):
            return None

    def test_actor_only_loaders_do_not_require_training_only_networks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(2):
                (root / f"actor_{index}.weights.h5").write_bytes(b"actor")

            for controller_type in (training.OptimizedMADDPG, training.OptimizedMATD3):
                controller = controller_type.__new__(controller_type)
                controller.eval_actor_only = True
                controller.agents = [
                    {"actor": self._FakeModel()},
                    {"actor": self._FakeModel()},
                ]
                self.assertTrue(controller.load_models(str(root), strict=True))
                self.assertTrue(all(agent["actor"].loaded for agent in controller.agents))

    def test_matd3_strict_resume_rejects_incomplete_twin_critics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "actor_0.weights.h5").write_bytes(b"actor")
            (root / "critic1_0.weights.h5").write_bytes(b"critic1")
            controller = training.OptimizedMATD3.__new__(training.OptimizedMATD3)
            controller.eval_actor_only = False
            controller.agents = [{
                "actor": self._FakeModel(),
                "target_actor": self._FakeModel(),
                "critic1": self._FakeModel(),
                "critic2": self._FakeModel(),
                "target_critic1": self._FakeModel(),
                "target_critic2": self._FakeModel(),
            }]
            with self.assertRaisesRegex(RuntimeError, "critics"):
                controller.load_models(str(root), strict=True)


class OfficialEvaluationCacheTests(unittest.TestCase):
    @staticmethod
    def _valid_summary():
        return {
            "episodes": 1,
            "team_success_rate": 0.0,
            "agent_success_rates": [0.0, 0.0, 0.0],
            "all_reached_without_safe_team_success_rate": 0.0,
            "collision_free_rate": 1.0,
            "avg_collision_count": 0.0,
            "avg_team_final_goal_distance": 10.0,
            "avg_team_total_path_length": 20.0,
        }

    def test_changed_actor_weights_invalidate_same_path_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model" / "final"
            model_dir.mkdir(parents=True)
            actor = model_dir / "actor_0.weights.h5"
            actor.write_bytes(b"old")
            old_signature = official._compute_model_signature(model_dir)
            old_spec = {
                "episodes": 1,
                "validation_role": "checkpoint_selection",
                "evaluated_model_path": str(model_dir.resolve()),
                "evaluated_model_signature": old_signature,
            }
            spec_path = root / "post_eval_spec.json"
            spec_path.write_text(json.dumps(old_spec), encoding="utf-8")
            results = {
                "episodes": 1,
                "summary": self._valid_summary(),
                "episode_details": [{"episode": 0}],
                "model_path": str(model_dir),
                "evaluation_setup": {"action_force_ratio_source": "checkpoint_variant"},
            }
            actor.write_bytes(b"new")
            new_spec = dict(old_spec)
            new_spec["evaluated_model_signature"] = official._compute_model_signature(model_dir)
            errors = official._existing_eval_reuse_errors(
                results=results,
                spec_path=spec_path,
                expected_spec=new_spec,
                model_path=model_dir,
            )
            self.assertTrue(any("post_eval_spec" in item for item in errors))


class EvaluationNoiseIntegrityTests(unittest.TestCase):
    @staticmethod
    def _evaluator(noise_scale=0.11, random_action_prob=0.01, seed=101):
        evaluator = evaluation.ModelEvaluator.__new__(evaluation.ModelEvaluator)
        evaluator._eval_noise_scale = float(noise_scale)
        evaluator._eval_random_action_prob = float(random_action_prob)
        evaluator._eval_noise_seed = int(seed)
        return evaluator

    def test_episode_stream_is_independent_of_batch_and_early_completion(self):
        actions_batch = evaluation.tf.zeros((2, 3, 7), dtype=evaluation.tf.float32)
        evaluator = self._evaluator()
        streams = [
            evaluator._make_eval_noise_streams(episode_idx)
            for episode_idx in (10, 11)
        ]
        first_batch = evaluator._apply_eval_action_noise(actions_batch, streams).numpy()

        solo_evaluator = self._evaluator()
        solo_stream = solo_evaluator._make_eval_noise_streams(10)
        first_solo = solo_evaluator._apply_eval_action_noise(
            evaluation.tf.zeros((3, 7), dtype=evaluation.tf.float32),
            solo_stream,
        ).numpy()
        np.testing.assert_array_equal(first_batch[0], first_solo)

        second_after_peer_completion = evaluator._apply_eval_action_noise(
            evaluation.tf.zeros((3, 7), dtype=evaluation.tf.float32),
            streams[1],
        ).numpy()
        fresh_evaluator = self._evaluator()
        fresh_stream = fresh_evaluator._make_eval_noise_streams(11)
        fresh_evaluator._apply_eval_action_noise(
            evaluation.tf.zeros((3, 7), dtype=evaluation.tf.float32),
            fresh_stream,
        )
        expected_second = fresh_evaluator._apply_eval_action_noise(
            evaluation.tf.zeros((3, 7), dtype=evaluation.tf.float32),
            fresh_stream,
        ).numpy()
        np.testing.assert_array_equal(second_after_peer_completion, expected_second)

    def test_random_replacement_stream_is_independent_of_gaussian_stream(self):
        actions = evaluation.tf.zeros((3, 7), dtype=evaluation.tf.float32)
        random_only = self._evaluator(noise_scale=0.0, random_action_prob=1.0)
        combined = self._evaluator(noise_scale=0.11, random_action_prob=1.0)
        random_only_actions = random_only._apply_eval_action_noise(
            actions,
            random_only._make_eval_noise_streams(5),
        ).numpy()
        combined_actions = combined._apply_eval_action_noise(
            actions,
            combined._make_eval_noise_streams(5),
        ).numpy()
        np.testing.assert_array_equal(random_only_actions, combined_actions)


class ProcessShardMergeTests(unittest.TestCase):
    def test_merge_preserves_obstacle_sequence_and_global_best_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shard_specs = []
            for index, reward in enumerate((1.0, 2.0)):
                shard_dir = root / f"shard_{index}"
                shard_dir.mkdir()
                best_html = shard_dir / "best_reward_interactive.html"
                best_html.write_text(f"best-{index}", encoding="utf-8")
                episode_html = shard_dir / f"episode_{index}.html"
                episode_html.write_text(f"episode-{index}", encoding="utf-8")
                result = {
                    "model_path": "/tmp/model/final",
                    "scenario": "paper3d_terrain_vectorized",
                    "episodes": 1,
                    "summary": {},
                    "evaluation_setup": {},
                    "episode_details": [{
                        "episode": index,
                        "reward": reward,
                        "terrain_complexity_level": 3,
                        "terrain_seed": 88 + index,
                        "terrain_variant_seed": 188 + index,
                        "obstacle_seed": 288 + index,
                        "collision_count": 0,
                        "terrain_collision_count": 0,
                        "obstacle_collision_count": 0,
                        "inter_agent_collision_count": 0,
                        "team_success": int(index == 1),
                        "success": int(index == 1),
                        "agent_success_flags": [int(index == 1)] * 3,
                        "agent_safe_flags": [1, 1, 1],
                        "agent_first_reach_steps": [-1, -1, -1],
                        "path_length": 20.0,
                        "agent_path_lengths": [6.0, 7.0, 7.0],
                        "final_goal_distance": 10.0,
                        "agent_final_goal_distances": [3.0, 3.0, 4.0],
                    }],
                    "visualization_artifacts": {
                        "episode_visualizations": [{
                            "episode": index,
                            "reward": reward,
                            "team_success": int(index == 1),
                            "files": {"html_path": episode_html.name},
                        }],
                        "best_reward_html": str(best_html),
                        "team_success_best_html": str(best_html) if index == 1 else None,
                    },
                }
                (shard_dir / "evaluation_results.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                shard_specs.append({
                    "index": index,
                    "start": index,
                    "count": 1,
                    "dir": str(shard_dir),
                    "log": str(shard_dir / "worker.log"),
                })

            args = SimpleNamespace(
                eval_episodes=2,
                collision_distance_threshold=0.5,
                load_model_path="/tmp/model/final",
                scenario_name="paper3d_terrain_vectorized",
                eval_process_workers=2,
                eval_episode_parallelism=2,
                eval_env_step_threads=2,
                eval_noise_seed=101,
            )
            merged = evaluation._merge_process_shard_results(
                args,
                shard_specs,
                root / "shards",
                root / "output",
                "start",
            )
            self.assertEqual(merged["obstacle_seed_sequence"], [288, 289])
            artifacts = merged["visualization_artifacts"]
            self.assertEqual(
                Path(artifacts["best_reward_html"]).read_text(encoding="utf-8"),
                "best-1",
            )
            self.assertTrue(
                all(Path(entry["files"]["html_path"]).exists() for entry in artifacts["episode_visualizations"])
            )


if __name__ == "__main__":
    unittest.main()

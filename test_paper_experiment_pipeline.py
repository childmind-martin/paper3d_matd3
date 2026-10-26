import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from paper_experiment_pipeline import (
    _clean_subprocess_environment,
    _evaluation_device_environment,
    _load_preflight_record,
    _load_candidate_index,
    _load_training_record,
    _preflight_record_path,
    _training_launch_contract,
    _validate_resolved_training_manifest,
    run_training,
)
from paper_experiment_protocol import (
    FORMAL_EVALUATION_DEVICE_REQUIREMENT,
    _FORMAL_COMMON_TRAINING_HYPERPARAMETERS,
    execution_environment_allowlist,
    method_spec,
)
from paper_experiment_schema import (
    PaperSchemaError,
    atomic_write_json,
    content_sha256,
    fingerprint_artifact,
)
from paper_curriculum import paper_physics_contract, paper_task_semantics_contract


class PaperExperimentPipelineTests(unittest.TestCase):
    def test_formal_evaluation_device_is_cpu_deterministic(self):
        protocol = {
            "execution_environment": {
                "evaluation_device_requirement": dict(
                    FORMAL_EVALUATION_DEVICE_REQUIREMENT
                )
            }
        }
        environment = _evaluation_device_environment(protocol)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(environment["TF_DETERMINISTIC_OPS"], "1")
        self.assertEqual(environment["TF_ENABLE_ONEDNN_OPTS"], "0")

    def test_training_record_rejects_model_root_configuration_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "model"
            model_root.mkdir()
            results_path = model_root / "results.json"
            results_path.write_text('{"action_force_ratio": 0.32}', encoding="utf-8")
            candidate_index = root / "candidate_index.json"
            candidate_index.write_text("{}", encoding="utf-8")
            record = {
                "schema_version": "paper_training_run_record_v3",
                "protocol_content_sha256": "protocol-content",
                "method_id": "EC-MATD3-APF",
                "train_seed": 101,
                "candidate_index": fingerprint_artifact(candidate_index),
                "model_root": fingerprint_artifact(model_root),
            }
            record["content_sha256"] = content_sha256(record)
            record_path = (
                root
                / "records"
                / "training"
                / "EC-MATD3-APF"
                / "seed_101.json"
            )
            atomic_write_json(record_path, record)
            protocol = {"content_sha256": "protocol-content"}
            _load_training_record(
                protocol, root, "EC-MATD3-APF", 101
            )

            results_path.write_text('{"action_force_ratio": 0.50}', encoding="utf-8")
            with self.assertRaisesRegex(PaperSchemaError, "model root changed"):
                _load_training_record(
                    protocol, root, "EC-MATD3-APF", 101
                )

    def test_candidate_index_freezes_checkpoint_specific_force_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protocol_file = root / "protocol.json"
            protocol_file.write_text("{}", encoding="utf-8")
            checkpoint_dir = root / "candidate"
            checkpoint_dir.mkdir()
            metadata = {
                "schema_version": "paper_training_checkpoint_v5",
                "completed_training_iterations": 320,
                "checkpoint_action_force_ratio": 0.47,
                "checkpoint_action_force_ratio_episode": 320,
            }
            metadata["content_sha256"] = content_sha256(metadata)
            metadata_path = checkpoint_dir / "paper_training_checkpoint.json"
            atomic_write_json(metadata_path, metadata)
            candidate = {
                "checkpoint": fingerprint_artifact(checkpoint_dir),
                "training_metadata_path": str(metadata_path),
                "training_metadata_content_sha256": metadata["content_sha256"],
                "checkpoint_action_force_ratio": 0.47,
            }
            index = {
                "schema_version": "paper_training_candidate_index_v2",
                "candidate_count": 1,
                "candidates": [candidate],
                "experiment_protocol": fingerprint_artifact(protocol_file),
                "experiment_protocol_content_sha256": "protocol-content",
            }
            index["content_sha256"] = content_sha256(index)
            index_path = root / "candidate_index.json"
            atomic_write_json(index_path, index)
            loaded = _load_candidate_index(index_path)
            self.assertEqual(
                loaded["candidates"][0]["checkpoint_action_force_ratio"],
                0.47,
            )

            index["candidates"][0]["checkpoint_action_force_ratio"] = 0.32
            index["content_sha256"] = content_sha256(
                {key: value for key, value in index.items() if key != "content_sha256"}
            )
            atomic_write_json(index_path, index)
            with self.assertRaisesRegex(PaperSchemaError, "force ratio mismatch"):
                _load_candidate_index(index_path)

    def test_formal_training_forbids_trajectory_changing_resume(self):
        with self.assertRaisesRegex(PaperSchemaError, "forbid.*resume"):
            run_training(
                "/does/not/matter/protocol.json",
                "/does/not/matter/experiment",
                "EC-MATD3-APF",
                101,
                resume_existing=True,
            )

    def test_training_requires_unchanged_preflight_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protocol_file = root / "protocol.json"
            launcher_file = root / "resolved_launcher.json"
            source_file = root / "training_source.py"
            protocol_file.write_text("{}", encoding="utf-8")
            launcher_file.write_text("{}", encoding="utf-8")
            source_file.write_text("VALUE = 1\n", encoding="utf-8")
            method = method_spec("EC-MATD3-APF")
            protocol = {
                "content_sha256": "protocol-content",
                "methods": {"EC-MATD3-APF": method},
            }
            record = {
                "schema_version": "paper_training_preflight_record_v1",
                "protocol": fingerprint_artifact(protocol_file),
                "protocol_content_sha256": "protocol-content",
                "method_id": "EC-MATD3-APF",
                "method_spec": method,
                "train_seed": 101,
                "resolved_launcher_manifest": fingerprint_artifact(
                    launcher_file
                ),
                "source_artifacts": [
                    {
                        **fingerprint_artifact(source_file),
                        "logical_name": "training_source.py",
                    }
                ],
                "checks": {
                    "protocol_loaded": True,
                    "formal_task_bank_validated": True,
                    "clean_environment_policy_validated": True,
                    "launcher_metadata_validated": True,
                    "launcher_environment_validated": True,
                    "launcher_arguments_validated": True,
                    "training_started": False,
                },
            }
            record["content_sha256"] = content_sha256(record)
            record_path = _preflight_record_path(
                root, "EC-MATD3-APF", 101
            )
            atomic_write_json(record_path, record)
            loaded = _load_preflight_record(
                protocol, root, "EC-MATD3-APF", 101
            )
            self.assertEqual(loaded["content_sha256"], record["content_sha256"])

            source_file.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(PaperSchemaError, "source changed"):
                _load_preflight_record(
                    protocol, root, "EC-MATD3-APF", 101
                )

    def test_clean_environment_drops_unfrozen_scientific_state(self):
        protocol = {
            "execution_environment": {
                "policy": "clean_allowlist_v2",
                "inherited_variable_names": execution_environment_allowlist()
            }
        }
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "HOME": "/tmp/paper-home",
                "CROSS_AGENT_REFERENCE_ENABLED": "1",
                "REPLAY_TYPE": "contaminated",
            },
            clear=True,
        ):
            environment = _clean_subprocess_environment(
                protocol, random_seed=101
            )
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["HOME"], "/tmp/paper-home")
        self.assertEqual(environment["PYTHONHASHSEED"], "101")
        self.assertNotIn("CROSS_AGENT_REFERENCE_ENABLED", environment)
        self.assertNotIn("REPLAY_TYPE", environment)

    def test_resolved_launcher_separately_checks_method_and_algorithm(self):
        protocol_file = Path("/tmp/frozen-paper-protocol.json").resolve()
        protocol = {
            "protocol_id": "protocol",
            "methods": {"EC-MATD3-APF": method_spec("EC-MATD3-APF")},
            "train_seeds": [101],
            "execution_environment": {
                "policy": "clean_allowlist_v2",
                "inherited_variable_names": execution_environment_allowlist(),
                "training_device_requirement": {
                    "policy": "tensorflow_gpu_required_v1",
                    "minimum_physical_gpus": 1,
                    "minimum_logical_gpus": 1,
                },
                "python_runtimes": {
                    "training": {
                        "executable_artifact": {
                            "path": str(Path(sys.executable).resolve())
                        }
                    }
                },
            },
            "task_bank": {"manifest_path": "/tmp/formal-task-bank.json"},
            "training": {
                "synchronous_iterations": 800,
                "parallel_environments": 8,
                "batch_size": 1024,
                "interruption_policy": "restart_train_seed_from_scratch_v1",
                "common_hyperparameter_contract": (
                    _FORMAL_COMMON_TRAINING_HYPERPARAMETERS
                ),
                "physics_contract": paper_physics_contract(),
                "task_semantics_contract": paper_task_semantics_contract(),
                "stage_transition_runtime": {
                    "replay_keep_recent_transitions": 0,
                    "learning_rate_factor": 0.5,
                    "optimizer_slot_policy": "reset_adam_moments",
                },
            },
        }
        with mock.patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "HOME": "/tmp/paper-home"},
            clear=True,
        ):
            launch = _training_launch_contract(
                protocol, protocol_file, "EC-MATD3-APF", 101
            )
        method_environment = launch["method"]["environment"]
        self.assertEqual(launch["environment"]["MATD3_REQUIRE_GPU"], "1")
        self.assertEqual(
            launch["environment"]["SAVE_TRAINING_RESUME_STATE"], "0"
        )
        argv = [
            "--scenario", "paper3d_terrain_energy",
            "--algo", "matd3",
            "--train-episodes", "800",
            "--batch-size", "1024",
            "--num-envs", "8",
            "--seed", "101",
            "--enable-role-shuffle", "0",
            "--simulation-dt", "0.08",
            "--agent-size", "0.5",
            "--agent-max-speed", "42.5",
            "--agent-accel", "4.6",
            "--gravity", "0.0",
            "--control-accel-gain", "1.0",
            "--damping", "0.12",
            "--action-range-x", "2.5",
            "--action-range-y", "2.5",
            "--action-range-z", "2.2",
            "--z-action-bias", "0.0",
            "--quadrotor-attitude-response-time", "0.05",
            "--quadrotor-psi-cmd", "0.0",
            "--success-distance-threshold", "2.0",
            "--collision-distance-threshold", "0.5",
            "--matd3-use-dual-q", method_environment["MATD3_USE_DUAL_Q"],
            "--matd3-use-separated-gradient",
            method_environment["MATD3_USE_SEPARATED_GRADIENT"],
            "--matd3-use-hybrid-actor-objective",
            method_environment["MATD3_USE_HYBRID_ACTOR_OBJECTIVE"],
            "--matd3-action-semantics-mode",
            method_environment["MATD3_ACTION_SEMANTICS_MODE"],
            "--matd3-reconstruct-corrected-target",
            method_environment["MATD3_RECONSTRUCT_CORRECTED_TARGET"],
            "--actor-update-delay",
            method_environment["ACTOR_UPDATE_DELAY"],
            "--enable-action-correction",
            method_environment["ENABLE_ACTION_CORRECTION"],
            "--use-tf-potential-field",
            method_environment["USE_TF_POTENTIAL_FIELD"],
            "--use-fr-feature", method_environment["USE_FR_FEATURE"],
            "--use-pf-feature", method_environment["USE_PF_FEATURE"],
            "--action-force-ratio", method_environment["ACTION_FORCE_RATIO"],
            "--action-force-ratio-schedule-pct",
            method_environment["ACTION_FORCE_RATIO_SCHEDULE_PCT"],
            "--save-model",
        ]
        for option, value in _FORMAL_COMMON_TRAINING_HYPERPARAMETERS[
            "launcher_flags"
        ].items():
            argv.extend([option, value])
        resolved = {
            "version": 1,
            "resolve_only": True,
            "cwd": str(launch["repo_root"]),
            "python_script": str(
                Path(launch["repo_root"]) / "paper3d_train_optimized.py"
            ),
            "python_executable": str(Path(sys.executable).resolve()),
            "argv": argv,
            "exec_env": launch["environment"],
            "meta": {
                "episodes": 800,
                "batch_size": 1024,
                "use_weighted_reward": "0",
                "algorithm": "matd3",
                "exp_name": launch["base_name"],
                "seed": "101",
                "num_envs": 8,
            },
        }
        _validate_resolved_training_manifest(
            resolved,
            launch=launch,
            protocol=protocol,
            protocol_file=protocol_file,
            method_id="EC-MATD3-APF",
            train_seed=101,
        )
        tampered = dict(resolved)
        tampered["argv"] = list(argv)
        tampered["argv"][
            tampered["argv"].index("--matd3-action-semantics-mode") + 1
        ] = "dual"
        with self.assertRaisesRegex(PaperSchemaError, "argument mismatch"):
            _validate_resolved_training_manifest(
                tampered,
                launch=launch,
                protocol=protocol,
                protocol_file=protocol_file,
                method_id="EC-MATD3-APF",
                train_seed=101,
            )


if __name__ == "__main__":
    unittest.main()

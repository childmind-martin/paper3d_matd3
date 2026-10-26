import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from paper_checkpoint_selection import (
    PAPER_CHECKPOINT_SELECTION_SCHEMA_VERSION,
    _candidate_from_manifest,
    _candidate_rank_key,
    _validation_eligibility,
    validate_selection_result,
)
from paper_curriculum import validation_checkpoint_gate
from paper_experiment_schema import (
    PAPER_AGGREGATE_SCHEMA_VERSION,
    PAPER_RUN_MANIFEST_SCHEMA_VERSION,
    PaperSchemaError,
    content_sha256,
    fingerprint_artifact,
    wilson_interval,
)
from paper_experiment_protocol import FORMAL_EVALUATION_DEVICE_REQUIREMENT


def _candidate(label, successes, checkpoint_sha, *, collision_count=3, spl=0.7):
    candidate = {
        "label": label,
        "training_difficulty_bin": "C3",
        "experiment_protocol_content_sha256": "p" * 64,
        "episodes": 64,
        "team_success": wilson_interval(successes, 64),
        "collision_rate": wilson_interval(collision_count, 64),
        "agent_spl": {"mean": spl},
        "control_effort": {"mean": 4.0},
        "checkpoint": {"path": f"/models/{label}", "sha256": checkpoint_sha},
    }
    candidate["validation_eligibility"] = _validation_eligibility(
        candidate, validation_checkpoint_gate("C3")
    )
    return candidate


def _selection_payload(*candidates):
    ranked = sorted(list(candidates), key=_candidate_rank_key)
    selected = ranked[0]
    payload = {
        "schema_version": PAPER_CHECKPOINT_SELECTION_SCHEMA_VERSION,
        "selection_data_role": "validation",
        "adequacy_required_for_selection": True,
        "experiment_protocol_content_sha256": "p" * 64,
        "validation_adequacy_gate": validation_checkpoint_gate("C3"),
        "eligible_candidate_count": sum(
            int(candidate["validation_eligibility"]["eligible"])
            for candidate in ranked
        ),
        "candidate_count": len(ranked),
        "candidates_ranked": ranked,
        "selected_label": selected["label"],
        "selected_checkpoint": selected["checkpoint"],
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


class PaperCheckpointSelectionTests(unittest.TestCase):
    def test_candidate_separates_paper_method_id_from_training_algorithm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            protocol_path.write_text("{}", encoding="utf-8")
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "weights.bin").write_bytes(b"weights")
            task_bank_dir = root / "task_bank"
            task_bank_dir.mkdir()
            (task_bank_dir / "manifest.json").write_text("bank", encoding="utf-8")

            aggregate = {
                "schema_version": PAPER_AGGREGATE_SCHEMA_VERSION,
                "episodes": 64,
                "unique_tasks": 64,
                "train_seeds": [101],
                "eval_seeds": [1701],
                "difficulty_bins": ["C3"],
                "team_success": wilson_interval(48, 64),
                "terminal_reason_rates": {
                    "collision": wilson_interval(3, 64)
                },
                "agent_spl": {"mean": 0.75},
                "control_effort": {"mean": 4.0},
            }
            aggregate["content_sha256"] = content_sha256(aggregate)
            aggregate_path = root / "aggregate.json"
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

            protocol_hash = "p" * 64
            training_metadata = {
                "schema_version": "paper_training_checkpoint_v5",
                "completed_training_iterations": 800,
                "checkpoint_action_force_ratio": 0.32,
                "checkpoint_action_force_ratio_episode": 800,
                "train_seed": 101,
                "algorithm": "matd3",
                "scenario": "paper3d_terrain_energy",
                "paper_method_id": "EC-MATD3-APF",
                "experiment_protocol_content_sha256": protocol_hash,
                "experiment_protocol": fingerprint_artifact(protocol_path),
                "trained_on_difficulty_bin": "C3",
                "training_run_id": "training-run",
                "training_source_bundle_sha256": "s" * 64,
            }
            training_metadata["content_sha256"] = content_sha256(
                training_metadata
            )
            checkpoint = fingerprint_artifact(checkpoint_dir)
            checkpoint["training_checkpoint_metadata"] = training_metadata
            task_bank = fingerprint_artifact(task_bank_dir)
            task_bank.update(
                {
                    "split": "validation_C3",
                    "split_role": "validation",
                    "split_task_count": 64,
                    "split_task_ids_sha256": "t" * 64,
                }
            )
            manifest = {
                "schema_version": PAPER_RUN_MANIFEST_SCHEMA_VERSION,
                "phase": "validation",
                "split_role": "validation",
                "run_id": "validation-run",
                "algorithm": "EC-MATD3-APF",
                "scenario": "paper3d_terrain_energy",
                "split": "validation_C3",
                "train_seed": 101,
                "eval_seed": 1701,
                "experiment_protocol": {
                    **fingerprint_artifact(protocol_path),
                    "protocol_content_sha256": protocol_hash,
                },
                "outputs": {
                    "aggregate_json": fingerprint_artifact(aggregate_path)
                },
                "source_artifacts": [],
                "task_bank": task_bank,
                "checkpoint": checkpoint,
                "runtime_environment": {
                    "action_force_ratio": 0.32,
                    "force_action_force_ratio": None,
                    "evaluation_device_provenance": {
                        "requirement": dict(
                            FORMAL_EVALUATION_DEVICE_REQUIREMENT
                        ),
                        "environment": {
                            "CUDA_VISIBLE_DEVICES": "-1",
                            "TF_DETERMINISTIC_OPS": "1",
                            "TF_ENABLE_ONEDNN_OPTS": "0",
                            "TF_NUM_INTRAOP_THREADS": "1",
                            "TF_NUM_INTEROP_THREADS": "1",
                            "OMP_NUM_THREADS": "1",
                        },
                        "physical_gpus": [],
                        "logical_gpus": [],
                    },
                },
            }
            manifest["content_sha256"] = content_sha256(manifest)
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            protocol = {
                "content_sha256": protocol_hash,
                "methods": {
                    "EC-MATD3-APF": {"algorithm": "matd3"}
                },
            }
            with mock.patch(
                "paper_checkpoint_selection.load_protocol",
                return_value=protocol,
            ):
                candidate = _candidate_from_manifest("candidate", manifest_path)
            self.assertEqual(candidate["algorithm"], "EC-MATD3-APF")
            self.assertEqual(candidate["training_algorithm"], "matd3")

    def test_c3_gate_requires_high_observed_rate_and_statistical_lower_bound(self):
        passing = _candidate("passing", 44, "a" * 64)
        failing = _candidate("failing", 43, "b" * 64)
        self.assertTrue(passing["validation_eligibility"]["eligible"])
        self.assertFalse(failing["validation_eligibility"]["eligible"])
        self.assertIn(
            "validation successes 43 < 44",
            failing["validation_eligibility"]["rejection_reasons"],
        )

    def test_selection_validator_recomputes_eligibility(self):
        payload = _selection_payload(
            _candidate("best", 48, "a" * 64),
            _candidate("second", 45, "b" * 64),
        )
        validate_selection_result(payload)
        tampered = copy.deepcopy(payload)
        tampered["candidates_ranked"][0]["validation_eligibility"][
            "observed_successes"
        ] = 64
        tampered["content_sha256"] = content_sha256(
            {key: value for key, value in tampered.items() if key != "content_sha256"}
        )
        with self.assertRaisesRegex(PaperSchemaError, "eligibility mismatch"):
            validate_selection_result(tampered)

    def test_selection_validator_recomputes_ranking(self):
        payload = _selection_payload(
            _candidate("best", 48, "a" * 64),
            _candidate("second", 45, "b" * 64),
        )
        tampered = copy.deepcopy(payload)
        tampered["candidates_ranked"].reverse()
        tampered["selected_label"] = tampered["candidates_ranked"][0]["label"]
        tampered["selected_checkpoint"] = tampered["candidates_ranked"][0][
            "checkpoint"
        ]
        tampered["content_sha256"] = content_sha256(
            {key: value for key, value in tampered.items() if key != "content_sha256"}
        )
        with self.assertRaisesRegex(PaperSchemaError, "ranking order mismatch"):
            validate_selection_result(tampered)

    def test_ineligible_baseline_selection_is_valid_when_predeclared(self):
        payload = _selection_payload(
            _candidate("best", 20, "a" * 64),
            _candidate("second", 10, "b" * 64),
        )
        payload["adequacy_required_for_selection"] = False
        payload["content_sha256"] = content_sha256(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
        validated = validate_selection_result(payload)
        self.assertEqual(validated["selected_label"], "best")
        self.assertEqual(validated["eligible_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()

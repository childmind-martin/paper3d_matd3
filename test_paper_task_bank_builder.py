import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from build_paper_task_bank import (
    PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION,
    _cached_or_generate_split,
    _prepare_work_dir,
    generate_bank,
)
from paper_experiment_schema import content_sha256


def _spec(bank_id="bank"):
    payload = {
        "schema_version": PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION,
        "cache_kind": "generation_spec",
        "bank_id": bank_id,
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


class PaperTaskBankBuilderTests(unittest.TestCase):
    def test_generation_worker_count_must_be_positive(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "generation_workers must be positive"
            ):
                generate_bank(
                    Path(temporary) / "bank",
                    bank_id="generation-worker-count",
                    base_terrain_seed=67,
                    held_out_terrain_seeds=[171],
                    train_count=1,
                    validation_count=1,
                    test_count=1,
                    generalization_count_per_seed=1,
                    generation_workers=0,
                )

    def test_dynamic_audit_worker_count_must_be_positive(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "dynamic_audit_workers must be positive"
            ):
                generate_bank(
                    Path(temporary) / "bank",
                    bank_id="worker-count",
                    base_terrain_seed=67,
                    held_out_terrain_seeds=[171],
                    train_count=1,
                    validation_count=1,
                    test_count=1,
                    generalization_count_per_seed=1,
                    dynamic_audit_workers=0,
                )

    def test_generated_split_cache_is_reused_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            spec = _spec()
            _prepare_work_dir(work_dir, spec)
            layouts = [{"task_id": "task-a"}]
            with mock.patch(
                "build_paper_task_bank._generate_layouts",
                return_value=(
                    layouts,
                    {
                        "accepted": 1,
                        "candidates_examined": 1,
                        "rejected": 0,
                        "acceptance_rate": 1.0,
                        "rejection_reason_incidence": {},
                    },
                ),
            ) as generator:
                first = _cached_or_generate_split(
                    work_dir=work_dir,
                    generation_spec_sha256=spec["content_sha256"],
                    config={},
                    count=1,
                    bank_id="bank",
                    split_name="train_C0",
                    role="train",
                )
                second = _cached_or_generate_split(
                    work_dir=work_dir,
                    generation_spec_sha256=spec["content_sha256"],
                    config={},
                    count=1,
                    bank_id="bank",
                    split_name="train_C0",
                    role="train",
                )
            self.assertEqual(first["layouts"], layouts)
            self.assertEqual(second["layouts"], layouts)
            self.assertEqual(first["generation_audit"]["accepted"], 1)
            generator.assert_called_once()

    def test_work_directory_refuses_a_different_generation_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            _prepare_work_dir(work_dir, _spec("first"))
            with self.assertRaisesRegex(RuntimeError, "different generation spec"):
                _prepare_work_dir(work_dir, _spec("second"))

    def test_rehashed_cache_with_wrong_task_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            spec = _spec()
            _prepare_work_dir(work_dir, spec)
            cache_path = work_dir / "generated_splits" / "train_C0.json"
            cache_path.parent.mkdir(parents=True)
            payload = {
                "schema_version": PAPER_TASK_BANK_BUILD_CACHE_SCHEMA_VERSION,
                "cache_kind": "split",
                "generation_spec_sha256": spec["content_sha256"],
                "split": "train_C0",
                "role": "train",
                "task_count": 0,
                "layouts": [],
            }
            payload["content_sha256"] = content_sha256(payload)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "task count mismatch"):
                _cached_or_generate_split(
                    work_dir=work_dir,
                    generation_spec_sha256=spec["content_sha256"],
                    config={},
                    count=1,
                    bank_id="bank",
                    split_name="train_C0",
                    role="train",
                )


if __name__ == "__main__":
    unittest.main()

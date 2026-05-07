#!/usr/bin/env python3
"""Check the reviewer artifact layout without touching source results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "RESULTS_MANIFEST.json",
    "docs/PAPER_RESULT_MAP.md",
    "docs/SIMULATION_BOUNDARY.md",
    "docs/METRIC_INTERPRETATION.md",
    "docs/APF_SANITY_EVIDENCE_NOTE.md",
]

REQUIRED_DIRS = [
    "level2_partial_summary_20260427_141546",
    "level2_dual_semantics_partial_summary_20260427_141546",
    "diagnostics/level2_official_eval_multiseed_summary_20260428_163815",
]

MANIFEST_FILE_FIELDS = [
    "raw_csv",
    "aggregate_csv",
    "summary_json",
]


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def is_csv_or_json(value: str) -> bool:
    return Path(value).suffix.lower() in {".csv", ".json"}


def add_check(checks: list[tuple[bool, str]], ok: bool, label: str) -> None:
    checks.append((ok, label))


def load_manifest(checks: list[tuple[bool, str]]) -> dict | None:
    manifest_path = ROOT / "RESULTS_MANIFEST.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        add_check(checks, False, f"RESULTS_MANIFEST.json valid JSON ({exc})")
        return None

    add_check(checks, True, "RESULTS_MANIFEST.json valid JSON")
    return manifest


def check_manifest_files(
    manifest: dict | None,
    checks: list[tuple[bool, str]],
) -> None:
    if not manifest:
        return

    entries = manifest.get("entries", {})
    if not isinstance(entries, dict):
        add_check(checks, False, "manifest entries is an object")
        return

    add_check(checks, True, "manifest entries is an object")

    expected_paths: list[tuple[str, Path]] = []
    for entry_name, entry in entries.items():
        if not isinstance(entry, dict):
            add_check(checks, False, f"{entry_name}: entry is an object")
            continue

        result_dir = entry.get("result_dir")
        if result_dir:
            path = resolve_repo_path(result_dir)
            expected_paths.append((f"{entry_name}: result_dir {result_dir}", path))

        for field in MANIFEST_FILE_FIELDS:
            value = entry.get(field)
            if not value:
                continue
            if not isinstance(value, str) or not is_csv_or_json(value):
                continue
            path = resolve_repo_path(value)
            expected_paths.append((f"{entry_name}: {field} {value}", path))

        source_files = entry.get("source_files") or []
        if not isinstance(source_files, list):
            add_check(checks, False, f"{entry_name}: source_files is a list")
            continue

        for value in source_files:
            if not isinstance(value, str) or not is_csv_or_json(value):
                continue
            path = resolve_repo_path(value)
            expected_paths.append((f"{entry_name}: source_file {value}", path))

    missing = [
        label
        for label, path in expected_paths
        if not (path.is_dir() if "result_dir" in label else path.is_file())
    ]
    if missing:
        add_check(checks, False, f"manifest result directories and CSV/JSON files ({len(missing)} missing)")
        for label in missing:
            add_check(checks, False, f"missing {label}")
    else:
        add_check(
            checks,
            True,
            f"manifest result directories and CSV/JSON files ({len(expected_paths)} checked)",
        )


def main() -> int:
    checks: list[tuple[bool, str]] = []

    for rel_path in REQUIRED_FILES:
        add_check(checks, (ROOT / rel_path).is_file(), f"{rel_path} exists")

    for rel_path in REQUIRED_DIRS:
        add_check(checks, (ROOT / rel_path).is_dir(), f"{rel_path}/ exists")

    manifest = load_manifest(checks)
    check_manifest_files(manifest, checks)

    failed = [(ok, label) for ok, label in checks if not ok]

    print("Reviewer artifact layout check")
    print("=" * 31)
    for ok, label in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {label}")

    print("-" * 31)
    if failed:
        print(f"FAIL  {len(failed)} required artifact check(s) failed.")
        return 1

    print("PASS  all required artifact checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

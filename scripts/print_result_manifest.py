#!/usr/bin/env python3
"""Print a human-readable summary of RESULTS_MANIFEST.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "RESULTS_MANIFEST.json"


def main() -> int:
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError:
        print(f"ERROR: missing {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in RESULTS_MANIFEST.json: {exc}", file=sys.stderr)
        return 1

    print(manifest.get("artifact_name", "Reviewer result manifest"))
    print("=" * 72)
    purpose = manifest.get("purpose")
    if purpose:
        print(f"Purpose: {purpose}")
        print()

    entries = manifest.get("entries", {})
    if not isinstance(entries, dict):
        print("ERROR: manifest entries must be a JSON object", file=sys.stderr)
        return 1

    for key, entry in entries.items():
        if not isinstance(entry, dict):
            print(f"{key}: malformed entry")
            continue

        print(f"[{key}]")
        print(f"  Paper item: {entry.get('paper_item', 'NA')}")
        print(f"  Evidence role: {entry.get('evidence_role', 'NA')}")
        print(f"  Result directory: {entry.get('result_dir', 'NA')}")
        print(f"  Used for main claim: {entry.get('used_for_main_claim', 'NA')}")
        print(f"  Notes: {entry.get('notes', 'NA')}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

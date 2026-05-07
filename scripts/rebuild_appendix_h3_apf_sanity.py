#!/usr/bin/env python3
"""Summarize Appendix H.3 APF sanity/reference evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "ablation_experiments" / "batch_20260403_132242"
OUTPUT_PATH = ROOT / "artifacts" / "rebuilt_tables" / "appendix_h3_apf_sanity.md"

SOURCE_CANDIDATES = [
    "plots/summary_20260403_221128.json",
    "plots/latest_summary.json",
]

METRICS = [
    ("Method", ["name_en", "name", "display_name", "label"], []),
    ("Design role", ["description", "role"], []),
    ("Train team success", ["train_team_success_rate"], []),
    ("Avg reward", ["avg_reward"], []),
    ("Final reward", ["final_reward"], []),
    ("Max reward", ["max_reward"], []),
    ("Eval team success", ["eval_team_success_rate"], []),
    ("Eval collision-free", ["eval_collision_free_rate"], []),
]


def find_existing(names: Iterable[str]) -> Path | None:
    for name in names:
        path = RESULT_DIR / name
        if path.exists():
            return path
    csv_files = sorted(RESULT_DIR.rglob("*.csv"))
    if csv_files:
        return csv_files[0]
    json_files = sorted(RESULT_DIR.rglob("*.json"))
    if json_files:
        return json_files[0]
    return None


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def load_json(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        rows = data.get("experiments") or data.get("aggregated") or data.get("runs") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    if not isinstance(rows, list):
        rows = []

    fields: list[str] = []
    seen: set[str] = set()
    clean_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean_rows.append(row)
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return clean_rows, fields


def get_value(row: dict, candidates: list[str]) -> object | None:
    lower_to_key = {str(key).lower(): key for key in row}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return None


def format_scalar(value: object) -> str:
    if value in (None, ""):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(number) >= 1000:
        return f"{number:.2f}"
    return f"{number:.3f}"


def format_metric(row: dict, value_candidates: list[str], std_candidates: list[str]) -> str:
    value = get_value(row, value_candidates)
    if value is None:
        return "NA"
    std = get_value(row, std_candidates) if std_candidates else None
    if std is None:
        return format_scalar(value)
    return f"{format_scalar(value)} +/- {format_scalar(std)}"


def missing_metric_names(fields: list[str]) -> list[str]:
    lower_fields = {field.lower() for field in fields}
    missing = []
    for name, value_candidates, _ in METRICS:
        if not any(candidate.lower() in lower_fields for candidate in value_candidates):
            missing.append(name)
    return missing


def build_markdown(rows: list[dict], fields: list[str], source_path: Path) -> str:
    headers = [metric[0] for metric in METRICS]
    lines = [
        "# Appendix H.3 APF Sanity Evidence",
        "",
        "**Evidence role:** supplementary sanity/reference evidence.",
        "",
        f"Source: `{source_path.relative_to(ROOT)}`",
        "",
        "This output summarizes the APF-only/action-only/action-plus-APF sanity evidence when source files are available.",
        "",
        "It is not used to establish the necessity of dual-semantic replay, critic construction, target reconstruction, or separated-gradient routing.",
        "",
        "It should not be presented as equal to Table 5 or Table 6.",
        "",
    ]

    missing = missing_metric_names(fields)
    if missing:
        lines.extend(
            [
                "## Column Notes",
                "",
                "The following expected metric groups were not found and are shown as `NA`: "
                + ", ".join(missing)
                + ".",
                "",
                "Available columns: `" + "`, `".join(fields) + "`",
                "",
            ]
        )

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [format_metric(row, candidates, std_candidates) for _, candidates, std_candidates in METRICS]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not RESULT_DIR.is_dir():
        print(f"ERROR: missing result directory {RESULT_DIR.relative_to(ROOT)}", file=sys.stderr)
        return 1

    source_path = find_existing(SOURCE_CANDIDATES)
    if not source_path:
        print(f"ERROR: no CSV/JSON source found under {RESULT_DIR.relative_to(ROOT)}", file=sys.stderr)
        return 1

    if source_path.suffix.lower() == ".csv":
        rows, fields = load_csv(source_path)
    else:
        rows, fields = load_json(source_path)

    if not rows:
        print(f"ERROR: no rows found in {source_path.relative_to(ROOT)}", file=sys.stderr)
        return 1

    missing = missing_metric_names(fields)
    if missing:
        print("WARNING: missing expected metric groups: " + ", ".join(missing))
        print("Available columns: " + ", ".join(fields))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_markdown(rows, fields, source_path), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} from {source_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

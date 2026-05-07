#!/usr/bin/env python3
"""Rebuild a Markdown summary for Table 5 from processed result files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "level2_dual_semantics_partial_summary_20260427_141546"
OUTPUT_PATH = ROOT / "artifacts" / "rebuilt_tables" / "table5_semantic_ablation.md"

CSV_CANDIDATES = [
    "official_eval_multiseed_aggregated.csv",
    "official_eval_cross_algo_summary.csv",
    "official_eval_multiseed_all_runs.csv",
]
JSON_CANDIDATES = [
    "official_eval_multiseed_summary.json",
    "official_eval_cross_algo_summary.json",
]

METRICS = [
    (
        "Method",
        ["display_name", "method", "algorithm", "name", "name_en", "label"],
        [],
    ),
    (
        "n",
        ["seed_count", "n", "num_seeds", "episodes", "seeds"],
        [],
    ),
    (
        "Team success",
        ["team_success_rate_mean", "team_success_mean", "team_success_rate", "eval_team_success_rate"],
        ["team_success_rate_std", "team_success_std"],
    ),
    (
        "Dense reward",
        ["avg_reward_mean", "dense_reward_mean", "reward_mean", "avg_reward"],
        ["avg_reward_std", "dense_reward_std", "reward_std"],
    ),
    (
        "Collision-free",
        ["collision_free_rate_mean", "collision_free_mean", "collision_free_rate", "eval_collision_free_rate"],
        ["collision_free_rate_std", "collision_free_std"],
    ),
    (
        "Final distance",
        [
            "avg_team_final_goal_distance_mean",
            "final_goal_distance_mean",
            "avg_final_goal_distance_mean",
            "avg_team_final_goal_distance",
            "eval_avg_team_final_goal_distance",
        ],
        ["avg_team_final_goal_distance_std", "final_goal_distance_std", "avg_final_goal_distance_std"],
    ),
    (
        "Total collisions",
        ["avg_collision_count_mean", "total_collision_burden_mean", "total_collisions_mean", "avg_collision_count"],
        ["avg_collision_count_std", "total_collision_burden_std", "total_collisions_std"],
    ),
]


def find_existing(names: Iterable[str]) -> Path | None:
    for name in names:
        path = RESULT_DIR / name
        if path.exists():
            return path
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
        rows = data.get("aggregated") or data.get("runs") or data.get("experiments") or []
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
    if isinstance(value, list):
        return str(len(value))
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


def format_count(row: dict, candidates: list[str]) -> str:
    value = get_value(row, candidates)
    if value in (None, ""):
        return "NA"
    if isinstance(value, list):
        return str(len(value))
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


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
        "# Rebuilt Table 5 Semantic Ablation",
        "",
        "**Evidence role:** core mechanism evidence.",
        "",
        f"Source: `{source_path.relative_to(ROOT)}`",
        "",
        "This table is rebuilt from processed CSV/JSON artifacts. It does not modify source result files.",
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
        values = []
        for name, candidates, std_candidates in METRICS:
            if name == "n":
                values.append(format_count(row, candidates))
            else:
                values.append(format_metric(row, candidates, std_candidates))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not RESULT_DIR.is_dir():
        print(f"ERROR: missing result directory {RESULT_DIR.relative_to(ROOT)}", file=sys.stderr)
        return 1

    source_path = find_existing(CSV_CANDIDATES)
    if source_path:
        rows, fields = load_csv(source_path)
    else:
        source_path = find_existing(JSON_CANDIDATES)
        if not source_path:
            print(f"ERROR: no CSV/JSON source found under {RESULT_DIR.relative_to(ROOT)}", file=sys.stderr)
            return 1
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

#!/usr/bin/env python3
"""Rebuild a Markdown summary for Table 6 from processed Level-2 results."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "diagnostics" / "level2_official_eval_multiseed_summary_20260428_163815"
OUTPUT_PATH = ROOT / "artifacts" / "rebuilt_tables" / "table6_level2_checkpoint_fr.md"

CSV_CANDIDATES = [
    "official_eval_multiseed_aggregated_model_fr.csv",
    "official_eval_multiseed_aggregated.csv",
    "official_eval_cross_algo_summary.csv",
    "official_eval_multiseed_all_runs_model_fr.csv",
    "official_eval_multiseed_all_runs.csv",
]
JSON_CANDIDATES = [
    "official_eval_multiseed_summary_model_fr.json",
    "official_eval_multiseed_summary.json",
    "official_eval_cross_algo_summary.json",
]

ALL_RUNS_CANDIDATES = [
    "official_eval_multiseed_all_runs_model_fr.csv",
    "official_eval_multiseed_all_runs.csv",
]

METRICS = [
    ("Method", ["display_name", "method", "algorithm", "name", "name_en", "label"], []),
    ("n", ["seed_count", "n", "num_seeds", "episodes", "seeds"], []),
    (
        "Team success",
        ["team_success_rate_mean", "team_success_mean", "team_success_rate", "eval_team_success_rate"],
        ["team_success_rate_std", "team_success_std"],
    ),
    (
        "Any-agent arrival",
        [
            "agent_success_rate_any_mean",
            "agent_success_rate_any",
            "any_agent_arrival_rate_mean",
            "any_agent_arrival_mean",
            "any_agent_arrival_rate",
            "any_agent_arrival",
            "at_least_one_agent_arrival_rate_mean",
            "at_least_one_arrival_rate_mean",
            "one_or_more_agent_arrival_rate_mean",
        ],
        ["any_agent_arrival_rate_std", "any_agent_arrival_std", "at_least_one_agent_arrival_rate_std"],
    ),
    (
        "Two-agent arrival",
        [
            "agent_success_rate_two_or_more_mean",
            "agent_success_rate_two_or_more",
            "two_agent_arrival_rate_mean",
            "two_agent_arrival_mean",
            "two_agent_arrival_rate",
            "two_agent_arrival",
            "at_least_two_agent_arrival_rate_mean",
            "at_least_two_arrival_rate_mean",
        ],
        ["two_agent_arrival_rate_std", "two_agent_arrival_std", "at_least_two_agent_arrival_rate_std"],
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

PERCENT_METRICS = {"Team success", "Any-agent arrival", "Two-agent arrival", "Collision-free"}


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


def safe_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def partial_rates_from_result(path: Path) -> tuple[int, int, int]:
    """Return any-agent count, two-or-more-agent count, and counted episodes."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0

    episodes = data.get("episode_details")
    if not isinstance(episodes, list):
        return 0, 0, 0

    any_count = 0
    two_count = 0
    counted = 0
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        flags = episode.get("agent_success_flags")
        if not isinstance(flags, list) or not flags:
            continue

        parsed = []
        for flag in flags:
            value = safe_int(flag)
            if value is not None:
                parsed.append(1 if value > 0 else 0)
        if not parsed:
            continue

        success_count = sum(parsed)
        any_count += int(success_count >= 1)
        two_count += int(success_count >= 2)
        counted += 1

    return any_count, two_count, counted


def enrich_partial_arrivals(rows: list[dict]) -> tuple[list[str], str | None]:
    all_runs_path = find_existing(ALL_RUNS_CANDIDATES)
    if not all_runs_path:
        return [], None

    all_runs, _ = load_csv(all_runs_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_runs:
        label = row.get("label")
        if label:
            grouped[str(label)].append(row)

    enriched_fields: list[str] = []
    for row in rows:
        label = row.get("label")
        if not label:
            continue

        any_total = 0
        two_total = 0
        counted_total = 0
        for run in grouped.get(str(label), []):
            result_value = run.get("results_path")
            if not result_value:
                continue
            result_path = Path(result_value)
            if not result_path.is_absolute():
                result_path = ROOT / result_path
            any_count, two_count, counted = partial_rates_from_result(result_path)
            any_total += any_count
            two_total += two_count
            counted_total += counted

        if counted_total <= 0:
            continue
        row["agent_success_rate_any_mean"] = any_total / counted_total
        row["agent_success_rate_two_or_more_mean"] = two_total / counted_total
        for field in ("agent_success_rate_any_mean", "agent_success_rate_two_or_more_mean"):
            if field not in enriched_fields:
                enriched_fields.append(field)

    return enriched_fields, str(all_runs_path.relative_to(ROOT))


def get_value(row: dict, candidates: list[str]) -> object | None:
    lower_to_key = {str(key).lower(): key for key in row}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return None


def format_scalar(value: object, *, scale: float = 1.0) -> str:
    if value in (None, ""):
        return "NA"
    if isinstance(value, list):
        return str(len(value))
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return str(value)

    if abs(number) >= 1000:
        return f"{number:.2f}"
    return f"{number:.3f}"


def format_metric(name: str, row: dict, value_candidates: list[str], std_candidates: list[str]) -> str:
    value = get_value(row, value_candidates)
    if value is None:
        return "NA"
    std = get_value(row, std_candidates) if std_candidates else None
    scale = 100.0 if name in PERCENT_METRICS else 1.0
    if std is None:
        return format_scalar(value, scale=scale)
    return f"{format_scalar(value, scale=scale)} +/- {format_scalar(std, scale=scale)}"


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


def build_markdown(rows: list[dict], fields: list[str], source_path: Path, partial_source: str | None) -> str:
    headers = [metric[0] for metric in METRICS]
    lines = [
        "# Rebuilt Table 6 Level-2 Checkpoint-FR",
        "",
        "**Evidence role:** primary deployment evidence.",
        "",
        f"Source: `{source_path.relative_to(ROOT)}`",
        "",
        "Any-agent and two-agent arrival are recomputed from `episode_details[].agent_success_flags` "
        "in the `evaluation_results.json` files referenced by the all-runs CSV.",
        "",
        "This table is rebuilt from processed CSV/JSON artifacts. It does not modify source result files.",
        "",
        "Interpret this as a matched-protocol deployment diagnostic, not as a universal MARL ranking.",
        "",
    ]
    if partial_source:
        lines.extend([f"Partial-arrival all-runs source: `{partial_source}`", ""])

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
                values.append(format_metric(name, row, candidates, std_candidates))
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

    enriched_fields, partial_source = enrich_partial_arrivals(rows)
    for field in enriched_fields:
        if field not in fields:
            fields.append(field)

    missing = missing_metric_names(fields)
    if missing:
        print("WARNING: missing expected metric groups: " + ", ".join(missing))
        print("Available columns: " + ", ".join(fields))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_markdown(rows, fields, source_path, partial_source), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} from {source_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

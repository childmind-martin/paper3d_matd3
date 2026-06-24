#!/usr/bin/env python3
"""Aggregate eval reward diagnostics for Level-3 reward audits."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METHOD_ORDER = [
    "DS Original",
    "DS Uniform",
    "DS Legacy PER",
    "MATD3 Full Dual Semantic",
    "MATD3 Collapsed Replay",
    "MATD3 No Corrected Target Reconstruction",
]

NUMERIC_FIELDS = {
    "train_seed",
    "eval_seed",
    "episode_id",
    "success",
    "any_arrival",
    "two_arrival",
    "collision_free",
    "final_goal_distance",
    "episode_length",
    "total_collisions",
    "reward_total",
    "reward_progress",
    "reward_clearance",
    "reward_height",
    "reward_stagnation",
    "reward_sync",
    "reward_energy",
    "reward_safety_penalty",
    "reward_collision_penalty",
    "reward_terrain_penalty",
    "reward_obstacle_penalty",
    "reward_inter_agent_penalty",
    "reward_boundary_penalty",
    "reward_terminal_success",
    "reward_terminal_failure",
    "reward_terminal_quality",
    "reward_apf_or_action_cost_if_any",
    "n_terrain_collisions",
    "n_obstacle_collisions",
    "n_inter_agent_collisions",
    "n_boundary_violations",
    "mean_semantic_gap",
    "max_semantic_gap",
    "mean_force_ratio",
    "path_length",
    "avg_speed",
    "avg_action_norm",
    "avg_corr_action_norm",
    "avg_action_delta_norm",
    "mean_pf_force_norm",
    "reward_diag_step_count",
    "reward_diag_missing_steps",
    "reward_total_before_clip_sum",
    "reward_clip_delta_sum",
}

SUMMARY_FIELDS = [
    "reward_total",
    "reward_progress",
    "reward_terminal_total",
    "reward_safety_penalty",
    "reward_collision_penalty",
    "reward_collision_family_penalty",
    "reward_sync",
    "reward_stagnation",
    "reward_energy",
    "episode_length",
    "total_collisions",
    "final_goal_distance",
    "success",
    "any_arrival",
    "two_arrival",
    "collision_free",
    "mean_semantic_gap",
    "max_semantic_gap",
    "mean_force_ratio",
    "avg_action_norm",
    "avg_corr_action_norm",
    "avg_action_delta_norm",
    "mean_pf_force_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        action="append",
        dest="patterns",
        required=True,
        help="Glob pattern for reward_decomposition_eval.csv files. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for aggregate reports.")
    return parser.parse_args()


def to_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def mean(values: Iterable[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return statistics.fmean(clean) if clean else math.nan


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def enrich_row(row: Dict[str, str], source_file: str) -> Dict[str, object]:
    out: Dict[str, object] = dict(row)
    out["source_file"] = source_file
    for field in NUMERIC_FIELDS:
        if field in out:
            out[field] = to_float(out[field])
    out["episode_key"] = f"{int(to_float(out.get('train_seed'), -1))}:{int(to_float(out.get('eval_seed'), -1))}:{int(to_float(out.get('episode_id'), -1))}"
    out["reward_terminal_total"] = (
        to_float(out.get("reward_terminal_success"), 0.0)
        + to_float(out.get("reward_terminal_failure"), 0.0)
        + to_float(out.get("reward_terminal_quality"), 0.0)
    )
    out["reward_collision_family_penalty"] = (
        to_float(out.get("reward_collision_penalty"), 0.0)
        + to_float(out.get("reward_terrain_penalty"), 0.0)
        + to_float(out.get("reward_obstacle_penalty"), 0.0)
        + to_float(out.get("reward_inter_agent_penalty"), 0.0)
        + to_float(out.get("reward_boundary_penalty"), 0.0)
    )
    out["reward_safety_family_penalty"] = (
        to_float(out.get("reward_safety_penalty"), 0.0)
        + to_float(out.get("reward_collision_family_penalty"), 0.0)
    )
    return out


def load_rows(patterns: Sequence[str]) -> List[Dict[str, object]]:
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    rows: List[Dict[str, object]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(enrich_row(row, path))
    return rows


def expand_paths(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(set(paths))


def method_sort_key(method: str) -> tuple:
    try:
        return (0, METHOD_ORDER.index(method))
    except ValueError:
        return (1, method)


def choose_comparison_methods(methods: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    ordered = [str(m) for m in sorted(set(methods), key=method_sort_key) if str(m)]
    if not ordered:
        return None, None
    baseline = "DS Original" if "DS Original" in ordered else ordered[0]
    candidates = [m for m in ordered if m != baseline]
    candidate = candidates[-1] if candidates else None
    return baseline, candidate


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_by_method(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method", ""))].append(row)

    out: List[Dict[str, object]] = []
    for method in sorted(grouped, key=method_sort_key):
        method_rows = grouped[method]
        summary: Dict[str, object] = {
            "method": method,
            "episode_count": len(method_rows),
            "train_seed_count": len({int(to_float(r.get("train_seed"), -1)) for r in method_rows}),
            "eval_seed_count": len({int(to_float(r.get("eval_seed"), -1)) for r in method_rows}),
        }
        for field in SUMMARY_FIELDS:
            summary[f"mean_{field}"] = mean(to_float(r.get(field)) for r in method_rows)
        out.append(summary)
    return out


def write_reward_gap_report(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    by_method = {str(row["method"]): row for row in summaries}
    baseline_name, candidate_name = choose_comparison_methods(by_method.keys())
    baseline = by_method.get(baseline_name or "")
    candidate = by_method.get(candidate_name or "")
    lines = [
        "Paired reward decomposition report",
        "",
        f"Method-level mean deltas use {candidate_name or 'candidate'} minus {baseline_name or 'baseline'}.",
    ]
    if not baseline or not candidate:
        lines.append("Missing baseline or candidate method, so gap decomposition is unavailable.")
    else:
        components = [
            ("reward_gap_vs_DSOriginal", "mean_reward_total"),
            ("progress_delta", "mean_reward_progress"),
            ("terminal_delta", "mean_reward_terminal_total"),
            ("safety_penalty_delta", "mean_reward_safety_penalty"),
            ("collision_penalty_delta", "mean_reward_collision_penalty"),
            ("collision_family_penalty_delta", "mean_reward_collision_family_penalty"),
            ("sync_delta", "mean_reward_sync"),
            ("stagnation_delta", "mean_reward_stagnation"),
            ("energy_delta", "mean_reward_energy"),
            ("episode_length_delta", "mean_episode_length"),
            ("total_collisions_delta", "mean_total_collisions"),
            ("final_goal_distance_delta", "mean_final_goal_distance"),
            ("team_success_rate_delta", "mean_success"),
            ("any_arrival_delta", "mean_any_arrival"),
            ("two_arrival_delta", "mean_two_arrival"),
        ]
        for label, field in components:
            delta = to_float(candidate.get(field)) - to_float(baseline.get(field))
            lines.append(f"{label}: {delta:.6g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_difference_report(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    by_key: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_key[str(row["episode_key"])][str(row.get("method", ""))] = row

    baseline_name, candidate_name = choose_comparison_methods(
        method for methods in by_key.values() for method in methods.keys()
    )
    pairs = []
    for key, methods in by_key.items():
        baseline = methods.get(baseline_name or "")
        candidate = methods.get(candidate_name or "")
        if baseline and candidate:
            pairs.append((key, baseline, candidate))

    lines = [
        "Paired reward difference report",
        "",
        f"paired_episode_count: {len(pairs)}",
        f"Deltas use {candidate_name or 'candidate'} minus {baseline_name or 'baseline'} on identical train_seed/eval_seed/episode_id.",
    ]
    fields = [
        "reward_total",
        "reward_progress",
        "reward_terminal_total",
        "reward_safety_penalty",
        "reward_collision_penalty",
        "reward_collision_family_penalty",
        "reward_sync",
        "reward_stagnation",
        "reward_energy",
        "episode_length",
        "total_collisions",
        "final_goal_distance",
        "success",
        "any_arrival",
        "two_arrival",
        "mean_semantic_gap",
        "mean_force_ratio",
        "avg_action_delta_norm",
    ]
    for field in fields:
        deltas = [to_float(full.get(field)) - to_float(ds.get(field)) for _, ds, full in pairs]
        lines.append(f"mean_delta_{field}: {mean(deltas):.6g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_success_failure_distribution(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("method", "")), int(to_float(row.get("success"), 0.0) >= 0.5))
        grouped[key].append(to_float(row.get("reward_total")))

    out = []
    for method, success in sorted(grouped, key=lambda x: (method_sort_key(x[0]), x[1])):
        rewards = grouped[(method, success)]
        out.append(
            {
                "method": method,
                "success": success,
                "episode_count": len(rewards),
                "mean_reward_total": mean(rewards),
                "min_reward_total": min(rewards) if rewards else math.nan,
                "p25_reward_total": percentile(rewards, 0.25),
                "p50_reward_total": percentile(rewards, 0.50),
                "p75_reward_total": percentile(rewards, 0.75),
                "max_reward_total": max(rewards) if rewards else math.nan,
            }
        )
    write_csv(
        path,
        out,
        [
            "method",
            "success",
            "episode_count",
            "mean_reward_total",
            "min_reward_total",
            "p25_reward_total",
            "p50_reward_total",
            "p75_reward_total",
            "max_reward_total",
        ],
    )


def write_candidate_case_report(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    _, candidate_name = choose_comparison_methods(str(r.get("method", "")) for r in rows)
    candidate_rows = [r for r in rows if r.get("method") == candidate_name]
    success_low = sorted(
        [r for r in candidate_rows if to_float(r.get("success"), 0.0) >= 0.5],
        key=lambda r: to_float(r.get("reward_total")),
    )[:5]
    failure_high = sorted(
        [r for r in candidate_rows if to_float(r.get("success"), 0.0) < 0.5],
        key=lambda r: to_float(r.get("reward_total")),
        reverse=True,
    )[:5]

    def describe(title: str, cases: Sequence[Dict[str, object]]) -> List[str]:
        lines = [title]
        if not cases:
            return lines + ["none"]
        for row in cases:
            lines.append(
                "episode_key={episode_key} reward_total={reward_total:.6g} "
                "success={success:.0f} final_distance={final_goal_distance:.6g} "
                "collisions={total_collisions:.6g} length={episode_length:.6g} "
                "safety_penalty={reward_safety_penalty:.6g} collision_penalty={reward_collision_penalty:.6g} "
                "terminal={reward_terminal_total:.6g}".format(
                    episode_key=row.get("episode_key", ""),
                    reward_total=to_float(row.get("reward_total")),
                    success=to_float(row.get("success"), 0.0),
                    final_goal_distance=to_float(row.get("final_goal_distance")),
                    total_collisions=to_float(row.get("total_collisions")),
                    episode_length=to_float(row.get("episode_length")),
                    reward_safety_penalty=to_float(row.get("reward_safety_penalty")),
                    reward_collision_penalty=to_float(row.get("reward_collision_penalty")),
                    reward_terminal_total=to_float(row.get("reward_terminal_total")),
                )
            )
        return lines

    lines = []
    display_name = candidate_name or "candidate"
    lines.extend(describe(f"{display_name}: successful episodes with lowest reward", success_low))
    lines.append("")
    lines.extend(describe(f"{display_name}: failed episodes with highest reward", failure_high))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_success_report(path: Path, csv_patterns: Sequence[str]) -> None:
    lines = [
        "Aggregate success/reward consistency report",
        "",
        "Each section is copied from the per-run evaluation_official/success_reward_consistency_report.txt file.",
    ]
    for csv_path in expand_paths(csv_patterns):
        report_path = Path(csv_path).with_name("success_reward_consistency_report.txt")
        if not report_path.exists():
            continue
        lines.append("")
        lines.append(f"## source={report_path}")
        lines.append(report_path.read_text(encoding="utf-8").rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_write_plots(out_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def scatter(x_field: str, y_field: str, filename: str) -> None:
        plt.figure(figsize=(9, 6))
        for method in sorted({str(r.get("method", "")) for r in rows}, key=method_sort_key):
            method_rows = [r for r in rows if r.get("method") == method]
            xs = [to_float(r.get(x_field)) for r in method_rows]
            ys = [to_float(r.get(y_field)) for r in method_rows]
            plt.scatter(xs, ys, s=12, alpha=0.55, label=method)
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()

    scatter("total_collisions", "reward_total", "collision_count_vs_reward_total.png")
    scatter("total_collisions", "reward_safety_penalty", "collision_count_vs_safety_penalty.png")
    scatter("mean_semantic_gap", "reward_total", "semantic_gap_vs_reward_total.png")
    scatter("mean_semantic_gap", "final_goal_distance", "semantic_gap_vs_final_distance.png")
    scatter("mean_force_ratio", "total_collisions", "force_ratio_vs_collision_count.png")
    scatter("avg_corr_action_norm", "total_collisions", "corrected_action_norm_vs_collision_count.png")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.patterns)
    if not rows:
        raise SystemExit("No reward_decomposition_eval.csv rows found.")

    base_fields = [
        "episode_key",
        "source_file",
        "reward_terminal_total",
        "reward_collision_family_penalty",
        "reward_safety_family_penalty",
    ]
    original_fields = [field for field in rows[0].keys() if field not in base_fields]
    combined_fields = base_fields + original_fields
    write_csv(out_dir / "reward_decomposition_eval.csv", rows, combined_fields)

    paired_fields = [
        "episode_key",
        "method",
        "train_seed",
        "eval_seed",
        "episode_id",
        "reward_total",
        "reward_progress",
        "reward_terminal_total",
        "reward_safety_penalty",
        "reward_collision_penalty",
        "reward_collision_family_penalty",
        "reward_sync",
        "reward_stagnation",
        "reward_energy",
        "success",
        "any_arrival",
        "two_arrival",
        "final_goal_distance",
        "total_collisions",
        "episode_length",
        "mean_semantic_gap",
        "max_semantic_gap",
        "mean_force_ratio",
        "avg_action_norm",
        "avg_corr_action_norm",
        "avg_action_delta_norm",
        "mean_pf_force_norm",
    ]
    write_csv(out_dir / "paired_eval_by_episode.csv", rows, paired_fields)

    summaries = summarize_by_method(rows)
    summary_fields = ["method", "episode_count", "train_seed_count", "eval_seed_count"] + [
        f"mean_{field}" for field in SUMMARY_FIELDS
    ]
    write_csv(out_dir / "method_reward_summary.csv", summaries, summary_fields)
    write_reward_gap_report(out_dir / "reward_gap_vs_ds_original_report.txt", summaries)
    write_paired_difference_report(out_dir / "paired_reward_difference_report.txt", rows)
    write_success_failure_distribution(out_dir / "success_failure_reward_distribution.csv", rows)
    write_candidate_case_report(out_dir / "candidate_reward_case_report.txt", rows)
    write_combined_success_report(out_dir / "success_reward_consistency_report.txt", args.patterns)

    scatter_fields = [
        "episode_key",
        "method",
        "success",
        "total_collisions",
        "reward_total",
        "reward_safety_penalty",
        "reward_collision_penalty",
        "final_goal_distance",
        "mean_semantic_gap",
        "mean_force_ratio",
        "avg_corr_action_norm",
    ]
    write_csv(out_dir / "collision_reward_scatter.csv", rows, scatter_fields)
    try_write_plots(out_dir, rows)
    print(f"Wrote reward diagnostics reports to {out_dir}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

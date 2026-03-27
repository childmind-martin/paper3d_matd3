#!/usr/bin/env python3
"""
Summarize a fixed-config multi-seed training sweep and generate plots.

Input can be:
1) a batch directory produced by seed_sweep_run.py
2) a manifest.json produced by seed_sweep_run.py
3) one or more explicit run directories containing episode_rewards.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(f"matplotlib is required: {exc}")

try:
    from scipy.ndimage import uniform_filter1d
except ImportError:
    uniform_filter1d = None

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a multi-seed run_optimized sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", nargs="?", default=None, help="Batch directory or manifest.json")
    parser.add_argument("--run-dir", action="append", default=[], help="Explicit run_dir containing episode_rewards.json")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tail", type=int, default=100, help="Tail window used for final-window statistics")
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def smooth_curve(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or window <= 1:
        return arr
    if uniform_filter1d is not None:
        return uniform_filter1d(arr, size=window, mode="nearest")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="same")


def rolling_mean_binary(flags: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(flags, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    out = np.zeros_like(arr, dtype=np.float64)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        out[idx] = arr[start:idx + 1].mean()
    return out


def t_critical(confidence: float, df: int) -> float:
    alpha = 1.0 - confidence
    if student_t is not None:
        return float(student_t.ppf(1.0 - alpha / 2.0, df))
    lookup = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    }
    if df in lookup:
        return lookup[df]
    return 1.96


def mean_std_ci(values: Sequence[float], confidence: float) -> Dict[str, Optional[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        half_width = t_critical(confidence, arr.size - 1) * std / math.sqrt(arr.size)
    else:
        half_width = 0.0
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci_low": float(mean - half_width),
        "ci_high": float(mean + half_width),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / float(total)
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def pad_series(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    max_len = max((len(series) for series in series_list), default=0)
    if max_len == 0:
        return np.empty((0, 0), dtype=np.float64)
    padded = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for row_idx, series in enumerate(series_list):
        arr = np.asarray(series, dtype=np.float64)
        padded[row_idx, :arr.size] = arr
    return padded


def nanmean_std(series_list: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    padded = pad_series(series_list)
    if padded.size == 0:
        return np.array([]), np.array([])
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0, ddof=0)


def resolve_min_clearance_series(raw_items: Sequence) -> List[float]:
    values: List[float] = []
    for item in raw_items:
        value: Optional[float]
        if isinstance(item, dict):
            raw_value = item.get("min", None)
            value = float(raw_value) if raw_value is not None else None
        elif item is None:
            value = None
        else:
            try:
                value = float(item)
            except Exception:
                value = None
        values.append(np.nan if value is None else value)
    return values


def resolve_run_entries(args: argparse.Namespace) -> Tuple[List[Dict], Optional[Path]]:
    if args.run_dir:
        entries: List[Dict] = []
        for idx, run_dir_str in enumerate(args.run_dir):
            run_dir = Path(run_dir_str).resolve()
            entries.append({
                "seed": None,
                "run_dir": str(run_dir),
                "status": "completed",
                "label": f"run_{idx + 1}",
            })
        return entries, None

    if args.source is None:
        raise ValueError("Provide either a batch directory / manifest.json source or one or more --run-dir values.")

    source = Path(args.source).resolve()
    if source.is_dir():
        manifest_path = source / "manifest.json"
    else:
        manifest_path = source

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = [row for row in manifest.get("runs", []) if row.get("status") == "completed" and row.get("run_dir")]
    return entries, manifest_path


@dataclass
class RunMetrics:
    seed: Optional[int]
    label: str
    run_dir: Path
    rewards: List[float]
    collisions: List[float]
    team_success_flags: List[int]
    agent_success_flags: List[List[int]]
    min_clearance: List[float]
    results: Dict


def load_run_metrics(entry: Dict) -> RunMetrics:
    run_dir = Path(entry["run_dir"]).resolve()
    metrics_path = run_dir / "episode_rewards.json"
    results_path = run_dir / "results.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing episode_rewards.json: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    results: Dict = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    rewards = [float(x) for x in metrics.get("episode_rewards", [])]
    collisions = [float(x) for x in metrics.get("collision_counts", [])]
    team_success_flags = [int(x) for x in metrics.get("team_success_flags", [])]
    agent_success_flags = [list(map(int, flags)) for flags in metrics.get("agent_success_flags", [])]
    min_clearance = resolve_min_clearance_series(metrics.get("min_distances_to_obstacle", []))

    seed = entry.get("seed", None)
    if seed is None:
        seed = (results.get("args", {}) or {}).get("seed", None)
    label = f"seed={seed}" if seed is not None else entry.get("label", run_dir.name)

    return RunMetrics(
        seed=int(seed) if seed is not None else None,
        label=label,
        run_dir=run_dir,
        rewards=rewards,
        collisions=collisions,
        team_success_flags=team_success_flags,
        agent_success_flags=agent_success_flags,
        min_clearance=min_clearance,
        results=results,
    )


def tail_slice(values: Sequence[float], tail: int) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    n = min(len(values), tail)
    return np.asarray(values[-n:], dtype=np.float64)


def compute_agent_tail_success(agent_flags: Sequence[Sequence[int]], tail: int) -> List[float]:
    if not agent_flags:
        return []
    n = min(len(agent_flags), tail)
    tail_flags = agent_flags[-n:]
    max_agents = max((len(flags) for flags in tail_flags), default=0)
    rates: List[float] = []
    for agent_idx in range(max_agents):
        hits = sum(1 for flags in tail_flags if len(flags) > agent_idx and flags[agent_idx] == 1)
        rates.append(hits / float(n) if n > 0 else 0.0)
    return rates


def build_run_summary(run: RunMetrics, tail: int) -> Dict:
    rewards = np.asarray(run.rewards, dtype=np.float64)
    collisions = np.asarray(run.collisions, dtype=np.float64)
    team_flags = np.asarray(run.team_success_flags, dtype=np.float64)
    clearances = np.asarray(run.min_clearance, dtype=np.float64)

    tail_rewards = tail_slice(run.rewards, tail)
    tail_collisions = tail_slice(run.collisions, tail)
    tail_team_flags = tail_slice(run.team_success_flags, tail)
    tail_clearances = tail_slice(run.min_clearance, tail)

    best_episode_idx = int(np.nanargmax(rewards)) if rewards.size else -1
    args_obj = run.results.get("args", {}) if isinstance(run.results, dict) else {}

    summary = {
        "seed": run.seed,
        "label": run.label,
        "run_dir": str(run.run_dir),
        "episodes": int(len(run.rewards)),
        "train_episodes": int(run.results.get("episodes", len(run.rewards))) if isinstance(run.results, dict) else int(len(run.rewards)),
        "scenario_seed": args_obj.get("terrain_seed", None),
        "tail_window": int(min(len(run.rewards), tail)),
        "final_reward": float(rewards[-1]) if rewards.size else None,
        "mean_reward": float(np.nanmean(rewards)) if rewards.size else None,
        "best_reward": float(np.nanmax(rewards)) if rewards.size else None,
        "best_episode_1based": best_episode_idx + 1 if best_episode_idx >= 0 else None,
        "full_run_team_success_rate": float(np.nanmean(team_flags)) if team_flags.size else None,
        "tail_reward_mean": float(np.nanmean(tail_rewards)) if tail_rewards.size else None,
        "tail_reward_std": float(np.nanstd(tail_rewards, ddof=1)) if tail_rewards.size > 1 else 0.0 if tail_rewards.size else None,
        "tail_team_success_rate": float(np.nanmean(tail_team_flags)) if tail_team_flags.size else None,
        "tail_collision_mean": float(np.nanmean(tail_collisions)) if tail_collisions.size else None,
        "tail_collision_std": float(np.nanstd(tail_collisions, ddof=1)) if tail_collisions.size > 1 else 0.0 if tail_collisions.size else None,
        "tail_min_clearance_mean": float(np.nanmean(tail_clearances)) if tail_clearances.size else None,
        "tail_min_clearance_std": float(np.nanstd(tail_clearances, ddof=1)) if tail_clearances.size > 1 else 0.0 if tail_clearances.size else None,
        "tail_agent_success_rates": compute_agent_tail_success(run.agent_success_flags, tail),
    }

    if tail_team_flags.size:
        successes = int(np.nansum(tail_team_flags))
        ci_low, ci_high = wilson_interval(successes, int(tail_team_flags.size))
        summary["tail_team_success_count"] = successes
        summary["tail_team_success_wilson_low"] = ci_low
        summary["tail_team_success_wilson_high"] = ci_high
    else:
        summary["tail_team_success_count"] = 0
        summary["tail_team_success_wilson_low"] = None
        summary["tail_team_success_wilson_high"] = None

    return summary


def ensure_output_dir(args: argparse.Namespace, manifest_path: Optional[Path]) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    elif manifest_path is not None:
        if manifest_path.name == "manifest.json":
            out_dir = manifest_path.parent / "summary"
        else:
            out_dir = manifest_path.parent / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        out_dir = Path("seed_sweep_summary_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_reward_overlay(runs: Sequence[RunMetrics], out_path: Path, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    reward_series = []
    for run in runs:
        rewards = np.asarray(run.rewards, dtype=np.float64)
        reward_series.append(rewards)
        episodes = np.arange(1, rewards.size + 1)
        ax.plot(episodes, rewards, alpha=0.12, linewidth=0.9)
        ax.plot(episodes, smooth_curve(rewards, smooth_window), linewidth=2.0, label=run.label)

    mean_curve, std_curve = nanmean_std(reward_series)
    if mean_curve.size:
        xs = np.arange(1, mean_curve.size + 1)
        ax.plot(xs, mean_curve, color="black", linewidth=3.0, label="Mean")
        ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, color="black", alpha=0.12, label="Mean ± std")

    ax.set_title("Reward Curves by Training Seed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_overlay(
    runs: Sequence[RunMetrics],
    out_path: Path,
    rolling_window: int,
    smooth_window: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    success_series = []
    collision_series = []
    clearance_series = []

    for run in runs:
        success = rolling_mean_binary(run.team_success_flags, rolling_window)
        collision = smooth_curve(run.collisions, smooth_window)
        clearance = smooth_curve(run.min_clearance, smooth_window)

        success_series.append(success)
        collision_series.append(collision)
        clearance_series.append(clearance)

        xs_success = np.arange(1, len(success) + 1)
        xs_collision = np.arange(1, len(collision) + 1)
        xs_clearance = np.arange(1, len(clearance) + 1)

        axes[0].plot(xs_success, success, linewidth=2.0, label=run.label)
        axes[1].plot(xs_collision, collision, linewidth=1.8, label=run.label)
        axes[2].plot(xs_clearance, clearance, linewidth=1.8, label=run.label)

    mean_success, _ = nanmean_std(success_series)
    mean_collision, _ = nanmean_std(collision_series)
    mean_clearance, _ = nanmean_std(clearance_series)

    if mean_success.size:
        axes[0].plot(np.arange(1, mean_success.size + 1), mean_success, color="black", linewidth=3.0, label="Mean")
    if mean_collision.size:
        axes[1].plot(np.arange(1, mean_collision.size + 1), mean_collision, color="black", linewidth=3.0, label="Mean")
    if mean_clearance.size:
        axes[2].plot(np.arange(1, mean_clearance.size + 1), mean_clearance, color="black", linewidth=3.0, label="Mean")

    axes[0].set_title(f"Rolling Team Success Rate (window={rolling_window})")
    axes[0].set_ylabel("Success Rate")
    axes[1].set_title(f"Collision Count (smoothed window={smooth_window})")
    axes[1].set_ylabel("Collisions")
    axes[2].set_title(f"Minimum Clearance (smoothed window={smooth_window})")
    axes[2].set_ylabel("Clearance (m)")
    axes[2].set_xlabel("Episode")
    axes[2].axhline(0.4, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Collision threshold")

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_tail_summary(summary_rows: Sequence[Dict], out_path: Path) -> None:
    seeds = [row.get("seed", idx + 1) for idx, row in enumerate(summary_rows)]
    x = np.arange(len(summary_rows))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        ("tail_reward_mean", "Final-window Reward Mean"),
        ("tail_team_success_rate", "Final-window Team Success Rate"),
        ("tail_collision_mean", "Final-window Collision Mean"),
        ("tail_min_clearance_mean", "Final-window Min Clearance Mean"),
    ]

    for ax, (key, title) in zip(axes.flat, specs):
        values = np.asarray([row.get(key, np.nan) for row in summary_rows], dtype=np.float64)
        ax.scatter(x, values, s=80, alpha=0.85)
        ax.plot(x, values, alpha=0.5)
        mean_value = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        std_value = float(np.nanstd(values, ddof=1)) if np.count_nonzero(np.isfinite(values)) > 1 else 0.0
        ax.axhline(mean_value, color="black", linewidth=2.0, linestyle="-", label=f"Mean={mean_value:.4f}")
        ax.axhspan(mean_value - std_value, mean_value + std_value, color="black", alpha=0.08, label="Mean ± std")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(seed) for seed in seeds], rotation=0)
        ax.set_xlabel("Seed")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_success_rate_ci(summary_rows: Sequence[Dict], aggregate_ci: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    y_positions = np.arange(len(summary_rows) + 1)

    for idx, row in enumerate(summary_rows):
        center = row.get("tail_team_success_rate", 0.0)
        low = row.get("tail_team_success_wilson_low", center)
        high = row.get("tail_team_success_wilson_high", center)
        left = max(0.0, float(center) - float(low))
        right = max(0.0, float(high) - float(center))
        ax.errorbar(
            x=center,
            y=idx,
            xerr=[[left], [right]],
            fmt="o",
            capsize=4,
            label=None,
            color="tab:blue",
        )

    if aggregate_ci.get("mean") is not None:
        center = aggregate_ci["mean"]
        low = aggregate_ci["ci_low"]
        high = aggregate_ci["ci_high"]
        left = max(0.0, float(center) - float(low))
        right = max(0.0, float(high) - float(center))
        ax.errorbar(
            x=center,
            y=len(summary_rows),
            xerr=[[left], [right]],
            fmt="s",
            capsize=5,
            color="black",
            markersize=8,
        )

    y_labels = [f"seed={row.get('seed', idx + 1)}" for idx, row in enumerate(summary_rows)] + ["across-seed mean"]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Final-window Team Success Rate")
    ax.set_title("Final-window Team Success Rate Intervals")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    entries, manifest_path = resolve_run_entries(args)
    if not entries:
        raise SystemExit("No completed run entries were found.")

    output_dir = ensure_output_dir(args, manifest_path)
    runs = [load_run_metrics(entry) for entry in entries]
    runs.sort(key=lambda run: (run.seed is None, run.seed if run.seed is not None else run.label))

    summary_rows = [build_run_summary(run, args.tail) for run in runs]

    aggregate = {
        "tail_reward_mean": mean_std_ci([row["tail_reward_mean"] for row in summary_rows if row["tail_reward_mean"] is not None], args.ci_level),
        "tail_team_success_rate": mean_std_ci([row["tail_team_success_rate"] for row in summary_rows if row["tail_team_success_rate"] is not None], args.ci_level),
        "tail_collision_mean": mean_std_ci([row["tail_collision_mean"] for row in summary_rows if row["tail_collision_mean"] is not None], args.ci_level),
        "tail_min_clearance_mean": mean_std_ci([row["tail_min_clearance_mean"] for row in summary_rows if row["tail_min_clearance_mean"] is not None], args.ci_level),
        "full_run_team_success_rate": mean_std_ci([row["full_run_team_success_rate"] for row in summary_rows if row["full_run_team_success_rate"] is not None], args.ci_level),
    }

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(manifest_path) if manifest_path is not None else None,
        "tail": args.tail,
        "smooth_window": args.smooth_window,
        "rolling_window": args.rolling_window,
        "runs": summary_rows,
        "aggregate": aggregate,
    }

    write_csv(output_dir / "seed_summary.csv", summary_rows)
    with open(output_dir / "seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    plot_reward_overlay(runs, output_dir / "reward_overlay.png", args.smooth_window)
    plot_metrics_overlay(runs, output_dir / "metrics_overlay.png", args.rolling_window, args.smooth_window)
    plot_final_tail_summary(summary_rows, output_dir / "final_tail_summary.png")
    plot_success_rate_ci(summary_rows, aggregate["tail_team_success_rate"], output_dir / "final_tail_success_rate_ci.png")

    print(f"Summary directory: {output_dir}")
    print(f"  - CSV : {output_dir / 'seed_summary.csv'}")
    print(f"  - JSON: {output_dir / 'seed_summary.json'}")
    print(f"  - Plot: {output_dir / 'reward_overlay.png'}")
    print(f"  - Plot: {output_dir / 'metrics_overlay.png'}")
    print(f"  - Plot: {output_dir / 'final_tail_summary.png'}")
    print(f"  - Plot: {output_dir / 'final_tail_success_rate_ci.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

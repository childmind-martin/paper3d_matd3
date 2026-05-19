#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = [
    "matd3_full_dual_semantic",
    "matd3_collapsed_replay",
    "matd3_no_corrected_target_reconstruction",
]

DISPLAY_NAMES = {
    "matd3_full_dual_semantic": "Full Dual-Semantic",
    "matd3_collapsed_replay": "Collapsed Replay",
    "matd3_no_corrected_target_reconstruction": "No Corrected Target Recon",
}

COLORS = {
    "matd3_full_dual_semantic": "#1F77B4",
    "matd3_collapsed_replay": "#B07AA1",
    "matd3_no_corrected_target_reconstruction": "#E45756",
}


@dataclass
class RunCurve:
    label: str
    seed: int
    model_dir: Path
    rewards: np.ndarray
    team_success: np.ndarray
    force_ratios: np.ndarray


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=np.float64)
    return np.asarray([_safe_float(v) for v in values], dtype=np.float64)


def _as_binary_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=np.float64)
    return np.asarray([1.0 if _safe_int(v) > 0 else 0.0 for v in values], dtype=np.float64)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, int(window))
    if window == 1:
        return values.astype(np.float64, copy=True)
    weights = np.ones(window, dtype=np.float64)
    valid = np.isfinite(values).astype(np.float64)
    clean = np.where(np.isfinite(values), values, 0.0)
    numerator = np.convolve(clean, weights, mode="same")
    denominator = np.convolve(valid, weights, mode="same")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _label_from_dir(path: Path, run_tag: str) -> Optional[str]:
    prefix = f"{run_tag}_"
    name = path.name
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    match = re.search(r"_seed\d+_", rest)
    if not match:
        return None
    return rest[: match.start()]


def _seed_from_dir(path: Path) -> Optional[int]:
    match = re.search(r"_seed(\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1))


def _load_curves(model_root: Path, run_tag: str, labels: Sequence[str], expected_episodes: int) -> List[RunCurve]:
    curves: List[RunCurve] = []
    label_set = set(labels)
    for model_dir in sorted(model_root.glob(f"{run_tag}_*_seed*_*")):
        if not model_dir.is_dir():
            continue
        label = _label_from_dir(model_dir, run_tag)
        seed = _seed_from_dir(model_dir)
        if label not in label_set or seed is None:
            continue
        state_path = model_dir / "checkpoint" / "checkpoint_state.json"
        if not state_path.exists():
            continue
        data = json.loads(state_path.read_text(encoding="utf-8"))
        rewards = _as_float_array(data.get("episode_rewards"))
        team_success = _as_binary_array(data.get("team_success_flags", data.get("success_flags")))
        force_ratios = _as_float_array(data.get("episode_force_ratios"))
        n = min(rewards.size, team_success.size, force_ratios.size)
        if n <= 0:
            continue
        if expected_episodes > 0 and n < expected_episodes:
            continue
        n = expected_episodes if expected_episodes > 0 else n
        curves.append(
            RunCurve(
                label=label,
                seed=seed,
                model_dir=model_dir,
                rewards=rewards[:n],
                team_success=team_success[:n],
                force_ratios=force_ratios[:n],
            )
        )
    return curves


def _group_curves(curves: Sequence[RunCurve], labels: Sequence[str]) -> Dict[str, List[RunCurve]]:
    grouped: Dict[str, List[RunCurve]] = {label: [] for label in labels}
    for curve in curves:
        grouped.setdefault(curve.label, []).append(curve)
    for items in grouped.values():
        items.sort(key=lambda item: item.seed)
    return grouped


def _stack_metric(items: Sequence[RunCurve], metric: str, smooth_window: int = 1) -> np.ndarray:
    rows = []
    for item in items:
        values = getattr(item, metric)
        if smooth_window > 1:
            values = _moving_average(values, smooth_window)
        rows.append(values)
    return np.vstack(rows) if rows else np.asarray([], dtype=np.float64)


def _plot_metric(
    grouped: Dict[str, List[RunCurve]],
    labels: Sequence[str],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for label in labels:
        items = grouped.get(label, [])
        if not items:
            continue
        stacked = _stack_metric(items, metric, smooth_window=smooth_window)
        if stacked.size == 0:
            continue
        episodes = np.arange(1, stacked.shape[1] + 1)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        color = COLORS.get(label, "#4C78A8")
        ax.plot(episodes, mean, label=DISPLAY_NAMES.get(label, label), color=color, linewidth=2.2)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(
    grouped: Dict[str, List[RunCurve]],
    labels: Sequence[str],
    output_path: Path,
    reward_window: int,
    success_window: int,
    fr_window: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True)
    specs = [
        ("rewards", reward_window, "Smoothed Reward", None),
        ("team_success", success_window, f"Rolling Team Success Rate (window={success_window})", (0.0, 0.60)),
        ("force_ratios", fr_window, "Action Force Ratio", (0.28, 0.52)),
    ]
    for ax, (metric, window, ylabel, ylim) in zip(axes, specs):
        for label in labels:
            items = grouped.get(label, [])
            if not items:
                continue
            stacked = _stack_metric(items, metric, smooth_window=window)
            if stacked.size == 0:
                continue
            episodes = np.arange(1, stacked.shape[1] + 1)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            color = COLORS.get(label, "#4C78A8")
            ax.plot(episodes, mean, label=DISPLAY_NAMES.get(label, label), color=color, linewidth=2.0)
            ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.28, linestyle="--")
    axes[0].set_title("Level2 Dual-Semantic Ablation Training Curves (Mean ± Std)", fontweight="bold", fontsize=15)
    axes[-1].set_xlabel("Training Episode")
    axes[0].legend(frameon=True, ncol=3, loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(grouped: Dict[str, List[RunCurve]], labels: Sequence[str], path: Path, success_window: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "display_name",
        "seed",
        "episodes",
        "final_reward",
        "avg_reward_last_100",
        "team_success_rate",
        "team_success_rate_last_100",
        "best_rolling_team_success_rate",
        "best_rolling_team_success_episode",
        "final_force_ratio",
        "model_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            for item in grouped.get(label, []):
                rolling_success = _moving_average(item.team_success, success_window)
                if rolling_success.size and np.any(np.isfinite(rolling_success)):
                    best_idx = int(np.nanargmax(rolling_success))
                    best_value = float(rolling_success[best_idx])
                    best_episode = best_idx + 1
                else:
                    best_value = math.nan
                    best_episode = 0
                writer.writerow(
                    {
                        "label": label,
                        "display_name": DISPLAY_NAMES.get(label, label),
                        "seed": item.seed,
                        "episodes": int(item.rewards.size),
                        "final_reward": float(item.rewards[-1]),
                        "avg_reward_last_100": float(np.nanmean(item.rewards[-100:])),
                        "team_success_rate": float(np.nanmean(item.team_success)),
                        "team_success_rate_last_100": float(np.nanmean(item.team_success[-100:])),
                        "best_rolling_team_success_rate": best_value,
                        "best_rolling_team_success_episode": best_episode,
                        "final_force_ratio": float(item.force_ratios[-1]),
                        "model_dir": str(item.model_dir),
                    }
                )


def _write_curve_csv(
    grouped: Dict[str, List[RunCurve]],
    labels: Sequence[str],
    path: Path,
    reward_window: int,
    success_window: int,
    fr_window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "display_name",
        "episode",
        "reward_mean",
        "reward_std",
        "rolling_team_success_mean",
        "rolling_team_success_std",
        "force_ratio_mean",
        "force_ratio_std",
        "seed_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            items = grouped.get(label, [])
            if not items:
                continue
            rewards = _stack_metric(items, "rewards", smooth_window=reward_window)
            team_success = _stack_metric(items, "team_success", smooth_window=success_window)
            force_ratios = _stack_metric(items, "force_ratios", smooth_window=fr_window)
            n = min(rewards.shape[1], team_success.shape[1], force_ratios.shape[1])
            for idx in range(n):
                writer.writerow(
                    {
                        "label": label,
                        "display_name": DISPLAY_NAMES.get(label, label),
                        "episode": idx + 1,
                        "reward_mean": float(np.nanmean(rewards[:, idx])),
                        "reward_std": float(np.nanstd(rewards[:, idx])),
                        "rolling_team_success_mean": float(np.nanmean(team_success[:, idx])),
                        "rolling_team_success_std": float(np.nanstd(team_success[:, idx])),
                        "force_ratio_mean": float(np.nanmean(force_ratios[:, idx])),
                        "force_ratio_std": float(np.nanstd(force_ratios[:, idx])),
                        "seed_count": len(items),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Level2 dual-semantics training curves from completed checkpoints.")
    parser.add_argument("--model-root", default="/home/tang/matd3/models")
    parser.add_argument("--run-tag", default="level2_dual_semantics_ablation")
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--reward-window", type=int, default=50)
    parser.add_argument("--success-window", type=int, default=50)
    parser.add_argument("--fr-window", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/tang/matd3/ablation_experiments/multi_seed_groupB_20260428_155347/plots",
    )
    parser.add_argument("--timestamp", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    labels = list(args.labels)
    curves = _load_curves(model_root, args.run_tag, labels, int(args.expected_episodes))
    grouped = _group_curves(curves, labels)
    missing = [label for label in labels if not grouped.get(label)]
    if missing:
        print(f"Missing labels: {missing}")
        return 2

    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S")
    reward_png = output_dir / f"dual_semantics_train_reward_curve_{timestamp}.png"
    success_png = output_dir / f"dual_semantics_train_team_sr_curve_{timestamp}.png"
    fr_png = output_dir / f"dual_semantics_train_force_ratio_curve_{timestamp}.png"
    combined_png = output_dir / f"dual_semantics_train_reward_success_fr_curve_{timestamp}.png"
    summary_csv = output_dir / f"dual_semantics_training_seed_summary_{timestamp}.csv"
    curve_csv = output_dir / f"dual_semantics_training_curve_points_{timestamp}.csv"

    _plot_metric(
        grouped,
        labels,
        "rewards",
        f"Level2 Dual-Semantic Ablation Reward Curves (window={args.reward_window})",
        "Episode Reward",
        reward_png,
        smooth_window=int(args.reward_window),
    )
    _plot_metric(
        grouped,
        labels,
        "team_success",
        f"Level2 Dual-Semantic Ablation Rolling Team SR (window={args.success_window})",
        "Rolling Team Success Rate",
        success_png,
        smooth_window=int(args.success_window),
        ylim=(0.0, 0.60),
    )
    _plot_metric(
        grouped,
        labels,
        "force_ratios",
        "Level2 Dual-Semantic Ablation Action Force Ratio",
        "Action Force Ratio",
        fr_png,
        smooth_window=int(args.fr_window),
        ylim=(0.28, 0.52),
    )
    _plot_combined(
        grouped,
        labels,
        combined_png,
        reward_window=int(args.reward_window),
        success_window=int(args.success_window),
        fr_window=int(args.fr_window),
    )
    _write_summary_csv(grouped, labels, summary_csv, success_window=int(args.success_window))
    _write_curve_csv(
        grouped,
        labels,
        curve_csv,
        reward_window=int(args.reward_window),
        success_window=int(args.success_window),
        fr_window=int(args.fr_window),
    )

    print(f"Loaded runs: {len(curves)}")
    for label in labels:
        seeds = [item.seed for item in grouped.get(label, [])]
        print(f"  {DISPLAY_NAMES.get(label, label)}: seeds={seeds}")
    print("Artifacts:")
    for path in (reward_png, success_png, fr_png, combined_png, summary_csv, curve_csv):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

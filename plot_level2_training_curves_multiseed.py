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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_level2_official_eval import COLOR_MAP, DEFAULT_LABELS, DISPLAY_NAME_MAP, DUAL_SEMANTICS_LABELS


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


def _float_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=np.float64)
    return np.asarray([_safe_float(value) for value in values], dtype=np.float64)


def _binary_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=np.float64)
    return np.asarray([1.0 if _safe_int(value) > 0 else 0.0 for value in values], dtype=np.float64)


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


def _display_name(label: str) -> str:
    return DISPLAY_NAME_MAP.get(label, label)


def _model_dirs_newest_first(model_root: Path, run_tag: str, label: str, seed: int) -> List[Path]:
    pattern = f"{run_tag}_{label}_seed{int(seed)}_*"
    candidates = [path for path in model_root.glob(pattern) if path.is_dir()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _read_checkpoint_state(model_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = [model_dir / "checkpoint" / "checkpoint_state.json"]
    candidates.extend(sorted(model_dir.glob("*/checkpoint_state.json"), key=lambda path: path.stat().st_mtime, reverse=True))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _load_curve_from_model(model_dir: Path, label: str, seed: int, expected_episodes: int) -> Optional[RunCurve]:
    state = _read_checkpoint_state(model_dir)
    if not state:
        return None
    rewards = _float_array(state.get("episode_rewards"))
    team_success = _binary_array(state.get("team_success_flags", state.get("success_flags")))
    force_ratios = _float_array(state.get("episode_force_ratios"))
    n = min(rewards.size, team_success.size)
    if force_ratios.size:
        n = min(n, force_ratios.size)
    if n <= 0:
        return None
    if expected_episodes > 0 and n < expected_episodes:
        return None
    n = int(expected_episodes) if expected_episodes > 0 else int(n)
    if force_ratios.size == 0:
        force_ratios = np.full(n, np.nan, dtype=np.float64)
    return RunCurve(
        label=label,
        seed=int(seed),
        model_dir=model_dir,
        rewards=rewards[:n],
        team_success=team_success[:n],
        force_ratios=force_ratios[:n],
    )


def _load_curves(
    model_root: Path,
    run_tag: str,
    labels: Sequence[str],
    seeds: Sequence[int],
    expected_episodes: int,
) -> Tuple[List[RunCurve], List[Dict[str, Any]]]:
    curves: List[RunCurve] = []
    missing: List[Dict[str, Any]] = []
    for label in labels:
        for seed in seeds:
            loaded: Optional[RunCurve] = None
            candidates = _model_dirs_newest_first(model_root, run_tag, label, int(seed))
            for model_dir in candidates:
                loaded = _load_curve_from_model(model_dir, label, int(seed), expected_episodes)
                if loaded is not None:
                    break
            if loaded is None:
                missing.append(
                    {
                        "label": label,
                        "seed": int(seed),
                        "candidate_count": len(candidates),
                    }
                )
                continue
            curves.append(loaded)
    return curves, missing


def _group_curves(curves: Sequence[RunCurve], labels: Sequence[str]) -> Dict[str, List[RunCurve]]:
    grouped: Dict[str, List[RunCurve]] = {label: [] for label in labels}
    for curve in curves:
        grouped.setdefault(curve.label, []).append(curve)
    for items in grouped.values():
        items.sort(key=lambda item: item.seed)
    return grouped


def _stack(items: Sequence[RunCurve], metric: str, window: int) -> np.ndarray:
    rows = []
    for item in items:
        values = getattr(item, metric)
        if window > 1:
            values = _moving_average(values, window)
        rows.append(values)
    return np.vstack(rows) if rows else np.asarray([], dtype=np.float64)


def _plot_metric(
    grouped: Dict[str, List[RunCurve]],
    labels: Sequence[str],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    window: int,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    for label in labels:
        stacked = _stack(grouped.get(label, []), metric, window)
        if stacked.size == 0:
            continue
        episodes = np.arange(1, stacked.shape[1] + 1)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        color = COLOR_MAP.get(label, "#4C78A8")
        ax.plot(episodes, mean, label=_display_name(label), color=color, linewidth=1.9)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.legend(frameon=True, fontsize=8, ncol=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_run_summary(curves: Sequence[RunCurve], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "display_name",
            "seed",
            "episodes",
            "reward_mean",
            "reward_final",
            "team_success_mean",
            "team_success_final",
            "force_ratio_initial",
            "force_ratio_final",
            "model_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for curve in sorted(curves, key=lambda item: (item.label, item.seed)):
            writer.writerow(
                {
                    "label": curve.label,
                    "display_name": _display_name(curve.label),
                    "seed": curve.seed,
                    "episodes": int(curve.rewards.size),
                    "reward_mean": float(np.nanmean(curve.rewards)),
                    "reward_final": float(curve.rewards[-1]),
                    "team_success_mean": float(np.nanmean(curve.team_success)),
                    "team_success_final": float(curve.team_success[-1]),
                    "force_ratio_initial": float(curve.force_ratios[0]) if curve.force_ratios.size else None,
                    "force_ratio_final": float(curve.force_ratios[-1]) if curve.force_ratios.size else None,
                    "model_dir": str(curve.model_dir),
                }
            )


def _write_curve_points(
    grouped: Dict[str, List[RunCurve]],
    labels: Sequence[str],
    path: Path,
    reward_window: int,
    success_window: int,
    fr_window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "display_name",
            "episode",
            "seed_count",
            "reward_mean",
            "reward_std",
            "rolling_team_success_mean",
            "rolling_team_success_std",
            "force_ratio_mean",
            "force_ratio_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            items = grouped.get(label, [])
            if not items:
                continue
            rewards = _stack(items, "rewards", reward_window)
            success = _stack(items, "team_success", success_window)
            force_ratios = _stack(items, "force_ratios", fr_window)
            n = min(rewards.shape[1], success.shape[1], force_ratios.shape[1])
            for idx in range(n):
                writer.writerow(
                    {
                        "label": label,
                        "display_name": _display_name(label),
                        "episode": idx + 1,
                        "seed_count": len(items),
                        "reward_mean": float(np.nanmean(rewards[:, idx])),
                        "reward_std": float(np.nanstd(rewards[:, idx])),
                        "rolling_team_success_mean": float(np.nanmean(success[:, idx])),
                        "rolling_team_success_std": float(np.nanstd(success[:, idx])),
                        "force_ratio_mean": float(np.nanmean(force_ratios[:, idx])),
                        "force_ratio_std": float(np.nanstd(force_ratios[:, idx])),
                    }
                )


def _default_labels_for_run_tag(run_tag: str) -> List[str]:
    if "dual_semantics" in run_tag or "semantic" in run_tag:
        return list(DUAL_SEMANTICS_LABELS)
    return list(DEFAULT_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Level-2 multi-seed training curves from completed checkpoints.")
    parser.add_argument("--model-root", type=Path, default=Path("/home/tang/matd3/models"))
    parser.add_argument("--run-tag", default="level2_ms_official")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--reward-window", type=int, default=50)
    parser.add_argument("--success-window", type=int, default=50)
    parser.add_argument("--fr-window", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = list(args.labels) if args.labels is not None and len(args.labels) > 0 else _default_labels_for_run_tag(args.run_tag)
    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(f"/home/tang/matd3/diagnostics/{args.run_tag}_training_curves_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    curves, missing = _load_curves(
        model_root=args.model_root.resolve(),
        run_tag=str(args.run_tag),
        labels=labels,
        seeds=list(args.seeds),
        expected_episodes=int(args.expected_episodes),
    )
    if missing and args.strict:
        missing_text = ", ".join(f"{item['label']}@{item['seed']}" for item in missing)
        raise SystemExit(f"Missing complete training curves: {missing_text}")
    if not curves:
        raise SystemExit("No complete training curves found.")

    grouped = _group_curves(curves, labels)
    reward_png = output_dir / f"{args.run_tag}_train_reward_curve_{timestamp}.png"
    success_png = output_dir / f"{args.run_tag}_train_team_sr_curve_{timestamp}.png"
    fr_png = output_dir / f"{args.run_tag}_train_force_ratio_curve_{timestamp}.png"
    run_summary_csv = output_dir / f"{args.run_tag}_training_seed_summary_{timestamp}.csv"
    curve_points_csv = output_dir / f"{args.run_tag}_training_curve_points_{timestamp}.csv"
    missing_json = output_dir / f"{args.run_tag}_training_missing_{timestamp}.json"

    _plot_metric(
        grouped,
        labels,
        "rewards",
        f"{args.run_tag} Training Reward Curves (Mean +/- Std, window={args.reward_window})",
        "Episode Reward",
        reward_png,
        int(args.reward_window),
    )
    _plot_metric(
        grouped,
        labels,
        "team_success",
        f"{args.run_tag} Rolling Team Success (Mean +/- Std, window={args.success_window})",
        "Rolling Team Success Rate",
        success_png,
        int(args.success_window),
        ylim=(0.0, 1.0),
    )
    _plot_metric(
        grouped,
        labels,
        "force_ratios",
        f"{args.run_tag} Action Force Ratio Schedule (Mean +/- Std)",
        "Action Force Ratio",
        fr_png,
        int(args.fr_window),
    )
    _write_run_summary(curves, run_summary_csv)
    _write_curve_points(grouped, labels, curve_points_csv, int(args.reward_window), int(args.success_window), int(args.fr_window))
    missing_json.write_text(json.dumps({"missing": missing}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Output dir: {output_dir}")
    print(f"Reward curve: {reward_png}")
    print(f"Team-success curve: {success_png}")
    print(f"Force-ratio curve: {fr_png}")
    print(f"Run summary CSV: {run_summary_csv}")
    print(f"Curve points CSV: {curve_points_csv}")
    if missing:
        print("Missing complete curves:")
        for item in missing:
            print(f"  - {item['label']} @ seed {item['seed']} (candidates={item['candidate_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

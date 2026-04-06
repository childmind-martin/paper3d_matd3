#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared color palette for algorithm-ablation plots."""

from __future__ import annotations

from typing import Optional


# Keep the same algorithm label on the same color across reward/loss/success plots.
ALGORITHM_ABLATION_COLOR_BY_LABEL = {
    "matd3_separated_gradient": "#0066CC",  # blue
    "matd3_dual_q": "#CC0000",              # red
    "matd3_separated_hybrid_actor": "#B8860B",  # dark goldenrod
    "matd3_single_q": "#00AA00",            # green
    "maddpg_separated_gradient": "#9900CC", # purple
    "maddpg_dual_q": "#00CCCC",             # cyan
    "maddpg_baseline": "#FF8800",           # orange
    "per_uniform_baseline": "#7F7F7F",      # gray
    "per_improved_mainline": "#0066CC",     # blue
}

ALGORITHM_ABLATION_FALLBACK_COLORS = [
    "#0066CC",
    "#CC0000",
    "#00AA00",
    "#9900CC",
    "#00CCCC",
    "#FF8800",
    "#8B4513",
    "#7F7F7F",
]


def get_algorithm_ablation_color(
    label: Optional[str],
    idx: int = 0,
    default: str = "#444444",
) -> str:
    normalized_label = str(label or "").strip()
    if normalized_label in ALGORITHM_ABLATION_COLOR_BY_LABEL:
        return ALGORITHM_ABLATION_COLOR_BY_LABEL[normalized_label]
    if not ALGORITHM_ABLATION_FALLBACK_COLORS:
        return default
    return ALGORITHM_ABLATION_FALLBACK_COLORS[idx % len(ALGORITHM_ABLATION_FALLBACK_COLORS)]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared visual palette for algorithm-ablation plots."""

from __future__ import annotations

from typing import Any, Dict, Optional


# Keep the same algorithm label on the same color and linestyle across
# reward/loss/success/post-eval plots.
ALGORITHM_ABLATION_STYLE_BY_LABEL: Dict[str, Dict[str, Any]] = {
    "matd3_separated_gradient": {
        "color": "#1F77B4",  # blue
        "linestyle": "-",
        "marker": "o",
        "hatch": "",
    },
    "matd3_dual_q": {
        "color": "#D62728",  # red
        "linestyle": (0, (3.2, 1.6)),
        "marker": "s",
        "hatch": "//",
    },
    "matd3_separated_hybrid_actor": {
        "color": "#FF7F0E",  # orange
        "linestyle": (0, (8.0, 2.4)),
        "marker": "D",
        "hatch": "xx",
    },
    "matd3_separated_hybrid_actor_alpha20": {
        "color": "#BCBD22",  # olive gold
        "linestyle": (0, (6.0, 1.5, 1.6, 1.5)),
        "marker": "P",
        "hatch": "++",
    },
    "matd3_single_q": {
        "color": "#2CA02C",  # green
        "linestyle": ":",
        "marker": "^",
        "hatch": "..",
    },
    "matd3_full_dual_semantic": {
        "color": "#1F77B4",  # blue
        "linestyle": "-",
        "marker": "o",
        "hatch": "",
    },
    "matd3_collapsed_replay": {
        "color": "#B279A2",  # muted mauve
        "linestyle": (0, (3.0, 1.5)),
        "marker": "s",
        "hatch": "//",
    },
    "matd3_no_corrected_target_reconstruction": {
        "color": "#E45756",  # coral red
        "linestyle": (0, (7.0, 2.0)),
        "marker": "D",
        "hatch": "xx",
    },
    "matd3_full_dual_semantic_cross_agent_ref": {
        "color": "#2CA02C",  # green
        "linestyle": "-",
        "marker": "^",
        "hatch": "",
    },
    "matd3_cross_agent_ref_agent_success": {
        "color": "#009E73",  # teal green
        "linestyle": "-",
        "marker": "^",
        "hatch": "",
    },
    "matd3_cross_agent_ref_agent_quality": {
        "color": "#56B4E9",  # sky blue
        "linestyle": (0, (5.0, 1.8)),
        "marker": "o",
        "hatch": "..",
    },
    "matd3_cross_agent_ref_soft_advantage": {
        "color": "#CC79A7",  # magenta
        "linestyle": (0, (6.0, 2.0)),
        "marker": "v",
        "hatch": "oo",
    },
    "matd3_cross_agent_ref_selector_mix": {
        "color": "#D55E00",  # vermillion
        "linestyle": "-",
        "marker": "*",
        "hatch": "**",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_tail0": {
        "color": "#009E73",  # green
        "linestyle": (0, (1.5, 1.5)),
        "marker": "P",
        "hatch": "++",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_tail01": {
        "color": "#0072B2",  # blue
        "linestyle": (0, (3.0, 1.6)),
        "marker": "D",
        "hatch": "//",
    },
    "matd3_cross_agent_ref_reward_to_success_selector": {
        "color": "#6A3D9A",  # deep purple
        "linestyle": (0, (4.5, 1.6, 1.2, 1.6)),
        "marker": "h",
        "hatch": "OO",
    },
    "matd3_cross_agent_ref_reward_to_success_head_tail_selector": {
        "color": "#56B4E9",  # sky blue
        "linestyle": (0, (5.0, 1.4)),
        "marker": "s",
        "hatch": "..",
    },
    "matd3_cross_agent_ref_closed_loop_team_head_tail_selector": {
        "color": "#009E73",  # teal green
        "linestyle": (0, (6.0, 1.5, 1.2, 1.5)),
        "marker": "P",
        "hatch": "++",
    },
    "matd3_cross_agent_ref_reward_to_success_head_tail_split_teacher_selector": {
        "color": "#E69F00",  # orange
        "linestyle": (0, (2.0, 1.2)),
        "marker": "X",
        "hatch": "xx",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_fr_current": {
        "color": "#6A3D9A",  # deep purple
        "linestyle": "-",
        "marker": "h",
        "hatch": "",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_fr_floor40": {
        "color": "#9467BD",  # purple
        "linestyle": (0, (5.0, 1.6)),
        "marker": "D",
        "hatch": "//",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_fr_fixed045": {
        "color": "#B279A2",  # muted mauve
        "linestyle": (0, (2.0, 1.4)),
        "marker": "P",
        "hatch": "++",
    },
    "matd3_cross_agent_ref_reward_to_success_head_tail_selector_fr_current": {
        "color": "#0072B2",  # blue
        "linestyle": "-",
        "marker": "s",
        "hatch": "",
    },
    "matd3_cross_agent_ref_reward_to_success_head_tail_selector_fr_floor40": {
        "color": "#009E73",  # teal green
        "linestyle": (0, (5.0, 1.6)),
        "marker": "^",
        "hatch": "..",
    },
    "matd3_cross_agent_ref_reward_to_success_head_tail_selector_fr_fixed045": {
        "color": "#56B4E9",  # sky blue
        "linestyle": (0, (2.0, 1.4)),
        "marker": "v",
        "hatch": "oo",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_clean_label": {
        "color": "#9467BD",  # purple
        "linestyle": (0, (2.0, 1.2)),
        "marker": "8",
        "hatch": "\\\\",
    },
    "matd3_cross_agent_ref_reward_to_success_selector_tail10": {
        "color": "#E69F00",  # amber
        "linestyle": (0, (7.0, 2.0)),
        "marker": "X",
        "hatch": "xx",
    },
    "matd3_cross_agent_ref_progress_gate": {
        "color": "#2CA02C",  # legacy green
        "linestyle": (0, (2.0, 1.6)),
        "marker": "^",
        "hatch": "\\\\",
    },
    "matd3_cross_agent_ref_agent_success_behavior_label": {
        "color": "#8C564B",  # brown gray
        "linestyle": ":",
        "marker": "X",
        "hatch": "--",
    },
    "matd3_cross_agent_ref_no_quality_gate": {
        "color": "#FF7F0E",  # orange
        "linestyle": (0, (4.0, 1.8)),
        "marker": "P",
        "hatch": "++",
    },
    "matd3_cross_agent_ref_behavior_label": {
        "color": "#7F7F7F",  # neutral gray
        "linestyle": ":",
        "marker": "X",
        "hatch": "--",
    },
    "matd3_cross_agent_ref_behavior_label_current_baseline": {
        "color": "#4D4D4D",  # dark neutral gray
        "linestyle": "-",
        "marker": "o",
        "hatch": "",
    },
    "matd3_cross_agent_ref_behavior_label_agent_quality_gate": {
        "color": "#4D4D4D",  # neutral control
        "linestyle": "-",
        "marker": "o",
        "hatch": "",
    },
    "matd3_cross_agent_ref_aqual_split_teacher": {
        "color": "#0072B2",  # blue
        "linestyle": (0, (5.0, 1.6)),
        "marker": "s",
        "hatch": "//",
    },
    "matd3_cross_agent_ref_adaptive_twin_advantage": {
        "color": "#E69F00",  # amber
        "linestyle": (0, (2.0, 1.2)),
        "marker": "D",
        "hatch": "xx",
    },
    "matd3_cross_agent_ref_shared_twin_advantage_selector": {
        "color": "#009E73",  # teal green
        "linestyle": (0, (6.0, 1.5, 1.2, 1.5)),
        "marker": "P",
        "hatch": "++",
    },
    "matd3_cross_agent_ref_behavior_label_agent_quality_safe_reward": {
        "color": "#D55E00",  # vermillion
        "linestyle": (0, (2.0, 1.2)),
        "marker": "D",
        "hatch": "xx",
    },
    "matd3_cross_agent_ref_behavior_label_team_feedback_selector_v2": {
        "color": "#009E73",  # teal green
        "linestyle": (0, (6.0, 1.5, 1.2, 1.5)),
        "marker": "P",
        "hatch": "++",
    },
    "maddpg_separated_gradient": {
        "color": "#9467BD",  # purple
        "linestyle": "-",
        "marker": "v",
        "hatch": "",
    },
    "maddpg_dual_q": {
        "color": "#17BECF",  # cyan
        "linestyle": "--",
        "marker": "<",
        "hatch": "\\\\",
    },
    "maddpg_baseline": {
        "color": "#7F7F7F",  # neutral gray
        "linestyle": ":",
        "marker": ">",
        "hatch": "--",
    },
    "mappo_baseline": {
        "color": "#D81B60",  # rose
        "linestyle": ":",
        "marker": "o",
        "hatch": "oo",
    },
    "mappo_fusion_only": {
        "color": "#5E3C99",  # indigo
        "linestyle": "--",
        "marker": "s",
        "hatch": "OO",
    },
    "mappo_separated_gradient": {
        "color": "#00897B",  # teal
        "linestyle": "-",
        "marker": "D",
        "hatch": "**",
    },
    "per_uniform_baseline": {
        "color": "#7F7F7F",
        "linestyle": ":",
        "marker": "o",
        "hatch": "..",
    },
    "per_improved_mainline": {
        "color": "#1F77B4",
        "linestyle": "-",
        "marker": "o",
        "hatch": "",
    },
}

ALGORITHM_ABLATION_COLOR_BY_LABEL = {
    label: style["color"]
    for label, style in ALGORITHM_ABLATION_STYLE_BY_LABEL.items()
}

ALGORITHM_ABLATION_FALLBACK_COLORS = [
    "#1F77B4",
    "#D62728",
    "#2CA02C",
    "#9467BD",
    "#17BECF",
    "#FF7F0E",
    "#D81B60",
    "#7F7F7F",
]

ALGORITHM_ABLATION_FALLBACK_LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
]

ALGORITHM_ABLATION_FALLBACK_MARKERS = [
    "o",
    "s",
    "D",
    "^",
    "v",
    "<",
    ">",
    "P",
]

ALGORITHM_ABLATION_FALLBACK_HATCHES = [
    "",
    "//",
    "xx",
    "..",
    "\\\\",
    "++",
    "oo",
    "**",
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


def get_algorithm_ablation_style(
    label: Optional[str],
    idx: int = 0,
) -> Dict[str, Any]:
    normalized_label = str(label or "").strip()
    if normalized_label in ALGORITHM_ABLATION_STYLE_BY_LABEL:
        return dict(ALGORITHM_ABLATION_STYLE_BY_LABEL[normalized_label])
    return {
        "color": get_algorithm_ablation_color(normalized_label, idx=idx),
        "linestyle": ALGORITHM_ABLATION_FALLBACK_LINESTYLES[idx % len(ALGORITHM_ABLATION_FALLBACK_LINESTYLES)],
        "marker": ALGORITHM_ABLATION_FALLBACK_MARKERS[idx % len(ALGORITHM_ABLATION_FALLBACK_MARKERS)],
        "hatch": ALGORITHM_ABLATION_FALLBACK_HATCHES[idx % len(ALGORITHM_ABLATION_FALLBACK_HATCHES)],
    }

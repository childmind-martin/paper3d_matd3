#!/usr/bin/env python3
"""Generate a clean vector framework diagram for the dual-semantic pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path("figures")
BASE_NAME = "fig3_execution_consistent_pipeline_code_aligned"


COL = {
    "panel": "#f1f1ef",
    "box": "#eeeeec",
    "blue": "#bcd7ed",
    "orange": "#ffc48a",
    "line": "#151515",
    "muted": "#555555",
    "accent_blue": "#4f93ca",
    "accent_orange": "#eea045",
}


def rounded(ax, x, y, w, h, text="", fc="#ffffff", ec=COL["line"], lw=1.3, r=0.07,
            fontsize=9, weight="normal", ha="center", va="center", z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.015,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=weight,
            color=COL["line"],
            linespacing=1.12,
            zorder=z + 1,
        )
    return patch


def arrow(ax, start, end, color=COL["line"], lw=1.6, ms=12, z=5, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def line(ax, xs, ys, color=COL["line"], lw=1.5, z=4):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round")


def nn_icon(ax, x, y, w, h, out_label, bottom_label, fc, title, dims):
    rounded(ax, x, y, w, h, fc=fc, lw=1.2, r=0.07)
    ax.text(x + w / 2, y + h - 0.14, f"{title}\n{dims}", ha="center", va="top",
            fontsize=8.6, fontweight="bold", linespacing=1.0, zorder=5)
    xs = [x + 0.22 * w, x + 0.50 * w, x + 0.78 * w]
    layers = [
        [y + 0.43 * h, y + 0.58 * h, y + 0.73 * h],
        [y + 0.35 * h, y + 0.50 * h, y + 0.65 * h, y + 0.80 * h],
        [y + 0.58 * h],
    ]
    for li in range(2):
        for yy0 in layers[li]:
            for yy1 in layers[li + 1]:
                line(ax, [xs[li], xs[li + 1]], [yy0, yy1], color="#606060", lw=0.55, z=3)
    for li, vals in enumerate(layers):
        for yy in vals:
            color = "#92bddf" if li < 2 else "#f3ae6c"
            ax.add_patch(Circle((xs[li], yy), 0.065, fc=color, ec="#555555", lw=0.8, zorder=4))
    ax.text(x + w - 0.08, y + 0.59 * h, out_label, ha="right", va="center",
            fontsize=9.5, fontweight="bold", zorder=5)
    ax.text(x + w / 2, y + 0.10, bottom_label, ha="center", va="bottom",
            fontsize=7.4, zorder=5)


def gradient_arrow(ax, x0, y0, x1, y1, c0, c1):
    ax.add_patch(
        Polygon(
            [[x0, y0 - 0.08], [x1 - 0.24, y0 - 0.08], [x1 - 0.24, y0 - 0.18],
             [x1, y1], [x1 - 0.24, y0 + 0.18], [x1 - 0.24, y0 + 0.08], [x0, y0 + 0.08]],
            closed=True,
            fc=c0,
            ec="none",
            alpha=0.90,
            zorder=2,
        )
    )
    ax.plot([x0, x1 - 0.25], [y0, y0], color=c1, lw=1.2, alpha=0.7, zorder=3)


def draw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(7.75, 8.25, "Execution-Consistent Dual-Semantic Learning Pipeline",
            ha="center", va="center", fontsize=24, fontweight="bold")

    # Main panels.
    rounded(ax, 0.25, 0.45, 4.85, 7.35, fc=COL["panel"], ec="none", lw=0, r=0.13)
    rounded(ax, 5.25, 0.45, 3.95, 7.35, fc=COL["panel"], ec="none", lw=0, r=0.13)
    rounded(ax, 9.50, 0.45, 5.75, 7.35, fc=COL["panel"], ec="none", lw=0, r=0.13)
    ax.text(2.68, 7.45, "Online Interaction\n& Policy Generation",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(7.23, 7.45, "Action Fusion\n& Execution",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(12.38, 7.45, "Centralized Training &\nSeparated-Gradient Routing",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # Actor network.
    rounded(ax, 1.05, 1.35, 2.70, 5.85, fc="#f8f8f8", lw=1.2, r=0.10)
    ax.text(2.40, 6.92, "Actor Network (A)", ha="center", va="top",
            fontsize=12, fontweight="bold")
    nn_icon(
        ax, 1.15, 4.32, 2.50, 2.35, "$m$",
        "Raw motion intention, ($m$)", COL["box"],
        "Motion Head", "(dims 1:3)"
    )
    nn_icon(
        ax, 1.15, 1.50, 2.50, 2.35, "$u$",
        "Correction-law parameters, ($u$)", COL["blue"],
        "APF-Modulation Tail", "(dims 4:7)"
    )
    ax.text(0.58, 4.30, "Local\nobs\n($o_i$)", ha="center", va="center", fontsize=11)
    arrow(ax, (0.85, 5.30), (1.15, 5.30), lw=1.4)
    arrow(ax, (0.85, 2.65), (1.15, 2.65), lw=1.4)
    line(ax, [0.73, 0.73, 0.85], [2.65, 5.30, 5.30], lw=1.4)
    line(ax, [0.73, 0.85], [2.65, 2.65], lw=1.4)
    ax.text(4.18, 4.18, "Raw Action\n($a_{raw}$)\n= [$m$, $u$]\nRaw policy\nintention",
            ha="center", va="center", fontsize=11, linespacing=1.0)
    arrow(ax, (3.65, 5.48), (4.05, 4.78), lw=1.5)
    arrow(ax, (3.65, 2.70), (4.05, 3.55), lw=1.5)
    arrow(ax, (4.45, 4.27), (5.25, 4.27), lw=1.8)
    line(ax, [3.65, 4.02, 4.02], [1.60, 1.60, 4.70], lw=1.4)
    arrow(ax, (3.98, 1.60), (4.45, 1.60), lw=1.4)
    ax.text(4.40, 1.68, "$a_{raw}$", fontsize=12, fontweight="bold", ha="left", va="bottom")

    # Fusion module.
    rounded(ax, 5.43, 1.15, 3.55, 6.05, fc="#f8f8f8", lw=1.2, r=0.10)
    ax.text(7.20, 6.92, "APF Correction / Fusion Module",
            ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(6.05, 5.73, "Raw motion ($m$)", ha="center", va="center", fontsize=11)
    line(ax, [5.25, 7.05, 7.05], [4.27, 4.27, 4.98], lw=1.5)
    arrow(ax, (7.05, 4.98), (7.05, 5.22), lw=1.5)
    rounded(ax, 6.22, 4.15, 2.02, 0.78, "APF Correction\n(parameterized by $u$, FR)",
            fc=COL["box"], lw=1.1, r=0.09, fontsize=9.2)
    ax.text(8.55, 5.75, "APF tail\nparameterizes local\ncorrection law",
            ha="center", va="center", fontsize=9.3)
    arrow(ax, (8.43, 4.54), (8.26, 4.54), lw=1.2, ms=10)
    ax.text(8.49, 4.64, "$u$", fontsize=12, fontweight="bold")
    ax.text(6.33, 3.10, "APF tail ($u$)\n\nAPF tail\npreserved\nunchanged",
            ha="center", va="center", fontsize=10, linespacing=1.0)
    arrow(ax, (5.25, 3.03), (7.05, 3.03), lw=1.5)
    ax.text(7.88, 3.95, "Corrected Motion\n($m_{corr}$)", ha="center", va="center",
            fontsize=11)
    line(ax, [7.05, 7.05], [4.15, 2.42], lw=1.5)
    arrow(ax, (7.05, 2.42), (7.05, 1.92), lw=1.5)
    ax.text(7.12, 1.42, "Corrected Action\n($a_{corr}$) = [$m + fr(a_{pf}-m)$, $u$]\nAPF-corrected execution",
            ha="center", va="center", fontsize=11, linespacing=1.0)

    # Data path to training.
    line(ax, [8.95, 9.12, 9.12, 9.50], [1.42, 1.42, 6.45, 6.45], lw=1.5)
    line(ax, [4.45, 4.45, 9.12], [1.60, 0.92, 0.92], lw=1.5)
    line(ax, [9.12, 9.12], [0.92, 6.02], lw=1.5)
    arrow(ax, (9.12, 6.45), (9.50, 6.45), lw=1.4)
    arrow(ax, (9.12, 6.02), (9.50, 6.02), lw=1.4)
    ax.text(9.55, 6.58, "$a_{raw}$", fontsize=11, fontweight="bold", ha="left")
    ax.text(9.55, 6.15, "$a_{corr}$", fontsize=11, fontweight="bold", ha="left")
    ax.text(9.53, 5.65, "state,\nnext", fontsize=10, ha="left", va="center")

    # Replay buffer.
    rounded(ax, 10.20, 5.78, 4.78, 1.42, fc="#e8e8e6", lw=1.2, r=0.08)
    ax.text(12.59, 7.02, "Dual-Semantic Replay Buffer", ha="center", va="top",
            fontsize=12, fontweight="bold")
    ax.text(12.59, 6.62, "Saves both raw ($a_{raw}$) and corrected ($a_{corr}$) actions",
            ha="center", va="center", fontsize=9.5)
    rounded(ax, 10.42, 5.92, 2.22, 0.52, "Raw policy intention\n($a_{raw}$)",
            fc=COL["orange"], ec="none", lw=0, r=0.05, fontsize=8.7, weight="bold")
    rounded(ax, 12.77, 5.92, 1.96, 0.52, "APF-corrected execution\n($a_{corr}$)",
            fc="#9dc4e4", ec="none", lw=0, r=0.05, fontsize=8.5, weight="bold")
    ax.text(12.59, 5.55, "A batch of dual-semantic transitions is sampled",
            ha="center", va="center", fontsize=10)
    arrow(ax, (12.59, 5.38), (12.59, 5.02), lw=1.6)
    ax.text(12.78, 5.18, "$s$, $a_{raw}$, $a_{corr}$", fontsize=10.5,
            ha="left", va="center", fontweight="bold")

    # Critic.
    rounded(ax, 10.05, 2.82, 5.00, 2.05, fc="#f8f8f8", lw=1.2, r=0.10)
    ax.text(12.55, 4.70, "Centralized Dual-Head Critic",
            ha="center", va="top", fontsize=12, fontweight="bold")
    rounded(ax, 10.65, 3.70, 1.95, 0.55, "Raw-Intention\nBranch ($Q_{head}$)",
            fc=COL["box"], lw=1.0, r=0.06, fontsize=9.5)
    rounded(ax, 12.95, 3.70, 1.82, 0.55, "Execution-Consistent\nBranch ($Q_{tail}$)",
            fc=COL["box"], lw=1.0, r=0.06, fontsize=9.2)
    ax.text(11.62, 3.47, "Evaluates raw intention", ha="center", va="center", fontsize=8.6)
    ax.text(13.86, 3.47, "Evaluates execution correction", ha="center", va="center", fontsize=8.3)
    ax.text(10.97, 4.34, "$s$", fontsize=10.5, fontweight="bold")
    ax.text(11.69, 4.34, "$a_{raw}$", fontsize=10.5, fontweight="bold", color="#bf6d1d")
    ax.text(13.26, 4.34, "$s$", fontsize=10.5, fontweight="bold")
    ax.text(14.05, 4.34, "$a_{corr}$", fontsize=10.5, fontweight="bold", color=COL["accent_blue"])
    arrow(ax, (11.04, 4.28), (11.04, 4.12), lw=1.0, ms=8)
    arrow(ax, (11.82, 4.28), (11.82, 4.12), color=COL["accent_orange"], lw=1.0, ms=8)
    arrow(ax, (13.32, 4.28), (13.32, 4.12), lw=1.0, ms=8)
    arrow(ax, (14.14, 4.28), (14.14, 4.12), color=COL["accent_blue"], lw=1.0, ms=8)
    ax.text(12.55, 3.06, "Target branch: target actor at $s'$ -> APF reconstruct $a'_{corr}$ (optional);\nTD target uses $\\min(Q_{tot}^{1},Q_{tot}^{2})$ over twin critics",
            ha="center", va="center", fontsize=8.2, color=COL["muted"], linespacing=1.1)

    # Actor update and routing.
    rounded(ax, 9.67, 1.02, 0.72, 3.55, "Actor\nUpdate", fc="#eeeeec", lw=1.1, r=0.06, fontsize=10)
    arrow(ax, (10.65, 3.97), (10.39, 3.72), lw=1.3)
    ax.text(12.55, 2.55, "Separated-Gradient Routing", ha="center", va="center",
            fontsize=12, fontweight="bold")
    gradient_arrow(ax, 10.50, 2.18, 13.90, 2.18, "#f2b060", "#d47b18")
    gradient_arrow(ax, 10.50, 1.35, 13.90, 1.35, "#b7d6ee", "#3f84bd")
    rounded(ax, 13.78, 1.85, 1.12, 0.58, "Actor Motion\nHead ($m$)",
            fc=COL["box"], lw=1.0, r=0.06, fontsize=9)
    rounded(ax, 13.78, 1.02, 1.12, 0.58, "Actor APF\nTail ($u$)",
            fc=COL["box"], lw=1.0, r=0.06, fontsize=9)
    ax.text(11.55, 2.33, "$\\nabla_{m} Q_{head}$", fontsize=9.5, fontweight="bold")
    ax.text(12.78, 2.33, "stop-grad($u$)", fontsize=9.4)
    ax.text(11.55, 1.50, "$\\nabla_{u} Q_{tail}$", fontsize=9.5, fontweight="bold")
    ax.text(12.78, 1.50, "stop-grad($m$)", fontsize=9.4)
    ax.text(11.72, 1.88, "Motion head optimized via\nraw-intention branch",
            fontsize=8.2, ha="center", va="top")
    ax.text(11.95, 0.88, "APF tail optimized via\nexecution-consistent branch",
            fontsize=8.2, ha="center", va="top")
    for x, y in [(13.12, 2.17), (13.12, 1.34)]:
        ax.add_patch(Circle((x, y), 0.105, fc="none", ec="#b83232", lw=2, zorder=6))
        line(ax, [x - 0.07, x + 0.07], [y - 0.07, y + 0.07], color="#b83232", lw=2, z=7)

    for ext in ("pdf", "svg", "png"):
        out = OUT_DIR / f"{BASE_NAME}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)

    for ext in ("pdf", "svg", "png"):
        print(OUT_DIR / f"{BASE_NAME}.{ext}")


def draw_v2() -> None:
    """Generate the publication-draft version with a roomier three-panel layout."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18.2, 8.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 8.9)
    ax.axis("off")

    ax.text(
        9.1,
        8.55,
        "Execution-Consistent Dual-Semantic Learning Pipeline",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    # Panel backgrounds.
    left = (0.25, 0.50, 5.05, 7.45)
    middle = (5.60, 0.50, 4.65, 7.45)
    right = (10.55, 0.50, 7.40, 7.45)
    for panel in (left, middle, right):
        rounded(ax, *panel, fc=COL["panel"], ec="none", lw=0, r=0.14)

    ax.text(2.78, 7.56, "Online Interaction\n& Policy Generation",
            ha="center", va="center", fontsize=12.6, fontweight="bold")
    ax.text(7.92, 7.56, "Action Fusion\n& Execution",
            ha="center", va="center", fontsize=12.6, fontweight="bold")
    ax.text(14.25, 7.56, "Centralized Training &\nSeparated-Gradient Routing",
            ha="center", va="center", fontsize=12.6, fontweight="bold")

    # Left panel: actor with two output semantics.
    rounded(ax, 1.05, 1.20, 3.18, 5.58, fc="#f8f8f8", lw=1.15, r=0.10)
    ax.text(2.64, 6.50, "Actor Network (A)", ha="center", va="top",
            fontsize=11.5, fontweight="bold")
    nn_icon(ax, 1.22, 4.05, 2.62, 2.06, "$m$",
            "Raw motion intention ($m$)", COL["box"], "Motion Head", "(dims 1:3)")
    nn_icon(ax, 1.22, 1.50, 2.62, 2.06, "$u$",
            "Correction-law parameters ($u$)", COL["blue"], "APF-Modulation Tail", "(dims 4:7)")

    ax.text(0.66, 4.08, "Local\nobs.\n($o_i$)", ha="center", va="center",
            fontsize=10.2)
    line(ax, [0.86, 0.86, 1.17], [2.55, 5.02, 5.02], lw=1.25)
    line(ax, [0.86, 1.17], [2.55, 2.55], lw=1.25)
    arrow(ax, (0.98, 5.02), (1.22, 5.02), lw=1.25, ms=10)
    arrow(ax, (0.98, 2.55), (1.22, 2.55), lw=1.25, ms=10)

    ax.text(4.55, 4.30, "Raw Action\n$a_{raw}=[m,u]$\nraw policy\nintention",
            ha="center", va="center", fontsize=9.8, linespacing=1.05)
    arrow(ax, (3.84, 5.14), (4.25, 4.70), lw=1.35, ms=11)
    arrow(ax, (3.84, 2.47), (4.25, 3.83), lw=1.35, ms=11)
    arrow(ax, (4.92, 4.30), (5.60, 4.30), lw=1.75, ms=13)
    line(ax, [3.84, 4.30, 4.30, 5.75], [1.68, 1.68, 1.05, 1.05], lw=1.25)
    arrow(ax, (4.05, 1.68), (4.75, 1.68), lw=1.25, ms=10)
    ax.text(4.83, 1.76, "$a_{raw}$", fontsize=11.5, fontweight="bold",
            ha="left", va="bottom")

    # Middle panel: APF correction/fusion.
    rounded(ax, 5.85, 1.18, 4.15, 5.60, fc="#f8f8f8", lw=1.15, r=0.10)
    ax.text(7.92, 6.50, "APF Correction / Fusion Module",
            ha="center", va="top", fontsize=11.4, fontweight="bold")
    ax.text(6.55, 5.48, "Raw motion ($m$)", ha="center", va="center",
            fontsize=10.2)
    arrow(ax, (5.60, 4.30), (6.95, 4.30), lw=1.35, ms=11)
    arrow(ax, (7.75, 5.05), (7.75, 4.83), lw=1.2, ms=9)

    rounded(ax, 6.95, 4.12, 1.90, 0.70,
            "APF Correction\n(parameterized by $u$, FR)",
            fc=COL["box"], lw=1.0, r=0.08, fontsize=8.4)
    ax.text(9.08, 5.42, "APF tail\nparameterizes local\ncorrection law",
            ha="center", va="center", fontsize=8.4, linespacing=1.0)
    arrow(ax, (9.00, 4.47), (8.87, 4.47), lw=1.0, ms=8)
    ax.text(9.08, 4.58, "$u$", fontsize=10.5, fontweight="bold")

    ax.text(6.65, 3.12, "APF tail ($u$)\n\nAPF tail\npreserved\nunchanged",
            ha="center", va="center", fontsize=9.3, linespacing=1.0)
    arrow(ax, (5.60, 3.02), (7.72, 3.02), lw=1.35, ms=11)
    ax.text(8.62, 3.80, "Corrected Motion\n($m_{corr}$)",
            ha="center", va="center", fontsize=10)
    line(ax, [7.75, 7.75], [4.12, 2.34], lw=1.35)
    arrow(ax, (7.75, 2.34), (7.75, 1.84), lw=1.35, ms=11)
    ax.text(
        7.94,
        1.55,
        "Corrected Action\n$a_{corr}=[m+fr(a_{pf}-m),u]$\nAPF-corrected execution",
        ha="center",
        va="center",
        fontsize=10.0,
        linespacing=1.0,
    )

    # Paths into training: keep them visually separate.
    line(ax, [9.88, 10.20, 10.20, 10.55], [1.55, 1.55, 6.47, 6.47], lw=1.35)
    line(ax, [5.75, 10.20], [1.05, 1.05], lw=1.25)
    line(ax, [10.20, 10.20], [1.05, 6.05], lw=1.25)
    arrow(ax, (10.20, 6.47), (10.55, 6.47), lw=1.25, ms=10)
    arrow(ax, (10.20, 6.05), (10.55, 6.05), lw=1.25, ms=10)
    ax.text(10.60, 6.59, "$a_{raw}$", fontsize=10.5, fontweight="bold",
            ha="left", va="bottom")
    ax.text(10.60, 6.17, "$a_{corr}$", fontsize=10.5, fontweight="bold",
            ha="left", va="bottom")
    ax.text(10.60, 5.70, "state,\nnext", fontsize=9.4, ha="left", va="center")

    # Right panel: replay buffer.
    rounded(ax, 11.20, 5.58, 6.35, 1.22, fc="#e8e8e6", lw=1.15, r=0.08)
    ax.text(14.38, 6.64, "Dual-Semantic Replay Buffer", ha="center", va="top",
            fontsize=11.5, fontweight="bold")
    ax.text(14.38, 6.27, "Stores both raw ($a_{raw}$) and corrected ($a_{corr}$) actions",
            ha="center", va="center", fontsize=8.9)
    rounded(ax, 11.48, 5.74, 2.65, 0.48, "Raw policy intention\n($a_{raw}$)",
            fc=COL["orange"], ec="none", lw=0, r=0.05, fontsize=8.8, weight="bold")
    rounded(ax, 14.34, 5.74, 2.72, 0.48, "APF-corrected execution\n($a_{corr}$)",
            fc="#9dc4e4", ec="none", lw=0, r=0.05, fontsize=8.8, weight="bold")
    ax.text(14.38, 5.36, "A batch of dual-semantic transitions is sampled",
            ha="center", va="center", fontsize=9.8)
    arrow(ax, (14.38, 5.20), (14.38, 4.98), lw=1.4, ms=10)
    ax.text(14.58, 5.08, "$s$, $a_{raw}$, $a_{corr}$", fontsize=10.2,
            ha="left", va="center", fontweight="bold")

    # Right panel: critic and target note.
    rounded(ax, 11.35, 2.90, 6.10, 2.05, fc="#f8f8f8", lw=1.15, r=0.10)
    ax.text(14.40, 4.77, "Centralized Dual-Head Critic",
            ha="center", va="top", fontsize=11.5, fontweight="bold")
    rounded(ax, 12.02, 3.74, 2.12, 0.55, "Raw-Intention\nBranch ($Q_{head}$)",
            fc=COL["box"], lw=0.95, r=0.055, fontsize=9.0)
    rounded(ax, 14.62, 3.74, 2.18, 0.55, "Execution-Consistent\nBranch ($Q_{tail}$)",
            fc=COL["box"], lw=0.95, r=0.055, fontsize=8.8)
    ax.text(13.08, 3.50, "Evaluates raw intention",
            ha="center", va="center", fontsize=8.2)
    ax.text(15.71, 3.50, "Evaluates execution correction",
            ha="center", va="center", fontsize=8.1)
    for x, label, color in [
        (12.42, "$s$", COL["line"]),
        (13.24, "$a_{raw}$", "#bf6d1d"),
        (15.02, "$s$", COL["line"]),
        (15.92, "$a_{corr}$", COL["accent_blue"]),
    ]:
        ax.text(x, 4.35, label, fontsize=10.0, fontweight="bold", color=color, ha="center")
        arrow(ax, (x, 4.28), (x, 4.13), color=color, lw=1.0, ms=7)
    ax.text(
        14.38,
        3.12,
        "Target branch: target actor at $s'$ -> APF reconstruct $a'_{corr}$ (optional);\n"
        "TD target uses $\\min(Q_{tot}^{1},Q_{tot}^{2})$ over twin critics",
        ha="center",
        va="center",
        fontsize=7.8,
        color=COL["muted"],
        linespacing=1.1,
    )

    # Actor update and separated routing.
    rounded(ax, 10.82, 1.02, 0.70, 3.70, "Actor\nUpdate", fc="#eeeeec",
            lw=1.05, r=0.06, fontsize=9.4)
    arrow(ax, (12.02, 4.02), (11.52, 3.66), lw=1.2, ms=10)
    ax.text(14.38, 2.53, "Separated-Gradient Routing", ha="center", va="center",
            fontsize=11.5, fontweight="bold")
    gradient_arrow(ax, 12.00, 2.16, 16.30, 2.16, "#f2b060", "#d47b18")
    gradient_arrow(ax, 12.00, 1.36, 16.30, 1.36, "#b7d6ee", "#3f84bd")
    rounded(ax, 16.05, 1.86, 1.30, 0.58, "Actor Motion\nHead ($m$)",
            fc=COL["box"], lw=0.95, r=0.055, fontsize=8.7)
    rounded(ax, 16.05, 1.06, 1.30, 0.58, "Actor APF\nTail ($u$)",
            fc=COL["box"], lw=0.95, r=0.055, fontsize=8.7)
    ax.text(12.64, 2.32, "$\\nabla_m Q_{head}$", fontsize=9.0, fontweight="bold")
    ax.text(14.02, 2.32, "stop-grad($u$)", fontsize=9.0)
    ax.text(12.64, 1.52, "$\\nabla_u Q_{tail}$", fontsize=9.0, fontweight="bold")
    ax.text(14.02, 1.52, "stop-grad($m$)", fontsize=9.0)
    ax.text(13.00, 1.90, "Motion head optimized via\nraw-intention branch",
            fontsize=7.8, ha="center", va="top")
    ax.text(13.12, 0.94, "APF tail optimized via\nexecution-consistent branch",
            fontsize=7.8, ha="center", va="top")
    for x, y in [(15.34, 2.15), (15.34, 1.35)]:
        ax.add_patch(Circle((x, y), 0.09, fc="none", ec="#b83232", lw=1.8, zorder=6))
        line(ax, [x - 0.06, x + 0.06], [y - 0.06, y + 0.06], color="#b83232", lw=1.8, z=7)

    for ext in ("pdf", "svg", "png"):
        out = OUT_DIR / f"{BASE_NAME}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)

    for ext in ("pdf", "svg", "png"):
        print(OUT_DIR / f"{BASE_NAME}.{ext}")


def draw_reference_matched() -> None:
    """Redraw the framework with fixed coordinates close to the provided reference."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42

    width, height = 1408, 768
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def text(x, y, s, size=12, weight="normal", ha="center", va="center",
             color=COL["line"], linespacing=1.05, z=10, bbox_fc=None):
        kwargs = {}
        if bbox_fc:
            kwargs["bbox"] = {
                "facecolor": bbox_fc,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.10,rounding_size=0.8",
                "alpha": 0.96,
            }
        ax.text(
            x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color, linespacing=linespacing, zorder=z, **kwargs
        )

    def rbox(x, y, w, h, s="", fc="#ffffff", ec=COL["line"], lw=1.4, radius=10,
             size=12, weight="normal", z=2, linespacing=1.05):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z
        )
        ax.add_patch(patch)
        if s:
            text(x + w / 2, y + h / 2, s, size=size, weight=weight,
                 linespacing=linespacing, z=z + 1)
        return patch

    def arr(x0, y0, x1, y1, color=COL["line"], lw=1.6, ms=13, z=8):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z
        ))

    def polyline(points, color=COL["line"], lw=1.6, z=7):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=z)

    def node(x, y, r=9, fc="#9dc4e4"):
        ax.add_patch(Circle((x, y), r, fc=fc, ec="#555555", lw=1.0, zorder=5))

    def network_icon(x, y, w, h, title, dims, label, out_label, fill):
        rbox(x, y, w, h, fc=fill, lw=1.2, radius=7)
        text(x + w / 2, y + 19, title, size=13, weight="bold")
        text(x + w / 2, y + 37, dims, size=11, weight="bold")
        ins = [(x + 34, y + 74), (x + 34, y + 103), (x + 34, y + 132)]
        hid = [(x + 103, y + 55), (x + 103, y + 88), (x + 103, y + 122), (x + 103, y + 156)]
        out = (x + 176, y + 103)
        for p0 in ins:
            for p1 in hid:
                polyline([p0, p1], color="#5f5f5f", lw=0.75, z=3)
        for p0 in hid:
            polyline([p0, out], color="#5f5f5f", lw=0.75, z=3)
        for p in ins + hid:
            node(*p, r=9, fc="#9dc4e4")
        node(*out, r=11, fc="#f3ae6c")
        text(x + w - 20, out[1], out_label, size=16, weight="bold", ha="left")
        text(x + w / 2, y + h - 18, label, size=10, weight="normal")

    def stop_sign(x, y, r=10):
        ax.add_patch(Circle((x, y), r, fc="none", ec="#b83232", lw=2.4, zorder=12))
        polyline([(x - 7, y + 7), (x + 7, y - 7)], color="#b83232", lw=2.4, z=13)

    def gradient_left_arrow(x0, x1, y, fill, line_color, line_end=None):
        # x0 is the left arrow tip, x1 is the right tail.
        if line_end is None:
            line_end = x1 - 3
        ax.add_patch(Polygon(
            [(x0, y), (x0 + 44, y - 21), (x0 + 44, y - 11),
             (x1, y - 11), (x1, y + 11), (x0 + 44, y + 11), (x0 + 44, y + 21)],
            closed=True, fc=fill, ec="none", alpha=0.34, zorder=3
        ))
        ax.add_patch(Polygon(
            [(x0 + 10, y), (x0 + 52, y - 14), (x0 + 52, y - 6),
             (x1, y - 6), (x1, y + 6), (x0 + 52, y + 6), (x0 + 52, y + 14)],
            closed=True, fc=fill, ec="none", alpha=0.62, zorder=4
        ))
        if line_end is not False:
            ax.plot([x0 + 48, line_end], [y, y], color=line_color, lw=2.2,
                    alpha=0.68, zorder=5, solid_capstyle="round")

    def gradient_right_arrow(x0, x1, y, fill, line_color):
        # x0 is the left tail, x1 is the right arrow tip.
        ax.add_patch(Polygon(
            [(x0, y - 11), (x1 - 44, y - 11), (x1 - 44, y - 21),
             (x1, y), (x1 - 44, y + 21), (x1 - 44, y + 11), (x0, y + 11)],
            closed=True, fc=fill, ec="none", alpha=0.34, zorder=3
        ))
        ax.add_patch(Polygon(
            [(x0, y - 6), (x1 - 52, y - 6), (x1 - 52, y - 14),
             (x1 - 10, y), (x1 - 52, y + 14), (x1 - 52, y + 6), (x0, y + 6)],
            closed=True, fc=fill, ec="none", alpha=0.62, zorder=4
        ))
        return

    # Title and panels.
    text(704, 38, "Execution-Consistent Dual-Semantic Learning Pipeline",
         size=27, weight="bold")
    rbox(10, 85, 455, 662, fc=COL["panel"], ec="none", lw=0, radius=15)
    rbox(477, 85, 365, 662, fc=COL["panel"], ec="none", lw=0, radius=15)
    rbox(866, 85, 528, 662, fc=COL["panel"], ec="none", lw=0, radius=15)
    text(238, 123, "Online Interaction\n& Policy Generation",
         size=18, weight="bold")
    text(660, 123, "Action Fusion\n& Execution", size=18, weight="bold")
    text(1130, 123, "Centralized Training &\nSeparated-Gradient Routing",
         size=18, weight="bold")

    # Actor panel.
    rbox(89, 163, 249, 503, fc="#f9f9f8", lw=1.25, radius=7)
    text(214, 188, "Actor Network (A)", size=16, weight="bold")
    network_icon(98, 207, 227, 199, "Motion Head", "(dims 1:3)",
                 "Raw motion intention, ($m$)", "$m$", COL["box"])
    network_icon(98, 432, 227, 219, "APF-Modulation Tail", "(dims 4:7)",
                 "Correction-law parameters, ($u$)", "$u$", COL["blue"])

    text(49, 384, "State\n($s$)", size=15, weight="bold")
    polyline([(77, 311), (77, 549)], lw=1.6)
    polyline([(77, 311), (98, 311)], lw=1.6)
    polyline([(77, 549), (98, 549)], lw=1.6)
    arr(84, 311, 98, 311, lw=1.6, ms=12)
    arr(84, 549, 98, 549, lw=1.6, ms=12)

    # Raw-action assembly and outgoing path. The actor emits one 7-D action,
    # conceptually split as m (dims 1:3) and u (dims 4:7), then concatenated.
    polyline([(325, 310), (346, 310), (346, 549), (325, 549)], lw=1.7)
    polyline([(346, 416), (358, 416)], lw=1.7)
    arr(450, 416, 477, 416, lw=1.9, ms=16)
    text(407, 414, "Raw Action\n($a_{raw}$)\n= [$m$, $u$]\nRaw policy\nintention",
         size=13.2, linespacing=0.90, bbox_fc=COL["panel"])

    # Raw semantic record line to replay.
    polyline([(405, 455), (405, 697), (853, 697), (853, 201)], lw=1.6)
    arr(853, 201, 935, 201, lw=1.6, ms=12)
    text(386, 684, "$a_{raw}$", size=16, weight="bold", ha="left",
         bbox_fc=COL["panel"])

    # APF fusion panel.
    rbox(492, 163, 335, 428, fc="#f9f9f8", lw=1.25, radius=7)
    text(660, 188, "APF Correction / Fusion Module", size=12.8, weight="bold")
    text(575, 260, "Raw motion ($m$)", size=13.2)
    polyline([(492, 416), (492, 270), (656, 270), (656, 335)], lw=1.6)
    arr(656, 270, 656, 335, lw=1.6, ms=12)
    rbox(564, 336, 186, 69, "APF Correction\n(Parameterized by $u$)",
         fc=COL["box"], lw=1.2, radius=8, size=11.8)
    text(742, 260, "APF tail\nparameterizes local\ncorrection law",
         size=11.0, linespacing=0.95)
    arr(795, 370, 752, 370, lw=1.3, ms=11)
    text(790, 352, "$u$", size=14, weight="bold", ha="left",
         bbox_fc="#f9f9f8")

    polyline([(477, 509), (656, 509)], lw=1.6)
    arr(640, 509, 656, 509, lw=1.6, ms=12)
    text(574, 499, "APF tail ($u$)", size=13.2)
    text(573, 545, "APF tail\npreserved\nunchanged", size=11.8, linespacing=0.90)
    text(737, 445, "Corrected Motion\n($m_{corr}$)", size=13.5)
    polyline([(656, 405), (656, 590)], lw=1.6)
    arr(656, 535, 656, 590, lw=1.6, ms=12)
    text(661, 650, "Corrected Action\n($a_{corr}$) = [$m_{corr}$, $u$]\nAPF-corrected execution",
         size=14.2, linespacing=0.98)

    # Corrected path to replay.
    polyline([(827, 650), (853, 650), (853, 253)], lw=1.6)
    arr(853, 253, 935, 253, lw=1.6, ms=12)
    text(873, 198, "$\\mathbf{a}_{\\mathbf{raw}}$", size=12.2, weight="bold", ha="left",
         bbox_fc=COL["panel"])
    text(873, 245, "$\\mathbf{a}_{\\mathbf{corr}}$", size=12.2, weight="bold", ha="left",
         bbox_fc=COL["panel"])
    text(882, 292, "state,\nnext", size=13.0, weight="bold", ha="left", linespacing=0.9,
         bbox_fc=COL["panel"])

    # Replay buffer.
    rbox(935, 163, 448, 125, fc="#e8e8e6", lw=1.25, radius=7)
    text(1159, 189, "Dual-Semantic Replay Buffer", size=15.5, weight="bold")
    text(1159, 214, "Saves both raw ($a_{raw}$) and corrected ($a_{corr}$) actions",
         size=11.4, weight="bold")
    rbox(950, 228, 207, 53, "Raw policy intention\n($a_{raw}$)",
         fc=COL["orange"], ec="none", lw=0, radius=5, size=10.3, weight="bold")
    rbox(1172, 228, 207, 53, "APF-corrected execution\n($a_{corr}$)",
         fc="#9dc4e4", ec="none", lw=0, radius=5, size=10.0, weight="bold")
    text(1159, 314, "A batch of dual-semantic transitions is sampled", size=13.0)
    arr(1159, 329, 1159, 340, lw=1.5, ms=12)
    text(1174, 335, "$\\mathbf{s},\\ \\mathbf{a}_{\\mathbf{raw}},\\ \\mathbf{a}_{\\mathbf{corr}}$",
         size=12.0, weight="bold", ha="left", bbox_fc="white")

    # Critic and actor update.
    rbox(920, 344, 463, 250, fc="#f9f9f8", lw=1.25, radius=7)
    rbox(881, 410, 74, 298, "Actor\nUpdate", fc="#eeeeec", lw=1.2,
         radius=7, size=15, linespacing=0.95)
    arr(1008, 448, 955, 448, lw=1.7, ms=14)
    text(1152, 369, "Centralized Dual-Head Critic", size=16.2, weight="bold")
    text(1033, 400, "$\\mathbf{s}$", size=11.5, weight="bold")
    text(1095, 400, "$\\mathbf{a}_{\\mathbf{raw}}$", size=11.5, weight="bold", color="#c87620")
    text(1204, 400, "$\\mathbf{s}$", size=11.5, weight="bold")
    text(1271, 400, "$\\mathbf{a}_{\\mathbf{corr}}$", size=11.5, weight="bold", color=COL["accent_blue"])
    arr(1033, 410, 1033, 418, lw=1.2, ms=8)
    arr(1095, 410, 1095, 418, color="#c87620", lw=1.2, ms=8)
    arr(1204, 410, 1204, 418, lw=1.2, ms=8)
    arr(1271, 410, 1271, 418, color=COL["accent_blue"], lw=1.2, ms=8)
    rbox(1008, 417, 149, 59, "Raw-Intention\nBranch ($Q_{head}$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.8)
    rbox(1175, 417, 178, 59, "Execution-Consistent\nBranch ($Q_{tail}$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.3)
    text(1078, 499, "Evaluates raw intention", size=9.3)
    text(1272, 499, "Evaluates execution correction", size=9.2)
    text(1151, 530, "Separated-Gradient Routing", size=15.2, weight="bold")

    # Separated-gradient arrows point toward Actor Update, as in the reference.
    gradient_left_arrow(958, 1250, 575, "#f2b060", "#d47b18", line_end=False)
    gradient_right_arrow(958, 1248, 672, "#b7d6ee", "#3f84bd")
    text(1018, 552, "$\\nabla_{a_{raw}} Q_{head}$", size=10.6, weight="bold")
    text(1152, 552, "stop-grad", size=10.6)
    text(1018, 649, "$\\nabla_{a_{raw}} Q_{tail}$", size=10.6, weight="bold")
    text(1152, 649, "stop-grad", size=10.6)
    stop_sign(1190, 575, r=8)
    stop_sign(1190, 672, r=8)
    rbox(1250, 545, 116, 59, "Actor Motion\nHead ($m$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.5)
    rbox(1250, 648, 116, 59, "Actor\nAPF Tail ($u$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.5)
    text(1053, 616, "Motion head optimized via\nraw-intention branch",
         size=9.7, linespacing=0.95)
    text(1076, 721, "APF tail optimized via\nexecution-consistent branch",
         size=9.7, linespacing=0.95)

    for ext in ("pdf", "svg", "png"):
        out = OUT_DIR / f"{BASE_NAME}.{ext}"
        fig.savefig(out, dpi=100)
    plt.close(fig)

    for ext in ("pdf", "svg", "png"):
        print(OUT_DIR / f"{BASE_NAME}.{ext}")


def draw_code_aligned_reference_style() -> None:
    """Draw a reference-style framework diagram aligned with the actual code path.

    The code uses a single actor output layer with action_dim=7. The diagram
    therefore shows one 7-D raw action followed by a semantic split into m and u,
    instead of implying two independent actor output layers.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42

    width, height = 1408, 768
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def text(x, y, s, size=12, weight="normal", ha="center", va="center",
             color=COL["line"], linespacing=1.05, z=10, bbox_fc=None):
        kwargs = {}
        if bbox_fc:
            kwargs["bbox"] = {
                "facecolor": bbox_fc,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.11,rounding_size=0.8",
                "alpha": 0.97,
            }
        ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight,
                color=color, linespacing=linespacing, zorder=z, **kwargs)

    def rbox(x, y, w, h, s="", fc="#ffffff", ec=COL["line"], lw=1.35,
             radius=8, size=12, weight="normal", z=2, linespacing=1.05):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z
        )
        ax.add_patch(patch)
        if s:
            text(x + w / 2, y + h / 2, s, size=size, weight=weight,
                 linespacing=linespacing, z=z + 1)
        return patch

    def line(points, color=COL["line"], lw=1.6, z=7):
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=color, lw=lw, solid_capstyle="round", zorder=z)

    def arr(x0, y0, x1, y1, color=COL["line"], lw=1.6, ms=13, z=8):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z
        ))

    def node(x, y, r=8, fc="#9dc4e4"):
        ax.add_patch(Circle((x, y), r, fc=fc, ec="#555555", lw=1.0, zorder=5))

    def shared_actor_icon(x, y):
        rbox(x, y, 220, 145, fc=COL["box"], lw=1.15, radius=7)
        text(x + 110, y + 22, "Shared Actor MLP", size=13.5, weight="bold")
        text(x + 110, y + 42, "Dense(7, tanh) output", size=10.8, weight="bold")
        ins = [(x + 38, y + 70), (x + 38, y + 95), (x + 38, y + 120)]
        hid = [(x + 100, y + 60), (x + 100, y + 85), (x + 100, y + 110), (x + 100, y + 135)]
        out = (x + 170, y + 98)
        for p0 in ins:
            for p1 in hid:
                line([p0, p1], color="#5f5f5f", lw=0.75, z=3)
        for p0 in hid:
            line([p0, out], color="#5f5f5f", lw=0.75, z=3)
        for p in ins + hid:
            node(*p, r=8, fc="#9dc4e4")
        node(*out, r=10, fc="#f3ae6c")
        text(x + 194, y + 98, "$a_{raw}$", size=13.5, weight="bold", ha="left")

    def stop_sign(x, y, r=9):
        ax.add_patch(Circle((x, y), r, fc="none", ec="#b83232", lw=2.3, zorder=12))
        line([(x - 6.5, y + 6.5), (x + 6.5, y - 6.5)],
             color="#b83232", lw=2.3, z=13)

    def gradient_left_arrow(x0, x1, y, fill):
        ax.add_patch(Polygon(
            [(x0, y), (x0 + 42, y - 20), (x0 + 42, y - 10),
             (x1, y - 10), (x1, y + 10), (x0 + 42, y + 10), (x0 + 42, y + 20)],
            closed=True, fc=fill, ec="none", alpha=0.48, zorder=3
        ))

    def gradient_right_arrow(x0, x1, y, fill):
        ax.add_patch(Polygon(
            [(x0, y - 10), (x1 - 42, y - 10), (x1 - 42, y - 20),
             (x1, y), (x1 - 42, y + 20), (x1 - 42, y + 10), (x0, y + 10)],
            closed=True, fc=fill, ec="none", alpha=0.48, zorder=3
        ))

    # Canvas and panels.
    text(704, 38, "Execution-Consistent Dual-Semantic MATD3 Pipeline",
         size=30, weight="bold")
    rbox(10, 85, 455, 662, fc=COL["panel"], ec="none", lw=0, radius=15)
    rbox(477, 85, 365, 662, fc=COL["panel"], ec="none", lw=0, radius=15)
    rbox(866, 85, 528, 662, fc=COL["panel"], ec="none", lw=0, radius=15)
    text(238, 123, "Online Interaction\n& Policy Generation",
         size=19, weight="bold")
    text(660, 123, "Action Fusion\n& Execution", size=19, weight="bold")
    text(1130, 123, "Centralized Training &\nSeparated-Gradient Routing",
         size=19, weight="bold")

    # Actor: single 7-D output, then semantic split.
    rbox(89, 163, 249, 503, fc="#f9f9f8", lw=1.25, radius=7)
    text(214, 188, "Actor Network (A)", size=17, weight="bold")
    shared_actor_icon(103, 214)
    rbox(104, 392, 220, 68, "Raw 7-D Action\n$a_{raw}=[m,u]$",
         fc="#f3f3f1", lw=1.1, radius=7, size=14, weight="bold")
    text(214, 486, "semantic split\n(same 7-D action)",
         size=8.2, weight="bold", color=COL["muted"], linespacing=0.90,
         bbox_fc="#f9f9f8")
    rbox(104, 515, 104, 88, "$m$\ndims 1:3\nraw motion",
         fc=COL["box"], lw=1.05, radius=7, size=10.0, weight="bold")
    rbox(220, 515, 104, 88, "$u$\ndims 4:7\nAPF params",
         fc=COL["blue"], lw=1.05, radius=7, size=10.0, weight="bold")
    line([(213, 359), (213, 392)], lw=1.35)
    line([(213, 460), (213, 500)], lw=1.25)
    line([(156, 500), (272, 500)], lw=1.25)
    arr(156, 500, 156, 515, lw=1.25, ms=10)
    arr(272, 500, 272, 515, lw=1.25, ms=10)

    text(49, 384, "Local\nobs.\n($o_i$)", size=14.5, weight="bold")
    line([(77, 310), (103, 310)], lw=1.6)
    line([(77, 310), (103, 310)], lw=1.6)
    arr(84, 310, 103, 310, lw=1.6, ms=12)

    # Raw action to fusion and raw semantic to replay.
    text(407, 416, "Raw Behavior\nAction ($a_{raw}$)\n= [$m$, $u$]\nActor + OU,\nbefore APF",
         size=12.7, linespacing=0.90, bbox_fc=COL["panel"])
    line([(324, 426), (356, 426)], lw=1.7)
    arr(452, 426, 477, 426, lw=1.9, ms=16)
    line([(405, 458), (405, 697), (853, 697), (853, 201)], lw=1.6)
    text(385, 697, "$\\mathbf{a}_{raw}$", size=15.5, weight="bold",
         ha="left", bbox_fc=COL["panel"])
    arr(853, 201, 935, 201, lw=1.6, ms=12)

    # APF correction / fusion.
    rbox(492, 163, 335, 428, fc="#f9f9f8", lw=1.25, radius=7)
    text(660, 188, "APF Correction / Fusion Module", size=12.8, weight="bold")
    line([(477, 426), (510, 426), (510, 270), (656, 270), (656, 335)], lw=1.6)
    arr(656, 270, 656, 335, lw=1.6, ms=12)
    line([(510, 426), (510, 509), (656, 509)], lw=1.6)
    arr(640, 509, 656, 509, lw=1.6, ms=12)
    text(575, 260, "Raw motion ($m$)", size=14.0, bbox_fc="#f9f9f8")
    text(574, 499, "APF tail ($u$)", size=14.0, bbox_fc="#f9f9f8")
    rbox(546, 336, 208, 69, "APF Correction\n(parameterized by $u$, FR)",
         fc=COL["box"], lw=1.2, radius=8, size=10.8)
    text(742, 260, "APF tail\nparameterizes local\ncorrection law",
         size=11.6, linespacing=0.95)
    arr(795, 370, 752, 370, lw=1.3, ms=11)
    text(790, 352, "$u$", size=14, weight="bold", ha="left", bbox_fc="#f9f9f8")
    text(573, 545, "APF tail\npreserved\nunchanged", size=12.2, linespacing=0.90)
    text(737, 445, "Corrected Motion\n$m_{corr}=m+fr(a_{pf}-m)$",
         size=12.2, linespacing=0.95)
    line([(656, 405), (656, 590)], lw=1.6)
    arr(656, 535, 656, 590, lw=1.6, ms=12)
    text(661, 650, "Corrected Action\n($a_{corr}$) = [$m_{corr}$, $u$]\nAPF-corrected execution",
         size=14.8, linespacing=0.98)
    line([(827, 650), (853, 650), (853, 253)], lw=1.6)
    arr(853, 253, 935, 253, lw=1.6, ms=12)
    text(873, 198, "$\\mathbf{a}_{raw}$", size=12.2, weight="bold",
         ha="left", bbox_fc=COL["panel"])
    text(873, 245, "$\\mathbf{a}_{corr}$", size=12.2, weight="bold",
         ha="left", bbox_fc=COL["panel"])
    text(882, 292, "$s, s'$", size=13.0, weight="bold",
         ha="left", linespacing=0.9, bbox_fc=COL["panel"])

    # Replay buffer.
    rbox(935, 163, 448, 125, fc="#e8e8e6", lw=1.25, radius=7)
    text(1159, 189, "Dual-Semantic Replay Buffer", size=16.8, weight="bold")
    text(1159, 214, "Stores raw ($a_{raw}$) and corrected ($a_{corr}$) actions",
         size=10.8, weight="bold")
    rbox(950, 228, 207, 53, "Raw behavior\n($a_{raw}$)",
         fc=COL["orange"], ec="none", lw=0, radius=5, size=10.8, weight="bold")
    rbox(1172, 228, 207, 53, "APF-corrected execution\n($a_{corr}$)",
         fc="#9dc4e4", ec="none", lw=0, radius=5, size=10.5, weight="bold")
    text(1159, 314, "A batch of dual-semantic transitions is sampled", size=14.0)
    arr(1159, 329, 1159, 345, lw=1.5, ms=12)
    text(1174, 335, "$\\mathbf{s},\\mathbf{s}',\\ \\mathbf{a}_{raw},\\ \\mathbf{a}_{corr}$",
         size=12.2, weight="bold", ha="left", bbox_fc="white")

    # Critic and gradient routing.
    rbox(920, 344, 463, 224, fc="#f9f9f8", lw=1.25, radius=7)
    rbox(881, 410, 74, 298, "Actor\nUpdate", fc="#eeeeec", lw=1.2,
         radius=7, size=15, linespacing=0.95)
    text(1152, 366, "Twin Centralized Critics", size=15.8, weight="bold")
    text(1152, 386, "each critic has dual semantic heads", size=10.2, weight="bold")
    text(1033, 400, "$\\mathbf{s}$", size=11.5, weight="bold")
    text(1095, 400, "$\\mathbf{a}_{raw}$", size=11.5, weight="bold", color="#c87620")
    text(1204, 400, "$\\mathbf{s}$", size=11.5, weight="bold")
    text(1271, 400, "$\\mathbf{a}_{corr}$", size=11.5, weight="bold", color=COL["accent_blue"])
    arr(1033, 410, 1033, 418, lw=1.2, ms=8)
    arr(1095, 410, 1095, 418, color="#c87620", lw=1.2, ms=8)
    arr(1204, 410, 1204, 418, lw=1.2, ms=8)
    arr(1271, 410, 1271, 418, color=COL["accent_blue"], lw=1.2, ms=8)
    rbox(1008, 417, 149, 59, "Raw-Intention\nHead ($Q_{head}$)",
         fc=COL["box"], lw=1.1, radius=5, size=11.0)
    rbox(1175, 417, 178, 59, "Execution-Consistent\nHead ($Q_{tail}$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.4)
    arr(1008, 448, 955, 448, lw=1.7, ms=14)
    text(1078, 499, "Evaluates raw intention", size=9.4)
    text(1272, 499, "Evaluates execution correction", size=9.3)
    text(1151, 517, "$Q_{tot}=Q_{head}(s,a_{raw})+Q_{tail}(s,a_{corr})$",
         size=10.0, weight="bold")
    text(1151, 535, "TD target: twin-min over $Q_{tot}^{1}, Q_{tot}^{2}$",
         size=8.8, color=COL["muted"])
    text(1151, 560, "Separated-Gradient Routing", size=13.5, weight="bold")
    gradient_left_arrow(958, 1250, 607, "#f2b060")
    gradient_right_arrow(958, 1248, 685, "#b7d6ee")
    text(1018, 588, "$\\nabla_m Q_{head}$", size=10.5, weight="bold")
    text(1152, 588, "stop-grad($u$)", size=10.3)
    text(1018, 666, "$\\nabla_u Q_{tail}$", size=10.5, weight="bold")
    text(1152, 666, "stop-grad($m$)", size=10.3)
    stop_sign(1190, 607, r=8)
    stop_sign(1190, 685, r=8)
    rbox(1250, 577, 116, 56, "Actor Motion\nSlice ($m$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.3)
    rbox(1250, 655, 116, 56, "Actor APF\nSlice ($u$)",
         fc=COL["box"], lw=1.1, radius=5, size=10.3)
    text(1053, 640, "Motion slice optimized via\nraw-intention head",
         size=8.9, linespacing=0.93)
    text(1076, 730, "APF slice optimized via\nexecution-consistent head",
         size=8.9, linespacing=0.93)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"{BASE_NAME}.{ext}", dpi=100)
    plt.close(fig)

    for ext in ("pdf", "svg", "png"):
        print(OUT_DIR / f"{BASE_NAME}.{ext}")


if __name__ == "__main__":
    draw_code_aligned_reference_style()

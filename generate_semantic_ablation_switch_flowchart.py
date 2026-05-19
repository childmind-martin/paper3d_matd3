#!/usr/bin/env python3
"""Generate a compact vector flowchart for the focused semantic ablation switches."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("figures")
BASE_NAME = "fig_semantic_ablation_switches"


COLORS = {
    "bg": "#ffffff",
    "panel": "#f7f7f5",
    "full": "#e8f3ff",
    "collapsed": "#fff1df",
    "notarget": "#f1edff",
    "line": "#252525",
    "muted": "#666666",
    "green": "#2f8f46",
    "red": "#c73a3a",
    "blue": "#3478bd",
    "orange": "#d17a00",
    "purple": "#6d57b8",
}


def box(ax, xy, wh, text, *, fc, ec=COLORS["line"], lw=1.2, fontsize=9.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["line"],
        fontweight=weight,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, *, color=COLORS["line"], lw=1.4, mutation_scale=12, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


def cross(ax, center, size=0.09, *, color=COLORS["red"], lw=2.4):
    x, y = center
    ax.plot([x - size, x + size], [y - size, y + size], color=color, lw=lw, solid_capstyle="round")
    ax.plot([x - size, x + size], [y + size, y - size], color=color, lw=lw, solid_capstyle="round")


def draw_row(ax, y, label, lane_fc, replay, critic, target, effect, accent, crosses=()):
    box(ax, (0.25, y), (1.35, 0.58), label, fc=lane_fc, ec=accent, lw=1.6, fontsize=10.2, weight="bold")
    box(ax, (1.88, y), (1.75, 0.58), replay, fc="#ffffff", ec=accent, lw=1.2, fontsize=8.8)
    box(ax, (3.92, y), (1.98, 0.58), critic, fc="#ffffff", ec=accent, lw=1.2, fontsize=8.8)
    box(ax, (6.20, y), (2.08, 0.58), target, fc="#ffffff", ec=accent, lw=1.2, fontsize=8.8)
    box(ax, (8.58, y), (1.82, 0.58), effect, fc="#ffffff", ec=accent, lw=1.2, fontsize=8.4)

    y_mid = y + 0.29
    arrow(ax, (1.60, y_mid), (1.88, y_mid), color=accent)
    arrow(ax, (3.63, y_mid), (3.92, y_mid), color=accent)
    arrow(ax, (5.90, y_mid), (6.20, y_mid), color=accent)
    arrow(ax, (8.28, y_mid), (8.58, y_mid), color=accent)

    for c in crosses:
        cross(ax, c)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    ax.text(
        5.35,
        4.35,
        "Focused Semantic Ablation Switches",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        5.35,
        4.08,
        "All variants keep the same actor, APF interface, terrain/obstacle rule, training budget, and official evaluation protocol.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    headers = [
        ("Variant", 0.925),
        ("Replay Record", 2.755),
        ("Critic Update", 4.91),
        ("TD Target Construction", 7.24),
        ("Isolated Effect", 9.49),
    ]
    for text, x in headers:
        ax.text(x, 3.67, text, ha="center", va="center", fontsize=10.5, fontweight="bold")

    draw_row(
        ax,
        2.86,
        "Full\nDual-Semantic",
        COLORS["full"],
        "store both\n$a_{raw}$ and $a_{corr}$",
        "$Q_{head}(a_{raw})$\n+\n$Q_{tail}(a_{corr})$",
        "reconstruct\n$a'_{corr}$ under\nnext-state APF",
        "complete\nsemantic\nalignment",
        COLORS["blue"],
    )

    draw_row(
        ax,
        1.90,
        "Collapsed\nReplay",
        COLORS["collapsed"],
        "store $a_{corr}$\nin both channels\n(raw/corr collapsed)",
        "critic no longer\nsees raw intention\nas a distinct record",
        "target correction\nkept unchanged",
        "replay-level\nsemantic\ncollapse",
        COLORS["orange"],
        crosses=[(2.78, 2.19)],
    )

    draw_row(
        ax,
        0.94,
        "No Corrected\nTarget Recon.",
        COLORS["notarget"],
        "store both\n$a_{raw}$ and $a_{corr}$",
        "$Q_{head}(a_{raw})$\n+\n$Q_{tail}(a_{corr})$",
        "do not reconstruct\nAPF-corrected\n$a'_{corr}$",
        "target-side\nexecution\nmismatch",
        COLORS["purple"],
        crosses=[(7.25, 1.23)],
    )

    ax.text(
        5.35,
        0.35,
        "Purpose: isolate whether degradation comes from replay-level semantic collapse or from removing execution-consistent target reconstruction.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    for ext in ("pdf", "svg", "png"):
        out = OUT_DIR / f"{BASE_NAME}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Generated: {OUT_DIR / (BASE_NAME + '.pdf')}")
    print(f"Generated: {OUT_DIR / (BASE_NAME + '.svg')}")
    print(f"Generated: {OUT_DIR / (BASE_NAME + '.png')}")


if __name__ == "__main__":
    main()

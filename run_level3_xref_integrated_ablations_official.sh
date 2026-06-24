#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tang/matd3"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/tang/miniconda3/envs/maddpg_env/bin/python3}"

RUN_TAG="${RUN_TAG:-level3_xref_integrated_seed101_matchedval_official10x30_$(date +%Y%m%d_%H%M%S)}"
BASE_SPEC="${BASE_SPEC:-$ROOT_DIR/ablation_experiments/multi_seed_groupB_20260609_130927/seed_batches/batch_groupB_seed101_20260609_130927/results/post_eval_shared_spec.json}"

TRAIN_SEEDS="${TRAIN_SEEDS:-101}"
EVAL_SEEDS="${EVAL_SEEDS:-30088 30188 30288 30388 30488 30588 30688 30788 30888 30988}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
EPISODES="${EPISODES:-${TRAIN_EPISODES:-1000}}"
TRAIN_EPISODES="${TRAIN_EPISODES:-$EPISODES}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
XLA_GLOBAL="${XLA_GLOBAL:-1}"
JIT_COMPILE="${JIT_COMPILE:-1}"
FORCE_OUTER_JIT_COMPILE="${FORCE_OUTER_JIT_COMPILE:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
TF_SYNC_ON_FINISH="${TF_SYNC_ON_FINISH:-0}"
XLA_COMPILE_PARALLELISM="${XLA_COMPILE_PARALLELISM:-1}"

MODEL_VARIANT="${MODEL_VARIANT:-auto}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-shared_matched_validation}"
SELECTION_VALIDATION_SEEDS="${SELECTION_VALIDATION_SEEDS:-114817}"
SELECTION_VALIDATION_EPISODES="${SELECTION_VALIDATION_EPISODES:-10}"
SELECTION_VALIDATION_CANDIDATES="${SELECTION_VALIDATION_CANDIDATES:-best_by_team_sr,best,checkpoint,final,latest_ep}"
SUMMARY_FILENAME_TAG="${SUMMARY_FILENAME_TAG:-model_fr}"

MISSING_TRAIN_LABELS="${MISSING_TRAIN_LABELS:-matd3_cross_agent_ref_no_quality_gate matd3_cross_agent_ref_behavior_label matd3_cross_agent_ref_agent_success_behavior_label matd3_cross_agent_ref_reward_to_success_selector_clean_label}"
COMPLETED_EVAL_LABELS="${COMPLETED_EVAL_LABELS:-matd3_full_dual_semantic matd3_cross_agent_ref_agent_success matd3_cross_agent_ref_agent_quality matd3_cross_agent_ref_soft_advantage matd3_cross_agent_ref_selector_mix matd3_cross_agent_ref_progress_gate}"
TRAIN_LABELS_OVERRIDE="${TRAIN_LABELS_OVERRIDE:-$MISSING_TRAIN_LABELS}"
EVAL_LABELS_OVERRIDE="${EVAL_LABELS_OVERRIDE:-$COMPLETED_EVAL_LABELS $MISSING_TRAIN_LABELS}"

R2S_SUMMARY_DIR="${R2S_SUMMARY_DIR:-$ROOT_DIR/diagnostics/level3_r2s_selector_matchedval_official10x30_20260608_125056_summary_20260609_113654}"
R2S_AGG_CSV="${R2S_AGG_CSV:-$R2S_SUMMARY_DIR/dual_semantics_eval10x30_aggregated_model_fr.csv}"
R2S_BY_SEED_CSV="${R2S_BY_SEED_CSV:-$R2S_SUMMARY_DIR/dual_semantics_eval10x30_by_train_seed_model_fr.csv}"

INTEGRATED_LABEL_ORDER="${INTEGRATED_LABEL_ORDER:-matd3_full_dual_semantic matd3_cross_agent_ref_agent_success matd3_cross_agent_ref_agent_quality matd3_cross_agent_ref_soft_advantage matd3_cross_agent_ref_selector_mix matd3_cross_agent_ref_progress_gate matd3_cross_agent_ref_no_quality_gate matd3_cross_agent_ref_behavior_label matd3_cross_agent_ref_agent_success_behavior_label matd3_cross_agent_ref_reward_to_success_selector_tail0 matd3_cross_agent_ref_reward_to_success_selector_tail01 matd3_cross_agent_ref_reward_to_success_selector matd3_cross_agent_ref_reward_to_success_selector_clean_label matd3_cross_agent_ref_reward_to_success_selector_tail10}"
R2S_REUSE_LABELS="${R2S_REUSE_LABELS:-matd3_cross_agent_ref_reward_to_success_selector_tail0 matd3_cross_agent_ref_reward_to_success_selector_tail01 matd3_cross_agent_ref_reward_to_success_selector matd3_cross_agent_ref_reward_to_success_selector_tail10}"
STRICT_INTEGRATION="${STRICT_INTEGRATION:-1}"

echo "[xref-integrated] Step 1/2: train only missing ablations, then eval completed+missing labels"
echo "[xref-integrated] RUN_TAG=$RUN_TAG"
echo "[xref-integrated] train labels: $TRAIN_LABELS_OVERRIDE"
echo "[xref-integrated] eval labels : $EVAL_LABELS_OVERRIDE"
echo "[xref-integrated] reused R2S CSV: $R2S_AGG_CSV"
echo "[xref-integrated] acceleration: XLA_GLOBAL=$XLA_GLOBAL JIT_COMPILE=$JIT_COMPILE FORCE_OUTER_JIT_COMPILE=$FORCE_OUTER_JIT_COMPILE CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING TF_SYNC_ON_FINISH=$TF_SYNC_ON_FINISH XLA_COMPILE_PARALLELISM=$XLA_COMPILE_PARALLELISM"

BASE_SPEC="$BASE_SPEC" \
RUN_TAG="$RUN_TAG" \
TRAIN_LABELS_OVERRIDE="$TRAIN_LABELS_OVERRIDE" \
EVAL_LABELS_OVERRIDE="$EVAL_LABELS_OVERRIDE" \
TRAIN_SEEDS="$TRAIN_SEEDS" \
EVAL_SEEDS="$EVAL_SEEDS" \
EVAL_EPISODES="$EVAL_EPISODES" \
EPISODES="$EPISODES" \
TRAIN_EPISODES="$TRAIN_EPISODES" \
MAX_PARALLEL="$MAX_PARALLEL" \
XLA_GLOBAL="$XLA_GLOBAL" \
JIT_COMPILE="$JIT_COMPILE" \
FORCE_OUTER_JIT_COMPILE="$FORCE_OUTER_JIT_COMPILE" \
CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
TF_SYNC_ON_FINISH="$TF_SYNC_ON_FINISH" \
XLA_COMPILE_PARALLELISM="$XLA_COMPILE_PARALLELISM" \
MODEL_VARIANT="$MODEL_VARIANT" \
SELECTION_PROTOCOL="$SELECTION_PROTOCOL" \
SELECTION_VALIDATION_SEEDS="$SELECTION_VALIDATION_SEEDS" \
SELECTION_VALIDATION_EPISODES="$SELECTION_VALIDATION_EPISODES" \
SELECTION_VALIDATION_CANDIDATES="$SELECTION_VALIDATION_CANDIDATES" \
SUMMARY_FILENAME_TAG="$SUMMARY_FILENAME_TAG" \
FORCE_RERUN="${FORCE_RERUN:-0}" \
REUSE="${REUSE:-1}" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$ROOT_DIR/run_level3_dual_semantics_train_eval_official.sh"

echo "[xref-integrated] Step 2/2: merge new official eval with reused R2S/tail official CSV"

NEW_SUMMARY_DIR="$(find "$ROOT_DIR/diagnostics" -maxdepth 1 -type d -name "${RUN_TAG}_summary_*" -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {$1=""; sub(/^ /, ""); print}')"
if [ -z "$NEW_SUMMARY_DIR" ] || [ ! -d "$NEW_SUMMARY_DIR" ]; then
  echo "[xref-integrated] cannot find summary dir for RUN_TAG=$RUN_TAG" >&2
  exit 1
fi

NEW_AGG_CSV="$NEW_SUMMARY_DIR/dual_semantics_eval10x30_aggregated_${SUMMARY_FILENAME_TAG}.csv"
NEW_BY_SEED_CSV="$NEW_SUMMARY_DIR/dual_semantics_eval10x30_by_train_seed_${SUMMARY_FILENAME_TAG}.csv"
INTEGRATED_SUMMARY_DIR="${INTEGRATED_SUMMARY_DIR:-$ROOT_DIR/diagnostics/${RUN_TAG}_integrated_summary_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$INTEGRATED_SUMMARY_DIR"

MPLCONFIGDIR=/tmp/mplconfig "$PYTHON_BIN" - \
  "$INTEGRATED_SUMMARY_DIR" \
  "$NEW_AGG_CSV" \
  "$R2S_AGG_CSV" \
  "$NEW_BY_SEED_CSV" \
  "$R2S_BY_SEED_CSV" \
  "$INTEGRATED_LABEL_ORDER" \
  "$R2S_REUSE_LABELS" \
  "$STRICT_INTEGRATION" \
  "$RUN_TAG" \
  "$NEW_SUMMARY_DIR" \
  "$R2S_SUMMARY_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

(
    out_dir_raw,
    new_agg_raw,
    r2s_agg_raw,
    new_by_seed_raw,
    r2s_by_seed_raw,
    label_order_raw,
    r2s_labels_raw,
    strict_raw,
    run_tag,
    new_summary_dir_raw,
    r2s_summary_dir_raw,
) = sys.argv[1:]

out_dir = Path(out_dir_raw)
new_agg = Path(new_agg_raw)
r2s_agg = Path(r2s_agg_raw)
new_by_seed = Path(new_by_seed_raw)
r2s_by_seed = Path(r2s_by_seed_raw)
label_order = label_order_raw.split()
r2s_labels = set(r2s_labels_raw.split())
strict = str(strict_raw).strip().lower() not in {"0", "false", "no", "off"}

def read_rows(path: Path):
    if not path.exists():
        raise SystemExit(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            raise SystemExit(f"empty CSV header: {path}")
        return reader.fieldnames, rows

def write_rows(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def as_float(row, key, default=0.0):
    try:
        value = row.get(key)
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default

def short_name(row):
    label = str(row.get("label") or "").strip()
    label_aliases = {
        "matd3_full_dual_semantic": "Full",
        "matd3_cross_agent_ref_agent_success": "Agent Succ",
        "matd3_cross_agent_ref_agent_quality": "Agent Qual",
        "matd3_cross_agent_ref_soft_advantage": "Soft Adv",
        "matd3_cross_agent_ref_selector_mix": "Selector Mix",
        "matd3_cross_agent_ref_progress_gate": "Progress",
        "matd3_cross_agent_ref_no_quality_gate": "No Gate",
        "matd3_cross_agent_ref_behavior_label": "Beh Label",
        "matd3_cross_agent_ref_agent_success_behavior_label": "AS Beh",
        "matd3_cross_agent_ref_reward_to_success_selector_tail0": "R2S tail0",
        "matd3_cross_agent_ref_reward_to_success_selector_tail01": "R2S tail01",
        "matd3_cross_agent_ref_reward_to_success_selector": "R2S",
        "matd3_cross_agent_ref_reward_to_success_head_tail_selector": "R2S H/T",
        "matd3_cross_agent_ref_reward_to_success_selector_clean_label": "R2S clean",
        "matd3_cross_agent_ref_reward_to_success_selector_tail10": "R2S tail10",
    }
    if label in label_aliases:
        return label_aliases[label]
    name = str(row.get("display_name") or row.get("label") or "").strip()
    replacements = {
        "Cross-Agent Ref - ": "",
        "CrossRef ": "",
        "MATD3 ": "",
        "Reward-to-Success Selector": "R2S",
        "Full Dual-Semantic": "Full",
        "Agent Success Behavior Label": "AS Beh",
        "Behavior Label": "Beh Label",
        "No Quality Gate": "No Gate",
        "Agent Success": "Agent Succ",
        "Agent Quality": "Agent Qual",
        "Soft Advantage": "Soft Adv",
        "Selector Mix": "Selector Mix",
        "Progress Gate": "Progress",
    }
    for src, dst in replacements.items():
        name = name.replace(src, dst)
    return name

def annotate_bars(ax, bars, values, percent=False):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if percent:
            text = f"{value * 100:.1f}%"
        elif abs(value) >= 1000:
            text = f"{value / 1000:.0f}k"
        else:
            text = f"{value:.1f}"
        va = "bottom" if height >= 0 else "top"
        y = height + (0.01 if height >= 0 else -0.01)
        ax.text(bar.get_x() + bar.get_width() / 2, y, text, ha="center", va=va, fontsize=7, rotation=90)

def plot_dashboard(rows, output_path: Path):
    if not rows:
        return None
    labels = [short_name(row) for row in rows]
    x = np.arange(len(rows))
    palette = plt.get_cmap("tab20").colors
    colors = [palette[idx % len(palette)] for idx in range(len(rows))]

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    axes = axes.flatten()

    team_success = [as_float(row, "team_success_rate_mean") for row in rows]
    team_success_err = [as_float(row, "team_success_rate_std") for row in rows]
    bars = axes[0].bar(x, team_success, yerr=team_success_err, color=colors, capsize=3)
    axes[0].set_title("Team Success Rate", fontweight="bold")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25, linestyle="--")
    annotate_bars(axes[0], bars, team_success, percent=True)

    collision_free = [as_float(row, "collision_free_rate_mean") for row in rows]
    collision_free_err = [as_float(row, "collision_free_rate_std") for row in rows]
    bars = axes[1].bar(x, collision_free, yerr=collision_free_err, color=colors, capsize=3)
    axes[1].set_title("Collision-Free Rate", fontweight="bold")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25, linestyle="--")
    annotate_bars(axes[1], bars, collision_free, percent=True)

    distance = [as_float(row, "avg_team_final_goal_distance_mean") for row in rows]
    distance_err = [as_float(row, "avg_team_final_goal_distance_std") for row in rows]
    bars = axes[2].bar(x, distance, yerr=distance_err, color=colors, capsize=3)
    axes[2].set_title("Team Final Goal Distance", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=28, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25, linestyle="--")
    annotate_bars(axes[2], bars, distance)

    rewards = [as_float(row, "avg_reward_mean") for row in rows]
    reward_err = [as_float(row, "avg_reward_std") for row in rows]
    bars = axes[3].bar(x, rewards, yerr=reward_err, color=colors, capsize=3)
    axes[3].set_title("Average Reward", fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=28, ha="right")
    axes[3].grid(True, axis="y", alpha=0.25, linestyle="--")
    annotate_bars(axes[3], bars, rewards)

    terrain = np.asarray([as_float(row, "avg_terrain_collision_count_mean") for row in rows])
    obstacle = np.asarray([as_float(row, "avg_obstacle_collision_count_mean") for row in rows])
    inter_agent = np.asarray([as_float(row, "avg_inter_agent_collision_count_mean") for row in rows])
    axes[4].bar(x, terrain, color="#6BAED6", label="Terrain")
    axes[4].bar(x, obstacle, bottom=terrain, color="#FD8D3C", label="Obstacle")
    axes[4].bar(x, inter_agent, bottom=terrain + obstacle, color="#74C476", label="Inter-Agent")
    axes[4].set_title("Collision Breakdown", fontweight="bold")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(labels, rotation=28, ha="right")
    axes[4].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[4].legend(fontsize=9)

    any_success = [as_float(row, "agent_success_rate_any_mean") for row in rows]
    two_success = [as_float(row, "agent_success_rate_two_or_more_mean") for row in rows]
    width = 0.36
    bars_any = axes[5].bar(x - width / 2, any_success, width=width, color="#2CA02C", label="Any-Agent")
    bars_two = axes[5].bar(x + width / 2, two_success, width=width, color="#9467BD", label="Two-Agent")
    axes[5].set_title("Partial Arrival Rates", fontweight="bold")
    axes[5].set_ylim(0.0, 1.05)
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(labels, rotation=28, ha="right")
    axes[5].grid(True, axis="y", alpha=0.25, linestyle="--")
    axes[5].legend(fontsize=9)
    annotate_bars(axes[5], bars_any, any_success, percent=True)
    annotate_bars(axes[5], bars_two, two_success, percent=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path

def merge_csv(new_path: Path, r2s_path: Path, output_name: str):
    new_fields, new_rows = read_rows(new_path)
    r2s_fields, r2s_rows = read_rows(r2s_path)
    if new_fields != r2s_fields:
        raise SystemExit(
            f"CSV headers differ:\nnew={new_path}: {new_fields}\nr2s={r2s_path}: {r2s_fields}"
        )
    rows_by_label = {}
    for row in new_rows:
        label = str(row.get("label", "")).strip()
        if label and label not in r2s_labels:
            rows_by_label[label] = row
    for row in r2s_rows:
        label = str(row.get("label", "")).strip()
        if label in r2s_labels:
            rows_by_label[label] = row
    missing = [label for label in label_order if label not in rows_by_label]
    if missing and strict:
        raise SystemExit(f"missing labels in integrated CSV {output_name}: {missing}")
    ordered = [rows_by_label[label] for label in label_order if label in rows_by_label]
    extra = [
        row for label, row in sorted(rows_by_label.items())
        if label not in set(label_order)
    ]
    merged = ordered + extra
    output_path = out_dir / output_name
    write_rows(output_path, new_fields, merged)
    return output_path, missing, len(merged)

agg_out, agg_missing, agg_count = merge_csv(
    new_agg,
    r2s_agg,
    "dual_semantics_eval10x30_aggregated_integrated_model_fr.csv",
)
by_seed_out, by_seed_missing, by_seed_count = merge_csv(
    new_by_seed,
    r2s_by_seed,
    "dual_semantics_eval10x30_by_train_seed_integrated_model_fr.csv",
)
with agg_out.open(newline="", encoding="utf-8") as f:
    integrated_rows = list(csv.DictReader(f))
dashboard_out = plot_dashboard(
    integrated_rows,
    out_dir / "dual_semantics_eval10x30_dashboard_integrated_model_fr.png",
)

manifest = {
    "run_tag": run_tag,
    "new_summary_dir": str(Path(new_summary_dir_raw)),
    "reused_r2s_summary_dir": str(Path(r2s_summary_dir_raw)),
    "new_aggregated_csv": str(new_agg),
    "reused_r2s_aggregated_csv": str(r2s_agg),
    "integrated_aggregated_csv": str(agg_out),
    "integrated_by_train_seed_csv": str(by_seed_out),
    "integrated_dashboard_png": str(dashboard_out) if dashboard_out else "",
    "label_order": label_order,
    "r2s_reuse_labels": sorted(r2s_labels),
    "aggregated_rows": agg_count,
    "by_train_seed_rows": by_seed_count,
    "missing_aggregated_labels": agg_missing,
    "missing_by_train_seed_labels": by_seed_missing,
}
(out_dir / "integration_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"[xref-integrated] integrated aggregated CSV: {agg_out}")
print(f"[xref-integrated] integrated by-train-seed CSV: {by_seed_out}")
if dashboard_out:
    print(f"[xref-integrated] integrated dashboard PNG: {dashboard_out}")
print(f"[xref-integrated] integration manifest: {out_dir / 'integration_manifest.json'}")
PY

echo "[xref-integrated] complete"

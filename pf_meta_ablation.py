#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PF 预热元优化消融实验脚本

自动运行以下三组对比，并生成奖励/损失曲线：
1. 有预热（无元优化） vs 无预热
2. 有预热+元优化 vs 无预热
3. 有预热+元优化 vs 有预热无元优化

运行要求：
- 依赖 matplotlib
- 需要使用 run_optimized.sh 生成的 episode_rewards.json / loss_history.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("缺少 matplotlib，请先安装：pip install matplotlib")
    sys.exit(1)


COMPARISONS = [
    {
        "name": "warmup_vs_no_warmup",
        "title": "有预热(无元优化) vs 无预热",
        "configs": [
            {"label": "ablate_warmup_no_meta", "env": {"LEARNING_WARMUP_ENABLED": "1", "PF_META_WARMUP_ENABLED": "0"}},
            {"label": "ablate_no_warmup", "env": {"LEARNING_WARMUP_ENABLED": "0", "PF_META_WARMUP_ENABLED": "0"}},
        ],
    },
    {
        "name": "warmup_meta_vs_no_warmup",
        "title": "有预热+元优化 vs 无预热",
        "configs": [
            {"label": "ablate_warmup_meta", "env": {"LEARNING_WARMUP_ENABLED": "1", "PF_META_WARMUP_ENABLED": "1"}},
            {"label": "ablate_no_warmup", "env": {"LEARNING_WARMUP_ENABLED": "0", "PF_META_WARMUP_ENABLED": "0"}},
        ],
    },
    {
        "name": "warmup_meta_vs_warmup_nometa",
        "title": "有预热+元优化 vs 有预热无元优化",
        "configs": [
            {"label": "ablate_warmup_meta", "env": {"LEARNING_WARMUP_ENABLED": "1", "PF_META_WARMUP_ENABLED": "1"}},
            {"label": "ablate_warmup_no_meta", "env": {"LEARNING_WARMUP_ENABLED": "1", "PF_META_WARMUP_ENABLED": "0"}},
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="PF 预热元优化消融实验")
    parser.add_argument("--script", type=str, default="./run_optimized.sh",
                        help="训练启动脚本路径 (默认 ./run_optimized.sh)")
    parser.add_argument("--episodes", type=int, default=50, help="每个实验的训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024, help="训练批次大小")
    parser.add_argument("--use-weighted-reward", type=int, default=1, choices=[0, 1],
                        help="是否使用分项加权奖励 (传给 run_optimized.sh)")
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3"],
                        help="训练算法选择")
    parser.add_argument("--output-dir", type=str, default="ablation_outputs",
                        help="图表输出目录")
    parser.add_argument("--logs-root", type=str, default="logs",
                        help="训练日志根目录 (与 run_optimized.sh 一致)")
    parser.add_argument("--reuse", action="store_true",
                        help="若检测到同名实验已存在，则跳过重新训练，直接复用最新日志")
    return parser.parse_args()


def run_experiment(label: str, env_overrides: Dict[str, str], args, cache: Dict[str, Dict]):
    if args.reuse and label in cache:
        print(f"[复用] {label}")
        return cache[label]

    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault("LEARNING_WARMUP_ENABLED", "0")
    env.setdefault("PF_META_WARMUP_ENABLED", "0")
    env["EXP_NAME"] = label

    cmd = [
        args.script,
        str(args.episodes),
        str(args.batch_size),
        label,
        str(args.use_weighted_reward),
        args.algorithm,
    ]

    print(f"[运行] {label} 环境变量: LEARNING_WARMUP_ENABLED={env.get('LEARNING_WARMUP_ENABLED')}, "
          f"PF_META_WARMUP_ENABLED={env.get('PF_META_WARMUP_ENABLED')}")
    subprocess.run(cmd, check=True, env=env)

    log_dir = find_latest_log_dir(label, args.logs_root)
    metrics = load_metrics(log_dir)
    result = {"label": label, "log_dir": log_dir, "metrics": metrics}
    cache[label] = result
    return result


def find_latest_log_dir(exp_name: str, logs_root: str) -> str:
    base = Path(logs_root) / exp_name
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"未找到日志目录: {base}")
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"{base} 下没有训练记录")
    return str(subdirs[-1])


def load_metrics(log_dir: str) -> Dict:
    metrics = {}
    ep_path = Path(log_dir) / "episode_rewards.json"
    if ep_path.exists():
        with open(ep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metrics["episode_rewards"] = data.get("episode_rewards", [])
        else:
            metrics["episode_rewards"] = data
    else:
        metrics["episode_rewards"] = []

    loss_path = Path(log_dir) / "loss_history.json"
    if loss_path.exists():
        with open(loss_path, "r", encoding="utf-8") as f:
            metrics["loss_history"] = json.load(f)
    else:
        metrics["loss_history"] = []
    return metrics


def plot_rewards(series: List[Dict], title: str, output_path: Path):
    plt.figure(figsize=(10, 5))
    has_data = False
    for item in series:
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        plt.plot(range(1, len(rewards) + 1), rewards, label=item["label"])
    plt.title(f"{title} - 奖励曲线")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if has_data:
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
    else:
        print(f"[警告] {title} 没有可用的奖励数据，跳过绘图：{output_path}")
    plt.close()


def plot_losses(series: List[Dict], title: str, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    has_data = False
    for item in series:
        history = item["metrics"].get("loss_history", [])
        if not history:
            continue
        has_data = True
        steps = [entry.get("step", idx) for idx, entry in enumerate(history)]
        critic = [entry.get("critic_loss") for entry in history]
        actor = [entry.get("actor_loss") for entry in history]
        axes[0].plot(steps, critic, label=item["label"])
        axes[1].plot(steps, actor, label=item["label"])
    axes[0].set_title("Critic Loss")
    axes[1].set_title("Actor Loss")
    for ax in axes:
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{title} - Loss 曲线")
    plt.tight_layout()
    if has_data:
        plt.savefig(output_path, dpi=200)
    else:
        print(f"[警告] {title} 没有可用的loss数据，跳过绘图：{output_path}")
    plt.close(fig)


def main():
    args = parse_args()
    script_path = os.path.abspath(args.script)
    if not os.path.isfile(script_path):
        print(f"[错误] 找不到训练脚本: {script_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, Dict] = {}
    summary = {}

    for comp in COMPARISONS:
        print(f"\n==== 对比: {comp['title']} ====")
        series = []
        for cfg in comp["configs"]:
            label = cfg["label"]
            env_overrides = cfg["env"].copy()
            result = run_experiment(label, env_overrides, args, cache)
            series.append(result)
            summary[label] = result["log_dir"]

        reward_png = output_dir / f"{comp['name']}_reward.png"
        loss_png = output_dir / f"{comp['name']}_loss.png"
        plot_rewards(series, comp["title"], reward_png)
        plot_losses(series, comp["title"], loss_png)
        print(f"[完成] 奖励曲线: {reward_png}")
        print(f"[完成] Loss曲线: {loss_png}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=False, indent=2)
    print(f"\n所有图表与汇总已输出到 {output_dir}")


if __name__ == "__main__":
    main()



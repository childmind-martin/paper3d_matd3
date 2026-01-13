#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于批次结构的消融实验图表生成脚本（简化版）

使用方法：
    # 使用最新批次
    python3 regenerate_plots_from_batch.py
    
    # 指定批次ID
    python3 regenerate_plots_from_batch.py --batch-id batch_20251228_031438
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ Matplotlib已配置英文字体")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装依赖: pip install matplotlib scipy numpy")
    sys.exit(1)

from ablation_batch_manager import AblationBatchManager


def smooth_curve(data: np.ndarray, method: str = "moving_average", window: int = 10) -> np.ndarray:
    """平滑曲线"""
    if len(data) < 2:
        return data
    
    if method == "moving_average":
        return uniform_filter1d(data.astype(float), size=window, mode='nearest')
    elif method == "poly":
        if len(data) < 3:
            return data
        x = np.arange(len(data))
        degree = min(5, len(data) - 1)
        coeffs = np.polyfit(x, data, degree)
        poly = np.poly1d(coeffs)
        return poly(x)
    else:
        return data


def plot_rewards(series, output_path):
    """绘制奖励对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']
    
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if rewards:
            episodes = range(1, len(rewards) + 1)
            smoothed = smooth_curve(np.array(rewards), method="moving_average", window=10)
            ax.plot(episodes, smoothed, 
                   label=item["name_en"], 
                   color=colors[idx % len(colors)], 
                   linewidth=2.5, 
                   alpha=0.9)
    
    ax.set_title("Reward Comparison", fontsize=16, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 奖励对比图: {output_path.name}")


def plot_collisions(series, output_path):
    """绘制碰撞次数对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']
    
    for idx, item in enumerate(series):
        collisions = item["metrics"].get("collision_counts", [])
        if collisions:
            episodes = range(1, len(collisions) + 1)
            smoothed = smooth_curve(np.array(collisions), method="moving_average", window=10)
            ax.plot(episodes, smoothed, 
                   label=item["name_en"], 
                   color=colors[idx % len(colors)], 
                   linewidth=2.5, 
                   alpha=0.9)
    
    ax.set_title("Collision Counts (Smoothed)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Collision Count", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 碰撞次数对比图: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="基于批次结构生成消融实验图表")
    parser.add_argument("--batch-id", type=str, default=None,
                       help="批次ID（默认使用latest）")
    args = parser.parse_args()
    
    print("="*70)
    print("基于批次结构生成消融实验图表")
    print("="*70)
    
    # 使用批次管理器
    manager = AblationBatchManager()
    
    try:
        batch_dir = manager.get_batch_dir(args.batch_id)
        print(f"\n✅ 使用批次: {batch_dir.name}")
        print(f"   路径: {batch_dir}")
        
        # 读取配置
        config_file = batch_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"   创建时间: {config.get('created_at', 'unknown')}")
            print(f"   回合数: {config.get('episodes', 'unknown')}")
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n可用的批次:")
        for batch in manager.list_batches():
            print(f"  - {batch['batch_id']}")
            print(f"    创建时间: {batch['config'].get('created_at', 'unknown')}")
        return 1
    
    # 输出目录
    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 实验配置
    experiments = [
        {"label": "apf_learnable", "name_en": "APF Learnable"},
        {"label": "apf_traditional", "name_en": "APF Traditional"},
        {"label": "action_apf_fusion", "name_en": "Action+APF Fusion"},
        {"label": "action_only", "name_en": "Action Only"}
    ]
    
    # 加载数据
    print("\n加载实验数据...")
    series = []
    for exp_config in experiments:
        label = exp_config["label"]
        name_en = exp_config["name_en"]
        
        print(f"  {label}...", end=" ")
        
        try:
            # 加载 episode_rewards.json
            data = manager.load_experiment_data(label, batch_dir.name, "episode_rewards.json")
            
            metrics = {}
            if isinstance(data, dict):
                metrics["episode_rewards"] = data.get("episode_rewards", [])
                metrics["success_flags"] = data.get("success_flags", [])
                metrics["collision_counts"] = data.get("collision_counts", [])
                metrics["min_distances_to_obstacle"] = data.get("min_distances_to_obstacle", [])
                metrics["agent_success_flags"] = data.get("agent_success_flags", [])
                metrics["team_success_flags"] = data.get("team_success_flags", [])
                metrics["team_success_rate"] = data.get("team_success_rate", 0.0)
            else:
                metrics["episode_rewards"] = data
                metrics["collision_counts"] = []
            
            # 加载 loss_history.json
            try:
                loss_data = manager.load_experiment_data(label, batch_dir.name, "loss_history.json")
                metrics["loss_history"] = loss_data
            except FileNotFoundError:
                metrics["loss_history"] = []
            
            series.append({
                "label": label,
                "name_en": name_en,
                "metrics": metrics,
                "log_dir": str(manager.get_experiment_dir(label, batch_dir.name))
            })
            
            print("✓")
            
        except FileNotFoundError:
            print("❌ 数据文件不存在")
        except Exception as e:
            print(f"❌ {e}")
    
    if not series:
        print("\n❌ 错误：未找到任何有效数据")
        return 1
    
    print(f"\n成功加载 {len(series)} 个实验的数据")
    
    # 生成图表
    print("\n生成图表...")
    
    # 1. 奖励对比图
    reward_path = output_dir / f"reward_comparison_{timestamp}.png"
    plot_rewards(series, reward_path)
    
    # 2. 碰撞次数对比图
    collision_path = output_dir / f"collision_comparison_{timestamp}.png"
    plot_collisions(series, collision_path)
    
    # 生成摘要
    summary = {
        "batch_id": batch_dir.name,
        "timestamp": timestamp,
        "experiments": [
            {
                "label": item["label"],
                "name": item["name_en"],
                "episodes": len(item["metrics"].get("episode_rewards", [])),
                "final_reward": float(item["metrics"]["episode_rewards"][-1]) if item["metrics"].get("episode_rewards") else None,
                "avg_reward": float(np.mean(item["metrics"]["episode_rewards"])) if item["metrics"].get("episode_rewards") else None,
                "team_success_rate": item["metrics"].get("team_success_rate", 0.0)
            }
            for item in series
        ]
    }
    
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ 图表生成完成")
    print("="*70)
    print(f"\n输出目录: {output_dir}/")
    print(f"  - reward_comparison_{timestamp}.png")
    print(f"  - collision_comparison_{timestamp}.png")
    print(f"  - summary_{timestamp}.json")
    
    print("\n实验结果汇总:")
    for exp in summary["experiments"]:
        print(f"  - {exp['name']}:")
        print(f"      Episodes: {exp['episodes']}")
        if exp['final_reward'] is not None:
            print(f"      Final Reward: {exp['final_reward']:.2f}")
            print(f"      Avg Reward: {exp['avg_reward']:.2f}")
        print(f"      Team Success Rate: {exp['team_success_rate']:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

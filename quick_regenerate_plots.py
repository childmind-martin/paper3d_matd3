#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速重新生成消融实验图表
直接读取已保存的数据，重新生成图片（使用英文标签）
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 导入绘图模块
sys.path.insert(0, str(Path(__file__).parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    import numpy as np
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 导入绘图函数
from ablation_action_pf_comparison import (
    plot_comparison_rewards,
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance,
    plot_comparison_losses,
    generate_interactive_comparison
)


def load_experiment_data(log_dir):
    """加载实验数据"""
    log_path = Path(log_dir)
    
    # 🔧 修复：优先查找episode_rewards.json（包含完整指标），如果没有再查找results.json
    episode_rewards_file = None
    loss_history_file = None
    results_file = None
    
    # 先尝试直接路径
    if (log_path / "episode_rewards.json").exists():
        episode_rewards_file = log_path / "episode_rewards.json"
    elif (log_path / "results.json").exists():
        results_file = log_path / "results.json"
    else:
        # 查找子目录中的文件
        subdirs = [d for d in log_path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            if (subdir / "episode_rewards.json").exists():
                episode_rewards_file = subdir / "episode_rewards.json"
                break
            elif (subdir / "results.json").exists() and results_file is None:
                results_file = subdir / "results.json"
    
    # 查找loss_history.json
    if episode_rewards_file:
        # episode_rewards.json在同一目录下
        loss_history_file = episode_rewards_file.parent / "loss_history.json"
    elif results_file:
        # results.json在同一目录下
        loss_history_file = results_file.parent / "loss_history.json"
    else:
        # 查找子目录中的loss_history.json
        subdirs = [d for d in log_path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            if (subdir / "loss_history.json").exists():
                loss_history_file = subdir / "loss_history.json"
                break
    
    metrics = {
        "episode_rewards": [],
        "loss_history": [],
        "success_flags": [],
        "collision_counts": [],
        "min_distances_to_obstacle": [],
        "agent_success_flags": [],
        "team_success_flags": [],
        "agent_success_rates": [],
        "team_success_rate": 0.0
    }
    
    # 优先从episode_rewards.json加载（包含完整指标）
    if episode_rewards_file:
        print(f"    读取: {episode_rewards_file}")
        with open(episode_rewards_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        metrics["episode_rewards"] = data.get("episode_rewards", [])
        metrics["success_flags"] = data.get("success_flags", [])
        metrics["collision_counts"] = data.get("collision_counts", [])
        metrics["min_distances_to_obstacle"] = data.get("min_distances_to_obstacle", [])
        metrics["agent_success_flags"] = data.get("agent_success_flags", [])
        metrics["team_success_flags"] = data.get("team_success_flags", [])
        metrics["agent_success_rates"] = data.get("agent_success_rates", [])
        metrics["team_success_rate"] = data.get("team_success_rate", 0.0)
    elif results_file:
        # 从results.json加载（可能只有rewards字段）
        print(f"    读取: {results_file}")
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 🔧 修复：results.json中可能使用"rewards"字段而不是"episode_rewards"
        metrics["episode_rewards"] = data.get("episode_rewards", data.get("rewards", []))
        # results.json通常不包含其他指标，所以保持默认值
    else:
        print(f"    警告：找不到 episode_rewards.json 或 results.json 在 {log_dir}")
        return metrics
    
    # 加载损失数据
    if loss_history_file and loss_history_file.exists():
        print(f"    读取: {loss_history_file}")
        with open(loss_history_file, 'r', encoding='utf-8') as f:
            metrics["loss_history"] = json.load(f)
    
    return metrics


def main():
    print("="*70)
    print("快速重新生成消融实验图表（基于已保存数据）")
    print("="*70)
    
    # 输出目录
    output_dir = Path("ablation_action_pf_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 查找最新的实验数据
    logs_root = Path("logs")
    
    # 实验配置（使用英文名称）
    experiments = [
        {"label": "apf_learnable", "name_en": "APF Learnable"},
        {"label": "apf_traditional", "name_en": "APF Traditional"},
        {"label": "action_apf_fusion", "name_en": "Action+APF Fusion"}
    ]
    
    series = []
    
    print("\n加载实验数据...")
    for exp in experiments:
        label = exp["label"]
        name_en = exp["name_en"]
        
        # 查找最新的日志目录
        matching_dirs = [d for d in logs_root.iterdir() 
                        if d.is_dir() and d.name.startswith(label)]
        
        if not matching_dirs:
            print(f"  ✗ {label}: 未找到数据")
            continue
        
        # 获取最新目录
        latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
        print(f"  ✓ {label}: {latest_dir.name}")
        
        # 加载数据
        metrics = load_experiment_data(latest_dir)
        
        if not metrics.get("episode_rewards"):
            print(f"    警告: 无有效数据")
            continue
        
        series.append({
            "label": label,
            "name": name_en,  # 使用英文名称
            "name_en": name_en,
            "metrics": metrics,
            "log_dir": str(latest_dir)
        })
    
    if not series:
        print("\n错误：未找到任何有效数据")
        return 1
    
    print(f"\n成功加载 {len(series)} 个实验的数据")
    
    # 生成图表
    print("\n生成图表...")
    
    # 1. 奖励对比图
    print("  1/5 生成奖励对比图...")
    reward_path = output_dir / f"reward_comparison_{timestamp}.png"
    plot_comparison_rewards(
        series, 
        "Action vs APF Correction Ablation Comparison",
        reward_path,
        smooth_window=10,
        fit_method="moving_average"
    )
    
    # 2. 成功率/碰撞/净空对比图
    print("  2/5 生成成功率/碰撞/净空对比图...")
    scc_path = output_dir / f"success_collision_clearance_comparison_{timestamp}.png"
    plot_comparison_success_collision_clearance(
        series,
        "Action vs APF Correction Ablation Comparison",
        scc_path,
        smooth_window=10,
        fit_method="moving_average"
    )
    
    # 3. 成功率与最小安全距离对比图
    print("  3/5 生成成功率与最小安全距离对比图...")
    src_path = output_dir / f"success_rate_and_clearance_comparison_{timestamp}.png"
    plot_comparison_success_rate_and_clearance(
        series,
        "Action vs APF Correction Ablation Comparison",
        src_path,
        smooth_window=10,
        fit_method="moving_average"
    )
    
    # 4. 损失对比图
    print("  4/5 生成损失对比图...")
    loss_path = output_dir / f"loss_comparison_{timestamp}.png"
    plot_comparison_losses(
        series,
        "Action vs APF Correction Ablation Comparison",
        loss_path
    )
    
    # 5. 交互式图表（可选）
    print("  5/5 生成交互式图表...")
    try:
        interactive_path = output_dir / f"interactive_comparison_{timestamp}.html"
        generate_interactive_comparison(
            series,
            "Action vs APF Correction Ablation Comparison",
            interactive_path,
            smooth_window=10,
            fit_method="moving_average"
        )
    except Exception as e:
        print(f"    警告: 交互式图表生成失败: {e}")
    
    # 生成摘要
    summary = {
        "timestamp": timestamp,
        "experiments": [
            {
                "label": item["label"],
                "name": item["name_en"],
                "log_dir": item["log_dir"],
                "episodes": len(item["metrics"].get("episode_rewards", [])),
                "team_success_rate": item["metrics"].get("team_success_rate", 0.0)
            }
            for item in series
        ]
    }
    
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ 所有图表生成完成")
    print("="*70)
    print(f"\n输出目录: {output_dir}/")
    print(f"  - reward_comparison_{timestamp}.png")
    print(f"  - success_collision_clearance_comparison_{timestamp}.png")
    print(f"  - success_rate_and_clearance_comparison_{timestamp}.png")
    print(f"  - loss_comparison_{timestamp}.png")
    print(f"  - interactive_comparison_{timestamp}.html")
    print(f"  - summary_{timestamp}.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


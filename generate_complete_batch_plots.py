#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的消融实验批次图表生成器
为批次生成所有对比图和单独分析图
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    from scipy.interpolate import make_interp_spline
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ Matplotlib configured with English fonts")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

from ablation_batch_manager import AblationBatchManager


def smooth_curve(data: np.ndarray, method: str = "moving_average", window: int = 10) -> np.ndarray:
    """平滑曲线"""
    if len(data) < 2:
        return data
    
    if method == "moving_average":
        return uniform_filter1d(data.astype(float), size=window, mode='nearest')
    elif method == "spline":
        if len(data) < 4:
            return data
        x = np.arange(len(data))
        try:
            spl = make_interp_spline(x, data, k=min(3, len(data)-1))
            return spl(x)
        except:
            return data
    else:
        return data


# ============================================================================
# 第1部分: 4张完整对比图
# ============================================================================

def plot_1_reward_comparison(series, output_path, smooth_window=20):
    """对比图1: 奖励对比"""
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    has_data = False
    
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        
        episodes = np.arange(1, len(rewards) + 1)
        rewards_array = np.array(rewards, dtype=float)
        smoothed = smooth_curve(rewards_array, method="moving_average", window=smooth_window)
        
        name = item.get('name_en', item.get('name', 'Unknown'))
        color = colors[idx % len(colors)]
        
        # 绘制原始数据（半透明）
        ax.plot(episodes, rewards_array, color=color, alpha=0.2, linewidth=0.8)
        # 绘制平滑数据
        ax.plot(episodes, smoothed, label=name, color=color, linewidth=2.5, alpha=0.9)
    
    if has_data:
        ax.set_title("Reward Comparison Across Experiments", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Episode", fontsize=13)
        ax.set_ylabel("Episode Reward", fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Generated: {output_path.name}")
    else:
        print(f"  ✗ No reward data available")
    plt.close(fig)


def plot_2_loss_comparison(series, output_path):
    """对比图2: 损失对比（Critic和Actor）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    has_data = False
    
    for idx, item in enumerate(series):
        history = item["metrics"].get("loss_history", [])
        if not history:
            continue
        has_data = True
        
        steps = [entry.get("step", i) for i, entry in enumerate(history)]
        critic = [entry.get("critic_loss", 0) for entry in history]
        actor = [entry.get("actor_loss", 0) for entry in history]
        
        name = item.get('name_en', item.get('name', 'Unknown'))
        color = colors[idx % len(colors)]
        
        axes[0].plot(steps, critic, label=name, color=color, linewidth=2.0, alpha=0.85)
        axes[1].plot(steps, actor, label=name, color=color, linewidth=2.0, alpha=0.85)
    
    if has_data:
        axes[0].set_title("Critic Loss Comparison", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Critic Loss", fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper right', fontsize=10)
        
        axes[1].set_title("Actor Loss Comparison", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Training Step", fontsize=12)
        axes[1].set_ylabel("Actor Loss", fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper right', fontsize=10)
        
        fig.suptitle("Loss Comparison Across Experiments", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Generated: {output_path.name}")
    else:
        print(f"  ✗ No loss data available")
    plt.close(fig)


def plot_3_success_collision_clearance(series, output_path, smooth_window=10):
    """对比图3: 成功率、碰撞次数、净空距离综合对比"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    has_data = False
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        name = item.get('name_en', item.get('name', 'Unknown'))
        color = colors[idx % len(colors)]
        
        # 1. 成功率（移动平均）
        success_flags = metrics.get("success_flags", [])
        if success_flags:
            has_data = True
            window_size = 50
            success_array = np.array(success_flags, dtype=float)
            success_rate = []
            for i in range(len(success_array)):
                start_idx = max(0, i - window_size + 1)
                window_data = success_array[start_idx:i+1]
                rate = np.mean(window_data) if len(window_data) > 0 else 0.0
                success_rate.append(rate)
            
            episodes = np.arange(1, len(success_rate) + 1)
            axes[0].plot(episodes, success_rate, label=name, color=color, linewidth=2.5, alpha=0.9)
        
        # 2. 碰撞次数
        collision_counts = metrics.get("collision_counts", [])
        if collision_counts:
            has_data = True
            episodes = np.arange(1, len(collision_counts) + 1)
            collisions_array = np.array(collision_counts, dtype=float)
            smoothed = smooth_curve(collisions_array, method="moving_average", window=smooth_window)
            axes[1].plot(episodes, smoothed, label=name, color=color, linewidth=2.5, alpha=0.9)
        
        # 3. 平均净空距离
        min_distances = metrics.get("min_distances_to_obstacle", [])
        if min_distances:
            has_data = True
            episodes = np.arange(1, len(min_distances) + 1)
            
            # 处理字典格式的距离数据
            if isinstance(min_distances[0], dict):
                dist_values = []
                for d in min_distances:
                    mean_val = d.get('mean', None) if isinstance(d, dict) else None
                    if mean_val is not None and np.isfinite(mean_val):
                        dist_values.append(float(mean_val))
                    else:
                        dist_values.append(np.nan)
            else:
                dist_values = [float(d) if d is not None and np.isfinite(d) else np.nan 
                              for d in min_distances]
            
            # 过滤有效数据
            valid_mask = np.isfinite(dist_values)
            if np.any(valid_mask):
                dist_array = np.array(dist_values)
                dist_array[~valid_mask] = np.nan
                smoothed = smooth_curve(dist_array[valid_mask], method="moving_average", window=smooth_window)
                valid_episodes = episodes[valid_mask]
                axes[2].plot(valid_episodes, smoothed, label=name, color=color, linewidth=2.5, alpha=0.9)
    
    if has_data:
        axes[0].set_title("Success Rate (Moving Average, Window=50)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Success Rate", fontsize=11)
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='lower right', fontsize=10)
        
        axes[1].set_title("Collision Counts (Smoothed)", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Collision Count", fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper right', fontsize=10)
        
        axes[2].set_title("Average Clearance Distance (Smoothed)", fontsize=13, fontweight='bold')
        axes[2].set_xlabel("Episode", fontsize=11)
        axes[2].set_ylabel("Distance (m)", fontsize=11)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(loc='lower right', fontsize=10)
        
        fig.suptitle("Success Rate, Collision, and Clearance Comparison", 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Generated: {output_path.name}")
    else:
        print(f"  ✗ No success/collision/clearance data available")
    plt.close(fig)


def plot_4_detailed_safety_analysis(series, output_path):
    """对比图4: 详细安全性分析（分布、CDF、分位数）"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    collision_threshold = 1.5
    has_data = False
    
    # 创建子图
    ax1 = fig.add_subplot(gs[0, :])  # 最小距离时间序列
    ax2 = fig.add_subplot(gs[1, 0])  # 直方图
    ax3 = fig.add_subplot(gs[1, 1])  # CDF
    ax4 = fig.add_subplot(gs[2, 0])  # 分位数
    ax5 = fig.add_subplot(gs[2, 1])  # 违反概率
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        name = item.get('name_en', item.get('name', 'Unknown'))
        color = colors[idx % len(colors)]
        
        min_distances = metrics.get("min_distances_to_obstacle", [])
        if not min_distances:
            continue
        has_data = True
        
        # 提取有效的距离值
        if isinstance(min_distances[0], dict):
            dist_values = [d.get('mean', np.nan) if isinstance(d, dict) else np.nan 
                          for d in min_distances]
        else:
            dist_values = [float(d) if d is not None and np.isfinite(d) else np.nan 
                          for d in min_distances]
        
        valid_distances = [d for d in dist_values if np.isfinite(d) and d > -1000]
        if not valid_distances:
            continue
        
        episodes = np.arange(1, len(valid_distances) + 1)
        dist_array = np.array(valid_distances)
        
        # 1. 时间序列（平滑）
        smoothed = smooth_curve(dist_array, method="moving_average", window=10)
        ax1.plot(episodes, smoothed, label=name, color=color, linewidth=2.0, alpha=0.85)
        ax1.axhline(y=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 2. 直方图
        ax2.hist(dist_array, bins=30, alpha=0.5, label=name, color=color, edgecolor='black')
        
        # 3. CDF
        sorted_dist = np.sort(dist_array)
        cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        ax3.plot(sorted_dist, cdf, label=name, color=color, linewidth=2.5, alpha=0.9)
        ax3.axvline(x=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 4. 分位数
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_values = np.quantile(dist_array, quantiles)
        x_pos = np.arange(len(quantiles)) + idx * 0.15
        ax4.bar(x_pos, quantile_values, width=0.15, label=name, color=color, alpha=0.85)
        
        # 5. 违反概率（随episode变化）
        window_size = 50
        violation_probs = []
        for i in range(len(dist_array)):
            start_idx = max(0, i - window_size + 1)
            window = dist_array[start_idx:i+1]
            prob = np.mean(window < collision_threshold)
            violation_probs.append(prob)
        ax5.plot(episodes, violation_probs, label=name, color=color, linewidth=2.0, alpha=0.85)
    
    if has_data:
        # 格式化子图
        ax1.set_title("Minimum Distance Time Series (Smoothed)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Episode", fontsize=10)
        ax1.set_ylabel("Distance (m)", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        ax2.set_title("Distance Distribution (Histogram)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Distance (m)", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=9)
        
        ax3.set_title("Cumulative Distribution Function (CDF)", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Distance (m)", fontsize=10)
        ax3.set_ylabel("Cumulative Probability", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        ax4.set_title("Distance Quantiles", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Quantile", fontsize=10)
        ax4.set_ylabel("Distance (m)", fontsize=10)
        ax4.set_xticks(np.arange(len(quantiles)) + 0.225)
        ax4.set_xticklabels(['10%', '25%', '50%', '75%', '90%'])
        ax4.axhline(y=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(fontsize=9)
        
        ax5.set_title("Safety Violation Probability (Window=50)", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Episode", fontsize=10)
        ax5.set_ylabel("Violation Probability", fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
        
        fig.suptitle("Detailed Safety Analysis Comparison", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Generated: {output_path.name}")
    else:
        print(f"  ✗ No safety analysis data available")
    plt.close(fig)


# ============================================================================
# 第2部分: 每个实验的单独详细分析图
# ============================================================================

def plot_individual_experiment(exp_name, metrics, output_path):
    """为单个实验生成详细分析图"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    has_data = False
    
    # 1. 奖励曲线
    ax1 = fig.add_subplot(gs[0, :])
    rewards = metrics.get("episode_rewards", [])
    if rewards:
        has_data = True
        episodes = np.arange(1, len(rewards) + 1)
        rewards_array = np.array(rewards, dtype=float)
        smoothed = smooth_curve(rewards_array, method="moving_average", window=20)
        ax1.plot(episodes, rewards_array, color='lightblue', alpha=0.3, linewidth=0.8, label='Raw')
        ax1.plot(episodes, smoothed, color='#1f77b4', linewidth=2.5, label='Smoothed')
        ax1.set_title("Episode Rewards", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Reward", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. 成功率
    ax2 = fig.add_subplot(gs[1, 0])
    success_flags = metrics.get("success_flags", [])
    if success_flags:
        has_data = True
        window_size = 50
        success_array = np.array(success_flags, dtype=float)
        success_rate = []
        for i in range(len(success_array)):
            start_idx = max(0, i - window_size + 1)
            window_data = success_array[start_idx:i+1]
            rate = np.mean(window_data) if len(window_data) > 0 else 0.0
            success_rate.append(rate)
        episodes = np.arange(1, len(success_rate) + 1)
        ax2.plot(episodes, success_rate, color='#2ca02c', linewidth=2.5)
        ax2.set_title("Success Rate (Window=50)", fontsize=13, fontweight='bold')
        ax2.set_ylabel("Success Rate", fontsize=11)
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
    
    # 3. 碰撞次数
    ax3 = fig.add_subplot(gs[1, 1])
    collision_counts = metrics.get("collision_counts", [])
    if collision_counts:
        has_data = True
        episodes = np.arange(1, len(collision_counts) + 1)
        collisions = np.array(collision_counts, dtype=float)
        smoothed = smooth_curve(collisions, method="moving_average", window=10)
        ax3.plot(episodes, smoothed, color='#d62728', linewidth=2.5)
        ax3.set_title("Collision Counts (Smoothed)", fontsize=13, fontweight='bold')
        ax3.set_ylabel("Collisions", fontsize=11)
        ax3.grid(True, alpha=0.3)
    
    # 4. 最小距离时间序列
    ax4 = fig.add_subplot(gs[2, :])
    min_distances = metrics.get("min_distances_to_obstacle", [])
    if min_distances:
        has_data = True
        if isinstance(min_distances[0], dict):
            dist_values = [d.get('mean', np.nan) if isinstance(d, dict) else np.nan 
                          for d in min_distances]
        else:
            dist_values = [float(d) if d is not None and np.isfinite(d) else np.nan 
                          for d in min_distances]
        
        valid_distances = [d for d in dist_values if np.isfinite(d) and d > -1000]
        if valid_distances:
            episodes = np.arange(1, len(valid_distances) + 1)
            dist_array = np.array(valid_distances)
            smoothed = smooth_curve(dist_array, method="moving_average", window=10)
            ax4.plot(episodes, smoothed, color='#ff7f0e', linewidth=2.5)
            ax4.axhline(y=1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision Threshold')
            ax4.set_title("Minimum Distance to Obstacles", fontsize=13, fontweight='bold')
            ax4.set_ylabel("Distance (m)", fontsize=11)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
    
    # 5. 距离分布直方图
    ax5 = fig.add_subplot(gs[3, 0])
    if min_distances and valid_distances:
        has_data = True
        ax5.hist(dist_array, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
        ax5.axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax5.set_title("Distance Distribution", fontsize=13, fontweight='bold')
        ax5.set_xlabel("Distance (m)", fontsize=11)
        ax5.set_ylabel("Frequency", fontsize=11)
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 损失曲线
    ax6 = fig.add_subplot(gs[3, 1])
    loss_history = metrics.get("loss_history", [])
    if loss_history:
        has_data = True
        steps = [entry.get("step", i) for i, entry in enumerate(loss_history)]
        critic = [entry.get("critic_loss", 0) for entry in loss_history]
        actor = [entry.get("actor_loss", 0) for entry in loss_history]
        ax6.plot(steps, critic, label='Critic Loss', color='#e377c2', linewidth=2.0)
        ax6.plot(steps, actor, label='Actor Loss', color='#7f7f7f', linewidth=2.0)
        ax6.set_title("Training Losses", fontsize=13, fontweight='bold')
        ax6.set_xlabel("Step", fontsize=11)
        ax6.set_ylabel("Loss", fontsize=11)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    if has_data:
        fig.suptitle(f"Detailed Analysis: {exp_name}", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Generated individual analysis: {output_path.name}")
    else:
        print(f"  ✗ No data for {exp_name}")
    plt.close(fig)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """生成完整的批次图表"""
    print("=" * 80)
    print("完整消融实验批次图表生成器")
    print("=" * 80)
    
    # 初始化批次管理器
    manager = AblationBatchManager()
    
    # 获取最新批次
    batch_dir = manager.get_batch_dir()
    if not batch_dir:
        print("✗ 未找到任何批次数据")
        return 1
    
    print(f"\n使用批次: {batch_dir.name}")
    print("-" * 80)
    
    # 实验配置
    experiments = [
        {"name": "action_only", "name_en": "Action Only", "color": "#1f77b4"},
        {"name": "apf_traditional", "name_en": "APF Traditional", "color": "#ff7f0e"},
        {"name": "apf_learnable", "name_en": "APF Learnable", "color": "#2ca02c"},
        {"name": "action_apf_fusion", "name_en": "Action+APF Fusion", "color": "#d62728"}
    ]
    
    # 加载所有实验数据
    series = []
    for exp in experiments:
        print(f"\n加载实验: {exp['name_en']}")
        try:
            metrics = manager.load_all_experiment_metrics(exp['name'], batch_dir.name)
            if metrics:
                series.append({
                    "name": exp['name'],
                    "name_en": exp['name_en'],
                    "color": exp['color'],
                    "metrics": metrics
                })
                print(f"  ✓ 数据加载成功")
                print(f"    - Episodes: {len(metrics.get('episode_rewards', []))}")
                print(f"    - Loss entries: {len(metrics.get('loss_history', []))}")
            else:
                print(f"  ✗ 数据加载失败")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    if not series:
        print("\n✗ 未找到任何有效实验数据")
        return 1
    
    print(f"\n成功加载 {len(series)} 个实验")
    print("-" * 80)
    
    # 创建plots目录
    plots_dir = batch_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # 生成4张对比图
    # ========================================================================
    print("\n生成对比图:")
    print("-" * 80)
    
    plot_1_reward_comparison(
        series, 
        plots_dir / f"comparison_1_rewards_{timestamp}.png"
    )
    
    plot_2_loss_comparison(
        series, 
        plots_dir / f"comparison_2_losses_{timestamp}.png"
    )
    
    plot_3_success_collision_clearance(
        series, 
        plots_dir / f"comparison_3_success_collision_clearance_{timestamp}.png"
    )
    
    plot_4_detailed_safety_analysis(
        series, 
        plots_dir / f"comparison_4_detailed_safety_{timestamp}.png"
    )
    
    # ========================================================================
    # 为每个实验生成单独详细分析图
    # ========================================================================
    print("\n生成单独实验分析图:")
    print("-" * 80)
    
    for item in series:
        exp_name = item['name_en']
        metrics = item['metrics']
        output_path = plots_dir / f"individual_{item['name']}_{timestamp}.png"
        plot_individual_experiment(exp_name, metrics, output_path)
    
    # ========================================================================
    # 生成摘要报告
    # ========================================================================
    print("\n生成摘要报告:")
    print("-" * 80)
    
    summary = {
        "batch_name": batch_dir.name,
        "generated_at": timestamp,
        "experiments": []
    }
    
    for item in series:
        metrics = item['metrics']
        rewards = metrics.get("episode_rewards", [])
        success_flags = metrics.get("success_flags", [])
        collision_counts = metrics.get("collision_counts", [])
        
        exp_summary = {
            "name": item['name_en'],
            "total_episodes": len(rewards),
            "final_reward": float(rewards[-1]) if rewards else 0.0,
            "avg_reward_last_100": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else 0.0,
            "success_rate_last_100": float(np.mean(success_flags[-100:])) if len(success_flags) >= 100 else 0.0,
            "avg_collisions_last_100": float(np.mean(collision_counts[-100:])) if len(collision_counts) >= 100 else 0.0
        }
        summary["experiments"].append(exp_summary)
    
    summary_path = plots_dir / f"summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 摘要报告: {summary_path.name}")
    
    print("\n" + "=" * 80)
    print("✓ 所有图表生成完成!")
    print(f"输出目录: {plots_dir}")
    print("=" * 80)
    
    # 打印摘要
    print("\n实验摘要:")
    print("-" * 80)
    for exp in summary["experiments"]:
        print(f"\n{exp['name']}:")
        print(f"  Episodes: {exp['total_episodes']}")
        print(f"  Final Reward: {exp['final_reward']:.2f}")
        print(f"  Avg Reward (last 100): {exp['avg_reward_last_100']:.2f}")
        print(f"  Success Rate (last 100): {exp['success_rate_last_100']:.2%}")
        print(f"  Avg Collisions (last 100): {exp['avg_collisions_last_100']:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

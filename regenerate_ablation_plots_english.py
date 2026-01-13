#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速重新生成消融实验图表（英文标签版本）
直接读取已保存的数据（episode_rewards.json, loss_history.json），重新生成图片
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
    
    # 🔧 关键：设置英文字体，避免中文显示为方框
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ Matplotlib已配置英文字体")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装依赖: pip install matplotlib scipy numpy")
    sys.exit(1)


def smooth_curve(data: np.ndarray, method: str = "moving_average", window: int = 10) -> np.ndarray:
    """平滑曲线以减少振幅"""
    if len(data) < 2:
        return data
    
    if method == "moving_average":
        # 移动平均
        return uniform_filter1d(data.astype(float), size=window, mode='nearest')
    elif method == "poly":
        # 多项式拟合
        if len(data) < 3:
            return data
        x = np.arange(len(data))
        degree = min(5, len(data) - 1)
        coeffs = np.polyfit(x, data, degree)
        poly = np.poly1d(coeffs)
        return poly(x)
    else:
        return data


def load_experiment_data(log_dir):
    """加载实验数据（从episode_rewards.json和loss_history.json）"""
    log_path = Path(log_dir)
    
    print(f"    查找数据文件: {log_dir}")
    
    # 🔧 修复：查找episode_rewards.json和loss_history.json
    # 数据可能在以下位置：
    # 1. log_dir/episode_rewards.json
    # 2. log_dir/timestamp/episode_rewards.json
    
    episode_rewards_file = None
    loss_history_file = None
    
    # 先尝试直接路径
    if (log_path / "episode_rewards.json").exists():
        episode_rewards_file = log_path / "episode_rewards.json"
    if (log_path / "loss_history.json").exists():
        loss_history_file = log_path / "loss_history.json"
    
    # 如果没找到，查找子目录
    if not episode_rewards_file or not loss_history_file:
        subdirs = sorted([d for d in log_path.iterdir() if d.is_dir() and d.name != 'evaluation'],
                        key=lambda x: x.stat().st_mtime, reverse=True)
        for subdir in subdirs:
            if not episode_rewards_file and (subdir / "episode_rewards.json").exists():
                episode_rewards_file = subdir / "episode_rewards.json"
            if not loss_history_file and (subdir / "loss_history.json").exists():
                loss_history_file = subdir / "loss_history.json"
            if episode_rewards_file and loss_history_file:
                break
    
    # 加载数据
    metrics = {
        "episode_rewards": [],
        "success_flags": [],
        "collision_counts": [],
        "min_distances_to_obstacle": [],
        "agent_success_flags": [],
        "team_success_flags": [],
        "agent_success_rates": [],
        "team_success_rate": 0.0,
        "loss_history": []
    }
    
    if episode_rewards_file:
        print(f"      ✓ episode_rewards: {episode_rewards_file.relative_to(log_path.parent)}")
        with open(episode_rewards_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                metrics["episode_rewards"] = data.get("episode_rewards", [])
                metrics["success_flags"] = data.get("success_flags", [])
                metrics["collision_counts"] = data.get("collision_counts", [])
                metrics["min_distances_to_obstacle"] = data.get("min_distances_to_obstacle", [])
                metrics["agent_success_flags"] = data.get("agent_success_flags", [])
                metrics["team_success_flags"] = data.get("team_success_flags", [])
                metrics["agent_success_rates"] = data.get("agent_success_rates", [])
                metrics["team_success_rate"] = data.get("team_success_rate", 0.0)
            else:
                metrics["episode_rewards"] = data
    else:
        print(f"      ✗ 未找到episode_rewards.json")
    
    if loss_history_file:
        print(f"      ✓ loss_history: {loss_history_file.relative_to(log_path.parent)}")
        with open(loss_history_file, 'r', encoding='utf-8') as f:
            metrics["loss_history"] = json.load(f)
    else:
        print(f"      ✗ 未找到loss_history.json")
    
    return metrics


def plot_comparison_rewards(series, output_path, smooth_window=10, fit_method="moving_average"):
    """绘制对比奖励曲线（英文标签）"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']
    has_data = False
    
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        episodes = range(1, len(rewards) + 1)
        rewards_array = np.array(rewards)
        
        color = colors[idx % len(colors)]
        name_en = item.get('name_en', item.get('name', 'Unknown'))
        
        # 原始曲线（半透明）
        ax.plot(episodes, rewards, 
                label=f"{name_en} (Raw)", 
                color=color, 
                alpha=0.3, 
                linewidth=1,
                linestyle='-')
        
        # 拟合曲线（实线，粗线）
        smoothed = smooth_curve(rewards_array, method=fit_method, window=smooth_window)
        ax.plot(episodes, smoothed, 
                label=f"{name_en} (Fitted)", 
                color=color, 
                alpha=0.9, 
                linewidth=2.5,
                linestyle='-')
    
    if has_data:
        ax.set_title(f"Action vs APF Correction Ablation Comparison\n(Fit Method: {fit_method}, Window: {smooth_window})", 
                     fontsize=16, fontweight='bold', pad=20, fontfamily='DejaVu Sans')
        ax.set_xlabel("Episode", fontsize=14, fontfamily='DejaVu Sans')
        ax.set_ylabel("Reward", fontsize=14, fontfamily='DejaVu Sans')
        legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.9, prop={'family': 'DejaVu Sans'})
        for text in legend.get_texts():
            text.set_fontfamily('DejaVu Sans')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ 奖励对比图: {output_path.name}")
    else:
        print(f"  ✗ 没有可用的奖励数据")
    plt.close(fig)


def plot_comparison_success_collision_clearance(series, output_path, smooth_window=10, fit_method="moving_average"):
    """绘制成功率、碰撞次数、平均净空对比图（英文标签）"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    has_data = False
    
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        name_en = item.get('name_en', item.get('name', 'Unknown'))
        color = colors[idx % len(colors)]
        
        # 1. 成功率
        success_flags = metrics.get("success_flags", [])
        if success_flags:
            has_data = True
            episodes = range(1, len(success_flags) + 1)
            success_array = np.array(success_flags, dtype=float)
            
            # 滑动窗口成功率
            window_size = 50
            success_rate = []
            for i in range(len(success_array)):
                start_idx = max(0, i - window_size + 1)
                window_data = success_array[start_idx:i+1]
                rate = np.mean(window_data) if len(window_data) > 0 else 0.0
                success_rate.append(rate)
            
            axes[0].plot(episodes, success_rate, 
                       label=name_en, 
                       color=color, 
                       linewidth=2.5, 
                       alpha=0.9)
        
        # 2. 碰撞次数
        collision_counts = metrics.get("collision_counts", [])
        if collision_counts:
            has_data = True
            episodes = range(1, len(collision_counts) + 1)
            collisions_array = np.array(collision_counts, dtype=float)
            smoothed = smooth_curve(collisions_array, method=fit_method, window=smooth_window)
            axes[1].plot(episodes, smoothed, 
                       label=name_en, 
                       color=color, 
                       linewidth=2.5, 
                       alpha=0.9)
        
        # 3. 平均净空
        min_distances = metrics.get("min_distances_to_obstacle", [])
        if min_distances:
            has_data = True
            episodes = range(1, len(min_distances) + 1)
            
            if isinstance(min_distances[0], dict):
                min_dist_values = []
                valid_episodes = []
                for ep_idx, d in enumerate(min_distances):
                    mean_val = d.get('mean', None) if isinstance(d, dict) else None
                    if mean_val is not None and np.isfinite(mean_val):
                        min_dist_values.append(float(mean_val))
                        valid_episodes.append(ep_idx + 1)
            else:
                min_dist_values = [float(d) if d is not None and np.isfinite(d) else None 
                                  for d in min_distances]
                valid_episodes = [ep_idx + 1 for ep_idx, d in enumerate(min_distances) 
                                if d is not None and np.isfinite(d)]
                min_dist_values = [d for d in min_dist_values if d is not None]
            
            if min_dist_values:
                min_distances_array = np.array(min_dist_values, dtype=float)
                valid_mask = np.isfinite(min_distances_array) & (min_distances_array > -1000)
                if np.any(valid_mask):
                    smoothed = smooth_curve(min_distances_array, method=fit_method, window=smooth_window)
                    plot_episodes = valid_episodes if valid_episodes else episodes[:len(smoothed)]
                    axes[2].plot(plot_episodes, smoothed, 
                               label=name_en, 
                               color=color, 
                               linewidth=2.5, 
                               alpha=0.9)
    
    if has_data:
        axes[0].set_title("Success Rate (Moving Average, Window=50)", 
                         fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[0].set_ylabel("Success Rate", fontsize=12, fontfamily='DejaVu Sans')
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3, linestyle='--')
        legend0 = axes[0].legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
        if legend0:
            for text in legend0.get_texts():
                text.set_fontfamily('DejaVu Sans')
        
        axes[1].set_title("Collision Counts (Smoothed)", 
                         fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[1].set_ylabel("Collision Count", fontsize=12, fontfamily='DejaVu Sans')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        legend1 = axes[1].legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
        if legend1:
            for text in legend1.get_texts():
                text.set_fontfamily('DejaVu Sans')
        
        axes[2].set_title("Average Clearance (Average Distance to Obstacle, Smoothed)", 
                         fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[2].set_xlabel("Episode", fontsize=12, fontfamily='DejaVu Sans')
        axes[2].set_ylabel("Average Distance (m)", fontsize=12, fontfamily='DejaVu Sans')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        legend2 = axes[2].legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
        if legend2:
            for text in legend2.get_texts():
                text.set_fontfamily('DejaVu Sans')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ 成功率/碰撞/净空对比图: {output_path.name}")
    else:
        print(f"  ✗ 没有可用的数据")
    plt.close(fig)


def plot_comparison_losses(series, output_path):
    """绘制对比损失曲线（英文标签）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    has_data = False
    
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']
    
    for idx, item in enumerate(series):
        history = item["metrics"].get("loss_history", [])
        if not history:
            continue
        has_data = True
        steps = [entry.get("step", idx) for idx, entry in enumerate(history)]
        critic = [entry.get("critic_loss", 0) for entry in history]
        actor = [entry.get("actor_loss", 0) for entry in history]
        color = colors[idx % len(colors)]
        name_en = item.get('name_en', item.get('name', 'Unknown'))
        
        axes[0].plot(steps, critic, label=f"{name_en} (Critic)", 
                    color=color, linewidth=2.5, alpha=0.9)
        axes[1].plot(steps, actor, label=f"{name_en} (Actor)", 
                    color=color, linewidth=2.5, alpha=0.9)
    
    if has_data:
        axes[0].set_title("Critic Loss", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[0].set_ylabel("Loss", fontsize=12, fontfamily='DejaVu Sans')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        legend0 = axes[0].legend(loc='upper right', fontsize=11, prop={'family': 'DejaVu Sans'})
        for text in legend0.get_texts():
            text.set_fontfamily('DejaVu Sans')
        axes[0].set_facecolor('#fafafa')
        
        axes[1].set_title("Actor Loss", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[1].set_xlabel("Update Step", fontsize=12, fontfamily='DejaVu Sans')
        axes[1].set_ylabel("Loss", fontsize=12, fontfamily='DejaVu Sans')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        legend1 = axes[1].legend(loc='upper right', fontsize=11, prop={'family': 'DejaVu Sans'})
        for text in legend1.get_texts():
            text.set_fontfamily('DejaVu Sans')
        axes[1].set_facecolor('#fafafa')
        
        fig.suptitle("Action vs APF Correction Ablation Comparison - Loss Curves", 
                     fontsize=16, fontweight='bold', y=0.995, fontfamily='DejaVu Sans')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Loss对比图: {output_path.name}")
    else:
        print(f"  ✗ 没有可用的loss数据")
    plt.close(fig)


def plot_comparison_success_rate_and_clearance(series, output_path, smooth_window=10, fit_method="moving_average"):
    """绘制成功率和最小安全距离分布对比图（英文标签）"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    has_data = False
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']
    collision_threshold = 1.5
    
    # 🔧 关键修复：在循环外创建所有子图，避免每次循环覆盖
    ax1 = fig.add_subplot(gs[0, 0])  # 单智能体成功率
    ax2 = fig.add_subplot(gs[0, 1])  # 团队成功率
    ax3 = fig.add_subplot(gs[1, :])  # 最小安全距离时间序列
    ax4 = fig.add_subplot(gs[2, 0])  # 最小安全距离直方图
    ax5 = fig.add_subplot(gs[2, 1])  # CDF曲线
    ax6 = fig.add_subplot(gs[3, :])  # 分位数和违反概率
    ax6_twin = ax6.twinx()  # 为违反概率创建右侧y轴
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        name_en = item.get('name_en', item.get('name', 'Unknown'))
        color = colors[idx % len(colors)]
        
        # 1. 单智能体成功率
        agent_success_flags = metrics.get("agent_success_flags", [])
        if agent_success_flags and len(agent_success_flags) > 0:
            has_data = True
            max_agents = max(len(flags) for flags in agent_success_flags if flags) if agent_success_flags else 0
            if max_agents > 0:
                for agent_idx in range(max_agents):
                    success_rates = []
                    window_size = 50
                    for ep_idx in range(len(agent_success_flags)):
                        start_idx = max(0, ep_idx - window_size + 1)
                        window_flags = [flags[agent_idx] for flags in agent_success_flags[start_idx:ep_idx+1] 
                                       if len(flags) > agent_idx]
                        rate = np.mean(window_flags) if window_flags else 0.0
                        success_rates.append(rate)
                    
                    episodes = range(1, len(success_rates) + 1)
                    ax1.plot(episodes, success_rates, 
                           label=f"{name_en} Agent {agent_idx+1}", 
                           color=color, 
                           linewidth=2.0, 
                           alpha=0.8)
        
        # 2. 团队成功率
        team_success_flags = metrics.get("team_success_flags", [])
        if team_success_flags:
            has_data = True
            episodes = range(1, len(team_success_flags) + 1)
            team_success_array = np.array(team_success_flags, dtype=float)
            
            window_size = 50
            success_rate = []
            for i in range(len(team_success_array)):
                start_idx = max(0, i - window_size + 1)
                window_data = team_success_array[start_idx:i+1]
                rate = np.mean(window_data) if len(window_data) > 0 else 0.0
                success_rate.append(rate)
            
            ax2.plot(episodes, success_rate, 
                   label=name_en, 
                   color=color, 
                   linewidth=2.5, 
                   alpha=0.9)
        
        # 3. 最小安全距离分布
        min_distances = metrics.get("min_distances_to_obstacle", [])
        if min_distances:
            has_data = True
            all_min_clearances = []
            for d in min_distances:
                if isinstance(d, dict):
                    min_val = d.get('min', None)
                    if min_val is not None and np.isfinite(min_val):
                        all_min_clearances.append(float(min_val))
                elif d is not None and np.isfinite(d):
                    all_min_clearances.append(float(d))
            
            if all_min_clearances:
                all_min_clearances = np.array(all_min_clearances)
                
                # 3.1 时间序列
                episodes = range(1, len(min_distances) + 1)
                min_values = []
                valid_episodes = []
                for ep_idx, d in enumerate(min_distances):
                    if isinstance(d, dict):
                        min_val = d.get('min', None)
                    else:
                        min_val = d if d is not None and np.isfinite(d) else None
                    if min_val is not None and np.isfinite(min_val):
                        min_values.append(float(min_val))
                        valid_episodes.append(ep_idx + 1)
                
                if min_values:
                    min_values_array = np.array(min_values, dtype=float)
                    smoothed = smooth_curve(min_values_array, method=fit_method, window=smooth_window)
                    ax3.plot(valid_episodes, smoothed, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
                
                # 3.2 直方图
                valid_clearances = all_min_clearances[np.isfinite(all_min_clearances) & (all_min_clearances > -1000) & (all_min_clearances < 1000)]
                if len(valid_clearances) > 0:
                    ax4.hist(valid_clearances, bins=50, alpha=0.6, color=color, label=name_en, edgecolor='black', linewidth=0.5)
                    mean_val = np.mean(valid_clearances)
                    median_val = np.median(valid_clearances)
                    ax4.axvline(x=mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'{name_en} Mean')
                    ax4.axvline(x=median_val, color=color, linestyle=':', linewidth=2, alpha=0.8, label=f'{name_en} Median')
                
                # 3.3 CDF曲线
                if len(valid_clearances) > 0:
                    sorted_clearances = np.sort(valid_clearances)
                    cdf_values = np.arange(1, len(sorted_clearances) + 1) / len(sorted_clearances)
                    
                    ax5.plot(sorted_clearances, cdf_values, 
                           color=color, linewidth=2.5, alpha=0.9, 
                           label=f'{name_en} CDF')
                
                # 3.4 分位数和违反概率
                if len(valid_clearances) > 0:
                    quantiles = [5, 10, 25, 50, 75, 90, 95]
                    quantile_values = [np.percentile(valid_clearances, q) for q in quantiles]
                    
                    violation_probs = []
                    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
                    for threshold in thresholds:
                        prob = np.mean(valid_clearances < threshold)
                        violation_probs.append(prob)
                    
                    # 简化：每个实验叠加绘制，不需要复杂的位置调整
                    ax6.bar([f'Q{q}%' for q in quantiles], quantile_values, 
                           alpha=0.4, color=color, label=f'{name_en} Quantiles', 
                           edgecolor='black', linewidth=0.5)
                    
                    # 在twin轴上绘制violation prob曲线
                    ax6_twin.plot(thresholds, violation_probs, 
                                 color=color, marker='o', linewidth=2.5, markersize=6, 
                                 label=f'{name_en} Violation Prob', alpha=0.9)
    
    # 🔧 修复完成：在循环外设置所有子图的格式
    # 设置ax1（单智能体成功率）格式
    ax1.set_title("Single Agent Success Rate (SR_i)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax1.set_ylabel("Success Rate", fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--')
    legend1 = ax1.legend(loc='upper right', fontsize=9, prop={'family': 'DejaVu Sans'})
    if legend1:
        for text in legend1.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    # 设置ax2（团队成功率）格式
    ax2.set_title("Team Success Rate (SR_team)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax2.set_ylabel("Success Rate", fontsize=12, fontfamily='DejaVu Sans')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, linestyle='--')
    legend2 = ax2.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    if legend2:
        for text in legend2.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    # 设置ax3（最小安全距离时间序列）格式
    ax3.axhline(y=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision Threshold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_title("Episode-Level Minimum Clearance (d_min^{i,(k)})", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax3.set_xlabel("Episode", fontsize=12, fontfamily='DejaVu Sans')
    ax3.set_ylabel("Minimum Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax3.grid(True, alpha=0.3, linestyle='--')
    legend3 = ax3.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    if legend3:
        for text in legend3.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    # 设置ax4（直方图）格式
    ax4.axvline(x=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_title("Minimum Clearance Distribution", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax4.set_xlabel("Minimum Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax4.set_ylabel("Frequency", fontsize=12, fontfamily='DejaVu Sans')
    ax4.grid(True, alpha=0.3, linestyle='--')
    legend4 = ax4.legend(loc='upper right', fontsize=9, prop={'family': 'DejaVu Sans'})
    if legend4:
        for text in legend4.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    # 设置ax5（CDF曲线）格式
    ax5.axvline(x=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision Threshold')
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax5.set_title("CDF: Pr(D_min ≤ δ)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax5.set_xlabel("Minimum Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax5.set_ylabel("Cumulative Probability", fontsize=12, fontfamily='DejaVu Sans')
    ax5.grid(True, alpha=0.3, linestyle='--')
    legend5 = ax5.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    if legend5:
        for text in legend5.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    # 设置ax6（分位数和违反概率）格式
    ax6.set_title("Quantiles (Q5%, Q50%, Q95%) & Violation Probability P_viol(δ)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax6.set_xlabel("Quantile / Threshold (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax6.set_ylabel("Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax6.grid(True, alpha=0.3, linestyle='--')
    legend6 = ax6.legend(loc='upper left', fontsize=9, prop={'family': 'DejaVu Sans'})
    if legend6:
        for text in legend6.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    # 设置ax6_twin（违反概率右侧y轴）格式
    ax6_twin.set_ylabel("Violation Probability", fontsize=12, fontfamily='DejaVu Sans')
    legend6_twin = ax6_twin.legend(loc='upper right', fontsize=9, prop={'family': 'DejaVu Sans'})
    if legend6_twin:
        for text in legend6_twin.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    if has_data:
        plt.suptitle("Action vs APF Correction Ablation Comparison", 
                     fontsize=16, fontweight='bold', fontfamily='DejaVu Sans', y=0.995)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ 成功率与最小安全距离对比图: {output_path.name}")
    else:
        print(f"  ✗ 没有可用的数据")
    plt.close(fig)


def main():
    print("="*70)
    print("快速重新生成消融实验图表（英文版本，基于已保存数据）")
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
        {"label": "action_apf_fusion", "name_en": "Action+APF Fusion"},
        {"label": "action_only", "name_en": "Action Only"}
    ]
    
    series = []
    
    print("\n加载实验数据...")
    for exp in experiments:
        label = exp["label"]
        name_en = exp["name_en"]
        
        # 查找最新的日志目录
        matching_dirs = [d for d in logs_root.iterdir() 
                        if d.is_dir() and d.name.startswith(label + "_")]
        
        if not matching_dirs:
            print(f"  ✗ {label}: 未找到数据目录")
            continue
        
        # 获取最新目录
        latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
        print(f"  ✓ {label}: {latest_dir.name}")
        
        # 加载数据
        metrics = load_experiment_data(latest_dir)
        
        if not metrics.get("episode_rewards"):
            print(f"      警告: 无有效的episode_rewards数据")
            continue
        
        series.append({
            "label": label,
            "name": name_en,
            "name_en": name_en,
            "metrics": metrics,
            "log_dir": str(latest_dir)
        })
    
    if not series:
        print("\n错误：未找到任何有效数据")
        return 1
    
    print(f"\n成功加载 {len(series)} 个实验的数据")
    print(f"  回合数: {[len(s['metrics'].get('episode_rewards', [])) for s in series]}")
    
    # 生成图表
    print("\n生成图表...")
    
    # 1. 奖励对比图
    print("  [1/4] 奖励对比图...")
    reward_path = output_dir / f"reward_comparison_english_{timestamp}.png"
    plot_comparison_rewards(series, reward_path, smooth_window=10, fit_method="moving_average")
    
    # 2. 成功率/碰撞/净空对比图
    print("  [2/4] 成功率/碰撞/净空对比图...")
    scc_path = output_dir / f"success_collision_clearance_comparison_english_{timestamp}.png"
    plot_comparison_success_collision_clearance(series, scc_path, smooth_window=10, fit_method="moving_average")
    
    # 3. 成功率与最小安全距离对比图
    print("  [3/4] 成功率与最小安全距离对比图...")
    src_path = output_dir / f"success_rate_and_clearance_comparison_english_{timestamp}.png"
    plot_comparison_success_rate_and_clearance(series, src_path, smooth_window=10, fit_method="moving_average")
    
    # 4. 损失对比图
    print("  [4/4] 损失对比图...")
    loss_path = output_dir / f"loss_comparison_english_{timestamp}.png"
    plot_comparison_losses(series, loss_path)
    
    # 生成摘要
    summary = {
        "timestamp": timestamp,
        "experiments": [
            {
                "label": item["label"],
                "name": item["name_en"],
                "log_dir": item["log_dir"],
                "episodes": len(item["metrics"].get("episode_rewards", [])),
                "final_reward": item["metrics"]["episode_rewards"][-1] if item["metrics"].get("episode_rewards") else None,
                "avg_reward": float(np.mean(item["metrics"]["episode_rewards"])) if item["metrics"].get("episode_rewards") else None,
                "team_success_rate": item["metrics"].get("team_success_rate", 0.0)
            }
            for item in series
        ]
    }
    
    summary_path = output_dir / f"summary_english_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✓ 所有图表生成完成（英文标签）")
    print("="*70)
    print(f"\n输出目录: {output_dir}/")
    print(f"  - reward_comparison_english_{timestamp}.png")
    print(f"  - success_collision_clearance_comparison_english_{timestamp}.png")
    print(f"  - success_rate_and_clearance_comparison_english_{timestamp}.png")
    print(f"  - loss_comparison_english_{timestamp}.png")
    print(f"  - summary_english_{timestamp}.json")
    print("\n实验结果汇总:")
    for exp in summary["experiments"]:
        print(f"  - {exp['name']}:")
        print(f"      Episodes: {exp['episodes']}")
        print(f"      Final Reward: {exp['final_reward']:.2f}" if exp['final_reward'] else "      Final Reward: N/A")
        print(f"      Avg Reward: {exp['avg_reward']:.2f}" if exp['avg_reward'] else "      Avg Reward: N/A")
        print(f"      Team Success Rate: {exp['team_success_rate']:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成Dual Q消融实验的对比图表
修复颜色重复和Loss图问题
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# 设置字体
def setup_fonts():
    """设置matplotlib字体"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

setup_fonts()

# 🎨 定义6种高对比度的不同颜色
COLORS_6 = [
    '#0066CC',  # 深蓝 - MADDPG Baseline
    '#9900CC',  # 深紫 - MATD3 Separated Gradient  
    '#00AA00',  # 深绿 - MADDPG Dual Q
    '#CC0000',  # 深红 - MADDPG Separated Gradient
    '#FF6600',  # 橙色 - MATD3 Single Q (新增)
    '#00CCCC',  # 青色 - MATD3 Dual Q (新增)
]

def smooth_curve(data, method='moving_average', window=10):
    """平滑曲线"""
    if method == 'moving_average' and len(data) >= window:
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='valid')
    elif method == 'spline' and len(data) > 3:
        from scipy.interpolate import UnivariateSpline
        x = np.arange(len(data))
        spl = UnivariateSpline(x, data, s=len(data)*10)
        return spl(x)
    elif method == 'poly' and len(data) > 1:
        x = np.arange(len(data))
        degree = min(5, len(data) - 1)
        coeffs = np.polyfit(x, data, degree)
        poly = np.poly1d(coeffs)
        return poly(x)
    else:
        return data


def plot_rewards(series: List[Dict], title: str, output_path: Path, 
                smooth_window: int = 10, fit_method: str = "moving_average"):
    """绘制奖励对比图（6种颜色）"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    has_data = False
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        episodes = range(1, len(rewards) + 1)
        rewards_array = np.array(rewards)
        
        color = COLORS_6[idx % len(COLORS_6)]
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        
        # 原始曲线（半透明）
        ax.plot(episodes, rewards, 
                label=f"{name_en} (Raw)", 
                color=color, 
                alpha=0.3, 
                linewidth=1,
                linestyle='-')
        
        # 拟合曲线（粗线）
        smoothed = smooth_curve(rewards_array, method=fit_method, window=smooth_window)
        if fit_method == 'moving_average' and len(smoothed) < len(rewards):
            # 调整x轴以匹配平滑后的长度
            offset = (len(rewards) - len(smoothed)) // 2
            smooth_episodes = range(1 + offset, 1 + offset + len(smoothed))
            ax.plot(smooth_episodes, smoothed, 
                    label=f"{name_en} (Fitted)", 
                    color=color, 
                    alpha=0.9, 
                    linewidth=2.5,
                    linestyle='-')
        else:
            ax.plot(episodes, smoothed, 
                    label=f"{name_en} (Fitted)", 
                    color=color, 
                    alpha=0.9, 
                    linewidth=2.5,
                    linestyle='-')
    
    if has_data:
        ax.set_title(f"{title}\n(Fit Method: {fit_method}, Window: {smooth_window})", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Reward", fontsize=14)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✅ 奖励对比图已生成: {output_path}")
    else:
        print(f"⚠️  没有奖励数据: {output_path}")
    plt.close(fig)


def plot_losses(series: List[Dict], title: str, output_path: Path):
    """绘制Loss对比图（6种颜色）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    has_critic_data = False
    has_actor_data = False
    
    for idx, item in enumerate(series):
        history = item["metrics"].get("loss_history", [])
        if not history:
            print(f"⚠️  {item.get('name_en', 'Unknown')} 没有loss_history数据")
            continue
        
        steps = [entry.get("step", i) for i, entry in enumerate(history)]
        critic = [entry.get("critic_loss", 0) for entry in history]
        actor = [entry.get("actor_loss", 0) for entry in history]
        
        # 过滤掉None和NaN值
        valid_critic = [(s, c) for s, c in zip(steps, critic) 
                       if c is not None and not (isinstance(c, float) and np.isnan(c)) and c != 0]
        valid_actor = [(s, a) for s, a in zip(steps, actor) 
                      if a is not None and not (isinstance(a, float) and np.isnan(a)) and abs(a) < 1000]
        
        color = COLORS_6[idx % len(COLORS_6)]
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        
        if valid_critic:
            has_critic_data = True
            steps_c, values_c = zip(*valid_critic)
            axes[0].plot(steps_c, values_c, 
                        label=f"{name_en} (Critic)", 
                        color=color, 
                        linewidth=2.5, 
                        alpha=0.9, 
                        linestyle='-')
        
        if valid_actor:
            has_actor_data = True
            steps_a, values_a = zip(*valid_actor)
            axes[1].plot(steps_a, values_a, 
                        label=f"{name_en} (Actor)", 
                        color=color, 
                        linewidth=2.5, 
                        alpha=0.9, 
                        linestyle='-')
    
    if has_critic_data or has_actor_data:
        axes[0].set_title("Critic Loss", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper right', fontsize=10, ncol=2)
        axes[0].set_facecolor('#fafafa')
        
        axes[1].set_title("Actor Loss", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Update Step", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper right', fontsize=10, ncol=2)
        axes[1].set_facecolor('#fafafa')
        
        fig.suptitle(f"{title} - Loss Curves", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✅ Loss对比图已生成: {output_path}")
    else:
        print(f"⚠️  没有Loss数据: {output_path}")
    plt.close(fig)


def plot_success_collision(series: List[Dict], title: str, output_path: Path,
                          smooth_window: int = 50):
    """绘制成功率、碰撞和净空对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        success_rates = metrics.get("episode_success_rates", [])
        collision_counts = metrics.get("episode_collision_counts", [])
        clearances = metrics.get("episode_min_clearances", [])
        
        color = COLORS_6[idx % len(COLORS_6)]
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        
        if success_rates:
            episodes = range(1, len(success_rates) + 1)
            # 移动平均平滑
            if len(success_rates) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(success_rates, kernel, mode='valid')
                offset = (len(success_rates) - len(smoothed)) // 2
                smooth_episodes = range(1 + offset, 1 + offset + len(smoothed))
                axes[0].plot(smooth_episodes, smoothed, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
            else:
                axes[0].plot(episodes, success_rates, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
        
        if collision_counts:
            episodes = range(1, len(collision_counts) + 1)
            if len(collision_counts) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(collision_counts, kernel, mode='valid')
                offset = (len(collision_counts) - len(smoothed)) // 2
                smooth_episodes = range(1 + offset, 1 + offset + len(smoothed))
                axes[1].plot(smooth_episodes, smoothed, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
            else:
                axes[1].plot(episodes, collision_counts, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
        
        if clearances:
            episodes = range(1, len(clearances) + 1)
            if len(clearances) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(clearances, kernel, mode='valid')
                offset = (len(clearances) - len(smoothed)) // 2
                smooth_episodes = range(1 + offset, 1 + offset + len(smoothed))
                axes[2].plot(smooth_episodes, smoothed, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
            else:
                axes[2].plot(episodes, clearances, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9)
    
    axes[0].set_title(f"Success Rate (Moving Average, Window={smooth_window})", 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Success Rate", fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    axes[1].set_title("Collision Counts (Smoothed)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Collision Count", fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10, ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title("Average Clearance (Average Distance to Obstacle, Smoothed)", 
                     fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Episode", fontsize=12)
    axes[2].set_ylabel("Average Distance (m)", fontsize=12)
    axes[2].legend(loc='upper right', fontsize=10, ncol=2)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ 成功率/碰撞/净空对比图已生成: {output_path}")
    plt.close(fig)


def load_batch_results(batch_dir: Path):
    """加载批次结果"""
    series = []
    
    # 读取summary文件获取实验信息
    summary_file = batch_dir / 'plots' / 'latest_summary.json'
    if not summary_file.exists():
        print(f"❌ 未找到summary文件: {summary_file}")
        return []
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    experiments = summary.get('experiments', [])
    
    for exp in experiments:
        label = exp.get('label')
        name_en = exp.get('name_en')
        log_dir = exp.get('log_dir')
        
        if not log_dir:
            print(f"⚠️  实验 {label} 没有log_dir")
            continue
        
        # 构建结果文件路径（相对于工作目录）
        work_dir = Path.cwd()
        log_path = work_dir / log_dir
        
        # 加载3个独立的JSON文件
        results_file = log_path / 'results.json'
        rewards_file = log_path / 'episode_rewards.json'
        loss_file = log_path / 'loss_history.json'
        
        if not results_file.exists():
            print(f"⚠️  未找到results.json: {results_file}")
            continue
        
        # 合并所有数据
        metrics = {}
        
        # 加载results.json
        with open(results_file, 'r') as f:
            results_data = json.load(f)
            metrics.update(results_data)
        
        # 加载episode_rewards.json（如果存在）
        if rewards_file.exists():
            with open(rewards_file, 'r') as f:
                rewards_data = json.load(f)
                metrics.update(rewards_data)
        
        # 加载loss_history.json（如果存在）
        if loss_file.exists():
            with open(loss_file, 'r') as f:
                loss_data = json.load(f)
                metrics['loss_history'] = loss_data
        
        series.append({
            'label': label,
            'name_en': name_en,
            'metrics': metrics
        })
        print(f"✅ 加载实验: {name_en}")
    
    return series


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重新生成Dual Q消融实验图表")
    parser.add_argument('--batch-dir', type=str, required=True,
                       help='批次目录路径（例如：ablation_experiments/batch_20260121_154619）')
    parser.add_argument('--smooth-window', type=int, default=10,
                       help='平滑窗口大小')
    parser.add_argument('--fit-method', type=str, default='moving_average',
                       choices=['moving_average', 'spline', 'poly'],
                       help='拟合方法')
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"❌ 批次目录不存在: {batch_dir}")
        return 1
    
    print(f"\n{'='*80}")
    print(f"重新生成Dual Q消融实验图表")
    print(f"批次目录: {batch_dir}")
    print(f"{'='*80}\n")
    
    # 加载所有实验结果
    series = load_batch_results(batch_dir)
    
    if not series:
        print(f"❌ 没有找到任何实验结果")
        return 1
    
    print(f"\n找到 {len(series)} 个实验结果\n")
    
    # 创建输出目录
    output_dir = batch_dir / 'plots_regenerated'
    output_dir.mkdir(exist_ok=True)
    
    title = "Dual Q & Separated Gradient Ablation Comparison"
    
    # 生成图表
    print("\n生成图表...")
    plot_rewards(series, title, output_dir / 'reward_comparison.png',
                smooth_window=args.smooth_window, fit_method=args.fit_method)
    
    plot_losses(series, title, output_dir / 'loss_comparison.png')
    
    plot_success_collision(series, title, output_dir / 'success_collision_clearance.png',
                          smooth_window=50)
    
    print(f"\n{'='*80}")
    print(f"✅ 所有图表已生成到: {output_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

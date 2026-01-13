#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量从多个 episode_rewards.json 文件生成奖励曲线图

用法:
    # 处理指定目录下的所有实验
    python3 plot_rewards_batch.py logs/ --pattern "action_*"
    
    # 处理所有消融实验
    python3 plot_rewards_batch.py logs/ --pattern "action_*|apf_*"
    
    # 指定平滑窗口
    python3 plot_rewards_batch.py logs/ --smooth-window 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
import re
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    HAS_MATPLOTLIB = True
except ImportError:
    print("错误: 缺少依赖，请安装: pip install matplotlib scipy")
    sys.exit(1)

# 导入单个绘图函数
sys.path.insert(0, str(Path(__file__).parent))
from plot_reward_from_json import load_episode_rewards, smooth_curve, setup_english_fonts


def find_experiment_dirs(logs_root: Path, pattern: str = None):
    """查找所有包含 episode_rewards.json 的实验目录"""
    experiment_dirs = []
    
    if not logs_root.exists():
        print(f"错误: 目录不存在: {logs_root}")
        return experiment_dirs
    
    # 搜索所有可能的日志目录结构
    # 结构1: logs/exp_name_timestamp/timestamp/episode_rewards.json
    # 结构2: logs/exp_name/episode_rewards.json
    for item in logs_root.iterdir():
        if not item.is_dir():
            continue
        
        # 检查是否有子目录（结构1）
        subdirs = sorted([d for d in item.iterdir() if d.is_dir()])
        if subdirs:
            # 使用最新的子目录
            for subdir in subdirs:
                json_path = subdir / "episode_rewards.json"
                if json_path.exists():
                    if pattern is None or re.search(pattern, item.name):
                        experiment_dirs.append((item.name, subdir))
        else:
            # 直接检查当前目录（结构2）
            json_path = item / "episode_rewards.json"
            if json_path.exists():
                if pattern is None or re.search(pattern, item.name):
                    experiment_dirs.append((item.name, item))
    
    return experiment_dirs


def plot_reward_curve(rewards: list, output_path: Path, smooth_window: int = 0, 
                     title: str = None, exp_name: str = None):
    """绘制奖励曲线图（从 plot_reward_from_json.py 复制）"""
    if not rewards:
        print(f"警告: 没有奖励数据，跳过绘图: {output_path}")
        return False
    
    setup_english_fonts()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    episodes = range(1, len(rewards) + 1)
    rewards_array = np.array(rewards)
    
    # 原始曲线（半透明，细线）
    ax.plot(episodes, rewards, 
            label="Raw Reward", 
            color='#1f77b4', 
            alpha=0.3, 
            linewidth=1,
            linestyle='-')
    
    # 如果启用平滑，绘制拟合曲线
    if smooth_window > 0:
        smoothed = smooth_curve(rewards_array, window=smooth_window)
        ax.plot(episodes, smoothed, 
                label=f"Smoothed (window={smooth_window})", 
                color='#1f77b4', 
                alpha=0.9, 
                linewidth=2.5,
                linestyle='-')
    
    # 设置标题
    if title:
        plot_title = title
    elif exp_name:
        plot_title = f"Reward Curve - {exp_name}"
    else:
        plot_title = "Training Reward Curve"
    
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    
    # 设置x轴范围
    ax.set_xlim(1, len(rewards))
    
    # 添加统计信息文本
    if len(rewards) > 0:
        final_reward = rewards[-1]
        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        
        stats_text = f"Final: {final_reward:.2f} | Avg: {avg_reward:.2f} | Max: {max_reward:.2f} | Min: {min_reward:.2f}"
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, 
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="批量从 episode_rewards.json 生成奖励曲线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有实验
  python3 plot_rewards_batch.py logs/
  
  # 只处理匹配模式的实验
  python3 plot_rewards_batch.py logs/ --pattern "action_*|apf_*"
  
  # 使用平滑
  python3 plot_rewards_batch.py logs/ --smooth-window 10
        """
    )
    parser.add_argument("logs_root", type=str,
                       help="日志根目录（通常是 logs/）")
    parser.add_argument("--pattern", type=str, default=None,
                       help="实验名称匹配模式（正则表达式，如 'action_*|apf_*'）")
    parser.add_argument("--smooth-window", type=int, default=0,
                       help="平滑窗口大小（0=不平滑，默认0）")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认: 每个实验的日志目录）")
    parser.add_argument("--suffix", type=str, default="from_json",
                       help="输出文件名后缀（默认: from_json，生成 reward_plot_from_json.png）")
    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    
    logs_root = Path(args.logs_root).resolve()
    experiment_dirs = find_experiment_dirs(logs_root, args.pattern)
    
    if not experiment_dirs:
        print(f"❌ 未找到匹配的实验目录（模式: {args.pattern or '全部'}）")
        sys.exit(1)
    
    print(f"✅ 找到 {len(experiment_dirs)} 个实验目录")
    print(f"   平滑窗口: {args.smooth_window if args.smooth_window > 0 else '无'}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for exp_name, log_dir in experiment_dirs:
        json_path = log_dir / "episode_rewards.json"
        
        try:
            rewards, train_episodes, timestamp = load_episode_rewards(json_path)
            
            # 确定输出路径
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{exp_name}_reward_plot_{args.suffix}.png"
            else:
                output_path = log_dir / f"reward_plot_{args.suffix}.png"
            
            # 绘制曲线
            if plot_reward_curve(rewards, output_path,
                               smooth_window=args.smooth_window,
                               exp_name=exp_name):
                success_count += 1
                print(f"  ✅ {exp_name}: {len(rewards)} 回合 → {output_path}")
            else:
                failed_count += 1
                print(f"  ❌ {exp_name}: 无数据")
                
        except Exception as e:
            failed_count += 1
            print(f"  ❌ {exp_name}: 错误 - {e}")
    
    print()
    print(f"完成: 成功 {success_count} 个，失败 {failed_count} 个")


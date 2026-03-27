#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 episode_rewards.json 文件生成奖励曲线图

用法:
    python3 plot_reward_from_json.py <log_dir>
    python3 plot_reward_from_json.py logs/exp_name_timestamp/timestamp/
    python3 plot_reward_from_json.py logs/exp_name_timestamp/timestamp/ --smooth-window 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    HAS_MATPLOTLIB = True
except ImportError:
    print("错误: 缺少依赖，请安装: pip install matplotlib scipy")
    sys.exit(1)


def setup_english_fonts():
    """设置英文字体，避免中文显示问题"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False


def smooth_curve(data: np.ndarray, window: int = 10) -> np.ndarray:
    """平滑曲线以减少振幅（移动平均）"""
    if len(data) < 2:
        return data
    return uniform_filter1d(data.astype(float), size=window, mode='nearest')


def load_episode_rewards(json_path: Path):
    """从JSON文件加载奖励数据"""
    if not json_path.exists():
        raise FileNotFoundError(f"文件不存在: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        rewards = data.get("episode_rewards", [])
        train_episodes = data.get("train_episodes", len(rewards))
        timestamp = data.get("timestamp", "unknown")
    else:
        # 兼容旧格式（直接是数组）
        rewards = data
        train_episodes = len(rewards)
        timestamp = "unknown"
    
    return rewards, train_episodes, timestamp


def plot_reward_curve(rewards: list, output_path: Path, smooth_window: int = 0, 
                     title: str = None, exp_name: str = None):
    """绘制奖励曲线图"""
    if not rewards:
        print(f"警告: 没有奖励数据，跳过绘图: {output_path}")
        return
    
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
    print(f"✅ 奖励曲线图已保存: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="从 episode_rewards.json 生成奖励曲线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个日志目录
  python3 plot_reward_from_json.py logs/exp_name_timestamp/timestamp/
  
  # 使用平滑
  python3 plot_reward_from_json.py logs/exp_name_timestamp/timestamp/ --smooth-window 10
  
  # 指定输出路径和标题
  python3 plot_reward_from_json.py logs/exp_name_timestamp/timestamp/ \\
      --output custom_reward_plot.png --title "My Experiment"
        """
    )
    parser.add_argument("log_dir", type=str, 
                       help="日志目录路径（包含 episode_rewards.json 的目录）")
    parser.add_argument("--smooth-window", type=int, default=0,
                       help="平滑窗口大小（0=不平滑，默认0）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径（默认: <log_dir>/reward_plot_from_json.png）")
    parser.add_argument("--title", type=str, default=None,
                       help="图表标题（默认: 从目录名推断）")
    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    
    log_dir = Path(args.log_dir).resolve()
    if not log_dir.exists():
        print(f"错误: 目录不存在: {log_dir}")
        sys.exit(1)
    
    json_path = log_dir / "episode_rewards.json"
    
    try:
        rewards, train_episodes, timestamp = load_episode_rewards(json_path)
        print(f"✅ 成功加载奖励数据: {len(rewards)} 个回合")
        print(f"   训练回合数: {train_episodes}")
        print(f"   时间戳: {timestamp}")
        
        # 确定输出路径
        if args.output:
            output_path = Path(args.output).resolve()
        else:
            output_path = log_dir / "reward_plot_from_json.png"
        
        # 确定标题
        title = args.title
        if not title:
            # 从目录名推断实验名称
            exp_name = log_dir.parent.name if log_dir.parent.name != "logs" else log_dir.name
            title = f"Reward Curve - {exp_name}"
        
        # 绘制曲线
        plot_reward_curve(rewards, output_path, 
                         smooth_window=args.smooth_window,
                         title=title,
                         exp_name=exp_name if 'exp_name' in locals() else None)
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


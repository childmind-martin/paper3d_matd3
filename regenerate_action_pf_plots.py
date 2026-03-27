#!/usr/bin/env python3
"""
重新生成 action vs APF ablation 实验的对比图
专门用于修复 batch_20260121_154619 的空图问题

使用方法:
  python regenerate_action_pf_plots.py --batch batch_20260121_154619
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

# 导入绘图函数
sys.path.insert(0, str(Path(__file__).parent))
from ablation_action_pf_comparison import (
    plot_comparison_rewards,
    plot_comparison_losses,
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance
)


def load_metrics(log_dir: str) -> Dict:
    """加载训练指标（与 ablation_action_pf_comparison.py 保持一致）"""
    metrics = {}
    
    # 加载奖励数据
    ep_path = Path(log_dir) / "episode_rewards.json"
    if ep_path.exists():
        with open(ep_path) as f:
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
            metrics["success_flags"] = []
            metrics["collision_counts"] = []
            metrics["min_distances_to_obstacle"] = []
            metrics["agent_success_flags"] = []
            metrics["team_success_flags"] = []
            metrics["agent_success_rates"] = []
            metrics["team_success_rate"] = 0.0
    else:
        metrics["episode_rewards"] = []
        metrics["success_flags"] = []
        metrics["collision_counts"] = []
        metrics["min_distances_to_obstacle"] = []
        metrics["agent_success_flags"] = []
        metrics["team_success_flags"] = []
        metrics["agent_success_rates"] = []
        metrics["team_success_rate"] = 0.0
    
    # 加载损失数据
    loss_path = Path(log_dir) / "loss_history.json"
    if loss_path.exists():
        with open(loss_path) as f:
            loss_data = json.load(f)
        if isinstance(loss_data, dict):
            metrics["loss_history"] = loss_data.get("loss_history", [])
        elif isinstance(loss_data, list):
            metrics["loss_history"] = loss_data
        else:
            metrics["loss_history"] = []
    else:
        metrics["loss_history"] = []
    
    return metrics


def regenerate_plots(batch_dir: Path, smooth_window: int = 10, fit_method: str = "moving_average"):
    """重新生成图表"""
    
    print(f"{'='*70}")
    print(f"重新生成图表: {batch_dir.name}")
    print(f"{'='*70}\n")
    
    # 读取 latest_summary.json
    summary_path = batch_dir / "plots" / "latest_summary.json"
    if not summary_path.exists():
        print(f"❌ 错误: 未找到 summary 文件: {summary_path}")
        sys.exit(1)
    
    with open(summary_path) as f:
        summary_data = json.load(f)
    
    # 重建 series（从实际的 log_dir 加载数据）
    series = []
    workspace_root = Path("/home/tang/Desktop")
    
    for exp_info in summary_data['experiments']:
        log_dir = exp_info['log_dir']
        
        if not log_dir:
            print(f"⚠️  {exp_info['label']}: 没有 log_dir，跳过")
            continue
        
        # 加载 metrics
        full_log_dir = workspace_root / log_dir
        if not full_log_dir.exists():
            print(f"⚠️  {exp_info['label']}: log_dir 不存在 ({full_log_dir})，跳过")
            continue
        
        metrics = load_metrics(str(full_log_dir))
        
        # 检查是否有数据
        has_data = (len(metrics.get('episode_rewards', [])) > 0)
        
        if not has_data:
            print(f"⚠️  {exp_info['label']}: 没有训练数据，跳过")
            continue
        
        item = {
            "label": exp_info['label'],
            "name": exp_info['name'],
            "name_en": exp_info.get('name_en', exp_info['label']),
            "description": exp_info.get('description', ''),
            "log_dir": log_dir,
            "metrics": metrics
        }
        
        series.append(item)
        
        print(f"✓ {exp_info['label']}: {len(metrics['episode_rewards'])} episodes")
    
    if not series:
        print(f"\n❌ 错误: 没有找到任何有效的实验数据")
        sys.exit(1)
    
    print(f"\n找到 {len(series)} 个有效实验，开始生成图表...\n")
    
    # 生成图表
    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title = "Action vs APF Correction Ablation Comparison"
    timestamp = summary_data.get('timestamp', 'regenerated')
    
    # 1. Reward comparison
    print("1. 生成 Reward 对比图...")
    reward_png = output_dir / f"reward_comparison_{timestamp}_FIXED.png"
    try:
        plot_comparison_rewards(
            series, title, reward_png,
            smooth_window=smooth_window,
            fit_method=fit_method
        )
        print(f"   ✓ {reward_png.name}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 2. Loss comparison
    print("2. 生成 Loss 对比图...")
    loss_png = output_dir / f"loss_comparison_{timestamp}_FIXED.png"
    try:
        plot_comparison_losses(series, title, loss_png)
        print(f"   ✓ {loss_png.name}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 3. Success/Collision/Clearance comparison
    print("3. 生成 Success/Collision/Clearance 对比图...")
    success_collision_png = output_dir / f"success_collision_clearance_comparison_{timestamp}_FIXED.png"
    try:
        plot_comparison_success_collision_clearance(
            series, title, success_collision_png,
            smooth_window=smooth_window,
            fit_method=fit_method
        )
        print(f"   ✓ {success_collision_png.name}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 4. Success Rate and Clearance comparison (这是用户说有问题的图)
    print("4. 生成 Success Rate and Clearance 对比图...")
    success_clearance_png = output_dir / f"success_rate_and_clearance_comparison_{timestamp}_FIXED.png"
    try:
        plot_comparison_success_rate_and_clearance(
            series, title, success_clearance_png,
            smooth_window=smooth_window,
            fit_method=fit_method
        )
        print(f"   ✓ {success_clearance_png.name}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ 图表重新生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    # 显示结果摘要
    print("实验数据摘要:")
    for item in series:
        metrics = item["metrics"]
        name = item.get('name_en', item['name'])
        n_episodes = len(metrics.get('episode_rewards', []))
        n_success = len(metrics.get('team_success_flags', []))
        n_collision = len(metrics.get('collision_counts', []))
        n_clearance = len(metrics.get('min_distances_to_obstacle', []))
        
        print(f"\n  {name}:")
        print(f"    - Episodes: {n_episodes}")
        print(f"    - Team Success Flags: {n_success}")
        print(f"    - Collision Counts: {n_collision}")
        print(f"    - Min Clearance Data: {n_clearance}")


def main():
    parser = argparse.ArgumentParser(description="重新生成 action vs APF ablation 对比图")
    parser.add_argument("--batch", type=str, required=True,
                       help="批次目录名称（例如：batch_20260121_154619）")
    parser.add_argument("--smooth-window", type=int, default=10,
                       help="平滑窗口大小（默认: 10）")
    parser.add_argument("--fit-method", type=str, default="moving_average",
                       choices=["moving_average", "spline", "poly"],
                       help="拟合方法（默认: moving_average）")
    
    args = parser.parse_args()
    
    # 构建批次目录路径
    batch_dir = Path("/home/tang/Desktop/ablation_experiments") / args.batch
    
    if not batch_dir.exists():
        print(f"❌ 错误: 批次目录不存在: {batch_dir}")
        sys.exit(1)
    
    regenerate_plots(batch_dir, args.smooth_window, args.fit_method)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速重新生成完整的消融实验图表（包含所有4个实验）
直接读取已保存的数据，使用英文标签生成图表

注意：apf_learnable 使用指定的数据目录（logs/apf_learnable_20251226_234343）
      其他实验自动查找最新的数据目录
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
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ Matplotlib已配置英文字体")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装依赖: pip install matplotlib scipy numpy")
    sys.exit(1)


# 导入绘图函数
sys.path.insert(0, str(Path(__file__).parent))
from regenerate_ablation_plots_english import (
    load_experiment_data,
    plot_comparison_rewards,
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance,
    plot_comparison_losses
)


def main():
    print("="*70)
    print("重新生成完整的消融实验图表（包含所有4个实验）")
    print("="*70)
    
    output_dir = Path("ablation_action_pf_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logs_root = Path("logs")
    
    # 🔧 关键：手动指定apf_learnable的数据目录
    # 因为最新的目录可能是空的，需要使用已知的完整数据目录
    APF_LEARNABLE_DIR = "logs/apf_learnable_20251226_234343"
    
    # 实验配置（使用英文名称）
    experiments = [
        {"label": "apf_learnable", "name_en": "APF Learnable", "dir": APF_LEARNABLE_DIR},
        {"label": "apf_traditional", "name_en": "APF Traditional", "dir": None},
        {"label": "action_apf_fusion", "name_en": "Action+APF Fusion", "dir": None},
        {"label": "action_only", "name_en": "Action Only", "dir": None}
    ]
    
    series = []
    
    print("\n加载实验数据...")
    for exp in experiments:
        label = exp["label"]
        name_en = exp["name_en"]
        
        if exp["dir"]:
            # 使用指定目录
            latest_dir = Path(exp["dir"])
            if not latest_dir.exists():
                print(f"  ✗ {label}: 指定目录不存在: {exp['dir']}")
                continue
        else:
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
        metrics = load_experiment_data(str(latest_dir))
        
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
    reward_path = output_dir / f"reward_comparison_complete_{timestamp}.png"
    plot_comparison_rewards(series, reward_path, smooth_window=10, fit_method="moving_average")
    
    # 2. 成功率/碰撞/净空对比图
    print("  [2/4] 成功率/碰撞/净空对比图...")
    scc_path = output_dir / f"success_collision_clearance_comparison_complete_{timestamp}.png"
    plot_comparison_success_collision_clearance(series, scc_path, smooth_window=10, fit_method="moving_average")
    
    # 3. 成功率与最小安全距离对比图
    print("  [3/4] 成功率与最小安全距离对比图...")
    src_path = output_dir / f"success_rate_and_clearance_comparison_complete_{timestamp}.png"
    plot_comparison_success_rate_and_clearance(series, src_path, smooth_window=10, fit_method="moving_average")
    
    # 4. 损失对比图
    print("  [4/4] 损失对比图...")
    loss_path = output_dir / f"loss_comparison_complete_{timestamp}.png"
    plot_comparison_losses(series, loss_path)
    
    # 生成摘要
    summary = {
        "timestamp": timestamp,
        "note": f"APF Learnable数据目录: {APF_LEARNABLE_DIR}",
        "experiments": [
            {
                "label": item["label"],
                "name": item["name_en"],
                "log_dir": item["log_dir"],
                "episodes": len(item["metrics"].get("episode_rewards", [])),
                "final_reward": item["metrics"]["episode_rewards"][-1] if item["metrics"].get("episode_rewards") else None,
                "avg_reward": float(np.mean(item["metrics"]["episode_rewards"])) if item["metrics"].get("episode_rewards") else None,
                "max_reward": float(np.max(item["metrics"]["episode_rewards"])) if item["metrics"].get("episode_rewards") else None,
                "team_success_rate": item["metrics"].get("team_success_rate", 0.0)
            }
            for item in series
        ]
    }
    
    summary_path = output_dir / f"summary_complete_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 同时保存为最新汇总
    latest_summary_path = output_dir / "latest_summary_complete.json"
    with open(latest_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✓ 所有图表生成完成（包含完整的4个实验）")
    print("="*70)
    print(f"\n输出目录: {output_dir}/")
    print(f"  - reward_comparison_complete_{timestamp}.png")
    print(f"  - success_collision_clearance_comparison_complete_{timestamp}.png")
    print(f"  - success_rate_and_clearance_comparison_complete_{timestamp}.png")
    print(f"  - loss_comparison_complete_{timestamp}.png")
    print(f"  - summary_complete_{timestamp}.json")
    print(f"  - latest_summary_complete.json (最新汇总)")
    
    print("\n实验结果汇总:")
    for exp in summary["experiments"]:
        print(f"  - {exp['name']}:")
        print(f"      Episodes: {exp['episodes']}")
        if exp['final_reward']:
            print(f"      Final Reward: {exp['final_reward']:,.2f}")
        if exp['avg_reward']:
            print(f"      Avg Reward: {exp['avg_reward']:,.2f}")
        if exp['max_reward']:
            print(f"      Max Reward: {exp['max_reward']:,.2f}")
        print(f"      Team Success Rate: {exp['team_success_rate']:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


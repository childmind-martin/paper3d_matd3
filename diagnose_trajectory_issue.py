#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断蓝色智能体轨迹异常问题
检查APF的Z方向力和地形高度采样
"""

import json
import numpy as np
from pathlib import Path

def analyze_evaluation_results(eval_results_path):
    """分析评估结果，找出异常轨迹"""
    eval_results_path = Path(eval_results_path)
    results_json = eval_results_path / "evaluation_results.json"
    
    if not results_json.exists():
        print(f"❌ 找不到评估结果: {results_json}")
        return
    
    with open(results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"="*70)
    print(f"评估结果诊断")
    print(f"="*70)
    print(f"路径: {eval_results_path}")
    print(f"")
    
    # 分析每个episode
    episode_details = results.get('episode_details', [])
    
    for ep_idx, ep_data in enumerate(episode_details):
        print(f"\n{'='*70}")
        print(f"Episode {ep_idx + 1}")
        print(f"{'='*70}")
        
        # 基本信息
        reward = ep_data.get('reward', 0)
        success = ep_data.get('success', 0)
        collisions = ep_data.get('collision_count', 0)
        
        print(f"奖励: {reward:.2f}")
        print(f"成功: {success}")
        print(f"碰撞: {collisions}")
        
        # 分析轨迹
        trajectory = ep_data.get('trajectory', [])
        if not trajectory:
            print("⚠️  没有轨迹数据")
            continue
        
        trajectory = np.array(trajectory)
        print(f"轨迹形状: {trajectory.shape} (steps, agents, xyz)")
        
        # 分析每个智能体
        n_agents = trajectory.shape[1]
        for agent_idx in range(n_agents):
            agent_traj = trajectory[:, agent_idx, :]  # (steps, 3)
            
            # 统计Z坐标
            z_coords = agent_traj[:, 2]
            z_min = np.min(z_coords)
            z_max = np.max(z_coords)
            z_mean = np.mean(z_coords)
            z_std = np.std(z_coords)
            
            # 检测异常
            is_abnormal = z_max > 100 or z_min < -10
            agent_names = ["黑色", "红色", "蓝色"]
            agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent{agent_idx}"
            
            status = "❌ 异常" if is_abnormal else "✅ 正常"
            print(f"\n  {agent_name} 智能体 {status}:")
            print(f"    Z坐标范围: [{z_min:.2f}, {z_max:.2f}]")
            print(f"    Z坐标均值: {z_mean:.2f} ± {z_std:.2f}")
            
            if is_abnormal:
                # 找到异常发生的时间步
                abnormal_steps = np.where((z_coords > 100) | (z_coords < -10))[0]
                print(f"    ⚠️  异常步数: {len(abnormal_steps)} / {len(z_coords)}")
                print(f"    ⚠️  首次异常: 步数 {abnormal_steps[0]} (Z={z_coords[abnormal_steps[0]]:.2f})")
                
                # 显示异常前后的轨迹
                if abnormal_steps[0] > 0:
                    before_idx = abnormal_steps[0] - 1
                    after_idx = abnormal_steps[0]
                    print(f"    异常前 (步{before_idx}): XYZ={agent_traj[before_idx]}")
                    print(f"    异常时 (步{after_idx}): XYZ={agent_traj[after_idx]}")
                    print(f"    Z方向变化: {agent_traj[after_idx, 2] - agent_traj[before_idx, 2]:.2f} 米")

def main():
    """查找并分析最近的评估结果"""
    # 查找terrain_sensing实验结果
    results_dirs = list(Path("terrain_sensing_experiments").glob("batch_*/evaluation_results/*"))
    
    if not results_dirs:
        # 尝试evaluation_results目录
        results_dirs = list(Path("evaluation_results").glob("*"))
    
    if not results_dirs:
        print("❌ 找不到评估结果目录")
        print("请提供评估结果路径，例如：")
        print("  python diagnose_trajectory_issue.py terrain_sensing_experiments/batch_20260122_xxx/evaluation_results/apf_learnable_oracle_same")
        return
    
    # 按修改时间排序，分析最新的
    results_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"找到 {len(results_dirs)} 个评估结果目录")
    print(f"\n分析最新的 3 个:")
    
    for i, results_dir in enumerate(results_dirs[:3]):
        if i > 0:
            print(f"\n{'='*70}\n")
        analyze_evaluation_results(results_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_evaluation_results(sys.argv[1])
    else:
        main()

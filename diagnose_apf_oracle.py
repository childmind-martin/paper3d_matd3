#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断Oracle模式下APF的问题
分析地形高度、净空、PF力方向
"""

import json
import numpy as np
from pathlib import Path

def analyze_trajectory_detail(eval_results_path, episode_idx=3):
    """详细分析单个episode的轨迹"""
    eval_results_path = Path(eval_results_path)
    results_json = eval_results_path / "evaluation_results.json"
    
    if not results_json.exists():
        print(f"❌ 找不到评估结果: {results_json}")
        return
    
    with open(results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    episode_details = results.get('episode_details', [])
    
    if episode_idx >= len(episode_details):
        print(f"❌ Episode {episode_idx + 1} 不存在（共{len(episode_details)}个episode）")
        return
    
    ep_data = episode_details[episode_idx]
    
    print(f"="*70)
    print(f"Episode {episode_idx + 1} 详细分析")
    print(f"="*70)
    
    # 基本信息
    reward = ep_data.get('reward', 0)
    success = ep_data.get('success', 0)
    collisions = ep_data.get('collision_count', 0)
    
    print(f"\n基本信息:")
    print(f"  奖励: {reward:.2f}")
    print(f"  成功: {success}")
    print(f"  碰撞: {collisions}")
    
    # 分析轨迹
    trajectory = np.array(ep_data.get('trajectory', []))
    if trajectory.size == 0:
        print("⚠️  没有轨迹数据")
        return
    
    print(f"\n轨迹形状: {trajectory.shape} (steps, agents, xyz)")
    
    # 找出蓝色智能体（agent 2）
    agent_idx = 2
    agent_traj = trajectory[:, agent_idx, :]  # (steps, 3)
    agent_names = ["黑色", "红色", "蓝色"]
    agent_name = agent_names[agent_idx]
    
    print(f"\n{'='*70}")
    print(f"{agent_name} 智能体轨迹分析")
    print(f"{'='*70}")
    
    # Z坐标统计
    z_coords = agent_traj[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    z_mean = np.mean(z_coords)
    z_start = z_coords[0]
    z_end = z_coords[-1]
    
    print(f"\nZ坐标统计:")
    print(f"  起点Z: {z_start:.2f}m")
    print(f"  终点Z: {z_end:.2f}m")
    print(f"  最小Z: {z_min:.2f}m")
    print(f"  最大Z: {z_max:.2f}m")
    print(f"  均值Z: {z_mean:.2f}m ± {np.std(z_coords):.2f}m")
    print(f"  Z变化: {z_end - z_start:+.2f}m")
    
    # 找到Z坐标突变的位置
    z_diff = np.diff(z_coords)
    large_changes = np.where(np.abs(z_diff) > 5.0)[0]
    
    if len(large_changes) > 0:
        print(f"\n⚠️  检测到 {len(large_changes)} 处Z坐标大幅变化 (>5m/step):")
        for idx in large_changes[:5]:  # 只显示前5个
            print(f"    步数 {idx} → {idx+1}: Z从 {z_coords[idx]:.2f}m 变化到 {z_coords[idx+1]:.2f}m (Δ{z_diff[idx]:+.2f}m)")
    
    # XY平面移动分析
    xy_traj = agent_traj[:, :2]
    xy_dist = np.linalg.norm(np.diff(xy_traj, axis=0), axis=1)
    total_xy_dist = np.sum(xy_dist)
    
    print(f"\n水平移动:")
    print(f"  XY平面总距离: {total_xy_dist:.2f}m")
    print(f"  平均速度: {total_xy_dist / len(z_coords):.2f}m/step")
    
    # 速度分析（3D）
    velocity_3d = np.linalg.norm(np.diff(agent_traj, axis=0), axis=1)
    print(f"  3D平均速度: {np.mean(velocity_3d):.2f}m/step")
    print(f"  3D最大速度: {np.max(velocity_3d):.2f}m/step")
    
    # 分析轨迹的不同阶段
    n_steps = len(z_coords)
    stages = [
        ("起始阶段", 0, min(100, n_steps)),
        ("中间阶段", n_steps//2 - 50, n_steps//2 + 50),
        ("结束阶段", max(0, n_steps - 100), n_steps)
    ]
    
    print(f"\n分阶段分析:")
    for stage_name, start, end in stages:
        if start < 0 or end > n_steps or start >= end:
            continue
        z_stage = z_coords[start:end]
        print(f"  {stage_name} (步数{start}-{end}):")
        print(f"    Z范围: [{np.min(z_stage):.2f}, {np.max(z_stage):.2f}]m")
        print(f"    Z均值: {np.mean(z_stage):.2f}m")
        print(f"    Z变化趋势: {z_stage[-1] - z_stage[0]:+.2f}m")

def main():
    """查找并分析Oracle Same Probes模式的Episode 4"""
    # 查找最新的Oracle Same Probes结果
    results_dirs = list(Path("terrain_sensing_experiments").glob("batch_*/evaluation_results/*oracle_same*"))
    
    if not results_dirs:
        print("❌ 找不到Oracle Same Probes的评估结果")
        return
    
    # 按修改时间排序，使用最新的
    results_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_result = results_dirs[0]
    
    print(f"分析最新的Oracle Same Probes结果:")
    print(f"路径: {latest_result}\n")
    
    # 分析Episode 4（索引3，因为从0开始）
    analyze_trajectory_detail(latest_result, episode_idx=3)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        analyze_trajectory_detail(sys.argv[1], int(sys.argv[2]) - 1)
    elif len(sys.argv) > 1:
        analyze_trajectory_detail(sys.argv[1], 3)
    else:
        main()

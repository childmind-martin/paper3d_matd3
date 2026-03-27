#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试半随机地形生成
验证：
1. 基准山峰位置是否固定
2. 实际山峰位置是否在合理范围内波动
3. 起始点和目标点距离是否保持稳定
"""

import os
import sys
import numpy as np
from pathlib import Path

# 设置环境变量
os.environ['SEMI_RANDOM_TERRAIN'] = '1'  # 启用半随机地形
os.environ['TERRAIN_BASE_SEED'] = '67'  # 基准种子
os.environ['PEAK_JITTER_RANGE'] = '15.0'  # 波动范围
os.environ['SCENARIO_SEED'] = '67'  # 场景种子
os.environ['USE_FIXED_POSITIONS'] = '1'
os.environ['DYNAMIC_FIRST_TIME'] = '0'
os.environ['RANDOM_TERRAIN'] = '0'
os.environ['MAP_SIZE'] = '200'
os.environ['TERRAIN_COMPLEXITY_LEVEL'] = '3'
os.environ['SUPPRESS_TERRAIN_OUTPUT'] = '0'
os.environ['SUPPRESS_MA_PROMPT'] = '1'  # 禁止MA环境提示

# 导入场景
sys.path.insert(0, str(Path(__file__).parent))
from multiagent.scenarios.paper3d_terrain_energy import Scenario

def test_semi_random_terrain(num_trials=10):
    """测试多次重置，验证地形生成的一致性和波动性"""
    
    print("="*80)
    print("半随机地形生成测试")
    print("="*80)
    print(f"测试次数: {num_trials}")
    print(f"基准种子: {os.environ['TERRAIN_BASE_SEED']}")
    print(f"波动范围: {os.environ['PEAK_JITTER_RANGE']}m")
    print("="*80)
    
    # 创建场景
    scenario = Scenario(
        seed=67,
        use_fixed_positions=True,
        dynamic_first_time=False,
        terrain_complexity_level=3
    )
    world = scenario.make_world()
    
    # 记录每次重置的山峰位置
    all_peak_positions = []
    all_agent_positions = []
    all_goal_positions = []
    
    for trial in range(num_trials):
        print(f"\n--- 试验 {trial + 1}/{num_trials} ---")
        
        # 使用不同的随机种子重置（模拟不同回合）
        scenario.seed = 67 + trial * 100  # 改变场景种子
        scenario.rng = np.random.RandomState(scenario.seed)
        
        # 重置世界
        scenario.reset_world(world)
        
        # 获取山峰位置
        peak_positions = scenario.mountain_centers if hasattr(scenario, 'mountain_centers') else []
        all_peak_positions.append(peak_positions)
        
        # 获取智能体位置
        agent_positions = [agent.state.p_pos.copy() for agent in world.agents]
        all_agent_positions.append(agent_positions)
        
        # 获取目标位置
        goal_position = scenario.goal_pos.copy()
        all_goal_positions.append(goal_position)
        
        print(f"  山峰数量: {len(peak_positions)}")
        print(f"  目标位置: [{goal_position[0]:.2f}, {goal_position[1]:.2f}, {goal_position[2]:.2f}]")
        
        # 计算起始点到目标的距离
        avg_start_pos = np.mean([pos for pos in agent_positions], axis=0)
        distance = np.linalg.norm(avg_start_pos - goal_position)
        print(f"  起始-目标距离: {distance:.2f}m")
    
    # 分析结果
    print("\n" + "="*80)
    print("分析结果")
    print("="*80)
    
    # 1. 分析山峰位置波动
    if len(all_peak_positions) > 1 and len(all_peak_positions[0]) > 0:
        num_peaks = len(all_peak_positions[0])
        print(f"\n1. 山峰位置波动分析（山峰数量: {num_peaks}）")
        
        for peak_idx in range(num_peaks):
            positions = [trial[peak_idx][:2] for trial in all_peak_positions if len(trial) > peak_idx]
            positions = np.array(positions)
            
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            max_deviation = np.max([np.linalg.norm(pos - mean_pos) for pos in positions])
            
            print(f"  山峰 {peak_idx + 1}:")
            print(f"    平均位置: [{mean_pos[0]:.2f}, {mean_pos[1]:.2f}]")
            print(f"    标准差: X={std_pos[0]:.2f}m, Y={std_pos[1]:.2f}m")
            print(f"    最大偏移: {max_deviation:.2f}m")
    
    # 2. 分析起始-目标距离稳定性
    print(f"\n2. 起始-目标距离稳定性分析")
    distances = []
    for agents, goal in zip(all_agent_positions, all_goal_positions):
        avg_start = np.mean([pos for pos in agents], axis=0)
        dist = np.linalg.norm(avg_start - goal)
        distances.append(dist)
    
    distances = np.array(distances)
    print(f"  平均距离: {np.mean(distances):.2f}m")
    print(f"  标准差: {np.std(distances):.2f}m")
    print(f"  最小距离: {np.min(distances):.2f}m")
    print(f"  最大距离: {np.max(distances):.2f}m")
    print(f"  距离范围: {np.max(distances) - np.min(distances):.2f}m")
    
    # 3. 分析目标位置稳定性
    print(f"\n3. 目标位置稳定性分析")
    goal_positions = np.array(all_goal_positions)
    goal_mean = np.mean(goal_positions, axis=0)
    goal_std = np.std(goal_positions, axis=0)
    
    print(f"  平均位置: [{goal_mean[0]:.2f}, {goal_mean[1]:.2f}, {goal_mean[2]:.2f}]")
    print(f"  标准差: X={goal_std[0]:.2f}m, Y={goal_std[1]:.2f}m, Z={goal_std[2]:.2f}m")
    
    # 4. 验证结论
    print("\n" + "="*80)
    print("验证结论")
    print("="*80)
    
    jitter_range = float(os.environ['PEAK_JITTER_RANGE'])
    
    if len(all_peak_positions) > 1 and len(all_peak_positions[0]) > 0:
        # 检查山峰偏移是否在合理范围内
        peak_check = True
        for peak_idx in range(num_peaks):
            positions = [trial[peak_idx][:2] for trial in all_peak_positions if len(trial) > peak_idx]
            positions = np.array(positions)
            mean_pos = np.mean(positions, axis=0)
            max_deviation = np.max([np.linalg.norm(pos - mean_pos) for pos in positions])
            
            if max_deviation > jitter_range * 1.5:  # 允许一定容差
                peak_check = False
                print(f"⚠️  山峰 {peak_idx + 1} 偏移超出预期范围: {max_deviation:.2f}m > {jitter_range * 1.5:.2f}m")
        
        if peak_check:
            print(f"✅ 山峰位置波动正常（最大偏移 < {jitter_range * 1.5:.2f}m）")
    
    # 检查距离稳定性（应该非常小的标准差）
    if np.std(distances) < 5.0:
        print(f"✅ 起始-目标距离稳定（标准差={np.std(distances):.2f}m < 5.0m）")
    else:
        print(f"⚠️  起始-目标距离波动较大（标准差={np.std(distances):.2f}m）")
    
    # 检查目标位置稳定性
    if np.max(goal_std[:2]) < 2.0:  # XY方向应该很稳定
        print(f"✅ 目标位置稳定（XY标准差 < 2.0m）")
    else:
        print(f"⚠️  目标位置波动较大（XY标准差: X={goal_std[0]:.2f}m, Y={goal_std[1]:.2f}m）")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)

if __name__ == "__main__":
    test_semi_random_terrain(num_trials=10)

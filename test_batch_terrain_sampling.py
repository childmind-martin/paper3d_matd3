#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证批量地形采样的正确性和性能
确保批量版本与单点版本数值完全一致，并测量性能提升
"""

import numpy as np
import sys
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_terrain_sampling_correctness():
    """测试批量地形采样的数值正确性"""
    print("="*70)
    print("测试1: 批量地形采样数值正确性")
    print("="*70)
    
    # 导入场景
    from multiagent.scenarios.paper3d_terrain_energy import Scenario
    
    # 创建场景并生成地形
    import os
    os.environ['MAP_SIZE'] = '200'
    os.environ['TERRAIN_COMPLEXITY_LEVEL'] = '3'
    os.environ['NUM_OBSTACLES'] = '10'
    os.environ['SCENARIO_SEED'] = '42'
    
    scenario = Scenario()
    world = scenario.make_world()
    
    # 测试用例1: 单点测试
    print("\n测试用例1: 单点查询")
    test_coords_single = [
        [100.5, 100.5],
        [0.0, 0.0],
        [199.0, 199.0],
        [50.3, 150.7],
    ]
    
    for coord in test_coords_single:
        x, y = coord
        # 单点版本
        h_single = scenario.get_terrain_height(x, y)
        # 批量版本（单个坐标）
        h_batch = scenario.batch_get_terrain_height(np.array([coord]))[0]
        
        diff = abs(h_single - h_batch)
        status = "✅ PASS" if diff < 1e-5 else "❌ FAIL"
        print(f"  坐标({x:6.1f}, {y:6.1f}): 单点={h_single:7.3f}, 批量={h_batch:7.3f}, 差值={diff:.2e} {status}")
    
    # 测试用例2: 批量测试
    print("\n测试用例2: 批量查询 (100个随机点)")
    np.random.seed(42)
    test_coords_batch = np.random.uniform(0, 199, (100, 2)).astype(np.float32)
    
    # 单点版本（逐个查询）
    heights_single = np.array([scenario.get_terrain_height(x, y) for x, y in test_coords_batch])
    
    # 批量版本（一次查询）
    heights_batch = scenario.batch_get_terrain_height(test_coords_batch)
    
    # 比较结果
    max_diff = np.max(np.abs(heights_single - heights_batch))
    mean_diff = np.mean(np.abs(heights_single - heights_batch))
    
    print(f"  最大差值: {max_diff:.2e}")
    print(f"  平均差值: {mean_diff:.2e}")
    print(f"  相对误差: {mean_diff / (np.mean(np.abs(heights_single)) + 1e-6) * 100:.4f}%")
    
    if max_diff < 1e-4:
        print(f"  ✅ PASS: 批量版本与单点版本数值一致（误差 < 1e-4）")
        return True
    else:
        print(f"  ❌ FAIL: 批量版本与单点版本存在显著差异")
        # 打印前10个差异较大的点
        diff_indices = np.argsort(np.abs(heights_single - heights_batch))[-10:]
        print("\n  差异最大的10个点:")
        for idx in reversed(diff_indices):
            x, y = test_coords_batch[idx]
            print(f"    坐标({x:6.1f}, {y:6.1f}): 单点={heights_single[idx]:7.3f}, "
                  f"批量={heights_batch[idx]:7.3f}, 差值={heights_single[idx] - heights_batch[idx]:.2e}")
        return False


def test_terrain_sampling_performance():
    """测试批量地形采样的性能提升"""
    print("\n" + "="*70)
    print("测试2: 批量地形采样性能提升")
    print("="*70)
    
    # 导入场景
    from multiagent.scenarios.paper3d_terrain_energy import Scenario
    
    # 创建场景并生成地形
    import os
    os.environ['MAP_SIZE'] = '200'
    os.environ['TERRAIN_COMPLEXITY_LEVEL'] = '3'
    os.environ['NUM_OBSTACLES'] = '10'
    os.environ['SCENARIO_SEED'] = '42'
    
    scenario = Scenario()
    world = scenario.make_world()
    
    # 生成测试数据：模拟observation()中的41个采样点
    np.random.seed(42)
    num_agents = 3
    num_samples_per_agent = 41  # 每个agent在observation()中的采样点数量
    
    # 为每个agent生成随机采样坐标
    all_coords = []
    for i in range(num_agents):
        agent_coords = np.random.uniform(0, 199, (num_samples_per_agent, 2)).astype(np.float32)
        all_coords.append(agent_coords)
    
    # 测试1: 单点版本（逐个查询，模拟原始代码）
    print(f"\n性能测试: {num_agents}个agent × {num_samples_per_agent}个采样点 = {num_agents * num_samples_per_agent}次查询")
    
    num_iterations = 100  # 重复次数
    
    # 预热
    for coords in all_coords:
        _ = scenario.batch_get_terrain_height(coords)
    
    # 单点版本计时
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        for coords in all_coords:
            heights = [scenario.get_terrain_height(x, y) for x, y in coords]
    time_single = time.perf_counter() - start_time
    
    # 批量版本计时
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        for coords in all_coords:
            heights = scenario.batch_get_terrain_height(coords)
    time_batch = time.perf_counter() - start_time
    
    # 计算性能提升
    speedup = time_single / time_batch
    time_per_call_single = (time_single / num_iterations / num_agents) * 1000  # ms
    time_per_call_batch = (time_batch / num_iterations / num_agents) * 1000  # ms
    
    print(f"\n  单点版本: {time_single:.3f}s ({time_per_call_single:.3f}ms/agent)")
    print(f"  批量版本: {time_batch:.3f}s ({time_per_call_batch:.3f}ms/agent)")
    print(f"  ⚡ 加速比: {speedup:.2f}x")
    print(f"  ⚡ 节省时间: {time_per_call_single - time_per_call_batch:.3f}ms/agent")
    
    if speedup > 2.0:
        print(f"  ✅ PASS: 批量版本性能显著提升 (>{speedup:.1f}x)")
        return True
    elif speedup > 1.2:
        print(f"  ⚠️  WARNING: 批量版本性能提升有限 ({speedup:.2f}x)")
        return True
    else:
        print(f"  ❌ FAIL: 批量版本性能提升不明显 ({speedup:.2f}x)")
        return False


def test_observation_integration():
    """测试observation()函数集成后的正确性"""
    print("\n" + "="*70)
    print("测试3: observation()函数集成验证")
    print("="*70)
    
    # 导入必要的模块
    from multiagent.scenarios.paper3d_terrain_energy import Scenario
    from multiagent.environment import MultiAgentEnv
    
    # 创建环境
    import os
    os.environ['MAP_SIZE'] = '200'
    os.environ['TERRAIN_COMPLEXITY_LEVEL'] = '3'
    os.environ['NUM_OBSTACLES'] = '10'
    os.environ['SCENARIO_SEED'] = '42'
    
    scenario = Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation
    )
    
    print("\n测试observation()函数...")
    try:
        # 重置环境
        result = env.reset()
        print(f"  ✅ env.reset() 成功")
        
        # 处理MultiAgentEnv的返回格式：(observations, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs_n, info = result
            obs_list = obs_n if isinstance(obs_n, list) else [obs_n]
        elif isinstance(result, (list, tuple)):
            obs_list = result
        else:
            obs_list = [result]
        
        # 转换为numpy数组
        obs_arrays = [np.asarray(obs) for obs in obs_list if not isinstance(obs, dict)]
        print(f"  观察维度: {[obs.shape for obs in obs_arrays]}")
        
        # 检查每个agent的观察
        for i, obs in enumerate(obs_arrays):
            if obs.shape[0] != 81:
                print(f"  ❌ FAIL: Agent {i} 观察维度错误: {obs.shape[0]} (期望81)")
                return False
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"  ❌ FAIL: Agent {i} 观察包含NaN或Inf")
                return False
        
        # 执行几步，确保没有错误
        print(f"  执行10步测试...")
        for step in range(10):
            # 生成随机动作
            actions = [np.random.randn(3) for _ in range(3)]  # 3个agent，每个3维动作
            result = env.step(actions)
            # 处理返回值
            if isinstance(result, tuple) and len(result) >= 3:
                pass  # 成功
            else:
                print(f"  ❌ FAIL: env.step()返回值格式错误")
                return False
        
        print(f"  ✅ 执行10步，无错误")
        print(f"  ✅ PASS: observation()函数集成正确")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: observation()函数出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("批量地形采样验证测试")
    print("="*70)
    
    results = []
    
    # 测试1: 数值正确性
    try:
        result1 = test_terrain_sampling_correctness()
        results.append(("数值正确性", result1))
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("数值正确性", False))
    
    # 测试2: 性能提升
    try:
        result2 = test_terrain_sampling_performance()
        results.append(("性能提升", result2))
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("性能提升", False))
    
    # 测试3: 集成验证
    try:
        result3 = test_observation_integration()
        results.append(("集成验证", result3))
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("集成验证", False))
    
    # 汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "="*70)
    if all_passed:
        print("✅ 所有测试通过！批量地形采样优化成功！")
        print("="*70)
        return 0
    else:
        print("❌ 部分测试失败，请检查代码")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

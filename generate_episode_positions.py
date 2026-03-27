#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个episode生成固定的位置文件
确保所有评估模式（Oracle Same、Oracle Dense、Local）使用相同的初始条件
"""

import json
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from multiagent.environment import MultiAgentEnv
    from multiagent.scenarios import load
except ImportError:
    # 如果导入失败，尝试添加项目路径
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from multiagent.environment import MultiAgentEnv
    from multiagent.scenarios import load


def generate_episode_positions(terrain_seed, episode_idx, output_dir, base_env_vars=None):
    """
    为单个episode生成固定的位置文件
    
    Args:
        terrain_seed: 地形种子
        episode_idx: episode索引（用于文件名）
        output_dir: 输出目录
        base_env_vars: 基础环境变量字典
    
    Returns:
        位置文件路径
    """
    if base_env_vars is None:
        base_env_vars = {}
    
    # 设置环境变量
    env_vars = os.environ.copy()
    env_vars.update(base_env_vars)
    
    # 设置地形种子
    env_vars["SCENARIO_SEED"] = str(terrain_seed)
    env_vars["USE_SCENARIO_SEED"] = "1"
    env_vars["RANDOM_TERRAIN"] = "0"
    env_vars["PER_EPISODE_TERRAIN"] = "0"
    env_vars["USE_FIXED_POSITIONS"] = "0"  # 禁用固定位置，让系统生成
    env_vars["DYNAMIC_FIRST_TIME"] = "1"  # 启用动态首次，生成位置
    
    # 设置其他必要的环境变量
    env_vars.setdefault("MAP_SIZE", "200")
    env_vars.setdefault("TERRAIN_COMPLEXITY_LEVEL", "3")
    env_vars.setdefault("MOUNTAIN_MIN_DISTANCE", "55")
    
    # 临时设置环境变量
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        # 解析关键配置，确保与评估/训练一致
        try:
            terrain_level = int(env_vars.get("TERRAIN_COMPLEXITY_LEVEL", 3))
        except Exception:
            terrain_level = 3
        try:
            map_size = float(env_vars.get("MAP_SIZE", 200))
        except Exception:
            map_size = 200.0

        # 加载场景（与评估保持一致）
        scenario = load("paper3d_terrain_weighted").Scenario(
            seed=int(terrain_seed),
            use_fixed_positions=False,
            dynamic_first_time=True,
            fixed_positions_file=None,
            random_terrain=False,
            terrain_complexity_level=terrain_level,
            map_size=map_size,
        )
        world = scenario.make_world()
        
        # 重置世界（这会生成初始位置）
        scenario.reset_world(world)
        
        # 🚨 关键修复：验证并调整智能体位置，确保所有智能体都在地形上方
        # 原因：reset_world后，智能体位置应该已经在地形上方，但为了确保，再次验证
        altitude_offset = float(env_vars.get('START_ALTITUDE_OFFSET', '7.0'))
        min_air_gap = max(1.0, altitude_offset)
        
        # 提取智能体和目标位置
        agent_positions = []
        for i, agent in enumerate(world.agents):
            pos = agent.state.p_pos.copy()
            # 验证Z坐标是否在地形上方
            terrain_h = scenario.get_terrain_height(pos[0], pos[1])
            required_height = terrain_h + min_air_gap
            if pos[2] < required_height:
                # 如果Z坐标太低，调整到地形上方
                old_z = pos[2]
                pos[2] = required_height
                print(f"⚠️  [位置生成] Agent{i}: Z坐标从{old_z:.2f}调整到{pos[2]:.2f}（地形高度={terrain_h:.2f}）")
            agent_positions.append(pos.tolist())
        
        # 🚨 关键修复：验证并调整目标位置，确保在地形上方
        goal_pos = scenario.goal_pos.copy() if scenario.goal_pos is not None else np.array([100.0, 100.0, 50.0])
        if scenario.goal_pos is not None:
            goal_terrain_h = scenario.get_terrain_height(goal_pos[0], goal_pos[1])
            goal_altitude = float(env_vars.get('GOAL_ALTITUDE', '12.0'))
            required_goal_height = goal_terrain_h + goal_altitude
            if goal_pos[2] < required_goal_height:
                old_goal_z = goal_pos[2]
                goal_pos[2] = required_goal_height
                print(f"⚠️  [位置生成] 目标: Z坐标从{old_goal_z:.2f}调整到{goal_pos[2]:.2f}（地形高度={goal_terrain_h:.2f}）")
        goal_pos = goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else goal_pos
        
        # 构建位置数据
        positions_data = {
            "terrain_seed": terrain_seed,
            "episode_idx": episode_idx,
            "agents": agent_positions,
            "goal": goal_pos
        }
        
        # 保存到文件
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        positions_file = output_dir / f"episode_{episode_idx:03d}_seed_{terrain_seed}.json"
        
        with open(positions_file, 'w', encoding='utf-8') as f:
            json.dump(positions_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 生成Episode {episode_idx}位置文件: {positions_file}")
        print(f"   地形种子: {terrain_seed}")
        print(f"   智能体数量: {len(agent_positions)}")
        print(f"   智能体位置: {[f'({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})' for p in agent_positions]}")
        print(f"   目标位置: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f})")
        
        return positions_file
        
    except Exception as e:
        print(f"❌ 生成Episode {episode_idx}位置文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # 恢复原始环境变量
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def generate_all_episode_positions(terrain_seeds, output_dir, base_env_vars=None):
    """
    为所有episode生成固定的位置文件
    
    Args:
        terrain_seeds: 地形种子列表
        output_dir: 输出目录
        base_env_vars: 基础环境变量字典
    
    Returns:
        位置文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    positions_files = []
    
    print(f"\n{'='*70}")
    print(f"开始为 {len(terrain_seeds)} 个episode生成固定位置文件")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    for episode_idx, terrain_seed in enumerate(terrain_seeds):
        positions_file = generate_episode_positions(
            terrain_seed, episode_idx, output_dir, base_env_vars
        )
        if positions_file:
            positions_files.append(positions_file)
    
    print(f"\n{'='*70}")
    print(f"✅ 成功生成 {len(positions_files)}/{len(terrain_seeds)} 个位置文件")
    print(f"{'='*70}\n")
    
    return positions_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="为每个episode生成固定的位置文件")
    parser.add_argument("--terrain-seeds", type=str, required=True,
                        help="地形种子序列（逗号分隔）")
    parser.add_argument("--output-dir", type=str, default="./saved_positions/episode_positions",
                        help="输出目录（默认: ./saved_positions/episode_positions）")
    parser.add_argument("--map-size", type=str, default="200",
                        help="地图大小（默认: 200）")
    parser.add_argument("--terrain-complexity", type=str, default="3",
                        help="地形复杂度（默认: 3）")
    
    args = parser.parse_args()
    
    # 解析地形种子序列
    terrain_seeds = [int(s.strip()) for s in args.terrain_seeds.split(',') if s.strip()]
    
    # 基础环境变量
    base_env_vars = {
        "MAP_SIZE": args.map_size,
        "TERRAIN_COMPLEXITY_LEVEL": args.terrain_complexity,
    }
    
    # 生成所有位置文件
    positions_files = generate_all_episode_positions(
        terrain_seeds, args.output_dir, base_env_vars
    )
    
    print(f"\n生成的位置文件列表:")
    for f in positions_files:
        print(f"  - {f}")

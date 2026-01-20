#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
numpy向量化优化的分项加权求和奖励场景
针对并行环境进行性能优化，使用批量计算提升效率
"""

import numpy as np
import sys
import os

# 添加utils路径以导入向量化奖励计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
try:
    from vectorized_reward_calculator import VectorizedRewardCalculator
except ImportError as e:
    print(f"Warning: Failed to import VectorizedRewardCalculator: {e}")
    print("Please ensure vectorized_reward_calculator.py is in the utils/ directory")
    VectorizedRewardCalculator = None
from multiagent.scenarios.paper3d_terrain_weighted import Scenario as BaseWeightedScenario

# 可选：导入ARW（与 weighted 场景保持一致的接口）
try:
    from .reward_calculator import AdaptiveRewardWeighting
    _ARW_OK = True
except Exception:
    try:
        from reward_calculator import AdaptiveRewardWeighting
        _ARW_OK = True
    except Exception:
        _ARW_OK = False


class Scenario(BaseWeightedScenario):
    """
    numpy向量化优化的分项加权求和奖励场景
    继承原有场景的所有功能，但使用向量化奖励计算
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 🚨 关键修复：优先使用kwargs，而不是环境变量，确保并行环境配置独立
        # 原因：并行运行时，多个实验共享父进程环境变量，会导致配置冲突
        # 修复：优先使用kwargs传递的参数，只有在kwargs中没有时才使用环境变量
        import os
        # 优先使用kwargs，如果kwargs中没有或为None，才使用环境变量
        terrain_complexity_from_kwargs = kwargs.get('terrain_complexity_level')
        if terrain_complexity_from_kwargs is not None:
            self.terrain_complexity_level = int(terrain_complexity_from_kwargs)
        else:
            self.terrain_complexity_level = int(os.environ.get('TERRAIN_COMPLEXITY_LEVEL', 2))
        
        # 🔧 性能优化：添加奖励缓存机制，避免每步重复计算
        # 缓存键：(world_id, current_step)，缓存值：batch_rewards数组
        self._reward_cache = {}
        self._reward_cache_step = -1  # 当前缓存的步数
        self._reward_cache_world_id = None  # 当前缓存的世界ID
        
        # 🚨 关键修复：优先使用kwargs，确保并行环境配置独立
        no_collision_reward_from_kwargs = kwargs.get('no_collision_reward_value')
        if no_collision_reward_from_kwargs is not None:
            self.no_collision_reward_value = float(no_collision_reward_from_kwargs)
        else:
            self.no_collision_reward_value = float(os.environ.get('NO_COLLISION_REWARD_VALUE', 0.0))
        
        # 🔧 修复：直接调用paper3d_terrain_energy的复杂度参数设置方法
        # 因为BaseWeightedScenario没有这个方法，需要调用更上层的父类
        from multiagent.scenarios.paper3d_terrain_energy import Scenario as BaseTerrainScenario
        BaseTerrainScenario._setup_complexity_parameters(self)
        
        # 初始化向量化奖励计算器，传递所有关键参数
        vectorized_kwargs = {}
        if hasattr(self, 'height_reward_enabled'):
            vectorized_kwargs['height_reward_enabled'] = self.height_reward_enabled
        if hasattr(self, 'height_ideal_min'):
            vectorized_kwargs['height_ideal_min'] = self.height_ideal_min
        if hasattr(self, 'height_ideal_max'):
            vectorized_kwargs['height_ideal_max'] = self.height_ideal_max
        # 轨迹平滑（最小拐弯角）奖励权重（如有）
        if hasattr(self, 'turn_smooth_weight'):
            vectorized_kwargs['turn_smooth_weight'] = self.turn_smooth_weight
            
        if VectorizedRewardCalculator is not None:
            self.vectorized_calculator = VectorizedRewardCalculator(
                reward_weights=self.reward_weights,
                max_reward=self.max_reward,
                min_reward=self.min_reward,
                success_reward_value=self.success_reward_value,
                success_distance_threshold=self.success_distance_threshold,
                collision_penalty_value=self.collision_penalty_value,
                collision_distance_threshold=self.collision_distance_threshold,
                no_collision_reward_value=getattr(self, 'no_collision_reward_value', 0.0),
                global_reward_mode=self.global_reward_mode,
                shaping_gamma=self.shaping_gamma,
                **vectorized_kwargs
            )
        else:
            print("Warning: VectorizedRewardCalculator not available, falling back to base implementation")
            self.vectorized_calculator = None

        # 可选：创建 ARW，并暴露统一句柄（与训练侧兼容）
        self.arw = None
        if _ARW_OK and kwargs.get('enable_arw', True):
            try:
                self.arw = AdaptiveRewardWeighting(
                    base_penalty=kwargs.get('arw_base_penalty', -10.0),
                    alpha=kwargs.get('arw_alpha', 0.5),
                    base_c=kwargs.get('arw_base_c', 10.0),
                    beta=kwargs.get('arw_beta', 0.3),
                    max_episodes=kwargs.get('arw_max_episodes', 20000),
                    warmup_episodes=kwargs.get('arw_warmup_episodes', 200),
                    enable_adaptive=kwargs.get('arw_enable', True),
                )
                # 与训练侧保持兼容：reward_calculator 作为统一访问入口
                self.reward_calculator = self.arw
                print("[ARW] (vectorized) Adaptive Reward Weighting enabled")
            except Exception as _e:
                print(f"[ARW] (vectorized) init failed: {_e}")
                self.arw = None
        
        # 批量处理缓存
        self._batch_cache = {
            'agents_batch': None,
            'world_batch': None,
            'last_batch_size': 0,
            'last_n_agents': 0
        }
        
        # 性能统计
        self._performance_stats = {
            'total_calls': 0,
            'batch_calls': 0,
            'single_calls': 0,
            'vectorization_savings': 0.0
        }
    
    def make_world(self):
        """创建世界并设置场景引用，支持地面约束"""
        world = super().make_world()
        
        # 添加场景引用到世界对象，用于地面约束
        world.scenario = self
        
        return world

    # 在每个回合开始（reset_world 调用）时由基类触发。
    def on_episode_start(self):
        # 🚨 关键修复：在向量化场景中，确保每回合开始时重置碰撞计数
        # 注意：向量化场景的world可能不在self.world中，而是在环境对象中
        # 因此碰撞计数重置主要在initialize_agents_for_reward中进行
        # 这里只作为备用路径，尝试从基类获取world
        try:
            # 尝试从多个可能的路径获取world对象（备用路径）
            world_obj = None
            if hasattr(self, 'world'):
                world_obj = self.world
            elif hasattr(self, '_world'):
                world_obj = self._world
            
            if world_obj and hasattr(world_obj, 'agents'):
                for agent in world_obj.agents:
                    # 🚨 关键修复：每回合都重置total_penetration_count和initialized_for_reward，确保从0开始累积
                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    if not isinstance(agent.debug_info, dict):
                        agent.debug_info = {}
                    agent.debug_info['total_penetration_count'] = 0
                    # 🚨 关键修复：重置initialized_for_reward标志，确保initialize_agents_for_reward能正确检测新回合
                    agent.initialized_for_reward = False
                    # 🚨 关键修复：重置防重复计数标志，确保每回合开始时重置
                    agent._last_collision_counted_step = -1
        except Exception:
            # 静默失败，主要重置逻辑在initialize_agents_for_reward中
            pass
        
        # 🔧 性能优化：清除奖励缓存，确保新回合开始时重新计算
        self._reward_cache.clear()
        self._reward_cache_step = -1
        self._reward_cache_world_id = None
        
        try:
            # 推进 ARW 内部调度
            if getattr(self, 'arw', None) is not None:
                # 基类会多次创建并行 env，这里使用自增回合，不依赖外部计数
                ep = getattr(self, '_arw_episode', 0) + 1
                setattr(self, '_arw_episode', ep)
                self.arw.on_episode_start(ep)

                # 若有向量化计算器，按 ARW 比例动态更新奖励权重
                if getattr(self, 'vectorized_calculator', None) is not None:
                    vc = self.vectorized_calculator
                    try:
                        base_w = getattr(vc, '_base_reward_weights', None)
                        if base_w is None:
                            # 兼容：若无基线，使用当前值作为基线
                            base_w = getattr(vc, 'reward_weights')
                        w = np.array(base_w, dtype=np.float32).copy()
                        # 0: distance, 10: collision（见 calculator 定义）
                        if w.shape[0] >= 11:
                            w[0] = float(base_w[0]) * float(self.arw.distance_scale)
                            w[10] = float(base_w[10]) * float(self.arw.collision_scale)
                        vc.reward_weights = w
                    except Exception as _e:
                        # 出错不影响训练
                        pass
        except Exception:
            pass
    
    def reward(self, agent, world):
        """
        单个智能体奖励计算（保持兼容性）
        内部调用向量化版本以提高性能
        🔧 性能优化：使用缓存机制，每步只计算一次所有智能体的奖励，后续调用直接返回缓存结果
        """
        self._performance_stats['total_calls'] += 1
        self._performance_stats['single_calls'] += 1
        
        # 🚨 关键修复：在首次调用时初始化智能体（确保碰撞计数从0开始）
        # 原因：向量化场景中，initialize_agents_for_reward可能没有被调用
        # 因此需要在reward函数中确保碰撞计数被正确初始化
        if not hasattr(agent, 'debug_info') or not isinstance(agent.debug_info, dict):
            agent.debug_info = {}
        # 🚨 关键：如果total_penetration_count不存在，初始化为0
        # 注意：这里不重置已存在的计数，因为计数应该在每回合开始时重置（在on_episode_start中）
        if 'total_penetration_count' not in agent.debug_info:
            agent.debug_info['total_penetration_count'] = 0
        
        # 修复：传入所有智能体而不是单个智能体，以便全局奖励计算正确
        # 找到当前智能体的索引
        agent_idx = -1
        if hasattr(world, 'agents') and world.agents:
            for i, ag in enumerate(world.agents):
                if ag == agent or (hasattr(ag, 'name') and hasattr(agent, 'name') and ag.name == agent.name):
                    agent_idx = i
                    break
            # 为了保持兼容性，将所有智能体转换为批量格式
            agents_batch = [world.agents]  # ✅ 传入所有智能体
        else:
            # 回退：如果没有world.agents，则只传入单个智能体
            agents_batch = [[agent]]
            agent_idx = 0
        
        world_batch = [world]
        
        # 🔧 性能优化：检查缓存，避免重复计算
        # 获取当前步数和世界ID（用于缓存键）
        try:
            current_step = world.current_step if hasattr(world, 'current_step') else -1
        except (AttributeError, TypeError):
            current_step = -1
        
        # 使用world对象的id作为缓存键的一部分（区分不同的world实例）
        world_id = id(world)
        
        # 🔧 性能优化：如果步数变化，清除旧缓存（节省内存）
        if self._reward_cache_step != current_step or self._reward_cache_world_id != world_id:
            self._reward_cache.clear()
            self._reward_cache_step = current_step
            self._reward_cache_world_id = world_id
        
        # 检查缓存是否有效（同一world、同一步）
        cache_key = (world_id, current_step)
        if cache_key in self._reward_cache:
            # 缓存命中：直接返回缓存结果
            batch_rewards = self._reward_cache[cache_key]
            if agent_idx >= 0 and agent_idx < batch_rewards.shape[1]:
                return batch_rewards[0, agent_idx]
            else:
                return batch_rewards[0, 0] if batch_rewards.shape[1] > 0 else 0.0
        
        # 缓存未命中：计算所有智能体的奖励
        if self.vectorized_calculator is not None:
            batch_rewards = self.vectorized_calculator.batch_calculate_rewards(agents_batch, world_batch)
            # 缓存结果
            self._reward_cache[cache_key] = batch_rewards
            # 返回当前智能体的奖励
            if agent_idx >= 0 and agent_idx < batch_rewards.shape[1]:
                return batch_rewards[0, agent_idx]
            else:
                return batch_rewards[0, 0]
        else:
            # 回退到父类实现
            return super().reward(agent, world)
    
    def batch_reward(self, agents_batch, world_batch):
        """
        批量奖励计算 - 主要优化接口
        
        Args:
            agents_batch: (batch_size, n_agents) 智能体批次
            world_batch: (batch_size,) 世界批次
            
        Returns:
            rewards: (batch_size, n_agents) 奖励数组
        """
        self._performance_stats['total_calls'] += 1
        self._performance_stats['batch_calls'] += 1
        
        # 使用向量化计算器（向后兼容：最小参数集）
        if self.vectorized_calculator is not None:
            return self.vectorized_calculator.batch_calculate_rewards(agents_batch, world_batch)
        else:
            # 回退到逐个计算
            batch_size = len(agents_batch)
            n_agents = len(agents_batch[0]) if batch_size > 0 else 0
            rewards = np.zeros((batch_size, n_agents), dtype=np.float32)
            for b in range(batch_size):
                for a in range(n_agents):
                    rewards[b, a] = super().reward(agents_batch[b][a], world_batch[b])
            return rewards
    
    def initialize_agents_for_reward(self, agents_batch, world_batch):
        """
        批量初始化智能体的奖励相关属性
        这是向量化优化的关键部分
        """
        batch_size = len(agents_batch)
        n_agents = len(agents_batch[0]) if batch_size > 0 else 0
        
        for b in range(batch_size):
            world = world_batch[b]
            for a in range(n_agents):
                agent = agents_batch[b][a]
                
                # 🚨 关键修复：只在回合开始时重置total_penetration_count，而不是每次调用都重置
                # 原因：如果每次调用都重置，会导致回合中间的计数被清零，无法正确记录碰撞
                # 修复：只在initialized_for_reward为False时（回合开始时）重置计数
                if not hasattr(agent, 'debug_info'):
                    agent.debug_info = {}
                if not isinstance(agent.debug_info, dict):
                    agent.debug_info = {}
                
                # 初始化智能体属性
                is_new_episode = not hasattr(agent, 'initialized_for_reward') or not agent.initialized_for_reward
                if is_new_episode:
                    # 🚨 关键修复：只在回合开始时重置计数，而不是每次调用都重置
                    agent.debug_info['total_penetration_count'] = 0
                    # 🚨 关键修复：重置防重复计数标志，确保每回合开始时重置
                    agent._last_collision_counted_step = -1
                
                if is_new_episode:
                    # 获取目标位置
                    if hasattr(agent, 'goal_a') and agent.goal_a.state.p_pos is not None:
                        goal_pos = agent.goal_a.state.p_pos
                        dist_to_goal = np.linalg.norm(agent.state.p_pos - goal_pos)
                        
                        # 初始化所有需要的属性
                        agent.last_goal_dist = dist_to_goal
                        agent.stationary_count = 0
                        agent.last_position = agent.state.p_pos.copy()
                        agent.last_velocity = np.zeros(3)
                        agent.visited_cells = set()
                        agent.initialized_for_reward = True
                        agent.start_position = agent.state.p_pos.copy()
                        
                        # 计算初始距离和方向（按每智能体真实目标）
                        agent.initial_distance_to_goal = dist_to_goal
                        agent.start_to_goal_dir = None
                        try:
                            if hasattr(agent, 'goal_a') and agent.goal_a is not None and agent.goal_a.state.p_pos is not None:
                                _g = agent.goal_a.state.p_pos
                            else:
                                _g = self.goal_pos if hasattr(self, 'goal_pos') else None
                            if _g is not None:
                                start_to_goal = _g - agent.state.p_pos
                                _d = np.linalg.norm(start_to_goal)
                                if _d > 1e-6:
                                    agent.start_to_goal_dir = start_to_goal / _d
                        except Exception:
                            pass
    
    def get_reward_breakdown(self, agents_batch, world_batch):
        """
        获取奖励分项详情（用于调试和分析）
        
        Returns:
            Dict[str, np.ndarray]: 各分项奖励的详细数据
        """
        if self.vectorized_calculator is not None:
            return self.vectorized_calculator.get_reward_breakdown(agents_batch, world_batch)
        else:
            # 返回空的breakdown
            return {}
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = self._performance_stats.copy()
        
        # 计算向量化效率
        if stats['total_calls'] > 0:
            batch_ratio = stats['batch_calls'] / stats['total_calls']
            stats['vectorization_efficiency'] = batch_ratio
            stats['potential_speedup'] = 1.0 + batch_ratio * 2.0  # 估算加速比
        
        return stats

    def is_done(self, agent, world):
        """统一终止条件：越界或非目标附近的地形落地/穿透即终止"""
        try:
            pos = getattr(getattr(agent, 'state', None), 'p_pos', None)
            if pos is None or len(pos) < 3:
                return False
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            # 越界检测（可选）
            if hasattr(world, 'is_within_bounds') and callable(world.is_within_bounds):
                try:
                    if not world.is_within_bounds(pos):
                        return True
                except Exception:
                    pass
            # 地形落地/穿透且不在目标附近
            terrain_h = None
            try:
                if hasattr(self, 'get_terrain_height'):
                    terrain_h = float(self.get_terrain_height(x, y))
            except Exception:
                terrain_h = None
            if terrain_h is not None:
                # 与目标距离
                gx = getattr(self, 'goal_pos', None)
                dist_to_goal = 1e9
                if gx is not None:
                    gx, gy, gz = float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2])
                    dist_to_goal = ((x-gx)**2 + (y-gy)**2 + (z-gz)**2) ** 0.5
                eps = 0.03
                thr_succ = float(getattr(self, 'success_distance_threshold', 2.0))
                if z <= terrain_h + eps and dist_to_goal > thr_succ:
                    return True
                # 提前碰撞：与地物最近距离阈值（使用父类weighted逻辑近似）
                try:
                    dmin = None
                    if hasattr(world, 'landmarks'):
                        for landmark in world.landmarks:
                            lp = getattr(getattr(landmark, 'state', None), 'p_pos', None)
                            if lp is None:
                                continue
                            r = float(getattr(landmark, 'size', 0.0)) + float(getattr(agent, 'size', 0.0))
                            d = float(np.linalg.norm(np.array([x,y,z], dtype=np.float32) - lp) - r)
                            dmin = d if dmin is None else min(dmin, d)
                    thr_coll = float(getattr(self, 'collision_distance_threshold', 0.5))
                    if dmin is not None and dmin <= thr_coll and dist_to_goal > thr_succ:
                        return True
                except Exception:
                    pass
            # 注意：agent.collide 只是表示是否参与碰撞检测，不是实际碰撞状态
            # 这里不应该基于 agent.collide 来判断是否终止，因为它始终为 True
            # 实际的碰撞检测应该基于物理约束（地形穿透、越界等）
        except Exception:
            return False
        return False
    
    def clear_caches(self):
        """清理所有缓存以释放内存"""
        if self.vectorized_calculator is not None:
            self.vectorized_calculator.clear_cache()
        self._batch_cache = {
            'agents_batch': None,
            'world_batch': None,
            'last_batch_size': 0,
            'last_n_agents': 0
        }
    
    def update_reward_weights(self, new_weights):
        """动态更新奖励权重"""
        self.reward_weights.update(new_weights)
        
        # 准备向量化计算器的参数，包括高度奖励配置
        vectorized_kwargs = {}
        if hasattr(self, 'height_reward_enabled'):
            vectorized_kwargs['height_reward_enabled'] = self.height_reward_enabled
        if hasattr(self, 'height_ideal_min'):
            vectorized_kwargs['height_ideal_min'] = self.height_ideal_min
        if hasattr(self, 'height_ideal_max'):
            vectorized_kwargs['height_ideal_max'] = self.height_ideal_max
        
        # 更新向量化计算器的权重
        if VectorizedRewardCalculator is not None:
            self.vectorized_calculator = VectorizedRewardCalculator(
                reward_weights=self.reward_weights,
                max_reward=self.max_reward,
                min_reward=self.min_reward,
                **vectorized_kwargs
            )
        else:
            print("Warning: VectorizedRewardCalculator not available, using base implementation")
            self.vectorized_calculator = None
    
    def benchmark_reward_calculation(self, agents_batch, world_batch, iterations=100):
        """
        基准测试奖励计算性能
        
        Args:
            agents_batch: 智能体批次
            world_batch: 世界批次
            iterations: 测试迭代次数
            
        Returns:
            Dict: 性能测试结果
        """
        import time
        
        # 测试向量化版本
        start_time = time.time()
        for _ in range(iterations):
            vectorized_rewards = self.batch_reward(agents_batch, world_batch)
        vectorized_time = time.time() - start_time
        
        # 测试原始版本（单个计算）
        start_time = time.time()
        for _ in range(iterations):
            original_rewards = []
            for b in range(len(agents_batch)):
                env_rewards = []
                for a in range(len(agents_batch[b])):
                    reward = super().reward(agents_batch[b][a], world_batch[b])
                    env_rewards.append(reward)
                original_rewards.append(env_rewards)
        original_time = time.time() - start_time
        
        # 计算性能提升
        speedup = original_time / vectorized_time if vectorized_time > 0 else 0
        
        return {
            'vectorized_time': vectorized_time,
            'original_time': original_time,
            'speedup': speedup,
            'iterations': iterations,
            'batch_size': len(agents_batch),
            'n_agents': len(agents_batch[0]) if len(agents_batch) > 0 else 0
        }


# 创建场景实例的独立函数
def make_world():
    scenario = Scenario()
    return scenario.make_world()

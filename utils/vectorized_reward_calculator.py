#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
numpy向量化奖励计算器
针对并行环境进行性能优化，使用批量计算提升效率
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Any


def _bilinear_interpolate_terrain(terrain: np.ndarray, positions: np.ndarray, terrain_heights: np.ndarray, map_size: int):
    """
    双线性插值获取地形高度（向量化实现）
    用于修复降采样导致的穿透检测误判问题
    
    🚨 关键修复：确保地形高度计算的准确性
    - 使用双线性插值而不是最近邻，提高精度
    - 处理边界情况（越界、NaN等）
    - 确保索引不越界
    
    Args:
        terrain: 地形高度图 (H, W)
        positions: 智能体位置 (N, 2) 或 (N, 3)，只使用前2维
        terrain_heights: 输出数组 (N,)，用于存储插值后的高度
        map_size: 地图尺寸
    """
    if terrain is None or positions is None or len(positions) == 0:
        return
    
    # 确保positions是2D数组
    if positions.ndim == 1:
        positions_2d = positions[:2].reshape(1, -1)
    else:
        positions_2d = positions[:, :2]
    
    num_pos = len(positions_2d)
    
    # 🚨 关键修复：确保地形尺寸与map_size一致
    # 如果地形尺寸与map_size不一致（降采样情况），需要缩放坐标
    terrain_h, terrain_w = terrain.shape[0], terrain.shape[1]
    if terrain_h != map_size or terrain_w != map_size:
        # 地形已降采样，需要缩放坐标
        scale_x = float(terrain_w) / float(map_size)
        scale_y = float(terrain_h) / float(map_size)
        x_coords = np.clip(positions_2d[:, 0] * scale_x, 0.0, float(terrain_w - 1))
        y_coords = np.clip(positions_2d[:, 1] * scale_y, 0.0, float(terrain_h - 1))
    else:
        # 地形尺寸与map_size一致，直接使用
        x_coords = np.clip(positions_2d[:, 0], 0.0, float(map_size - 1))
        y_coords = np.clip(positions_2d[:, 1], 0.0, float(map_size - 1))
    
    # 计算四个角点的索引
    x_low = np.floor(x_coords).astype(np.int32)
    y_low = np.floor(y_coords).astype(np.int32)
    x_high = np.minimum(x_low + 1, terrain_w - 1)
    y_high = np.minimum(y_low + 1, terrain_h - 1)
    
    # 🚨 关键修复：确保索引不越界
    x_low = np.clip(x_low, 0, terrain_w - 1)
    y_low = np.clip(y_low, 0, terrain_h - 1)
    x_high = np.clip(x_high, 0, terrain_w - 1)
    y_high = np.clip(y_high, 0, terrain_h - 1)
    
    # 计算插值权重
    x_weight = x_coords - x_low.astype(np.float32)
    y_weight = y_coords - y_low.astype(np.float32)
    
    # 获取四个角点的高度值
    h00 = terrain[y_low, x_low]  # 左下
    h10 = terrain[y_low, x_high]  # 右下
    h01 = terrain[y_high, x_low]  # 左上
    h11 = terrain[y_high, x_high]  # 右上
    
    # 🚨 关键修复：处理NaN和Inf值
    # 如果某个角点的高度无效，使用其他有效值或默认值
    valid_mask = np.isfinite(h00) & np.isfinite(h10) & np.isfinite(h01) & np.isfinite(h11)
    if not np.all(valid_mask):
        # 对于无效值，使用最近邻值
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            # 使用最近的有效值（优先使用h00）
            h00[invalid_mask] = np.where(np.isfinite(h00[invalid_mask]), h00[invalid_mask], 
                                        np.where(np.isfinite(h10[invalid_mask]), h10[invalid_mask],
                                                np.where(np.isfinite(h01[invalid_mask]), h01[invalid_mask], h11[invalid_mask])))
            h10[invalid_mask] = np.where(np.isfinite(h10[invalid_mask]), h10[invalid_mask], h00[invalid_mask])
            h01[invalid_mask] = np.where(np.isfinite(h01[invalid_mask]), h01[invalid_mask], h00[invalid_mask])
            h11[invalid_mask] = np.where(np.isfinite(h11[invalid_mask]), h11[invalid_mask], h00[invalid_mask])
    
    # 双线性插值
    h0 = h00 * (1.0 - x_weight) + h10 * x_weight  # 下边插值
    h1 = h01 * (1.0 - x_weight) + h11 * x_weight  # 上边插值
    terrain_heights[:] = h0 * (1.0 - y_weight) + h1 * y_weight  # 最终插值
    
    # 🚨 关键修复：确保输出值有效
    invalid_output = ~np.isfinite(terrain_heights)
    if np.any(invalid_output):
        # 对于无效输出，使用最近邻值
        x_nearest = np.clip(x_low[invalid_output], 0, terrain_w - 1)
        y_nearest = np.clip(y_low[invalid_output], 0, terrain_h - 1)
        terrain_heights[invalid_output] = terrain[y_nearest, x_nearest]


class VectorizedRewardCalculator:
    """numpy向量化奖励计算器"""
    
    def __init__(self, reward_weights: Dict[str, float], max_reward: float = 800.0, min_reward: float = -800.0,
                 success_reward_value: float = 150.0, success_distance_threshold: float = 2.0,
                 collision_penalty_value: float = 30.0, collision_distance_threshold: float = 0.5,
                 no_collision_reward_value: float = 0.0, global_reward_mode: str = 'success_rate', shaping_gamma: float = 0.95, **kwargs):
        """初始化向量化奖励计算器
        
        Args:
            reward_weights: 奖励权重字典
            max_reward: 最大奖励值
            min_reward: 最小奖励值
        """
        # 将权重转换为numpy数组，按固定顺序排列（14通道，必须与分项索引严格一致）
        self.reward_weights = np.array([
            reward_weights['distance'],          # 0
            reward_weights['exploration'],       # 1
            reward_weights['stationary'],        # 2
            reward_weights['direction'],         # 3
            reward_weights['deviation'],         # 4
            reward_weights['start_area'],        # 5
            reward_weights['approach'],          # 6
            reward_weights['energy'],            # 7
            reward_weights['height'],            # 8
            reward_weights.get('success', 0.0),  # 9
            reward_weights.get('collision', 0.0),# 10
            reward_weights.get('global', 0.0),   # 11
            reward_weights.get('shaping', 0.0),  # 12
            reward_weights.get('clearance', 0.0) # 13
        ], dtype=np.float32)
        self._base_reward_weights = self.reward_weights.copy()
        
        self.max_reward = max_reward
        self.min_reward = min_reward
        
        # 预分配数组缓存和性能优化
        self._cache = {}
        self._obstacle_cache = {}  # 障碍物数据缓存
        self._terrain_cache = {}   # 地形数据缓存
        self._goal_cache = {}      # 目标数据缓存
        
        # 其他参数（预转换为合适类型避免运行时转换）
        self.success_reward_value = np.float32(success_reward_value)
        self.success_distance_threshold = np.float32(success_distance_threshold)
        self.collision_penalty_value = np.float32(collision_penalty_value)
        self.collision_distance_threshold = np.float32(collision_distance_threshold)
        self.no_collision_reward_value = np.float32(no_collision_reward_value)
        self.global_reward_mode = str(global_reward_mode)
        self.shaping_gamma = np.float32(shaping_gamma)
        # 新增：穿透深度系数与探索严格模式（环境变量可覆盖）以及轨迹平滑权重
        try:
            import os
            self.penetration_alpha = np.float32(float(os.getenv('PENETRATION_ALPHA', '0.5')))
            self.expl_reward_strict = str(os.getenv('EXPL_REWARD_STRICT', '0')) not in ('0', 'false', 'False')
            # 新增：地形穿透基础惩罚值，默认等于碰撞惩罚
            terrain_base = os.getenv('PENETRATION_BASE_PENALTY', '')
            if terrain_base is not None and str(terrain_base).strip() != '':
                self.terrain_penalty_value = np.float32(float(terrain_base))
            else:
                self.terrain_penalty_value = self.collision_penalty_value
            # 新增：地形接触阈值（触地即罚/终止的高度缓冲）
            # 🚨 调试：默认值从2.5改为10.0，用于诊断碰撞计数问题
            # 原因：大幅提高阈值，让碰撞检测更频繁触发，判断是计数过多还是没有记录上
            # 大幅提高阈值以确保能检测到所有碰撞和接近地形的接触
            self.terrain_contact_eps = np.float32(float(os.getenv('TERRAIN_CONTACT_EPS', '10.0')))
            # 轨迹平滑奖励权重：优先使用kwargs，回退到环境变量TURN_SMOOTH_WEIGHT
            self.turn_smooth_weight = np.float32(
                float(kwargs.get('turn_smooth_weight', os.getenv('TURN_SMOOTH_WEIGHT', '0.0')))
            )
        except Exception:
            self.penetration_alpha = np.float32(0.5)
            self.expl_reward_strict = False
            self.terrain_penalty_value = self.collision_penalty_value
            # 🚨 关键修复：异常处理中的默认值也从1.5改为2.5，与run_optimized.sh保持一致
            self.terrain_contact_eps = np.float32(2.5)
            self.turn_smooth_weight = np.float32(0.0)
        
        # 性能优化开关
        self.debug_mode = False  # 关闭调试输出
        self.use_fast_path = True  # 启用快速路径
        self._printed_once = False  # 首回合一次性打印关键配置
        
        # 奖励分项名称（用于调试，14通道，对应索引0-13）
        self.reward_names = [
            'distance',    # 0
            'exploration', # 1
            'stationary',  # 2
            'direction',   # 3
            'deviation',   # 4
            'start_area',  # 5
            'approach',    # 6
            'energy',      # 7
            'height',      # 8
            'success',     # 9
            'collision',   # 10
            'global',      # 11
            'shaping',     # 12
            'clearance'    # 13
        ]
        
        # 奖励多样性检测
        self.reward_history = []
        self.diversity_threshold = 0.95  # 如果连续奖励相似度超过95%，认为缺乏多样性
        self.diversity_window = 10  # 检测窗口大小
        self.last_warning_episode = 0  # 上次警告的回合数
        self.warning_cooldown = 20  # 警告冷却期（回合数）
        
        # 高度奖励可配置开关与范围（支持kwargs与环境变量）
        try:
            import os
            _env_enabled = os.getenv('HEIGHT_REWARD_ENABLED', '1')
            self.height_reward_enabled = kwargs.get('height_reward_enabled', _env_enabled not in ('0', 'false', 'False'))
            self.height_ideal_min = np.float32(kwargs.get('height_ideal_min', float(os.getenv('HEIGHT_IDEAL_MIN', '2.0'))))
            self.height_ideal_max = np.float32(kwargs.get('height_ideal_max', float(os.getenv('HEIGHT_IDEAL_MAX', '5.0'))))
            # 安全保护：保证最小值不大于最大值
            if self.height_ideal_min > self.height_ideal_max:
                self.height_ideal_min, self.height_ideal_max = self.height_ideal_max, self.height_ideal_min
        except Exception:
            # 兜底到默认范围
            self.height_reward_enabled = True
            self.height_ideal_min = np.float32(2.0)
            self.height_ideal_max = np.float32(5.0)
    
    def update_collision_parameters(self, collision_weight: float, collision_penalty_value: float = None):
        """动态更新碰撞权重与惩罚。"""
        try:
            self.reward_weights[10] = np.float32(collision_weight)
        except Exception:
            pass
        if collision_penalty_value is not None:
            try:
                self.collision_penalty_value = np.float32(collision_penalty_value)
                self.terrain_penalty_value = np.float32(collision_penalty_value)
            except Exception:
                pass

    def _update_caches(self, world: Any, scenario: Any) -> Dict[str, Any]:
        """预处理和缓存环境数据，减少重复计算"""
        cache_key = id(world)
        
        # 检查缓存是否有效
        if cache_key in self._cache and self.use_fast_path:
            return self._cache[cache_key]
        
        cached_data = {
            'terrain': None,
            'obstacles_centers': None,
            'obstacles_radii': None,
            'goal_positions': None,
            'goal_radii': None,
            'map_size': 200
        }
        
        try:
            # 缓存地形数据
            terrain = None
            if hasattr(scenario, 'terrain') and scenario.terrain is not None:
                terrain = scenario.terrain
            elif hasattr(world, 'terrain') and world.terrain is not None:
                terrain = world.terrain
            
            if terrain is not None:
                # 🚨 关键优化：如果地形是降采样的，预先插值生成高分辨率地形并缓存
                # 这样后续计算时可以直接使用索引，不需要每次都调用插值函数
                map_size = int(getattr(scenario, 'map_size', terrain.shape[0]))
                terrain_downsampled = getattr(scenario, 'terrain_downsampled', False)
                
                if terrain_downsampled and hasattr(scenario, 'get_terrain_height'):
                    # 地形已降采样，预先插值生成高分辨率地形
                    # 生成 map_size × map_size 的高分辨率地形数组
                    high_res_terrain = np.zeros((map_size, map_size), dtype=np.float32)
                    try:
                        # 向量化插值：为每个坐标点计算地形高度
                        for y in range(map_size):
                            for x in range(map_size):
                                try:
                                    high_res_terrain[y, x] = scenario.get_terrain_height(float(x), float(y))
                                except Exception:
                                    # 如果插值失败，使用降采样地形的最近邻值
                                    terrain_y = int(y * terrain.shape[0] / map_size)
                                    terrain_x = int(x * terrain.shape[1] / map_size)
                                    terrain_y = min(terrain_y, terrain.shape[0] - 1)
                                    terrain_x = min(terrain_x, terrain.shape[1] - 1)
                                    high_res_terrain[y, x] = terrain[terrain_y, terrain_x]
                        cached_data['terrain'] = high_res_terrain.astype(np.float32)
                        cached_data['map_size'] = map_size
                        cached_data['terrain_interpolated'] = True  # 标记已插值
                    except Exception:
                        # 如果插值失败，回退到原始地形
                        cached_data['terrain'] = terrain.astype(np.float32)
                        cached_data['map_size'] = terrain.shape[0]
                        cached_data['terrain_interpolated'] = False
                else:
                    # 地形未降采样，直接使用原始地形
                    cached_data['terrain'] = terrain.astype(np.float32)
                    cached_data['map_size'] = terrain.shape[0]
                    cached_data['terrain_interpolated'] = False
            
            # 缓存障碍物数据
            if hasattr(scenario, 'obstacles') and scenario.obstacles:
                centers = []
                radii = []
                for obs in scenario.obstacles:
                    if isinstance(obs, dict):
                        center = None
                        if 'center' in obs and obs['center'] is not None:
                            center = obs['center'][:3]  # 取x,y,z
                        elif 'pos' in obs and obs['pos'] is not None:
                            center = obs['pos'][:3]
                        
                        if center is not None:
                            centers.append(center)
                            radius = obs.get('radius', obs.get('r', 1.0))
                            radii.append(radius)
                
                if centers:
                    cached_data['obstacles_centers'] = np.asarray(centers, dtype=np.float32)
                    cached_data['obstacles_radii'] = np.asarray(radii, dtype=np.float32)
            
            # 缓存目标数据
            if hasattr(world, 'agents') and world.agents:
                goal_positions = []
                goal_radii = []
                for agent in world.agents:
                    if hasattr(agent, 'goal_a') and agent.goal_a is not None:
                        if hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                            goal_positions.append(agent.goal_a.state.p_pos[:3])
                            # 获取目标半径
                            goal_radius = getattr(agent.goal_a, 'size', getattr(agent.goal_a, 'radius', 2.0))
                            goal_radii.append(goal_radius)
                
                if goal_positions:
                    cached_data['goal_positions'] = np.asarray(goal_positions, dtype=np.float32)
                    cached_data['goal_radii'] = np.asarray(goal_radii, dtype=np.float32)
        
        except Exception:
            # 静默处理异常，使用默认值
            pass
        
        # 更新缓存
        self._cache[cache_key] = cached_data
        return cached_data
    
    def clear_caches(self):
        """清理缓存，释放内存"""
        self._cache.clear()
        self._obstacle_cache.clear()
        self._terrain_cache.clear()
        self._goal_cache.clear()
    
    def set_debug_mode(self, enabled: bool):
        """设置调试模式"""
        self.debug_mode = enabled
    
    def set_fast_path(self, enabled: bool):
        """设置快速路径优化"""
        self.use_fast_path = enabled
    
    def batch_calculate_rewards(
        self,
        agents_batch: List[List[Any]],
        world_batch: List[Any],
        scenario_batch: List[Any] = None,
        positions_batch: np.ndarray = None,
        prev_positions_batch: np.ndarray = None,
        actions_batch: np.ndarray = None,
        start_positions_batch: np.ndarray = None,
        episode: int = -1
    ) -> np.ndarray:
        """
        批量计算奖励 - numpy向量化版本
        
        Args:
            agents_batch: (batch_size, n_agents) 智能体批次
            world_batch: (batch_size,) 世界批次
            scenario_batch: (batch_size,) 场景批次
            positions_batch: (batch_size, n_agents, 3) 智能体位置批次
            prev_positions_batch: (batch_size, n_agents, 3) 智能体上一时刻位置批次
            actions_batch: (batch_size, n_agents, 3) 智能体动作批次
            start_positions_batch: (batch_size, n_agents, 3) 智能体起始位置批次
            episode: 当前回合数
            
        Returns:
            rewards: (batch_size, n_agents) 奖励数组
        """
        try:
            batch_size = len(agents_batch)
            n_agents = len(agents_batch[0]) if batch_size > 0 else 0
            
            if batch_size == 0 or n_agents == 0:
                return np.zeros((batch_size, n_agents), dtype=np.float32)
            
            # 预处理缓存数据（性能优化）
            cached_data_batch = []
            for b in range(batch_size):
                world = world_batch[b] if world_batch else None
                scenario = scenario_batch[b] if scenario_batch and b < len(scenario_batch) else None
                cached_data = self._update_caches(world, scenario) if self.use_fast_path else None
                cached_data_batch.append(cached_data)
            
            # 预分配数组
            arrays = self._get_preallocated_arrays(batch_size, n_agents)
            
            # 向后兼容：若未提供批次数据，则从agents提取，或用保守默认值
            if positions_batch is None or prev_positions_batch is None or actions_batch is None or start_positions_batch is None:
                # 提取当前位置与速度
                self._extract_batch_states(agents_batch, arrays)
            
                # prev_positions：优先使用agent.last_position，否则用当前位置兜底
                prev_positions = np.zeros_like(arrays['positions'])
                for b in range(batch_size):
                    for a in range(n_agents):
                        ag = agents_batch[b][a]
                        if hasattr(ag, 'last_position') and isinstance(ag.last_position, np.ndarray):
                            prev_positions[b, a] = ag.last_position
                        else:
                            prev_positions[b, a] = arrays['positions'][b, a]
                arrays['prev_positions'][:] = prev_positions
                
                # actions：未知时置零（能量分项内部做保护）
                arrays['actions'][:] = 0.0
                
                # start_positions：优先使用agent.start_position，否则用当前位置兜底
                starts = np.zeros_like(arrays['positions'])
                for b in range(batch_size):
                    for a in range(n_agents):
                        ag = agents_batch[b][a]
                        if hasattr(ag, 'start_position') and isinstance(ag.start_position, np.ndarray):
                            starts[b, a] = ag.start_position
                        else:
                            starts[b, a] = arrays['positions'][b, a]
                arrays['start_positions'][:] = starts
            else:
                arrays['positions'][:] = positions_batch
                arrays['prev_positions'][:] = prev_positions_batch
                arrays['actions'][:] = actions_batch
                arrays['start_positions'][:] = start_positions_batch

            # 准备scenario_batch（向后兼容：若未提供，则从world读取）
            if scenario_batch is None:
                scenario_batch = []
                for w in world_batch:
                    scenario_batch.append(getattr(w, 'scenario', None))

            # 核心奖励计算
            try:
                self._calculate_all_rewards_vectorized(agents_batch, world_batch, scenario_batch, arrays, cached_data_batch)
            except Exception as e:
                print(f"批量奖励计算异常: {e}")
                import traceback
                traceback.print_exc()

            # 应用权重并求和（对齐通道数，防御性处理）
            rewards_mat = arrays['rewards']
            weights_vec = self.reward_weights
            try:
                ch = rewards_mat.shape[2]
                wlen = int(weights_vec.shape[0]) if hasattr(weights_vec, 'shape') else len(weights_vec)
                if wlen != ch:
                    if wlen < ch:
                        pad = np.zeros((ch - wlen,), dtype=weights_vec.dtype if hasattr(weights_vec, 'dtype') else np.float32)
                        weights_vec = np.concatenate([weights_vec, pad], axis=0)
                    else:
                        weights_vec = weights_vec[:ch]
            except Exception:
                pass
            total_rewards = np.sum(rewards_mat * weights_vec, axis=2)
            
            # 检查奖励值是否异常 - 只在调试模式和异常时输出
            if self.debug_mode and (np.isnan(total_rewards).any() or np.isinf(total_rewards).any()):
                print(f"\n{'='*80}")
                print(f"🚨 奖励计算中检测到异常值")
                print(f"{'='*80}")
                print(f"📊 奖励形状: {total_rewards.shape}")
                print(f"🔍 异常值统计:")
                print(f"   - NaN: {np.sum(np.isnan(total_rewards))} 个")
                print(f"   - Inf: {np.sum(np.isinf(total_rewards))} 个")
                print(f"📈 奖励值范围: [{np.min(total_rewards):.6f}, {np.max(total_rewards):.6f}]")
                
                # 检查分项奖励是否异常
                for i, reward_name in enumerate(self.reward_names):
                    reward_values = arrays['rewards'][:, :, i]
                    if np.isnan(reward_values).any() or np.isinf(reward_values).any():
                        print(f"⚠️  {reward_name} 奖励异常: NaN={np.sum(np.isnan(reward_values))}, Inf={np.sum(np.isinf(reward_values))}")
                
                print(f"{'='*80}\n")
            
            # 静默修复异常值
            if np.isnan(total_rewards).any() or np.isinf(total_rewards).any():
                total_rewards = np.nan_to_num(total_rewards, nan=0.0, posinf=self.max_reward, neginf=self.min_reward)
            
            # （已禁用）奖励统计调试输出，减少日志噪声
            
            # 首回合打印关键裁剪与权重配置
            if not self._printed_once:
                try:
                    print(f"[VecRew] min/max clip: [{float(self.min_reward):.1f},{float(self.max_reward):.1f}] weights[height,collision,exploration]={self.reward_weights[8]:.2f},{self.reward_weights[10]:.2f},{self.reward_weights[1]:.2f}")
                except Exception:
                    pass
                self._printed_once = True
            # 检测奖励多样性（添加回合数参数）
            self._check_reward_diversity(total_rewards, episode=episode)
            
            # 限制范围
            return np.clip(total_rewards, self.min_reward, self.max_reward)
            
        except Exception as e:
            print(f"批量奖励计算异常: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((batch_size, n_agents), dtype=np.float32)
    
    def _get_preallocated_arrays(self, batch_size: int, n_agents: int) -> Dict[str, np.ndarray]:
        """获取预分配的numpy数组"""
        key = (batch_size, n_agents)
        if key not in self._cache:
            self._cache[key] = {
                'positions': np.zeros((batch_size, n_agents, 3), dtype=np.float32),
                'prev_positions': np.zeros((batch_size, n_agents, 3), dtype=np.float32),
                'actions': np.zeros((batch_size, n_agents, 3), dtype=np.float32),
                'start_positions': np.zeros((batch_size, n_agents, 3), dtype=np.float32),
                'velocities': np.zeros((batch_size, n_agents, 3), dtype=np.float32),
                'goals': np.zeros((batch_size, n_agents, 3), dtype=np.float32),
                'distances': np.zeros((batch_size, n_agents), dtype=np.float32),
                'rewards': np.zeros((batch_size, n_agents, 14), dtype=np.float32),
                'pos_changes': np.zeros((batch_size, n_agents), dtype=np.float32),
                'relative_distances': np.zeros((batch_size, n_agents), dtype=np.float32),
                'speed_efficiency': np.zeros((batch_size, n_agents), dtype=np.float32)
            }
        return self._cache[key]
    
    def _extract_batch_states(self, agents_batch: List[List[Any]], arrays: Dict[str, np.ndarray]):
        """批量提取智能体状态 - 向量化版本"""
        batch_size, n_agents = arrays['positions'].shape[:2]
        
        # 使用numpy向量化操作批量提取状态
        for b in range(batch_size):
            for a in range(n_agents):
                agent = agents_batch[b][a]
                arrays['positions'][b, a] = agent.state.p_pos
                arrays['velocities'][b, a] = agent.state.p_vel
                
                if hasattr(agent, 'goal_a') and agent.goal_a.state.p_pos is not None:
                    arrays['goals'][b, a] = agent.goal_a.state.p_pos
        
        # 向量化计算距离
        arrays['distances'] = np.linalg.norm(
            arrays['positions'] - arrays['goals'], axis=2
        )
    
    def _calculate_all_rewards_vectorized(self, agents_batch: List[List[Any]], world_batch: List[Any], scenario_batch: List[Any], arrays: Dict[str, np.ndarray], cached_data_batch: List[Any] = None):
        """向量化计算所有奖励分项"""
        batch_size, n_agents = arrays['rewards'].shape[:2]
        
        # 重置奖励数组
        arrays['rewards'].fill(0.0)
        
        # 批量计算各项奖励
        for b, (agents, world, scenario) in enumerate(zip(agents_batch, world_batch, scenario_batch)):
            # 取出该批次的缓存数据
            cached_data = None
            if cached_data_batch is not None and b < len(cached_data_batch):
                cached_data = cached_data_batch[b]
            for a, agent in enumerate(agents):
                # 计算各项奖励（转换为标量形式以匹配现有函数签名）
                pos = arrays['positions'][b, a]
                prev_pos = arrays['prev_positions'][b, a] 
                start_pos = arrays['start_positions'][b, a]
                action = arrays['actions'][b, a]
                
                # 1. 距离奖励
                dist_reward = self._distance_reward_vectorized(agent, scenario, pos.reshape(1, -1), start_pos.reshape(1, -1))
                arrays['rewards'][b, a, 0] = dist_reward[0] if isinstance(dist_reward, np.ndarray) else dist_reward
                
                # 2. 探索奖励  
                expl_reward = self._exploration_reward_vectorized(agent, scenario, pos.reshape(1, -1))
                arrays['rewards'][b, a, 1] = expl_reward[0] if isinstance(expl_reward, np.ndarray) else expl_reward
                
                # 3. 停滞惩罚
                stat_penalty = self._stationary_penalty_vectorized(agent, pos.reshape(1, -1), prev_pos.reshape(1, -1))
                arrays['rewards'][b, a, 2] = stat_penalty[0] if isinstance(stat_penalty, np.ndarray) else stat_penalty
                
                # 4. 方向奖励
                dir_reward = self._direction_reward_vectorized(agent, pos.reshape(1, -1), prev_pos.reshape(1, -1))
                arrays['rewards'][b, a, 3] = dir_reward[0] if isinstance(dir_reward, np.ndarray) else dir_reward
                
                # 5. 偏离惩罚
                dev_penalty = self._deviation_reward_vectorized(agent, scenario, start_pos.reshape(1, -1), pos.reshape(1, -1))
                arrays['rewards'][b, a, 4] = dev_penalty[0] if isinstance(dev_penalty, np.ndarray) else dev_penalty
                
                # 6. 起始区奖励
                start_reward = self._start_area_reward_vectorized(agent, scenario, pos.reshape(1, -1), start_pos.reshape(1, -1))
                arrays['rewards'][b, a, 5] = start_reward[0] if isinstance(start_reward, np.ndarray) else start_reward
                
                # 7. 接近奖励
                appr_reward = self._approach_reward_vectorized(agent, scenario, pos.reshape(1, -1), prev_pos.reshape(1, -1))
                arrays['rewards'][b, a, 6] = appr_reward[0] if isinstance(appr_reward, np.ndarray) else appr_reward
                
                # 8. 能量消耗惩罚
                energy_reward = self._energy_reward_vectorized(action.reshape(1, -1))
                arrays['rewards'][b, a, 7] = energy_reward[0] if isinstance(energy_reward, np.ndarray) else energy_reward
                
                # 9. 高度奖励（传入上一时刻位置和动作，用于判断向上飞行）
                height_reward = self._height_reward_vectorized(
                    agent, scenario, pos.reshape(1, -1), 
                    prev_pos.reshape(1, -1) if prev_pos is not None else None,
                    action.reshape(1, -1) if action is not None else None
                )
                arrays['rewards'][b, a, 8] = height_reward[0] if isinstance(height_reward, np.ndarray) else height_reward
                
                # 10. 成功奖励（使用缓存数据）
                # 🚨 关键修复：传入循环索引a作为agent_idx，避免对象身份比较失败
                success_reward = self._success_reward_vectorized(agent, scenario, pos.reshape(1, -1), cached_data, agent_idx=a)
                arrays['rewards'][b, a, 9] = success_reward[0] if isinstance(success_reward, np.ndarray) else success_reward
                
                # 11. 碰撞惩罚（使用循环索引a作为agent索引，避免依赖agent.index，使用缓存数据）
                collision_penalty = self._collision_penalty_vectorized(a, world, scenario, pos.reshape(1, -1), cached_data)
                arrays['rewards'][b, a, 10] = collision_penalty[0] if isinstance(collision_penalty, np.ndarray) else collision_penalty
                
                # 12. 全局奖励（临时设为0，将在批处理结束后统一计算）
                arrays['rewards'][b, a, 11] = 0.0
                
                # 13. 塑形奖励（按每智能体真实目标计算距离）
                try:
                    if hasattr(agent, 'goal_a') and agent.goal_a is not None and agent.goal_a.state.p_pos is not None:
                        _g = agent.goal_a.state.p_pos
                    else:
                        _g = scenario.goal_pos if hasattr(scenario, 'goal_pos') else None
                    dist_to_goal = float(np.linalg.norm(pos - _g)) if _g is not None else 0.0
                except Exception:
                    dist_to_goal = 0.0
                arrays['rewards'][b, a, 12] = self._potential_shaping_vectorized(agent, dist_to_goal)
                
                # 14. 间隙奖励
                clear_reward = self._clearance_reward_vectorized(agent, world, pos.reshape(1, -1))
                arrays['rewards'][b, a, 13] = clear_reward[0] if isinstance(clear_reward, np.ndarray) else clear_reward


            # 批内全局奖励（同一b共享）- 修复：统一计算全局奖励
            global_val = self._global_reward_vectorized(agents_batch[b])
            arrays['rewards'][b, :, 11] = global_val
            
            # 🚨 关键修复：无碰撞奖励是全队奖励，应该给所有智能体（而不是只给到达目标的智能体）
            # 在所有智能体计算完成后，统一给所有智能体加上无碰撞奖励（类似全局奖励的处理方式）
            try:
                world = world_batch[b]
                if world is not None and hasattr(world, '_team_no_collision_reward'):
                    team_no_collision_reward = getattr(world, '_team_no_collision_reward', 0.0)
                    if team_no_collision_reward > 0.0:
                        # 给所有智能体都加上无碰撞奖励（全队共享）
                        arrays['rewards'][b, :, 9] = arrays['rewards'][b, :, 9] + team_no_collision_reward
                        # 重置标志，避免重复添加
                        world._team_no_collision_reward = 0.0
            except Exception:
                pass
            
            # 全局奖励调试输出已删除
    
    def _distance_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray, start_positions: np.ndarray) -> np.ndarray:
        """向量化距离奖励计算（按每智能体目标对齐环境实现）
        - 目标优先使用 agent.goal_a.state.p_pos；若不可用则回退 scenario.goal_pos。
        - 与场景reward保持一致：基于相对初始距离的比例 1 - (d/denom)。
        """
        rewards = np.zeros(len(positions), dtype=np.float32)
        try:
            # 目标位置：优先每智能体目标
            goal_pos = None
            try:
                if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                    goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
            except Exception:
                goal_pos = None
            if goal_pos is None and scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                goal_pos = np.asarray(scenario.goal_pos, dtype=np.float32)
            if goal_pos is None:
                return rewards

            start_pos = np.asarray(start_positions[0], dtype=np.float32)
            initial_dist = float(np.linalg.norm(start_pos - goal_pos))
            current_dist = np.linalg.norm(positions - goal_pos, axis=-1)

            # 稳健归一化
            denom = max(initial_dist, 1.0)
            ratio = np.clip(current_dist / denom, 0.0, 2.0)
            rewards = 1.0 - ratio
            return rewards * 10.0
        except Exception:
            return np.zeros(len(positions), dtype=np.float32)
    
    def _exploration_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray) -> np.ndarray:
        """向量化探索奖励计算 - 增强版"""
        rewards = np.zeros(len(positions), dtype=np.float32)
        try:
            if scenario is None or not hasattr(scenario, 'exploration_grid') or not hasattr(scenario, 'grid_cell_size'):
                return rewards
            cell_size = scenario.grid_cell_size
            # 初始化访问状态
            if not hasattr(agent, 'visited_cells'):
                agent.visited_cells = set()
            if not hasattr(agent, 'cell_visit_counts'):
                agent.cell_visit_counts = {}
            if not hasattr(agent, 'random_exploration_counter'):
                agent.random_exploration_counter = 0
            exploration_reward = np.zeros(len(positions), dtype=np.float32)
            for i, pos in enumerate(positions):
                current_cell = tuple((pos / cell_size).astype(int))
                if current_cell not in agent.visited_cells:
                    agent.visited_cells.add(current_cell)
                    exploration_reward[i] += (1.0 if self.expl_reward_strict else 5.0)
                visit_count = agent.cell_visit_counts.get(current_cell, 0) + 1
                agent.cell_visit_counts[current_cell] = visit_count
                if not self.expl_reward_strict and visit_count <= 3:
                    exploration_reward[i] += (4 - visit_count) * 1.0
                agent.random_exploration_counter += 1
                if (not self.expl_reward_strict) and agent.random_exploration_counter % 50 == 0:
                    exploration_reward[i] += float(np.random.uniform(0.5, 2.0))
            return exploration_reward
        except Exception:
            return np.zeros(len(positions), dtype=np.float32)
    
    def _stationary_penalty_vectorized(self, agent: Any, position: np.ndarray, prev_position: np.ndarray) -> np.ndarray:
        """向量化停滞惩罚计算（已修改以支持悬停和动态惩罚）"""
        rewards = np.zeros(len(position), dtype=np.float32)
        
        # --- 新增豁免逻辑 ---
        # 如果在目标成功圈内，则不施加停滞惩罚
        dist_to_goal = np.linalg.norm(position - agent.goal_a.state.p_pos, axis=-1)
        if dist_to_goal <= self.success_distance_threshold:
            return rewards

        # --- 原有停滞惩罚逻辑 ---
        # 安全检查：如果last_position不存在则初始化
        if not hasattr(agent, 'last_position'):
            agent.last_position = position.copy()
            return rewards
        
        pos_change = np.linalg.norm(position - agent.last_position, axis=-1)
        agent.last_position = position.copy()
        
        # 安全检查：如果stationary_count不存在则初始化
        if not hasattr(agent, 'stationary_count'):
            agent.stationary_count = 0
        
        is_stationary = pos_change < 0.005
        if is_stationary:
            agent.stationary_count += 1
        else:
            agent.stationary_count = 0
        
        if agent.stationary_count > 3:
            # 安全检查：如果start_position不存在则初始化
            if not hasattr(agent, 'start_position'):
                agent.start_position = position.copy()
            
            dist_to_start = np.linalg.norm(position - agent.start_position, axis=-1)
            dist_to_goal = dist_to_goal
            
            # 动态惩罚：根据停滞时间增加惩罚强度
            penalty_multiplier = min(agent.stationary_count / 10.0, 3.0)  # 最大3倍惩罚
            
            if dist_to_start < 15:
                rewards -= 10.0 * penalty_multiplier
            elif dist_to_goal < 5:
                rewards -= 0.5 * penalty_multiplier
            else:
                rewards -= 5.0 * penalty_multiplier
        
        return rewards
    
    def _direction_reward_vectorized(self, agent: Any, position: np.ndarray, prev_position: np.ndarray) -> np.ndarray:
        """方向一致性奖励 + 高度平滑奖励
        规则：
          - 基础奖励 = dot(vel_dir, goal_dir) + speed_bonus（仅在速度>阈值时计算）
          - 高度平滑奖励：奖励高度（Z坐标）的平滑变化，避免突然的高度跳跃
          - 高度平滑度 = exp(-高度变化率 * scale)，高度变化越小，奖励越高（接近1）
          - 对所有位置都计算高度平滑奖励，无速度阈值限制
        """
        rewards = np.zeros(len(position), dtype=np.float32)
        try:
            vel = getattr(agent.state, 'p_vel', None)
            if vel is None:
                return rewards
            vel = np.asarray(vel, dtype=np.float32).reshape(1, -1)
            speed = np.linalg.norm(vel, axis=-1)  # (1,)
            
            # 基础方向奖励（仅在速度足够时计算）
            speed_thr = 0.3
            active_mask = speed > speed_thr
            base_reward = np.zeros_like(speed)

            if np.any(active_mask):
                goal_pos = None
                try:
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                        goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
                except Exception:
                    goal_pos = None
                
                if goal_pos is not None:
                    pos_now = position  # (1,3)
                    to_goal = goal_pos - pos_now
                    dist = np.linalg.norm(to_goal, axis=-1, keepdims=True)
                    goal_dir = to_goal / np.maximum(dist, 1e-6)
                    vel_dir = vel / np.maximum(speed.reshape(-1, 1), 1e-6)

                    alignment = np.sum(vel_dir * goal_dir, axis=-1)  # (1,)
                    dir_reward = alignment
                    speed_bonus = speed * 0.2
                    base_reward[active_mask] = (dir_reward + speed_bonus)[active_mask]

            # 高度平滑奖励：奖励高度（Z坐标）的平滑变化
            height_smooth_term = np.zeros(len(position), dtype=np.float32)
            try:
                # 获取当前高度（Z坐标，索引为2）
                # position 形状为 (1, 3)，取 position[0, 2] 获取当前高度
                if position.ndim == 2 and position.shape[0] > 0:
                    current_height = np.float32(position[0, 2])
                elif position.ndim == 1 and len(position) >= 3:
                    current_height = np.float32(position[2])
                else:
                    current_height = np.float32(0.0)
                
                # 获取上一帧高度
                if not hasattr(agent, 'last_height'):
                    agent.last_height = current_height
                
                prev_height = np.float32(agent.last_height)
                
                # 计算高度变化率：|current_height - prev_height| / max(|current_height|, |prev_height|, 1.0)
                # 使用1.0作为最小分母，避免除零
                max_height = np.maximum(np.abs(current_height), np.maximum(np.abs(prev_height), 1.0))
                height_change = np.abs(current_height - prev_height) / max_height
                
                # 将高度变化率映射到[0, 1]，高度变化越小，smooth越接近1
                # 使用exp衰减：exp(-height_change * scale)，scale控制敏感度
                height_smooth_scale = np.float32(5.0)  # 高度变化敏感度系数（可调整）
                # 🚨 关键修复：防止exp溢出和NaN
                # 1. 清理height_change中的NaN/Inf
                height_change = np.nan_to_num(height_change, nan=0.0, posinf=100.0, neginf=-100.0)
                # 2. 限制exp输入范围，防止溢出（exp(-700)接近0，exp(700)会溢出）
                exp_input = np.clip(-height_change * height_smooth_scale, -700.0, 700.0)
                # 3. 计算exp
                height_smooth_score = np.exp(exp_input)
                # 4. 确保输出是有限值
                height_smooth_score = np.nan_to_num(height_smooth_score, nan=0.0, posinf=1.0, neginf=0.0)
                height_smooth_score = np.clip(height_smooth_score, 0.0, 1.0)
                
                # 应用到所有位置（因为高度平滑对所有情况都适用）
                height_smooth_term[:] = height_smooth_score
                
                # 更新上一帧高度
                agent.last_height = float(current_height)
                    
            except Exception:
                height_smooth_term[:] = 0.0

            # 总奖励 = 基础方向奖励 + 高度平滑奖励
            total = base_reward.copy()
            if getattr(self, 'turn_smooth_weight', np.float32(0.0)) != 0.0:
                # 将高度平滑奖励应用到所有位置（不限于active_mask）
                total = total + self.turn_smooth_weight * height_smooth_term

            rewards[:] = total[:]
            return rewards
        except Exception:
            return rewards
    
    def _deviation_reward_vectorized(self, agent: Any, scenario: Any, start_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
        """
        惩罚智能体相对于 起点-目标 连线段的侧向偏离（越靠近直线越好）——按每智能体真实目标
        """
        rewards = np.zeros(len(current_pos), dtype=np.float32)
        try:
            # 需要目标与起点（优先每智能体真实目标）
            goal_vec = None
            try:
                if hasattr(agent, 'goal_a') and agent.goal_a is not None and agent.goal_a.state.p_pos is not None:
                    goal_vec = agent.goal_a.state.p_pos
                elif scenario is not None and hasattr(scenario, 'goal_pos'):
                    goal_vec = scenario.goal_pos
            except Exception:
                goal_vec = None
            if goal_vec is None:
                return rewards

            # 统一为(1,3)形状以便向量化计算（当前调用路径即为(1,3)）
            p0 = start_pos.reshape(-1)[:3][None, :]            # 起点 (1,3)
            p1 = np.asarray(goal_vec, dtype=np.float32).reshape(-1)[:3][None, :]  # 目标 (1,3)
            p  = current_pos.reshape(-1)[:3][None, :]          # 当前 (1,3)

            # 线段向量与长度
            v = p1 - p0                                         # (1,3)
            v_len = np.linalg.norm(v, axis=-1, keepdims=True)   # (1,1)
            # 防止极小分母
            denom = np.maximum(v_len, 1.0)                      # (1,1)

            # 起点到当前点
            w = p - p0                                          # (1,3)
            # 点到线段的投影系数 t∈[0,1]
            t = np.sum(w * v, axis=-1, keepdims=True) / (denom * denom)
            t = np.clip(t, 0.0, 1.0)                            # (1,1)

            # 最近点与垂直偏离
            proj = p0 + t * v                                   # (1,3)
            d_perp = np.linalg.norm(p - proj, axis=-1)          # (1,)

            # 将侧向偏离按路径长度归一化并裁剪到[0,2]，再映射为[ -1, 1 ]区间中的正向奖励
            norm_dev = np.clip(d_perp / (denom.squeeze() if denom.size == 1 else denom.reshape(-1)), 0.0, 2.0)
            rewards = 1.0 - norm_dev

            return rewards.astype(np.float32)
        except Exception:
            # 出错时返回零，不影响其他分项
            return rewards
    
    def _start_area_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray, start_positions: np.ndarray) -> np.ndarray:
        """向量化起始区域奖励计算"""
        rewards = np.zeros(len(positions), dtype=np.float32)
        # 安全检查：如果start_position不存在则初始化
        if not hasattr(agent, 'start_position'):
            agent.start_position = positions[0].copy() # Assuming start_position is the same for the batch
        
        dist_to_start = np.linalg.norm(positions - agent.start_position, axis=-1)
        
        if dist_to_start < 20 and np.linalg.norm(agent.state.p_vel, axis=-1) > 0.1:
            rewards = (20 - dist_to_start) * 0.5
        
        return rewards
    
    def _approach_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray, prev_positions: np.ndarray) -> np.ndarray:
        """向量化接近目标奖励计算（按每智能体目标对齐环境实现）"""
        rewards = np.zeros(len(positions), dtype=np.float32)
        try:
            goal_pos = None
            try:
                if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                    goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
            except Exception:
                goal_pos = None
            if goal_pos is None and scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                goal_pos = np.asarray(scenario.goal_pos, dtype=np.float32)
            if goal_pos is None:
                return rewards

            prev_dist = np.linalg.norm(prev_positions - goal_pos, axis=-1)
            current_dist = np.linalg.norm(positions - goal_pos, axis=-1)
            rewards = prev_dist - current_dist
            return rewards * 5.0
        except Exception:
            return np.zeros(len(positions), dtype=np.float32)
    
    def _energy_reward_vectorized(self, velocity: np.ndarray) -> np.ndarray:
        """向量化能量效率奖励计算"""
        rewards = np.zeros(len(velocity), dtype=np.float32)
        try:
                # 如果没有加速度信息，使用速度的模长作为能量消耗的近似
            energy_consumption = np.linalg.norm(velocity, axis=-1) * 0.1
        except (AttributeError, TypeError):
            # 如果访问失败，使用默认值
            energy_consumption = 1.0
        
        current_speed = np.linalg.norm(velocity, axis=-1)
        
        if np.any(energy_consumption > 0.1):
            speed_efficiency = current_speed / np.maximum(energy_consumption, 1e-6)
            rewards = np.minimum(speed_efficiency * 0.1, 2.0)
        
        return rewards
    
    def _height_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray, prev_positions: np.ndarray = None, actions: np.ndarray = None) -> np.ndarray:
        """向量化高度适应性奖励：相对于当地地形高度的相对高度（可禁用/可配置理想高度范围）
        
        Args:
            agent: 智能体对象
            scenario: 场景对象
            positions: 当前位置数组 (n, 3)
            prev_positions: 上一时刻位置数组 (n, 3)，用于计算速度方向
            actions: 动作数组 (n, act_dim)，用于判断z轴动作方向
        """
        rewards = np.zeros(len(positions), dtype=np.float32)
        
        # 允许完全关闭该分项（优先使用VectorizedRewardCalculator自身的配置）
        if not getattr(self, 'height_reward_enabled', True):
            return rewards
        
        try:
            if scenario is None or not hasattr(scenario, 'get_terrain_height'):
                return rewards
            
            # 获取地形高度（真实坐标系）
            terrain_heights = np.array(
                [scenario.get_terrain_height(pos[0], pos[1]) for pos in positions],
                dtype=np.float32
            )
            height_diff = positions[:, 2] - terrain_heights  # 智能体离地高度（米）
            
            # 可配置的理想高度范围（与场景 paper3d_terrain_weighted._calculate_height_reward 保持一致）
            ideal_min = float(getattr(self, 'height_ideal_min', 2.0))
            ideal_max = float(getattr(self, 'height_ideal_max', 5.0))
            if ideal_min > ideal_max:
                ideal_min, ideal_max = ideal_max, ideal_min
            
            # 三段式高度奖励（完全对齐场景中的标量实现）
            in_range = (height_diff >= ideal_min) & (height_diff <= ideal_max)
            below_range = height_diff < ideal_min
            above_range = height_diff > ideal_max
            
            # 1) 理想高度区间：给出固定正奖励
            rewards[in_range] = 1.0
            
            # 2) 低于理想高度：线性惩罚 + 危险高度额外惩罚 + 穿透惩罚 + 🚨 新增：向上飞行奖励
            if np.any(below_range):
                shortage = ideal_min - height_diff[below_range]
                # 与场景实现一致：线性惩罚系数 1.5
                low_reward = -shortage * 1.5
                
                # 危险高度（<3m）额外惩罚：线性 3.0（paper3d_terrain_weighted 616-618）
                danger_mask = height_diff[below_range] < 3.0
                if np.any(danger_mask):
                    danger_level = 3.0 - height_diff[below_range][danger_mask]
                    low_reward[danger_mask] -= danger_level * 3.0
                
                # 穿透地形（height_diff<0）额外惩罚：线性 15.0（paper3d_terrain_weighted 620-624）
                penetration_mask = height_diff[below_range] < 0.0
                if np.any(penetration_mask):
                    penetration_depth = -height_diff[below_range][penetration_mask]
                    low_reward[penetration_mask] -= penetration_depth * 15.0
                
                # 🚨 新增：在低高度时，如果向上飞行，给予奖励
                # 目的：鼓励智能体在低高度时主动向上飞，避免陷入"向下飞"的局部最优
                upward_reward = np.zeros(len(below_range), dtype=np.float32)
                if prev_positions is not None and len(prev_positions) == len(positions):
                    # 计算z轴速度（向上为正）
                    z_velocity = positions[:, 2] - prev_positions[:, 2]  # 位置变化（米/步）
                    # 只对低高度的智能体检查
                    below_indices = np.where(below_range)[0]
                    if len(below_indices) > 0:
                        below_z_velocity = z_velocity[below_indices]
                        # 如果z轴速度为正（向上飞），给予奖励
                        # 奖励强度：与向上速度成正比，但不超过2.0
                        upward_mask = below_z_velocity > 0.0
                        if np.any(upward_mask):
                            # 奖励公式：min(向上速度 * 2.0, 2.0)
                            # 例如：向上0.5米/步 → 奖励1.0，向上1.0米/步 → 奖励2.0
                            upward_reward[below_indices[upward_mask]] = np.clip(
                                below_z_velocity[upward_mask] * 2.0, 0.0, 2.0
                            )
                
                # 如果提供了动作信息，也可以从动作中判断向上倾向
                if actions is not None and len(actions) == len(positions):
                    below_indices = np.where(below_range)[0]
                    if len(below_indices) > 0 and actions.shape[1] >= 3:
                        # 获取z轴动作（归一化到[-1, 1]）
                        below_actions = actions[below_indices]
                        z_actions = below_actions[:, 2]  # z轴动作
                        # 如果z轴动作为正（向上），给予额外奖励
                        # 奖励强度：与动作值成正比，但不超过1.0
                        upward_action_mask = z_actions > 0.0
                        if np.any(upward_action_mask):
                            # 奖励公式：z_action * 1.0（如果z_action=1.0，奖励1.0）
                            upward_reward[below_indices[upward_action_mask]] += np.clip(
                                z_actions[upward_action_mask] * 1.0, 0.0, 1.0
                            )
                
                # 将向上飞行奖励加到低高度奖励中
                low_reward = low_reward + upward_reward
                
                rewards[below_range] = low_reward
            
            # 3) 高于理想高度：平衡的线性惩罚（系数 0.5，提高以平衡低高度惩罚）
            # 🚨 修复：提高高高度惩罚系数，从0.1提高到0.5，减少奖励函数的不对称性
            # 原因：低高度惩罚系数1.5，高高度惩罚系数0.1，导致网络倾向于向下飞
            if np.any(above_range):
                rewards[above_range] = -(height_diff[above_range] - ideal_max) * 0.5
            
            # 首回合调试打印：确认配置与场景一致
            if not self._printed_once:
                try:
                    print(
                        f"[VecRew] HEIGHT on: {self.height_reward_enabled} "
                        f"ideal=[{ideal_min:.2f},{ideal_max:.2f}] "
                        f"clip=[{float(self.min_reward):.1f},{float(self.max_reward):.1f}] "
                        f"strict_expl={self.expl_reward_strict}"
                    )
                except Exception:
                    pass
        except Exception:
            return np.zeros(len(positions), dtype=np.float32)
        return rewards

    def _success_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray, cached_data: Dict[str, Any] = None, agent_idx: int = None) -> np.ndarray:
        """
        到达目标一次性奖励 + 悬停奖励：优化版本，减少冗余检查
        修复重复成功奖励问题，鼓励智能体在终点处持续停留
        
        Args:
            agent: 智能体对象
            scenario: 场景对象
            positions: 位置数组
            cached_data: 缓存数据
            agent_idx: 智能体索引（可选，如果提供则优先使用，避免对象身份比较失败）
        """
        if positions.size == 0:
            return np.array([], dtype=np.float32)
        
        rewards = np.zeros(len(positions), dtype=np.float32)
        
        # 🚨 关键修复：优先使用传入的agent_idx，避免对象身份比较失败
        # 原因：调用位置已经有循环索引a，直接使用它比对象身份比较更可靠
        agent_id = None
        if agent_idx is not None and agent_idx >= 0:
            # 验证agent_idx是否有效
            try:
                world = getattr(scenario, 'world', None)
                if world is not None and hasattr(world, 'agents'):
                    if agent_idx < len(world.agents):
                        # agent_idx有效，直接使用
                        agent_id = agent_idx
            except Exception:
                pass
        
        # 如果传入的agent_idx无效或未提供，尝试其他方式获取
        if agent_id is None:
            # 方式1：从agent.id获取
            if hasattr(agent, 'id') and agent.id is not None:
                try:
                    agent_id = int(agent.id)
                except (ValueError, TypeError):
                    agent_id = None
            
            # 方式2：从agent.name获取
            if agent_id is None:
                agent_name = getattr(agent, 'name', '')
                if 'agent_' in agent_name.lower():
                    try:
                        agent_id = int(agent_name.split('_')[1])
                    except (ValueError, IndexError):
                        agent_id = None
            
            # 方式3：从agent.index获取
            if agent_id is None:
                if hasattr(agent, 'index') and agent.index is not None:
                    try:
                        agent_id = int(agent.index)
                    except (ValueError, TypeError):
                        agent_id = None
            
            # 方式4：从world.agents中查找索引（最后手段，但这是最可靠的方法）
            if agent_id is None:
                try:
                    world = getattr(scenario, 'world', None)
                    if world is not None and hasattr(world, 'agents'):
                        for idx, ag in enumerate(world.agents):
                            if ag is agent:
                                agent_id = idx
                                break
                except Exception:
                    pass
            
            # 🚨 关键修复：如果所有方式都失败，不要使用默认值0，而是再次尝试从world.agents中查找
            # 原因：如果agent_id获取失败，使用默认值0会导致所有agent都被识别为agent0
            # 修复：如果第一次查找失败，再次尝试查找（可能world还没有初始化）
            if agent_id is None:
                try:
                    # 再次尝试从world.agents中查找（可能第一次查找时world还没有初始化）
                    world = getattr(scenario, 'world', None)
                    if world is not None and hasattr(world, 'agents') and len(world.agents) > 0:
                        for idx, ag in enumerate(world.agents):
                            if ag is agent:
                                agent_id = idx
                                break
                except Exception:
                    pass
            
            # 🚨 关键修复：如果仍然失败，使用-1作为标记，而不是0
            # 原因：使用0会导致所有agent都被识别为agent0，使用-1可以区分"未找到"和"确实是agent0"
            # 在打印时，如果agent_id是-1，会再次尝试从world.agents中查找
            if agent_id is None:
                agent_id = -1  # 使用-1作为标记，表示未找到
                # 🔧 可选：输出警告，帮助调试
                import os as _debug_os
                if _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on'):
                    print(f"[DEBUG] Warning: Failed to get agent_id for agent, using -1 as marker. agent.name={getattr(agent, 'name', 'N/A')}, agent.id={getattr(agent, 'id', 'N/A')}, agent_idx={agent_idx}")
        
        # 初始化成功状态跟踪
        if not hasattr(agent, '_success_state'):
            agent._success_state = {
                'success_reward_given': False,
                'first_success_step': None,
                'hover_reward_count': 0
            }
        
        success_state = agent._success_state
        # 每回合重置：检测到步数“回绕”（cur_step < 上次看到的步）或首次观测步数即视为新回合
        # 🚨 关键修复：将cur_step和should_update_last_seen_step声明为函数级变量，确保在函数结束前可用
        cur_step = None
        should_update_last_seen_step = False
        try:
            world = getattr(scenario, 'world', None)
            if world is not None and hasattr(world, 'current_step'):
                cur_step = int(getattr(world, 'current_step', -1))
            else:
                # 回退字段（不推荐）：仅当world无该属性时回退
                cur_step = int(getattr(scenario, 'current_step', -1))

            last_seen_step = getattr(agent, '_last_seen_step', None)
            # 🔧 增强新回合判断：添加更严格的检查，防止同一回合中重复触发
            # 判断条件：
            # 1. 首次观测（last_seen_step is None）
            # 2. 步数回绕（cur_step < last_seen_step，说明新回合开始）- 这是最可靠的判断
            # 3. 步数为0且上次不是0（明确的新回合开始）
            # 🚨 关键修复：移除cur_step==-1的判断，因为cur_step获取失败不应该被视为新回合
            # 如果cur_step获取失败，应该保持当前状态，而不是重置（避免误重置导致重复触发）
            # 只有在cur_step有效（>=0）且明确小于last_seen_step时，才认为是新回合
            is_new_episode = (last_seen_step is None) or \
                            (cur_step >= 0 and last_seen_step is not None and cur_step < last_seen_step) or \
                            (cur_step == 0 and last_seen_step is not None and last_seen_step != 0)
            
            # 🔧 添加调试日志（可选）
            import os as _debug_os
            debug_success = _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on')
            if debug_success and is_new_episode:
                print(f"[DEBUG] New episode detected: cur_step={cur_step}, last_seen_step={last_seen_step}, agent_id={agent_id}")
            
            # 🚨 关键修复：只有在明确检测到新回合时，才重置状态
            # 并且要确保不会在同一个step中多次重置（防止重复触发）
            if is_new_episode:
                # 🔧 额外检查：如果已经给过奖励，且cur_step没有明显回绕，不重置
                # 这可以防止在同一个step中多次调用时误重置
                if success_state.get('success_reward_given', False):
                    # 如果已经给过奖励，需要更严格的检查才能重置
                    # 只有当cur_step明确小于last_seen_step（步数回绕）时，才重置
                    if not (cur_step >= 0 and last_seen_step is not None and cur_step < last_seen_step):
                        # 不是明确的步数回绕，可能是误判，不重置
                        if debug_success:
                            print(f"[DEBUG] Skipping reset: already given reward, cur_step={cur_step}, last_seen_step={last_seen_step}")
                        is_new_episode = False  # 取消重置
                
                # 🚨 额外保护：如果cur_step和last_seen_step相同或接近，不重置
                # 这可以防止在同一个step中多次调用时误重置
                if is_new_episode and cur_step >= 0 and last_seen_step is not None:
                    # 如果cur_step和last_seen_step相同或只差1，可能是同一个step中的多次调用
                    if cur_step == last_seen_step or (cur_step == last_seen_step + 1 and success_state.get('success_reward_given', False)):
                        # 可能是同一个step中的多次调用，不重置
                        if debug_success:
                            print(f"[DEBUG] Skipping reset: same step detected, cur_step={cur_step}, last_seen_step={last_seen_step}")
                        is_new_episode = False  # 取消重置
                
                if is_new_episode:
                    success_state['success_reward_given'] = False
                    success_state['first_success_step'] = None
                    success_state['hover_reward_count'] = 0
                    success_state['no_collision_reward_given'] = False  # 🔧 新增：重置无碰撞奖励标志（agent级别，已废弃但保留兼容性）
                    
                    # 🚨 关键修复：重置world级别的无碰撞奖励标志（确保新回合开始时重置）
                    try:
                        world = getattr(scenario, 'world', None)
                        if world is not None:
                            world._no_collision_reward_given = False
                    except Exception:
                        pass
                    
                    # 🔧 添加确认日志
                    if debug_success:
                        print(f"[DEBUG] Success state reset for agent {agent_id}: success_reward_given=False, cur_step={cur_step}, last_seen_step={last_seen_step}")
                    # 同步重置本回合的安全标志（用于回合成功判定）
                    try:
                        agent._had_penetration_or_collision = False
                        agent._had_obstacle_collision = False
                        agent._had_terrain_contact_or_penetration = False
                    except Exception:
                        pass
            # 🚨 关键修复：last_seen_step的更新应该在检查成功条件之后，而不是在之前
            # 这样可以确保在同一个step中的多次调用时，last_seen_step保持一致
            # 暂时不更新last_seen_step，等到检查完成功条件后再更新
            should_update_last_seen_step = (cur_step >= 0)
        except Exception as e:
            # 🔧 修复：记录异常信息帮助调试
            import os as _debug_os
            if _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on'):
                print(f"[WARNING] Episode detection failed for agent {agent_id}: {e}")
            # 🚨 关键修复：异常情况下，不重置状态，避免误重置导致重复触发
            # 如果cur_step获取失败，应该保持当前状态，而不是重置
            # 只有在明确检测到新回合时（cur_step < last_seen_step），才重置状态
            try:
                # 如果这是首次调用（last_seen_step为None），则初始化状态
                if not hasattr(agent, '_last_seen_step') or getattr(agent, '_last_seen_step', None) is None:
                    # 首次调用，确保状态已初始化
                    if 'success_reward_given' not in success_state:
                        success_state['success_reward_given'] = False
                    if 'no_collision_reward_given' not in success_state:
                        success_state['no_collision_reward_given'] = False
            except Exception:
                pass
        
        # 快速路径：使用缓存数据
        if self.use_fast_path and cached_data is not None:
            goal_positions = cached_data.get('goal_positions')
            goal_radii = cached_data.get('goal_radii')
            
            if goal_positions is not None and len(goal_positions) > 0:
                # 🚨 关键修复：使用函数开头获取的agent_id，而不是重新获取agent_idx
                # 原因：函数开头已经有完整的agent_id获取逻辑，使用它确保一致性
                # 如果agent_id获取失败（为-1），再次尝试从world.agents中查找
                agent_idx_for_goal = agent_id if agent_id is not None and agent_id >= 0 else -1
                
                # 🔧 如果agent_id是-1或None，尝试从world.agents中获取正确的智能体索引
                if agent_idx_for_goal < 0:
                    try:
                        world = getattr(scenario, 'world', None)
                        if world is not None and hasattr(world, 'agents'):
                            for idx, ag in enumerate(world.agents):
                                if ag is agent:
                                    agent_idx_for_goal = idx
                                    break
                    except Exception:
                        pass
                
                # 🚨 关键修复：如果仍然找不到，尝试使用传入的agent_idx或保持-1标记
                # 原因：使用0会导致所有agent都被识别为agent0
                if agent_idx_for_goal < 0:
                    # 尝试使用传入的agent_idx（如果有效）
                    if agent_idx is not None and agent_idx >= 0:
                        agent_idx_for_goal = agent_idx
                    else:
                        # 如果agent_idx也无效，使用-1标记，并在打印时明确标识
                        agent_idx_for_goal = -1
                        import os as _debug_os
                        if _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on'):
                            print(f"[DEBUG] Warning: Failed to get agent_id for goal selection, using -1. agent.name={getattr(agent, 'name', 'N/A')}, agent.id={getattr(agent, 'id', 'N/A')}, agent_idx={agent_idx}")
                
                # 🔧 如果agent_idx_for_goal仍然是-1，使用0作为最后手段（但会记录警告）
                # 注意：这应该很少发生，因为我们已经优先使用了传入的agent_idx
                if agent_idx_for_goal < 0:
                    agent_idx_for_goal = 0
                    import os as _debug_os
                    if _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on'):
                        print(f"[DEBUG] Warning: All methods failed for goal selection, using 0 as last resort. agent.name={getattr(agent, 'name', 'N/A')}, agent.id={getattr(agent, 'id', 'N/A')}, agent_idx={agent_idx}")
                
                # 使用当前智能体对应的目标位置
                if agent_idx_for_goal < len(goal_positions):
                    goal_pos = goal_positions[agent_idx_for_goal]
                    goal_radius = goal_radii[agent_idx_for_goal] if goal_radii is not None and agent_idx_for_goal < len(goal_radii) else None
                else:
                    # 回退：如果索引超出范围，使用第一个目标
                    goal_pos = goal_positions[0]
                    goal_radius = goal_radii[0] if goal_radii is not None and len(goal_radii) > 0 else None
                
                # 向量化距离计算
                distances = np.linalg.norm(positions - goal_pos, axis=-1)
                
                # 两步判定：先进入目标范围，再应用阈值
                if goal_radius is not None and goal_radius > 0:
                    in_area_mask = distances <= goal_radius
                    success_mask = in_area_mask & (distances <= self.success_distance_threshold)
                else:
                    success_mask = distances <= self.success_distance_threshold
                
                if np.any(success_mask):
                    # 一次性成功奖励（防重复）
                    # 🚨 关键修复：使用原子性检查-设置操作，防止并发调用导致重复触发
                    # 在检查成功条件之前，先检查并设置标志，确保同一回合中只触发一次
                    reward_should_be_given = False
                    if not success_state.get('success_reward_given', False):
                        # 🔧 修复：立即设置标志，防止在后续处理过程中再次触发
                        # 这是关键：在检查成功条件之后、计算奖励之前就设置标志
                        success_state['success_reward_given'] = True
                        success_state['first_success_step'] = getattr(scenario, 'current_step', 0)
                        success_state['hover_reward_count'] = 0
                        reward_should_be_given = True
                    
                    # 🚨 关键修复：只有在标志刚被设置时才给予奖励
                    if reward_should_be_given:
                        
                        # 🔧 新增：计算无碰撞比例（基于回合总步数和实际碰撞次数）
                        # 无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                        no_collision_ratio = 1.0  # 默认值：假设没有碰撞
                        no_collision_reward = 0.0
                        total_collision_count = 0
                        episode_length = 2800  # 默认值
                        try:
                            world = getattr(scenario, 'world', None)
                            if world is not None:
                                # 获取回合总步数
                                if hasattr(world, 'episode_length') and world.episode_length is not None:
                                    episode_length = int(world.episode_length)
                                elif hasattr(world, 'max_steps') and world.max_steps is not None:
                                    episode_length = int(world.max_steps)
                                else:
                                    # 从环境变量获取
                                    import os
                                    episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                                    try:
                                        episode_length = int(episode_length_str)
                                    except (ValueError, TypeError):
                                        episode_length = 2800
                                
                                # 统计所有智能体的总碰撞次数
                                if hasattr(world, 'agents') and world.agents is not None:
                                    for ag in world.agents:
                                        penetration_count = 0
                                        if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                            penetration_count = ag.debug_info.get('total_penetration_count', 0)
                                            try:
                                                penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                            except (ValueError, TypeError, OverflowError):
                                                penetration_count = 0
                                        total_collision_count += penetration_count
                                    
                                    # 计算无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                                    if episode_length > 0:
                                        collision_ratio = float(total_collision_count) / float(episode_length)
                                        no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                        
                                        # 🚨 非线性映射：降低少量碰撞时的奖励，确保"碰了一半及以上就小于0.2"
                                        # 如果碰撞比例 < 0.1（少量碰撞，如50次/2800=0.018），使用更严厉的惩罚
                                        if collision_ratio < 0.1:
                                            # 少量碰撞时，使用更严厉的指数惩罚，让梯度更明显
                                            # 例如：50次碰撞(0.018) -> 比例 = 0.982^4.5 ≈ 0.920
                                            # 使用指数4.5，让少量碰撞时的奖励显著降低，形成更明显的梯度
                                            no_collision_ratio = no_collision_ratio ** 4.5  # 更严厉的指数惩罚，让少量碰撞时比例更低
                                        # 如果碰撞比例 >= 0.5（碰了一半及以上），使用更严厉的惩罚
                                        elif collision_ratio >= 0.5:
                                            # 碰撞比例在[0.5, 1.0]范围内，映射到[0.0, 0.2]
                                            # 使用线性映射：0.5 -> 0.2, 1.0 -> 0.0
                                            no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                            no_collision_ratio = max(0.0, no_collision_ratio)
                                    else:
                                        no_collision_ratio = 1.0 if total_collision_count == 0 else 0.0
                                    
                                    # 🚨 关键修复：使用严格的碰撞检查逻辑，确保判断准确
                                    # 检查所有智能体是否都没有碰撞（使用与显示逻辑一致的检查）
                                    all_no_collision_strict = True
                                    try:
                                        if hasattr(world, 'agents') and world.agents is not None:
                                            for ag in world.agents:
                                                penetration_count_check = 0
                                                if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                                    penetration_count_check = ag.debug_info.get('total_penetration_count', 0)
                                                    try:
                                                        penetration_count_check = int(penetration_count_check) if np.isfinite(penetration_count_check) else 0
                                                    except (ValueError, TypeError, OverflowError):
                                                        penetration_count_check = 0
                                                if (getattr(ag, '_had_penetration_or_collision', False) or 
                                                    penetration_count_check > 0 or 
                                                    getattr(ag, '_had_terrain_contact_or_penetration', False) or
                                                    getattr(ag, '_had_obstacle_collision', False)):
                                                    all_no_collision_strict = False
                                                    break
                                    except Exception:
                                        all_no_collision_strict = False
                                    
                                    # 如果所有智能体都没有碰撞，给予无碰撞奖励
                                    if all_no_collision_strict and total_collision_count == 0 and self.no_collision_reward_value > 0.0:
                                        # 初始化无碰撞奖励状态
                                        if 'no_collision_reward_given' not in success_state:
                                            success_state['no_collision_reward_given'] = False
                                        
                                        # 只在第一次成功到达时给予无碰撞奖励
                                        if not success_state.get('no_collision_reward_given', False):
                                            success_state['no_collision_reward_given'] = True
                                            no_collision_reward = self.no_collision_reward_value
                            else:
                                # 如果没有world，回退到只检查当前智能体
                                penetration_count = 0
                                if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                    penetration_count = agent.debug_info.get('total_penetration_count', 0)
                                    try:
                                        penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                    except (ValueError, TypeError, OverflowError):
                                        penetration_count = 0
                                
                                total_collision_count = penetration_count
                                
                                # 获取回合总步数（回退方案）
                                import os
                                episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                                try:
                                    episode_length = int(episode_length_str)
                                except (ValueError, TypeError):
                                    episode_length = 2800
                                
                                # 计算无碰撞比例
                                if episode_length > 0:
                                    collision_ratio = float(total_collision_count) / float(episode_length)
                                    no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                    
                                    # 🚨 非线性映射：确保"碰了一半及以上就小于0.2"
                                    if collision_ratio >= 0.5:
                                        no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                        no_collision_ratio = max(0.0, no_collision_ratio)
                                else:
                                    no_collision_ratio = 1.0 if total_collision_count == 0 else 0.0
                                
                                # 🚨 关键修复：回退路径也使用严格的碰撞检查
                                all_no_collision_strict = (penetration_count == 0 and 
                                                          not getattr(agent, '_had_penetration_or_collision', False) and
                                                          not getattr(agent, '_had_terrain_contact_or_penetration', False) and
                                                          not getattr(agent, '_had_obstacle_collision', False))
                                
                                if all_no_collision_strict and total_collision_count == 0 and self.no_collision_reward_value > 0.0:
                                    if 'no_collision_reward_given' not in success_state:
                                        success_state['no_collision_reward_given'] = False
                                    if not success_state.get('no_collision_reward_given', False):
                                        success_state['no_collision_reward_given'] = True
                                        no_collision_reward = self.no_collision_reward_value
                        except Exception:
                            pass
                        
                        # 🚨 关键修改：成功奖励 = 成功奖励 × 无碰撞比例
                        # 如果所有智能体都没有碰撞，无碰撞比例 = 1.0，成功奖励 = 原值
                        # 如果部分智能体有碰撞，无碰撞比例 < 1.0，成功奖励会按比例减少
                        # 如果所有智能体都有碰撞，无碰撞比例 = 0.0，成功奖励 = 0
                        success_reward_scaled = self.success_reward_value * no_collision_ratio
                        rewards[success_mask] = success_reward_scaled
                        
                        # 🚨 关键修复：无碰撞奖励是全队奖励，应该给所有智能体（而不是只给到达目标的智能体）
                        # 只有在所有智能体都到达目标且无碰撞时，才给所有智能体无碰撞奖励
                        # 注意：无碰撞奖励不在_success_reward_vectorized中直接加到rewards，而是通过world级别的标志
                        # 在_calculate_all_rewards_vectorized中统一给所有智能体加上（类似全局奖励的处理方式）
                        # 🔧 关键修复：在快速路径中也需要检查所有智能体是否都到达目标
                        all_agents_reached = False
                        try:
                            world = getattr(scenario, 'world', None)
                            if world is not None and hasattr(world, 'agents') and world.agents is not None:
                                all_reached_count = 0
                                for ag in world.agents:
                                    ag_pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                                    if ag_pos is not None:
                                        ag_goal = None
                                        if hasattr(ag, 'goal_a') and hasattr(ag.goal_a, 'state') and getattr(ag.goal_a.state, 'p_pos', None) is not None:
                                            ag_goal = ag.goal_a.state.p_pos
                                        if ag_goal is None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                                            ag_goal = scenario.goal_pos
                                        if ag_goal is not None:
                                            ag_dist = np.linalg.norm(np.array(ag_pos) - np.array(ag_goal))
                                            if ag_dist <= self.success_distance_threshold:
                                                all_reached_count += 1
                                all_agents_reached = (all_reached_count == len(world.agents))
                        except Exception:
                            all_agents_reached = False
                        
                        if all_agents_reached and no_collision_reward > 0.0:
                            # 将无碰撞奖励值存储到world中，供后续统一分配
                            if not hasattr(world, '_team_no_collision_reward'):
                                world._team_no_collision_reward = 0.0
                            world._team_no_collision_reward = no_collision_reward
                        # 🔧 改进日志：显示环境ID和智能体ID，减少混淆
                        try:
                            # 🚨 关键修复：尝试从多个地方获取环境ID，确保能正确显示
                            world = getattr(scenario, 'world', None)
                            env_id = None
                            # 尝试1：从world获取
                            if world is not None:
                                env_id = getattr(world, 'env_id', None)
                            # 尝试2：从scenario获取
                            if env_id is None:
                                env_id = getattr(scenario, 'env_id', None)
                            # 尝试3：从agent获取（某些场景中agent可能包含环境信息）
                            if env_id is None and hasattr(agent, 'world'):
                                agent_world = getattr(agent, 'world', None)
                                if agent_world is not None:
                                    env_id = getattr(agent_world, 'env_id', None)
                            # 尝试4：从agent的某个属性获取（如果存在）
                            if env_id is None and hasattr(agent, 'env_id'):
                                env_id = getattr(agent, 'env_id', None)
                            
                            # 只在第一个环境打印，或者显示完整信息
                            success_count = int(np.sum(success_mask))
                            env_display = f"Env {env_id}" if env_id is not None else "Env ?"
                            agent_info = f"Agent {agent_id}" if agent_id is not None else "Agent"
                            
                            # 🚨 修复：移除env_id限制，确保所有智能体都能输出信息
                            # 原代码：if env_id is None or env_id == 0: 只允许第一个环境打印
                            # 问题：这可能导致某些智能体的信息被隐藏
                            # 修复：所有环境都打印，但添加环境ID标识（如果env_id > 0）
                            if np.any(success_mask):
                                min_dist = np.min(distances[success_mask]) if np.any(success_mask) else 999.0
                                # 🚨 关键修复：使用实际的智能体ID（agent_id），而不是位置索引
                                # 因为success_mask只包含当前智能体的位置，索引总是0
                                # 🔧 修复：如果agent_id是-1（未找到标记），再次尝试从world.agents中查找
                                actual_agent_id = agent_id if agent_id is not None and agent_id >= 0 else -1
                                
                                # 🔧 改进：如果agent_id是-1或None，尝试从world.agents中获取正确的智能体索引
                                if actual_agent_id < 0:
                                    try:
                                        world = getattr(scenario, 'world', None)
                                        if world is not None and hasattr(world, 'agents'):
                                            for idx, ag in enumerate(world.agents):
                                                if ag is agent:
                                                    actual_agent_id = idx
                                                    break
                                    except Exception:
                                        pass
                                
                                # 🚨 关键修复：如果仍然找不到，尝试使用传入的agent_idx或保持-1标记
                                # 原因：使用0会导致所有agent都被识别为agent0
                                if actual_agent_id < 0:
                                    # 尝试使用传入的agent_idx（如果有效）
                                    if agent_idx is not None and agent_idx >= 0:
                                        actual_agent_id = agent_idx
                                    else:
                                        # 如果agent_idx也无效，使用-1标记，并在打印时明确标识
                                        actual_agent_id = -1
                                        import os as _debug_os
                                        if _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on'):
                                            print(f"[DEBUG] Warning: Failed to get agent_id for agent in print, using -1. agent.name={getattr(agent, 'name', 'N/A')}, agent.id={getattr(agent, 'id', 'N/A')}, agent_idx={agent_idx}")
                                
                                # 🔧 如果actual_agent_id仍然是-1，使用0作为最后手段（但会记录警告）
                                # 注意：这应该很少发生，因为我们已经优先使用了传入的agent_idx
                                if actual_agent_id < 0:
                                    actual_agent_id = 0
                                    import os as _debug_os
                                    if _debug_os.getenv('DEBUG_SUCCESS_REWARD', '0').lower() in ('1','true','yes','on'):
                                        print(f"[DEBUG] Warning: All methods failed for print, using 0 as last resort. agent.name={getattr(agent, 'name', 'N/A')}, agent.id={getattr(agent, 'id', 'N/A')}, agent_idx={agent_idx}")
                                
                                # 🚨 显示无碰撞比例和缩放后的成功奖励
                                env_display = f"Env{env_id}" if env_id is not None else "Env?"
                                print(f"[VecSuccessReward] {env_display} Agent{actual_agent_id}: reached goal at {min_dist:.2f}m, collisions={total_collision_count}/{episode_length}, no_collision_ratio={no_collision_ratio:.3f}, reward={success_reward_scaled:.1f} (scaled from {self.success_reward_value:.1f}, one-time)")
                                
                                # 🔧 新增：输出无碰撞奖励提示
                                # 🚨 关键修复：no_collision_reward > 0.0 表示所有智能体都到达目标且无碰撞
                                # 只有在所有智能体都到达且无碰撞时，才打印无碰撞奖励
                                if no_collision_reward > 0.0:
                                    # 🚨 关键修复：再次检查所有智能体是否都到达（确保打印时条件仍然满足）
                                    all_agents_reached_for_print = False
                                    try:
                                        if world is not None and hasattr(world, 'agents') and world.agents is not None:
                                            all_reached_count = 0
                                            for ag in world.agents:
                                                ag_pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                                                if ag_pos is not None:
                                                    ag_goal = None
                                                    if hasattr(ag, 'goal_a') and hasattr(ag.goal_a, 'state') and getattr(ag.goal_a.state, 'p_pos', None) is not None:
                                                        ag_goal = ag.goal_a.state.p_pos
                                                    if ag_goal is None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                                                        ag_goal = scenario.goal_pos
                                                    
                                                    if ag_goal is not None:
                                                        ag_dist = np.linalg.norm(np.array(ag_pos) - np.array(ag_goal))
                                                        if ag_dist <= self.success_distance_threshold:
                                                            all_reached_count += 1
                                            
                                            all_agents_reached_for_print = (all_reached_count == len(world.agents))
                                    except Exception:
                                        all_agents_reached_for_print = False
                                    
                                    # 只有在所有智能体都到达时才打印
                                    if all_agents_reached_for_print:
                                        # 🚨 关键修复：使用实际的智能体ID，而不是位置索引
                                        # 🔧 修复：如果agent_id是-1（未找到标记），再次尝试从world.agents中查找
                                        actual_agent_id = agent_id if agent_id is not None and agent_id >= 0 else -1
                                        
                                        # 🔧 改进：如果agent_id是-1或None，尝试从world.agents中获取正确的智能体索引
                                        if actual_agent_id < 0:
                                            try:
                                                world = getattr(scenario, 'world', None)
                                                if world is not None and hasattr(world, 'agents'):
                                                    for idx, ag in enumerate(world.agents):
                                                        if ag is agent:
                                                            actual_agent_id = idx
                                                            break
                                            except Exception:
                                                pass
                                        
                                        # 🚨 关键修复：如果仍然找不到，尝试使用传入的agent_idx或保持-1标记
                                        # 原因：使用0会导致所有agent都被识别为agent0
                                        if actual_agent_id < 0:
                                            # 尝试使用传入的agent_idx（如果有效）
                                            if agent_idx is not None and agent_idx >= 0:
                                                actual_agent_id = agent_idx
                                            else:
                                                # 如果agent_idx也无效，使用-1标记
                                                actual_agent_id = -1
                                        
                                        # 🔧 如果actual_agent_id仍然是-1，使用0作为最后手段
                                        # 注意：这应该很少发生，因为我们已经优先使用了传入的agent_idx
                                        if actual_agent_id < 0:
                                            actual_agent_id = 0
                                        
                                        env_display = f"Env{env_id}" if env_id is not None else "Env?"
                                        # 🔧 修复：no_collision_reward > 0.0 表示所有智能体都没有碰撞，应该显示"All agents"而不是单个Agent
                                        print(f"[VecNoCollisionReward] {env_display} All agents: no collision in episode, reward={no_collision_reward} (one-time)")
                        except Exception:
                            pass
                    else:
                        # 悬停奖励：鼓励稳定悬停
                        current_speed = np.linalg.norm(getattr(agent.state, 'p_vel', np.zeros(3)))
                        hover_speed_threshold = 1.0
                        hover_reward_max = 5.0
                        
                        if current_speed < hover_speed_threshold:
                            hover_reward = (1.0 - current_speed / hover_speed_threshold) * hover_reward_max
                            success_state['hover_reward_count'] += 1
                            
                            # 限制悬停奖励频率
                            if success_state['hover_reward_count'] % 10 == 0:
                                rewards[success_mask] = hover_reward
                        else:
                            # 高速惩罚
                            speed_penalty = -current_speed * 0.5
                            rewards[success_mask] = speed_penalty
                else:
                    # 曾到达过目标点就开放悬停奖励（区外也按间隔发放）
                    if success_state['success_reward_given']:
                        current_speed = np.linalg.norm(getattr(agent.state, 'p_vel', np.zeros(3)))
                        hover_speed_threshold = 1.0
                        hover_reward_max = 5.0
                        if current_speed < hover_speed_threshold:
                            hover_reward = (1.0 - current_speed / hover_speed_threshold) * hover_reward_max
                            success_state['hover_reward_count'] += 1
                            if success_state['hover_reward_count'] % 10 == 0:
                                rewards[:] = hover_reward
                return rewards
        
        # 回退路径：原有逻辑（简化版）
        # 🔧 修复：优先使用智能体的独立目标，如果没有则使用中央目标
        goal_pos_fallback = None
        try:
            # 尝试获取当前智能体的独立目标
            if hasattr(agent, 'goal_a') and agent.goal_a is not None:
                if hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                    goal_pos_fallback = agent.goal_a.state.p_pos
        except Exception:
            pass
        
        # 如果没有独立目标，使用中央目标
        if goal_pos_fallback is None:
            if scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                goal_pos_fallback = scenario.goal_pos
        
        if goal_pos_fallback is not None:
            distances = np.linalg.norm(positions - goal_pos_fallback, axis=-1)
            success_mask = distances <= self.success_distance_threshold
            
            if np.any(success_mask):
                # 一次性成功奖励（防重复）
                # 🚨 关键修复：使用get方法，避免KeyError，并防止并发调用导致重复触发
                if not success_state.get('success_reward_given', False):
                    # 🔧 修复：立即设置标志，防止在后续处理过程中再次触发
                    success_state['success_reward_given'] = True
                    success_state['first_success_step'] = getattr(scenario, 'current_step', 0)
                    success_state['hover_reward_count'] = 0
                    
                    # 🔧 新增：计算无碰撞比例（基于回合总步数和实际碰撞次数）
                    # 无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                    no_collision_ratio = 1.0  # 默认值：假设没有碰撞
                    no_collision_reward = 0.0
                    total_collision_count = 0
                    episode_length = 2800  # 默认值
                    try:
                        world = getattr(scenario, 'world', None)
                        if world is not None:
                            # 获取回合总步数
                            if hasattr(world, 'episode_length') and world.episode_length is not None:
                                episode_length = int(world.episode_length)
                            elif hasattr(world, 'max_steps') and world.max_steps is not None:
                                episode_length = int(world.max_steps)
                            else:
                                # 从环境变量获取
                                import os
                                episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                                try:
                                    episode_length = int(episode_length_str)
                                except (ValueError, TypeError):
                                    episode_length = 2800
                            
                            # 统计所有智能体的总碰撞次数
                            if hasattr(world, 'agents') and world.agents is not None:
                                for ag in world.agents:
                                    penetration_count = 0
                                    if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                        penetration_count = ag.debug_info.get('total_penetration_count', 0)
                                        try:
                                            penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                        except (ValueError, TypeError, OverflowError):
                                            penetration_count = 0
                                    total_collision_count += penetration_count
                                
                                # 计算无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                                if episode_length > 0:
                                    collision_ratio = float(total_collision_count) / float(episode_length)
                                    no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                    
                                    # 🚨 非线性映射：降低少量碰撞时的奖励，确保"碰了一半及以上就小于0.2"
                                    # 如果碰撞比例 < 0.1（少量碰撞，如50次/2800=0.018），使用更严厉的惩罚
                                    if collision_ratio < 0.1:
                                        # 少量碰撞时，使用更严厉的指数惩罚，让梯度更明显
                                        # 例如：50次碰撞(0.018) -> 比例 = 0.982^4.5 ≈ 0.920
                                        # 更严厉：50次碰撞(0.018) -> 比例 = 0.982^5.0 ≈ 0.910
                                        # 使用指数4.5，让少量碰撞时的奖励显著降低，形成更明显的梯度
                                        no_collision_ratio = no_collision_ratio ** 4.5  # 更严厉的指数惩罚，让少量碰撞时比例更低
                                    # 如果碰撞比例 >= 0.5（碰了一半及以上），使用更严厉的惩罚
                                    elif collision_ratio >= 0.5:
                                        # 碰撞比例在[0.5, 1.0]范围内，映射到[0.0, 0.2]
                                        # 使用线性映射：0.5 -> 0.2, 1.0 -> 0.0
                                        no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                        no_collision_ratio = max(0.0, no_collision_ratio)
                                else:
                                    no_collision_ratio = 1.0 if total_collision_count == 0 else 0.0
                                
                                # 🚨 关键修复：无碰撞奖励应该只在所有智能体都到达目标且无碰撞时才给一次
                                # 检查所有智能体是否都到达目标
                                all_agents_reached = False
                                if hasattr(world, 'agents') and world.agents is not None:
                                    all_reached_count = 0
                                    for ag in world.agents:
                                        # 检查当前智能体是否到达目标
                                        ag_pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                                        if ag_pos is not None:
                                            # 获取智能体的目标位置
                                            ag_goal = None
                                            if hasattr(ag, 'goal_a') and hasattr(ag.goal_a, 'state') and getattr(ag.goal_a.state, 'p_pos', None) is not None:
                                                ag_goal = ag.goal_a.state.p_pos
                                            if ag_goal is None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                                                ag_goal = scenario.goal_pos
                                            
                                            if ag_goal is not None:
                                                ag_dist = np.linalg.norm(np.array(ag_pos) - np.array(ag_goal))
                                                if ag_dist <= self.success_distance_threshold:
                                                    all_reached_count += 1
                                    
                                    # 所有智能体都到达目标
                                    all_agents_reached = (all_reached_count == len(world.agents))
                                
                                # 如果所有智能体都到达目标且无碰撞，给予无碰撞奖励（只给一次）
                                if all_agents_reached and total_collision_count == 0 and self.no_collision_reward_value > 0.0:
                                    # 🚨 关键修复：使用world级别的状态，确保只给一次（而不是每个agent都检查）
                                    if not hasattr(world, '_no_collision_reward_given'):
                                        world._no_collision_reward_given = False
                                    
                                    # 只在第一次所有智能体都到达且无碰撞时给予无碰撞奖励
                                    if not world._no_collision_reward_given:
                                        world._no_collision_reward_given = True
                                        no_collision_reward = self.no_collision_reward_value
                        else:
                            # 如果没有world，回退到只检查当前智能体
                            penetration_count = 0
                            if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                penetration_count = agent.debug_info.get('total_penetration_count', 0)
                                try:
                                    penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                except (ValueError, TypeError, OverflowError):
                                    penetration_count = 0
                            
                            total_collision_count = penetration_count
                            
                            # 获取回合总步数（回退方案）
                            import os
                            episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                            try:
                                episode_length = int(episode_length_str)
                            except (ValueError, TypeError):
                                episode_length = 2800
                            
                            # 计算无碰撞比例
                            if episode_length > 0:
                                collision_ratio = float(total_collision_count) / float(episode_length)
                                no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                
                                # 🚨 非线性映射：降低少量碰撞时的奖励，确保"碰了一半及以上就小于0.2"
                                # 如果碰撞比例 < 0.1（少量碰撞），使用更严厉的惩罚
                                if collision_ratio < 0.1:
                                    # 少量碰撞时，使用更严厉的指数惩罚，让梯度更明显
                                    no_collision_ratio = no_collision_ratio ** 4.5  # 更严厉的指数惩罚
                                # 如果碰撞比例 >= 0.5（碰了一半及以上），使用更严厉的惩罚
                                elif collision_ratio >= 0.5:
                                    no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                    no_collision_ratio = max(0.0, no_collision_ratio)
                            else:
                                no_collision_ratio = 1.0 if total_collision_count == 0 else 0.0
                            
                            # 🚨 关键修复：回退路径也使用严格的碰撞检查和所有智能体到达检查
                            all_no_collision_strict_fallback = True
                            all_agents_reached_fallback = False
                            try:
                                world = getattr(scenario, 'world', None)
                                if world is not None and hasattr(world, 'agents') and world.agents is not None:
                                    all_reached_count = 0
                                    for ag in world.agents:
                                        # 检查碰撞
                                        penetration_count_check = 0
                                        if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                            penetration_count_check = ag.debug_info.get('total_penetration_count', 0)
                                            try:
                                                penetration_count_check = int(penetration_count_check) if np.isfinite(penetration_count_check) else 0
                                            except (ValueError, TypeError, OverflowError):
                                                penetration_count_check = 0
                                        if (getattr(ag, '_had_penetration_or_collision', False) or 
                                            penetration_count_check > 0 or 
                                            getattr(ag, '_had_terrain_contact_or_penetration', False) or
                                            getattr(ag, '_had_obstacle_collision', False)):
                                            all_no_collision_strict_fallback = False
                                        
                                        # 检查是否到达目标
                                        ag_pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                                        if ag_pos is not None:
                                            ag_goal = None
                                            if hasattr(ag, 'goal_a') and hasattr(ag.goal_a, 'state') and getattr(ag.goal_a.state, 'p_pos', None) is not None:
                                                ag_goal = ag.goal_a.state.p_pos
                                            if ag_goal is None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                                                ag_goal = scenario.goal_pos
                                            
                                            if ag_goal is not None:
                                                ag_dist = np.linalg.norm(np.array(ag_pos) - np.array(ag_goal))
                                                if ag_dist <= self.success_distance_threshold:
                                                    all_reached_count += 1
                                    
                                    # 所有智能体都到达目标
                                    all_agents_reached_fallback = (all_reached_count == len(world.agents))
                            except Exception:
                                all_no_collision_strict_fallback = False
                                all_agents_reached_fallback = False
                            
                            # 🚨 关键修复：只有在所有智能体都到达且无碰撞时，才给无碰撞奖励
                            if all_agents_reached_fallback and all_no_collision_strict_fallback and total_collision_count == 0 and self.no_collision_reward_value > 0.0:
                                # 🚨 关键修复：使用world级别的状态，确保只给一次
                                world = getattr(scenario, 'world', None)
                                if world is not None:
                                    if not hasattr(world, '_no_collision_reward_given'):
                                        world._no_collision_reward_given = False
                                    
                                    if not world._no_collision_reward_given:
                                        world._no_collision_reward_given = True
                                        no_collision_reward = self.no_collision_reward_value
                    except Exception:
                        pass
                    
                    # 🚨 关键修改：成功奖励 = 成功奖励 × 无碰撞比例
                    # 无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                    # 如果碰撞比例 >= 0.5，使用非线性映射确保比例 < 0.2
                    success_reward_scaled = self.success_reward_value * no_collision_ratio
                    rewards[success_mask] = success_reward_scaled
                    
                    # 🚨 关键修复：无碰撞奖励是全队奖励，应该给所有智能体（而不是只给到达目标的智能体）
                    # 只有在所有智能体都到达目标且无碰撞时，才给所有智能体无碰撞奖励（回退路径）
                    # 注意：无碰撞奖励不在_success_reward_vectorized中直接加到rewards，而是通过world级别的标志
                    # 在_calculate_all_rewards_vectorized中统一给所有智能体加上（类似全局奖励的处理方式）
                    if all_agents_reached_fallback and no_collision_reward > 0.0:
                        # 将无碰撞奖励值存储到world中，供后续统一分配
                        world = getattr(scenario, 'world', None)
                        if world is not None:
                            if not hasattr(world, '_team_no_collision_reward'):
                                world._team_no_collision_reward = 0.0
                            world._team_no_collision_reward = no_collision_reward
                    # 🔧 减少重复打印：只在第一个环境打印，避免并行环境重复输出
                    try:
                        # 🚨 关键修复：尝试从多个地方获取环境ID，确保能正确显示（回退路径）
                        world = getattr(scenario, 'world', None)
                        env_id = None
                        # 尝试1：从world获取
                        if world is not None:
                            env_id = getattr(world, 'env_id', None)
                        # 尝试2：从scenario获取
                        if env_id is None:
                            env_id = getattr(scenario, 'env_id', None)
                        # 尝试3：从agent获取（某些场景中agent可能包含环境信息）
                        if env_id is None and hasattr(agent, 'world'):
                            agent_world = getattr(agent, 'world', None)
                            if agent_world is not None:
                                env_id = getattr(agent_world, 'env_id', None)
                        # 尝试4：从agent的某个属性获取（如果存在）
                        if env_id is None and hasattr(agent, 'env_id'):
                            env_id = getattr(agent, 'env_id', None)
                        
                        # 🚨 修复：移除env_id限制，确保所有智能体都能输出信息（fallback路径）
                        # 原代码：打印语句在if env_id is None and hasattr(agent, 'env_id'):块内，导致某些agent的打印被跳过
                        # 修复：将打印语句移出条件块，确保所有agent都能打印
                        success_count = int(np.sum(success_mask))
                        agent_info = f"Agent {agent_id}" if agent_id is not None else "Agent"
                        # 🔧 修复：显示距离信息帮助调试
                        min_dist = np.min(distances[success_mask]) if np.any(success_mask) else 999.0
                        # 🚨 关键修复：使用实际的智能体ID，而不是位置索引（与主路径保持一致）
                        # 🔧 修复：如果agent_id是-1（未找到标记），再次尝试从world.agents中查找
                        actual_agent_id = agent_id if agent_id is not None and agent_id >= 0 else -1
                        
                        # 🔧 改进：如果agent_id是-1或None，尝试从world.agents中获取正确的智能体索引
                        if actual_agent_id < 0:
                            try:
                                world = getattr(scenario, 'world', None)
                                if world is not None and hasattr(world, 'agents'):
                                    for idx, ag in enumerate(world.agents):
                                        if ag is agent:
                                            actual_agent_id = idx
                                            break
                            except Exception:
                                pass
                        
                        # 🚨 关键修复：如果仍然找不到，尝试使用传入的agent_idx或保持-1标记
                        # 原因：使用0会导致所有agent都被识别为agent0
                        if actual_agent_id < 0:
                            # 尝试使用传入的agent_idx（如果有效）
                            if agent_idx is not None and agent_idx >= 0:
                                actual_agent_id = agent_idx
                            else:
                                # 如果agent_idx也无效，使用-1标记
                                actual_agent_id = -1
                        
                        # 🔧 如果actual_agent_id仍然是-1，使用0作为最后手段
                        # 注意：这应该很少发生，因为我们已经优先使用了传入的agent_idx
                        if actual_agent_id < 0:
                            actual_agent_id = 0
                        
                        # 🚨 显示无碰撞比例和缩放后的成功奖励
                        env_display = f"Env{env_id}" if env_id is not None else "Env?"
                        print(f"[VecSuccessReward] {env_display} Agent{actual_agent_id}: reached goal at {min_dist:.2f}m (fallback), collisions={total_collision_count}/{episode_length}, no_collision_ratio={no_collision_ratio:.3f}, reward={success_reward_scaled:.1f} (scaled from {self.success_reward_value:.1f})")
                        
                        # 🔧 新增：输出无碰撞奖励提示
                        # 🚨 关键修复：no_collision_reward > 0.0 表示所有智能体都到达目标且无碰撞
                        # 只有在所有智能体都到达且无碰撞时，才打印无碰撞奖励（回退路径）
                        if no_collision_reward > 0.0:
                            # 🚨 关键修复：再次检查所有智能体是否都到达（确保打印时条件仍然满足）
                            all_agents_reached_for_print_fallback = False
                            try:
                                world = getattr(scenario, 'world', None)
                                if world is not None and hasattr(world, 'agents') and world.agents is not None:
                                    all_reached_count = 0
                                    for ag in world.agents:
                                        ag_pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                                        if ag_pos is not None:
                                            ag_goal = None
                                            if hasattr(ag, 'goal_a') and hasattr(ag.goal_a, 'state') and getattr(ag.goal_a.state, 'p_pos', None) is not None:
                                                ag_goal = ag.goal_a.state.p_pos
                                            if ag_goal is None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                                                ag_goal = scenario.goal_pos
                                            
                                            if ag_goal is not None:
                                                ag_dist = np.linalg.norm(np.array(ag_pos) - np.array(ag_goal))
                                                if ag_dist <= self.success_distance_threshold:
                                                    all_reached_count += 1
                                    
                                    all_agents_reached_for_print_fallback = (all_reached_count == len(world.agents))
                            except Exception:
                                all_agents_reached_for_print_fallback = False
                            
                            # 只有在所有智能体都到达时才打印
                            if all_agents_reached_for_print_fallback:
                                # 🚨 关键修复：使用实际的智能体ID，而不是位置索引
                                # 🔧 修复：如果agent_id是-1（未找到标记），再次尝试从world.agents中查找
                                actual_agent_id = agent_id if agent_id is not None and agent_id >= 0 else -1
                                
                                # 🔧 改进：如果agent_id是-1或None，尝试从world.agents中获取正确的智能体索引
                                if actual_agent_id < 0:
                                    try:
                                        world = getattr(scenario, 'world', None)
                                        if world is not None and hasattr(world, 'agents'):
                                            for idx, ag in enumerate(world.agents):
                                                if ag is agent:
                                                    actual_agent_id = idx
                                                    break
                                    except Exception:
                                        pass
                                
                                # 🚨 关键修复：如果仍然找不到，尝试使用传入的agent_idx或保持-1标记
                                # 原因：使用0会导致所有agent都被识别为agent0
                                if actual_agent_id < 0:
                                    # 尝试使用传入的agent_idx（如果有效）
                                    if agent_idx is not None and agent_idx >= 0:
                                        actual_agent_id = agent_idx
                                    else:
                                        # 如果agent_idx也无效，使用-1标记
                                        actual_agent_id = -1
                                
                                # 🔧 如果actual_agent_id仍然是-1，使用0作为最后手段
                                # 注意：这应该很少发生，因为我们已经优先使用了传入的agent_idx
                                if actual_agent_id < 0:
                                    actual_agent_id = 0
                                
                                env_display = f"Env{env_id}" if env_id is not None else "Env?"
                                print(f"[VecNoCollisionReward] {env_display} All agents: no collision in episode, reward={no_collision_reward} (one-time)")
                    except Exception:
                        pass
                else:
                    # 悬停奖励
                    current_speed = np.linalg.norm(getattr(agent.state, 'p_vel', np.zeros(3)))
                    hover_speed_threshold = 1.0
                    hover_reward_max = 5.0
                    
                    if current_speed < hover_speed_threshold:
                        hover_reward = (1.0 - current_speed / hover_speed_threshold) * hover_reward_max
                        success_state['hover_reward_count'] += 1
                        
                        if success_state['hover_reward_count'] % 10 == 0:
                            rewards[success_mask] = hover_reward
                    else:
                        speed_penalty = -current_speed * 0.5
                        rewards[success_mask] = speed_penalty
            else:
                # 曾到达过目标点就开放悬停奖励（区外也按间隔发放）
                if success_state['success_reward_given']:
                    current_speed = np.linalg.norm(getattr(agent.state, 'p_vel', np.zeros(3)))
                    hover_speed_threshold = 1.0
                    hover_reward_max = 5.0
                    if current_speed < hover_speed_threshold:
                        hover_reward = (1.0 - current_speed / hover_speed_threshold) * hover_reward_max
                        success_state['hover_reward_count'] += 1
                        if success_state['hover_reward_count'] % 10 == 0:
                            rewards[:] = hover_reward
            
            # 🔧 修复：不再重置成功状态，确保整个回合只给一次成功奖励
            # if not np.any(success_mask):
            #     success_state['success_reward_given'] = False
        
        # 🚨 关键修复：在函数结束前更新last_seen_step，确保在同一个step中的多次调用时保持一致
        # 这样可以防止在同一个step中的多次调用时，last_seen_step被提前更新，导致is_new_episode判断出错
        # 问题根源：在同一个step中，对同一个agent，_success_reward_vectorized被多次调用（每个agent计算奖励时都会调用一次）
        # 如果last_seen_step在检查成功条件之前就被更新，会导致后续调用时is_new_episode判断出错
        try:
            if should_update_last_seen_step and cur_step is not None and cur_step >= 0:
                setattr(agent, '_last_seen_step', cur_step)
        except Exception:
            pass  # 如果更新失败，不影响主流程
        
        return rewards

    def _collision_penalty_vectorized(self, agent_idx: int, world: Any, scenario: Any, positions: np.ndarray, cached_data: Dict[str, Any] = None) -> np.ndarray:
        """
        优化版碰撞惩罚计算：使用缓存数据，减少重复计算
        
        🚨 关键修复：使用综合最小距离（d_min_current）检测碰撞，而不是只检查Z坐标穿透
        原因：图表显示 min_distance_to_obstacle 多次为0，但碰撞计数为0，说明Z坐标检测无法捕获侧面碰撞
        修复：计算障碍物和地形的综合最小距离，如果 < collision_distance_threshold 或 < 0，触发碰撞
        """
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        penalties = np.zeros(len(positions), dtype=np.float32)
        
        # 🚨 关键修复：首先计算综合最小距离（障碍物和地形的综合距离）
        # 这是检测碰撞的主要方法，比Z坐标检测更准确（能捕获侧面碰撞）
        d_min_current = None
        try:
            # 计算到障碍物的距离
            obstacle_min_dist = np.full(len(positions), np.inf, dtype=np.float32)
            has_obstacles = bool(hasattr(world, 'scenario') and 
                                 hasattr(world.scenario, 'obstacles') and 
                                 world.scenario.obstacles)
            
            if has_obstacles:
                for obstacle_data in world.scenario.obstacles:
                    if 'center' in obstacle_data and 'radius' in obstacle_data:
                        obstacle_center = np.array(obstacle_data['center'], dtype=np.float32)
                        obstacle_radius = float(obstacle_data['radius'])
                        # 计算3D距离
                        dist_3d = np.linalg.norm(positions - obstacle_center, axis=-1)
                        dist_to_surface = dist_3d - obstacle_radius  # 到障碍物表面的距离
                        obstacle_min_dist = np.minimum(obstacle_min_dist, dist_to_surface)
            
            # 计算到地形的距离（使用Z坐标差作为近似，因为精确的XY距离计算较慢）
            terrain_min_dist = np.full(len(positions), np.inf, dtype=np.float32)
            if scenario is not None and hasattr(scenario, 'get_terrain_height'):
                try:
                    if hasattr(scenario, 'get_terrain_height_vectorized'):
                        terrain_heights = scenario.get_terrain_height_vectorized(positions[:, 0], positions[:, 1])
                        if terrain_heights.ndim == 0:
                            terrain_heights = np.full(len(positions), float(terrain_heights), dtype=np.float32)
                    else:
                        terrain_heights = np.array([scenario.get_terrain_height(float(p[0]), float(p[1])) for p in positions], dtype=np.float32)
                    # 地形距离 = Z坐标差（简化版本，用于快速检测）
                    terrain_min_dist = positions[:, 2] - terrain_heights
                except Exception:
                    pass
            
            # 综合最小距离：取障碍物和地形距离的最小值
            d_min_current = np.minimum(obstacle_min_dist, terrain_min_dist)
        except Exception:
            d_min_current = np.full(len(positions), np.inf, dtype=np.float32)
        
        # 🚨 关键修复：使用综合最小距离检测碰撞
        # 如果 d_min_current < collision_distance_threshold 或 < 0，视为碰撞
        # 这是主要的碰撞检测方法，比Z坐标检测更准确（能捕获侧面碰撞）
        collision_threshold = float(self.collision_distance_threshold)
        distance_based_collision_mask = (d_min_current < collision_threshold) | (d_min_current < 0.0)
        
        # 🚨 关键修复：如果检测到距离碰撞，立即应用惩罚和更新计数
        if np.any(distance_based_collision_mask):
            # 计算碰撞深度（负数表示穿透）
            collision_depths = np.maximum(-d_min_current[distance_based_collision_mask], 0.0)
            penalties[distance_based_collision_mask] = np.minimum(
                penalties[distance_based_collision_mask],
                -self.collision_penalty_value - collision_depths * float(self.penetration_alpha)
            )
            
            # 🚨 关键修复：更新碰撞计数
            try:
                has_world_agents = hasattr(world, 'agents')
                world_agents_len = len(world.agents) if has_world_agents else 0
                agent_idx_valid = (has_world_agents and 0 <= agent_idx < world_agents_len)
                
                if agent_idx_valid:
                    ag = world.agents[agent_idx]
                    ag._had_penetration_or_collision = True
                    if not hasattr(ag, 'debug_info'):
                        ag.debug_info = {}
                    if not isinstance(ag.debug_info, dict):
                        ag.debug_info = {}
                    
                    # 统计本批次中的碰撞次数
                    collision_count = np.sum(distance_based_collision_mask)
                    old_count = ag.debug_info.get('total_penetration_count', 0)
                    try:
                        collision_count_int = int(collision_count)
                        if not np.isfinite(collision_count_int) or collision_count_int < 0:
                            collision_count_int = 0
                    except (ValueError, TypeError, OverflowError):
                        collision_count_int = 0
                    new_count = old_count + collision_count_int
                    if new_count > 1000000:
                        new_count = 1000000
                    ag.debug_info['total_penetration_count'] = int(new_count)
                    
                    # 🚨 调试输出：大幅减少输出频率，避免日志过多
                    # 只在每100次碰撞或第一次碰撞时输出，或者完全关闭（通过环境变量控制）
                    import os
                    enable_collision_debug = os.getenv('ENABLE_COLLISION_DEBUG', '0').lower() in ('1','true','yes','on')
                    if enable_collision_debug and collision_count_int > 0:
                        # 只在特定条件下输出：每100次碰撞，或者新计数是100的倍数
                        if new_count % 100 == 0 or old_count == 0:
                            min_dist_val = float(np.min(d_min_current[distance_based_collision_mask]))
                            print(f"[距离碰撞检测] ✅ agent_idx={agent_idx}, 碰撞数={collision_count_int}, "
                                  f"旧计数={old_count}, 新计数={new_count}, "
                                  f"最小距离={min_dist_val:.3f}, 阈值={collision_threshold:.3f}")
            except Exception as e:
                import os
                if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                    print(f"[距离碰撞检测异常] {type(e).__name__}: {e}")
                pass
        
        # 快速路径：使用缓存数据（保留原有的Z坐标检测作为补充，但主要依赖距离检测）
        if self.use_fast_path and cached_data is not None:
            # 地形碰撞检测
            terrain = cached_data.get('terrain')
            if terrain is not None:
                # 🚨 关键优化：越界/插值失败视为碰撞，不再静默放过
                terrain_heights = np.full(len(positions), np.nan, dtype=np.float32)
                terrain_interpolated = cached_data.get('terrain_interpolated', False)
                map_size = cached_data.get('map_size', terrain.shape[0])
                
                if terrain_interpolated:
                    # 🔧 修复降采样误判：使用双线性插值而不是整数索引
                    # 问题：使用np.floor会丢失小数精度，导致在陡峭地形边缘误判穿透
                    # 解决：即使地形已预先插值，也使用双线性插值获取精确高度
                    # 优先使用向量化插值函数（如果支持），否则使用双线性插值
                    if hasattr(scenario, 'get_terrain_height_vectorized'):
                        try:
                            terrain_heights = scenario.get_terrain_height_vectorized(positions[:, 0], positions[:, 1])
                            # 确保形状正确
                            if terrain_heights.ndim == 0:
                                terrain_heights = np.full(len(positions), float(terrain_heights), dtype=np.float32)
                            elif terrain_heights.shape[0] != len(positions):
                                terrain_heights = np.zeros(len(positions), dtype=np.float32)
                        except Exception:
                            # 回退：使用双线性插值
                            _bilinear_interpolate_terrain(terrain, positions, terrain_heights, map_size)
                    else:
                        # 使用双线性插值
                        _bilinear_interpolate_terrain(terrain, positions, terrain_heights, map_size)
                elif hasattr(scenario, 'get_terrain_height'):
                    # 地形未预先插值，使用插值函数（较慢但准确）
                    try:
                        for i, pos in enumerate(positions):
                            try:
                                terrain_heights[i] = scenario.get_terrain_height(float(pos[0]), float(pos[1]))
                            except Exception:
                                pass
                    except Exception:
                        pass
                else:
                    # 回退：如果没有插值函数，使用直接索引
                    int_pos = np.floor(positions[:, :2]).astype(np.int32)
                    valid_mask = ((int_pos[:, 0] >= 0) & (int_pos[:, 0] < map_size) & 
                                 (int_pos[:, 1] >= 0) & (int_pos[:, 1] < map_size))
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        terrain_heights[valid_indices] = terrain[int_pos[valid_indices, 1], int_pos[valid_indices, 0]]

                # 越界/插值失败 → 直接视为接触/碰撞
                invalid_mask = ~np.isfinite(terrain_heights)
                if np.any(invalid_mask):
                    penalties[invalid_mask] = np.minimum(
                        penalties[invalid_mask],
                        -self.terrain_penalty_value
                    )
                    # 填入当前位置高度，避免后续出现 NaN/Inf
                    try:
                        terrain_heights[invalid_mask] = positions[invalid_mask, 2]
                    except Exception:
                        terrain_heights[invalid_mask] = 0.0
                    try:
                        if hasattr(world, 'agents') and 0 <= agent_idx < len(world.agents):
                            ag = world.agents[agent_idx]
                            ag._had_penetration_or_collision = True
                            ag._had_terrain_contact_or_penetration = True
                            if not hasattr(ag, 'debug_info') or not isinstance(ag.debug_info, dict):
                                ag.debug_info = {}
                            ag.debug_info['total_penetration_count'] = ag.debug_info.get('total_penetration_count', 0) + int(np.sum(invalid_mask))
                    except Exception:
                        pass
                
                # 地形穿透惩罚（Z坐标检测，作为距离检测的补充）
                # 🚨 关键修复：Z坐标检测只作为补充，主要依赖距离检测（已在前面完成）
                # 原因：距离检测能捕获侧面碰撞，Z坐标检测只能捕获垂直穿透
                # 修复：Z坐标检测只检测距离检测未捕获的情况，避免重复计数
                eps = float(self.terrain_contact_eps)
                # Z坐标穿透检测：智能体Z坐标 < 地形高度 + eps
                penetration_mask = positions[:, 2] < terrain_heights + eps
                # 🚨 关键修复：排除已经被距离检测捕获的碰撞，避免重复计数
                # 如果距离检测已经检测到碰撞，Z坐标检测不再重复计数
                penetration_mask = penetration_mask & (~distance_based_collision_mask)
                
                # 🚨 调试：输出碰撞检测详细信息（仅前几个回合或调试模式）
                try:
                    import os
                    debug_collision = not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on'))
                    if debug_collision and (len(positions) > 0):
                        # 计算一些统计信息用于调试
                        z_positions = positions[:, 2]
                        z_min = float(np.min(z_positions)) if len(z_positions) > 0 else 0.0
                        z_max = float(np.max(z_positions)) if len(z_positions) > 0 else 0.0
                        terrain_h_min = float(np.min(terrain_heights)) if len(terrain_heights) > 0 and np.any(np.isfinite(terrain_heights)) else 0.0
                        terrain_h_max = float(np.max(terrain_heights)) if len(terrain_heights) > 0 and np.any(np.isfinite(terrain_heights)) else 0.0
                        penetration_count_debug = int(np.sum(penetration_mask))
                        # 🚨 关键调试：检查world.agents是否存在
                        has_world_agents = hasattr(world, 'agents')
                        world_agents_len = len(world.agents) if has_world_agents else 0
                        agent_idx_valid = (has_world_agents and 0 <= agent_idx < world_agents_len)
                        current_count = 0
                        if agent_idx_valid:
                            ag = world.agents[agent_idx]
                            if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                current_count = ag.debug_info.get('total_penetration_count', 0)
                        
                        # 🚨 关键调试：输出world.agents信息
                        print(f"[碰撞检测调试] agent_idx={agent_idx}, 本批次位置数={len(positions)}, "
                              f"穿透检测={penetration_count_debug}, 当前累计计数={current_count}, "
                              f"eps={eps:.3f}, Z范围=[{z_min:.2f}, {z_max:.2f}], "
                              f"地形高度范围=[{terrain_h_min:.2f}, {terrain_h_max:.2f}], "
                              f"穿透条件满足={penetration_count_debug > 0}, "
                              f"world.agents存在={has_world_agents}, world.agents长度={world_agents_len}, "
                              f"agent_idx有效={agent_idx_valid}, "
                              f"位置Z={z_positions[0] if len(z_positions) > 0 else 'N/A':.2f}, "
                              f"地形高度={terrain_heights[0] if len(terrain_heights) > 0 and np.isfinite(terrain_heights[0]) else 'N/A':.2f}, "
                              f"条件检查: Z({z_positions[0] if len(z_positions) > 0 else 'N/A':.2f}) < 地形({terrain_heights[0] if len(terrain_heights) > 0 and np.isfinite(terrain_heights[0]) else 'N/A':.2f}) + eps({eps:.2f}) = {terrain_heights[0] + eps if len(terrain_heights) > 0 and np.isfinite(terrain_heights[0]) else 'N/A':.2f}")
                except Exception as e:
                    import os
                    if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                        print(f"[碰撞检测调试异常] {type(e).__name__}: {e}")
                    pass
                
                if np.any(penetration_mask):
                    # 🚨 关键修复：计算穿透深度（地形高度 - 智能体Z坐标），用于惩罚计算
                    penetration_depth = terrain_heights[penetration_mask] - positions[penetration_mask, 2]
                    penalties[penetration_mask] = -self.terrain_penalty_value - np.maximum(penetration_depth, 0.0) * float(self.penetration_alpha)
                    # 🚨 关键修复：标记本回合发生过地形穿透/接触，并更新穿透计数
                    try:
                        has_world_agents = hasattr(world, 'agents')
                        world_agents_len = len(world.agents) if has_world_agents else 0
                        agent_idx_valid = (has_world_agents and 0 <= agent_idx < world_agents_len)
                        
                        # 🚨 关键调试：输出world.agents信息
                        import os
                        debug_output = not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on'))
                        if debug_output:
                            print(f"[碰撞检测更新] agent_idx={agent_idx}, world.agents存在={has_world_agents}, "
                                  f"world.agents长度={world_agents_len}, agent_idx有效={agent_idx_valid}, "
                                  f"穿透检测数={np.sum(penetration_mask)}")
                        
                        if agent_idx_valid:
                            ag = world.agents[agent_idx]
                            ag._had_penetration_or_collision = True
                            ag._had_terrain_contact_or_penetration = True
                            # 🚨 新增：更新穿透计数（用于成功判定）
                            if not hasattr(ag, 'debug_info'):
                                ag.debug_info = {}
                            if not isinstance(ag.debug_info, dict):
                                ag.debug_info = {}
                            # 统计本批次中的穿透次数
                            penetration_count = np.sum(penetration_mask)
                            old_count = ag.debug_info.get('total_penetration_count', 0)
                            # 🚨 关键修复：防止NaN和溢出，确保计数是有效的整数
                            try:
                                penetration_count_int = int(penetration_count)
                                if not np.isfinite(penetration_count_int) or penetration_count_int < 0:
                                    penetration_count_int = 0
                            except (ValueError, TypeError, OverflowError):
                                penetration_count_int = 0
                            new_count = old_count + penetration_count_int
                            # 🚨 关键修复：防止溢出，限制最大值为合理范围
                            if new_count > 1000000:  # 防止溢出
                                new_count = 1000000
                            ag.debug_info['total_penetration_count'] = int(new_count)
                            # 🚨 调试：大幅减少输出频率，避免日志过多
                            # 只在每100次碰撞或第一次碰撞时输出
                            if penetration_count > 0:
                                try:
                                    import os
                                    enable_collision_debug = os.getenv('ENABLE_COLLISION_DEBUG', '0').lower() in ('1','true','yes','on')
                                    if enable_collision_debug and (new_count % 100 == 0 or old_count == 0):
                                        # 计算穿透深度范围用于调试
                                        penetration_depths = terrain_heights[penetration_mask] - positions[penetration_mask, 2]
                                        depth_min = float(np.min(penetration_depths)) if len(penetration_depths) > 0 else 0.0
                                        depth_max = float(np.max(penetration_depths)) if len(penetration_depths) > 0 else 0.0
                                        print(f"[碰撞检测] ✅ 成功更新计数: agent_idx={agent_idx}, 本批次穿透={penetration_count}, "
                                              f"旧计数={old_count}, 新计数={new_count}, "
                                              f"world.agents长度={world_agents_len}, "
                                              f"eps={eps}, 穿透深度范围=[{depth_min:.3f}, {depth_max:.3f}]")
                                except Exception as e:
                                    if debug_output:
                                        print(f"[碰撞检测调试异常] {type(e).__name__}: {e}")
                        else:
                            # 🚨 关键调试：如果agent_idx无效，输出详细信息
                            if debug_output:
                                print(f"[碰撞检测] ❌ agent_idx无效: agent_idx={agent_idx}, "
                                      f"world.agents存在={has_world_agents}, world.agents长度={world_agents_len}")
                    except Exception as e:
                        import os
                        if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                            print(f"[碰撞检测更新异常] {type(e).__name__}: {e}")
                        pass

                # 地面接触惩罚：当 z 接近地形高度（<= 阈值）时触发，但"目标点范围内"不惩罚
                # 🚨 关键修复：地形接触检测应该只使用 terrain_contact_eps，不使用 collision_distance_threshold
                # 原因：
                #   - collision_distance_threshold 用于障碍物碰撞检测（距离阈值）
                #   - terrain_contact_eps 用于地形穿透/接触检测（高度容差）
                #   - 两者应该独立使用，不应该混合
                # 修复：移除错误的 collision_distance_threshold 使用，只使用 terrain_contact_eps
                contact_mask = positions[:, 2] <= (terrain_heights + float(self.terrain_contact_eps))

                # 目标范围判定（使用缓存的中央目标位置与 success_distance_threshold）
                success_thresh = float(getattr(self, 'success_distance_threshold', 2.0))
                goal_positions = cached_data.get('goal_positions')
                if goal_positions is not None and len(goal_positions) > 0:
                    goal_pos = goal_positions[0]
                    # 计算与中央目标的距离
                    dists_to_goal = np.linalg.norm(positions - goal_pos, axis=-1)
                    in_goal_area = dists_to_goal <= success_thresh
                else:
                    in_goal_area = np.zeros(len(positions), dtype=bool)

                # 接触且不在目标范围内 → 惩罚（不叠加重复惩罚，取更大绝对值）
                # 🚨 关键修复：排除已经被距离检测捕获的碰撞，避免重复计数
                contact_penalty_mask = contact_mask & (~in_goal_area) & (~distance_based_collision_mask)
                if np.any(contact_penalty_mask):
                    contact_penalties = np.full(len(positions), 0.0, dtype=np.float32)
                    contact_penalties[contact_penalty_mask] = -self.terrain_penalty_value
                    # 合并穿透/接触两种惩罚（取更负者）
                    penalties = np.minimum(penalties, contact_penalties)
                    # 🚨 关键修复：地面接触也要标记碰撞，并更新穿透计数
                    # 🔧 修复：检测接近地形的接触，即使没有完全穿透也要更新碰撞计数
                    # 原因：从可视化图可以看到轨迹被地形掩盖，说明确实发生了碰撞
                    # 注意：代码中没有复位机制，智能体位置是真实物理模拟的结果
                    try:
                        if hasattr(world, 'agents') and 0 <= agent_idx < len(world.agents):
                            ag = world.agents[agent_idx]
                            ag._had_penetration_or_collision = True
                            ag._had_terrain_contact_or_penetration = True
                            # 🔧 修复：更新穿透计数，检测接近地形的接触（z <= terrain_height + eps）
                            # 原因：地形高度可能因为降采样或插值误差而不完全准确，需要容差
                            if not hasattr(ag, 'debug_info'):
                                ag.debug_info = {}
                            if not isinstance(ag.debug_info, dict):
                                ag.debug_info = {}
                            # 🔧 修复：检查是否有接近穿透（z <= terrain_height + eps）
                            # 🚨 调试：使用TERRAIN_CONTACT_EPS作为阈值，与穿透检测阈值一致，大幅提高到10.0米（用于诊断）
                            eps = float(self.terrain_contact_eps)  # 使用TERRAIN_CONTACT_EPS（默认10.0米），大幅提高阈值用于诊断
                            actual_penetration_mask = contact_penalty_mask & (positions[:, 2] <= terrain_heights + eps)
                            if np.any(actual_penetration_mask):
                                penetration_count = np.sum(actual_penetration_mask)
                                old_count = ag.debug_info.get('total_penetration_count', 0)
                                try:
                                    penetration_count_int = int(penetration_count)
                                    if not np.isfinite(penetration_count_int) or penetration_count_int < 0:
                                        penetration_count_int = 0
                                except (ValueError, TypeError, OverflowError):
                                    penetration_count_int = 0
                                new_count = old_count + penetration_count_int
                                if new_count > 1000000:
                                    new_count = 1000000
                                ag.debug_info['total_penetration_count'] = int(new_count)
                    except Exception:
                        pass
            
            # 障碍物碰撞检测（向量化版本）
            # 🚨 关键修复：障碍物碰撞检测已经在距离检测中完成，这里只处理距离检测未捕获的情况
            # 原因：距离检测已经检测了障碍物距离 < threshold 的情况，这里只检测完全穿透（距离 < 0）
            obstacles_centers = cached_data.get('obstacles_centers')
            obstacles_radii = cached_data.get('obstacles_radii')
            
            if obstacles_centers is not None and obstacles_radii is not None:
                # 向量化计算所有障碍物距离（改为3D距离，而非仅XY平面）
                centers = obstacles_centers[:, :3]  # (M, 3)
                pts = positions[:, :3]              # (N, 3)
                # 广播计算欧氏距离: (N, M)
                diff = pts[:, None, :] - centers[None, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=-1))  # (N, M)
                
                # 计算穿透深度
                penetrations = obstacles_radii[None, :] - distances  # (N, M)
                collision_masks = penetrations > 0
                
                # 取最大穿透深度
                max_penetrations = np.where(collision_masks, penetrations, 0).max(axis=1)
                obs_collision_mask = max_penetrations > 0
                
                # 🚨 关键修复：排除已经被距离检测捕获的碰撞，避免重复计数和重复惩罚
                # 距离检测已经处理了 d_min_current < threshold 的情况，这里只处理完全穿透
                obs_collision_mask = obs_collision_mask & (~distance_based_collision_mask)
                
                # 应用障碍物碰撞惩罚（只对未被距离检测捕获的碰撞）
                if np.any(obs_collision_mask):
                    penalties[obs_collision_mask] = np.minimum(
                        penalties[obs_collision_mask],
                        -self.collision_penalty_value - max_penetrations[obs_collision_mask] * float(self.penetration_alpha)
                    )
                
                if np.any(obs_collision_mask):
                    # 🚨 关键修复：障碍物碰撞也要更新穿透计数
                    # 原因：从图表可以看到 min_distance_to_obstacle 经常小于0（穿透障碍物），但碰撞计数始终为0
                    # 问题：之前的代码只检查地形穿透，不检查障碍物穿透，导致障碍物碰撞没有被记录
                    try:
                        if hasattr(world, 'agents') and 0 <= agent_idx < len(world.agents):
                            ag = world.agents[agent_idx]
                            ag._had_penetration_or_collision = True
                            ag._had_obstacle_collision = True
                            # 🚨 关键修复：更新穿透计数（障碍物穿透）
                            if not hasattr(ag, 'debug_info'):
                                ag.debug_info = {}
                            if not isinstance(ag.debug_info, dict):
                                ag.debug_info = {}
                            # 统计本批次中的障碍物穿透次数
                            obstacle_penetration_count = np.sum(obs_collision_mask)
                            old_count = ag.debug_info.get('total_penetration_count', 0)
                            # 🚨 关键修复：防止NaN和溢出，确保计数是有效的整数
                            try:
                                penetration_count_int = int(obstacle_penetration_count)
                                if not np.isfinite(penetration_count_int) or penetration_count_int < 0:
                                    penetration_count_int = 0
                            except (ValueError, TypeError, OverflowError):
                                penetration_count_int = 0
                            new_count = old_count + penetration_count_int
                            # 🚨 关键修复：防止溢出，限制最大值为合理范围
                            if new_count > 1000000:  # 防止溢出
                                new_count = 1000000
                            ag.debug_info['total_penetration_count'] = int(new_count)
                            # 🚨 调试：大幅减少输出频率，避免日志过多
                            # 只在每100次碰撞或第一次碰撞时输出
                            if obstacle_penetration_count > 0:
                                try:
                                    import os
                                    enable_collision_debug = os.getenv('ENABLE_COLLISION_DEBUG', '0').lower() in ('1','true','yes','on')
                                    if enable_collision_debug and (new_count % 100 == 0 or old_count == 0):
                                        # 计算穿透深度范围用于调试
                                        penetration_depths = max_penetrations[obs_collision_mask]
                                        depth_min = float(np.min(penetration_depths)) if len(penetration_depths) > 0 else 0.0
                                        depth_max = float(np.max(penetration_depths)) if len(penetration_depths) > 0 else 0.0
                                        print(f"[障碍物碰撞检测] ✅ 成功更新计数: agent_idx={agent_idx}, 本批次障碍物穿透={obstacle_penetration_count}, "
                                              f"旧计数={old_count}, 新计数={new_count}, "
                                              f"world.agents长度={len(world.agents) if hasattr(world, 'agents') else 0}, "
                                              f"穿透深度范围=[{depth_min:.3f}, {depth_max:.3f}]")
                                except Exception as e:
                                    import os
                                    if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                                        print(f"[障碍物碰撞检测调试异常] {type(e).__name__}: {e}")
                    except Exception as e:
                        import os
                        if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                            print(f"[障碍物碰撞检测更新异常] {type(e).__name__}: {e}")
                        pass
            
            return penalties
        
        # 🚨 关键修复：回退路径也应该使用距离检测
        # 如果距离检测已经检测到碰撞，直接返回惩罚（避免重复计算）
        if np.any(distance_based_collision_mask):
            # 距离检测已经在前面处理了惩罚和计数，这里直接返回
            return penalties
        
        # 回退路径：简化版原有逻辑（仅在距离检测未捕获时使用）
        terrain = None
        if hasattr(scenario, 'terrain') and scenario.terrain is not None:
            terrain = scenario.terrain
        elif hasattr(world, 'terrain') and world.terrain is not None:
            terrain = world.terrain
        
        if terrain is not None:
            # 🚨 关键修复：使用插值获取地形高度，而不是直接使用整数索引
            # 原因：地形可能已降采样（50×50），直接使用整数索引会导致访问错误或越界
            # 解决方案：使用scenario.get_terrain_height进行插值，确保正确获取地形高度
            terrain_heights = np.full(len(positions), -1e6, dtype=np.float32)
            if hasattr(scenario, 'get_terrain_height'):
                # 🚨 关键修复：使用向量化方式调用get_terrain_height（避免循环性能问题）
                # 注意：get_terrain_height本身支持任意坐标，但我们需要向量化调用
                try:
                    # 尝试向量化调用（如果scenario支持）
                    if hasattr(scenario, 'get_terrain_height_vectorized'):
                        terrain_heights = scenario.get_terrain_height_vectorized(positions[:, 0], positions[:, 1])
                    else:
                        # 回退：逐个调用（虽然慢但准确）
                        for i, pos in enumerate(positions):
                            try:
                                terrain_heights[i] = scenario.get_terrain_height(float(pos[0]), float(pos[1]))
                            except Exception:
                                pass
                except Exception:
                    # 如果插值失败，尝试直接访问（兼容旧逻辑）
                    try:
                        map_size = terrain.shape[0]
                        int_pos = np.floor(positions[:, :2]).astype(np.int32)
                        valid_mask = ((int_pos[:, 0] >= 0) & (int_pos[:, 0] < map_size) & 
                                     (int_pos[:, 1] >= 0) & (int_pos[:, 1] < map_size))
                        if np.any(valid_mask):
                            valid_indices = np.where(valid_mask)[0]
                            terrain_heights[valid_indices] = terrain[int_pos[valid_indices, 1], int_pos[valid_indices, 0]]
                    except Exception:
                        pass
            else:
                # 回退：如果没有插值函数，使用直接索引（可能不准确，特别是降采样地形）
                map_size = terrain.shape[0]
                int_pos = np.floor(positions[:, :2]).astype(np.int32)
                valid_mask = ((int_pos[:, 0] >= 0) & (int_pos[:, 0] < map_size) & 
                             (int_pos[:, 1] >= 0) & (int_pos[:, 1] < map_size))
                
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    terrain_heights[valid_indices] = terrain[int_pos[valid_indices, 1], int_pos[valid_indices, 0]]
            
            penetration = terrain_heights - positions[:, 2]
            collision_mask = penetration > 0
            penalties[collision_mask] = -self.terrain_penalty_value - penetration[collision_mask] * float(self.penetration_alpha)
            if np.any(collision_mask):
                try:
                    if hasattr(world, 'agents') and 0 <= agent_idx < len(world.agents):
                        ag = world.agents[agent_idx]
                        ag._had_penetration_or_collision = True
                        ag._had_terrain_contact_or_penetration = True
                        # 🚨 新增：更新穿透计数（用于成功判定）
                        if not hasattr(ag, 'debug_info'):
                            ag.debug_info = {}
                        if not isinstance(ag.debug_info, dict):
                            ag.debug_info = {}
                        # 统计本批次中的穿透次数
                        penetration_count = np.sum(collision_mask)
                        ag.debug_info['total_penetration_count'] = ag.debug_info.get('total_penetration_count', 0) + int(penetration_count)
                except Exception:
                    pass
        
        return penalties

    def _global_reward_vectorized(self, agents: List[Any]) -> float:
        try:
            dists = []
            progresses = []
            successes = []
            valid_agents = 0
            
            # 🚨 关键修复：检测新回合并重置全局奖励标志
            # 尝试从agents中获取world和current_step
            world = None
            cur_step = None
            is_new_episode = False
            
            try:
                if agents and hasattr(agents[0], 'world'):
                    world = agents[0].world
                    if world is not None and hasattr(world, 'current_step'):
                        cur_step = int(getattr(world, 'current_step', -1))
                    else:
                        # 回退：尝试从scenario获取
                        if hasattr(agents[0], 'scenario'):
                            scenario = agents[0].scenario
                            cur_step = int(getattr(scenario, 'current_step', -1))
                    
                    # 检测新回合：步数回绕或首次观测
                    if world is not None:
                        last_seen_step = getattr(world, '_global_reward_last_step', None)
                        if last_seen_step is None:
                            # 首次观测
                            is_new_episode = True
                        elif cur_step >= 0 and last_seen_step is not None and cur_step < last_seen_step:
                            # 步数回绕，新回合开始
                            is_new_episode = True
                        elif cur_step == 0 and last_seen_step is not None and last_seen_step != 0:
                            # 步数为0且上次不是0，新回合开始
                            is_new_episode = True
                        
                        # 如果是新回合，重置全局奖励标志
                        if is_new_episode:
                            world._global_reward_given = False
                            if cur_step >= 0:
                                world._global_reward_last_step = cur_step
                        elif cur_step >= 0:
                            world._global_reward_last_step = cur_step
            except Exception:
                # 如果检测失败，假设不是新回合
                pass
            
            for i, ag in enumerate(agents):
                # 调试：检查智能体属性
                has_goal_a = hasattr(ag, 'goal_a')
                has_state = hasattr(ag, 'state')
                has_p_pos = hasattr(ag.state, 'p_pos') if has_state else False
                goal_pos_valid = hasattr(ag.goal_a, 'state') and hasattr(ag.goal_a.state, 'p_pos') and ag.goal_a.state.p_pos is not None if has_goal_a else False
                
                # 属性检查（不输出调试信息）
                
                if has_goal_a and goal_pos_valid and has_state and has_p_pos:
                    try:
                        d = np.linalg.norm(ag.state.p_pos - ag.goal_a.state.p_pos)
                        dists.append(d)
                        valid_agents += 1
                        
                        # 修复：确保last_goal_dist存在并正确初始化
                        if not hasattr(ag, 'last_goal_dist') or ag.last_goal_dist is None:
                            ag.last_goal_dist = d  # 初始化为当前距离
                        
                        progress = max(0.0, ag.last_goal_dist - d)
                        progresses.append(progress)
                        
                        # 距离和进度计算（不输出调试信息）
                        
                        ag.last_goal_dist = d  # 更新为当前距离
                        successes.append(1.0 if d <= self.success_distance_threshold else 0.0)
                        
                    except Exception as e:
                        # 属性错误处理（不输出调试信息）
                        continue
                else:
                    # 属性检查失败（不输出调试信息）
                    pass
            
            # 统计信息（不输出调试信息）
            
            if not dists:
                return 0.0
            
            # 🚨 关键修复：全局奖励也应该考虑无碰撞比例，与成功奖励保持一致
            # 计算无碰撞比例（基于回合总步数和实际碰撞次数）
            no_collision_ratio = 1.0  # 默认值：假设没有碰撞
            total_collision_count = 0
            episode_length = 2800  # 默认值
            
            try:
                # 尝试从agents中获取world和episode_length
                if world is None and agents and hasattr(agents[0], 'world'):
                    world = agents[0].world
                
                if world is not None:
                    if hasattr(world, 'episode_length') and world.episode_length is not None:
                        episode_length = int(world.episode_length)
                    elif hasattr(world, 'max_steps') and world.max_steps is not None:
                        episode_length = int(world.max_steps)
                    else:
                        import os
                        episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                        try:
                            episode_length = int(episode_length_str)
                        except (ValueError, TypeError):
                            episode_length = 2800
                    
                    # 统计所有智能体的总碰撞次数
                    if hasattr(world, 'agents') and world.agents is not None:
                        for ag in world.agents:
                            penetration_count = 0
                            if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                penetration_count = ag.debug_info.get('total_penetration_count', 0)
                                try:
                                    penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                except (ValueError, TypeError, OverflowError):
                                    penetration_count = 0
                            total_collision_count += penetration_count
                        
                        # 计算无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                        if episode_length > 0:
                            collision_ratio = float(total_collision_count) / float(episode_length)
                            no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                            
                            # 🚨 非线性映射：与成功奖励保持一致
                            # 如果碰撞比例 < 0.1（少量碰撞），使用更严厉的指数惩罚
                            if collision_ratio < 0.1:
                                no_collision_ratio = no_collision_ratio ** 4.5
                            # 如果碰撞比例 >= 0.5（碰了一半及以上），使用更严厉的惩罚
                            elif collision_ratio >= 0.5:
                                no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                no_collision_ratio = max(0.0, no_collision_ratio)
                        else:
                            no_collision_ratio = 1.0 if total_collision_count == 0 else 0.0
            except Exception:
                # 如果计算失败，默认使用1.0（假设没有碰撞）
                pass
            
            # 🚨 关键修复：全局奖励改为一次性，只在第一次所有智能体都到达目标时给一次
            if self.global_reward_mode == 'avg_progress' and progresses:
                avg_progress = float(np.mean(progresses)) * 10.0
                # avg_progress模式：只在第一次所有智能体都有进展时给一次（简化处理，主要关注success_rate模式）
                if world is not None and not getattr(world, '_global_reward_given', False):
                    world._global_reward_given = True
                    return avg_progress * no_collision_ratio
                else:
                    return 0.0
            if self.global_reward_mode == 'min_distance':
                # min_distance模式：每步都给（负奖励，用于引导）
                return float(-np.min(dists)) * no_collision_ratio
            if self.global_reward_mode == 'success_rate' and successes:
                # 🚨 关键修复：全局奖励改为一次性，只在第一次所有智能体都到达目标时给一次
                # 检查是否所有智能体都到达目标
                all_success = len(successes) > 0 and all(s == 1.0 for s in successes)
                
                if all_success and world is not None and not getattr(world, '_global_reward_given', False):
                    # 第一次所有智能体都到达目标，给一次性奖励
                    world._global_reward_given = True
                    success_rate = float(np.mean(successes))
                    return success_rate * self.success_reward_value * no_collision_ratio
                else:
                    # 已经给过奖励或不是所有智能体都到达目标，返回0
                    return 0.0
        except Exception as e:
            # 异常处理（不输出调试信息）
            return 0.0
        return 0.0

    def _clearance_reward_vectorized(self, agent: Any, world: Any, positions: np.ndarray) -> np.ndarray:
        """
        向量化净空奖励计算（统一版 - 同时考虑障碍物和地形）
        
        设计理念：
        1. 保持安全距离的奖励：当距离大于安全距离时，根据距离目标的距离动态调整权重
        2. 避障惩罚：当距离小于安全距离时，无论距离目标多远，都使用高权重惩罚
        3. 同时考虑障碍物和地形：取两者中的最小距离作为安全距离
        
        奖励组成：
        - 条件化净空奖励：根据距离目标的距离动态调整权重，防止刷分同时保持碰撞避免
        """
        # 🔧 修复：确保positions是2D数组，防止CUDA内存访问错误
        if positions.ndim == 1:
            # 1D数组，需要reshape为(1, 3)
            if len(positions) >= 3:
                positions = positions[:3].reshape(1, -1)
            else:
                # 长度不足，返回零奖励
                return np.zeros(1, dtype=np.float32)
        elif positions.ndim > 2:
            # 多维数组，展平为2D
            positions = positions.reshape(-1, positions.shape[-1])
        
        # 确保positions至少有3列（x, y, z）
        if positions.shape[-1] < 3:
            # 列数不足，返回零奖励
            return np.zeros(positions.shape[0], dtype=np.float32)
        
        rewards = np.zeros(positions.shape[0], dtype=np.float32)
        
        # 获取安全距离参数（可通过环境变量调整）
        safe_distance = float(os.getenv('OBSTACLE_SAFE_DISTANCE', '15.0'))  # 默认15米安全距离
        
        num_positions = positions.shape[0]
        min_distances = []
        
        # === 1. 计算到障碍物的距离 ===
        nearest_obstacle_centers = np.zeros((num_positions, 3), dtype=np.float32)
        nearest_obstacle_radii = np.zeros(num_positions, dtype=np.float32)
        nearest_obstacle_distances = np.full(num_positions, np.inf, dtype=np.float32)
        
        # 确保 has_obstacles 是布尔值，而不是字典或其他类型
        has_obstacles = bool(hasattr(world, 'scenario') and 
                             hasattr(world.scenario, 'obstacles') and 
                             world.scenario.obstacles)
        
        if has_obstacles:
            for obstacle_data in world.scenario.obstacles:
                if 'center' in obstacle_data and 'radius' in obstacle_data:
                    obstacle_center = np.array(obstacle_data['center'], dtype=np.float32)
                    obstacle_radius = float(obstacle_data['radius'])
                    # 计算3D距离
                    dist_3d = np.linalg.norm(positions - obstacle_center, axis=-1)
                    dist_to_surface = dist_3d - obstacle_radius  # 到障碍物表面的距离
                    min_distances.append(dist_to_surface)
                    
                    # 更新最近障碍物信息（向量化）
                    mask = dist_to_surface < nearest_obstacle_distances
                    nearest_obstacle_centers[mask] = obstacle_center
                    nearest_obstacle_radii[mask] = obstacle_radius
                    nearest_obstacle_distances[mask] = dist_to_surface[mask]
        
        # === 2. 计算到地形的距离（同一高度的XY平面距离）===
        # 🚨 关键修复：不使用高度差，而是使用同一高度的XY平面距离
        # 原因：智能体可能从侧面贴着山地飞行，此时高度差很大但实际距离很近（会相撞）
        # 正确方法：在智能体高度z处，找到地形轮廓的XY位置，计算水平距离
        # 🚨 GPU加速优化：使用向量化方法，批量计算所有采样点，避免Python循环
        terrain_distances = np.full(num_positions, np.inf, dtype=np.float32)
        terrain_heights = np.zeros(num_positions, dtype=np.float32)
        has_terrain = False
        
        scenario = getattr(world, 'scenario', None)
        cached_data = getattr(self, '_cache', {})
        terrain = cached_data.get('terrain') if isinstance(cached_data, dict) else None
        
        # 🚨 GPU加速优化：向量化计算同一高度的XY平面距离
        if scenario is not None and hasattr(scenario, 'get_terrain_height'):
            has_terrain = True
            # 采样方向数（8个主方向 + 8个对角线方向 = 16个方向）
            directions = np.array([
                [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],  # 四个主方向
                [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707],  # 四个对角线
                [0.923, 0.383], [-0.923, 0.383], [0.923, -0.383], [-0.923, -0.383],  # 额外方向
                [0.383, 0.923], [-0.383, 0.923], [0.383, -0.923], [-0.383, -0.923]
            ], dtype=np.float32)
            n_directions = len(directions)
            
            # 搜索半径范围（从近到远）
            search_radii = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 50.0], dtype=np.float32)
            n_radii = len(search_radii)
            
            try:
                map_size = float(getattr(scenario, 'map_size', 200.0))
                agent_zs = positions[:, 2].astype(np.float32)  # (num_positions,)
                agent_xys = positions[:, :2].astype(np.float32)  # (num_positions, 2)
                
                # 🚨 向量化：生成所有采样点的XY坐标
                # 形状：(num_positions, n_directions, n_radii, 2)
                # 使用广播：agent_xys[:, None, None, :] + directions[None, :, None, :] * search_radii[None, None, :, None]
                sample_xys = agent_xys[:, None, None, :] + directions[None, :, None, :] * search_radii[None, None, :, None]
                # 展平为 (num_positions * n_directions * n_radii, 2)
                sample_xys_flat = sample_xys.reshape(-1, 2)
                
                # 检查是否在地图范围内
                valid_mask = (sample_xys_flat[:, 0] >= 0) & (sample_xys_flat[:, 0] < map_size) & \
                            (sample_xys_flat[:, 1] >= 0) & (sample_xys_flat[:, 1] < map_size)
                
                # 🚨 GPU加速：批量获取所有采样点的地形高度
                if hasattr(scenario, 'get_terrain_height_vectorized') and np.any(valid_mask):
                    # 只计算有效范围内的点
                    valid_sample_xys = sample_xys_flat[valid_mask]
                    try:
                        # 批量获取地形高度
                        sample_terrain_heights_flat = scenario.get_terrain_height_vectorized(
                            valid_sample_xys[:, 0], valid_sample_xys[:, 1]
                        )
                        if sample_terrain_heights_flat.ndim == 0:
                            sample_terrain_heights_flat = np.full(len(valid_sample_xys), float(sample_terrain_heights_flat), dtype=np.float32)
                        else:
                            sample_terrain_heights_flat = sample_terrain_heights_flat.astype(np.float32)
                        
                        # 重建完整数组（无效位置设为-inf）
                        sample_terrain_heights_full = np.full(len(sample_xys_flat), -np.inf, dtype=np.float32)
                        sample_terrain_heights_full[valid_mask] = sample_terrain_heights_flat
                    except Exception:
                        # 如果向量化失败，回退到逐个调用
                        sample_terrain_heights_full = np.full(len(sample_xys_flat), -np.inf, dtype=np.float32)
                        for i, (xy, valid) in enumerate(zip(sample_xys_flat, valid_mask)):
                            if valid:
                                try:
                                    sample_terrain_heights_full[i] = scenario.get_terrain_height(float(xy[0]), float(xy[1]))
                                except Exception:
                                    pass
                else:
                    # 回退：逐个调用
                    sample_terrain_heights_full = np.full(len(sample_xys_flat), -np.inf, dtype=np.float32)
                    for i, (xy, valid) in enumerate(zip(sample_xys_flat, valid_mask)):
                        if valid:
                            try:
                                sample_terrain_heights_full[i] = scenario.get_terrain_height(float(xy[0]), float(xy[1]))
                            except Exception:
                                pass
                
                # 重塑为 (num_positions, n_directions, n_radii)
                sample_terrain_heights = sample_terrain_heights_full.reshape(num_positions, n_directions, n_radii)
                sample_xys_reshaped = sample_xys_flat.reshape(num_positions, n_directions, n_radii, 2)
                
                # 🚨 向量化：找到地形高度等于或接近智能体高度z的点（容差1米）
                agent_zs_expanded = agent_zs[:, None, None]  # (num_positions, 1, 1)
                height_diff = np.abs(sample_terrain_heights - agent_zs_expanded)  # (num_positions, n_directions, n_radii)
                match_mask = height_diff < 1.0  # 容差1米
                
                # 计算XY距离
                agent_xys_expanded = agent_xys[:, None, None, :]  # (num_positions, 1, 1, 2)
                xy_distances = np.linalg.norm(sample_xys_reshaped - agent_xys_expanded, axis=-1)  # (num_positions, n_directions, n_radii)
                
                # 🚨 向量化优化：对于每个智能体，在每个方向上找到第一个匹配点（最近的点）
                # 使用argmax找到每个方向上第一个匹配的半径索引（True的索引）
                # 将不匹配的位置的距离设为inf，然后取最小值
                xy_distances_masked = np.where(match_mask, xy_distances, np.inf)  # (num_positions, n_directions, n_radii)
                # 在每个方向上取最小值（第一个匹配点就是最近的）
                min_distances_per_direction = np.min(xy_distances_masked, axis=2)  # (num_positions, n_directions)
                # 取所有方向的最小值
                terrain_distances = np.min(min_distances_per_direction, axis=1)  # (num_positions,)
                
                # 如果找不到匹配点，回退到高度差方法
                fallback_mask = ~np.isfinite(terrain_distances) | (terrain_distances == np.inf)
                if np.any(fallback_mask):
                    try:
                        if hasattr(scenario, 'get_terrain_height_vectorized'):
                            fallback_positions = positions[fallback_mask]
                            terrain_heights_fallback = scenario.get_terrain_height_vectorized(
                                fallback_positions[:, 0], fallback_positions[:, 1]
                            )
                            if terrain_heights_fallback.ndim == 0:
                                terrain_heights_fallback = np.full(np.sum(fallback_mask), float(terrain_heights_fallback), dtype=np.float32)
                            else:
                                terrain_heights_fallback = terrain_heights_fallback.astype(np.float32)
                            terrain_distances[fallback_mask] = fallback_positions[:, 2] - terrain_heights_fallback
                        else:
                            for i in np.where(fallback_mask)[0]:
                                try:
                                    terrain_heights[i] = scenario.get_terrain_height(float(positions[i, 0]), float(positions[i, 1]))
                                    terrain_distances[i] = positions[i, 2] - terrain_heights[i]
                                except Exception:
                                    terrain_distances[i] = np.inf
                    except Exception:
                        pass
            except Exception:
                # 如果向量化计算失败，回退到高度差方法
                try:
                    if hasattr(scenario, 'get_terrain_height_vectorized'):
                        terrain_heights_raw = scenario.get_terrain_height_vectorized(positions[:, 0], positions[:, 1])
                        if terrain_heights_raw.ndim == 0:
                            terrain_heights = np.full(num_positions, float(terrain_heights_raw), dtype=np.float32)
                        elif terrain_heights_raw.shape[0] != num_positions:
                            terrain_heights = np.zeros(num_positions, dtype=np.float32)
                        else:
                            terrain_heights = terrain_heights_raw.astype(np.float32)
                        terrain_distances = positions[:, 2] - terrain_heights
                    else:
                        for i, pos in enumerate(positions):
                            try:
                                terrain_heights[i] = scenario.get_terrain_height(float(pos[0]), float(pos[1]))
                                terrain_distances[i] = positions[i, 2] - terrain_heights[i]
                            except Exception:
                                terrain_distances[i] = np.inf
                except Exception:
                    pass
        
        # === 3. 合并障碍物和地形的距离，取最小值 ===
        if min_distances:
            try:
                # 🔧 修复：确保min_distances是2D数组，防止形状不匹配
                min_distances_array = np.array(min_distances)  # (n_obstacles, num_positions)
                if min_distances_array.ndim == 1:
                    # 如果只有一个障碍物，min_distances_array是1D，需要reshape
                    obstacle_min_dist = min_distances_array.reshape(-1)
                else:
                    # 多个障碍物，沿着axis=0取最小值
                    obstacle_min_dist = np.min(min_distances_array, axis=0)  # (num_positions,)
                
                # 🔧 修复：确保obstacle_min_dist的形状正确
                if obstacle_min_dist.ndim == 0:
                    # 标量，需要扩展
                    obstacle_min_dist = np.full(num_positions, float(obstacle_min_dist), dtype=np.float32)
                elif obstacle_min_dist.shape[0] != num_positions:
                    # 形状不匹配，使用默认值
                    obstacle_min_dist = np.full(num_positions, np.inf, dtype=np.float32)
            except Exception:
                # 计算失败，使用默认值
                obstacle_min_dist = np.full(num_positions, np.inf, dtype=np.float32)
        else:
            obstacle_min_dist = np.full(num_positions, np.inf, dtype=np.float32)
        
        # 🔧 修复：确保terrain_distances的形状正确
        if terrain_distances.shape[0] != num_positions:
            terrain_distances = np.full(num_positions, np.inf, dtype=np.float32)
        
        # 合并：取障碍物距离和地形距离的最小值
        d_min_current = np.minimum(obstacle_min_dist, terrain_distances)
        
        # 🔧 修复：确保d_min_current的形状正确
        if d_min_current.ndim == 0:
            d_min_current = np.full(num_positions, float(d_min_current), dtype=np.float32)
        elif d_min_current.shape[0] != num_positions:
            d_min_current = np.full(num_positions, np.inf, dtype=np.float32)
        
        # === 4. 确定最近的危险源（用于计算向上绕行特性）===
        # 🔧 修复：确保所有中间变量的形状正确
        # 🔧 XLA友好优化：使用向量化操作，避免Python循环
        # 如果地形距离更近，使用地形作为参考；否则使用障碍物
        try:
            use_terrain_as_reference = terrain_distances < obstacle_min_dist
            # 确保use_terrain_as_reference是1D布尔数组
            if use_terrain_as_reference.ndim == 0:
                use_terrain_as_reference = np.full(num_positions, bool(use_terrain_as_reference), dtype=bool)
            elif use_terrain_as_reference.shape[0] != num_positions:
                use_terrain_as_reference = np.zeros(num_positions, dtype=bool)
        except Exception:
            use_terrain_as_reference = np.zeros(num_positions, dtype=bool)
        
        # 初始化参考点（障碍物或地形）- 向量化构建
        # 地形参考点：x, y使用智能体位置，z使用地形高度
        try:
            terrain_reference_centers = np.zeros((num_positions, 3), dtype=np.float32)
            terrain_reference_centers[:, 0] = positions[:, 0]  # x
            terrain_reference_centers[:, 1] = positions[:, 1]  # y
            terrain_reference_centers[:, 2] = terrain_heights  # z（地形高度）
        except Exception:
            terrain_reference_centers = np.zeros((num_positions, 3), dtype=np.float32)
        
        # 使用np.where进行向量化选择
        # 如果地形距离更近且地形可用，使用地形参考；否则使用障碍物参考
        # 确保 has_terrain 和 has_obstacles 是布尔值，然后与数组进行广播操作
        has_terrain_bool = bool(has_terrain)
        has_obstacles_bool = bool(has_obstacles)
        try:
            terrain_mask = use_terrain_as_reference & has_terrain_bool
            obstacle_mask = (~use_terrain_as_reference) & has_obstacles_bool
            default_terrain_mask = (~terrain_mask) & (~obstacle_mask) & has_terrain_bool
            
            # 确保所有mask都是1D布尔数组
            if terrain_mask.ndim == 0:
                terrain_mask = np.full(num_positions, bool(terrain_mask), dtype=bool)
            elif terrain_mask.shape[0] != num_positions:
                terrain_mask = np.zeros(num_positions, dtype=bool)
            
            if obstacle_mask.ndim == 0:
                obstacle_mask = np.full(num_positions, bool(obstacle_mask), dtype=bool)
            elif obstacle_mask.shape[0] != num_positions:
                obstacle_mask = np.zeros(num_positions, dtype=bool)
            
            if default_terrain_mask.ndim == 0:
                default_terrain_mask = np.full(num_positions, bool(default_terrain_mask), dtype=bool)
            elif default_terrain_mask.shape[0] != num_positions:
                default_terrain_mask = np.zeros(num_positions, dtype=bool)
        except Exception:
            terrain_mask = np.zeros(num_positions, dtype=bool)
            obstacle_mask = np.zeros(num_positions, dtype=bool)
            default_terrain_mask = np.zeros(num_positions, dtype=bool)
        
        # 向量化构建参考点
        try:
            reference_centers = np.where(
                terrain_mask[:, None] | default_terrain_mask[:, None],
                terrain_reference_centers,
                nearest_obstacle_centers
            )
            # 确保reference_centers是2D数组
            if reference_centers.ndim == 1:
                reference_centers = reference_centers.reshape(1, -1)
            elif reference_centers.shape[0] != num_positions:
                reference_centers = np.zeros((num_positions, 3), dtype=np.float32)
            
            reference_heights = np.where(
                terrain_mask | default_terrain_mask,
                terrain_heights,
                nearest_obstacle_centers[:, 2]
            )
            # 确保reference_heights是1D数组
            if reference_heights.ndim == 0:
                reference_heights = np.full(num_positions, float(reference_heights), dtype=np.float32)
            elif reference_heights.shape[0] != num_positions:
                reference_heights = np.zeros(num_positions, dtype=np.float32)
        except Exception:
            reference_centers = np.zeros((num_positions, 3), dtype=np.float32)
            reference_heights = np.zeros(num_positions, dtype=np.float32)
        
        # === 5. 🔧 条件化净空奖励：根据距离目标的距离动态调整权重 ===
        # 问题：高权重净空奖励（3.5）可以防止APF撞地形，但会被"仅有动作"方法利用来刷分
        # 解决：距离目标远时降低权重（防止刷分），距离目标近时提高权重（防止撞地形）
        goal_pos = None
        try:
            if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                goal_pos_raw = agent.goal_a.state.p_pos
                goal_pos = np.asarray(goal_pos_raw, dtype=np.float32)
                # 🔧 修复：确保goal_pos是1D数组(3,)，而不是标量或其他形状
                if goal_pos.ndim == 0:
                    # 标量，无法使用
                    goal_pos = None
                elif goal_pos.ndim > 1:
                    # 多维数组，取前3个元素并展平
                    goal_pos = goal_pos.flatten()[:3]
                elif len(goal_pos) < 3:
                    # 长度不足，无法使用
                    goal_pos = None
                else:
                    # 确保是1D数组(3,)
                    goal_pos = goal_pos[:3].flatten()
        except Exception:
            goal_pos = None
        
        if goal_pos is None:
            try:
                scenario = getattr(world, 'scenario', None)
                if scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                    goal_pos_raw = scenario.goal_pos
                    goal_pos = np.asarray(goal_pos_raw, dtype=np.float32)
                    # 🔧 修复：确保goal_pos是1D数组(3,)
                    if goal_pos.ndim == 0:
                        goal_pos = None
                    elif goal_pos.ndim > 1:
                        goal_pos = goal_pos.flatten()[:3]
                    elif len(goal_pos) < 3:
                        goal_pos = None
                    else:
                        goal_pos = goal_pos[:3].flatten()
            except Exception:
                goal_pos = None
        
        # 计算每个位置到目标的距离
        # 🔧 注意：positions已经在函数开始处被确保为2D数组 (num_positions, 3)
        if goal_pos is not None and len(goal_pos) >= 3:
            try:
                # positions形状: (num_positions, 3)，已经在函数开始处确保
                # goal_pos形状: (3,)
                # 确保positions的最后一维是3
                if positions.shape[-1] >= 3:
                    positions_3d = positions[..., :3]
                    # 计算距离：positions_3d (num_positions, 3) - goal_pos (3,) -> (num_positions, 3)
                    # 然后计算norm得到 (num_positions,)
                    dists_to_goal = np.linalg.norm(positions_3d - goal_pos, axis=-1)
                else:
                    # 形状不匹配，使用默认值
                    dists_to_goal = np.full(positions.shape[0], 100.0, dtype=np.float32)
            except Exception:
                # 计算失败，使用默认值
                dists_to_goal = np.full(positions.shape[0], 100.0, dtype=np.float32)
        else:
            # 如果没有目标，使用默认权重（保守，假设距离较远）
            dists_to_goal = np.full(positions.shape[0], 100.0, dtype=np.float32)
        
        # 动态权重计算
        FAR_THRESHOLD = float(os.getenv('CLEARANCE_FAR_THRESHOLD', '50.0'))  # 远距离阈值（米）
        NEAR_THRESHOLD = float(os.getenv('CLEARANCE_NEAR_THRESHOLD', '20.0'))  # 近距离阈值（米）
        WEIGHT_FAR = float(os.getenv('CLEARANCE_WEIGHT_FAR', '0.5'))  # 远距离权重（防止刷分）
        WEIGHT_NEAR = float(os.getenv('CLEARANCE_WEIGHT_NEAR', '5.0'))  # 近距离权重（防止撞地形）
        
        # 🔧 修复：确保dists_to_goal的长度与positions匹配
        # positions已经在函数开始处被确保为2D数组，所以直接使用shape[0]
        num_pos = positions.shape[0]
        
        # 确保dists_to_goal的长度匹配
        if len(dists_to_goal) != num_pos:
            # 长度不匹配，重新创建dists_to_goal
            dists_to_goal = np.full(num_pos, 100.0, dtype=np.float32)
        
        # 向量化计算动态权重
        try:
            far_mask = dists_to_goal > FAR_THRESHOLD
            near_mask = dists_to_goal < NEAR_THRESHOLD
            transition_mask = ~(far_mask | near_mask)
            
            dynamic_weights = np.zeros(num_pos, dtype=np.float32)
            dynamic_weights[far_mask] = WEIGHT_FAR
            dynamic_weights[near_mask] = WEIGHT_NEAR
            
            # 过渡区域：线性插值
            if np.any(transition_mask):
                ratio = (dists_to_goal[transition_mask] - NEAR_THRESHOLD) / (FAR_THRESHOLD - NEAR_THRESHOLD)
                dynamic_weights[transition_mask] = WEIGHT_NEAR - ratio * (WEIGHT_NEAR - WEIGHT_FAR)
        except Exception:
            # 计算失败，使用默认权重
            dynamic_weights = np.full(num_pos, WEIGHT_FAR, dtype=np.float32)
        
        # 🔧 关键修复：分离"避障"和"保持安全距离"
        # 避障（距离<安全距离）：无论距离目标多远，都使用高权重惩罚
        # 保持安全距离（距离>安全距离）：根据距离目标距离动态调整权重
        PENALTY_WEIGHT = float(os.getenv('CLEARANCE_PENALTY_WEIGHT', '5.0'))  # 避障惩罚权重（固定高权重）
        CLEARANCE_WEIGHT = float(os.getenv('CLEARANCE_WEIGHT', '3.5'))  # 净空奖励基础权重
        
        # 🔧 修复：确保所有数组形状匹配，防止CUDA内存访问错误
        # d_min_current形状: (num_positions,)
        # dynamic_weights形状: (num_positions,)
        # 需要确保它们都有一致的形状
        
        # 确保d_min_current的形状正确
        try:
            # 检查d_min_current的形状
            if d_min_current.ndim == 0:
                # 标量，需要扩展
                d_min_current = np.full(num_pos, float(d_min_current), dtype=np.float32)
            elif d_min_current.shape[0] != num_pos:
                # 形状不匹配，使用默认值
                rewards = np.zeros(num_pos, dtype=np.float32)
                # 更新last_min_distance后返回
                if not hasattr(agent, 'last_min_distance'):
                    agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
                else:
                    d_min_previous = agent.last_min_distance
                    if isinstance(d_min_previous, np.ndarray) and len(d_min_previous) != len(d_min_current):
                        agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
                    else:
                        agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
                return rewards
            
            # 检查dynamic_weights的形状
            if dynamic_weights.shape[0] != num_pos:
                # 形状不匹配，重新创建dynamic_weights
                dynamic_weights = np.full(num_pos, WEIGHT_FAR, dtype=np.float32)
            
            # 🚨 关键改进：从"基于绝对距离"改为"基于距离变化"
            # 核心思想：
            # 1. 距离增加时：给正奖励（鼓励避障）
            # 2. 距离减少时：给负奖励（惩罚接近危险）
            # 3. 距离不变时：给零奖励（避免刷分）
            # 这样既能保持避障引导，又能避免"保持安全距离但不接近目标"的刷分行为
            
            # 🚨 关键修复：防止除以零和NaN/Inf
            # 确保safe_distance不为0或NaN
            safe_distance_safe = max(float(safe_distance), 1e-6)  # 防止除以零
            
            # 🚨 关键修复：防止NaN/Inf传播
            d_min_current = np.nan_to_num(d_min_current, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 获取上一时刻的最小距离
            if not hasattr(agent, 'last_min_distance'):
                # 首次调用，初始化
                agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
                d_min_previous = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
            else:
                d_min_previous = agent.last_min_distance
                # 确保形状匹配
                if isinstance(d_min_previous, np.ndarray):
                    if d_min_previous.shape[0] != num_pos:
                        d_min_previous = np.full(num_pos, float(d_min_previous[0]) if len(d_min_previous) > 0 else 0.0, dtype=np.float32)
                else:
                    d_min_previous = np.full(num_pos, float(d_min_previous), dtype=np.float32)
            
            # 清理d_min_previous中的NaN/Inf
            d_min_previous = np.nan_to_num(d_min_previous, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 计算距离变化
            distance_change = d_min_current - d_min_previous
            
            # 归一化距离变化
            clearance_d_max = float(os.getenv('CLEARANCE_D_MAX', '80.0'))
            clearance_d_max_safe = max(clearance_d_max, 1e-6)  # 防止除以零
            normalized_change = np.clip(distance_change / clearance_d_max_safe, -1.0, 1.0)
            
            # 清理normalized_change中的NaN/Inf
            normalized_change = np.nan_to_num(normalized_change, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 🚨 关键改进：基于距离变化计算奖励
            # 距离增加时：给正奖励（鼓励避障）
            # 距离减少时：给负奖励（惩罚接近危险，惩罚因子2.0）
            # 距离不变时：给零奖励（避免刷分）
            distance_increase_mask = distance_change > 1e-6  # 距离增加（容差1e-6米）
            distance_decrease_mask = distance_change < -1e-6  # 距离减少（容差1e-6米）
            distance_unchanged_mask = ~(distance_increase_mask | distance_decrease_mask)  # 距离不变
            
            # 初始化奖励
            clearance_reward_base = np.zeros(num_pos, dtype=np.float32)
            
            # 距离增加：给正奖励（鼓励避障）
            if np.any(distance_increase_mask):
                clearance_reward_base[distance_increase_mask] = dynamic_weights[distance_increase_mask] * normalized_change[distance_increase_mask]
            
            # 距离减少：给负奖励（惩罚接近危险，惩罚因子2.0）
            if np.any(distance_decrease_mask):
                penalty_factor = 2.0  # 惩罚因子，让接近危险的行为受到更严厉的惩罚
                clearance_reward_base[distance_decrease_mask] = dynamic_weights[distance_decrease_mask] * normalized_change[distance_decrease_mask] * penalty_factor
            
            # 距离不变：给零奖励（避免刷分）
            if np.any(distance_unchanged_mask):
                clearance_reward_base[distance_unchanged_mask] = 0.0
            
            # 所有形状匹配，正常计算
            # 🚨 关键修复：防止除以零（safe_distance可能为0）
            safe_distance_for_penalty = max(float(safe_distance), 1e-6)
            rewards = np.where(
                d_min_current < safe_distance_for_penalty,
                # 避障：固定高权重惩罚（无论距离目标多远）
                -PENALTY_WEIGHT * (1.0 - d_min_current / safe_distance_for_penalty),
                # 保持安全距离：基于距离变化（而非绝对距离）
                clearance_reward_base
            )
            # 🚨 关键修复：确保最终奖励是有限值
            # 限制奖励范围，防止CLEARANCE_WEIGHT=16.0时奖励过大
            max_reward = CLEARANCE_WEIGHT * 2.0  # 允许奖励达到权重的2倍
            min_reward = -PENALTY_WEIGHT * 2.0  # 允许惩罚达到权重的2倍
            rewards = np.clip(rewards, min_reward, max_reward)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=max_reward, neginf=min_reward)
        except Exception as e:
            # 计算失败，使用默认值
            rewards = np.zeros(num_pos, dtype=np.float32)
        
        # 🚨 关键改进：更新上一时刻的最小距离（用于下次计算距离变化）
        # 注意：这个更新应该在计算奖励之后，确保下次调用时能正确计算距离变化
        if not hasattr(agent, 'last_min_distance'):
            agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
        else:
            # 确保形状匹配
            if isinstance(agent.last_min_distance, np.ndarray):
                if agent.last_min_distance.shape[0] != num_pos:
                    agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
                else:
                    agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
            else:
                agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
        
        # 🚨 关键修复：同时更新debug_info中的d_min_current（用于数据收集）
        # 取最后一个值（当前时刻的值）作为标量存储
        try:
            if not hasattr(agent, 'debug_info'):
                agent.debug_info = {}
            if not isinstance(agent.debug_info, dict):
                agent.debug_info = {}
            # 如果d_min_current是数组，取最后一个值；否则直接使用
            if isinstance(d_min_current, np.ndarray) and d_min_current.size > 0:
                d_min_scalar = float(d_min_current[-1] if d_min_current.ndim > 0 else d_min_current.item())
            else:
                d_min_scalar = float(d_min_current) if not isinstance(d_min_current, np.ndarray) else float(d_min_current.item())
            agent.debug_info['d_min_current'] = d_min_scalar
        except Exception:
            # 如果更新失败，不影响奖励计算
            pass
        
        return rewards

    def _potential_shaping_vectorized(self, agent: Any, dist_to_goal: float) -> float:
        try:
            if not hasattr(agent, '_phi_last'):
                agent._phi_last = -dist_to_goal * 0.01
                return 0.0
            phi_now = -dist_to_goal * 0.01
            r_shape = self.shaping_gamma * phi_now - agent._phi_last
            agent._phi_last = phi_now
            return float(r_shape)
        except Exception:
            return 0.0
    
    def get_reward_breakdown(self, agents_batch: List[List[Any]], world_batch: List[Any], scenario_batch: List[Any] = None) -> Dict[str, np.ndarray]:
        """获取奖励分项详情（用于调试和分析）"""
        batch_size = len(agents_batch)
        n_agents = len(agents_batch[0]) if batch_size > 0 else 0
        
        if batch_size == 0 or n_agents == 0:
            return {}
        
        # 如果没有提供 scenario_batch，从 world_batch 中提取
        if scenario_batch is None:
            scenario_batch = []
            for world in world_batch:
                if hasattr(world, 'scenario'):
                    scenario_batch.append(world.scenario)
                else:
                    scenario_batch.append(None)
        
        arrays = self._get_preallocated_arrays(batch_size, n_agents)
        self._extract_batch_states(agents_batch, arrays)
        self._calculate_all_rewards_vectorized(agents_batch, world_batch, scenario_batch, arrays)
        
        breakdown = {}
        for i, name in enumerate(self.reward_names):
            breakdown[name] = arrays['rewards'][:, :, i]
        
        return breakdown
    
    def _check_reward_diversity(self, rewards: np.ndarray, episode: int = 0):
        """检测奖励多样性，如果缺乏多样性则输出警告（带冷却期）"""
        try:
            # 计算平均奖励
            avg_reward = np.mean(rewards)
            
            # 添加到历史记录
            self.reward_history.append(avg_reward)
            
            # 保持窗口大小
            if len(self.reward_history) > self.diversity_window:
                self.reward_history = self.reward_history[-self.diversity_window:]
            
            # 检查冷却期，避免频繁警告
            if episode - self.last_warning_episode < self.warning_cooldown:
                return
            
            # 如果历史记录足够，检测多样性
            if len(self.reward_history) >= self.diversity_window:
                recent_rewards = np.array(self.reward_history)
                
                # 计算奖励的标准差
                reward_std = np.std(recent_rewards)
                reward_mean = np.mean(recent_rewards)
                
                # 计算变异系数（标准差/均值）
                if abs(reward_mean) > 1e-6:
                    cv = reward_std / abs(reward_mean)
                else:
                    cv = float('inf')
                
                # 调整检测阈值，使其更适合实际奖励范围
                # 对于小奖励值（-10到+10），使用更宽松的阈值
                if abs(reward_mean) < 10:
                    # 小奖励值：变异系数小于2%或标准差小于0.3
                    diversity_threshold_cv = 0.02
                    diversity_threshold_std = 0.3
                else:
                    # 大奖励值：变异系数小于1%或标准差小于100
                    diversity_threshold_cv = 0.01
                    diversity_threshold_std = 100
                
                # 检查是否真的停滞（连续多个窗口都满足条件）
                is_stagnant = False
                if cv < diversity_threshold_cv or reward_std < diversity_threshold_std:
                    # 检查是否连续多个窗口都满足停滞条件
                    if len(self.reward_history) >= self.diversity_window * 2:
                        # 检查前一个窗口是否也满足停滞条件
                        prev_window = self.reward_history[-self.diversity_window*2:-self.diversity_window]
                        if len(prev_window) >= self.diversity_window:
                            prev_std = np.std(prev_window)
                            prev_mean = np.mean(prev_window)
                            if abs(prev_mean) > 1e-6:
                                prev_cv = prev_std / abs(prev_mean)
                            else:
                                prev_cv = float('inf')
                            
                            # 如果前一个窗口也停滞，才认为是真正的停滞
                            if (prev_cv < diversity_threshold_cv or prev_std < diversity_threshold_std):
                                is_stagnant = True
                    else:
                        # 历史记录不够，暂时不判断为停滞
                        is_stagnant = False
                
                # 只有在真正停滞时才输出警告
                if is_stagnant:
                    if cv < diversity_threshold_cv:
                        print(f"\n⚠️  奖励多样性警告: 最近{self.diversity_window}次奖励变异系数过低 ({cv:.6f})")
                        print(f"   平均奖励: {reward_mean:.2f}, 标准差: {reward_std:.2f}")
                        print(f"   建议: 增加探索噪声或调整奖励权重\n")
                    
                    if reward_std < diversity_threshold_std:
                        print(f"\n⚠️  奖励停滞警告: 最近{self.diversity_window}次奖励标准差过小 ({reward_std:.2f})")
                        print(f"   平均奖励: {reward_mean:.2f}")
                        print(f"   建议: 检查智能体是否陷入局部最优\n")
                    
                    # 更新警告时间
                    self.last_warning_episode = episode
                    
        except Exception as e:
            # 静默处理异常，避免影响主流程
            pass
    
    def clear_cache(self):
        """清理缓存以释放内存"""
        self._cache.clear()

    def reset_agent_state(self, agent: Any):
        """重置单个智能体的内部状态，用于新回合开始时"""
        # 要重置的属性列表
        attributes_to_reset = [
            'visited_cells',
            'cell_visit_counts',
            'random_exploration_counter',
            'last_position',
            'stationary_count',
            'start_position',
            'last_min_distance',
            '_phi_last'
        ]
        for attr in attributes_to_reset:
            if hasattr(agent, attr):
                delattr(agent, attr)

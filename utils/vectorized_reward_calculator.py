#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
numpy向量化奖励计算器
针对并行环境进行性能优化，使用批量计算提升效率
"""

import numpy as np
import os
import time
from typing import Dict, List, Tuple, Any


def _bilinear_interpolate_terrain_xy(
    terrain: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    terrain_heights: np.ndarray,
    map_size: int,
):
    """双线性插值获取地形高度（xy 专用向量化实现）。"""
    if terrain is None or x_coords is None or y_coords is None or len(x_coords) == 0:
        return

    x_coords = np.asarray(x_coords, dtype=np.float32).reshape(-1)
    y_coords = np.asarray(y_coords, dtype=np.float32).reshape(-1)
    if x_coords.shape[0] != y_coords.shape[0]:
        return

    terrain_h, terrain_w = terrain.shape[0], terrain.shape[1]
    if terrain_h != map_size or terrain_w != map_size:
        scale_x = float(terrain_w) / float(map_size)
        scale_y = float(terrain_h) / float(map_size)
        x_coords = np.clip(x_coords * scale_x, 0.0, float(terrain_w - 1))
        y_coords = np.clip(y_coords * scale_y, 0.0, float(terrain_h - 1))
    else:
        x_coords = np.clip(x_coords, 0.0, float(map_size - 1))
        y_coords = np.clip(y_coords, 0.0, float(map_size - 1))

    x_low = np.floor(x_coords).astype(np.int32)
    y_low = np.floor(y_coords).astype(np.int32)
    x_high = np.minimum(x_low + 1, terrain_w - 1)
    y_high = np.minimum(y_low + 1, terrain_h - 1)

    x_low = np.clip(x_low, 0, terrain_w - 1)
    y_low = np.clip(y_low, 0, terrain_h - 1)
    x_high = np.clip(x_high, 0, terrain_w - 1)
    y_high = np.clip(y_high, 0, terrain_h - 1)

    x_weight = x_coords - x_low.astype(np.float32)
    y_weight = y_coords - y_low.astype(np.float32)

    h00 = terrain[y_low, x_low]
    h10 = terrain[y_low, x_high]
    h01 = terrain[y_high, x_low]
    h11 = terrain[y_high, x_high]

    valid_mask = np.isfinite(h00) & np.isfinite(h10) & np.isfinite(h01) & np.isfinite(h11)
    if not np.all(valid_mask):
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            h00[invalid_mask] = np.where(np.isfinite(h00[invalid_mask]), h00[invalid_mask],
                                        np.where(np.isfinite(h10[invalid_mask]), h10[invalid_mask],
                                                np.where(np.isfinite(h01[invalid_mask]), h01[invalid_mask], h11[invalid_mask])))
            h10[invalid_mask] = np.where(np.isfinite(h10[invalid_mask]), h10[invalid_mask], h00[invalid_mask])
            h01[invalid_mask] = np.where(np.isfinite(h01[invalid_mask]), h01[invalid_mask], h00[invalid_mask])
            h11[invalid_mask] = np.where(np.isfinite(h11[invalid_mask]), h11[invalid_mask], h00[invalid_mask])

    h0 = h00 * (1.0 - x_weight) + h10 * x_weight
    h1 = h01 * (1.0 - x_weight) + h11 * x_weight
    terrain_heights[:] = h0 * (1.0 - y_weight) + h1 * y_weight

    invalid_output = ~np.isfinite(terrain_heights)
    if np.any(invalid_output):
        x_nearest = np.clip(x_low[invalid_output], 0, terrain_w - 1)
        y_nearest = np.clip(y_low[invalid_output], 0, terrain_h - 1)
        terrain_heights[invalid_output] = terrain[y_nearest, x_nearest]


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
    
    _bilinear_interpolate_terrain_xy(
        terrain,
        positions_2d[:, 0],
        positions_2d[:, 1],
        terrain_heights,
        map_size,
    )


class VectorizedRewardCalculator:
    """numpy向量化奖励计算器"""
    REWARD_TIMING_KEYS = (
        'rew_cache',
        'rew_state',
        'rew_numeric',
        'rew_explore',
        'rew_motion',
        'rew_success',
        'rew_collision',
        'rew_clearance',
        'rew_lateral',
        'rew_team',
        'rew_reduce',
    )
    
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
        # 将权重转换为numpy数组，按固定顺序排列（16通道，必须与分项索引严格一致）
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
            reward_weights.get('clearance', 0.0),# 13
            reward_weights.get('lateral', 0.0),  # 14
            reward_weights.get('collision_reduction', 0.0)  # 15
        ], dtype=np.float32)
        self._base_reward_weights = self.reward_weights.copy()
        
        self.max_reward = max_reward
        self.min_reward = min_reward
        
        # 预分配数组缓存和性能优化
        self._cache = {}
        self._obstacle_cache = {}  # 障碍物数据缓存
        self._terrain_cache = {}   # 地形数据缓存
        self._goal_cache = {}      # 目标数据缓存
        self._terrain_distance_scratch = {}
        
        # 其他参数（预转换为合适类型避免运行时转换）
        self.success_reward_value = np.float32(success_reward_value)
        self.success_distance_threshold = np.float32(success_distance_threshold)
        self.collision_penalty_value = np.float32(collision_penalty_value)
        self.collision_distance_threshold = np.float32(collision_distance_threshold)
        self.no_collision_reward_value = np.float32(no_collision_reward_value)
        self.global_reward_mode = str(global_reward_mode)
        self.shaping_gamma = np.float32(shaping_gamma)
        try:
            raw_reward_profile = kwargs.get(
                'reward_profile',
                os.getenv('CURRICULUM_REWARD_PROFILE', os.getenv('REWARD_PROFILE', 'fixed_route'))
            )
            self.reward_profile = self._normalize_reward_profile_name(raw_reward_profile)
            self.curriculum_stage_id = int(kwargs.get('curriculum_stage_id', os.getenv('CURRICULUM_STAGE_ID', '0')))
        except Exception:
            self.reward_profile = 'fixed_route'
            self.curriculum_stage_id = 0
        try:
            self.lateral_activation_distance = np.float32(
                float(kwargs.get('lateral_activation_distance', os.getenv('LATERAL_ACTIVATION_DISTANCE', '15.0')))
            )
        except Exception:
            self.lateral_activation_distance = np.float32(15.0)
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
            # 🚨 关键修复：区分"真实碰撞"和"净空不足"两个概念
            # TERRAIN_COLLISION_EPS：真实碰撞阈值（0.3米）- 用于统计碰撞次数，影响成功判定
            #   - 智能体在地形+0.3米以内才算真正碰撞，会增加 total_penetration_count
            #   - 这个阈值应该很小，只检测实际接触或极度接近地形的情况
            # TERRAIN_CLEARANCE_EPS：净空监测阈值（1.5米）- 用于净空不足惩罚，鼓励保持安全距离
            #   - 智能体在地形+1.5米以内会受到净空不足的惩罚（通过奖励计算）
            #   - 但不会增加碰撞计数，不影响成功判定
            # 原问题：之前用1.5米作为碰撞阈值，导致智能体飞在0.5-1.0米高度也被误报为"碰撞"
            # 修复：碰撞阈值降到0.3米，只有真正接触地形才算碰撞
            self.terrain_collision_eps = np.float32(float(os.getenv('TERRAIN_COLLISION_EPS', '0.3')))  # 真实碰撞阈值
            self.terrain_clearance_eps = np.float32(float(os.getenv('TERRAIN_CLEARANCE_EPS', '1.5')))  # 净空监测阈值
            # 兼容旧代码：保留 terrain_contact_eps 变量（指向 collision_eps）
            self.terrain_contact_eps = self.terrain_collision_eps
            self.goal_hold_reward = np.float32(float(kwargs.get('goal_hold_reward', os.getenv('GOAL_HOLD_REWARD', '5.0'))))
            self.leave_goal_penalty = np.float32(float(kwargs.get('leave_goal_penalty', os.getenv('LEAVE_GOAL_PENALTY', '5.0'))))
            self.hover_speed_threshold = np.float32(float(kwargs.get('hover_speed_threshold', os.getenv('HOVER_SPEED_THRESHOLD', '1.0'))))
            self.hover_reward_interval = int(kwargs.get('hover_reward_interval', os.getenv('HOVER_REWARD_INTERVAL', '5')))
            # 轨迹平滑奖励权重：优先使用kwargs，回退到环境变量TURN_SMOOTH_WEIGHT
            self.turn_smooth_weight = np.float32(
                float(kwargs.get('turn_smooth_weight', os.getenv('TURN_SMOOTH_WEIGHT', '0.0')))
            )
            self.distance_reward_near_goal_radius = np.float32(
                float(os.getenv('DISTANCE_REWARD_NEAR_GOAL_RADIUS', '12.0'))
            )
            self.distance_reward_near_goal_factor = np.float32(
                float(os.getenv('DISTANCE_REWARD_NEAR_GOAL_FACTOR', '0.2'))
            )
            self.distance_reward_progress_only_near_goal = (
                str(os.getenv('DISTANCE_REWARD_PROGRESS_ONLY_NEAR_GOAL', '0')).lower()
                in ('1', 'true', 'yes', 'on')
            )
            self.height_penalty_only_near_goal_radius = np.float32(
                float(os.getenv('HEIGHT_PENALTY_ONLY_NEAR_GOAL_RADIUS', '10.0'))
            )
            self.height_near_goal_positive_factor = np.float32(
                float(os.getenv('HEIGHT_NEAR_GOAL_POSITIVE_FACTOR', '0.08'))
            )
            self.clearance_penalty_only_near_goal_radius = np.float32(
                float(os.getenv('CLEARANCE_PENALTY_ONLY_NEAR_GOAL_RADIUS', '12.0'))
            )
            self.clearance_near_goal_positive_factor = np.float32(
                float(os.getenv('CLEARANCE_NEAR_GOAL_POSITIVE_FACTOR', '0.06'))
            )
            self.approach_near_goal_threshold = np.float32(
                float(os.getenv('APPROACH_NEAR_GOAL_THRESHOLD', '50.0'))
            )
            _app_mx = float(os.getenv('APPROACH_NEAR_GOAL_MAX_MULT', '1.22'))
            self.approach_near_goal_max_mult = np.float32(max(_app_mx, 1.0))
            self.stationary_near_goal_radius = np.float32(
                float(os.getenv('STATIONARY_NEAR_GOAL_RADIUS', '8.0'))
            )
            self.stationary_near_goal_threshold = np.float32(
                float(os.getenv('STATIONARY_NEAR_GOAL_THRESHOLD', '0.02'))
            )
            self.stationary_near_goal_min_penalty = np.float32(
                float(os.getenv('STATIONARY_NEAR_GOAL_MIN_PENALTY', '6.0'))
            )
            self.stationary_near_goal_max_penalty = np.float32(
                float(os.getenv('STATIONARY_NEAR_GOAL_MAX_PENALTY', '16.0'))
            )
            self.terminal_failure_penalty_base = np.float32(
                float(os.getenv('TERMINAL_FAILURE_PENALTY_BASE', '30.0'))
            )
            self.terminal_failure_penalty_per_meter = np.float32(
                float(os.getenv('TERMINAL_FAILURE_PENALTY_PER_METER', '120.0'))
            )
            self.terminal_failure_penalty_max = np.float32(
                float(os.getenv('TERMINAL_FAILURE_PENALTY_MAX', '180.0'))
            )
            goal_ring_schedule = os.getenv(
                'GOAL_RING_REWARD_SCHEDULE',
                '18:20,10:35,6:55,3.5:80'
            )
            self.goal_ring_radii, self.goal_ring_bonus_values = self._parse_goal_ring_schedule(goal_ring_schedule)
        except Exception:
            self.penetration_alpha = np.float32(0.5)
            self.expl_reward_strict = False
            self.terrain_penalty_value = self.collision_penalty_value
            # 🚨 关键修复：异常处理中的默认值也从1.5改为2.5，与run_optimized.sh保持一致
            self.terrain_collision_eps = np.float32(0.3)
            self.terrain_clearance_eps = np.float32(1.5)
            self.terrain_contact_eps = np.float32(2.5)
            self.goal_hold_reward = np.float32(5.0)
            self.leave_goal_penalty = np.float32(5.0)
            self.hover_speed_threshold = np.float32(1.0)
            self.hover_reward_interval = 5
            self.turn_smooth_weight = np.float32(0.0)
            self.distance_reward_near_goal_radius = np.float32(12.0)
            self.distance_reward_near_goal_factor = np.float32(0.2)
            self.distance_reward_progress_only_near_goal = False
            self.height_penalty_only_near_goal_radius = np.float32(10.0)
            self.height_near_goal_positive_factor = np.float32(0.08)
            self.clearance_penalty_only_near_goal_radius = np.float32(12.0)
            self.clearance_near_goal_positive_factor = np.float32(0.06)
            self.approach_near_goal_threshold = np.float32(50.0)
            self.approach_near_goal_max_mult = np.float32(1.22)
            self.stationary_near_goal_radius = np.float32(8.0)
            self.stationary_near_goal_threshold = np.float32(0.02)
            self.stationary_near_goal_min_penalty = np.float32(6.0)
            self.stationary_near_goal_max_penalty = np.float32(16.0)
            self.terminal_failure_penalty_base = np.float32(30.0)
            self.terminal_failure_penalty_per_meter = np.float32(120.0)
            self.terminal_failure_penalty_max = np.float32(180.0)
            self.goal_ring_radii, self.goal_ring_bonus_values = self._parse_goal_ring_schedule(
                '18:20,10:35,6:55,3.5:80'
            )
        
        # 性能优化开关
        self.debug_mode = False  # 关闭调试输出
        self.use_fast_path = True  # 启用快速路径
        self._printed_once = False  # 首回合一次性打印关键配置
        
        # 奖励分项名称（用于调试，16通道，对应索引0-15）
        self.reward_names = [
            'distance',    # 0 (主训练中承载 merged progress)
            'exploration', # 1
            'stationary',  # 2
            'direction',   # 3
            'deviation',   # 4
            'start_area',  # 5
            'approach',    # 6 (legacy，占位保留，默认主路径已并入 progress)
            'energy',      # 7
            'height',      # 8
            'success',     # 9
            'collision',   # 10
            'global',      # 11 (主训练中承载 team-sync dense + 终局团队语义)
            'shaping',     # 12
            'clearance',   # 13
            'lateral',     # 14
            'collision_reduction'  # 15
        ]

        # === Reward structure (dense vs terminal) ===
        # 默认启用“结构收缩”：
        # - 主 dense reward：仅保留少数可学性必需项（progress/collision/clearance/height/stationary/success/team-sync）
        # - terminal 奖励/惩罚：在 episode end 额外结算
        # - 其余分项：保留计算与日志，但从主 reward 移出
        try:
            import os as _os
            self.restructured_reward_enabled = _os.getenv('RESTRUCTURED_REWARD', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self.restructured_reward_enabled = True

        # dense 主通道索引（按 reward_names 顺序）
        # progress(distance:0), stationary(2), success(9), collision(10), team-sync/global(11), clearance(13), height(8)
        self._dense_indices_core_fixed = (0, 2, 8, 9, 10, 11, 13)
        self._dense_indices_core_obstacle = (0, 2, 8, 9, 10, 11, 13, 14)
        self._dense_indices_core = (
            self._dense_indices_core_obstacle
            if self._is_obstacle_reward_route()
            else self._dense_indices_core_fixed
        )

        # progress 合并：将 distance(状态锚点) 与 approach(步进结果) 融合成单一主通道
        self.progress_merge_enabled = True

        # 可选弱正则：energy(7)
        try:
            import os as _os
            self.dense_energy_enabled = _os.getenv('DENSE_ENERGY_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self.dense_energy_enabled = True

        # 结构收缩时是否仍计算全部 16 通道（默认否：只算主 dense 用到的通道，省算力；设 1 则恢复全通道便于对照日志）
        try:
            import os as _os
            self.restructured_full_channel_compute = _os.getenv(
                'RESTRUCTURED_FULL_CHANNEL_COMPUTE', '0'
            ).lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self.restructured_full_channel_compute = False

        # terminal 奖励/惩罚（仅 episode end 生效）
        try:
            import os as _os
            self.team_success_bonus = np.float32(float(_os.getenv('TEAM_SUCCESS_BONUS', '3000.0')))
        except Exception:
            self.team_success_bonus = np.float32(3000.0)
        try:
            import os as _os
            self.unsafe_arrival_penalty = np.float32(float(_os.getenv('UNSAFE_ARRIVAL_PENALTY', '1200.0')))
        except Exception:
            self.unsafe_arrival_penalty = np.float32(1200.0)

        # success-only 质量奖励（仅 team_success=True 时给）
        try:
            import os as _os
            self.clearance_quality_bonus_weight = np.float32(float(_os.getenv('CLEARANCE_QUALITY_BONUS_WEIGHT', '800.0')))
        except Exception:
            self.clearance_quality_bonus_weight = np.float32(800.0)
        try:
            import os as _os
            self.efficiency_bonus_weight = np.float32(float(_os.getenv('EFFICIENCY_BONUS_WEIGHT', '800.0')))
        except Exception:
            self.efficiency_bonus_weight = np.float32(800.0)

        # === Team-sync dense reward ===
        try:
            import os as _os
            self.team_sync_enabled = _os.getenv('TEAM_SYNC_REWARD_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self.team_sync_enabled = True
        try:
            import os as _os
            self.team_goal_occupancy_scale = np.float32(float(_os.getenv('TEAM_GOAL_OCCUPANCY_SCALE', '1.0')))
        except Exception:
            self.team_goal_occupancy_scale = np.float32(1.0)
        try:
            import os as _os
            self.team_bottleneck_progress_scale = np.float32(float(_os.getenv('TEAM_BOTTLENECK_PROGRESS_SCALE', '4.0')))
        except Exception:
            self.team_bottleneck_progress_scale = np.float32(4.0)
        try:
            import os as _os
            self.team_waiting_scale = np.float32(float(_os.getenv('TEAM_WAITING_SCALE', '0.6')))
        except Exception:
            self.team_waiting_scale = np.float32(0.6)
        try:
            import os as _os
            self.team_waiting_speed_threshold = np.float32(
                float(_os.getenv('TEAM_WAITING_SPEED_THRESHOLD', str(kwargs.get('hover_speed_threshold', 1.0))))
            )
        except Exception:
            self.team_waiting_speed_threshold = np.float32(kwargs.get('hover_speed_threshold', 1.0))
        try:
            import os as _os
            self.team_bottleneck_delta_clip = np.float32(float(_os.getenv('TEAM_BOTTLENECK_DELTA_CLIP', '1.0')))
        except Exception:
            self.team_bottleneck_delta_clip = np.float32(1.0)
        
        # 奖励多样性检测
        self.reward_history = []
        self.diversity_threshold = 0.95  # 如果连续奖励相似度超过95%，认为缺乏多样性
        self.diversity_window = 10  # 检测窗口大小
        self.last_warning_episode = 0  # 上次警告的回合数
        self.warning_cooldown = 20  # 警告冷却期（回合数）
        try:
            import os as _os
            self.enable_reward_diversity_check = _os.getenv(
                'ENABLE_REWARD_DIVERSITY_CHECK', '0'
            ).lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self.enable_reward_diversity_check = False
        try:
            import os as _os
            self.reward_diversity_check_interval = max(
                1, int(_os.getenv('REWARD_DIVERSITY_CHECK_INTERVAL', '1'))
            )
        except Exception:
            self.reward_diversity_check_interval = 1
        self._last_diversity_checked_episode = None
        
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

        self._timing_detail_enabled_cache = None
        self._last_reward_timing = None
        self._terrain_search_directions = np.array([
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707],
            [0.923, 0.383], [-0.923, 0.383], [0.923, -0.383], [-0.923, -0.383],
            [0.383, 0.923], [-0.383, 0.923], [0.383, -0.923], [-0.383, -0.923],
        ], dtype=np.float32)
        self._terrain_search_radii = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 50.0], dtype=np.float32)
        terrain_sample_offsets = (
            self._terrain_search_directions[:, None, :] * self._terrain_search_radii[None, :, None]
        ).astype(np.float32)
        self._terrain_sample_offsets_flat = terrain_sample_offsets.reshape(-1, 2)
        self._terrain_sample_offset_sq_flat = np.sum(
            self._terrain_sample_offsets_flat * self._terrain_sample_offsets_flat,
            axis=1,
            dtype=np.float32,
        ).astype(np.float32)

    @staticmethod
    def _normalize_reward_profile_name(raw_profile):
        try:
            profile = str(raw_profile).strip().lower()
        except Exception:
            profile = 'fixed_route'
        if profile == 'obstacle_route':
            return 'obstacle_route'
        return 'fixed_route'

    def _is_obstacle_reward_route(self) -> bool:
        return getattr(self, 'reward_profile', 'fixed_route') == 'obstacle_route'

    def _attenuate_distance_reward_near_goal(
        self,
        base_rewards: np.ndarray,
        current_dist: np.ndarray,
    ) -> np.ndarray:
        """处理成功圈外近目标区域的 distance 状态奖励，减少“停在目标外刷分”局部最优。"""
        adjusted = np.asarray(base_rewards, dtype=np.float32).copy()
        try:
            radius = float(self.distance_reward_near_goal_radius)
            floor_factor = float(self.distance_reward_near_goal_factor)
            success_thr = float(self.success_distance_threshold)
            if radius <= success_thr or floor_factor >= 1.0:
                return adjusted

            near_mask = (
                (current_dist > success_thr) &
                (current_dist < radius) &
                (adjusted > 0.0)
            )
            if not np.any(near_mask):
                return adjusted

            if bool(getattr(self, 'distance_reward_progress_only_near_goal', False)):
                adjusted[near_mask] = 0.0
                return adjusted

            span = max(radius - success_thr, 1e-6)
            ramp = (current_dist[near_mask] - success_thr) / span
            ramp = np.clip(ramp, 0.0, 1.0)
            scale = floor_factor + (1.0 - floor_factor) * ramp
            adjusted[near_mask] = adjusted[near_mask] * scale.astype(np.float32)
            return adjusted
        except Exception:
            return adjusted

    def _parse_goal_ring_schedule(self, schedule_text: str) -> Tuple[np.ndarray, np.ndarray]:
        """解析阶段性 goal ring 奖励配置，格式为 '半径:奖励,半径:奖励,...'。"""
        radii = []
        bonuses = []
        try:
            success_thr = float(getattr(self, 'success_distance_threshold', 2.0))
            for chunk in str(schedule_text or '').split(','):
                item = chunk.strip()
                if not item or ':' not in item:
                    continue
                radius_text, bonus_text = item.split(':', 1)
                radius = float(radius_text.strip())
                bonus = float(bonus_text.strip())
                if radius <= success_thr or bonus <= 0.0:
                    continue
                radii.append(radius)
                bonuses.append(bonus)
        except Exception:
            radii = []
            bonuses = []

        if not radii:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        order = np.argsort(np.asarray(radii, dtype=np.float32))
        return (
            np.asarray(radii, dtype=np.float32)[order],
            np.asarray(bonuses, dtype=np.float32)[order],
        )

    def _goal_ring_bonus_vectorized(
        self,
        agent: Any,
        distances: np.ndarray,
        success_state: Dict[str, Any],
    ) -> np.ndarray:
        """一次性阶段奖励：首次进入若干目标半径时发放小额 bonus。"""
        rewards = np.zeros(len(distances), dtype=np.float32)
        try:
            if self.goal_ring_radii.size == 0 or self.goal_ring_bonus_values.size == 0:
                return rewards

            ring_state = success_state.get('goal_ring_rewards_given')
            if not isinstance(ring_state, list) or len(ring_state) != int(self.goal_ring_radii.size):
                ring_state = [False] * int(self.goal_ring_radii.size)

            for idx, (radius, bonus) in enumerate(zip(self.goal_ring_radii, self.goal_ring_bonus_values)):
                if ring_state[idx]:
                    continue
                ring_mask = distances <= float(radius)
                if np.any(ring_mask):
                    rewards[ring_mask] += float(bonus)
                    ring_state[idx] = True

            success_state['goal_ring_rewards_given'] = ring_state
            return rewards
        except Exception:
            return rewards

    def _merge_progress_rewards(
        self,
        distance_reward: np.ndarray,
        approach_reward: np.ndarray,
        distance_weight: float = None,
        approach_weight: float = None,
    ) -> np.ndarray:
        """
        将 distance(状态锚点) 与 approach(步进结果) 合并成单一 progress 通道。

        设计目标：
        - 保留 distance 的“全局接近度”锚点；
        - 保留 approach 的“本步真实推进”梯度；
        - 避免两个通道并列时对“接近目标”重复计分。
        """
        distance_reward = np.asarray(distance_reward, dtype=np.float32)
        approach_reward = np.asarray(approach_reward, dtype=np.float32)

        if distance_weight is None:
            distance_weight = float(self.reward_weights[0]) if len(self.reward_weights) > 0 else 1.0
        if approach_weight is None:
            approach_weight = float(self.reward_weights[6]) if len(self.reward_weights) > 6 else 1.0

        total_weight = abs(float(distance_weight)) + abs(float(approach_weight))
        if total_weight < 1e-6:
            return np.zeros_like(distance_reward, dtype=np.float32)

        merged = (
            float(distance_weight) * distance_reward +
            float(approach_weight) * approach_reward
        ) / total_weight
        return merged.astype(np.float32)

    def _relax_approach_for_obstacle_route(
        self,
        approach_reward: np.ndarray,
        geometry_context: Dict[str, Any],
    ) -> np.ndarray:
        """
        第二阶段（随机障碍路线）的最简 progress 调整：
        仅在靠近随机障碍时放宽负 approach，正 approach 保持不变，不新增调参项。
        """
        adjusted = np.asarray(approach_reward, dtype=np.float32).copy()
        if not self._is_obstacle_reward_route():
            return adjusted
        if geometry_context is None or not isinstance(geometry_context, dict):
            return adjusted

        try:
            obstacle_min_dist = np.asarray(geometry_context['obstacle_min_dist'], dtype=np.float32)
            if obstacle_min_dist.shape != adjusted.shape:
                return adjusted
            activation_distance = max(float(getattr(self, 'lateral_activation_distance', 15.0)), 1e-6)
            relax = np.clip(obstacle_min_dist / activation_distance, 0.0, 1.0).astype(np.float32)
            negative_mask = adjusted < 0.0
            if np.any(negative_mask):
                adjusted[negative_mask] = adjusted[negative_mask] * relax[negative_mask]
            return adjusted
        except Exception:
            return adjusted

    def _team_sync_reward_vectorized(
        self,
        agents: List[Any],
        world: Any,
        scenario: Any,
        positions: np.ndarray,
        goal_positions: np.ndarray,
        valid_goal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        团队同步过程奖励（dense）：
        - 所有阶段统一：occupancy/waiting 按 Succ_i（Reach_i ∧ Safe_i）
        - bottleneck progress：统一看未成功体

        该项不改严格 TeamSuccess 定义，只补“过程梯度”。
        """
        rewards = np.zeros(len(agents), dtype=np.float32)
        if not getattr(self, 'team_sync_enabled', True):
            return rewards
        if world is None or len(agents) == 0:
            return rewards

        try:
            cur_step = int(getattr(world, 'current_step', -1))
        except Exception:
            cur_step = -1

        cache = getattr(world, '_team_sync_step_cache', None)
        if cache is not None and cache[0] == cur_step:
            cached_rewards = np.asarray(cache[1], dtype=np.float32)
            if cached_rewards.shape == rewards.shape:
                return cached_rewards.copy()

        state = getattr(world, '_team_sync_state', None)
        is_new_episode = False
        if state is None or not isinstance(state, dict):
            state = {}
            is_new_episode = True
        else:
            last_step = state.get('last_step', None)
            if last_step is None:
                is_new_episode = True
            elif cur_step >= 0 and last_step is not None and cur_step < int(last_step):
                is_new_episode = True
            elif cur_step == 0 and last_step is not None and int(last_step) != 0:
                is_new_episode = True

        n_agents = max(len(agents), 1)
        current_dists = np.full((len(agents),), np.inf, dtype=np.float32)
        if goal_positions is not None and len(goal_positions) == len(agents):
            valid_idx = np.where(valid_goal_mask)[0]
            if valid_idx.size > 0:
                current_dists[valid_idx] = np.linalg.norm(
                    positions[valid_idx] - goal_positions[valid_idx], axis=-1
                ).astype(np.float32)

        safe_flags = np.array([bool(self._agent_safe_so_far(ag)) for ag in agents], dtype=bool)
        reach_mask = np.isfinite(current_dists) & (current_dists <= float(self.success_distance_threshold))
        succ_mask = reach_mask & safe_flags
        progress_mask = succ_mask

        occupancy_ratio = float(np.count_nonzero(progress_mask)) / float(n_agents)
        incomplete_mask = np.isfinite(current_dists) & (~progress_mask)
        if np.any(incomplete_mask):
            bottleneck_dist = float(np.max(current_dists[incomplete_mask]))
        else:
            bottleneck_dist = 0.0

        if is_new_episode:
            state['last_bottleneck_dist'] = bottleneck_dist
        last_bottleneck_dist = float(state.get('last_bottleneck_dist', bottleneck_dist))
        bottleneck_delta = max(0.0, last_bottleneck_dist - bottleneck_dist)
        bottleneck_delta = min(bottleneck_delta, float(self.team_bottleneck_delta_clip))

        speeds = np.zeros((len(agents),), dtype=np.float32)
        for idx, ag in enumerate(agents):
            try:
                speeds[idx] = float(np.linalg.norm(getattr(getattr(ag, 'state', None), 'p_vel', np.zeros(3))))
            except Exception:
                speeds[idx] = 0.0

        all_progress = bool(np.all(progress_mask)) if len(progress_mask) > 0 else False
        waiting_mask = progress_mask & (speeds <= float(self.team_waiting_speed_threshold)) & (~all_progress)
        waiting_ratio = float(np.count_nonzero(waiting_mask)) / float(n_agents)

        reward_scalar = (
            float(self.team_goal_occupancy_scale) * occupancy_ratio +
            float(self.team_bottleneck_progress_scale) * bottleneck_delta +
            float(self.team_waiting_scale) * waiting_ratio
        )

        state['last_bottleneck_dist'] = bottleneck_dist
        state['last_step'] = cur_step
        world._team_sync_state = state

        rewards.fill(np.float32(reward_scalar))
        world._team_sync_step_cache = (cur_step, rewards.copy())
        world._team_sync_reward = float(reward_scalar)
        world._team_sync_occupancy_ratio = float(occupancy_ratio)
        world._team_sync_bottleneck_dist = float(bottleneck_dist)
        world._team_sync_bottleneck_delta = float(bottleneck_delta)
        world._team_sync_waiting_ratio = float(waiting_ratio)
        world._team_sync_occupancy_basis = 'succ'

        for ag in agents:
            try:
                if not hasattr(ag, 'debug_info') or not isinstance(ag.debug_info, dict):
                    ag.debug_info = {}
                ag.debug_info['team_sync_reward'] = float(reward_scalar)
                ag.debug_info['team_sync_occupancy_ratio'] = float(occupancy_ratio)
                ag.debug_info['team_sync_bottleneck_dist'] = float(bottleneck_dist)
                ag.debug_info['team_sync_bottleneck_delta'] = float(bottleneck_delta)
                ag.debug_info['team_sync_waiting_ratio'] = float(waiting_ratio)
                ag.debug_info['team_sync_occupancy_basis'] = 'succ'
            except Exception:
                pass
        return rewards

    def _is_episode_finished(self, world: Any) -> bool:
        """判断当前回合是否结束，用于仅在终局结算的奖励/惩罚。"""
        try:
            if world is None:
                return False

            if bool(getattr(world, '_episode_success', False)):
                return True

            episode_length = None
            if hasattr(world, 'episode_length') and world.episode_length is not None:
                episode_length = int(world.episode_length)
            elif hasattr(world, 'max_steps') and world.max_steps is not None:
                episode_length = int(world.max_steps)
            else:
                episode_length = int(os.getenv('EPISODE_LENGTH', '2800'))

            cur_step = int(getattr(world, 'current_step', -1))
            return bool(episode_length is not None and episode_length > 0 and cur_step >= episode_length)
        except Exception:
            return False

    def _agent_reached_goal_ever(self, agent: Any, current_dist: float) -> bool:
        """与成功统计保持一致：当前进入成功圈，或本回合曾经进入过成功圈。"""
        try:
            if current_dist <= float(self.success_distance_threshold):
                return True
            if bool(getattr(agent, '_ever_reached_goal', False)):
                return True
            if hasattr(agent, '_success_state') and isinstance(agent._success_state, dict):
                if bool(agent._success_state.get('success_reward_given', False)):
                    return True
        except Exception:
            pass
        return False

    def _terminal_failure_penalty_batch(
        self,
        agents: List[Any],
        positions: np.ndarray,
        goal_positions: np.ndarray,
        valid_goal_mask: np.ndarray,
        start_positions: np.ndarray,
        world: Any,
    ) -> np.ndarray:
        """回合结束时，对从未成功进入目标圈的智能体按剩余距离施加终局惩罚。"""
        num_positions = positions.shape[0]
        penalties = np.zeros(num_positions, dtype=np.float32)
        try:
            if not self._is_episode_finished(world):
                return penalties

            base_penalty = max(float(self.terminal_failure_penalty_base), 0.0)
            per_meter = max(float(self.terminal_failure_penalty_per_meter), 0.0)
            max_penalty = max(float(self.terminal_failure_penalty_max), 0.0)
            success_thr = float(self.success_distance_threshold)
            if max_penalty <= 0.0 or (base_penalty <= 0.0 and per_meter <= 0.0):
                return penalties

            for idx, agent in enumerate(agents):
                if not valid_goal_mask[idx]:
                    continue

                current_dist = float(np.linalg.norm(positions[idx] - goal_positions[idx]))
                if self._agent_reached_goal_ever(agent, current_dist):
                    continue

                excess_dist = max(current_dist - success_thr, 0.0)
                initial_dist = float(np.linalg.norm(start_positions[idx] - goal_positions[idx]))
                denom = max(initial_dist, success_thr, 1e-6)
                remaining_ratio = np.clip(excess_dist / denom, 0.0, 1.0)

                raw_penalty = base_penalty + per_meter * remaining_ratio
                penalties[idx] = -min(max_penalty, raw_penalty)
        except Exception:
            return np.zeros(num_positions, dtype=np.float32)
        return penalties

    def _reward_timing_enabled(self) -> bool:
        cached = self._timing_detail_enabled_cache
        if cached is None:
            try:
                level = int(os.getenv('TIMING_LEVEL', '1'))
            except Exception:
                level = 1
            try:
                detail_flag = os.getenv('TIMING_DETAIL', '0').lower() in ('1', 'true', 'yes', 'on')
            except Exception:
                detail_flag = False
            cached = bool(level >= 2 or detail_flag)
            self._timing_detail_enabled_cache = cached
        return bool(cached)

    def _new_reward_timing_bucket(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.REWARD_TIMING_KEYS}

    def get_last_reward_timing(self) -> Dict[str, float]:
        if isinstance(self._last_reward_timing, dict):
            return dict(self._last_reward_timing)
        return {}
    
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

    def _piecewise_collision_penalty(self, signed_distances: np.ndarray, threshold: float, base_penalty: float = None) -> np.ndarray:
        """真实碰撞带内的分段线性惩罚。

        - d >= threshold: 0
        - 0 <= d < threshold: 从 0 线性下降到 -base_penalty
        - d < 0: 在 -base_penalty 的基础上继续按穿透深度增加惩罚
        """
        distances = np.asarray(signed_distances, dtype=np.float32)
        safe_threshold = max(float(threshold), 1e-6)
        base = float(self.collision_penalty_value if base_penalty is None else base_penalty)

        distances = np.nan_to_num(distances, nan=safe_threshold, posinf=safe_threshold, neginf=-safe_threshold)
        band_ratio = np.clip((safe_threshold - distances) / safe_threshold, 0.0, 1.0).astype(np.float32)
        penetration_depth = np.maximum(-distances, 0.0).astype(np.float32)

        penalties = -np.float32(base) * band_ratio - penetration_depth * float(self.penetration_alpha)
        penalties = np.where(distances >= safe_threshold, 0.0, penalties)
        return penalties.astype(np.float32)

    def _get_agent_velocity_array(self, agent: Any, num_positions: int) -> np.ndarray:
        """将智能体当前速度转换为与 positions 对齐的二维数组。"""
        velocities = np.zeros((num_positions, 3), dtype=np.float32)
        try:
            raw_vel = getattr(getattr(agent, 'state', None), 'p_vel', None)
            if raw_vel is None:
                return velocities

            vel_arr = np.asarray(raw_vel, dtype=np.float32)
            if vel_arr.ndim == 0:
                return velocities
            if vel_arr.ndim == 1:
                if vel_arr.shape[0] >= 3:
                    velocities[:] = vel_arr[:3]
                return velocities

            flat_vel = vel_arr.reshape(-1, vel_arr.shape[-1])
            if flat_vel.shape[1] < 3:
                return velocities
            if flat_vel.shape[0] == num_positions:
                velocities = flat_vel[:, :3].astype(np.float32)
            else:
                velocities[:] = flat_vel[0, :3]
        except Exception:
            pass
        return velocities

    def _query_terrain_heights_batch(
        self,
        scenario: Any,
        cached_data: Dict[str, Any],
        positions: np.ndarray,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """批量查询地形高度，优先使用缓存地形和向量化接口。"""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        query_xy = np.asarray(positions[:, :2], dtype=np.float32)
        if out is not None and isinstance(out, np.ndarray) and out.shape[0] >= query_xy.shape[0]:
            heights = out[:query_xy.shape[0]]
            heights.fill(np.nan)
        else:
            heights = np.full(query_xy.shape[0], np.nan, dtype=np.float32)
        if scenario is None or query_xy.shape[0] == 0:
            return heights

        cached_data = cached_data if isinstance(cached_data, dict) else {}
        cached_terrain = cached_data.get('terrain')
        cached_map_size = int(cached_data.get('map_size', getattr(scenario, 'map_size', 200)))

        try:
            if cached_terrain is not None:
                _bilinear_interpolate_terrain_xy(
                    cached_terrain,
                    query_xy[:, 0],
                    query_xy[:, 1],
                    heights,
                    cached_map_size,
                )
                if np.all(np.isfinite(heights)):
                    return heights

            if hasattr(scenario, 'get_terrain_height_vectorized'):
                sampled = scenario.get_terrain_height_vectorized(query_xy[:, 0], query_xy[:, 1])
                sampled = np.asarray(sampled, dtype=np.float32)
                if sampled.ndim == 0:
                    heights[:] = np.float32(sampled)
                    return heights
                if sampled.shape[0] == query_xy.shape[0]:
                    return sampled

            if hasattr(scenario, 'get_terrain_height'):
                for idx, xy in enumerate(query_xy):
                    try:
                        heights[idx] = scenario.get_terrain_height(float(xy[0]), float(xy[1]))
                    except Exception:
                        continue
        except Exception:
            pass
        return heights

    def _get_terrain_distance_scratch(self, num_positions: int) -> Dict[str, np.ndarray]:
        """获取 terrain distance 计算用的预分配 scratch buffer。"""
        sample_count = int(self._terrain_sample_offsets_flat.shape[0])
        key = (int(num_positions), sample_count)
        scratch = self._terrain_distance_scratch.get(key)
        total_samples = int(num_positions) * sample_count
        merged_count = int(num_positions) + total_samples
        if scratch is None:
            scratch = {
                'sample_xys_flat': np.zeros((total_samples, 2), dtype=np.float32),
                'valid_map_mask': np.zeros(total_samples, dtype=bool),
                'sample_heights_flat': np.full(total_samples, np.nan, dtype=np.float32),
                'merged_queries': np.zeros((merged_count, 2), dtype=np.float32),
                'merged_heights': np.full(merged_count, np.nan, dtype=np.float32),
            }
            self._terrain_distance_scratch[key] = scratch
        return scratch

    def _compute_obstacle_distance_data(
        self,
        positions: np.ndarray,
        cached_data: Dict[str, Any] = None,
        world: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量计算到最近障碍物表面的距离与参考点。"""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        positions = np.asarray(positions[:, :3], dtype=np.float32)
        num_positions = positions.shape[0]

        obstacle_min_dist = np.full(num_positions, np.inf, dtype=np.float32)
        nearest_obstacle_centers = np.zeros((num_positions, 3), dtype=np.float32)
        nearest_obstacle_radii = np.zeros(num_positions, dtype=np.float32)

        cached_data = cached_data if isinstance(cached_data, dict) else {}
        obstacles_centers = cached_data.get('obstacles_centers')
        obstacles_radii = cached_data.get('obstacles_radii')

        if obstacles_centers is not None and obstacles_radii is not None:
            try:
                centers = np.asarray(obstacles_centers, dtype=np.float32)
                radii = np.asarray(obstacles_radii, dtype=np.float32)
                if (
                    centers.ndim == 2
                    and centers.shape[0] > 0
                    and centers.shape[1] >= 3
                    and radii.ndim == 1
                    and radii.shape[0] == centers.shape[0]
                ):
                    diff = positions[:, None, :] - centers[None, :, :3]
                    center_dist = np.linalg.norm(diff, axis=-1)
                    surface_dist = center_dist - radii[None, :]
                    nearest_indices = np.argmin(surface_dist, axis=1)
                    obstacle_min_dist = surface_dist[np.arange(num_positions), nearest_indices].astype(np.float32)
                    nearest_obstacle_centers = centers[nearest_indices, :3].astype(np.float32)
                    nearest_obstacle_radii = radii[nearest_indices].astype(np.float32)
                    return obstacle_min_dist, nearest_obstacle_centers, nearest_obstacle_radii
            except Exception:
                obstacle_min_dist.fill(np.inf)

        scenario = getattr(world, 'scenario', None)
        if bool(hasattr(scenario, 'obstacles') and scenario.obstacles):
            for obstacle_data in scenario.obstacles:
                if 'center' not in obstacle_data or 'radius' not in obstacle_data:
                    continue
                obstacle_center = np.asarray(obstacle_data['center'][:3], dtype=np.float32)
                obstacle_radius = float(obstacle_data['radius'])
                dist_3d = np.linalg.norm(positions - obstacle_center, axis=-1)
                dist_to_surface = dist_3d - obstacle_radius
                update_mask = dist_to_surface < obstacle_min_dist
                if np.any(update_mask):
                    nearest_obstacle_centers[update_mask] = obstacle_center
                    nearest_obstacle_radii[update_mask] = obstacle_radius
                    obstacle_min_dist[update_mask] = dist_to_surface[update_mask]

        return obstacle_min_dist, nearest_obstacle_centers, nearest_obstacle_radii

    def _compute_terrain_distance_data(
        self,
        positions: np.ndarray,
        scenario: Any,
        cached_data: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算当前位置对应的地形高度、地形最小距离和参考点。"""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        positions = np.asarray(positions[:, :3], dtype=np.float32)
        num_positions = positions.shape[0]

        terrain_heights_current = np.full(num_positions, np.nan, dtype=np.float32)
        terrain_min_dist = np.full(num_positions, np.inf, dtype=np.float32)
        terrain_reference_points = positions.copy()

        if num_positions == 0 or scenario is None or not hasattr(scenario, 'get_terrain_height'):
            return terrain_heights_current, terrain_min_dist, terrain_reference_points

        cached_data = cached_data if isinstance(cached_data, dict) else {}
        cached_map_size = int(cached_data.get('map_size', getattr(scenario, 'map_size', 200.0)))

        try:
            map_size = float(getattr(scenario, 'map_size', cached_map_size if cached_map_size > 0 else 200.0))
            sample_offsets_flat = self._terrain_sample_offsets_flat
            offset_sq_flat = self._terrain_sample_offset_sq_flat
            num_samples = int(sample_offsets_flat.shape[0])
            cached_terrain = cached_data.get('terrain')
            scratch = self._get_terrain_distance_scratch(num_positions)
            sample_xys_flat = scratch['sample_xys_flat']
            sample_xys = sample_xys_flat.reshape(num_positions, num_samples, 2)
            np.add(positions[:, None, :2], sample_offsets_flat[None, :, :], out=sample_xys)
            valid_map_mask = scratch['valid_map_mask']
            valid_map_mask[:] = (
                (sample_xys_flat[:, 0] >= 0.0) & (sample_xys_flat[:, 0] < map_size) &
                (sample_xys_flat[:, 1] >= 0.0) & (sample_xys_flat[:, 1] < map_size)
            )

            sample_heights_flat = scratch['sample_heights_flat']
            sample_heights_flat.fill(np.nan)
            if isinstance(cached_terrain, np.ndarray) and cached_terrain.size > 0:
                _bilinear_interpolate_terrain_xy(
                    cached_terrain,
                    positions[:, 0],
                    positions[:, 1],
                    terrain_heights_current,
                    cached_map_size,
                )
                _bilinear_interpolate_terrain_xy(
                    cached_terrain,
                    sample_xys_flat[:, 0],
                    sample_xys_flat[:, 1],
                    sample_heights_flat,
                    cached_map_size,
                )
                sample_heights_flat[~valid_map_mask] = np.nan
            else:
                merged_queries = scratch['merged_queries']
                merged_queries[:num_positions] = positions[:, :2]
                merged_count = num_positions
                if np.any(valid_map_mask):
                    valid_sample_xys = sample_xys_flat[valid_map_mask]
                    valid_count = valid_sample_xys.shape[0]
                    merged_queries[num_positions:num_positions + valid_count] = valid_sample_xys
                    merged_count += valid_count

                merged_heights = self._query_terrain_heights_batch(
                    scenario,
                    cached_data,
                    merged_queries[:merged_count],
                    out=scratch['merged_heights'],
                )
                terrain_heights_current = np.asarray(merged_heights[:num_positions], dtype=np.float32).copy()
                if merged_count > num_positions:
                    sample_heights_flat[valid_map_mask] = merged_heights[num_positions:merged_count]

            valid_current_terrain = np.isfinite(terrain_heights_current)
            if np.any(valid_current_terrain):
                terrain_min_dist[valid_current_terrain] = (
                    positions[valid_current_terrain, 2] - terrain_heights_current[valid_current_terrain]
                ).astype(np.float32)
                terrain_reference_points[valid_current_terrain, 0] = positions[valid_current_terrain, 0]
                terrain_reference_points[valid_current_terrain, 1] = positions[valid_current_terrain, 1]
                terrain_reference_points[valid_current_terrain, 2] = terrain_heights_current[valid_current_terrain]

            sample_heights = sample_heights_flat.reshape(num_positions, num_samples)
            delta_heights = sample_heights - positions[:, None, 2]
            sample_distances = np.sqrt(offset_sq_flat[None, :] + delta_heights * delta_heights).astype(
                np.float32,
                copy=False,
            )
            sample_distances[~np.isfinite(sample_heights)] = np.inf

            nearest_sample_idx = np.argmin(sample_distances, axis=1)
            nearest_sample_dist = sample_distances[np.arange(num_positions), nearest_sample_idx]
            nearest_sample_points = np.zeros((num_positions, 3), dtype=np.float32)
            nearest_sample_points[:, :2] = sample_xys[np.arange(num_positions), nearest_sample_idx]
            nearest_sample_points[:, 2] = sample_heights[np.arange(num_positions), nearest_sample_idx]

            use_sample_mask = np.isfinite(nearest_sample_dist) & (nearest_sample_dist < terrain_min_dist)
            terrain_min_dist = np.where(
                use_sample_mask,
                nearest_sample_dist.astype(np.float32),
                terrain_min_dist,
            )
            terrain_reference_points[use_sample_mask] = nearest_sample_points[use_sample_mask].astype(np.float32)
        except Exception:
            pass

        return terrain_heights_current, terrain_min_dist.astype(np.float32), terrain_reference_points.astype(np.float32)

    def _build_reward_geometry_context(
        self,
        positions: np.ndarray,
        world: Any,
        scenario: Any,
        cached_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """构建同一步共享的几何上下文，供 clearance/collision 复用。"""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        positions = np.asarray(positions[:, :3], dtype=np.float32)

        obstacle_min_dist, nearest_obstacle_centers, nearest_obstacle_radii = self._compute_obstacle_distance_data(
            positions,
            cached_data=cached_data,
            world=world,
        )
        terrain_heights_current, terrain_min_dist, terrain_reference_points = self._compute_terrain_distance_data(
            positions,
            scenario,
            cached_data=cached_data,
        )

        return {
            'num_positions': int(positions.shape[0]),
            'obstacle_min_dist': obstacle_min_dist.astype(np.float32, copy=False),
            'nearest_obstacle_centers': nearest_obstacle_centers.astype(np.float32, copy=False),
            'nearest_obstacle_radii': nearest_obstacle_radii.astype(np.float32, copy=False),
            'terrain_heights_current': terrain_heights_current.astype(np.float32, copy=False),
            'terrain_min_dist': terrain_min_dist.astype(np.float32, copy=False),
            'terrain_reference_points': terrain_reference_points.astype(np.float32, copy=False),
        }

    def _batch_clearance_reward_vectorized(
        self,
        agents: List[Any],
        world: Any,
        positions: np.ndarray,
        cached_data: Dict[str, Any] = None,
        geometry_context: Dict[str, Any] = None,
    ) -> np.ndarray:
        """按当前 world 的所有智能体批量计算净空奖励。"""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        positions = np.asarray(positions[:, :3], dtype=np.float32)
        num_positions = positions.shape[0]
        if num_positions == 0:
            return np.zeros(0, dtype=np.float32)

        if len(agents) != num_positions:
            rewards = np.zeros(num_positions, dtype=np.float32)
            for idx in range(min(len(agents), num_positions)):
                single = self._clearance_reward_vectorized(agents[idx], world, positions[idx:idx + 1])
                rewards[idx] = single[0] if isinstance(single, np.ndarray) else float(single)
            return rewards

        safe_distance = max(float(os.getenv('OBSTACLE_SAFE_DISTANCE', '15.0')), 1e-6)
        collision_threshold = max(float(self.collision_distance_threshold), 1e-6)
        warning_span = max(safe_distance - collision_threshold, 1e-6)
        penalty_weight = float(os.getenv('CLEARANCE_PENALTY_WEIGHT', '5.0'))
        clearance_weight = float(os.getenv('CLEARANCE_WEIGHT', '3.5'))
        clearance_d_max = max(float(os.getenv('CLEARANCE_D_MAX', '80.0')), 1e-6)
        trend_penalty_factor = float(os.getenv('CLEARANCE_APPROACH_PENALTY_FACTOR', '2.0'))
        velocity_weight = float(os.getenv('CLEARANCE_VELOCITY_WEIGHT', '0.5'))
        speed_scale = max(float(os.getenv('CLEARANCE_SPEED_SCALE', '5.0')), 1e-6)
        trend_eps = float(os.getenv('CLEARANCE_TREND_EPS', '1e-6'))
        speed_eps = float(os.getenv('CLEARANCE_SPEED_EPS', '1e-4'))
        far_distance_fill = safe_distance + clearance_d_max

        scenario = getattr(world, 'scenario', None)
        cached_data = cached_data if isinstance(cached_data, dict) else {}
        geometry_ready = False
        if isinstance(geometry_context, dict):
            try:
                obstacle_min_dist = np.asarray(geometry_context['obstacle_min_dist'], dtype=np.float32)
                nearest_obstacle_centers = np.asarray(geometry_context['nearest_obstacle_centers'], dtype=np.float32)
                nearest_obstacle_radii = np.asarray(geometry_context['nearest_obstacle_radii'], dtype=np.float32)
                terrain_warning_dist = np.asarray(geometry_context['terrain_min_dist'], dtype=np.float32)
                terrain_reference_points = np.asarray(geometry_context['terrain_reference_points'], dtype=np.float32)
                terrain_heights_current = np.asarray(geometry_context['terrain_heights_current'], dtype=np.float32)
                geometry_ready = (
                    int(geometry_context.get('num_positions', -1)) == num_positions
                    and obstacle_min_dist.shape == (num_positions,)
                    and nearest_obstacle_centers.shape == (num_positions, 3)
                    and nearest_obstacle_radii.shape == (num_positions,)
                    and terrain_warning_dist.shape == (num_positions,)
                    and terrain_reference_points.shape == (num_positions, 3)
                    and terrain_heights_current.shape == (num_positions,)
                )
            except Exception:
                geometry_ready = False

        if not geometry_ready:
            obstacle_min_dist, nearest_obstacle_centers, nearest_obstacle_radii = self._compute_obstacle_distance_data(
                positions,
                cached_data=cached_data,
                world=world,
            )
            terrain_heights_current, terrain_warning_dist, terrain_reference_points = self._compute_terrain_distance_data(
                positions,
                scenario,
                cached_data=cached_data,
            )

        obstacle_reference_points = positions.copy()
        obstacle_vector = positions - nearest_obstacle_centers
        obstacle_norm = np.linalg.norm(obstacle_vector, axis=1, keepdims=True)
        default_dirs = np.zeros_like(obstacle_vector)
        default_dirs[:, 2] = 1.0
        obstacle_dirs = np.where(obstacle_norm > 1e-6, obstacle_vector / np.maximum(obstacle_norm, 1e-6), default_dirs)
        valid_obstacle_mask = np.isfinite(obstacle_min_dist)
        if np.any(valid_obstacle_mask):
            obstacle_reference_points[valid_obstacle_mask] = (
                nearest_obstacle_centers[valid_obstacle_mask]
                + obstacle_dirs[valid_obstacle_mask] * nearest_obstacle_radii[valid_obstacle_mask, None]
            )

        d_min_current = np.minimum(obstacle_min_dist, terrain_warning_dist).astype(np.float32)
        use_terrain_reference = terrain_warning_dist <= obstacle_min_dist
        reference_points = np.where(use_terrain_reference[:, None], terrain_reference_points, obstacle_reference_points).astype(np.float32)
        invalid_reference = ~np.all(np.isfinite(reference_points), axis=1)
        if np.any(invalid_reference):
            reference_points[invalid_reference] = positions[invalid_reference]

        goal_positions = np.zeros((num_positions, 3), dtype=np.float32)
        valid_goal_mask = np.zeros(num_positions, dtype=bool)
        cached_goal_positions = cached_data.get('goal_positions')
        fallback_goal = None
        try:
            if scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                fallback_goal = np.asarray(scenario.goal_pos, dtype=np.float32).reshape(-1)[:3]
                if fallback_goal.shape[0] < 3:
                    fallback_goal = None
        except Exception:
            fallback_goal = None

        for idx, agent in enumerate(agents):
            goal_pos = None
            if cached_goal_positions is not None and idx < len(cached_goal_positions):
                try:
                    goal_pos = np.asarray(cached_goal_positions[idx], dtype=np.float32).reshape(-1)[:3]
                    if goal_pos.shape[0] < 3:
                        goal_pos = None
                except Exception:
                    goal_pos = None
            if goal_pos is None:
                try:
                    if hasattr(agent, 'goal_a') and agent.goal_a is not None:
                        if hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                            goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32).reshape(-1)[:3]
                            if goal_pos.shape[0] < 3:
                                goal_pos = None
                except Exception:
                    goal_pos = None
            if goal_pos is None:
                goal_pos = fallback_goal
            if goal_pos is not None:
                goal_positions[idx] = goal_pos
                valid_goal_mask[idx] = True

        FAR_THRESHOLD = float(os.getenv('CLEARANCE_FAR_THRESHOLD', '50.0'))
        NEAR_THRESHOLD = float(os.getenv('CLEARANCE_NEAR_THRESHOLD', '20.0'))
        WEIGHT_FAR = float(os.getenv('CLEARANCE_WEIGHT_FAR', '0.5'))
        WEIGHT_NEAR = float(os.getenv('CLEARANCE_WEIGHT_NEAR', '12.0'))

        dists_to_goal = np.full(num_positions, 100.0, dtype=np.float32)
        if np.any(valid_goal_mask):
            try:
                dists_to_goal[valid_goal_mask] = np.linalg.norm(
                    positions[valid_goal_mask] - goal_positions[valid_goal_mask],
                    axis=-1,
                ).astype(np.float32)
            except Exception:
                pass

        dynamic_weights = np.full(num_positions, WEIGHT_FAR, dtype=np.float32)
        try:
            far_mask = dists_to_goal > FAR_THRESHOLD
            near_mask = dists_to_goal < NEAR_THRESHOLD
            transition_mask = ~(far_mask | near_mask)
            dynamic_weights[far_mask] = WEIGHT_FAR
            dynamic_weights[near_mask] = WEIGHT_NEAR
            if np.any(transition_mask):
                ratio = (dists_to_goal[transition_mask] - NEAR_THRESHOLD) / (FAR_THRESHOLD - NEAR_THRESHOLD)
                dynamic_weights[transition_mask] = WEIGHT_NEAR - ratio * (WEIGHT_NEAR - WEIGHT_FAR)
        except Exception:
            dynamic_weights.fill(WEIGHT_FAR)

        d_min_current = np.nan_to_num(
            d_min_current,
            nan=far_distance_fill,
            posinf=far_distance_fill,
            neginf=-far_distance_fill,
        )

        d_min_previous = np.full(num_positions, far_distance_fill, dtype=np.float32)
        velocities = np.zeros((num_positions, 3), dtype=np.float32)
        for idx, agent in enumerate(agents):
            try:
                prev_value = getattr(agent, 'last_min_distance')
                if isinstance(prev_value, np.ndarray):
                    if prev_value.size > 0:
                        prev_scalar = float(prev_value.item() if prev_value.ndim == 0 else prev_value.reshape(-1)[-1])
                    else:
                        prev_scalar = far_distance_fill
                else:
                    prev_scalar = float(prev_value)
                if np.isfinite(prev_scalar):
                    d_min_previous[idx] = prev_scalar
            except Exception:
                d_min_previous[idx] = d_min_current[idx]

            try:
                raw_vel = getattr(getattr(agent, 'state', None), 'p_vel', None)
                if raw_vel is not None:
                    vel_arr = np.asarray(raw_vel, dtype=np.float32).reshape(-1)
                    if vel_arr.shape[0] >= 3:
                        velocities[idx] = vel_arr[:3]
            except Exception:
                pass

        distance_change = d_min_current - d_min_previous
        normalized_change = np.clip(distance_change / clearance_d_max, -1.0, 1.0)

        hazard_vectors = positions - reference_points
        hazard_norms = np.linalg.norm(hazard_vectors, axis=1, keepdims=True)
        hazard_dirs = np.zeros_like(hazard_vectors)
        valid_hazard_mask = hazard_norms[:, 0] > 1e-6
        if np.any(valid_hazard_mask):
            hazard_dirs[valid_hazard_mask] = hazard_vectors[valid_hazard_mask] / hazard_norms[valid_hazard_mask]

        speeds = np.linalg.norm(velocities, axis=1)
        radial_speed = np.sum(velocities * hazard_dirs, axis=1)
        motion_signal = np.zeros(num_positions, dtype=np.float32)
        moving_mask = valid_hazard_mask & (speeds > speed_eps)
        if np.any(moving_mask):
            motion_signal[moving_mask] = np.clip(radial_speed[moving_mask] / speed_scale, -1.0, 1.0)

        combined_trend = np.clip(normalized_change + velocity_weight * motion_signal, -1.0, 1.0)
        trend_reward = np.zeros(num_positions, dtype=np.float32)
        improving_mask = combined_trend > trend_eps
        worsening_mask = combined_trend < -trend_eps
        if np.any(improving_mask):
            trend_reward[improving_mask] = dynamic_weights[improving_mask] * combined_trend[improving_mask]
        if np.any(worsening_mask):
            trend_reward[worsening_mask] = (
                dynamic_weights[worsening_mask]
                * combined_trend[worsening_mask]
                * trend_penalty_factor
            )

        try:
            goal_focus_radius = float(self.clearance_penalty_only_near_goal_radius)
            success_thr = float(self.success_distance_threshold)
            if goal_focus_radius > success_thr:
                clearance_goal_focus_mask = (
                    (dists_to_goal > success_thr) &
                    (dists_to_goal < goal_focus_radius)
                )
                if np.any(clearance_goal_focus_mask):
                    positive_mask = clearance_goal_focus_mask & (trend_reward > 0.0)
                    if np.any(positive_mask):
                        trend_reward[positive_mask] *= float(self.clearance_near_goal_positive_factor)
        except Exception:
            pass

        warning_ratio = np.clip((safe_distance - d_min_current) / warning_span, 0.0, 1.0)
        warning_penalty = -penalty_weight * warning_ratio
        trend_active_mask = d_min_current >= collision_threshold
        rewards = warning_penalty + np.where(trend_active_mask, trend_reward, 0.0)

        max_positive = max(clearance_weight, WEIGHT_NEAR)
        min_negative = penalty_weight + max(clearance_weight, WEIGHT_NEAR) * trend_penalty_factor
        rewards = np.clip(rewards, -min_negative, max_positive * 2.0)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=max_positive * 2.0, neginf=-min_negative).astype(np.float32)

        for idx, agent in enumerate(agents):
            agent.last_min_distance = float(d_min_current[idx])
            try:
                if not hasattr(agent, 'debug_info') or not isinstance(agent.debug_info, dict):
                    agent.debug_info = {}
                agent.debug_info['d_min_current'] = float(d_min_current[idx])
                agent.debug_info['clearance_radial_speed'] = float(radial_speed[idx])
                agent.debug_info['clearance_motion_signal'] = float(motion_signal[idx])
                agent.debug_info['clearance_reference'] = 'terrain' if bool(use_terrain_reference[idx]) else 'obstacle'
                # 维护 episode 级最小安全距离（用于 success-only 质量奖励）
                try:
                    if world is not None:
                        prev_min = getattr(world, '_episode_dmin_min', None)
                        cur = float(d_min_current[idx])
                        if prev_min is None or (np.isfinite(cur) and (not np.isfinite(prev_min) or cur < float(prev_min))):
                            setattr(world, '_episode_dmin_min', cur)
                except Exception:
                    pass
            except Exception:
                pass

        return rewards

    def _batch_collision_penalty_vectorized(
        self,
        agents: List[Any],
        world: Any,
        scenario: Any,
        positions: np.ndarray,
        cached_data: Dict[str, Any] = None,
        geometry_context: Dict[str, Any] = None,
    ) -> np.ndarray:
        """按当前 world 的所有智能体批量计算碰撞惩罚。"""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        positions = np.asarray(positions[:, :3], dtype=np.float32)
        num_positions = positions.shape[0]
        if num_positions == 0:
            return np.zeros(0, dtype=np.float32)

        if len(agents) != num_positions or cached_data is None or not self.use_fast_path:
            penalties = np.zeros(num_positions, dtype=np.float32)
            for idx in range(min(len(agents), num_positions)):
                single = self._collision_penalty_vectorized(idx, world, scenario, positions[idx:idx + 1], cached_data)
                penalties[idx] = single[0] if isinstance(single, np.ndarray) else float(single)
            return penalties

        penalties = np.zeros(num_positions, dtype=np.float32)
        cached_data = cached_data if isinstance(cached_data, dict) else {}
        collision_threshold = float(self.collision_distance_threshold)
        geometry_ready = False
        if isinstance(geometry_context, dict):
            try:
                obstacle_min_dist = np.asarray(geometry_context['obstacle_min_dist'], dtype=np.float32)
                terrain_min_dist = np.asarray(geometry_context['terrain_min_dist'], dtype=np.float32)
                terrain_heights_current = np.asarray(geometry_context['terrain_heights_current'], dtype=np.float32)
                geometry_ready = (
                    int(geometry_context.get('num_positions', -1)) == num_positions
                    and obstacle_min_dist.shape == (num_positions,)
                    and terrain_min_dist.shape == (num_positions,)
                    and terrain_heights_current.shape == (num_positions,)
                )
            except Exception:
                geometry_ready = False

        if not geometry_ready:
            obstacle_min_dist, _, _ = self._compute_obstacle_distance_data(
                positions,
                cached_data=cached_data,
                world=world,
            )
            try:
                terrain_heights_current, terrain_min_dist, _ = self._compute_terrain_distance_data(
                    positions,
                    scenario,
                    cached_data=cached_data,
                )
            except Exception:
                terrain_heights_current = np.full(num_positions, np.nan, dtype=np.float32)
                terrain_min_dist = np.full(num_positions, np.inf, dtype=np.float32)
        d_min_current = np.minimum(obstacle_min_dist, terrain_min_dist)

        distance_based_collision_mask = (d_min_current < collision_threshold) | (d_min_current < 0.0)
        real_distance_collision_mask = d_min_current < 0.0
        if np.any(distance_based_collision_mask):
            penalties[distance_based_collision_mask] = self._piecewise_collision_penalty(
                d_min_current[distance_based_collision_mask],
                collision_threshold,
                self.collision_penalty_value,
            )

        terrain = cached_data.get('terrain')
        terrain_heights = np.full(num_positions, np.nan, dtype=np.float32)
        invalid_mask = np.zeros(num_positions, dtype=bool)
        penetration_mask = np.zeros(num_positions, dtype=bool)
        contact_penalty_mask = np.zeros(num_positions, dtype=bool)
        actual_penetration_mask = np.zeros(num_positions, dtype=bool)
        obs_collision_mask = np.zeros(num_positions, dtype=bool)

        if terrain is not None:
            terrain_heights = terrain_heights_current.copy() if geometry_ready else self._query_terrain_heights_batch(scenario, cached_data, positions)
            invalid_mask = ~np.isfinite(terrain_heights)

            if np.any(invalid_mask):
                penalties[invalid_mask] = np.minimum(
                    penalties[invalid_mask],
                    -float(self.terrain_penalty_value),
                )
                terrain_heights = terrain_heights.copy()
                terrain_heights[invalid_mask] = positions[invalid_mask, 2]

            eps = float(self.terrain_collision_eps)
            penetration_mask = (positions[:, 2] < terrain_heights + eps) & (~distance_based_collision_mask)
            if np.any(penetration_mask):
                terrain_signed_clearance = positions[penetration_mask, 2] - terrain_heights[penetration_mask]
                penalties[penetration_mask] = np.minimum(
                    penalties[penetration_mask],
                    self._piecewise_collision_penalty(
                        terrain_signed_clearance,
                        eps,
                        self.terrain_penalty_value,
                    ),
                )

            success_thresh = float(getattr(self, 'success_distance_threshold', 2.0))
            goal_positions = cached_data.get('goal_positions')
            in_goal_area = np.zeros(num_positions, dtype=bool)
            if goal_positions is not None and len(goal_positions) > 0:
                goal_pos = np.asarray(goal_positions[0], dtype=np.float32).reshape(-1)[:3]
                if goal_pos.shape[0] == 3:
                    dists_to_goal = np.linalg.norm(positions - goal_pos, axis=-1)
                    in_goal_area = dists_to_goal <= success_thresh

            contact_mask = positions[:, 2] <= (terrain_heights + float(self.terrain_contact_eps))
            contact_penalty_mask = contact_mask & (~in_goal_area) & (~distance_based_collision_mask)
            if np.any(contact_penalty_mask):
                contact_signed_clearance = positions[contact_penalty_mask, 2] - terrain_heights[contact_penalty_mask]
                penalties[contact_penalty_mask] = np.minimum(
                    penalties[contact_penalty_mask],
                    self._piecewise_collision_penalty(
                        contact_signed_clearance,
                        float(self.terrain_contact_eps),
                        self.terrain_penalty_value,
                    ),
                )
                collision_eps = float(self.terrain_collision_eps)
                actual_penetration_mask = contact_penalty_mask & (
                    positions[:, 2] <= terrain_heights + collision_eps
                )

        obs_collision_mask = (obstacle_min_dist < 0.0) & (~distance_based_collision_mask)
        if np.any(obs_collision_mask):
            penalties[obs_collision_mask] = np.minimum(
                penalties[obs_collision_mask],
                self._piecewise_collision_penalty(
                    obstacle_min_dist[obs_collision_mask],
                    collision_threshold,
                    self.collision_penalty_value,
                ),
            )

        for idx, agent in enumerate(agents):
            if not hasattr(agent, 'debug_info') or not isinstance(agent.debug_info, dict):
                agent.debug_info = {}
            agent.debug_info.setdefault('terrain_penetration_count', 0)
            agent.debug_info.setdefault('obstacle_collision_count', 0)

            total_add = 0
            terrain_add = 0
            obstacle_add = 0

            if real_distance_collision_mask[idx]:
                agent._episode_has_collision = True
                agent._had_penetration_or_collision = True
                total_add += 1
                if terrain_min_dist[idx] <= obstacle_min_dist[idx]:
                    agent._had_terrain_contact_or_penetration = True
                    terrain_add += 1
                else:
                    agent._had_obstacle_collision = True
                    obstacle_add += 1

            if invalid_mask[idx]:
                agent._episode_has_collision = True
                agent._had_penetration_or_collision = True
                agent._had_terrain_contact_or_penetration = True
                total_add += 1
                terrain_add += 1

            if penetration_mask[idx]:
                agent._episode_has_collision = True
                agent._had_penetration_or_collision = True
                agent._had_terrain_contact_or_penetration = True
                total_add += 1
                terrain_add += 1

            if contact_penalty_mask[idx]:
                if actual_penetration_mask[idx]:
                    agent._episode_has_collision = True
                    agent._had_penetration_or_collision = True
                    agent._had_terrain_contact_or_penetration = True
                    total_add += 1
                    terrain_add += 1

            if obs_collision_mask[idx]:
                agent._episode_has_collision = True
                agent._had_penetration_or_collision = True
                agent._had_obstacle_collision = True
                total_add += 1
                obstacle_add += 1

            if total_add > 0:
                new_total = int(agent.debug_info.get('total_penetration_count', 0)) + total_add
                agent.debug_info['total_penetration_count'] = min(new_total, 1000000)
            if terrain_add > 0:
                agent.debug_info['terrain_penetration_count'] = (
                    agent.debug_info.get('terrain_penetration_count', 0) + terrain_add
                )
            if obstacle_add > 0:
                agent.debug_info['obstacle_collision_count'] = (
                    agent.debug_info.get('obstacle_collision_count', 0) + obstacle_add
                )

        return penalties

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
        reward_timing_enabled = self._reward_timing_enabled()
        reward_timing = self._new_reward_timing_bucket() if reward_timing_enabled else None
        _reward_perf_counter = time.perf_counter if reward_timing_enabled else None
        self._last_reward_timing = None
        try:
            batch_size = len(agents_batch)
            n_agents = len(agents_batch[0]) if batch_size > 0 else 0
            
            if batch_size == 0 or n_agents == 0:
                if reward_timing is not None:
                    self._last_reward_timing = reward_timing
                return np.zeros((batch_size, n_agents), dtype=np.float32)
            
            # 预处理缓存数据（性能优化）
            if reward_timing is not None:
                _t_reward_seg = _reward_perf_counter()
            cached_data_batch = []
            for b in range(batch_size):
                world = world_batch[b] if world_batch else None
                scenario = scenario_batch[b] if scenario_batch and b < len(scenario_batch) else None
                cached_data = self._update_caches(world, scenario) if self.use_fast_path else None
                cached_data_batch.append(cached_data)
            if reward_timing is not None:
                reward_timing['rew_cache'] += _reward_perf_counter() - _t_reward_seg
            
            # 预分配数组
            arrays = self._get_preallocated_arrays(batch_size, n_agents)
            
            # 向后兼容：若未提供批次数据，则从agents提取，或用保守默认值
            if reward_timing is not None:
                _t_reward_seg = _reward_perf_counter()
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
            if reward_timing is not None:
                reward_timing['rew_state'] += _reward_perf_counter() - _t_reward_seg

            # 核心奖励计算
            try:
                self._calculate_all_rewards_vectorized(
                    agents_batch,
                    world_batch,
                    scenario_batch,
                    arrays,
                    cached_data_batch,
                    reward_timing=reward_timing,
                )
            except Exception as e:
                print(f"批量奖励计算异常: {e}")
                import traceback
                traceback.print_exc()

            # 应用权重并求和（对齐通道数，防御性处理）
            if reward_timing is not None:
                _t_reward_seg = _reward_perf_counter()
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
            # === 主奖励：默认启用“结构收缩”的 dense reward ===
            # dense: 仅保留少数核心项（progress / success / collision / clearance / height / stationary / team-sync）；
            # 其余通道仍可保留实现与日志，但不参与主优化目标。
            if getattr(self, 'restructured_reward_enabled', False):
                dense_indices = list(getattr(self, '_dense_indices_core', ()))
                if getattr(self, 'dense_energy_enabled', False):
                    dense_indices.append(7)  # energy 作为弱正则
                dense_indices = [int(i) for i in dense_indices if 0 <= int(i) < rewards_mat.shape[2]]
                if dense_indices:
                    if getattr(self, 'progress_merge_enabled', False) and weights_vec.shape[0] > 6:
                        dense_weights_vec = np.array(weights_vec, copy=True)
                        dense_weights_vec[0] = np.float32(abs(float(weights_vec[0])) + abs(float(weights_vec[6])))
                        dense_weights_vec[6] = np.float32(0.0)
                    else:
                        dense_weights_vec = weights_vec
                    dense_rewards = rewards_mat[:, :, dense_indices]
                    dense_weights = dense_weights_vec[dense_indices]
                    total_rewards = np.sum(dense_rewards * dense_weights[None, None, :], axis=2)
                else:
                    total_rewards = np.zeros(rewards_mat.shape[:2], dtype=np.float32)
            else:
                total_rewards = np.sum(rewards_mat * weights_vec, axis=2)

            # === 终局项：episode end 单独结算（不参与逐步累积竞争） ===
            # - team_success_bonus（团队成功主导项）
            # - unsafe_arrival_penalty（全员到达但非安全成功）
            # - terminal_failure_penalty（从未进入成功圈的终局惩罚）
            # - success-only quality bonuses（仅 team_success=True：clearance/efficiency）
            try:
                for b, world in enumerate(world_batch):
                    if not self._is_episode_finished(world):
                        continue

                    n_agents_b = int(total_rewards[b].shape[0]) if hasattr(total_rewards[b], 'shape') else 0
                    n_agents_b = max(n_agents_b, 1)

                    # 1) 终局失败惩罚：未曾进入成功圈者按剩余距离惩罚
                    try:
                        valid_goal_mask = np.isfinite(arrays['goals'][b]).all(axis=1)
                        penalties = self._terminal_failure_penalty_batch(
                            agents_batch[b],
                            arrays['positions'][b],
                            arrays['goals'][b],
                            valid_goal_mask,
                            arrays['start_positions'][b],
                            world,
                        ).astype(np.float32)
                        total_rewards[b] = total_rewards[b] + penalties
                    except Exception:
                        pass

                    team_success = bool(getattr(world, '_episode_success', False))
                    all_reached = bool(getattr(world, '_episode_all_reached', False))

                    if team_success:
                        # 2) 团队成功终局奖励：平均分配
                        tsb = float(getattr(self, 'team_success_bonus', 0.0))
                        if tsb != 0.0:
                            total_rewards[b] = total_rewards[b] + (tsb / float(n_agents_b))

                        # 3) success-only 质量奖励：净空（D_min^(k)）越大越好
                        try:
                            dmin = float(getattr(world, '_episode_dmin_min', np.inf))
                            if np.isfinite(dmin):
                                safe_d = None
                                try:
                                    scenario = scenario_batch[b] if scenario_batch and b < len(scenario_batch) else None
                                    if scenario is not None:
                                        safe_d = getattr(scenario, 'obstacle_safe_distance', None)
                                except Exception:
                                    safe_d = None
                                if safe_d is None:
                                    safe_d = 5.0
                                safe_d = float(max(float(safe_d), 1e-6))
                                q = float(np.clip(dmin / safe_d, 0.0, 1.0))
                                cqb = float(getattr(self, 'clearance_quality_bonus_weight', 0.0)) * q
                                if cqb != 0.0:
                                    total_rewards[b] = total_rewards[b] + (cqb / float(n_agents_b))
                        except Exception:
                            pass

                        # 4) success-only 质量奖励：效率（effective_steps 越少越好）
                        try:
                            eff_steps = float(getattr(world, 'current_step', 0))
                            ep_len = None
                            try:
                                if hasattr(world, 'episode_length') and world.episode_length is not None:
                                    ep_len = float(world.episode_length)
                                elif hasattr(world, 'max_steps') and world.max_steps is not None:
                                    ep_len = float(world.max_steps)
                            except Exception:
                                ep_len = None
                            if ep_len is None or ep_len <= 0:
                                import os as _os
                                ep_len = float(_os.getenv('EPISODE_LENGTH', '2800'))
                            ep_len = float(max(ep_len, 1.0))
                            eff = float(np.clip(1.0 - (eff_steps / ep_len), 0.0, 1.0))
                            eb = float(getattr(self, 'efficiency_bonus_weight', 0.0)) * eff
                            if eb != 0.0:
                                total_rewards[b] = total_rewards[b] + (eb / float(n_agents_b))
                        except Exception:
                            pass
                    else:
                        # 全员到达但不安全成功：终局惩罚（平均分配）
                        uap = float(getattr(self, 'unsafe_arrival_penalty', 0.0))
                        if all_reached and uap != 0.0:
                            total_rewards[b] = total_rewards[b] - (uap / float(n_agents_b))
            except Exception:
                pass
            
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
            # 奖励多样性诊断默认关闭；显式启用时也只在有效 episode 编号上按回合间隔检查一次。
            if self.enable_reward_diversity_check:
                try:
                    episode_int = int(episode)
                except Exception:
                    episode_int = -1
                if episode_int >= 0:
                    should_check_diversity = (
                        self._last_diversity_checked_episode != episode_int
                        and (episode_int % self.reward_diversity_check_interval == 0)
                    )
                    if should_check_diversity:
                        self._check_reward_diversity(total_rewards, episode=episode_int)
                        self._last_diversity_checked_episode = episode_int
            if reward_timing is not None:
                reward_timing['rew_reduce'] += _reward_perf_counter() - _t_reward_seg
                self._last_reward_timing = reward_timing
            
            # 限制范围
            return np.clip(total_rewards, self.min_reward, self.max_reward)
            
        except Exception as e:
            if reward_timing is not None:
                self._last_reward_timing = reward_timing
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
                'rewards': np.zeros((batch_size, n_agents, len(self.reward_names)), dtype=np.float32),
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
    
    def _calculate_all_rewards_vectorized(
        self,
        agents_batch: List[List[Any]],
        world_batch: List[Any],
        scenario_batch: List[Any],
        arrays: Dict[str, np.ndarray],
        cached_data_batch: List[Any] = None,
        reward_timing: Dict[str, float] = None,
    ):
        """向量化计算所有奖励分项"""
        batch_size, n_agents = arrays['rewards'].shape[:2]
        _reward_perf_counter = time.perf_counter if reward_timing is not None else None
        
        # 重置奖励数组
        arrays['rewards'].fill(0.0)
        
        # 批量计算各项奖励
        for b, (agents, world, scenario) in enumerate(zip(agents_batch, world_batch, scenario_batch)):
            # 取出该批次的缓存数据
            cached_data = None
            if cached_data_batch is not None and b < len(cached_data_batch):
                cached_data = cached_data_batch[b]
            pos_batch = arrays['positions'][b]
            prev_pos_batch = arrays['prev_positions'][b]
            start_pos_batch = arrays['start_positions'][b]
            action_batch = arrays['actions'][b]
            rewards_batch = arrays['rewards'][b]

            _full_ch = (
                not getattr(self, 'restructured_reward_enabled', False)
                or getattr(self, 'restructured_full_channel_compute', False)
            )
            if _full_ch:
                _need_ch = set(range(16))
            else:
                _need_ch = {int(i) for i in getattr(self, '_dense_indices_core', ())}
                if getattr(self, 'dense_energy_enabled', False):
                    _need_ch.add(7)

            goal_positions = np.zeros((n_agents, 3), dtype=np.float32)
            valid_goal_mask = np.zeros(n_agents, dtype=bool)

            fallback_goal = None
            try:
                if scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                    fallback_goal = np.asarray(scenario.goal_pos, dtype=np.float32).reshape(-1)[:3]
                    if fallback_goal.shape[0] < 3:
                        fallback_goal = None
            except Exception:
                fallback_goal = None

            for a, agent in enumerate(agents):
                goal_pos = None
                try:
                    if hasattr(agent, 'goal_a') and agent.goal_a is not None:
                        if hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                            goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32).reshape(-1)[:3]
                            if goal_pos.shape[0] < 3:
                                goal_pos = None
                except Exception:
                    goal_pos = None

                if goal_pos is None:
                    goal_pos = fallback_goal

                if goal_pos is not None:
                    goal_positions[a] = goal_pos
                    valid_goal_mask[a] = True

            if np.any(valid_goal_mask) and (
                _full_ch or (0 in _need_ch) or (4 in _need_ch) or (6 in _need_ch) or (9 in _need_ch) or (11 in _need_ch)
            ):
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                valid_pos = pos_batch[valid_goal_mask]
                valid_prev_pos = prev_pos_batch[valid_goal_mask]
                valid_start_pos = start_pos_batch[valid_goal_mask]
                valid_goals = goal_positions[valid_goal_mask]

                current_dist = np.linalg.norm(valid_pos - valid_goals, axis=-1)
                prev_dist = np.linalg.norm(valid_prev_pos - valid_goals, axis=-1)

                distance_reward = np.zeros_like(current_dist, dtype=np.float32)
                approach_reward = np.zeros_like(current_dist, dtype=np.float32)

                if _full_ch or (0 in _need_ch) or (6 in _need_ch):
                    # 1. progress 合并前的 distance 锚点：接近目标时逐步衰减成功圈外的状态型正奖励，
                    # 防止策略停在目标外几米稳定刷分。
                    initial_dist = np.linalg.norm(valid_start_pos - valid_goals, axis=-1)
                    denom = np.maximum(initial_dist, 1.0)
                    ratio = np.clip(current_dist / denom, 0.0, 2.0)
                    distance_reward = (1.0 - ratio) * 10.0
                    distance_reward = self._attenuate_distance_reward_near_goal(
                        distance_reward,
                        current_dist,
                    )

                if _full_ch or (4 in _need_ch):
                    # 5. 偏离惩罚：与原实现一致，按起点-目标线段计算侧向偏离
                    path_vec = valid_goals - valid_start_pos
                    path_len = np.linalg.norm(path_vec, axis=-1)
                    denom_dev = np.maximum(path_len, 1.0)
                    w = valid_pos - valid_start_pos
                    t = np.sum(w * path_vec, axis=-1) / (denom_dev * denom_dev)
                    t = np.clip(t, 0.0, 1.0)
                    proj = valid_start_pos + t[:, None] * path_vec
                    d_perp = np.linalg.norm(valid_pos - proj, axis=-1)
                    norm_dev = np.clip(d_perp / denom_dev, 0.0, 2.0)
                    rewards_batch[valid_goal_mask, 4] = 1.0 - norm_dev

                if _full_ch or (0 in _need_ch) or (6 in _need_ch):
                    # 2. progress 合并前的 approach：近目标区倍数由 APPROACH_NEAR_GOAL_MAX_MULT 限制（默认弱化，避免终点外强刷）
                    approach_reward = prev_dist - current_dist
                    near_goal_threshold = float(
                        getattr(self, 'approach_near_goal_threshold', np.float32(50.0))
                    )
                    near_goal_threshold = max(near_goal_threshold, 1e-3)
                    app_max = float(getattr(self, 'approach_near_goal_max_mult', np.float32(1.22)))
                    app_max = max(app_max, 1.0)
                    approach_weight_multipliers = np.ones_like(current_dist, dtype=np.float32)
                    near_goal_mask = current_dist < near_goal_threshold
                    if np.any(near_goal_mask):
                        t = current_dist[near_goal_mask] / near_goal_threshold
                        approach_weight_multipliers[near_goal_mask] = (
                            np.float32(1.0) + (np.float32(app_max) - np.float32(1.0)) * (np.float32(1.0) - t)
                        ).astype(np.float32)
                    approach_weight_multipliers = np.clip(
                        approach_weight_multipliers, 1.0, max(app_max, 1.0)
                    )
                    approach_reward = approach_reward * 5.0 * approach_weight_multipliers

                if _full_ch or (0 in _need_ch):
                    # 主 dense progress：将 distance(状态锚点) 与 approach(步进结果) 收成单一通道
                    merged_progress = self._merge_progress_rewards(
                        distance_reward,
                        approach_reward,
                        distance_weight=float(self.reward_weights[0]) if len(self.reward_weights) > 0 else 1.0,
                        approach_weight=float(self.reward_weights[6]) if len(self.reward_weights) > 6 else 1.0,
                    )
                    rewards_batch[valid_goal_mask, 0] = merged_progress

                # legacy approach 通道保留占位，但默认主路径不再单独计分
                rewards_batch[valid_goal_mask, 6] = 0.0
                if reward_timing is not None:
                    reward_timing['rew_numeric'] += _reward_perf_counter() - _t_reward_seg

            # 8. 能量消耗惩罚：保持现有公式，避免每个 agent 单独构造小数组
            if _full_ch or (7 in _need_ch):
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                energy_consumption = np.linalg.norm(action_batch, axis=-1) * 0.1
                current_speed = np.linalg.norm(action_batch, axis=-1)
                energy_reward = np.zeros_like(current_speed, dtype=np.float32)
                active_energy_mask = energy_consumption > 0.1
                if np.any(active_energy_mask):
                    speed_efficiency = current_speed[active_energy_mask] / np.maximum(
                        energy_consumption[active_energy_mask], 1e-6
                    )
                    energy_reward[active_energy_mask] = np.minimum(speed_efficiency * 0.1, 2.0)
                rewards_batch[:, 7] = energy_reward
                if reward_timing is not None:
                    reward_timing['rew_numeric'] += _reward_perf_counter() - _t_reward_seg

            geometry_context = None
            _need_geom = _full_ch or (10 in _need_ch) or (13 in _need_ch)
            if _need_geom and self.use_fast_path and len(agents) == pos_batch.shape[0]:
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                try:
                    geometry_context = self._build_reward_geometry_context(
                        pos_batch,
                        world,
                        scenario,
                        cached_data=cached_data,
                    )
                except Exception:
                    geometry_context = None
                if reward_timing is not None:
                    reward_timing['rew_state'] += _reward_perf_counter() - _t_reward_seg

            # 9. 高度奖励：保留原数值规则，优先复用几何上下文中的当前位置地形高度
            if (
                (_full_ch or (8 in _need_ch))
                and getattr(self, 'height_reward_enabled', True)
                and scenario is not None
                and hasattr(scenario, 'get_terrain_height')
            ):
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                terrain_heights = np.full(pos_batch.shape[0], np.nan, dtype=np.float32)
                try:
                    geometry_ready = False
                    if isinstance(geometry_context, dict):
                        try:
                            terrain_heights_current = np.asarray(
                                geometry_context['terrain_heights_current'],
                                dtype=np.float32
                            )
                            geometry_ready = (
                                int(geometry_context.get('num_positions', -1)) == pos_batch.shape[0]
                                and terrain_heights_current.shape == (pos_batch.shape[0],)
                            )
                            if geometry_ready:
                                terrain_heights = terrain_heights_current
                        except Exception:
                            geometry_ready = False

                    if not geometry_ready:
                        cached_terrain = cached_data.get('terrain') if isinstance(cached_data, dict) else None
                        cached_map_size = int(
                            cached_data.get('map_size', getattr(scenario, 'map_size', 200))
                        ) if isinstance(cached_data, dict) else int(getattr(scenario, 'map_size', 200))

                        if cached_terrain is not None:
                            query_positions = np.zeros((pos_batch.shape[0], 3), dtype=np.float32)
                            query_positions[:, :2] = pos_batch[:, :2]
                            _bilinear_interpolate_terrain(cached_terrain, query_positions, terrain_heights, cached_map_size)

                        invalid_terrain = ~np.isfinite(terrain_heights)
                        if np.any(invalid_terrain) and hasattr(scenario, 'get_terrain_height_vectorized'):
                            sampled = scenario.get_terrain_height_vectorized(
                                pos_batch[invalid_terrain, 0], pos_batch[invalid_terrain, 1]
                            )
                            sampled = np.asarray(sampled, dtype=np.float32)
                            if sampled.ndim == 0:
                                terrain_heights[invalid_terrain] = np.float32(sampled)
                            elif sampled.shape[0] == np.count_nonzero(invalid_terrain):
                                terrain_heights[invalid_terrain] = sampled

                        invalid_terrain = ~np.isfinite(terrain_heights)
                        if np.any(invalid_terrain):
                            for idx in np.where(invalid_terrain)[0]:
                                try:
                                    terrain_heights[idx] = scenario.get_terrain_height(
                                        float(pos_batch[idx, 0]), float(pos_batch[idx, 1])
                                    )
                                except Exception:
                                    continue
                except Exception:
                    terrain_heights.fill(np.nan)

                if np.any(np.isfinite(terrain_heights)):
                    height_reward = np.zeros(pos_batch.shape[0], dtype=np.float32)
                    finite_terrain = np.isfinite(terrain_heights)
                    terrain_heights = np.nan_to_num(terrain_heights, nan=0.0)
                    height_diff = pos_batch[:, 2] - terrain_heights
                    height_goal_focus_mask = np.zeros(pos_batch.shape[0], dtype=bool)
                    try:
                        goal_focus_radius = float(self.height_penalty_only_near_goal_radius)
                        success_thr = float(self.success_distance_threshold)
                        if goal_focus_radius > success_thr and np.any(valid_goal_mask):
                            height_goal_dists = np.full(pos_batch.shape[0], np.inf, dtype=np.float32)
                            height_goal_dists[valid_goal_mask] = np.linalg.norm(
                                pos_batch[valid_goal_mask] - goal_positions[valid_goal_mask],
                                axis=-1,
                            ).astype(np.float32)
                            height_goal_focus_mask = (
                                (height_goal_dists > success_thr) &
                                (height_goal_dists < goal_focus_radius)
                            )
                    except Exception:
                        height_goal_focus_mask = np.zeros(pos_batch.shape[0], dtype=bool)

                    ideal_min = float(getattr(self, 'height_ideal_min', 2.0))
                    ideal_max = float(getattr(self, 'height_ideal_max', 5.0))
                    if ideal_min > ideal_max:
                        ideal_min, ideal_max = ideal_max, ideal_min

                    in_range = finite_terrain & (height_diff >= ideal_min) & (height_diff <= ideal_max)
                    below_range = finite_terrain & (height_diff < ideal_min)
                    above_range = finite_terrain & (height_diff > ideal_max)

                    height_reward[in_range & (~height_goal_focus_mask)] = 1.0
                    if np.any(in_range & height_goal_focus_mask):
                        height_reward[in_range & height_goal_focus_mask] = float(
                            self.height_near_goal_positive_factor
                        )

                    if np.any(below_range):
                        shortage = ideal_min - height_diff[below_range]
                        low_reward = -shortage * 1.5

                        low_height = height_diff[below_range]
                        danger_mask = low_height < 3.0
                        if np.any(danger_mask):
                            danger_level = 3.0 - low_height[danger_mask]
                            low_reward[danger_mask] -= danger_level * 3.0

                        penetration_mask = low_height < 0.0
                        if np.any(penetration_mask):
                            penetration_depth = -low_height[penetration_mask]
                            low_reward[penetration_mask] -= penetration_depth * 15.0

                        upward_reward = np.zeros(np.count_nonzero(below_range), dtype=np.float32)
                        below_indices = np.where(below_range)[0]
                        below_z_velocity = pos_batch[below_indices, 2] - prev_pos_batch[below_indices, 2]
                        upward_mask = below_z_velocity > 0.0
                        if np.any(upward_mask):
                            upward_reward[upward_mask] = np.clip(
                                below_z_velocity[upward_mask] * 2.0, 0.0, 2.0
                            )

                        if action_batch.shape[1] >= 3:
                            z_actions = action_batch[below_indices, 2]
                            upward_action_mask = z_actions > 0.0
                            if np.any(upward_action_mask):
                                upward_reward[upward_action_mask] += np.clip(
                                    z_actions[upward_action_mask] * 1.0, 0.0, 1.0
                                )

                        if np.any(height_goal_focus_mask[below_indices]):
                            upward_reward[height_goal_focus_mask[below_indices]] *= float(
                                self.height_near_goal_positive_factor
                            )

                        height_reward[below_range] = low_reward + upward_reward

                    if np.any(above_range):
                        height_reward[above_range] = -(height_diff[above_range] - ideal_max) * 0.5

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

                    rewards_batch[:, 8] = height_reward
            if reward_timing is not None:
                reward_timing['rew_numeric'] += _reward_perf_counter() - _t_reward_seg

            if _full_ch or (10 in _need_ch):
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                batch_collision_penalties = self._batch_collision_penalty_vectorized(
                    agents,
                    world,
                    scenario,
                    pos_batch,
                    cached_data=cached_data,
                    geometry_context=geometry_context,
                )
                rewards_batch[:, 10] = batch_collision_penalties
                if reward_timing is not None:
                    reward_timing['rew_collision'] += _reward_perf_counter() - _t_reward_seg

            if _full_ch or (13 in _need_ch):
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                batch_clearance_rewards = self._batch_clearance_reward_vectorized(
                    agents,
                    world,
                    pos_batch,
                    cached_data=cached_data,
                    geometry_context=geometry_context,
                )
                rewards_batch[:, 13] = batch_clearance_rewards
                if reward_timing is not None:
                    reward_timing['rew_clearance'] += _reward_perf_counter() - _t_reward_seg

            if (_full_ch or (0 in _need_ch)) and self._is_obstacle_reward_route():
                if reward_timing is not None:
                    _t_reward_seg = _reward_perf_counter()
                try:
                    adjusted_approach_reward = self._relax_approach_for_obstacle_route(
                        approach_reward,
                        geometry_context,
                    )
                    rewards_batch[valid_goal_mask, 0] = self._merge_progress_rewards(
                        distance_reward,
                        adjusted_approach_reward,
                        distance_weight=float(self.reward_weights[0]) if len(self.reward_weights) > 0 else 1.0,
                        approach_weight=float(self.reward_weights[6]) if len(self.reward_weights) > 6 else 1.0,
                    )
                except Exception:
                    pass
                if reward_timing is not None:
                    reward_timing['rew_numeric'] += _reward_perf_counter() - _t_reward_seg

            for a, agent in enumerate(agents):
                pos = pos_batch[a]
                prev_pos = prev_pos_batch[a]
                start_pos = start_pos_batch[a]

                if _full_ch or (1 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    expl_reward = self._exploration_reward_vectorized(agent, scenario, pos.reshape(1, -1))
                    rewards_batch[a, 1] = expl_reward[0] if isinstance(expl_reward, np.ndarray) else expl_reward
                    if reward_timing is not None:
                        reward_timing['rew_explore'] += _reward_perf_counter() - _t_reward_seg

                if _full_ch or (2 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    stat_penalty = self._stationary_penalty_vectorized(
                        agent, pos.reshape(1, -1), prev_pos.reshape(1, -1)
                    )
                    rewards_batch[a, 2] = stat_penalty[0] if isinstance(stat_penalty, np.ndarray) else stat_penalty
                    if reward_timing is not None:
                        reward_timing['rew_motion'] += _reward_perf_counter() - _t_reward_seg

                if _full_ch or (3 in _need_ch):
                    dir_reward = self._direction_reward_vectorized(
                        agent, pos.reshape(1, -1), prev_pos.reshape(1, -1)
                    )
                    rewards_batch[a, 3] = dir_reward[0] if isinstance(dir_reward, np.ndarray) else dir_reward

                if _full_ch or (5 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    start_reward = self._start_area_reward_vectorized(
                        agent, scenario, pos.reshape(1, -1), start_pos.reshape(1, -1)
                    )
                    rewards_batch[a, 5] = start_reward[0] if isinstance(start_reward, np.ndarray) else start_reward
                    if reward_timing is not None:
                        reward_timing['rew_motion'] += _reward_perf_counter() - _t_reward_seg

                if _full_ch or (9 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    success_reward = self._success_reward_vectorized(
                        agent, scenario, pos.reshape(1, -1), cached_data, agent_idx=a, world=world
                    )
                    rewards_batch[a, 9] = success_reward[0] if isinstance(success_reward, np.ndarray) else success_reward
                    if reward_timing is not None:
                        reward_timing['rew_success'] += _reward_perf_counter() - _t_reward_seg

                if _full_ch:
                    rewards_batch[a, 11] = 0.0

                if _full_ch or (12 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    try:
                        if hasattr(agent, 'goal_a') and agent.goal_a is not None and agent.goal_a.state.p_pos is not None:
                            _g = agent.goal_a.state.p_pos
                        else:
                            _g = scenario.goal_pos if hasattr(scenario, 'goal_pos') else None
                        dist_to_goal = float(np.linalg.norm(pos - _g)) if _g is not None else 0.0
                    except Exception:
                        dist_to_goal = 0.0
                    rewards_batch[a, 12] = self._potential_shaping_vectorized(agent, dist_to_goal)
                    if reward_timing is not None:
                        reward_timing['rew_motion'] += _reward_perf_counter() - _t_reward_seg

                if _full_ch or (14 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    lateral_reward = self._lateral_reward_vectorized(agent, world, scenario, pos.reshape(1, -1))
                    rewards_batch[a, 14] = lateral_reward[0] if isinstance(lateral_reward, np.ndarray) else lateral_reward
                    if reward_timing is not None:
                        reward_timing['rew_lateral'] += _reward_perf_counter() - _t_reward_seg

                if _full_ch or (15 in _need_ch):
                    if reward_timing is not None:
                        _t_reward_seg = _reward_perf_counter()
                    collision_reduction_reward = self._collision_reduction_reward_vectorized(
                        agent, world, scenario, pos.reshape(1, -1)
                    )
                    rewards_batch[a, 15] = (
                        collision_reduction_reward[0]
                        if isinstance(collision_reduction_reward, np.ndarray)
                        else collision_reduction_reward
                    )
                    if reward_timing is not None:
                        reward_timing['rew_motion'] += _reward_perf_counter() - _t_reward_seg

            if reward_timing is not None:
                _t_reward_seg = _reward_perf_counter()
            if _full_ch or (11 in _need_ch):
                team_sync_reward = self._team_sync_reward_vectorized(
                    agents_batch[b],
                    world_batch[b],
                    scenario_batch[b] if scenario_batch and b < len(scenario_batch) else None,
                    pos_batch,
                    goal_positions,
                    valid_goal_mask,
                )
                arrays['rewards'][b, :, 11] = team_sync_reward
            else:
                team_sync_reward = np.zeros((len(agents_batch[b]),), dtype=np.float32)

            if _full_ch:
                global_val = self._global_reward_vectorized(agents_batch[b])
            else:
                global_val = 0.0
            if _full_ch and global_val > 0.0:
                # 🚨 按个体贡献分配全局奖励，表现好的智能体得到更多
                try:
                    # 计算每个智能体的贡献度（基于个体成功奖励）
                    individual_contributions = []
                    for a, agent in enumerate(agents_batch[b]):
                        # 贡献度 = 个体成功奖励（已按无碰撞比例缩放）
                        individual_success = arrays['rewards'][b, a, 9]  # 个体成功奖励
                        individual_contributions.append(max(0.0, float(individual_success)))
                    
                    # 按贡献度分配全局奖励
                    total_contribution = sum(individual_contributions)
                    if total_contribution > 0.0:
                        for a in range(len(agents_batch[b])):
                            # 按贡献比例分配全局奖励
                            contribution_ratio = individual_contributions[a] / total_contribution
                            arrays['rewards'][b, a, 11] = arrays['rewards'][b, a, 11] + (global_val * contribution_ratio)
                    else:
                        # 如果所有智能体都没有成功奖励，平均分配（但这种情况不应该发生）
                        arrays['rewards'][b, :, 11] = arrays['rewards'][b, :, 11] + (global_val / len(agents_batch[b]))
                except Exception:
                    arrays['rewards'][b, :, 11] = arrays['rewards'][b, :, 11] + (global_val / len(agents_batch[b]))
            elif _full_ch:
                arrays['rewards'][b, :, 11] = arrays['rewards'][b, :, 11]
                terminal_failure_penalty = self._terminal_failure_penalty_batch(
                    agents_batch[b],
                    pos_batch,
                    goal_positions,
                    valid_goal_mask,
                    start_pos_batch,
                    world,
                )
                if np.any(terminal_failure_penalty < 0.0):
                    arrays['rewards'][b, :, 11] += terminal_failure_penalty

            # 结构收缩且不算全通道时：不往 ch9 写团队无碰撞奖励（主目标不含该通道），但必须清零标志避免累积
            try:
                world = world_batch[b]
                if not _full_ch and world is not None and hasattr(world, '_team_no_collision_reward'):
                    world._team_no_collision_reward = 0.0
            except Exception:
                pass

            # 🚨 关键修复：无碰撞奖励按个体表现分配，只给无碰撞的智能体
            try:
                world = world_batch[b]
                if _full_ch and world is not None and hasattr(world, '_team_no_collision_reward'):
                    team_no_collision_reward = getattr(world, '_team_no_collision_reward', 0.0)
                    if team_no_collision_reward > 0.0:
                        # 🚨 按个体表现分配：只给无碰撞且到达目标的智能体
                        for a, agent in enumerate(agents_batch[b]):
                            # 检查当前智能体是否到达目标
                            agent_reached = False
                            try:
                                if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                                    agent_pos = agent.state.p_pos
                                    agent_goal = None
                                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                                        agent_goal = agent.goal_a.state.p_pos
                                    elif hasattr(scenario_batch[b], 'goal_pos') and scenario_batch[b].goal_pos is not None:
                                        agent_goal = scenario_batch[b].goal_pos
                                    
                                    if agent_goal is not None:
                                        agent_dist = np.linalg.norm(np.array(agent_pos) - np.array(agent_goal))
                                        agent_reached = agent_dist <= self.success_distance_threshold
                            except Exception:
                                agent_reached = False
                            
                            # 检查当前智能体是否有真实碰撞/真实穿透（与成功判断逻辑保持一致）
                            agent_has_collision = True
                            try:
                                agent_has_collision = not self._agent_safe_so_far(agent)
                            except Exception:
                                # 如果检查失败，保守地认为有碰撞（不给奖励）
                                agent_has_collision = True
                            
                            # 🚨 关键修复：只有到达目标且无碰撞的智能体才得到无碰撞奖励
                            # 原因：用户明确指出"有专门的部分到达奖励，无碰撞奖励就应该无碰撞且到达才给"
                            # 修复：使用严格的碰撞检查（与成功判断逻辑一致），确保只有真正无碰撞的智能体才得到奖励
                            if agent_reached and not agent_has_collision:
                                arrays['rewards'][b, a, 9] = arrays['rewards'][b, a, 9] + team_no_collision_reward
                            # 表现不好的智能体（未到达目标或有碰撞）不得到无碰撞奖励
                        
                        # 重置标志，避免重复添加
                        world._team_no_collision_reward = 0.0
            except Exception:
                pass
            if reward_timing is not None:
                reward_timing['rew_team'] += _reward_perf_counter() - _t_reward_seg
            
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
            rewards = (1.0 - ratio) * 10.0
            rewards = self._attenuate_distance_reward_near_goal(rewards, current_dist)
            return rewards
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

        near_goal_radius = float(self.stationary_near_goal_radius)
        base_stationary_threshold = 0.005
        stationary_threshold = np.full(len(position), base_stationary_threshold, dtype=np.float32)
        near_goal_mask = dist_to_goal < near_goal_radius
        if np.any(near_goal_mask):
            stationary_threshold[near_goal_mask] = max(
                base_stationary_threshold,
                float(self.stationary_near_goal_threshold),
            )

        is_stationary = pos_change < stationary_threshold
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
            elif dist_to_goal < near_goal_radius:
                span = max(near_goal_radius - float(self.success_distance_threshold), 1e-6)
                proximity = 1.0 - np.clip(
                    (dist_to_goal - float(self.success_distance_threshold)) / span,
                    0.0,
                    1.0,
                )
                near_goal_penalty = (
                    float(self.stationary_near_goal_min_penalty) +
                    proximity * (
                        float(self.stationary_near_goal_max_penalty) -
                        float(self.stationary_near_goal_min_penalty)
                    )
                )
                rewards -= near_goal_penalty * penalty_multiplier
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

            # 🚨 关键修复：确保prev_positions和positions形状匹配
            # positions形状通常是(1, 3)或(n, 3)，prev_positions应该是相同的形状
            if prev_positions is None or len(prev_positions) != len(positions):
                # 如果prev_positions无效，使用当前位置作为prev_positions（第一步的情况）
                prev_positions = positions.copy()
            
            # 🚨 关键修复：确保goal_pos可以广播到positions和prev_positions的形状
            # goal_pos是(3,)，需要reshape为(1, 3)以便与positions (1, 3)相减
            if goal_pos.ndim == 1:
                goal_pos = goal_pos.reshape(1, -1) if positions.ndim == 2 else goal_pos
            
            prev_dist = np.linalg.norm(prev_positions - goal_pos, axis=-1)
            current_dist = np.linalg.norm(positions - goal_pos, axis=-1)
            rewards = prev_dist - current_dist
            
            # 🚨 关键修复：根据距离目标的距离动态调整接近奖励权重
            # 原设计：距离目标越近权重越低（0.3倍），导致智能体缺乏接近目标的动机
            # 修复：距离目标越近权重越高（2.0倍），鼓励智能体接近目标
            # 🔧 规划改进：NEAR_GOAL_THRESHOLD 从30米扩大到50米，让更多“朝目标走”的阶段获得增强接近奖励
            # - 距离目标0米时：权重倍数 = 2.0（大幅增强接近奖励，鼓励到达目标）
            # - 距离目标50米时：权重倍数 = 1.0（正常接近奖励）
            # - 距离目标>50米时：权重倍数 = 1.0（正常接近奖励）
            NEAR_GOAL_THRESHOLD = 50.0  # 距离目标50米内增强接近奖励（从30.0提高，利于规划到达）
            approach_weight_multipliers = np.ones(len(positions), dtype=np.float32)
            near_goal_mask = current_dist < NEAR_GOAL_THRESHOLD
            if np.any(near_goal_mask):
                # 线性插值：距离0米时倍数2.0（大幅增强），距离30米时倍数1.0（正常）
                approach_weight_multipliers[near_goal_mask] = 2.0 - (current_dist[near_goal_mask] / NEAR_GOAL_THRESHOLD) * 1.0
            approach_weight_multipliers = np.clip(approach_weight_multipliers, 1.0, 2.0)
            
            return rewards * 5.0 * approach_weight_multipliers
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
            height_goal_focus_mask = np.zeros(len(positions), dtype=bool)
            try:
                goal_pos = None
                if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                    goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32).reshape(-1)[:3]
                elif hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                    goal_pos = np.asarray(scenario.goal_pos, dtype=np.float32).reshape(-1)[:3]
                if goal_pos is not None and goal_pos.shape[0] >= 3:
                    goal_focus_radius = float(self.height_penalty_only_near_goal_radius)
                    success_thr = float(self.success_distance_threshold)
                    if goal_focus_radius > success_thr:
                        goal_dists = np.linalg.norm(positions - goal_pos[:3], axis=-1)
                        height_goal_focus_mask = (
                            (goal_dists > success_thr) &
                            (goal_dists < goal_focus_radius)
                        )
            except Exception:
                height_goal_focus_mask = np.zeros(len(positions), dtype=bool)
            
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
            rewards[in_range & (~height_goal_focus_mask)] = 1.0
            if np.any(in_range & height_goal_focus_mask):
                rewards[in_range & height_goal_focus_mask] = float(
                    self.height_near_goal_positive_factor
                )
            
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

                below_indices = np.where(below_range)[0]
                if len(below_indices) > 0 and np.any(height_goal_focus_mask[below_indices]):
                    upward_reward[height_goal_focus_mask[below_indices]] *= float(
                        self.height_near_goal_positive_factor
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

    def _agent_safe_so_far(self, agent: Any) -> bool:
        try:
            if getattr(agent, '_episode_has_collision', False):
                return False
        except Exception:
            pass
        try:
            if getattr(agent, '_had_obstacle_collision', False):
                return False
        except Exception:
            pass
        try:
            if getattr(agent, '_had_terrain_contact_or_penetration', False):
                return False
        except Exception:
            pass

        pen_count = 0
        if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
            pen_count = agent.debug_info.get('total_penetration_count', 0)
        try:
            pen_count = int(pen_count) if np.isfinite(pen_count) else 0
        except Exception:
            pen_count = 0
        return pen_count == 0

    def _success_reward_vectorized(self, agent: Any, scenario: Any, positions: np.ndarray, cached_data: Dict[str, Any] = None, agent_idx: int = None, world: Any = None) -> np.ndarray:
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
                # 🚀 关键修复：优先使用传入的world参数
                if world is None:
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
                    # 🚀 关键修复：优先使用传入的world参数
                    if world is None:
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
                'hover_reward_count': 0,
                'was_in_goal_zone': False,
                'goal_ring_rewards_given': [False] * int(getattr(self.goal_ring_radii, 'size', 0)),
            }

        success_state = agent._success_state
        success_state.setdefault('was_in_goal_zone', False)
        if 'goal_ring_rewards_given' not in success_state or not isinstance(success_state.get('goal_ring_rewards_given'), list):
            success_state['goal_ring_rewards_given'] = [False] * int(getattr(self.goal_ring_radii, 'size', 0))
        # 每回合重置：检测到步数“回绕”（cur_step < 上次看到的步）或首次观测步数即视为新回合
        # 🚨 关键修复：将cur_step和should_update_last_seen_step声明为函数级变量，确保在函数结束前可用
        cur_step = None
        should_update_last_seen_step = False
        try:
            # 🚀 关键修复：优先使用传入的world参数
            if world is None:
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
                    success_state['goal_ring_rewards_given'] = [False] * int(getattr(self.goal_ring_radii, 'size', 0))
                    success_state['no_collision_reward_given'] = False  # 🔧 新增：重置无碰撞奖励标志（agent级别，已废弃但保留兼容性）
                    
                    # 🚨 关键修复：重置world级别的无碰撞奖励标志（确保新回合开始时重置）
                    try:
                        # 🚀 关键修复：优先使用传入的world参数
                        if world is None:
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
                        agent._episode_has_collision = False
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
                        # 🚀 关键修复：优先使用传入的world参数
                        if world is None:
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

                ring_bonus = self._goal_ring_bonus_vectorized(agent, distances, success_state)
                if np.any(ring_bonus > 0.0):
                    rewards = np.maximum(rewards, ring_bonus.astype(np.float32))
                
                if np.any(success_mask):
                    # 🚨 关键修复：奖励值必须总是被计算和设置，即使success_reward_given已经是True
                    # 原因：一次性奖励机制只控制打印和悬停奖励，不应该阻止基础成功奖励的发放
                    # 修复：将奖励值计算移出if reward_should_be_given块，确保总是执行
                    
                    # 一次性成功奖励（防重复）- 只用于控制打印和悬停奖励
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
                    
                    # 🚨 关键修复：计算奖励值（总是执行，不依赖于reward_should_be_given）
                    # 奖励值计算和设置应该在检测到到达时立即执行，而不是只在第一次到达时执行
                    # 原因：即使success_reward_given已经是True，奖励值仍然应该被正确设置
                    # 一次性奖励机制只控制打印和悬停奖励，不应该阻止基础成功奖励的发放
                    
                    # 计算无碰撞比例（解耦版）：个体成功奖励只看当前agent自己的碰撞
                    # 注意：total_collision_count 仍用于团队无碰撞奖励判定，不影响此处的个体奖励缩放
                    no_collision_ratio = 1.0
                    no_collision_reward = 0.0
                    total_collision_count = 0
                    current_agent_collision_count = 0
                    episode_length = 2800
                    try:
                        if world is None:
                            world = getattr(scenario, 'world', None)
                        if world is not None:
                            # 获取回合总步数
                            if hasattr(world, 'episode_length') and world.episode_length is not None:
                                episode_length = int(world.episode_length)
                            elif hasattr(world, 'max_steps') and world.max_steps is not None:
                                episode_length = int(world.max_steps)
                            else:
                                episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                                try:
                                    episode_length = int(episode_length_str)
                                except (ValueError, TypeError):
                                    episode_length = 2800

                            # 统计全队碰撞（用于团队奖励）并提取当前agent碰撞（用于个体成功奖励缩放）
                            if hasattr(world, 'agents') and world.agents is not None:
                                matched_agent = False
                                for ag in world.agents:
                                    penetration_count = 0
                                    if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                        penetration_count = ag.debug_info.get('total_penetration_count', 0)
                                        try:
                                            penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                        except (ValueError, TypeError, OverflowError):
                                            penetration_count = 0
                                    total_collision_count += penetration_count
                                    if ag is agent:
                                        current_agent_collision_count = penetration_count
                                        matched_agent = True

                                # 在极端情况下agent对象不在world.agents中，回退到当前agent自身统计
                                if not matched_agent and hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                    _cnt = agent.debug_info.get('total_penetration_count', 0)
                                    try:
                                        current_agent_collision_count = int(_cnt) if np.isfinite(_cnt) else 0
                                    except (ValueError, TypeError, OverflowError):
                                        current_agent_collision_count = 0
                            elif hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                _cnt = agent.debug_info.get('total_penetration_count', 0)
                                try:
                                    current_agent_collision_count = int(_cnt) if np.isfinite(_cnt) else 0
                                except (ValueError, TypeError, OverflowError):
                                    current_agent_collision_count = 0

                            # 个体成功奖励缩放只由当前agent碰撞比例决定
                            if episode_length > 0:
                                collision_ratio = float(current_agent_collision_count) / float(episode_length)
                                no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                if collision_ratio < 0.1:
                                    no_collision_ratio = no_collision_ratio ** 12
                                elif collision_ratio >= 0.5:
                                    no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                    no_collision_ratio = max(0.0, no_collision_ratio)
                            else:
                                no_collision_ratio = 1.0 if current_agent_collision_count == 0 else 0.0
                    except Exception as e:
                        if os.getenv('ENABLE_REWARD_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on'):
                            print(f"[警告] 获取碰撞计数失败: {e}, 使用默认值current_agent_collision_count=0")
                        pass
                    
                    # 个体成功奖励严格按回合只发一次。
                    reward_value = self.success_reward_value * no_collision_ratio
                    if not self._agent_safe_so_far(agent):
                        reward_value = 0.0
                    reward_value = min(reward_value, self.success_reward_value)
                    success_reward_scaled = np.full(len(positions), reward_value, dtype=np.float32)
                    if reward_should_be_given:
                        rewards[success_mask] = success_reward_scaled[success_mask]
                    
                    # 🚨 关键修复：打印语句应该在第一次到达时总是执行
                    # 原因：用户需要看到奖励信息，即使success_reward_given已经是True
                    # 修复：使用reward_should_be_given标志，它在第一次到达时为True
                    # 注意：reward_should_be_given在success_reward_given被设置为True时也会被设置为True
                    # 所以如果success_reward_given为False，reward_should_be_given也会是True（第一次到达）
                    # 如果success_reward_given为True，reward_should_be_given会是False（已经到达过）
                    # 因此，使用reward_should_be_given可以确保只在第一次到达时打印
                    if reward_should_be_given:
                        # 🚨 关键修复：重新获取world用于打印（确保使用正确的world对象）
                        # 🚀 关键修复：优先使用传入的world参数，如果不存在再从agent.world或scenario.world获取
                        world_for_print = world if world is not None else None
                        if world_for_print is None:
                            world_for_print = getattr(agent, 'world', None)
                        if world_for_print is None:
                            world_for_print = getattr(scenario, 'world', None)
                        
                        # 🔧 获取环境ID用于打印
                        env_id = None
                        try:
                            if world_for_print is not None:
                                env_id = getattr(world_for_print, 'env_id', None)
                            if env_id is None:
                                env_id = getattr(scenario, 'env_id', None)
                            if env_id is None and hasattr(agent, 'env_id'):
                                env_id = getattr(agent, 'env_id', None)
                        except Exception:
                            pass
                        
                        # 获取智能体ID用于打印
                        actual_agent_id = agent_id if agent_id is not None and agent_id >= 0 else -1
                        if actual_agent_id < 0:
                            try:
                                if world_for_print is not None and hasattr(world_for_print, 'agents'):
                                    for idx, ag in enumerate(world_for_print.agents):
                                        if ag is agent:
                                            actual_agent_id = idx
                                            break
                            except Exception:
                                pass
                        if actual_agent_id < 0:
                            if agent_idx is not None and agent_idx >= 0:
                                actual_agent_id = agent_idx
                            else:
                                actual_agent_id = 0
                        
                        min_dist = np.min(distances[success_mask]) if np.any(success_mask) else 999.0
                        env_display = f"Env{env_id}" if env_id is not None else "Env?"
                        
                        # 🚨 关键修复：重新计算total_collision_count用于打印（确保使用正确的world对象）
                        # 🚀 关键修复：优先使用传入的world参数，确保获取正确的碰撞计数
                        # 使用与奖励计算相同的逻辑，但使用world_for_print确保一致性
                        total_collision_count_for_print = total_collision_count  # 默认使用之前计算的值
                        episode_length_for_print = episode_length  # 默认使用之前计算的值
                        try:
                            # 🚀 关键修复：优先使用传入的world参数，如果不存在再使用world_for_print
                            world_for_collision_count = world if world is not None else world_for_print
                            if world_for_collision_count is not None:
                                # 重新获取episode_length
                                if hasattr(world_for_collision_count, 'episode_length') and world_for_collision_count.episode_length is not None:
                                    episode_length_for_print = int(world_for_collision_count.episode_length)
                                elif hasattr(world_for_collision_count, 'max_steps') and world_for_collision_count.max_steps is not None:
                                    episode_length_for_print = int(world_for_collision_count.max_steps)
                                
                                # 🚀 关键修复：重新统计所有智能体的总碰撞次数（使用正确的world对象）
                                if hasattr(world_for_collision_count, 'agents') and world_for_collision_count.agents is not None:
                                    total_collision_count_for_print = 0
                                    for ag in world_for_collision_count.agents:
                                        penetration_count = 0
                                        if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                            penetration_count = ag.debug_info.get('total_penetration_count', 0)
                                            try:
                                                penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                            except (ValueError, TypeError, OverflowError):
                                                penetration_count = 0
                                        total_collision_count_for_print += penetration_count
                        except Exception:
                            pass  # 如果重新计算失败，使用之前的值
                        
                        # 🔧 计算所有智能体的到达状态（用于打印）
                        all_reached_count = 0
                        total_agents = 0
                        try:
                            if world_for_print is not None and hasattr(world_for_print, 'agents') and world_for_print.agents is not None:
                                total_agents = len(world_for_print.agents)
                                for ag in world_for_print.agents:
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
                        except Exception:
                            pass
                        
                        # 获取当前智能体自己的碰撞次数（用于打印）
                        # 奖励缩放已改为个体碰撞统计，打印继续展示当前agent碰撞与全队碰撞（便于诊断）
                        current_agent_collision_count = 0
                        try:
                            if agent is not None:
                                if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                    current_agent_collision_count = agent.debug_info.get('total_penetration_count', 0)
                                    try:
                                        current_agent_collision_count = int(current_agent_collision_count) if np.isfinite(current_agent_collision_count) else 0
                                    except (ValueError, TypeError, OverflowError):
                                        current_agent_collision_count = 0
                        except Exception:
                            pass
                        
                        # 计算当前智能体的无碰撞比例（与当前个体成功奖励缩放一致）
                        current_agent_no_collision_ratio = 1.0
                        if episode_length_for_print > 0:
                            current_agent_collision_ratio = float(current_agent_collision_count) / float(episode_length_for_print)
                            current_agent_no_collision_ratio = max(0.0, 1.0 - current_agent_collision_ratio)
                        
                        # 判断是部分到达还是全部到达
                        debug_reward_events = os.getenv('DEBUG_REWARD_EVENTS', '0').lower() in ('1','true','yes','on')
                        if debug_reward_events:
                            if total_agents > 0 and 0 < all_reached_count < total_agents:
                                # 部分到达：显示部分到达奖励
                                # 🚨 关键修复：显示当前智能体自己的碰撞次数，而不是全局的
                                print(f"[部分到达奖励] {env_display} Agent{actual_agent_id}: {all_reached_count}/{total_agents}个智能体到达目标, 距离={min_dist:.2f}m, 碰撞次数={current_agent_collision_count}/{episode_length_for_print}, 无碰撞比例={current_agent_no_collision_ratio:.3f}, 奖励={success_reward_scaled[success_mask][0] if np.any(success_mask) else 0.0:.1f} (基础值={self.success_reward_value:.1f}, 一次性, 全局碰撞={total_collision_count_for_print})")
                            else:
                                # 全部到达或单个智能体到达：显示标准成功奖励
                                # 🚨 关键修复：显示当前智能体自己的碰撞次数，而不是全局的
                                print(f"[VecSuccessReward] {env_display} Agent{actual_agent_id}: reached goal at {min_dist:.2f}m, collisions={current_agent_collision_count}/{episode_length_for_print}, no_collision_ratio={current_agent_no_collision_ratio:.3f}, reward={success_reward_scaled[success_mask][0] if np.any(success_mask) else 0.0:.1f} (scaled from {self.success_reward_value:.1f}, one-time, global_collisions={total_collision_count_for_print})")
                    
                    # 🚨 关键修复：只有在标志刚被设置时才给予额外奖励（如无碰撞奖励）
                    if reward_should_be_given:
                        
                        # 🚨 关键修复：这里不应该重新计算no_collision_ratio，应该使用之前计算的值
                        # 原因：之前已经计算过了（1395-1451行），重新计算会导致不一致和覆盖
                        # 修复：直接使用之前计算的no_collision_ratio和total_collision_count
                        # 只重新获取world对象用于无碰撞奖励检查（如果需要）
                        no_collision_reward = 0.0
                        try:
                            # 🚨 关键修复：优先从agent.world获取world，向量化环境中scenario.world可能不存在
                            world = getattr(agent, 'world', None)
                            if world is None:
                                world = getattr(scenario, 'world', None)
                            
                            # 🚨 关键修复：使用严格的碰撞检查逻辑，确保判断准确
                            # 检查所有智能体是否都没有碰撞（使用与显示逻辑一致的检查）
                            all_no_collision_strict = True
                            if world is not None:
                                try:
                                    if hasattr(world, 'agents') and world.agents is not None:
                                        for ag in world.agents:
                                            if not self._agent_safe_so_far(ag):
                                                all_no_collision_strict = False
                                                break
                                except Exception:
                                    all_no_collision_strict = False
                            else:
                                # 如果没有world，回退到只检查当前智能体
                                all_no_collision_strict = self._agent_safe_so_far(agent)
                            
                            # 如果所有智能体都没有碰撞，给予无碰撞奖励
                            # 🚨 关键修复：使用之前计算的total_collision_count，而不是重新计算
                            if all_no_collision_strict and total_collision_count == 0 and self.no_collision_reward_value > 0.0:
                                # 初始化无碰撞奖励状态
                                if 'no_collision_reward_given' not in success_state:
                                    success_state['no_collision_reward_given'] = False
                                
                                # 只在第一次成功到达时给予无碰撞奖励
                                if not success_state.get('no_collision_reward_given', False):
                                    success_state['no_collision_reward_given'] = True
                                    no_collision_reward = self.no_collision_reward_value
                        except Exception:
                            pass
                        
                        # 🚨 关键修改：成功奖励 = 成功奖励 × 无碰撞比例
                        # 如果所有智能体都没有碰撞，无碰撞比例 = 1.0，成功奖励 = 原值
                        # 如果部分智能体有碰撞，无碰撞比例 < 1.0，成功奖励会按比例减少
                        # 如果所有智能体都有碰撞，无碰撞比例 = 0.0，成功奖励 = 0
                        # 🚀 关键修复：移除precision_bonus，确保奖励不超过基础值（按照碰撞比例来给奖励值）
                        # 原因：用户要求奖励应该按照碰撞比例来给，不应该超过基础值2000
                        # 🚀 关键修复：确保success_reward_scaled是数组而不是标量，以便使用数组索引
                        reward_value = self.success_reward_value * no_collision_ratio
                        if not self._agent_safe_so_far(agent):
                            reward_value = 0.0
                        # 确保奖励值不超过基础值（防止数值误差导致超过）
                        reward_value = min(reward_value, self.success_reward_value)
                        # 创建与positions长度相同的数组
                        success_reward_scaled = np.full(len(positions), reward_value, dtype=np.float32)
                        # 🚨 关键修复：奖励值必须总是被设置，即使reward_should_be_given为False
                        # 原因：如果success_reward_given已经是True（例如由于某种原因），奖励值仍然应该被设置
                        # 修复：将奖励值设置移出if reward_should_be_given块，确保总是执行
                        rewards[success_mask] = success_reward_scaled[success_mask]
                        
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
                        
                        # 🚨 关键修复：无碰撞奖励应该只在"所有智能体都到达目标且所有智能体都无碰撞"时才给予
                        # 原因：用户明确指出"有专门的部分到达奖励，无碰撞奖励就应该无碰撞且到达才给"
                        # 修复：不仅检查all_agents_reached，还要再次验证all_no_collision_strict
                        # 注意：all_no_collision_strict在第1719-1753行已经计算过，但为了确保逻辑清晰，我们再次检查
                        if all_agents_reached and no_collision_reward > 0.0:
                            # 🚨 关键修复：再次验证所有智能体都无碰撞（确保逻辑正确）
                            # 即使no_collision_reward > 0.0，也要再次检查无碰撞条件，防止逻辑错误
                            all_no_collision_verify = True
                            try:
                                world_for_verify = getattr(scenario, 'world', None)
                                if world_for_verify is None:
                                    world_for_verify = getattr(agent, 'world', None)
                                
                                if world_for_verify is not None and hasattr(world_for_verify, 'agents') and world_for_verify.agents is not None:
                                    for ag in world_for_verify.agents:
                                        if not self._agent_safe_so_far(ag):
                                            all_no_collision_verify = False
                                            break
                            except Exception:
                                all_no_collision_verify = False
                            
                            # 🚨 关键修复：只有在"所有智能体都到达目标"且"所有智能体都无碰撞"时，才设置无碰撞奖励
                            if all_no_collision_verify:
                                # 将无碰撞奖励值存储到world中，供后续统一分配
                                if not hasattr(world, '_team_no_collision_reward'):
                                    world._team_no_collision_reward = 0.0
                                world._team_no_collision_reward = no_collision_reward
                            else:
                                # 🚨 关键修复：如果有任何智能体有碰撞，即使所有智能体都到达目标，也不给予无碰撞奖励
                                # 原因：无碰撞奖励应该只在"无碰撞且到达"时才给，部分到达奖励已经通过success_reward处理
                                if hasattr(world, '_team_no_collision_reward'):
                                    world._team_no_collision_reward = 0.0
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
                            # 🔧 注意：打印逻辑已经移到了if reward_should_be_given块之前（第1509行），这里不再重复打印
                            # 只在if reward_should_be_given块内处理无碰撞奖励的打印
                            if np.any(success_mask) and reward_should_be_given:
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

            ring_bonus = self._goal_ring_bonus_vectorized(agent, distances, success_state)
            if np.any(ring_bonus > 0.0):
                rewards = np.maximum(rewards, ring_bonus.astype(np.float32))
            
            if np.any(success_mask):
                # 一次性成功奖励（防重复）
                # 🚨 关键修复：使用get方法，避免KeyError，并防止并发调用导致重复触发
                if not success_state.get('success_reward_given', False):
                    # 🔧 修复：立即设置标志，防止在后续处理过程中再次触发
                    success_state['success_reward_given'] = True
                    success_state['first_success_step'] = getattr(scenario, 'current_step', 0)
                    success_state['hover_reward_count'] = 0
                    
                    # 计算无碰撞比例（解耦版）：个体成功奖励只看当前agent自己的碰撞
                    no_collision_ratio = 1.0
                    no_collision_reward = 0.0
                    total_collision_count = 0
                    current_agent_collision_count = 0
                    episode_length = 2800
                    all_agents_reached_fallback = False
                    try:
                        if world is None:
                            world = getattr(scenario, 'world', None)
                        if world is not None:
                            # 获取回合总步数
                            if hasattr(world, 'episode_length') and world.episode_length is not None:
                                episode_length = int(world.episode_length)
                            elif hasattr(world, 'max_steps') and world.max_steps is not None:
                                episode_length = int(world.max_steps)
                            else:
                                episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                                try:
                                    episode_length = int(episode_length_str)
                                except (ValueError, TypeError):
                                    episode_length = 2800

                            # 全队碰撞用于团队奖励；个体碰撞用于当前agent成功奖励缩放
                            if hasattr(world, 'agents') and world.agents is not None:
                                matched_agent = False
                                for ag in world.agents:
                                    penetration_count = 0
                                    if hasattr(ag, 'debug_info') and isinstance(ag.debug_info, dict):
                                        penetration_count = ag.debug_info.get('total_penetration_count', 0)
                                        try:
                                            penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                                        except (ValueError, TypeError, OverflowError):
                                            penetration_count = 0
                                    total_collision_count += penetration_count
                                    if ag is agent:
                                        current_agent_collision_count = penetration_count
                                        matched_agent = True

                                if not matched_agent and hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                    _cnt = agent.debug_info.get('total_penetration_count', 0)
                                    try:
                                        current_agent_collision_count = int(_cnt) if np.isfinite(_cnt) else 0
                                    except (ValueError, TypeError, OverflowError):
                                        current_agent_collision_count = 0
                            elif hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                                _cnt = agent.debug_info.get('total_penetration_count', 0)
                                try:
                                    current_agent_collision_count = int(_cnt) if np.isfinite(_cnt) else 0
                                except (ValueError, TypeError, OverflowError):
                                    current_agent_collision_count = 0

                            if episode_length > 0:
                                collision_ratio = float(current_agent_collision_count) / float(episode_length)
                                no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                if collision_ratio < 0.1:
                                    no_collision_ratio = no_collision_ratio ** 12
                                elif collision_ratio >= 0.5:
                                    no_collision_ratio = 0.2 * (1.0 - (collision_ratio - 0.5) / 0.5)
                                    no_collision_ratio = max(0.0, no_collision_ratio)
                            else:
                                no_collision_ratio = 1.0 if current_agent_collision_count == 0 else 0.0
                                
                                # 🚨 关键修复：无碰撞奖励应该只在所有智能体都到达目标且无碰撞时才给一次
                                # 检查所有智能体是否都到达目标
                                all_agents_reached = False
                                all_agents_safe = False
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
                                    all_agents_safe = all(self._agent_safe_so_far(ag) for ag in world.agents)
                                
                                # 如果所有智能体都到达目标且无碰撞，给予无碰撞奖励（只给一次）
                                if all_agents_reached and all_agents_safe and total_collision_count == 0 and self.no_collision_reward_value > 0.0:
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
                            episode_length_str = os.getenv('EPISODE_LENGTH', '2800')
                            try:
                                episode_length = int(episode_length_str)
                            except (ValueError, TypeError):
                                episode_length = 2800
                            
                            # 计算无碰撞比例
                            if episode_length > 0:
                                collision_ratio = float(total_collision_count) / float(episode_length)
                                no_collision_ratio = max(0.0, 1.0 - collision_ratio)
                                
                                # 🚨 非线性映射：大幅压低“有碰撞”时的成功奖励，激励完全避免碰撞
                                if collision_ratio < 0.1:
                                    no_collision_ratio = no_collision_ratio ** 12  # 强惩罚少量碰撞，推动零碰撞
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
                                        if not self._agent_safe_so_far(ag):
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
                    # 🚀 关键修复：确保success_reward_scaled是数组而不是标量，以便使用数组索引
                    reward_value = self.success_reward_value * no_collision_ratio
                    if not self._agent_safe_so_far(agent):
                        reward_value = 0.0
                    # 确保奖励值不超过基础值（防止数值误差导致超过）
                    reward_value = min(reward_value, self.success_reward_value)
                    # 创建与positions长度相同的数组
                    success_reward_scaled = np.full(len(positions), reward_value, dtype=np.float32)
                    rewards[success_mask] = success_reward_scaled[success_mask]
                    
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
                        # 🚀 关键修复：success_reward_scaled现在是数组，需要提取标量值用于打印
                        reward_for_print = float(success_reward_scaled[success_mask][0]) if np.any(success_mask) else float(self.success_reward_value * no_collision_ratio)
                        if os.getenv('DEBUG_REWARD_EVENTS', '0').lower() in ('1','true','yes','on'):
                            print(f"[VecSuccessReward] {env_display} Agent{actual_agent_id}: reached goal at {min_dist:.2f}m (fallback), collisions={total_collision_count}/{episode_length}, no_collision_ratio={no_collision_ratio:.3f}, reward={reward_for_print:.1f} (scaled from {self.success_reward_value:.1f})")
                        
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
                                if os.getenv('DEBUG_REWARD_EVENTS', '0').lower() in ('1','true','yes','on'):
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

        current_in_goal_zone = False
        try:
            if 'success_mask' in locals():
                current_in_goal_zone = bool(np.any(success_mask))
        except Exception:
            current_in_goal_zone = False

        if not current_in_goal_zone:
            rewards[:] = 0.0
            if success_state.get('was_in_goal_zone', False):
                rewards[:] = -float(self.leave_goal_penalty)
                success_state['hover_reward_count'] = 0
        elif not self._agent_safe_so_far(agent):
            rewards = np.minimum(rewards, 0.0).astype(np.float32, copy=False)

        success_state['was_in_goal_zone'] = current_in_goal_zone
        
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

    def _terrain_min_distance_3d(self, positions: np.ndarray, scenario: Any, cached_data: Dict[str, Any] = None, world: Any = None) -> np.ndarray:
        """
        计算位置到地形的 3D 最小距离，与净空奖励 _clearance_reward_vectorized 中地形距离定义一致。
        用于碰撞惩罚与净空奖励共用同一套 d_min 定义，避免信号矛盾。
        返回形状 (len(positions),) 的 float32 数组，无有效地形时为 np.inf。
        """
        cached_data = cached_data if isinstance(cached_data, dict) else (self._cache.get(id(world), {}) if world is not None else {})
        _, terrain_min_dist, _ = self._compute_terrain_distance_data(
            positions,
            scenario,
            cached_data=cached_data,
        )
        return terrain_min_dist

    def _collision_penalty_vectorized(self, agent_idx: int, world: Any, scenario: Any, positions: np.ndarray, cached_data: Dict[str, Any] = None) -> np.ndarray:
        """
        优化版碰撞惩罚计算：使用缓存数据，减少重复计算
        
        🚨 关键修复：使用综合最小距离（d_min_current）检测碰撞，而不是只检查Z坐标穿透
        原因：图表显示 min_distance_to_obstacle 多次为0，但碰撞计数为0，说明Z坐标检测无法捕获侧面碰撞
        修复：计算障碍物和地形的综合最小距离，如果 < collision_distance_threshold 或 < 0，触发碰撞
        地形距离与净空奖励一致：使用 3D 最近距离（_terrain_min_distance_3d），避免碰撞与净空信号矛盾。
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
            
            # 地形距离：与净空路径一致，使用 3D 最近距离（垂向 + 多方向采样取最小）
            terrain_min_dist = self._terrain_min_distance_3d(positions, scenario, cached_data=cached_data, world=world)
            
            # 综合最小距离：取障碍物和地形距离的最小值
            d_min_current = np.minimum(obstacle_min_dist, terrain_min_dist)
        except Exception:
            d_min_current = np.full(len(positions), np.inf, dtype=np.float32)
        
        # 🚨 关键修复：使用综合最小距离检测碰撞
        # 如果 d_min_current < collision_distance_threshold 或 < 0，视为碰撞
        # 这是主要的碰撞检测方法，比Z坐标检测更准确（能捕获侧面碰撞）
        collision_threshold = float(self.collision_distance_threshold)
        distance_based_collision_mask = (d_min_current < collision_threshold) | (d_min_current < 0.0)
        real_distance_collision_mask = (d_min_current < 0.0)
        
        # 🚨 关键修复：如果检测到距离碰撞，立即应用惩罚和更新计数
        if np.any(distance_based_collision_mask):
            # 🚀 新增：根据距离目标的距离动态调整碰撞惩罚强度
            # 距离目标越近，碰撞惩罚越强，鼓励智能体在接近目标时更加谨慎
            goal_dist = None
            try:
                # 尝试从agent获取目标位置
                if agent_idx is not None and hasattr(world, 'agents') and 0 <= agent_idx < len(world.agents):
                    agent = world.agents[agent_idx]
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                        goal_pos = np.asarray(agent.goal_a.state.p_pos[:3], dtype=np.float32)
                        if goal_pos.ndim == 1:
                            goal_pos = goal_pos.reshape(1, -1)
                        goal_dist = np.linalg.norm(positions - goal_pos, axis=-1)
                # 如果无法从agent获取，尝试从scenario获取
                if goal_dist is None and scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                    goal_pos = np.asarray(scenario.goal_pos[:3], dtype=np.float32)
                    if goal_pos.ndim == 1:
                        goal_pos = goal_pos.reshape(1, -1)
                    goal_dist = np.linalg.norm(positions - goal_pos, axis=-1)
            except Exception:
                goal_dist = None
            
            # 🚨 关键修复：碰撞惩罚不再按距离目标减弱，鼓励完全避免碰撞
            # 原修复（近目标减弱惩罚）会导致“始终出现小碰撞”：近目标碰撞成本低，智能体倾向接受少量碰撞换到达
            # 现改为全程使用全额惩罚（1.0倍），使任意一步碰撞都受到完整惩罚，从而激励零碰撞
            penalty_multipliers = np.ones(len(positions), dtype=np.float32)
            
            collision_penalties = self._piecewise_collision_penalty(
                d_min_current[distance_based_collision_mask],
                collision_threshold,
                self.collision_penalty_value
            )
            dynamic_penalties = collision_penalties * penalty_multipliers[distance_based_collision_mask]
            penalties[distance_based_collision_mask] = np.minimum(
                penalties[distance_based_collision_mask],
                dynamic_penalties
            )
            
            # 🚨 关键修复：更新碰撞计数
            try:
                has_world_agents = hasattr(world, 'agents')
                world_agents_len = len(world.agents) if has_world_agents else 0
                agent_idx_valid = (has_world_agents and 0 <= agent_idx < world_agents_len)
                
                if agent_idx_valid and np.any(real_distance_collision_mask):
                    ag = world.agents[agent_idx]
                    ag._episode_has_collision = True
                    ag._had_penetration_or_collision = True
                    if not hasattr(ag, 'debug_info'):
                        ag.debug_info = {}
                    if not isinstance(ag.debug_info, dict):
                        ag.debug_info = {}
                    
                    # 统计本批次中的碰撞次数
                    collision_count = np.sum(real_distance_collision_mask)
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
                    # 分项计数：距离检测为地形与障碍综合，按最小距离来源区分
                    ag.debug_info.setdefault('terrain_penetration_count', 0)
                    ag.debug_info.setdefault('obstacle_collision_count', 0)
                    try:
                        terrain_add = int(np.sum(real_distance_collision_mask & (terrain_min_dist <= obstacle_min_dist)))
                        obstacle_add = int(np.sum(real_distance_collision_mask & (obstacle_min_dist < terrain_min_dist)))
                        if terrain_add > 0:
                            ag._had_terrain_contact_or_penetration = True
                            ag.debug_info['terrain_penetration_count'] = ag.debug_info.get('terrain_penetration_count', 0) + terrain_add
                        if obstacle_add > 0:
                            ag._had_obstacle_collision = True
                            ag.debug_info['obstacle_collision_count'] = ag.debug_info.get('obstacle_collision_count', 0) + obstacle_add
                    except Exception:
                        pass
                    
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
                            ag._episode_has_collision = True
                            ag._had_penetration_or_collision = True
                            ag._had_terrain_contact_or_penetration = True
                            if not hasattr(ag, 'debug_info') or not isinstance(ag.debug_info, dict):
                                ag.debug_info = {}
                            add_invalid = int(np.sum(invalid_mask))
                            ag.debug_info['total_penetration_count'] = ag.debug_info.get('total_penetration_count', 0) + add_invalid
                            ag.debug_info.setdefault('terrain_penetration_count', 0)
                            ag.debug_info.setdefault('obstacle_collision_count', 0)
                            if add_invalid > 0:
                                ag.debug_info['terrain_penetration_count'] = ag.debug_info.get('terrain_penetration_count', 0) + add_invalid
                    except Exception:
                        pass
                
                # 地形穿透惩罚（Z坐标检测，作为距离检测的补充）
                # 🚨 关键修复：Z坐标检测只作为补充，主要依赖距离检测（已在前面完成）
                # 原因：距离检测能捕获侧面碰撞，Z坐标检测只能捕获垂直穿透
                # 修复：Z坐标检测只检测距离检测未捕获的情况，避免重复计数
                # 🚨 关键修复2：使用真实碰撞阈值（0.3米），而不是净空监测阈值（1.5米）
                # 原问题：使用1.5米阈值导致智能体在0.5-1.0米高度飞行也被误报为"穿透"
                eps = float(self.terrain_collision_eps)  # 使用真实碰撞阈值（0.3米）
                # Z坐标穿透检测：智能体Z坐标 < 地形高度 + eps（只检测真实接触）
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
                    terrain_signed_clearance = positions[penetration_mask, 2] - terrain_heights[penetration_mask]
                    terrain_penalties = self._piecewise_collision_penalty(
                        terrain_signed_clearance,
                        float(self.terrain_collision_eps),
                        self.terrain_penalty_value
                    )
                    penalties[penetration_mask] = np.minimum(
                        penalties[penetration_mask],
                        terrain_penalties
                    )
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
                            ag._episode_has_collision = True
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
                            ag.debug_info.setdefault('terrain_penetration_count', 0)
                            ag.debug_info.setdefault('obstacle_collision_count', 0)
                            if penetration_count_int > 0:
                                ag.debug_info['terrain_penetration_count'] = ag.debug_info.get('terrain_penetration_count', 0) + penetration_count_int
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
                    contact_signed_clearance = positions[contact_penalty_mask, 2] - terrain_heights[contact_penalty_mask]
                    contact_penalties[contact_penalty_mask] = self._piecewise_collision_penalty(
                        contact_signed_clearance,
                        float(self.terrain_contact_eps),
                        self.terrain_penalty_value
                    )
                    # 合并穿透/接触两种惩罚（取更负者）
                    penalties = np.minimum(penalties, contact_penalties)
                    # ⚖️ 语义收紧：仅当满足“真实碰撞阈值”（terrain_collision_eps）时，
                    # 才拉起 `_had_penetration_or_collision` / `_had_terrain_contact_or_penetration` 并累计计数。
                    # 处于 contact_eps 带内但未触及 collision_eps 的低空飞行，仅施加数值惩罚，不再作为离散碰撞事件统计。
                    try:
                        if hasattr(world, 'agents') and 0 <= agent_idx < len(world.agents):
                            ag = world.agents[agent_idx]
                            if not hasattr(ag, 'debug_info'):
                                ag.debug_info = {}
                            if not isinstance(ag.debug_info, dict):
                                ag.debug_info = {}
                            # 只有 z <= terrain_height + collision_eps 才视为“真实碰撞事件”
                            collision_eps = float(self.terrain_collision_eps)  # 默认0.3米，真实碰撞阈值
                            actual_penetration_mask = contact_penalty_mask & (positions[:, 2] <= terrain_heights + collision_eps)
                            if np.any(actual_penetration_mask):
                                ag._had_penetration_or_collision = True
                                ag._had_terrain_contact_or_penetration = True
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
                                ag.debug_info.setdefault('terrain_penetration_count', 0)
                                ag.debug_info.setdefault('obstacle_collision_count', 0)
                                if penetration_count_int > 0:
                                    ag.debug_info['terrain_penetration_count'] = ag.debug_info.get('terrain_penetration_count', 0) + penetration_count_int
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
                    obstacle_signed_distance = -max_penetrations[obs_collision_mask]
                    obstacle_penalties = self._piecewise_collision_penalty(
                        obstacle_signed_distance,
                        collision_threshold,
                        self.collision_penalty_value
                    )
                    penalties[obs_collision_mask] = np.minimum(
                        penalties[obs_collision_mask],
                        obstacle_penalties
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
                            ag.debug_info.setdefault('terrain_penetration_count', 0)
                            ag.debug_info.setdefault('obstacle_collision_count', 0)
                            if penetration_count_int > 0:
                                ag.debug_info['obstacle_collision_count'] = ag.debug_info.get('obstacle_collision_count', 0) + penetration_count_int
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
            if np.any(collision_mask):
                terrain_signed_clearance = positions[collision_mask, 2] - terrain_heights[collision_mask]
                terrain_penalties = self._piecewise_collision_penalty(
                    terrain_signed_clearance,
                    float(self.terrain_collision_eps),
                    self.terrain_penalty_value
                )
                penalties[collision_mask] = np.minimum(
                    penalties[collision_mask],
                    terrain_penalties
                )
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
                        add_cnt = int(penetration_count)
                        ag.debug_info['total_penetration_count'] = ag.debug_info.get('total_penetration_count', 0) + add_cnt
                        ag.debug_info.setdefault('terrain_penetration_count', 0)
                        ag.debug_info.setdefault('obstacle_collision_count', 0)
                        if add_cnt > 0:
                            ag.debug_info['terrain_penetration_count'] = ag.debug_info.get('terrain_penetration_count', 0) + add_cnt
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
                            
                            # 🚨 非线性映射：与成功奖励一致，强惩罚少量碰撞以推动零碰撞
                            if collision_ratio < 0.1:
                                no_collision_ratio = no_collision_ratio ** 12
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
                import os as _os
                scenario = None
                if world is not None:
                    scenario = getattr(world, 'scenario', None)
                if scenario is None and agents and hasattr(agents[0], 'scenario'):
                    scenario = agents[0].scenario

                if world is None:
                    return 0.0

                episode_length = None
                try:
                    if hasattr(world, 'episode_length') and world.episode_length is not None:
                        episode_length = int(world.episode_length)
                    elif hasattr(world, 'max_steps') and world.max_steps is not None:
                        episode_length = int(world.max_steps)
                    else:
                        episode_length = int(_os.getenv('EPISODE_LENGTH', '2800'))
                except Exception:
                    episode_length = None

                try:
                    cur_step = int(getattr(world, 'current_step', -1))
                except Exception:
                    cur_step = -1

                # 全局团队奖励仅在“回合结束”时结算：
                # 1. 正常跑到 episode_length；
                # 2. 环境已判定本回合团队成功并提前终止。
                episode_finished = bool(getattr(world, '_episode_success', False))
                if not episode_finished and not (
                    episode_length is not None and episode_length > 0 and cur_step >= episode_length
                ):
                    return 0.0

                team_success_flags = []
                thr_success = float(getattr(self, 'success_distance_threshold', 2.0))
                for ag in agents:
                    ag_goal = None
                    try:
                        if hasattr(ag, 'goal_a') and hasattr(ag.goal_a, 'state') and getattr(ag.goal_a.state, 'p_pos', None) is not None:
                            ag_goal = ag.goal_a.state.p_pos
                    except Exception:
                        ag_goal = None
                    if ag_goal is None and scenario is not None:
                        ag_goal = getattr(scenario, 'goal_pos', None)

                    pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                    if pos is None or ag_goal is None or len(pos) < 3:
                        team_success_flags.append(False)
                        continue

                    dist_goal_3d = float(np.linalg.norm(np.asarray(pos[:3]) - np.asarray(ag_goal[:3])))
                    # 统一阈值：不再乘 1.2
                    reach_i = dist_goal_3d <= thr_success
                    safe_i = self._agent_safe_so_far(ag)
                    team_success_flags.append(bool(reach_i and safe_i))

                all_success = len(team_success_flags) > 0 and all(team_success_flags)
                if all_success and not getattr(world, '_global_reward_given', False):
                    world._global_reward_given = True
                    return float(self.success_reward_value)
                return 0.0
        except Exception as e:
            # 异常处理（不输出调试信息）
            return 0.0
        return 0.0

    def _estimate_terrain_normal(self, scenario: Any, x: float, y: float, delta: float = 1.0) -> np.ndarray:
        """估计地形法向量，失败时返回竖直向上向量。"""
        default_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        try:
            if scenario is None or not hasattr(scenario, 'get_terrain_height'):
                return default_normal
            h_x_plus = float(scenario.get_terrain_height(x + delta, y))
            h_x_minus = float(scenario.get_terrain_height(x - delta, y))
            h_y_plus = float(scenario.get_terrain_height(x, y + delta))
            h_y_minus = float(scenario.get_terrain_height(x, y - delta))
            dz_dx = (h_x_plus - h_x_minus) / (2.0 * delta)
            dz_dy = (h_y_plus - h_y_minus) / (2.0 * delta)
            normal = np.array([-dz_dx, -dz_dy, 1.0], dtype=np.float32)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-6 or not np.isfinite(norm):
                return default_normal
            return normal / norm
        except Exception:
            return default_normal

    def _lateral_reward_vectorized(self, agent: Any, world: Any, scenario: Any, positions: np.ndarray) -> np.ndarray:
        """
        向量化侧向/绕行奖励。
        与 weighted 版本对齐：在危险区域内，奖励切向绕行或向外远离动作。
        """
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        rewards = np.zeros(len(positions), dtype=np.float32)
        try:
            vel = getattr(getattr(agent, 'state', None), 'p_vel', None)
            if vel is None:
                return rewards
            vel = np.asarray(vel, dtype=np.float32).reshape(-1)
            speed = float(np.linalg.norm(vel))
            if speed < 0.1:
                return rewards
            vel_dir = vel / max(speed, 1e-6)

            activation_dist = float(getattr(scenario, 'lateral_activation_distance', 15.0)) if scenario is not None else 15.0
            activation_dist = max(activation_dist, 1e-6)

            for i, pos in enumerate(positions):
                min_dist = float('inf')
                danger_normal = None

                # 障碍物法向量（从障碍物中心指向智能体）
                if world is not None and hasattr(world, 'landmarks'):
                    for landmark in getattr(world, 'landmarks', []):
                        if landmark is None or not hasattr(landmark, 'state') or landmark.state.p_pos is None:
                            continue
                        obstacle_pos = np.asarray(landmark.state.p_pos, dtype=np.float32)
                        to_obstacle = obstacle_pos - pos
                        dist_center = float(np.linalg.norm(to_obstacle))
                        radius = float(getattr(landmark, 'size', getattr(landmark, 'radius', 1.0)))
                        dist_surface = max(0.0, dist_center - radius)
                        if dist_surface < min_dist and dist_surface < activation_dist:
                            min_dist = dist_surface
                            if dist_center > 1e-6:
                                danger_normal = (-to_obstacle / dist_center).astype(np.float32)
                            else:
                                danger_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                # 地形法向量（优先选更近的危险面）
                if scenario is not None and hasattr(scenario, 'get_terrain_height'):
                    try:
                        terrain_h = float(scenario.get_terrain_height(float(pos[0]), float(pos[1])))
                        dist_terrain = max(0.0, float(pos[2]) - terrain_h)
                        if dist_terrain < min_dist and dist_terrain < activation_dist:
                            min_dist = dist_terrain
                            danger_normal = self._estimate_terrain_normal(scenario, float(pos[0]), float(pos[1]))
                    except Exception:
                        pass

                if danger_normal is None:
                    continue

                cos_angle = float(np.dot(vel_dir, danger_normal))
                if cos_angle > -0.05:
                    dist_factor = 1.0 - (min_dist / activation_dist)
                    dist_factor = float(np.clip(dist_factor, 0.0, 1.0))
                    rewards[i] = float((0.3 + 0.7 * np.clip(cos_angle, 0.0, 1.0)) * dist_factor)
        except Exception:
            return np.zeros(len(positions), dtype=np.float32)
        return rewards

    def _collision_reduction_reward_vectorized(self, agent: Any, world: Any, scenario: Any, positions: np.ndarray) -> np.ndarray:
        """
        向量化碰撞次数减少奖励。
        与 weighted 版本语义保持一致：默认仅在回合最后一步产生非零奖励。
        """
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        rewards = np.zeros(len(positions), dtype=np.float32)
        try:
            if world is None:
                return rewards

            current_step = int(getattr(world, 'current_step', 0))
            episode_length = int(getattr(world, 'episode_length', 2800))
            if episode_length <= 0:
                episode_length = 2800

            # 兼容字段：当前回合碰撞计数直接使用累计穿透计数
            current_count = 0
            if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                current_count = agent.debug_info.get('total_penetration_count', 0)
            try:
                current_count = int(current_count) if np.isfinite(current_count) else 0
            except (ValueError, TypeError, OverflowError):
                current_count = 0
            current_count = max(current_count, 0)
            agent.current_episode_collision_count = current_count

            state = getattr(agent, '_collision_reduction_state', None)
            if not isinstance(state, dict):
                state = {
                    'last_seen_step': -1,
                    'reward_given': False,
                    'prev_collision_count': int(getattr(agent, 'previous_episode_collision_count', 0) or 0),
                }
            last_seen_step = int(state.get('last_seen_step', -1))

            # 新回合检测：step回绕到0或变小
            is_new_episode = False
            if last_seen_step >= 0:
                if current_step == 0 and last_seen_step > 0:
                    is_new_episode = True
                elif current_step < last_seen_step:
                    is_new_episode = True

            if is_new_episode:
                prev_collision_count = int(getattr(agent, '_last_episode_collision_count', state.get('prev_collision_count', 0)))
                prev_collision_count = max(prev_collision_count, 0)
                state['prev_collision_count'] = prev_collision_count
                state['reward_given'] = False
                agent.collision_reduction_reward_given = False

            is_last_step = current_step >= (episode_length - 1)
            if (not is_last_step) or bool(state.get('reward_given', False)):
                state['last_seen_step'] = current_step
                agent._collision_reduction_state = state
                return rewards

            prev_count = int(state.get('prev_collision_count', getattr(agent, 'previous_episode_collision_count', 0) or 0))
            prev_count = max(prev_count, 0)

            reward_val = 0.0
            if prev_count > 0 and current_count < prev_count:
                reward_val = float(np.clip(float(prev_count - current_count) / float(prev_count), 0.0, 1.0))
            elif prev_count == 0 and current_count == 0:
                reward_val = 0.1

            rewards[:] = reward_val

            # 标记一次性发放，并记录本回合碰撞计数供下一回合比较
            state['reward_given'] = True
            state['last_seen_step'] = current_step
            agent._collision_reduction_state = state
            agent.collision_reduction_reward_given = True
            agent._last_episode_collision_count = current_count
            agent.previous_episode_collision_count = current_count
        except Exception:
            return np.zeros(len(positions), dtype=np.float32)
        return rewards

    def _clearance_reward_vectorized(self, agent: Any, world: Any, positions: np.ndarray) -> np.ndarray:
        """
        净空奖励负责碰撞前预警与趋势引导，不再承担真实碰撞事件惩罚。

        分工：
        1. clearance: safe_distance 外只做趋势 shaping；进入预警带后给连续惩罚
        2. collision: 仅在真实碰撞阈值内给事件惩罚
        3. 趋势项同时考虑几何净空变化和当前速度相对危险源的径向分量
        """
        if positions.ndim == 1:
            if len(positions) >= 3:
                positions = positions[:3].reshape(1, -1)
            else:
                return np.zeros(1, dtype=np.float32)
        elif positions.ndim > 2:
            positions = positions.reshape(-1, positions.shape[-1])

        if positions.shape[-1] < 3:
            return np.zeros(positions.shape[0], dtype=np.float32)

        positions = positions[:, :3].astype(np.float32, copy=False)
        num_positions = positions.shape[0]
        rewards = np.zeros(num_positions, dtype=np.float32)

        safe_distance = max(float(os.getenv('OBSTACLE_SAFE_DISTANCE', '15.0')), 1e-6)
        collision_threshold = max(float(self.collision_distance_threshold), 1e-6)
        warning_span = max(safe_distance - collision_threshold, 1e-6)
        penalty_weight = float(os.getenv('CLEARANCE_PENALTY_WEIGHT', '5.0'))
        clearance_weight = float(os.getenv('CLEARANCE_WEIGHT', '3.5'))
        clearance_d_max = max(float(os.getenv('CLEARANCE_D_MAX', '80.0')), 1e-6)
        trend_penalty_factor = float(os.getenv('CLEARANCE_APPROACH_PENALTY_FACTOR', '2.0'))
        velocity_weight = float(os.getenv('CLEARANCE_VELOCITY_WEIGHT', '0.5'))
        speed_scale = max(float(os.getenv('CLEARANCE_SPEED_SCALE', '5.0')), 1e-6)
        trend_eps = float(os.getenv('CLEARANCE_TREND_EPS', '1e-6'))
        speed_eps = float(os.getenv('CLEARANCE_SPEED_EPS', '1e-4'))
        far_distance_fill = safe_distance + clearance_d_max

        scenario = getattr(world, 'scenario', None)
        cached_data = self._cache.get(id(world), {}) if isinstance(getattr(self, '_cache', None), dict) else {}
        obstacle_min_dist, nearest_obstacle_centers, nearest_obstacle_radii = self._compute_obstacle_distance_data(
            positions,
            cached_data=cached_data,
            world=world,
        )

        obstacle_reference_points = positions.copy()
        obstacle_vector = positions - nearest_obstacle_centers
        obstacle_norm = np.linalg.norm(obstacle_vector, axis=1, keepdims=True)
        default_dirs = np.zeros_like(obstacle_vector)
        default_dirs[:, 2] = 1.0
        obstacle_dirs = np.where(obstacle_norm > 1e-6, obstacle_vector / np.maximum(obstacle_norm, 1e-6), default_dirs)
        valid_obstacle_mask = np.isfinite(obstacle_min_dist)
        obstacle_reference_points[valid_obstacle_mask] = (
            nearest_obstacle_centers[valid_obstacle_mask]
            + obstacle_dirs[valid_obstacle_mask] * nearest_obstacle_radii[valid_obstacle_mask, None]
        )
        terrain_heights_current, terrain_warning_dist, terrain_reference_points = self._compute_terrain_distance_data(
            positions,
            scenario,
            cached_data=cached_data,
        )

        d_min_current = np.minimum(obstacle_min_dist, terrain_warning_dist).astype(np.float32)
        use_terrain_reference = terrain_warning_dist <= obstacle_min_dist
        reference_points = np.where(use_terrain_reference[:, None], terrain_reference_points, obstacle_reference_points).astype(np.float32)

        invalid_reference = ~np.all(np.isfinite(reference_points), axis=1)
        if np.any(invalid_reference):
            reference_points[invalid_reference] = positions[invalid_reference]

        goal_pos = None
        try:
            if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                goal_pos_raw = agent.goal_a.state.p_pos
                goal_pos = np.asarray(goal_pos_raw, dtype=np.float32)
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
        
        if goal_pos is None:
            try:
                scenario = getattr(world, 'scenario', None)
                if scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                    goal_pos_raw = scenario.goal_pos
                    goal_pos = np.asarray(goal_pos_raw, dtype=np.float32)
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

        if goal_pos is not None and len(goal_pos) >= 3:
            try:
                dists_to_goal = np.linalg.norm(positions - goal_pos[:3], axis=-1)
            except Exception:
                dists_to_goal = np.full(num_positions, 100.0, dtype=np.float32)
        else:
            dists_to_goal = np.full(num_positions, 100.0, dtype=np.float32)

        FAR_THRESHOLD = float(os.getenv('CLEARANCE_FAR_THRESHOLD', '50.0'))
        NEAR_THRESHOLD = float(os.getenv('CLEARANCE_NEAR_THRESHOLD', '20.0'))
        WEIGHT_FAR = float(os.getenv('CLEARANCE_WEIGHT_FAR', '0.5'))
        WEIGHT_NEAR = float(os.getenv('CLEARANCE_WEIGHT_NEAR', '12.0'))

        try:
            far_mask = dists_to_goal > FAR_THRESHOLD
            near_mask = dists_to_goal < NEAR_THRESHOLD
            transition_mask = ~(far_mask | near_mask)

            dynamic_weights = np.zeros(num_positions, dtype=np.float32)
            dynamic_weights[far_mask] = WEIGHT_FAR
            dynamic_weights[near_mask] = WEIGHT_NEAR

            if np.any(transition_mask):
                ratio = (dists_to_goal[transition_mask] - NEAR_THRESHOLD) / (FAR_THRESHOLD - NEAR_THRESHOLD)
                dynamic_weights[transition_mask] = WEIGHT_NEAR - ratio * (WEIGHT_NEAR - WEIGHT_FAR)
        except Exception:
            dynamic_weights = np.full(num_positions, WEIGHT_FAR, dtype=np.float32)

        try:
            if d_min_current.ndim == 0:
                d_min_current = np.full(num_positions, float(d_min_current), dtype=np.float32)
            elif d_min_current.shape[0] != num_positions:
                d_min_current = np.full(num_positions, far_distance_fill, dtype=np.float32)

            d_min_current = np.nan_to_num(
                d_min_current,
                nan=far_distance_fill,
                posinf=far_distance_fill,
                neginf=-far_distance_fill
            )

            if not hasattr(agent, 'last_min_distance'):
                agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
                d_min_previous = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current
            else:
                d_min_previous = agent.last_min_distance
                if isinstance(d_min_previous, np.ndarray):
                    if d_min_previous.shape[0] != num_positions:
                        seed_value = float(d_min_previous[0]) if len(d_min_previous) > 0 else far_distance_fill
                        d_min_previous = np.full(num_positions, seed_value, dtype=np.float32)
                else:
                    d_min_previous = np.full(num_positions, float(d_min_previous), dtype=np.float32)

            d_min_previous = np.nan_to_num(
                d_min_previous,
                nan=far_distance_fill,
                posinf=far_distance_fill,
                neginf=-far_distance_fill
            )
            distance_change = d_min_current - d_min_previous
            normalized_change = np.clip(distance_change / clearance_d_max, -1.0, 1.0)

            velocities = self._get_agent_velocity_array(agent, num_positions)
            hazard_vectors = positions - reference_points
            hazard_norms = np.linalg.norm(hazard_vectors, axis=1, keepdims=True)
            hazard_dirs = np.zeros_like(hazard_vectors)
            valid_hazard_mask = hazard_norms[:, 0] > 1e-6
            if np.any(valid_hazard_mask):
                hazard_dirs[valid_hazard_mask] = (
                    hazard_vectors[valid_hazard_mask] / hazard_norms[valid_hazard_mask]
                )

            speeds = np.linalg.norm(velocities, axis=1)
            radial_speed = np.sum(velocities * hazard_dirs, axis=1)
            motion_signal = np.zeros(num_positions, dtype=np.float32)
            moving_mask = valid_hazard_mask & (speeds > speed_eps)
            if np.any(moving_mask):
                motion_signal[moving_mask] = np.clip(radial_speed[moving_mask] / speed_scale, -1.0, 1.0)

            combined_trend = np.clip(normalized_change + velocity_weight * motion_signal, -1.0, 1.0)
            trend_reward = np.zeros(num_positions, dtype=np.float32)
            improving_mask = combined_trend > trend_eps
            worsening_mask = combined_trend < -trend_eps
            if np.any(improving_mask):
                trend_reward[improving_mask] = dynamic_weights[improving_mask] * combined_trend[improving_mask]
            if np.any(worsening_mask):
                trend_reward[worsening_mask] = (
                    dynamic_weights[worsening_mask]
                    * combined_trend[worsening_mask]
                    * trend_penalty_factor
                )

            try:
                goal_focus_radius = float(self.clearance_penalty_only_near_goal_radius)
                success_thr = float(self.success_distance_threshold)
                if goal_focus_radius > success_thr:
                    clearance_goal_focus_mask = (
                        (dists_to_goal > success_thr) &
                        (dists_to_goal < goal_focus_radius)
                    )
                    if np.any(clearance_goal_focus_mask):
                        positive_mask = clearance_goal_focus_mask & (trend_reward > 0.0)
                        if np.any(positive_mask):
                            trend_reward[positive_mask] *= float(self.clearance_near_goal_positive_factor)
            except Exception:
                pass

            warning_ratio = np.clip((safe_distance - d_min_current) / warning_span, 0.0, 1.0)
            warning_penalty = -penalty_weight * warning_ratio
            trend_active_mask = d_min_current >= collision_threshold
            rewards = warning_penalty + np.where(trend_active_mask, trend_reward, 0.0)

            max_positive = max(clearance_weight, WEIGHT_NEAR)
            min_negative = penalty_weight + max(clearance_weight, WEIGHT_NEAR) * trend_penalty_factor
            rewards = np.clip(rewards, -min_negative, max_positive * 2.0)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=max_positive * 2.0, neginf=-min_negative)
        except Exception:
            rewards = np.zeros(num_positions, dtype=np.float32)

        agent.last_min_distance = d_min_current.copy() if hasattr(d_min_current, 'copy') else d_min_current

        try:
            if not hasattr(agent, 'debug_info'):
                agent.debug_info = {}
            if not isinstance(agent.debug_info, dict):
                agent.debug_info = {}
            if isinstance(d_min_current, np.ndarray) and d_min_current.size > 0:
                d_min_scalar = float(d_min_current[-1] if d_min_current.ndim > 0 else d_min_current.item())
            else:
                d_min_scalar = float(d_min_current) if not isinstance(d_min_current, np.ndarray) else float(d_min_current.item())
            agent.debug_info['d_min_current'] = d_min_scalar
            if 'radial_speed' in locals():
                agent.debug_info['clearance_radial_speed'] = float(radial_speed[-1]) if radial_speed.size > 0 else 0.0
            if 'motion_signal' in locals():
                agent.debug_info['clearance_motion_signal'] = float(motion_signal[-1]) if motion_signal.size > 0 else 0.0
            if 'use_terrain_reference' in locals():
                agent.debug_info['clearance_reference'] = 'terrain' if bool(use_terrain_reference[-1]) else 'obstacle'
            # 维护 episode 级最小安全距离（用于 success-only 质量奖励）
            try:
                if world is not None:
                    prev_min = getattr(world, '_episode_dmin_min', None)
                    cur = float(d_min_scalar)
                    if prev_min is None or (np.isfinite(cur) and (not np.isfinite(prev_min) or cur < float(prev_min))):
                        setattr(world, '_episode_dmin_min', cur)
            except Exception:
                pass
        except Exception:
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

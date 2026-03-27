#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分项加权求和奖励机制的3D地形场景
将奖励分解为独立的分项，通过权重配置进行加权求和
适合随机地形训练，提高奖励稳定性
"""

import numpy as np
from multiagent.scenario import BaseScenario
from multiagent.scenarios.paper3d_terrain_energy import Scenario as BaseTerrainScenario

# 导入ARW模块（优先相对导入，回退顶层导入）
try:
    from .reward_calculator import AdaptiveRewardWeighting
    ARW_AVAILABLE = True
except Exception:
    try:
        from reward_calculator import AdaptiveRewardWeighting
        ARW_AVAILABLE = True
    except Exception:
        ARW_AVAILABLE = False
        print("[Warning] ARW module not available, adaptive reward weighting disabled")

class Scenario(BaseTerrainScenario):
    """
    基于分项加权求和的3D地形场景
    继承原有场景的所有功能，但使用新的奖励机制
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 课程阶段绑定的 reward profile：
        # - fixed_route：阶段1/普通固定环境，保持当前主路线
        # - obstacle_route：阶段2+，针对随机障碍强化局部绕障与团队过程梯度
        try:
            import os
            raw_reward_profile = kwargs.get(
                'reward_profile',
                os.getenv('CURRICULUM_REWARD_PROFILE', os.getenv('REWARD_PROFILE', 'fixed_route'))
            )
            self.reward_profile = self._normalize_reward_profile_name(raw_reward_profile)
            self.curriculum_stage_id = int(
                kwargs.get('curriculum_stage_id', os.getenv('CURRICULUM_STAGE_ID', '0'))
            )
        except Exception:
            self.reward_profile = 'fixed_route'
            self.curriculum_stage_id = 0

        try:
            import os
            self.restructured_reward_enabled = kwargs.get(
                'restructured_reward_enabled',
                os.getenv('RESTRUCTURED_REWARD', '1').lower() in ('1', 'true', 'yes', 'on')
            )
            self.dense_energy_enabled = kwargs.get(
                'dense_energy_enabled',
                os.getenv('DENSE_ENERGY_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
            )
        except Exception:
            self.restructured_reward_enabled = True
            self.dense_energy_enabled = True
        self._dense_reward_keys_fixed = (
            'distance', 'stationary', 'height', 'success', 'collision', 'global', 'clearance'
        )
        self._dense_reward_keys_obstacle = (
            'distance', 'stationary', 'height', 'success', 'collision', 'global', 'clearance', 'lateral'
        )
        
        # 初始化奖励权重配置（可通过参数调整）
        self.reward_weights = {
            'distance': kwargs.get('distance_weight', 1.0),
            'exploration': kwargs.get('exploration_weight', 0.5),
            'stationary': kwargs.get('stationary_weight', 1.0),
            'direction': kwargs.get('direction_weight', 0.3),
            'deviation': kwargs.get('deviation_weight', -2.0),
            'start_area': kwargs.get('start_area_weight', 0.2),
            'approach': kwargs.get('approach_weight', 1.0),
            'energy': kwargs.get('energy_weight', 0.1),
            'height': kwargs.get('height_weight', 0.2),
            # 新增分项权重
            'success': kwargs.get('success_weight', 0.0),
            'collision': kwargs.get('collision_weight', 0.0),
            'global': kwargs.get('global_weight', 0.0),
            'shaping': kwargs.get('shaping_weight', 0.0),
            'lateral': kwargs.get('lateral_weight', 0.4),
            'clearance': kwargs.get('clearance_weight', 0.4),
            'collision_reduction': kwargs.get('collision_reduction_weight', 0.0)  # 🔧 新增：碰撞次数减少奖励
        }
        
        # 奖励范围限制（默认值与Shell脚本对齐，防止过度裁剪）
        self.max_reward = kwargs.get('max_reward', 1000.0)
        self.min_reward = kwargs.get('min_reward', -2500.0)
        
        # 新增分项参数
        self.success_reward_value = kwargs.get('success_reward_value', 150.0)
        self.success_distance_threshold = kwargs.get('success_distance_threshold', 2.0)
        self.collision_penalty_value = kwargs.get('collision_penalty_value', 30.0)
        self.collision_distance_threshold = kwargs.get('collision_distance_threshold', 0.5)
        self.global_reward_mode = kwargs.get('global_reward_mode', 'success_rate')
        self.shaping_gamma = kwargs.get('shaping_gamma', 0.95)
        
        # 🔧 悬停奖励参数（支持环境变量）
        try:
            import os
            self.hover_reward_max = float(kwargs.get('hover_reward_max', os.getenv('HOVER_REWARD_MAX', '12.0')))
            self.hover_speed_threshold = float(kwargs.get('hover_speed_threshold', os.getenv('HOVER_SPEED_THRESHOLD', '1.0')))
            self.hover_reward_interval = int(kwargs.get('hover_reward_interval', os.getenv('HOVER_REWARD_INTERVAL', '5')))
            self.goal_hold_reward = float(kwargs.get('goal_hold_reward', os.getenv('GOAL_HOLD_REWARD', str(self.hover_reward_max))))
            self.leave_goal_penalty = float(kwargs.get('leave_goal_penalty', os.getenv('LEAVE_GOAL_PENALTY', str(self.hover_reward_max))))
        except Exception:
            self.hover_reward_max = 12.0  # 🔧 从5.0提高到12.0，增强悬停奖励
            self.hover_speed_threshold = 1.0
            self.hover_reward_interval = 5  # 🔧 从10改为5，增加悬停奖励频率
            self.goal_hold_reward = self.hover_reward_max
            self.leave_goal_penalty = self.hover_reward_max
        
        # 侧向/净空奖励参数
        self.clearance_d_max = kwargs.get('clearance_d_max', 66.0)
        self.lateral_activation_distance = kwargs.get('lateral_activation_distance', 15.0)
        self.terrain_gradient_threshold = kwargs.get('terrain_gradient_threshold', 0.5)
        self.team_goal_occupancy_scale = float(kwargs.get('team_goal_occupancy_scale', os.getenv('TEAM_GOAL_OCCUPANCY_SCALE', '1.0')))
        self.team_bottleneck_progress_scale = float(kwargs.get('team_bottleneck_progress_scale', os.getenv('TEAM_BOTTLENECK_PROGRESS_SCALE', '4.0')))
        self.team_waiting_scale = float(kwargs.get('team_waiting_scale', os.getenv('TEAM_WAITING_SCALE', '0.6')))
        self.team_waiting_speed_threshold = float(kwargs.get('team_waiting_speed_threshold', os.getenv('TEAM_WAITING_SPEED_THRESHOLD', str(self.hover_speed_threshold))))

        # 性能优化：缓存环境变量（训练期间不变，避免每步重复os.getenv）
        import os
        self._terrain_contact_eps = float(os.getenv('TERRAIN_CONTACT_EPS', '0.3'))
        self._terrain_collision_eps = float(os.getenv('TERRAIN_COLLISION_EPS', '0.3'))
        self._enable_collision_debug = os.getenv('ENABLE_COLLISION_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
        self._early_stop_mode = os.getenv('EARLY_STOP_MODE', 'never').lower()
        self._quiet_output = os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')

        # 轨迹平滑（最小拐弯角）奖励权重：默认从环境变量读取，可通过CLI覆盖
        try:
            import os
            self.turn_smooth_weight = float(
                kwargs.get('turn_smooth_weight', os.getenv('TURN_SMOOTH_WEIGHT', '0.0'))
            )
        except Exception:
            self.turn_smooth_weight = 0.0
        
        # 高度奖励可配置开关与范围（支持kwargs与环境变量）
        try:
            import os
            _env_enabled = os.getenv('HEIGHT_REWARD_ENABLED', '1')
            self.height_reward_enabled = kwargs.get('height_reward_enabled', _env_enabled not in ('0', 'false', 'False'))
            self.height_ideal_min = float(kwargs.get('height_ideal_min', os.getenv('HEIGHT_IDEAL_MIN', '2.0')))
            self.height_ideal_max = float(kwargs.get('height_ideal_max', os.getenv('HEIGHT_IDEAL_MAX', '5.0')))
            # 安全保护：保证最小值不大于最大值
            if self.height_ideal_min > self.height_ideal_max:
                self.height_ideal_min, self.height_ideal_max = self.height_ideal_max, self.height_ideal_min
            
            # 调试输出：高度奖励配置
            print(f"[高度奖励配置] enabled={self.height_reward_enabled} | weight={self.reward_weights['height']} | ideal_range=[{self.height_ideal_min:.1f},{self.height_ideal_max:.1f}]")
        except Exception:
            # 兜底到默认范围
            self.height_reward_enabled = True
            self.height_ideal_min = 2.0
            self.height_ideal_max = 5.0
            print(f"[高度奖励配置] 使用默认值: enabled={self.height_reward_enabled} | ideal_range=[{self.height_ideal_min:.1f},{self.height_ideal_max:.1f}]")
        
        # 初始化ARW模块
        self.arw = None
        if ARW_AVAILABLE and kwargs.get('enable_arw', True):
            arw_config = {
                'base_penalty': kwargs.get('arw_base_penalty', -10.0),
                'alpha': kwargs.get('arw_alpha', 0.5),
                'base_c': kwargs.get('arw_base_c', 10.0),
                'beta': kwargs.get('arw_beta', 0.3),
                'max_episodes': kwargs.get('arw_max_episodes', 20000),
                'warmup_episodes': kwargs.get('arw_warmup_episodes', 200),
                'enable_adaptive': kwargs.get('arw_enable', True)
            }
            self.arw = AdaptiveRewardWeighting(**arw_config)
            # 暴露统一句柄，便于训练侧写入/读取（与现有代码的reward_calculator占位兼容）
            try:
                self.reward_calculator = self.arw
            except Exception:
                pass
            print(f"[ARW] Adaptive Reward Weighting enabled with config: {arw_config}")
        else:
            print("[ARW] Adaptive Reward Weighting disabled")

    @staticmethod
    def _normalize_reward_profile_name(raw_profile):
        try:
            profile = str(raw_profile).strip().lower()
        except Exception:
            profile = 'fixed_route'
        if profile == 'obstacle_route':
            return 'obstacle_route'
        return 'fixed_route'

    def _is_obstacle_reward_route(self):
        return getattr(self, 'reward_profile', 'fixed_route') == 'obstacle_route'

    def _dense_reward_keys(self):
        dense_keys = (
            self._dense_reward_keys_obstacle
            if self._is_obstacle_reward_route()
            else self._dense_reward_keys_fixed
        )
        if getattr(self, 'dense_energy_enabled', False):
            dense_keys = dense_keys + ('energy',)
        return dense_keys

    def _adjust_approach_reward_for_obstacle_route(self, agent, approach_reward):
        """
        第二阶段（随机障碍路线）的最简 progress 调整：
        仅在靠近障碍时放宽“短暂远离目标”的负 approach，不新增额外调参项。
        """
        try:
            approach_reward = float(approach_reward)
            if approach_reward >= 0.0 or not self._is_obstacle_reward_route():
                return approach_reward

            activation_distance = max(float(getattr(self, 'lateral_activation_distance', 15.0)), 1e-6)
            obstacle_min_dist = getattr(agent, '_rc_dmin', None)
            if obstacle_min_dist is None or not np.isfinite(obstacle_min_dist):
                return approach_reward

            relax = float(np.clip(float(obstacle_min_dist) / activation_distance, 0.0, 1.0))
            return approach_reward * relax
        except Exception:
            return float(approach_reward)
    
    def reward(self, agent, world):
        """
        分项加权求和奖励函数
        将奖励分解为独立的分项，通过权重配置进行加权求和
        适合随机地形训练，提高奖励稳定性
        """
        try:
            # 获取智能体的独立目标位置
            if not hasattr(agent, 'goal_a') or agent.goal_a.state.p_pos is None:
                return 0.0 # Agent has no goal, no reward
            goal_pos = agent.goal_a.state.p_pos
            
            # 计算到目标的距离
            agent_pos = agent.state.p_pos
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            
                # 初始化智能体需要的所有属性
            if not hasattr(agent, 'initialized_for_reward') or not agent.initialized_for_reward:
                agent.last_goal_dist = dist_to_goal
                agent.stationary_count = 0
                agent.last_position = agent.state.p_pos.copy()
                agent.last_velocity = np.zeros(3)
                agent.visited_cells = set()
                agent.debug_info = {}
                agent.initialized_for_reward = True
                agent.start_position = agent.state.p_pos.copy()
                agent.current_episode_collision_count = 0
                agent.previous_episode_collision_count = 0
                agent.collision_reduction_reward_given = False
                agent.debug_info['total_penetration_count'] = 0
                agent.debug_info['terrain_penetration_count'] = 0
                agent.debug_info['obstacle_collision_count'] = 0
                # 回合开始时重置全局奖励标记（所有agent共享world，只需置一次）
                if hasattr(world, '_global_reward_given'):
                    world._global_reward_given = False
                if hasattr(world, '_team_sync_step_cache'):
                    world._team_sync_step_cache = None
                if hasattr(world, '_team_sync_state'):
                    world._team_sync_state = None
                
                # 计算从起始点到目标点的初始距离和方向（按每智能体的真实目标）
                agent.initial_distance_to_goal = dist_to_goal  # 记录初始距离
                agent.start_to_goal_dir = None
                try:
                    if hasattr(agent, 'goal_a') and agent.goal_a is not None and agent.goal_a.state.p_pos is not None:
                        goal_pos_true = agent.goal_a.state.p_pos
                    else:
                        goal_pos_true = self.goal_pos
                    if goal_pos_true is not None:
                        start_to_goal = goal_pos_true - agent.state.p_pos
                        _d = np.linalg.norm(start_to_goal)
                        if _d > 1e-6:
                            agent.start_to_goal_dir = start_to_goal / _d
                except Exception:
                    pass
                
                return 0.0

            # 性能优化：预计算本步共享值，用属性存储避免每步分配 dict
            _th = self.get_terrain_height(agent_pos[0], agent_pos[1]) if hasattr(self, 'get_terrain_height') else 0.0
            _dmin_obs = None
            if hasattr(world, 'nearest_obstacle_distance'):
                try:
                    _dmin_obs = world.nearest_obstacle_distance(agent)
                except Exception:
                    pass
            _dist_to_start = np.linalg.norm(agent_pos - agent.start_position) if hasattr(agent, 'start_position') else 0.0
            agent._rc_th = _th
            agent._rc_dgoal = dist_to_goal
            agent._rc_dstart = _dist_to_start
            agent._rc_dmin = _dmin_obs
            agent._rc_collision = False

            # collision 必须在 collision_reduction 之前求值
            distance_reward = self._calculate_distance_reward(agent, world, dist_to_goal)
            approach_reward = self._calculate_approach_reward(agent, world, dist_to_goal)
            approach_reward = self._adjust_approach_reward_for_obstacle_route(agent, approach_reward)
            progress_reward = self._merge_progress_reward(distance_reward, approach_reward)

            rewards = {
                'distance': progress_reward,  # distance 通道承载 merged progress
                'exploration': 0.0,          # legacy shaping：已归档，默认主路径移出
                'stationary': self._calculate_stationary_penalty(agent, world),
                'direction': self._calculate_direction_reward(agent, world),
                'deviation': 0.0,            # legacy shaping：已归档，默认主路径移出
                'start_area': 0.0,           # legacy shaping：已归档，默认主路径移出
                'approach': 0.0,             # legacy 占位：已并入 progress
                'energy': self._calculate_energy_reward(agent, world),
                'height': self._calculate_height_reward(agent, world),
                'lateral': self._calculate_lateral_reward(agent, world),
                'clearance': self._calculate_clearance_reward(agent, world),
                'success': self._calculate_success_reward(agent, world, dist_to_goal),
                'collision': self._calculate_collision_penalty(agent, world),
                'collision_reduction': 0.0,  # legacy shaping：已归档，默认主路径移出
                'global': self._calculate_team_sync_reward(agent, world) + self._calculate_global_reward(world),
                'shaping': 0.0               # legacy shaping：已归档，默认主路径移出
            }
            
            # 加权求和
            if getattr(self, 'restructured_reward_enabled', False):
                dense_keys = self._dense_reward_keys()
                total_reward = sum(self.reward_weights.get(key, 0.0) * rewards.get(key, 0.0) for key in dense_keys)
            else:
                total_reward = sum(self.reward_weights[key] * rewards[key] for key in rewards.keys())
            total_before_clip = total_reward
            
            # 应用ARW调整
            if self.arw:
                # 创建奖励项字典用于ARW调整
                reward_terms = {
                    'collision': rewards.get('collision', 0.0),
                    'distance': rewards.get('distance', 0.0)
                }
                
                # 应用ARW调整
                adjusted_terms = self.arw.apply_to_rewards(reward_terms)
                
                # 重新计算总奖励
                if getattr(self, 'restructured_reward_enabled', False):
                    dense_keys = self._dense_reward_keys()
                    total_reward = sum(self.reward_weights.get(key, 0.0) * rewards.get(key, 0.0) for key in dense_keys)
                else:
                    total_reward = sum(self.reward_weights[key] * rewards[key] for key in rewards.keys())
                
                # 应用ARW调整的差异
                collision_adjustment = adjusted_terms.get('collision', 0.0) - reward_terms.get('collision', 0.0)
                distance_adjustment = adjusted_terms.get('distance', 0.0) - reward_terms.get('distance', 0.0)
                
                total_reward += collision_adjustment + distance_adjustment
                
                # 记录ARW调整信息
                if hasattr(agent, 'debug_info'):
                    agent.debug_info['arw_adjustments'] = {
                        'collision_adjustment': collision_adjustment,
                        'distance_adjustment': distance_adjustment,
                        'arw_weights': self.arw.get_current_weights()
                    }
            
            # 记录调试信息（在clip之前）
            if hasattr(agent, 'debug_info'):
                agent.debug_info.update({
                    'rewards': rewards,
                    'weights': self.reward_weights,
                    'total_before_clip': total_before_clip,
                    'reward_profile': getattr(self, 'reward_profile', 'fixed_route'),
                })
            
            # 限制奖励范围
            total_reward = np.clip(total_reward, self.min_reward, self.max_reward)
            
            # 记录clip后的值
            if hasattr(agent, 'debug_info'):
                agent.debug_info['total_after_clip'] = total_reward
                agent.debug_info['clipped'] = abs(total_before_clip - total_reward) > 0.01
            
            # === 调试记录器集成 ===
            try:
                if getattr(world, 'enable_reward_debug', False):
                    from reward_debugger import get_debugger
                    import os as _os_inner
                    # 从环境变量读取log_dir（由主进程设置）
                    log_dir = _os_inner.getenv('REWARD_DEBUG_LOG_DIR', './reward_debug_logs')
                    debugger = get_debugger(log_dir=log_dir, enabled=True)
                    if debugger.enabled:
                        # 获取step和索引信息
                        current_step = getattr(world, 'current_step', 0)
                        env_idx = getattr(agent, 'env_idx', 0)
                        # agent可能没有id属性，使用name或其他标识
                        if hasattr(agent, 'id'):
                            agent_idx = agent.id
                        elif hasattr(agent, 'name'):
                            # 从name中提取数字，如"agent_0" -> 0
                            try:
                                agent_idx = int(agent.name.split('_')[-1])
                            except:
                                agent_idx = 0
                        else:
                            agent_idx = 0
                        
                        debugger.log_step(
                            step=current_step,
                            env_idx=env_idx,
                            agent_idx=agent_idx,
                            components=rewards,
                            weights=self.reward_weights,
                            total_before_clip=total_before_clip,
                            total_after_clip=total_reward
                        )
            except Exception as debug_error:
                # 调试功能不应该影响正常运行
                if not hasattr(self, '_debug_error_printed'):
                    print(f"[调试器警告] 无法记录奖励调试信息: {debug_error}")
                    import traceback
                    traceback.print_exc()
                    self._debug_error_printed = True
            
            return total_reward
            
        except Exception as e:
            print(f"奖励计算异常: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # 出错时返回零奖励

    # 由基类 reset_world 在每回合开始时触发（见 paper3d_terrain_energy.reset_world）
    def on_episode_start(self):
        try:
            if getattr(self, 'arw', None) is not None:
                ep = getattr(self, '_arw_episode', 0) + 1
                setattr(self, '_arw_episode', ep)
                self.arw.on_episode_start(ep)
        except Exception:
            pass
        try:
            if hasattr(self, 'world') and self.world is not None:
                self.world._team_sync_step_cache = None
                self.world._team_sync_state = None
        except Exception:
            pass
        
        # 🔧 新增：重置碰撞计数（在回合开始时）
        # 注意：这里会在world.reset_world时被调用，此时agents已经重置
        # 我们需要在reward函数中初始化，但这里可以确保上一回合的计数被保存
        try:
            # 尝试从多个可能的路径获取world对象
            world_obj = None
            if hasattr(self, 'world'):
                world_obj = self.world
            elif hasattr(self, '_world'):
                world_obj = self._world
            
            if world_obj and hasattr(world_obj, 'agents'):
                for agent in world_obj.agents:
                    if hasattr(agent, 'current_episode_collision_count'):
                        # 保存上一回合的碰撞次数
                        agent.previous_episode_collision_count = agent.current_episode_collision_count
                        # 重置当前回合碰撞计数
                        agent.current_episode_collision_count = 0
                        agent.collision_reduction_reward_given = False
                    
                    # 🔧 修复：重置debug_info中的total_penetration_count
                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    if not isinstance(agent.debug_info, dict):
                        agent.debug_info = {}
                    agent.debug_info['total_penetration_count'] = 0
                    agent.debug_info['terrain_penetration_count'] = 0
                    agent.debug_info['obstacle_collision_count'] = 0
                    
                    # 🔧 修复：重置min_distance相关属性
                    if hasattr(agent, 'd_min_prev'):
                        agent.d_min_prev = None
                    if hasattr(agent, 'last_min_distance'):
                        agent.last_min_distance = None
                    if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                        agent.debug_info['d_min_current'] = None
                    
                    # 🚨 关键修复：重置防重复计数标志，确保每回合开始时重置
                    agent._last_collision_counted_step = -1
        except Exception:
            pass

    def _get_agent_goal_pos(self, agent):
        try:
            if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                return agent.goal_a.state.p_pos
        except Exception:
            pass
        return getattr(self, 'goal_pos', None)

    def _get_success_state(self, agent):
        if not hasattr(agent, '_success_state') or not isinstance(agent._success_state, dict):
            agent._success_state = {}
        agent._success_state.setdefault('success_reward_given', False)
        agent._success_state.setdefault('first_success_step', None)
        agent._success_state.setdefault('hover_reward_count', 0)
        agent._success_state.setdefault('was_in_goal_zone', False)
        return agent._success_state

    def _agent_safe_so_far(self, agent):
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

    def _record_termination_reason(self, world, agent, reason):
        try:
            if not hasattr(world, '_termination_reasons') or not isinstance(world._termination_reasons, dict):
                world._termination_reasons = {}
            agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
            world._termination_reasons[agent_name] = [reason]
        except Exception:
            pass
    
    def _calculate_distance_reward(self, agent, world, dist_to_goal):
        """计算距离相关奖励（相对化，适合随机地形）"""
        # 确保 initial_distance_to_goal 存在
        if not hasattr(agent, 'initial_distance_to_goal'):
            agent.initial_distance_to_goal = dist_to_goal
        
        # 使用相对距离，减少地形变化影响
        if agent.initial_distance_to_goal > 0:
            relative_dist = dist_to_goal / agent.initial_distance_to_goal
            # 🔧 修复：降低距离奖励系数（10.0→3.0），减少每步奖励累积
            # 原因：距离奖励每步都计算，在目标附近停留会累积大量奖励，导致奖励值波动大
            # 新值3.0：降低70%，减少每步奖励，使奖励更平滑
            return -(relative_dist - 1.0) * 3.0  # 归一化到合理范围
        else:
            return -dist_to_goal * 0.01  # 回退到绝对距离

    def _merge_progress_reward(self, distance_reward, approach_reward):
        """
        将 distance(状态锚点) 与 approach(步进结果) 合并为单一 progress 通道。
        使用当前脚本中的 distance/approach 权重做加权平均，避免两条通道重复计分。
        """
        try:
            dw = abs(float(self.reward_weights.get('distance', 0.0)))
            aw = abs(float(self.reward_weights.get('approach', 0.0)))
            denom = dw + aw
            if denom < 1e-6:
                return 0.0
            return float((dw * float(distance_reward) + aw * float(approach_reward)) / denom)
        except Exception:
            return float(distance_reward)

    def _calculate_team_sync_reward(self, agent, world):
        """
        团队同步过程奖励（dense）：
        - 所有阶段统一：occupancy/waiting 按 Succ_i（Reach_i ∧ Safe_i）
        - bottleneck：统一看未成功体

        返回值对同一步内所有 agent 相同，并通过 world 级缓存保证只计算一次。
        """
        try:
            agents = getattr(world, 'agents', [])
            if not agents:
                return 0.0

            cur_step = int(getattr(world, 'current_step', -1))
            cache = getattr(world, '_team_sync_step_cache', None)
            if cache is not None and cache[0] == cur_step:
                return float(cache[1])

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

            speeds = []
            reach_flags = []
            succ_flags = []
            remaining_dists = []
            thr_success = float(self.success_distance_threshold)
            for ag in agents:
                pos = getattr(getattr(ag, 'state', None), 'p_pos', None)
                goal = self._get_agent_goal_pos(ag)
                if pos is None or goal is None:
                    reach_flags.append(False)
                    succ_flags.append(False)
                    speeds.append(0.0)
                    continue
                dist = float(np.linalg.norm(pos - goal))
                reach_i = bool(dist <= thr_success)
                safe_i = bool(self._agent_safe_so_far(ag))
                succ_i = bool(reach_i and safe_i)
                reach_flags.append(reach_i)
                succ_flags.append(succ_i)
                progress_flag = succ_i
                if not progress_flag:
                    remaining_dists.append(dist)
                try:
                    speeds.append(float(np.linalg.norm(getattr(getattr(ag, 'state', None), 'p_vel', np.zeros(3)))))
                except Exception:
                    speeds.append(0.0)

            n_agents = max(len(agents), 1)
            progress_flags = succ_flags
            occupancy_ratio = float(sum(1 for v in progress_flags if v)) / float(n_agents)
            bottleneck_dist = max(remaining_dists) if remaining_dists else 0.0
            if is_new_episode:
                state['last_bottleneck_dist'] = bottleneck_dist
            last_bottleneck_dist = float(state.get('last_bottleneck_dist', bottleneck_dist))
            bottleneck_delta = max(0.0, last_bottleneck_dist - bottleneck_dist)
            bottleneck_delta = min(bottleneck_delta, 1.0)

            waiting_speed_thr = float(getattr(self, 'team_waiting_speed_threshold', getattr(self, 'hover_speed_threshold', 1.0)))
            all_progress = all(progress_flags) if progress_flags else False
            waiting_ratio = float(
                sum(1 for progress_i, spd in zip(progress_flags, speeds) if progress_i and spd <= waiting_speed_thr and not all_progress)
            ) / float(n_agents)

            reward_scalar = (
                float(getattr(self, 'team_goal_occupancy_scale', 1.0)) * occupancy_ratio +
                float(getattr(self, 'team_bottleneck_progress_scale', 4.0)) * bottleneck_delta +
                float(getattr(self, 'team_waiting_scale', 0.6)) * waiting_ratio
            )

            state['last_bottleneck_dist'] = bottleneck_dist
            state['last_step'] = cur_step
            world._team_sync_state = state
            world._team_sync_step_cache = (cur_step, float(reward_scalar))
            world._team_sync_reward = float(reward_scalar)
            world._team_sync_occupancy_ratio = float(occupancy_ratio)
            world._team_sync_bottleneck_dist = float(bottleneck_dist)
            world._team_sync_bottleneck_delta = float(bottleneck_delta)
            world._team_sync_waiting_ratio = float(waiting_ratio)
            world._team_sync_occupancy_basis = 'reach' if use_reach_progress else 'succ'
            return float(reward_scalar)
        except Exception:
            return 0.0
    
    def _calculate_exploration_reward(self, agent, world):
        """计算探索奖励"""
        # 将连续空间离散化为网格
        cell_size = 3.0
        x_cell = int(agent.state.p_pos[0] / cell_size)
        y_cell = int(agent.state.p_pos[1] / cell_size)
        z_cell = int(agent.state.p_pos[2] / cell_size)
        current_cell = (x_cell, y_cell, z_cell)
        
        # 如果访问了新的区域，给予奖励
        if current_cell not in agent.visited_cells:
            agent.visited_cells.add(current_cell)
            return 5.0  # 探索奖励
        return 0.0
    
    def _calculate_stationary_penalty(self, agent, world):
        """计算停滞惩罚 - 修复：避免因复位机制导致的惩罚爆炸"""
        # 计算位置变化（仅XY平面），避免Z轴动作影响停滞判定
        try:
            pos_change = np.linalg.norm((agent.state.p_pos - agent.last_position)[:2])
        except Exception:
            pos_change = np.linalg.norm(agent.state.p_pos - agent.last_position)
        # 更新上次位置
        agent.last_position = agent.state.p_pos.copy()
        
        # 🔧 修复：放宽停滞阈值，避免复位机制导致的误判
        # 从0.01米放宽到0.05米，给智能体更多缓冲空间
        is_stationary = pos_change < 0.05
        if is_stationary:
            agent.stationary_count += 1
        else:
            agent.stationary_count = 0
        
        # 🔧 修复：改用线性惩罚+上限，避免指数爆炸
        # 只有长时间停滞（>10步）才给予惩罚，且惩罚有明确上限
        if agent.stationary_count > 10:
            dist_to_start = getattr(agent, '_rc_dstart', None)
            if dist_to_start is None:
                dist_to_start = np.linalg.norm(agent.state.p_pos - agent.start_position)
            dist_to_goal = getattr(agent, '_rc_dgoal', None)
            if dist_to_goal is None:
                dist_to_goal = np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos)
            
            # 🔧 修复：使用固定惩罚+线性增长，设置明确上限
            base_penalty = -5.0  # 降低基础惩罚（-25 → -5）
            # 线性增长，但设置合理上限
            extra_penalty = min((agent.stationary_count - 10) * 0.1, 10.0)  # 最多额外-10
            
            if dist_to_start < 15:  # 在起始区域附近
                # 起始区域稍微加重，但不再乘以2倍
                return base_penalty - extra_penalty * 1.5  # 最多 -5 - 15 = -20
            elif dist_to_goal < 5:  # 非常接近目标
                return -0.5  # 轻微惩罚（已接近目标，允许稳定）
            else:
                return base_penalty - extra_penalty  # 最多 -5 - 10 = -15
        
        return 0.0
    
    def _calculate_direction_reward(self, agent, world):
        """计算方向一致性奖励 + 基础移动奖励 + 轨迹平滑（最小拐弯角）奖励"""
        vel = getattr(agent.state, 'p_vel', np.zeros(3))
        speed = np.linalg.norm(vel)
        base_movement_reward = 0.0
        if speed > 0.05:
            base_movement_reward = min(speed * 2.0, 5.0)
        _dstart = getattr(agent, '_rc_dstart', None)
        dist_to_start = _dstart if _dstart is not None else np.linalg.norm(agent.state.p_pos - agent.start_position)
        alignment_bonus = 0.0
        if dist_to_start < 20 and hasattr(agent, 'start_to_goal_dir') and agent.start_to_goal_dir is not None:
            if speed > 0.1:  # 有明确的移动（3D速度）
                vel_dir = vel / speed
                init_alignment = np.dot(vel_dir, agent.start_to_goal_dir)
                if init_alignment > 0.0:
                    alignment_bonus = init_alignment * 4.0 * (20 - dist_to_start) / 20.0
        
        # 轨迹平滑奖励：鼓励相邻两步速度方向夹角尽量小，抑制"过山车式"剧烈拐弯
        # 🔧 改进：结合距离目标点信息，距离越近平滑度要求越高
        smooth_reward = 0.0
        try:
            if not hasattr(agent, 'last_velocity'):
                agent.last_velocity = np.zeros(3)
            prev_vel = agent.last_velocity
            prev_speed = np.linalg.norm(prev_vel)
            if prev_speed > 0.1 and speed > 0.1:
                prev_dir = prev_vel / prev_speed
                cur_dir = vel / speed
                cos_turn = float(np.dot(prev_dir, cur_dir))
                
                # 计算到目标的距离（用于调节平滑权重）
                goal_pos = None
                if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state'):
                    goal_pos = getattr(agent.goal_a.state, 'p_pos', None)
                
                if goal_pos is not None:
                    _dgoal = getattr(agent, '_rc_dgoal', None)
                    dist_to_goal = _dgoal if _dgoal is not None else float(np.linalg.norm(agent.state.p_pos - goal_pos))
                    # 使用平滑过渡函数：sigmoid((50-dist)/20)映射到[0.3, 1.2]
                    dist_factor = 0.3 + 0.9 / (1.0 + np.exp((dist_to_goal - 35.0) / 10.0))
                else:
                    # 无法获取目标位置时使用默认权重
                    dist_factor = 1.0
                
                # 平滑度评分：只在转向朝向目标时给奖励（cos>0且朝目标方向转）
                # 如果转向背离目标（cos<0），给予惩罚
                if goal_pos is not None:
                    to_goal = goal_pos - agent.state.p_pos
                    goal_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)
                    # 检查当前速度方向与目标方向的夹角
                    vel_to_goal_align = float(np.dot(cur_dir, goal_dir))
                    # 如果朝向目标(>0)：根据cos_turn给平滑奖励
                    # 如果背离目标(<0)：轻微惩罚转向（除非是必要的调头）
                    if vel_to_goal_align > 0.3:
                        # 朝向目标：奖励平滑转向，cos接近1奖励高
                        smooth_score = max(min(cos_turn, 1.0), 0.0) * dist_factor
                    elif cos_turn < -0.5:
                        # 背离目标但在大幅度转向（可能是调头）：轻微奖励
                        smooth_score = 0.2 * dist_factor
                    else:
                        # 背离目标且小幅转向：不奖励
                        smooth_score = 0.0
                else:
                    # 无目标信息时的原有逻辑
                    smooth_score = max(min(cos_turn, 1.0), 0.0)
                
                smooth_reward = smooth_score
        except Exception:
            smooth_reward = 0.0
        # 更新上一帧速度，用于下一步计算转向角
        try:
            agent.last_velocity = vel.copy()
        except Exception:
            pass
        
        total = base_movement_reward + alignment_bonus
        if getattr(self, 'turn_smooth_weight', 0.0) != 0.0:
            total += float(self.turn_smooth_weight) * smooth_reward
        return float(total)
    
    def _calculate_deviation_reward(self, agent, world, dist_to_goal):
        """计算偏离奖励：奖励贴近起点-目标直线的行为（侧向偏离越小越好）"""
        # 🔧 改进：完全取消偏离奖励，直接返回0
        # 这样可以避免过度限制智能体的探索路径，允许更灵活的路径规划
        return 0.0
    
    def _calculate_start_area_reward(self, agent, world):
        """计算起始区域奖励"""
        _dstart = getattr(agent, '_rc_dstart', None)
        dist_to_start = _dstart if _dstart is not None else np.linalg.norm(agent.state.p_pos - agent.start_position)
        
        if dist_to_start < 20 and np.linalg.norm(agent.state.p_vel) > 0.1:
            return (20 - dist_to_start) * 0.5  # 越接近起点，奖励越大
        
        return 0.0
    
    def _calculate_approach_reward(self, agent, world, dist_to_goal):
        """计算接近目标奖励"""
        # 确保 last_goal_dist 存在
        if not hasattr(agent, 'last_goal_dist'):
            agent.last_goal_dist = dist_to_goal
        
        # 计算距离变化
        dist_change = agent.last_goal_dist - dist_to_goal
        # 供其它分项（如能量效率）复用：本步的距离变化，不在其它分项重复更新last_goal_dist
        try:
            agent._last_dist_change = float(dist_change)
        except Exception:
            pass
        # 更新上一次距离
        agent.last_goal_dist = dist_to_goal
        
        if dist_change > 0:  # 接近目标
            # 🔧 修复：进一步降低接近奖励系数（2.0→0.5），减少每步奖励累积
            # 原因：接近奖励每步都计算，持续接近目标会累积大量奖励，导致奖励值波动大
            # 新值0.5：降低75%，减少每步奖励，使奖励更平滑
            return dist_change * 0.5
        elif dist_change < -0.1:  # 明显远离目标
            # 关键修复：远离目标不再给予正奖励
            # - 之前为了“路径寻找”给了正奖励，导致策略学会远离目标
            # - 现在统一改为轻微惩罚；若确实需要探索，请通过EXPLORATION_WEIGHT控制
            retreat_distance = abs(dist_change)
            penalty = -retreat_distance * 1.0  # 远离目标给予线性惩罚
            return penalty
        
        return 0.0
    
    def _calculate_energy_reward(self, agent, world):
        """能量效率分项（按‘接近/远离’与高度区间自适应）：
        - 仅在“明显远离目标”时对能量消耗施加惩罚；
        - 当 dist_change>0（接近目标）且 height_diff < 理想区间上界时，给予能量效率奖励；
        - 其余情形弱化（或不计）能量项，避免误导策略朝Z向下。"""

        # 估算能量消耗：优先使用速度变化近似加速度，退化到速度幅值
        try:
            # 记录并使用上一步速度（若不存在，初始化为零）
            if not hasattr(agent, 'last_velocity'):
                agent.last_velocity = np.zeros(3)
            velocity_change = agent.state.p_vel - agent.last_velocity
            acc_mag = float(np.linalg.norm(velocity_change))
            energy_consumption = acc_mag * acc_mag * 0.1  # 与加速度平方成正比，匹配energy场景做法
        except Exception:
            energy_consumption = float(np.linalg.norm(getattr(agent.state, 'p_vel', np.zeros(3)))) * 0.1

        # 从 approach 分项读取最近一次距离变化（避免在此更新 last_goal_dist 与其竞态）
        dist_change = getattr(agent, '_last_dist_change', None)

        # 获取高度信息及理想上界
        height_ok = False
        height_diff = None
        ideal_max = None
        try:
            _th = getattr(agent, '_rc_th', None)
            if _th is not None:
                terrain_h = _th
            elif hasattr(self, 'get_terrain_height'):
                pos = agent.state.p_pos
                terrain_h = self.get_terrain_height(pos[0], pos[1])
            else:
                terrain_h = None
            if terrain_h is not None:
                height_diff = float(agent.state.p_pos[2] - terrain_h)
                ideal_max = float(getattr(self, 'height_ideal_max', 5.0))
                height_ok = (height_diff <= ideal_max)
        except Exception:
            pass

        # 无法判定接近/远离时，弱化能量项影响
        if dist_change is None:
            return 0.0

        # 明显远离：仅此时施加惩罚，且与远离幅度成比例
        if dist_change < -0.1:
            # 归一化“远离强度”，避免过大惩罚
            try:
                goal_pos = agent.goal_a.state.p_pos if hasattr(agent, 'goal_a') and agent.goal_a is not None else None
                d_now = float(np.linalg.norm(agent.state.p_pos - goal_pos)) if goal_pos is not None else 1.0
            except Exception:
                d_now = 1.0
            worsen_ratio = min(1.0, abs(dist_change) / max(1e-6, d_now))
            return float(-energy_consumption * 6.0 * worsen_ratio)  # 适中权重，避免盖过其他项

        # 接近目标：若高度不超过“理想区间上界”，按能量效率奖励
        if dist_change > 0.0 and (height_ok or ideal_max is None):
            speed = float(np.linalg.norm(getattr(agent.state, 'p_vel', np.zeros(3))))
            if energy_consumption > 1e-3:
                efficiency = min(speed / energy_consumption, 20.0)
                return float(min(efficiency * 0.08, 2.0))  # 上限限制，平滑贡献
            return 0.0

        # 其它情况：不计或弱化能量影响，避免误导（如高度过高还在接近）
        return 0.0
    
    def _calculate_height_reward(self, agent, world):
        """计算高度适应性奖励（可禁用/可配置理想高度范围）"""
        # 允许完全关闭该分项
        if not getattr(self, 'height_reward_enabled', True):
            return 0.0
        
        _th = getattr(agent, '_rc_th', None)
        if _th is not None or hasattr(self, 'get_terrain_height'):
            current_pos = agent.state.p_pos
            terrain_height = _th if _th is not None else self.get_terrain_height(current_pos[0], current_pos[1])
            height_diff = current_pos[2] - terrain_height
            
            # 可配置的理想高度：地形高度 + [min, max] 米
            ideal_min = float(getattr(self, 'height_ideal_min', 2.0))
            ideal_max = float(getattr(self, 'height_ideal_max', 5.0))
            if ideal_min > ideal_max:
                ideal_min, ideal_max = ideal_max, ideal_min
            
            # 计算奖励 - 渐进式高度控制策略
            # 🔧 修复：降低惩罚强度，避免高度奖励过度影响导航奖励
            if ideal_min <= height_diff <= ideal_max:
                reward = 1.0  # 在理想高度范围内
            elif height_diff < ideal_min:
                # 🔧 修复：降低低高度惩罚系数，避免过度惩罚
                height_shortage = ideal_min - height_diff
                
                # 🔧 修复：从3.0降低到1.5，使用线性惩罚而非平方，减少惩罚强度
                # 例如：短缺1米惩罚-1.5，短缺2米惩罚-3.0，短缺3米惩罚-4.5
                reward = -height_shortage * 1.5  # 🔧 修复：改为线性惩罚，从平方改为线性
                
                # 🔧 修复：降低危险高度惩罚（<3米时）
                if height_diff < 3.0:
                    danger_level = 3.0 - height_diff
                    reward -= danger_level * 3.0  # 🔧 修复：从平方惩罚10.0降低到线性惩罚3.0
                
                # 🔧 修复：降低地形穿透惩罚
                if height_diff < 0.0:
                    penetration_depth = -height_diff  # 穿透深度（正值）
                    # 🔧 修复：从平方惩罚50.0降低到线性惩罚15.0
                    reward -= penetration_depth * 15.0  # 🔧 修复：改为线性惩罚，降低强度
            else:
                reward = -(height_diff - ideal_max) * 0.1  # 🔧 修复：降低过高惩罚（从0.2降到0.1）
            
            # 调试输出（每10步输出一次，可通过 world.suppress_height_debug 关闭）
            try:
                suppress = bool(getattr(world, 'suppress_height_debug', True))
            except Exception:
                suppress = True
            if (not suppress) and hasattr(world, 'current_step') and world.current_step % 10 == 0:
                print(f"[高度奖励调试] step={world.current_step} | agent={getattr(agent,'name','?')} | pos=({current_pos[0]:.2f},{current_pos[1]:.2f},{current_pos[2]:.2f}) | terrain_h={terrain_height:.2f} | height_diff={height_diff:.2f} | ideal_range=[{ideal_min:.1f},{ideal_max:.1f}] | reward={reward:.3f}")
            
            return reward
        
        return 0.0

    def _calculate_success_reward(self, agent, world, dist_to_goal):
        """
        目标区奖励：首次进入目标区奖励 + 目标区保持奖励 + 离开目标区惩罚
        其中“成功”只按安全到达给一次性奖励；保持/离开用于让终止帧语义可学习。
        """
        try:
            success_state = self._get_success_state(agent)
            in_goal_zone = dist_to_goal <= float(self.success_distance_threshold)
            was_in_goal_zone = bool(success_state.get('was_in_goal_zone', False))
            safe_so_far = self._agent_safe_so_far(agent)

            if in_goal_zone:
                if not success_state['success_reward_given']:
                    agent_id = getattr(agent, 'id', None)
                    if agent_id is None:
                        agent_name = getattr(agent, 'name', '')
                        if 'agent_' in agent_name:
                            try:
                                agent_id = int(agent_name.split('_')[1])
                            except (ValueError, IndexError):
                                agent_id = 0
                        else:
                            agent_id = 0

                    current_step = getattr(world, 'current_step', 0)
                    if current_step == 0:
                        scenario_obj = getattr(world, 'scenario', None)
                        env_ref = getattr(scenario_obj, '_env_ref', None) if scenario_obj is not None else None
                        if env_ref is not None and hasattr(env_ref, '_current_step'):
                            try:
                                env_step = int(getattr(env_ref, '_current_step', 0))
                                if env_step > 0:
                                    current_step = env_step
                                    if hasattr(world, 'current_step'):
                                        world.current_step = env_step
                            except Exception:
                                pass
                    if current_step == 0:
                        current_step = 1

                    success_state['success_reward_given'] = True
                    success_state['first_success_step'] = current_step
                    success_state['hover_reward_count'] = 0

                    reward_val = float(self.success_reward_value) if safe_so_far else 0.0
                    episode = getattr(world, '_episode_count', getattr(world, 'episode_count', 0))
                    agent_pos = agent.state.p_pos
                    goal_pos = self._get_agent_goal_pos(agent)

                    safe_str = "Safe" if safe_so_far else "Unsafe"
                    print(f"[成功奖励] Agent{agent_id} 到达目标区 | 回合={episode} 步={current_step} | {safe_str} | 奖励={reward_val:.0f}")
                    if goal_pos is not None:
                        dist = np.linalg.norm(agent_pos - goal_pos)
                        print(f"  位置=({agent_pos[0]:.1f},{agent_pos[1]:.1f},{agent_pos[2]:.1f}) 目标距离={dist:.2f}m")

                    success_state['was_in_goal_zone'] = True
                    return reward_val

                if safe_so_far:
                    current_speed = np.linalg.norm(agent.state.p_vel)
                    if current_speed < self.hover_speed_threshold:
                        hold_reward = (1.0 - current_speed / self.hover_speed_threshold) * float(self.goal_hold_reward)
                        success_state['hover_reward_count'] += 1
                        success_state['was_in_goal_zone'] = True
                        if success_state['hover_reward_count'] % self.hover_reward_interval == 0:
                            return hold_reward
                        return 0.0
                    success_state['was_in_goal_zone'] = True
                    return -current_speed * 0.5

                success_state['was_in_goal_zone'] = True
                return 0.0
        except Exception as e:
            print(f"[SuccessReward] Error: {e}")
            pass

        try:
            success_state = self._get_success_state(agent)
            if success_state.get('was_in_goal_zone', False):
                success_state['was_in_goal_zone'] = False
                success_state['hover_reward_count'] = 0
                return -float(self.leave_goal_penalty)
            success_state['was_in_goal_zone'] = False
        except Exception:
            pass
        return 0.0

    def _calculate_collision_penalty(self, agent, world):
        """碰撞惩罚：接触障碍或越界给予强惩罚"""
        try:
            has_collision = False
            has_terrain_collision = False  # 本步是否因地形触发
            has_obstacle_collision = False  # 本步是否因球形障碍触发
            real_terrain_collision = False
            real_obstacle_collision = False
            penalty = 0.0
            eps = self._terrain_contact_eps
            terrain_h = getattr(agent, '_rc_th', None)
            dmin = getattr(agent, '_rc_dmin', None)

            try:
                if dmin is None and hasattr(world, 'nearest_obstacle_distance'):
                    dmin = world.nearest_obstacle_distance(agent)
                if dmin is None:
                    if terrain_h is None and hasattr(self, 'get_terrain_height'):
                        pos = agent.state.p_pos
                        terrain_h = self.get_terrain_height(pos[0], pos[1])
                    if terrain_h is not None:
                        dmin = max(agent.state.p_pos[2] - terrain_h, 1e-3)
                if dmin is not None and dmin < float(self.collision_distance_threshold):
                    has_obstacle_collision = True
                    has_collision = True
                    penalty = -float(self.collision_penalty_value)
                    if dmin <= 0.0:
                        real_obstacle_collision = True
                    if self._enable_collision_debug:
                        collision_count = getattr(agent, 'current_episode_collision_count', 0)
                        if collision_count == 0 or collision_count % 100 == 0:
                            print(f"[障碍物碰撞检测] Agent碰撞: dmin={dmin:.3f}, threshold={self.collision_distance_threshold}, 计数={collision_count}")
            except Exception:
                pass

            try:
                pos = agent.state.p_pos
                if terrain_h is None and hasattr(self, 'get_terrain_height'):
                    terrain_h = self.get_terrain_height(pos[0], pos[1])
                if terrain_h is not None and pos[2] <= terrain_h + eps:
                    base = float(self.collision_penalty_value)
                    has_terrain_collision = True
                    has_collision = True
                    penalty = -base
                    if pos[2] <= terrain_h + float(self._terrain_collision_eps):
                        real_terrain_collision = True
            except Exception:
                pass

            real_collision = real_terrain_collision or real_obstacle_collision
            if real_collision:
                agent._episode_has_collision = True
                agent._had_penetration_or_collision = True
                if real_terrain_collision:
                    agent._had_terrain_contact_or_penetration = True
                if real_obstacle_collision:
                    agent._had_obstacle_collision = True

                try:
                    current_step = world.current_step if hasattr(world, 'current_step') else -1
                except (AttributeError, TypeError):
                    current_step = -1

                try:
                    last_counted_step = agent._last_collision_counted_step if hasattr(agent, '_last_collision_counted_step') else -1
                except (AttributeError, TypeError):
                    last_counted_step = -1

                if current_step != last_counted_step:
                    if not hasattr(agent, 'current_episode_collision_count'):
                        agent.current_episode_collision_count = 0
                    agent.current_episode_collision_count += 1

                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    if not isinstance(agent.debug_info, dict):
                        agent.debug_info = {}
                    if 'total_penetration_count' not in agent.debug_info:
                        agent.debug_info['total_penetration_count'] = 0
                    if 'terrain_penetration_count' not in agent.debug_info:
                        agent.debug_info['terrain_penetration_count'] = 0
                    if 'obstacle_collision_count' not in agent.debug_info:
                        agent.debug_info['obstacle_collision_count'] = 0
                    old_count = agent.debug_info.get('total_penetration_count', 0)
                    agent.debug_info['total_penetration_count'] = old_count + 1
                    # 分项统计（仅显示用）：地形优先归因，若同时触发则计为地形
                    if real_terrain_collision:
                        agent.debug_info['terrain_penetration_count'] = agent.debug_info.get('terrain_penetration_count', 0) + 1
                    else:
                        agent.debug_info['obstacle_collision_count'] = agent.debug_info.get('obstacle_collision_count', 0) + 1
                    agent._last_collision_counted_step = current_step

                if self._enable_collision_debug:
                    new_count = agent.debug_info.get('total_penetration_count', 0) if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict) else 0
                    if new_count == 1 or new_count % 100 == 0:
                        try:
                            pos = agent.state.p_pos
                            th = terrain_h if terrain_h is not None else (self.get_terrain_height(pos[0], pos[1]) if hasattr(self, 'get_terrain_height') else None)
                            print(f"[碰撞检测] Agent碰撞触发: pos={pos}, terrain_h={th}, "
                                  f"height_diff={pos[2] - th if th is not None else 'N/A'}, "
                                  f"eps={eps}, dmin={dmin}, total_count={new_count}")
                        except Exception:
                            print(f"[碰撞检测] Agent碰撞触发: total_count={new_count}")

            agent._rc_collision = has_collision

            if has_collision and penalty < 0:
                cum_count = 0
                if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                    cum_count = agent.debug_info.get('total_penetration_count', 0)
                progressive = 1.0 + min(cum_count / 30.0, 3.0)
                penalty = penalty * progressive

            return penalty
        except Exception:
            pass
        return 0.0

    def update_arw_stats(self, episode_idx: int, success_flags: list, critic_losses: list = None):
        """
        更新ARW统计信息
        
        Args:
            episode_idx: 当前回合索引
            success_flags: 各智能体成功标志列表
            critic_losses: Critic loss列表
        """
        if self.arw:
            # 更新成功率（使用平均成功率）
            avg_success = np.mean(success_flags) if success_flags else False
            self.arw.update_success_rate(avg_success)
            
            # 更新Critic loss
            if critic_losses:
                for loss in critic_losses:
                    self.arw.update_critic_loss(loss)
            
            # 调整权重
            self.arw.adjust_weights(episode_idx)

    def get_arw_debug_info(self) -> dict:
        """
        获取ARW调试信息
        
        Returns:
            ARW调试信息字典
        """
        if self.arw:
            return self.arw.get_current_weights()
        return {}

    def reset_arw(self):
        """重置ARW状态"""
        if self.arw:
            self.arw.reset()

    # 统一各场景的"地形碰撞即终止（非目标附近落地）"规则
    def is_done(self, agent, world):
        try:
            try:
                early_stop_mode = getattr(self, '_early_stop_mode', 'never')
                if early_stop_mode in ('never', 'disabled'):
                    # 完全禁用早停，只记录不终止
                    try:
                        step_idx = int(getattr(world, 'current_step', -1))
                        ep_len = int(getattr(world, 'episode_length', -1))
                        # 只在开始时打印一次提示
                        if step_idx == 0 and hasattr(agent, '_early_stop_disabled_logged'):
                            pass
                        else:
                            if not hasattr(agent, '_early_stop_disabled_logged'):
                                print(f"[早停禁用] agent={getattr(agent,'name','?')} | 早停已禁用，智能体将运行完整回合")
                                agent._early_stop_disabled_logged = True
                    except Exception:
                        pass
                    return False
            except Exception:
                pass
            
            # 1. 首先检查严重越界（由environment标记的）
            if hasattr(agent, 'out_of_bounds_info') and agent.out_of_bounds_info.get('out_of_bounds', False):
                self._record_termination_reason(world, agent, "越界")
                try:
                    step_idx = int(getattr(world, 'current_step', -1))
                    ep_len = int(getattr(world, 'episode_length', -1))
                    reason = agent.out_of_bounds_info.get('reason', '未知原因')
                    pos = agent.out_of_bounds_info.get('position', agent.state.p_pos)
                    print(f"[终止] 严重越界 | step={step_idx+1}/{ep_len} | agent={getattr(agent,'name','?')} | {reason} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
                except Exception:
                    pass
                return True
            
            # 2. 起飞前保护：若仍在起始XY附近且未离地到阈值高度，则忽略地形接触/穿透判定，避免刚重置即早停
            try:
                start_pos = getattr(agent, 'start_position', None)
                if start_pos is not None:
                    dx = float(agent.state.p_pos[0] - start_pos[0])
                    dy = float(agent.state.p_pos[1] - start_pos[1])
                    dist_xy = (dx*dx + dy*dy) ** 0.5
                    # 从world读取可配置阈值，提供默认
                    start_radius = float(getattr(world, 'pre_takeoff_start_radius', 1.0))
                    airborne_thr = float(getattr(world, 'pre_takeoff_airborne_threshold', 0.5))
                    _th0 = getattr(agent, '_rc_th', None)
                    terrain_h0 = _th0 if _th0 is not None else self.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
                    hdiff = float(agent.state.p_pos[2]) - float(terrain_h0)
                    still_in_start = dist_xy <= start_radius
                    not_airborne = hdiff <= airborne_thr
                    if still_in_start and not_airborne:
                        # 若已实际穿透/接触地形，则不予豁免，直接早停
                        eps0 = float(self._terrain_collision_eps)
                        if float(agent.state.p_pos[2]) <= float(terrain_h0) + eps0:
                            self._record_termination_reason(world, agent, "地形穿透")
                            try:
                                step_idx = int(getattr(world, 'current_step', -1))
                                ep_len = int(getattr(world, 'episode_length', -1))
                                print(f"[终止] 起飞前穿透 | step={step_idx+1}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({agent.state.p_pos[0]:.2f},{agent.state.p_pos[1]:.2f},{agent.state.p_pos[2]:.2f}) | terrain_h={terrain_h0:.2f}")
                            except Exception:
                                pass
                            return True
                        # 否则在起飞前给予一次性豁免，避免刚重置即早停
                        try:
                            step_idx = int(getattr(world, 'current_step', -1))
                            if step_idx <= 5:  # 只在前5步打印保护信息
                                print(f"[保护] 起飞前豁免 | step={step_idx+1} | agent={getattr(agent,'name','?')} | dist_xy={dist_xy:.2f}/{start_radius} | hdiff={hdiff:.2f}/{airborne_thr}")
                        except Exception:
                            pass
                        return False
            except Exception:
                pass
            # 越界
            if hasattr(world, 'is_within_bounds'):
                pos = agent.state.p_pos
                if not world.is_within_bounds(pos):
                    self._record_termination_reason(world, agent, "越界")
                    try:
                        step_idx = int(getattr(world, 'current_step', -1))
                        ep_len = int(getattr(world, 'episode_length', -1))
                        print(f"[终止] 越界 | step={step_idx+1}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
                    except Exception:
                        pass
                    return True
            pos = agent.state.p_pos
            _th_done = getattr(agent, '_rc_th', None)
            terrain_h = _th_done if _th_done is not None else self.get_terrain_height(pos[0], pos[1])
            gx, gy, gz = float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2]) if hasattr(self, 'goal_pos') and self.goal_pos is not None else (0.0, 0.0, 0.0)
            dist_to_goal = np.linalg.norm(np.asarray(pos) - np.asarray([gx, gy, gz], dtype=np.float32)) if hasattr(self, 'goal_pos') and self.goal_pos is not None else 1e9
            eps = float(self._terrain_collision_eps)
            # 全局接触/穿透宽限期：在训练前K步内不因地形接触而终止，减少无意义早停
            try:
                cur_step_glob = int(getattr(world, 'current_step', 0))
                grace_steps = int(getattr(world, 'global_contact_grace_steps', 200))
            except Exception:
                cur_step_glob = 0
                grace_steps = 200
            if cur_step_glob < grace_steps:
                # 在宽限期内，仅记录但不终止（使用宽松阈值）
                if pos[2] <= terrain_h + eps:
                    try:
                        if cur_step_glob % 50 == 0:
                            ep_len = int(getattr(world, 'episode_length', -1))
                            print(f"[宽限] 地形接触 | step={cur_step_glob+1}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | terrain_h={terrain_h:.2f}")
                    except Exception:
                        pass
                    return False
            # 任何穿透地形均视为失败（不再要求远离目标）
            # 🔧 修复：严格判断穿透（pos[2] < terrain_h才是真正的穿透，但为了容忍数值误差使用<=）
            # 注意：这里使用<=是为了容忍数值误差，但严格来说应该是< terrain_h
            if pos[2] <= terrain_h + eps:
                self._record_termination_reason(world, agent, "地形穿透")
                try:
                    step_idx = int(getattr(world, 'current_step', -1))
                    ep_len = int(getattr(world, 'episode_length', -1))
                    # 显示为物理积分后的第 step+1 帧，避免首帧显示为0的困惑
                    print(f"[终止] 地形穿透 | step={step_idx+1}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | terrain_h={terrain_h:.2f}")
                except Exception:
                    pass
                return True
            # 障碍/实体碰撞（早于穿透的接触判定）：与任何地物的最近距离小于阈值
            try:
                dmin = None
                if hasattr(world, 'landmarks'):
                    for landmark in world.landmarks:
                        lp = getattr(getattr(landmark, 'state', None), 'p_pos', None)
                        if lp is None:
                            continue
                        r = float(getattr(landmark, 'size', 0.0)) + float(getattr(agent, 'size', 0.0))
                        d = float(np.linalg.norm(pos - lp) - r)
                        dmin = d if dmin is None else min(dmin, d)
                # 任何实体近距离碰撞均视为失败
                if dmin is not None and dmin <= 0.0:
                    self._record_termination_reason(world, agent, "实体碰撞")
                    try:
                        step_idx = int(getattr(world, 'current_step', -1))
                        ep_len = int(getattr(world, 'episode_length', -1))
                        print(f"[终止] 实体碰撞 | step={step_idx}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | min_dist={dmin:.3f}")
                    except Exception:
                        pass
                    return True
            except Exception:
                pass
        except Exception:
            return False
        return False

    def _calculate_global_reward(self, world):
        """全局奖励：一次性团队奖励，仅在所有智能体首次全部到达目标时触发。"""
        try:
            if getattr(world, '_global_reward_given', False):
                return 0.0

            # 步级缓存：同一步内只遍历一次 agents 列表
            cur_step = getattr(world, 'current_step', -1)
            cache = getattr(world, '_global_reward_step_cache', None)
            if cache is not None and cache[0] == cur_step:
                return cache[1]

            agents = getattr(world, 'agents', [])
            if not agents:
                world._global_reward_step_cache = (cur_step, 0.0)
                return 0.0

            all_reached = True
            all_safe = True
            for ag in agents:
                if not (hasattr(ag, 'goal_a') and ag.goal_a.state.p_pos is not None):
                    all_reached = False
                    break
                d = np.linalg.norm(ag.state.p_pos - ag.goal_a.state.p_pos)
                if d > float(self.success_distance_threshold):
                    all_reached = False
                    break
                # 统一“回合安全”语义：episode 内任意碰撞/穿透即不安全
                # 注意：global reward 不引入“末帧无风险(d_min_last)”约束，避免与回合成功/解锁口径混淆
                if not self._agent_safe_so_far(ag):
                    all_safe = False

            if not all_reached:
                world._global_reward_step_cache = (cur_step, 0.0)
                return 0.0

            world._global_reward_given = True
            result = float(self.success_reward_value) if all_safe else 0.0
            world._global_reward_step_cache = (cur_step, result)
            return result
        except Exception:
            pass
        return 0.0

    def _phi_potential(self, dist_to_goal):
        # 潜势函数 Φ(s) = - dist / D_max（这里用初始距离近似D_max）
        try:
            Dmax = getattr(self, 'initial_global_distance', None)
            if Dmax is None:
                # 初始化：取场景目标到第一个agent起点的平均距离
                try:
                    agents = getattr(self, 'world', None)
                    agents = getattr(agents, 'agents', [])
                    ds = []
                    for ag in agents:
                        if hasattr(ag, 'initial_distance_to_goal'):
                            ds.append(ag.initial_distance_to_goal)
                    if ds:
                        Dmax = float(np.mean(ds))
                        self.initial_global_distance = Dmax
                except Exception:
                    Dmax = None
            if Dmax and Dmax > 1e-6:
                return -float(dist_to_goal) / float(Dmax)
        except Exception:
            pass
        return -float(dist_to_goal) * 0.01

    def _calculate_potential_shaping(self, agent, dist_to_goal):
        """潜势函数shaping奖励：γΦ(s') − Φ(s)（近似：需要上一距离）"""
        try:
            if not hasattr(agent, '_phi_last'):
                agent._phi_last = self._phi_potential(dist_to_goal)
                return 0.0
            phi_now = self._phi_potential(dist_to_goal)
            gamma = float(getattr(self, 'shaping_gamma', 0.95))
            r_shape = gamma * phi_now - agent._phi_last
            agent._phi_last = phi_now
            return float(r_shape)
        except Exception:
            return 0.0
    
    def _calculate_lateral_reward(self, agent, world):
        """
        计算侧向/绕行奖励：增强版（3D法向量 + 宽容判定 + Z轴逃逸支持）
        
        实现逻辑：
        1. 获取最危险的障碍物或地形表面，计算其3D法向量（指向外/安全方向）。
        2. 计算速度方向与法向量的夹角余弦。
        3. 如果夹角 <= 90度（即速度没有指向障碍物内部），给予奖励。
           - 奖励切向运动（90度）和向外远离（<90度）。
           - 特别地，当在障碍物侧面或下方时，向上（Z+）拉升也是一种有效的"侧向"逃逸。
        """
        try:
            # 获取智能体位置和速度
            agent_pos = agent.state.p_pos
            agent_vel = agent.state.p_vel
            vel_norm = np.linalg.norm(agent_vel)
            
            # 如果速度太小，无法判断方向，返回0
            if vel_norm < 0.1:
                return 0.0
            
            vel_dir = agent_vel / vel_norm  # 归一化速度方向
            
            # 危险检测变量初始化
            min_dist = float('inf')
            danger_normal = None  # 危险表面的法向量（指向外）
            
            # === 1. 检测障碍物法向量 ===
            if hasattr(world, 'landmarks'):
                for landmark in world.landmarks:
                    if landmark is None or not hasattr(landmark, 'state') or landmark.state.p_pos is None:
                        continue
                    
                    obstacle_pos = landmark.state.p_pos
                    to_obstacle = obstacle_pos - agent_pos
                    dist_center = np.linalg.norm(to_obstacle)
                    
                    # 减去半径得到表面距离
                    radius = getattr(landmark, 'size', getattr(landmark, 'radius', 1.0))
                    dist_surface = max(0.0, dist_center - radius)
                    
                    if dist_surface < min_dist:
                        # 仅当在激活距离内才考虑
                        if dist_surface < self.lateral_activation_distance:
                            min_dist = dist_surface
                            # 法向量：从障碍物中心指向智能体（即指向外）
                            if dist_center > 1e-6:
                                danger_normal = -to_obstacle / dist_center
                            else:
                                danger_normal = np.array([0, 0, 1.0]) # 重合时假设向上
            
            # === 2. 检测3D地形法向量 ===
            _th = getattr(agent, '_rc_th', None)
            if _th is not None or hasattr(self, 'get_terrain_height'):
                try:
                    h0 = _th if _th is not None else self.get_terrain_height(agent_pos[0], agent_pos[1])
                    dist_terrain = max(0.0, agent_pos[2] - h0)
                    
                    if dist_terrain < min_dist and dist_terrain < self.lateral_activation_distance:
                        # 获取3D地形法向量
                        terrain_normal = self._get_terrain_normal(agent_pos[0], agent_pos[1])
                        if terrain_normal is not None:
                            min_dist = dist_terrain
                            danger_normal = terrain_normal
                except Exception:
                    pass
            
            # === 3. 计算奖励 ===
            # 如果没有检测到危险，返回0
            if danger_normal is None:
                return 0.0
            
            # 计算速度与法向量的夹角余弦
            # dot > 0: 夹角 < 90度 (远离或切向) -> 安全，给分
            # dot < 0: 夹角 > 90度 (靠近/撞击) -> 危险，不给分
            cos_angle = np.dot(vel_dir, danger_normal)
            
            if cos_angle > -0.05: # 稍微宽容一点点
                # 距离衰减：越近，奖励权重越大
                dist_factor = 1.0 - (min_dist / self.lateral_activation_distance)
                dist_factor = np.clip(dist_factor, 0.0, 1.0)
                
                # 奖励设计：
                # 0.3: 基础存活分（只要不撞）
                # 0.7 * cos_angle: 鼓励更大幅度的远离/爬升/绕行
                # 这样既奖励"擦肩而过"（cos~0），也奖励"调头离开"（cos~1）
                # 特别是向上拉升通常意味着 cos > 0，会得到高分
                lateral_reward = (0.3 + 0.7 * np.clip(cos_angle, 0.0, 1.0)) * dist_factor
                
                return float(lateral_reward)
            
            return 0.0
            
        except Exception as e:
            return 0.0

    def _get_terrain_normal(self, x, y):
        """
        计算指定位置的地形3D法向量
        使用中心差分法估算梯度
        """
        try:
            delta = 1.0  # 采样间距
            
            # 获取周围点高度
            h_x_plus = self.get_terrain_height(x + delta, y)
            h_x_minus = self.get_terrain_height(x - delta, y)
            h_y_plus = self.get_terrain_height(x, y + delta)
            h_y_minus = self.get_terrain_height(x, y - delta)
            
            # 计算梯度 (dz/dx, dz/dy)
            dz_dx = (h_x_plus - h_x_minus) / (2 * delta)
            dz_dy = (h_y_plus - h_y_minus) / (2 * delta)
            
            # 构造法向量 (-dz/dx, -dz/dy, 1)
            # 数学推导：切平面方程 z = f(x,y) -> f(x,y) - z = 0 -> grad = (df/dx, df/dy, -1)
            # 我们需要指向上方的法向量，取负 -> (-df/dx, -df/dy, 1)
            normal = np.array([-dz_dx, -dz_dy, 1.0])
            
            # 归一化
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6:
                return normal / norm_len
            else:
                return np.array([0, 0, 1.0])  # 默认向上
                
        except Exception:
            return np.array([0, 0, 1.0])
    
    def _calculate_clearance_reward(self, agent, world):
        """
        计算净空奖励：奖励智能体主动增加与障碍物和地形的距离
        
        实现逻辑：
        1. 计算当前智能体到所有障碍物和地形的最小距离
        2. 与上一步的最小距离比较
        3. 如果距离增加，说明成功"绕开"，给予正奖励
        4. 如果距离减少，说明在接近危险，给予负奖励或零奖励
        """
        try:
            agent_pos = agent.state.p_pos
            
            # === 1. 计算到所有障碍物的最小距离 ===
            min_obstacle_dist = float('inf')
            
            # 检查所有障碍物
            if hasattr(world, 'landmarks'):
                for landmark in world.landmarks:
                    if landmark is None or not hasattr(landmark, 'state') or landmark.state.p_pos is None:
                        continue
                    
                    obstacle_pos = landmark.state.p_pos
                    dist = np.linalg.norm(agent_pos - obstacle_pos)
                    
                    # 减去障碍物半径
                    if hasattr(landmark, 'size'):
                        dist = dist - landmark.size
                    
                    if dist < min_obstacle_dist:
                        min_obstacle_dist = dist
            
            # === 2. 计算到地形的最小安全距离 ===
            # 这里定义为：当前高度与地形高度的差值
            terrain_clearance = float('inf')
            
            _th = getattr(agent, '_rc_th', None)
            if _th is not None or hasattr(self, 'get_terrain_height'):
                try:
                    terrain_h = _th if _th is not None else self.get_terrain_height(agent_pos[0], agent_pos[1])
                    check_radius = 3.0
                    directions = [
                        [1, 0], [-1, 0], [0, 1], [0, -1],  # 四个主方向
                        [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]  # 四个对角线
                    ]
                    
                    max_nearby_terrain_h = terrain_h
                    for dx, dy in directions:
                        check_x = agent_pos[0] + dx * check_radius
                        check_y = agent_pos[1] + dy * check_radius
                        nearby_h = self.get_terrain_height(check_x, check_y)
                        max_nearby_terrain_h = max(max_nearby_terrain_h, nearby_h)
                    
                    # 地形净空 = 当前高度 - 周围最高地形高度
                    terrain_clearance = agent_pos[2] - max_nearby_terrain_h
                    
                    # 如果低于地形，净空为负（危险）
                    terrain_clearance = max(terrain_clearance, -5.0)  # 限制最小值
                    
                except Exception:
                    pass
            
            # === 3. 综合最小距离（取障碍物距离和地形净空的最小值） ===
            current_min_dist = min(min_obstacle_dist, max(0, terrain_clearance))
            
            # 限制最大值（避免过大的数值）
            current_min_dist = min(current_min_dist, self.clearance_d_max)
            
            # 🔧 修复：始终保存当前最小距离（无论是否第一次调用）
            # 保存当前最小距离到agent属性（用于数据收集）
            agent.last_min_distance = float(current_min_dist)
            
            # 同时保存到debug_info（用于数据收集）
            if not hasattr(agent, 'debug_info'):
                agent.debug_info = {}
            if not isinstance(agent.debug_info, dict):
                agent.debug_info = {}
            agent.debug_info['d_min_current'] = float(current_min_dist)
            
            # === 4. 与上一步比较 ===
            if not hasattr(agent, 'd_min_prev'):
                agent.d_min_prev = current_min_dist
                return 0.0  # 第一次没有奖励
            
            # 距离变化
            d_change = current_min_dist - agent.d_min_prev
            
            # === 5. 计算奖励 ===
            # 归一化到[-1, 1]范围
            clearance_reward = d_change / self.clearance_d_max
            
            # 正值：距离增加（成功绕开）→ 正奖励
            # 负值：距离减少（接近危险）→ 负奖励
            clearance_reward = np.clip(clearance_reward, -1.0, 1.0)
            
            # === 6. 更新上一步距离 ===
            agent.d_min_prev = current_min_dist
            
            return float(clearance_reward)
            
        except Exception as e:
            # 出错时返回0，不影响其他奖励
            return 0.0
    
    def _calculate_collision_reduction_reward(self, agent, world):
        """碰撞次数减少奖励（仅回合最后一步产生非零值）"""
        try:
            # 短路：非最后一步直接返回0（避免每步执行is_done和碰撞检测）
            current_step = int(getattr(world, 'current_step', 0))
            episode_length = int(getattr(world, 'episode_length', 2500))
            if current_step < episode_length - 1:
                return 0.0

            if not hasattr(agent, 'current_episode_collision_count'):
                agent.current_episode_collision_count = 0
            if not hasattr(agent, 'previous_episode_collision_count'):
                agent.previous_episode_collision_count = 0
            if not hasattr(agent, 'collision_reduction_reward_given'):
                agent.collision_reduction_reward_given = False

            if agent.collision_reduction_reward_given:
                return 0.0

            agent_is_done = False
            try:
                agent_is_done = self.is_done(agent, world)
            except Exception:
                try:
                    if hasattr(world, '_agent_done_flags'):
                        agent_key = getattr(agent, 'name', f'agent_{id(agent)}')
                        agent_is_done = world._agent_done_flags.get(agent_key, False)
                except Exception:
                    pass
            if agent_is_done:
                return 0.0

            prev_count = agent.previous_episode_collision_count
            curr_count = agent.current_episode_collision_count

            agent.collision_reduction_reward_given = True
            if prev_count > 0 and curr_count < prev_count:
                reduction = prev_count - curr_count
                return float(np.clip(float(reduction) / float(prev_count), 0.0, 1.0))
            elif prev_count == 0 and curr_count == 0:
                return 0.1
            return 0.0
        except Exception:
            return 0.0
    
    def update_reward_weights(self, new_weights):
        """更新奖励权重配置"""
        self.reward_weights.update(new_weights)
    
    def get_reward_weights(self):
        """获取当前奖励权重配置"""
        return self.reward_weights.copy()
    
    def set_reward_weights(self, weights):
        """设置奖励权重配置"""
        self.reward_weights = weights.copy()

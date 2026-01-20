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
        except Exception:
            self.hover_reward_max = 12.0  # 🔧 从5.0提高到12.0，增强悬停奖励
            self.hover_speed_threshold = 1.0
            self.hover_reward_interval = 5  # 🔧 从10改为5，增加悬停奖励频率
        
        # 侧向/净空奖励参数
        self.clearance_d_max = kwargs.get('clearance_d_max', 66.0)  # 最大可能距离（用于归一化）
        self.lateral_activation_distance = kwargs.get('lateral_activation_distance', 15.0)  # 激活侧向奖励的距离阈值
        self.terrain_gradient_threshold = kwargs.get('terrain_gradient_threshold', 0.5)  # 地形梯度阈值（米/米）

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
                agent.visited_cells = set()  # 用于记录已访问区域
                agent.debug_info = {}  # 添加调试信息字典
                agent.initialized_for_reward = True
                agent.start_position = agent.state.p_pos.copy()  # 记录起始位置
                # 🔧 新增：初始化碰撞次数跟踪
                agent.current_episode_collision_count = 0  # 当前回合碰撞次数
                agent.previous_episode_collision_count = 0  # 上一回合碰撞次数
                agent.collision_reduction_reward_given = False  # 标记是否已给予减少奖励
                # 🔧 修复：初始化debug_info中的total_penetration_count
                agent.debug_info['total_penetration_count'] = 0
                
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
                
                # 第一次计算默认值
                return 0.0  # 第一帧没有奖励
            
            # 计算各项独立奖励
            rewards = {
                'distance': self._calculate_distance_reward(agent, world, dist_to_goal),
                'exploration': self._calculate_exploration_reward(agent, world),
                'stationary': self._calculate_stationary_penalty(agent, world),
                'direction': self._calculate_direction_reward(agent, world),
                'deviation': self._calculate_deviation_reward(agent, world, dist_to_goal),
                'start_area': self._calculate_start_area_reward(agent, world),
                'approach': self._calculate_approach_reward(agent, world, dist_to_goal),
                'energy': self._calculate_energy_reward(agent, world),
                'height': self._calculate_height_reward(agent, world),
                'lateral': self._calculate_lateral_reward(agent, world),
                'clearance': self._calculate_clearance_reward(agent, world),
                'success': self._calculate_success_reward(agent, world, dist_to_goal),
                'collision': self._calculate_collision_penalty(agent, world),
                'collision_reduction': self._calculate_collision_reduction_reward(agent, world),  # 🔧 新增：碰撞次数减少奖励
                'global': self._calculate_global_reward(world),
                'shaping': self._calculate_potential_shaping(agent, dist_to_goal)
            }
            
            # 加权求和
            total_reward = sum(self.reward_weights[key] * rewards[key] for key in rewards.keys())
            total_before_clip = total_reward
            
            # 应用ARW调整
            if self.arw:
                # 创建奖励项字典用于ARW调整
                reward_terms = {
                    'collision': rewards.get('collision', 0.0),
                    'distance': rewards.get('distance', 0.0) + rewards.get('approach', 0.0)  # 合并距离相关奖励
                }
                
                # 应用ARW调整
                adjusted_terms = self.arw.apply_to_rewards(reward_terms)
                
                # 重新计算总奖励
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
                    'total_before_clip': total_before_clip
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
        if agent.stationary_count > 10:  # 提高触发阈值（2→10）
            dist_to_start = np.linalg.norm(agent.state.p_pos - agent.start_position)
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
        # 当前速度及基础移动奖励（鼓励"至少要动起来"）
        vel = getattr(agent.state, 'p_vel', np.zeros(3))
        speed = np.linalg.norm(vel)
        base_movement_reward = 0.0
        if speed > 0.05:  # 最低速度阈值
            base_movement_reward = min(speed * 2.0, 5.0)
        
        # 与起点→目标方向的一致性（保持原有逻辑）
        dist_to_start = np.linalg.norm(agent.state.p_pos - agent.start_position)
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
                    dist_to_goal = float(np.linalg.norm(agent.state.p_pos - goal_pos))
                    # 距离调节因子：远距离(>50m)允许大转向(权重0.3)，近距离(<15m)要求高平滑度(权重1.2)
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
        dist_to_start = np.linalg.norm(agent.state.p_pos - agent.start_position)
        
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
            if hasattr(self, 'get_terrain_height'):
                pos = agent.state.p_pos
                terrain_h = self.get_terrain_height(pos[0], pos[1])
                height_diff = float(pos[2] - terrain_h)
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
        
        if hasattr(self, 'get_terrain_height'):
            current_pos = agent.state.p_pos
            terrain_height = self.get_terrain_height(current_pos[0], current_pos[1])
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
        成功奖励：到达目标一次性大额奖励 + 悬停奖励
        修复重复成功奖励问题，鼓励智能体在终点处持续停留
        """
        try:
            if dist_to_goal <= float(self.success_distance_threshold):
                # 获取智能体唯一标识（确保使用正确的ID）
                agent_id = getattr(agent, 'id', None)
                if agent_id is None:
                    # 如果agent.id不存在，尝试从agent.name中提取
                    agent_name = getattr(agent, 'name', '')
                    if 'agent_' in agent_name:
                        try:
                            agent_id = int(agent_name.split('_')[1])
                        except (ValueError, IndexError):
                            agent_id = 0  # 默认值
                    else:
                        agent_id = 0  # 默认值
                
                # 初始化成功状态跟踪
                if not hasattr(agent, '_success_state'):
                    agent._success_state = {
                        'success_reward_given': False,
                        'first_success_step': None,
                        'hover_reward_count': 0
                    }
                
                success_state = agent._success_state
                
                # 一次性成功奖励（防重复）
                if not success_state['success_reward_given']:
                    # 🔧 修复：正确获取当前步数
                    # 注意：reward函数在step方法中被调用，此时步数应该已经更新（在world.step()之后）
                    # 优先从world.current_step获取（应该已经同步）
                    current_step = getattr(world, 'current_step', 0)
                    
                    # 🔧 如果world.current_step为0或不存在，尝试从环境对象获取
                    # 环境对象可能通过world.scenario._env_ref访问，或者通过其他方式
                    if current_step == 0:
                        # 方法1：尝试从场景对象获取环境引用（如果场景保存了环境引用）
                        scenario_obj = getattr(world, 'scenario', None)
                        if scenario_obj is not None:
                            # 检查场景是否有环境引用
                            env_ref = getattr(scenario_obj, '_env_ref', None)
                            if env_ref is not None and hasattr(env_ref, '_current_step'):
                                env_step = getattr(env_ref, '_current_step', 0)
                                if env_step > 0:
                                    current_step = env_step
                                    # 同步回world，确保一致性
                                    if hasattr(world, 'current_step'):
                                        world.current_step = int(current_step)
                    
                    # 🔧 关键修复：如果步数仍然为0，尝试从环境对象获取
                    # 注意：在评估时，world.current_step可能在某个时刻被重置或未同步
                    # 尝试从场景的环境引用获取步数
                    if current_step == 0:
                        # 方法2：尝试通过world.scenario._env_ref获取环境对象的_current_step
                        scenario_obj = getattr(world, 'scenario', None)
                        if scenario_obj is not None:
                            env_ref = getattr(scenario_obj, '_env_ref', None)
                            if env_ref is not None and hasattr(env_ref, '_current_step'):
                                env_step = getattr(env_ref, '_current_step', 0)
                                if env_step > 0:
                                    current_step = env_step
                                    # 同步回world，确保一致性
                                    if hasattr(world, 'current_step'):
                                        world.current_step = int(current_step)
                    
                    # 🔧 如果步数仍然为0，使用默认值1（不输出警告）
                    # 因为可能是正常情况（重置后的第一步就到达目标）
                    if current_step == 0:
                        current_step = 1  # 默认使用1（第一步）
                    
                    # 保存首次到达步数
                    success_state['success_reward_given'] = True
                    success_state['first_success_step'] = current_step
                    success_state['hover_reward_count'] = 0
                    
                    # 🔧 增强调试输出：显示智能体位置和目标位置，确保能够验证成功判断
                    # 🔧 修复：使用保存的首次到达步数，而不是当前步数
                    episode = getattr(world, '_episode_count', getattr(world, 'episode_count', 0))
                    agent_pos = agent.state.p_pos
                    goal_pos = agent.goal_a.state.p_pos if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') else None
                    
                    print(f"✅ [成功奖励] Agent{agent_id} 到达目标！")
                    print(f"   回合={episode} | 步数={success_state['first_success_step']} | 奖励值={self.success_reward_value:.1f}")
                    print(f"   智能体位置: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})")
                    if goal_pos is not None:
                        dist = np.linalg.norm(agent_pos - goal_pos)
                        print(f"   目标位置: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}) | 距离={dist:.2f}m")
                    
                    return float(self.success_reward_value)
                
                # 悬停奖励：鼓励稳定悬停
                current_speed = np.linalg.norm(agent.state.p_vel)
                hover_speed_threshold = self.hover_speed_threshold  # 🔧 使用配置的悬停速度阈值
                hover_reward_max = self.hover_reward_max  # 🔧 使用配置的最大悬停奖励
                
                if current_speed < hover_speed_threshold:
                    # 速度越低，奖励越高
                    hover_reward = (1.0 - current_speed / hover_speed_threshold) * hover_reward_max
                    success_state['hover_reward_count'] += 1
                    
                    # 🔧 限制悬停奖励频率，使用配置的间隔
                    if success_state['hover_reward_count'] % self.hover_reward_interval == 0:
                        return hover_reward
                    else:
                        return 0.0
                else:
                    # 在成功范围内但速度过快，给予轻微惩罚
                    speed_penalty = -current_speed * 0.5
                    return speed_penalty
        except Exception as e:
            print(f"[SuccessReward] Error: {e}")
            pass
        
        # 不在成功范围内：若曾到达过目标点，则开放悬停奖励（区外也按间隔发放）
        try:
            if hasattr(agent, '_success_state') and agent._success_state.get('success_reward_given', False):
                current_speed = np.linalg.norm(agent.state.p_vel)
                hover_speed_threshold = self.hover_speed_threshold
                hover_reward_max = self.hover_reward_max
                if current_speed < hover_speed_threshold:
                    hover_reward = (1.0 - current_speed / hover_speed_threshold) * hover_reward_max
                    agent._success_state['hover_reward_count'] += 1
                    if agent._success_state['hover_reward_count'] % self.hover_reward_interval == 0:
                        return hover_reward
        except Exception:
            pass
        return 0.0

    def _calculate_collision_penalty(self, agent, world):
        """碰撞惩罚：接触障碍或越界给予强惩罚"""
        try:
            has_collision = False
            penalty = 0.0
            
            # 🚨 关键修复：先获取TERRAIN_CONTACT_EPS，确保在异常情况下也能使用
            try:
                import os
                eps = float(os.getenv('TERRAIN_CONTACT_EPS', '0.3'))  # 🔧 统一阈值：从0.75降低到0.3米，与shell脚本保持一致
            except Exception:
                eps = 0.75  # 回退到默认值0.75米（合理的接触阈值）
            
            # 🚨 关键修复：检查障碍物碰撞（使用world.nearest_obstacle_distance）
            # 这是主要的碰撞检测逻辑，必须正确触发
            dmin = None
            try:
                if hasattr(world, 'nearest_obstacle_distance'):
                    dmin = world.nearest_obstacle_distance(agent)
                # 如果world没有提供障碍物距离，基于地形高度计算
                if dmin is None and hasattr(self, 'get_terrain_height'):
                    pos = agent.state.p_pos
                    terrain_h = self.get_terrain_height(pos[0], pos[1])
                    # 距地过近也记为潜在碰撞
                    dmin = max(pos[2] - terrain_h, 1e-3)
                # 🚨 关键修复：地形贴近/接触惩罚（必须正确触发）
                if dmin is not None and dmin < float(self.collision_distance_threshold):
                    has_collision = True
                    penalty = -float(self.collision_penalty_value)
                    # 🔧 调试：输出碰撞检测信息（仅在非安静模式下）
                    import os
                    if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                        print(f"[障碍物碰撞检测] Agent碰撞: dmin={dmin:.3f}, threshold={self.collision_distance_threshold}")
            except Exception as e:
                # 🚨 关键修复：即使异常也要记录，帮助诊断问题
                import os
                if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                    print(f"[警告] 障碍物碰撞检测异常: {e}")
                pass

            # 非目标点附近的地形"落地/穿透" -> 直接给大额负值（统一规则）
            # 🚨 关键修复：将地形穿透检测移出内层try-except，确保has_collision能被正确设置
            try:
                # 🚨 关键：严格按照当前坐标位置判断穿透
                pos = agent.state.p_pos
                # 🔧 修复：使用当前坐标位置获取地形高度
                terrain_h = self.get_terrain_height(pos[0], pos[1])
                # 🚨 关键：使用更大的eps（2.5米）可以检测到侧面穿透，即使Z坐标没有完全低于地形高度
                # 🔧 修复：严格判断穿透（pos[2] <= terrain_h + eps表示接近或穿透地形）
                if pos[2] <= terrain_h + eps:
                    # 固定值惩罚：避免随剩余步数放大导致奖励饱和
                    base = float(self.collision_penalty_value)
                    has_collision = True
                    penalty = -base
            except Exception as e:
                # 🚨 关键修复：即使异常也要记录，帮助诊断问题
                import os
                if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                    print(f"[警告] 地形穿透检测异常: {e}")
                pass
            
            # 🔧 新增：更新碰撞计数（用于碰撞减少奖励和统计）
            # 🚨 关键修复：确保在has_collision为True时，无论penalty是否被设置，都要更新计数
            # 🚨 关键修复：添加防重复计数机制，确保每步只计数一次
            if has_collision:
                # 获取当前步数，用于防重复计数
                # 🔧 性能优化：直接访问属性，避免getattr开销
                try:
                    current_step = world.current_step if hasattr(world, 'current_step') else -1
                except (AttributeError, TypeError):
                    current_step = -1
                
                try:
                    last_counted_step = agent._last_collision_counted_step if hasattr(agent, '_last_collision_counted_step') else -1
                except (AttributeError, TypeError):
                    last_counted_step = -1
                
                # 🚨 关键修复：只有在当前步数不同时才计数，防止同一步重复计数
                if current_step != last_counted_step:
                    if not hasattr(agent, 'current_episode_collision_count'):
                        agent.current_episode_collision_count = 0
                    agent.current_episode_collision_count += 1
                    
                    # 🔧 修复：同时更新debug_info中的total_penetration_count（用于数据收集）
                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    if not isinstance(agent.debug_info, dict):
                        agent.debug_info = {}
                    # 🚨 关键修复：确保total_penetration_count被正确初始化
                    if 'total_penetration_count' not in agent.debug_info:
                        agent.debug_info['total_penetration_count'] = 0
                    old_count = agent.debug_info.get('total_penetration_count', 0)
                    agent.debug_info['total_penetration_count'] = old_count + 1
                    
                    # 标记当前步已计数
                    agent._last_collision_counted_step = current_step
                
                # 🚨 调试：输出每次碰撞检测的详细信息，帮助诊断问题
                import os
                if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                    try:
                        pos = agent.state.p_pos
                        terrain_h = self.get_terrain_height(pos[0], pos[1]) if hasattr(self, 'get_terrain_height') else None
                        print(f"[碰撞检测] Agent碰撞触发: pos={pos}, terrain_h={terrain_h}, "
                              f"height_diff={pos[2] - terrain_h if terrain_h is not None else 'N/A'}, "
                              f"eps={eps}, dmin={dmin}, "
                              f"计数更新: {old_count} -> {agent.debug_info['total_penetration_count']}")
                    except Exception:
                        print(f"[碰撞检测] Agent碰撞触发: 计数更新: {old_count} -> {agent.debug_info['total_penetration_count']}")
            
            return penalty
        except Exception as e:
            # 🚨 关键修复：即使外层异常也要记录，帮助诊断问题
            import os
            if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                print(f"[警告] _calculate_collision_penalty异常: {e}")
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
            # 🔧 检查早停模式：如果设置为 never 或 disabled，则完全禁用终止条件
            try:
                import os
                early_stop_mode = os.getenv('EARLY_STOP_MODE', 'never').lower()
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
                    terrain_h0 = self.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
                    hdiff = float(agent.state.p_pos[2]) - float(terrain_h0)
                    still_in_start = dist_xy <= start_radius
                    not_airborne = hdiff <= airborne_thr
                    if still_in_start and not_airborne:
                        # 若已实际穿透/接触地形，则不予豁免，直接早停
                        eps0 = 0.03
                        if float(agent.state.p_pos[2]) <= float(terrain_h0) + eps0:
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
                    try:
                        step_idx = int(getattr(world, 'current_step', -1))
                        ep_len = int(getattr(world, 'episode_length', -1))
                        print(f"[终止] 越界 | step={step_idx+1}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
                    except Exception:
                        pass
                    return True
            # 地形穿透/落地并且不在目标附近
            # 🚨 关键：严格按照当前坐标位置判断穿透
            pos = agent.state.p_pos
            # 🔧 修复：使用当前坐标位置获取地形高度
            terrain_h = self.get_terrain_height(pos[0], pos[1])
            gx, gy, gz = float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2]) if hasattr(self, 'goal_pos') and self.goal_pos is not None else (0.0, 0.0, 0.0)
            dist_to_goal = np.linalg.norm(np.asarray(pos) - np.asarray([gx, gy, gz], dtype=np.float32)) if hasattr(self, 'goal_pos') and self.goal_pos is not None else 1e9
            # 🔧 修复：使用与碰撞检测一致的阈值（0.03），但is_done中使用稍大的阈值（0.15）以容忍更多误差
            # 注意：is_done中的阈值可以稍大，因为这是终止条件，需要更宽松以避免误判
            eps = 0.15  # 放宽地形穿透判定阈值，从8cm增加到15cm（用于is_done终止判断）
            # 🚨 关键修复：使用TERRAIN_CONTACT_EPS环境变量，与_calculate_collision_penalty保持一致
            try:
                import os
                eps_strict = float(os.getenv('TERRAIN_CONTACT_EPS', '0.75'))  # 🚨 修复：使用环境变量（默认0.75米），而不是硬编码0.03
            except Exception:
                eps_strict = 0.75  # 回退到默认值0.75米
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
                if dmin is not None and dmin <= float(self.collision_distance_threshold):
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
        """全局奖励：基于全体智能体的全局指标"""
        try:
            agents = getattr(world, 'agents', [])
            # 仅当每个agent有goal_a时启用
            dists = []
            progresses = []
            successes = []
            for ag in agents:
                if hasattr(ag, 'goal_a') and ag.goal_a.state.p_pos is not None:
                    d = np.linalg.norm(ag.state.p_pos - ag.goal_a.state.p_pos)
                    dists.append(d)
                    if hasattr(ag, 'last_goal_dist'):
                        progresses.append(max(0.0, ag.last_goal_dist - d))
                    successes.append(1.0 if d <= float(self.success_distance_threshold) else 0.0)
            if not dists:
                return 0.0
            mode = getattr(self, 'global_reward_mode', 'success_rate')
            if mode == 'avg_progress' and progresses:
                return float(np.mean(progresses)) * 10.0
            if mode == 'min_distance':
                return float(-np.min(dists))  # 越小越好
            if mode == 'success_rate' and successes:
                return float(np.mean(successes)) * float(self.success_reward_value)
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
            
            # === 2. 检测3D地形法向量（新增） ===
            # 使用辅助函数获取地形法向量
            if hasattr(self, 'get_terrain_height'):
                try:
                    # 检查当前高度与地形距离
                    h0 = self.get_terrain_height(agent_pos[0], agent_pos[1])
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
            
            if hasattr(self, 'get_terrain_height'):
                try:
                    terrain_h = self.get_terrain_height(agent_pos[0], agent_pos[1])
                    # 检查周围8个方向的地形高度变化
                    check_radius = 3.0  # 检查半径3米
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
        """
        🔧 新增：碰撞次数减少奖励
        如果当前回合的碰撞次数比上一回合少，给予奖励
        奖励值 = (上一回合碰撞次数 - 当前回合碰撞次数) / max(上一回合碰撞次数, 1)
        """
        try:
            # 确保属性已初始化
            if not hasattr(agent, 'current_episode_collision_count'):
                agent.current_episode_collision_count = 0
            if not hasattr(agent, 'previous_episode_collision_count'):
                agent.previous_episode_collision_count = 0
            if not hasattr(agent, 'collision_reduction_reward_given'):
                agent.collision_reduction_reward_given = False
            
            # 🚨 关键修复：检查智能体是否已经done，如果已经done则不应该计算这个奖励
            # 注意：必须在碰撞检测之前检查done状态，避免done后仍然更新碰撞计数
            agent_is_done = False
            try:
                # 方法1：直接调用is_done函数检查（最可靠的方法）
                agent_is_done = self.is_done(agent, world)
            except Exception:
                # 方法2：如果is_done调用失败，尝试通过world的done标志检查（如果可用）
                try:
                    if hasattr(world, '_agent_done_flags'):
                        agent_key = getattr(agent, 'name', f'agent_{id(agent)}')
                        agent_is_done = world._agent_done_flags.get(agent_key, False)
                except Exception:
                    pass
            
            # 如果智能体已经done，不应该计算这个奖励（早停后不应该有奖励）
            # 🔧 关键：早停的智能体不应该在done之后还获得无碰撞奖励
            if agent_is_done:
                return 0.0
            
            # 检测当前步是否发生碰撞（复用碰撞惩罚检测逻辑）
            # 🔧 修复：使用与_calculate_collision_penalty一致的阈值和判断逻辑
            has_collision = False
            try:
                # 检查地形碰撞：使用当前坐标位置严格判断
                pos = agent.state.p_pos
                if hasattr(self, 'get_terrain_height'):
                    # 🚨 关键：严格按照当前坐标位置获取地形高度
                    terrain_h = self.get_terrain_height(pos[0], pos[1])
                    # 🚨 关键修复：使用TERRAIN_CONTACT_EPS环境变量，与_calculate_collision_penalty保持一致
                    try:
                        import os
                        eps = float(os.getenv('TERRAIN_CONTACT_EPS', '0.75'))  # 🚨 修复：使用环境变量（默认0.75米），而不是硬编码0.1
                    except Exception:
                        eps = 0.75  # 回退到默认值0.75米
                    # 🔧 修复：严格判断穿透（pos[2] < terrain_h才是真正的穿透，但为了容忍数值误差使用<=）
                    # 注意：这里使用<=是为了容忍数值误差，但严格来说应该是< terrain_h
                    if pos[2] <= terrain_h + eps:
                        has_collision = True
                
                # 检查障碍物碰撞
                if not has_collision:
                    dmin = None
                    if hasattr(world, 'nearest_obstacle_distance'):
                        dmin = world.nearest_obstacle_distance(agent)
                    if dmin is None and hasattr(self, 'get_terrain_height'):
                        # 使用当前坐标位置计算距离
                        dmin = max(pos[2] - terrain_h, 1e-3)
                    if dmin is not None and dmin < float(self.collision_distance_threshold):
                        has_collision = True
            except Exception:
                pass
            
            # 如果发生碰撞，增加当前回合碰撞计数
            if has_collision:
                agent.current_episode_collision_count += 1
            
            # 注意：done检查已提前到函数开头，这里不再重复检查
            # 问题：当启用早停时，智能体可能在current_step < episode_length - 1时就done了
            # 但如果环境继续运行到current_step >= episode_length - 1，这个奖励仍然会被触发
            # 解决方案：在计算奖励前检查智能体是否已经done
            agent_is_done = False
            try:
                # 方法1：直接调用is_done函数检查（最可靠的方法）
                agent_is_done = self.is_done(agent, world)
            except Exception:
                # 方法2：如果is_done调用失败，尝试通过world的done标志检查（如果可用）
                try:
                    if hasattr(world, '_agent_done_flags'):
                        agent_key = getattr(agent, 'name', f'agent_{id(agent)}')
                        agent_is_done = world._agent_done_flags.get(agent_key, False)
                except Exception:
                    pass
            
            # 如果智能体已经done，不应该计算这个奖励（早停后不应该有奖励）
            # 🔧 关键：早停的智能体不应该在done之后还获得无碰撞奖励
            if agent_is_done:
                return 0.0
            
            # 在回合结束时（通过检查world.current_step）计算减少奖励
            # 注意：这里在每步都检查，但只在回合结束时给予一次奖励
            try:
                current_step = int(getattr(world, 'current_step', 0))
                episode_length = int(getattr(world, 'episode_length', 2500))
                
                # 🔧 修复：改为检查是否到达回合结束（current_step >= episode_length - 1）
                # 但前提是智能体还没有done（早停情况下，done的智能体不应该获得这个奖励）
                if current_step >= episode_length - 1 and not agent.collision_reduction_reward_given:
                    # 比较当前回合和上一回合的碰撞次数
                    prev_count = agent.previous_episode_collision_count
                    curr_count = agent.current_episode_collision_count
                    
                    if prev_count > 0 and curr_count < prev_count:
                        # 碰撞次数减少，给予奖励
                        reduction = prev_count - curr_count
                        # 归一化奖励：减少的碰撞数 / 上一回合碰撞数
                        reward = float(reduction) / float(prev_count)
                        # 限制在[0, 1]范围
                        reward = np.clip(reward, 0.0, 1.0)
                        agent.collision_reduction_reward_given = True
                        return reward
                    elif prev_count == 0 and curr_count == 0:
                        # 两回合都没有碰撞，给予小奖励（保持无碰撞状态）
                        reward = 0.1
                        agent.collision_reduction_reward_given = True
                        return reward
                    else:
                        # 碰撞次数未减少或增加，无奖励
                        agent.collision_reduction_reward_given = True
                        return 0.0
            except Exception:
                pass
            
            return 0.0
            
        except Exception as e:
            # 出错时返回0，不影响其他奖励
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

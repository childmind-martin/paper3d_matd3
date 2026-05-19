try:
    # 首先尝试导入 gymnasium
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import EnvSpec
except ImportError:
    # 如果 gymnasium 不可用，尝试导入 gym
    import gym
    from gym import spaces
    from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import copy
import pygame
import sys
import os  # 🚀 性能优化：用于QUIET_OUTPUT检查
import time
# 为3D渲染添加必要的导入
from pygame.locals import *
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("警告: 未能导入 OpenGL 模块，3D 渲染功能将不可用")

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=False):

        self.world = world
        self.agents = self.world.policy_agents
        # 保存场景对象引用
        if hasattr(world, 'scenario'):
            self.scenario = world.scenario
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback if observation_callback is not None else lambda agent, world: []
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback
        # environment parameters
        # 使用传入的 discrete_action 参数配置动作空间类型，避免被硬编码为离散导致策略网络维度错误
        self.discrete_action_space = bool(discrete_action)
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # 添加这一行来保存render_callback属性
        self.render_callback = None

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.obs_dim = None  # 保存观察维度，确保系统一致性
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(1) if not hasattr(world, 'dim_c') or world.dim_c <= 0 else spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            if self.observation_callback is not None:
                obs_dim = len(self.observation_callback(agent, self.world))
                # 保存观察维度，确保系统一致性
                if self.obs_dim is None:
                    self.obs_dim = obs_dim
                elif self.obs_dim != obs_dim:
                    print(f"警告: 智能体观察维度不一致! 期望{self.obs_dim}, 得到{obs_dim}")
                    self.obs_dim = obs_dim  # 使用最新的维度
                self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering - 使用字典而非列表
        self.shared_viewer = shared_viewer
        self.viewers = {}  # 改为字典初始化
        self._reset_render()

        # 记录每个智能体的完成状态，避免重复判定/重复打印（使用name作为键）
        self._agent_done_flags = {}
        self._agent_done_steps = {}  # 记录智能体完成的步数
        self._timing_detail_enabled_cache = None
        self._last_step_timing = None
        self._refresh_runtime_flags()

    def _refresh_runtime_flags(self):
        """缓存热路径中会频繁读取的运行时标志。"""
        try:
            self._quiet_output_enabled = os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self._quiet_output_enabled = True
        try:
            self._disable_trajectory_recording = os.getenv('DISABLE_TRAJECTORY_RECORDING', '0').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self._disable_trajectory_recording = False
        try:
            self._debug_episode_summary_enabled = os.getenv('DEBUG_EPISODE_SUMMARY', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self._debug_episode_summary_enabled = True
        try:
            self._debug_collision_summary_enabled = os.getenv('DEBUG_COLLISION_SUMMARY', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self._debug_collision_summary_enabled = True
        try:
            light_default = os.getenv('EVAL_LIGHT_MODE', '0')
            self._eval_light_action_path = os.getenv('EVAL_LIGHT_ACTION_PATH', light_default).lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self._eval_light_action_path = False
        try:
            light_default = os.getenv('EVAL_LIGHT_MODE', '0')
            self._eval_light_info = os.getenv('EVAL_LIGHT_INFO', light_default).lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self._eval_light_info = False

    def _timing_detail_enabled(self):
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

    def _policy_episode_collision_reasons(self, agent):
        """与训练侧 Safe_i 判定一致的真实碰撞/穿透原因列表（空表示安全）。"""
        reasons = []
        try:
            if getattr(agent, '_episode_has_collision', False):
                reasons.append("_episode_has_collision")
        except Exception:
            pass
        try:
            if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                penetration_count = agent.debug_info.get('total_penetration_count', 0)
                try:
                    penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                except (ValueError, TypeError, OverflowError):
                    penetration_count = 0
                if penetration_count > 0:
                    reasons.append(f"穿透计数={penetration_count}")
        except Exception:
            pass
        try:
            if getattr(agent, '_had_obstacle_collision', False):
                reasons.append("_had_obstacle_collision")
        except Exception:
            pass
        try:
            if getattr(agent, '_had_terrain_contact_or_penetration', False):
                reasons.append("_had_terrain_contact_or_penetration")
        except Exception:
            pass
        return reasons

    def _policy_agent_unsafe_for_episode_success(self, agent):
        return len(self._policy_episode_collision_reasons(agent)) > 0

    def _sync_world_team_success_snapshot(self):
        """每步末更新 world 上团队成功权威快照（仅 policy_agents，与回合层判定同源），供训练侧日志读取。"""
        w = self.world
        agents = list(self.agents) if self.agents is not None else []
        scn = getattr(self, 'scenario', None)
        thr = float(getattr(scn, 'success_distance_threshold', 2.0)) if scn is not None else 2.0
        reach = []
        safe = []
        succ = []
        for idx, agent in enumerate(agents):
            goal_pos = None
            if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                goal_pos = agent.goal_a.state.p_pos
            elif scn is not None and getattr(scn, 'goal_pos', None) is not None:
                goal_pos = scn.goal_pos
            if goal_pos is None:
                raise RuntimeError(f"success snapshot同步失败: agent[{idx}] 缺少goal_pos")
            pos = getattr(getattr(agent, 'state', None), 'p_pos', None)
            if pos is None or len(pos) < 3:
                raise RuntimeError(f"success snapshot同步失败: agent[{idx}] 缺少有效位置")
            dist = float(np.linalg.norm(pos - goal_pos))
            r_flag = 1 if dist <= thr else 0
            reach.append(r_flag)
            s_flag = 0 if self._policy_agent_unsafe_for_episode_success(agent) else 1
            safe.append(s_flag)
            succ.append(1 if (r_flag and s_flag) else 0)
        team = 1 if (succ and all(v == 1 for v in succ)) else 0
        w._episode_agent_reach_flags = reach
        w._episode_agent_safe_flags = safe
        w._episode_agent_success_flags = succ
        w._episode_team_success_flag = team
        w._episode_all_reached = bool(reach and all(v == 1 for v in reach))
        w._episode_success = bool(team == 1)
        w._episode_success_thr_snapshot = thr

    def step(self, action_n):
        # 修改签名以符合gym标准
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        timing_detail_enabled = self._timing_detail_enabled()
        _perf_counter = time.perf_counter if timing_detail_enabled else None
        step_timing = None
        if timing_detail_enabled:
            step_timing = {
                'env_set': 0.0,
                'env_world': 0.0,
                'env_pre': 0.0,
                'env_traj': 0.0,
                'env_obs': 0.0,
                'env_reward': 0.0,
                'env_done': 0.0,
                'env_info': 0.0,
                'env_post': 0.0,
                'env_outer': 0.0,
            }
        self._last_step_timing = None
        
        # 设置动作（对已完成的智能体，强制零动作，避免无意义动态与重复碰撞）
        if timing_detail_enabled:
            _t_env_seg = _perf_counter()
        for i, agent in enumerate(self.agents):
            # 避免索引越界
            if i < len(action_n):
                action = action_n[i]
                # 获取智能体标识符（优先使用name，回退到索引）
                agent_key = getattr(agent, 'name', f'agent_{i}')
                # 若该智能体此前已完成，则忽略外部动作，直接施加零动作
                if self._agent_done_flags.get(agent_key, False):
                    try:
                        if hasattr(agent, 'action') and hasattr(agent.action, 'u'):
                            agent.action.u = np.zeros_like(agent.action.u)
                    except Exception:
                        pass
                else:
                    self._set_action(action, agent, self.action_space[i])
            else:
                # 如果动作数量不足，使用默认动作（零向量或静止）
                # 改进对动作形状的检测，处理 MultiDiscrete 情况
                action_shape = 0
                if hasattr(self.action_space[i], 'shape'):
                    if isinstance(self.action_space[i].shape, int):
                        # MultiDiscrete 的 shape 是整数
                        action_shape = self.action_space[i].shape
                    elif hasattr(self.action_space[i].shape, '__getitem__'):
                        # 正常的 shape 元组/列表
                        action_shape = self.action_space[i].shape[0]
                default_action = np.zeros(action_shape) if action_shape > 0 else 0
                self._set_action(default_action, agent, self.action_space[i])
        if timing_detail_enabled:
            step_timing['env_set'] += _perf_counter() - _t_env_seg
                
        # 执行世界动态更新
        if timing_detail_enabled:
            _t_env_seg = _perf_counter()
        self.world.step()
        if timing_detail_enabled:
            step_timing['env_world'] += _perf_counter() - _t_env_seg
        
        # 立即更新步数计数器（在is_done检查之前）
        if timing_detail_enabled:
            _t_env_seg = _perf_counter()
        try:
            if hasattr(self, '_current_step'):
                self._current_step += 1
            else:
                self._current_step = 1
            # 同步到world
            if hasattr(self.world, 'current_step'):
                self.world.current_step = int(self._current_step)
        except Exception:
            pass

        # 起飞前安全钳制（仅在回合最初1步内，且仍在起始区域内生效）：
        # 目的：避免刚重置因数值或初始化误差导致的瞬时穿透而被判早停；
        # 一旦离开起始区或已离地，或过了首步，则不再钳制，若向下穿透则按规则早停。
        try:
            cur_step = int(getattr(self.world, 'current_step', 0))
            if cur_step <= 20 and hasattr(self, 'scenario') and hasattr(self.scenario, 'get_terrain_height'):
                start_radius = float(getattr(self.world, 'pre_takeoff_start_radius', 1.0))
                airborne_thr = float(getattr(self.world, 'pre_takeoff_airborne_threshold', 0.5))
                for agent in self.agents:
                    if not hasattr(agent, 'state') or not hasattr(agent.state, 'p_pos'):
                        continue
                    # 判定是否仍在起始区域
                    start_pos = getattr(agent, 'start_position', agent.state.p_pos)
                    # 🔧 已彻底删除：起飞前强制Z提升机制（复位机制）
                    # 原因：干扰重力模拟，导致智能体无法学习真实物理行为
                    pass
        except Exception:
            pass
        if timing_detail_enabled:
            step_timing['env_pre'] += _perf_counter() - _t_env_seg
        
        # 记录智能体位置到轨迹属性中（可通过环境变量禁用以提升性能）
        if timing_detail_enabled:
            _t_env_seg = _perf_counter()
        if not self._disable_trajectory_recording:
            for i, agent in enumerate(self.agents):
                if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                    # 🔧 修复：如果智能体已经done，不再记录轨迹（避免记录下落帧）
                    agent_key = getattr(agent, 'name', f'agent_{i}')
                    if self._agent_done_flags.get(agent_key, False):
                        continue  # 跳过已完成的智能体
                    
                    # 如果智能体没有轨迹属性，初始化它
                    if not hasattr(agent, '_trajectory'):
                        agent._trajectory = []
                    
                    # 🔧 修复：确保记录的是修正后的位置
                    position_copy = agent.state.p_pos.copy()
                    
                    # 🔧 已彻底删除：轨迹记录时的二次Z修正机制（复位机制）
                    # 原因：干扰重力模拟，应该记录真实位置而不是修正后的位置
                    pass
                    
                    # 记录当前位置到轨迹中
                    agent._trajectory.append(position_copy)
                    
                    # 增强调试输出（已禁用，减少开销）
                    # if i < 3:  # 只输出前3个智能体的轨迹信息
                    #     trajectory_id = id(agent._trajectory)  # 获取轨迹对象的唯一ID
                    #     position_id = id(position_copy)  # 获取位置对象的唯一ID
                    #    # print(f"DEBUG: 步骤轨迹 - 智能体{i}位置: {position_copy}, 轨迹点数: {len(agent._trajectory)}, 轨迹ID: {trajectory_id}, 位置对象ID: {position_id}")
        if timing_detail_enabled:
            step_timing['env_traj'] += _perf_counter() - _t_env_seg
        
        # 记录最新的观察、奖励、是否完成和信息
        # 使用异常处理确保即使某个智能体的回调失败，也能返回完整数据
        agent_count = len(self.agents)

        # observation 在当前主场景里本来就是 step 级批量计算 + cache，
        # 这里直接一次性取整批，避免再为每个 agent 走一遍回调包装。
        if timing_detail_enabled:
            _t_obs_batch_seg = _perf_counter()
        try:
            obs_batch = self._get_obs_batch(self.agents)
        except Exception:
            obs_batch = [np.zeros(self._get_default_obs_dim(), dtype=np.float32) for _ in range(agent_count)]
        if timing_detail_enabled:
            step_timing['env_obs'] += _perf_counter() - _t_obs_batch_seg

        # 🚀 性能优化：批量处理智能体数据，减少循环开销
        # 使用try-except确保总是返回与智能体数量匹配的值
        try:
            # 批量收集所有智能体的数据（减少循环开销）
            agent_count = len(self.agents)
            obs_list = []
            reward_list = []
            done_list = []
            info_list = []
            
            # 预先获取常用属性，避免重复查找
            reward_pos_scale = float(getattr(self.world, 'reward_pos_scale', 1.0))
            reward_neg_scale = float(getattr(self.world, 'reward_neg_scale', 1.0))
            enable_collision_autoreset = getattr(self.world, 'enable_collision_autoreset', False)
            default_obs_dim = self._get_default_obs_dim()
            reward_batch_values = None

            reward_owner = getattr(self.reward_callback, '__self__', None) if self.reward_callback is not None else None
            if reward_owner is not None and hasattr(reward_owner, '_compute_batch_rewards'):
                if timing_detail_enabled:
                    _t_reward_batch_seg = _perf_counter()
                try:
                    current_step = int(getattr(self.world, 'current_step', -1))
                    world_id = id(self.world)
                    cache_key = (world_id, current_step)
                    if hasattr(reward_owner, '_ensure_world_reward_initialized'):
                        reward_owner._ensure_world_reward_initialized(self.world)
                    reward_batch = reward_owner._compute_batch_rewards(
                        [self.agents],
                        [self.world],
                        cache_key=cache_key,
                    )
                    if (
                        isinstance(reward_batch, np.ndarray)
                        and reward_batch.ndim == 2
                        and reward_batch.shape[0] >= 1
                        and reward_batch.shape[1] >= agent_count
                    ):
                        reward_batch_values = np.asarray(reward_batch[0], dtype=np.float32)
                except Exception:
                    reward_batch_values = None
                if timing_detail_enabled:
                    step_timing['env_reward'] += _perf_counter() - _t_reward_batch_seg
            
            # 批量处理所有智能体
            for i, agent in enumerate(self.agents):
                # 观察值（已由 batch helper 预取）
                try:
                    if i < len(obs_batch):
                        obs = obs_batch[i]
                    else:
                        obs = np.zeros(default_obs_dim, dtype=np.float32)
                    obs_list.append(obs)
                except Exception as e:
                    if not self._quiet_output_enabled:
                        print(f"智能体 {i} 观察值计算错误: {e}")
                    obs_list.append(np.zeros(default_obs_dim, dtype=np.float32))
                
                # 奖励值
                if reward_batch_values is None and timing_detail_enabled:
                    _t_reward_seg = _perf_counter()
                try:
                    if reward_batch_values is not None and i < reward_batch_values.shape[0]:
                        r = float(reward_batch_values[i])
                    else:
                        r = self._get_reward(agent)
                    # 应用正负奖励缩放（使用预获取的值）
                    if r >= 0:
                        r = r * reward_pos_scale
                    else:
                        r = r * reward_neg_scale
                    reward_list.append(r)
                except Exception as e:
                    if not self._quiet_output_enabled:
                        print(f"智能体 {i} 奖励计算错误: {e}")
                    reward_list.append(0.0)
                if reward_batch_values is None and timing_detail_enabled:
                    step_timing['env_reward'] += _perf_counter() - _t_reward_seg
                
                # Done状态
                if timing_detail_enabled:
                    _t_done_seg = _perf_counter()
                try:
                    agent_key = getattr(agent, 'name', f'agent_{i}')
                    if self._agent_done_flags.get(agent_key, False):
                        done_list.append(True)
                    else:
                        d = self._get_done(agent)
                        if d:
                            current_step = int(getattr(self, '_current_step', 0))
                            self._agent_done_flags[agent_key] = True
                            self._agent_done_steps[agent_key] = current_step
                            
                            if enable_collision_autoreset:
                                # 🔧 关键修复：智能体完成时，修正轨迹的最后一个点
                                if hasattr(agent, '_trajectory') and agent._trajectory:
                                    try:
                                        if hasattr(self.scenario, 'get_terrain_height_cached'):
                                            # 🚀 性能优化：使用缓存版本的地形高度查询
                                            terrain_h = self.scenario.get_terrain_height_cached(agent.state.p_pos[0], agent.state.p_pos[1])
                                            corrected_pos = agent.state.p_pos.copy()
                                            corrected_pos[2] = terrain_h
                                            agent._trajectory[-1] = corrected_pos
                                            if not self._quiet_output_enabled:
                                                print(f"[轨迹修正] {agent_key}: 修正终点 Z={agent.state.p_pos[2]:.2f} -> {terrain_h:.2f}")
                                    except Exception as e:
                                        if not self._quiet_output_enabled:
                                            print(f"[轨迹修正失败] {agent_key}: {e}")
                            # 统一惩罚：早停触发时确保奖励中包含碰撞惩罚
                            try:
                                termination_reasons = []
                                if hasattr(self, '_termination_reasons'):
                                    termination_reasons = getattr(self, '_termination_reasons', {}).get(agent_key, [])
                                penalty_triggers = {'地形穿透', '实体碰撞', '越界'}
                                if any(reason in penalty_triggers for reason in termination_reasons):
                                    base_penalty = float(getattr(self.scenario, 'collision_penalty_value', getattr(self.world, 'collision_penalty_value', 30.0)))
                                    collision_weight = None
                                    try:
                                        if hasattr(self.scenario, 'reward_weights') and isinstance(self.scenario.reward_weights, dict):
                                            collision_weight = self.scenario.reward_weights.get('collision', None)
                                    except Exception:
                                        collision_weight = None
                                    if collision_weight is None:
                                        collision_weight = getattr(self.world, 'collision_weight', 1.0)
                                    try:
                                        collision_weight = float(collision_weight)
                                    except Exception:
                                        collision_weight = 1.0
                                    total_penalty = base_penalty * max(collision_weight, 0.0)
                                    if total_penalty != 0.0 and i < len(reward_list):
                                        try:
                                            current_reward = reward_list[i]
                                            if current_reward > -0.5 * total_penalty:
                                                reward_list[i] = current_reward - total_penalty
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        done_list.append(d)
                except Exception as e:
                    if not self._quiet_output_enabled:
                        print(f"智能体 {i} 完成状态计算错误: {e}")
                    done_list.append(False)
                if timing_detail_enabled:
                    step_timing['env_done'] += _perf_counter() - _t_done_seg
                
                # 信息
                if timing_detail_enabled:
                    _t_info_seg = _perf_counter()
                try:
                    if self._eval_light_info:
                        base_info = {}
                    else:
                        base_info = self._get_info(agent)
                        if not isinstance(base_info, dict):
                            base_info = {}
                        # 附加动作通道调试：返回原始网络动作与环境最终施加的连续动作（前三维力）
                        try:
                            raw = None if self._eval_light_action_path else (agent.current_action if hasattr(agent, 'current_action') else None)
                            applied = agent.action.u if hasattr(agent, 'action') and hasattr(agent.action, 'u') else None
                            if isinstance(raw, np.ndarray):
                                # 🚀 性能优化：只在需要时才copy
                                base_info['raw_action'] = raw.copy() if raw.size < 100 else raw  # 小数组才copy
                            if isinstance(applied, np.ndarray):
                                base_info['applied_force3'] = applied[:3].copy()
                        except Exception:
                            pass
                    info_list.append(base_info)
                except Exception as e:
                    if not self._quiet_output_enabled:
                        print(f"智能体 {i} 信息计算错误: {e}")
                    info_list.append({})
                if timing_detail_enabled:
                    step_timing['env_info'] += _perf_counter() - _t_info_seg

            if timing_detail_enabled and hasattr(self.scenario, 'get_last_reward_timing'):
                try:
                    reward_detail = self.scenario.get_last_reward_timing(self.world)
                    if isinstance(reward_detail, dict):
                        for key, value in reward_detail.items():
                            try:
                                step_timing[key] = float(value)
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # 批量赋值（减少append开销）
            obs_n = obs_list
            reward_n = reward_list
            done_n = done_list
            info_n['n'] = info_list
            if timing_detail_enabled:
                _t_env_post_seg = _perf_counter()
            
            # 验证长度与智能体数量是否一致
            if len(obs_n) != agent_count:
                print(f"警告: 观察值数量 ({len(obs_n)}) 与智能体数量 ({agent_count}) 不匹配!")
                # 修复不匹配问题
                if len(obs_n) > agent_count:
                    # 如果观察值过多，截断
                    obs_n = obs_n[:agent_count]
                else:
                    # 如果观察值不足，填充
                    obs_dim = self._get_default_obs_dim()
                    if len(obs_n) > 0:
                        obs_dim = len(obs_n[0])
                    while len(obs_n) < agent_count:
                        obs_n.append(np.zeros(obs_dim))
            
            # 确保其他返回值数量也匹配
            while len(reward_n) < agent_count:
                reward_n.append(0.0)
            while len(done_n) < agent_count:
                done_n.append(False)
            while len(info_n['n']) < agent_count:
                info_n['n'].append({})
                
            # 裁剪超出的值
            if len(reward_n) > agent_count:
                reward_n = reward_n[:agent_count]
            if len(done_n) > agent_count:
                done_n = done_n[:agent_count]
            if len(info_n['n']) > agent_count:
                info_n['n'] = info_n['n'][:agent_count]
            
            # 🚨 关键修改：检查所有智能体是否都到达目标
            # - 如果所有智能体都到达：立即终止回合进入下一回合（确保数据记录正确）
            # - 如果只有部分智能体到达：继续运行，保持悬停奖励（让已到达的智能体继续获得悬停奖励）
            try:
                if hasattr(self, 'scenario') and hasattr(self.scenario, 'success_distance_threshold'):
                    thr_succ = getattr(self.scenario, 'success_distance_threshold', 2.0)
                    # 统一阈值：回合层不再额外放宽到 1.2 倍，保持与 reward/训练一致
                    thr_succ_actual = thr_succ
                    all_reached = True
                    reached_agents = []
                    agent_distances = []  # 用于调试输出
                    
                    for i, agent in enumerate(self.agents):
                        try:
                            pos = agent.state.p_pos
                            # 获取目标位置（优先每智能体目标，其次全局目标）
                            goal_pos = None
                            if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                                goal_pos = agent.goal_a.state.p_pos
                            elif hasattr(self.scenario, 'goal_pos') and self.scenario.goal_pos is not None:
                                goal_pos = self.scenario.goal_pos
                            
                            if goal_pos is not None:
                                dist_to_goal = np.linalg.norm(pos - goal_pos)
                                agent_distances.append((i, dist_to_goal, thr_succ_actual))
                                if dist_to_goal <= thr_succ_actual:
                                    reached_agents.append(i)
                                    # 记录“本回合曾到达”，供训练侧在回合末按 ∃t 语义统计 Reach_i
                                    try:
                                        agent._ever_reached_goal = True
                                    except Exception:
                                        pass
                                else:
                                    all_reached = False
                            else:
                                all_reached = False
                                agent_distances.append((i, None, None))
                        except Exception as e:
                            all_reached = False
                            agent_distances.append((i, None, f"Error: {e}"))
                    
                    # 🚨 关键修复：统一成功判断逻辑，明确成功定义
                    # 成功定义：所有智能体都到达目标 AND 所有智能体都无碰撞
                    # 到达目标定义：所有智能体都到达目标（距离 <= 阈值），不管是否有碰撞
                    # 修复：区分"到达目标"和"成功"，使用与训练脚本一致的碰撞检查逻辑
                    if all_reached and len(reached_agents) == agent_count:
                        # 🚨 关键修复：检查所有智能体是否都无碰撞（与训练脚本逻辑一致）
                        all_no_collision = True
                        collision_info = []
                        try:
                            for i, agent in enumerate(self.agents):
                                collision_reasons = self._policy_episode_collision_reasons(agent)
                                if collision_reasons:
                                    all_no_collision = False
                                    collision_info.append(f"Agent{i}: {', '.join(collision_reasons)}")
                        except Exception as e:
                            # 如果检查失败，保守地认为有碰撞
                            all_no_collision = False
                            collision_info.append(f"碰撞检查失败: {e}")
                        
                        # 只在首次达到时打印一次
                        if not hasattr(self.world, '_all_agents_reached_logged'):
                            self.world._all_agents_reached_logged = False
                        
                        if not self.world._all_agents_reached_logged:
                            self.world._all_agents_reached_logged = True
                            try:
                                if self._debug_episode_summary_enabled:
                                    step_idx = int(getattr(self.world, 'current_step', -1))
                                    ep_len = int(getattr(self.world, 'episode_length', -1))
                                    # 🔧 添加详细调试信息
                                    dist_info = ", ".join([f"Agent{i}: {d:.2f}m" if d is not None else f"Agent{i}: N/A" for i, d, _ in agent_distances])
                                    
                                    # 🚨 关键修复：明确区分"到达目标"和"成功"
                                    if all_no_collision:
                                        # 成功：所有智能体都到达目标且无碰撞
                                        print(f"[成功] 所有智能体都到达目标且无碰撞 | step={step_idx}/{ep_len} | 立即终止回合进入下一回合")
                                        print(f"  [调试] 到达的智能体: {reached_agents}, 阈值={thr_succ_actual:.2f}m, 距离信息: {dist_info}")
                                    else:
                                        # 到达目标但可能有碰撞：只说明到达目标，不说明成功
                                        print(f"[到达目标] 所有智能体都到达目标（但可能有碰撞） | step={step_idx}/{ep_len} | 立即终止回合进入下一回合")
                                        print(f"  [调试] 到达的智能体: {reached_agents}, 阈值={thr_succ_actual:.2f}m, 距离信息: {dist_info}")
                                        if collision_info:
                                            print(f"  [碰撞信息] {', '.join(collision_info)}")
                            except Exception as e:
                                if self._debug_episode_summary_enabled:
                                    print(f"[到达目标] 所有智能体都到达目标，但调试输出失败: {e}")
                        
                        # 🚨 关键修复：在world中记录成功状态（只有无碰撞才算成功）
                        # 原因：统一成功定义，确保与训练脚本一致
                        if not hasattr(self.world, '_episode_success'):
                            self.world._episode_success = False
                        # 只有所有智能体都到达目标且无碰撞时，才记录为成功
                        self.world._episode_success = all_no_collision
                        # 全员到达标志（与安全成功解耦）：用于终局 unsafe_arrival_penalty 与日志分析
                        try:
                            self.world._episode_all_reached = True
                        except Exception:
                            pass
                        
                        # 🚨 关键：设置所有智能体的done为True，立即终止回合
                        # 确保数据记录正确（episode_rewards等会正确记录）
                        for i in range(len(done_n)):
                            if i < len(done_n):
                                done_n[i] = True
                                # 标记智能体已完成
                                if i < len(self.agents):
                                    agent_key = getattr(self.agents[i], 'name', f'agent_{i}')
                                    self._agent_done_flags[agent_key] = True
                                    self._agent_done_steps[agent_key] = int(getattr(self.world, 'current_step', -1))
                    elif len(reached_agents) > 0 and len(reached_agents) < agent_count:
                        # 部分智能体到达：继续运行，保持悬停奖励
                        # 悬停奖励逻辑在paper3d_terrain_weighted.py中已实现，无需修改
                        # 🔧 添加调试输出（每500步输出一次，降低输出频率）
                        try:
                            step_idx = int(getattr(self.world, 'current_step', -1))
                            if step_idx % 500 == 0 and not self._quiet_output_enabled:  # 每500步输出一次
                                dist_info = ", ".join([f"Agent{i}: {d:.2f}m" if d is not None else f"Agent{i}: N/A" for i, d, _ in agent_distances])
                                print(f"[部分到达] step={step_idx} | 到达的智能体: {reached_agents}/{agent_count} | 阈值={thr_succ_actual:.2f}m | 距离: {dist_info}")
                        except Exception:
                            pass
                    else:
                        # 没有智能体到达：每500步输出一次调试信息（降低输出频率）
                        try:
                            step_idx = int(getattr(self.world, 'current_step', -1))
                            if step_idx % 500 == 0 and not self._quiet_output_enabled:  # 每500步输出一次
                                dist_info = ", ".join([f"Agent{i}: {d:.2f}m" if d is not None else f"Agent{i}: N/A" for i, d, _ in agent_distances])
                                print(f"[未到达] step={step_idx} | 到达的智能体: {reached_agents}/{agent_count} | 阈值={thr_succ_actual:.2f}m | 距离: {dist_info}")
                        except Exception:
                            pass
            except Exception as e:
                # 🔧 关键修复：输出异常信息，帮助诊断问题
                import traceback
                print(f"[错误] 检查所有智能体到达目标时发生异常: {e}")
                if not self._quiet_output_enabled:
                    traceback.print_exc()
                pass
                
        except Exception as e:
            print(f"步骤执行期间发生严重错误: {e}")
            # 提供完整的默认返回值
            default_obs_dim = self._get_default_obs_dim()
            obs_n = [np.zeros(default_obs_dim) for _ in range(agent_count)]
            reward_n = [0.0 for _ in range(agent_count)]
            done_n = [False for _ in range(agent_count)]
            info_n = {'n': [{} for _ in range(agent_count)]}
            if timing_detail_enabled:
                _t_env_post_seg = _perf_counter()
            
        # 权威团队成功快照（policy_agents + scenario 成功阈值），与 [成功记录] 同源
        self._sync_world_team_success_snapshot()

        # 更新3D视图中的实体位置（如果正在使用3D渲染）
        if timing_detail_enabled and '_t_env_post_seg' not in locals():
            _t_env_post_seg = _perf_counter()
        self._update_3d_objects()
        if timing_detail_enabled:
            step_timing['env_post'] += _perf_counter() - _t_env_post_seg
            if not isinstance(info_n, dict):
                info_n = {'n': [{} for _ in range(len(done_n) if hasattr(done_n, '__len__') else 0)]}
            info_n['_timing'] = step_timing
            self._last_step_timing = step_timing
        
        # 检查是否使用的是 gymnasium (通过检查是否有 step 方法的特定参数)
        if hasattr(gym.Env, 'step') and gym.Env.step.__code__.co_varnames.count('truncated') > 0:
            # 使用新版 API 返回 (observations, rewards, terminated, truncated, info)
            terminated_n = done_n  # 在这个环境中，done 等同于 terminated
            truncated_n = [False] * len(done_n)  # 这个环境不使用 truncated
            return obs_n, reward_n, terminated_n, truncated_n, info_n
        else:
            # 使用旧版 API 返回 (observations, rewards, done, info)
            return obs_n, reward_n, done_n, info_n

    def reset(self, seed=None, options=None):
        """
        重置环境，并返回初始观察。
        如果使用 gymnasium 则符合新 API，如果使用 gym 则提供向下兼容。
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            如果使用 gymnasium: (obs_n, info)
            如果使用 gym: obs_n
        """
        # 设置随机种子（如果提供）
        if seed is not None:
            np.random.seed(seed)

        # 在 reset_callback 之前推进 episode 索引，使场景可在 reset_world 内按 episode_idx 决定地形/障碍实例。
        try:
            episode_index = int(getattr(self.world, '_episode_index_counter', 0))
        except Exception:
            episode_index = 0
        try:
            self.world.episode_index = int(episode_index)
            self.world._episode_index_counter = int(episode_index) + 1
        except Exception:
            pass
            
        # reset world
        if self.reset_callback is not None:
            self.reset_callback(self.world)
        self._refresh_runtime_flags()
        # reset renderer
        self._reset_render()
        # 清空完成标记和步数记录
        self._agent_done_flags = {}
        self._agent_done_steps = {}
        
        # 重置步数计数器
        self._current_step = 0
        
        # 🔧 修复：同步到world，确保vectorized_reward_calculator能正确检测新回合
        # 问题：如果不同步，reward计算可能使用上一回合的步数，导致success_reward_given不会重置
        if hasattr(self.world, 'current_step'):
            self.world.current_step = 0
        
        # 🔧 重置成功状态标志（所有智能体到达目标）
        if hasattr(self.world, '_all_agents_reached_logged'):
            self.world._all_agents_reached_logged = False
        if hasattr(self.world, '_episode_success'):
            self.world._episode_success = False
        # 终局结算辅助标志：全员到达（不要求安全）与 episode 级最小安全距离
        try:
            self.world._episode_all_reached = False
        except Exception:
            pass
        try:
            self.world._episode_dmin_min = float('inf')
        except Exception:
            pass
        try:
            self.world._episode_agent_reach_flags = []
            self.world._episode_agent_safe_flags = []
            self.world._episode_agent_success_flags = []
            self.world._episode_team_success_flag = 0
            if hasattr(self.world, '_episode_success_thr_snapshot'):
                delattr(self.world, '_episode_success_thr_snapshot')
        except Exception:
            pass
        
        # 保存智能体引用
        self.agents = self.world.policy_agents
        
        # 初始化智能体轨迹
        for i, agent in enumerate(self.agents):
            # 重置“本回合曾到达目标”标志，避免跨回合污染
            try:
                agent._ever_reached_goal = False
            except Exception:
                pass
            if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                # 重置轨迹
                agent._trajectory = [agent.state.p_pos.copy()]
                
                # 不再输出调试信息，减少输出量
                # if i < 3:  # 只输出前3个智能体的轨迹信息
                #     print(f"DEBUG: 重置轨迹 - 智能体{i}初始位置: {agent.state.p_pos}, 轨迹列表ID: {id(agent._trajectory)}")
        
        # record observations for each agent
        obs_n = []

        try:
            obs_n = self._get_obs_batch(self.agents)

            # 最终验证：确保观察值数量与智能体数量匹配
            if len(obs_n) != len(self.agents):
                if len(obs_n) > len(self.agents):
                    obs_n = obs_n[:len(self.agents)]
                else:
                    default_obs_dim = self._get_default_obs_dim()
                    while len(obs_n) < len(self.agents):
                        obs_n.append(np.zeros(default_obs_dim, dtype=np.float32))
        except Exception as e:
            print(f"重置环境期间发生错误: {e}")
            # 确保始终返回与智能体数量匹配的观察值
            default_obs_dim = self._get_default_obs_dim()
            obs_n = [np.zeros(default_obs_dim, dtype=np.float32) for _ in range(len(self.agents))]
        
        # 添加调试信息（已禁用以减少输出）
        # print(f"DEBUG: environment.py reset() - 准备返回观察值")
        # print(f"DEBUG: obs_n类型: {type(obs_n)}, 长度: {len(obs_n)}")
        # print(f"DEBUG: 智能体数量: {len(self.agents)}")
        # for i, obs in enumerate(obs_n):
        #     print(f"DEBUG: 智能体{i}观察值形状: {obs.shape if hasattr(obs, 'shape') else 'no_shape'}")
        
        # 检查是否使用的是 gymnasium (通过检查是否有特定方法)
        if hasattr(gym.Env, 'reset') and 'options' in gym.Env.reset.__code__.co_varnames:
            # gymnasium API 返回 (observations, info)
            info = {}
            # print(f"DEBUG: 检测到gymnasium API，返回(obs_n, info)元组")
            return obs_n, info
        else:
            # 旧版 gym API 只返回 observations
            # print(f"DEBUG: 检测到旧版gym API，直接返回obs_n")
            return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def _try_fast_obs_value(self, obs, expected_dim=None):
        """批量 observation 热路径的零拷贝快路径。"""
        if not isinstance(obs, np.ndarray):
            return None
        if obs.dtype != np.float32 or obs.ndim != 1:
            return None
        if expected_dim is not None and int(obs.size) != int(expected_dim):
            return None
        if obs.flags['C_CONTIGUOUS']:
            return obs
        return np.ascontiguousarray(obs, dtype=np.float32)

    def _normalize_obs_value(self, obs):
        """统一 observation 的形状与副本语义，供单 agent 与 batch 路径复用。"""
        default_obs_dim = self._get_default_obs_dim()

        if not isinstance(obs, np.ndarray):
            try:
                if isinstance(obs, list) and len(obs) > 0:
                    if isinstance(obs[0], np.ndarray):
                        obs = obs[0]
                    else:
                        obs = np.array(obs)
                else:
                    obs = np.array(obs)
            except Exception:
                return np.zeros(default_obs_dim, dtype=np.float32)

        try:
            obs = np.asarray(obs)
            if obs.ndim == 0:
                obs = obs.reshape(1)
            elif obs.ndim > 1:
                obs = obs.flatten()
        except Exception:
            return np.zeros(default_obs_dim, dtype=np.float32)

        actual_dim = int(obs.size)
        if hasattr(self, 'obs_dim') and self.obs_dim is not None:
            expected_dim = int(self.obs_dim)
        else:
            expected_dim = actual_dim
            self.obs_dim = actual_dim

        fast_obs = self._try_fast_obs_value(obs, expected_dim)
        if fast_obs is not None:
            return fast_obs

        if actual_dim != expected_dim:
            if actual_dim > expected_dim:
                obs = np.ascontiguousarray(obs[:expected_dim], dtype=np.float32)
            else:
                padded_obs = np.zeros(expected_dim, dtype=np.float32)
                padded_obs[:actual_dim] = obs
                obs = padded_obs
        else:
            if obs.dtype != np.float32:
                obs = obs.astype(np.float32, copy=False)
            if not obs.flags['C_CONTIGUOUS']:
                obs = np.ascontiguousarray(obs, dtype=np.float32)

        return obs

    def _get_obs_batch(self, agents=None):
        """优先命中场景已有的批量 observation 语义，失败时回退到逐 agent。"""
        if agents is None:
            agents = self.agents
        if self.observation_callback is None:
            return [np.zeros(0, dtype=np.float32) for _ in agents]
        default_obs_dim = self._get_default_obs_dim()

        callback_owner = getattr(self.observation_callback, '__self__', None)
        if callback_owner is not None and hasattr(callback_owner, '_compute_observations_batch_uncached'):
            try:
                cache_key = (id(self.world), int(getattr(self.world, 'current_step', -1)))
            except Exception:
                cache_key = None

            try:
                if cache_key is not None and getattr(callback_owner, '_obs_step_cache_key', None) == cache_key:
                    cache = getattr(callback_owner, '_obs_step_cache', None)
                    if isinstance(cache, dict):
                        obs_n = []
                        complete = True
                        for agent in agents:
                            cached_obs = cache.get(id(agent))
                            if cached_obs is None:
                                complete = False
                                break
                            fast_obs = self._try_fast_obs_value(cached_obs, default_obs_dim)
                            obs_n.append(fast_obs if fast_obs is not None else self._normalize_obs_value(cached_obs))
                        if complete and len(obs_n) == len(agents):
                            return obs_n

                batch_obs = callback_owner._compute_observations_batch_uncached(self.world)
                if isinstance(batch_obs, dict) and batch_obs:
                    if cache_key is not None:
                        callback_owner._obs_step_cache_key = cache_key
                        callback_owner._obs_step_cache = dict(batch_obs)

                    obs_n = []
                    complete = True
                    for agent in agents:
                        obs = batch_obs.get(id(agent))
                        if obs is None:
                            complete = False
                            break
                        fast_obs = self._try_fast_obs_value(obs, default_obs_dim)
                        obs_n.append(fast_obs if fast_obs is not None else self._normalize_obs_value(obs))
                    if complete and len(obs_n) == len(agents):
                        return obs_n
            except Exception:
                try:
                    callback_owner._obs_step_cache_key = None
                    callback_owner._obs_step_cache = {}
                except Exception:
                    pass

        obs_n = []
        for agent in agents:
            try:
                obs_n.append(self._get_obs(agent))
            except Exception:
                obs_n.append(np.zeros(default_obs_dim, dtype=np.float32))
        return obs_n

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0, dtype=np.float32)
        obs = self.observation_callback(agent, self.world)
        return self._normalize_obs_value(obs)

    def _get_default_obs_dim(self):
        """获取默认观察维度，确保系统一致性"""
        if hasattr(self, 'obs_dim') and self.obs_dim is not None:
            return self.obs_dim
        else:
            # 🔧 修复：如果还没有初始化观察维度，尝试从observation_space获取
            # 优先使用observation_space中的维度，避免硬编码的后备值导致截断
            if hasattr(self, 'observation_space') and len(self.observation_space) > 0:
                # 使用第一个智能体的观察空间维度
                return self.observation_space[0].shape[0]
            elif hasattr(self, 'scenario') and hasattr(self.scenario, 'observation_dim'):
                try:
                    return int(self.scenario.observation_dim)
                except Exception:
                    pass
            else:
                # 最终后备值：使用92维（当前场景的实际维度）
                return 92

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        
        # 🔧 检查早停模式：如果设置为 never 或 disabled，则完全禁用终止条件
        try:
            import os
            early_stop_mode = os.getenv('EARLY_STOP_MODE', 'never').lower()
            if early_stop_mode in ('never', 'disabled'):
                # 完全禁用早停，清除done标志并返回False
                agent_key = getattr(agent, 'name', f'agent_{id(agent)}')
                self._agent_done_flags[agent_key] = False  # 清除done标志
                return False
        except Exception:
            pass
        
        # 获取智能体标识符（使用name，确保跨进程一致）
        agent_key = getattr(agent, 'name', f'agent_{id(agent)}')
        
        # 调用原始的done_callback
        # 已完成的智能体直接返回True，避免重复打印/重复计算
        if self._agent_done_flags.get(agent_key, False):
            return True
        is_done = self.done_callback(agent, self.world)
        
        # 如果智能体完成，尝试获取终止原因
        if is_done:
            
            # 记录完成标记，后续不再重复
            current_step = int(getattr(self, '_current_step', 0))
            self._agent_done_flags[agent_key] = True
            self._agent_done_steps[agent_key] = current_step
            try:
                # 初始化终止原因列表
                if not hasattr(self, '_termination_reasons'):
                    self._termination_reasons = {}
                
                # 获取智能体名称
                agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
                
                # 🔧 关键修复：优先从world._termination_reasons获取终止原因（由is_done设置）
                # 原因：is_done在检测到终止条件时已经设置了终止原因，直接使用可以避免重复检测和状态不一致
                termination_reasons = []
                
                # 方法1：从world._termination_reasons获取（由is_done设置，最可靠）
                if hasattr(self.world, '_termination_reasons') and isinstance(self.world._termination_reasons, dict):
                    termination_reasons = self.world._termination_reasons.get(agent_name, [])
                
                # 方法2：如果方法1没有获取到，则重新检测（向后兼容）
                if not termination_reasons:
                    # 1. 检查越界
                    if hasattr(self.world, 'is_within_bounds'):
                        pos = agent.state.p_pos
                        if not self.world.is_within_bounds(pos):
                            termination_reasons.append("越界")
                    
                    # 2. 检查地形穿透（与is_done逻辑保持一致）
                    if hasattr(self.scenario, 'get_terrain_height'):
                        try:
                            import os
                            eps = float(os.getenv('TERRAIN_COLLISION_EPS', '0.3'))  # 使用真实碰撞阈值，与is_done保持一致
                        except Exception:
                            eps = 0.3
                        try:
                            terrain_h = self.scenario.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
                            if agent.state.p_pos[2] <= terrain_h + eps:
                                termination_reasons.append("地形穿透")
                        except Exception:
                            pass
                    
                    # 3. 检查实体碰撞（与is_done逻辑保持一致）
                    if hasattr(self.world, 'landmarks'):
                        pos = agent.state.p_pos
                        dmin = None
                        for landmark in self.world.landmarks:
                            lp = getattr(getattr(landmark, 'state', None), 'p_pos', None)
                            if lp is None:
                                continue
                            r = float(getattr(landmark, 'size', 0.0)) + float(getattr(agent, 'size', 0.0))
                            d = float(np.linalg.norm(pos - lp) - r)
                            dmin = d if dmin is None else min(dmin, d)
                        
                        if dmin is not None and dmin <= 0.0:
                            termination_reasons.append("实体碰撞")
                
                # 存储终止原因
                if termination_reasons:
                    self._termination_reasons[agent_name] = termination_reasons
                    # 调试：直接输出终止原因
                    print(f"[DEBUG] 智能体 {agent_name} 终止原因: {termination_reasons}")
                else:
                    self._termination_reasons[agent_name] = ["未知原因"]
                    print(f"[DEBUG] 智能体 {agent_name} 终止原因: 未知原因")
                
                # 输出详细的终止信息（恢复原有的详细输出）
                try:
                    # 使用当前环境的步数，而不是累计步数
                    step_idx = int(getattr(self, '_current_step', -1))
                    ep_len = int(getattr(self.world, 'episode_length', -1))
                    pos = agent.state.p_pos
                    
                    # 根据终止原因输出不同的详细信息
                    if "地形穿透" in termination_reasons:
                        terrain_h = self.scenario.get_terrain_height(pos[0], pos[1]) if hasattr(self.scenario, 'get_terrain_height') else 0.0
                        print(f"[终止] 地形穿透 | step={step_idx+1}/{ep_len} | agent={agent_name} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | terrain_h={terrain_h:.2f}")
                    elif "越界" in termination_reasons:
                        print(f"[终止] 越界 | step={step_idx+1}/{ep_len} | agent={agent_name} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
                    elif "实体碰撞" in termination_reasons:
                        # 计算最小距离
                        dmin = None
                        if hasattr(self.world, 'landmarks'):
                            for landmark in self.world.landmarks:
                                lp = getattr(getattr(landmark, 'state', None), 'p_pos', None)
                                if lp is None:
                                    continue
                                r = float(getattr(landmark, 'size', 0.0)) + float(getattr(agent, 'size', 0.0))
                                d = float(np.linalg.norm(pos - lp) - r)
                                dmin = d if dmin is None else min(dmin, d)
                        print(f"[终止] 实体碰撞 | step={step_idx+1}/{ep_len} | agent={agent_name} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | min_dist={dmin:.3f}")
                    else:
                        print(f"[终止] 未知原因 | step={step_idx+1}/{ep_len} | agent={agent_name} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
                except Exception as e:
                    print(f"[终止] 输出错误: {e}")
                    pass
                    
            except Exception as e:
                # 如果获取终止原因失败，记录错误但不影响主要逻辑
                pass
        
        return is_done

    def get_termination_reasons(self, env_id=None):
        """获取终止原因"""
        if not hasattr(self, '_termination_reasons'):
            return []
        
        if env_id is not None:
            # 返回特定环境的终止原因
            return list(self._termination_reasons.values())
        else:
            # 返回所有终止原因
            all_reasons = []
            for reasons in self._termination_reasons.values():
                all_reasons.extend(reasons)
            return all_reasons

    def get_latest_termination_reasons(self):
        """获取最新的终止原因（用于调试）"""
        if not hasattr(self, '_termination_reasons'):
            return []
        
        # 返回所有智能体的终止原因
        all_reasons = []
        for agent_name, reasons in self._termination_reasons.items():
            all_reasons.extend(reasons)
        return all_reasons

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        # 确保动作是numpy数组并且形状正确
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, np.ndarray) and len(action.shape) > 1:
            action = np.squeeze(action)
        
        # 处理动作中的NaN和Inf值 - 只在异常时输出调试信息
        if isinstance(action, np.ndarray) and (np.isnan(action).any() or np.isinf(action).any()):
            # 详细分析无效值
            original_action = action.copy()
            nan_mask = np.isnan(action)
            inf_mask = np.isinf(action)
            posinf_mask = np.isposinf(action)
            neginf_mask = np.isneginf(action)
            
            # 统计无效值
            nan_count = np.sum(nan_mask)
            inf_count = np.sum(inf_mask)
            posinf_count = np.sum(posinf_mask)
            neginf_count = np.sum(neginf_mask)
            
            # 获取调用栈信息
            import traceback
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
            
            # 尝试获取更多上下文信息
            try:
                # 获取智能体信息
                agent_info = f"Agent: {getattr(agent, 'name', 'Unknown')}"
                if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                    pos = agent.state.p_pos
                    agent_info += f" at pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                
                # 获取动作空间信息
                action_space_info = f"ActionSpace: {type(action_space).__name__}"
                if hasattr(action_space, 'shape'):
                    action_space_info += f" shape{action_space.shape}"
                
                # 获取时间信息
                time_info = f"Time: {time}" if time is not None else "Time: Unknown"
                
            except Exception:
                agent_info = "Agent: Info unavailable"
                action_space_info = "ActionSpace: Info unavailable"
                time_info = "Time: Info unavailable"
            
            # 输出详细警告信息
            print(f"\n{'='*60}")
            print(f"⚠️  动作值异常警告")
            print(f"{'='*60}")
            print(f"📍 调用位置: {caller_info}")
            print(f"🤖 {agent_info}")
            print(f"📦 {action_space_info}")
            print(f"⏰ {time_info}")
            print(f"📊 原始动作值: {original_action}")
            print(f"🔍 异常值统计:")
            print(f"   - NaN值: {nan_count} 个")
            print(f"   - Inf值: {inf_count} 个 (正无穷: {posinf_count}, 负无穷: {neginf_count})")
            
            # 显示具体位置
            if nan_count > 0:
                nan_indices = np.where(nan_mask)[0]
                print(f"   - NaN位置: {nan_indices}")
            if inf_count > 0:
                inf_indices = np.where(inf_mask)[0]
                print(f"   - Inf位置: {inf_indices}")
            
            # 替换无效值
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"✅ 替换后动作值: {action}")
            print(f"{'='*60}\n")
        
        # 3D环境的连续动作处理 - 最高优先级处理
        if hasattr(self.world, 'dim_p') and self.world.dim_p == 3 and isinstance(action, np.ndarray) and len(action) >= 3:
            # 对于3D环境，优先处理连续动作，无论action_space类型
            if not hasattr(agent.action, 'u') or agent.action.u is None:
                agent.action.u = np.zeros(self.world.dim_p)
            
            # 深拷贝动作，避免修改原始数据
            continuous_action = action[:self.world.dim_p].copy()
            
            # 🔧 统一Z轴映射（零中心）：[-1,1] → [-1,1]（不做偏置）
            # 记录原始Z以供后续调试信息使用
            z_action = continuous_action[2]
            continuous_action[2] = np.clip(continuous_action[2], -1.0, 1.0)
            
            # 4. 记录原始动作和处理后动作，便于调试
            if (not self._eval_light_action_path) and hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                if 'action_history' not in agent.debug_info:
                    agent.debug_info['action_history'] = []
                # 仅保留最近的100个记录
                if len(agent.debug_info['action_history']) > 100:
                    agent.debug_info['action_history'].pop(0)
                agent.debug_info['action_history'].append({
                    'original': action.copy() if isinstance(action, np.ndarray) else action,
                    'processed': continuous_action.copy(),
                    'z_original': z_action,
                    'z_processed': continuous_action[2]
                })
            
            # 5. 将处理后的连续动作应用到智能体
            agent.action.u = continuous_action
            
            # 6. 将动作信息传递给智能体，供奖励函数使用
            if not self._eval_light_action_path:
                agent.current_action = action.copy() if isinstance(action, np.ndarray) else action
            
            return
        
        # 下面是原始条件的处理逻辑
        # 检查是否需要处理连续动作输入但环境是离散动作空间
        if (isinstance(action_space, spaces.Discrete) or isinstance(action_space, spaces.MultiDiscrete)) and isinstance(action, np.ndarray) and len(action.shape) > 0:
            # 处理具有连续输入但期望离散动作的情况
            # 将连续动作向量转换为离散动作
            if hasattr(self.world, 'dim_p'):
                # 如果没有.u属性，或者.u是None，创建一个全零向量
                if not hasattr(agent.action, 'u') or agent.action.u is None:
                    agent.action.u = np.zeros(self.world.dim_p)
                
                # 使用连续动作向量
                if len(action) == self.world.dim_p:
                    # 对于完全匹配维度的连续向量，直接使用
                    agent.action.u = action.copy()
                elif len(action) > self.world.dim_p:
                    # 如果动作维度大于世界维度，截取前dim_p个维度
                    agent.action.u = action[:self.world.dim_p].copy()
                else:
                    # 如果动作维度小于世界维度，填充零
                    temp_action = np.zeros(self.world.dim_p)
                    temp_action[:len(action)] = action
                    agent.action.u = temp_action
                
                # 将动作信息传递给智能体，供奖励函数使用
                if not self._eval_light_action_path:
                    agent.current_action = action.copy() if isinstance(action, np.ndarray) else action
            return
            
        # 物理控制
        if agent.movable:
            # 确保u是初始化的
            if not hasattr(agent.action, 'u') or agent.action.u is None:
                agent.action.u = np.zeros(self.world.dim_p)
            
            # 处理动作应用
            if self.discrete_action_input:
                # 离散动作空间处理 (保持原有逻辑)
                agent.action.u = np.zeros(self.world.dim_p)
                # 处理3D离散动作
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
                # 添加Z轴离散动作支持
                if self.world.dim_p > 2 and len(action) > 0:
                    if action[0] == 5: agent.action.u[2] = +1.0
                    if action[0] == 6: agent.action.u[2] = -1.0
            else:
                # 连续动作空间处理
                if self.force_discrete_action:
                    d = np.argmax(action[0:4])
                    direction = [0,0,0,0][d]
                    agent.action.u[0] = direction
                    d = np.argmax(action[4:])
                    direction = [0,0,0,0][d]
                    agent.action.u[1] = direction
                else:
                    # 使用连续动作值，确保动作维度匹配
                    if isinstance(action, np.ndarray) and len(action) >= self.world.dim_p:
                        agent.action.u = action[:self.world.dim_p].copy()
                    else:
                        # 尝试转换为numpy数组
                        try:
                            act_array = np.array(action, dtype=float)
                            if len(act_array) >= self.world.dim_p:
                                agent.action.u = act_array[:self.world.dim_p].copy()
                            else:
                                # 无法获取足够的维度，保持现有动作
                                pass
                        except:
                            # 无法转换，保持现有动作
                            pass
            
            # 将动作信息传递给智能体，供奖励函数使用
            if not self._eval_light_action_path:
                agent.current_action = action.copy() if isinstance(action, np.ndarray) else action

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0] - 1] = 1.0
            else:
                agent.action.c = action[self.world.dim_p:] if len(action) > self.world.dim_p else np.zeros(self.world.dim_c)
        
        # 对极端情况进行额外检查
        self._apply_position_safety_constraints(agent)

    # reset rendering assets
    def _reset_render(self):
        # 使用hasattr检查属性是否存在
        if hasattr(self, 'render_callback') and self.render_callback is not None:
            return self.render_callback(self.world)
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            # 尝试使用现有的查看器 - 使用字典访问方式
            if mode not in self.viewers or self.viewers[mode] is None:
                # 检测维度并创建相应查看器
                if self.world.dim_p == 3:  # 如果是3D环境
                    try:
                        # 使用相对导入路径
                        try:
                            from . import rendering_3d
                        except ImportError:
                            # 备用方案
                            import multiagent.rendering_3d as rendering_3d
                        self.viewers[mode] = rendering_3d.Viewer3D()
                    except ImportError:
                        print("无法导入rendering_3d，回退到默认渲染器")
                        from gym.envs.classic_control import rendering
                        self.viewers[mode] = rendering.Viewer(700, 700)
                else:  # 默认为2D环境
                    try:
                        from . import rendering
                    except ImportError:
                        import multiagent.rendering as rendering
                    self.viewers[mode] = rendering.Viewer(700, 700)

            # 更新场景
            if self.world.dim_p == 3:  # 3D环境特殊处理
                viewer = self.viewers[mode]
                # 首先清空实体
                viewer.entities = []
                # 重新添加所有实体
                for entity in self.world.entities:
                    entity_type = getattr(entity, 'entity_type', entity.name if hasattr(entity, 'name') else 'unspecified')
                    geom = viewer.add_entity(
                        position=entity.state.p_pos,
                        size=entity.size,
                        color=entity.color,
                        entity_type=entity_type,
                        name=entity.name if hasattr(entity, 'name') else None
                    )
                # 更新查看器
                viewer.update()

                # 处理事件队列以确保窗口响应
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return None
                    viewer.handle_event(event)

                return viewer
            else:  # 2D环境正常处理
                # 从世界中获取每个geom并添加到查看器
                # ... 现有2D渲染代码 ...
                return self.viewers[mode].render(return_rgb_array=(mode=='rgb_array'))
        elif mode == 'rgb_array':
            # 如果需要RGB数组，首先确保查看器存在
            self.render(mode='human')
            if self.world.dim_p == 3:  # 3D环境特殊处理
                return self.viewers['human'].render(return_rgb_array=True)
            else:  # 2D环境
                return self.viewers['human'].render(return_rgb_array=True)

    def close(self):
        """关闭环境和相关资源"""
        for viewer in self.viewers.values():
            if hasattr(viewer, 'close'):
                viewer.close()
        self.viewers = {}
        return super().close() if hasattr(super(), 'close') else None

    def _render_3d_human(self):
        """3D人类模式渲染"""
        try:
            # 确保导入所需的模块
            try:
                from . import rendering_3d
            except ImportError:
                import multiagent.rendering_3d as rendering_3d
            
            # 创建3D查看器（如果不存在）
            if not hasattr(self, '_3d_viewer') or self._3d_viewer is None:
                print("创建新的3D查看器...")
                self._3d_viewer = rendering_3d.create_3d_world(width=800, height=600)
                
                # 添加所有实体
                print("添加实体到3D场景...")
                
                # 检查是否有山脉对象，并添加到场景
                if hasattr(self, 'scenario') and hasattr(self.scenario, 'mountain'):
                    print("检测到山脉对象，添加到3D场景...")
                    self._3d_viewer.add_mountain(mountain_obj=self.scenario.mountain)
                
                # 添加智能体
                for agent in self.world.agents:
                    color = agent.color if hasattr(agent, 'color') else [0.35, 0.35, 0.85]
                    self._3d_viewer.add_entity(
                        position=agent.state.p_pos,
                        size=agent.size if hasattr(agent, 'size') else 0.1,
                        color=color,
                        entity_type='agent',
                        name=agent.name if hasattr(agent, 'name') else None
                    )
                
                # 添加地标（障碍物和目标）
                for landmark in self.world.landmarks:
                    entity_type = 'obstacle'
                    if hasattr(landmark, 'name'):
                        if 'goal' in landmark.name.lower():
                            entity_type = 'goal'
                        elif 'obstacle' in landmark.name.lower():
                            entity_type = 'obstacle'
                    
                    color = landmark.color if hasattr(landmark, 'color') else [0.75, 0.25, 0.25]
                    self._3d_viewer.add_entity(
                        position=landmark.state.p_pos,
                        size=landmark.size if hasattr(landmark, 'size') else 0.1,
                        color=color,
                        entity_type=entity_type,
                        name=landmark.name if hasattr(landmark, 'name') else None
                    )
                
                print(f"已添加 {len(self.world.agents)} 个智能体和 {len(self.world.landmarks)} 个地标到3D场景")
                
                # 设置键盘重复率，使键盘控制更流畅
                pygame.key.set_repeat(10, 10)  # 10ms延迟，10ms间隔
            
            # 更新实体位置
            self._update_entity_positions()
            
            # 处理积累的所有事件
            import pygame
            events = pygame.event.get()
            if events:  # 只有在有事件时才处理
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return False
                    elif hasattr(self._3d_viewer, 'handle_event'):
                        # 传递事件给3D查看器处理
                        self._3d_viewer.handle_event(event)
            
            # 更新键盘状态
            keys = pygame.key.get_pressed()
            if hasattr(self._3d_viewer, 'target_distance'):
                # 放大缩小
                if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                    self._3d_viewer.target_distance -= 0.2
                if keys[pygame.K_MINUS]:
                    self._3d_viewer.target_distance += 0.2
                    
                # WASD控制相机位置平移
                move_speed = 0.05
                if keys[pygame.K_w]:
                    self._3d_viewer.camera_position_offset[1] += move_speed
                if keys[pygame.K_s]:
                    self._3d_viewer.camera_position_offset[1] -= move_speed
                if keys[pygame.K_a]:
                    self._3d_viewer.camera_position_offset[0] -= move_speed
                if keys[pygame.K_d]:
                    self._3d_viewer.camera_position_offset[0] += move_speed
                if keys[pygame.K_q]:
                    self._3d_viewer.center[2] += move_speed
                if keys[pygame.K_e]:
                    self._3d_viewer.center[2] -= move_speed
                # 上下左右控制视角旋转
                if keys[pygame.K_LEFT]:
                    self._3d_viewer.target_azimuth -= 1.0
                if keys[pygame.K_RIGHT]:
                    self._3d_viewer.target_azimuth += 1.0
                if keys[pygame.K_UP]:
                    self._3d_viewer.target_elevation += 1.0
                if keys[pygame.K_DOWN]:
                    self._3d_viewer.target_elevation -= 1.0
            
            # 更新并渲染3D场景
            if hasattr(self._3d_viewer, 'update'):
                self._3d_viewer.update()
            render_result = self._3d_viewer.render()
            
            # 确保处理事件队列，防止窗口无响应
            pygame.event.pump()
            
            # 短暂的延迟让事件处理有时间执行
            pygame.time.delay(10)
            
            return render_result
        except Exception as e:
            print(f"3D渲染失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _update_entity_positions(self):
        """更新3D实体位置"""
        if not hasattr(self, '_3d_viewer'):
            return
        
        # 更新智能体位置
        for i, agent in enumerate(self.world.agents):
            if i < len(self._3d_viewer.entities):
                if hasattr(self._3d_viewer, 'update_entity'):
                    self._3d_viewer.update_entity(i, agent.state.p_pos)
                elif hasattr(self._3d_viewer.entities[i], 'set_position'):
                    self._3d_viewer.entities[i].set_position(agent.state.p_pos)
                
        # 更新目标点位置
        offset = len(self.world.agents)
        for i, landmark in enumerate(self.world.landmarks):
            idx = offset + i
            if idx < len(self._3d_viewer.entities):
                if hasattr(self._3d_viewer, 'update_entity'):
                    self._3d_viewer.update_entity(idx, landmark.state.p_pos)
                elif hasattr(self._3d_viewer.entities[idx], 'set_position'):
                    self._3d_viewer.entities[idx].set_position(landmark.state.p_pos)

    def _render_3d_array(self):
        """生成3D渲染的RGB数组"""
        try:
            # 确保3D渲染器存在
            if not hasattr(self, '_3d_viewer'):
                self._render_3d_human()
            
            # 获取RGB数组
            try:
                return self._3d_viewer.render(return_rgb_array=True)
            except TypeError as e:
                # 处理旧版本渲染器不支持return_rgb_array参数的情况
                if "unexpected keyword argument 'return_rgb_array'" in str(e):
                    print("3D渲染器不支持return_rgb_array参数，正在使用备用方法...")
                    # 设置需要返回RGB数组的标志（如果渲染器支持这种方式）
                    if hasattr(self._3d_viewer, 'need_return_rgb_array'):
                        self._3d_viewer.need_return_rgb_array = True
                        return self._3d_viewer.render()
                    else:
                        # 如果都不支持，回退到简单RGB数组
                        print("3D渲染器不支持RGB数组模式，使用简单RGB数组代替")
                        return self._simple_rgb_array()
                else:
                    # 其他TypeError异常，重新抛出
                    raise
        except Exception as e:
            print(f"生成3D RGB数组失败: {e}")
            # 回退到简单RGB数组
            return self._simple_rgb_array()

    def _render_human(self):
        """人类模式渲染 - 用于直接显示"""
        try:
            # 尝试使用scenario的render方法
            if hasattr(self, 'scenario') and hasattr(self.scenario, 'render'):
                self.scenario.render(self.world)
            else:
                # 简单的后备渲染
                img = self._simple_rgb_array()
                
                # 显示图像
                try:
                    import cv2
                    cv2.imshow('3D环境', img)
                    cv2.waitKey(1)
                except ImportError:
                    print("警告: 无法导入cv2库，无法显示图像")
        except Exception as e:
            print(f"人类模式渲染失败: {e}")
        return None

    def _simple_rgb_array(self):
        """生成一个简单的RGB数组，但具有3D效果"""
        # 创建图像
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        img[:,:] = [10, 10, 30]  # 深蓝色背景
        
        # 获取所有实体位置
        entities = self.world.agents + self.world.landmarks
        positions = []
        for entity in entities:
            if hasattr(entity, 'state') and hasattr(entity.state, 'p_pos'):
                positions.append(entity.state.p_pos)
        
        if not positions:
            return img
        
        # 找到坐标范围
        positions = np.array(positions)
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        
        # 创建3D网格效果
        grid_size = 20
        for i in range(grid_size):
            # X-Y平面网格
            y = int(50 + i * (500/grid_size))
            cv2.line(img, (50, y), (750, y), [40, 40, 80], 1)
            x = int(50 + i * (700/grid_size))
            cv2.line(img, (x, 50), (x, 550), [40, 40, 80], 1)
            
            # 添加透视效果的网格线 (模拟Z轴)
            if i < grid_size // 2:
                # 左侧透视线
                cv2.line(img, (50, 300 - i*20), (400, 300 - i*5), [30, 30, 60], 1)
                # 右侧透视线
                cv2.line(img, (750, 300 - i*20), (400, 300 - i*5), [30, 30, 60], 1)
        
        # 绘制坐标轴
        # X轴 (红色)
        cv2.line(img, (400, 300), (600, 300), [200, 0, 0], 2)
        # Y轴 (绿色)
        cv2.line(img, (400, 300), (400, 100), [0, 200, 0], 2)
        # Z轴 (蓝色，带透视效果)
        cv2.line(img, (400, 300), (550, 200), [0, 0, 200], 2)
        
        # 绘制每个实体，考虑Z轴透视
        for i, entity in enumerate(entities):
            if hasattr(entity, 'state') and hasattr(entity.state, 'p_pos'):
                pos = entity.state.p_pos
                # 归一化到2D图像坐标，添加透视效果
                norm_x = (pos[0] - min_pos[0]) / (max_pos[0] - min_pos[0] + 1e-10)
                norm_y = (pos[1] - min_pos[1]) / (max_pos[1] - min_pos[1] + 1e-10)
                norm_z = 0
                if len(pos) > 2:
                    norm_z = (pos[2] - min_pos[2]) / (max_pos[2] - min_pos[2] + 1e-10)
                
                # 基础坐标
                x_base = 50 + norm_x * 700
                y_base = 50 + (1-norm_y) * 500  # Y轴反转
                
                # 添加Z轴透视效果
                z_effect = norm_z * 100  # Z值影响透视效果的程度
                
                # 最终坐标 (透视校正)
                x = int(x_base + z_effect * 0.5)  # Z越大，X越偏右
                y = int(y_base - z_effect)        # Z越大，Y越偏上
                
                # 根据实体类型绘制不同颜色和大小
                if i < len(self.world.agents):
                    # 智能体，按索引使用不同颜色
                    colors = [
                        [255, 0, 0],    # 红色
                        [0, 0, 255],    # 蓝色
                        [255, 255, 0],  # 黄色
                        [255, 0, 255],  # 紫色
                        [0, 255, 255]   # 青色
                    ]
                    color = colors[i % len(colors)]
                    # Z值越大，大小越大，表示更靠近视图
                    size = int(10 + norm_z * 15)
                    cv2.circle(img, (x, y), size, color, -1)
                    
                    # 添加阴影效果
                    shadow_y = 550 - int(norm_z * 100)
                    shadow_size = max(5, size - int(norm_z * 8))
                    cv2.circle(img, (x, shadow_y), shadow_size, [80, 80, 80], -1)
                else:
                    # 目标点或障碍物
                    if hasattr(entity, 'name') and entity.name:
                        name = entity.name.lower()
                        if 'goal' in name or 'target' in name:
                            color = [0, 255, 0]  # 绿色目标
                            size = 15
                        else:
                            color = [150, 150, 150]  # 灰色障碍物
                            size = 12
                    else:
                        color = [200, 200, 200]  # 默认颜色
                        size = 12
                    
                    # Z值影响大小
                    size = int(size + norm_z * 10)
                    cv2.circle(img, (x, y), size, color, -1)
        
        # 添加轨迹：如果世界对象中有智能体轨迹，则绘制
        for i, agent in enumerate(self.world.agents):
            if hasattr(agent, '_trajectory') and len(agent._trajectory) > 1:
                traj = agent._trajectory
                colors = [[255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
                color = colors[i % len(colors)]
                
                # 绘制轨迹线
                for j in range(1, len(traj)):
                    p1 = traj[j-1]
                    p2 = traj[j]
                    
                    # 归一化到图像坐标
                    x1 = int(50 + (p1[0] - min_pos[0]) / (max_pos[0] - min_pos[0] + 1e-10) * 700)
                    y1 = int(50 + (1-(p1[1] - min_pos[1]) / (max_pos[1] - min_pos[1] + 1e-10)) * 500)
                    
                    x2 = int(50 + (p2[0] - min_pos[0]) / (max_pos[0] - min_pos[0] + 1e-10) * 700)
                    y2 = int(50 + (1-(p2[1] - min_pos[1]) / (max_pos[1] - min_pos[1] + 1e-10)) * 500)
                    
                    # 添加Z轴透视
                    if len(p1) > 2 and len(p2) > 2:
                        z1 = (p1[2] - min_pos[2]) / (max_pos[2] - min_pos[2] + 1e-10)
                        z2 = (p2[2] - min_pos[2]) / (max_pos[2] - min_pos[2] + 1e-10)
                        
                        x1 += int(z1 * 50)
                        y1 -= int(z1 * 100)
                        x2 += int(z2 * 50)
                        y2 -= int(z2 * 100)
                    
                    cv2.line(img, (x1, y1), (x2, y2), color, 2)
        
        # 添加简单文本说明
        for i in range(len(self.world.agents)):
            color_rect = np.zeros((20, 20, 3), dtype=np.uint8)
            colors = [[255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
            color_rect[:,:] = colors[i % len(colors)]
            x_pos = 20
            y_pos = 20 + i * 25
            img[y_pos:y_pos+20, x_pos:x_pos+20] = color_rect
        
        return img

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

    def _update_3d_objects(self):
        """更新3D对象的位置（如果3D渲染器存在）"""
        if hasattr(self, '_3d_viewer') and self._3d_viewer is not None:
            self._update_entity_positions()
            
            # 处理积累的事件，防止窗口无响应
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    # 传递事件给3D查看器
                    if hasattr(self._3d_viewer, 'handle_event'):
                        self._3d_viewer.handle_event(event)
            except Exception as e:
                # print(f"处理3D事件时出错: {e}")
                pass

    def _apply_position_safety_constraints(self, agent):
        """检查并修正智能体的极端位置，确保它们不会逃离环境太远"""
        # 最大允许距离 - 设置为100，这是更合理的值
        max_allowed_distance = 1000.0  # 之前设置为1000.0，改回100.0
        
        # 检查智能体当前位置是否超出范围
        position = agent.state.p_pos
        distance_from_origin = np.linalg.norm(position)
        
        # 记录当前位置和距离，便于调试
        if np.random.random() < 0.005:  # 0.5%概率打印
            # print(f"安全检查: 距离={distance_from_origin:.2f}, 位置={position}, 最大允许={max_allowed_distance}")
            pass
        
        # 如果智能体距离原点太远
        if distance_from_origin > max_allowed_distance:
            # 计算超出比例
            excess_ratio = distance_from_origin / max_allowed_distance
            
            # 如果严重超出范围(10倍以上)，才直接拉回到边界
            if excess_ratio > 10.0:  # 从5.0增加到10.0，非常宽松的限制
                # 向原点方向的单位向量
                direction_to_origin = -position / (distance_from_origin + 1e-8)
                
                # 新的位置在最大允许距离处
                new_position = direction_to_origin * max_allowed_distance * 0.9
                
                # 强制更新位置和速度
                agent.state.p_pos = new_position
                agent.state.p_vel = np.zeros_like(agent.state.p_vel)  # 停止移动
                
                # 记录强制修正
                # print(f"强制修正位置: 原位置={position}, 距离={distance_from_origin:.2f}, 新位置={new_position}")
            # 对于轻微超出范围的情况，不做强制限制，只非常轻微向内拉
            else:
                # 计算向内拉的比例，超出越多拉得越厉害，但比例非常小
                pull_ratio = 0.005 * (excess_ratio - 1.0) + 0.001  # 进一步减小拉力
                
                # 向原点方向的单位向量
                direction_to_origin = -position / (distance_from_origin + 1e-8)
                
                # 施加一个向内的力
                correction_force = direction_to_origin * pull_ratio * distance_from_origin
                
                # 应用这个力作为负速度
                agent.state.p_vel += correction_force
                
                if np.random.random() < 0.01:  # 1%的概率记录
                    # print(f"应用微弱向心力: 位置={position}, 距离={distance_from_origin:.2f}, 力={correction_force}")
                    pass
        
        # 检查并修正任何NaN或无穷值
        if not np.all(np.isfinite(agent.state.p_pos)):
            # print(f"检测到无效位置: {agent.state.p_pos}，重置到原点")
            agent.state.p_pos = np.zeros(self.world.dim_p)
            agent.state.p_vel = np.zeros(self.world.dim_p)


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n):
        # 修改签名以符合gym标准
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, info = env.step(action_n[i:(i+env.n)])
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

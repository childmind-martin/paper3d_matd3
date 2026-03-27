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
        self.discrete_action_space = True
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
                self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering - 使用字典而非列表
        self.shared_viewer = shared_viewer
        self.viewers = {}  # 改为字典初始化
        self._reset_render()

    def step(self, action_n, time=None):
        """
        步进环境，应用动作并更新状态
        
        参数：
            action_n - 所有智能体的动作列表
            time - 时间步（默认为None）
        
        返回：
            obs_n - 所有智能体的观察列表
            reward_n - 所有智能体的奖励列表
            done_n - 所有智能体的完成标志列表
            info_n - 所有智能体的信息字典列表
        """
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        
        try:
            # 设置动作
            self._set_action(action_n, self.agents, time)
            
            # 准备力向量数组
            p_force = [np.zeros(self.world.dim_p) for _ in range(len(self.world.entities) + len(self.world.agents))]
            
            # 应用动作产生的力
            self.world.apply_action_force(p_force)
            
            # 应用环境力
            self.world.apply_environment_force(p_force)
            
            # 集成物理
            self.world.integrate_state(p_force)
            
            # 对于每个智能体收集信息
            for agent in self.world.agents:
                try:
                    # 收集观察
                    obs = self._get_obs(agent)
                    # 收集奖励
                    reward = self._get_reward(agent)
                    # 检查是否完成
                    done = self._get_done(agent)
                    # 收集信息
                    info = self._get_info(agent)
                    
                    # 添加到结果列表
                    obs_n.append(obs)
                    reward_n.append(reward)
                    done_n.append(done)
                    info_n['n'].append(info)
                except Exception as e:
                    import traceback
                    print(f"处理智能体 {agent.name} 的步骤结果时出错: {e}")
                    traceback.print_exc()
                    # 添加默认值
                    obs_n.append(np.zeros(self.observation_space[0].shape[0]))
                    reward_n.append(0.0)
                    done_n.append(False)
                    info_n['n'].append({})
            
            # 所有智能体完成即整体完成
            done = all(done_n)
            info = {'done': done}
            
            # 统一格式以兼容gym和gymnasium
            if hasattr(self, 'gymnasium_mode') and self.gymnasium_mode:
                return obs_n, reward_n, done_n, info
            else:
                return obs_n, reward_n, done_n, info
            
        except Exception as e:
            import traceback
            print(f"环境步骤执行失败: {e}")
            traceback.print_exc()
            
            # 返回默认值
            default_obs = [np.zeros(space.shape[0]) for space in self.observation_space]
            default_reward = [0.0 for _ in range(len(self.agents))]
            default_done = [False for _ in range(len(self.agents))]
            default_info = {'n': [{} for _ in range(len(self.agents))], 'done': False}
            
            if hasattr(self, 'gymnasium_mode') and self.gymnasium_mode:
                return default_obs, default_reward, default_done, default_info
            else:
                return default_obs, default_reward, default_done, {'done': False}

    def _set_action(self, action_n, agent_actions, time=None):
        """
        设置智能体动作，将接收的动作分配给各智能体
        
        参数：
            action_n - 所有智能体的动作列表
            agent_actions - 分配动作的智能体对象列表
            time - 时间步（默认为None）
        """
        # 确保动作是列表格式
        if not isinstance(action_n, list):
            if isinstance(action_n, dict):
                # 将字典转换为列表
                action_list = [None] * len(self.agents)
                for agent_id, action in action_n.items():
                    try:
                        agent_idx = next(i for i, ag in enumerate(self.agents) if ag.name == agent_id)
                        action_list[agent_idx] = action
                    except (StopIteration, TypeError) as e:
                        print(f"警告: 无法为ID {agent_id} 找到智能体: {e}")
                action_n = action_list
            else:
                # 如果是单一动作，转换为列表
                action_n = [action_n]
        
        # 确保动作列表长度与智能体数量匹配
        if len(action_n) < len(agent_actions):
            print(f"警告: 动作数量 {len(action_n)} 少于智能体数量 {len(agent_actions)}，将使用默认动作")
            # 扩展动作列表
            action_n.extend([np.zeros(self.world.dim_p) for _ in range(len(agent_actions) - len(action_n))])
        elif len(action_n) > len(agent_actions):
            print(f"警告: 动作数量 {len(action_n)} 多于智能体数量 {len(agent_actions)}，将忽略多余动作")
            action_n = action_n[:len(agent_actions)]
        
        # 将动作分配给智能体
        for i, agent in enumerate(agent_actions):
            try:
                action = action_n[i]
                if action is None:
                    # 如果动作为None，使用默认动作
                    if self.discrete_action_space:
                        # 离散动作空间使用零动作（不移动）
                        action = np.zeros(7)  # 7D离散动作
                        action[0] = 1  # 不移动的动作
                    else:
                        # 连续动作空间使用零力
                        action = np.zeros(self.world.dim_p)
                
                # 处理不同格式的动作
                if not isinstance(action, np.ndarray):
                    try:
                        action = np.array(action)
                    except:
                        print(f"警告: 无法将智能体 {i} 的动作转换为numpy数组，使用默认动作")
                        if self.discrete_action_space:
                            action = np.zeros(7)
                            action[0] = 1  # 不移动
                        else:
                            action = np.zeros(self.world.dim_p)
                
                # 确保离散动作符合格式
                if self.discrete_action_space:
                    # 检查是否是离散动作格式
                    if action.shape[0] != 7:
                        print(f"警告: 智能体 {i} 的离散动作维度应为7，但收到 {action.shape[0]}，调整格式")
                        # 将其他格式转换为7D离散动作
                        new_action = np.zeros(7)
                        if action.shape[0] == 0:  # 空动作
                            new_action[0] = 1  # 不移动
                        elif len(action.shape) == 0 or action.shape[0] == 1:  # 标量
                            # 将标量转换为one-hot格式
                            idx = int(action.item() if len(action.shape) == 0 else action[0]) % 7
                            new_action[idx] = 1
                        else:  # 其他维度
                            # 直接设置不移动
                            new_action[0] = 1
                        action = new_action
                else:
                    # 连续动作空间
                    if action.shape[0] != self.world.dim_p:
                        print(f"警告: 智能体 {i} 的连续动作维度应为 {self.world.dim_p}，但收到 {action.shape[0]}，调整格式")
                        # 将其他格式转换为3D连续动作
                        new_action = np.zeros(self.world.dim_p)
                        if action.shape[0] == 0:  # 空动作
                            pass  # 使用零向量
                        elif len(action.shape) == 0:  # 标量
                            # 只设置x方向
                            new_action[0] = float(action)
                        elif action.shape[0] == 1:  # 1D
                            # 只设置x方向
                            new_action[0] = action[0]
                        elif action.shape[0] == 2:  # 2D
                            # 设置x和y方向
                            new_action[:2] = action[:2]
                        else:  # 其他维度
                            # 截取前三个维度
                            new_action = action[:self.world.dim_p]
                        action = new_action
                
                # 设置智能体动作
                agent.action = action
                
            except Exception as e:
                import traceback
                print(f"为智能体 {i} 设置动作时出错: {e}")
                traceback.print_exc()
                # 设置默认动作
                if self.discrete_action_space:
                    self.agents[i].action = np.zeros(7)
                    self.agents[i].action[0] = 1  # 不移动
                else:
                    self.agents[i].action = np.zeros(self.world.dim_p)

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
            
        # reset world
        if self.reset_callback is not None:
            self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        
        # 保存智能体引用
        self.agents = self.world.policy_agents
        
        # record observations for each agent
        obs_n = []
        
        try:
            # 为每个智能体单独获取观察值，并加入更多错误处理
            for i, agent in enumerate(self.agents):
                try:
                    # 获取智能体的观察值
                    # print(f"为智能体 {i}/{len(self.agents)-1} 获取观察值")
                    obs = self._get_obs(agent)
                    
                    # 确保观察值是numpy数组
                    if not isinstance(obs, np.ndarray):
                        try:
                            if isinstance(obs, list) and len(obs) > 0:
                                if isinstance(obs[0], np.ndarray):
                                    obs = obs[0]  # 取第一个数组
                                else:
                                    obs = np.array(obs)
                            else:
                                obs = np.array(obs)
                        except Exception as e:
                            # print(f"转换观察值出错: {e}")
                            # 创建默认观察值
                            obs = np.zeros(36)
                    
                    # 形状验证
                    if len(obs.shape) > 1 or obs.shape[0] != 36:
                        # print(f"观察值形状不正确: {obs.shape}，尝试调整")
                        if len(obs.shape) > 1:
                            try:
                                obs = obs.flatten()[:36]  # 展平并截断
                            except:
                                obs = np.zeros(36)
                        
                        # 填充或截断到36维
                        if len(obs) < 36:
                            obs = np.pad(obs, (0, 36 - len(obs)), 'constant')
                        elif len(obs) > 36:
                            obs = obs[:36]
                    
                    # 添加到观察值列表
                    obs_n.append(obs)
                    # print(f"智能体 {i} 观察值获取成功，形状={obs.shape}")
                except Exception as e:
                    # print(f"获取智能体 {i} 的观察值时出错: {e}")
                    # 添加默认的零向量
                    obs_n.append(np.zeros(36))
            
            # 最终验证：确保观察值数量与智能体数量匹配
            if len(obs_n) != len(self.agents):
                # print(f"警告: 最终观察值数量 ({len(obs_n)}) 与智能体数量 ({len(self.agents)}) 不匹配!")
                
                # 调整观察值数量以匹配智能体数量
                if len(obs_n) > len(self.agents):
                    # print(f"裁剪多余的观察值: {len(obs_n)} -> {len(self.agents)}")
                    obs_n = obs_n[:len(self.agents)]
                else:
                    # print(f"添加缺失的观察值: {len(obs_n)} -> {len(self.agents)}")
                    while len(obs_n) < len(self.agents):
                        obs_n.append(np.zeros(36))
            else:
                # print(f"观察值数量验证通过: {len(obs_n)} 个，与智能体数量匹配")
                pass
        except Exception as e:
            print(f"重置环境期间发生错误: {e}")
            # 确保始终返回与智能体数量匹配的观察值
            obs_n = [np.zeros(36) for _ in range(len(self.agents))]
            
        # 检查是否使用的是 gymnasium (通过检查是否有特定方法)
        if hasattr(gym.Env, 'reset') and 'options' in gym.Env.reset.__code__.co_varnames:
            # gymnasium API 返回 (observations, info)
            info = {}
            return obs_n, info
        else:
            # 旧版 gym API 只返回 observations
            return obs_n

    # get info used for benchmarking
    def _get_info(self, agent, pre_positions, pre_velocities):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world, pre_positions, pre_velocities)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent, pre_positions, pre_velocities):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world, pre_positions, pre_velocities)

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

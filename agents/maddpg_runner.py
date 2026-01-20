import numpy as np
import tensorflow as tf
import tqdm
from agents.maddpg import MADDPGAgent
import time
import copy
import os
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import psutil
import threading
import platform
import cv2
from importlib import import_module

# 导入DEBUG_OUTPUT变量
DEBUG_OUTPUT = False  # 默认值

class MADDPGRunner():

    def run_episode(self, max_episode_len=200, policy_param='last', render=True, waitTime=0.0, mode=None, test_mode=True, log_interval=50, fast_mode=False, visualization_callback=None):
        """运行一个回合（用于评估）
        
        Args:
            max_episode_len: 最大步数
            policy_param: 使用哪种策略权重('best_overall', 'best_single', 'best_average', 'last')
            render: 是否渲染环境
            waitTime: 每步等待时间
            mode: 渲染模式('human', 'rgb_array', None)
            test_mode: 是否为测试模式
            log_interval: 日志记录间隔
            fast_mode: 快速模式，减少日志和跳过部分渲染
            visualization_callback: 可视化回调函数，接收(step, trajectories, scenario)参数
        
        Returns:
            final_trajectories: 智能体轨迹
            reward_episode: 回合总奖励
            info: 步骤信息
            images: 收集的图像
        """
        print("\n=== 开始执行回合 ===")
        
        # 设置使用哪个策略
        for agent in self.agents:
            if hasattr(agent, 'set_policy_for_execution'):
                agent.set_policy_for_execution(policy_param)
        
        # 打印智能体数量和环境信息
        print(f"环境中智能体数量: {len(self.env.world.agents)}")
        for i, agent in enumerate(self.env.world.agents):
            if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                print(f"智能体 {i} 初始位置: {agent.state.p_pos}")
        
        # 初始化状态
        states = self.env.reset()
        
        # 打印观察空间信息
        print(f"环境返回观察数据类型: {type(states)}")
        if isinstance(states, list):
            print(f"观察列表长度: {len(states)}")
            for i, obs in enumerate(states):
                if obs is not None:
                    print(f"智能体 {i} 观察数据形状: {obs.shape if hasattr(obs, 'shape') else '未知'}")
                else:
                    print(f"智能体 {i} 观察数据为None")
        
        # 检查states类型并转换为列表
        if not isinstance(states, list):
            # 如果states不是列表，转换为列表
            if isinstance(states, tuple):
                states = list(states)
            else:
                # 单个状态值，转换为只有一个元素的列表
                states = [states]
                
        # 确保states列表长度与智能体数量匹配
        if len(states) < self.n_agents:
            print(f"警告: 状态数量({len(states)})小于智能体数量({self.n_agents})")
            # 查找哪些智能体缺少状态数据
            for i in range(self.n_agents):
                if i >= len(states) or states[i] is None:
                    print(f"智能体 {i} 缺少状态数据")
                    
            # 使用复制第一个状态来扩展states列表
            while len(states) < self.n_agents:
                agent_idx = len(states)
                # 尝试获取适当的观察空间维度
                obs_dim = 0
                if agent_idx < len(self.agents) and hasattr(self.agents[agent_idx], 'obs_dim'):
                    obs_dim = self.agents[agent_idx].obs_dim
                else:
                    # 尝试从其他智能体或已有状态获取维度
                    for a in self.agents:
                        if hasattr(a, 'obs_dim'):
                            obs_dim = a.obs_dim
                            break
                    # 如果还是无法确定维度，尝试从已有状态推断
                    if obs_dim <= 0 and states and states[0] is not None:
                        if isinstance(states[0], np.ndarray):
                            obs_dim = states[0].shape[0]
                    # 最后的默认值
                    if obs_dim <= 0:
                        obs_dim = 61  # 默认观察空间维度
                
                # 使用零向量作为默认观察
                print(f"为智能体 {agent_idx} 生成默认观察数据，维度: {obs_dim}")
                states.append(np.zeros(obs_dim))
        
        reward_episode = 0
        images = []
        info = []
        
        # 初始化智能体轨迹数组 - 无论快速模式与否，都预先分配空间
        trajectories = []
        # 为每个智能体预分配足够空间的轨迹数组
        for i in range(self.n_agents):
            # 预分配足够大的数组（max_episode_len + 1个元素）
            agent_trajectory = [None] * (max_episode_len + 1) 
            # 记录初始位置
            if i < len(self.env.world.agents) and hasattr(self.env.world.agents[i], 'state'):
                agent_trajectory[0] = self.env.world.agents[i].state.p_pos.copy()
                if not fast_mode:  # 快速模式下减少日志输出
                    print(f"智能体 {i} 初始位置: {agent_trajectory[0]}")
            else:
                # 如果无法获取实际初始位置，使用零向量
                print(f"警告: 无法获取智能体 {i} 的初始位置")
                agent_trajectory[0] = np.zeros(3)  # 默认为3D位置向量
            trajectories.append(agent_trajectory)
        
        # 轨迹索引 - 当前轨迹在数组中的位置
        traj_indices = [1] * self.n_agents
        
        # 确保3D渲染器设置正确
        if self.is_3d and hasattr(self.env, '_3d_viewer') and render:
            if not fast_mode:
                print("设置3D渲染器...")
            self.env._3d_viewer.auto_rotate = False
            self.env._3d_viewer.camera_angle = 45.0  # 45度角
            self.env._3d_viewer.camera_height = 8.0  # 增加高度更好观察
            self.env._3d_viewer.camera_distance = 15.0  # 增加距离看到更多场景
        
        print(f"开始执行，最大步数: {max_episode_len}")
        
        # 使用tqdm进度条显示测试进度（仅在非快速模式下）
        if not fast_mode:
            try:
                import tqdm
                steps_iter = tqdm.tqdm(range(max_episode_len), desc="测试进度")
            except ImportError:
                steps_iter = range(max_episode_len)
        else:
            steps_iter = range(max_episode_len)
        
        for step in steps_iter:
            # 获取动作 - 使用policy
            actions = []
            for i, agent in enumerate(self.agents):
                # 使用策略网络选择动作 - 确保处理各种输入形式
                if i < len(states):
                    agent_obs = states[i]
                else:
                    print(f"警告: 智能体 {i} 的观察数据不存在，使用零向量代替...")
                    # 尝试确定观察空间维度
                    obs_dim = 0
                    if hasattr(agent, 'obs_dim'):
                        obs_dim = agent.obs_dim
                    else:
                        # 尝试从其他智能体或默认值获取维度
                        for a in self.agents:
                            if hasattr(a, 'obs_dim'):
                                obs_dim = a.obs_dim
                                break
                        if obs_dim <= 0:
                            obs_dim = 61  # 默认观察空间维度
                    # 使用零向量作为默认观察
                    agent_obs = np.zeros(obs_dim)
                
                # 确保输入是正确形状并选择动作
                try:
                    if isinstance(agent_obs, np.ndarray) and agent_obs.ndim == 1:
                        action = agent.policy(agent_obs, add_noise=True)
                    else:
                        action = agent.policy(agent_obs, add_noise=True)
                    
                    # 确保action是NumPy数组
                    if isinstance(action, tf.Tensor):
                        action_np = action.numpy()
                    else:
                        action_np = np.array(action)
                    
                    actions.append(action_np)
                except Exception as e:
                    if not fast_mode:
                        print(f"获取智能体{i}动作时出错: {e}")
                    # 提供默认动作防止崩溃
                    action_dim = self.env.world.dim_p
                    actions.append(np.zeros(action_dim))
            
            # 应用动作处理器（如果存在）
            if hasattr(self, '_action_processor') and callable(self._action_processor):
                try:
                    actions = self._action_processor(actions)
                except Exception as e:
                    print(f"动作处理器错误: {e}")
            
            # 执行动作
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # 确保next_states是列表
            if not isinstance(next_states, list):
                if isinstance(next_states, tuple):
                    next_states = list(next_states)
                else:
                    next_states = [next_states]
                    
            # 确保next_states列表长度与智能体数量匹配
            if len(next_states) < self.n_agents:
                print(f"警告: next_states数量({len(next_states)})小于智能体数量({self.n_agents})")
                # 扩展next_states列表，使用零向量而不是简单复制
                while len(next_states) < self.n_agents:
                    i = len(next_states)  # 当前处理的智能体索引
                    # 尝试获取适当的观察空间维度
                    obs_dim = 0
                    if i < len(self.agents) and hasattr(self.agents[i], 'obs_dim'):
                        obs_dim = self.agents[i].obs_dim
                    else:
                        # 尝试从其他智能体或已有状态获取维度
                        for a in self.agents:
                            if hasattr(a, 'obs_dim'):
                                obs_dim = a.obs_dim
                                break
                        # 如果还是无法确定维度，尝试从已有状态推断
                        if obs_dim <= 0 and next_states and next_states[0] is not None:
                            if isinstance(next_states[0], np.ndarray):
                                obs_dim = next_states[0].shape[0]
                        # 最后的默认值
                        if obs_dim <= 0:
                            obs_dim = 61  # 默认观察空间维度
                    # 使用零向量作为默认观察
                    next_states.append(np.zeros(obs_dim))
            
            # 记录这一步的信息
            step_info = {}
            
            # 记录每个智能体的位置 - 确保每个时间步都记录
            agent_positions = []
            for i, agent in enumerate(self.env.world.agents):
                if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                    pos = agent.state.p_pos
                    # 始终复制位置数据以避免引用问题
                    pos_copy = pos.copy()
                    agent_positions.append(pos_copy)
                    
                    # 记录位置到轨迹中 - 添加安全检查
                    if i < len(trajectories) and traj_indices[i] < len(trajectories[i]):
                        trajectories[i][traj_indices[i]] = pos_copy
                        traj_indices[i] += 1
                    else:
                        print(f"警告: 轨迹索引超出范围 - 智能体 {i}, 索引 {traj_indices[i] if i < len(traj_indices) else 'N/A'}")
                    
                    # 记录位置到step_info中
                    step_info[f'agent_{i}_pos'] = pos_copy
                    
                    # 减少日志输出频率
                    if not fast_mode and step % log_interval == 0 and DEBUG_OUTPUT:
                        print(f"步数 {step}, 智能体 {i} 位置: {pos}")
            
            # 每20步比较一次所有智能体位置，检查是否存在完全相同的位置
            if step % 20 == 0 and len(agent_positions) >= 3:
                print(f"\n步数 {step}, 智能体位置比较:")
                print(f"智能体0位置: {agent_positions[0]}")
                print(f"智能体1位置: {agent_positions[1]}")
                print(f"智能体2位置: {agent_positions[2]}")
                
                # 检查位置是否相同
                pos_same_01 = np.allclose(agent_positions[0], agent_positions[1], atol=1e-5)
                pos_same_02 = np.allclose(agent_positions[0], agent_positions[2], atol=1e-5)
                pos_same_12 = np.allclose(agent_positions[1], agent_positions[2], atol=1e-5)
                
                print(f"位置比较结果:")
                print(f"  智能体0和智能体1位置相同: {pos_same_01}")
                print(f"  智能体0和智能体2位置相同: {pos_same_02}")
                print(f"  智能体1和智能体2位置相同: {pos_same_12}")
            
            info.append(step_info)
            reward_episode += sum(rewards)
            
            # 处理Pygame事件（仅在非快速模式且渲染开启时）
            if render and not fast_mode:
                try:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return [traj[:idx] for traj, idx in zip(trajectories, traj_indices)], reward_episode, info, images
                        
                        # 传递事件到3D查看器
                        if hasattr(self.env, '_3d_viewer'):
                            if hasattr(self.env._3d_viewer, '_handle_event'):
                                self.env._3d_viewer._handle_event(event)
                            elif hasattr(self.env._3d_viewer, 'handle_event'):
                                self.env._3d_viewer.handle_event(event)
                except Exception:
                    pass
            
            # 渲染（仅在非快速模式且渲染开启时）
            if render and not fast_mode:
                try:
                    # 第一帧时，设置固定视角（用于GIF录制）
                    if step == 0 and mode == 'rgb_array' and hasattr(self.env, '_3d_viewer'):
                        self.env._3d_viewer.auto_rotate = False
                        self.env._3d_viewer.camera_angle = 45.0
                        self.env._3d_viewer.camera_height = 8.0
                        self.env._3d_viewer.camera_distance = 15.0
                        print("已设置固定GIF录制视角")
                    
                    # 渲染当前帧（降低渲染频率以提高性能）
                    if step % 5 == 0:  # 每5步渲染一次
                        # 更新3D对象位置
                        if hasattr(self.env, '_update_3d_objects'):
                            self.env._update_3d_objects()
                        elif hasattr(self.env, '_3d_viewer') and hasattr(self.env._3d_viewer, 'update_agent_positions'):
                            self.env._3d_viewer.update_agent_positions()
                        
                        # 根据模式渲染
                        if mode == 'rgb_array':
                            img = self.env.render(mode='rgb_array')
                            if img is not None:
                                images.append(img.copy())
                        else:
                            self.env.render(mode='human')
                except Exception as e:
                    if not fast_mode:
                        print(f'渲染失败: {e}')
            
            states = next_states
            
            # 日志：每N步记录一次轨迹长度
            if not fast_mode and step > 0 and step % 100 == 0 and DEBUG_OUTPUT:
                print(f"步数 {step}: 轨迹长度 = {traj_indices}")
            
            # 调用可视化回调（如果提供）
            if visualization_callback is not None:
                # 传递当前步数、轨迹和场景
                # 为回调创建当前轨迹快照
                current_trajectories = [traj[:idx] for traj, idx in zip(trajectories, traj_indices)]
                # 调用回调
                stop_requested = visualization_callback(step, current_trajectories, self.env.world)
                # 检查回调是否请求停止执行
                if stop_requested:
                    print(f"用户请求在步数 {step} 停止执行")
                    break
            
            # 检查是否所有智能体都达到完成状态
            if all(dones):
                if not fast_mode:
                    print(f"所有智能体在步数 {step} 完成任务，提前结束回合")
                break
        
        # 裁剪预分配的轨迹数组到实际使用的长度
        final_trajectories = [traj[:idx] for traj, idx in zip(trajectories, traj_indices)]
        
        # 打印轨迹信息
        for i, traj in enumerate(final_trajectories):
            print(f"智能体 {i} 轨迹点数量: {len(traj)}")
            if len(traj) >= 2:
                print(f"  起点: {traj[0]}")
                print(f"  终点: {traj[-1]}")
                print(f"  移动距离: {np.linalg.norm(np.array(traj[-1]) - np.array(traj[0]))}")
        
        if images and not fast_mode:
            print(f"收集了 {len(images)} 帧图像，准备生成GIF")
        
        return final_trajectories, reward_episode, info, images

    def plot_3d_trajectory(self, info, output_path=None):
        """绘制3D轨迹图并保存"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            
            # 检查输入数据
            if not info or len(info) == 0:
                print("警告: 没有轨迹信息可用于绘图")
                return None
            
            print(f"开始绘制3D轨迹图，数据点数: {len(info)}")
            
            # 避免中文字体问题，直接使用英文
            plt.rcParams['font.sans-serif'] = ['Arial']
            # 使用ASCII负号替代Unicode负号
            plt.rcParams['axes.unicode_minus'] = False
            
            # 提取轨迹数据
            trajectories = []
            for i in range(self.n_agents):
                traj = []
                for step_info in info:
                    if 'positions' in step_info and i < len(step_info['positions']) and step_info['positions'][i] is not None:
                        traj.append(step_info['positions'][i])
                    elif 'positions' not in step_info:
                        print(f"警告: 步骤 {step_info.get('step', '未知')} 没有位置信息")
                
                if traj:
                    trajectories.append(traj)
                else:
                    print(f"警告: 智能体 {i} 没有可用轨迹数据")
            
            if not trajectories:
                print("错误: 没有可用的轨迹数据")
                return None
            
            print(f"提取了 {len(trajectories)} 个智能体的轨迹，每个轨迹的点数: {[len(t) for t in trajectories]}")
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 定义颜色列表
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            
            # 绘制智能体轨迹
            for i, trajectory in enumerate(trajectories):
                if not trajectory or len(trajectory) < 2:
                    print(f"  警告: 智能体 {i} 的轨迹为空或不足两个点，跳过绘制")
                    continue
                
                # 绘制轨迹数据
                x = [p[0] for p in trajectory if p is not None]
                y = [p[1] for p in trajectory if p is not None]
                z = [p[2] if len(p) > 2 else 0 for p in trajectory if p is not None]
                
                # 绘制轨迹
                color = colors[i % len(colors)]
                ax.plot(x, y, z, color=color, label=f'Agent {i}', linewidth=2)
                
                # 标记起点和终点
                ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=100, label=f'Start {i}')
                ax.scatter(x[-1], y[-1], z[-1], color=color, marker='x', s=100, label=f'End {i}')
            
            # 设置轴标签和标题
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Agent Trajectories')
            
            # 设置图例
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # 保存图像
            if output_path:
                plt.savefig(output_path)
                print(f"3D轨迹图保存至: {output_path}")
            
            return fig
        except Exception as e:
            print(f"绘制3D轨迹图失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __init__(self, env, gamma=0.95, batch_size=1024, 
                actor_learning_rate=0.001, critic_learning_rate=0.002, 
                update_target_period=0.01, noise_std_dev=0.2, 
                is_3d=False, save_dir="./weights",
                continuous_actions=False,
                build_actor_fn=None,
                build_critic_fn=None):
        """
        初始化MADDPG运行器
        
        参数:
            env: 环境实例
            gamma: 折扣因子
            batch_size: 批量大小
            actor_learning_rate: actor网络学习率
            critic_learning_rate: critic网络学习率
            update_target_period: 目标网络更新周期
            noise_std_dev: 探索噪声标准差
            is_3d: 是否为3D环境
            save_dir: 模型保存目录
            continuous_actions: 是否使用连续动作空间
            build_actor_fn: 自定义Actor网络构建函数
            build_critic_fn: 自定义Critic网络构建函数
        """
        self.env = env
        self.save_dir = save_dir
        # 添加维度调试变量
        self.debug_dimension_mismatch = True  # 是否输出维度不匹配信息
        self.observation_dims_history = []  # 记录观察空间维度变化
        self.expected_input_dims = []  # 记录网络期望的输入维度
        self.dimension_adjustments_count = 0  # 记录维度调整次数
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.tau = update_target_period
        self.noise_std_dev = noise_std_dev
        self.continuous_actions = continuous_actions
        self.build_actor_fn = build_actor_fn
        self.build_critic_fn = build_critic_fn
        
        # 检测是否为3D环境
        self.is_3d = is_3d or env.world.dim_p == 3
        if env.world.dim_p == 3:
            print("检测到3D环境，自动设置is_3d=True")
            self.is_3d = True
            
        # 获取环境信息
        self.n_agents = len(env.world.agents)
        self.dim_p = env.world.dim_p
        
        print(f"环境: 维度={self.dim_p}D, 智能体数量={self.n_agents}")
        if self.continuous_actions:
            print("使用连续动作空间")
        else:
            print("使用离散动作空间")
        
        # 计算最大回合
        self.max_episode_len = 200
        
        # 创建maddpg智能体
        self.agents = []
        for i in range(self.n_agents):
            print(f"创建智能体 {i}...")
            agent = MADDPGAgent(
                env=env, 
                agent_index=i, 
                gamma=self.gamma,
                tau=self.tau,
                critic_lr=self.critic_lr,
                actor_lr=self.actor_lr,
                noise_std_dev=self.noise_std_dev,
                buffer_size=1000000,
                batch_size=self.batch_size,
                is_3d=self.is_3d,
                continuous_actions=self.continuous_actions,
                build_actor_fn=self.build_actor_fn,
                build_critic_fn=self.build_critic_fn
            )
            self.agents.append(agent)
        
        # 记录最好的奖励
        self.best_score = -np.inf
        self.best_average_score = -np.inf
        
        # 确保权重目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 内存监控阈值（MB）
        self.memory_warning_threshold = 4000  # 4GB
        
        # 检测操作系统
        self.is_windows = platform.system() == 'Windows'
        if self.is_windows:
            print("检测到Windows系统，将使用兼容的性能优化方法")
        
        # 预先检查必要的库是否可用
        self._check_required_libraries()
    
    def _check_required_libraries(self):
        """检查和导入所需的库"""
        # 检查psutil库
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            print("警告: psutil库未安装，内存监控功能将被禁用")
            self.psutil_available = False
        
        # 检查其他可能需要的库
        try:
            import matplotlib
            self.matplotlib_available = True
        except ImportError:
            print("警告: matplotlib库未安装，可视化功能将被禁用")
            self.matplotlib_available = False
            
        # 检查CPU支持
        try:
            import multiprocessing
            self.cpu_count = multiprocessing.cpu_count()
            print(f"检测到{self.cpu_count}个CPU核心，将优化CPU性能")
        except:
            self.cpu_count = 1
            print("无法检测CPU核心数，将使用单线程模式")
    
    def _clean_memory(self, aggressive=False):
        """清理内存"""
        try:
            # 标准清理
            tf.keras.backend.clear_session()
            gc.collect()
            
            if aggressive:
                # 更激进的清理
                # 强制执行Python的垃圾回收
                gc.collect(2)
                
                # 在Linux系统上尝试释放未使用的内存回操作系统
                if not self.is_windows:
                    try:
                        import ctypes
                        libc = ctypes.CDLL('libc.so.6')
                        libc.malloc_trim(0)
                    except Exception as e:
                        print(f"在Linux上释放内存时出错: {e}")
            
            print("内存清理完成")
        except Exception as e:
            print(f"清理内存时出错: {e}")
    
    def _get_memory_usage(self):
        """获取当前进程的内存使用情况（MB）"""
        try:
            # 使用安全的方式调用psutil
            # 直接引用全局导入的psutil模块
            import psutil as ps  # 使用别名避免与全局psutil冲突
            process = ps.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # 转换为MB
        except ImportError:
            print("警告: psutil库未安装，无法获取内存使用信息")
            return 0  # 返回0表示无法获取内存信息
        except Exception as e:
            print(f"获取内存使用信息时出错: {e}")
            return 0  # 返回0表示无法获取内存信息
    
    def _check_and_handle_high_memory(self):
        """检查并处理高内存使用情况"""
        # 检查psutil是否可用
        if not hasattr(self, 'psutil_available') or not self.psutil_available:
            # 如果psutil不可用，直接执行普通清理
            self._clean_memory(aggressive=False)
            return False
            
        mem_usage = self._get_memory_usage()
        if mem_usage > self.memory_warning_threshold:
            print(f"警告: 内存使用量高 ({mem_usage:.2f} MB)，执行激进清理...")
            self._clean_memory(aggressive=True)
            return True
        return False
    
    def save_agents(self, suffix=""):
        """保存所有智能体的模型权重"""
        try:
            # 确保目录存在
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                
            print(f"保存智能体模型到 {self.save_dir}")
            for i, agent in enumerate(self.agents):
                try:
                    agent_dir = os.path.join(self.save_dir, f"agent_{i}{suffix}")
                    if not os.path.exists(agent_dir):
                        os.makedirs(agent_dir)
                    
                    # 保存模型
                    agent.save_models(suffix=f"_{i}{suffix}")
                    print(f"保存智能体 {i} 模型成功")
                except Exception as e:
                    print(f"保存智能体 {i} 模型失败: {e}")
        except Exception as e:
            print(f"保存智能体模型失败: {e}")
    
    def load_agents(self, suffix=""):
        """加载所有智能体的模型权重"""
        try:
            print(f"加载智能体模型从 {self.save_dir}")
            all_success = True  # 添加标志跟踪所有智能体的加载状态
            for i, agent in enumerate(self.agents):
                try:
                    success = agent.load_models(suffix=f"_{i}{suffix}")
                    if success:
                        print(f"加载智能体 {i} 模型成功")
                    else:
                        print(f"加载智能体 {i} 模型失败")
                        all_success = False
                except Exception as e:
                    print(f"加载智能体 {i} 模型失败: {e}")
                    all_success = False
            return all_success  # 返回整体加载状态
        except Exception as e:
            print(f"加载智能体模型失败: {e}")
            return False
    
    def train(self, num_episodes=1200, num_steps=1200, batch_size=2048, render=False, render_freq=10, event_handling=False, visualization_callback=None, viz_interval=20):
        """训练智能体
        
        Args:
            num_episodes: 训练回合数
            num_steps: 每回合最大步数
            batch_size: 批次大小
            render: 是否渲染环境
            render_freq: 渲染频率，每render_freq回合渲染一次
            event_handling: 是否处理事件
            visualization_callback: 可视化回调函数，接收(episode, step, trajectories, scenario)参数
            viz_interval: 可视化更新间隔步数
        """
        # 初始化奖励记录
        rewards = []
        avg_rewards = []
        
        # 获取场景对象以便操作3D渲染
        scenario = None
        if self.is_3d and hasattr(self.env, 'world') and hasattr(self.env.world, 'scenario'):
            scenario = self.env.world.scenario
            print("场景对象已保存用于3D渲染")
        
        # 记录训练信息
        info = {
            'best_reward': -np.inf,
            'best_avg_reward': -np.inf,
            'best_reward_episode': 0,
            'best_avg_reward_episode': 0,
            'consecutive_no_improve': 0  # 添加计数器跟踪连续无改进回合
        }
        
        # 添加学习率衰减
        initial_lr_actor = 0.0003  # 初始演员网络学习率
        initial_lr_critic = 0.0005  # 初始评论家网络学习率
        min_lr = 0.00001  # 最低学习率
        
        # 添加探索噪声衰减 - 修改这里增加探索能力
        initial_noise = 0.3  # 从1.0降低到0.3
        final_noise = 0.05   # 从0.1降低到0.05
        
        # 导入tqdm进行进度显示
        import tqdm
        
        # 创建单一进度条，总回合数为用户指定的回合数
        print(f"开始训练，总回合数: {num_episodes}")
        progress_bar = tqdm.tqdm(range(num_episodes), desc="训练")
        
        # 直接训练指定回合数
        for episode in progress_bar:
            # 设置当前回合的学习率衰减和噪声衰减 - 减缓衰减速度
            progress = min(1.0, episode / (num_episodes * 0.9))  # 从80%增加到90%，减缓衰减
            
            # 学习率衰减 - 使用平滑的余弦衰减
            current_lr_actor = max(min_lr, initial_lr_actor * (0.5 * (1 + np.cos(np.pi * progress))))
            current_lr_critic = max(min_lr, initial_lr_critic * (0.5 * (1 + np.cos(np.pi * progress))))
            
            # 噪声衰减 - 改为更慢的衰减函数
            current_noise = max(final_noise, initial_noise * (1 - progress**2))  # 使用二次函数减缓衰减
            
            # 更新每个智能体的学习率和噪声水平
            for agent in self.agents:
                if hasattr(agent, '_actor_opt') and hasattr(agent._actor_opt, 'learning_rate'):
                    agent._actor_opt.learning_rate.assign(current_lr_actor)
                if hasattr(agent, '_critic_opt') and hasattr(agent._critic_opt, 'learning_rate'):
                    agent._critic_opt.learning_rate.assign(current_lr_critic)
                if hasattr(agent, '_noise') and hasattr(agent._noise, 'set_std_dev'):
                    agent._noise.set_std_dev(current_noise)
            
            # 重置环境
            states = self.env.reset()
            episode_reward = 0
            
            # 保存智能体轨迹 - 改为与run_episode相同的预分配方式，确保完整记录
            trajectories = []
            # 预分配足够大的轨迹数组
            for i in range(self.n_agents):
                agent_trajectory = [None] * (num_steps + 1)
                # 记录初始位置
                if hasattr(self.env.world.agents[i], 'state'):
                    agent_trajectory[0] = self.env.world.agents[i].state.p_pos.copy()
                trajectories.append(agent_trajectory)
            
            # 轨迹索引
            traj_indices = [1] * self.n_agents
            
            # 执行回合
            for step in range(num_steps):
                # 每10步更新一次进度条信息
                if step % 10 == 0:
                    progress_bar.set_description(f"E:{episode+1}/{num_episodes} S:{step}/{num_steps}")
                
                # 每个智能体选择动作
                actions = []
                for i, agent in enumerate(self.agents):
                    # 使用策略网络选择动作
                    agent_obs = states[i]
                    
                    # 确保输入是正确形状
                    try:
                        if isinstance(agent_obs, np.ndarray) and agent_obs.ndim == 1:
                            action = agent.policy(agent_obs, add_noise=True)
                        else:
                            action = agent.policy(agent_obs, add_noise=True)
                        
                        # 确保action是NumPy数组
                        if isinstance(action, tf.Tensor):
                            action_np = action.numpy()
                        else:
                            action_np = np.array(action)
                        
                        actions.append(action_np)
                    except Exception as e:
                        print(f"获取智能体{i}动作时出错: {e}")
                        # 提供默认动作防止崩溃
                        action_dim = self.env.world.dim_p
                        actions.append(np.zeros(action_dim))
                
                # 执行动作
                next_states, reward, done, _ = self.env.step(actions)
                
                # 记录每个智能体的位置 - 确保每个时间步都记录
                for i, agent in enumerate(self.env.world.agents):
                    if hasattr(agent, 'state'):
                        pos = agent.state.p_pos
                        # 始终复制位置数据以避免引用问题
                        pos_copy = pos.copy()
                        
                        # 记录位置到轨迹中
                        trajectories[i][traj_indices[i]] = pos_copy
                        traj_indices[i] += 1
                
                # 调用可视化回调（如果提供）
                if visualization_callback is not None and step % viz_interval == 0:
                    # 创建当前轨迹快照用于可视化
                    current_trajectories = [traj[:idx] for traj, idx in zip(trajectories, traj_indices)]
                    # 调用回调
                    stop_requested = visualization_callback(episode, step, current_trajectories, self.env.world)
                    # 检查回调是否请求停止执行
                    if stop_requested:
                        print(f"用户请求在回合 {episode}, 步数 {step} 停止训练")
                        # 保存当前模型
                        for agent in self.agents:
                            if hasattr(agent, 'save_models'):
                                agent.save_models(suffix=f'interrupted_ep{episode}')
                        return rewards, avg_rewards, info
                
                # 处理Pygame事件以保持窗口响应
                if render and (episode % render_freq == 0):
                    try:
                        import pygame
                        
                        # 确保3D对象位置已更新
                        if hasattr(self.env, '_update_3d_objects'):
                            self.env._update_3d_objects()
                        elif hasattr(self.env, '_3d_viewer') and hasattr(self.env._3d_viewer, 'update_agent_positions'):
                            self.env._3d_viewer.update_agent_positions()

                        # 处理所有待处理事件，确保响应性
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return rewards, avg_rewards, info
                            elif event.type == pygame.KEYDOWN:
                                # ESC键退出
                                if event.key == pygame.K_ESCAPE:
                                    pygame.quit()
                                    return rewards, avg_rewards, info
                            
                            # 将事件传递给3D查看器处理
                            if hasattr(self.env, '_3d_viewer'):
                                self.env._3d_viewer.handle_event(event)
                        
                        # 渲染环境
                        self.env.render()
                        
                        # 使用时间延迟确保渲染可见，并允许事件处理
                        pygame.time.delay(10)  # 减小延迟到10毫秒，提高训练速度
                        
                    except Exception as e:
                        print(f"渲染失败: {e}")
                
                # 无论是否在渲染回合，都需要定期处理事件，确保UI响应
                if step % 10 == 0:  # 每10步处理一次事件，即使不渲染
                    try:
                        import pygame
                        pygame.event.pump()  # 处理所有待处理事件，保持窗口响应
                    except Exception:
                        pass
                
                # 更新状态
                states = next_states
                
                # 累积奖励
                episode_reward += sum(reward)
                
                # 添加小延迟确保交互充分，但减少到最小值
                time.sleep(0.003)  # 从0.005秒减少到0.003秒，进一步减少延迟
                
                # 如果回合结束，则跳出循环
                if all(done):
                    break
            
            # 裁剪预分配的轨迹数组到实际使用的长度
            final_trajectories = [traj[:idx] for traj, idx in zip(trajectories, traj_indices)]
            
            # 记录回合奖励
            rewards.append(episode_reward)
            # 计算平均奖励（最近100个回合或所有回合）
            window_size = min(100, len(rewards))
            avg_reward = np.mean(rewards[-window_size:])
            avg_rewards.append(avg_reward)
            
            # 更新进度条 - 更精简的格式，但包含更多信息
            progress_bar.set_postfix({
                'R': f'{episode_reward:.1f}',  # 当前奖励，精简格式
                'Avg': f'{avg_reward:.1f}',    # 平均奖励，精简格式
                'Best': f'{info["best_reward"]:.1f}',  # 最佳奖励，精简格式
                'N': f'{current_noise:.2f}',   # 当前噪声水平，精简格式
                'LR': f'{current_lr_actor:.2e}' # 当前学习率，使用科学计数法更精简
            })
            
            # 更新最佳奖励记录
            best_reward_updated = False
            best_avg_updated = False
            
            if episode_reward > info['best_reward']:
                info['best_reward'] = episode_reward
                info['best_reward_episode'] = episode
                best_reward_updated = True
                # 保存最佳模型
                for agent in self.agents:
                    if hasattr(agent, 'cache_best_single'):
                        agent.cache_best_single()
            
            # 更新最佳平均奖励记录
            if avg_reward > info['best_avg_reward']:
                info['best_avg_reward'] = avg_reward
                info['best_avg_reward_episode'] = episode
                best_avg_updated = True
                # 保存最佳平均奖励模型
                for agent in self.agents:
                    if hasattr(agent, 'cache_best_average'):
                        agent.cache_best_average()
            
            # 检查是否连续无改进
            if best_reward_updated or best_avg_updated:
                info['consecutive_no_improve'] = 0
            else:
                info['consecutive_no_improve'] += 1
            
            # 如果连续50个回合没有改进，增大探索噪声
            if info['consecutive_no_improve'] >= 50 and info['consecutive_no_improve'] % 50 == 0:
                # 临时增加噪声来跳出局部最优
                temp_noise = min(1.2, current_noise * 3.0)  # 临时将噪声增大3倍，最大不超过1.2
                
                # 只在达到里程碑时输出一条信息，减少输出量
                if info['consecutive_no_improve'] % 100 == 0:
                    print(f"\n连续{info['consecutive_no_improve']}回合无改进，临时增大噪声至{temp_noise:.4f}")
                
                # 应用临时噪声
                for agent in self.agents:
                    if hasattr(agent, '_noise') and hasattr(agent._noise, 'set_std_dev'):
                        agent._noise.set_std_dev(temp_noise)
            
            # 定期保存模型
            if episode % 50 == 0 or episode == num_episodes - 1:
                for agent in self.agents:
                    if hasattr(agent, 'save_models'):
                        agent.save_models(suffix='last')
                        
                # 每100个回合打印一次当前训练状态 - 减少输出频率
                if episode % 200 == 0: # 从100改为200
                    print(f"\n当前训练状态 [回合 {episode}/{num_episodes}]:")
                    print(f"  当前奖励: {episode_reward:.2f}")
                    print(f"  最近平均奖励: {avg_reward:.2f}")
                    print(f"  最佳奖励: {info['best_reward']:.2f} (回合 {info['best_reward_episode']})")
                    print(f"  最佳平均奖励: {info['best_avg_reward']:.2f} (回合 {info['best_avg_reward_episode']})")
                    print(f"  连续无改进回合: {info['consecutive_no_improve']}")
                    print(f"  当前学习率: {current_lr_actor:.6f}")
                    print(f"  当前噪声水平: {current_noise:.4f}")
            
            # 定期清理内存
            if episode % 50 == 0:
                self._clean_memory()
                
            # 确保每个回合结束时正确处理pygame事件，防止窗口无响应
            if render and (episode % render_freq == 0):
                try:
                    import pygame
                    pygame.event.pump()  # 处理所有待处理事件，保持窗口响应
                except:
                    pass
        
        # 关闭进度条
        progress_bar.close()
        
        # 完成训练后保存最终模型
        for agent in self.agents:
            if hasattr(agent, 'save_models'):
                agent.save_models(suffix='last')
        
        print("训练完成!")
        print(f"最佳回合奖励: {info['best_reward']:.2f} (回合 {info['best_reward_episode']})")
        print(f"最佳平均奖励: {info['best_avg_reward']:.2f} (回合 {info['best_avg_reward_episode']})")
        
        return rewards, avg_rewards, info
    
    def _enhance_2d_image(self, img, step, max_steps):
        """增强2D图像，添加更多信息"""
        # 添加步数信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Step: {step}/{max_steps}"
        cv2.putText(img, text, (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # 添加虚拟视角提示
        cv2.putText(img, "3D View", (img.shape[1]-100, 30), font, 0.7, (255, 255, 255), 2)
        
        return img
    
    def _save_trajectory_plot(self, trajectories, obstacle_positions=None, goal_position=None, episode=None, output_dir=None):
        """保存轨迹图，包括障碍物和目标位置"""
        if not trajectories:
            print("轨迹为空，无法保存轨迹图")
            return
            
        # 打印轨迹信息以便调试
        print("轨迹信息:")
        for i, traj in enumerate(trajectories):
            if i < 3:  # 只显示前3个智能体
                if not traj:
                    print(f"  智能体 {i} 轨迹为空")
                else:
                    print(f"  智能体 {i} 轨迹点数量: {len(traj)}")
        
        # 确保轨迹列表至少有三个元素（对应三个智能体）
        while len(trajectories) < 3:
            trajectories.append([])
            
        # 如果所有轨迹都为空，返回
        if all(not t for t in trajectories):
            print("所有轨迹都为空，无法保存轨迹图")
            return
        
        try:
            # 避免中文字体问题，直接使用英文
            plt.rcParams['font.sans-serif'] = ['Arial']
            # 使用ASCII负号替代Unicode负号
            plt.rcParams['axes.unicode_minus'] = False
            
            # 确定输出目录
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "outputs")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            episode_str = f"_ep{episode}" if episode is not None else ""
            filename = f"trajectory_plot_{timestamp}{episode_str}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 获取环境中的障碍物和目标位置
            if obstacle_positions is None and hasattr(self.env.world, 'landmarks'):
                obstacle_positions = []
                for landmark in self.env.world.landmarks:
                    if hasattr(landmark, 'name') and 'obstacle' in landmark.name:
                        obstacle_positions.append(landmark.state.p_pos.copy())
                        print(f"找到障碍物: {landmark.name} 位置: {landmark.state.p_pos}")
                    elif hasattr(landmark, 'name') and 'target' in landmark.name:
                        goal_position = landmark.state.p_pos.copy()
                        print(f"找到目标: {landmark.name} 位置: {landmark.state.p_pos}")
            
            # 绘制智能体轨迹
            for i in range(3):  # 强制处理所有3个智能体
                trajectory = trajectories[i] if i < len(trajectories) else []
                
                if not trajectory or len(trajectory) < 2:
                    print(f"  警告: 智能体 {i} 的轨迹为空或不足两个点，跳过绘制")
                    continue
                
                # 确保数据是浮点数
                x = [float(point[0]) for point in trajectory if point is not None]
                y = [float(point[1]) for point in trajectory if point is not None]
                z = [float(point[2]) if len(point) > 2 else 0 for point in trajectory if point is not None]
                
                if not x or not y or not z:
                    print(f"  警告: 智能体 {i} 的轨迹中存在无效点，跳过绘制")
                    continue
                
                # 使用鲜明颜色
                line_color = ['red', 'blue', 'green', 'purple', 'orange'][i % 5]
                
                # 绘制完整轨迹线
                ax.plot(x, y, z, color=line_color, label=f'Agent {i}', linewidth=3.5)
                
                # 标记起点和终点
                ax.scatter(x[0], y[0], z[0], color='green', s=150, marker='^', label='Start' if i == 0 else "")
                ax.scatter(x[-1], y[-1], z[-1], color='red', s=150, marker='o', label='End' if i == 0 else "")
            
            # 绘制障碍物和目标
            if obstacle_positions:
                for i, pos in enumerate(obstacle_positions):
                    pos_x = float(pos[0])
                    pos_y = float(pos[1])
                    pos_z = float(pos[2]) if len(pos) > 2 else 0
                    ax.scatter(pos_x, pos_y, pos_z, 
                              color='black', s=300, marker='o', alpha=0.7,
                              label='Obstacle' if i == 0 else "")
            
            if goal_position is not None:
                goal_x = float(goal_position[0])
                goal_y = float(goal_position[1])
                goal_z = float(goal_position[2]) if len(goal_position) > 2 else 0
                ax.scatter(goal_x, goal_y, goal_z,
                          color='gold', s=400, marker='*', label='Goal')
            
            # 设置图形属性
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(f'Agent Trajectories (Episode {episode})' if episode else 'Agent Trajectories')
            
            # 调整轴比例和视角
            ax.view_init(elev=30, azim=45)
            
            # 添加图例
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # 保存图像
            plt.savefig(filepath)
            print(f"轨迹图已保存到: {filepath}")
            
            return fig
        except Exception as e:
            print(f"绘制轨迹图失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_test(self):
        """运行测试"""
        # 运行一个回合（用于测试）
        trajectories, reward_episode, info, images = self.run_episode(max_episode_len=self.max_episode_len, policy_param='last', render=True, test_mode=True)
        
        # 测试部分
        print("开始测试最终性能...")
        print("测试期间您可以使用以下控制:")
        print("- WASDQE键: 移动相机位置")
        print("- 方向键: 旋转相机视角")
        print("- 鼠标拖动: 旋转视角")
        print("- R键: 开关自动旋转")
        print("- 空格键: 重置视角")
        
        return trajectories, reward_episode, info, images

    def _update_policy(self, states, rewards, next_states, dones):
        """更新所有智能体的策略"""
        print(f"执行策略更新 - 批次大小: {len(states)}")
        # 收集所有代理的当前动作和下一个动作
        current_actions = []
        next_actions = []
        critic_losses = []
        actor_losses = []

        # 定义本地版本的get_agent_action函数
        def local_get_agent_action(agent, obs, add_noise=False):
            """本地版本的get_agent_action函数
            
            Args:
                agent: 智能体实例
                obs: 观察值
                add_noise: 是否添加噪声
                
            Returns:
                action: 智能体选择的动作
            """
            try:
                # 检查policy方法的参数
                if hasattr(agent, 'policy'):
                    import inspect
                    try:
                        policy_params = inspect.signature(agent.policy).parameters
                        if 'add_noise' in policy_params:
                            # 使用add_noise参数接口
                            return agent.policy(obs, add_noise=add_noise)
                        else:
                            # 无参数接口，直接调用
                            return agent.policy(obs)
                    except (ValueError, TypeError):
                        # signature获取失败，直接调用
                        return agent.policy(obs)
                elif hasattr(agent, 'act'):
                    # 使用act方法
                    try:
                        import inspect
                        act_params = inspect.signature(agent.act).parameters
                        if 'add_noise' in act_params:
                            return agent.act(obs, add_noise=add_noise)
                        else:
                            return agent.act(obs)
                    except (ValueError, TypeError):
                        # signature获取失败，直接调用
                        return agent.act(obs)
                elif hasattr(agent, 'get_action'):
                    # 尝试使用get_action方法
                    return agent.get_action(obs)
                elif hasattr(agent, 'select_action'):
                    # 尝试使用select_action方法
                    return agent.select_action(obs)
                else:
                    # 没有找到合适的方法
                    raise AttributeError("智能体没有可用的动作选择方法")
            except Exception as e:
                print(f"获取智能体动作时出错: {str(e)}")
                # 返回零向量作为默认动作
                if isinstance(obs, np.ndarray):
                    if len(obs.shape) > 1:
                        # 如果观察是批次形式，返回相同形状的零动作
                        return np.zeros((obs.shape[0], 7))
                    else:
                        # 单个观察
                        return np.zeros(7)
                else:
                    # 默认返回7维零向量（连续动作空间）
                    return np.zeros(7)

        # 获取所有智能体的下一个动作
        for i, agent in enumerate(self.agents):
            try:
                # 获取单个智能体的观察和下一个观察
                agent_obs = states[i]
                agent_next_obs = next_states[i]
                
                # 确保观察值有批次维度
                if isinstance(agent_obs, np.ndarray) and len(agent_obs.shape) == 1:
                    agent_obs = np.expand_dims(agent_obs, axis=0)
                if isinstance(agent_next_obs, np.ndarray) and len(agent_next_obs.shape) == 1:
                    agent_next_obs = np.expand_dims(agent_next_obs, axis=0)
                
                # 转换为张量
                agent_obs_tensor = tf.convert_to_tensor(agent_obs, dtype=tf.float32)
                agent_next_obs_tensor = tf.convert_to_tensor(agent_next_obs, dtype=tf.float32)
                
                # 使用目标策略网络预测下一个动作 - 不添加噪声
                if hasattr(agent, 'target_policy'):
                    next_action = agent.target_policy(agent_next_obs_tensor, add_noise=False)
                else:
                    # 使用适配器函数
                    next_action = local_get_agent_action(agent, agent_next_obs_tensor, add_noise=False)
                
                # 确保next_action是numpy数组并且具有正确的形状 (batch_size, action_dim)
                if isinstance(next_action, tf.Tensor):
                    next_action = next_action.numpy()
                
                # 检查维度并确保有批次维度
                if len(next_action.shape) == 1:  # 如果只有一维 (action_dim,)
                    next_action = np.expand_dims(next_action, axis=0)  # 变为 (1, action_dim)
                
                next_actions.append(next_action)

                # 当前动作从策略网络获取
                if hasattr(agent, 'policy'):
                    try:
                        import inspect
                        if 'add_noise' in inspect.signature(agent.policy).parameters:
                            current_action = agent.policy(agent_obs_tensor, add_noise=False)
                        else:
                            current_action = agent.policy(agent_obs_tensor)
                    except (ValueError, TypeError):
                        # signature获取失败，直接调用
                        current_action = agent.policy(agent_obs_tensor)
                else:
                    # 使用适配器函数
                    current_action = local_get_agent_action(agent, agent_obs_tensor, add_noise=False)
                
                # 确保current_action是numpy数组并且具有正确的形状 (batch_size, action_dim)
                if isinstance(current_action, tf.Tensor):
                    current_action = current_action.numpy()
                
                # 检查维度并确保有批次维度
                if len(current_action.shape) == 1:  # 如果只有一维 (action_dim,)
                    current_action = np.expand_dims(current_action, axis=0)  # 变为 (1, action_dim)
                
                current_actions.append(current_action)
            except Exception as e:
                print(f"获取智能体动作时出错: {e}")
                import traceback
                traceback.print_exc()
                # 使用零向量作为默认动作，确保形状正确: (batch_size, action_dim)
                action_dim = 7  # 3D环境中是7维: 3维动作 + 4维势场力参数
                batch_size = 1  # 默认批次大小
                
                # 尝试获取批次大小
                if isinstance(states, list) and i < len(states):
                    if isinstance(states[i], np.ndarray) and len(states[i].shape) > 0:
                        if len(states[i].shape) > 1:
                            batch_size = states[i].shape[0]
                
                next_actions.append(np.zeros((batch_size, action_dim)))
                current_actions.append(np.zeros((batch_size, action_dim)))

        # 对每个智能体执行更新
        for i, agent in enumerate(self.agents):
            try:
                # 检查并适配状态维度 - 解决维度不匹配问题
                if isinstance(states[i], np.ndarray) and len(states[i].shape) == 2:
                    batch_size, obs_dim = states[i].shape
                    
                    # 动态检测网络期望的输入维度，而不是硬编码61
                    expected_dim = None
                    
                    # 检查actor_model的输入形状
                    if hasattr(agent, 'actor_model') and hasattr(agent.actor_model, 'input_shape'):
                        # 获取第一个输入层的形状
                        input_shape = agent.actor_model.input_shape
                        if input_shape and len(input_shape) > 1:
                            expected_dim = input_shape[1]  # 特征维度通常是第二个元素
                    
                    # 如果无法从actor_model获取，尝试从actor_network获取
                    if expected_dim is None and hasattr(agent, 'actor_network') and hasattr(agent.actor_network, 'input_shape'):
                        input_shape = agent.actor_network.input_shape
                        if input_shape and len(input_shape) > 1:
                            expected_dim = input_shape[1]
                    
                    # 如果仍然无法获取，使用传统的61维
                    if expected_dim is None:
                        expected_dim = 61  # 默认值，与以前保持一致
                    
                    if obs_dim < expected_dim:
                        print(f"智能体 {i} - 将状态维度从{obs_dim}扩展到{expected_dim} (网络期望的输入维度)")
                        # 创建一个新的数组，填充零值
                        padded_states = np.zeros((batch_size, expected_dim), dtype=np.float32)
                        # 复制原始数据
                        padded_states[:, :obs_dim] = states[i]
                        states_i = padded_states
                    else:
                        states_i = states[i]
                else:
                    states_i = states[i]
                
                # 同样处理next_states
                if isinstance(next_states[i], np.ndarray) and len(next_states[i].shape) == 2:
                    batch_size, obs_dim = next_states[i].shape
                    
                    # 动态检测网络期望的输入维度，而不是硬编码61
                    expected_dim = None
                    
                    # 检查actor_model的输入形状
                    if hasattr(agent, 'actor_model') and hasattr(agent.actor_model, 'input_shape'):
                        # 获取第一个输入层的形状
                        input_shape = agent.actor_model.input_shape
                        if input_shape and len(input_shape) > 1:
                            expected_dim = input_shape[1]  # 特征维度通常是第二个元素
                    
                    # 如果无法从actor_model获取，尝试从actor_network获取
                    if expected_dim is None and hasattr(agent, 'actor_network') and hasattr(agent.actor_network, 'input_shape'):
                        input_shape = agent.actor_network.input_shape
                        if input_shape and len(input_shape) > 1:
                            expected_dim = input_shape[1]
                    
                    # 如果仍然无法获取，使用传统的61维
                    if expected_dim is None:
                        expected_dim = 61  # 默认值，与以前保持一致
                    
                    if obs_dim < expected_dim:
                        print(f"智能体 {i} - 将下一个状态维度从{obs_dim}扩展到{expected_dim} (网络期望的输入维度)")
                        # 创建一个新的数组，填充零值
                        padded_next_states = np.zeros((batch_size, expected_dim), dtype=np.float32)
                        # 复制原始数据
                        padded_next_states[:, :obs_dim] = next_states[i]
                        next_states_i = padded_next_states
                    else:
                        next_states_i = next_states[i]
                else:
                    next_states_i = next_states[i]
                
                # 创建更新所需的数据
                agent_data = (
                    states_i, 
                    current_actions[i][0] if len(current_actions[i]) == 1 else current_actions[i], 
                    rewards[i], 
                    next_states_i, 
                    dones[i],
                    next_actions[i]  # 只传递当前智能体的下一个动作，而不是所有智能体的动作列表
                )
                
                # 执行策略更新 - 打印调试信息
                critic_loss, actor_loss = agent.update(*agent_data)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                
                # 打印关键调试信息
                print(f"智能体 {i} 更新 - Critic损失: {critic_loss:.6f}, Actor损失: {actor_loss:.6f}")
                
                # 软更新目标网络
                try:
                    if hasattr(agent, 'target_actor') and hasattr(agent, 'actor'):
                        agent.update_target(agent.target_actor.variables, agent.actor.variables)
                        agent.update_target(agent.target_critic.variables, agent.critic.variables)
                    elif hasattr(agent, 'update_target_networks'):
                        # 使用agent自己的更新方法
                        agent.update_target_networks()
                    elif hasattr(agent, '_update_target_networks'):
                        # 调用内部方法
                        agent._update_target_networks()
                    else:
                        print(f"警告: 智能体 {i} 没有目标网络更新方法")
                except Exception as e:
                    print(f"智能体 {i} 更新失败: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"智能体 {i} 更新失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 返回平均损失
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        
        # 打印平均损失
        print(f"平均损失 - Critic: {avg_critic_loss:.6f}, Actor: {avg_actor_loss:.6f}")
        
        return avg_critic_loss, avg_actor_loss
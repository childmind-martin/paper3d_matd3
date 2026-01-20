# Adopted from https://keras.io/examples/rl/ddpg_pendulum/

import numpy as np
import tensorflow as tf
# 直接导入keras模块
from tensorflow import keras
import os
import time
import pickle
import random
from .nets.actor_network import generate_actor_network
from .nets.critic_network import generate_critic_network
from .util import * 
import tqdm
from .parameter_space_noise import ParameterSpaceNoise

# 尝试导入 gymnasium，如果失败则导入 gym
try:
    import gymnasium as gym
    print("使用 gymnasium 模块")
except ImportError:
    try:
        import gym
        print("使用 gym 模块")
    except ImportError:
        print("警告: 无法导入 gymnasium 或 gym 模块，请安装其中之一")

# 限制TensorFlow内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 设置随机种子以提高可重复性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 设置更加保守的学习率
CRITIC_LR = 0.0005  # 降低评论家网络学习率
ACTOR_LR = 0.0003   # 降低演员网络学习率
NOISE_STDDEV = 0.2  # 降低噪声标准差
TAU = 0.005         # 降低目标网络更新率

# Our basic MADDPG: Currenty assumes access to all other agents actions/policies and observations while training
# Could be interesting to explore their policy estimation and ensemble suggestions
class MADDPGAgent():
    def  __init__(
           self,
           env,
           agent_index,
           gamma=0.95,
           tau=0.01,             # 软更新率
           critic_lr=0.002,      # 评论家（Critic）网络学习率
           actor_lr=0.001,       # 演员（Actor）网络学习率
           noise_std_dev=0.02,   # 动作噪音的标准差
           buffer_size=10e6,     # 经验回放缓冲区大小
           batch_size=1024,      # 批次大小
           is_3d=False,          # 是否为3D环境
           continuous_actions=False,  # 是否使用连续动作空间
           build_actor_fn=None,   # 自定义Actor网络构建函数
           build_critic_fn=None,   # 自定义Critic网络构建函数
           use_param_noise=True,   # 是否使用参数空间噪声
           actor_update_delay=2    # Actor延迟更新频率
           ):
        """初始化MADDPG智能体
        
        Args:
            env: 多智能体环境
            agent_index: 该智能体的索引
            gamma: 折扣因子
            tau: 软更新率（目标网络更新速度）
            critic_lr: 评论家网络学习率
            actor_lr: 演员网络学习率
            noise_std_dev: 动作噪音标准差
            buffer_size: 经验回放缓冲区大小
            batch_size: 训练批次大小
            is_3d: 是否为3D环境
            continuous_actions: 是否使用连续动作空间
            build_actor_fn: 自定义Actor网络构建函数
            build_critic_fn: 自定义Critic网络构建函数
            use_param_noise: 是否使用参数空间噪声
            actor_update_delay: Actor延迟更新频率
        """
        self.env = env
        self.index = agent_index
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.continuous_actions = continuous_actions
        self.actor_update_delay = actor_update_delay
        
        # 训练计数器，用于延迟更新
        self.update_count = 0
        
        # 降噪（参数空间噪声）
        self.use_param_noise = use_param_noise
        
        # 获取观察和动作空间
        self._observation_space = env.observation_space[agent_index]
        self._action_space = env.action_space[agent_index]
        print(f"\n动作空间信息:")
        print(f"类型: {type(self._action_space)}")
        print(f"形状: {self._action_space.shape if hasattr(self._action_space, 'shape') else 'N/A'}")
        print(f"维度: {self._action_space.n if hasattr(self._action_space, 'n') else 'N/A'}")
        print(f"范围: high={self._action_space.high if hasattr(self._action_space, 'high') else 'N/A'}, low={self._action_space.low if hasattr(self._action_space, 'low') else 'N/A'}")
        
        # 创建经验回放缓冲区
        self._replay_buffer = ReplayBuffer(int(buffer_size))
        
        # 创建优化器
        self._actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self._critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        # 计算总的观察和动作维度
        self._total_obs_size = self._observation_space.shape[0]
        self._total_act_size = 0
        if isinstance(self._action_space, gym.spaces.Box):
            self._total_act_size += self._action_space.shape[0]
        elif isinstance(self._action_space, gym.spaces.Discrete):
            self._total_act_size += self._action_space.n
        else:
            try:
                self._total_act_size += len(self._action_space.sample())
            except:
                self._total_act_size += 1
                print(f"警告：无法确定智能体 {agent_index} 的动作空间维度，使用默认值 1")
        
        print(f"智能体 {agent_index} 网络维度:")
        print(f"- 观察空间: {self._total_obs_size}")
        print(f"- 动作空间: {self._total_act_size}")
        
        # 创建网络
        self._create_networks(is_3d)
        
        # 参数空间噪声设置
        if self.use_param_noise:
            self.param_noise = ParameterSpaceNoise(
                initial_stddev=float(self._noise_std_dev),
                desired_action_stddev=float(self._noise_std_dev),
                adoption_coefficient=1.01
            )
            # 创建带参数噪声的Actor网络
            self.actor_perturbed = None
            self._update_perturbed_actor()
        
        # 编译模型
        self._actor_model.compile(optimizer=self._actor_opt)
        self._critic_model.compile(optimizer=self._critic_opt)
        
        # 打印网络结构
        print("\nActor网络结构:")
        self._actor_model.summary()
        print("\nCritic网络结构:")
        self._critic_model.summary()

    def _create_networks(self, is_3d=False):
        """创建DDPG所需的深度神经网络

        根据观察空间和动作空间的维度创建Actor和Critic网络
        """
        # 观察空间维度
        obs_dim = self._observation_space.shape[0]
        
        # 动作空间维度 - 处理不同类型的动作空间
        if isinstance(self._action_space, gym.spaces.Box):
            # 连续动作空间
            if hasattr(self._action_space, 'shape') and hasattr(self._action_space.shape, '__len__'):
                action_dim = self._action_space.shape[0]
            elif hasattr(self._action_space, 'shape') and isinstance(self._action_space.shape, int):
                action_dim = self._action_space.shape
            else:
                action_dim = 1
        elif isinstance(self._action_space, gym.spaces.Discrete):
            # 离散动作空间
            action_dim = self._action_space.n
        else:
            # 尝试MultiDiscrete或其他类型
            try:
                if hasattr(self._action_space, 'n'):
                    action_dim = self._action_space.n
                elif hasattr(self._action_space, 'nvec'):
                    action_dim = sum(self._action_space.nvec)
                else:
                    action_dim = 2  # 默认值
                    print(f"警告：无法确定动作空间维度，使用默认值: {action_dim}")
            except:
                action_dim = 2
                print(f"警告：无法确定智能体 {self.index} 的动作空间维度，使用默认值 {action_dim}")
        
        # 设置Actor和Critic的动作维度变量
        self._original_action_dim = action_dim  # 原始维度以供引用
        self._actor_action_dim = action_dim     # Actor网络输出维度
        self._critic_action_dim = action_dim    # Critic网络输入维度
        
        # 在3D环境中，保持原始动作维度，确保与环境匹配
        if is_3d and self.continuous_actions:
            # 不再强制修改为3维，保持原始动作维度
            print(f"3D环境: 使用原始动作维度={action_dim}, Actor动作维度={self._actor_action_dim}, Critic动作维度={self._critic_action_dim}")
        
        self._total_act_size = self._actor_action_dim
        
        print(f"创建智能体 {self.index} 网络，状态维度: {obs_dim}, 动作维度: Actor={self._actor_action_dim}, Critic={self._critic_action_dim}")
        
        # 使用自定义构建函数（如果提供）
        if self.continuous_actions and self.build_actor_fn is not None:
            print(f"使用自定义Actor网络构建函数")
            self._actor_model = self.build_actor_fn(input_shape=(obs_dim,), action_dim=self._actor_action_dim)
        else:
            # 创建Actor网络（策略网络）
            self._actor_model = generate_actor_network(obs_dim, self._actor_action_dim)
        
        if self.continuous_actions and self.build_critic_fn is not None:
            print(f"使用自定义Critic网络构建函数")
            # 为Critic使用原始动作维度，确保与环境动作空间匹配
            self._critic_model = self.build_critic_fn(state_shape=(obs_dim,), action_dim=self._critic_action_dim)
        else:
            # 创建Critic网络（值函数网络）
            self._critic_model = generate_critic_network(obs_dim, self._critic_action_dim)
        
        # 创建目标网络（用于稳定训练）
        self._target_actor = tf.keras.models.clone_model(self._actor_model)
        self._target_critic = tf.keras.models.clone_model(self._critic_model)
        
        # 确保目标网络权重初始化与主网络相同
        self._target_actor.set_weights(self._actor_model.get_weights())
        self._target_critic.set_weights(self._critic_model.get_weights())
        
        # 动作噪声 - 仅在不使用参数空间噪声时创建
        if not self.use_param_noise:
            # 确保噪声维度与动作维度匹配
            action_dim = self._total_act_size
            self._noise = OUNoise(
                mean=np.zeros(action_dim), 
                std_dev=float(self._noise_std_dev) * np.ones(action_dim), 
                decay=0.9995
            )
            print(f"创建OU噪声，维度: {action_dim}")

    def _update_perturbed_actor(self):
        """更新带参数噪声的Actor网络"""
        if not self.use_param_noise:
            return
            
        self.actor_perturbed = self.param_noise.add_noise_to_weights(self._actor_model)

    # @tf.function  # 移除tf.function装饰器以提高灵活性
    def policy(self, state, noise_scale=1.0):
        """
        根据当前状态选择动作，添加噪声进行探索
        """
        # 检查state是否为列表，如果是，则转换为numpy数组
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        # 确保state是tensor并且形状正确
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # 检查输入维度，解决(3, 8)到(None, 61)的形状不匹配问题
        expected_dim = 61  # 从错误信息中获取的期望维度
        
        # 根据state形状调整维度
        if len(state.shape) == 1:  # 单个观察值(8,)
            current_dim = state.shape[0]
            if current_dim < expected_dim:
                # 扩展维度
                padded_state = tf.zeros(expected_dim, dtype=tf.float32)
                # 复制原始数据到前面的维度
                padded_state = tf.tensor_scatter_nd_update(
                    padded_state,
                    [[i] for i in range(current_dim)],
                    state
                )
                state = padded_state
        elif len(state.shape) == 2:  # 批次形式(batch_size, 8)
            batch_size, current_dim = state.shape
            if current_dim < expected_dim:
                # 扩展维度
                padded_state = tf.zeros((batch_size, expected_dim), dtype=tf.float32)
                
                # 创建更新索引和值
                indices = []
                values = []
                for i in range(batch_size):
                    for j in range(current_dim):
                        indices.append([i, j])
                        values.append(state[i, j])
                
                # 使用scatter_nd更新
                padded_state = tf.tensor_scatter_nd_update(padded_state, indices, values)
                state = padded_state
        
        # 添加噪声，用于探索
        if noise_scale > 0 and not self.use_param_noise:
            # 使用OU噪声
            noise = self._noise()
        else:
            # 不添加噪声
            noise = np.zeros(self._total_act_size, dtype=np.float32)
        
        try:
            # 获取动作和力场参数
            if self.use_param_noise and noise_scale > 0 and self.actor_perturbed is not None:
                # 使用带参数噪声的网络
                outputs = self.actor_perturbed(state)
                # 每隔一段时间更新噪声网络
                if np.random.uniform() < 0.01:  # 1%概率更新噪声网络
                    self._update_perturbed_actor()
            else:
                # 使用原始网络
                try:
                    # 尝试直接调用模型
                    outputs = self._actor_model(state)
                except Exception as e:
                    print(f"直接调用Actor网络时出错: {e}")
                    
                    # 尝试重新整形状态后再调用
                    try:
                        if tf.is_tensor(state):
                            # 检查并处理维度
                            state_shape = tf.shape(state)
                            if len(state_shape) == 3:
                                state = tf.reshape(state, [state_shape[0], state_shape[2]])
                            elif len(state_shape) > 1 and state_shape[0] == 1:
                                # 确保只有一个批次维度
                                state = tf.reshape(state, [1, -1])
                        
                        outputs = self._actor_model(state)
                    except Exception as e2:
                        print(f"处理后调用Actor网络时仍然出错: {e2}")
                        # 返回默认的零动作
                        return tf.zeros((1, self._total_act_size), dtype=tf.float32)
            
            # 处理多输出模型情况
            action_probs = None
            force_params = None
            
            if isinstance(outputs, list) and len(outputs) > 0:
                # 多输出模型，第一个是动作，第二个是力场参数
                action_probs = outputs[0]
                if len(outputs) > 1:
                    force_params = outputs[1]
            else:
                # 单输出模型
                action_probs = outputs
            
            # 存储力场参数以供其他函数使用
            if force_params is not None:
                if hasattr(force_params, 'numpy'):
                    self._last_force_params = force_params.numpy()
                else:
                    self._last_force_params = force_params
                
                # 如果不使用参数空间噪声，则添加OU噪声
                if not self.use_param_noise and noise_scale > 0:
                    noise = self._noise() * noise_scale
                    # 确保噪声维度与动作维度匹配
                    if hasattr(action_probs, 'numpy'):
                        action_probs_np = action_probs.numpy()
                        # 检查并调整噪声维度以匹配动作维度
                        if noise.shape != action_probs_np.shape:
                            # 调整噪声的维度以匹配动作维度
                            noise = np.resize(noise, action_probs_np.shape)
                        action_probs_np = action_probs_np + noise
                        action_probs = tf.convert_to_tensor(action_probs_np, dtype=tf.float32)
                    else:
                        # 处理张量情况
                        action_shape = tf.shape(action_probs)
                        noise_tensor = tf.convert_to_tensor(noise, dtype=tf.float32)
                        if tf.shape(noise_tensor)[0] != action_shape[0]:
                            # 调整噪声张量维度
                            noise_tensor = tf.broadcast_to(noise_tensor, action_shape)
                        action_probs = action_probs + noise_tensor
            
            # 裁剪概率值到合法范围
            if hasattr(action_probs, 'numpy'):
                action_probs_np = action_probs.numpy()
                action_probs_np = np.clip(action_probs_np, -1.0, 1.0)
                action_probs = tf.convert_to_tensor(action_probs_np, dtype=tf.float32)
            else:
                action_probs = tf.clip_by_value(action_probs, -1.0, 1.0)
            
            return action_probs  # 直接返回连续动作值
            
        except Exception as e:
            print(f"执行policy时出错，返回默认动作: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认的零动作
            return tf.zeros((1, self._total_act_size), dtype=tf.float32)

    # @tf.function  # 移除tf.function装饰器以提高灵活性
    def non_exploring_policy(self, state):
        """
        根据当前状态选择动作，不添加探索噪声
        """
        # 检查state是否为列表，如果是，则转换为numpy数组
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        # 确保状态是tensor并且有正确的维度和类型
        if isinstance(state, np.ndarray):
            # 检查维度是否正确
            if len(state.shape) == 1 and state.shape[0] != self._total_obs_size:
                print(f"警告: 状态维度不匹配! 预期 {self._total_obs_size}，实际 {state.shape[0]}")
                # 尝试调整维度
                if state.shape[0] < self._total_obs_size:
                    # 填充
                    padding = np.zeros(self._total_obs_size - state.shape[0], dtype=np.float32)
                    state = np.concatenate([state, padding])
                else:
                    # 截断
                    state = state[:self._total_obs_size]
            
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        elif tf.is_tensor(state) and state.dtype != tf.float32:
            state = tf.cast(state, tf.float32)
        
        # 处理维度 - 修复形状问题
        if len(tf.shape(state)) == 1:
            state = tf.expand_dims(state, 0)
        elif len(tf.shape(state)) == 3:
            # 处理(1, 1, N)形状 - 去除多余维度
            state = tf.squeeze(state, axis=1)
        
        try:
            # 获取动作和力场参数
            try:
                # 直接调用网络
                outputs = self._actor_model(state)
            except Exception as e:
                print(f"直接调用Actor网络时出错: {e}")
                
                # 尝试重新整形状态后再调用
                if tf.is_tensor(state):
                    # 检查并处理维度
                    state_shape = tf.shape(state)
                    if len(state_shape) == 3:
                        state = tf.reshape(state, [state_shape[0], state_shape[2]])
                    elif len(state_shape) > 1 and state_shape[0] == 1:
                        # 确保只有一个批次维度
                        state = tf.reshape(state, [1, -1])
                
                try:
                    outputs = self._actor_model(state)
                except Exception as e2:
                    print(f"处理后调用Actor网络时仍然出错: {e2}")
                    # 返回默认的零动作
                    return tf.zeros((1, self._total_act_size), dtype=tf.float32)
            
            # 处理多输出模型
            action_probs = None
            force_params = None
            
            if isinstance(outputs, list) and len(outputs) > 0:
                # 多输出模型，第一个是动作，第二个是力场参数
                action_probs = outputs[0]
                if len(outputs) > 1:
                    force_params = outputs[1]
            else:
                # 单输出模型
                action_probs = outputs
            
            # 存储力场参数以供其他函数使用
            if force_params is not None:
                if hasattr(force_params, 'numpy'):
                    self._last_force_params = force_params.numpy()
                else:
                    self._last_force_params = force_params
            
            # 直接返回连续动作值，不进行softmax或argmax处理
            return action_probs
            
        except Exception as e:
            print(f"执行non_exploring_policy时出错，返回默认动作: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认的零动作
            return tf.zeros((1, self._total_act_size), dtype=tf.float32)

    # Custom NN updates based on the MADDPG paper基于MADDPG论文的自定义神经网络更新
    def update(self, states, actions, rewards, next_states, dones, next_actions):
        """
        更新智能体网络，实现Actor的延迟更新。
        - Critic网络在每次调用时都会更新。
        - Actor网络和目标网络根据 actor_update_delay 的设置延迟更新。
        """
        self.update_count += 1
        
        # 1. 总是更新Critic网络
        critic_loss = self._update_critic(states, actions, rewards, next_states, next_actions, dones)
        
        actor_loss = tf.constant(0.0) # 默认actor loss为0
        # 2. 根据延迟频率更新Actor网络和目标网络
        if self.update_count % self.actor_update_delay == 0:
            actor_loss = self._update_actor(states)
            self._update_target_networks()
        
        return critic_loss, actor_loss

    def _update_critic(self, states, actions, rewards, next_states, next_actions, dones):
        """更新Critic网络
        
        Args:
            states: 状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一个状态批次
            next_actions: 下一个动作批次
            dones: 完成标志批次
            
        Returns:
            critic_loss: Critic网络损失
        """
        try:
            # 检查输入数据是否包含NaN或Inf
            if tf.reduce_any(tf.math.is_nan(states)) or tf.reduce_any(tf.math.is_inf(states)):
                print(f"警告: 输入状态包含NaN或Inf值，已替换为有限值")
                states = tf.where(tf.math.is_nan(states), tf.zeros_like(states), states)
                states = tf.where(tf.math.is_inf(states), tf.ones_like(states), states)
                
            if tf.reduce_any(tf.math.is_nan(actions)) or tf.reduce_any(tf.math.is_inf(actions)):
                print(f"警告: 输入动作包含NaN或Inf值，已替换为有限值")
                actions = tf.where(tf.math.is_nan(actions), tf.zeros_like(actions), actions)
                actions = tf.where(tf.math.is_inf(actions), tf.clip_by_value(actions, -1.0, 1.0), actions)
                
            if tf.reduce_any(tf.math.is_nan(rewards)) or tf.reduce_any(tf.math.is_inf(rewards)):
                print(f"警告: 输入奖励包含NaN或Inf值，已替换为有限值")
                rewards = tf.where(tf.math.is_nan(rewards), tf.zeros_like(rewards), rewards)
                rewards = tf.where(tf.math.is_inf(rewards), tf.clip_by_value(rewards, -100.0, 100.0), rewards)
                
            if tf.reduce_any(tf.math.is_nan(next_states)) or tf.reduce_any(tf.math.is_inf(next_states)):
                print(f"警告: 输入下一状态包含NaN或Inf值，已替换为有限值")
                next_states = tf.where(tf.math.is_nan(next_states), tf.zeros_like(next_states), next_states)
                next_states = tf.where(tf.math.is_inf(next_states), tf.ones_like(next_states), next_states)
                
            if tf.reduce_any(tf.math.is_nan(next_actions)) or tf.reduce_any(tf.math.is_inf(next_actions)):
                print(f"警告: 输入下一动作包含NaN或Inf值，已替换为有限值")
                next_actions = tf.where(tf.math.is_nan(next_actions), tf.zeros_like(next_actions), next_actions)
                next_actions = tf.where(tf.math.is_inf(next_actions), tf.clip_by_value(next_actions, -1.0, 1.0), next_actions)
            
            # 打印输入形状信息
            print(f"_update_critic - 输入形状: states={states.shape}, actions={actions.shape}, next_states={next_states.shape}, next_actions={next_actions.shape}")
            
            # 确保输入形状正确
            Critic模型期望输入形状 = f"states={self._critic_model.input_shape[0]}, actions={self._critic_model.input_shape[1]}"
            print(f"Critic模型期望输入形状: {Critic模型期望输入形状}")
            
            # 调整输入形状以匹配模型期望
            print(f"调整后输入形状: states={states.shape}, actions={actions.shape}")
            
            # 使用目标网络计算目标Q值
            with tf.GradientTape() as tape:
                # 计算目标Q值
                target_q = self._target_critic([next_states, next_actions])
                
                # 检查target_q是否包含NaN或Inf
                if tf.reduce_any(tf.math.is_nan(target_q)) or tf.reduce_any(tf.math.is_inf(target_q)):
                    print(f"警告: 目标Q值包含NaN或Inf值，已替换为有限值")
                    target_q = tf.where(tf.math.is_nan(target_q), tf.zeros_like(target_q), target_q)
                    target_q = tf.where(tf.math.is_inf(target_q), tf.clip_by_value(target_q, -100.0, 100.0), target_q)
                
                # 应用折扣因子和奖励
                target_q = rewards + self.gamma * target_q * (1 - dones)
                
                # 计算当前Q值
                q = self._critic_model([states, actions])
                
                # 检查q是否包含NaN或Inf
                if tf.reduce_any(tf.math.is_nan(q)) or tf.reduce_any(tf.math.is_inf(q)):
                    print(f"警告: 当前Q值包含NaN或Inf值，已替换为有限值")
                    q = tf.where(tf.math.is_nan(q), tf.zeros_like(q), q)
                    q = tf.where(tf.math.is_inf(q), tf.clip_by_value(q, -100.0, 100.0), q)
                
                # 计算均方误差损失
                critic_loss = tf.reduce_mean(tf.square(target_q - q))
                
                # 检查损失值是否为NaN或Inf
                if tf.math.is_nan(critic_loss) or tf.math.is_inf(critic_loss):
                    print(f"警告: Critic损失值为NaN或Inf，使用默认损失值")
                    critic_loss = tf.constant(1.0, dtype=tf.float32)
            
            # 计算梯度
            critic_grads = tape.gradient(critic_loss, self._critic_model.trainable_variables)
            
            # 检查梯度是否包含NaN或Inf
            has_nan_or_inf = False
            for g in critic_grads:
                if g is not None and (tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))):
                    has_nan_or_inf = True
                    break
            
            if has_nan_or_inf:
                print("警告: Critic网络梯度包含NaN或Inf值，跳过此次更新")
                return float(critic_loss)
            
            # 过滤None梯度
            critic_grads = [tf.clip_by_norm(g, 1.0) for g in critic_grads if g is not None]
            self._critic_opt.apply_gradients(zip(critic_grads, self._critic_model.trainable_variables))
            
            return critic_loss
        except Exception as e:
            print(f"Critic网络更新失败: {e}")
            import traceback
            traceback.print_exc()
            return tf.constant(1.0, dtype=tf.float32)  # 返回默认损失值

    def _update_actor(self, states):
        """更新Actor网络
        
        Args:
            states: 状态批次
            
        Returns:
            actor_loss: Actor网络损失
        """
        try:
            with tf.GradientTape() as tape:
                # 确保输入形状正确
                print(f"_update_actor - 输入状态形状: {states.shape}, 期望输入形状: {self._actor_model.input_shape}")
                
                # 检查输入数据是否包含NaN或Inf
                if tf.reduce_any(tf.math.is_nan(states)) or tf.reduce_any(tf.math.is_inf(states)):
                    print(f"警告: 输入状态包含NaN或Inf值，已替换为有限值")
                    states = tf.where(tf.math.is_nan(states), tf.zeros_like(states), states)
                    states = tf.where(tf.math.is_inf(states), tf.ones_like(states), states)
                
                # 检查期望的输入维度
                expected_dim = self._total_obs_size
                input_dim = states.shape[1]
                
                # 处理输入形状不匹配问题
                if input_dim != expected_dim:
                    print(f"需要调整状态维度: 当前={input_dim}, 期望={expected_dim}")
                    
                    # 创建正确形状的输入
                    actor_inputs = tf.zeros((states.shape[0], expected_dim), dtype=tf.float32)
                    
                    # 决定如何复制数据
                    if input_dim > expected_dim:
                        # 状态维度过大，只取前expected_dim个维度
                        actor_inputs = tf.slice(states, [0, 0], [states.shape[0], expected_dim])
                        print(f"截取状态维度: {input_dim} -> {expected_dim}")
                    else:
                        # 状态维度过小，复制并填充剩余维度
                        actor_inputs = tf.pad(states, [[0, 0], [0, expected_dim - input_dim]])
                        print(f"填充状态维度: {input_dim} -> {expected_dim}")
                else:
                    # 无需调整
                    actor_inputs = states
                
                # 获取Actor网络输出
                actor_outputs = self._actor_model(actor_inputs, training=True)
                
                # 处理多输出网络
                if isinstance(actor_outputs, list) and len(actor_outputs) > 0:
                    actions = actor_outputs[0]  # 只使用第一个输出（动作）
                    # 保存力场参数，但不参与梯度计算
                    if len(actor_outputs) > 1:
                        self._force_params = actor_outputs[1]
                else:
                    actions = actor_outputs
                
                # 检查动作是否包含NaN或Inf
                if tf.reduce_any(tf.math.is_nan(actions)) or tf.reduce_any(tf.math.is_inf(actions)):
                    print(f"警告: Actor网络输出包含NaN或Inf值，已替换为有限值")
                    actions = tf.where(tf.math.is_nan(actions), tf.zeros_like(actions), actions)
                    actions = tf.where(tf.math.is_inf(actions), tf.clip_by_value(actions, -1.0, 1.0), actions)
                
                # 获取Actor和Critic期望的动作维度
                actor_action_dim = actions.shape[1] if len(actions.shape) > 1 else 1
                critic_action_dim = getattr(self, '_critic_action_dim', 7)  # 默认使用7D动作维度（3D位置+4D势场力）
                
                print(f"Actor输出动作维度: {actor_action_dim}, Critic期望动作维度: {critic_action_dim}")
                    
                # 检查并调整动作维度以匹配Critic输入
                if actor_action_dim != critic_action_dim:
                    print(f"调整Actor输出动作维度({actor_action_dim})为Critic期望维度({critic_action_dim})")
                    
                    # 保留3D动作和势场力参数
                    if actor_action_dim == 7 and critic_action_dim == 7:
                        # 完整使用7D动作（3D位置+4D势场力）
                        critic_actions = actions
                        print("保持完整7D动作: 使用x,y,z坐标和势场力参数进行Critic评估")
                    elif actor_action_dim > critic_action_dim:
                        # 截断动作维度，只保留前critic_action_dim个维度
                        critic_actions = tf.slice(actions, [0, 0], [actions.shape[0], critic_action_dim])
                    else:
                        # 扩展动作维度，填充零值
                        paddings = tf.constant([[0, 0], [0, critic_action_dim - actor_action_dim]])
                        critic_actions = tf.pad(actions, paddings)
                        
                    print(f"调整后动作维度: {critic_actions.shape}")
                else:
                    critic_actions = actions
                
                # 计算梯度更新方向
                # 我们的目标是最大化Q值，因此使用负号来表示梯度下降方向
                # 为避免"Cannot iterate over a scalar tensor"错误，修改此处的处理方式
                critic_output = self._critic_model([actor_inputs, critic_actions])
                
                # 检查critic输出是否包含NaN或Inf
                if tf.reduce_any(tf.math.is_nan(critic_output)) or tf.reduce_any(tf.math.is_inf(critic_output)):
                    print(f"警告: Critic网络输出包含NaN或Inf值，已替换为有限值")
                    critic_output = tf.where(tf.math.is_nan(critic_output), tf.zeros_like(critic_output), critic_output)
                    critic_output = tf.where(tf.math.is_inf(critic_output), tf.clip_by_value(critic_output, -100.0, 100.0), critic_output)
                
                # 处理critic_output是否为标量的情况
                if len(tf.shape(critic_output)) == 0:  # 如果是标量
                    actor_loss = -critic_output  # 直接取负值
                else:
                    actor_loss = -tf.reduce_mean(critic_output)  # 取平均值的负值
                
                # 检查损失值是否为NaN或Inf
                if tf.math.is_nan(actor_loss) or tf.math.is_inf(actor_loss):
                    print(f"警告: Actor损失值为NaN或Inf，使用默认损失值")
                    actor_loss = tf.constant(1.0, dtype=tf.float32)
            
            # 计算梯度并应用
            actor_grads = tape.gradient(actor_loss, self._actor_model.trainable_variables)
            
            # 检查梯度是否包含NaN或Inf
            has_nan_or_inf = False
            for g in actor_grads:
                if g is not None and (tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))):
                    has_nan_or_inf = True
                    break
            
            if has_nan_or_inf:
                print("警告: Actor网络梯度包含NaN或Inf值，跳过此次更新")
                return float(actor_loss)
            
            # 检查梯度是否为None
            if None in actor_grads:
                print("警告: Actor网络梯度包含None值")
                valid_grads = []
                valid_vars = []
                for g, v in zip(actor_grads, self._actor_model.trainable_variables):
                    if g is not None:
                        # 应用梯度裁剪
                        clipped_g = tf.clip_by_norm(g, 1.0)
                        valid_grads.append(clipped_g)
                        valid_vars.append(v)
                
                if valid_grads:
                    self._actor_opt.apply_gradients(zip(valid_grads, valid_vars))
            else:
                # 梯度裁剪，防止梯度爆炸
                actor_grads = [tf.clip_by_norm(g, 1.0) for g in actor_grads]
                self._actor_opt.apply_gradients(zip(actor_grads, self._actor_model.trainable_variables))
            
            # 如果actor_loss是张量，将其转换为Python标量
            if tf.is_tensor(actor_loss):
                actor_loss = actor_loss.numpy()
                
            return actor_loss
            
        except Exception as e:
            print(f"智能体 {self.index} 更新失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个默认的损失值
            return tf.constant(1.0, dtype=tf.float32)

    def update_target_networks(self):
        """更新目标网络的权重"""
        # 使用软更新方式更新目标网络
        for target_var, source_var in zip(self._target_actor.variables, self._actor_model.variables):
            target_var.assign((1 - self.tau) * target_var + self.tau * source_var)
        
        for target_var, source_var in zip(self._target_critic.variables, self._critic_model.variables):
            target_var.assign((1 - self.tau) * target_var + self.tau * source_var)

    def set_policy_for_execution(self, policy_param='last'):
        """设置执行策略，选择使用哪个模型权重"""
        # 默认使用最后训练的模型
        if policy_param == 'last':
            # 使用默认设置，不需要切换
            pass
        elif policy_param == 'best_overall':
            if hasattr(self, '_best_model_weights'):
                self._actor_model.set_weights(self._best_model_weights)
            else:
                print(f"警告: 智能体 {self.index} 没有最佳整体权重，使用当前权重")
        elif policy_param == 'best_average':
            if hasattr(self, '_best_avg_model_weights'):
                self._actor_model.set_weights(self._best_avg_model_weights)
            else:
                print(f"警告: 智能体 {self.index} 没有最佳平均权重，使用当前权重")
        else:
            print(f"警告: 未知策略参数 '{policy_param}'，使用当前权重")

    def act(self, obs, add_noise=True):
        """
        根据观察选择动作，可选是否添加噪声
        
        Args:
            obs: 观察值
            add_noise: 是否添加探索噪声
            
        Returns:
            action: 选择的动作
        """
        try:
        # 如果观察是空字典，返回零动作
            if isinstance(obs, dict) and len(obs) == 0:
                print(f"警告: 接收到空字典作为观察值，返回零动作")
                return np.zeros(self._total_act_size)
                
            # 处理字典类型的观察（gymnasium 有时使用字典类型）
            if isinstance(obs, dict):
                # 尝试获取observation键或将所有值连接起来
                if 'observation' in obs:
                    obs = obs['observation']
                else:
                    # 连接所有数值类型的值
                    combined_obs = []
                    for k, v in obs.items():
                        if isinstance(v, (np.ndarray, list, float, int)):
                            if isinstance(v, np.ndarray):
                                combined_obs.append(v.flatten())
                            elif isinstance(v, list):
                                combined_obs.append(np.array(v).flatten())
                            else:
                                combined_obs.append(np.array([v]))
                    
                    if combined_obs:
                        obs = np.concatenate(combined_obs)
                    else:
                        print(f"警告: 无法从字典中提取观察值，返回零动作")
                        return np.zeros(self._total_act_size)
            
            # 确保输入是tensorflow张量
            if isinstance(obs, np.ndarray):
                obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            elif not isinstance(obs, tf.Tensor):
                # 处理其他类型，例如列表
                try:
                    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                except Exception as e:
                    print(f"无法转换观察值为Tensor: {e}，类型: {type(obs)}")
                    print(f"观察值内容: {obs}")
                    return np.zeros(self._total_act_size)
            
            # 添加批次维度如果需要
            if len(obs.shape) == 1:
                obs = tf.expand_dims(obs, axis=0)
                
                # 获取基础动作（处理多输出网络）
                outputs = self._actor_model(obs)
                    
                # 处理多输出模型
                if isinstance(outputs, list) and len(outputs) > 0:
                    # 使用第一个输出作为动作
                    action = outputs[0]
                    # 如果有第二个输出，它是力场参数
                    if len(outputs) > 1 and hasattr(self, '_last_force_params'):
                        self._last_force_params = outputs[1].numpy() if hasattr(outputs[1], 'numpy') else outputs[1]
                else:
                    # 单输出模型
                    action = outputs
                    
            action = tf.squeeze(action)  # 移除不必要的维度
            
                # 添加探索噪声
            if add_noise:
                    try:
                        noise = self._noise()  # 使用__call__方法
                        # 确保噪声和动作的形状匹配
                        if isinstance(noise, tf.Tensor) and noise.shape != action.shape:
                            noise = tf.reshape(noise, action.shape)
                        action = action + noise
                    except Exception as e:
                        print(f"添加噪声时出错: {e}，跳过添加噪声")
            
            # 确保动作在合法范围内
            action = tf.clip_by_value(action, self._action_space.low[0], self._action_space.high[0])
            
            # 转换为numpy数组
            if hasattr(action, 'numpy'):
                action = action.numpy()
            
            return action
            
        except Exception as e:
            print(f"执行act方法时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回零动作作为后备
            return np.zeros(self._total_act_size)

    def save(self, save_path):
        """保存模型"""
        try:
            # 确保目录存在
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            # 保存各个智能体的模型
            for i, agent in enumerate(self.agents):
                agent_save_path = os.path.join(save_path, f'agent_{i}')
                if not os.path.exists(agent_save_path):
                    os.makedirs(agent_save_path)
                
                # 保存actor和critic网络
                if hasattr(agent, 'actor') and agent.actor is not None:
                    agent.actor.save_weights(os.path.join(agent_save_path, 'actor'))
                if hasattr(agent, 'critic') and agent.critic is not None:
                    agent.critic.save_weights(os.path.join(agent_save_path, 'critic'))
                
                # 保存目标网络
                if hasattr(agent, 'target_actor') and agent.target_actor is not None:
                    agent.target_actor.save_weights(os.path.join(agent_save_path, 'target_actor'))
                if hasattr(agent, 'target_critic') and agent.target_critic is not None:
                    agent.target_critic.save_weights(os.path.join(agent_save_path, 'target_critic'))
                
            print(f'成功保存模型到：{save_path}')
            return True
        except Exception as e:
            print(f'保存模型时出错：{str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def update_learning_rate(self, new_lr):
        """更新所有网络的学习率"""
        self.lr = new_lr
        
        # 更新所有actor和critic的优化器学习率
        for agent in self.agents:
            # 尝试不同的优化器属性名
            optimizer_attrs = ['a_optimizer', 'c_optimizer', 'actor_optimizer', 
                              'critic_optimizer', 'a_opt', 'c_opt']
            
            for attr in optimizer_attrs:
                if hasattr(agent, attr):
                    optimizer = getattr(agent, attr)
                    if hasattr(optimizer, 'learning_rate'):
                        optimizer.learning_rate.assign(new_lr)
                    # 支持TF优化器中不同的学习率访问方式
                    elif hasattr(optimizer, '_learning_rate'):
                        if hasattr(optimizer._learning_rate, 'assign'):
                            optimizer._learning_rate.assign(new_lr)
                        else:
                            print(f"警告: 无法修改优化器学习率，优化器类型: {type(optimizer)}")
        
        print(f"已将MADDPG学习率更新为: {new_lr}")
    
    def partial_reset_weights(self):
        """部分重置网络权重以跳出局部最优"""
        print("部分重置网络权重以帮助跳出局部最优...")
        
        import numpy as np
        reset_factor = 0.3  # 只重置30%的权重
        
        for agent in self.agents:
            # 随机重置actor网络的部分权重
            if hasattr(agent, 'actor'):
                for layer in agent.actor.layers:
                    if hasattr(layer, 'kernel'):
                        # 获取当前权重
                        weights = layer.kernel.numpy()
                        # 创建随机掩码决定哪些权重需要重置
                        mask = np.random.random(weights.shape) < reset_factor
                        # 生成新的随机权重
                        new_weights = np.random.normal(0, 0.02, weights.shape)
                        # 只重置掩码为True的权重
                        weights[mask] = new_weights[mask]
                        # 设置回网络
                        layer.kernel.assign(weights)
            
            # 类似地，对critic网络进行部分重置
            if hasattr(agent, 'critic'):
                for layer in agent.critic.layers:
                    if hasattr(layer, 'kernel'):
                        weights = layer.kernel.numpy()
                        mask = np.random.random(weights.shape) < reset_factor
                        new_weights = np.random.normal(0, 0.02, weights.shape)
                        weights[mask] = new_weights[mask]
                        layer.kernel.assign(weights)
        
        print("网络权重部分重置完成")

    def adapt_parameter_noise(self, states):
        """自适应调整参数噪声大小"""
        if not self.use_param_noise or self.actor_perturbed is None:
            return 0.0
            
        # 调整参数噪声
        distance = self.param_noise.adapt(
            self._actor_model, self.actor_perturbed, states
        )
        
        # 更新带噪声的Actor网络
        self._update_perturbed_actor()
        
        return distance

    def get_force_params(self, state=None):
        """
        获取力场参数，如果提供了状态则执行一次前向传播获取最新参数
        
        参数:
            state: 可选，当前状态
            
        返回:
            force_params: 力场参数
        """
        try:
            # 状态处理逻辑优化，减少调用policy
            if state is not None:
                # 检查状态是否为空或无效
                is_empty_state = False
                
                if isinstance(state, dict) and not state:
                    print("警告: get_force_params收到空字典状态，使用默认力场参数")
                    is_empty_state = True
                elif isinstance(state, list) and not state:
                    print("警告: get_force_params收到空列表状态，使用默认力场参数")
                    is_empty_state = True
                elif isinstance(state, np.ndarray) and state.size == 0:
                    print("警告: get_force_params收到空数组状态，使用默认力场参数")
                    is_empty_state = True
                elif tf.is_tensor(state) and tf.size(state) == 0:
                    print("警告: get_force_params收到空张量状态，使用默认力场参数")
                    is_empty_state = True
                
                # 只在状态非空时才尝试调用网络
                if not is_empty_state:
                    # 直接使用Actor网络获取输出，而不是通过policy函数
                    try:
                        # 预处理状态
                        processed_state = state
                        
                        # 转换字典状态为数组
                        if isinstance(processed_state, dict):
                            try:
                                values = []
                                for value in processed_state.values():
                                    if isinstance(value, (list, np.ndarray)):
                                        values.extend(value)
                                    else:
                                        values.append(value)
                                processed_state = np.array(values, dtype=np.float32)
                            except Exception as e:
                                print(f"处理字典状态时出错: {e}")
                                processed_state = np.zeros(self._total_obs_size, dtype=np.float32)
                        
                        # 转换列表为数组
                        if isinstance(processed_state, list):
                            processed_state = np.array(processed_state, dtype=np.float32)
                        
                        # 确保维度正确
                        if isinstance(processed_state, np.ndarray):
                            if len(processed_state.shape) == 1:
                                if processed_state.shape[0] != self._total_obs_size:
                                    print(f"警告: 状态维度不匹配! 预期 {self._total_obs_size}，实际 {processed_state.shape[0]}")
                                    # 调整维度
                                    if processed_state.shape[0] < self._total_obs_size:
                                        padding = np.zeros(self._total_obs_size - processed_state.shape[0], dtype=np.float32)
                                        processed_state = np.concatenate([processed_state, padding])
                                    else:
                                        processed_state = processed_state[:self._total_obs_size]
                                
                                # 扩展批次维度
                                processed_state = np.expand_dims(processed_state, axis=0)
                            
                            # 转换为张量
                            processed_state = tf.convert_to_tensor(processed_state, dtype=tf.float32)
                        
                        # 检查并调整输入维度 - 新增处理额外维度
                        if tf.is_tensor(processed_state):
                            state_shape = tf.shape(processed_state)
                            if len(state_shape) == 3:
                                # 处理(1, 1, N)形状 - 去除多余维度
                                processed_state = tf.squeeze(processed_state, axis=1)
                        
                        # 直接调用Actor网络
                        try:
                            outputs = self._actor_model(processed_state)
                        except Exception as e:
                            print(f"直接调用Actor网络时出错: {e}")
                            # 尝试进一步调整状态的形状
                            if tf.is_tensor(processed_state):
                                try:
                                    # 再次检查并处理维度
                                    state_shape = tf.shape(processed_state)
                                    if len(state_shape) == 3:
                                        processed_state = tf.reshape(processed_state, [state_shape[0], state_shape[2]])
                                    elif len(state_shape) > 1 and state_shape[0] == 1:
                                        # 确保只有一个批次维度
                                        processed_state = tf.reshape(processed_state, [1, -1])
                                    
                                    # 再次尝试调用模型
                                    outputs = self._actor_model(processed_state)
                                except Exception as e2:
                                    print(f"调整形状后调用网络仍然失败: {e2}")
                                    # 设置outputs为None，让后面的逻辑处理
                                    outputs = None
                        
                        # 处理输出，获取力场参数
                        if outputs is not None:
                            if isinstance(outputs, list) and len(outputs) > 1:
                                force_params = outputs[1]  # 第二个输出是力场参数
                                if hasattr(force_params, 'numpy'):
                                    self._last_force_params = force_params.numpy()
                                else:
                                    self._last_force_params = force_params
                    except Exception as e:
                        print(f"直接调用Actor网络时出错: {e}")
                        # 错误处理，不中断执行
            
            # 返回最后一次存储的力场参数
            if hasattr(self, '_last_force_params') and self._last_force_params is not None:
                # 检查参数维度，处理批次维度
                if isinstance(self._last_force_params, np.ndarray) and len(self._last_force_params.shape) > 1:
                    # 确保返回的是一维数组，并且包含4个参数
                    params = self._last_force_params[0]
                    # 验证并修复参数长度
                    if len(params) != 4:
                        if len(params) > 4:
                            params = params[:4]  # 只取前4个参数
                        else:
                            # 如果参数少于4个，填充到4个
                            params = np.pad(params, (0, 4 - len(params)), 'constant', constant_values=0.5)
                    return params
                
                # 直接检查参数是一维数组的情况
                params = self._last_force_params
                if isinstance(params, np.ndarray) and len(params) != 4:
                    if len(params) > 4:
                        params = params[:4]  # 只取前4个参数
                    else:
                        # 如果参数少于4个，填充到4个
                        params = np.pad(params, (0, 4 - len(params)), 'constant', constant_values=0.5)
                return params
            else:
                # 返回默认值 - 4个势场参数
                default_params = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)  # 默认中间值
                self._last_force_params = default_params
                return default_params
        except Exception as e:
            print(f"获取力场参数时出错: {e}")
            # 返回默认力场参数 - 4个参数
            return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

class MADDPG:
    def __init__(self, n_agents, obs_shapes, action_dims, gamma=0.95, tau=0.01, critic_lr=0.002, actor_lr=0.001, noise_std_dev=0.02, buffer_size=10e6, batch_size=1024, is_3d=False):
        """初始化MADDPG算法
        
        参数:
            n_agents: 智能体数量
            obs_shapes: 各智能体观察空间维度列表
            action_dims: 各智能体动作空间维度列表(7维包含3维加速度+4维势场参数)
            gamma: 折扣因子
            tau: 软更新系数
            critic_lr: 评论家网络学习率
            actor_lr: 演员网络学习率
            noise_std_dev: 噪声标准差
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            is_3d: 是否为3D环境
        """
        self.n_agents = n_agents
        self.is_3d = is_3d
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std_dev = noise_std_dev
        
        # 创建每个智能体的网络和优化器
        self.agents = []
        
        for i in range(n_agents):
            agent = MADDPGAgent(
                env=None,  # 不需要环境对象
                agent_index=i,
                gamma=gamma,
                tau=tau,
                critic_lr=critic_lr,
                actor_lr=actor_lr,
                noise_std_dev=noise_std_dev,
                buffer_size=buffer_size,
                batch_size=batch_size,
                is_3d=is_3d
            )
            self.agents.append(agent)
        
        print(f"创建了 {n_agents} 个智能体")
        print(f"- 观察空间维度: {obs_shapes}")
        print(f"- 动作空间维度: {action_dims}")
        print(f"- 3D环境: {is_3d}")
        print(f"- 折扣因子: {gamma}")
        print(f"- 软更新率: {tau}")
        print(f"- 评论家学习率: {critic_lr}")
        print(f"- 演员学习率: {actor_lr}")
        print(f"- 噪声标准差: {noise_std_dev}")
        print(f"- 经验回放缓冲区大小: {buffer_size}")
        print(f"- 批次大小: {batch_size}")
    
    def select_action(self, agent_idx, state, add_noise=True):
        """为给定智能体选择动作
        
        Args:
            agent_idx: 智能体索引
            state: 观察值
            add_noise: 是否添加探索噪声
            
        Returns:
            action: 选择的动作
        """
        return self.agents[agent_idx].act(state, add_noise)
    
    def update(self, agent_idx, state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_actions):
        """更新指定智能体的网络
        
        Args:
            agent_idx: 智能体索引
            state_batch: 当前状态批次
            action_batch: 当前动作批次
            reward_batch: 奖励批次
            next_state_batch: 下一个状态批次
            done_batch: 结束标志批次
            next_actions: 下一个动作批次
        """
        # 处理布尔类型的done_batch
        if isinstance(done_batch, bool):
            done_batch_processed = float(done_batch)
        else:
            done_batch_processed = done_batch
            
        return self.agents[agent_idx].update(
            state_batch, 
            action_batch, 
            reward_batch, 
            next_state_batch, 
            done_batch_processed, 
            next_actions
        )
    
    def save_models(self, suffix=""):
        """保存所有智能体的模型权重"""
        for i, agent in enumerate(self.agents):
            agent.save_models(f"{suffix}_agent{i}")
    
    def load_models(self, suffix=""):
        """加载所有智能体的模型权重"""
        for i, agent in enumerate(self.agents):
            agent.load_models(f"{suffix}_agent{i}")
    
    def set_policy_for_execution(self, policy_param='last'):
        """设置所有智能体的执行策略"""
        for agent in self.agents:
            agent.set_policy_for_execution(policy_param)

    def save(self, save_path):
        """保存模型"""
        try:
            # 确保目录存在
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            # 保存各个智能体的模型
            for i, agent in enumerate(self.agents):
                agent_save_path = os.path.join(save_path, f'agent_{i}')
                if not os.path.exists(agent_save_path):
                    os.makedirs(agent_save_path)
                
                # 保存actor和critic网络
                if hasattr(agent, 'actor') and agent.actor is not None:
                    agent.actor.save_weights(os.path.join(agent_save_path, 'actor'))
                if hasattr(agent, 'critic') and agent.critic is not None:
                    agent.critic.save_weights(os.path.join(agent_save_path, 'critic'))
                
                # 保存目标网络
                if hasattr(agent, 'target_actor') and agent.target_actor is not None:
                    agent.target_actor.save_weights(os.path.join(agent_save_path, 'target_actor'))
                if hasattr(agent, 'target_critic') and agent.target_critic is not None:
                    agent.target_critic.save_weights(os.path.join(agent_save_path, 'target_critic'))
                
            print(f'成功保存模型到：{save_path}')
            return True
        except Exception as e:
            print(f'保存模型时出错：{str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def update_learning_rate(self, new_lr):
        """更新所有网络的学习率"""
        self.lr = new_lr
        
        # 更新所有actor和critic的优化器学习率
        for agent in self.agents:
            # 尝试不同的优化器属性名
            optimizer_attrs = ['a_optimizer', 'c_optimizer', 'actor_optimizer', 
                              'critic_optimizer', 'a_opt', 'c_opt']
            
            for attr in optimizer_attrs:
                if hasattr(agent, attr):
                    optimizer = getattr(agent, attr)
                    if hasattr(optimizer, 'learning_rate'):
                        optimizer.learning_rate.assign(new_lr)
                    # 支持TF优化器中不同的学习率访问方式
                    elif hasattr(optimizer, '_learning_rate'):
                        if hasattr(optimizer._learning_rate, 'assign'):
                            optimizer._learning_rate.assign(new_lr)
                        else:
                            print(f"警告: 无法修改优化器学习率，优化器类型: {type(optimizer)}")
        
        print(f"已将MADDPG学习率更新为: {new_lr}")
    
    def partial_reset_weights(self):
        """部分重置网络权重以跳出局部最优"""
        print("部分重置网络权重以帮助跳出局部最优...")
        
        import numpy as np
        reset_factor = 0.3  # 只重置30%的权重
        
        for agent in self.agents:
            # 随机重置actor网络的部分权重
            if hasattr(agent, 'actor'):
                for layer in agent.actor.layers:
                    if hasattr(layer, 'kernel'):
                        # 获取当前权重
                        weights = layer.kernel.numpy()
                        # 创建随机掩码决定哪些权重需要重置
                        mask = np.random.random(weights.shape) < reset_factor
                        # 生成新的随机权重
                        new_weights = np.random.normal(0, 0.02, weights.shape)
                        # 只重置掩码为True的权重
                        weights[mask] = new_weights[mask]
                        # 设置回网络
                        layer.kernel.assign(weights)
            
            # 类似地，对critic网络进行部分重置
            if hasattr(agent, 'critic'):
                for layer in agent.critic.layers:
                    if hasattr(layer, 'kernel'):
                        weights = layer.kernel.numpy()
                        mask = np.random.random(weights.shape) < reset_factor
                        new_weights = np.random.normal(0, 0.02, weights.shape)
                        weights[mask] = new_weights[mask]
                        layer.kernel.assign(weights)
        
        print("网络权重部分重置完成")

    def get_force_params(self, observations=None):
        """获取所有智能体的力场参数
        
        参数:
            observations: 智能体的观察值

        返回:
            force_params_list: 包含所有智能体力场参数的列表
        """
        force_params_list = []
        
        if observations is not None:
            # 为每个智能体获取力场参数
            for i, agent in enumerate(self.agents):
                # 从7维动作中提取力场参数(最后4维)
                if i < len(observations):
                    # 获取完整的7维动作输出
                    action = agent.non_exploring_policy(observations[i])
                    
                    # 从7维动作中提取后4维作为力场参数
                    if isinstance(action, np.ndarray):
                        if len(action.shape) > 0 and action.shape[0] >= 7:
                            # 提取后4维作为力场参数
                            force_params = action[3:7]
                        else:
                            # 维度不足，可能是旧版格式，使用默认值
                            force_params = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                    else:
                        # 尝试转换为numpy数组并提取
                        try:
                            if hasattr(action, 'numpy'):
                                np_action = action.numpy()
                                if len(np_action.shape) > 0 and np_action.shape[0] >= 7:
                                    force_params = np_action[3:7]
                                else:
                                    force_params = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                            else:
                                force_params = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                        except:
                            force_params = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                    
                    force_params_list.append(force_params)
                else:
                    # 如果没有对应的观察值，使用默认力场参数
                    force_params_list.append(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        else:
            # 如果没有提供观察值，返回所有智能体的默认力场参数
            for _ in range(self.n_agents):
                force_params_list.append(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        
        return force_params_list

def create_actor_network(num_states, num_actions):
    """
    创建演员网络
    """
    inputs = tf.keras.layers.Input(shape=(num_states,))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_critic_network(num_states, num_actions):
    """
    创建评论家网络
    """
    state_input = tf.keras.layers.Input(shape=(num_states,))
    action_input = tf.keras.layers.Input(shape=(num_actions,))
    
    # 合并状态和动作
    concat = tf.keras.layers.Concatenate()([state_input, action_input])
    
    x = tf.keras.layers.Dense(256, activation='relu')(concat)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)

class ReplayBuffer:
    """
    实现经验回放缓冲区
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        将经验添加到缓冲区
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        从缓冲区采样经验
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.float32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

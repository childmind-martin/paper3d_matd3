"""
神经网络架构定义模块
包含Actor和Critic网络的定义
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


def build_continuous_action_network(input_shape, action_dim=7, hidden_units=(256, 128, 64), use_residual=True):
    """构建连续动作Actor网络
    
    参数:
        input_shape: 输入形状 (观察空间维度)
        action_dim: 动作空间维度 (默认7: 3维加速度 + 4维势场参数)
        hidden_units: 隐藏层单元数
        use_residual: 是否使用残差连接
    
    返回:
        actor_model: Keras模型
    """
    inputs = layers.Input(shape=(input_shape,))
    
    # 批标准化层
    x = layers.BatchNormalization()(inputs)
    
    # 构建隐藏层
    prev_x = None
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, kernel_initializer='glorot_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)
        
        # 残差连接（如果启用且维度匹配）
        if use_residual and prev_x is not None and prev_x.shape[-1] == x.shape[-1]:
            x = layers.Add()([x, prev_x])
        prev_x = x
    
    # 分支输出：动作和势场参数
    # 动作分支 (3维加速度)
    action_branch = layers.Dense(64, activation='relu', name='action_hidden')(x)
    action_branch = layers.BatchNormalization()(action_branch)
    action_output = layers.Dense(3, activation='tanh', name='action_output')(action_branch)
    
    # 势场参数分支 (4维参数)
    force_branch = layers.Dense(32, activation='relu', name='force_hidden')(x)
    force_branch = layers.BatchNormalization()(force_branch)
    force_params = layers.Dense(4, activation='sigmoid', name='force_params')(force_branch)
    
    # 合并输出
    outputs = layers.Concatenate(name='combined_output')([action_output, force_params])
    
    # 构建模型
    actor_model = Model(inputs=inputs, outputs=outputs, name='actor_network')
    
    return actor_model


def build_continuous_critic_network(state_shape, action_dim=7, hidden_units=(128, 64, 32), n_agents=3, use_residual=True):
    """构建连续动作Critic网络
    
    参数:
        state_shape: 状态空间形状
        action_dim: 动作空间维度
        hidden_units: 隐藏层单元数
        n_agents: 智能体数量
        use_residual: 是否使用残差连接
    
    返回:
        critic_model: Keras模型
    """
    # 输入层：所有智能体的状态和动作
    state_inputs = []
    action_inputs = []
    
    for i in range(n_agents):
        state_inputs.append(layers.Input(shape=(state_shape,), name=f'state_{i}'))
        action_inputs.append(layers.Input(shape=(action_dim,), name=f'action_{i}'))
    
    # 合并所有输入
    states = layers.Concatenate()(state_inputs) if len(state_inputs) > 1 else state_inputs[0]
    actions = layers.Concatenate()(action_inputs) if len(action_inputs) > 1 else action_inputs[0]
    
    # 特征提取
    state_features = layers.Dense(64, activation='relu')(states)
    state_features = layers.BatchNormalization()(state_features)
    
    action_features = layers.Dense(32, activation='relu')(actions)
    action_features = layers.BatchNormalization()(action_features)
    
    # 合并特征
    x = layers.Concatenate()([state_features, action_features])
    
    # 隐藏层
    prev_x = None
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, kernel_initializer='glorot_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)
        
        # 残差连接
        if use_residual and prev_x is not None and prev_x.shape[-1] == x.shape[-1]:
            x = layers.Add()([x, prev_x])
        prev_x = x
    
    # 输出层：Q值
    outputs = layers.Dense(1, kernel_initializer='glorot_uniform', name='q_value')(x)
    
    # 构建模型
    critic_model = Model(inputs=state_inputs + action_inputs, outputs=outputs, name='critic_network')
    
    return critic_model


class NetworkBuilder:
    """网络构建器类，提供统一的网络创建接口"""
    
    @staticmethod
    def create_actor(obs_dim, act_dim, hidden_units=(256, 128, 64), use_residual=True):
        """创建Actor网络"""
        return build_continuous_action_network(obs_dim, act_dim, hidden_units, use_residual)
    
    @staticmethod
    def create_critic(state_dim, act_dim, n_agents=3, hidden_units=(128, 64, 32), use_residual=True):
        """创建Critic网络"""
        return build_continuous_critic_network(state_dim, act_dim, hidden_units, n_agents, use_residual)
    
    @staticmethod
    def create_target_network(source_network):
        """创建目标网络（复制源网络的结构和权重）"""
        import copy
        target_network = tf.keras.models.clone_model(source_network)
        target_network.set_weights(source_network.get_weights())
        return target_network
    
    @staticmethod
    def soft_update(target_network, source_network, tau=0.01):
        """软更新目标网络权重"""
        target_weights = target_network.get_weights()
        source_weights = source_network.get_weights()
        
        updated_weights = []
        for target_weight, source_weight in zip(target_weights, source_weights):
            updated_weight = tau * source_weight + (1 - tau) * target_weight
            updated_weights.append(updated_weight)
        
        target_network.set_weights(updated_weights)

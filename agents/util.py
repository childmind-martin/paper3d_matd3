# Adapted partiall from https://keras.io/examples/rl/ddpg_pendulum/

import numpy as np
import random
import tensorflow as tf


class ReplayBuffer(object):

    def __init__(self, capacity, num_actions=None):
        """初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            num_actions: 动作空间维度（可选）
        """
        self.capacity = int(capacity)  # 确保容量是整数
        self.buffer = []
        self.position = 0
        self.num_actions = num_actions

    def push(self, state, action, reward, next_state, done):
        """添加一条经验到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """返回缓冲区中经验的数量"""
        return len(self.buffer)


# Class for Ornstein-Uhlenbeck Process
class OUNoise(object):
    """Ornstein-Uhlenbeck过程噪声（支持独立随机种子）"""
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, decay=0.9995, seed=None):
        self._theta = theta
        self._dt = dt
        self._mean = mean
        self._std_dev = std_dev
        self._decay = decay
        self._x_prev = np.zeros_like(self._mean)
        # 独立随机数发生器，避免多个实例间共享全局随机态
        try:
            self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        except Exception:
            self._rng = np.random.RandomState()
        self.reset()  # 将clear()改为reset()

    def __call__(self):
        """更新并返回噪声样本"""
        x = (
            self._x_prev
            + self._theta * (self._mean - self._x_prev) * self._dt
            + self._std_dev * np.sqrt(self._dt) * self._rng.normal(size=self._mean.shape)
        )
        self._x_prev = x
        return x

    def reset(self):
        """重置噪声状态"""
        self._x_prev = np.zeros_like(self._mean)

    def set_std_dev(self, std_dev):
        """设置噪声的标准差
        
        Args:
            std_dev: 新的标准差值
        """
        self._std_dev = std_dev
    
    def reseed(self, seed):
        """重新设置随机种子（可选）"""
        try:
            self._rng = np.random.RandomState(seed)
        except Exception:
            self._rng = np.random.RandomState()
        
    @property
    def std_dev(self):
        """获取噪声的标准差"""
        return self._std_dev
        
    @property
    def decay(self):
        """获取噪声衰减率"""
        return self._decay
    
    @decay.setter
    def decay(self, value):
        """设置噪声衰减率"""
        self._decay = value


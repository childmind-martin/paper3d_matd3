import numpy as np
import tensorflow as tf

class OUNoise:
    """Ornstein-Uhlenbeck过程噪声（支持独立随机种子）"""
    def __init__(self, size, mu=0.0, theta=0.15, std_dev=0.2, decay=0.9995, seed=None):
        """初始化参数和噪声过程
        
        Args:
            size: 动作空间的维度
            mu: 均值
            theta: 均值回归参数
            std_dev: 噪声标准差
            decay: 噪声衰减率
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.std_dev = std_dev
        self.decay = decay
        self.state = None
        # 独立随机状态，避免并行环境共享全局随机流
        try:
            self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        except Exception:
            self._rng = np.random.RandomState()
        self.reset()
        
    def reset(self):
        """重置内部状态（噪声）到均值"""
        self.state = np.ones(self.size) * self.mu
        
    def __call__(self):
        """更新内部状态并返回噪声样本"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.std_dev * self._rng.randn(self.size)
        self.state = x + dx
        # 注意：不要在这里衰减std_dev，应该在外部控制衰减
        return tf.convert_to_tensor(self.state, dtype=tf.float32)
    
    def get_noise(self):
        """获取当前噪声状态"""
        return tf.convert_to_tensor(self.state, dtype=tf.float32)
    
    def set_std_dev(self, std_dev):
        """设置噪声标准差
        
        Args:
            std_dev: 新的噪声标准差
        """
        self.std_dev = std_dev 
    
    def reseed(self, seed):
        """重新设置随机种子"""
        try:
            self._rng = np.random.RandomState(seed)
        except Exception:
            self._rng = np.random.RandomState()
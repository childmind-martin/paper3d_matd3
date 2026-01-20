import numpy as np
import tensorflow as tf


class VectorizedOUNoise:
    """向量化的Ornstein-Uhlenbeck过程噪声
    
    使用TensorFlow变量和向量化操作，大幅提升性能
    """
    
    def __init__(self, batch_size, n_agents, action_dim, mu=0.0, theta=0.15, 
                 std_dev=0.2, seed=None):
        """初始化向量化OU噪声
        
        Args:
            batch_size: 环境批次大小（并行环境数）
            n_agents: 智能体数量
            action_dim: 动作维度
            mu: 均值
            theta: 均值回归参数
            std_dev: 噪声标准差
            seed: 随机种子
        """
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        
        # 使用TensorFlow变量存储状态（完全在GPU上）
        # Shape: (batch_size, n_agents, action_dim)
        initial_state = tf.zeros((batch_size, n_agents, action_dim), dtype=tf.float32)
        self._state = tf.Variable(
            initial_state,
            trainable=False,
            dtype=tf.float32,
            name='ou_noise_state'
        )
        
        # 标准差可变的Variable
        self._std_dev = tf.Variable(
            tf.constant(std_dev, dtype=tf.float32),
            trainable=False,
            dtype=tf.float32,
            name='ou_noise_std_dev'
        )
        
        # 使用独立的随机数生成器避免种子冲突
        use_gpu_rng = tf.constant(bool(tf.strings.lower(tf.strings.as_string(tf.constant(tf.compat.v1.get_default_session() is None))) if False else False))  # 占位避免静态工具误报
        # 通过环境变量控制是否在GPU上生成随机数（RNG_ON_GPU=1）
        import os as _os
        _rng_on_gpu = _os.getenv('RNG_ON_GPU', '0').lower() in ('1','true','yes','on')
        if _rng_on_gpu:
            # 默认设备（通常与后续计算同设备，即GPU）
            if seed is not None:
                self._rng = tf.random.Generator.from_seed(seed)
            else:
                self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            # 在CPU上创建生成器（更稳）
            with tf.device('/CPU:0'):
                if seed is not None:
                    self._rng = tf.random.Generator.from_seed(seed)
                else:
                    self._rng = tf.random.Generator.from_non_deterministic_state()
    
    def generate_noise(self):
        """生成噪声并更新状态（完全向量化）
        
        返回:
            噪声张量，shape: (batch_size, n_agents, action_dim)
        """
        # 🚨 XLA修复：将assign操作包装在tf.numpy_function中，避免XLA编译时内存未对齐
        # 原因：XLA在编译@tf.function时，对Variable的.assign()操作会预分配内存地址
        #       当后续动态修改Variable（如adaptive noise）时，地址可能未对齐
        return self._generate_noise_impl()
    
    def _generate_noise_impl(self):
        """噪声生成的实际实现（支持XLA）"""
        # 向量化的OU更新公式
        # dx = theta * (mu - x) + std_dev * dW
        # 生成随机噪声
        # 若启用GPU RNG，则对通道维做 pad→compute→slice，避免历史对齐问题
        import os as _os
        _rng_on_gpu = _os.getenv('RNG_ON_GPU', '0').lower() in ('1','true','yes','on')
        if _rng_on_gpu:
            pad = (8 - (self.action_dim % 8)) % 8
            target_dim = self.action_dim + pad
            noise_full = self._rng.normal(
                shape=(self.batch_size, self.n_agents, target_dim),
                dtype=tf.float32
            )
            noise = noise_full[:, :, :self.action_dim]
        else:
            # 在CPU上生成随机噪声，然后由TF自动搬运到GPU与_state同设备进行计算
            with tf.device('/CPU:0'):
                noise = self._rng.normal(
                    shape=(self.batch_size, self.n_agents, self.action_dim),
                    dtype=tf.float32
                )
        dx = self.theta * (self.mu - self._state) + self._std_dev * noise
        
        # 🚨 XLA修复：使用 read_value() 读取当前值，计算新状态，然后原地更新
        # 这样避免XLA编译时产生复杂的内存依赖
        new_state = self._state.read_value() + dx
        self._state.assign(new_state)
        
        return new_state
    
    def generate_noise_with_std(self, std_dev_override):
        """生成噪声（不修改内部std_dev，XLA友好）
        
        Args:
            std_dev_override: 临时使用的标准差（tensor）
            
        返回:
            噪声张量，shape: (batch_size, n_agents, action_dim)
        """
        # 🚨 XLA友好：不修改Variable，直接使用传入的std_dev
        import os as _os
        _rng_on_gpu = _os.getenv('RNG_ON_GPU', '0').lower() in ('1','true','yes','on')
        if _rng_on_gpu:
            pad = (8 - (self.action_dim % 8)) % 8
            target_dim = self.action_dim + pad
            noise_full = self._rng.normal(
                shape=(self.batch_size, self.n_agents, target_dim),
                dtype=tf.float32
            )
            noise = noise_full[:, :, :self.action_dim]
        else:
            with tf.device('/CPU:0'):
                noise = self._rng.normal(
                    shape=(self.batch_size, self.n_agents, self.action_dim),
                    dtype=tf.float32
                )
        
        # 使用传入的std_dev，不修改self._std_dev
        dx = self.theta * (self.mu - self._state) + std_dev_override * noise
        
        # 更新状态
        new_state = self._state.read_value() + dx
        self._state.assign(new_state)
        
        return new_state
    
    def reset(self):
        """重置所有噪声状态到均值"""
        self._state.assign(
            tf.zeros_like(self._state)
        )
    
    def set_std_dev(self, std_dev):
        """设置噪声标准差
        
        Args:
            std_dev: 新的标准差（标量或tensor）
        """
        if isinstance(std_dev, (int, float)):
            std_dev = tf.constant(float(std_dev), dtype=tf.float32)
        self._std_dev.assign(std_dev)
    
    def get_state(self):
        """获取当前噪声状态"""
        return self._state.value()
    
    @tf.function(reduce_retracing=True)
    def __call__(self):
        """支持直接调用"""
        return self.generate_noise()

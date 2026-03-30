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

        # 关键约束：OU 的状态、标准差和随机数生成器必须共置于同一设备。
        # 否则在 XLA/GPU 图里读取 CPU 资源变量会触发跨设备 resource 错误。
        has_gpu = bool(tf.config.list_logical_devices('GPU'))
        self._resource_device = '/GPU:0' if has_gpu else '/CPU:0'
        self._pad_dim = (8 - (self.action_dim % 8)) % 8
        self._target_action_dim = self.action_dim + self._pad_dim

        with tf.device(self._resource_device):
            # 使用 TensorFlow 变量存储状态
            initial_state = tf.zeros((batch_size, n_agents, action_dim), dtype=tf.float32)
            self._state = tf.Variable(
                initial_state,
                trainable=False,
                dtype=tf.float32,
                name='ou_noise_state'
            )

            # 标准差可变的 Variable
            self._std_dev = tf.Variable(
                tf.constant(std_dev, dtype=tf.float32),
                trainable=False,
                dtype=tf.float32,
                name='ou_noise_std_dev'
            )

            # 使用同设备的随机数生成器，避免 CPU/GPU resource 交叉访问
            if seed is not None:
                self._rng = tf.random.Generator.from_seed(seed)
            else:
                self._rng = tf.random.Generator.from_non_deterministic_state()

    def _sample_noise(self):
        """在资源同设备上生成噪声，避免 XLA 的跨设备 resource 访问。"""
        with tf.device(self._resource_device):
            if self._pad_dim > 0:
                noise_full = self._rng.normal(
                    shape=(self.batch_size, self.n_agents, self._target_action_dim),
                    dtype=tf.float32
                )
                return noise_full[:, :, :self.action_dim]

            return self._rng.normal(
                shape=(self.batch_size, self.n_agents, self.action_dim),
                dtype=tf.float32
            )
    
    def generate_noise(self):
        """生成噪声并更新状态（完全向量化）
        
        返回:
            噪声张量，shape: (batch_size, n_agents, action_dim)
        """
        return self._generate_noise_impl()
    
    def _generate_noise_impl(self):
        """噪声生成的实际实现（支持XLA）"""
        with tf.device(self._resource_device):
            noise = self._sample_noise()
            dx = self.theta * (self.mu - self._state) + self._std_dev * noise

            # 使用 read_value() 读取当前值，计算新状态，然后原地更新
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
        with tf.device(self._resource_device):
            noise = self._sample_noise()
            std_dev_override = tf.cast(std_dev_override, tf.float32)

            # 使用传入的 std_dev，不修改 self._std_dev
            dx = self.theta * (self.mu - self._state) + std_dev_override * noise

            # 更新状态
            new_state = self._state.read_value() + dx
            self._state.assign(new_state)
            return new_state
    
    def reset(self):
        """重置所有噪声状态到均值"""
        with tf.device(self._resource_device):
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
        with tf.device(self._resource_device):
            self._std_dev.assign(std_dev)
    
    def get_state(self):
        """获取当前噪声状态"""
        with tf.device(self._resource_device):
            return self._state.value()
    
    @tf.function(reduce_retracing=True)
    def __call__(self):
        """支持直接调用"""
        return self.generate_noise()

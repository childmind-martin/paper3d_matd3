"""
观察数据处理模块
优化观察数据的预处理和内存管理
"""
import numpy as np
import tensorflow as tf


class ObservationProcessor:
    """观察数据处理器，支持内存池化和批量处理"""
    
    def __init__(self, n_agents=3, obs_dim=66, pool_size=64):
        """初始化观察处理器
        
        参数:
            n_agents: 智能体数量
            obs_dim: 观察空间维度
            pool_size: 内存池大小
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.pool_size = pool_size
        
        # 初始化内存池
        self._init_memory_pool()
        
        # 类型缓存
        self._type_cache = {}
        
        # 预分配的零向量
        self._zero_obs = np.zeros(obs_dim, dtype=np.float32)
        self._zero_batch = np.zeros((n_agents, obs_dim), dtype=np.float32)
    
    def _init_memory_pool(self):
        """初始化内存池"""
        self.memory_pool = {
            'obs_buffers': np.zeros((self.pool_size, self.obs_dim), dtype=np.float32),
            'batch_buffers': np.zeros((self.pool_size, self.n_agents, self.obs_dim), dtype=np.float32),
            'available_indices': list(range(self.pool_size)),
            'batch_available': list(range(self.pool_size))
        }
    
    def process_observation(self, obs_data, agent_index=0):
        """处理单个观察数据
        
        参数:
            obs_data: 原始观察数据
            agent_index: 智能体索引
            
        返回:
            processed_obs: 处理后的观察数据
        """
        # 快速类型检查
        obs_type = type(obs_data)
        
        # 使用类型缓存加速
        if obs_type in self._type_cache:
            processor = self._type_cache[obs_type]
            return processor(obs_data)
        
        # tuple 按 list 处理
        if obs_type is tuple:
            obs_data = list(obs_data)
            obs_type = list

        # 处理不同类型的观察数据
        if obs_type is np.ndarray:
            processed_obs = self._process_numpy_array(obs_data)
        elif obs_type is list:
            processed_obs = self._process_list(obs_data)
        elif obs_type is dict:
            processed_obs = self._process_dict(obs_data)
        else:
            # 尝试将任意可迭代对象转为浮点数组
            try:
                processed_obs = self._process_numpy_array(np.asarray(obs_data, dtype=np.float32))
            except Exception:
                processed_obs = self._zero_obs.copy()
        
        # 缓存处理方法
        if obs_type is np.ndarray:
            self._type_cache[obs_type] = self._process_numpy_array
        elif obs_type is list:
            self._type_cache[obs_type] = self._process_list
        elif obs_type is dict:
            self._type_cache[obs_type] = self._process_dict
        
        return processed_obs
    
    @staticmethod
    def _flatten_numeric(data):
        """递归收集 data 中的数值型标量，返回一维 float32 ndarray。"""
        out = []
        stack = [data]
        while stack:
            x = stack.pop()
            # numpy 标量
            if isinstance(x, (np.generic,)):
                out.append(float(x))
            # 纯标量
            elif isinstance(x, (int, float)):
                out.append(float(x))
            # ndarray
            elif isinstance(x, np.ndarray):
                if x.dtype == object:
                    # 对象数组：逐元素展开
                    stack.extend(list(x[::-1]))
                else:
                    out.extend(map(float, x.reshape(-1)))
            # 可迭代（list/tuple等）
            elif isinstance(x, (list, tuple)):
                stack.extend(list(x[::-1]))
            # 其他类型忽略
            else:
                continue
        return np.asarray(out, dtype=np.float32)

    def _process_numpy_array(self, obs_data):
        """处理numpy数组观察数据"""
        processed = self._flatten_numeric(obs_data)
        return self._adjust_dimension(processed)
    
    def _process_list(self, obs_data):
        """处理列表观察数据"""
        processed = self._flatten_numeric(obs_data)
        return self._adjust_dimension(processed)
    
    def _process_dict(self, obs_data):
        """处理字典观察数据"""
        values = []
        for k in sorted(obs_data.keys()):
            v = obs_data[k]
            if isinstance(v, (int, float)):
                values.append(v)
            elif isinstance(v, np.ndarray):
                values.extend(v.flatten())
        processed = np.array(values, dtype=np.float32)
        return self._adjust_dimension(processed)
    
    def _adjust_dimension(self, obs):
        """调整观察数据维度"""
        # 统一为一维 float32
        try:
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        except Exception:
            obs = self._zero_obs.copy()
        obs_size = obs.size
        
        if obs_size == self.obs_dim:
            return obs
        
        # 从内存池获取缓冲区
        if self.memory_pool['available_indices']:
            buffer_idx = self.memory_pool['available_indices'].pop()
            result = self.memory_pool['obs_buffers'][buffer_idx]
            result.fill(0)
            
            if obs_size < self.obs_dim:
                # 填充
                result[:obs_size] = obs
            else:
                # 截断
                result[:] = obs[:self.obs_dim]
            
            # 归还缓冲区
            self.memory_pool['available_indices'].append(buffer_idx)
            return result.copy()
        else:
            # 内存池已满，创建新数组
            result = np.zeros(self.obs_dim, dtype=np.float32)
            if obs_size < self.obs_dim:
                result[:obs_size] = obs
            else:
                result[:] = obs[:self.obs_dim]
            return result
    
    def batch_process_observations(self, obs_list):
        """批量处理观察数据
        
        参数:
            obs_list: 观察数据列表
            
        返回:
            processed_batch: 处理后的批量数据
        """
        batch_size = len(obs_list)
        
        # 直接创建结果数组
        result = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            processed = self.process_observation(obs, i)
            result[i] = processed
        
        return result
    
    @tf.function
    def tf_batch_process(self, obs_tensor):
        """TensorFlow优化的批量处理
        
        参数:
            obs_tensor: TensorFlow张量
            
        返回:
            processed_tensor: 处理后的张量
        """
        # 确保正确的形状
        shape = tf.shape(obs_tensor)
        batch_size = shape[0]
        
        # 调整维度
        if shape[-1] != self.obs_dim:
            if shape[-1] < self.obs_dim:
                # 填充
                padding = tf.zeros([batch_size, self.obs_dim - shape[-1]], dtype=tf.float32)
                processed = tf.concat([obs_tensor, padding], axis=-1)
            else:
                # 截断
                processed = obs_tensor[:, :self.obs_dim]
        else:
            processed = obs_tensor
        
        # 归一化处理（可选）
        # processed = tf.nn.l2_normalize(processed, axis=-1)
        
        return processed
    
    def reset_memory_pool(self):
        """重置内存池"""
        self.memory_pool['available_indices'] = list(range(self.pool_size))
        self.memory_pool['batch_available'] = list(range(self.pool_size))
        self.memory_pool['obs_buffers'].fill(0)
        self.memory_pool['batch_buffers'].fill(0)
    
    def get_memory_stats(self):
        """获取内存池统计信息"""
        return {
            'pool_size': self.pool_size,
            'available_obs_buffers': len(self.memory_pool['available_indices']),
            'available_batch_buffers': len(self.memory_pool['batch_available']),
            'memory_usage_mb': (self.memory_pool['obs_buffers'].nbytes + 
                              self.memory_pool['batch_buffers'].nbytes) / (1024 * 1024)
        }

    def batch_process_observations_parallel(self, obs_batch):
        """
        处理来自并行环境的批量观察数据。
        兼容多种输入：
          - ndarray 形状为 (num_envs, n_agents, obs_dim_raw)
          - list[n_agents]（单环境多智能体）
          - list[num_envs][n_agents]（多环境多智能体）
        """
        # 情况1：numpy 三维
        if isinstance(obs_batch, np.ndarray) and obs_batch.ndim == 3:
            num_envs = obs_batch.shape[0]
            processed_batch = np.zeros((num_envs, self.n_agents, self.obs_dim), dtype=np.float32)
            for i in range(num_envs):
                processed_batch[i] = self.batch_process_observations(list(obs_batch[i]))
            return processed_batch

        # 情况2：list 结构
        if isinstance(obs_batch, (list, tuple)):
            # 判定是否为单环境：长度等于智能体数，且其中元素不是“环境集合”
            if len(obs_batch) == self.n_agents and not (
                isinstance(obs_batch[0], (list, tuple)) and len(obs_batch[0]) == self.n_agents
            ):
                single_env = self.batch_process_observations(list(obs_batch))
                return single_env.reshape(1, self.n_agents, self.obs_dim)
            else:
                # 视为多环境，每个元素是一个[n_agents]列表
                processed_envs = []
                for env_obs in obs_batch:
                    processed_envs.append(self.batch_process_observations(list(env_obs)))
                return np.stack(processed_envs, axis=0)

        # 兜底：当成单环境
        single_env = self.batch_process_observations([obs_batch] * self.n_agents)
        return single_env.reshape(1, self.n_agents, self.obs_dim)

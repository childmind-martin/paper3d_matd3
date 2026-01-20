import numpy as np
import tensorflow as tf
import random
import time

class ParameterSpaceNoise:
    """参数空间噪声实现，为Actor网络的权重添加噪声而非动作"""
    
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, 
                 adoption_coefficient=1.01, scale_factor=1.0, decay=0.9995):
        """初始化参数空间噪声
        
        Args:
            initial_stddev: 初始权重扰动标准差
            desired_action_stddev: 期望的动作输出空间标准差
            adoption_coefficient: 噪声调整系数
            scale_factor: 噪声缩放因子
            decay: 噪声衰减率
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev
        self.scale_factor = scale_factor
        self.decay = decay
        
    def _get_random_seed(self):
        """生成随机种子，确保每次调用都不同"""
        seed = int(time.time() * 1000000) % 2**32
        seed = (seed + random.randint(0, 1000000)) % 2**32
        return seed
        
    def add_noise_to_weights(self, model, seed=None):
        """向模型权重添加噪声，返回新的噪声模型
        
        Args:
            model: 原始模型
            seed: 随机种子，如果为None则自动生成
            
        Returns:
            noisy_model: 添加噪声后的模型副本
        """
        if seed is None:
            seed = self._get_random_seed()
            
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        # 克隆模型
        noisy_model = tf.keras.models.clone_model(model)
        
        # 获取原始权重
        weights = model.get_weights()
        noisy_weights = []
        
        # 为每个权重添加高斯噪声
        for w in weights:
            # 根据权重形状生成噪声
            noise = np.random.normal(0, self.current_stddev * self.scale_factor, size=w.shape)
            noisy_weights.append(w + noise)
        
        # 设置噪声权重
        noisy_model.set_weights(noisy_weights)
        
        # 应用噪声衰减
        self.scale_factor *= self.decay
        
        return noisy_model
        
    def adapt(self, model, noisy_model, states):
        """自适应调整噪声标准差
        
        Args:
            model: 原始模型
            noisy_model: 噪声模型
            states: 用于比较动作差异的状态批次
            
        Returns:
            distance: 原始输出和噪声输出之间的差距
        """
        # 获取原始模型和噪声模型的预测
        states = np.array(states, dtype=np.float32)
        original_outputs = model(states)
        noisy_outputs = noisy_model(states)
        
        # 处理多输出模型情况（只比较第一个输出，即动作输出）
        if isinstance(original_outputs, list) and len(original_outputs) > 0:
            original_actions = original_outputs[0]
        else:
            original_actions = original_outputs
            
        if isinstance(noisy_outputs, list) and len(noisy_outputs) > 0:
            noisy_actions = noisy_outputs[0]
        else:
            noisy_actions = noisy_outputs
        
        # 转换为numpy数组进行计算
        if hasattr(original_actions, 'numpy'):
            original_actions = original_actions.numpy()
        if hasattr(noisy_actions, 'numpy'):
            noisy_actions = noisy_actions.numpy()
        
        # 计算两个动作序列之间的距离
        distance = np.sqrt(np.mean(np.square(original_actions - noisy_actions)))
        
        # 调整噪声大小
        if distance < self.desired_action_stddev:
            # 增加噪声
            self.current_stddev *= self.adoption_coefficient
        else:
            # 减少噪声
            self.current_stddev /= self.adoption_coefficient
            
        return distance
    
    @property
    def stddev(self):
        """获取当前标准差"""
        return self.current_stddev
    
    def set_stddev(self, stddev):
        """设置当前标准差"""
        self.current_stddev = stddev
        
    def reset_scale_factor(self):
        """重置缩放因子"""
        self.scale_factor = 1.0 
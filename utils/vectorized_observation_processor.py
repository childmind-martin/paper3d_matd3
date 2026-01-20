#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量化优化的观察数据处理器
针对并行环境进行性能优化，使用numpy向量化操作提升效率
"""

import numpy as np
import tensorflow as tf
from typing import Union, List, Tuple, Any
from .observation_processor import ObservationProcessor


class VectorizedObservationProcessor(ObservationProcessor):
    """向量化优化的观察数据处理器"""
    
    def __init__(self, n_agents=3, obs_dim=66, pool_size=64, use_vectorization=True):
        """初始化向量化观察处理器
        
        Args:
            n_agents: 智能体数量
            obs_dim: 观察空间维度
            pool_size: 内存池大小
            use_vectorization: 是否启用向量化优化
        """
        super().__init__(n_agents, obs_dim, pool_size)
        
        self.use_vectorization = use_vectorization
        # 调试/行为开关
        self.disable_clip = False
        self.obs_debug_sample = False
        
        # 向量化处理缓存
        self._vectorized_cache = {}
        
        # 预分配的大型数组（用于批量处理）
        self._large_arrays = {
            'max_batch_size': 32,
            'max_n_agents': 10,
            'temp_buffer': np.zeros((32, 10, obs_dim), dtype=np.float32),
            'clip_buffer': np.zeros((32, 10, obs_dim), dtype=np.float32),
            'normalize_buffer': np.zeros((32, 10, obs_dim), dtype=np.float32)
        }
        
        # 性能统计
        self._performance_stats = {
            'total_calls': 0,
            'vectorized_calls': 0,
            'fallback_calls': 0,
            'total_elements_processed': 0,
            'vectorization_savings': 0.0
        }
    
    def batch_process_observations_vectorized(self, obs_batch: Union[np.ndarray, List, Tuple]) -> np.ndarray:
        """
        完全向量化的批量观察处理
        
        Args:
            obs_batch: 观察数据批次
            
        Returns:
            processed_batch: 处理后的观察数据 (num_envs, n_agents, obs_dim)
        """
        self._performance_stats['total_calls'] += 1
        
        if not self.use_vectorization:
            return self.batch_process_observations_parallel(obs_batch)
        
        try:
            # 情况1：numpy 三维数组 - 完全向量化处理
            if isinstance(obs_batch, np.ndarray) and obs_batch.ndim == 3:
                out = self._process_3d_array_vectorized(obs_batch)
                # 首回合样本摘要（可选）
                if self.obs_debug_sample and not hasattr(self, '_obs_debug_once'):
                    try:
                        arr = out
                        print(f"[ObsDebug] shape={arr.shape} min={float(arr.min()):.3f} max={float(arr.max()):.3f} mean={float(arr.mean()):.3f}")
                        self._obs_debug_once = True
                    except Exception:
                        pass
                return out
            
            # 情况2：列表结构 - 转换为numpy后向量化处理
            elif isinstance(obs_batch, (list, tuple)):
                return self._process_list_vectorized(obs_batch)
            
            # 兜底：使用原始方法
            else:
                self._performance_stats['fallback_calls'] += 1
                return self.batch_process_observations_parallel(obs_batch)
                
        except Exception as e:
            print(f"向量化处理失败，回退到原始方法: {e}")
            self._performance_stats['fallback_calls'] += 1
            return self.batch_process_observations_parallel(obs_batch)
    
    def _process_3d_array_vectorized(self, obs_batch: np.ndarray) -> np.ndarray:
        """处理3D numpy数组 - 完全向量化"""
        num_envs, n_agents, obs_dim_raw = obs_batch.shape
        
        # 更新统计信息
        self._performance_stats['vectorized_calls'] += 1
        self._performance_stats['total_elements_processed'] += num_envs * n_agents * obs_dim_raw
        
        # 预分配输出数组
        processed_batch = np.zeros((num_envs, n_agents, self.obs_dim), dtype=np.float32)
        
        # 向量化处理：批量标准化、裁剪、填充
        if obs_dim_raw == self.obs_dim:
            # 维度匹配
            if self.disable_clip:
                processed_batch = obs_batch.astype(np.float32)
            else:
                processed_batch = np.clip(obs_batch, -1e6, 1e6).astype(np.float32)
            
        elif obs_dim_raw > self.obs_dim:
            # 维度过大，截断
            if self.disable_clip:
                processed_batch = obs_batch[:, :, :self.obs_dim].astype(np.float32)
            else:
                processed_batch = np.clip(obs_batch[:, :, :self.obs_dim], -1e6, 1e6).astype(np.float32)
            
        else:
            # 维度不足，填充
            if self.disable_clip:
                processed_batch[:, :, :obs_dim_raw] = obs_batch.astype(np.float32)
            else:
                processed_batch[:, :, :obs_dim_raw] = np.clip(obs_batch, -1e6, 1e6).astype(np.float32)
            # 其余部分保持为0（已在初始化时设置）
        
        return processed_batch
    
    def _process_list_vectorized(self, obs_batch: Union[List, Tuple]) -> np.ndarray:
        """处理列表结构 - 转换为numpy后向量化处理"""
        # 判定是否为单环境
        if len(obs_batch) == self.n_agents and not (
            isinstance(obs_batch[0], (list, tuple)) and len(obs_batch[0]) == self.n_agents
        ):
            # 单环境多智能体
            single_env = self._process_single_env_vectorized(obs_batch)
            return single_env.reshape(1, self.n_agents, self.obs_dim)
        else:
            # 多环境多智能体
            return self._process_multi_env_vectorized(obs_batch)
    
    def _process_single_env_vectorized(self, env_obs: List) -> np.ndarray:
        """向量化处理单环境观察数据"""
        # 转换为numpy数组
        env_array = np.array(env_obs, dtype=np.float32)
        
        # 确保形状正确
        if env_array.ndim == 1:
            env_array = env_array.reshape(1, -1)
        
        # 向量化处理
        processed_env = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        
        for i in range(min(self.n_agents, env_array.shape[0])):
            obs_raw = env_array[i]
            obs_dim_raw = len(obs_raw)
            
            # 标准化和裁剪
            obs_clipped = np.clip(obs_raw, -1e6, 1e6)
            
            if obs_dim_raw == self.obs_dim:
                processed_env[i] = obs_clipped
            elif obs_dim_raw > self.obs_dim:
                processed_env[i] = obs_clipped[:self.obs_dim]
            else:
                processed_env[i, :obs_dim_raw] = obs_clipped
        
        return processed_env
    
    def _process_multi_env_vectorized(self, obs_batch: List) -> np.ndarray:
        """向量化处理多环境观察数据"""
        num_envs = len(obs_batch)
        processed_envs = []
        
        for env_obs in obs_batch:
            processed_env = self._process_single_env_vectorized(env_obs)
            processed_envs.append(processed_env)
        
        return np.stack(processed_envs, axis=0)
    
    def batch_process_observations_parallel(self, obs_batch):
        """
        重写父类方法，优先使用向量化处理
        """
        if self.use_vectorization:
            return self.batch_process_observations_vectorized(obs_batch)
        else:
            return super().batch_process_observations_parallel(obs_batch)
    
    def preprocess_observations_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        批量预处理观察数据 - 高级向量化操作
        
        Args:
            obs_batch: 原始观察数据 (num_envs, n_agents, obs_dim_raw)
            
        Returns:
            processed_batch: 预处理后的数据 (num_envs, n_agents, obs_dim)
        """
        if not isinstance(obs_batch, np.ndarray) or obs_batch.ndim != 3:
            raise ValueError("输入必须是3D numpy数组")
        
        num_envs, n_agents, obs_dim_raw = obs_batch.shape
        
        # 预分配输出数组
        processed_batch = np.zeros((num_envs, n_agents, self.obs_dim), dtype=np.float32)
        
        # 高级向量化操作
        if obs_dim_raw == self.obs_dim:
            # 维度匹配，使用高级标准化
            processed_batch = self._advanced_normalize(obs_batch)
            
        elif obs_dim_raw > self.obs_dim:
            # 维度过大，使用特征选择
            processed_batch = self._feature_selection(obs_batch)
            
        else:
            # 维度不足，使用智能填充
            processed_batch = self._intelligent_padding(obs_batch)
        
        return processed_batch
    
    def _advanced_normalize(self, obs_batch: np.ndarray) -> np.ndarray:
        """高级标准化处理"""
        # 使用numpy的高级操作进行标准化
        obs_normalized = np.clip(obs_batch, -1e6, 1e6)
        
        # 可选：添加更复杂的标准化逻辑
        # 例如：按特征类型分组标准化
        # 这里保持简单，但可以扩展
        
        return obs_normalized.astype(np.float32)
    
    def _feature_selection(self, obs_batch: np.ndarray) -> np.ndarray:
        """特征选择处理"""
        # 简单的截断策略
        obs_selected = obs_batch[:, :, :self.obs_dim]
        
        # 可选：添加更智能的特征选择逻辑
        # 例如：基于重要性的特征选择
        
        return np.clip(obs_selected, -1e6, 1e6).astype(np.float32)
    
    def _intelligent_padding(self, obs_batch: np.ndarray) -> np.ndarray:
        """智能填充处理"""
        num_envs, n_agents, obs_dim_raw = obs_batch.shape
        processed_batch = np.zeros((num_envs, n_agents, self.obs_dim), dtype=np.float32)
        
        # 填充现有数据
        processed_batch[:, :, :obs_dim_raw] = np.clip(obs_batch, -1e6, 1e6)
        
        # 可选：添加更智能的填充策略
        # 例如：基于历史数据的填充、基于其他智能体数据的填充等
        
        return processed_batch
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        stats = self._performance_stats.copy()
        
        # 计算向量化效率
        if stats['total_calls'] > 0:
            vectorization_ratio = stats['vectorized_calls'] / stats['total_calls']
            stats['vectorization_efficiency'] = vectorization_ratio
            
            # 估算性能提升
            if vectorization_ratio > 0:
                stats['estimated_speedup'] = 1.0 + vectorization_ratio * 1.5
        
        # 计算处理效率
        if stats['total_elements_processed'] > 0:
            stats['elements_per_call'] = stats['total_elements_processed'] / stats['total_calls']
        
        return stats
    
    def clear_caches(self):
        """清理所有缓存"""
        super()._init_memory_pool()  # 重置父类缓存
        self._vectorized_cache.clear()
        
        # 重置性能统计
        self._performance_stats = {
            'total_calls': 0,
            'vectorized_calls': 0,
            'fallback_calls': 0,
            'total_elements_processed': 0,
            'vectorization_savings': 0.0
        }
    
    def benchmark_processing(self, obs_batch: np.ndarray, iterations: int = 100) -> dict:
        """
        基准测试处理性能
        
        Args:
            obs_batch: 测试数据
            iterations: 测试迭代次数
            
        Returns:
            Dict: 性能测试结果
        """
        import time
        
        # 测试向量化版本
        start_time = time.time()
        for _ in range(iterations):
            vectorized_result = self.batch_process_observations_vectorized(obs_batch)
        vectorized_time = time.time() - start_time
        
        # 测试原始版本
        self.use_vectorization = False
        start_time = time.time()
        for _ in range(iterations):
            original_result = self.batch_process_observations_parallel(obs_batch)
        original_time = time.time() - start_time
        self.use_vectorization = True
        
        # 计算性能提升
        speedup = original_time / vectorized_time if vectorized_time > 0 else 0
        
        return {
            'vectorized_time': vectorized_time,
            'original_time': original_time,
            'speedup': speedup,
            'iterations': iterations,
            'data_shape': obs_batch.shape,
            'result_shape': vectorized_result.shape
        }

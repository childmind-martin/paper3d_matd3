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
        
        # 单环境与多环境共用同一最后两维布局；这里只保留小型工作缓冲区。
        self._large_arrays = {
            'max_n_agents': 10,
            'temp_buffer': np.zeros((10, obs_dim), dtype=np.float32),
            'clip_buffer': np.zeros((10, obs_dim), dtype=np.float32),
            'normalize_buffer': np.zeros((10, obs_dim), dtype=np.float32)
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
        向量化处理单环境或多环境观察。
        
        Args:
            obs_batch:
              - list[n_agents] / ndarray (n_agents, obs_dim_raw)
              - list[num_envs][n_agents] /
                ndarray (num_envs, n_agents, obs_dim_raw)
            
        Returns:
            processed_batch: (num_envs, n_agents, obs_dim)。单环境的
            num_envs 为 1。
        """
        self._performance_stats['total_calls'] += 1
        
        if not self.use_vectorization:
            return super().batch_process_observations_parallel(obs_batch)
        
        try:
            if isinstance(obs_batch, np.ndarray):
                out = self._process_array_vectorized(obs_batch)
            elif isinstance(obs_batch, (list, tuple)):
                out = self._process_list_vectorized(obs_batch)
            else:
                raise TypeError(
                    f"不支持的观察批次类型: {type(obs_batch).__name__}"
                )

            if self.obs_debug_sample and not hasattr(self, '_obs_debug_once'):
                try:
                    print(
                        f"[ObsDebug] shape={out.shape} "
                        f"min={float(out.min()):.3f} "
                        f"max={float(out.max()):.3f} "
                        f"mean={float(out.mean()):.3f}"
                    )
                    self._obs_debug_once = True
                except Exception:
                    pass
            return out
                
        except Exception as e:
            print(f"向量化处理失败，回退到原始方法: {e}")
            self._performance_stats['fallback_calls'] += 1
            # 直接调用父类，避免通过本类 override 再次递归进入这里。
            return super().batch_process_observations_parallel(obs_batch)

    def _process_array_vectorized(self, obs_batch: np.ndarray) -> np.ndarray:
        """处理 (agents, features) 或 (envs, agents, features) 数组。"""
        obs_array = np.asarray(obs_batch)
        if obs_array.ndim == 2:
            obs_array = obs_array[np.newaxis, ...]
        elif obs_array.ndim != 3:
            raise ValueError(
                f"观察数组必须是2D或3D，收到 shape={obs_array.shape}"
            )

        num_envs, input_agents, obs_dim_raw = obs_array.shape
        processed_batch = np.zeros(
            (num_envs, self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        copied_agents = min(self.n_agents, input_agents)
        copied_features = min(self.obs_dim, obs_dim_raw)
        if copied_agents > 0 and copied_features > 0:
            source = obs_array[
                :, :copied_agents, :copied_features
            ].astype(np.float32, copy=False)
            if not self.disable_clip:
                source = np.clip(source, -1e6, 1e6)
            processed_batch[
                :, :copied_agents, :copied_features
            ] = source

        self._performance_stats['vectorized_calls'] += 1
        self._performance_stats['total_elements_processed'] += int(
            obs_array.size
        )
        return processed_batch
    
    def _process_2d_array_vectorized(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        处理2D numpy数组 (n_agents, obs_dim_raw)。
        """
        return self._process_array_vectorized(obs_batch)
    
    def _process_list_vectorized(self, obs_batch: Union[List, Tuple]) -> np.ndarray:
        """
        处理单环境或多环境列表；规则数组走同一向量化路径。
        """
        try:
            obs_array = np.asarray(obs_batch, dtype=np.float32)
        except (TypeError, ValueError):
            obs_array = None
        if obs_array is not None and obs_array.ndim in (2, 3):
            return self._process_array_vectorized(obs_array)

        # 兼容不规则列表。顶层元素为二维结构时代表多环境；
        # 否则按单环境的 agent 列表处理。
        if obs_batch:
            try:
                first_ndim = np.asarray(obs_batch[0], dtype=object).ndim
            except Exception:
                first_ndim = 0
            if first_ndim >= 2:
                return np.stack(
                    [
                        self._process_single_env_vectorized(env_obs)
                        for env_obs in obs_batch
                    ],
                    axis=0,
                )
        single_env = self._process_single_env_vectorized(obs_batch)
        return single_env[np.newaxis, ...]
    
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
        批量预处理单环境或多环境观察数据。
        
        Args:
            obs_batch: (num_envs, n_agents, obs_dim_raw) 或
                       (n_agents, obs_dim_raw)
            
        Returns:
            processed_batch: (num_envs, n_agents, obs_dim)
        """
        if not isinstance(obs_batch, np.ndarray):
            raise ValueError("输入必须是numpy数组")
        return self._process_array_vectorized(obs_batch)
    
    def _advanced_normalize_2d(self, obs_batch: np.ndarray) -> np.ndarray:
        """🔧 单环境模式：高级标准化处理 (n_agents, obs_dim)"""
        # 使用numpy的高级操作进行标准化
        obs_normalized = np.clip(obs_batch, -1e6, 1e6)
        return obs_normalized.astype(np.float32)
    
    def _feature_selection_2d(self, obs_batch: np.ndarray) -> np.ndarray:
        """🔧 单环境模式：特征选择处理 (n_agents, obs_dim_raw) -> (n_agents, obs_dim)"""
        # 简单的截断策略
        obs_selected = obs_batch[:, :self.obs_dim]
        return np.clip(obs_selected, -1e6, 1e6).astype(np.float32)
    
    def _intelligent_padding_2d(self, obs_batch: np.ndarray) -> np.ndarray:
        """🔧 单环境模式：智能填充处理 (n_agents, obs_dim_raw) -> (n_agents, obs_dim)"""
        n_agents, obs_dim_raw = obs_batch.shape
        processed_batch = np.zeros((n_agents, self.obs_dim), dtype=np.float32)
        
        # 填充现有数据
        processed_batch[:, :obs_dim_raw] = np.clip(obs_batch, -1e6, 1e6)
        
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

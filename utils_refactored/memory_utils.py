"""
内存管理模块 - 接口层
直接调用 paper3d_train_optimized.py 中的函数
"""
from paper3d_train_optimized import (
    _get_current_rss_mb,
    _get_mem_available_mb,
    _trim_memory,
    _clear_gpu_cache,
)

__all__ = [
    '_get_current_rss_mb',
    '_get_mem_available_mb',
    '_trim_memory',
    '_clear_gpu_cache',
]


"""
网络构建模块 - 接口层
直接调用 paper3d_train_optimized.py 中的函数
"""
from paper3d_train_optimized import (
    build_continuous_action_network,
    build_continuous_critic_network,
    build_continuous_critic_network_matd3,
)

__all__ = [
    'build_continuous_action_network',
    'build_continuous_critic_network',
    'build_continuous_critic_network_matd3',
]

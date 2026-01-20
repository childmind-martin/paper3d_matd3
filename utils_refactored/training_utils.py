"""
训练辅助模块 - 接口层
直接调用 paper3d_train_optimized.py 中的函数
"""
from paper3d_train_optimized import (
    _compute_loss_stats,
    _episode_success_without_collision,
    _compute_effective_steps_time_major,
)

__all__ = [
    '_compute_loss_stats',
    '_episode_success_without_collision',
    '_compute_effective_steps_time_major',
]


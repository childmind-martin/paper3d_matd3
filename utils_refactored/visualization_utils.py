"""
可视化工具模块 - 接口层
直接调用 paper3d_train_optimized.py 中的函数
"""
from paper3d_train_optimized import (
    get_plt,
    setup_english_fonts,
    _snapshot_env_trajectory,
    _derive_extent_from_trajectory,
    _plot_terrain_and_obstacles,
    _plot_trajectories_2d,
)

__all__ = [
    'get_plt',
    'setup_english_fonts',
    '_snapshot_env_trajectory',
    '_derive_extent_from_trajectory',
    '_plot_terrain_and_obstacles',
    '_plot_trajectories_2d',
]


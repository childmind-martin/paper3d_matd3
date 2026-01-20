"""
重构后的工具函数模块 - 接口层
所有函数都直接调用 paper3d_train_optimized.py 中的实现
保持模块化结构，但避免代码重复
"""
# 从各个子模块导入并重新导出
from .helpers import (
    _broadcast_force_ratio,
    _populate_common_cached_constants,
    parse_hidden_units,
    _scale_to_01,
    _extract_goal_positions_from_env,
    _apply_force_params_to_corrector,
    try_apply_scenario_params,
)
from .gpu_config import configure_gpu
from .scenario_loader import load_scenario_module
from .networks import (
    build_continuous_action_network,
    build_continuous_critic_network,
    build_continuous_critic_network_matd3,
)
from .visualization_utils import (
    get_plt,
    setup_english_fonts,
    _snapshot_env_trajectory,
    _derive_extent_from_trajectory,
    _plot_terrain_and_obstacles,
    _plot_trajectories_2d,
)
from .memory_utils import (
    _get_current_rss_mb,
    _get_mem_available_mb,
    _trim_memory,
    _clear_gpu_cache,
)
from .training_utils import (
    _compute_loss_stats,
    _episode_success_without_collision,
    _compute_effective_steps_time_major,
)

__all__ = [
    # Helpers
    '_broadcast_force_ratio',
    '_populate_common_cached_constants',
    'parse_hidden_units',
    '_scale_to_01',
    '_extract_goal_positions_from_env',
    '_apply_force_params_to_corrector',
    'try_apply_scenario_params',
    # GPU Config
    'configure_gpu',
    # Scenario Loader
    'load_scenario_module',
    # Networks
    'build_continuous_action_network',
    'build_continuous_critic_network',
    'build_continuous_critic_network_matd3',
    # Visualization
    'get_plt',
    'setup_english_fonts',
    '_snapshot_env_trajectory',
    '_derive_extent_from_trajectory',
    '_plot_terrain_and_obstacles',
    '_plot_trajectories_2d',
    # Memory
    '_get_current_rss_mb',
    '_get_mem_available_mb',
    '_trim_memory',
    '_clear_gpu_cache',
    # Training
    '_compute_loss_stats',
    '_episode_success_without_collision',
    '_compute_effective_steps_time_major',
]

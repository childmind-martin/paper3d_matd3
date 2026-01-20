"""
辅助工具函数模块 - 接口层
直接调用 paper3d_train_optimized.py 中的函数，避免代码重复
"""
# 直接从原文件导入所有工具函数
from paper3d_train_optimized import (
    _broadcast_force_ratio,
    parse_hidden_units,
    _populate_common_cached_constants,
    _scale_to_01,
    _apply_force_params_to_corrector,
    try_apply_scenario_params,
    _extract_goal_positions_from_env,
)

# 重新导出，保持接口一致
__all__ = [
    '_broadcast_force_ratio',
    'parse_hidden_units',
    '_populate_common_cached_constants',
    '_scale_to_01',
    '_apply_force_params_to_corrector',
    'try_apply_scenario_params',
    '_extract_goal_positions_from_env',
]

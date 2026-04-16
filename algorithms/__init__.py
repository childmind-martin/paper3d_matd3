"""
算法模块 - 接口层
直接调用 paper3d_train_optimized.py 中的算法类
"""
from paper3d_train_optimized import OptimizedMADDPG, OptimizedMATD3

try:
    from algorithms.mappo import OptimizedMAPPO
except ModuleNotFoundError:
    OptimizedMAPPO = None

__all__ = ['OptimizedMADDPG', 'OptimizedMATD3']
if OptimizedMAPPO is not None:
    __all__.append('OptimizedMAPPO')

"""
势场修正模块 - 接口层
直接调用 paper3d_train_optimized.py 中的势场相关函数和类

注意：势场修正的主要方法在 OptimizedMADDPG 和 OptimizedMATD3 类中
如果需要独立使用，可以从 algorithms 模块导入这些类
"""
# 势场修正器类（如果原文件中有独立导出）
try:
    from potential_field_corrector import ContinuousPotentialFieldCorrector
    __all__ = ['ContinuousPotentialFieldCorrector']
except ImportError:
    # 如果不存在独立导出，则只导出算法类中的方法
    __all__ = []


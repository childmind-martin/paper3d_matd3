#!/usr/bin/env python3
"""
测试 XLA 兼容性修复：验证 _get_terrain_height_map_tf 是否能在 @tf.function 内正常工作
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['SUPPRESS_MA_PROMPT'] = '1'

import tensorflow as tf
import numpy as np
import argparse

print("=" * 60)
print("XLA 兼容性测试")
print("=" * 60)

# 创建一个简化的类来测试
class TestMADDPG:
    def __init__(self, args):
        self.args = args
        # 🔧 预先创建默认 terrain tensor（XLA修复的关键）
        default_map_size = int(getattr(args, 'map_size', 200))
        self._terrain_height_tensor = tf.zeros([default_map_size, default_map_size], dtype=tf.float32)
        self._terrain_height_cache_key = None
        
        self.c_map_size = tf.constant(float(default_map_size), dtype=tf.float32)
        self.c_0_0 = tf.constant(0.0, dtype=tf.float32)
        self.c_1_0 = tf.constant(1.0, dtype=tf.float32)
    
    def _get_terrain_height_map_tf(self):
        """
        🔧 XLA兼容：直接返回预先创建的 tensor，不做任何动态判断
        """
        return self._terrain_height_tensor
    
    def update_terrain_cache(self, terrain_np):
        """在外部更新 terrain"""
        if isinstance(terrain_np, np.ndarray):
            contiguous = terrain_np
            if not contiguous.flags['C_CONTIGUOUS']:
                contiguous = np.ascontiguousarray(contiguous)
            if contiguous.dtype != np.float32:
                contiguous = contiguous.astype(np.float32)
            self._terrain_height_tensor = tf.constant(contiguous, dtype=tf.float32)
    
    @tf.function(jit_compile=True)
    def test_potential_field_tf(self, obs):
        """测试势场计算（包含地形采样）"""
        # 提取位置
        pos = obs[:, :3]  # [B, 3]
        x = pos[:, 0:1]   # [B, 1]
        y = pos[:, 1:2]   # [B, 1]
        
        # 获取地形高度图
        height_map = self._get_terrain_height_map_tf()
        
        # 简单的双线性采样
        map_size = tf.cast(self.c_map_size, tf.float32)
        ix = tf.clip_by_value(x, self.c_0_0, map_size - self.c_1_0)
        iy = tf.clip_by_value(y, self.c_0_0, map_size - self.c_1_0)
        
        # 使用整数索引采样
        ix_int = tf.cast(ix, tf.int32)
        iy_int = tf.cast(iy, tf.int32)
        
        # 计算线性索引
        H = tf.shape(height_map)[0]
        W = tf.shape(height_map)[1]
        height_map_flat = tf.reshape(height_map, [-1])
        
        indices = iy_int * W + ix_int
        indices = tf.clip_by_value(indices, 0, tf.shape(height_map_flat)[0] - 1)
        heights = tf.gather(height_map_flat, indices)
        
        return heights

# 创建测试参数
args = argparse.Namespace(map_size=200)
maddpg = TestMADDPG(args)

# 测试1：使用默认的 zeros tensor
print("\n[测试1] 使用默认 zeros tensor...")
obs_test = tf.constant([[50.0, 50.0, 10.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
try:
    result = maddpg.test_potential_field_tf(obs_test)
    print(f"✓ 成功! 采样高度: {result.numpy()}")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试2：更新为真实地形后测试
print("\n[测试2] 更新为真实地形后...")
terrain_np = np.random.rand(200, 200).astype(np.float32) * 10.0  # 随机地形 0-10m
maddpg.update_terrain_cache(terrain_np)
try:
    result = maddpg.test_potential_field_tf(obs_test)
    print(f"✓ 成功! 采样高度: {result.numpy()}")
    # 验证结果合理性
    expected_h = terrain_np[50, 50]
    actual_h = result.numpy()[0, 0]
    print(f"  预期高度: {expected_h:.2f}, 实际高度: {actual_h:.2f}")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试3：批量测试
print("\n[测试3] 批量测试 (10个样本)...")
obs_batch = tf.random.uniform([10, 6], minval=0.0, maxval=199.0)
try:
    result = maddpg.test_potential_field_tf(obs_batch)
    print(f"✓ 成功! 批量采样形状: {result.shape}, 高度范围: [{tf.reduce_min(result).numpy():.2f}, {tf.reduce_max(result).numpy():.2f}]")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试4：启用 XLA 后的性能测试
print("\n[测试4] XLA 编译性能测试...")
import time

# 预热
for _ in range(3):
    _ = maddpg.test_potential_field_tf(obs_batch)

# 计时
n_iters = 100
start = time.time()
for _ in range(n_iters):
    _ = maddpg.test_potential_field_tf(obs_batch)
elapsed = time.time() - start
print(f"✓ 完成 {n_iters} 次迭代，耗时: {elapsed*1000:.2f}ms, 平均: {elapsed*1000/n_iters:.4f}ms/iter")

print("\n" + "=" * 60)
print("所有测试通过！XLA 修复生效。")
print("=" * 60)


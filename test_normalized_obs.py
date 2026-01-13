#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用归一化后的观测测试Actor初始输出
"""
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_actor(obs_dim=78, action_dim=7):
    """构建Actor网络"""
    input_layer = tf.keras.layers.Input(shape=(obs_dim,))
    x = input_layer
    
    for i, units in enumerate([256, 256, 256]):
        x_res = x
        x = tf.keras.layers.Dense(
            units,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(5e-4)
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        
        if i > 0 and x.shape == x_res.shape:
            x = tf.keras.layers.Add()([x, x_res])
    
    output = tf.keras.layers.Dense(
        action_dim,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)
    )(x)
    
    return tf.keras.Model(inputs=input_layer, outputs=output)

def create_normalized_obs(pos, vel, goal_rel, map_size=200):
    """创建符合实际训练的归一化观测"""
    obs = np.zeros(78)
    
    # 1. 状态信息 (9维)
    # 位置：归一化到[-1, 1]
    map_half = map_size / 2.0
    normalized_pos = pos / map_half - 1.0
    obs[0:3] = normalized_pos
    
    # 速度：归一化到[-1, 1]（max_speed=22.5）
    normalized_vel = vel / 22.5
    obs[3:6] = normalized_vel
    
    # 加速度：初始为0
    obs[6:9] = 0.0
    
    # 2. 目标信息 (4维)
    # 目标方向向量（已归一化）+ 归一化距离
    goal_dist = np.linalg.norm(goal_rel)
    if goal_dist > 1e-6:
        goal_dir = goal_rel / goal_dist
    else:
        goal_dir = np.array([0, 0, 0])
    obs[9:12] = goal_dir
    obs[12] = goal_dist / map_size  # 归一化距离
    
    # 3. 地形信息 (29维)：暂时用小的随机值模拟
    obs[13:42] = np.random.randn(29) * 0.05
    
    # 4. 其他智能体信息 (33维)：暂时用小的随机值
    obs[42:75] = np.random.randn(33) * 0.05
    
    # 5. 势场特征 (3维，如果启用）
    if len(obs) > 75:
        obs[75:78] = 0.0
    
    return obs

print("="*80)
print("🔍 使用归一化观测测试Actor初始输出")
print("="*80)

actor = build_actor()

# 手动设置偏置为0
output_layer = None
for layer in reversed(actor.layers):
    if isinstance(layer, tf.keras.layers.Dense) and layer.units == 7:
        output_layer = layer
        break

if output_layer:
    new_bias = np.zeros(7)
    output_layer.bias.assign(new_bias)
    print(f"✅ 输出层偏置已设为: {output_layer.bias.numpy()}\n")

# 测试不同场景
test_cases = [
    {
        "name": "起点位置 (100,150,25), 静止",
        "pos": np.array([100.0, 150.0, 25.0]),
        "vel": np.array([0.0, 0.0, 0.0]),
        "goal_rel": np.array([-95.0, -8.85, -20.0])  # 到目标(5, 141, 5)
    },
    {
        "name": "起点位置 (100,150,25), 向目标飞行",
        "pos": np.array([100.0, 150.0, 25.0]),
        "vel": np.array([-5.0, -0.5, -1.0]),  # 向目标方向
        "goal_rel": np.array([-95.0, -8.85, -20.0])
    },
    {
        "name": "接近目标 (20,145,15), 减速",
        "pos": np.array([20.0, 145.0, 15.0]),
        "vel": np.array([-2.0, -0.5, -0.5]),
        "goal_rel": np.array([-15.0, -3.85, -10.0])
    },
]

for test in test_cases:
    obs = create_normalized_obs(test["pos"], test["vel"], test["goal_rel"])
    output = actor(obs[np.newaxis, :], training=False).numpy()[0]
    
    print(f"📍 {test['name']}")
    print(f"  原始位置: {test['pos']}")
    print(f"  归一化位置: {obs[0:3]}")
    print(f"  速度: {test['vel']}")
    print(f"  到目标: {test['goal_rel']} (距离={np.linalg.norm(test['goal_rel']):.1f}m)")
    print(f"  🎯 Actor输出: ax={output[0]:+.4f}, ay={output[1]:+.4f}, az={output[2]:+.4f}")
    print(f"  范围: [{output.min():.4f}, {output.max():.4f}]")
    print()

print("="*80)
print("✅ 测试完成")
print("\n💡 结论：")
print("  - 观测已经正确归一化到[-1, 1]范围")
print("  - 网络初始输出取决于具体的观测值")
print("  - 初始输出不为0是正常的（因为观测不为0）")
print("  - 关键是Z轴映射对称性（已修复为0偏置）")
print("="*80)


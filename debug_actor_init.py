#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断Actor网络初始化状态
检查初始权重、偏置和输出
"""
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_actor(obs_dim=78, action_dim=7):
    """构建Actor网络（与训练脚本相同）"""
    input_layer = tf.keras.layers.Input(shape=(obs_dim,))
    x = input_layer
    
    # 3层隐藏层
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
    
    # 输出层
    output = tf.keras.layers.Dense(
        action_dim,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)
    )(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

def main():
    print("="*80)
    print("🔍 Actor网络初始化诊断")
    print("="*80)
    
    # 构建网络
    actor = build_actor()
    
    # 检查输出层权重和偏置
    print("\n📊 输出层参数检查：")
    output_layer = None
    for layer in reversed(actor.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 7:
            output_layer = layer
            break
    
    if output_layer:
        weights = output_layer.kernel.numpy()
        bias = output_layer.bias.numpy()
        
        print(f"  - 权重形状: {weights.shape}")
        print(f"  - 权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
        print(f"  - 权重均值: {weights.mean():.6f}, 标准差: {weights.std():.6f}")
        print(f"  - 偏置值: {bias}")
        print(f"  - 偏置范围: [{bias.min():.6f}, {bias.max():.6f}]")
    
    # 手动设置偏置为0（模拟训练脚本的操作）
    print("\n🔧 手动设置所有偏置为0...")
    if output_layer:
        new_bias = np.zeros_like(bias)
        output_layer.bias.assign(new_bias)
        print(f"  - 设置后偏置: {output_layer.bias.numpy()}")
    
    # 创建不同的测试输入
    test_cases = [
        ("全零输入", np.zeros((1, 78))),
        ("标准正态输入", np.random.randn(1, 78)),
        ("小范围均匀输入", np.random.uniform(-0.1, 0.1, (1, 78))),
        ("典型观测输入（模拟）", np.concatenate([
            np.array([[100, 150, 25]]),  # 位置
            np.array([[0, 0, 0]]),  # 速度
            np.random.randn(1, 72) * 0.1  # 其他观测
        ], axis=1))
    ]
    
    print("\n📈 不同输入下的初始输出：")
    for name, obs in test_cases:
        output = actor(obs, training=False).numpy()[0]
        print(f"\n  {name}:")
        print(f"    ax={output[0]:+.4f}, ay={output[1]:+.4f}, az={output[2]:+.4f}")
        print(f"    范围: [{output.min():.4f}, {output.max():.4f}]")
        print(f"    均值: {output.mean():.4f}, 标准差: {output.std():.4f}")
    
    # 检查中间层输出
    print("\n🔬 中间层输出分析（全零输入）：")
    obs_zero = np.zeros((1, 78))
    layer_outputs = []
    x = obs_zero
    for i, layer in enumerate(actor.layers):
        if len(layer.weights) > 0 or 'normalization' in layer.name.lower():
            x_prev = x
            x = layer(x, training=False).numpy()
            print(f"  Layer {i} ({layer.name}):")
            print(f"    输出范围: [{x.min():.6f}, {x.max():.6f}]")
            print(f"    输出均值: {x.mean():.6f}, 标准差: {x.std():.6f}")
            if x.shape[-1] == 7:  # 输出层
                print(f"    动作值: ax={x[0,0]:+.4f}, ay={x[0,1]:+.4f}, az={x[0,2]:+.4f}")
    
    print("\n" + "="*80)
    print("✅ 诊断完成")
    print("="*80)

if __name__ == '__main__':
    main()


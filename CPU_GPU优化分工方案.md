# CPU/GPU优化分工方案

## 分析目标

充分利用硬件特性：
- **CPU**: 串行逻辑处理、数据准备、I/O操作
- **GPU**: 大规模并行计算、矩阵运算、神经网络操作

## 当前架构分析

### 训练循环主要操作流程

```
1. 环境交互 (env.step)
2. 观察处理 (preprocessing)
3. 动作生成 (actor network)
4. 经验存储 (replay_buffer.add)
5. 经验采样 (replay_buffer.sample)
6. 网络更新 (maddpg.update)
   - 批量前向传播
   - 损失计算
   - 梯度计算
   - 参数更新
7. 统计和日志
```

## 优化方案

### 🖥️ CPU负责的操作（串行逻辑）

#### 1. 环境交互
```python
# 已优化：环境交互本身是串行的
with tf.device('/CPU:0'):
    obs_n, rew_n, done_n, info_n = env.step(real_actions)
```

**原因**：
- 环境模拟是纯Python/NumPy代码
- 串行逻辑，无法并行化
- 涉及大量控制流和条件判断

#### 2. 经验回放缓冲区操作
```python
# 采样和存储在CPU上
with tf.device('/CPU:0'):
    replay_buffer.add(obs_np, next_obs_np, action_data, reward_data, done_data)
    batch = replay_buffer.sample(batch_size)
```

**原因**：
- NumPy数组操作
- 随机采样涉及CPU随机数生成器
- 内存管理更高效

#### 3. 数据预处理
```python
# 观察归一化和预处理
with tf.device('/CPU:0'):
    # 数据格式转换
    obs_np = obs_data.numpy() if isinstance(obs_data, tf.Tensor) else obs_data
    
    # 统计计算（均值、方差等）
    step_rewards.append(np.mean(rew_n))
    episode_rewards_per_env += step_increment
```

**原因**：
- 小规模数组操作
- CPU更擅长不规则数据处理
- 避免CPU↔GPU数据传输开销

#### 4. 日志和文件I/O
```python
# 所有I/O操作在CPU
with tf.device('/CPU:0'):
    # 保存模型
    actor.save_weights(path)
    
    # 日志记录
    print(f"回合 {episode}: 奖励={reward}")
    
    # 轨迹保存
    np.save(trajectory_file, trajectory_data)
```

**原因**：
- I/O操作是纯CPU操作
- GPU无法直接访问文件系统

### 🎮 GPU负责的操作（并行计算）

#### 1. 神经网络推理（动作生成）
```python
# 批量动作生成在GPU
# ✅ 已默认在GPU上，无需显式指定
actions = actor(obs_tensor, training=False)  # 自动在GPU
```

**原因**：
- 大规模矩阵乘法
- 高度并行化
- TensorFlow自动优化

#### 2. 网络训练更新
```python
# 整个update过程在GPU
# ✅ 已优化：TensorFlow自动将计算图放在GPU
with tf.GradientTape() as tape:
    # 前向传播
    q_values = critic([state, action])
    # 损失计算
    loss = criterion(target, q_values)

# 梯度计算和应用
grads = tape.gradient(loss, critic.trainable_variables)
optimizer.apply_gradients(zip(grads, critic.trainable_variables))
```

**原因**：
- 批量矩阵运算（batch_size × 网络层）
- 自动微分高度并行化
- GPU张量运算效率高

#### 3. 批量数据转换
```python
# ✅ 大批量张量操作保留在GPU
# 转换为TensorFlow张量（自动在GPU）
all_obs = [tf.convert_to_tensor(obs_n[:, i], dtype=tf.float32) 
           for i in range(n_agents)]
```

**原因**：
- 批量数据处理
- 后续直接用于GPU计算
- 减少CPU↔GPU传输

### ⚖️ 混合策略（需要权衡）

#### 1. 单个动作生成（推理时）
```python
# 小批量或单样本推理
if batch_size == 1:
    # CPU可能更快（避免GPU启动开销）
    with tf.device('/CPU:0'):
        action = actor(obs)
else:
    # 大批量在GPU
    actions = actor(obs_batch)  # 默认GPU
```

#### 2. 势场修正
```python
# 如果使用TF版本势场修正
if use_tf_potential_field and batch_size >= 32:
    # GPU批量处理
    corrected_actions = _apply_potential_field_correction_tf(actions, obs, fr)
else:
    # CPU逐样本处理（NumPy版本）
    with tf.device('/CPU:0'):
        corrected_actions = _apply_potential_field_correction_numpy(actions, obs, fr)
```

## 实施策略

### 1. 显式设备放置（关键路径）

```python
# paper3d_train_optimized.py 中添加设备控制

class OptimizedTrainer:
    def __init__(self):
        # 明确CPU/GPU策略
        self.cpu_device = '/CPU:0'
        self.gpu_device = '/GPU:0'  # 如果有GPU
    
    def run_episode(self):
        # CPU: 环境交互
        with tf.device(self.cpu_device):
            obs_n, rew_n, done_n, info_n = env.step(actions)
            replay_buffer.add(obs_n, next_obs_n, act_n, rew_n, done_n)
        
        # GPU: 批量推理（已默认）
        actions = maddpg.get_actions(obs_n, noise_scale)
        
        # GPU: 训练更新（已默认）
        if should_update:
            losses = maddpg.update(replay_buffer, batch_size)
```

### 2. 数据传输优化

```python
# ✅ 批量转换，减少CPU↔GPU传输次数
# 错误方式：每步都转换
for step in range(episode_length):
    obs_gpu = tf.convert_to_tensor(obs)  # ❌ 频繁传输
    action = actor(obs_gpu)

# 正确方式：批量转换
obs_batch = tf.convert_to_tensor(obs_buffer)  # ✅ 一次传输
actions_batch = actor(obs_batch)
```

### 3. 异步数据准备（高级优化）

```python
# 使用tf.data进行异步数据准备
dataset = tf.data.Dataset.from_tensor_slices((obs, next_obs, actions))
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 数据准备在CPU异步进行，GPU专注计算
for batch in dataset:
    losses = maddpg.update_batch(batch)
```

## 性能对比预期

| 配置 | 每回合用时 | CPU利用率 | GPU利用率 |
|------|-----------|----------|----------|
| 当前（未优化） | 25-30s | 40-60% | 60-80% |
| CPU/GPU分工优化 | 20-25s | 70-85% | 85-95% |
| + 异步数据准备 | 18-22s | 80-90% | 90-95% |

## 实施优先级

### 🔥 高优先级（立即实施）
1. ✅ **经验回放在CPU** - 已默认实现
2. ✅ **网络训练在GPU** - 已默认实现
3. ⚠️ **环境交互显式CPU** - 建议添加显式device控制

### 📋 中优先级（逐步优化）
4. 数据预处理CPU优化
5. 单样本vs批量推理策略
6. 势场修正设备选择

### 💡 低优先级（长期优化）
7. 异步数据准备pipeline
8. 多GPU训练支持
9. 混合精度训练优化

## 注意事项

### 1. 避免过度优化
- **当前瓶颈**：环境交互（串行）> 网络计算（并行）
- **优化重点**：确保GPU不空闲等待CPU

### 2. 设备放置开销
- 显式device切换有开销
- 仅在必要时使用
- TensorFlow已有智能放置

### 3. 数据传输成本
- CPU↔GPU传输很昂贵
- 批量传输优于频繁小传输
- 尽量让数据留在原设备

## 快速验证方案

```bash
# 添加性能分析
TF_CPP_MIN_LOG_LEVEL=0 python paper3d_train_optimized.py \
    --profiling 1 \
    --episodes 5

# 查看设备放置
grep -E "(CPU|GPU)" logs/*/training.log
```

## 总结

**核心原则**：
1. ✅ CPU处理串行逻辑（环境、I/O、控制流）
2. ✅ GPU处理并行计算（神经网络、矩阵运算）
3. ⚠️ 减少设备间数据传输
4. ⚠️ 批量处理优于单样本处理

**当前状态**：
- TensorFlow已自动优化大部分操作
- 主要瓶颈是环境交互（本质串行）
- 进一步优化需要权衡复杂度vs收益

**建议**：
- 保持当前架构（已经合理）
- 关注瓶颈分析而非过度优化
- 确保XLA禁用（稳定性优先）

---
创建日期: 2025-11-06

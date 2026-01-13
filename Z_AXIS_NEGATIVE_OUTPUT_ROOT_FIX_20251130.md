# Z轴负向输出问题根本修复 - 2025-11-30

## 🎯 问题根源

### 用户诊断（正确）

**问题描述**：训练若干回合后，Actor网络总是趋向于Z轴负向输出，导致智能体下坠。

**根本原因**：梯度回传错误。

之前的代码逻辑：
1. Actor 输出原始动作（例如 `z = -0.5`，向下）
2. 势场修正 → `z_corrected = +0.3`（向上）
3. **环境执行修正后的动作**
4. **回放区存储修正后的动作**
5. Critic 学习 `Q(s, a_corrected)` = 高分（因为修正后是好的）
6. Actor 优化 `Q(s, a_corrected)`，梯度告诉它："你的 `-0.5` 很好！"
7. Actor 错误地学到：**输出负 Z 是对的**

**本质矛盾**：Actor 输出负 Z → 势场修正成正 Z → Critic 给高分 → Actor 继续输出负 Z

## ✅ 修复方案

### 核心思想

**分离"存储动作"和"执行动作"**：

1. **原始动作**（Actor 输出 + 噪声，**无势场修正**）→ 存入回放区
2. **修正动作**（Actor 输出 + 噪声 + **势场修正**）→ 交给环境执行
3. Critic 学习 `Q(s, a_raw)`，能正确评估**原始动作**的价值
4. Actor 优化 `Q(s, a_raw)`，梯度告诉它："输出负 Z 会导致低 Q 值"
5. Actor 学会：**应该输出正 Z**

### 关键思想

- **势场修正是执行时的安全保障**，不是学习目标
- **Actor 应该学会本身就安全的动作**，而不是依赖势场"托住"它
- **Critic 评估的是原始策略的价值**，而不是修正后的

## 🔧 代码修改

### 1. `batch_select_actions_vectorized` 返回三个值

**文件**：`paper3d_train_optimized.py`
**位置**：第 7229-7316 行

#### 修改前：
```python
# 返回两个值：
# 1. actions: 最终动作（修正后）
# 2. pf_forces: 势场力
return actions, pf_forces
```

#### 修改后：
```python
# 1. 先对原始动作添加噪声（用于探索）
if add_noise:
    raw_actions_with_noise = raw_actor_actions + OU_noise
else:
    raw_actions_with_noise = raw_actor_actions

# 2. 对加噪声后的动作进行势场修正（用于安全执行）
if should_apply_pf:
    network_actions_corrected, pf_forces = apply_pf_correction(raw_actions_with_noise)
else:
    network_actions_corrected = raw_actions_with_noise

# 3. 混合随机动作
actions_for_storage = mix(rand_actions_raw, raw_actions_with_noise)  # 无修正
actions_for_execution = mix(rand_actions_raw, network_actions_corrected)  # 有修正

# 返回三个值：
# 1. actions_for_storage: 存入回放区（原始+噪声，无修正）
# 2. actions_for_execution: 交给环境（原始+噪声+修正）
# 3. pf_forces_final: 势场力矢量
return actions_for_storage, actions_for_execution, pf_forces_final
```

### 2. 训练循环接收三个返回值

**位置**：第 10966-10977 行

#### 修改：
```python
# 旧代码：
# actions_tensor, pf_forces_tensor = maddpg.batch_select_actions_vectorized(...)

# 新代码：
actions_for_storage_tensor, actions_for_execution_tensor, pf_forces_tensor = \
    maddpg.batch_select_actions_vectorized(...)
```

### 3. Tensor 转 NumPy 处理三个值

**位置**：第 10995-11096 行

#### 修改：
```python
# 处理三个返回值
if isinstance(actions_tensor, tuple) and len(actions_tensor) == 3:
    actions_for_storage_tf, actions_for_execution_tf, pf_forces_tf = actions_tensor
    # 转换为 NumPy
    actions_for_storage = actions_for_storage_tf.numpy()
    actions = actions_for_execution_tf.numpy()  # 执行动作
    pf_forces_np = pf_forces_tf.numpy()
```

### 4. 存储动作使用 `actions_for_storage`

**位置**：第 11357-11382 行

#### 修改前：
```python
real_actions = actions.copy()  # 使用环境执行的动作（有修正）
# 映射到环境尺度
real_actions[..., 0] = actions[..., 0] * arx
real_actions[..., 1] = actions[..., 1] * ary
real_actions[..., 2] = (actions[..., 2] + z_bias) * arz * gz
```

#### 修改后：
```python
real_actions = actions_for_storage.copy()  # 使用存储动作（无修正）
# 映射到环境尺度
real_actions[..., 0] = actions_for_storage[..., 0] * arx
real_actions[..., 1] = actions_for_storage[..., 1] * ary
real_actions[..., 2] = (actions_for_storage[..., 2] + z_bias) * arz * gz
```

**注意**：`real_actions` 随后会通过 `action_data` 存入回放区。

## 📊 修复效果

### 修复前的问题流程

```
训练流程：
1. Actor 输出 z_raw = -0.5
2. 势场修正 → z_corrected = +0.3
3. 环境执行 z_corrected（安全）
4. 回放区存 z_corrected
5. Critic 学习 Q(s, z_corrected) = 高分
6. Actor 梯度：∂Q/∂z_raw 基于 z_corrected → "负 Z 很好"
7. Actor 持续输出负 Z
```

**结果**：Actor 依赖势场修正，无法独立飞行。

### 修复后的正确流程

```
训练流程：
1. Actor 输出 z_raw = -0.5
2. 势场修正 → z_corrected = +0.3
3. 环境执行 z_corrected（安全，训练不崩）
4. 回放区存 z_raw（关键修改！）
5. Critic 学习 Q(s, z_raw) = 低分（因为会掉下去）
6. Actor 梯度：∂Q/∂z_raw 基于 z_raw → "负 Z 很差"
7. Actor 学会输出正 Z
```

**结果**：Actor 学会直接输出安全的正 Z 动作。

## 🔬 与之前修复的对比

### 2025-11-29 的修复（已回退）

**目标**：让 Actor 梯度基于原始动作计算
**方法**：在 Actor 更新时，使用 `raw_action_mapped` 而不是 `corrected_action`
**问题**：
- 回放区存的是修正后的动作
- Critic 学习的是 `Q(s, a_corrected)`
- 但 Actor 优化的是 `Q(s, a_raw)`
- **训练-执行不匹配**

### 2025-11-30 的修复（当前）

**目标**：分离存储动作和执行动作
**方法**：
- 回放区存**原始动作**（无修正）
- 环境执行**修正动作**（有修正）
- Critic 学习 `Q(s, a_raw)`
- Actor 优化 `Q(s, a_raw)`
- **训练-执行一致**

**关键区别**：
- 之前：只改 Actor 更新逻辑，但回放区数据没变 → 不一致
- 现在：改回放区存储的数据 → 一致

## 🎯 预期效果

1. **Z 轴控制改善**：
   - Actor 输出的原始 Z 轴动作逐渐变正
   - 不再依赖势场修正
   - 智能体能主动维持高度

2. **训练稳定性提升**：
   - Critic 的 Q 值估计更准确（基于真实执行的策略）
   - Actor 的梯度更有意义（基于原始动作的价值）
   - 训练和执行完全一致

3. **轨迹质量提升**：
   - 智能体不会无缘无故下坠
   - Z 轴控制更稳定
   - 整体路径规划更合理

## ✅ 验证方法

### 1. 监控 Actor 输出

在训练日志中添加 Actor 原始输出的统计：
```python
# 在 batch_select_actions_vectorized 中
raw_z_mean = tf.reduce_mean(raw_actor_actions[:, :, 2])
tf.print("Raw Z output:", raw_z_mean)
```

**预期**：随着训练进行，`raw_z_mean` 从负值逐渐变为正值。

### 2. 对比修正前后动作

```python
# 在训练循环中
storage_z = actions_for_storage[0, 0, 2]  # 存储的 Z（原始）
execution_z = actions[0, 0, 2]  # 执行的 Z（修正后）
print(f"Storage Z: {storage_z:.3f}, Execution Z: {execution_z:.3f}")
```

**预期**：初期差异大，后期趋近（Actor 学会输出安全动作）。

### 3. 观察 Critic Loss 和 Actor Loss

**预期**：
- Critic Loss：初期可能上升（重新学习 `Q(s, a_raw)`），然后稳定下降
- Actor Loss：绝对值逐渐变大（Q 值提升），且更稳定

## 📝 总结

这次修复的核心是：

1. **用户诊断正确**：梯度回传错误，Actor 学到了依赖势场修正的策略
2. **之前修复不完整**：只改 Actor 更新，但回放区数据没变，导致不一致
3. **当前修复完整**：分离存储和执行动作，保证训练和执行完全一致
4. **关键思想**：势场修正是执行时的安全保障，Actor 应该学会本身就安全的动作

## 🔧 修改文件清单

1. ✅ `paper3d_train_optimized.py` (第 7229-7316 行) - `batch_select_actions_vectorized` 返回三个值
2. ✅ `paper3d_train_optimized.py` (第 10966-10977 行) - 训练循环接收三个值
3. ✅ `paper3d_train_optimized.py` (第 10995-11096 行) - Tensor 转 NumPy 处理三个值
4. ✅ `paper3d_train_optimized.py` (第 11357-11382 行) - 存储使用 `actions_for_storage`

修复时间：2025-11-30


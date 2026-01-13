# 最终方案：势场可导性与训练策略 - 2025-11-30

## 🎯 核心问题重述

用户提出了两个关键质疑：
1. **Tensor转NumPy会破坏XLA加速**
2. **势场优化也是学习的一部分，应该参与梯度回传**

## 📊 系统架构分析

### Actor 输出的 7 维动作

```
Actor 输出 = [x, y, z, k_att, lambda_1, k_rep, radius]
             └─前3维─┘  └────────后4维势场参数────────┘
```

**后 4 维势场参数**（可学习的）：
- `k_att`: 目标吸引力系数（base=5.0, delta=±2.0）
- `lambda_1`: 距离阈值 d0（base=6.5, delta=±2.0）
- `k_rep`: 地形斥力系数（base=60.0, delta=±2.0）
- `radius`: 检测半径（base=15.0, delta=±2.0）

### 势场修正的作用

```python
# _apply_potential_field_correction_tf (4439行)
def PF_correct(action_7d, obs, force_ratio):
    xyz = action_7d[:, :3]  # 前3维加速度
    pf_params = action_7d[:, 3:7]  # 后4维势场参数
    
    # 使用势场参数计算势场力
    goal_force = calc_goal_attraction(obs, k_att, lambda_1)
    terrain_force = calc_terrain_repulsion(obs, k_rep, radius)
    agent_force = calc_agent_repulsion(obs, k_rep, radius)
    obstacle_force = calc_obstacle_repulsion(obs, k_rep, radius)
    
    total_pf_force = goal_force + terrain_force + agent_force + obstacle_force
    
    # 混合原始动作和势场力
    xyz_corrected = (1 - force_ratio) * xyz + force_ratio * total_pf_force
    
    return xyz_corrected
```

**关键发现**：
- 势场修正过程是**完全可导的**（注释明确说明）
- 梯度可以从 `xyz_corrected` 回传到：
  - 前3维 `xyz`
  - 后4维 `pf_params`

## 🤔 三种可能的训练策略

### 策略 A：存储并训练修正后的动作（原始方案）

```
流程：
1. Actor输出 a_7d = [xyz, pf_params]
2. 势场修正 a_corrected = PF_correct(a_7d)
3. 环境执行 a_corrected
4. 回放区存储 a_corrected（只存前3维）
5. Critic学习 Q(s, a_corrected)
6. Actor优化 Q(s, a_corrected)
```

**问题**：
- 回放区只存前 3 维（修正后的 xyz）
- **丢失了后 4 维势场参数**
- Actor 无法学习如何调整势场参数

**适用场景**：
- 势场参数是固定的（不需要学习）
- 只想让 Actor 学习加速度

### 策略 B：存储原始动作，训练也用原始动作（当前修复）

```
流程：
1. Actor输出 a_7d = [xyz, pf_params]
2. 势场修正 a_corrected = PF_correct(a_7d)
3. 环境执行 a_corrected（安全）
4. 回放区存储 a_7d（完整7维）
5. Critic学习 Q(s, a_7d)
6. Actor优化 Q(s, a_7d)
```

**优点**：
- 保留完整的 7 维动作
- Critic 评估原始策略的价值
- Actor 学会输出本身就安全的动作

**问题**：
- **训练和执行不一致**：
  - 训练时 Critic 评估 `Q(s, a_7d)`
  - 执行时环境执行 `a_corrected = PF_correct(a_7d)`
- **梯度不经过势场修正层**：
  - Actor 无法利用势场修正的可导性
  - 无法学习最优的势场参数

### 策略 C：存储原始动作，训练时应用势场修正（推荐）

```
流程：
1. Actor输出 a_7d = [xyz, pf_params]
2. 势场修正 a_corrected = PF_correct(a_7d)
3. 环境执行 a_corrected（安全）
4. 回放区存储 a_7d（完整7维）
5. Critic学习 Q(s, a_corrected)，但a_corrected是从a_7d推导的
6. Actor优化时：
   new_a_7d = Actor(s)
   new_a_corrected = PF_correct(new_a_7d)  # 梯度可回传！
   Q = Critic(s, new_a_corrected)
   Actor_loss = -Q
```

**优点**：
- ✅ 训练和执行完全一致
- ✅ 梯度经过势场修正层回传
- ✅ Actor 可以同时学习：
  - 更好的加速度 xyz
  - 更好的势场参数 pf_params
- ✅ 保留完整的 7 维动作信息

**实现**：
- 回放区存储完整的 7 维原始动作
- Critic 训练时，从 7 维动作推导修正后的动作
- Actor 更新时，应用势场修正后再计算 Q 值

## 🔧 推荐的代码修改（策略 C）

### 1. 回放区存储完整 7 维动作（已完成）

当前的修改已经做到了这一点：
- `actions_for_storage` = 完整 7 维原始动作
- `actions_for_execution` = 修正后的动作（给环境）

### 2. Critic 训练时从原始动作推导修正动作

**位置**：`_multi_agent_update_step`，Critic 更新部分

**修改**：
```python
# 当前代码（简化）：
global_actions = tf.convert_to_tensor(act_n)  # 从回放区采样的原始动作
current_q1, current_q2 = Critic([global_state, global_actions], training=True)

# 修改后：
# 从原始动作推导修正后的动作
global_actions_raw = tf.convert_to_tensor(act_n)  # 7维原始动作
# 应用势场修正（与环境交互时一致）
global_actions_corrected = self._apply_pf_correction_for_training(
    global_actions_raw, obs_n, current_force_ratio
)
# Critic 评估修正后的动作
current_q1, current_q2 = Critic([global_state, global_actions_corrected], training=True)
```

### 3. Actor 更新时应用势场修正

**位置**：`_multi_agent_update_step`，Actor 更新部分（8510-8570行）

**修改**：
```python
# 当前代码（简化）：
new_action = Actor(obs, training=True)  # 7维输出
raw_action_mapped = map_to_env_scale(new_action)  # 映射但不修正
global_actions_actor = concat([..., raw_action_mapped, ...])
actor_q1 = Critic([global_state, global_actions_actor], training=False)

# 修改后：
new_action = Actor(obs, training=True)  # 7维输出
# 应用势场修正（梯度可回传！）
corrected_action = self._apply_pf_correction_for_actor_update(
    new_action, obs, current_force_ratio
)
# 映射到环境尺度
corrected_action_mapped = map_to_env_scale(corrected_action)
global_actions_actor = concat([..., corrected_action_mapped, ...])
actor_q1 = Critic([global_state, global_actions_actor], training=False)
```

**关键**：
- 势场修正在 `tf.GradientTape` 内部
- 梯度可以回传到 Actor 的 7 维输出
- Actor 学会调整加速度和势场参数

### 4. Tensor转NumPy的优化

**问题**：环境是 Python/NumPy 实现的，必须调用 `.numpy()`

**当前无法避免的转换**：
```python
# 动作选择（在TensorFlow图内）
actions_for_storage, actions_for_execution, pf_forces = \
    maddpg.batch_select_actions_vectorized(obs_tensor)  # GPU

# 必须转换（因为环境是NumPy）
actions_np = actions_for_execution.numpy()  # GPU→CPU，同步

# 环境执行（CPU）
next_obs, reward, done, info = env.step(actions_np)
```

**优化方向**：
1. **异步执行**：在等待环境返回时，GPU 可以继续训练
2. **批量处理**：一次处理多个环境，减少同步次数
3. **尽量延迟转换**：只在真正需要时才调用 `.numpy()`

**不影响训练的转换**：
- 训练更新（`_multi_agent_update_step`）完全在 TensorFlow 图内
- XLA 编译的部分不受影响
- 只有环境交互部分需要转换

## 📊 三种策略的对比

| 特性 | 策略A（修正后） | 策略B（原始） | 策略C（原始+修正）|
|------|--------------|-------------|----------------|
| 回放区存储 | 修正后（3维） | 原始（7维）✅ | 原始（7维）✅ |
| Critic评估 | Q(s,a_corrected) | Q(s,a_raw) | Q(s,a_corrected) ✅ |
| Actor优化 | Q(s,a_corrected) | Q(s,a_raw) | Q(s,a_corrected) ✅ |
| 训练-执行一致 | ✅ | ❌ | ✅ |
| 梯度经过PF | ❌（参数丢失） | ❌（无修正） | ✅ |
| 学习PF参数 | ❌ | ❌ | ✅ |
| 学习安全动作 | 可能依赖PF | ✅ | ✅（通过PF梯度）|

## ✅ 最终推荐

**采用策略 C**：
1. 回放区存储完整 7 维原始动作（已完成）
2. Critic 训练时，从原始动作推导修正动作
3. Actor 更新时，应用势场修正后计算 Q 值
4. 梯度通过势场修正层回传到 Actor

**关键优势**：
- 训练和执行完全一致
- Actor 可以学习最优的加速度和势场参数
- 梯度利用了势场修正的可导性
- 保留了势场作为"软约束"的优势

## 🔬 关于 XLA 和 Tensor转NumPy

**现状**：
- 环境交互必须转 NumPy（环境是 Python 实现的）
- 这是**无法避免的瓶颈**

**不影响训练的原因**：
- 训练更新（Critic/Actor 梯度计算）在 TensorFlow 图内
- XLA 编译的 `@tf.function` 不受影响
- 只有环境 `step()` 需要等待，但可以异步化

**优化建议**：
1. 使用多个环境并行，减少单次同步开销
2. 考虑将环境也用 TensorFlow 实现（长期目标）
3. 当前的批量处理（num_envs=3）已经在优化这个问题

## 📝 总结

用户的质疑非常有价值：
1. ✅ **势场参数是可学习的**，应该参与梯度回传
2. ✅ **Tensor转NumPy确实影响性能**，但环境交互无法避免
3. ✅ **当前修复不完整**，应该让梯度经过势场修正层

**下一步**：
- 实施策略 C 的修改
- 让 Critic 和 Actor 都使用修正后的动作
- 但保留完整的 7 维原始动作在回放区
- 梯度通过势场修正层回传

这样既解决了 Z 轴负向输出问题，又充分利用了势场参数的可学习性。


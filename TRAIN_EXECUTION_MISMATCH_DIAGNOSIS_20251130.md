# 训练-执行不匹配问题诊断 - 2025-11-30

## 🚨 核心问题

发现了一个**训练-执行不匹配（Train-Execution Mismatch）**的严重问题：

### 当前流程分析

#### 1. 环境交互阶段 (`batch_select_actions_vectorized`)

```python
# 第 7211-7230 行
raw_actor_actions = Actor(state)  # Actor 输出原始动作

# 第 7236-7265 行
if should_apply_pf:
    final_network_actions, pf_forces = _apply_pf_correction(raw_actor_actions)  # 势场修正
else:
    final_network_actions = raw_actor_actions

# 第 7267-7286 行
if add_noise:
    final_actions_with_noise = final_network_actions + OU_noise  # 添加噪声

# 第 7288-7289 行
actions = mix(random_actions, final_actions_with_noise)  # 混合随机和网络动作

# 返回 actions → 这是势场修正后的动作
```

**结论**：环境执行的是**势场修正后的动作** `a_corrected`

#### 2. 数据存储阶段

```python
# 训练循环中（第 10954 行左右）
actions_tensor, pf_forces_tensor = maddpg.batch_select_actions_vectorized(...)
actions = actions_tensor.numpy()  # 转换为 NumPy

# 环境执行
next_obs_n, reward_n, done_n, info_n = env.step(actions)

# 存入回放区
replay_buffer.add(obs_n, actions, ...)  # 存储的是 actions（势场修正后）
```

**结论**：回放区存储的是**势场修正后的动作** `a_corrected`

#### 3. Critic 学习阶段 (`_multi_agent_update_step`)

```python
# 从回放区采样
obs_n, next_obs_n, act_n, rew_n, done_n, ... = replay_buffer.sample(batch_size)

# Critic 更新（简化版）
current_q1 = Critic(state, act_n)  # act_n 是势场修正后的动作
target_q = reward + gamma * Critic_target(next_state, next_action_target)
critic_loss = huber_loss(current_q1 - target_q) + Q_regularization
```

**结论**：Critic 学习的是 `Q(s, a_corrected)`，这是**正确的**，因为环境确实执行了 `a_corrected`

#### 4. Actor 更新阶段（刚刚修复的部分，第 8507-8545 行）

```python
# 第 8502-8505 行
new_action = actor(obs)  # 原始 Actor 输出 a_raw

# 第 8510-8528 行
# 将 a_raw 映射到环境尺度（但不做势场修正）
raw_na_x = new_action[:, 0:1] * arx
raw_na_y = new_action[:, 1:2] * ary
raw_na_z = (new_action[:, 2:3] + z_bias) * arz * gz
raw_action_mapped = concat([raw_na_x, raw_na_y, raw_na_z, ...])

# 第 8530-8540 行
# 构建全局动作（使用原始映射动作）
global_actions_actor = concat([..., raw_action_mapped, ...])

# 第 8541-8551 行
# 计算 Actor Loss
actor_q1 = Critic(global_state, global_actions_actor)  # 使用 raw_action_mapped
actor_loss = -mean(actor_q1)  # 最大化 Q 值
```

**结论**：Actor 优化的是 `Q(s, a_raw)`，而不是 `Q(s, a_corrected)`

### ⚠️ 问题本质

- **Critic 学习的是** `Q(s, a_corrected)`（正确）
- **Actor 优化的是** `Q(s, a_raw)`（错误）
- **两者的动作 `a` 不一致！**

这导致：
1. **梯度不匹配**：Actor 的梯度是基于一个 Critic 没有充分训练的区域（`a_raw`）
2. **策略偏差**：Actor 学习的策略与实际执行的策略（经过势场修正）不一致
3. **性能次优**：Actor 可能学会输出一些"需要势场修正"的动作，但 Critic 给这些动作的 Q 值评估不准确

## 🤔 应该怎么修复？

### 方案对比

#### 方案A：Actor 优化 `Q(s, a_corrected)`（推荐）

**理由**：
- 环境确实执行的是 `a_corrected`
- Critic 学习的也是 `Q(s, a_corrected)`
- 保持训练和执行的一致性

**实现**：
```python
# Actor 更新时
new_action = actor(obs)  # 原始输出
# 应用势场修正（与环境交互时一致）
corrected_action = apply_pf_correction(new_action, obs, force_ratio)
# 构建全局动作
global_actions_actor = concat([..., corrected_action, ...])
# 计算 Q 值
actor_q1 = Critic(global_state, global_actions_actor)
actor_loss = -mean(actor_q1)
```

**优点**：
- 训练和执行完全一致
- Critic 的 Q 值估计准确
- 策略学习稳定

**缺点**：
- Actor 可能学会"依赖势场修正"
- 但这不是问题，因为：
  1. 势场修正是环境的一部分（类似于物理约束）
  2. Actor 的目标是学习在有势场修正的情况下的最优策略
  3. 如果不希望依赖势场，应该逐渐减小 `force_ratio`，而不是改变训练逻辑

#### 方案B：Actor 优化 `Q(s, a_raw)` + 环境执行 `a_raw`（需要大改）

**理由**：
- 让 Actor 学习直接输出安全动作
- 势场仅作为额外的安全保障

**实现**：
- 修改 `batch_select_actions_vectorized`，不做势场修正
- 修改环境交互，执行原始动作
- 修改回放区，存储原始动作
- Critic 学习 `Q(s, a_raw)`

**优点**：
- Actor 学会输出本质上安全的动作
- 不依赖势场修正

**缺点**：
- 需要大量修改，风险高
- 训练初期可能不稳定（智能体容易坠毁）
- 与当前代码框架不一致

#### 方案C：混合方案（不推荐）

- Critic 学习 `Q(s, a_corrected)`
- Actor 优化 `Q(s, a_raw)`
- 但在 Actor 更新时，确保 Critic 也能评估 `Q(s, a_raw)`

**问题**：
- Critic 需要同时学习两个不同的 Q 函数
- 训练不稳定
- 逻辑复杂

## 📊 当前训练表现的解释

基于当前的训练日志（Critic Loss 上升，奖励提升）：

1. **奖励提升**：说明策略在改善（至少在短期内）
2. **Critic Loss 上升**：
   - 部分原因：Q² 正则项随 Q 值增大而增大（正常）
   - 部分原因：训练-执行不匹配导致 Critic 的 Q 值估计不准确

3. **潜在风险**：
   - Actor 可能学到一些次优策略
   - Critic 的 Q 值可能在 `a_raw` 区域估计不准
   - 长期训练可能不稳定

## ✅ 推荐修复方案

**采用方案A：Actor 优化 `Q(s, a_corrected)`**

### 理由：

1. **最小修改原则**：只需修改 Actor 更新部分，不影响环境交互和数据存储
2. **逻辑一致性**：训练和执行完全一致
3. **稳定性**：Critic 的 Q 值估计准确，训练稳定
4. **符合当前框架**：势场修正被视为环境的一部分

### 实现细节：

在 `_multi_agent_update_step` 中，Actor 更新时：
1. Actor 输出原始动作 `new_action`
2. 应用势场修正 `corrected_action = apply_pf_correction(new_action, obs, force_ratio)`
3. 构建全局动作 `global_actions_actor = concat([..., corrected_action, ...])`
4. 计算 Q 值 `actor_q1 = Critic(global_state, global_actions_actor)`
5. 计算 Actor Loss `actor_loss = -mean(actor_q1)`

### 关于"Actor 依赖势场修正"的问题：

这**不是问题**，而是**特性**：
- 势场修正是环境约束的一部分（类似于物理限制）
- Actor 的目标是在这些约束下学习最优策略
- 如果想让 Actor 学习更"自主"的策略，应该：
  - 逐渐降低 `force_ratio`（课程学习）
  - 而不是改变训练逻辑

### 与之前修复的对比：

**之前的修复（2025-11-29）**：
- 目标：让 Actor 学会输出正向 Z 轴加速度，不依赖势场
- 方法：Actor 梯度基于 `a_raw` 计算
- 问题：与环境执行 `a_corrected` 不一致

**新的理解**：
- 势场修正不是"作弊"，而是环境的一部分
- Actor 应该学习在有势场的情况下的最优策略
- 训练和执行必须保持一致

## 📝 总结

当前代码存在**训练-执行不匹配**问题：
- 环境执行势场修正后的动作
- Critic 学习势场修正后的 Q 值
- 但 Actor 梯度基于未修正的动作计算

**推荐修复**：
- 让 Actor 梯度也基于势场修正后的动作计算
- 保持训练和执行的完全一致
- 这符合强化学习的基本原则

**下一步**：
- 实施方案A的代码修改
- 重新训练并观察效果
- 如果需要减少对势场的依赖，采用课程学习逐渐降低 `force_ratio`


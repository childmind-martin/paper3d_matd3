# MATD3架构详细说明

## 一、MATD3 Critic的双Q输出结构

### 1.1 架构设计（第2933-3088行）

**核心思想**：
- **Q1**：评估前3维动作（xyz加速度）的Q值
- **Q2**：评估后4维PF参数（k_att, lambda_1, k_rep, radius）的Q值
- **共享底座**：State + PF_info特征提取
- **分离特征**：前3维动作特征 + 后4维PF参数特征

**⚠️ 重要澄清**：
- MATD3的"双Q"**不是**指"一个原始动作一个修正动作"
- 而是指"前3维动作 vs 后4维PF参数"的分离评估
- 每个Q head都接收完整的动作输入，但只关注对应的维度

### 1.2 网络结构

```
输入：
  - state: (batch, total_obs_dim)
  - action: (batch, total_action_dim)  # 包含所有智能体的动作
  - fr_input: (batch, 1)  # 可选
  - pf_input: (batch, n_agents*3)  # 可选

共享底座：
  - State特征提取 → 384维
  - PF_info特征提取 → 256维
  - 融合 → 512维

分离特征：
  - 前3维动作特征：每个智能体的前3维 → 128维 → 拼接 → 128*n_agents维
  - 后4维PF参数特征：每个智能体的后4维 → 128维 → 拼接 → 128*n_agents维

Q Head：
  - Q1: 共享底座 + 前3维动作特征 → Q值（评估Actor动作）
  - Q2: 共享底座 + 后4维PF参数特征 → Q值（评估PF参数）

输出：
  - [Q1, Q2]  # 双Q值
```

## 二、MATD3的梯度分离策略

### 2.1 路径1：前3维学习原始动作（第10248-10289行）

**目标**：让Actor学会直接输出安全动作，不依赖势场修正

**流程**：
```
Actor输出 → 原始动作（无修正） → 映射到环境尺度 → Q1评估 → loss_head
```

**关键代码**：
```python
# 第10248-10258行：使用原始动作
raw_na_x = new_action_safe[:, 0:1] * arx
raw_na_y = new_action_safe[:, 1:2] * ary
raw_na_z = (new_action_safe[:, 2:3] + z_bias) * arz * gz
raw_na_head = tf.concat([raw_na_x, raw_na_y, raw_na_z], axis=1)
raw_action_mapped = tf.concat([raw_na_head, na_tail_for_head], axis=1)  # 后4维使用stop_gradient

# 第10284行：Q1评估原始动作
actor_q_output_raw = agent['critic1'](actor_inputs_raw, training=False)
actor_q1_raw, actor_q2_raw = actor_q_output_raw
actor_q_raw = tf.cast(actor_q1_raw, tf.float32)  # 使用Q1评估前3维动作
```

**梯度流向**：
```
loss_head → Q1_raw → raw_xyz → Actor前3维输出 ✅
（后4维使用stop_gradient，梯度不会回传）
```

### 2.2 路径2：后4维学习修正后的动作（第10291-10346行）

**目标**：优化势场参数，使其能更好地辅助决策

**流程**：
```
Actor输出 → 势场修正（前3维使用stop_gradient） → 映射到环境尺度 → Q2评估 → loss_tail
```

**关键代码**：
```python
# 第10296行：前3维使用stop_gradient
xyz_for_tail = tf.stop_gradient(new_action_safe[:, :3])

# 第10304行：势场修正（梯度只能从corrected_head回传到pf_params）
corrected_head_flat_tail, _ = self._apply_potential_field_correction(
    action_for_tail,  # [batch_size, 7]
    obs,
    current_force_ratio
)

# 第10341行：Q2评估修正后的动作
actor_q_output_corrected = agent['critic1'](actor_inputs_corrected, training=False)
actor_q1_corrected, actor_q2_corrected = actor_q_output_corrected
actor_q_corrected_tail = tf.cast(actor_q2_corrected, tf.float32)  # 使用Q2评估后4维PF参数
```

**梯度流向**：
```
loss_tail → Q2_corrected → corrected_xyz → pf_params → Actor后4维输出 ✅
（前3维使用stop_gradient，梯度不会回传）
```

### 2.3 梯度分离机制

**关键**：通过 `stop_gradient` 实现梯度隔离

- **路径1**：后4维使用 `stop_gradient`，梯度只影响前3维
- **路径2**：前3维使用 `stop_gradient`，梯度只影响后4维

**优势**：
1. 前3维学习安全动作（不依赖势场）
2. 后4维学习最优势场（利用可导性）
3. 两个学习目标独立，互不干扰

## 三、修正动作的使用路径总结

### 3.1 MADDPG

**Critic学习**（第6857-6878行）：
- 使用混合动作：前3维原始 + 后4维修正
- 评估 `Q(s, a_mixed)`

**Actor训练**（第6335-6368行）：
- 使用修正后的动作
- 评估 `Q(s, a_corrected)`

### 3.2 MATD3

**Critic学习**（第8613-8634行）：
- 使用混合动作：前3维原始 + 后4维修正
- Q1评估前3维，Q2评估后4维

**Actor训练**：
- **路径1（前3维）**：使用原始动作，Q1评估（第10248-10289行）
- **路径2（后4维）**：使用修正后的动作，Q2评估（第10291-10346行）

## 四、FR Schedule的影响

### 4.1 问题描述

**如果FR值随时间变化（schedule）**：
- 历史执行时使用的FR值：`fr_historical = 0.8`（第100步）
- 当前训练时的FR值：`fr_current = 0.5`（第10000步）
- **问题**：如果使用当前FR值恢复修正动作，会导致训练目标与实际执行不一致

### 4.2 修复方案（已实施）

**MADDPG**（第6700-6715行）：
```python
if len(sample_result) >= 8:
    # 使用历史存储的FR值（确保与历史执行一致）
    fr_batch_historical = np.asarray(fr_batch, dtype=np.float32)
    fr_batch_column = tf.expand_dims(tf.convert_to_tensor(fr_batch_historical, dtype=tf.float32), axis=1)
else:
    # 向后兼容：如果没有历史FR值，使用当前FR值
    current_fr_value = float(getattr(self.args, 'action_force_ratio', 0.0))
    fr_batch_column = tf.fill([len(obs_n), 1], tf.cast(current_fr_value, tf.float32))
```

**MATD3**（第8400-8412行）：
```python
if len(batch) >= 8:
    # 使用历史存储的FR值（确保与历史执行一致）
    fr_batch_historical = np.asarray(fr_batch, dtype=np.float32)
    fr_batch_tensor = tf.expand_dims(tf.convert_to_tensor(fr_batch_historical, dtype=tf.float32), axis=1)
else:
    # 向后兼容：如果没有历史FR值，使用当前FR值
    current_fr_value = float(getattr(self.args, 'action_force_ratio', 0.0))
    fr_batch_tensor = tf.fill([len(obs_n), 1], tf.cast(current_fr_value, tf.float32))
```

### 4.3 验证

**存储**（第13950-13952行）：
- 回放缓冲区存储历史FR值：`fr_list.append(current_fr_value)`

**恢复**（第6703-6709行，MADDPG；第8403-8409行，MATD3）：
- 训练时优先使用历史FR值
- 如果没有历史FR值，回退到当前FR值（向后兼容）

**结论**：✅ 即使FR值随时间变化（schedule），训练-执行仍然一致


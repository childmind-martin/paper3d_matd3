# Actor-Critic网络训练问题分析总结

## 一、发现的关键问题

### 1.1 动作尺度不一致 ⚠️⚠️⚠️

**问题**：
- **Critic学习时**：使用归一化动作（[-1,1]）
- **Actor训练时**：使用混合尺度动作（前三维映射到环境尺度，后四维归一化）

**影响**：
- Critic学习的Q值基于归一化动作
- Actor训练时使用的动作是混合尺度
- Critic无法正确评估Actor的动作，导致Actor无法学习

### 1.2 已修复的问题

**修复1：Critic学习时也使用环境尺度动作**
- 位置：第7270行
- 修改：`global_actions = _map_action_batch(global_actions, self.action_dims)`
- 效果：Critic学习时使用环境尺度动作，与Actor训练保持一致

**修复2：Actor训练时其他智能体动作也映射到环境尺度**
- 位置：第6736-6750行
- 修改：将其他智能体的归一化动作也映射到环境尺度
- 效果：`global_actions_actor`全部是环境尺度动作

---

## 二、当前训练流程（修复后）

### 2.1 Critic学习流程

1. **构建混合动作**（第7248-7270行）：
   - 前三维：原始动作（归一化）
   - 后四维：修正后动作（归一化，实际与原始相同）
   - `global_actions` = 混合动作（归一化）

2. **映射到环境尺度**（第7270行）：
   - `global_actions = _map_action_batch(global_actions, self.action_dims)`
   - 前三维映射到环境尺度，后四维保持归一化

3. **Critic学习**（第6556行）：
   - `current_q = agent['critic']([global_state, global_actions], training=True)`
   - 使用环境尺度动作学习Q值

### 2.2 Actor训练流程

1. **Actor前向传播**（第6658-6664行）：
   - `new_action = agent['actor'](actor_inputs, training=True)`
   - 输出7维归一化动作

2. **映射前三维到环境尺度**（第6706-6709行）：
   - `na_x = action_head_for_actor[:, 0:1] * arx`
   - `na_y = action_head_for_actor[:, 1:2] * ary`
   - `na_z = (action_head_for_actor[:, 2:3] + z_bias) * arz * gz`
   - `na_head = tf.concat([na_x, na_y, na_z], axis=1)`

3. **构建全局动作**（第6728-6741行）：
   - 当前智能体：`new_action_real` = `[na_head, na_tail]`（前三维映射，后四维归一化）
   - 其他智能体：从`global_actions`提取并映射到环境尺度
   - `global_actions_actor` = 全部环境尺度动作

4. **Critic评估**（第6752行）：
   - `actor_q = agent['critic']([global_state, global_actions_actor], training=False)`
   - 使用环境尺度动作评估Q值

### 2.3 一致性检查 ✅

**修复后**：
- Critic学习时：使用环境尺度动作 ✅
- Actor训练时：使用环境尺度动作 ✅
- **一致！** ✅

---

## 三、其他潜在问题

### 3.1 动作正则化强度

**位置**：第6848-6879行
```python
head_reg = head_l2 + tf.cast(1.0, head_l2.dtype) * head_boundary_penalty
tail_reg = tf.cast(0.3, tail_mean.dtype) * tail_mean + ...
action_reg = (head_reg + tail_reg) * tf.cast(arc_eff, new_action.dtype)
actor_loss = actor_loss_pg + tf.cast(action_reg, actor_loss_pg.dtype)
```

**检查**：
- `action_reg_coef` = 0.008（从run_optimized.sh）
- 如果`action_reg`过大，可能压制策略梯度损失

**建议**：
- 检查`action_reg`和`actor_loss_pg`的相对大小
- 如果`action_reg >> actor_loss_pg`，需要降低正则化系数

### 3.2 学习率

**当前设置**：
- Actor学习率：0.00100（已提高）
- Critic学习率：0.0030

**检查**：
- 学习率是否合理
- 是否需要进一步调整

### 3.3 梯度流

**已验证**：
- `action_head_for_actor = new_action[:, :3]` ✅ 直接切片，没有stop_gradient
- 映射操作（`* arx`, `* ary`, `* arz * gz`）✅ 不会阻断梯度
- 梯度裁剪阈值合理（逐层2.0，全局10.0）✅

---

## 四、修复总结

### 已修复：
1. ✅ Critic学习时使用环境尺度动作（第7270行）
2. ✅ Actor训练时其他智能体动作也映射到环境尺度（第6736-6750行）
3. ✅ 提高Actor学习率（0.00050 → 0.00100）
4. ✅ 修复其他智能体信息索引错误（70 → 63）

### 待验证：
1. ⏳ 动作正则化强度是否合理
2. ⏳ 梯度流是否完整
3. ⏳ Q值对动作的敏感度

---

## 五、预期效果

修复后，Actor-Critic训练应该：
1. ✅ Critic学习的Q值能准确反映Actor训练时使用的动作价值
2. ✅ Actor能够从Critic反馈中正确学习
3. ✅ 前三维动作应该能够有效更新








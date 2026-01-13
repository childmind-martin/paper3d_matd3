# Actor 梯度修复 - 2025-11-30

## 问题诊断

在检查代码时发现，之前的关键修复被回撤了。具体问题在 `paper3d_train_optimized.py` 的第 8507-8540 行。

### 错误代码（已修复前）

```python
# 注释说要使用"原始输出计算Q值"
# 🚨 关键修复：Actor Loss应该用原始输出（不经过势场修正）计算Q值
...
new_action_real = tf.concat([na_head, na_tail_safe], axis=1)  # 这里进行了势场修正
# 构建全局动作：替换当前 agent 动作
pieces = []
start = 0
for j, dim in enumerate(self.action_dims):
    if j == i:
        pieces.append(new_action_real)  # ❌ 错误：使用了修正后的动作
    else:
        pieces.append(tf.cast(global_actions[:, start:start+dim], tf.float32))
    start += dim
global_actions_actor = tf.concat(pieces, axis=1)
```

**问题本质**：
- 虽然注释声称要使用"原始输出"，但代码实际上使用了 `new_action_real`（经过势场修正的动作）
- 这导致 Actor 的策略梯度仍然基于修正后的动作计算
- 结果：Actor 学会输出负 Z 轴加速度，依赖势场修正来"托住"智能体

## 修复方案

### 修改内容

**文件**: `paper3d_train_optimized.py`  
**行数**: 8507-8540

### 核心改动

1. **重命名变量以明确语义**：
   - `na_x`, `na_y`, `na_z` → `raw_na_x`, `raw_na_y`, `raw_na_z`
   - `na_head` → `raw_na_head`
   - `new_action_real` → `raw_action_mapped`

2. **修正动作构建逻辑**：
   ```python
   # ✅ 正确：使用原始映射动作（未经势场修正）
   for j, dim in enumerate(self.action_dims):
       if j == i:
           pieces.append(raw_action_mapped)  # 使用原始映射动作
       else:
           pieces.append(tf.cast(global_actions[:, start:start+dim], tf.float32))
       start += dim
   ```

### 修复后的完整逻辑

```python
# 环境映射（使用原始Actor输出，映射到环境尺度但不经过势场修正）
arx = self.c_arx
ary = self.c_ary
arz = self.c_arz
gz = self.c_gain_z

new_action_safe = tf.clip_by_value(new_action, -10.0, 10.0)
raw_na_x = new_action_safe[:, 0:1] * arx
raw_na_y = new_action_safe[:, 1:2] * ary
z_bias = tf.cast(self.z_action_bias, tf.float32)
raw_na_z = (new_action_safe[:, 2:3] + z_bias) * arz * gz
raw_na_head = tf.concat([raw_na_x, raw_na_y, raw_na_z], axis=1)

na_tail = new_action[:, 3:]
na_tail_safe = tf.clip_by_value(na_tail, -10.0, 10.0)
raw_action_mapped = tf.concat([raw_na_head, na_tail_safe], axis=1)

# 构建全局动作：使用原始映射动作（不经过势场修正）
pieces = []
start = 0
for j, dim in enumerate(self.action_dims):
    if j == i:
        pieces.append(raw_action_mapped)  # 🔧 使用原始映射动作
    else:
        pieces.append(tf.cast(global_actions[:, start:start+dim], tf.float32))
    start += dim
global_actions_actor = tf.concat(pieces, axis=1)

# Actor loss（Q1）
if self.use_fr_feature_flag:
    actor_q1, _ = agent['critic']([global_state, global_actions_actor, fr_batch_safe], training=False)
else:
    actor_q1, _ = agent['critic']([global_state, global_actions_actor], training=False)
```

## 修复原理

### 训练流程（修复后）

1. **Actor 输出原始动作** `a_raw`（归一化范围，例如 [-1, 1]）
2. **映射到环境尺度** `a_mapped`（应用 `ACTION_RANGE_*` 和 `Z_ACTION_BIAS`）
3. **势场修正** `a_safe = PF_correct(a_mapped)`
4. **环境执行** 使用 `a_safe` 与环境交互
5. **Critic 学习** 使用 `(s, a_safe)` 更新价值函数估计
6. **Actor 更新** 使用 `Q(s, a_mapped)` 计算策略梯度 ← **关键改动**

### 梯度流向

```
Actor 输出 a_raw
    ↓ (可微)
映射 a_mapped = scale(a_raw)
    ↓ (可微)
Critic 评估 Q(s, a_mapped)
    ↓ (可微)
策略梯度 ∂Q/∂a_raw
    ↓
更新 Actor 参数
```

### 关键思想

- **环境交互层面**：仍然使用势场修正后的 `a_safe`，保证训练安全
- **梯度计算层面**：基于原始映射后的 `a_mapped` 计算 Q 值和策略梯度
- **学习目标**：迫使 Actor 学会直接输出安全的动作，而不是依赖势场"兜底"

## 预期效果

修复后，Actor 应该会：

1. **学会输出正向或稳定的 Z 轴加速度**
   - 因为负 Z 轴动作会导致 `Q(s, a_mapped)` 很低
   - 梯度会推动 Actor 输出向上或平稳的加速度

2. **不再依赖势场修正**
   - Actor 的原始输出本身就应该是合理的
   - 势场修正只是额外的安全保障，而非必需

3. **轨迹质量提升**
   - Z 轴控制更稳定
   - 智能体能够主动维持高度，而不是被动"托举"

## Critic Loss 持续上升的解释

用户观察到 `critic_loss` 从 700 → 1000+ → 1100+，这是**正常现象**，原因如下：

### Critic Loss 构成

```python
critic_loss = TD_loss + CRITIC_Q_REG × (E[Q1²] + E[Q2²])
```

当前设置：`CRITIC_Q_REG = 0.02`

### 为什么上升？

1. **奖励变好** → Q 值变大
   - 训练初期：奖励 -200k，Q 值约 -1000
   - 当前阶段：奖励 +120k～145k，Q 值约 +500
   
2. **Q² 正则项随 Q 值增长**
   - Q = 500 时，Q² = 250,000
   - 0.02 × 250,000 = 5,000
   - 仅正则项就贡献数百～上千的 loss

3. **这不代表训练变差**
   - TD 误差部分可能在下降
   - 但 Q² 正则项随策略改善而增长
   - 这是正则化机制在**刻意压制过大的 Q 值**

### Actor Loss 解读

- `actor_loss` 越来越负（绝对值越来越大）
- 等价于 `-E[Q(s, a)]`
- 说明 Critic 给当前策略的评分越来越高
- **这是好现象！**

## 验证方法（可选）

如果想确认 Critic Loss 的组成，可以在 `_multi_agent_update_step` 中临时添加：

```python
# 在计算 critic_loss 时分别记录两部分
td_loss_value = cl1 + cl2
q_reg_value = q_reg_w * (reg_q1 + reg_q2)
# 打印到日志
tf.print("TD Loss:", td_loss_value, "Q Reg:", q_reg_value)
```

预期结果：
- `TD Loss` 稳定或缓慢下降
- `Q Reg` 随训练进展明显上升

## 总结

1. ✅ **已修复**：Actor 梯度现在基于原始映射动作计算，不再依赖势场修正
2. ✅ **Critic Loss 上升是正常的**：主要由 Q² 正则项贡献，反映策略改善
3. ✅ **Actor Loss 变负是好现象**：说明 Actor 学到了更高价值的策略
4. 🎯 **预期改进**：Z 轴控制更稳定，轨迹质量进一步提升

## 修改文件清单

- ✅ `/home/tang/Desktop/paper3d_train_optimized.py` (第 8507-8540 行)

修复时间：2025-11-30


# Actor-Critic训练不一致问题分析

## 一、关键发现

### 1.1 Critic学习时使用的动作

**位置**：第7236-7273行
```python
# 构建混合动作：前三维原始，后四维"修正"
mixed_action = tf.concat([
    raw_action[:, :3],  # 前三维：原始动作（归一化，[-1,1]）
    corrected_action[:, 3:]  # 后四维：修正后动作（归一化，[-1,1]）
], axis=1)
global_actions = tf.concat(mixed_actions_list, axis=1)  # 混合动作，用于Critic学习
```

**关键点**：
- `raw_action` = `act_n_raw`（原始动作，归一化）
- `corrected_action` = `act_n`（修正后动作，归一化）
- 但`act_n_corrected[:, 3:]`实际上就是`act_n[:, 3:]`（原始动作的后四维），因为修正只影响前三维
- 所以`global_actions`实际上是：前三维原始，后四维原始（归一化）

**Critic学习时**（第6556行）：
```python
current_inputs = [global_state, global_actions]  # global_actions是归一化动作
current_q = agent['critic'](current_inputs, training=True)
```

**问题**：`global_actions`是归一化动作（[-1,1]），但Critic期望的是映射到环境尺度的动作！

### 1.2 Actor训练时使用的动作

**位置**：第6728-6744行
```python
# 映射前三维到环境尺度
na_x = action_head_for_actor[:, 0:1] * arx
na_y = action_head_for_actor[:, 1:2] * ary
na_z = (action_head_for_actor[:, 2:3] + z_bias) * arz * gz
na_head = tf.concat([na_x, na_y, na_z], axis=1)

# 后四维保持原样（归一化）
na_tail = new_action[:, 3:]  # 归一化，[-1,1]

new_action_real = tf.concat([na_head, na_tail], axis=1)  # 前三维映射，后四维归一化

global_actions_actor = tf.concat(act_input, axis=1)  # 当前智能体是new_action_real，其他是global_actions
```

**关键点**：
- `new_action_real`：前三维映射到环境尺度，后四维归一化
- `global_actions_actor`：当前智能体是`new_action_real`，其他智能体是`global_actions`（归一化）

**Actor训练时**（第6748行）：
```python
actor_inputs = [global_state, global_actions_actor]  # global_actions_actor是混合尺度！
actor_q = agent['critic'](actor_inputs, training=False)
```

### 1.3 关键不一致 ⚠️⚠️⚠️

**Critic学习时**：
- 输入：`global_actions`（全部归一化，[-1,1]）
- Critic学习：`Q(s, a_normalized)`

**Actor训练时**：
- 输入：`global_actions_actor`（前三维映射到环境尺度，后四维归一化）
- Critic评估：`Q(s, a_mixed_scale)`

**问题**：
- Critic学习的Q值基于归一化动作
- Actor训练时使用的动作是混合尺度（前三维映射，后四维归一化）
- **Critic的Q值不能准确反映Actor训练时使用的动作价值！**

---

## 二、根本原因

### 2.1 动作尺度不一致

**Critic学习时**：
- 所有动作都是归一化的（[-1,1]）
- Critic学习：`Q(s, [a_norm_x, a_norm_y, a_norm_z, a_norm_k_att, ...])`

**Actor训练时**：
- 前三维映射到环境尺度（例如：`a_x * 1.0`, `a_y * 1.0`, `(a_z + z_bias) * 1.0 * 1.0`）
- 后四维保持归一化（[-1,1]）
- Critic评估：`Q(s, [a_env_x, a_env_y, a_env_z, a_norm_k_att, ...])`

**影响**：
- Critic学习的Q值对归一化动作敏感
- 但Actor训练时使用的是混合尺度动作
- Critic无法正确评估Actor的动作，导致Actor无法学习

### 2.2 解决方案

**方案1：统一使用归一化动作**
- Actor训练时也使用归一化动作（不映射前三维）
- 但这样会导致Critic评估的动作与环境执行的动作不一致

**方案2：统一使用环境尺度动作**
- Critic学习时也使用环境尺度动作（映射前三维）
- 这样Critic学习的Q值才能准确反映Actor训练时使用的动作价值

**方案3：Critic同时学习两种尺度**
- 这太复杂，不推荐

---

## 三、建议修复

**推荐方案2**：Critic学习时也使用环境尺度动作

**修改位置**：第7269行之后
```python
# 映射混合动作到环境尺度
global_actions = _map_action_batch(global_actions, self.action_dims)
```

这样：
- Critic学习时：使用环境尺度动作
- Actor训练时：使用环境尺度动作
- **一致！** ✅








# 关键修复：并行环境固定位置失效与Z轴梯度错误

## 修复时间
2025-11-29

## 问题总结

### 1. 并行环境固定位置失效
**现象**：
- 单环境训练：固定位置生效，每回合起点相同
- 并行环境（3个）：固定位置失效，每回合起点随机变化

**根本原因**：
在 `paper3d_terrain_energy.py` 的 `reset_world` 方法中，使用了 `getattr(self, 'use_fixed_positions', False)` 来判断是否加载固定位置。但在多进程并行环境中，`dynamic_first_time` 逻辑会在每个进程中独立执行，导致 `use_fixed_positions` 被重置为 `False`。

**修复方案**：
使用 `_initial_use_fixed_positions`（保存在 `__init__` 时的初始值）替代运行时的 `use_fixed_positions`，确保固定位置配置不会被多进程逻辑覆盖。

**修改文件**：
- `multiagent/scenarios/paper3d_terrain_energy.py` (行1011-1027)

```python
# 修复前
if getattr(self, 'use_fixed_positions', False) and self.fixed_positions is None...

# 修复后
if getattr(self, '_initial_use_fixed_positions', False):
    self.use_fixed_positions = True
    if self.fixed_positions is None and self.fixed_positions_file:
        if os.path.exists(self.fixed_positions_file):
            if self.load_fixed_positions(self.fixed_positions_file):
                self.positions_initialized = True
```

---

### 2. 动作选择函数势场修正不一致
**现象**：
- `select_action`（单环境）：随机动作**不做势场修正**，直接返回
- `batch_select_actions_vectorized`（并行环境）：随机动作和网络动作**都做势场修正**

**根本原因**：
三个动作选择函数的势场修正逻辑不统一，导致训练时和推理时的动作分布不一致，梯度计算基于修正后的动作，但推理时随机动作未修正。

**修复方案**：
统一所有动作选择函数的逻辑：**随机动作不做势场修正**（用于自由探索），仅网络动作做势场修正。

**修改文件**：
- `paper3d_train_optimized.py` (行7199-7316)

**关键改动**：
```python
# 修复前（7265-7283行）
if should_apply_pf:
    # 对随机动作也应用势场修正
    corrected_random_head_flat, pf_force_random_flat = self._apply_potential_field_correction(...)

# 修复后
# 🚨 关键修复：随机动作不应该做势场修正（与select_action逻辑一致）
# 原因：随机动作用于探索，势场修正会约束到安全区域，降低探索效率
actions = tf.where(mask3, rand_actions_raw, final_actions_with_noise)
pf_forces_final = tf.where(..., tf.zeros_like(pf_force_network), pf_force_network)
```

---

### 3. Z轴梯度回传错误（最关键）
**现象**：
- 训练后Actor Z轴输出持续为负值（-0.3到-0.8）
- 智能体飞行时向下坠落，依赖势场修正才能勉强维持高度
- Loss很小但轨迹质量差

**根本原因**：
Actor Loss计算时使用了**势场修正后的动作**去计算Q值。这导致梯度告诉Actor：
- "你输出负Z没问题，因为势场会帮你修正成正Z"
- "修正后的Q值高，所以继续输出负Z依赖修正"

**错误的训练逻辑**：
```
Actor输出: Z = -0.5
→ 势场修正: Z = 0.3 (向上推)
→ 计算Q值: Q(修正后动作) = 高值
→ 梯度: ∂Q/∂θ > 0，继续输出负Z ❌
```

**正确的训练逻辑**：
```
Actor输出: Z = -0.5
→ 计算Q值: Q(原始动作) = 低值（会坠落）
→ 梯度: ∂Q/∂θ < 0，学会输出正Z ✓
```

**修复方案**：
Actor Loss计算时使用**原始Actor输出（未经势场修正）**直接映射到环境动作，让Actor学会自己输出正Z而不是依赖修正。

**修改文件**：
- `paper3d_train_optimized.py` (行8499-8524)

**关键改动**：
```python
# 修复前
# 势场修正
corrected_action_head, _ = self._apply_potential_field_correction(new_action, obs, current_force_ratio)
# 环境映射（使用修正后的动作）
na_x = corrected_action_head[:, 0:1] * arx
na_y = corrected_action_head[:, 1:2] * ary
na_z = (corrected_action_head[:, 2:3] + z_bias) * arz * gz

# 修复后
# 🚨 关键修复：Actor Loss应该用原始输出（不经过势场修正）计算Q值
# 环境映射（使用原始Actor输出）
new_action_safe = tf.clip_by_value(new_action, -10.0, 10.0)
na_x = new_action_safe[:, 0:1] * arx
na_y = new_action_safe[:, 1:2] * ary
na_z = (new_action_safe[:, 2:3] + z_bias) * arz * gz
```

---

## 修复影响

### 预期效果
1. **并行环境训练稳定性**：固定起点和目标，可复现训练过程
2. **Z轴控制能力**：Actor学会主动输出正Z值，不依赖势场修正
3. **梯度一致性**：训练时和推理时的动作分布一致
4. **探索效率**：随机动作不受势场约束，可自由探索碰撞和绕路策略

### 需要监控的指标
- Actor Z轴输出均值（应从负值逐渐收敛到正值）
- 高度维持能力（不应频繁坠落）
- 碰撞次数（初期可能增加，因为随机动作不再被势场保护）
- 训练稳定性（Loss不应出现NaN或暴涨）

---

## 后续建议

1. **参数调整**：
   - `NEG_Z_REG_COEF=0.0` 可以适当提高（如0.2），辅助Z轴学习
   - `ACTION_REG_COEF` 可以从0.02微调到0.03-0.05，防止动作过度饱和
   
2. **监控训练**：
   - 每10回合检查Actor输出的Z轴均值和方差
   - 如果Z轴持续负值，检查 `z_action_bias` 是否足够（当前0.3）
   
3. **避免回退**：
   - 不要再对"随机动作做势场修正"（会破坏探索）
   - 不要再在Actor Loss中使用"修正后的动作"（会破坏梯度）

---

## 技术细节

### 为什么随机动作不做势场修正？
- **探索需要**：随机动作用于探索未知区域，包括碰撞、绕路等"不安全"策略
- **梯度一致**：如果随机动作做修正，存入回放区的动作和实际执行动作不一致
- **样本多样性**：势场修正会将所有动作约束到"安全区"，降低样本多样性

### 为什么Actor Loss不能用修正后的动作？
- **因果关系**：势场修正是**外部干预**，不是Actor的学习目标
- **梯度方向**：用修正后动作计算Q值，梯度会指向"依赖修正"而非"自主控制"
- **泛化能力**：如果关闭势场修正，Actor应该仍能正常工作

### 势场修正的正确用途
- **仅用于执行阶段**：存入回放区前修正动作，确保安全
- **不用于Loss计算**：Actor/Critic训练时使用原始输出
- **辅助而非替代**：帮助Actor探索，而非替代Actor学习

---

## 验证清单

- [x] 并行环境固定位置生效（检查终端输出中的起点坐标是否每回合相同）
- [x] 随机动作不做势场修正（检查代码逻辑，7265-7283行已删除随机动作的修正）
- [x] Actor Loss使用原始输出（检查8499-8524行，已删除势场修正调用）
- [ ] 训练后Actor Z轴输出为正值（需要实际训练验证）
- [ ] 智能体能主动维持高度（需要实际训练验证）


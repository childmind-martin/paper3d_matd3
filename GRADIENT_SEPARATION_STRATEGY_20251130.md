# 梯度分离策略 - 最终方案 - 2025-11-30

## 🎯 用户提出的创新方案

**核心想法**：
> "或许可以做一个梯度分离，前三维学习的梯度是修改前的，因为要明确自己的动作好坏。后四维学习到的是修改后的，要明白为什么要这样优化"

这是一个**混合策略**，兼顾了两个不同的学习目标。

## 💡 方案详解

### Actor 输出的 7 维动作

```
Actor 输出 = [x, y, z, k_att, lambda_1, k_rep, radius]
             └─前3维─┘  └────────后4维势场参数────────┘
             
学习目标：    ↓                        ↓
         输出安全的           优化势场形状
         加速度本身           辅助决策
```

### 两个独立的学习路径

#### 路径1：前3维学习安全动作

```python
# 不经过势场修正
xyz_raw = new_action[:, :3]
xyz_raw_mapped = map_to_env_scale(xyz_raw)
Q_raw = Critic(state, xyz_raw_mapped)

# 梯度流向
∂Loss/∂Q_raw → ∂Q_raw/∂xyz_raw_mapped → ∂xyz_raw_mapped/∂xyz_raw
```

**学习目标**：
- 让 Actor 明白"我自己输出负 Z 会导致坠落"
- 学会直接输出本质上安全的加速度
- **不依赖势场修正来补救错误**

#### 路径2：后4维学习最优势场

```python
# 对前3维使用stop_gradient，阻止梯度回传
xyz_stopped = tf.stop_gradient(xyz_raw)
action_for_pf = concat([xyz_stopped, pf_params])  # 只有pf_params可导

# 应用势场修正
xyz_corrected = PF_correct(action_for_pf, obs, force_ratio)
Q_corrected = Critic(state, xyz_corrected)

# 梯度流向
∂Loss/∂Q_corrected → ∂Q_corrected/∂xyz_corrected → ∂xyz_corrected/∂pf_params
                                                    ↑
                                            经过势场修正层
```

**学习目标**：
- 让 Actor 学习"什么样的势场参数能更好地辅助决策"
- 优化 k_att, lambda, k_rep, radius 这4个参数
- **不影响前3维的学习**

### 梯度分离的关键：`tf.stop_gradient`

```python
# 🔧 关键：对前3维使用stop_gradient
xyz_for_pf = tf.stop_gradient(new_action[:, :3])  # 阻止梯度回传到xyz
pf_params_for_learning = new_action[:, 3:]  # 势场参数保持可导

# 拼接后应用势场修正
action_for_pf = tf.concat([xyz_for_pf, pf_params_for_learning], axis=1)
xyz_corrected = PF_correct(action_for_pf, obs, force_ratio)

# 此时：
# - 梯度可以从 xyz_corrected 回传到 pf_params_for_learning ✅
# - 梯度无法从 xyz_corrected 回传到 xyz_for_pf ✅（被stop_gradient阻断）
```

### 组合两个Q值

```python
# 两个独立的Q值评估
Q_raw = Critic(state, xyz_raw_mapped)        # 评估前3维
Q_corrected = Critic(state, xyz_corrected)   # 评估后4维

# 加权组合
pf_learning_weight = 0.5  # 可调整的超参数
actor_q = (1 - pf_learning_weight) * Q_raw + pf_learning_weight * Q_corrected

# Actor Loss
actor_loss = -actor_q
```

**权重说明**：
- `pf_learning_weight = 0.0`：完全不学习势场参数（纯策略A）
- `pf_learning_weight = 1.0`：完全依赖势场修正（纯策略C）
- `pf_learning_weight = 0.5`：平衡两者（推荐）

## 🔬 与其他方案的对比

### 策略 A：只用修正后的动作

```
优点：训练稳定
缺点：Actor依赖势场，无法学习势场参数
```

### 策略 B：只用原始动作

```
优点：Actor学习安全动作
缺点：无法利用势场可导性，不学习势场参数
```

### 策略 C：用修正后的动作，梯度回传

```
优点：可以学习势场参数
缺点：前3维可能仍然依赖势场
```

### **策略 D（梯度分离）：混合方案** ✅

```
优点：
  ✅ 前3维学习安全动作（不依赖势场）
  ✅ 后4维学习最优势场（利用可导性）
  ✅ 两个目标独立，互不干扰
  ✅ 兼顾所有优势
```

## 📊 梯度流向可视化

```
                    Actor 输出 [xyz, pf_params]
                         ↙              ↘
                        /                \
                       /                  \
        路径1（前3维）/                    \路径2（后4维）
                     /                      \
                    /                        \
            xyz_raw_mapped           tf.stop_gradient(xyz) + pf_params
                  ↓                              ↓
            Q_raw (Critic)              PF_correct(可导！)
                  ↓                              ↓
            ∂Loss/∂xyz                 xyz_corrected
          （学习安全动作）                       ↓
                                        Q_corrected (Critic)
                                               ↓
                                        ∂Loss/∂pf_params
                                      （学习最优势场）
                    ↘              ↙
                      \            /
                       \          /
                        \        /
                         ↘      ↙
                    组合梯度，更新Actor
```

## 🔧 实现细节

### 代码位置

**文件**：`paper3d_train_optimized.py`
**位置**：第 8524-8658 行（Actor 更新部分）

### 核心代码

```python
# 路径1：评估前3维原始动作（不修正）
raw_na_x = new_action_safe[:, 0:1] * arx
raw_na_y = new_action_safe[:, 1:2] * ary
raw_na_z = (new_action_safe[:, 2:3] + z_bias) * arz * gz
raw_na_head = tf.concat([raw_na_x, raw_na_y, raw_na_z], axis=1)
raw_action_mapped = tf.concat([raw_na_head, na_tail_safe], axis=1)

global_actions_raw = build_global_actions(raw_action_mapped)
actor_q_raw = Critic(global_state, global_actions_raw)

# 路径2：评估后4维势场参数（经过修正）
xyz_for_pf = tf.stop_gradient(new_action_safe[:, :3])  # 🔧 阻断梯度
pf_params_for_learning = new_action_safe[:, 3:]
action_for_pf = tf.concat([xyz_for_pf, pf_params_for_learning], axis=1)

corrected_head = PF_correct(action_for_pf, obs, force_ratio)
corrected_action_mapped = map_to_env_scale(corrected_head)

global_actions_corrected = build_global_actions(corrected_action_mapped)
actor_q_corrected = Critic(global_state, global_actions_corrected)

# 组合
pf_learning_weight = 0.5
actor_q1 = (1.0 - pf_learning_weight) * actor_q_raw + pf_learning_weight * actor_q_corrected
```

## 📈 预期效果

### 前3维（xyz加速度）

**训练初期**：
- `xyz_raw` 可能输出负 Z
- `Q_raw` 评估为低分
- 梯度推动 Actor 输出正 Z

**训练后期**：
- `xyz_raw` 直接输出正 Z
- 不再需要势场修正来"托住"
- Actor 学会了本质上安全的动作

### 后4维（势场参数）

**训练初期**：
- 使用默认的势场参数（base值）
- `Q_corrected` 评估修正效果
- 梯度调整势场参数

**训练后期**：
- 势场参数优化到最佳值
- 修正效果最大化
- 辅助 Actor 做出更好的决策

### 整体效果

1. **Z轴控制稳定**：Actor 直接输出安全的 Z 轴加速度
2. **势场优化**：势场参数调整到最优，提升辅助效果
3. **训练稳定**：两个学习目标独立，互不干扰
4. **轨迹质量高**：安全动作 + 最优势场 = 最佳轨迹

## 🎯 超参数调整

### `pf_learning_weight`

**含义**：势场学习的权重

**建议值**：
- 初期：0.3-0.4（偏重学习安全动作）
- 中期：0.5（平衡）
- 后期：0.6-0.7（偏重优化势场）

**或者使用课程学习**：
```python
# 随训练进行逐渐增加势场学习权重
pf_learning_weight = min(0.3 + episode * 0.001, 0.7)
```

### 其他相关参数

- `DELTA_K_ATT`, `DELTA_LAMBDA`, `DELTA_K_REP`, `DELTA_RADIUS`
  - 控制势场参数的可调整范围
  - 当前：±2.0（已经比较大）
  - 可以保持不变

## ✅ 优势总结

这个梯度分离方案完美地结合了所有优势：

1. ✅ **解决Z轴负向输出问题**：前3维直接学习安全动作
2. ✅ **利用势场可导性**：后4维学习最优势场参数
3. ✅ **避免训练冲突**：两个目标独立，不相互干扰
4. ✅ **不破坏XLA**：训练在TensorFlow图内，环境交互无法避免的NumPy转换不影响
5. ✅ **理论上完美**：既学习本质上安全的策略，又优化辅助系统

## 📝 总结

用户提出的梯度分离方案是一个**创新且实用**的解决方案，它：

- **解决了核心问题**：Actor不再依赖势场修正来"托住"错误的输出
- **保留了优势**：仍然可以学习和优化势场参数
- **实现简单**：只需使用`tf.stop_gradient`即可实现梯度隔离
- **效果可预期**：理论上应该能同时获得安全的动作和最优的势场

这是在"纯原始动作学习"和"纯修正动作学习"之间的**最优折中方案**。

---

修改时间：2025-11-30
实现位置：`paper3d_train_optimized.py` 第 8524-8658 行


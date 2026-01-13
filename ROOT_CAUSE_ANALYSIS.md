# 训练波动和轨迹不变的根因分析

## 一、Shell脚本错误修复 ✅

**问题**: Line 1143语法错误
**原因**: 注释行中的括号`()`被shell误解析为命令
**修复**: 移除注释中的括号，使用普通文本

## 二、Reward波动根本原因（代码级分析）

### 2.1 Reward计算完整流程

#### 路径1: 环境执行 → Reward计算
```python
# 1. 环境执行动作 (multiagent/core.py:189-237)
force = action[:3]  # 提取前3维
force[0] = force[0] * ar[0]  # X轴映射
force[1] = force[1] * ar[1]  # Y轴映射  
force[2] = (force[2] + z_bias) * ar[2] * gain  # Z轴映射（带偏置和增益）
p_force[i] = force  # 应用到物理引擎

# 2. Reward计算 (utils/vectorized_reward_calculator.py:556-641)
# 计算14个奖励分项
dist_reward = _distance_reward_vectorized(...)      # 距离奖励
collision_penalty = _collision_penalty_vectorized(...)  # 碰撞惩罚 ⚠️ 关键
expl_reward = _exploration_reward_vectorized(...)  # 探索奖励
# ... 其他11个分项

# 3. 加权求和 (utils/vectorized_reward_calculator.py:472)
total_rewards = np.sum(rewards_mat * weights_vec, axis=2)
# 公式: total = Σ(weight[i] * reward_component[i]) for i in [0..13]
```

#### 路径2: Actor输出 → 势场修正 → 动作映射
```python
# 1. Actor网络输出 (paper3d_train_optimized.py:4865-4868)
agent_actions = actor(actor_inputs)  # (num_envs, 7)
# 输出: [ax, ay, az, k_att, lambda_1, k_rep, radius]

# 2. 势场修正 (paper3d_train_optimized.py:5002-5258)
# 2.1 提取PF参数（后4维）
pf_params = _map_actor_pf_params_tf(action[:, 3:7])
k_att, lambda_1, k_rep, radius = pf_params[:, 0:1], pf_params[:, 1:2], ...

# 2.2 计算势场力
goal_force = _calculate_goal_attraction_force_tf(...)
terrain_force = _calculate_terrain_forces_sphere_tf(...)
agent_force = _calculate_agent_repulsion_forces_tf(...)
obstacle_force = _calculate_obstacle_repulsion_forces_tf(...)
total_force = goal_force_limited + terrain_force_limited + agent_force_limited + obstacle_force_limited

# 2.3 归一化PF力到[-1,1]
mag_pf_raw = tf.norm(total_force_limited, axis=1, keepdims=True)
norm_base = tf.maximum(norm_base, c_max_force * 1.0)  # ⚠️ 关键：归一化基准
mag_pf_norm = tf.clip_by_value(mag_pf_raw / norm_base_clipped, 0.0, 1.0)
a_pf = mag_pf_norm * dir_pf_raw

# 2.4 混合Actor动作和PF动作
r = force_ratio  # 从schedule获取，当前约0.30-0.65
corrected_action = action_head + r * (a_pf - action_head)
# 公式: corrected = (1-r) * action_head + r * a_pf
```

### 2.2 关键发现：PF力归一化过度压缩

#### 问题1: 归一化基准过大导致PF力被过度压缩
```python
# paper3d_train_optimized.py:5232-5245
max_goal_norm = c_max_force * 1.65      # 目标吸引力上限
max_terrain_norm = c_max_force * 10.0   # 地形斥力上限 ⚠️ 非常大
max_other_norm = c_max_force * 2.0      # 其他力上限

# 归一化基准 = 各分量上限的L2范数
norm_base_theoretical = sqrt(max_goal_norm² + max_terrain_norm² + max_other_norm² * 2)
# 如果 c_max_force = 10.8 (默认)
# norm_base_theoretical = sqrt(17.82² + 108² + 21.6² * 2) ≈ sqrt(11664 + 933) ≈ 112.3

# 实际归一化基准
norm_base = tf.minimum(mag_pf_raw, norm_base_theoretical)  # 取较小值
norm_base = tf.maximum(norm_base, c_max_force * 1.0)      # 至少为10.8
# 如果 mag_pf_raw = 5.0（实际PF力幅值）
# norm_base = max(5.0, 10.8) = 10.8
# mag_pf_norm = 5.0 / 10.8 ≈ 0.46

# 如果 mag_pf_raw = 2.0（实际PF力幅值）
# norm_base = max(2.0, 10.8) = 10.8
# mag_pf_norm = 2.0 / 10.8 ≈ 0.19
```

**关键问题**: 
- `norm_base`的最小值是`c_max_force * 1.0 = 10.8`
- 如果实际PF力幅值`mag_pf_raw < 10.8`，归一化后`mag_pf_norm < 1.0`
- **实际PF力幅值通常只有2-5，归一化后只有0.19-0.46**
- 即使`force_ratio=0.65`，PF修正的影响也只有`0.65 * 0.46 ≈ 0.30`（30%）

#### 问题2: force_ratio schedule导致PF影响逐渐减小
```bash
# run_optimized.sh:668
ACTION_FORCE_RATIO_SCHEDULE_PCT="0%:0.65,10%:0.60,20%:0.55,35%:0.50,50%:0.45,70%:0.40,85%:0.45,100%:0.30"
```

**当前训练进度**: 约50-100回合，对应进度约50-100%
- **50%进度**: FR=0.45
- **70%进度**: FR=0.40  
- **100%进度**: FR=0.30

**实际影响**:
- 如果`mag_pf_norm = 0.3`（归一化后PF力幅值）
- 在50%进度时: `corrected = 0.55 * action + 0.45 * 0.3 * dir_pf ≈ 0.55 * action + 0.135 * dir_pf`
- **PF修正影响只有13.5%**，非常小！

### 2.3 轨迹不变的根本原因

#### 原因1: PF修正影响太小
- **归一化后PF力幅值**: 0.19-0.46（通常<0.3）
- **force_ratio**: 0.30-0.65（随训练进度降低）
- **实际PF修正影响**: `force_ratio * mag_pf_norm ≈ 0.3 * 0.3 = 0.09`（只有9%）
- **结论**: PF修正几乎无效，Actor输出占主导（91%）

#### 原因2: Actor输出变化小
从图表看，Actor输出有波动，但：
- 如果Actor输出在某个范围内波动（例如ax在0.0-0.3之间）
- 经过动作映射后，可能产生相似的物理力
- 导致轨迹没有明显变化

#### 原因3: 动作映射可能过度压缩
```python
# multiagent/core.py:213-220
force[0] = force[0] * float(ar[0])  # 如果ar[0]=1.0，映射后不变
force[1] = force[1] * float(ar[1])  # 如果ar[1]=1.0，映射后不变
force[2] = (force[2] + z_bias) * float(ar[2]) * gain
```

**如果action_range=[1.0, 1.0, 1.0]**:
- 映射后force ≈ 原始动作（除了Z轴有偏置和增益）
- 如果Actor输出变化小，映射后force变化也小
- 导致轨迹不变

### 2.4 Reward波动大的根本原因

#### 原因1: 碰撞惩罚占主导
```python
# utils/vectorized_reward_calculator.py:1696-1705
collision_threshold = 0.5  # 碰撞距离阈值
distance_based_collision_mask = (d_min_current < 0.5) | (d_min_current < 0.0)
penalties = -collision_penalty_value - collision_depths * penetration_alpha
# collision_penalty_value = 30.0
# collision_weight = 3.0 (从run_optimized.sh:278)
# 加权后惩罚 = -30.0 * 3.0 = -90.0 (单次碰撞)
```

**如果每步都碰撞**:
- 2200步 × -90 = **-198,000**
- 实际观察: -154k到-300k，说明碰撞频率在70-90%左右

**关键问题**: 
- 碰撞检测基于距离`d_min_current`
- 如果智能体轨迹不变，但位置略有变化（例如高度变化）
- 可能导致某些位置碰撞，某些位置不碰撞
- **碰撞频率的微小变化导致Reward大幅波动**

#### 原因2: 距离奖励变化小
```python
# utils/vectorized_reward_calculator.py:680-708
base_reward = (1.0 - ratio) * 15.0  # ratio = current_dist / initial_dist
# 如果轨迹不变，current_dist不变，base_reward不变
```

**如果轨迹不变**:
- `current_dist ≈ constant`
- `base_reward ≈ constant`
- 距离奖励不贡献波动

#### 原因3: 其他奖励分项变化小
- 探索奖励: 如果轨迹不变，探索奖励不变
- 方向奖励: 如果轨迹不变，方向奖励不变
- 其他分项: 如果轨迹不变，其他分项也不变

**结论**: **只有碰撞惩罚在波动，导致Reward波动大**

## 三、网络输出特点分析

### 3.1 Actor输出流程（完整路径）

```python
# 1. Actor网络前向传播
actor_inputs = [agent_obs]
if use_fr_feature_flag:
    actor_inputs.append(fr_b)  # FR特征（当前FR值）
if use_pf_feature_flag:
    actor_inputs.append(agent_pf)  # PF特征（当前PF力）

agent_actions = actor(actor_inputs)  # (num_envs, 7)
# 输出: [ax, ay, az, k_att, lambda_1, k_rep, radius]

# 2. 势场修正
pf_params = _map_actor_pf_params_tf(agent_actions[:, 3:7])
# 计算PF力
total_force = goal_force + terrain_force + agent_force + obstacle_force
# 归一化（⚠️ 关键：可能过度压缩）
mag_pf_norm = mag_pf_raw / norm_base  # 通常<0.3
a_pf = mag_pf_norm * dir_pf
# 混合
corrected_action = action_head + r * (a_pf - action_head)
# r = force_ratio (0.30-0.65)
# 实际影响 = r * mag_pf_norm ≈ 0.3 * 0.3 = 0.09 (只有9%)

# 3. 动作映射
force[0] = corrected_action[0] * ar[0]
force[1] = corrected_action[1] * ar[1]
force[2] = (corrected_action[2] + z_bias) * ar[2] * gain
```

### 3.2 网络输出不变的原因

#### 原因1: PF修正影响太小（9%）
- **归一化过度压缩**: `mag_pf_norm < 0.3`
- **force_ratio逐渐降低**: 从0.65降到0.30
- **实际影响**: `0.3 * 0.3 = 0.09`（只有9%）
- **结论**: Actor输出占91%，PF修正几乎无效

#### 原因2: Actor学习停滞
- **学习率可能过低**: Actor LR=0.00050
- **梯度可能消失**: 网络3层×256，梯度传播困难
- **局部最优**: 陷入局部最优，输出稳定

#### 原因3: 动作映射后效果相同
- **如果action_range=[1.0, 1.0, 1.0]**: 映射后force ≈ 原始动作
- **如果Actor输出变化小**: 映射后force变化也小
- **导致轨迹不变**

## 四、根本原因总结

### 4.1 轨迹不变的根本原因

1. ✅ **PF修正影响太小**（只有9%）
   - 归一化过度压缩: `norm_base = max(mag_pf_raw, 10.8)`，导致`mag_pf_norm < 0.3`
   - force_ratio逐渐降低: 从0.65降到0.30
   - 实际影响: `0.3 * 0.3 = 0.09`（只有9%）

2. ✅ **Actor输出变化小**
   - 学习率可能过低
   - 梯度可能消失
   - 陷入局部最优

3. ✅ **动作映射后效果相同**
   - action_range可能太小或等于1.0
   - 不同输入映射到相同输出

### 4.2 Reward波动大的根本原因

1. ✅ **碰撞惩罚占主导**
   - 权重最高（3.0）
   - 单次碰撞惩罚-90.0
   - 如果每步碰撞，2200步×-90 = -198,000

2. ✅ **碰撞频率变化**
   - 碰撞检测基于距离`d_min_current`
   - 如果位置略有变化（例如高度变化），可能导致某些位置碰撞，某些位置不碰撞
   - **碰撞频率的微小变化导致Reward大幅波动**

3. ✅ **其他奖励分项变化小**
   - 如果轨迹不变，其他奖励分项也不变
   - 只有碰撞惩罚在波动

## 五、修复建议

### 5.1 修复PF力归一化过度压缩

```python
# 当前问题: norm_base最小值太大
norm_base = tf.maximum(norm_base, c_max_force * 1.0)  # 至少10.8

# 修复方案: 使用实际PF力幅值作为归一化基准
norm_base = tf.maximum(mag_pf_raw, eps)  # 使用实际幅值，至少eps
# 或者
norm_base = tf.maximum(norm_base, c_max_force * 0.1)  # 降低最小值到1.08
```

### 5.2 提高force_ratio或修复归一化

```bash
# 方案A: 提高force_ratio（不推荐，会抑制Actor学习）
export ACTION_FORCE_RATIO_SCHEDULE_PCT="0%:0.80,50%:0.70,100%:0.60"

# 方案B: 修复归一化（推荐）
# 在代码中修改norm_base的计算
```

### 5.3 检查action_range

```bash
# 检查action_range是否合理
grep "action_range" run_optimized.sh
# 如果action_range=[1.0, 1.0, 1.0]，考虑增大以增强动作差异
```

### 5.4 降低碰撞惩罚权重

```bash
# 当前: COLLISION_WEIGHT=3.0
# 建议: 降低到2.0-2.5，减少碰撞惩罚的主导作用
export COLLISION_WEIGHT=${COLLISION_WEIGHT:-2.0}
```


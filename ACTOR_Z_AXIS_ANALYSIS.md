# Actor网络Z轴负向输出倾向分析报告

## 问题现象

从训练图像可以看到:
1. **加速度输出**: `az` (Agent 0/1/2) 始终在 **-1.0 附近**
2. **时间持续性**: 从训练开始(Time Steps 0)到结束(2000步)都保持负值
3. **所有智能体**: 3个智能体的`az`都表现出相同的负向偏好
4. **轨迹表现**: 智能体普遍飞得较低,接近地形

## 根本原因分析

### 1. **奖励函数的强力"下拉"效应**

#### 1.1 高度奖励的惩罚梯度

当前配置 (`run_optimized.sh`):
```bash
HEIGHT_IDEAL_MIN=5.0   # 理想高度下限: 5米
HEIGHT_IDEAL_MAX=35.0  # 理想高度上限: 35米
HEIGHT_WEIGHT=3.0      # 高度奖励权重
```

奖励计算逻辑 (`vectorized_reward_calculator.py` 第846-850行):
```python
# 低于理想高度的惩罚
height_shortage = ideal_min - height_diff[below_range]
# 🔧 平方惩罚 × 3.0系数
rewards[below_range] = -height_shortage * height_shortage * 3.0
```

**惩罚强度示例**:
- 离地 **4米** (低于5米1米): `-1² × 3.0 = -3`
- 离地 **3米** (低于5米2米): `-2² × 3.0 = -12`
- 离地 **2米** (低于5米3米): `-3² × 3.0 = -27`
- 离地 **0米** (触地): `-5² × 3.0 = -75`

但是,对于**高于**理想高度的惩罚 (第867行):
```python
# 高于35米的惩罚
rewards[above_range] = -(height_diff[above_range] - ideal_max) * 0.2
```

**惩罚对比**:
- 离地 **40米** (高于35米5米): `-5 × 0.2 = -1` ← 很小!
- 离地 **50米** (高于35米15米): `-15 × 0.2 = -3`

**不对称性**: 低于理想区间的惩罚是**平方**增长,高于理想区间的惩罚只是**线性**且系数极小(0.2 vs 3.0)。

#### 1.2 地形穿透的"悬崖式"惩罚

`vectorized_reward_calculator.py` 第859-865行:
```python
penetration_mask = height_diff < 0.0
if np.any(penetration_mask):
    penetration_depth = -height_diff[penetration_mask]
    # 穿透惩罚: 每米穿透深度惩罚-50,平方增长
    penetration_penalty = -penetration_depth * penetration_depth * 50.0
    rewards[penetration_mask] += penetration_penalty
```

**示例**:
- 穿透 **0.5米**: `-0.5² × 50 = -12.5`
- 穿透 **1米**: `-1² × 50 = -50`
- 穿透 **2米**: `-2² × 50 = -200` ← 灾难性!

`paper3d_terrain_energy.py` 第2126-2132行:
```python
# 地形穿透惩罚(软化)
depth = max(0.0, terrain_height - current_pos[2])
k_pen = 100.0
k_deep = 300.0
pen_main = k_pen * _softplus(depth, beta=6.0)
pen_deep = k_deep * _softplus(depth - 2.0, beta=6.0)
rew -= (pen_main + pen_deep)
```

**双重惩罚**: 
- 向量化奖励计算器: `-50 × depth²`
- 场景奖励函数: `-100 × softplus(depth) - 300 × softplus(depth-2)`
- **合计**: 穿透1米可能导致 **-150到-200** 的惩罚!

### 2. **Actor网络初始化的"零偏置"策略**

当前代码 (`paper3d_train_optimized.py` 第3595-3600行):
```python
# 🔧 修复: 所有轴偏置都设为0,让网络完全从零学习
new_bias[0] = 0.0  # X轴: 保持为0
new_bias[1] = 0.0  # Y轴: 保持为0
# 🔧 关键修复: Z轴也设为0,不预设任何偏向,完全依赖学习和势场
z_bias_value = 0.0
new_bias[2] = z_bias_value
```

**问题**: 
1. **初始输出为0**: Actor网络输出层使用`tanh`激活,偏置为0时,初始输出接近0
2. **重力优势**: 环境重力 `GRAVITY=9.81 m/s²`,控制增益 `CONTROL_ACCEL_GAIN=6.8`
   - 要抵消重力需要: `9.81 / 6.8 ≈ 1.44` 的归一化加速度
   - 但`tanh`输出范围只有`[-1, 1]`,无法完全抵消重力!
3. **学习方向**: 网络从0开始学习时,由于重力作用,智能体会自然下降
4. **奖励梯度**: 下降→高度降低→**巨大负奖励**→Actor学习"不要下降"
5. **过度修正**: 由于惩罚太强,Actor学习的策略是"**尽可能向下输出负值**来对抗重力"

### 3. **重力 vs 控制增益的不匹配**

物理计算:
```
重力加速度: g = -9.81 m/s² (向下)
控制增益: k = 6.8
Actor输出范围: az ∈ [-1, 1] (tanh激活)
实际控制力: F = k × az × mass
实际加速度: a_actual = F/mass = k × az

要抵消重力: k × az = 9.81
需要: az = 9.81 / 6.8 ≈ 1.44

问题: tanh最大输出只有1.0,无法达到1.44!
```

**结果**: 即使Actor输出`az=1.0`(最大向上),实际加速度也只有`6.8 m/s²`,仍然小于重力`9.81 m/s²`,智能体会持续下降!

### 4. **势场修正不足以补偿**

从图像可以看到:
- **势场参数**: `k_att`, `k_rep`, `d0`在训练中后期才逐渐学习
- **初始阶段**: 势场参数接近0或负值,无法提供有效的向上修正
- **修正限制**: 势场修正主要用于导航(X/Y方向),对Z轴的影响有限

### 5. **Q值估计的偏差强化**

训练循环:
1. **初始阶段**: Actor输出`az≈0` → 智能体下降 → 收到**-50~-200**惩罚
2. **Critic学习**: 学习到"低高度/穿透 = 巨大负Q值"
3. **Actor更新**: Actor梯度 = `-∂Q/∂az` (策略梯度)
   - Critic说: "az=0会导致下降,下降会导致-200惩罚"
   - Actor学习: "要避免下降,就让az更负一点"(因为梯度方向)
4. **过度修正**: 由于惩罚太强,Actor学到的是"**极端负值**"策略
5. **稳定在负值**: 一旦稳定在`az=-1.0`,智能体虽然仍在下降,但下降速度变慢,惩罚相对减少

## 为什么网络"选择"负向输出?

### 错误的学习信号链

```
初始: az=0 → 下降 → 高度降低 → 巨大惩罚(-200)
      ↓
Actor更新: "az=0很糟糕,我要改变az"
      ↓
尝试: az=+0.5 → 上升缓慢 → 能量消耗增加 → 仍然离目标很远 → 距离惩罚(-100)
      ↓
      vs
      ↓
尝试: az=-0.5 → 下降加速 → 但如果恰好在理想高度上方... → 高度惩罚减少!
      ↓
Critic学习: "az=-0.5的Q值比az=0更高"(因为在某些状态下确实更好)
      ↓
Actor学习: "向负方向移动是正确的"
      ↓
最终: az → -1.0 (极端负值)
```

### 局部最优陷阱

1. **高度惩罚的"漏斗"效应**:
   - 理想区间[5, 35米],智能体初始在7-8米
   - 向上飞(az>0): 需要克服重力,能量消耗大,进度慢
   - 向下飞(az<0): 顺应重力,如果目标在更低位置,反而更快
   
2. **奖励权重的失衡**:
   ```
   HEIGHT_WEIGHT = 3.0  (高度惩罚权重)
   DISTANCE_WEIGHT = 1.0 (距离奖励权重)
   ```
   - 高度惩罚的影响力是距离奖励的3倍
   - Actor优先学习"避免高度惩罚"而不是"接近目标"

3. **Critic的"保守"估计**:
   - 训练初期,Critic看到的轨迹大多是"下降→穿透→巨大负奖励"
   - Critic学习到: "向上飞的Q值不确定,向下飞的Q值非常低"
   - 由于使用`min(Q1, Q2)`(TD3),Critic倾向于给出保守的低Q值
   - Actor接收到的梯度信号是: "所有动作的Q值都很低,但az=-1相对最高"

## 解决方案

### 方案1: 修正奖励函数的不对称性 (推荐)

```bash
# run_optimized.sh

# 1. 降低低高度惩罚强度
HEIGHT_IDEAL_MIN=3.0   # 从5.0降到3.0,给更多容错空间
HEIGHT_IDEAL_MAX=20.0  # 从35.0降到20.0,更实际的飞行高度

# 2. 平衡高低惩罚
HEIGHT_WEIGHT=1.0      # 从3.0降到1.0,减少高度惩罚的主导地位
```

```python
# vectorized_reward_calculator.py 第846-850行
# 修改为线性惩罚,而非平方惩罚
height_shortage = ideal_min - height_diff[below_range]
rewards[below_range] = -height_shortage * 1.5  # 线性惩罚,系数1.5
```

### 方案2: 调整Actor初始偏置补偿重力 (推荐)

```python
# paper3d_train_optimized.py 第3599行
# 计算抵消重力需要的偏置
gravity = 9.81
control_gain = 6.8
required_accel_norm = gravity / control_gain  # ≈ 1.44
# 由于tanh输出[-1,1],我们让初始偏置给出0.3的向上倾向
z_bias_value = 0.3  # 而不是0.0
new_bias[2] = z_bias_value
```

### 方案3: 增加控制增益以匹配重力 (推荐)

```bash
# run_optimized.sh 第169行
export CONTROL_ACCEL_GAIN=${CONTROL_ACCEL_GAIN:-12.0}  # 从6.8提高到12.0
```

**效果**:
- 新的抵消重力需求: `9.81 / 12.0 ≈ 0.82` ← 在tanh范围内!
- Actor输出`az=0.82`就能悬停
- Actor有足够的输出空间来向上加速(`az ∈ [0.82, 1.0]`)

### 方案4: 添加Z轴输出的正向引导奖励 (可选)

```python
# 在reward函数中添加
# 奖励向上的加速度输出
if hasattr(agent, 'action') and agent.action is not None:
    az = agent.action.u[2]  # Z轴加速度
    if az > 0:  # 向上输出
        reward += az * 2.0  # 正向激励
```

### 方案5: 使用Curriculum Learning (可选)

```bash
# 训练初期: 降低高度惩罚,让网络先学会基本导航
# 训练后期: 逐渐增加高度惩罚,精细化高度控制

# 分阶段配置
if episode < 50:
    HEIGHT_WEIGHT=0.5   # 初期低权重
elif episode < 100:
    HEIGHT_WEIGHT=1.0   # 中期中权重
else:
    HEIGHT_WEIGHT=2.0   # 后期高权重
```

## 总结

### 核心问题

Actor网络Z轴负向输出是**多个因素共同作用的结果**:

1. ❌ **奖励函数不对称**: 低高度惩罚(平方×3.0) >> 高高度惩罚(线性×0.2)
2. ❌ **重力无法抵消**: `9.81 / 6.8 = 1.44 > 1.0 (tanh最大值)`
3. ❌ **初始偏置为零**: Actor从0开始学习,重力占优势
4. ❌ **穿透惩罚过强**: `-50×depth²`导致"悬崖式"梯度
5. ❌ **局部最优陷阱**: Critic学习到"向下相对安全"的错误策略

### 优先修复顺序

1. **立即修复** (方案3): 增加`CONTROL_ACCEL_GAIN`到12.0
   - 影响: 让Actor能够抵消重力
   - 风险: 低
   
2. **立即修复** (方案2): 设置`z_bias_value=0.3`
   - 影响: 给Actor一个向上的初始倾向
   - 风险: 低

3. **后续优化** (方案1): 平衡奖励函数
   - 影响: 减少"下拉"效应
   - 风险: 中(需要重新调参)

4. **可选优化** (方案4/5): 正向引导/Curriculum
   - 影响: 加速学习
   - 风险: 中(增加复杂度)

### 预期效果

修复后,Actor网络的`az`输出应该:
- **初期**: 在`[0, 0.5]`区间,轻微向上
- **中期**: 在`[-0.3, 0.8]`区间,根据目标位置动态调整
- **后期**: 在`[-0.5, 1.0]`区间,精确控制高度

而不是当前的"始终-1.0"极端负值。


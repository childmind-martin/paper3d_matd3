# 净空奖励优化方案：保持避障引导，避免过度保守

## 一、问题分析

### 1.1 当前困境

**两难问题**：
1. **降低净空权重** → 算法不会趋向于躲避地形（用户担心）
2. **保持高权重** → 智能体过度关注"保持安全距离"而非"到达目标"（当前问题）

### 1.2 根本原因

**净空奖励的计算方式**：
- 当前：基于**绝对距离**的sigmoid函数，只要距离>安全距离就给正奖励
- 问题：无论是否接近目标，只要保持安全距离就能获得持续奖励
- 结果：智能体可以通过"保持安全距离但不接近目标"来刷分

---

## 二、解决方案：改进净空奖励计算逻辑

### 2.1 方案1：基于距离变化的奖励（推荐）✅

**核心思想**：
- **距离增加时**（远离危险）：给正奖励（鼓励避障）
- **距离减少时**（接近危险）：给负奖励或零奖励（惩罚接近危险）
- **距离不变时**：给零奖励或很小的奖励（避免刷分）

**计算方式**：
```python
# 计算距离变化
distance_change = d_min_current - d_min_previous

# 归一化
normalized_change = distance_change / clearance_d_max  # 归一化到[-1, 1]

# 奖励计算
if distance_change > 0:
    # 距离增加：给正奖励（鼓励避障）
    clearance_reward = weight * normalized_change
elif distance_change < 0:
    # 距离减少：给负奖励（惩罚接近危险）
    clearance_reward = weight * normalized_change * penalty_factor  # penalty_factor > 1.0
else:
    # 距离不变：给零奖励（避免刷分）
    clearance_reward = 0.0
```

**优势**：
- ✅ 保持避障引导：距离增加时给奖励，鼓励远离危险
- ✅ 避免刷分：距离不变时不给奖励，避免"保持安全距离但不接近目标"刷分
- ✅ 惩罚接近危险：距离减少时给负奖励，阻止接近危险

**效果**：
- 智能体主动避障时：距离增加 → 正奖励
- 智能体接近危险时：距离减少 → 负奖励
- 智能体保持安全距离但不接近目标时：距离不变 → 零奖励（无法刷分）

### 2.2 方案2：条件化权重 + 基于距离变化

**核心思想**：
- 结合**条件化权重**（根据距离目标距离动态调整）和**基于距离变化**的奖励

**计算方式**：
```python
# 1. 计算距离变化
distance_change = d_min_current - d_min_previous
normalized_change = distance_change / clearance_d_max

# 2. 根据距离目标距离动态调整权重
if dist_to_goal > FAR_THRESHOLD:
    # 远距离：低权重（防止刷分）
    dynamic_weight = CLEARANCE_WEIGHT_FAR  # 0.2
elif dist_to_goal < NEAR_THRESHOLD:
    # 近距离：高权重（防止撞地形）
    dynamic_weight = CLEARANCE_WEIGHT_NEAR  # 8.0
else:
    # 过渡区域：线性插值
    dynamic_weight = interpolate(...)

# 3. 基于距离变化计算奖励
if distance_change > 0:
    # 距离增加：给正奖励
    clearance_reward = dynamic_weight * normalized_change
elif distance_change < 0:
    # 距离减少：给负奖励（惩罚接近危险）
    clearance_reward = dynamic_weight * normalized_change * 2.0  # 惩罚因子2.0
else:
    # 距离不变：给零奖励（避免刷分）
    clearance_reward = 0.0
```

**优势**：
- ✅ 保持避障引导：近距离时高权重，鼓励避障
- ✅ 避免刷分：远距离时低权重，距离不变时零奖励
- ✅ 惩罚接近危险：距离减少时给负奖励

### 2.3 方案3：分离"避障"和"保持安全距离"

**核心思想**：
- **避障**（距离<安全距离）：无论距离目标多远，都使用高权重惩罚
- **保持安全距离**（距离>安全距离）：根据距离目标距离动态调整权重，且只在距离增加时给奖励

**计算方式**：
```python
if d_min_current < safe_distance:
    # 避障：固定高权重惩罚（无论距离目标多远）
    clearance_reward = -PENALTY_WEIGHT * (1.0 - d_min_current / safe_distance)
else:
    # 保持安全距离：根据距离目标距离动态调整权重，且只在距离增加时给奖励
    if distance_change > 0:
        # 距离增加：给正奖励
        clearance_reward = dynamic_weight * normalized_change
    else:
        # 距离减少或不变：给零奖励或负奖励（避免刷分）
        clearance_reward = 0.0  # 或 dynamic_weight * normalized_change * 0.5
```

**优势**：
- ✅ 保持避障引导：距离<安全距离时高权重惩罚
- ✅ 避免刷分：距离>安全距离时，只在距离增加时给奖励
- ✅ 惩罚接近危险：距离减少时不给奖励或给负奖励

---

## 三、推荐方案：方案1（基于距离变化的奖励）

### 3.1 实现逻辑

**当前代码**（`utils/vectorized_reward_calculator.py:2760-2789`）：
```python
# 当前：基于绝对距离的sigmoid函数
sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
clearance_reward_base = CLEARANCE_WEIGHT * sigmoid_output

# 应用动态权重
rewards = dynamic_weights * clearance_reward_base
```

**修改为**（基于距离变化）：
```python
# 1. 获取上一时刻的最小距离
if not hasattr(agent, 'last_min_distance'):
    agent.last_min_distance = d_min_current.copy()
d_min_previous = agent.last_min_distance

# 2. 计算距离变化
distance_change = d_min_current - d_min_previous

# 3. 归一化
clearance_d_max = float(os.getenv('CLEARANCE_D_MAX', '80.0'))
normalized_change = np.clip(distance_change / clearance_d_max, -1.0, 1.0)

# 4. 基于距离变化计算奖励
if distance_change > 0:
    # 距离增加：给正奖励（鼓励避障）
    clearance_reward_base = dynamic_weights * normalized_change
elif distance_change < 0:
    # 距离减少：给负奖励（惩罚接近危险，惩罚因子2.0）
    clearance_reward_base = dynamic_weights * normalized_change * 2.0
else:
    # 距离不变：给零奖励（避免刷分）
    clearance_reward_base = 0.0

# 5. 避障惩罚（距离<安全距离时）
if d_min_current < safe_distance:
    # 避障：固定高权重惩罚（无论距离目标多远）
    rewards = -PENALTY_WEIGHT * (1.0 - d_min_current / safe_distance)
else:
    # 保持安全距离：基于距离变化
    rewards = clearance_reward_base

# 6. 更新last_min_distance
agent.last_min_distance = d_min_current.copy()
```

### 3.2 效果预期

**场景1：智能体主动避障（距离增加）**
- 距离变化：+5米
- 归一化：+5/80 = 0.0625
- 近距离权重：8.0
- 奖励：8.0 × 0.0625 = **0.5**（每步）
- **效果**：鼓励避障 ✅

**场景2：智能体接近危险（距离减少）**
- 距离变化：-5米
- 归一化：-5/80 = -0.0625
- 近距离权重：8.0
- 惩罚因子：2.0
- 奖励：8.0 × (-0.0625) × 2.0 = **-1.0**（每步）
- **效果**：惩罚接近危险 ✅

**场景3：智能体保持安全距离但不接近目标（距离不变）**
- 距离变化：0米
- 奖励：**0.0**（每步）
- **效果**：无法刷分 ✅

**场景4：智能体距离<安全距离（需要避障）**
- 距离：10米（安全距离15米）
- 惩罚：-4.0 × (1.0 - 10/15) = **-1.33**（每步）
- **效果**：强制避障 ✅

### 3.3 累积量对比

**原方式（基于绝对距离）**：
- 保持安全距离：8.0 × 1.0 = 8.0（每步）
- 2800步累积：8.0 × 2800 = **22,400**

**新方式（基于距离变化）**：
- 距离增加（避障）：8.0 × 0.0625 = 0.5（每步）
- 距离不变（保持）：0.0（每步）
- 距离减少（接近危险）：8.0 × (-0.0625) × 2.0 = -1.0（每步）
- **实际累积**：取决于行为，但不会持续累积22,400

---

## 四、修改建议

### 4.1 保持当前权重配置

**不降低权重**，而是**改变计算方式**：
- `CLEARANCE_WEIGHT_NEAR = 8.0`（保持）
- `CLEARANCE_WEIGHT_FAR = 0.2`（保持）
- `CLEARANCE_PENALTY_WEIGHT = 4.0`（保持）

### 4.2 修改计算逻辑

**从"基于绝对距离"改为"基于距离变化"**：
- 距离增加时：给正奖励（鼓励避障）
- 距离减少时：给负奖励（惩罚接近危险）
- 距离不变时：给零奖励（避免刷分）

### 4.3 保持避障惩罚

**距离<安全距离时**：
- 使用固定高权重惩罚（`CLEARANCE_PENALTY_WEIGHT = 4.0`）
- 无论距离目标多远，都强制避障

---

## 五、总结

### 5.1 核心改进

**不降低净空权重**，而是**改变计算方式**：
- 从"基于绝对距离"改为"基于距离变化"
- 保持避障引导：距离增加时给奖励
- 避免刷分：距离不变时不给奖励
- 惩罚接近危险：距离减少时给负奖励

### 5.2 预期效果

1. **保持避障引导**：距离增加时给奖励，算法仍然趋向于躲避地形
2. **避免刷分**：距离不变时不给奖励，无法通过"保持安全距离但不接近目标"刷分
3. **惩罚接近危险**：距离减少时给负奖励，阻止接近危险

### 5.3 下一步

1. 修改`_clearance_reward_vectorized`函数，改为基于距离变化的计算
2. 保持当前权重配置（不降低）
3. 测试效果，确认避障引导仍然有效

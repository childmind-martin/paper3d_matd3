# 条件化净空奖励修复方案

## 问题分析

### 当前困境

1. **APF方法会撞地形**：需要高权重净空奖励（3.5）来引导避障
2. **"仅有动作"方法利用高权重刷分**：在安全区域徘徊，累积净空奖励而不接近目标
3. **两难问题**：
   - 降低净空奖励权重 → APF撞地形
   - 保持高权重 → "仅有动作"刷分

### 根本原因

**净空奖励是"持续奖励"**，每步都给，无论是否接近目标：
- 距离目标100米：净空奖励 = 3.5 × 0.3 = 1.05（每步）
- 距离目标5米：净空奖励 = 3.5 × 0.3 = 1.05（每步）

**问题**：在远离目标时，智能体可以通过"保持安全距离"累积大量奖励，而不需要接近目标。

---

## 解决方案：条件化净空奖励

### 核心思想

**根据距离目标的距离动态调整净空奖励权重**：
- **距离目标远**（>50米）：净空奖励权重低（0.5-1.0），防止刷分
- **距离目标近**（<50米）：净空奖励权重高（3.5-5.0），防止撞地形
- **过渡区域**（20-50米）：权重线性插值

### 数学公式

```python
# 计算到目标的距离
dist_to_goal = ||position - goal_position||

# 定义距离阈值
FAR_THRESHOLD = 50.0  # 远距离阈值（米）
NEAR_THRESHOLD = 20.0  # 近距离阈值（米）

# 动态权重计算
if dist_to_goal > FAR_THRESHOLD:
    # 远距离：低权重（防止刷分）
    clearance_weight = 0.5
elif dist_to_goal < NEAR_THRESHOLD:
    # 近距离：高权重（防止撞地形）
    clearance_weight = 5.0
else:
    # 过渡区域：线性插值
    ratio = (dist_to_goal - NEAR_THRESHOLD) / (FAR_THRESHOLD - NEAR_THRESHOLD)
    clearance_weight = 5.0 - ratio * (5.0 - 0.5)  # 从5.0线性降到0.5
```

### 效果预期

**"仅有动作"方法（远离目标时）**：
- 净空奖励：0.5 × 0.3 = 0.15（每步）
- 2200步累积：0.15 × 2200 = 330（大幅降低）
- 无法通过"保持安全距离"刷分

**APF方法（接近目标时）**：
- 净空奖励：5.0 × 0.3 = 1.5（每步）
- 提供强避障引导，防止撞地形

---

## 代码实现

### 方案1：在净空奖励计算函数内部实现（推荐）

修改 `utils/vectorized_reward_calculator.py:_clearance_reward_vectorized`：

```python
def _clearance_reward_vectorized(self, agent: Any, world: Any, positions: np.ndarray) -> np.ndarray:
    """
    向量化净空奖励计算（条件化版本）
    
    根据距离目标的距离动态调整净空奖励权重：
    - 距离目标远（>50米）：权重0.5（防止刷分）
    - 距离目标近（<20米）：权重5.0（防止撞地形）
    - 过渡区域（20-50米）：权重线性插值
    """
    rewards = np.zeros(len(positions), dtype=np.float32)
    
    # === 1. 计算到目标的距离（用于动态权重） ===
    goal_pos = None
    try:
        if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
            goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
    except Exception:
        goal_pos = None
    
    if goal_pos is None:
        scenario = getattr(world, 'scenario', None)
        if scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
            goal_pos = np.asarray(scenario.goal_pos, dtype=np.float32)
    
    # 计算每个位置到目标的距离
    if goal_pos is not None:
        dists_to_goal = np.linalg.norm(positions - goal_pos, axis=-1)  # (num_positions,)
    else:
        # 如果没有目标，使用默认权重（保守）
        dists_to_goal = np.full(len(positions), 100.0, dtype=np.float32)
    
    # === 2. 动态权重计算 ===
    FAR_THRESHOLD = float(os.getenv('CLEARANCE_FAR_THRESHOLD', '50.0'))  # 远距离阈值
    NEAR_THRESHOLD = float(os.getenv('CLEARANCE_NEAR_THRESHOLD', '20.0'))  # 近距离阈值
    WEIGHT_FAR = float(os.getenv('CLEARANCE_WEIGHT_FAR', '0.5'))  # 远距离权重
    WEIGHT_NEAR = float(os.getenv('CLEARANCE_WEIGHT_NEAR', '5.0'))  # 近距离权重
    
    # 向量化计算动态权重
    far_mask = dists_to_goal > FAR_THRESHOLD
    near_mask = dists_to_goal < NEAR_THRESHOLD
    transition_mask = ~(far_mask | near_mask)
    
    dynamic_weights = np.zeros(len(positions), dtype=np.float32)
    dynamic_weights[far_mask] = WEIGHT_FAR
    dynamic_weights[near_mask] = WEIGHT_NEAR
    
    # 过渡区域：线性插值
    if np.any(transition_mask):
        ratio = (dists_to_goal[transition_mask] - NEAR_THRESHOLD) / (FAR_THRESHOLD - NEAR_THRESHOLD)
        dynamic_weights[transition_mask] = WEIGHT_NEAR - ratio * (WEIGHT_NEAR - WEIGHT_FAR)
    
    # === 3. 原有的净空奖励计算（保持不变） ===
    # ... 原有的距离计算、向上绕行加成等逻辑 ...
    safe_distance_reward = ...  # 原有的净空奖励值（未加权）
    
    # === 4. 应用动态权重 ===
    rewards = dynamic_weights * safe_distance_reward
    
    return rewards
```

### 方案2：在奖励计算主函数中实现

修改 `utils/vectorized_reward_calculator.py:_calculate_all_rewards_vectorized`：

```python
# 在计算净空奖励时，根据距离目标距离动态调整权重
clear_reward = self._clearance_reward_vectorized(agent, world, pos.reshape(1, -1))

# 计算到目标的距离
goal_pos = ...
dist_to_goal = np.linalg.norm(pos - goal_pos)

# 动态权重
if dist_to_goal > 50.0:
    clearance_weight = 0.5
elif dist_to_goal < 20.0:
    clearance_weight = 5.0
else:
    ratio = (dist_to_goal - 20.0) / (50.0 - 20.0)
    clearance_weight = 5.0 - ratio * (5.0 - 0.5)

# 应用动态权重
arrays['rewards'][b, a, clearance_index] = clear_reward[0] * clearance_weight
```

**推荐方案1**，因为：
1. 逻辑集中，易于维护
2. 向量化计算，性能更好
3. 不影响其他奖励项的计算

---

## 配置参数

在 `run_optimized.sh` 中添加：

```bash
# === 条件化净空奖励参数 ===
export CLEARANCE_FAR_THRESHOLD=${CLEARANCE_FAR_THRESHOLD:-50.0}  # 远距离阈值（米）
export CLEARANCE_NEAR_THRESHOLD=${CLEARANCE_NEAR_THRESHOLD:-20.0}  # 近距离阈值（米）
export CLEARANCE_WEIGHT_FAR=${CLEARANCE_WEIGHT_FAR:-0.5}  # 远距离权重（防止刷分）
export CLEARANCE_WEIGHT_NEAR=${CLEARANCE_WEIGHT_NEAR:-5.0}  # 近距离权重（防止撞地形）

# 保留原有的CLEARANCE_WEIGHT作为默认值（用于向后兼容）
export CLEARANCE_WEIGHT=${CLEARANCE_WEIGHT:-3.5}  # 默认权重（仅在无条件化时使用）
```

---

## 额外优化：分离"避障"和"保持安全距离"

### 问题

当前净空奖励同时包含：
1. **"避障"**（距离<安全距离）：应该高权重惩罚
2. **"保持安全距离"**（距离>安全距离）：应该低权重奖励

### 解决方案

```python
# 在_clearance_reward_vectorized中
if d_effective < safe_distance:
    # 避障：高权重惩罚（无论距离目标多远）
    penalty_weight = 5.0  # 固定高权重
    rewards = -penalty_weight * safe_distance_weight * (1.0 - d_effective / safe_distance)
else:
    # 保持安全距离：根据距离目标动态调整权重
    if dist_to_goal > FAR_THRESHOLD:
        reward_weight = 0.5  # 远距离：低权重
    elif dist_to_goal < NEAR_THRESHOLD:
        reward_weight = 5.0  # 近距离：高权重
    else:
        ratio = (dist_to_goal - NEAR_THRESHOLD) / (FAR_THRESHOLD - NEAR_THRESHOLD)
        reward_weight = 5.0 - ratio * (5.0 - 0.5)
    
    rewards = reward_weight * safe_distance_reward
```

---

## 效果验证

### 预期效果

1. **"仅有动作"方法**：
   - 远离目标时：净空奖励大幅降低（0.5 × 0.3 = 0.15/步）
   - 无法通过"保持安全距离"刷分
   - 必须接近目标才能获得高奖励

2. **APF方法**：
   - 接近目标时：净空奖励保持高权重（5.0 × 0.3 = 1.5/步）
   - 提供强避障引导，防止撞地形
   - 远离目标时：权重降低，不影响探索

3. **训练稳定性**：
   - 减少"刷分"行为
   - 提高到达目标的成功率
   - 保持避障能力

### 验证指标

1. **"仅有动作"方法的平均奖励**：应该从-425,574提高到接近0或正值
2. **APF方法的撞地形次数**：应该保持低水平（<10次/回合）
3. **到达目标的成功率**：应该提高（>50%）

---

## 实施步骤

1. **修改净空奖励计算函数**：添加动态权重逻辑
2. **添加配置参数**：在`run_optimized.sh`中添加条件化参数
3. **测试验证**：运行消融实验，对比修复前后的效果
4. **参数调优**：根据实际效果调整阈值和权重

---

## 注意事项

1. **向后兼容**：保留原有的`CLEARANCE_WEIGHT`参数，作为默认值
2. **性能影响**：动态权重计算是向量化的，性能影响可忽略
3. **参数敏感性**：阈值和权重需要根据实际环境调优
4. **多智能体场景**：每个智能体独立计算到各自目标的距离



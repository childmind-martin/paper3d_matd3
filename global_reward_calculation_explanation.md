# 全局奖励计算详解

## 一、函数位置

**文件**: `multiagent/scenarios/paper3d_terrain_weighted.py`  
**函数**: `_calculate_global_reward(self, world)`  
**行数**: 1056-1082

---

## 二、计算流程

### 2.1 数据收集阶段

```python
def _calculate_global_reward(self, world):
    agents = getattr(world, 'agents', [])
    dists = []          # 所有智能体到目标的距离
    progresses = []     # 所有智能体的进展（距离减少量）
    successes = []      # 所有智能体的成功标志（0或1）
    
    for ag in agents:
        if hasattr(ag, 'goal_a') and ag.goal_a.state.p_pos is not None:
            # 1. 计算当前距离
            d = np.linalg.norm(ag.state.p_pos - ag.goal_a.state.p_pos)
            dists.append(d)
            
            # 2. 计算进展（距离减少量）
            if hasattr(ag, 'last_goal_dist'):
                progress = max(0.0, ag.last_goal_dist - d)  # 只取正值
                progresses.append(progress)
            
            # 3. 判断是否成功
            success = 1.0 if d <= success_distance_threshold else 0.0
            successes.append(success)
```

**关键变量说明**:
- `d`: 当前智能体到目标的距离
- `ag.last_goal_dist`: 上一步智能体到目标的距离（在`_calculate_approach_reward`中更新）
- `progress`: 距离减少量 = `last_goal_dist - d`（只取正值，即只奖励接近）
- `success`: 如果距离 <= 成功阈值（默认5.0米），则为1.0，否则为0.0

---

## 三、三种计算模式

### 3.1 avg_progress 模式（默认，最危险）⚠️⚠️⚠️

**当前配置**: `GLOBAL_REWARD_MODE=success_rate`（已修改）  
**但代码默认**: `'avg_progress'`（如果未设置环境变量）

**计算公式**:
```python
if mode == 'avg_progress' and progresses:
    return float(np.mean(progresses)) * 10.0
```

**详细计算**:
1. 计算每个智能体的进展: `progress = max(0.0, last_goal_dist - current_dist)`
2. 计算平均进展: `avg_progress = np.mean(progresses)`
3. 乘以系数: `reward = avg_progress * 10.0`
4. 应用权重: `final_reward = reward * GLOBAL_WEIGHT (5.0)`

**示例计算**:
假设3个智能体，每步进展分别为：
- Agent 0: 0.5米（从100米到99.5米）
- Agent 1: 0.3米（从80米到79.7米）
- Agent 2: 0.2米（从120米到119.8米）

```
avg_progress = (0.5 + 0.3 + 0.2) / 3 = 0.333米
reward = 0.333 * 10.0 = 3.33
final_reward = 3.33 * 5.0 = 16.65（每步）
```

**2800步累积**: 16.65 × 2800 = **46,620**

**问题**:
- ✅ 每步都计算，即使智能体不接近目标也能获得奖励（只要有任何进展）
- ✅ 智能体可以通过"持续接近但不到达"来刷分
- ✅ 即使所有智能体都失败，只要持续有进展就能获得大量奖励

---

### 3.2 success_rate 模式（推荐）✅

**当前配置**: `GLOBAL_REWARD_MODE=success_rate`

**计算公式**:
```python
if mode == 'success_rate' and successes:
    return float(np.mean(successes)) * float(self.success_reward_value)
```

**详细计算**:
1. 判断每个智能体是否成功: `success = 1.0 if dist <= threshold else 0.0`
2. 计算成功率: `success_rate = np.mean(successes)`（0.0到1.0之间）
3. 乘以成功奖励值: `reward = success_rate * success_reward_value (3000.0)`
4. 应用权重: `final_reward = reward * GLOBAL_WEIGHT (5.0)`

**示例计算**:
假设3个智能体：
- Agent 0: 距离=6米（未成功，success=0.0）
- Agent 1: 距离=4米（成功，success=1.0）
- Agent 2: 距离=3米（成功，success=1.0）

```
success_rate = (0.0 + 1.0 + 1.0) / 3 = 0.667
reward = 0.667 * 3000.0 = 2000.0
final_reward = 2000.0 * 5.0 = 10,000.0（每步）
```

**关键特性**:
- ✅ 只有智能体在目标附近（距离<=5米）时才能获得奖励
- ✅ 所有智能体都成功时，成功率=1.0，奖励最大
- ✅ 部分智能体成功时，奖励按比例减少
- ✅ 所有智能体都失败时，成功率=0.0，奖励=0

**2800步累积**:
- 如果所有智能体始终在目标附近: 10,000 × 2800 = **28,000,000**（但实际不会发生，因为成功率极低）
- 如果所有智能体都失败: 0 × 2800 = **0**

**问题**:
- ⚠️ 如果智能体到达目标后继续在目标附近徘徊，会持续获得奖励
- ⚠️ 但相比`avg_progress`模式，这个模式更合理，因为至少要求智能体接近目标

---

### 3.3 min_distance 模式

**计算公式**:
```python
if mode == 'min_distance':
    return float(-np.min(dists))  # 越小越好
```

**详细计算**:
1. 找到所有智能体到目标的最小距离: `min_dist = np.min(dists)`
2. 取负号（因为距离越小越好）: `reward = -min_dist`
3. 应用权重: `final_reward = reward * GLOBAL_WEIGHT (5.0)`

**示例计算**:
假设3个智能体到目标的距离：
- Agent 0: 100米
- Agent 1: 80米
- Agent 2: 120米

```
min_dist = min(100, 80, 120) = 80米
reward = -80.0
final_reward = -80.0 * 5.0 = -400.0（每步）
```

**关键特性**:
- ✅ 奖励与最小距离成反比（距离越小，奖励越大）
- ✅ 只关注最近的智能体，不关注其他智能体
- ⚠️ 可能导致部分智能体到达后，其他智能体不再努力

**2800步累积**: 
- 如果最小距离始终为80米: -400.0 × 2800 = **-1,120,000**（负奖励！）
- 如果最小距离逐渐减少到5米: 从-400逐渐增加到-25，平均约-200，累积约 **-560,000**

---

## 四、关键发现

### 4.1 当前配置分析

根据`run_optimized.sh`第484行：
```bash
export GLOBAL_REWARD_MODE=${GLOBAL_REWARD_MODE:-success_rate}
```

**当前使用的是`success_rate`模式**，但训练数据中成功率极低（0%），说明：
1. 全局奖励在`success_rate`模式下几乎为0（因为成功率=0）
2. 但实际奖励值很高（400万-540万），说明**其他奖励项**才是主要贡献者
3. 可能之前的训练使用了`avg_progress`模式，导致累积了大量奖励

### 4.2 代码默认值问题

在`_calculate_global_reward`函数中：
```python
mode = getattr(self, 'global_reward_mode', 'avg_progress')  # 默认是avg_progress！
```

**问题**: 如果环境变量`GLOBAL_REWARD_MODE`未正确传递，代码会回退到`avg_progress`模式，这是最危险的模式！

### 4.3 每步计算机制

**关键**: 全局奖励在**每个训练步**都会计算，这意味着：
- 每回合2800步，全局奖励会累积2800次
- 即使单步奖励很小，累积后也会很大

**示例**（avg_progress模式）:
- 单步奖励: 16.65
- 2800步累积: 16.65 × 2800 = 46,620
- 加上环境缩放（×2.0）: 46,620 × 2.0 = **93,240**

---

## 五、改进建议

### 5.1 确保使用success_rate模式

```bash
# 在run_optimized.sh中明确设置
export GLOBAL_REWARD_MODE=success_rate
```

### 5.2 修改代码默认值

在`paper3d_terrain_weighted.py`中修改：
```python
# 从
mode = getattr(self, 'global_reward_mode', 'avg_progress')

# 改为
mode = getattr(self, 'global_reward_mode', 'success_rate')  # 更安全的默认值
```

### 5.3 降低全局奖励权重

即使使用`success_rate`模式，如果智能体到达目标后继续徘徊，仍会获得持续奖励。建议：
```bash
export GLOBAL_WEIGHT=2.0  # 从5.0降低到2.0
```

### 5.4 添加奖励衰减机制

对于到达目标的智能体，可以添加奖励衰减：
- 第一次到达: 给全额奖励
- 后续在目标附近: 奖励逐渐衰减（如每10步衰减10%）

---

## 六、总结

### 6.1 三种模式对比

| 模式 | 每步奖励范围 | 2800步累积（估算） | 问题 |
|------|------------|------------------|------|
| **avg_progress** | 0-50 | **140,000** | ⚠️⚠️⚠️ 最危险，持续有进展就能刷分 |
| **success_rate** | 0-15,000 | **0-42,000,000** | ⚠️ 到达目标后持续奖励 |
| **min_distance** | -400到-25 | **-560,000** | ⚠️ 负奖励，可能导致训练困难 |

### 6.2 当前状态

- **配置**: `GLOBAL_REWARD_MODE=success_rate`（已设置）
- **实际效果**: 由于成功率=0%，全局奖励几乎为0
- **主要问题**: 其他奖励项（净空、高度、接近等）才是主要累积来源

### 6.3 建议

1. **保持`success_rate`模式**（当前配置正确）
2. **降低全局奖励权重**（从5.0降到2.0）
3. **重点修复其他累积项**（净空、高度、接近奖励）

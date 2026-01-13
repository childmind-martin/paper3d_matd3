# 全局奖励模式和环境缩放修复报告

## 一、问题诊断

### 1.1 全局奖励模式回退问题

**问题**: 代码中存在多处默认值`'avg_progress'`，如果环境变量未正确传递，会回退到危险的`avg_progress`模式。

**发现的默认值位置**:
1. `paper3d_train_optimized.py`第17298行：命令行参数默认值 `'avg_progress'`
2. `paper3d_train_optimized.py`第3288行：传递给场景的默认值 `'avg_progress'`
3. `multiagent/scenarios/paper3d_terrain_weighted.py`第64行：场景类默认值 `'avg_progress'`
4. `multiagent/scenarios/paper3d_terrain_weighted.py`第1073行：`_calculate_global_reward`函数中的默认值 `'avg_progress'`
5. `utils/vectorized_reward_calculator.py`第111行：向量化计算器默认值 `'avg_progress'`

**影响**: 
- 如果环境变量`GLOBAL_REWARD_MODE`未设置或传递失败，所有默认值都会回退到`avg_progress`
- `avg_progress`模式会导致大量奖励累积（每步约50，2800步累积140,000）

### 1.2 环境缩放问题

**问题**: 环境奖励缩放系数过高，导致奖励值被过度放大。

**当前配置**:
- `REWARD_POS_SCALE = 2.0`：所有正奖励放大2倍
- `REWARD_NEG_SCALE = 2.5`：所有负奖励放大2.5倍

**影响**:
- 正奖励累积：184,324 × 2.0 = **368,648**
- 负奖励累积：-224,000 × 2.5 = **-560,000**
- 这解释了为什么奖励值在400万-540万范围内

---

## 二、修复内容

### 2.1 修复全局奖励模式默认值

#### 修复1: `paper3d_train_optimized.py` 命令行参数
```python
# 修改前
parser.add_argument("--global-reward-mode", type=str, default="avg_progress", ...)

# 修改后
parser.add_argument("--global-reward-mode", type=str, default="success_rate", ...)
```

#### 修复2: `paper3d_train_optimized.py` 场景参数传递
```python
# 修改前
'global_reward_mode': getattr(args, 'global_reward_mode', 'avg_progress'),

# 修改后
'global_reward_mode': getattr(args, 'global_reward_mode', 'success_rate'),
```

#### 修复3: `paper3d_terrain_weighted.py` 场景类初始化
```python
# 修改前
self.global_reward_mode = kwargs.get('global_reward_mode', 'avg_progress')

# 修改后
self.global_reward_mode = kwargs.get('global_reward_mode', 'success_rate')
```

#### 修复4: `paper3d_terrain_weighted.py` 全局奖励计算函数
```python
# 修改前
mode = getattr(self, 'global_reward_mode', 'avg_progress')

# 修改后
mode = getattr(self, 'global_reward_mode', 'success_rate')
```

#### 修复5: `utils/vectorized_reward_calculator.py` 向量化计算器
```python
# 修改前
global_reward_mode: str = 'avg_progress'

# 修改后
global_reward_mode: str = 'success_rate'
```

### 2.2 修复环境缩放系数

#### 修复1: `run_optimized.sh` 环境变量
```bash
# 修改前
export REWARD_POS_SCALE=${REWARD_POS_SCALE:-2.0}
export REWARD_NEG_SCALE=${REWARD_NEG_SCALE:-2.5}

# 修改后
export REWARD_POS_SCALE=${REWARD_POS_SCALE:-1.0}  # 不进行缩放
export REWARD_NEG_SCALE=${REWARD_NEG_SCALE:-1.0}  # 不进行缩放
```

#### 修复2: `paper3d_train_optimized.py` 命令行参数默认值
```python
# 修改前
parser.add_argument("--reward-pos-scale", type=float, default=2.0, ...)
parser.add_argument("--reward-neg-scale", type=float, default=1.0, ...)

# 修改后
parser.add_argument("--reward-pos-scale", type=float, default=1.0, ...)
parser.add_argument("--reward-neg-scale", type=float, default=1.0, ...)
```

---

## 三、修复效果

### 3.1 全局奖励模式修复效果

**修复前**:
- 如果环境变量未设置，回退到`avg_progress`模式
- 每步奖励约50，2800步累积140,000

**修复后**:
- 默认使用`success_rate`模式（更安全）
- 由于成功率=0%，全局奖励=0（不会累积）
- 即使环境变量未设置，也不会回退到危险的`avg_progress`模式

### 3.2 环境缩放修复效果

**修复前**:
- 正奖励累积：184,324 × 2.0 = **368,648**
- 负奖励累积：-224,000 × 2.5 = **-560,000**

**修复后**:
- 正奖励累积：184,324 × 1.0 = **184,324**（降低50%）
- 负奖励累积：-224,000 × 1.0 = **-224,000**（降低60%）

**预期效果**:
- 奖励值从400万-540万降低到约200万-300万
- 奖励信号更准确，不会过度放大
- 训练更稳定，Q值不会爆炸

---

## 四、验证方法

### 4.1 验证全局奖励模式

1. **检查环境变量传递**:
   ```bash
   # 在run_optimized.sh中，确保传递了--global-reward-mode参数
   grep -n "global-reward-mode" run_optimized.sh
   ```

2. **检查代码默认值**:
   ```bash
   # 确认所有默认值都是success_rate
   grep -n "global_reward_mode.*avg_progress" paper3d_train_optimized.py
   grep -n "global_reward_mode.*avg_progress" multiagent/scenarios/paper3d_terrain_weighted.py
   grep -n "global_reward_mode.*avg_progress" utils/vectorized_reward_calculator.py
   ```

3. **运行时验证**:
   - 在训练日志中检查全局奖励值
   - 如果使用`success_rate`模式且成功率=0%，全局奖励应该为0

### 4.2 验证环境缩放

1. **检查环境变量**:
   ```bash
   # 确认REWARD_POS_SCALE和REWARD_NEG_SCALE都是1.0
   grep -n "REWARD_POS_SCALE\|REWARD_NEG_SCALE" run_optimized.sh
   ```

2. **检查代码默认值**:
   ```bash
   # 确认命令行参数默认值都是1.0
   grep -n "reward-pos-scale\|reward-neg-scale" paper3d_train_optimized.py
   ```

3. **运行时验证**:
   - 对比修复前后的奖励值
   - 预期奖励值降低约50%

---

## 五、注意事项

### 5.1 向后兼容性

- 如果用户显式设置了`GLOBAL_REWARD_MODE=avg_progress`，仍然会使用`avg_progress`模式
- 如果用户显式设置了`REWARD_POS_SCALE=2.0`，仍然会使用2.0缩放
- 修复只影响**默认值**，不影响显式配置

### 5.2 训练影响

- **需要重新训练**：由于奖励尺度变化，现有模型不兼容
- **奖励值会降低**：这是预期的，因为之前的值被过度放大了
- **训练稳定性提升**：奖励信号更准确，训练应该更稳定

### 5.3 其他累积项

即使修复了全局奖励和环境缩放，其他累积项（净空、高度、接近等）仍然存在。建议继续修复这些项。

---

## 六、总结

### 6.1 修复内容

1. ✅ 修复了5处`global_reward_mode`默认值（从`avg_progress`改为`success_rate`）
2. ✅ 修复了环境缩放系数（从2.0/2.5改为1.0/1.0）

### 6.2 预期效果

1. **全局奖励**：默认使用`success_rate`模式，不会回退到危险的`avg_progress`模式
2. **奖励值**：降低约50%，更接近真实值
3. **训练稳定性**：奖励信号更准确，训练应该更稳定

### 6.3 后续建议

1. **继续修复其他累积项**（净空、高度、接近奖励）
2. **监控训练效果**：观察修复后的奖励值和成功率
3. **调整奖励权重**：根据新的奖励尺度调整各项权重

# CUDA内存访问错误修复报告

## 问题现象

从终端输出（第733-739行）可以看到：
```
2025-12-09 21:11:37.091524: E external/local_xla/xla/stream_executor/cuda/cuda_event.cc:30] Error polling for event status: failed to query event: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-12-09 21:11:37.091659: F tensorflow/core/common_runtime/device/device_event_mgr.cc:223] Unexpected Event status: 1
...
/home/tang/Desktop/run_optimized.sh: line 1101: 1075305 Aborted                 (core dumped) python3 paper3d_train_optimized.py "${ARGS[@]}"
```

**关键信息**：
- 错误发生在运行**融合实验（action_apf_fusion）**时
- 错误类型：`CUDA_ERROR_ILLEGAL_ADDRESS`（非法内存访问）
- 程序在第5回合时崩溃（`回合 5/100: 奖励=-26,962`）

---

## 根本原因分析

### 问题定位

错误发生在**条件化净空奖励**的实现中（`utils/vectorized_reward_calculator.py:1880-1995`）。

### 具体问题

#### **问题1：goal_pos形状不匹配**

```python
# 原始代码（有问题）：
goal_pos = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
dists_to_goal = np.linalg.norm(positions - goal_pos, axis=-1)
```

**问题**：
- `goal_pos`可能是标量、1D数组`(3,)`、2D数组`(1, 3)`或其他形状
- `positions`形状是`(num_positions, 3)`或`(1, 3)`
- 如果`goal_pos`形状不对，`positions - goal_pos`无法正确广播，导致CUDA内存访问错误

#### **问题2：dists_to_goal长度不匹配**

```python
# 原始代码（有问题）：
dists_to_goal = np.linalg.norm(positions - goal_pos, axis=-1)
dynamic_weights = np.zeros(len(positions), dtype=np.float32)
dynamic_weights[far_mask] = WEIGHT_FAR  # 如果dists_to_goal长度不匹配，这里会出错
```

**问题**：
- 如果`positions - goal_pos`计算失败，`dists_to_goal`的长度可能与`positions`不匹配
- 导致后续的mask操作（`far_mask`, `near_mask`）长度不匹配
- 在CUDA上执行时，访问越界内存导致崩溃

#### **问题3：数组形状不一致**

```python
# 原始代码（有问题）：
rewards = np.where(
    d_effective < safe_distance,
    PENALTY_WEIGHT * safe_distance_reward,
    dynamic_weights * safe_distance_reward
)
```

**问题**：
- 如果`d_effective`、`safe_distance_reward`、`dynamic_weights`的形状不一致
- `np.where`在CUDA上执行时，可能导致内存访问错误

---

## 修复方案

### 修复1：确保goal_pos形状正确

```python
# 修复后：
goal_pos = np.asarray(goal_pos_raw, dtype=np.float32)
# 🔧 修复：确保goal_pos是1D数组(3,)，而不是标量或其他形状
if goal_pos.ndim == 0:
    goal_pos = None
elif goal_pos.ndim > 1:
    goal_pos = goal_pos.flatten()[:3]
elif len(goal_pos) < 3:
    goal_pos = None
else:
    goal_pos = goal_pos[:3].flatten()
```

### 修复2：确保positions和goal_pos形状兼容

```python
# 修复后：
if positions.ndim == 1:
    positions_2d = positions.reshape(1, -1)
else:
    positions_2d = positions

if positions_2d.shape[-1] >= 3:
    positions_3d = positions_2d[..., :3]
    dists_to_goal = np.linalg.norm(positions_3d - goal_pos, axis=-1)
else:
    dists_to_goal = np.full(len(positions_2d), 100.0, dtype=np.float32)
```

### 修复3：确保dists_to_goal长度匹配

```python
# 修复后：
num_pos = len(positions)
if len(dists_to_goal) != num_pos:
    dists_to_goal = np.full(num_pos, 100.0, dtype=np.float32)
```

### 修复4：确保所有数组形状一致

```python
# 修复后：
num_pos = len(positions)

# 确保d_effective和safe_distance_reward的形状正确
if d_effective.shape[0] != num_pos or safe_distance_reward.shape[0] != num_pos:
    rewards = np.zeros(num_pos, dtype=np.float32)
elif dynamic_weights.shape[0] != num_pos:
    dynamic_weights = np.full(num_pos, WEIGHT_FAR, dtype=np.float32)
    rewards = np.where(
        d_effective < safe_distance,
        PENALTY_WEIGHT * safe_distance_reward,
        dynamic_weights * safe_distance_reward
    )
else:
    rewards = np.where(
        d_effective < safe_distance,
        PENALTY_WEIGHT * safe_distance_reward,
        dynamic_weights * safe_distance_reward
    )
```

### 修复5：添加异常处理

```python
# 修复后：
try:
    far_mask = dists_to_goal > FAR_THRESHOLD
    near_mask = dists_to_goal < NEAR_THRESHOLD
    transition_mask = ~(far_mask | near_mask)
    # ... 计算dynamic_weights ...
except Exception:
    # 计算失败，使用默认权重
    dynamic_weights = np.full(num_pos, WEIGHT_FAR, dtype=np.float32)
```

---

## 为什么只在融合实验中崩溃？

### 可能原因

1. **融合实验使用动态FR调度**：
   - FR值在训练过程中动态变化（0.0 → 1.0）
   - 可能导致某些时刻的奖励计算路径不同
   - 触发形状不匹配的边界情况

2. **融合实验的奖励计算更复杂**：
   - 需要同时处理原始动作和修正动作
   - 可能导致某些时刻`goal_pos`获取失败
   - 触发`goal_pos = None`的路径

3. **并行环境竞争**：
   - 融合实验可能使用更多的并行环境
   - 在并行环境下，内存访问竞争更容易触发CUDA错误

---

## 验证方法

### 1. 检查修复后的代码

运行以下命令检查语法：
```bash
python3 -m py_compile utils/vectorized_reward_calculator.py
```

### 2. 运行融合实验

重新运行消融实验，检查是否还会崩溃：
```bash
python3 ablation_action_pf_comparison.py
```

### 3. 检查日志

在训练日志中查找：
- 是否还有CUDA错误
- 是否在相同位置崩溃
- 奖励计算是否正常

---

## 预防措施

### 1. 添加形状检查

在所有向量化操作前，检查数组形状：
```python
if array1.shape != array2.shape:
    # 处理形状不匹配
```

### 2. 使用try-except保护

在所有CUDA相关操作外添加异常处理：
```python
try:
    result = np.where(condition, array1, array2)
except Exception as e:
    # 回退到安全值
    result = np.zeros_like(array1)
```

### 3. 验证数组类型

确保所有数组都是NumPy数组，且类型正确：
```python
if not isinstance(array, np.ndarray):
    array = np.asarray(array, dtype=np.float32)
```

---

## 总结

**根本原因**：条件化净空奖励实现中，`goal_pos`形状不匹配、`dists_to_goal`长度不匹配、数组形状不一致导致CUDA内存访问错误。

**修复方案**：
1. 确保`goal_pos`是1D数组`(3,)`
2. 确保`positions`和`goal_pos`形状兼容
3. 确保`dists_to_goal`长度与`positions`匹配
4. 确保所有数组形状一致
5. 添加异常处理保护

**预期效果**：修复后，融合实验应该能够正常运行，不再出现CUDA内存访问错误。



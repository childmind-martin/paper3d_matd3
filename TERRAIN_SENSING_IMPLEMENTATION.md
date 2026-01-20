# 地形感知模式（Local vs Oracle）实现方案

## 概述

实现地形感知模式的消融实验，对比使用观测中的地形信息（local）vs 使用Oracle接口获取真值（oracle）对APF性能的影响。

## 实现步骤

### 1. 场景类增强（已完成）
- ✅ 在 `paper3d_terrain_energy.py` 中添加 `get_terrain_grad(x, y, dx=1.0, dy=1.0)` 方法
- ✅ 使用有限差分计算地形梯度

### 2. 训练脚本参数添加（已完成）
- ✅ 在 `paper3d_train_optimized.py` 中添加 `--terrain-sensing-mode` 参数
- ✅ 支持三种模式：`local`, `oracle_same_probes`, `oracle_dense`

### 3. OptimizedMADDPG 类修改（待实现）

#### 3.1 存储场景引用和感知模式
```python
def __init__(self, n_agents, obs_shapes, action_dims, args, env=None):
    # ... 现有代码 ...
    self.terrain_sensing_mode = getattr(args, 'terrain_sensing_mode', 'local')
    self.scenario_ref = None  # 场景引用（用于Oracle模式）
    if env is not None and hasattr(env, 'scenario'):
        self.scenario_ref = env.scenario
```

#### 3.2 修改 `_calculate_terrain_forces_sphere_tf` 函数

**关键修改点：**
1. 在函数开始处检查 `terrain_sensing_mode`
2. 如果是 `local` 模式：保持现有逻辑，从 `obs` 中提取地形信息
3. 如果是 `oracle_same_probes` 或 `oracle_dense` 模式：
   - 计算 probe 的真实世界坐标
   - 使用 `tf.py_function` 调用 `get_terrain_height` 和 `get_terrain_grad`
   - 使用真值替换观测中的地形信息

**Probe 布局（与 local 模式一致）：**
- 前方探测点：8个点，距离 [2, 4, 6, 10, 15, 20, 25, 30] 米
- 周围近距探测：8个方向，距离 5 米
- 周围远距探测：8个方向，距离 12 米（oracle_dense 模式可增加密度）

**坐标计算：**
```python
# 从观测中提取智能体位置和速度方向
agent_pos_xy = agent_pos[:, 0:2]  # (batch, 2)
agent_height = agent_pos[:, 2:3]   # (batch, 1)
vel_xy = obs[:, 3:5] * max_speed   # 还原速度
vel_dir = vel_xy / (norm(vel_xy) + eps)  # 归一化方向

# 计算前方探测点坐标
forward_distances = [2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0]
for dist in forward_distances:
    probe_x = agent_pos_xy[:, 0:1] + vel_dir[:, 0:1] * dist
    probe_y = agent_pos_xy[:, 1:2] + vel_dir[:, 1:2] * dist
    # 调用 get_terrain_height(probe_x, probe_y)
```

### 4. 消融实验脚本（已完成）
- ✅ 创建 `ablation_terrain_sensing.py`
- ✅ 定义实验配置：`apf_traditional_local`, `apf_traditional_oracle_same`, `apf_traditional_oracle_dense`, `action_apf_fusion`

### 5. 评估脚本（待实现）

创建 `evaluate_terrain_sensing.py`：
- 对每种方法在相同随机种子和地图集合上运行固定数量 episodes
- 记录指标：
  - SR_team（团队成功率）
  - 穿透/碰撞次数
  - d_min 分位数
  - P(d_min≤δ)
  - 回报曲线

## 注意事项

1. **Oracle 模式仅用于评估**：Oracle 模式只影响 APF 地形力计算，不注入到 RL 训练观测中
2. **性能考虑**：Oracle 模式使用 `tf.py_function`，可能影响 XLA 编译，需要在性能测试中验证
3. **坐标系统一致性**：确保 probe 坐标计算与观测生成时的逻辑一致
4. **边界处理**：probe 坐标可能超出地图范围，需要处理边界情况

## 实现优先级

1. **高优先级**：实现 `oracle_same_probes` 模式（probe 布局与 local 一致）
2. **中优先级**：实现 `oracle_dense` 模式（提升探测密度）
3. **低优先级**：优化性能，减少 `tf.py_function` 调用开销

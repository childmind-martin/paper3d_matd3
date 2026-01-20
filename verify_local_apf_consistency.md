# Local APF一致性验证报告

## 分析结果

经过仔细分析代码，**评估时的local APF完全采用的是`paper3d_train_optimized.py`中的方式**。

## 验证要点

### 1. APF计算函数调用路径

**训练时：**
```python
# paper3d_train_optimized.py
self._apply_potential_field_correction(action, obs, force_ratio)
  → self._apply_potential_field_correction_tf(action, obs, force_ratio)
    → self._calculate_terrain_forces_sphere_tf(pos, goal_pos, k_rep, radius, obs)
```

**评估时：**
```python
# evaluate_optimized.py
self.maddpg._apply_potential_field_correction(raw_actions_tf_for_correction, processed_obs_tf, action_force_ratio)
  → self.maddpg._apply_potential_field_correction_tf(...)  # 同一个函数
    → self.maddpg._calculate_terrain_forces_sphere_tf(pos, goal_pos, k_rep, radius, obs)  # 同一个函数
```

✅ **结论：评估和训练使用完全相同的APF计算函数**

### 2. Local模式的地形信息提取逻辑

**训练时（paper3d_train_optimized.py:5822-5837）：**
```python
# Local模式：从观测中提取地形信息
terrain_start_idx = 9  # 跳过状态9
terrain_info = tf.cast(obs[:, terrain_start_idx:terrain_start_idx + 32], dtype)  # 地形信息32维

# 归一化还原
relative_height = terrain_info[:, 0:1] * tf.cast(20.0, dtype)
current_height = terrain_info[:, 1:2] * tf.cast(100.0, dtype)
terrain_gradients = terrain_info[:, 2:6] * tf.cast(10.0, dtype)
forward_terrain_heights = terrain_info[:, 6:14] * tf.cast(100.0, dtype)
surround_near_heights = terrain_info[:, 14:22] * tf.cast(100.0, dtype)
surround_far_heights = terrain_info[:, 22:30] * tf.cast(100.0, dtype)
```

**评估时：**
- 使用相同的`_calculate_terrain_forces_sphere_tf`函数
- 当`terrain_sensing_mode='local'`时，执行相同的local模式逻辑
- 从相同的观测索引（9-40）提取32维地形信息
- 使用相同的归一化还原系数（20.0, 100.0, 10.0）

✅ **结论：评估和训练使用完全相同的地形信息提取和归一化还原逻辑**

### 3. Terrain Sensing Mode设置

**评估时（evaluate_optimized.py:704-705）：**
```python
terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
maddpg_args.terrain_sensing_mode = terrain_sensing_mode
```

**Local模式检查（paper3d_train_optimized.py:5612-5613）：**
```python
terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
use_oracle = terrain_sensing_mode.startswith('oracle')
```

**评估配置（ablation_terrain_sensing.py:314）：**
```python
env["TERRAIN_SENSING_MODE"] = "local"  # 明确设置为local
```

✅ **结论：评估时正确设置为local模式，不会进入Oracle分支**

### 4. 观测处理一致性

**训练时：**
- 使用`self.obs_processor.batch_process_observations(obs_n)`处理观测
- 处理后的观测传入`_apply_potential_field_correction`

**评估时：**
- 使用`self.maddpg.obs_processor.batch_process_observations(obs_n)`处理观测
- 使用相同的`ObservationProcessor`类（从训练模型加载）
- 如果`base_obs_dim > 78`，会截取前78维（去除PF特征），确保基础观测结构一致

✅ **结论：评估和训练使用相同的观测处理器，确保观测结构一致**

### 5. APF参数一致性

**训练时：**
- APF参数（k_att, lambda_1, k_rep, radius）从Actor网络输出映射得到
- 使用`_map_actor_pf_params_tf`进行参数映射

**评估时：**
- 使用相同的Actor网络（从训练模型加载）
- 使用相同的参数映射函数`_map_actor_pf_params_tf`
- APF参数计算逻辑完全一致

✅ **结论：评估和训练使用相同的APF参数计算逻辑**

## 总结

**评估时的local APF完全采用`paper3d_train_optimized.py`中的方式**，具体体现在：

1. ✅ **相同的函数调用路径**：使用完全相同的`_apply_potential_field_correction`和`_calculate_terrain_forces_sphere_tf`函数
2. ✅ **相同的地形信息提取**：从观测索引9-40提取32维地形信息，使用相同的归一化还原系数
3. ✅ **相同的模式判断**：正确设置为local模式，不会进入Oracle分支
4. ✅ **相同的观测处理**：使用相同的`ObservationProcessor`类处理观测
5. ✅ **相同的APF参数计算**：使用相同的Actor网络和参数映射函数

因此，**评估时的local APF与训练时完全一致**，确保了评估结果的可靠性和可比性。

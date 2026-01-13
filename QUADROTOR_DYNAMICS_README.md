# 四旋翼无人机动力学模型

## 概述

本实现将原有的质点运动模型重构为四旋翼无人机动力学模型，同时保持高层接口不变。外部策略/控制仍然输出世界坐标系下的期望加速度 `a_cmd = (ax, ay, az)`，但环境内部通过四旋翼动力学逆解成电机输入，再用刚体6自由度模型更新状态。

## 核心特性

1. **保持接口不变**：动作仍然是世界坐标系期望加速度 `a_cmd = (ax, ay, az)`
2. **完整状态**：在位置 `pos` 和速度 `vel` 基础上，增加姿态（四元数）和角速度状态
3. **理想姿态跟踪**：假设姿态能瞬时达到期望值（过渡阶段）
4. **电机转速输出**：每步输出4个电机转速，便于后续引入真实控制器和噪声模型

## 状态变量

### 新增状态（AgentState）

- `orientation`: 姿态四元数 `[w, x, y, z]`
- `angular_vel`: 机体系角速度 `[wx, wy, wz]` (rad/s)
- `p_acc`: 线加速度（用于观测）
- `motor_speeds`: 电机转速 `[omega1, omega2, omega3, omega4]` (rad/s)

### 原有状态（保持不变）

- `p_pos`: 位置 `[x, y, z]`
- `p_vel`: 速度 `[vx, vy, vz]`

## 动力学流程

在每个仿真步 `step(dt)` 中，执行以下流程：

1. **加速度逆解**：从期望加速度 `a_cmd` 反推期望姿态和总推力
   - 计算期望升力方向：`h = g*e3 - a_cmd`，`zB_des = h / ||h||`
   - 总推力：`f = m * ||h||`

2. **构造期望姿态**：使用期望偏航角 `psi_cmd` 和 `zB_des` 构造旋转矩阵 `R_des`

3. **理想姿态跟踪**：直接设置姿态为 `R_des`，角速度为0

4. **电机分配**：通过4×4分配矩阵将总推力和力矩转换为4个电机转速

5. **状态更新**：使用6自由度平动动力学更新位置和速度

## 使用方法

### 启用四旋翼动力学

在场景的 `make_world` 方法中，为每个智能体设置：

```python
import os

# 通过环境变量启用
agent.use_quadrotor_dynamics = os.getenv('USE_QUADROTOR_DYNAMICS', '0').lower() in ('1', 'true', 'yes', 'on')
```

或者在代码中直接设置：

```python
agent.use_quadrotor_dynamics = True
```

### 环境变量配置

- `USE_QUADROTOR_DYNAMICS`: 启用四旋翼动力学（默认：0/False）
- `QUADROTOR_PSI_CMD`: 期望偏航角（rad），默认0

### 示例

```bash
# 启用四旋翼动力学
export USE_QUADROTOR_DYNAMICS=1
export QUADROTOR_PSI_CMD=0.0  # 偏航角为0（可选）

# 运行训练
python paper3d_train_optimized.py
```

## 参数配置

四旋翼动力学参数在 `QuadrotorDynamics` 类中定义：

- `mass`: 无人机质量 (kg)，默认1.0
- `arm_length`: 螺旋桨臂长 (m)，默认0.25
- `thrust_coeff`: 推力系数 kf (N/(rad/s)^2)，默认1.0e-6
- `torque_coeff`: 反扭矩系数 km (N·m/(rad/s)^2)，默认1.0e-7
- `max_motor_speed`: 最大电机转速 (rad/s)，默认1000.0
- `g`: 重力加速度 (m/s^2)，默认9.81
- `inertia`: 惯性矩 `[Ixx, Iyy, Izz]` (kg·m^2)，默认[0.01, 0.01, 0.02]

## 向后兼容

- 默认情况下，系统使用原有的质点模型（`use_quadrotor_dynamics=False`）
- 所有可视化输出完全兼容（只依赖 `p_pos`）
- 观察空间兼容（`p_acc` 字段已存在，仅在启用动力学时更新）

## 文件结构

- `multiagent/quadrotor_dynamics.py`: 四旋翼动力学核心实现
- `multiagent/core.py`: 集成动力学模型到 `World.integrate_state`

## 后续扩展

1. **真实姿态控制器**：替换理想姿态跟踪为PD/PID控制器
2. **完整转动动力学**：实现 `J * domega/dt + omega × (J*omega) = tau`
3. **电机噪声模型**：在电机转速上添加噪声
4. **姿态响应延迟**：模拟真实姿态跟踪的延迟特性

## 注意事项

1. 在理想姿态跟踪假设下，飞行轨迹应尽可能接近原本质点模型
2. 电机转速输出可用于后续引入真实控制器和噪声模型
3. 姿态和角速度状态可用于扩展观察空间（未来功能）


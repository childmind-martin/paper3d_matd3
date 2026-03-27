# Reward Component Archive

本文件用于统一保存“计划移出主 reward 或从主 dense reward 中降级”的分项的当前函数形式、默认权重、依赖状态与代码位置，便于后续恢复、做消融或单独复用。

当前代码的主训练 reward 已经不是“16 通道全部参与主 dense 求和”的旧结构了。默认情况下，向量化主路径启用了 reward structure 收缩，只保留少数 dense 核心项参与主优化，见 [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L326)。

当前主 dense 核心通道为：

- `distance`（0，内部承载 merged progress）
- `stationary`（2）
- `height`（8）
- `success`（9）
- `collision`（10）
- `global`（11，内部承载 team-sync dense）
- `clearance`（13）

本归档主要覆盖两类对象：

1. 准备从主 reward 中移出的旧 shaping 项
2. 准备合并/重构的核心项（当前主要是 `distance` 与 `approach`）

说明：

- “归档”不代表立即删除代码，只表示后续不应再无控制地继续叠加到主 reward。
- 本文档记录的是当前代码中的实际实现，不做理想化改写。
- 默认权重均来自当前脚本 [run_optimized.sh](/home/tang/matd3/run_optimized.sh)。

## Summary

| Name | Reward Index | Default Weight | Current Status | Planned Handling |
| --- | ---: | ---: | --- | --- |
| `distance` | 0 | `0.60` | 已并入主 dense `progress` 通道 | 作为 merged progress 的状态锚点保留 |
| `approach` | 6 | `0.52` | 已并入主 dense `progress` 通道 | legacy 占位保留，不再单独主计分 |
| `exploration` | 1 | `0.00` | 已默认移出主 reward | 仅保留归档与后续恢复能力 |
| `direction` | 3 | `0.30` | 默认已不进主 dense reward | 移出主 reward，避免与 `approach/distance` 重叠 |
| `deviation` | 4 | `0.00` | 已默认移出主 reward | 仅保留归档与后续恢复能力 |
| `start_area` | 5 | `0.00` | 已默认移出主 reward | 仅保留归档与后续恢复能力 |
| `shaping` | 12 | `0.00` | 已默认移出主 reward | 仅保留归档与后续恢复能力 |
| `lateral` | 14 | `1.00` | 默认已不进主 dense reward | 移出主 reward，避免与 `clearance` 功能重叠 |
| `collision_reduction` | 15 | `0.00` | 已默认移出主 reward | 若保留应转为分析指标或 episode-level 辅助项 |

## 1. `distance`

- Reward index: `0`
- Current default weight: `0.60`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L337)
- Current active dense implementation:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L1945)
- Current standalone helper:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2400)

### Current formula

Current active fast-path formula:

```text
goal_pos = per-agent goal if available, else scenario.goal_pos
initial_dist = ||start_pos - goal_pos||
current_dist = ||current_pos - goal_pos||

denom = max(initial_dist, 1.0)
ratio = clip(current_dist / denom, 0.0, 2.0)
reward = (1.0 - ratio) * 10.0

reward = attenuate_distance_reward_near_goal(reward, current_dist)
```

Near-goal attenuation is additionally controlled by:

- `DISTANCE_REWARD_NEAR_GOAL_RADIUS`
- `DISTANCE_REWARD_NEAR_GOAL_FACTOR`
- `DISTANCE_REWARD_PROGRESS_ONLY_NEAR_GOAL`

### State dependencies

- `agent.goal_a.state.p_pos` or `scenario.goal_pos`
- `start_pos`
- `current_pos`
- `self.success_distance_threshold`
- `self.distance_reward_near_goal_radius`
- `self.distance_reward_near_goal_factor`
- `self.distance_reward_progress_only_near_goal`

### Why archived

该项提供“当前状态已经离目标更近”的全局锚点，但和 `approach` 在“朝目标推进”这条主轴上存在明显语义重叠。后续更合理的方向不是继续单独强化，而是与 `approach` 合并成一个统一的 progress 通道。

## 2. `approach`

- Reward index: `6`
- Current default weight: `0.52`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L341)
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L343)
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L344)
- Current active dense implementation:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L1971)
- Legacy helper still present:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2695)

### Current formula

Current active fast-path formula:

```text
goal_pos = per-agent goal if available, else scenario.goal_pos
prev_dist = ||prev_pos - goal_pos||
current_dist = ||current_pos - goal_pos||

approach_reward = prev_dist - current_dist

near_goal_threshold = APPROACH_NEAR_GOAL_THRESHOLD
app_max = APPROACH_NEAR_GOAL_MAX_MULT

if current_dist < near_goal_threshold:
    multiplier = 1 + (app_max - 1) * (1 - current_dist / near_goal_threshold)
else:
    multiplier = 1

multiplier = clip(multiplier, 1.0, app_max)

reward = approach_reward * 5.0 * multiplier
```

说明：

- 当前主训练实际用的是上面的 fast-path 版本
- 文件里仍保留了一个 legacy helper，其近目标倍率上限写死为 `2.0`，与当前主路径不完全一致，因此后续若归并，应以 fast-path 为准

### State dependencies

- `agent.goal_a.state.p_pos` or `scenario.goal_pos`
- `prev_pos`
- `current_pos`
- `self.approach_near_goal_threshold`
- `self.approach_near_goal_max_mult`

### Why archived

该项是“这一步是否真正朝目标推进”的结果型奖励，比 `distance` 更贴近任务推进语义；但两者继续并列存在，容易形成重复激励。后续应与 `distance` 合并，而不是继续单独堆强。

## 3. `exploration`

- Reward index: `1`
- Current default weight: `0.50`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L378)
- Current function:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2432)
- Current call site:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2218)

### Current formula

For each position:

```text
current_cell = floor(position / cell_size)

if current_cell is first visited:
    reward += 1.0                 if expl_reward_strict
    reward += 5.0                 otherwise

visit_count = previous_visit_count(current_cell) + 1

if not expl_reward_strict and visit_count <= 3:
    reward += (4 - visit_count) * 1.0

every 50th call in non-strict mode:
    reward += Uniform(0.5, 2.0)
```

### State dependencies

- `scenario.exploration_grid`
- `scenario.grid_cell_size`
- `agent.visited_cells`
- `agent.cell_visit_counts`
- `agent.random_exploration_counter`
- `self.expl_reward_strict`

### Why archived

该项更像早期 bootstrapping 补丁，不适合长期参与主 reward 排序；后续若保留，更适合做阶段 1 的短期辅助或单独分析项。

## 4. `direction`

- Reward index: `3`
- Current default weight: `0.30`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L384)
- Current function:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2535)
- Current call site:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2236)

### Current formula

This term has two parts.

#### 4.1 Goal-direction alignment reward

When speed exceeds threshold:

```text
speed = ||v||
speed_thr = 0.3

if speed > speed_thr:
    goal_dir = normalize(goal_pos - pos_now)
    vel_dir  = normalize(v)
    alignment = dot(vel_dir, goal_dir)
    dir_reward = alignment
    speed_bonus = 0.2 * speed
    base_reward = dir_reward + speed_bonus
else:
    base_reward = 0
```

#### 4.2 Height smoothness reward

```text
height_change = abs(z_now - z_prev) / max(abs(z_now), abs(z_prev), 1.0)
height_smooth_score = exp(-5.0 * height_change)
height_smooth_score is clipped into [0, 1]

total_reward = base_reward + turn_smooth_weight * height_smooth_score
```

### State dependencies

- `agent.state.p_vel`
- `agent.goal_a.state.p_pos`
- `agent.last_height`
- `self.turn_smooth_weight`

### Why archived

该项和 `approach / distance` 有明显语义重叠，同时“方向一致”会隐式偏好朝目标直冲，不一定适合需要绕障的场景。若后续保留，更适合只借用其中的平滑正则思想，而不是保留整条主 reward。

## 5. `deviation`

- Reward index: `4`
- Current default weight: `0.35`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L391)
- Current fast-path implementation:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L1958)
- Current standalone helper:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2633)

### Current formula

This term rewards staying close to the straight segment between start and goal:

```text
v = goal - start
t = clip( dot(current - start, v) / ||v||^2, 0, 1 )
proj = start + t * v
d_perp = ||current - proj||
norm_dev = clip(d_perp / max(||v||, 1), 0, 2)
reward = 1.0 - norm_dev
```

### State dependencies

- `agent.goal_a.state.p_pos` or `scenario.goal_pos`
- `start_pos`
- `current_pos`

### Why archived

该项会隐式奖励“更接近起点-目标直线”的路径，但你的任务经常需要主动绕山、绕障碍，因此它容易与真实可行路径冲突。

## 6. `start_area`

- Reward index: `5`
- Current default weight: `0.30`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L400)
- Current function:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2681)
- Current call site:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2242)

### Current formula

```text
if agent.start_position does not exist:
    agent.start_position = current_position

dist_to_start = ||position - agent.start_position||
speed = ||agent.state.p_vel||

if dist_to_start < 20 and speed > 0.1:
    reward = (20 - dist_to_start) * 0.5
else:
    reward = 0
```

### State dependencies

- `agent.start_position`
- `agent.state.p_vel`

### Why archived

该项主要是冷启动辅助，长期保留在主 reward 中会放大“离开起点”这个次要目标，不利于后期聚焦团队成功。

## 7. `shaping`

- Reward index: `12`
- Current default weight: `0.20`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L482)
- Current function:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L5407)
- Current call site:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2265)

### Current formula

Potential-based shaping with a goal-distance potential:

```text
phi_now = -0.01 * dist_to_goal

if _phi_last does not exist:
    _phi_last = phi_now
    reward = 0
else:
    reward = shaping_gamma * phi_now - _phi_last
    _phi_last = phi_now
```

### State dependencies

- `agent._phi_last`
- `self.shaping_gamma`
- `dist_to_goal`

### Why archived

该项容易形成一个额外的“隐藏奖励通道”，使得最终 episode 排序更难解释。若后续要恢复，建议只作为对照实验中的可切换项，而不是默认主 reward。

## 8. `lateral`

- Reward index: `14`
- Current default weight: `1.00`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L423)
- Current function:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L5008)
- Current call site:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2280)

### Current formula

This term activates only when the agent is moving and close to a danger source:

```text
speed = ||v||
if speed < 0.1:
    reward = 0

vel_dir = normalize(v)
activation_dist = scenario.lateral_activation_distance (default 15.0)

find nearest danger surface among:
    - spherical obstacles
    - terrain surface

danger_normal = outward normal from nearest danger source
cos_angle = dot(vel_dir, danger_normal)

if cos_angle > -0.05:
    dist_factor = 1 - min_dist / activation_dist
    reward = (0.3 + 0.7 * clip(cos_angle, 0, 1)) * clip(dist_factor, 0, 1)
else:
    reward = 0
```

### State dependencies

- `agent.state.p_vel`
- `world.landmarks`
- `landmark.state.p_pos`
- `landmark.size / landmark.radius`
- `scenario.get_terrain_height`
- `scenario.lateral_activation_distance`
- `self._estimate_terrain_normal(...)`

### Why archived

该项本质是在危险区奖励“切向绕行或远离危险源”的行为，但功能和 `clearance` 已有较强重叠，继续同时保留容易造成安全 shaping 叠加过多。

## 9. `collision_reduction`

- Reward index: `15`
- Current default weight: `0.90`
- Weight config:
  - [run_optimized.sh](/home/tang/matd3/run_optimized.sh#L372)
- Current function:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L5073)
- Current call site:
  - [utils/vectorized_reward_calculator.py](/home/tang/matd3/utils/vectorized_reward_calculator.py#L2288)

### Current formula

This term is effectively an episode-end comparison reward:

```text
current_count = agent.debug_info['total_penetration_count'] or 0

if not last step of episode:
    reward = 0

else:
    prev_count = previous_episode_collision_count

    if prev_count > 0 and current_count < prev_count:
        reward = clip((prev_count - current_count) / prev_count, 0, 1)
    elif prev_count == 0 and current_count == 0:
        reward = 0.1
    else:
        reward = 0
```

### State dependencies

- `world.current_step`
- `world.episode_length`
- `agent.debug_info['total_penetration_count']`
- `agent._collision_reduction_state`
- `agent.previous_episode_collision_count`
- `agent._last_episode_collision_count`

### Why archived

该项是跨回合比较量，不适合和单步 dense reward 混在一起。若后续保留，更适合作为：

- 课程学习解锁辅助指标
- 训练监控指标
- episode-level 辅助统计量

## Practical Note

如果后续要恢复这些分项，建议优先按以下方式回接，而不是直接重新塞回主 dense reward：

1. `distance + approach`：合并成单一 progress 通道，而不是双通道并列
2. `exploration / start_area`：仅阶段 1 使用
3. `direction / deviation / lateral`：只做消融开关，不默认开启
4. `shaping`：只做对照项
5. `collision_reduction`：改成日志或 curriculum 指标，而不是主 reward

这样可以保留可复用性，同时避免重新把主 reward 变回“语义重叠、排序不清”的结构。

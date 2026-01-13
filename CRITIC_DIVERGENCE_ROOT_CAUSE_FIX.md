# Critic发散根本原因修复方案
**日期：2025-12-01**  
**问题：奖励尺度爆炸 → Q值裁剪失效 → Critic Loss发散**

## 🔍 问题诊断

### 1. 数值链条分析
```
单步奖励：[-500, 100] (场景裁剪)
    ↓ × 2200步
回合奖励：[-1,100,000 ~ 220,000] (实际：~130,000)
    ↓ 配置期望：[-650 ~ 400] (差距325倍！)
    ↓ γ = 0.95累积
Q值应有范围：±250,000
    ↓ 裁剪到：±5,000 (Q_CLIP_VALUE)
实际Q值：5,000 (几乎所有样本)
    ↓ TD误差
TD误差：0 (裁剪后) vs 269,500 (实际应有)
    ↓ Huber Loss (delta=3.0)
Critic Loss：震荡在180-200 (发散)
```

### 2. 关键证据
- **训练日志**：回合奖励108,685~197,995，平均130,000
- **配置文件**：MAX_REWARD=400，期望与实际差距325倍
- **Critic Loss**：持续180-200，从未收敛
- **最佳回合**：仅在回合4出现（197,995），之后再无突破

## 🎯 三种修复方案对比

### 方案A：奖励归一化（推荐⭐⭐⭐⭐⭐）
**核心思想**：将回合奖励缩放到合理范围，保持Q值可学习

#### A1. 场景层归一化（最简单）
```python
# multiagent/scenarios/paper3d_terrain_energy.py
# line 2487修改为：
EPISODE_LENGTH = float(os.getenv('EPISODE_LENGTH', '2200'))
rew_scaled = rew / (EPISODE_LENGTH / 100.0)  # 缩放到100步等效
rew_final = np.clip(rew_scaled, -5.0, 1.0)  # 单步[-5, 1]
# 回合累积后：[-11,000 ~ 2,200]
```

**优点**：
- 简单，只改1行代码
- 奖励语义不变（比例缩放）
- 适配现有网络架构

**缺点**：
- 需要重新训练
- 旧模型不兼容

#### A2. TD目标归一化（最稳定）
```python
# paper3d_train_optimized.py
# line 5525修改为：
REWARD_SCALE = 1.0 / 2200.0  # 每步奖励贡献权重
target_q = (rewards * REWARD_SCALE) + gamma * target_q_next
# 等效Q值范围：±250 (可学习)
```

**优点**：
- 不影响场景奖励设计
- Q值在合理范围内
- 训练稳定性最好

**缺点**：
- 需同步修改训练和评估代码

### 方案B：增大Q裁剪阈值（不推荐❌）
```bash
# run_optimized.sh
Q_CLIP_VALUE=500000  # 从3000提高到500k
HUBER_DELTA=50000    # 从3.0提高到50k
```

**问题**：
- Q值数值不稳定（±50万）
- 梯度爆炸风险极高
- 网络权重容易饱和
- 只是掩盖问题，未解决根源

### 方案C：混合策略（折中）
```bash
# 1. 适度提高裁剪阈值
Q_CLIP_VALUE=50000   # 提高10倍
HUBER_DELTA=300      # 提高100倍

# 2. 降低学习率
LEARNING_RATE_CRITIC=0.00001  # 降低10倍

# 3. 增强正则化
CRITIC_Q_REG=0.1  # 提高5倍
```

**问题**：
- 学习速度极慢
- 参数难以调优
- 仍未解决尺度不匹配

## ✅ 推荐实施方案

### 第一阶段：立即修复（A2方案）

#### 1. 修改TD目标计算
```python
# paper3d_train_optimized.py
# 搜索所有 "target_q = rewards" 并修改为：

# Line 5525 (MADDPG版本)
REWARD_SCALE = 1.0 / 2200.0  # 添加在类初始化
target_q = (rewards[:, tf.newaxis] * REWARD_SCALE) + gamma_val * target_q_next

# Line 7668 (MATD3版本1)
target_q = (rewards[:, tf.newaxis] * self.c_reward_scale) + gamma_val * target_q_next_min

# Line 8069 (MATD3版本2)
target_q = (rewards[:, tf.newaxis] * self.c_reward_scale) + gamma_val * target_q_next_min
```

#### 2. 缓存常量
```python
# Line 2384附近添加：
target.c_reward_scale = _tf_const('reward_scale', 1.0 / 2200.0)
```

#### 3. 调整Q裁剪阈值
```bash
# run_optimized.sh line 582
export Q_CLIP_VALUE=${Q_CLIP_VALUE:-500.0}  # 缩放后合理范围

# line 90
export HUBER_DELTA=${HUBER_DELTA:-10.0}  # 匹配缩放后的TD误差
```

#### 4. 调整Critic学习率
```bash
# run_optimized.sh line 87
export LEARNING_RATE_CRITIC=${LEARNING_RATE_CRITIC:-0.0003}  # 提高3倍
```

### 第二阶段：长期优化（A1方案）

场景层归一化，用于新训练实验：
```python
# multiagent/scenarios/paper3d_terrain_energy.py
def reward(self, agent, world):
    # ... 现有奖励计算 ...
    
    # line 2487修改为：
    EPISODE_LENGTH = float(os.getenv('EPISODE_LENGTH', '2200'))
    REWARD_SCALE = 100.0 / EPISODE_LENGTH  # 归一化到100步等效
    rew_scaled = rew * REWARD_SCALE
    rew_final = np.clip(rew_scaled, -5.0, 1.0)
    return rew_final
```

## 📊 预期效果

### 修复前（当前状态）
```
回合奖励：~130,000
Q值范围：±5,000 (裁剪饱和)
TD误差：0~269,500 (极端不稳定)
Critic Loss：180-200 (发散)
训练收敛：❌ 无法学习
```

### 修复后（A2方案）
```
回合奖励：~130,000 (不变)
Q值范围：±300 (REWARD_SCALE后)
TD误差：-50~50 (稳定)
Critic Loss：预期5-15 (收敛)
训练收敛：✅ 预期20-40回合内看到改善
```

## 🚀 实施步骤

1. **备份当前代码**
   ```bash
   cp paper3d_train_optimized.py paper3d_train_optimized.py.backup
   cp run_optimized.sh run_optimized.sh.backup
   ```

2. **修改训练脚本**
   - 添加REWARD_SCALE常量
   - 修改所有TD目标计算

3. **修改配置文件**
   - 调整Q_CLIP_VALUE=500
   - 调整HUBER_DELTA=10
   - 提高LEARNING_RATE_CRITIC=0.0003

4. **验证修改**
   ```bash
   # 运行1个回合测试
   ./run_optimized.sh 1 1024 "reward_scale_test"
   # 检查日志中的Q值和Critic Loss
   ```

5. **完整训练**
   ```bash
   ./run_optimized.sh 200 1024 "reward_scale_fix_exp"
   ```

## 📈 监控指标

训练前20回合应观察到：
- ✅ Critic Loss从180降到50以下
- ✅ Q值范围在[-500, 500]之间
- ✅ TD误差绝对值<100
- ✅ 奖励开始逐渐上升
- ✅ Actor Loss更稳定

如果前20回合未见改善，检查：
1. REWARD_SCALE是否正确生效（打印target_q值）
2. Q_CLIP_VALUE是否过小/过大
3. 学习率是否需要进一步调整

## ⚠️ 注意事项

1. **旧模型不兼容**：修改REWARD_SCALE后需要重新训练
2. **评估代码同步**：evaluate_optimized.py也需相同修改
3. **PER权重**：TD误差尺度变化，PER采样分布会改变
4. **Gamma影响**：REWARD_SCALE会影响折扣因子的有效强度

## 🔗 相关文件

- `paper3d_train_optimized.py`: TD目标计算
- `run_optimized.sh`: 超参数配置
- `multiagent/scenarios/paper3d_terrain_energy.py`: 奖励函数
- `evaluate_optimized.py`: 评估脚本（需同步修改）


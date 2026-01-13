# 模型加载维度不匹配修复

## 问题诊断

从终端日志发现的严重问题：

### 1. Actor网络加载失败（严重）
```
variable.shape=(78, 512), Received: value.shape=(81, 512)
actor[0] 覆盖变量: 4/16 (25.0%) ⚠️ 极低！
```

**原因**：
- 训练时：观测维度 = 81（78基础 + 3势场力）
- 评估时：网络构建使用78维（错误）

### 2. Critic网络部分失败
```
variable.shape=(234, 384), Received: value.shape=(243, 384)
critic[0] 覆盖变量: 62/64 (96.9%)
```

**原因**：
- 训练时：状态维度 = 243（81×3智能体）
- 评估时：状态维度 = 234（78×3智能体）

### 3. 影响
虽然总体加载率显示82.5%，但**Actor的第一层完全未加载**，这意味着Actor几乎是随机初始化的策略，评估结果将毫无意义！

## 根本原因

评估脚本在构建网络时，直接从环境获取观测维度（78），没有考虑训练时追加的势场力特征（+3维）。

## 修复方案

### evaluate_optimized.py

在环境初始化部分添加观测维度调整逻辑：

```python
# 获取环境信息
self.n_agents = self.env.n
base_obs_shapes = [self.env.observation_space[i].shape[0] for i in range(self.n_agents)]

# 🔧 关键修复：如果训练时使用了势场力特征，观测维度需要+3
use_pf = getattr(self.args, 'use_pf_feature', False)
if use_pf:
    self.obs_shapes = [dim + 3 for dim in base_obs_shapes]
    print(f"✅ 检测到PF特征启用，观测维度调整: {base_obs_shapes} -> {self.obs_shapes}")
else:
    self.obs_shapes = base_obs_shapes
```

### run_evaluation.sh

已正确配置参数传递：
```bash
export USE_PF_FEATURE=${USE_PF_FEATURE:-1}  # 默认启用
export USE_FR_FEATURE=${USE_FR_FEATURE:-1}  # 默认启用
```

## 验证方法

### 快速验证
```bash
source maddpg_venv/bin/activate
bash verify_model_loading.sh
```

这将：
1. 验证网络构建维度是否正确（应该是81维）
2. 快速运行1个回合，检查是否还有维度不匹配警告
3. 验证模型加载率是否达到100%

### 完整评估
```bash
source maddpg_venv/bin/activate
bash run_evaluation.sh models/调试分离梯度、无重力、无早停、预热、随机地图、高变FR低高低_exp_20251201_112141/final 3
```

## 预期结果

修复后应该看到：
- ✅ 观测维度: [81, 81, 81]（不再是78）
- ✅ 状态维度: 243（不再是234）
- ✅ Actor加载率: 100%（不再是25%）
- ✅ Critic加载率: 100%（不再是96.9%）
- ✅ 总体加载率: 100%（不再是82.5%）
- ✅ 无维度不匹配警告

## 其他已修复内容

1. **AdaptiveRewardWeighting.get_current_weights()** - 添加缺失方法
2. **元优化配置加载** - 从pf_meta_baseline.json加载
3. **MATD3算法支持** - Twin Critic架构
4. **FR特征网络构建** - Actor接受[obs, fr]双输入
5. **隐藏层维度修正** - Actor=512×3层, Critic=256×4层
6. **训练代码select_action修复** - NumPy路径的tensor转换


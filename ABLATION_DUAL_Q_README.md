# 双Q头和分离式梯度设计消融实验说明

## 实验设计

本消融实验对比以下4个实验组：

| 实验组 | 算法框架 | Twin Critic | 双Q头 | 分离式梯度 | 说明 |
|--------|---------|------------|-------|-----------|------|
| **Baseline 1: MADDPG** | MADDPG | ❌ | ❌ | ❌ | 标准基线：单Critic，单Q头 |
| **Baseline 2: MATD3-单Q** | MATD3 | ✅ | ❌ | ❌ | MATD3框架但单Q头（验证Twin Critic效果） |
| **实验1: MATD3-双Q头** | MATD3 | ✅ | ✅ | ❌ | MATD3+双Q头但统一梯度（验证双Q头架构效果） |
| **实验2: MATD3-完整** | MATD3 | ✅ | ✅ | ✅ | 当前实现（验证分离式梯度设计的贡献） |

## 实验目的

1. **MADDPG vs MATD3-单Q**：验证Twin Critic的贡献
2. **MATD3-单Q vs MATD3-双Q头**：验证双Q头架构的贡献
3. **MATD3-双Q头 vs MATD3-完整**：验证分离式梯度设计的贡献

## 代码修改需求

由于当前代码中双Q头和分离式梯度是硬编码在MATD3实现中的，需要进行以下修改：

### 1. 添加环境变量支持

在 `paper3d_train_optimized.py` 中添加对以下环境变量的支持：

- `MATD3_USE_DUAL_Q`: 控制是否使用双Q头（0=单Q头，1=双Q头）
- `MATD3_USE_SEPARATED_GRADIENT`: 控制是否使用分离式梯度（0=统一梯度，1=分离式梯度）

### 2. 修改Critic网络构建

在 `build_continuous_critic_network_matd3` 函数中：
- 如果 `MATD3_USE_DUAL_Q=0`，修改为输出单个Q值（类似MADDPG）
- 如果 `MATD3_USE_DUAL_Q=1`，保持当前双Q头实现

### 3. 修改训练逻辑

在 `train_step_matd3` 函数中：
- 如果 `MATD3_USE_SEPARATED_GRADIENT=0`，使用统一梯度（Actor损失 = -(Q1+Q2)）
- 如果 `MATD3_USE_SEPARATED_GRADIENT=1`，保持当前分离式梯度实现

## 快速开始（当前可用）

如果暂时不想修改代码，可以使用以下简化方案：

### 方案1：只对比MADDPG和MATD3

```bash
# 运行MADDPG baseline
python ablation_dual_q_separated_gradient.py --episodes 150 --batch-size 1024

# 手动修改配置，只运行MATD3-完整实验
# 在配置文件中只保留 matd3_full 配置
```

### 方案2：手动修改代码后运行

1. 按照上述"代码修改需求"修改训练脚本
2. 运行完整消融实验：
```bash
python ablation_dual_q_separated_gradient.py --episodes 150 --batch-size 1024 --parallel --gpus 0 1 2 3
```

## 配置文件说明

配置文件：`ablation_dual_q_separated_gradient.py`

关键配置项：
- `ALGORITHM`: 选择算法（"maddpg" 或 "matd3"）
- `MATD3_USE_DUAL_Q`: 控制双Q头（需要代码支持）
- `MATD3_USE_SEPARATED_GRADIENT`: 控制分离式梯度（需要代码支持）
- `SEED`: 所有实验使用相同的随机种子（252488），确保网络初始化一致

## 预期结果

- **MADDPG baseline**: 提供标准基线性能
- **MATD3-单Q**: 应该比MADDPG好（Twin Critic的贡献）
- **MATD3-双Q头**: 应该比MATD3-单Q好（双Q头架构的贡献）
- **MATD3-完整**: 应该最好（分离式梯度设计的贡献）

## 注意事项

1. 所有实验使用相同的随机种子，确保网络初始化一致
2. 所有实验使用相同的固定位置和地形，确保公平对比
3. 所有实验使用相同的训练参数（除了算法相关参数）
4. 建议使用并行运行以节省时间

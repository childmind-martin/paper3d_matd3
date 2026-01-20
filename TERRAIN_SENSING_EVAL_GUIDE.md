# 地形感知模式评估指南

## 概述

本指南说明如何运行地形感知模式（local vs oracle）的评估对比实验，并查看结果。

## 运行方式

### 方式1：完整流程（训练+评估）

如果还没有训练好的模型，运行完整流程：

```bash
python3 ablation_terrain_sensing.py \
    --episodes 120 \
    --eval-episodes 20 \
    --eval-seed 42
```

这会：
1. 使用 `local` 模式训练一次模型（`apf_learnable`，可学习APF）
2. 使用同一个训练好的模型，分别用 `local`、`oracle_same_probes`、`oracle_dense` 三种模式进行评估
3. 结果保存在 `terrain_sensing_experiments/batch_*/results/experiment_results.json`

### 方式2：仅评估模式（使用已有模型）

如果已经有训练好的模型，直接进行评估：

```bash
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/apf_learnable/best \
    --eval-episodes 20 \
    --eval-seed 42
```

**支持的模型路径格式：**
- `models/apf_learnable/best` - 最佳模型（推荐）
- `models/apf_learnable/final` - 最终模型
- `models/apf_learnable_<timestamp>/best` - 带时间戳的模型
- `models/apf_learnable/checkpoint` - 检查点目录（持续训练）

这会：
1. 跳过训练，直接使用指定的模型进行评估
2. 分别用 `local`、`oracle_same_probes`、`oracle_dense` 三种模式进行评估
3. 结果保存在 `terrain_sensing_experiments/batch_*/results/experiment_results.json`

**详细说明请参考：** `TERRAIN_SENSING_MODEL_LOADING.md`

## 参数说明

- `--episodes`: 训练回合数（默认120）
- `--eval-episodes`: 评估回合数（默认20）
- `--eval-seed`: 评估随机种子（默认42）
- `--trained-model-path`: 训练好的模型路径（用于 `--eval-only` 模式）
- `--positions-file`: 固定位置文件路径（可选，会自动生成）

## 查看结果

### 1. JSON结果文件

评估结果保存在：
```
terrain_sensing_experiments/batch_*/results/experiment_results.json
```

该文件包含：
- `training`: 训练结果
- `evaluation`: 评估结果（包含local、oracle_same_probes、oracle_dense三种模式）

### 2. 评估输出目录

每次评估的结果保存在：
```
evaluation_results/apf_learnable_local_local/
evaluation_results/apf_learnable_oracle_same_oracle_same_probes/
evaluation_results/apf_learnable_oracle_dense_oracle_dense/
```

每个目录包含：
- `evaluation_summary.json`: 评估摘要（SR_team、平均奖励等）
- `episode_results.json`: 每个回合的详细结果
- `metrics.json`: 完整指标（碰撞次数、最小净空距离等）

### 3. 快速查看结果

```bash
# 查看最新批次的结果
cat terrain_sensing_experiments/batch_*/results/experiment_results.json | python3 -m json.tool

# 查看特定评估模式的详细结果
cat evaluation_results/apf_learnable_local_local/evaluation_summary.json | python3 -m json.tool
cat evaluation_results/apf_learnable_oracle_same_oracle_same_probes/evaluation_summary.json | python3 -m json.tool
```

## 关键指标

评估会记录以下指标：

1. **SR_team**: 团队成功率（所有智能体都到达目标的比例）
2. **碰撞次数**: 每个回合的碰撞次数
3. **最小净空距离 (d_min)**: 每个回合的最小安全距离
   - 分位数：Q5%, Q10%, Q25%, Q50%, Q75%, Q90%, Q95%
   - 违反概率：P(d_min ≤ δ)，其中 δ 为碰撞阈值（默认1.5米）
4. **回报曲线**: 每个回合的奖励

## 对比分析

运行评估后，可以对比：

- **Local vs Oracle Same Probes**: 相同探测布局下，真值地形信息 vs 观测地形信息的效果
- **Oracle Same Probes vs Oracle Dense**: 相同探测布局 vs 密集探测布局的效果（上界）

## 注意事项

1. **训练模式**: 训练时始终使用 `local` 模式，Oracle模式只在评估时使用
2. **模型一致性**: 所有评估模式使用同一个训练好的模型，确保公平对比
3. **固定位置**: 所有评估使用相同的固定位置文件，确保地形和起始位置一致
4. **随机种子**: 评估时使用固定的随机种子，确保可复现性

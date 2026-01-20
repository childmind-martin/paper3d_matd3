# 地形感知实验模型加载指南

## 概述

本指南说明如何在地形感知消融实验中加载和使用预训练的权重模型。

## 模型路径格式

训练好的模型通常保存在以下位置：

```
models/
├── apf_learnable/
│   ├── best/          # 训练过程中的最佳模型（推荐）
│   ├── final/         # 训练完成后的最终模型
│   └── checkpoint/    # 检查点目录（如果使用持续训练）
└── apf_learnable_<timestamp>/
    ├── best/
    └── final/
```

## 加载方式

### 方式1：仅评估模式（推荐）

如果已经有训练好的模型，直接使用 `--eval-only` 模式进行评估：

```bash
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/apf_learnable/best \
    --eval-episodes 20 \
    --eval-seed 42
```

**支持的模型路径格式：**

1. **相对路径（从项目根目录）：**
   ```bash
   --trained-model-path models/apf_learnable/best
   --trained-model-path models/apf_learnable/final
   ```

2. **带时间戳的模型：**
   ```bash
   --trained-model-path models/apf_learnable_20260114_165908/best
   --trained-model-path models/apf_learnable_20260114_165908/final
   ```

3. **绝对路径：**
   ```bash
   --trained-model-path /home/tang/Desktop/models/apf_learnable/best
   ```

4. **检查点路径（持续训练）：**
   ```bash
   --trained-model-path models/apf_learnable/checkpoint
   ```

### 方式2：完整流程（训练+评估）

如果还没有训练好的模型，运行完整流程会自动训练并加载：

```bash
python3 ablation_terrain_sensing.py \
    --episodes 120 \
    --eval-episodes 20 \
    --eval-seed 42
```

训练完成后，脚本会自动查找：
1. `models/apf_learnable/best` （优先）
2. `models/apf_learnable/final` （如果best不存在）

## 查找可用模型

### 方法1：使用命令行查找

```bash
# 查找所有apf_learnable模型
find models -type d -name "*apf_learnable*" | grep -E "(best|final|checkpoint)"

# 列出所有模型目录
ls -d models/apf_learnable*/*/
```

### 方法2：查看训练日志

训练完成后，日志会显示模型保存位置：

```bash
# 查看最新训练日志
ls -lt logs/apf_learnable* | head -1

# 查看results.json中的模型路径
cat logs/apf_learnable_*/results.json | grep -E "(model_path|best_model)"
```

## 模型路径验证

脚本会自动验证模型路径是否存在：

```python
# 如果路径不存在，会显示错误信息
❌ 错误: 训练模型路径不存在: models/apf_learnable/best

# 并提示可用的模型路径
```

## 常见问题

### Q1: 如何指定特定的检查点？

如果使用持续训练，模型可能保存在 `checkpoint/` 目录下：

```bash
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/apf_learnable/checkpoint \
    --eval-episodes 20
```

### Q2: 如何加载其他实验的模型？

可以加载任何符合格式的模型：

```bash
# 加载action_apf_fusion模型（如果兼容）
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/action_apf_fusion/best \
    --eval-episodes 20
```

**注意：** 确保模型架构兼容（相同的obs_shape和action_dim）。

### Q3: 模型路径包含特殊字符怎么办？

如果路径包含空格或特殊字符，使用引号：

```bash
--trained-model-path "models/apf learnable/best"
```

### Q4: 如何验证模型是否正确加载？

评估脚本会在开始时显示模型路径：

```
[评估-apf_learnable_local] 开始评估: 可学习APF (Local感知)
[评估-apf_learnable_local] 地形感知模式: local
[评估-apf_learnable_local] 模型路径: models/apf_learnable/best
```

如果模型加载失败，会在评估过程中报错。

## 评估脚本内部处理

`run_evaluation.sh` 接受模型路径作为第一个参数：

```bash
./run_evaluation.sh \
    models/apf_learnable/best \    # 模型路径
    20 \                            # 评估回合数
    evaluation_results/... \        # 保存路径
    saved_positions/... \           # 位置文件
    1 \                             # 使用固定位置
    true                            # 禁用提前终止
```

`ablation_terrain_sensing.py` 会自动调用 `run_evaluation.sh` 并传递正确的参数。

## 示例

### 示例1：使用最佳模型评估

```bash
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/apf_learnable/best \
    --eval-episodes 20 \
    --eval-seed 42
```

### 示例2：使用最终模型评估

```bash
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/apf_learnable/final \
    --eval-episodes 50 \
    --eval-seed 123
```

### 示例3：使用带时间戳的模型

```bash
python3 ablation_terrain_sensing.py \
    --eval-only \
    --trained-model-path models/apf_learnable_20260114_165908/best \
    --eval-episodes 20
```

### 示例4：完整流程（训练+评估）

```bash
# 训练新模型并评估
python3 ablation_terrain_sensing.py \
    --episodes 120 \
    --eval-episodes 20 \
    --eval-seed 42
```

## 注意事项

1. **模型兼容性**：确保加载的模型与评估配置兼容（相同的obs_shape、action_dim等）
2. **路径格式**：模型路径应该指向包含模型文件的目录（如 `best/` 或 `final/`），而不是单个文件
3. **文件结构**：模型目录应包含：
   - `actor_*.h5` 或 `actor_*.ckpt` 文件
   - `critic_*.h5` 或 `critic_*.ckpt` 文件
   - `results.json`（可选，用于读取训练参数）

## 调试技巧

如果模型加载失败，检查：

1. **模型目录是否存在：**
   ```bash
   ls -la models/apf_learnable/best/
   ```

2. **模型文件是否存在：**
   ```bash
   find models/apf_learnable/best -name "*.h5" -o -name "*.ckpt"
   ```

3. **查看评估日志：**
   ```bash
   tail -f evaluation_results/apf_learnable_local_local/evaluation.log
   ```

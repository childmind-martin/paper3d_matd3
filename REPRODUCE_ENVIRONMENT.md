# 运行环境与复现说明

这份文档只描述当前仓库实际采用的复现链，不再混用旧环境记录。

## 1. 复现目标

本仓库有两种常见复现方式：

1. 从 GitHub 克隆代码，在服务器上重新训练，然后再评估。
2. 不重新训练，直接把已有模型与最小必要产物带到服务器上做评估或续训。

这两种方式需要的文件集合不同，最小提交清单见 [GITHUB_REPRO_CHECKLIST.md](GITHUB_REPRO_CHECKLIST.md)。

## 2. 推荐运行环境

- 操作系统：Linux
- Python：`3.10.x`
- TensorFlow：`2.12.0`
- TensorBoard：`2.12.0`
- Gym：`0.26.2`
- NumPy：`1.23.5`
- SciPy：`1.15.2`
- Matplotlib：`3.10.1`
- Plotly：`5.22.0`
- tqdm：`4.67.1`
- 其他默认运行依赖：`opencv-python`、`pygame`、`imageio`、`imageio-ffmpeg`、`psutil`、`PyOpenGL`、`pyglet`

说明：

- 训练脚本默认会生成交互式 HTML 轨迹，因此默认复现链把 `plotly` 视为必需依赖。
- 仓库中的 `requirements.txt`、`setup_conda_env.sh`、`repair_conda_env.sh` 和 `tools/check_tf_env.py` 已统一到上述版本。
- `pandas` 不是当前主训练/评估链的硬依赖，不再作为默认复现前提。

## 3. 推荐安装方式

优先使用仓库自带脚本，而不是手动逐条安装：

```bash
bash setup_conda_env.sh
conda activate maddpg_env
python tools/check_tf_env.py --label "TensorFlow Environment Check"
```

如果服务器环境已经污染，使用：

```bash
bash repair_conda_env.sh
```

如果你只是想先看服务器缺什么，可先执行：

```bash
bash tools/check_server_env.sh
```

如果你想在上传前或服务器 clone 后检查“这份代码/模型包本身能不能跑”，再执行：

```bash
python tools/preflight_server_run.py --mode code --algorithm matd3
```

如果是直接评估已有模型包：

```bash
python tools/preflight_server_run.py \
  --mode eval-existing \
  --algorithm matd3 \
  --model-dir models/<exp_name>/best_by_team_sr
```

## 4. 代码复现的默认入口

训练：

```bash
./run_with_conda.sh
```

评估：

```bash
./run_evaluation.sh
```

当前默认约定：

- 固定位置文件默认使用 `./saved_positions/5.json`
- `run_evaluation.sh` 会优先尝试 `models/` 下最新实验目录；若没有显式给模型路径，不再写死某个本机时间戳目录
- 训练结束后会把训练配置额外镜像到 `models/<exp_name>/results.json`

最后这一点很重要，因为严格评估需要读取训练期 `results.json` 才能对齐训练参数。

## 5. 现有模型直评所需最小产物

如果你不打算在服务器上重新训练，而是想直接评估已有实验，请至少带上：

- 一个模型目录，例如 `models/<exp_name>/best_by_team_sr/`、`models/<exp_name>/best/` 或 `models/<exp_name>/final/`
- 同一实验根目录下的 `models/<exp_name>/results.json`
- 固定位置评估时使用的 `saved_positions/5.json`，或你实际训练/评估时对应的 positions 文件

MATD3/MADDPG 评估建议保留的权重文件：

- `actor_*.weights.h5`
- `critic1_*.weights.h5`
- `critic2_*.weights.h5`

如果你还要继续续训，再额外带上：

- `target_actor_*.weights.h5`
- `target_critic1_*.weights.h5`
- `target_critic2_*.weights.h5`
- `checkpoint_state.json`

如果你暂时无法提供 `models/<exp_name>/results.json`，也可以提供匹配实验的 `logs/<exp_name>/<timestamp>/results.json`；但从现在开始，推荐优先使用模型目录下镜像出的那份 `results.json`。

对于历史训练产物，如果模型目录旁还没有这份文件，可先回填：

```bash
python tools/backfill_model_results.py --exp-name <exp_name>
```

## 6. MAPPO 说明

当前仓库的默认复现链以 `matd3` 为主。

如果你要复现 `mappo`，除了主训练/评估脚本外，还必须一并提交：

- `algorithms/mappo/`
- `train_mappo_strict.py`
- `evaluate_mappo.py`

现在的 `evaluate_optimized.py` 已经改成：

- 仓库里没有 MAPPO 相关文件时，不会阻塞 `matd3` / `maddpg`
- 只有在显式选择 `--algorithm mappo` 时，才会要求这些文件存在

## 7. 不建议直接提交到仓库的大产物

下面这些内容通常不应该作为常规 Git 提交的一部分：

- `logs/`
- `evaluation_results/`
- 大批量 `models/` 权重
- `__pycache__/`
- 临时诊断输出、导出图片、HTML 预览、一次性实验缓存

如果确实需要分享已训练模型，优先考虑：

1. 只上传一个最小实验包
2. 使用 GitHub Release 或 Git LFS
3. 至少保证模型目录与对应 `results.json` 成对出现

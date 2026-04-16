# GitHub 最小复现清单

这份清单的目标不是“把整个工作目录都传上去”，而是只保留服务器下载后真正能复现所必需的内容。

推荐在上传前先跑一遍：

```bash
python tools/preflight_server_run.py --mode code --algorithm matd3
```

如果你要直接评估现有模型包，再跑：

```bash
python tools/preflight_server_run.py \
  --mode eval-existing \
  --algorithm matd3 \
  --model-dir models/<exp_name>/best_by_team_sr
```

## A. 代码复现最小清单

如果目标是“服务器克隆仓库后重新训练并评估”，至少要保证以下内容已经提交到 GitHub：

- `run_optimized.sh`
- `run_evaluation.sh`
- `run_with_conda.sh`
- `paper3d_train_optimized.py`
- `evaluate_optimized.py`
- `potential_field_corrector.py`
- `requirements.txt`
- `setup_conda_env.sh`
- `repair_conda_env.sh`
- `REPRODUCE_ENVIRONMENT.md`
- `tools/check_tf_env.py`
- `tools/check_server_env.sh`
- `multiagent/`
- `agents/`
- `core/`
- `utils/`
- `visualization/`
- `saved_positions/5.json`
- `src/multiagent/`

如果你还需要保留多算法入口，再额外提交：

- `algorithms/`
- `algorithms/mappo/`
- `train_mappo_strict.py`
- `evaluate_mappo.py`

说明：

- `saved_positions/5.json` 是当前默认固定位置文件，缺它会影响默认复现路径。
- `src/multiagent/` 仍建议保留，因为环境脚本会执行 `pip install -e src/multiagent`。
- `algorithms/mappo/` 这一组文件只有在你要复现 MAPPO 时才是必需项。

## B. 现有模型直评最小清单

如果目标是“服务器下载后不重新训练，直接评估已有模型”，除了上面的代码外，还至少要有：

- `models/<exp_name>/results.json`
- `models/<exp_name>/best_by_team_sr/` 或 `models/<exp_name>/best/` 或 `models/<exp_name>/final/`
- 与该实验一致的固定位置文件，例如 `saved_positions/5.json`

MATD3/MADDPG 目录内建议至少保留：

- `actor_*.weights.h5`
- `critic1_*.weights.h5`
- `critic2_*.weights.h5`

如果你要续训，再补充：

- `target_actor_*.weights.h5`
- `target_critic1_*.weights.h5`
- `target_critic2_*.weights.h5`
- `checkpoint_state.json`

如果模型目录里暂时还没有镜像出来的 `results.json`，那就必须额外提供与该实验精确匹配的：

- `logs/<exp_name>/<timestamp>/results.json`

## C. 不建议常规提交的内容

下面这些通常不该进入常规 Git 提交：

- `logs/`
- `evaluation_results/`
- 大批量 `models/` 目录
- `__pycache__/`
- `.mplcache/`
- 临时导出图片、HTML 预览、诊断目录、一次性实验缓存

## D. 推荐上传策略

最稳妥的做法是分成两层：

1. GitHub 仓库存代码与最小配置文件
2. 模型权重与大产物走 GitHub Release、Git LFS 或单独压缩包

如果只能上传一个最小实验包，优先保证这三样一起出现：

1. `models/<exp_name>/<model_variant>/`
2. `models/<exp_name>/results.json`
3. `saved_positions/5.json`

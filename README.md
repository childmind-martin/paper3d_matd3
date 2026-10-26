# EC-MATD3: Learning with APF-Corrected Actions for Multi-UAV Navigation

本仓库研究多无人机协同导航：Actor 提出运动动作和 APF 参数，人工势场（APF）修正动作后交给环境执行。`ec-matd3` 分支新增按执行控制学习的 EC 路线，以及配套任务、训练、评估和统计流程。

**截至 2026-10-26：实现与部分正确性检查已完成；当前修订版尚无完整 EC 训练、验证选模和正式测试结果。不能据此声称成功率提高、收敛、泛化更强或超过 selector。**

## 当前做到什么程度

| 工作 | 当前状态 | 证据与边界 |
| --- | --- | --- |
| EC 与 RawOnly 学习方式 | 已实现 | 有动作投影、回放、局部梯度和配置检查；不是性能实验 |
| 奖励、终止、剩余时间观测 | 已实现并有组件测试 | 正式课程显式启用新配置；并非所有旧入口自动启用 |
| 任务库、C0—C3 课程、固定协议 | 已实现 | 任务生成与审计工具齐备；旧任务库和协议不能自动对应当前源码 |
| checkpoint 选择、固定力比例、种子统计 | 已实现并有规则测试 | 尚未产出当前版本的完整正式结果矩阵 |
| 回归检查 | 此前报告 220 项通过 | 包含已有 selector 测试；本次文档更新未重跑，未附完整 unittest 原始日志 |
| 真实环境短程更新 | **RawOnly**：16 步交互、6 次更新通过 | [原始 JSON](docs/validation/raw_only_smoke_20261026.json)；不是 EC 的完整训练 |
| EC 历史启动 | 旧配置日志到 13/800 回合 | 本地审计记录；没有完整模型和正式测试，且旧源码与当前版本不同 |
| 当前 EC 的成功率、稳定性与泛化 | **待正式实验验证** | 没有可报告的新性能数值 |

完整依据、测试范围、历史运行记录和未完成事项见 [EC 状态与分析记录](docs/EC_MATD3_STATUS_20261026.md)。旧论文表格属于历史版本，不评价本分支的新算法、奖励或任务规则。

## 与 selector 版本的主要区别

分支从 `matd3-selector@a283524` 创建，首个 EC 源码提交为 [`bb41f69`](https://github.com/childmind-martin/paper3d_matd3/commit/bb41f69fd6ec3ca6bf21c5bf5a5478028bc46112)，包含 45 个代码、测试和说明文件。

- **selector**：在历史 Dual-Q／分离梯度路线中，决定哪些跨智能体参考经验值得用于学习。当前正式 EC 五方法矩阵没有启用 selector。
- **EC**：让 Critic 评价实际执行的控制，Actor 的运动输出与 APF 参数经过 APF 变换共同获得梯度。它保留 TD3 的两个 Critic，每个输出一个完整标量价值，取消每个 Critic 内部的加法分解。
- **公共训练修订**：MATD3 目标动作由 target actor 加平滑噪声产生，再按配置经过 APF；另增加正式奖励、终止规则、任务标签与课程训练。
- **评估修订**：固定任务划分，用 C3 验证集选一个 checkpoint，随后用于各测试集；测试时不重新选模型或调整力比例。统计以独立训练种子为单位。

| 正式方法 | Critic 的动作信息 | 环境执行 APF | 目的 |
| --- | --- | --- | --- |
| `EC-MATD3-APF` | 执行后的三维控制；接口尾部补零 | 是 | 当前主方法 |
| `RawOnly-MATD3-APF` | 完整七维原始动作，含 APF 参数 | 是 | 原始动作建模对照 |
| `EC-MATD3` | 不经 APF 的控制 | 否 | APF 贡献对照 |
| `DualQ-MATD3-APF` | 历史 head/tail 加法分支及分离更新 | 是 | 历史设计对照 |
| `MADDPG-APF` | 执行后的控制，采用 MADDPG 更新 | 是 | 算法对照 |

三架无人机的 Actor 分别建网，**不共享参数**。历史 M3 的 shared selector 不表示 Actor 共享。

当前协议的主要确认性对比是 `EC-MATD3-APF` 对 `EC-MATD3`，回答 APF 的贡献；它不能单独证明 EC 优于 RawOnly。协议还对主方法设置了额外的验证合格门槛。正式训练前需要确认这些规则与论文主问题一致，详见[分析与待办](docs/EC_MATD3_STATUS_20261026.md#尚未完成的论证与实验)。

## 应该使用哪个入口

| 文件 | 用途 |
| --- | --- |
| [paper_experiment_protocol.py](paper_experiment_protocol.py) | 固定方法、种子、训练预算、评估和源码指纹 |
| [paper_experiment_pipeline.py](paper_experiment_pipeline.py) | `preflight → train → validate → select → test` |
| [build_paper_task_bank.py](build_paper_task_bank.py) | 生成分离的训练、验证、测试任务并审计可行性 |
| [paper3d_train_optimized.py](paper3d_train_optimized.py) | 实际训练与回放实现 |
| [evaluate_optimized.py](evaluate_optimized.py) | 实际评估与逐任务指标 |
| [paper_results_summary.py](paper_results_summary.py) | 正式结果汇总与统计 |

**`ablation_dual_q_separated_gradient.py` 默认仍是历史 separated／dual-Q／single-Q 三组；直接运行 `run_optimized.sh` 也仍默认旧 dual＋sep 配置。切换到 EC 分支本身不会把这些入口自动改成 EC。** 正式 EC 通过 protocol/pipeline 显式选择 `--method EC-MATD3-APF`。

## 检查与复现

环境背景见 [REPRODUCE_ENVIRONMENT.md](REPRODUCE_ENVIRONMENT.md)。此前验证使用本机已有的 Linux／Python 3.10／TensorFlow 2.12 环境；本次发布没有重新验收全新环境安装。该文档的通用训练入口不是正式 EC 协议入口。

在仓库根目录、激活可用环境后，可以独立复核此前的三组测试：

```bash
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
python -m unittest -v test_paper_*.py test_executed_control_critic test_review_revision
python -m unittest -v test_experiment_integrity
python -m unittest -v test_cross_agent_reference_selector
```

对应范围为 131＋82＋7＝220 项。检查 RawOnly 的实际短程更新：

```bash
python tools/smoke_review_raw_only.py --output paper_artifacts/raw_only_smoke.json
```

上面的 smoke 不测 EC 收敛。正式实验需要重新生成并审计任务库、冻结协议，然后按同一协议训练各方法和种子。具体流程及现有预算计划见[状态记录中的继续实验说明](docs/EC_MATD3_STATUS_20261026.md#继续正式实验)。

## 文档与记录

- [EC 状态与分析记录（2026-10-26）](docs/EC_MATD3_STATUS_20261026.md)：当前程度、证据来源、审稿问题对应、未完成实验和发布记录。
- [RawOnly 原始 smoke JSON](docs/validation/raw_only_smoke_20261026.json)：保留实际输出；原始文件未包含 commit/hash，不能充当完整源码来源绑定。
- [9 月修订记录](docs/REVIEW_REVISION_20260907.md)：当时的修改、检查与后续实验计划；其中本机论文、备份、视频不随此次源码发布。
- [统计实现说明](docs/REVIEW_STATISTICS_20260907.md)：训练种子、bootstrap、符号翻转检验和多重比较的范围。
- [历史投稿说明](docs/HISTORICAL_SUBMITTED_ARTIFACT.md)：从旧 README 保留的机制表述、表格与复现入口。它不代表当前 EC 已经取得这些结果。

本分支发布源码、测试与说明，以及上述小型 smoke 记录。完整训练日志、模型、旧任务库、备份、审稿 PDF、论文源文件和演示视频未包含在本次发布中。

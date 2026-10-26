# EC-MATD3 当前状态、分析与验证记录

记录日期：2026-10-26。代码基准：`ec-matd3@bb41f69fd6ec3ca6bf21c5bf5a5478028bc46112`；分支来源：`matd3-selector@a28352490d63ce03c47f15134fc02fded48c1816`。本次只整理文档、保留既有 smoke 输出，没有更改算法或启动新的训练。

**当前程度：已实现修订算法与实验流程，部分组件和短程更新经过检查；尚未完成当前修订版 EC 的完整训练、验证选模和正式评估。**

## 实现了什么

| 部分 | 已实现内容 | 主要代码 |
| --- | --- | --- |
| 执行动作学习 | EC 的每个 Critic 输出一个标量，动作输入只保留执行控制；Actor 梯度经过 APF | [训练主文件](../paper3d_train_optimized.py)、[梯度测试](../test_executed_control_critic.py) |
| 自然对照 | RawOnly 保留完整 raw 七维供 replay、target、Actor 查询使用，环境仍执行 APF | [动作语义](../action_semantics.py)、[方法配置](../paper_experiment_protocol.py) |
| MATD3 target | target actor 的下一步动作加平滑噪声，再按配置重建 APF；不再优先使用 replay 中的下一步行为动作 | [训练主文件](../paper3d_train_optimized.py) |
| 正式任务反馈 | `paper_v1` 进度奖励、持续代价、成功/碰撞/越界/超时奖惩；正式课程开启剩余时间观测 | [场景](../multiagent/scenarios/paper3d_terrain_energy.py)、[课程](../paper_curriculum.py) |
| 回放与课程 | 记录真实终止、截断、任务编号、难度和阶段；阶段切换有回放清理等明确规则 | [回放测试](../test_paper_replay_contract.py)、[协议](../paper_experiment_protocol.py) |
| 正式评估 | 分离任务集、C3 验证选模、同一 checkpoint 跨测试集、固定力比例、CPU 确定性评估配置 | [流程](../paper_experiment_pipeline.py)、[选择器](../paper_checkpoint_selection.py) |
| 统计 | 训练种子层面推断，任务/地形配对 bootstrap，符号翻转检验和 Holm 诊断校正 | [统计](../paper_seed_statistics.py)、[汇总](../paper_results_summary.py) |

EC 仍保留 TD3 的两个 Critic；取消的是每个 Critic 内部的两个加法价值分支。EC 与 RawOnly 名义动作输入宽度相同，但 EC 尾部补零、RawOnly 尾部有效，不能宣称有效输入信息完全相同。

Actor 按智能体分别创建并分别维护 target actor，三个 Actor 不共享参数。历史 M3 的共享对象是额外的 reference selector，不是 Actor。当前正式五方法矩阵未启用该 selector。

配置边界也需要保留：`paper_v1` 奖励与剩余时间观测必须显式启用；公共 target 与部分终止修复会影响历史路线。bootstrap 的终止 mask 取 `terminated`，普通截断单独记录；本项目正式硬 deadline 被定义为任务终局，而不是外部截断。

## 已有验证及其边界

### 回归测试

此前本地审计报告以下三组共 **220 项通过**。这是已有执行记录的整理；本次文档更新没有重跑，仓库也未附该次完整 unittest 标准输出。

| 组别 | 数量 | 覆盖 |
| --- | ---: | --- |
| 16 个 `test_paper_*` 模块 | 115 | 任务、奖励、课程、回放、评估、选模、统计及来源约束 |
| `test_executed_control_critic` | 5 | 动作投影、局部梯度及 target 规则标签 |
| `test_review_revision` | 11 | RawOnly 配置/回放、观测一致性及统计边界 |
| `test_experiment_integrity` | 82 | 现有工程和实验规则回归 |
| `test_cross_agent_reference_selector` | 7 | 保留的 selector 核心运算回归 |

前三行合计 131；131＋82＋7＝220。9 月文档中的 213 项不包含额外检查的 selector 7 项。相对 selector 基线新增的是 132 项测试，并非 220 项全部新增。运行命令见[README](../README.md#检查与复现)。

这些检查有不同证据强度：

- APF 梯度测试使用人工几何输入调用真实变换函数，证明受测局部路径可微；没有检验完整策略长期优化的效果。
- 奖励/终止测试运行真实环境，但部分状态由测试直接设置到达、碰撞或超时条件；不能把它们当作策略自行导航成功。
- `test_target_action_contract_is_target_policy_not_replay_behavior` 只检查字符串常量；target 修订结论还来自源码路径核对，并非已完成完整数值等价测试。
- 选模和统计测试包含合成数据及人工候选成绩，验证的是程序规则。它们生成的数值不是论文性能结果。

### 真实环境短程更新：仅 RawOnly

[原始 JSON](validation/raw_only_smoke_20261026.json) 从本地已有文件原样复制，没有修改数值。对应工具为 [smoke_review_raw_only.py](../tools/smoke_review_raw_only.py)。

| 字段 | 实际记录 |
| --- | ---: |
| 环境交互步数 | 16 |
| 更新调用 | 6 |
| 其中缺失辅助动作字段的兼容更新 | 2 |
| 被检查的更新 batch | 9 |
| 最大 APF 干预范数 | 1.3586704730987549 |
| Actor 权重变化 L2 范数之和 | 0.031303128955187276 |
| Critic 权重变化 L2 范数之和 | 0.38284183447103715 |

配置为 CPU、单环境、32×32 小网络、低复杂度静态场景、固定力比例 0.5、关闭 selector。它验证了动作生成、APF 执行、环境交互、回放与实际网络更新能连通，且参数变化有限。它没有运行正式 EC 课程，也没有证明收敛或成功率。

原文件本地路径为 `/tmp/matd3_audit_raw_only_20261026_103200.json`，SHA256 为 `3edf068bd51df2642c51e70b51cbd421c2fce8048866fa19ca29baf868448c62`。**原始 JSON 没有源码 commit/hash 字段，所以此附件不是与当前发布 commit 完整绑定的训练记录。** 文档中的代码基准表示此次审计所阅读的版本，不补写原始运行缺失的来源字段。

### EC 历史训练与 preflight

以下是本地文件审计结论；完整原日志和 preflight 文件未随本次文档发布。

| 记录 | 本地观察 | 可以说明什么 |
| --- | --- | --- |
| 2026-08-30 第一次 EC 启动（r1） | `Path` 未导入导致失败；当前已有导入和回归检查 | 当时启动失败，不能算完成训练 |
| 2026-08-30 第二次 EC 启动（r2） | seed 101 的 stdout 到 13/800，含 Actor/Critic loss；对应模型目录为空 | 旧配置曾实际启动更新，未完成 |
| `formal_experiments_v1/v2/v3` | 共 7 个 preflight 记录，均记录 `training_started:false` | preflight 是启动前检查；该字段不能推翻随后 r2 的实际训练日志 |
| 正式训练、验证、测试产物 | 三个实验根目录未找到训练完成记录、validation 或 official-test manifest | 不能报告当前 EC 的正式实验成绩 |

r2 日志本地相对路径为 `logs/paper_formal_v1_20260830_r2__EC-MATD3-APF__seed101_20260830_214523/run_optimized_stdout.log`，第 240—241 行记录 13/800。最新旧 preflight 位于 `paper_artifacts/formal_experiments_v3/records/preflight/EC-MATD3-APF/seed_101.json`；其中训练脚本、实验协议和地形场景的源码指纹与此次审计版本不一致。

因此准确状态是：**旧 EC 启动过；当前修订版没有可核验的完整正式训练与测试结果。** 不能写成“从未启动”，也不能用旧 13 回合证明当前修订版已完成训练验证。

## 尚未完成的论证与实验

| 研究问题 | 当前回应 | 仍需完成 |
| --- | --- | --- |
| 为什么需要两个加法价值分支 | 新 EC 取消加法分解，旧 DualQ 保留作对照 | 新实现没有证明旧分支有独立语义，也未证明旧分解必要 |
| 分离梯度是否有独立收益 | 有历史路线和配置 | EC 对 DualQ 同时改变结构、动作查询和 Actor 更新，不能把差异全部归因于分离梯度；需控制其他因素的实验 |
| 是否必须显式评价执行动作 | 增加完整 raw 七维自然对照 | 同任务、奖励、预算和选模规则下比较 EC 与 RawOnly |
| 是否构成两层决策 | 当前描述为 Actor 输出加确定性 APF 变换 | 不凭“意图/执行”命名宣称时间层级或独立宏策略 |
| 成功率低 | 修订任务反馈、终止和课程；坚持严格团队成功率 | 完整训练曲线、所有独立训练种子和正式成功率，不能用距离或局部到达替代 |
| 选模与力比例混杂 | 固定验证集选模、单 checkpoint、测试禁止改力比例 | 在完整实验中执行并保留全部记录 |
| seed 太少与泛化 | 有独立训练种子统计和未见地形划分 | 实际完成多 seed 与未见地形测试；重复评估不能增加训练次数 |
| 基线与 selector | 正式矩阵有 no-APF、RawOnly、DualQ、MADDPG；历史 selector 保留 | 所需基线在新任务规则下重跑；当前五方法矩阵不包含 selector 或 MAPPO |

当前协议有两个必须明确披露的设计点：

1. **主要确认性对比是 APF 贡献。** `paper_experiment_protocol.py` 的 `_REPORTING_CONTRACT` 指定 `test_C3` 上 `EC-MATD3-APF` 对 `EC-MATD3` 的团队成功率；EC 对 RawOnly 尚不是该主对比。若论文主问题改为动作建模，需要在正式实验之前调整并冻结相应协议。
2. **验证合格门槛不对称。** 只有主方法设置 `require_validation_adequacy=true`，其余四方法为 false。C3 门槛要求 64 个验证任务至少 44 次成功，且 Wilson 下界达到 0.55。这是选模规则，不是实测成功率；不能描述为所有方法使用同一个合格门槛。正式报告需披露不合格运行，不能静默删掉失败 seed。

任务可行性审计回答参考控制器能否完成给定任务，不能替代 RL 策略的成功率。旧 selector、旧投稿表格、历史 Gazebo 回放也不能直接当作新 EC 的同条件性能证据。

## 继续正式实验

先确定论文要验证的主问题、比较方法、统一规则和预算，再冻结源码与协议。随后依次进行：

1. 使用 `build_paper_task_bank.py` 生成并审计当前源码对应的训练、验证、测试任务。
2. 使用 `paper_experiment_protocol.py` 固定方法、训练种子、预算、运行环境和源码指纹。保留失败记录，不手工改旧 hash 复用不兼容结果。
3. 对每个方法/seed 用 `paper_experiment_pipeline.py preflight` 检查，再 `train` 完成训练。
4. 执行 `validate`、`select`，仅由 C3 验证集选一个 checkpoint。
5. 对 `test_C0`—`test_C3` 和任务库中全部未见地形 split 分别 `test`，之后用 `paper_results_summary.py` 汇总完整协议。

CLI 当前默认是 5 个训练 seed、800 个训练迭代、8 个并行环境，即每方法/seed 6,400 条轨迹。它是默认配置，不是完成量。9 月修订文档给出的计划是 10 个 seed、1,600 次迭代、8 个环境，即每方法/seed 12,800 条轨迹、五方法共 50 个训练运行；**该计划尚未完成，也未在本次文档更新中启动。** 迭代数、并行轨迹数和环境步数不能互换。

任务生成和协议命令见[9 月修订说明](REVIEW_REVISION_20260907.md#在-wsl-中继续正式实验)。该文档以 RawOnly 为训练示例；EC 使用相同协议和工作目录，显式选择 `--method EC-MATD3-APF`。增加训练预算本身不能证明收敛；要报告固定预算曲线，并事先确定稳定性检查方式。

正式协议还规定中断后从头重启该训练 seed。旧通用入口的“续训”能力不能直接当作正式实验允许改变训练轨迹的许可。

## 发布与记录说明

| 日期/版本 | 记录 |
| --- | --- |
| `matd3-selector@a283524` | 此分支的历史代码基线 |
| 2026-09-07 | [修订记录](REVIEW_REVISION_20260907.md)及[统计说明](REVIEW_STATISTICS_20260907.md)，保留当时范围 |
| 2026-10-26，`bb41f69` | 上传 45 个 EC 相关源码、测试与说明文件；从 selector 新建 `ec-matd3` |
| 2026-10-26，本次文档补充 | 重写 README 为当前 EC 入口；原 README 历史正文移入[档案](HISTORICAL_SUBMITTED_ARTIFACT.md)；加入本状态记录和既有 smoke JSON |

9 月记录中的本地论文、视频、备份及“随交付保留”的测试日志指当时工作环境，不表示当前 GitHub 分支包含这些附件。此次发布不包含完整训练日志、模型或正式测试矩阵。

未来性能结果应同时提供方法配置、源码/协议/任务库指纹、训练 seed 和预算、训练完成记录、验证选择依据、最终模型以及逐任务测试记录。本次文档仅标明已知证据和缺口，不用计划、测试夹具或历史成绩填补新实验结果。

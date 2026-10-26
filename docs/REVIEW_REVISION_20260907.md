# MATD3 拒稿修订说明 — 2026-09-07

> 本文保留 2026-09-07 的本地修订记录。当前分支状态、2026-10-26 的 GitHub 发布与验证边界见 [EC 状态与分析记录](EC_MATD3_STATUS_20261026.md)。文中的本机论文、视频、备份及测试日志并不随此次源码发布；末尾“没有提交或推送 Git”描述的是当时状态。

本轮直接修改当前电脑 WSL2 Ubuntu 的 `/home/tang/matd3`，论文主文件在 `/home/tang/paper`。Windows 工作目录只用于补丁暂存、日志及交付副本。没有通过 Cursor 扩展 UI 下发任务，也没有修改远程服务器上的另一个同名仓库。

## 本轮代码修改

- `action_semantics.py` 与训练入口：增加 `raw_only`。保留完整原始 7 维动作（含 APF 参数尾部），回放、目标 critic 查询和 actor 查询保持原始坐标，环境仍执行 APF 修正。为兼容既有缓冲区，两个动作数组保存相同 raw 值；不宣称节省存储空间。缺失辅助数组时直接使用 raw，混用 raw/executed 别名则明确报错。缺失辅助数组时直接使用 raw，混用 raw/executed 别名则明确报错。
- 禁止 raw-only 与双分支 critic、分离/混合梯度、修正目标或绕过 APF 的 epsilon 随机探索组合，错误配置在建网前报错。
- 正式方法矩阵加入 `RawOnly-MATD3-APF`，其余结构和 APF 配置与 scalar executed-control 主方法保持一致。二者网络输入宽度一致，但 raw-only 的尾部坐标有效，执行控制 critic 尾部零填充；因此不声称有效输入信息量完全一致。
- 正式共同配置将 `random_action_prob` 从 0.01 改为 0，保留施加在原始动作上的 OU 探索。原因是旧 epsilon 动作绕过 APF，不符合固定纠正算子的组合转移假设。旧实验默认路径未被重解释成新结果。
- 修正批量观测缺少数值裁剪的问题，使其与标量观测路径均限制在 `[-10,10]`。测试覆盖智能体严重越界后的输入范围。
- 新增训练种子层面的配对符号翻转检验及 Holm 多重比较校正；保存到统计 JSON/CSV。保留原有 crossed bootstrap 和预先指定的主对比 CI 决策。检验假设与适用范围见独立统计说明。

现有的执行控制 critic、重新设计的奖励、碰撞/越界终止、时间观测、任务库、课程训练和严格评估链是本轮开始前已存在的未提交工作；本轮保留并验证这些内容，不将它们列为新实现。

## 与审稿意见的对应关系

- **M1：建模。** 论文统一状态、动作、观测、单位、转移与合作目标。明确中央 critic 的联合观测不等于完整 Markov 状态，个体 shaping 奖励与严格共享奖励目标的差别。
- **M2：障碍物。** 更正为回合间重采样的静态球形障碍物，给出数量、半径、分布与拒绝采样规则。代码中的 `use_dynamic_obstacles` 不能被解读为障碍物在回合内运动。
- **M3：critic 理论。** 加入加性分支不可辨识的讨论，分离 actor 梯度只能作为启发式；明确理想 Bellman 论证的前提。新主线使用 scalar executed-control critic，旧加性设计是待比较消融。
- **M4：消融解释。** 增加可运行 raw-only 自然控制，将 collapsed replay 失败同时解释为动作查询/分布不匹配；旧结果不足以证明保存两个动作通道必不可少。
- **M5：训练质量。** 修复观测差异，核验奖励/终止等现有契约及真实梯度更新。正文不再以 partial arrival 替代任务完成。收敛曲线与更高成功率仍需正式训练得到。
- **M6：统计。** 新实现按训练种子配对、保留 task/terrain 层面的 bootstrap 并处理多重比较。3 个独立训练种子的双侧精确符号翻转检验最小 p 值为 0.25；重复评估不能增加独立训练样本数。
- **M7：基线。** 论文明确 3,374 m 漂移和大量零成功不能支撑算法优越性。新共同环境契约仍需对各基线进行预算一致、只用验证集调参的正式验证；尚未完成 MAPPO 重调参。
- **M8：外部验证。** 提供已有模拟轨迹的成功/失败定性回放视频及来源记录。它不是本轮新训练结果、硬件验证或新的 Gazebo 闭环测试；泛化测试仍需执行。
- **M9：表达。** 简化标题、统一主要术语，并重写摘要、讨论和结论，区分历史结果与新实现。

## 验证与数据边界

WSL 实际验证使用 `/home/tang/miniconda3/envs/maddpg_env/bin/python`。raw-only smoke 采集 16 步真实环境数据、执行 6 次更新（含 2 次缺失辅助字段的兼容回放）；检查原始/目标动作坐标一致，环境 APF 干预非零，actor 与 critic 权重均发生有限变化。它证明代码路径可运行，不证明性能、收敛或显著优越性。

最终 213 项测试全部通过，零失败、零跳过；测试记录随交付保留。所有旧表中的数值保持历史出处，没有用测试数据或演示片段填充新算法的性能表。论文是可审阅修订草稿，仍有实验条件未满足，不能视作已消除全部拒稿理由的投稿终稿。

## 在 WSL 中继续正式实验

先进入项目和现有环境：

```bash
cd /home/tang/matd3
source /home/tang/miniconda3/etc/profile.d/conda.sh
conda activate maddpg_env
python tools/smoke_review_raw_only.py --output paper_artifacts/raw_only_smoke.json
```

源码发生变化后，需要重新生成和审计任务库。以下命令使用新的输出目录，不覆盖原实验：

```bash
python build_paper_task_bank.py \
  --output-dir paper_artifacts/task_bank_review_20260907 \
  --bank-id review_20260907 \
  --base-terrain-seed 67 --held-out-terrain-seeds 701,719,743,787

python paper_experiment_protocol.py \
  --output paper_artifacts/review_20260907_protocol.json \
  --protocol-id review_20260907 \
  --task-bank-manifest paper_artifacts/task_bank_review_20260907/task_bank_manifest.json \
  --train-seeds 101,202,303,404,505,606,707,808,909,1010 \
  --training-iterations 1600 --training-num-envs 8
```

该配置对应每个方法/种子 12,800 条环境轨迹，C0/C1/C2/C3 分别为 1,536/2,048/3,072/6,144 条，均完整遍历 256 项训练任务库；预算必须在看测试集之前冻结。10 个种子、5 个方法共 50 个训练运行，不能将这个计划描述为已完成。 按当前完整探索比较族（4 个分布内测试、4 个新地形测试及 1 个汇总，乘 4 个对照和 6 项指标），Holm 族含 216 项。10 个种子的精确双侧 p 值经此校正最低为 0.421875，因此这些 p 值只作探索诊断，主对比保留预先指定的 CI 判据；若计划以完整比较族作 p<0.05 声明，至少 14 个种子才在数学上可能，仍需事先作功效与预算设计，不能看结果后缩小检验族。更长预算本身也不证明收敛，应报告固定预算学习曲线，并预先规定加倍预算的稳定性检查。

以一个 raw-only 种子为例，先做启动检查再训练：

```bash
python paper_experiment_pipeline.py preflight \
  --protocol paper_artifacts/review_20260907_protocol.json \
  --experiment-root paper_artifacts/review_20260907_runs \
  --method RawOnly-MATD3-APF --train-seed 101

python paper_experiment_pipeline.py train \
  --protocol paper_artifacts/review_20260907_protocol.json \
  --experiment-root paper_artifacts/review_20260907_runs \
  --method RawOnly-MATD3-APF --train-seed 101
```

其余方法和种子必须使用同一冻结协议。训练完成后，依次使用 `paper_experiment_pipeline.py validate`、`select`、`test`，保持相同参数并使用 C3 验证选择的单一 checkpoint；测试覆盖 `test_C0` 至 `test_C3` 和任务库记录的全部 4 个 `test_generalization` 分区。最后使用 `paper_results_summary.py` 汇总完整协议。不要复用旧 protocol 的 hash，也不要跳过失败种子、挑选测试集 checkpoint 或将未完成结果混入完整矩阵。

## 原文件与回退

本轮覆盖前的代码原文件位于 `/home/tang/matd3/backups/reviewer_revision_20260907/`，论文备份位于 `/home/tang/paper/backups/reviewer_revision_20260907/`。这些备份包含本轮开始时已有的未提交修改。交付补丁仅表示本轮变化，不要求重置整个仓库；没有替用户提交或推送 Git。

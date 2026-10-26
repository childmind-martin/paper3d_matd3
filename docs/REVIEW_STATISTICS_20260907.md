# 拒稿修订：统计检验与复现说明（2026-09-07）

> 本文说明当时实现的统计方法与组件检查，不报告新的训练性能。当前实验完成程度与发布记录见 [EC 状态与分析记录](EC_MATD3_STATUS_20261026.md)。

本次修改位于本机 WSL2 Ubuntu 的 `/home/tang/matd3`。修改文件为 `paper_seed_statistics.py`、`paper_results_summary.py`；新增 `test_paper_seed_statistics.py`。没有生成或补写任何正式实验结果。

## 独立单位与检验

每个方法的训练随机种子是独立训练单位；同一 seed 内先对评估任务取平均，再对方法间的 seed 均值差做双侧配对符号翻转检验。三维泛化结果先在该 seed 内平均地形与任务。检验条件是 seed 差异独立且在原假设下对称，或方法标签在配对内可交换；仅有均值相等并不足以保证这一条件。[SciPy 配对置换检验文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)

3 个独立训练 seed 只有 `2^3=8` 种符号组合，双侧精确 p 值的理论下界是 `2/8=0.25`。增加评估 episode 不会增加独立训练次数；若 30 个 train/eval 组合来自同样的 3 个训练 seed，它们仍是 3 个独立训练单位，不能写成 30 次独立训练。深度强化学习的小样本评估必须同时报告训练次数与不确定性。[Agarwal 等，NeurIPS 2021](https://arxiv.org/abs/2108.13264)

`n<=20` 时穷举所有符号组合，包含观测组合且将相等统计量计入尾部；更大 n 使用固定随机种子的 Monte Carlo，并采用保守的 `(extreme+1)/(draws+1)`，避免零 p 值。代码修复了固定绝对容差导致微小尺度差异误判的问题：归一化用于统计量比较，原始效应值保持原单位。[Phipson 与 Smyth，2010](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)

## 校正范围与结果解释

新增 p 值全部标记为探索性诊断。Holm 校正使用同一个完整检验族：本次正式汇总的所有 split × baseline × 六项性能 metric，加上 G1 地形汇总的相同比较；该检验族也包含主对比的诊断 p 值。JSON 保存完整 family member 列表及数量，CSV 每行导出原始/校正 p 值、检验族、独立 seed 数、计算方式与推断范围。APF 机制指标仍仅为描述性结果，不产生额外检验。[Holm，1979](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf)

原预声明主分析的“定向效应 95% bootstrap CI 下界大于 0”决策规则保持不变，新增诊断 p 值不会替换或附加其判定条件。原有次要分析不据此产生二元优越性结论。小样本下主 CI 规则可能满足，而 seed 诊断检验不能拒绝原假设；应同时如实呈现，不能把两者写成相互等价的显著性证据。

原有 seed × immutable task 配对 bootstrap，以及 seed × held-out terrain × terrain 内 task 的 bootstrap 均保留。bootstrap 大于零的重复样本比例属于描述性重采样比例，不是 p 值。seed 符号翻转检验条件于已评估任务/地形集合，不单独证明对新地形总体的泛化能力；地形泛化不确定性仍由原三维 bootstrap 表达。

汇总 schema 升为 `paper_results_summary_v6`，来源指纹加入新统计模块。旧正式实验的来源绑定没有回填；需要按更新后的协议重新训练/评估，才可生成修订稿的正式统计结果。

## 验证

在 WSL2 的 `maddpg_env` 运行 `python -m unittest -v test_paper_seed_statistics test_paper_results_summary`，31 项测试全部通过，其中新增 20 项。覆盖 SciPy 精确配对置换对照、3-seed 下界、评估复制不增加样本量、计量单位不变性、Monte Carlo 可复现与非零 p、Holm 跨整个检验族、原 CI 决策保持、2D/3D bootstrap 区分 p 值，以及 JSON/CSV 集成导出与来源指纹。集成测试使用临时合成数据，仅验证程序，测试结束后自动清理。

修改前阶段备份保存在 `backups/reviewer_revision_20260907/statistics_before_holm/`，未覆盖此前的原始项目备份。

# MATD3/APF 代码创新点总结

本文档总结当前代码中可以用于论文创新点、方法贡献、实验贡献和工程支撑的内容。核心判断是：不是所有工程改动都适合作为论文创新。最有价值的主线是 **APF 修正执行动作后造成的动作语义错配问题**，以及代码中围绕该问题实现的 replay、critic、target 和 actor objective 的一致性设计。

## 0. 总体定位

当前代码最强的创新主线可以概括为：

> 在 APF-corrected multi-agent reinforcement learning 中，actor 输出的 raw policy action 和环境实际执行的 corrected action 具有不同语义。若 replay buffer、critic 输入、TD target 和 actor objective 混用这两类动作，会产生训练-执行语义错配。当前代码提出并实现了 dual-semantic replay、execution-consistent target construction、dual-Q critic 和 focused semantic ablation，用于显式建模和验证这种语义差异。

这条主线比简单的 “APF + MATD3” 更强。因为代码不是只做安全后处理，而是把安全修正导致的动作语义变化纳入了训练机制。

## 1. 核心创新一：Action Semantic Mismatch Problem

### 可以作为问题创新

当前代码隐含并实际解决的问题是：

- actor 输出的是 raw policy intention；
- APF 或安全层会把 raw action 修正为 corrected execution action；
- 环境状态转移由 corrected action 决定；
- 但 actor 学习和策略梯度又需要保留 raw intention；
- 如果 replay 或 critic 只保存一种 action，会造成语义坍缩或训练-执行不一致。

这一点可以写成论文中的问题定义：

> We identify an action-semantic mismatch in APF-corrected MARL: the policy intention produced by the actor and the safety-corrected action executed by the environment are not semantically identical, yet standard replay and critic learning usually treat them as a single action channel.

### 为什么重要

如果只存 corrected action，策略的原始意图被抹掉，critic 看到的是执行后的动作，而不是 actor 真正产生的动作。
如果只存 raw action，critic 又无法准确对应真实环境转移，因为环境实际执行的是 corrected action。
因此这不是一个普通实现问题，而是 APF-corrected RL 中的训练语义问题。

### 对应代码位置

- `paper3d_train_optimized.py`
  - `LiteReplayBuffer` 显式保存 raw/corrected/next-corrected 动作以及 APF 信息。
  - 参考：`paper3d_train_optimized.py:1960`
- `paper3d_train_optimized.py`
  - 训练时区分 storage action 和 execution action。
  - 参考：`paper3d_train_optimized.py:6185`

## 2. 核心创新二：Dual-Semantic Replay Buffer

### 创新内容

经验池不是普通 replay buffer，而是显式保存两套动作语义：

- `act`：raw policy action；
- `act_corrected`：APF-corrected executed action；
- `next_act_corrected`：下一状态对应的 corrected target action；
- `fr`：force ratio，用于记录 APF 修正强度；
- `pf_forces`：APF force；
- `pf_features` / `next_pf_features`：APF 相关状态特征。

这可以写成：

> A dual-semantic replay buffer that preserves both policy intention and corrected execution, preventing replay-level semantic collapse in APF-corrected actor-critic learning.

### 代码依据

- replay buffer 字段定义：`paper3d_train_optimized.py:1960`
- 单条经验写入 raw/corrected action：`paper3d_train_optimized.py:2270`
- 批量写入 corrected action：`paper3d_train_optimized.py:2570`
- sample 时返回 dual-semantic 信息：`paper3d_train_optimized.py:2827`

### 论文价值

这是最适合作为主贡献之一的内容。
原因是它不是普通数据结构优化，而是直接服务于论文核心问题：APF 修正后 raw action 与 executed action 的语义差异。

## 3. 核心创新三：Storage Action 与 Execution Action 分离

### 创新内容

训练采样动作时，代码明确返回并使用两套 action：

- `actions_for_storage`：用于 replay 存储的 raw/noisy actor action；
- `actions_for_execution`：用于环境交互的 APF-corrected action；
- `pf_forces`：APF 修正力；
- `raw_actor_outputs`：actor 原始输出；
- `pf_features_current`：当前 APF 特征。

### 代码依据

- vectorized action selection 返回五类信息：`paper3d_train_optimized.py:6185`
- raw action 和 corrected action 分离逻辑：`paper3d_train_optimized.py:6328`
- 训练循环中使用 storage/execution action：`paper3d_train_optimized.py:16761`
- replay 写入 raw 和 corrected action：`paper3d_train_optimized.py:17424`

### 论文表达

可以写成：

> During environment interaction, the framework separates the action used for replay storage from the action used for deployment execution. This preserves policy intention while keeping environment transitions aligned with APF-corrected execution.

### 为什么不是普通实现

普通实现往往只会把“实际执行的动作”塞进 replay。
当前代码保留了 actor intention 和 execution action 两条链路，这使后续 critic 和 actor objective 可以按语义分别处理。

## 4. 核心创新四：Execution-Consistent Target Reconstruction

### 创新内容

TD target 阶段不是简单使用 target actor 的 raw next action，而是可以根据 next state 的几何/APF 上下文重新构造 corrected target action。

也就是说，Bellman target 中的 next action 与部署时的 APF-corrected execution 保持一致。

### 代码依据

- `matd3_reconstruct_corrected_target` 配置：`paper3d_train_optimized.py:9453`
- target action smoothing 后进行 corrected target reconstruction：`paper3d_train_optimized.py:10876`
- 使用 next observation 几何上下文重构 APF-corrected target action：`paper3d_train_optimized.py:11035`
- raw smooth target 与 corrected target 分别构造：`paper3d_train_optimized.py:11092`

### 论文表达

可以写成：

> An execution-consistent Bellman target is constructed by reconstructing the APF-corrected target action under the next-state geometry, aligning target Q estimation with deployment-side execution semantics.

### 为什么重要

如果 target critic 使用 raw target action，但真实部署使用 corrected action，TD target 会和环境转移语义不一致。
这会导致训练端看起来有学习，但测试端表现退化。

这也是 `No corrected target reconstruction` 消融实验的核心意义。

## 5. 核心创新五：Dual-Q Critic for Hybrid Action Semantics

### 创新内容

actor action 不是单一语义动作，而是混合语义：

- 前 3 维：3D motion action；
- 后 4 维：APF parameter tail。

代码中的 critic 使用两个 Q head：

- `Q_head_1`：评价 motion action；
- `Q_head_2`：评价 APF parameter action。

这可以看作针对 hybrid action semantics 的 critic 结构。

### 代码依据

- dual-Q critic 网络定义：`paper3d_train_optimized.py:4247`
- MATD3 dual-Q 相关配置：`paper3d_train_optimized.py:9453`
- multi-agent update 中 head/tail Q 计算：`paper3d_train_optimized.py:12456`

### 论文表达

可以写成：

> A dual-head critic is introduced to separately model raw motion intention and APF-parameterized correction behavior under a shared state and potential-field representation.

### 注意事项

不要把它单独夸成完全独立的新 critic 范式。
更稳妥的写法是：它是 dual-semantic action framework 下的 critic design，用于配合 raw motion action 和 corrected/APF parameter action 的不同语义。

## 6. 核心创新六：Separated / Unified / Hybrid Actor Objective

### 创新内容

代码支持多种 actor objective：

- separated gradient objective；
- unified total-Q objective；
- hybrid objective；
- hybrid actor alpha 可调。

在 separated objective 中：

- motion head 可通过 raw-action Q 进行优化；
- APF parameter tail 可通过 corrected-action / APF-related Q 进行优化；
- hybrid objective 则混合 separated loss 和 unified total-Q loss。

### 代码依据

- actor objective 配置项：`paper3d_train_optimized.py:9453`
- actor path 中 raw/corrected global action 分离：`paper3d_train_optimized.py:12729`
- corrected actor head 重构：`paper3d_train_optimized.py:12820`
- separated objective：`paper3d_train_optimized.py:12927`
- hybrid objective：`paper3d_train_optimized.py:12947`
- unified objective：`paper3d_train_optimized.py:12967`

### 论文表达

可以写成：

> The actor objective is routed according to action semantics, allowing raw motion intention and APF-mediated parameter actions to receive semantically matched gradients.

### 论文定位

这是重要方法组件，但最好作为 dual-semantic framework 的一部分，而不是单独作为最大创新。

## 7. 方法创新七：Learnable APF Fusion Interface

### 创新内容

APF 不是固定后处理。actor 的后四维输出会映射为 APF 参数，例如：

- attraction gain；
- terrain-related weighting；
- repulsion gain；
- influence radius。

APF force 又融合了多种因素：

- goal attraction；
- terrain force；
- obstacle repulsion；
- inter-agent interaction。

最终通过 force ratio 把 raw action 与 APF force 融合：

```text
corrected = raw_action + force_ratio * (pf_force - raw_action)
```

### 代码依据

- actor action 拆分为 motion head 和 APF parameter tail：`paper3d_train_optimized.py:6471`
- APF geometry context：`paper3d_train_optimized.py:6497`
- goal/terrain/obstacle/inter-agent force 组合：`paper3d_train_optimized.py:6688`
- APF correction：`paper3d_train_optimized.py:6756`
- actor tail 到 APF 参数的映射：`paper3d_train_optimized.py:7131`

### 论文表达

可以写成：

> The APF module is formulated as a learnable execution interface: the actor controls APF parameters, while the resulting corrected action is explicitly tracked through replay, critic learning, and target construction.

### 注意事项

“APF + RL” 本身不是特别强的新颖点。
真正强的是：APF 修正不是孤立后处理，而是和 dual-semantic replay、critic、target 对齐机制绑定起来。

## 8. 实验创新一：Focused Semantic Ablation

### 创新内容

当前代码支持三个非常针对性的 semantic ablation：

1. Full dual-semantic method
   - raw action 和 corrected action 都进入 replay/critic/target；
   - TD target 重构 corrected action。

2. Collapsed replay
   - replay 只保留 corrected execution action；
   - raw/corrected 双语义被折叠。

3. No corrected target reconstruction
   - replay 保留双语义；
   - 但 TD target 阶段不重构 APF-corrected target action。

### 代码依据

- semantic ablation wrapper：`run_level2_dual_semantics_ablation_official.sh:6`
- full dual-semantic 配置：`run_level2_multiseed_all_algos_official.sh:361`
- collapsed replay 配置：`run_level2_multiseed_all_algos_official.sh:375`
- no corrected target reconstruction 配置：`run_level2_multiseed_all_algos_official.sh:389`
- action semantics mode：`paper3d_train_optimized.py:16727`
- 命令行参数：`paper3d_train_optimized.py:20365`

### 论文价值

这组消融非常适合支撑主创新，因为它直接验证：

- replay-level semantic collapse 是否有害；
- target-side execution inconsistency 是否导致部署退化；
- full dual-semantic design 是否更合理。

### 推荐表述

> The focused semantic ablation isolates replay-level semantic collapse and target-side execution inconsistency, providing mechanism-level evidence for the proposed dual-semantic design.

避免说：

- all components are universally necessary；
- statistically significant，如果没有正式统计检验；
- Full universally dominates。

建议说：

- focused mechanism evidence；
- deployment-side degradation；
- replay-level semantic collapse；
- target-side execution consistency。

## 9. 实验创新二：Matched Validation and Official Evaluation Protocol

### 创新内容

评估不是随机挑 checkpoint 或随机测试，而是：

- 生成固定 validation/test 地形与障碍物序列；
- 对不同方法使用一致的 position / terrain / obstacle seed；
- 先 matched validation 选择 checkpoint；
- 再 official evaluation；
- 测试阶段关闭 exploration noise 和 random action。

### 代码依据

- validation/test seed 序列生成：`official_eval_with_matched_validation.py:91`
- per-episode positions 与 seed metadata：`official_eval_with_matched_validation.py:199`
- checkpoint selection scoring：`official_eval_with_matched_validation.py:357`
- strict matched evaluation environment：`official_eval_with_matched_validation.py:374`
- matched validation 主流程：`official_eval_with_matched_validation.py:542`

### 论文表达

可以写成：

> A matched validation and official evaluation protocol is used to control terrain, obstacle, and start-goal sequences across methods, reducing stochastic evaluation noise and making cross-method comparison more reliable.

### 论文定位

这是实验方法贡献，不是算法贡献。
适合放在 Experimental Setup 或 Evaluation Protocol 中。

## 10. 实验创新三：Level 1 / Level 2 / Level 3 层级 benchmark

### 创新内容

当前实验体系不是单一地图，而是分层评估：

1. Level 1
   - fixed-map mechanism validation；
   - 适合验证基本框架机制。

2. Level 2
   - stochastic obstacle regime；
   - 固定 terrain seed，但有动态障碍物和多随机种子；
   - 用于比较主方法和 baseline。

3. Level 3
   - semi-random terrain-family stress evaluation；
   - terrain family、variant seed、peak jitter、height scale 等变化；
   - 用于更强泛化压力测试。

### 代码依据

- Level 2 official 脚本通用配置：`run_level2_multiseed_all_algos_official.sh:20`
- Level 2 多算法 label：`run_level2_multiseed_all_algos_official.sh:59`
- semi-random terrain 配置：`multiagent/scenarios/paper3d_terrain_energy.py:140`
- terrain variant / jitter / height scale：`multiagent/scenarios/paper3d_terrain_energy.py:158`
- terrain seed 生成：`multiagent/scenarios/paper3d_terrain_energy.py:573`
- terrain regeneration：`multiagent/scenarios/paper3d_terrain_energy.py:1233`
- dynamic obstacle deterministic seed / refresh：`multiagent/scenarios/paper3d_terrain_energy.py:2928`

### 论文表达

可以写成：

> A hierarchical 3D evaluation benchmark is constructed to separate fixed-map mechanism validation, stochastic-obstacle evaluation, and semi-random terrain-family stress testing.

### 注意事项

这是 benchmark/evaluation contribution。
如果论文篇幅紧张，不要让它抢走 dual-semantic 方法的主线。

## 11. 结果分析创新：Deployment-Side Diagnostic Metrics

### 创新内容

评估代码不仅记录 reward，还记录部署侧指标：

- team success；
- any-agent success；
- two-agent success；
- per-agent success；
- collision-free rate；
- final goal distance；
- terrain penetration；
- obstacle collision；
- inter-agent collision；
- minimum inter-agent clearance。

### 代码依据

- summary metric 初始化：`evaluate_optimized.py:604`
- summary metric 聚合：`evaluate_optimized.py:786`
- per-episode min clearance 和 agent success flags：`evaluate_optimized.py:3492`
- per-episode 评估详情保存：`evaluate_optimized.py:4508`

### 论文表达

可以写成：

> Deployment performance is interpreted using task-completion and safety diagnostics rather than reward alone, including team success, partial arrival, collision-free behavior, final goal distance, penetration, and clearance.

### 论文价值

这可以帮助解释：

- 高 reward 但 team success 为 0 的方法不能被称为任务最优；
- 低 team success 可能来自 partial completion、non-arrival、safety burden 或 mixed failure；
- Level 2 / Level 3 的结果应该结合 success、collision-free、distance 一起判断。

### 注意事项

如果某些旧 Level 3 日志只有 aggregate collision / penetration，而没有细分 obstacle/terrain/inter-agent 来源，论文中不能硬说具体失败来源。
应使用 coarse failure pattern。

## 12. 工程支撑一：Vectorized Training / Reward / Scenario Infrastructure

### 内容

代码中有较多 vectorized 环境、reward、action selection 逻辑：

- vectorized scenario；
- batch reward calculation；
- vectorized APF/action selection；
- 多环境并行训练；
- reward/collision/penetration counter 缓存。

### 代码依据

- vectorized scenario 定义：`multiagent/scenarios/paper3d_terrain_vectorized.py:34`
- vectorized reward calculator 初始化：`multiagent/scenarios/paper3d_terrain_vectorized.py:77`
- episode counter reset：`multiagent/scenarios/paper3d_terrain_vectorized.py:162`
- batch reward 计算：`multiagent/scenarios/paper3d_terrain_vectorized.py:425`
- reward weight sync：`multiagent/scenarios/paper3d_terrain_vectorized.py:605`

### 论文定位

这类内容建议放在 implementation details 或 appendix。
它可以证明实验规模和复现性，但不建议作为 abstract 级别主创新。

## 13. 工程支撑二：Crash Recovery and Incomplete-Run Handling

### 内容

Level 2 official 脚本中有训练完成检测、official eval 完成检测、未完成内容清理等逻辑：

- 检查 `results.json`；
- 检查 checkpoint episode；
- 检查 official evaluation episode 数；
- 对未完成实验删除 incomplete artifacts；
- 支持崩溃后重新补齐未完成实验。

### 代码依据

- model completed 检查：`run_level2_multiseed_all_algos_official.sh:160`
- official eval completed 检查：`run_level2_multiseed_all_algos_official.sh:195`
- incomplete artifact 删除：`run_level2_multiseed_all_algos_official.sh:257`
- official eval 运行：`run_level2_multiseed_all_algos_official.sh:297`

### 论文定位

这是工程可靠性，不建议作为算法创新。
可以在 reproducibility 或 experimental protocol 中简短说明。

## 14. 工程支撑三：Training Metadata and Reproducibility

### 内容

训练时会记录大量配置：

- actor objective mode；
- action semantics mode；
- target reconstruction；
- dual-Q/separated-gradient/hybrid alpha；
- replay semantic settings；
- 环境与训练超参数。

### 代码依据

- training hyperparameter capture：`paper3d_train_optimized.py:15104`
- CLI flags：`paper3d_train_optimized.py:20365`

### 论文定位

这可以作为 reproducibility 支撑。
适合放在实验设置、appendix 或 artifact description 中。

## 15. 建议的论文主贡献写法

建议把主贡献压缩成 4 条。

### Contribution 1

提出 APF-corrected MARL 中的 action semantic mismatch 问题。

推荐英文：

> We identify an action-semantic mismatch in APF-corrected multi-agent reinforcement learning, where raw policy intentions and safety-corrected execution actions play different roles in replay, critic learning, and target construction.

### Contribution 2

提出 dual-semantic replay 和 execution-consistent target construction。

推荐英文：

> We propose a dual-semantic replay and target construction mechanism that stores both raw and corrected actions and reconstructs corrected target actions under the next-state APF geometry.

### Contribution 3

提出 dual-Q / separated-gradient actor-critic 设计。

推荐英文：

> We design a dual-head critic and semantic actor-objective routing strategy to separately evaluate raw motion intention and APF-parameterized execution behavior.

### Contribution 4

构建 focused semantic ablation 和 matched official evaluation protocol。

推荐英文：

> We provide focused semantic ablations and matched official evaluations to isolate replay-level semantic collapse, target-side execution inconsistency, and deployment-side degradation under controlled terrain-obstacle conditions.

## 16. 哪些内容不建议作为主创新

以下内容有价值，但不建议作为论文最大创新点：

- CUDA 显存清理；
- `MAX_PARALLEL` 并行脚本；
- 断点后清理 incomplete runs；
- dashboard 图；
- reward 曲线绘图；
- 单纯的 APF 后处理；
- 单纯的 MAPPO/MADDPG baseline 接入；
- vectorized reward 的工程加速。

这些可以放在：

- implementation details；
- reproducibility；
- appendix；
- supplementary material。

## 17. 最强论文叙事主线

推荐整篇论文围绕这一句话展开：

> APF-based correction improves deployment safety, but it also changes the semantics of the action used by the environment. Therefore, APF-corrected MARL should not treat raw policy actions and corrected execution actions as a single replay/critic/target channel. The proposed method preserves and aligns these dual semantics throughout replay storage, critic learning, actor optimization, and target construction.

中文理解：

> APF 可以提高安全性，但它改变了动作语义。传统 replay/critic/target 把 raw action 和 corrected action 混成一个动作通道，会造成训练-执行错配。本文的核心贡献就是在 replay、critic、actor objective 和 TD target 中显式保留并对齐这两种动作语义。

这是当前代码中最有创新性、最容易写成论文核心贡献的内容。

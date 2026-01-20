#!/bin/bash

# 确保从脚本所在目录运行，避免相对路径问题
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MADDPG优化版本快速启动脚本

# 🔧 关键修复：清理可能干扰训练的环境变量残留（避免之前运行的shell变量影响）
unset TF_GPU_ALLOCATOR
unset CUDA_LAUNCH_BLOCKING  
unset TF_SYNC_ON_FINISH

# 🔧 关键修复：在脚本开头设置 TF_XLA_FLAGS，禁用 Auto JIT
# 必须在所有其他设置之前，确保在 Python 导入 TensorFlow 之前生效
# 说明：某些 TensorFlow 版本不再读取 XLA_FLAGS，而仅读取 TF_XLA_FLAGS。
#       若继续设置 XLA_FLAGS 可能触发 "Unknown flags in XLA_FLAGS" 并导致进程中止。
# 🚨 强制清除 XLA_FLAGS（TensorFlow 2.x 主要读取 TF_XLA_FLAGS，但XLA_FLAGS仍会被使用）
# 🔧 关键修复：不设置为空，而是在Python代码中设置稳定的XLA配置
# 原因：需要在启用XLA时设置稳定的配置（禁用Triton autotuner等）
# 注意：Python代码会在启用XLA时自动设置XLA_FLAGS
unset XLA_FLAGS 2>/dev/null || true
# 🚨 修复：移除不支持的 --xla_gpu_enable_triton_gemm flag（当前TF版本不识别）
# 注意：Triton GEMM 可能默认禁用，或者需要通过其他方式控制
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# 检测GPU是否可用（存在 nvidia-smi 且可列出设备，且未显式屏蔽 CUDA_VISIBLE_DEVICES）
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
            HAS_GPU=1
        fi
    fi
fi

if [ "$HAS_GPU" -eq 1 ]; then
    # 🔧 XLA配置：已通过 TF_XLA_FLAGS 禁用 Auto JIT
    # 注意：Triton GEMM 控制在当前TF版本中不可用，依赖默认行为
    : # 空操作，保持条件结构完整
fi

echo "======================================"
echo "MADDPG 优化版训练启动器 (已修复)"
echo "======================================"

# 设置默认参数
EPISODES=${1:-500}    
BATCH_SIZE=${2:-1024}  # 🚀 提升批次大小到4096，充分利用GPU并行能力
EXP_NAME=${3:-"双头q改进测试、变FR、改动作添加噪声、随机地图、复杂4_exp"}
USE_WEIGHTED_REWARD=${4:-1}  # 新增：是否使用分项加权求和奖励机制（1=启用，0=禁用）
ALGORITHM=${5:-"matd3"}     # 新增：选择训练算法（maddpg或matd3）
RESUME_MODEL=${6:-""}       # 🔧 新增：持续训练模型路径（可选，指定要恢复的模型目录）

# 🔧 持续训练模型配置（优先级：命令行参数 > 环境变量 > 空）
# 如果通过环境变量指定，优先使用环境变量
if [ -z "$RESUME_MODEL" ] && [ -n "${RESUME_MODEL_ENV:-}" ]; then
    RESUME_MODEL="${RESUME_MODEL_ENV}"
fi
if [ -z "$RESUME_MODEL" ] && [ -n "${CHECKPOINT_MODEL:-}" ]; then
    RESUME_MODEL="${CHECKPOINT_MODEL}"
fi

# 生成时间戳，确保每次训练都有唯一的目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME_WITH_TIMESTAMP="${EXP_NAME}_${TIMESTAMP}"

# PF元优化基准输出路径（默认写入当前实验日志目录）
if [ -z "${PF_META_BASELINE_FILE:-}" ]; then
    export PF_META_BASELINE_FILE="logs/$EXP_NAME_WITH_TIMESTAMP/pf_meta_baseline.json"
fi

# 🔧 随机种子初始化（在显示参数前生成，确保SEED已设置）
if [ -z "${SEED:-}" ]; then
    # 生成随机种子（基于时间戳，范围：100000-999999）
    # 使用纳秒级时间戳的后6位，确保每次运行都有不同的种子
    RANDOM_SEED=$(($(date +%s%N) % 900000 + 100000))
    export SEED=$RANDOM_SEED
    SEED_SOURCE="自动生成"
else
    SEED_SOURCE="环境变量（复现模式）"
fi

echo "" 
echo "训练参数:"
echo "  - 训练回合数: $EPISODES"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 实验名称: $EXP_NAME"
echo "  - 带时间戳的实验名称: $EXP_NAME_WITH_TIMESTAMP"
echo "  - 使用分项加权奖励: $USE_WEIGHTED_REWARD"
echo "  - 训练算法: $ALGORITHM"
echo "  - 随机种子: $SEED ($SEED_SOURCE)"
if [ -n "$RESUME_MODEL" ]; then
    echo "  - 🔧 持续训练: 从模型恢复 - $RESUME_MODEL"
else
    echo "  - 🔧 持续训练: 否（新训练）"
fi
echo ""

echo ""

# 创建必要的目录
mkdir -p logs/$EXP_NAME_WITH_TIMESTAMP
mkdir -p models/$EXP_NAME_WITH_TIMESTAMP

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# === 学习率参数 ===
# 🚨 修复训练波动：降低Critic学习率，提高稳定性
export LEARNING_RATE_ACTOR=${LEARNING_RATE_ACTOR:-0.0008}   # 🔧 保持Actor学习率
                                                              # 原因：0.0003是合理的Actor学习率
export LEARNING_RATE_CRITIC=${LEARNING_RATE_CRITIC:-0.002}  # 🚨 关键修复：提高Critic学习率（0.0005→0.001）
                                                               # 原因：之前学习率太小导致更新缓慢，Critic Loss过小（~0.01）
                                                               # 修复：提高到0.001，是Actor的2倍，加快Critic更新速度
                                                               # 预期：Q值估计更快收敛，Critic Loss提升到0.1-1.0范围

# === 🔧 Loss函数参数（增强梯度信号）===
export HUBER_DELTA=${HUBER_DELTA:-2.0}                 # 🔧 Huber Delta参数（从2.5降低到2.0）
                                                        # 原因：reward_scale改为1/1000后，Q值范围~[-500,500]，TD误差~2-20
                                                        # 作用：delta=1.5使TD误差>1.5进入线性区，<1.5在平方区，让Loss对误差更敏感
                                                        # 效果：梯度增强，网络更新更快，Critic Loss从0.01-0.015提升到0.1-1.0
                                                        # 建议范围：1.0-2.0（根据Q值范围调整，通常为Q值范围的0.3-0.5%）

# === 🔥 学习率衰减参数（保持学习能力）===
export LR_DECAY_ENABLED=${LR_DECAY_ENABLED:-1}                          # 启用学习率衰减，但设置更高的最小学习率
export LR_DECAY_STEPS=${LR_DECAY_STEPS:-20000}                           # 🔧 衰减步数放宽到20000，减缓衰减速度
export LR_DECAY_RATE=${LR_DECAY_RATE:-0.9996}                           # 🔧 衰减更慢（0.999→0.9995），每20000步衰减0.05%
export LR_STAIRCASE=${LR_STAIRCASE:-1}                                  # 是否使用阶梯式衰减（1=阶梯，0=平滑）
export LR_MIN_ACTOR=${LR_MIN_ACTOR:-0.00030}                            # 🔧 提高Actor最小学习率（0.00008→0.00015），保持后期学习能力
export LR_MIN_CRITIC=${LR_MIN_CRITIC:-0.00040}                          # 🔧 提高Critic最小学习率（0.00010→0.00020），保持后期学习能力

# === 网络架构参数 ===
# 🔙 恢复到原先结构（不改动逻辑，仅还原规模）
export ACTOR_HIDDEN=${ACTOR_HIDDEN:-"256,256,256"}     # 🔧 恢复到3层×256，平衡容量和稳定性
export CRITIC_HIDDEN=${CRITIC_HIDDEN:-"256,256,256"}   # 🚨 修复训练波动：从4层降到3层，降低网络复杂度（参数量减少约50%）
                                                         # 原因：Critic网络过于复杂（约3.2M参数），导致训练不稳定
                                                         # 简化后参数量约1.6M，仍然足够表达复杂策略  

# === 权重约束参数 ===
# 🔧 放宽权重约束：允许网络有更强的表达能力，避免过度限制学习
export MAX_WEIGHT_THRESHOLD=${MAX_WEIGHT_THRESHOLD:-0.999}   # 从0.15放宽到0.8，给予网络更大学习空间
export WEIGHT_SCALING_FACTOR=${WEIGHT_SCALING_FACTOR:-0.999}  # 从0.6提高到0.9，更温和的缩放

# === 🚀 加速配置（默认：关闭所有JIT，避免内存对齐问题）===
export AMP_MODE=off
export JIT_COMPILE=${JIT_COMPILE:-0}            # 🔧 修复：默认禁用（1→0），避免与XLA叠加导致内存对齐问题
export PF_JIT=0                                  # 势场JIT编译（默认禁用）

# === 🔧 XLA Global 配置（默认启用，异步执行模式）===
# 配置说明：
#   - XLA Global：启用全局XLA编译，显著提升性能
#   - 异步执行：使用异步CUDA执行，最大化GPU利用率
#   - 缓存清理：已优化到每10回合清理一次，防止长时间运行后内存累积
# 注意事项：
#   - 长时间运行（500+回合）后可能出现 CUDA_ERROR_ILLEGAL_ADDRESS
#   - 已通过频繁的GPU缓存清理（每10回合）来缓解此问题
export XLA_GLOBAL=${XLA_GLOBAL:-1}               # 默认启用（1），XLA Global + 异步执行
export CPU_THREADS=${CPU_THREADS:-12}              # 默认12线程（Zen4 7945HX 实测更稳）
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-0.3}  # 进度条刷新间隔（秒）
export TQDM_NCOLS=${TQDM_NCOLS:-100}              # 进度条列宽
export GPU_ID=${GPU_ID:-0}
export NUM_ENVS=${NUM_ENVS:-1}                     # 🔧 单环境运行，避免并行环境引入的干扰和复杂性
export TQDM_DISABLE=${TQDM_DISABLE:-1}            # 完全禁用tqdm进度条（默认安静）
export SAVE_POSITIONS=${SAVE_POSITIONS:-1}        # 默认关闭保存位置
export USE_LITE_BUFFER=${USE_LITE_BUFFER:-1}      # 默认启用内存友好的sumTree 优先经验回放
## 移除：历史TF Buffer开关（实现已移除）
# export USE_TF_BUFFER=${USE_TF_BUFFER:-0}
export PER_ENABLED=${PER_ENABLED:-1}              # 启用优先经验回放（对 Lite/ReplayBuffer 生效）
export BUFFER_SIZE=${BUFFER_SIZE:-500000}   # 经验缓冲区大小（增加到50万）
export UPDATE_RATE=${UPDATE_RATE:-40}       # 🔧 回滚到30：UPDATE_RATE不是主要问题，奖励尺度才是
export ACTOR_UPDATE_DELAY=${ACTOR_UPDATE_DELAY:-2}  # 🚨 关键修复：提高Actor更新延迟（从1到2），减少训练频率，提高训练稳定性
                                                      # 问题：每步都更新Actor导致训练不稳定，Loss波动大
                                                      # 修复：每2步更新一次Actor，让Critic有更多时间稳定Q值估计
                                                      # 预期：Loss波动减小，训练更稳定

# === MATD3算法特有参数 === 
export POLICY_NOISE=${POLICY_NOISE:-0.28}            # 🔧 修复：降低目标策略平滑噪声（从0.50降到0.25），提高稳定性
export NOISE_CLIP=${NOISE_CLIP:-0.32}                # 目标策略噪声裁剪幅度（MATD3）- 降低以减少数值波动
export POLICY_FREQ=${POLICY_FREQ:-1}                # 🔧 修复：降低Actor更新延迟（从2降到1），提高Actor更新频率（MATD3）

export BUFFER_DTYPE=${BUFFER_DTYPE:-fp32}         # 롤백: fp16 -> fp32 (안정성 우선)
export PER_REPLACE=${PER_REPLACE:-0}             # PER采样是否放回: 1/0（保持原有采样模式）
## === PER增强参数（新） ===
export PER_UNIFORM_MIX=${PER_UNIFORM_MIX:-0.45}      # PER与均匀采样混合比例（保持原值，不改变分布）
export PER_TD_WEIGHT=${PER_TD_WEIGHT:-0.80}           # 优先级中TD误差项权重
export PER_REWARD_WEIGHT=${PER_REWARD_WEIGHT:-0.12}   # 优先级中奖励幅值项权重
export PER_AGE_DECAY=${PER_AGE_DECAY:-0.95}           # 年龄衰减系数(0-1]，1不衰减
export MEM_DEBUG=${MEM_DEBUG:-0}                  # 打印内存/显存
export DEBUG_PF_FORCES=${DEBUG_PF_FORCES:-0}       # 调试势场力量级
export PROFILING=${PROFILING:-0}                  # 启用剖析（默认关闭）
export STEP_WARN_S=${STEP_WARN_S:-5}              # 单步耗时告警阈值
## 清理：脚本内未使用 STEP_WARN_ENABLE（仅保留阈值 STEP_WARN_S 透传）
# export STEP_WARN_ENABLE=${STEP_WARN_ENABLE:-0}
export STACK_DUMP_TIMEOUT=${STACK_DUMP_TIMEOUT:-0} # 超时打印堆栈（默认禁用）
export STACK_WATCHDOG_ENABLE=${STACK_WATCHDOG_ENABLE:-0} # 显式看门狗开关（默认禁用）
export QUIET_OUTPUT=${QUIET_OUTPUT:-0}            # 安静输出（1=安静, 0=详细）（启用详细输出以检查数据质量）
export DEBUG_EFF_SAMPLES=${DEBUG_EFF_SAMPLES:-0}   # 打印每批有效样本计数/批大小（调试数值稳定性）
## 清理：训练脚本未读取 DEBUG_ENV_OUTPUT，请使用 QUIET_OUTPUT/SUPPRESS_ENV_DEBUG 控制
# export DEBUG_ENV_OUTPUT=${DEBUG_ENV_OUTPUT:-0}
export LOG_INTERVAL_STEPS=${LOG_INTERVAL_STEPS:-5000}  # 训练步骤日志输出间隔（默认5000步，减少I/O）
export EPISODE_LOG_INTERVAL=${EPISODE_LOG_INTERVAL:-20}  # 回合日志输出间隔（默认10回合）
export GRAVITY=${GRAVITY:-0.0}                   # 环境重力（-Z）- 标准重力场模拟
export CONTROL_ACCEL_GAIN=${CONTROL_ACCEL_GAIN:-1.0} # 🚨 关键修复：提高到12.0，增强垂直控制（8.5→12.0）
                                                      # 原因：增益8.5时，需要1.15N才能抵消重力，限制了向上加速能力
                                                      # 新值12.0：降低重力相对影响，Actor更容易控制高度
export DAMPING=${DAMPING:-0.12}                       # 速度阻尼系数，减少振荡 - 降低阻尼，允许更多运动

# === 四旋翼动力学模型配置 ===
# 🔧 四旋翼动力学：启用后使用完整的四旋翼动力学模型（姿态、角速度、电机转速）
# 默认：0（未启用，使用简化的质点模型）
# 启用：1（使用四旋翼动力学模型，包含姿态、角速度、电机分配等）
# 说明：
#   - 启用后，网络输出仍然是世界坐标系期望加速度 a_cmd = (ax, ay, az)
#   - 环境内部通过四旋翼动力学逆解成电机输入，再用刚体6自由度模型更新状态
#   - 包含完整状态：位置、速度、姿态（四元数）、角速度、电机转速
#   - 在理想姿态跟踪假设下，飞行轨迹应尽可能接近原本质点模型
export USE_QUADROTOR_DYNAMICS=${USE_QUADROTOR_DYNAMICS:-1}  # 是否启用四旋翼动力学模型（默认0=质点模型，1=四旋翼模型）
export QUADROTOR_PSI_CMD=${QUADROTOR_PSI_CMD:-0.0}          # 期望偏航角（rad），默认0（保持初始朝向）
                                                             # 说明：四旋翼动力学模型会尝试保持该偏航角
                                                             # 建议范围：0.0-2π（0表示保持初始朝向）
export QUADROTOR_ATTITUDE_RESPONSE_TIME=${QUADROTOR_ATTITUDE_RESPONSE_TIME:-0.05}  # 🔧 启用真实动力学约束：姿态响应时间0.1秒
                                                             # 说明：>0时启用姿态响应延迟，模拟真实姿态控制器的延迟特性
                                                             # 建议范围：0.0-0.5（0.1表示姿态响应时间约0.1秒）
                                                             # 🔧 关键：设置为0时，姿态能瞬时达到期望值，没有动力学约束
                                                             # 设置为>0时，姿态需要时间响应，真正限制智能体的行为
                                                             # ✅ 修复：从0.0改为0.1，启用真实动力学约束，让智能体受限于动力学
export ACTION_RANGE_X=${ACTION_RANGE_X:-2.5}      # 🔧 增加动作范围（2.0→2.5），允许更大的加速度输出
export ACTION_RANGE_Y=${ACTION_RANGE_Y:-2.5}     # 🔧 增加动作范围（2.0→2.5），允许更大的加速度输出
export ACTION_RANGE_Z=${ACTION_RANGE_Z:-2.2}     # 🔧 增加Z轴动作范围（1.8→2.2），提高向上飞行能力
                                                                                                 # 问题：网络输出范围依然不大，无法直接给出较大的加速度效果
                                                                                                 # 修复：增加动作范围，配合降低的正则化，允许更大的加速度输出
## 已弃用：垂直力抑制参数移除
export REWARD_POS_SCALE=${REWARD_POS_SCALE:-1.0}   # 🔧 修复奖励累积：降低到1.0（2.0→1.0），避免过度放大奖励
                                                      # 原因：2.0的缩放会将所有正奖励放大2倍，导致奖励值虚高
                                                      # 新值1.0：不进行缩放，保持原始奖励值
export REWARD_NEG_SCALE=${REWARD_NEG_SCALE:-1.0}    # 🔧 修复奖励累积：降低到1.0（2.5→1.0），避免过度放大负奖励
                                                      # 原因：2.5的缩放会将所有负奖励放大2.5倍，导致惩罚过重
                                                      # 新值1.0：不进行缩放，保持原始奖励值

# 动作OU噪声参数（已替换原高斯白噪声）：std_dev/decay/min 与训练脚本一致
# 🔧 修复：进一步降低OU噪声，减少训练阶段智能体到处跑的问题
# 问题：训练阶段OU噪声过大，导致动作波动大，智能体无法稳定学习
# 解决方案：降低噪声初始值和最小值，减少对动作的干扰
export NOISE_SCALE=${NOISE_SCALE:-0.35}           # 🔧 降低初始噪声（0.45→0.35），减少过度探索
                                                   # 原因：FR降低时探索过强，动作抖动导致碰撞增多
                                                   # 新值0.35：减小噪声幅度，让策略更可控
export NOISE_DECAY=${NOISE_DECAY:-0.9995}        # 🔧 加快噪声衰减（0.999→0.9995），让训练后期更稳定
                                                   # 原因：奖励曲线有很好的峰值但没有稳住，说明噪声衰减太慢导致后期仍有过大探索
                                                   # 新值0.9995：加快衰减，让训练后期噪声更小，动作更稳定
export NOISE_MIN=${NOISE_MIN:-0.05}              # 🔧 进一步降低最小噪声（0.05→0.03），减少后期扰动
                                                   # 原因：后期仍存在不必要的探索，影响避障稳定性
                                                   # 新值0.03：更平滑的后期行为

# === 🔧 自适应调整参数（修复奖励下降问题）===
export ADAPTIVE_PATIENCE=${ADAPTIVE_PATIENCE:-15}              # 🔧 修复：降低耐心（20→15），更早触发探索增强
export ADAPTIVE_LR_DECAY=${ADAPTIVE_LR_DECAY:-0.95}            # 🚨 修复学习率过低：降低衰减力度（0.9→0.95），每次降低5%而非10%
                                                                 # 问题：当前Actor LR已降到0.000063，接近最小值0.00002，导致无法跳出局部最优
                                                                 # 解决：降低衰减力度，让学习率下降更慢，保持足够的学习能力
export ADAPTIVE_MIN_LR=${ADAPTIVE_MIN_LR:-5e-5}                # 🚨 修复学习率过低：提高最小学习率（2e-5→5e-5），保持学习能力
                                                                 # 问题：最小学习率2e-5过低，导致网络无法有效学习
                                                                 # 解决：提高到5e-5，确保网络始终有足够的学习能力
export ADAPTIVE_NOISE_MAX=${ADAPTIVE_NOISE_MAX:-0.4}           # 🔧 修复噪声过大：降低噪声上限（0.8→0.4），避免动作波动过大
                                                               # 说明：动作范围2.0，噪声0.4占20%，0.8占40%过大
                                                               # 建议：噪声上限应不超过动作范围的20-25%
export ADAPTIVE_NOISE_SMOOTH=${ADAPTIVE_NOISE_SMOOTH:-0.4}     # 噪声平滑更新系数（0-1）
export NOISE_RESTART_INTERVAL=${NOISE_RESTART_INTERVAL:-20}    # 🔧 修复局部最优：更频繁重启噪声（30→20），保持探索活力

# === 分阶段探索策略参数 ===
# 🔧 修复过拟合：提高随机动作概率，增强探索能力
export RANDOM_ACTION_PROB=${RANDOM_ACTION_PROB:-0.12}         # 🔧 预热随机动作降低到12%，避免过度发散
export RANDOM_ACTION_PROB_TRAINING=${RANDOM_ACTION_PROB_TRAINING:-0.00}  # 🔧 训练随机动作降低到5%，减少不必要探索
## 清理：训练脚本未读取以下两项（已统一采用"预热阶段/训练阶段"）：
# export EXPLORATION_PHASE_SWITCH=${EXPLORATION_PHASE_SWITCH:-"warmup"}
# export EXPLORATION_SWITCH_THRESHOLD=${EXPLORATION_SWITCH_THRESHOLD:-10}

# === 位置和地形控制参数 ===
export DYNAMIC_FIRST_TIME=${DYNAMIC_FIRST_TIME:-1}    # 🔧 启用动态首次：第一回合生成固定位置，后续回合使用该固定位置
                                                        # 配合 USE_FIXED_POSITIONS=1 使用，实现固定起点训练
export WARMUP_RANDOM_INIT=${WARMUP_RANDOM_INIT:-0}    # 🔧 临时禁用，避免多进程竞态条件
export TERRAIN_COMPLEXITY_LEVEL=${TERRAIN_COMPLEXITY_LEVEL:-3} # 地形复杂度等级 (1-4) - 默认等级2
export MOUNTAIN_MIN_DISTANCE=${MOUNTAIN_MIN_DISTANCE:-55}      # 山峰之间的最小距离（单位：地图单位，建议范围20-80）
export MAP_SIZE=${MAP_SIZE:-200}
export MIN_START_GOAL_DIST=${MIN_START_GOAL_DIST:-75.0}        # 起点与目标的最小水平距离
export MAX_START_GOAL_DIST=${MAX_START_GOAL_DIST:-120.0}       # 起点与目标的最大水平距离（控制随机幅度）
export START_POS_MARGIN=${START_POS_MARGIN:-5.0}               # 起点生成的地图边缘安全距离
export START_ALTITUDE_OFFSET=${START_ALTITUDE_OFFSET:-12.0}     # 🔧 起点离地高度：7米（平坦地形上方5-10米区间）
                                                                # 说明：让智能体从较低位置起飞，学习向上飞行轨迹
export GOAL_ALTITUDE=${GOAL_ALTITUDE:-25.0}                     # 🔧 目标离地高度：12米（略高于理想区间上限，引导向上飞）
                                                                # 说明：与HEIGHT_IDEAL_MAX(15米)配合，让智能体学习在合理高度完成任务
export START_POS_MAX_TRIALS=${START_POS_MAX_TRIALS:-2000}      # 起点采样最大尝试次数

# === 分项加权求和奖励权重参数 ===
# 这些参数控制分项加权求和奖励机制中各项奖励的权重
# 仅在 USE_WEIGHTED_REWARD=true 时生效
# 调优建议：权重表示强度，分项本身决定正/负；对返回负值表示惩罚的分项（如停滞/碰撞/低空），权重应为正以保留惩罚方向

# === 🎯 完整加权奖励分项：恢复所有奖励组件 ===
# 核心奖励1：距离奖励（基础导航）
export DISTANCE_WEIGHT=${DISTANCE_WEIGHT:-0.6}          # 🔧 降低距离奖励（0.8→0.6），减少奖励波动，让智能体更多学习躲避地形
                                                          # 原因：距离奖励每步累积，权重过高会导致奖励值波动过大
                                                          # 新策略：降低目标导向奖励，平衡避障与目标导向
                                                          # 新值0.6：降低25%，减少每步奖励，使奖励更平滑

# 核心奖励2：接近目标奖励（强化目标导向）
export APPROACH_WEIGHT=${APPROACH_WEIGHT:-0.8}          # 🔧 降低接近奖励权重（1.0→0.8），减少奖励波动，让智能体更多学习躲避地形
                                                        # 原因：接近奖励每步都计算，持续接近目标会累积大量奖励，导致奖励值波动大
                                                        # 新策略：降低目标导向奖励，平衡避障与目标导向
                                                        # 新值0.8：降低20%，减少奖励波动，让智能体更多关注地形避障

# 核心奖励3：停滞惩罚（防止卡住）
export STATIONARY_WEIGHT=${STATIONARY_WEIGHT:-0.3}      # 🔧 大幅降低停滞惩罚到0.3
                                                        # 作用：防止智能体停滞不前
                                                        # 建议范围：0.3-2.0（正数增大惩罚强度）

# 核心奖励4：成功奖励（最终目标）
export SUCCESS_WEIGHT=${SUCCESS_WEIGHT:-2.0}            # 🚨 修复"直冲"问题：降低成功奖励权重（4.0→2.0），平衡安全与成功
                                                        # 作用：明确的任务完成信号，但不应该让智能体为了成功而忽略安全
                                                        # 问题：4.0权重导致成功奖励(60000)远大于碰撞惩罚(62.5)，智能体"宁愿碰撞也直冲"
                                                        # 新值2.0：成功奖励降低到12000-16000，与碰撞惩罚更平衡
                                                        # 建议范围：1.5-2.5（平衡安全与成功）

# 核心奖励5：碰撞惩罚（安全约束）
export COLLISION_WEIGHT=${COLLISION_WEIGHT:-4.0}        # 🚨🚨🚨 加强碰撞避免：提高碰撞权重（2.0→4.0），确保无碰撞回合奖励更高
                                                        # 问题诊断：2.0权重导致碰撞惩罚不够强，有碰撞的回合仍可能获得高奖励
                                                        # 修复后效果：4.0权重 × 80惩罚值 = 单次碰撞-320
                                                        #           即使500次碰撞，总惩罚也有-16万，足以让有碰撞的回合奖励低于无碰撞回合
                                                        # 策略：让碰撞有足够代价，确保团队成功（无碰撞）的回合奖励最高
                                                        # 建议范围：3.5-5.0（确保碰撞惩罚足够强）
export COLLISION_WEIGHT_START_FACTOR=${COLLISION_WEIGHT_START_FACTOR:-0.75}   # 初期碰撞权重倍率
export COLLISION_WEIGHT_FULL_PCT=${COLLISION_WEIGHT_FULL_PCT:-1.0}           # 线性提升到完整权重的训练进度
                                                        # 建议范围：0.3-1.0

# 🔧 新增：碰撞次数减少奖励（鼓励减少碰撞）
export COLLISION_REDUCTION_WEIGHT=${COLLISION_REDUCTION_WEIGHT:-0.8}  # 碰撞次数减少奖励权重
                                                        # 作用：如果当前回合碰撞次数比上一回合少，给予奖励
                                                        # 奖励值 = (上一回合碰撞数 - 当前回合碰撞数) / 上一回合碰撞数
                                                        # 建议范围：0.3-1.0（正值奖励减少碰撞）

# === 探索与导航相关奖励 ===
export EXPLORATION_WEIGHT=${EXPLORATION_WEIGHT:-1.0}     # 🔧 修复局部最优：提高到1.0（0.6→1.0），激励探索新路径
                                                        # 作用：鼓励智能体探索环境，避免重复路径和原地悬停
                                                        # 问题：0.6的权重导致探索奖励(+3.0)无法抵消碰撞恐惧(-377.0)，智能体选择"安全绕圈"而非"探索接近"
                                                        # 新值1.0：让新格子奖励达到约5.0，提供足够的探索动力
                                                        # 建议范围：0.8-1.5

export DIRECTION_WEIGHT=${DIRECTION_WEIGHT:-0.3}         # 🔧🔧 降低方向奖励（0.8→0.3），避免诱导"直线冲刺"行为
                                                        # 作用：方向奖励过高会让智能体只关注"朝向目标"而忽略"绕路避障"
                                                        # 新策略：保留方向引导，但不能成为主导因素
                                                        # 增强特性：已提高对齐奖励和速度奖励系数，添加距离相关倍数
                                                        # 新值0.8：配合增强的方向奖励计算，提供更强的方向引导
                                                        # 建议范围：0.8-1.2

export DEVIATION_WEIGHT=${DEVIATION_WEIGHT:-0.35}         # 偏离奖励：奖励贴近起点-目标直线
                                                        # 作用：鼓励直线路径，减少侧向偏离
                                                        # 建议范围：0.5-2.0

# === 轨迹平滑/最小拐弯角奖励 ===
# 通过约束相邻两步速度方向的夹角，鼓励更平滑的轨迹，抑制“过山车式”剧烈转向
export TURN_SMOOTH_WEIGHT=${TURN_SMOOTH_WEIGHT:-0.3}     # 高度平滑奖励权重（0=关闭；建议0.1-0.8）- 奖励高度（Z坐标）的平滑变化

# === 区域与行为奖励 ===
export START_AREA_WEIGHT=${START_AREA_WEIGHT:-0.3}       # 起始区域奖励：鼓励快速离开起点
                                                        # 作用：避免在起点徘徊
                                                        # 建议范围：0.2-0.8

export ENERGY_WEIGHT=${ENERGY_WEIGHT:-0.2}               # 能量效率奖励：提高权重以强化能量项影响
                                                        # 作用：鼓励平滑运动，减少不必要的加速
                                                        # 建议范围：0.1-0.5

# === 高度控制奖励 ===
# 🔧 修复：降低高度奖励权重，避免过度惩罚导致奖励信号混乱
export HEIGHT_WEIGHT=${HEIGHT_WEIGHT:-3.5}               # 🚨🚨🚨 大幅提高高度奖励（1.50→3.5），引导安全飞行高度
                                                        # 问题诊断：APF让智能体贴地飞行（距地形1-3m），频繁碰撞
                                                        # 根本原因：高度奖励太弱(1.5)，无法对抗APF的"贴地引导"
                                                        # 解决策略：提高高度奖励权重，鼓励飞行在15-75m理想区间
                                                        # 建议范围：3.0-5.0（必须强到能引导飞行高度）
export HEIGHT_REWARD_ENABLED=${HEIGHT_REWARD_ENABLED:-1} # 1=启用高度奖励，0=完全关闭
export HEIGHT_IDEAL_MIN=${HEIGHT_IDEAL_MIN:-15.0}         # 🔧 理想高度下限（默认5.0m）
                                                          # 说明：允许智能体在地形起伏处飞行
export HEIGHT_IDEAL_MAX=${HEIGHT_IDEAL_MAX:-75.0}        # 🚨 扩大理想高度上限（35→60m），允许爬升到山峰高度
                                                          # 原因：当前35米限制导致智能体不敢飞高，但目标可能在50+米处
                                                          # 新值60米：覆盖大部分山峰高度，不惩罚合理爬升

# === 避障相关奖励 ===
export LATERAL_WEIGHT=${LATERAL_WEIGHT:-1.0}             # 侧向绕行奖励：奖励合理的避障轨迹
                                                        # 作用：在保持目标导向的同时绕过障碍
                                                        # 建议范围：0.3-1.0

export CLEARANCE_WEIGHT=${CLEARANCE_WEIGHT:-8.0}          # 🚨 保持净空奖励权重（不降低），但改变计算方式
                                                        # 问题诊断：基于绝对距离的sigmoid函数导致"保持安全距离但不接近目标"刷分
                                                        # 根本原因：只要距离>安全距离就给正奖励，无论距离是否变化
                                                        # 解决策略：改为基于距离变化的计算方式
                                                        #   - 距离增加时：给正奖励（鼓励避障）
                                                        #   - 距离减少时：给负奖励（惩罚接近危险）
                                                        #   - 距离不变时：给零奖励（避免刷分）
                                                        # 优势：保持避障引导，同时避免刷分
                                                        # 注意：实际权重会根据距离目标的距离动态调整（见下方条件化参数）

# === 🔧 条件化净空奖励参数（解决"仅有动作"刷分问题） ===
export CLEARANCE_FAR_THRESHOLD=${CLEARANCE_FAR_THRESHOLD:-100.0}  # 远距离阈值（米）：距离目标>此值时使用低权重
export CLEARANCE_NEAR_THRESHOLD=${CLEARANCE_NEAR_THRESHOLD:-20.0}  # 近距离阈值（米）：距离目标<此值时使用高权重
export CLEARANCE_WEIGHT_FAR=${CLEARANCE_WEIGHT_FAR:-0.2}  # 远距离权重：防止"仅有动作"方法在远离目标时刷分
export CLEARANCE_WEIGHT_NEAR=${CLEARANCE_WEIGHT_NEAR:-6.0}  # 🔧 提高净空奖励权重（5.5→6.0），让智能体更多学习躲避地形
                                                             # 作用：增强地形避障引导，减少奖励波动
                                                             # 调整：提高9%，与降低的目标导向奖励形成更好平衡
                                                             # 原因：当前训练回合存在奖励值波动过大的问题，提高净空奖励有助于稳定训练
export CLEARANCE_PENALTY_WEIGHT=${CLEARANCE_PENALTY_WEIGHT:-4.0}  # 🔧 避障惩罚权重：降低到2.5（之前4.5过高）
                                                                  # 作用：当距离<安全距离时，无论距离目标多远都使用此权重
                                                                  # 建议范围：2.0-3.0（过小无法引导避障，过大导致过度保守）

export CLEARANCE_D_MAX=${CLEARANCE_D_MAX:-80.0}          # 净空检测最大距离（米）

export OBSTACLE_SAFE_DISTANCE=${OBSTACLE_SAFE_DISTANCE:-8.0}  # 🚨 障碍物安全距离（米）
                                                               # 作用：定义"安全距离"，智能体应保持与障碍物的最小距离
                                                               # 建议范围：10.0-20.0米
                                                               # 当有效距离小于此值时，给予负奖励；大于此值时，给予正奖励

export SAFE_DISTANCE_WEIGHT=${SAFE_DISTANCE_WEIGHT:-0.4}  # 🚨 绝对距离奖励权重
                                                          # 作用：奖励保持安全距离的持续奖励
                                                          # 建议范围：0.2-0.5

# 🚨 已禁用：UPWARD_BONUS_FACTOR 存在诱导穿透障碍物的风险
# 问题：如果智能体在障碍物上方时有效距离会乘以加成因子，可能诱导智能体从障碍物下方穿透到上方
# 解决方案：已禁用此功能，避免诱导穿透行为
# 如果将来需要实现向上绕行奖励，应该：
#   1. 只有在智能体已经在障碍物上方时才给予奖励（不诱导穿透）
#   2. 或者使用其他机制（如高度奖励）来鼓励向上绕行
# export UPWARD_BONUS_FACTOR=${UPWARD_BONUS_FACTOR:-1.2}  # 已禁用：避免诱导穿透

# === 奖励裁剪范围 ===
export REWARD_CLIP_MIN=${REWARD_CLIP_MIN:--8000.0}       # 🔧 进一步放宽惩罚范围（-5000→-8000），保留完整惩罚信号
export REWARD_CLIP_MAX=${REWARD_CLIP_MAX:-4000.0}         # 🔧 进一步放宽奖励范围（2500→4000），保留完整奖励信号

# === 团队协作与势场塑形 ===
export GLOBAL_WEIGHT=${GLOBAL_WEIGHT:-5.0}               # 🚨 优化团队协作：大幅提高全局奖励权重（2.5→5.0），强化团队协作信号
                                                        # 作用：促进团队协作，鼓励所有智能体都到达目标
                                                        # 问题：2.5的权重仍然不足以鼓励所有智能体都到达目标（团队成功率仅0.05）
                                                        # 新值5.0：提供更强的团队协作信号，确保所有智能体都完成任务
                                                        # 建议范围：3.0-8.0（根据训练效果调整）

export SHAPING_WEIGHT=${SHAPING_WEIGHT:-0.3}             # 潜势函数shaping：基于势场的奖励塑形
                                                        # 作用：提供额外的梯度信号，加速学习
                                                        # 建议范围：0.2-0.6

# === Z轴动作映射偏置 ===
# 🔧 新实验：给予轻微向上偏置，引导智能体更偏向空中轨迹，而不是贴地飞行
# 说明：
#   - Actor输出范围仍为[-1,1]，映射关系为：real_az = (az + z_bias) * ACTION_RANGE_Z * CONTROL_ACCEL_GAIN
#   - 当 z_bias=0.35 时，理论映射区间约为 [-0.65, 1.35]，向上能力略强于向下，更容易克服重力和高度惩罚
#   - 环境 core.py 与训练侧映射都统一从该环境变量读取，保证训练/执行/回放完全一致
export Z_ACTION_BIAS=${Z_ACTION_BIAS:-0.0}

# === 🎯 大幅缩小奖励尺度：防止Q值爆炸→梯度爆炸→权重爆炸→tanh饱和 ===
# 这些参数限制最终奖励的范围，防止极端值影响训练稳定性
# 作用位置：paper3d_terrain_weighted.py 的 reward() 方法
# 作用时机：各分项加权求和后的第一次裁剪
# 后续还会在 environment.py 中进行缩放（负值×1.1，正值×1.3）

export MAX_REWARD=${MAX_REWARD:-800.0}                    # 🔧 提高到300，给予更多正奖励空间
                                                        # 作用：防止奖励过大导致Q值爆炸
                                                        # 建议范围：30-100

export MIN_REWARD=${MIN_REWARD:--800.0}                   # 🚨 增强穿透惩罚：扩大到-450，匹配更严厉的穿透惩罚（-400→-450）
                                                        # 作用：防止惩罚过大导致Q值爆炸，同时保留完整的穿透惩罚信号
export REWARD_CLIP_VALUE=${REWARD_CLIP_VALUE:--650.0} # 🚨 增强穿透惩罚：扩大到-400，匹配更严厉的穿透惩罚（-350→-400）
                                                        # 作用：防止极端负奖励淹没正常奖励信号，同时保留有效的穿透惩罚
                                                        # 注意：单步奖励范围扩大到-150到-200，需要更大的裁剪阈值
                                                        # 建议范围：-300到-500（根据实际奖励分布调整）

# === 新增分项参数 ===
export SUCCESS_REWARD_VALUE=${SUCCESS_REWARD_VALUE:-3000.0}  # 🚨🚨🚨 成功奖励基础值（会被无碰撞比例削弱）
                                                           # 🚨 关键修改：成功奖励 = 成功奖励 × 无碰撞比例
                                                           #   无碰撞比例 = 1 - (总碰撞次数 / 回合总步数)
                                                           #   回合总步数：2800步（从--episode-length获取）
                                                           #   总碰撞次数：所有智能体的total_penetration_count之和
                                                           # 非线性映射（确保"碰了一半及以上就小于0.2"）：
                                                           #   - 如果碰撞比例 < 0.5：无碰撞比例 = 1 - 碰撞比例（线性）
                                                           #   - 如果碰撞比例 >= 0.5：无碰撞比例 = 0.2 × (1 - (碰撞比例-0.5)/0.5)（非线性）
                                                           # 示例计算（回合总步数=2800）：
                                                           #   - 没有碰撞（0次）: 比例 = 1.0，成功奖励 = 3000 × 1.0 × 2.0 = 6000
                                                           #   - 只碰了几次（50次）: 比例 = (1-50/2800)^4.5 ≈ 0.920，成功奖励 = 3000 × 0.920 × 2.0 = 5520
                                                           #   - 碰了一半（1400次）: 比例 = 0.2（非线性映射），成功奖励 = 3000 × 0.2 × 2.0 = 1200
                                                           #   - 碰了全部（2800次）: 比例 = 0.0，成功奖励 = 3000 × 0.0 × 2.0 = 0
                                                           # 注意：少量碰撞时使用指数惩罚（^4.5），显著降低奖励，形成更明显的梯度信号
                                                           #   无碰撞奖励: 20000 × 2.0 = 40000（仅当总碰撞次数=0时）
                                                           #   合计正向奖励（无碰撞）: (3000+20000) × 2.0 = 46000
                                                           #   vs 碰撞惩罚: 100/次（可承受一定碰撞）
                                                           # 建议范围：2000-5000（与碰撞惩罚协调）

export NO_COLLISION_REWARD_VALUE=${NO_COLLISION_REWARD_VALUE:-20000.0}  # 🚨🚨🚨 加强碰撞避免：大幅提高无碰撞奖励（12000→20000），更强烈地鼓励无碰撞行为
                                                                         # 奖励体系重构（修复后）：
                                                                         #   成功+无碰撞: (3000+20000) × 2.0 = 46000
                                                                         #   成功+少量碰撞(500次): 3000 - 500×320 = -157000
                                                                         #   结论：无碰撞回合奖励远高于有碰撞回合，确保团队成功（无碰撞）成为最佳回合
                                                                         # 建议范围：15000-25000（确保无碰撞回合奖励足够高，强烈鼓励无碰撞策略）

export SUCCESS_DISTANCE_THRESHOLD=${SUCCESS_DISTANCE_THRESHOLD:-5.0} # 🔧 进一步放宽成功判定阈值（4.0→5.0），让所有智能体更容易成功
                                                                     # 实际判断阈值 = 1.2 * SUCCESS_DISTANCE_THRESHOLD = 4.8米
                                                                     # 问题：3.6米对某些智能体来说太严格，导致只有部分智能体到达
                                                                     # 新值4.8米：在保持精确性的同时，提高所有智能体到达的可能性
                                                                     # 建议范围：3.0-5.0

export COLLISION_PENALTY_VALUE=${COLLISION_PENALTY_VALUE:-60.0}     # 🚨🚨🚨 加强碰撞避免：提高碰撞惩罚（50→80），配合权重4.0确保碰撞有足够代价
                                                                     # 问题诊断：50×2.0=100/次，即使2000次碰撞也只有-20万，不足以区分有碰撞和无碰撞回合
                                                                     # 修复策略：
                                                                     #   1. 提高惩罚值到80（配合权重4.0 = -320/次）
                                                                     #   2. 确保有碰撞的回合（如500次碰撞=-16万）奖励低于无碰撞回合
                                                                     #   3. 让团队成功（无碰撞）的回合成为最佳回合
                                                                     # 建议范围：70-100（确保碰撞惩罚足够强，但不至于完全淹没奖励）

export COLLISION_DISTANCE_THRESHOLD=${COLLISION_DISTANCE_THRESHOLD:-0.5}  # 🚨🚨🚨 紧急修复：大幅降低碰撞阈值（1.5→0.5）
                                                                         # 问题诊断：1.5m阈值导致智能体在复杂地形中"无处不碰撞"
                                                                         #          Episode 6有2444次碰撞，意味着几乎每步都在"碰撞"
                                                                         # 根本原因：1.5m太大了！智能体半径通常<0.5m，1.5m等于"3倍体积"
                                                                         # 修复策略：只惩罚"真正的物理碰撞"（距离<0.5m），而不是"接近"
                                                                         # 建议范围：0.3-0.8米（真实碰撞距离）

export GLOBAL_REWARD_MODE=${GLOBAL_REWARD_MODE:-success_rate}                   # 🔧 改为success_rate模式，鼓励所有智能体都到达目标
                                                                     # 全局奖励模式：avg_progress|min_distance|success_rate
                                                                     # - avg_progress：所有智能体到各自目标的平均正向进展
                                                                     # - min_distance：全体最小距离（越小越好，取负号作为奖励）
                                                                     # - success_rate：达到阈值的智能体比例 * R_succ（✅ 推荐）
                                                                     # 问题：avg_progress模式可能导致部分智能体到达后停止努力
                                                                     # 新值success_rate：只有当所有智能体都到达时才会获得最大奖励
                                                                     # 选择建议：
                                                                     #   - 协同收敛：success_rate（✅ 推荐，确保所有智能体都到达）
                                                                     #   - 确保至少一人先到：min_distance
                                                                     #   - 强调整体完成率：success_rate

export SHAPING_GAMMA=${SHAPING_GAMMA:-0.9}                         # 潜势函数 gamma（0-1）
                                                                     # 越接近1越强调"未来潜势"差分
                                                                     # 建议：0.9-0.99；与SHAPING_WEIGHT联动调小以防主奖励被淹没

# === 🎯 奖励计算微调参数（增强穿透惩罚）===
export PENETRATION_ALPHA=${PENETRATION_ALPHA:-5.0}                  # 🔧 穿透深度系数：降低到5.0（之前10.8过高）
export PENETRATION_BASE_PENALTY=${PENETRATION_BASE_PENALTY:-120.0}   # 🔧 基础穿透惩罚：降低到120（之前300过高）
                                                                      # 穿透惩罚公式：-BASE_PENALTY - penetration_depth * ALPHA
                                                                      # 示例：穿透1米 = -120 - 5.0 = -125（适度惩罚，不会过度抑制）
                                                                      # 建议范围：BASE_PENALTY=100-150, ALPHA=4.0-6.0
export TERRAIN_CONTACT_EPS=${TERRAIN_CONTACT_EPS:-0.2}            # 🔧 地形接触阈值：降低到0.3米，更早触发穿透早停
                                                                  # 作用：更早检测地形穿透，提前终止回合
                                                                  # 建议范围：0.1-0.5米（更严格的穿透检测）
                                                                     # 问题：2.5米阈值过大，导致智能体在安全高度（12米）时被误判为接触地形
                                                                     # 影响：每回合产生1500-2900次误报碰撞，导致奖励波动极大（-216k到+70k）
                                                                     # 修复：0.75米是合理的接触阈值，只在实际接近地形时触发
                                                                     # 建议范围：0.5-1.0米（过小会漏检，过大会误触发）

export EXPL_REWARD_STRICT=${EXPL_REWARD_STRICT:-0}                   # 探索奖励严格模式
                                                                     # 作用：1=严格模式（新格奖励1.0，禁用随机奖励），0=宽松模式（新格奖励5.0，含随机奖励）
                                                                     # 建议：训练初期用0，中后期用1避免探索奖励盖过负向信号

# === 向量化优化参数 ===
# 这些参数控制numpy向量化优化功能，显著提升并行环境训练性能
# 建议在并行环境训练时启用，单环境训练时可选择性启用

export USE_VECTORIZATION=${USE_VECTORIZATION:-1}              # 是否启用向量化优化（默认启用）
                                                               # 1=启用向量化优化，0=使用原始实现
                                                               # 建议：并行环境训练时启用，可提升2-3x性能

export VECTORIZED_REWARDS=${VECTORIZED_REWARDS:-1}          # 是否启用向量化奖励计算
                                                               # 1=使用批量奖励计算，0=单个计算
                                                               # 建议：使用分项加权求和场景时启用，可提升3-5x奖励计算性能

export VECTORIZED_OBSERVATIONS=${VECTORIZED_OBSERVATIONS:-1} # 是否启用向量化观察处理
                                                               # 1=使用批量观察处理，0=单个处理
                                                               # 建议：并行环境训练时启用，可提升2-3x观察处理性能

export VECTORIZED_SCENARIO=${VECTORIZED_SCENARIO:-1}       # 是否使用向量化场景（实验性功能）
                                                               # 1=使用paper3d_terrain_vectorized场景，0=使用原始场景
                                                               # 注意：这是实验性功能，建议先测试稳定性
# === 学习预热机制参数 ===
# 🔧 修复：默认禁用预热机制，因为初期经验质量差，反而影响学习
# 问题分析：
#   - 预热阶段使用随机策略或未训练网络，产生的经验质量差（大量负奖励、无效动作）
#   - 这些差经验会污染回放缓冲区，导致网络学习到不好的模式
#   - 预热阶段不写入RB，但预热完成后立即写入的经验仍然是初期差经验
# 替代方案：
#   - 直接开始训练，让网络从少量经验中学习（PER会优先采样高质量经验）
#   - 使用势场引导（ACTION_FORCE_RATIO）提供初始引导，产生更好的初始经验
#   - 提高初始探索噪声（NOISE_SCALE）和随机动作概率（RANDOM_ACTION_PROB）来平衡探索
export LEARNING_WARMUP_ENABLED=${LEARNING_WARMUP_ENABLED:-0}              # 🔧 禁用预热，立即开始学习
export LEARNING_WARMUP_RATIO=${LEARNING_WARMUP_RATIO:-0.15}               # 预热比例25%
export LEARNING_WARMUP_MIN_STEPS=${LEARNING_WARMUP_MIN_STEPS:-8000}        # 🔧 大幅降低到3000步，快速开始学习
export LEARNING_WARMUP_MAX_STEPS=${LEARNING_WARMUP_MAX_STEPS:-15000}       # 🔧 降低到5000步
# 🔧 新增：完全随机预热（PF引导）配置
export WARMUP_RANDOM_POLICY=${WARMUP_RANDOM_POLICY:-0}          # 1=启用PF引导的完全随机预热（仅在预热阶段生效）
export WARMUP_RANDOM_SCALE=${WARMUP_RANDOM_SCALE:-1.0}          # 随机动作幅度（0-1），1.0表示覆盖全部动作范围
export WARMUP_FORCE_RATIO=${WARMUP_FORCE_RATIO:-0.75}            # 🔧 提高到0.75，预热阶段势场主导，防止向下飞
# 预热阶段势场base参数元优化开关：1=启用（在预热期轮换少量候选base，并在预热结束时固定最优base），0=禁用
export PF_META_WARMUP_ENABLED=${PF_META_WARMUP_ENABLED:-1}
export PF_META_GA_MIN=${PF_META_GA_MIN:-1.5}                      # 势场目标吸引力最小值
export PF_META_GA_MAX=${PF_META_GA_MAX:-5.5}                      # 势场目标吸引力最大值
export PF_META_TR_MIN=${PF_META_TR_MIN:-20.0}                     # 地形排斥力最小值
export PF_META_TR_MAX=${PF_META_TR_MAX:-80.0}                    # 地形排斥力最大值
export PF_META_RANGE_MIN=${PF_META_RANGE_MIN:-18.0}               # 影响范围最小值
export PF_META_RANGE_MAX=${PF_META_RANGE_MAX:-36.0}               # 影响范围最大值
export PF_META_GA_SAMPLES=${PF_META_GA_SAMPLES:-6}                # GA 采样数量
export PF_META_TR_SAMPLES=${PF_META_TR_SAMPLES:-6}                # TR 采样数量
export PF_META_RANGE_SAMPLES=${PF_META_RANGE_SAMPLES:-4}          # 影响范围采样数量
export PF_META_MAX_CANDIDATES=${PF_META_MAX_CANDIDATES:-12}        # 候选最大数量
export PF_META_MIN_EPISODES=${PF_META_MIN_EPISODES:-2}            # 每个候选至少评估的回合数
export PF_META_DISTANCE_WEIGHT=${PF_META_DISTANCE_WEIGHT:-0.8}    # 元优化评分：目标距离权重（越小越好）
export PF_META_OBSTACLE_WEIGHT=${PF_META_OBSTACLE_WEIGHT:-0.7}    # 元优化评分：障碍碰撞惩罚权重
export PF_META_PENETRATION_WEIGHT=${PF_META_PENETRATION_WEIGHT:-1.2} # 元优化评分：地形穿透惩罚权重
export PF_META_ENERGY_WEIGHT=${PF_META_ENERGY_WEIGHT:-0.3}        # 元优化评分：能量消耗惩罚权重
export PF_META_METHOD=${PF_META_METHOD:-bayes}                    # 元优化搜索方式：grid|bayes
export PF_META_BO_INIT_RANDOM=${PF_META_BO_INIT_RANDOM:-5}        # BO初始随机采样次数
export PF_META_BO_ACQ_SAMPLES=${PF_META_BO_ACQ_SAMPLES:-64}       # BO采集函数随机采样数
export PF_META_BO_NOISE=${PF_META_BO_NOISE:-0.12}                  # BO GP噪声项
export PF_META_BO_LENGTH_SCALE=${PF_META_BO_LENGTH_SCALE:-0.5}    # BO核函数长度尺度
export PF_META_BO_XI=${PF_META_BO_XI:-0.01}                       # BO EI探索系数
# === 起飞前重力补偿阈值（物理层） ===
export PRE_TAKEOFF_START_RADIUS=${PRE_TAKEOFF_START_RADIUS:-8.0}         # 起始XY半径
export PRE_TAKEOFF_AIRBORNE_THRESHOLD=${PRE_TAKEOFF_AIRBORNE_THRESHOLD:-1.0}  # 🔧 起飞前保护：确保智能体在地形上方至少0.5米（防止初始穿透）
                                                                              # 注意：这个参数不是重力补偿阈值！重力补偿由物理层内部控制

# === 🔧 关闭碰撞自动复位机制（让智能体真实体验穿透惩罚）===
export ENABLE_COLLISION_AUTORESET=${ENABLE_COLLISION_AUTORESET:-0}  # 1=开启复位，防止贴地后持续受惩罚

# === 🔧 delta+base 模式势场力参数配置（网络输出绝对调整量）===
# 🔧 势场力公式详解（基于观察数据的实际势场公式 + delta+base参数调整）：
# 
# 参数计算方式（delta+base 模式 - 绝对变化量）：
#   参数 = base_value + network_output * delta_amount
#   - network_output: Actor网络输出的[-1, 1]范围值
#   - base_value: 基准值（见下方配置）
#   - delta_amount: 绝对变化量（网络输出[-1,1]映射到实际变化范围）
#   
#   实际参数范围 = base ± delta_amount
#   例如：base=1.0, delta=0.5 → 范围[0.5, 1.5]
#         base=80.0, delta=40.0 → 范围[40.0, 120.0]
#
# 1. 目标吸引力公式（可微分段版本）：
#    近距离 (d < d0): F_att = k_att * d0 * dir_to_goal (常数吸引力)
#    远距离 (d > d0): F_att = 2 * k_att * d * dir_to_goal (线性增长吸引力)
#    - d0: 距离阈值（分段点），由 lambda_1 参数确定
#          lambda_1 = LAMBDA_1_BASE + output[1] * DELTA_LAMBDA_1
#          默认值: LAMBDA_1_BASE=6.5, DELTA_LAMBDA_1=4.0
#          实际范围: 6.5 ± 4.0 = [2.5, 10.5]（代码中限制在[3.0, 15.0]）
#          🔧 关键：d0控制吸引力强度的变化方式，默认约6.5米
#    - k_att: 目标吸引力系数（控制吸引力强度）
#             k_att = GOAL_ATTRACTION + output[0] * DELTA_K_ATT
#             默认值: GOAL_ATTRACTION=3.5, DELTA_K_ATT=1.8
#             实际范围: 3.5 ± 1.8 = [1.7, 5.3]
#    - dir_to_goal: 指向目标的单位方向向量
#    - 使用sigmoid函数实现平滑分段过渡，保持可微性
#    - ⚠️  重要：吸引力没有最大作用距离限制，从任意距离（>0）都会产生吸引力
#
# 2. 地形排斥力公式：
#    F_rep = λr * (1/r_min - 1/R_safe) * (1/r_min^n) * κ * n̂
#    - r_min: 智能体到地形表面的最小距离（基于观察数据）
#    - R_safe: 安全检测半径，由 radius 参数确定
#              radius = AGENT_INFLUENCE_RANGE + output[3] * DELTA_RADIUS
#              默认范围: 10.0 ± 5.0 = [5.0, 15.0]
#    - λr: 地形排斥力系数
#          λr = TERRAIN_REPULSION + output[2] * DELTA_K_REP
#          默认范围: 80.0 ± 40.0 = [40.0, 120.0]
#    - n: 距离指数 (通常为2.0)
#    - κ: 目标距离影响因子 (exp(-goal_dist/50.0))
#    - n̂: 地形法向量（基于观察数据中的地形梯度计算）
#
# 3. 智能体间排斥力公式：
#    F_agent = Σ(1/d - 1/R_detection) * (1/d²) * (p_self - p_other)/d
#    - d: 智能体间距离（基于观察数据中的其他智能体位置）
#    - R_detection: 检测半径（与上述R_safe共用radius参数）
#    - p_other: 其他智能体的绝对位置（从观察数据计算）
#    - Σ: 对所有其他智能体求和
#
# 4. 障碍物排斥力公式：
#    F_obstacle = Σ(1/d - 1/R_detection) * (1/d²) * (p_self - p_obstacle)/d
#    - d: 到障碍物的距离（基于观察数据中的障碍物位置）
#    - R_detection: 检测半径（与上述R_safe共用radius参数）
#    - p_obstacle: 障碍物的绝对位置（从观察数据计算）
#    - Σ: 对所有障碍物求和

# 参数空间范围映射模式已废弃，改用delta+base模式（见下方配置）

# === 🔧 delta+base 模式基准参数和绝对变化量（用于 TensorFlow 版本的势场参数调整） ===
# 基准值参数
export GOAL_ATTRACTION=${GOAL_ATTRACTION:-6.0}                                               # 🚨 适度降低目标吸引力（8.0→6.0），平衡目标导向和地形避障
                                                                                                 # 作用：目标吸引力的基准强度
                                                                                                 # 问题：目标吸引力过强（8.0）可能抵消地形排斥力，导致难以躲避地形
                                                                                                 # 解决策略：降低目标吸引力25%，让地形排斥力更容易占主导
                                                                                                 # 效果：目标吸引力从8.0降低到6.0，但仍保持足够的目标导向
                                                                                                 # 调优：增大增强目标导向，减小降低目标依赖
                                                                                                 # 建议：5.0-8.0（默认6.0，平衡目标导向和地形避障）
export LAMBDA_1_BASE=${LAMBDA_1_BASE:-8.5}                                                   # lambda_1基准值（目标吸引力分段距离阈值d0）
                                                                                                 # 🔧 关键参数：控制吸引力从何时开始产生作用
                                                                                                 # 作用：目标吸引力从常数切换到线性的分段点基准值（单位：米）
                                                                                                 # 吸引力作用机制：
                                                                                                 #   - 距离 < d0（约6.5米）：常数吸引力 = k_att * d0（稳定引导）
                                                                                                 #   - 距离 > d0（约6.5米）：线性增长吸引力 = 2 * k_att * d（距离越远，吸引力越强）
                                                                                                 #   - 注意：吸引力没有最大作用距离限制，从任意距离（>0）都会产生
                                                                                                 # 调优：增大延迟分段切换（远距离才增强），减小提前分段切换（近距离就增强）
                                                                                                 # 建议：3.0-15.0（默认6.5米，适合中等距离开始增强吸引力）
export TERRAIN_REPULSION=${TERRAIN_REPULSION:-8000.0}                                         # 🚨🚨🚨 进一步增强地形排斥力（6000→8000），防止贴地飞行
                                                                                                # 问题诊断：势场引导让智能体容易到达目标，但难以躲避地形
                                                                                                # 根本原因：目标吸引力过强，可能抵消地形排斥力
                                                                                                # 解决策略：进一步提高k_rep到8000，确保地形排斥力足够强
                                                                                                # 效果：地形斥力强度增加33%，即使在远距离也能有效避障
                                                                                                # 建议范围：7000-10000（必须强到对抗目标吸引力）
                                                                                                # 调优：增大增强避障能力，减小降低地形敏感度
export AGENT_INFLUENCE_RANGE=${AGENT_INFLUENCE_RANGE:-150.0}                                  # 🔧 扩大检测范围（120→150），让地形排斥力更早生效
                                                                                                # 作用：检测半径基准值，影响地形排斥力的作用范围
                                                                                                # 问题：120米可能不够大，导致排斥力反应较晚
                                                                                                # 新值150米：扩大25%，让排斥力在更远距离就开始生效
                                                                                                # 调优：根据环境大小调整，地图大时增加，地图小时减少
                                                                                                # 建议：120-180米（默认150米）
# 绝对变化量参数（网络输出[-1,1]映射到实际变化范围）
# 🚨 关键修复：缩小delta范围，防止Actor通过输出±1.0来破坏势场保护
# 原因：Actor学会输出-1.0来最小化k_rep（地形排斥力），导致穿透地形
# 解决方案：将delta缩小到原来的30%，限制Actor对PF参数的影响幅度
# 效果：Actor仍可微调PF参数，但无法彻底关闭势场保护

export DELTA_K_ATT=${DELTA_K_ATT:-5.0}                                                       # 🚨 从2.0降到0.6（30%），k_att范围[19.4, 20.6]
                                                                                                 # 作用：控制目标吸引力的调整范围
                                                                                                 # 计算公式：k_att = 20.0 + output * 0.6
                                                                                                 # 范围：20.0 ± 0.6 = [19.4, 20.6]（微调，不影响主导作用）
                                                                                                 
export DELTA_LAMBDA_1=${DELTA_LAMBDA_1:-2.2}                                                 # 🚨 从2.5降到0.75（30%），lambda_1范围[7.75, 9.25]
                                                                                                 # 作用：控制分段距离阈值的调整范围
                                                                                                 # 计算公式：lambda_1 = 8.5 + output * 0.75
                                                                                                 # 范围：8.5 ± 0.75 = [7.75, 9.25]（微调）
                                                                                                 
export DELTA_K_REP=${DELTA_K_REP:-1200.0}                                                      # 🚨 增大变化范围（1000→1200），配合tanh允许k_rep在[2300, 4700]调整
                                                                                                # 作用：控制地形排斥力的调整范围
                                                                                                # 计算公式：k_rep = 1200 + output * 75
                                                                                                # 范围：1200 ± 75 = [1125, 1275]（±6.25%变化，保持强排斥）
                                                                                                # 🎯 关键：即使Actor输出-1.0，k_rep仍有1125，足够强的排斥力！
                                                                                                
export DELTA_RADIUS=${DELTA_RADIUS:-80.0}                                                     # 🔧 扩大检测半径调整范围（60→80），让radius范围更大
                                                                                                 # 作用：控制检测半径的调整范围
                                                                                                 # 计算公式：radius = AGENT_INFLUENCE_RANGE + output * DELTA_RADIUS
                                                                                                 # 范围：150 ± 80 = [70, 230]（扩大检测范围，让排斥力更早生效）
                                                                                                 # 问题：60的范围可能不够大，无法充分利用扩大后的基准值
                                                                                                 # 新值80：允许radius在更大范围内调整，配合150的基准值，最大可达230米

# 🔧 势场力归一化说明（自动归一化到与网络动作同一量级）
# 工作原理：
#   1. 限制势场力最大幅度到max_force_magnitude（8.0）
#   2. 归一化到单位向量（保留方向，去除幅值）→ 范围[-1, +1]
#   3. 与网络动作混合：corrected_action = action + force_direction * force_ratio
# 网络动作范围是[-1, +1]，势场单位向量范围也是[-1, +1]，自然同一量级
# force_ratio控制混合比例：0.0=完全网络动作，1.0=完全势场方向

# === 网络动作和势场动作混合比例 ===
# 🔧 修复势场包裹过强问题：使用渐进式FR衰减，平衡势场引导和网络学习
# 原理：
#   - 训练初期：高FR（0.60），势场主导，提供强引导，避免碰撞
#   - 训练中期：中FR（0.35-0.45），势场与网络平衡，网络学习加速
#   - 训练后期：低FR（0.20），网络主导，势场仅提供安全修正
export ACTION_FORCE_RATIO=${ACTION_FORCE_RATIO:-0.50}                # 初始值（会被schedule覆盖）
# 🔧 修复：ACTION_FORCE_RATIO_SCHEDULE_PCT 的处理逻辑
# 1. 如果环境变量已设置且不为空，使用环境变量的值（允许从外部传入schedule配置）
# 2. 如果设置为 "DISABLED"，则设为空字符串（禁用schedule）
# 3. 如果未设置（变量不存在），则使用默认渐进式衰减schedule
# 4. 如果设置为空字符串，则保持为空（禁用schedule）
# 🚨 关键修复：先检查环境变量是否已设置，避免覆盖外部传入的值
if [ -z "${ACTION_FORCE_RATIO_SCHEDULE_PCT:-}" ]; then
    # 变量未设置，使用默认渐进式衰减schedule（平衡配置）
    # 🔧 修复：平衡初期引导和后期学习
    # 设计原则：
    #   - 初期（0-20%）：FR=0.85-0.80，高势场保护，帮助网络获得好的初始解
    #   - 中期（20-60%）：FR=0.80-0.50，逐渐降低，平衡势场和网络学习
    #   - 后期（60-100%）：FR=0.50-0.30，网络主导，势场仅提供安全修正
    # 这样既能保证初期有好的初始解，又能让网络在后期有足够的学习空间
    export ACTION_FORCE_RATIO_SCHEDULE_PCT="0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10"  # 🔧 修复：大幅提高初期PF比例，防止随机策略导致坠毁
                                                                                                    # 问题诊断：30%的FR仍导致智能体过度依赖APF贴地飞行
                                                                                                    # 修复策略：初期30%→后期2%，让Actor尽早主导
                                                                                                    # 目标：让智能体学习自主避障，而不是依赖APF0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10
fi
# 如果设置为"DISABLED"，则设为空字符串（禁用schedule）
if [ "${ACTION_FORCE_RATIO_SCHEDULE_PCT:-}" = "DISABLED" ]; then
    # 显式禁用schedule
    export ACTION_FORCE_RATIO_SCHEDULE_PCT=""
# else: 变量已设置且不为空且不是"DISABLED"，使用已有值（不做修改，允许schedule生效）
fi
export MAX_FORCE_MAGNITUDE=${MAX_FORCE_MAGNITUDE:-80.0}               # 🚨 关键修复：大幅提高（28.5→80），避免强斥力被过早裁剪（k_rep*factor可达3000+）
# 动作-势场混合模式已固定为 pre_tanh（未饱和空间叠加），不再通过环境变量配置
export SUCCESS_COUNT_MODE=${SUCCESS_COUNT_MODE:-any}                  # 并行环境成功计数聚合：any|all|majority（默认any）

# FR条件化与Q安全参数
export USE_FR_FEATURE=${USE_FR_FEATURE:-1}            # ✅ 启用FR特征（作为独立条件输入传递给网络）
export USE_PF_FEATURE=${USE_PF_FEATURE:-1}            # ✅ 启用势场矢量特征（训练脚本会在保存经验时追加到obs）
export Q_CLIP_VALUE=${Q_CLIP_VALUE:-300.0}          # 🚨 关键修复：降低Q裁剪值以匹配新的奖励缩放（1/2000）
                                                      # 原因：APF方法5M奖励→500，Action Only 45k→4.5，Q值在[-1000,1000]范围内
export CRITIC_Q_REG=${CRITIC_Q_REG:-0.005}           # 🔧 进一步降低Q正则（0.01→0.005），稍微放开critic的学习能力
                                                      # 作用：降低正则化约束，允许critic网络有更强的学习能力
                                                      # 调整：降低50%，减少对Q值的正则化惩罚，让critic能够更好地学习Q值估计
                                                      # 建议范围：0.003-0.01（过小可能导致Q值不稳定，过大可能限制学习能力）
export ACTION_REG_COEF=${ACTION_REG_COEF:-0.00014}    # 🔧 大幅降低动作正则（0.002→0.0005），让Actor敢于输出更大的加速度值
                                                                                                 # 问题：网络输出范围依然不大，无法直接给出较大的加速度效果
                                                                                                 # 修复：进一步降低正则化，允许Actor输出更大的动作幅值
                                                                                                 # 新值0.0005：降低75%，大幅减少对大幅动作的惩罚
                                                                                                 # 问题：Actor输出范围太小，轨迹高度偏低，长度不够到达终点
                                                                                                 # 修复：大幅降低正则化，允许Actor输出更大的加速度值，提高飞行高度和距离
                                                                                                 # 建议：0.001-0.005
export NEG_Z_REG_COEF=${NEG_Z_REG_COEF:-0.0}           # 🔧 默认关闭负向Z轴额外正则（仅用于诊断时手动开启），避免用正则掩盖潜在结构问题

# === 势场修正版本选择 ===
# 🔧 当前使用基于观察数据的势场修正：
#    - 地形斥力：基于观察数据中的地形高度和梯度计算真实法向量
#    - 智能体间斥力：基于观察数据中的其他智能体位置计算真实距离和方向
#    - 障碍物斥力：基于观察数据中的障碍物位置计算真实距离和方向
#    - 目标吸引力：使用sigmoid函数实现可微分段，实现NumPy版本的分段效果
#    - 所有斥力都基于实际的环境信息，不再使用随机方向
export USE_TF_POTENTIAL_FIELD=${USE_TF_POTENTIAL_FIELD:-1}           # 势场修正版本（统一使用TF版本）
# 🔧 注意：TERRAIN_SENSING_MODE 仅用于评估，训练时强制使用local模式
# Oracle模式仅在评估时通过 evaluate_optimized.py 传递，训练时不需要此参数
                                                                     # 
                                                                     # 1 = TF版本（默认，推荐）
                                                                     #   ✅ 训练-推理完全一致（梯度可回传）
                                                                     #   ✅ 支持XLA加速（整个计算图可编译）
                                                                     #   ✅ Actor可以学习调节势场参数（k_att, lambda_1等）
                                                                     #   ✅ 收敛速度快，效果最佳
                                                                     #   ⚠️ 首次编译时间长（~30-60秒）
                                                                     # 
                                                                     # 0 = 不使用势场修正
                                                                     #   ⚠️ 仅用于对比实验，不推荐训练
                                                                     #   ⚠️ 训练效果差（无引导信号）
                                                                     # 
                                                                     # 注意：NumPy版本已移除（梯度无法回传，训练效果差）

export MEM_TRIM=${MEM_TRIM:-1}                      # 每回合末执行malloc_trim
export GPU_CACHE_CLEAR_INTERVAL=${GPU_CACHE_CLEAR_INTERVAL:-0}   # 🔧 XLA友好修复：完全禁用GPU缓存清理
                                                                   # 原因：清理函数中的.numpy()调用会触发GPU→CPU同步，打断XLA编译
                                                                   # 导致CUDA_ERROR_MISALIGNED_ADDRESS（内存对齐问题）
                                                                   # XLA Global模式下，TensorFlow会自动管理编译缓存，无需手动清理
export SAVE_BEST_TRAJ=${SAVE_BEST_TRAJ:-1}          # 保存最佳回合的轨迹图
export SAVE_EPISODE_TRAJ=${SAVE_EPISODE_TRAJ:-0}    # 保存每个完成回合的轨迹图（默认关闭，会产生大量图片）
export SAVE_INTERACTIVE_TRAJ=${SAVE_INTERACTIVE_TRAJ:-1}    # 是否保存可交互HTML轨迹图（需要plotly）
export SAVE_INTERACTIVE_TRAJ_INDEPENDENT=${SAVE_INTERACTIVE_TRAJ_INDEPENDENT:-0}  # 默认不独立生成每回合交互图

# 早停策略（防误判）
export EARLY_STOP_MODE=${EARLY_STOP_MODE:-never}                # 🔧 any|majority|all|never|disabled（默认never：禁用早停）
                                                                # any: 任一智能体完成就早停
                                                                # majority: 多数智能体完成就早停
                                                                # all: 全部智能体完成才早停（原默认）
                                                                # never/disabled: 禁用早停，每回合跑满步数
export EARLY_STOP_MAJORITY_RATIO=${EARLY_STOP_MAJORITY_RATIO:-0.66}  # 当模式为majority时的比例阈值

# 悬停奖励参数（到达终点后鼓励稳定悬停）
export HOVER_REWARD_MAX=${HOVER_REWARD_MAX:-6.0}            # 🔧 降低悬停奖励（15.0→3.0），减少累积奖励
                                                              # 原因：悬停奖励每3步发放一次，在目标处停留会累积大量奖励，导致奖励值波动大
                                                              # 新值3.0：降低80%，减少累积奖励，使奖励更平滑
export HOVER_SPEED_THRESHOLD=${HOVER_SPEED_THRESHOLD:-0.3}   # 悬停速度阈值（m/s）：速度低于此值给予悬停奖励
export HOVER_REWARD_INTERVAL=${HOVER_REWARD_INTERVAL:-10}     # 🔧 增加悬停奖励间隔（3→10），减少奖励频率
                                                              # 原因：减少悬停奖励的累积频率，使奖励更平滑

# 位置/地形复现实验相关（可选开关）
# 说明：
#   - USE_FIXED_POSITIONS=1 且 DYNAMIC_FIRST_TIME=1 时：
#       第一次重置使用动态采样起点/目标，并保存到 POSITIONS_FILE；
#       后续回合与其他并行环境都会从该文件加载同一套固定起点/目标，不再变化。
#   - USE_SCENARIO_SEED=1 + SCENARIO_SEED 固定 → 地形高度图可复现（但障碍物默认仍按随机数生成，每回合会变化）。
export USE_FIXED_POSITIONS=${USE_FIXED_POSITIONS:-1}     # 启用固定起点/目标（配合 DYNAMIC_FIRST_TIME 和 POSITIONS_FILE）
export POSITIONS_FILE=${POSITIONS_FILE:-./saved_positions/5.json}  # 固定位置文件路径（dynamic_first_time 首回合会写入）
export USE_SCENARIO_SEED=${USE_SCENARIO_SEED:-1}         # 是否使用固定随机种子来固定地图 (0=随机、但是要把后面RANDOM_TERRAIN设置为1, 1=固定)
# 只有在 USE_SCENARIO_SEED=1 时才设置 SCENARIO_SEED 环境变量（用于可复现实验）
# 否则不设置该变量，让代码使用时间戳生成真正的随机种子
if [ "${USE_SCENARIO_SEED}" = "1" ] || [ "${USE_SCENARIO_SEED,,}" = "true" ] || [ "${USE_SCENARIO_SEED,,}" = "yes" ] || [ "${USE_SCENARIO_SEED,,}" = "on" ]; then
    export SCENARIO_SEED=${SCENARIO_SEED:-88}                  # 地图/场景随机种子  48 
fi
export RANDOM_TERRAIN=${RANDOM_TERRAIN:-0}             # 是否使用随机地形 (1=每回合随机, 0=固定地形)

# 关键修复：当启用随机地形时，必须启用每回合地形重生成
# 否则所有并行环境在所有回合都会使用相同的地形
if [ "${RANDOM_TERRAIN}" = "1" ] || [ "${RANDOM_TERRAIN,,}" = "true" ] || [ "${RANDOM_TERRAIN,,}" = "yes" ] || [ "${RANDOM_TERRAIN,,}" = "on" ]; then
    export PER_EPISODE_TERRAIN=${PER_EPISODE_TERRAIN:-1}   # 每回合重新生成地形（随机地形模式下默认启用）
else
    export PER_EPISODE_TERRAIN=${PER_EPISODE_TERRAIN:-0}   # 固定地形模式下默认不重新生成
fi

# === GPU基础配置（稳定性优先）===
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}     # 减少TensorFlow日志（0=全部 3=仅错误）

# === 🔧 GPU显存分配模式配置 ===
# TF_FORCE_GPU_ALLOW_GROWTH: 控制GPU显存是否按需动态增长
#   true  (默认): 按需增长模式 - 显存根据实际使用量动态分配，避免一次性占用全部显存
#   false        : 预分配模式 - 一次性分配全部可用显存，可能导致OOM或与其他程序冲突
# 推荐设置: true (按需增长)，特别是在多程序共享GPU或显存有限时
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}  # 🔧 显存按需增长（默认启用）

export CUDA_VISIBLE_DEVICES=${GPU_ID}                       # 指定GPU ID
export TF_ENABLE_ONEDNN_OPTS=${TF_ENABLE_ONEDNN_OPTS:-1}  

# === 🔧 GPU内存分配器配置 ===
# TF_GPU_ALLOCATOR: 选择GPU内存分配器实现
# 可选值：
#   "" (空字符串)           - 默认 BFC 分配器（Best-Fit with Coalescing，推荐用于稳定性）
#   "memory_guard"          - 内存保护分配器（用于调试内存问题，性能较低）
#   "cuda_malloc_async"     - CUDA 异步分配器（性能较高，但可能导致 CUDA_ERROR_INVALID_PC）
# 推荐设置: "" (默认BFC分配器)，与 FP16 混合精度兼容性最好，避免内存对齐问题
# 注意: 如果遇到性能瓶颈且稳定性良好，可尝试 "cuda_malloc_async" 提升性能
export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-""}  # 🔧 GPU内存分配器（默认: cuda_malloc_async） 
# TF32加速：对矩阵乘法启用19位浮点（性能提升~20%，精度损失<0.1%）
# 恢复TF32（之前的禁用可能导致cuModuleGetFunction错误）
export TF_ENABLE_CUBLAS_TF32=${TF_ENABLE_CUBLAS_TF32:-1}   # cuBLAS TF32（GEMM加速）
export TF_USE_CUDNN_TF32=${TF_USE_CUDNN_TF32:-1}           # cuDNN TF32（卷积加速）
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}     # 全局TF32开关
# 减少线程争用，提升GPU执行稳定性（可按需覆盖）
export TF_GPU_THREAD_MODE=${TF_GPU_THREAD_MODE:-gpu_private}
# 🔧 GPU内存限制（可选，如果遇到内存问题可以设置）
# export TF_GPU_MEMORY_FRACTION=${TF_GPU_MEMORY_FRACTION:-0.9}  # 限制GPU内存使用90%

# glibc/ptmalloc 相关内存管理（抑制碎片/峰值）
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-131072}  # 触发trim门限（字节）；128KB
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}  # 小块走heap，大块走mmap的阈值（字节）；128KB
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}                   # 限制arena数

# === GPU执行配置（XLA Global + TF32）===
# 🔧 已在脚本开头通过 TF_XLA_FLAGS 统一设置：禁用 Auto JIT
# 此处不再覆盖 TF_XLA_FLAGS，避免前面的设置被重置
# 注意：Triton GEMM 控制在当前TF版本中不可用，使用默认配置
export TF_CUDNN_USE_AUTOTUNE=1                       # 启用cuDNN自动调优（运行时选择最快算法）
# 异步执行模式（性能优先）
# 🔧 配置：XLA Global + 异步执行（最大化性能）
# 说明：
#   - 使用异步CUDA执行，最大化GPU利用率
#   - 已优化GPU缓存清理频率（每25回合），避免在异步+XLA模式下频繁清理导致CUDA_ERROR_ILLEGAL_ADDRESS
#   - 清理前会先同步所有GPU操作，确保XLA编译完成，避免清理时打断编译导致内存地址无效
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}  # 异步执行（0），性能优先
export TF_SYNC_ON_FINISH=${TF_SYNC_ON_FINISH:-0}       # 异步执行（0），性能优先

# 根据CPU线程数设置常见并行库线程（可选）
if [ -n "${CPU_THREADS}" ]; then
    export OMP_NUM_THREADS=${CPU_THREADS}
    export MKL_NUM_THREADS=${CPU_THREADS}
    export TF_NUM_INTRAOP_THREADS=${CPU_THREADS}
    # 经验值：INTEROP 设置为 INTRAOP 的一半
    export TF_NUM_INTEROP_THREADS=$(( CPU_THREADS > 1 ? CPU_THREADS/2 : 1 ))
fi

# === 配置摘要 ===
echo "=========================================="
echo "🚀 最优性能配置（已通过稳定性测试）"
echo "=========================================="
echo "加速特性:"
echo "  - GPU: RTX 4060 Laptop (CUDA 12.8)"
echo "  - 混合精度: ${AMP_MODE^^} $([ "${AMP_MODE}" != "off" ] && echo '✅' || echo '❌')"
echo "  - TF32: $([ "${TF_ENABLE_CUBLAS_TF32}" = "1" ] && echo '✅ ON' || echo '❌ OFF') (矩阵加速~20%)"
echo "  - train_step JIT: $([ "${JIT_COMPILE}" = "1" ] && echo '✅' || echo '❌')"
echo "  - XLA: $([ "${XLA_GLOBAL:-0}" = "1" ] && echo '✅ 启用（Global）' || echo '❌ 禁用')"
echo "  - XLA Global: $([ "${XLA_GLOBAL:-0}" = "1" ] && echo '✅ 启用' || echo '❌ 禁用')"
echo "  - Triton GEMM: ⚠️  使用默认配置（当前TF版本不支持手动控制）"
echo ""
echo "执行配置:"
echo "  - 加速模式: TensorFlow + cuDNN + cuBLAS + TF32"
echo "  - 执行方式: 异步执行（性能优先）"
if [ "${XLA_GLOBAL:-0}" = "1" ]; then
    echo "  - ✅ XLA Global + 异步执行（性能模式）"
    echo "     - GPU缓存清理: 每${GPU_CACHE_CLEAR_INTERVAL:-10}回合（防止内存累积）"
fi
echo "  - cuDNN调优: 启用（自动选择最快算法）"
echo "  - 显存分配: 动态增长"
echo "  - GPU缓存清理: 每${GPU_CACHE_CLEAR_INTERVAL:-10}回合自动清理（防止长时间运行后内存累积和CUDA错误）"
echo ""
echo "训练参数:"
echo "  - GPU: ${GPU_ID}"
echo "  - 批次大小: ${BATCH_SIZE}"
echo "  - 训练回合: ${EPISODES}"
echo "  - 缓冲区: ${BUFFER_SIZE}"
echo "  - CPU线程: ${CPU_THREADS}"
echo "  - 并行环境: ${NUM_ENVS}"


# 设置环境变量抑制多智能体环境交互式提示（必须启用，否则并行环境会阻塞/报EOF）
export SUPPRESS_MA_PROMPT=1
# 🔧 关键修复：抑制地形生成冗余输出（减少并行环境中的大量地形信息输出）
export SUPPRESS_TERRAIN_OUTPUT=1

# 🔧 关键修复：清理Python缓存，确保使用最新代码（81维观测）
echo "清理Python缓存..."
find /home/tang/Desktop/multiagent -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /home/tang/Desktop/multiagent -name "*.pyc" -delete 2>/dev/null || true
find /home/tang/Desktop -name "*.pyc" -delete 2>/dev/null || true
echo "缓存已清理"

# 运行优化后的训练
echo "开始训练..."
echo "======================================"

# 组装训练参数并标注影响说明（不改变原有默认取值）
# 根据参数选择场景
if [ "$USE_WEIGHTED_REWARD" = "1" ] || [ "${USE_WEIGHTED_REWARD,,}" = "true" ]; then
    if [ "$VECTORIZED_SCENARIO" = "1" ] || [ "${VECTORIZED_SCENARIO,,}" = "true" ]; then
        SCENARIO_NAME="paper3d_terrain_vectorized"
        echo "使用向量化分项加权求和奖励机制的场景: $SCENARIO_NAME"
    else
        SCENARIO_NAME="paper3d_terrain_weighted"
        echo "使用分项加权求和奖励机制的场景: $SCENARIO_NAME"
    fi
else
    SCENARIO_NAME="paper3d_terrain_energy"
    echo "使用原始奖励机制的场景: $SCENARIO_NAME"
fi

# 设置场景参数
ARGS=(
    --scenario "$SCENARIO_NAME"             # 场景名称：使用选择的场景
    --algo "$ALGORITHM"                      # 训练算法：maddpg或matd3
    --train-episodes "$EPISODES"        # 训练回合数：↑ 总时长↑，总体稳定性↑
    --batch-size "$BATCH_SIZE"    # 批大小：提升到1024以充分利用GPU
    --exp-name "$EXP_NAME_WITH_TIMESTAMP"              # 实验名称：用于日志与模型目录（带时间戳）
    --save-model                          # 开启周期保存模型
    --save-interval 800                # 保存间隔（回合）：↑ IO频率↑
    --episode-length 2800               # 🔧 回滚到2200：步数翻倍会导致奖励尺度翻倍，Critic Loss反而更高
    --update-rate "$UPDATE_RATE"       # 更新频率（步）：可通过环境变量覆盖
    --learning-rate-actor "$LEARNING_RATE_ACTOR"   # Actor学习率（初始值）
    --learning-rate-critic "$LEARNING_RATE_CRITIC"  # Critic学习率（初始值）
    
    # 🔥 学习率衰减参数
    --lr-decay-enabled "$LR_DECAY_ENABLED"           # 是否启用学习率衰减
    --lr-decay-steps "$LR_DECAY_STEPS"               # 衰减步数
    --lr-decay-rate "$LR_DECAY_RATE"                 # 衰减率
    --lr-staircase "$LR_STAIRCASE"                   # 是否阶梯式衰减
    --lr-min-actor "$LR_MIN_ACTOR"                   # Actor最小学习率
    --lr-min-critic "$LR_MIN_CRITIC"                 # Critic最小学习率
    
    --gamma 0.95                          # 🔧 保持0.95，平衡短期和长期奖励
    --tau 0.015                       # 🚨 关键修复：大幅提高tau值（0.0002→0.015），加快目标网络更新
                                       # 问题：tau=0.0002太小，目标网络几乎不更新（需要5000次更新才能改变1%）
                                       # 影响：Q值估计严重滞后，学习效率极低，训练1000回合仍然失败
                                       # 修复：提高到0.015（75倍），是更标准的tau值，目标网络可以及时更新
                                       # 说明：目标网络每步都进行软更新（与Actor延迟更新无关），确保Q值估计稳定
                                       # 预期：提高学习效率，加快收敛速度
    --huber-delta "$HUBER_DELTA"         # 🔧 修复：Huber Loss的delta参数（降低delta让loss对中等误差更敏感）
    --grad-clip-norm 10.0                 # 🚨 关键修复：提高梯度裁剪阈值（5.0→10.0），允许更大的梯度
                                         # 问题：代码中存在双重裁剪（先clip_by_norm到1.0，再clip_by_global_norm到10.0）
                                         #       第一次裁剪已经将梯度限制到1.0，第二次裁剪几乎无效，导致梯度被过度限制
                                         # 影响：网络学习能力严重受损，训练效果差
                                         # 修复：需要在Python代码中移除逐层裁剪（clip_by_norm），只保留全局裁剪（clip_by_global_norm）
                                         #       当前配置10.0是合理的全局裁剪阈值，但需要配合代码修复才能生效
                                         # 预期：移除双重裁剪后，网络可以正常学习，训练效果显著改善
    --action-reg-coef "$ACTION_REG_COEF"              # Actor 动作正则系数（惩罚大幅动作）
    --neg-z-reg-coef "$NEG_Z_REG_COEF"                # 负向Z轴专用正则系数（仅惩罚az<0，防止整体偏向负半轴）
    # 探索噪声：作用于Actor输出（训练与评估时如需关闭，可将NOISE_SCALE设为0）
    --noise-scale "$NOISE_SCALE"          # 初始噪声幅度：↑ 探索↑ 但轨迹更抖；↓ 更平滑但易早收敛
    --noise-decay  "$NOISE_DECAY"         # 回合级衰减：接近1表示衰减很慢；过小会过早失去探索
    --noise-min    "$NOISE_MIN"           # 噪声下限：保持少量探索，避免策略陷入局部最优
    --random-action-prob "$RANDOM_ACTION_PROB"  # 随机动作概率（预热阶段）：epsilon-greedy探索
    --random-action-prob-training "$RANDOM_ACTION_PROB_TRAINING"  # 🔧 随机动作概率（训练阶段）：epsilon-greedy探索
    --lite-buffer true                    # 启用轻量缓冲区（预分配连续内存，峰值更低）
    --buffer-size "$BUFFER_SIZE"          # 覆盖缓冲区大小
    --per-enabled "$PER_ENABLED"         # 是否启用PER
    --buffer-dtype "$BUFFER_DTYPE"       # 回放缓冲精度
    --per-replace "$PER_REPLACE"         # PER是否放回
    --per-uniform-mix "$PER_UNIFORM_MIX" # PER与均匀采样混合比例
    --per-td-weight "$PER_TD_WEIGHT"     # 优先级：TD误差权重
    --per-reward-weight "$PER_REWARD_WEIGHT" # 优先级：奖励幅值权重
    --per-age-decay "$PER_AGE_DECAY"     # 优先级：年龄衰减系数
    --mem-debug "$MEM_DEBUG"             # 打印内存与显存
    --debug-pf-forces "$DEBUG_PF_FORCES" # 调试势场力量级
    --profiling "$PROFILING"             # 启用剖析
    --step-warn-s "$STEP_WARN_S"         # 步骤告警阈值
    --stack-dump-timeout "$STACK_DUMP_TIMEOUT" # 堆栈超时
    --amp-mode "$AMP_MODE"               # AMP策略：off|fp16|bf16
    --jit-compile "$JIT_COMPILE"         # 关键tf.function启用jit_compile
    --num-envs "$NUM_ENVS"               # 传递并行环境数量
    
    --gravity "$GRAVITY"                 # 环境重力
    --control-accel-gain "$CONTROL_ACCEL_GAIN" # 控制增益
    # --ground-friction 已弃用
    --damping "$DAMPING"                 # 速度阻尼系数
    --action-range-x "$ACTION_RANGE_X"  # X轴动作范围
    --action-range-y "$ACTION_RANGE_Y"  # Y轴动作范围
    --action-range-z "$ACTION_RANGE_Z"  # Z轴动作范围
        # --vertical-force-suppress 已移除
    --reward-pos-scale "$REWARD_POS_SCALE" # 正向奖励缩放
    --reward-neg-scale "$REWARD_NEG_SCALE" # 负向奖励缩放
    # 起飞前重力补偿阈值（传给物理层）
    --pre-takeoff-start-radius "$PRE_TAKEOFF_START_RADIUS"
    --pre-takeoff-airborne-threshold "$PRE_TAKEOFF_AIRBORNE_THRESHOLD"
    # 全局地形接触宽限步数通过场景内部默认/环境变量读取，无需CLI传参
    
    # Actor/Critic 更新频率控制
    --actor-update-delay "${ACTOR_UPDATE_DELAY}" # Actor延迟更新频率
    
    # MATD3特有参数（仅在ALGORITHM=matd3时生效）
    --policy-noise "${POLICY_NOISE:-0.28}"        # 目标策略平滑噪声标准差
    --noise-clip "${NOISE_CLIP:-0.35}"           # 目标策略噪声裁剪幅度
    --policy-freq "${POLICY_FREQ:-1}"            # 🔧 修复：传递MATD3的Actor更新频率（每N次Critic更新后更新1次Actor）
    
    # 势场力参数已改用delta+base模式（见下方配置）
    
    # 网络动作和势场动作混合比例
    --action-force-ratio "$ACTION_FORCE_RATIO"                              # 网络动作和势场动作混合比例
    --max-force-magnitude "$MAX_FORCE_MAGNITUDE"                           # 势场最大力幅值
    --success-count-mode "$SUCCESS_COUNT_MODE"                             # 并行环境成功计数聚合
    --use-fr-feature "$USE_FR_FEATURE"                                     # 将FR作为网络条件特征
    --use-pf-feature "$USE_PF_FEATURE"                                     # 将势场矢量作为网络条件特征
    --q-clip-value "$Q_CLIP_VALUE"                                         # 目标Q裁剪阈值
    --critic-q-reg "$CRITIC_Q_REG"                                         # Critic Q正则权重
    --reward-clip-value "$REWARD_CLIP_VALUE"                               # TD目标计算的奖励裁剪值
    
    # 势场修正版本选择
    --use-tf-potential-field "$USE_TF_POTENTIAL_FIELD"                      # 是否使用TensorFlow版本的势场修正
    # 🔧 注意：--terrain-sensing-mode 参数已移除，训练时强制使用local模式
    # Oracle模式仅在评估时通过 evaluate_optimized.py 传递
    
    # 🔧 delta+base 模式参数（用于 TensorFlow 版本的势场参数调整）
    # 基准值参数
    --goal-attraction "$GOAL_ATTRACTION"                                     # 目标吸引力基准值
    --lambda-1-base "$LAMBDA_1_BASE"                                        # lambda_1基准值
    --terrain-repulsion "$TERRAIN_REPULSION"                                # 地形斥力基准值
    --agent-influence-range "$AGENT_INFLUENCE_RANGE"                       # 智能体影响范围基准值
    
    # 绝对变化量参数（网络输出[-1,1]映射到实际变化范围）
    --delta-k-att "$DELTA_K_ATT"                                             # k_att的绝对变化量
    --delta-lambda-1 "$DELTA_LAMBDA_1"                                       # lambda_1的绝对变化量
    --delta-k-rep "$DELTA_K_REP"                                             # k_rep的绝对变化量
    --delta-radius "$DELTA_RADIUS"                                           # radius的绝对变化量
    
    # 🔧 权重约束参数
    --max-weight-threshold "$MAX_WEIGHT_THRESHOLD"
    --weight-scaling-factor "$WEIGHT_SCALING_FACTOR"
    
    --agent-accel "4.6"                # 智能体基础加速度
    --agent-max-speed "42.5"              # 智能体最大速度
    
    # 动态地图切换参数（基于连续成功和奖励停滞）- 🔧 修复：提高开启门槛，避免过早切换到随机地形
    # 🚨 注意：如果不想启用课程学习，请将这两个参数都设置为0
    # 当前配置：满足条件后会自动切换到随机地形模式（即使RANDOM_TERRAIN=0）
    # 🔧 关键修复：优先从环境变量读取，允许消融实验等场景覆盖默认值
    # 如果环境变量未设置，则使用默认值（启用课程学习）
    # 🚨 提高课程学习开启门槛：需要更多连续成功或更长的奖励停滞期
    --unlock-env-on-success "${UNLOCK_ENV_ON_SUCCESS:-100}"           # 🔧 提高连续成功要求（20→50），确保智能体在固定地形上充分学习
                                                                     # 作用：需要连续50个回合成功才能解锁随机地形
                                                                     # 影响：提高门槛，避免过早切换到随机地形，让智能体在固定地形上充分训练
    --unlock-env-on-plateau "${UNLOCK_ENV_ON_PLATEAU:-200}"         # 🔧 提高奖励停滞要求（25→60），确保智能体真正达到性能瓶颈
                                                                     # 作用：需要连续60个回合奖励无提升才能解锁随机地形
                                                                     # 影响：提高门槛，避免因短期波动而误判为性能停滞
    # 🔧 禁用课程学习：将环境变量设置为0，例如：
    # export UNLOCK_ENV_ON_SUCCESS=0
    # export UNLOCK_ENV_ON_PLATEAU=0
    # 或者直接修改上面的默认值（50→0, 60→0）
    
    
    # 向量化优化参数
    --use-vectorization "$USE_VECTORIZATION"      # 是否启用向量化优化
    --vectorized-rewards "$VECTORIZED_REWARDS"    # 是否启用向量化奖励计算
    --vectorized-observations "$VECTORIZED_OBSERVATIONS"  # 是否启用向量化观察处理

    # 异常/崩坏检测与自恢复
    --auto-abort-on-nan "true"
    --restart-on-collapse "false"
    --collapse-patience "3"
    --collapse-loss-threshold "1000"
    --collapse-z-threshold "-50.0"
    --terrain-complexity-level "$TERRAIN_COMPLEXITY_LEVEL"
    --map-size "$MAP_SIZE"
)

# 若提供了分段日程，则将其追加到参数中
# 🔧 修复：如果 ACTION_FORCE_RATIO_SCHEDULE_PCT 为 "DISABLED" 或空字符串，则不传递给训练脚本
if [ -n "$ACTION_FORCE_RATIO_SCHEDULE_PCT" ] && [ "$ACTION_FORCE_RATIO_SCHEDULE_PCT" != "DISABLED" ]; then
    ARGS+=(--action-force-ratio-schedule-pct "$ACTION_FORCE_RATIO_SCHEDULE_PCT")
fi

# 可选：将 Critic 势场权重透传（未设置则在训练脚本中回退为 mix_beta/FR）
if [ -n "${CRITIC_PF_WEIGHT:-}" ]; then
    ARGS+=(--critic-pf-weight "${CRITIC_PF_WEIGHT}")
fi
## 可选：仅在 XLA_GLOBAL=1 时才传递布尔参数，避免传入"0"被误判为启用
if [ "${XLA_GLOBAL:-0}" = "1" ]; then
    ARGS+=(--xla-global "1")
fi

# 🔧 随机种子管理：每次训练使用随机种子，但支持复现实验
# 注意：SEED已在脚本开头初始化（第54-66行），这里直接使用
# 随机种子会通过--seed参数传递给训练脚本，并保存到results.json中
ARGS+=(--seed "$SEED")

# 🔧 持续训练模型配置（如果指定了模型路径）
if [ -n "$RESUME_MODEL" ]; then
    # 检查模型路径是否存在
    if [ ! -d "$RESUME_MODEL" ]; then
        echo "⚠️  警告: 指定的模型路径不存在: $RESUME_MODEL"
        echo "   将尝试自动查找检查点..."
        # 尝试查找checkpoint目录
        CHECKPOINT_DIR="${RESUME_MODEL}/checkpoint"
        if [ -d "$CHECKPOINT_DIR" ]; then
            echo "✅ 找到检查点目录: $CHECKPOINT_DIR"
            ARGS+=(--checkpoint "$CHECKPOINT_DIR")
        elif [ -d "${RESUME_MODEL}/final" ]; then
            echo "✅ 找到final目录，使用final作为检查点: ${RESUME_MODEL}/final"
            ARGS+=(--checkpoint "${RESUME_MODEL}/final")
        else
            echo "❌ 错误: 无法找到有效的检查点目录"
            echo "   请确保模型路径包含 checkpoint/ 或 final/ 目录"
            exit 1
        fi
    else
        # 检查是否存在checkpoint目录
        if [ -d "${RESUME_MODEL}/checkpoint" ]; then
            echo "✅ 使用检查点目录: ${RESUME_MODEL}/checkpoint"
            ARGS+=(--checkpoint "${RESUME_MODEL}/checkpoint")
        elif [ -d "${RESUME_MODEL}/final" ]; then
            echo "✅ 使用final目录作为检查点: ${RESUME_MODEL}/final"
            ARGS+=(--checkpoint "${RESUME_MODEL}/final")
        else
            # 直接使用指定的路径作为检查点
            echo "✅ 使用指定路径作为检查点: $RESUME_MODEL"
            ARGS+=(--checkpoint "$RESUME_MODEL")
        fi
    fi
fi

# ========= 奖励参数分组：根据 USE_WEIGHTED_REWARD 切换 =========
if [ "$USE_WEIGHTED_REWARD" = "1" ] || [ "${USE_WEIGHTED_REWARD,,}" = "true" ]; then
    echo "[奖励模式] 分项加权（weighted）"
    # 分项加权求和奖励参数（仅在加权模式下追加）
    ARGS+=(
        --distance-weight "$DISTANCE_WEIGHT"
        --exploration-weight "$EXPLORATION_WEIGHT"
        --stationary-weight "$STATIONARY_WEIGHT"
        --direction-weight "$DIRECTION_WEIGHT"
        --turn-smooth-weight "$TURN_SMOOTH_WEIGHT"
        --deviation-weight "$DEVIATION_WEIGHT"
        --start-area-weight "$START_AREA_WEIGHT"
        --approach-weight "$APPROACH_WEIGHT"
        --energy-weight "$ENERGY_WEIGHT"
        --height-weight "$HEIGHT_WEIGHT"
        --height-reward-enabled "$HEIGHT_REWARD_ENABLED"
        --height-ideal-min "$HEIGHT_IDEAL_MIN"
        --height-ideal-max "$HEIGHT_IDEAL_MAX"
        --lateral-weight "$LATERAL_WEIGHT"
        --clearance-weight "$CLEARANCE_WEIGHT"
        --clearance-d-max "$CLEARANCE_D_MAX"
        --success-weight "$SUCCESS_WEIGHT"
        --collision-weight "$COLLISION_WEIGHT"
        --collision-reduction-weight "$COLLISION_REDUCTION_WEIGHT"  # 🔧 新增：碰撞次数减少奖励权重
        --global-weight "$GLOBAL_WEIGHT"
        --shaping-weight "$SHAPING_WEIGHT"
        --max-reward "$MAX_REWARD"
        --min-reward "$MIN_REWARD"
        --success-reward-value "$SUCCESS_REWARD_VALUE"
        --no-collision-reward-value "$NO_COLLISION_REWARD_VALUE"
        --success-distance-threshold "$SUCCESS_DISTANCE_THRESHOLD"
        --collision-penalty-value "$COLLISION_PENALTY_VALUE"
        --collision-distance-threshold "$COLLISION_DISTANCE_THRESHOLD"
        --global-reward-mode "$GLOBAL_REWARD_MODE"
        --shaping-gamma "$SHAPING_GAMMA"
    )
else
    echo "[奖励模式] 原始（original）"
    # 原始奖励模式：全局缩放参数已在第449-450行传递，无需重复
fi

# 🔧 追加网络架构配置到ARGS（已在环境变量区定义）
if [ -n "$ACTOR_HIDDEN" ]; then
    ARGS+=(--actor-hidden "$ACTOR_HIDDEN")
fi
if [ -n "$CRITIC_HIDDEN" ]; then
    ARGS+=(--critic-hidden "$CRITIC_HIDDEN")
fi
# UPDATE_RATE参数已通过环境变量直接设置，无需额外覆盖逻辑

# 用环境变量覆盖回合步数（可选）
if [ -n "$EPISODE_LENGTH_ENV" ]; then
    for i in "${!ARGS[@]}"; do
        if [ "${ARGS[$i]}" = "--episode-length" ]; then
            ARGS[$((i+1))]="$EPISODE_LENGTH_ENV"
            break
        fi
    done
fi

# 提取实际使用的 episode-length 值并导出为环境变量（供并行worker使用）
for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--episode-length" ]; then
        export EPISODE_LENGTH="${ARGS[$((i+1))]}"
        break
    fi
done

# 条件追加：是否保存位置到文件
if [ "${SAVE_POSITIONS}" = "1" ] || [ "${SAVE_POSITIONS,,}" = "true" ] || [ "${SAVE_POSITIONS,,}" = "yes" ] || [ "${SAVE_POSITIONS,,}" = "on" ]; then
    ARGS+=(--save-positions)  # 保存当前运行中的初始/固定位置
fi

# 条件追加：是否使用固定位置文件（固定起点与目标）
if [ "${USE_FIXED_POSITIONS}" = "1" ] || [ "${USE_FIXED_POSITIONS,,}" = "true" ] || [ "${USE_FIXED_POSITIONS,,}" = "yes" ] || [ "${USE_FIXED_POSITIONS,,}" = "on" ]; then
    ARGS+=(--use-fixed-positions)
    ARGS+=(--positions-file "${POSITIONS_FILE}")
fi

# 条件追加：是否使用固定随机种子来固定地图
if [ "${USE_SCENARIO_SEED}" = "1" ] || [ "${USE_SCENARIO_SEED,,}" = "true" ] || [ "${USE_SCENARIO_SEED,,}" = "yes" ] || [ "${USE_SCENARIO_SEED,,}" = "on" ]; then
    if [ -n "${SCENARIO_SEED}" ]; then
        ARGS+=(--terrain-seed "${SCENARIO_SEED}")
    fi
fi

# 条件追加：是否使用随机地形
if [ "${RANDOM_TERRAIN}" = "1" ] || [ "${RANDOM_TERRAIN,,}" = "true" ] || [ "${RANDOM_TERRAIN,,}" = "yes" ] || [ "${RANDOM_TERRAIN,,}" = "on" ]; then
    ARGS+=(--random-terrain)
fi

# 条件追加：是否启用动态首次运行（首次动态生成位置，后续固定）
if [ "${DYNAMIC_FIRST_TIME}" = "1" ] || [ "${DYNAMIC_FIRST_TIME,,}" = "true" ] || [ "${DYNAMIC_FIRST_TIME,,}" = "yes" ] || [ "${DYNAMIC_FIRST_TIME,,}" = "on" ]; then
    ARGS+=(--dynamic-first-time)
fi

# Add terrain complexity level parameter
ARGS+=(--terrain-complexity-level "$TERRAIN_COMPLEXITY_LEVEL")

# 调试：检查ARGs数组
# 🔧 关键修复：使用更安全的数组长度获取方式，避免shell解析错误
if [ "${DEBUG_ARGS:-0}" = "1" ]; then
    ARGS_COUNT=${#ARGS[@]}
    echo "Total arguments: $ARGS_COUNT"
    for i in "${!ARGS[@]}"; do
        echo "  [$i] ${ARGS[$i]}"
    done
fi

# Fix: Use more stable way to call python, avoid array expansion issues
# Note: In some environments, cmd || STATUS=$? may trigger extra parsing after long logs
# Splitting command into two lines can completely avoid paper3d_train_optimized.py: command not found
# Key fix: Ensure XLA_FLAGS is completely cleared before Python starts - prevent inheritance from parent shell
# Python script already handles XLA_FLAGS cleanup, here just ensure SUPPRESS_MA_PROMPT is set
# 🔧 修复：使用显式路径和引号，避免文件名被错误解析
PYTHON_SCRIPT="paper3d_train_optimized.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到训练脚本 $PYTHON_SCRIPT" >&2
    exit 1
fi
python3 "$PYTHON_SCRIPT" "${ARGS[@]}"
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "❌ 训练失败: Python进程退出码 $STATUS" >&2
    exit $STATUS
fi

echo "✅ 训练成功完成（退出码: $STATUS）"

echo ""
echo "======================================"
echo "训练完成!"
echo "结果保存在:"
echo "  - 日志: logs/$EXP_NAME_WITH_TIMESTAMP/"
echo "  - 模型: models/$EXP_NAME_WITH_TIMESTAMP/"
echo ""
echo "模型目录结构:"
echo "  - best/     : 训练过程中的最佳模型"
echo "  - final/    : 训练完成后的最终模型"
echo "  - ep20/     : 第20回合的模型快照"
echo "  - ep40/     : 第40回合的模型快照"
echo "  - ...       : 其他回合快照"
echo ""
echo "🎲 随机种子信息:"
echo "  - 本次训练使用的随机种子: $SEED"
echo "  - 种子已保存到: logs/$EXP_NAME_WITH_TIMESTAMP/*/results.json"
# 🔧 关键修复：使用单引号包裹复现命令，防止shell误解析为可执行命令
if [ -n "$RESUME_MODEL" ]; then
    echo "  - 复现此实验请运行: SEED=$SEED ./run_optimized.sh $EPISODES $BATCH_SIZE '$EXP_NAME' $USE_WEIGHTED_REWARD $ALGORITHM '$RESUME_MODEL'"
else
    echo "  - 复现此实验请运行: SEED=$SEED ./run_optimized.sh $EPISODES $BATCH_SIZE '$EXP_NAME' $USE_WEIGHTED_REWARD $ALGORITHM"
fi
echo ""
echo "🔧 持续训练使用示例:"
echo "  # 方式1：从指定模型目录恢复训练（自动查找checkpoint或final目录）"
echo "  ./run_optimized.sh 200 1024 '继续训练' 1 matd3 'models/my_experiment_20240101_120000'"
echo ""
echo "  # 方式2：通过环境变量指定模型路径"
echo "  RESUME_MODEL_ENV='models/my_experiment_20240101_120000' ./run_optimized.sh 200 1024 '继续训练' 1 matd3"
echo ""
echo "  # 方式3：使用CHECKPOINT_MODEL环境变量"
echo "  CHECKPOINT_MODEL='models/my_experiment_20240101_120000/checkpoint' ./run_optimized.sh 200 1024 '继续训练' 1 matd3"
echo ""

# === 🔧 自动测试评估（使用相同环境配置）===
# 检查是否启用自动评估（可通过环境变量控制）
AUTO_EVAL=${AUTO_EVAL:-0}  # 默认启用自动评估
EVAL_EPISODES=${EVAL_EPISODES:-5}  # 默认评估5个回合

if [ "${AUTO_EVAL}" = "1" ] || [ "${AUTO_EVAL,,}" = "true" ] || [ "${AUTO_EVAL,,}" = "yes" ] || [ "${AUTO_EVAL,,}" = "on" ]; then
    echo "======================================"
    echo "🔬 开始自动测试评估（使用相同环境配置）"
    echo "======================================"
    
    # 确定要评估的模型（优先使用final，如果不存在则使用best）
    MODEL_TO_EVAL=""
    if [ -d "models/$EXP_NAME_WITH_TIMESTAMP/final" ]; then
        MODEL_TO_EVAL="models/$EXP_NAME_WITH_TIMESTAMP/final"
        echo "✅ 使用最终模型: $MODEL_TO_EVAL"
    elif [ -d "models/$EXP_NAME_WITH_TIMESTAMP/best" ]; then
        MODEL_TO_EVAL="models/$EXP_NAME_WITH_TIMESTAMP/best"
        echo "✅ 使用最佳模型: $MODEL_TO_EVAL"
    else
        echo "⚠️  警告: 未找到可评估的模型（final 或 best）"
        echo "   跳过自动评估"
        MODEL_TO_EVAL=""
    fi
    
    if [ -n "$MODEL_TO_EVAL" ]; then
        # 设置评估结果保存路径（保存到相同的logs目录）
        EVAL_SAVE_PATH="logs/$EXP_NAME_WITH_TIMESTAMP/evaluation"
        mkdir -p "$EVAL_SAVE_PATH"
        
        echo ""
        echo "评估配置:"
        echo "  - 模型路径: $MODEL_TO_EVAL"
        echo "  - 评估回合数: $EVAL_EPISODES"
        echo "  - 结果保存路径: $EVAL_SAVE_PATH"
        echo "  - 使用固定位置: ${USE_FIXED_POSITIONS:-1}"
        echo "  - 固定位置文件: ${POSITIONS_FILE:-./saved_positions/5.json}"
        echo "  - 地形复杂度: ${TERRAIN_COMPLEXITY_LEVEL:-3}"
        echo "  - 场景种子: ${SCENARIO_SEED:-88}"
        echo "  - 势场修正: ${USE_TF_POTENTIAL_FIELD:-1} [1=启用TF版本，0=禁用]"
        echo "  - FR值测试序列: 0.55, 0.65, 0.75, 0.85, 0.95 (每个回合使用不同值)"
        echo ""
        
        # 🔧 关键：使用与训练相同的环境配置
        # 1. 使用相同的固定位置文件（如果训练时使用了）
        # 2. 使用相同的地形配置（SCENARIO_SEED、TERRAIN_COMPLEXITY_LEVEL）
        # 3. 每个评估回合使用不同的FR值进行测试
        # 4. 禁用提前终止，确保完整轨迹
        
        # 准备评估环境变量（继承训练时的配置）
        EVAL_ENV=()
        
        # 地形和位置配置（与训练一致）
        if [ "${USE_FIXED_POSITIONS}" = "1" ] || [ "${USE_FIXED_POSITIONS,,}" = "true" ]; then
            EVAL_ENV+=("USE_FIXED_POSITIONS=1")
            if [ -n "${POSITIONS_FILE}" ]; then
                EVAL_ENV+=("POSITIONS_FILE=${POSITIONS_FILE}")
            fi
        fi
        
        # 地形配置（与训练一致）
        if [ -n "${SCENARIO_SEED}" ]; then
            EVAL_ENV+=("USE_SCENARIO_SEED=1")
            EVAL_ENV+=("SCENARIO_SEED=${SCENARIO_SEED}")
        fi
        if [ -n "${TERRAIN_COMPLEXITY_LEVEL}" ]; then
            EVAL_ENV+=("TERRAIN_COMPLEXITY_LEVEL=${TERRAIN_COMPLEXITY_LEVEL}")
        fi
        if [ -n "${MAP_SIZE}" ]; then
            EVAL_ENV+=("MAP_SIZE=${MAP_SIZE}")
        fi
        if [ -n "${MOUNTAIN_MIN_DISTANCE}" ]; then
            EVAL_ENV+=("MOUNTAIN_MIN_DISTANCE=${MOUNTAIN_MIN_DISTANCE}")
        fi
        
        # 🔧 关键修复：确保势场修正启用
        # 势场修正生效条件：USE_TF_POTENTIAL_FIELD=1 AND ACTION_FORCE_RATIO > 0.0
        if [ -n "${USE_TF_POTENTIAL_FIELD}" ]; then
            EVAL_ENV+=("USE_TF_POTENTIAL_FIELD=${USE_TF_POTENTIAL_FIELD}")
        else
            # 如果未设置，默认启用TF版本势场修正
            EVAL_ENV+=("USE_TF_POTENTIAL_FIELD=1")
        fi
        
        # 🔧 关键修复：传递训练时的DELTA参数（确保传统APF评估时DELTA_*=0.0）
        # 传统APF训练时DELTA_*=0.0，评估时必须保持一致，否则势场参数范围会变化
        if [ -n "${DELTA_K_ATT}" ]; then
            EVAL_ENV+=("DELTA_K_ATT=${DELTA_K_ATT}")
        fi
        if [ -n "${DELTA_LAMBDA_1}" ]; then
            EVAL_ENV+=("DELTA_LAMBDA_1=${DELTA_LAMBDA_1}")
        fi
        if [ -n "${DELTA_K_REP}" ]; then
            EVAL_ENV+=("DELTA_K_REP=${DELTA_K_REP}")
        fi
        if [ -n "${DELTA_RADIUS}" ]; then
            EVAL_ENV+=("DELTA_RADIUS=${DELTA_RADIUS}")
        fi
        
        # 🔧 关键修复：传递训练时的基准参数（确保与训练时一致）
        if [ -n "${GOAL_ATTRACTION}" ]; then
            EVAL_ENV+=("GOAL_ATTRACTION=${GOAL_ATTRACTION}")
        fi
        if [ -n "${LAMBDA_1_BASE}" ]; then
            EVAL_ENV+=("LAMBDA_1_BASE=${LAMBDA_1_BASE}")
        fi
        if [ -n "${TERRAIN_REPULSION}" ]; then
            EVAL_ENV+=("TERRAIN_REPULSION=${TERRAIN_REPULSION}")
        fi
        if [ -n "${AGENT_INFLUENCE_RANGE}" ]; then
            EVAL_ENV+=("AGENT_INFLUENCE_RANGE=${AGENT_INFLUENCE_RANGE}")
        fi
        
        # 🔧 关键修复：传递训练时的特征标志（确保与训练时一致）
        if [ -n "${USE_FR_FEATURE}" ]; then
            EVAL_ENV+=("USE_FR_FEATURE=${USE_FR_FEATURE}")
        fi
        if [ -n "${USE_PF_FEATURE}" ]; then
            EVAL_ENV+=("USE_PF_FEATURE=${USE_PF_FEATURE}")
        fi
        
        # 🔧 关键修复：评估时禁用FR schedule（使用固定的FR值进行测试）
        # 评估时每个回合使用不同的FR值（0.55, 0.65, 0.75, 0.85, 0.95），不需要schedule
        EVAL_ENV+=("ACTION_FORCE_RATIO_SCHEDULE_PCT=")  # 设置为空字符串，禁用schedule
        
        # 🔧 关键修复：确保评估时禁用训练相关功能
        EVAL_ENV+=("NOISE_SCALE=0.0")  # 评估时禁用噪声
        EVAL_ENV+=("RANDOM_ACTION_PROB=0.0")  # 评估时禁用随机动作
        EVAL_ENV+=("RANDOM_ACTION_PROB_TRAINING=0.0")  # 评估时禁用训练随机动作
        EVAL_ENV+=("PER_ENABLED=0")  # 评估时禁用PER（不需要经验回放）
        EVAL_ENV+=("LEARNING_WARMUP_ENABLED=0")  # 评估时禁用预热
        EVAL_ENV+=("SAVE_MODEL=0")  # 评估时禁用模型保存
        EVAL_ENV+=("ADAPTIVE_PATIENCE=999999")  # 评估时禁用自适应学习（设置极大值）
        
        # 场景配置（与训练一致）
        if [ -n "${SCENARIO_NAME}" ]; then
            EVAL_ENV+=("SCENARIO_NAME=${SCENARIO_NAME}")
        fi
        
        # 禁用提前终止，确保完整轨迹
        EVAL_ENV+=("DISABLE_EARLY_TERMINATION=true")
        
        # 🔧 启用交互式HTML和动作时序图（与训练时的最佳回合图效果一致）
        EVAL_ENV+=("SAVE_INTERACTIVE_TRAJ=1")  # 确保启用交互式轨迹图
        EVAL_ENV+=("ENABLE_OVERLAY=1")  # 启用overlay图片生成
        
        # 安静输出（减少评估时的日志）
        EVAL_ENV+=("QUIET_OUTPUT=1")
        EVAL_ENV+=("TQDM_DISABLE=1")
        
        # 🔧 关键修复：从训练配置（results.json）中读取训练时使用的ACTION_FORCE_RATIO
        # 如果是apf_learnable实验（FR=1.0），评估时也应该使用FR=1.0，而不是0.55-0.95
        TRAINING_FR=""
        RESULTS_JSON_PATH=""
        
        # 查找results.json（可能在logs目录中）
        if [ -d "logs/$EXP_NAME_WITH_TIMESTAMP" ]; then
            # 在logs目录及其子目录中查找results.json
            RESULTS_JSON_PATH=$(find "logs/$EXP_NAME_WITH_TIMESTAMP" -name "results.json" -type f | head -1)
        fi
        
        if [ -n "$RESULTS_JSON_PATH" ] && [ -f "$RESULTS_JSON_PATH" ]; then
            # 使用Python读取results.json中的action_force_ratio
            TRAINING_FR=$(python3 <<PYTHON_EOF
import json
import sys
try:
    with open("$RESULTS_JSON_PATH", 'r', encoding='utf-8') as f:
        results = json.load(f)
    # 优先从args中读取action_force_ratio
    if 'args' in results and isinstance(results['args'], dict):
        if 'action_force_ratio' in results['args']:
            print(results['args']['action_force_ratio'])
            sys.exit(0)
    # 如果没有，尝试从顶层读取
    if 'action_force_ratio' in results:
        print(results['action_force_ratio'])
        sys.exit(0)
except Exception as e:
    pass
sys.exit(1)
PYTHON_EOF
)
            if [ -n "$TRAINING_FR" ]; then
                echo "✅ 从训练配置读取ACTION_FORCE_RATIO: $TRAINING_FR"
            fi
        fi
        
        # 🔧 关键修复：根据训练时的FR值决定评估策略
        # 如果训练时FR=1.0（apf_learnable），评估时也应该使用FR=1.0
        # 如果训练时FR<1.0（action_apf_fusion），评估时使用多个FR值进行测试
        if [ -n "$TRAINING_FR" ]; then
            # 使用Python进行浮点数比较（更可靠）
            IS_APF_LEARNABLE=$(python3 <<PYTHON_EOF
try:
    fr = float("$TRAINING_FR")
    if fr >= 0.99:
        print("1")
    else:
        print("0")
except:
    print("0")
PYTHON_EOF
)
            if [ "$IS_APF_LEARNABLE" = "1" ]; then
                # 训练时FR=1.0（apf_learnable），评估时也使用FR=1.0
                FR_VALUES=(1.0)
                echo "🔧 检测到apf_learnable实验（训练时FR=$TRAINING_FR），评估时使用FR=1.0"
            else
                # 训练时FR<1.0（action_apf_fusion），评估时使用多个FR值进行测试
                FR_VALUES=(0.55 0.65 0.75 0.85 0.95)
                echo "🔧 检测到action_apf_fusion实验（训练时FR=$TRAINING_FR），评估时使用多个FR值进行测试"
            fi
        else
            # 如果无法读取训练时的FR值，默认使用多个FR值进行测试
            FR_VALUES=(0.55 0.65 0.75 0.85 0.95)
            echo "⚠️  无法读取训练时的ACTION_FORCE_RATIO，默认使用多个FR值进行测试"
        fi
        
        echo "  FR值序列: ${FR_VALUES[*]}"
        echo ""
        
        # 运行每个评估回合（每个回合使用不同的FR值）
        echo "正在运行评估（每个回合使用不同的FR值）..."
        echo ""
        
        EVAL_POSITIONS_FILE="${POSITIONS_FILE:-./saved_positions/5.json}"
        EVAL_USE_FIXED="${USE_FIXED_POSITIONS:-1}"
        EVAL_CMD="./run_evaluation.sh"
        
        EVAL_STATUS=0
        for ep_idx in $(seq 0 $((EVAL_EPISODES - 1))); do
            # 选择对应的FR值（循环使用）
            FR_INDEX=$((ep_idx % ${#FR_VALUES[@]}))
            CURRENT_FR=${FR_VALUES[$FR_INDEX]}
            
            echo "======================================"
            echo "🔬 评估回合 $((ep_idx + 1))/$EVAL_EPISODES"
            echo "  FR值: $CURRENT_FR"
            echo "======================================"
            
            # 为当前回合设置FR值
            EVAL_ENV_CURRENT=("${EVAL_ENV[@]}")
            EVAL_ENV_CURRENT+=("ACTION_FORCE_RATIO=${CURRENT_FR}")
            
            # 设置当前回合的保存路径（每个回合单独保存）
            EVAL_SAVE_PATH_EPISODE="${EVAL_SAVE_PATH}/episode_$((ep_idx + 1))_fr${CURRENT_FR}"
            mkdir -p "$EVAL_SAVE_PATH_EPISODE"
            
            # 构建评估命令（每个回合只评估1个episode）
            EVAL_CMD_ARGS=(
                "$MODEL_TO_EVAL"
                "1"  # 每个回合只评估1个episode
                "$EVAL_SAVE_PATH_EPISODE"
                "$EVAL_POSITIONS_FILE"
                "$EVAL_USE_FIXED"
                "true"  # 禁用提前终止
            )
            
            # 在评估环境中运行（设置当前FR值）
            for env_var in "${EVAL_ENV_CURRENT[@]}"; do
                export "$env_var"
            done
            
            # 🔧 关键修复：打印当前评估配置，便于调试
            echo "📋 当前评估配置:"
            echo "  ACTION_FORCE_RATIO=${CURRENT_FR}"
            echo "  USE_TF_POTENTIAL_FIELD=${USE_TF_POTENTIAL_FIELD:-1}"
            echo "  DELTA_K_ATT=${DELTA_K_ATT:-未设置}"
            echo "  DELTA_LAMBDA_1=${DELTA_LAMBDA_1:-未设置}"
            echo "  DELTA_K_REP=${DELTA_K_REP:-未设置}"
            echo "  DELTA_RADIUS=${DELTA_RADIUS:-未设置}"
            echo "  USE_FR_FEATURE=${USE_FR_FEATURE:-未设置}"
            echo "  USE_PF_FEATURE=${USE_PF_FEATURE:-未设置}"
            echo ""
            
            # 运行评估脚本
            "$EVAL_CMD" "${EVAL_CMD_ARGS[@]}"
            EPISODE_STATUS=$?
            
            if [ "${EPISODE_STATUS:-0}" -ne 0 ]; then
                echo "⚠️  回合 $((ep_idx + 1)) 评估失败，退出码: ${EPISODE_STATUS}"
                EVAL_STATUS=${EPISODE_STATUS}
            else
                echo "✅ 回合 $((ep_idx + 1)) 评估完成 (FR=${CURRENT_FR})"
            fi
            echo ""
        done
        
        echo ""
        echo "======================================"
        if [ "${EVAL_STATUS:-0}" -eq 0 ]; then
            echo "✅ 自动评估完成!"
            echo ""
            echo "评估结果保存在:"
            echo "  - 每个回合的评估结果保存在独立目录:"
            for ep_idx in $(seq 0 $((EVAL_EPISODES - 1))); do
                FR_INDEX=$((ep_idx % ${#FR_VALUES[@]}))
                CURRENT_FR=${FR_VALUES[$FR_INDEX]}
                EPISODE_DIR="${EVAL_SAVE_PATH}/episode_$((ep_idx + 1))_fr${CURRENT_FR}"
                echo "    - 回合 $((ep_idx + 1)) (FR=${CURRENT_FR}): $EPISODE_DIR"
                echo "      - 评估统计: $EPISODE_DIR/evaluation_results.json"
                echo "      - 轨迹图片: $EPISODE_DIR/trajectory_*.png"
                echo "      - 轨迹动画: $EPISODE_DIR/trajectory_*.gif"
                echo "      - 交互式HTML: $EPISODE_DIR/trajectory_*_interactive.html"
                echo "      - Overlay图片: $EPISODE_DIR/trajectory_*_overlay.png"
            done
            if [ -f "$EVAL_SAVE_PATH/evaluation_results.json" ]; then
                echo ""
                echo "📊 评估摘要:"
                # 尝试提取关键指标（如果JSON格式允许）
                if command -v python3 >/dev/null 2>&1; then
                    python3 <<PYTHON_EOF
import json
import sys
try:
    with open("$EVAL_SAVE_PATH/evaluation_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" in data:
        summary = data["summary"]
        print("  - 平均奖励: {:.2f}".format(summary.get('avg_reward', 0) if summary.get('avg_reward') != 'N/A' else 0))
        print("  - 成功率: {:.1%}".format(summary.get('success_rate', 0) if summary.get('success_rate') != 'N/A' else 0))
        print("  - 平均步数: {:.1f}".format(summary.get('avg_steps', 0) if summary.get('avg_steps') != 'N/A' else 0))
    elif "episodes" in data and len(data["episodes"]) > 0:
        rewards = [ep.get("total_reward", 0) for ep in data["episodes"]]
        successes = [ep.get("success", False) for ep in data["episodes"]]
        steps = [ep.get("steps", 0) for ep in data["episodes"]]
        if rewards:
            print("  - 平均奖励: {:.2f}".format(sum(rewards)/len(rewards)))
        if successes:
            print("  - 成功率: {:.1%}".format(sum(successes)/len(successes)))
        if steps:
            print("  - 平均步数: {:.1f}".format(sum(steps)/len(steps)))
except Exception as e:
    print("  ⚠️  无法解析评估结果: {}".format(e))
PYTHON_EOF
                fi
            fi
        else
            echo "⚠️  自动评估失败，退出码: ${EVAL_STATUS}"
            echo "   可以手动运行评估:"
            echo "   ./run_evaluation.sh $MODEL_TO_EVAL $EVAL_EPISODES"
        fi
        echo "======================================"
        echo ""
    fi
else
    echo "ℹ️  自动评估已禁用（设置 AUTO_EVAL=1 启用）"
    echo ""
fi

echo "评估命令示例:"
echo "  ./run_evaluation.sh models/$EXP_NAME_WITH_TIMESTAMP/best"
echo "  ./run_evaluation.sh models/$EXP_NAME_WITH_TIMESTAMP/final"
echo ""
echo "复现实验示例:"
echo "  # 方法1：从results.json中读取随机种子，然后复现实验"
echo "  # 查看随机种子: cat logs/$EXP_NAME_WITH_TIMESTAMP/*/results.json | grep '\"seed\"'"
echo "  # 然后使用提取的种子值设置SEED环境变量"
echo "  SEED=123456 ./run_optimized.sh 200 2048 \"复现实验\" 1 matd3"
echo ""
echo "  # 方法2：直接使用训练完成时显示的复现命令（见上方）"
echo ""
echo "向量化优化使用示例:"
echo "  # 启用所有向量化优化"
echo "  USE_VECTORIZATION=1 VECTORIZED_REWARDS=1 VECTORIZED_OBSERVATIONS=1 ./run_optimized.sh 200 2048 \"vectorized_exp\" 1"
echo ""
echo "  # 仅启用观察向量化"
echo "  VECTORIZED_OBSERVATIONS=1 VECTORIZED_REWARDS=0 ./run_optimized.sh 200 2048 \"obs_vectorized_exp\" 1"
echo ""
echo "  # 使用向量化场景（实验性）"
echo "  VECTORIZED_SCENARIO=1 ./run_optimized.sh 200 2048 \"vectorized_scenario_exp\" 1"
echo ""
echo "  # 性能对比测试"
echo "  USE_VECTORIZATION=0 ./run_optimized.sh 50 1024 \"baseline_exp\" 1  # 基线"
echo "  USE_VECTORIZATION=1 ./run_optimized.sh 50 1024 \"optimized_exp\" 1  # 优化版"
echo ""
echo "  # 禁用自动评估"
echo "  AUTO_EVAL=0 ./run_optimized.sh 200 2048 \"no_auto_eval_exp\" 1"
echo ""
echo "  # 自定义评估回合数"
echo "  EVAL_EPISODES=10 ./run_optimized.sh 200 2048 \"custom_eval_exp\" 1"
echo "======================================"

# 🔧 关键修复：显式退出脚本，防止意外继续执行
exit 0

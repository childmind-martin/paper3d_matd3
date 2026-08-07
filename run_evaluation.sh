#!/bin/bash

# MADDPG优化版模型评估一键运行脚本

# 🔧 关键修复：确保从脚本所在目录运行，避免相对路径问题
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 手动运行评估脚本时，使用与训练入口相同的项目解释器发现顺序。
# 不能默认落到系统 python3：该解释器在本项目机器上不包含 TensorFlow，
# 会让参数解析完成后才以 ModuleNotFoundError 失败。
if [ -z "${EVAL_PYTHON_BIN:-}" ]; then
    if [ -n "${TRAIN_PYTHON_BIN:-}" ]; then
        EVAL_PYTHON_BIN="$TRAIN_PYTHON_BIN"
    else
        for _MATD3_EVAL_CANDIDATE_PYTHON in \
            "$SCRIPT_DIR/.venv/bin/python3" \
            "/home/tang/miniconda3/envs/maddpg_env/bin/python3" \
            "${HOME:-}/miniconda3/envs/maddpg_env/bin/python3" \
            "${HOME:-}/anaconda3/envs/maddpg_env/bin/python3"
        do
            if [ -x "$_MATD3_EVAL_CANDIDATE_PYTHON" ] \
               && "$_MATD3_EVAL_CANDIDATE_PYTHON" -c \
                    'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)' \
                    >/dev/null 2>&1; then
                EVAL_PYTHON_BIN="$_MATD3_EVAL_CANDIDATE_PYTHON"
                break
            fi
        done
        unset _MATD3_EVAL_CANDIDATE_PYTHON
    fi
fi
EVAL_PYTHON_BIN=${EVAL_PYTHON_BIN:-python3}
export EVAL_PYTHON_BIN

resolve_default_model_path() {
    if [ -n "${MODEL_PATH:-}" ]; then
        printf '%s\n' "$MODEL_PATH"
        return
    fi

    local candidate=""
    local latest_exp=""

    while IFS= read -r exp_dir; do
        [ -z "$exp_dir" ] && continue
        latest_exp="$exp_dir"
        break
    done < <(find models -mindepth 1 -maxdepth 1 -type d \
        \( -name "*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" -o -name "optimized_exp*" \) \
        2>/dev/null | sort -r)

    if [ -n "$latest_exp" ]; then
        candidate="$latest_exp"
    elif [ -d "models/optimized_exp" ]; then
        candidate="models/optimized_exp"
    else
        candidate="models/optimized_exp"
    fi

    printf '%s\n' "$candidate"
}

# 设置默认参数
MODEL_PATH=${1:-"$(resolve_default_model_path)"}
EVAL_EPISODES=${2:-3}
MODEL_VARIANT=${MODEL_VARIANT:-auto}  # auto|final|best|best_by_team_sr|latest_ep

# 🔧 关键修复：规范化模型路径，移除多余的斜杠
MODEL_PATH=$(echo "$MODEL_PATH" | sed 's|//|/|g' | sed 's|/$||')
SAVE_PATH=${3:-"evaluation_results/$(basename "$MODEL_PATH")_$(date +%Y%m%d_%H%M%S)"}
POSITIONS_FILE=${4:-"./saved_positions/5.json"}
USE_FIXED_POSITIONS=${5:-false}
DISABLE_EARLY_TERMINATION=${6:-false}  # 默认允许提前终止，避免日常测试硬跑满全长
# 严格验证模式：默认开启。要求评估关键参数必须与训练配置一致，否则退出
STRICT_EVAL_MATCH=${STRICT_EVAL_MATCH:-1}
EVAL_EPISODE_LENGTH_MULTIPLIER=${EVAL_EPISODE_LENGTH_MULTIPLIER:-${POST_EVAL_EPISODE_LENGTH_MULTIPLIER:-1}}
EVAL_EPISODE_PARALLELISM=${EVAL_EPISODE_PARALLELISM:-1}
EVAL_ENV_STEP_THREADS=${EVAL_ENV_STEP_THREADS:-1}
EVAL_NOISE_SCALE=${EVAL_NOISE_SCALE:-0.0}
EVAL_RANDOM_ACTION_PROB=${EVAL_RANDOM_ACTION_PROB:-0.0}
EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-}
APF_BACKEND=${APF_BACKEND:-python_original}

echo ""
echo "评估参数:"
echo "  - 模型路径: $MODEL_PATH"
echo "  - 评估回合数: $EVAL_EPISODES"
echo "  - 模型选择: $MODEL_VARIANT"
echo "  - 结果保存路径: $SAVE_PATH"
echo "  - 固定位置文件: $POSITIONS_FILE"
echo "  - 使用固定位置: $USE_FIXED_POSITIONS"
echo "  - 禁用提前终止: $DISABLE_EARLY_TERMINATION"
echo "  - 严格对齐训练配置: $STRICT_EVAL_MATCH"
echo "  - Episode内并行(方案A): $EVAL_EPISODE_PARALLELISM"
echo "  - env.step线程数(方案A): $EVAL_ENV_STEP_THREADS"
echo "  - 评估动作噪声: scale=$EVAL_NOISE_SCALE, random_prob=$EVAL_RANDOM_ACTION_PROB, seed=${EVAL_NOISE_SEED:-<random>}"
echo "  - APF后端: $APF_BACKEND"
echo ""

# 🔧 关键修复：确保路径变量正确引用，支持中文路径
# 检查模型是否存在（确保路径被正确引用）
# 使用绝对路径检查，避免中文路径问题
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 找不到模型路径 $MODEL_PATH"
    echo ""
    echo "🔍 正在搜索可用的模型..."
    
    # 搜索所有可用的模型目录
    echo "可用的模型路径:"
    
    # 🔧 关键修复：搜索所有模型目录，包括中文路径
    # 搜索带时间戳的模型
    for model_dir in models/optimized_exp_*; do
        if [ -d "$model_dir" ]; then
            timestamp=$(basename "$model_dir" | sed 's/optimized_exp_//')
            echo "  📅 $model_dir/best_by_team_sr - Team SR最佳模型 (时间戳: $timestamp)"
            echo "  📅 $model_dir/best     - 训练过程中的最佳模型 (时间戳: $timestamp)"
            echo "  📅 $model_dir/final    - 训练完成后的最终模型 (时间戳: $timestamp)"
            echo "  📅 $model_dir/ep500    - 第500轮的模型快照 (时间戳: $timestamp)"
        fi
    done
    
    # 🔧 关键修复：搜索所有包含时间戳的模型目录（包括中文路径）
    for model_dir in models/*_*_*; do
        if [ -d "$model_dir" ] && [[ "$model_dir" =~ [0-9]{8}_[0-9]{6}$ ]]; then
            model_name=$(basename "$model_dir")
            echo "  📁 $model_dir/best_by_team_sr - Team SR最佳模型 ($model_name)"
            echo "  📁 $model_dir/best     - 最佳模型 ($model_name)"
            echo "  📁 $model_dir/final    - 最终模型 ($model_name)"
        fi
    done
    
    # 搜索不带时间戳的旧模型
    if [ -d "models/optimized_exp" ]; then
        echo "  📁 models/optimized_exp/best_by_team_sr - Team SR最佳模型 (旧版本)"
        echo "  📁 models/optimized_exp/best     - 训练过程中的最佳模型 (旧版本)"
        echo "  📁 models/optimized_exp/final    - 训练完成后的最终模型 (旧版本)"
        echo "  📁 models/optimized_exp/ep500    - 第500轮的模型快照 (旧版本)"
    fi
    
    echo ""
    echo "使用方法:"
    echo "  $0 [模型路径] [评估回合数] [保存路径] [固定位置文件] [是否使用固定位置] [是否禁用提前终止]"
    echo "  MODEL_VARIANT=best_by_team_sr $0 [实验目录] [评估回合数]"
    echo ""
    echo "示例:"
    echo "  # 直接评估 Team SR 最佳模型"
    echo "  $0 models/optimized_exp_20250121_143022/best_by_team_sr 1"
    echo "  # 只给实验目录，通过 MODEL_VARIANT 自动选择 Team SR 最佳模型"
    echo "  MODEL_VARIANT=best_by_team_sr $0 models/optimized_exp_20250121_143022 1"
    echo "  # 评估最新训练的最佳模型"
    echo "  $0 models/optimized_exp_20250121_143022/best 1"
    echo "  # 评估最新训练的最终模型"
    echo "  $0 models/optimized_exp_20250121_143022/final 1"
    echo "  # 使用固定位置"
    echo "  $0 models/optimized_exp_20250121_143022/best 1 eval_results ./saved_positions/my_positions.json true"
    echo "  # 生成完整轨迹GIF（禁用提前终止）"
    echo "  $0 models/optimized_exp_20250121_143022/best 1 full_trajectory '' false true"
    echo ""
    echo "势场参数环境变量:"
    echo "  ACTION_FORCE_RATIO=0.2        # 势场力强度系数 (0.0=无势场, 1.0=满势场)"
    echo "  ENABLE_ACTION_CORRECTION=false # 禁用势场修正"
    echo "  INFLUENCE_RANGE=3.0           # 势场影响范围"
    echo "  FORCE_PARAM_RATIO=0.8         # 势场参数调整基准系数"
    echo "  TERRAIN_COMPLEXITY_LEVEL=2     # 地形复杂度等级 (1-4)"
    echo ""
    echo "地形参数示例:"
    echo "  # 使用固定地形"
    echo "  RANDOM_TERRAIN=0 $0 models/optimized_exp/best 3"
    echo "  # 使用随机地形（默认；每回合生成新地形）"
    echo "  RANDOM_TERRAIN=1 $0 models/optimized_exp/best 3"
    echo "  # 调整地形复杂度"
    echo "  TERRAIN_COMPLEXITY_LEVEL=3 $0 models/optimized_exp/best 3"
    echo ""
    echo "势场参数示例:"
    echo "  ACTION_FORCE_RATIO=0.8 $0 models/optimized_exp/best 3"
    echo "  ENABLE_ACTION_CORRECTION=false $0 models/optimized_exp/best 3"
    echo ""
    echo "参数详细说明:"
    echo "  ACTION_FORCE_RATIO: 势场力强度系数"
    echo "    - 0.0: 完全禁用势场修正，只使用网络动作"
    echo "    - 0.05: 极低势场依赖，主要依赖网络输出（推荐测试）"
    echo "    - 0.2: 低势场依赖，平衡网络和势场"
    echo "    - 0.5: 势场力强度减半，平衡网络和势场"
    echo "    - 0.8: 势场力强度80%，主要依赖势场引导"
    echo "    - 1.0: 势场力强度100%，完全依赖势场"
    echo "  FORCE_PARAM_RATIO: 势场参数调整基准系数"
    echo "    - 控制势场参数的整体缩放比例"
    echo "    - 影响目标吸引力、地形排斥力、智能体排斥力的强度"
    echo "  RANDOM_TERRAIN: 地形模式选择"
    echo "    - 0: 固定地形"
    echo "    - 1: 随机地形（每回合生成新地形，当前默认）"
    echo "  TERRAIN_COMPLEXITY_LEVEL: 地形复杂度等级"
    echo "    - 1: 简单地形 (1个山峰, 4个障碍物)"
    echo "    - 2: 中等地形 (2个山峰, 8个障碍物)"
    echo "    - 3: 困难地形 (3个山峰, 12个障碍物, 有峡谷)"
    echo "    - 4: 极难地形 (4个山峰, 16个障碍物, 有峡谷)"
    echo "    - 未设置: 随机选择复杂度等级"
    echo ""
    echo "💡 提示: 可以使用最新的时间戳模型进行评估"
    exit 1
fi

# 🔧 关键修复：自动检测权重文件位置
# 如果指定的路径是目录但没有权重文件，自动查找子目录
if [ -d "$MODEL_PATH" ]; then
    # 检查当前目录是否有权重文件（至少有一个actor文件）
    HAS_WEIGHTS=$(find "$MODEL_PATH" -maxdepth 1 -name "actor_*.weights.h5" -type f 2>/dev/null | head -1)
    
    if [ -z "$HAS_WEIGHTS" ]; then
        # 没有权重文件，尝试在子目录中查找
        echo "🔍 在指定目录中未找到权重文件，正在搜索子目录..."
        
        # 优先级可通过 MODEL_VARIANT 控制
        FOUND_PATH=""
        MODEL_VARIANT_NORMALIZED=$(printf '%s' "$MODEL_VARIANT" | tr '[:upper:]' '[:lower:]')
        case "$MODEL_VARIANT_NORMALIZED" in
            best_by_team_sr|team_sr|sr)
                SEARCH_PRIORITY=("best_by_team_sr" "best" "final" "ep_latest")
                ;;
            best)
                SEARCH_PRIORITY=("best" "best_by_team_sr" "final" "ep_latest")
                ;;
            final)
                SEARCH_PRIORITY=("final" "best" "best_by_team_sr" "ep_latest")
                ;;
            latest_ep|ep|episode)
                SEARCH_PRIORITY=("ep_latest" "final" "best" "best_by_team_sr")
                ;;
            auto|"")
                SEARCH_PRIORITY=("best_by_team_sr" "final" "best" "ep_latest")
                ;;
            *)
                echo "⚠️ 未识别的 MODEL_VARIANT=$MODEL_VARIANT，回退为 auto"
                SEARCH_PRIORITY=("best_by_team_sr" "final" "best" "ep_latest")
                ;;
        esac
        
        for search_target in "${SEARCH_PRIORITY[@]}"; do
            if [ -n "$FOUND_PATH" ]; then
                break
            fi
            if [ "$search_target" = "ep_latest" ]; then
                while IFS= read -r ep_dir; do
                    [ -z "$ep_dir" ] && continue
                    EP_WEIGHTS=$(find "$ep_dir" -maxdepth 1 -name "actor_*.weights.h5" -type f 2>/dev/null | head -1)
                    if [ -n "$EP_WEIGHTS" ]; then
                        FOUND_PATH="$ep_dir"
                        echo "✅ 找到权重文件: $FOUND_PATH"
                        break
                    fi
                done < <(find "$MODEL_PATH" -maxdepth 1 -type d -name "ep*" 2>/dev/null | sort -V -r)
            else
                CANDIDATE_DIR="$MODEL_PATH/$search_target"
                if [ -d "$CANDIDATE_DIR" ]; then
                    CANDIDATE_WEIGHTS=$(find "$CANDIDATE_DIR" -maxdepth 1 -name "actor_*.weights.h5" -type f 2>/dev/null | head -1)
                    if [ -n "$CANDIDATE_WEIGHTS" ]; then
                        FOUND_PATH="$CANDIDATE_DIR"
                        echo "✅ 找到权重文件: $FOUND_PATH"
                    fi
                fi
            fi
        done
        
        # 如果找到了，更新 MODEL_PATH
        if [ -n "$FOUND_PATH" ]; then
            MODEL_PATH="$FOUND_PATH"
            echo "📁 使用模型路径: $MODEL_PATH"
        else
            echo "❌ 错误: 在 $MODEL_PATH 及其子目录中未找到权重文件"
            echo ""
            echo "请检查以下目录:"
            echo "  - $MODEL_PATH/best_by_team_sr"
            echo "  - $MODEL_PATH/final"
            echo "  - $MODEL_PATH/best"
            echo "  - $MODEL_PATH/ep*"
            exit 1
        fi
    else
        echo "✅ 在指定目录中找到权重文件: $MODEL_PATH"
    fi
fi

# 预置关键参数容器（后续优先从训练配置读取）
TRAINING_SCENARIO_NAME=""
TRAINING_ALGORITHM=""
TRAINING_EPISODE_LENGTH=""
TRAINING_SUCCESS_DISTANCE_THRESHOLD=""
TRAINING_COLLISION_DISTANCE_THRESHOLD=""
TRAINING_TERRAIN_CONTACT_EPS=""
TRAINING_GRAVITY=""
TRAINING_CONTROL_ACCEL_GAIN=""
TRAINING_AGENT_SIZE=""
TRAINING_AGENT_MAX_SPEED=""
TRAINING_AGENT_ACCEL=""
TRAINING_USE_QUADROTOR_DYNAMICS=""
TRAINING_START_ALTITUDE_OFFSET=""
TRAINING_GOAL_ALTITUDE=""
TRAINING_AGENT_GOAL_FORMATION_RADIUS=""
TRAINING_ENABLE_ROLE_SHUFFLE=""
TRAINING_INIT_VEL_JITTER_MAX=""
TRAINING_MOUNTAIN_MARGIN=""
TRAINING_OBSTACLE_MIN_CLEARANCE_START_GOAL=""
TRAINING_TERRAIN_COLLISION_EPS=""
TRAINING_USE_LEGACY_TERRAIN=""
TRAINING_RING_BASE_REWARD=""
TRAINING_MIN_START_GOAL_DIST=""
TRAINING_MAX_START_GOAL_DIST=""
TRAINING_START_POS_MARGIN=""
TRAINING_START_POS_MAX_TRIALS=""
TRAINING_RUNTIME_MANIFEST_PATH=""
TRAINING_USE_FIXED_POSITIONS=""
TRAINING_POSITIONS_FILE=""
TRAINING_RANDOM_TERRAIN=""
TRAINING_TERRAIN_SEED=""
TRAINING_PER_EPISODE_TERRAIN=""
TRAINING_PER_ENV_TERRAIN=""
TRAINING_SEMI_RANDOM_TERRAIN=""
TRAINING_TERRAIN_BASE_SEED=""
TRAINING_TERRAIN_COMPLEXITY_LEVEL=""
TRAINING_MAP_SIZE=""
TRAINING_MOUNTAIN_MIN_DISTANCE=""
TRAINING_OBSTACLE_OBSERVATION_MODE=""
TRAINING_OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT=""
TRAINING_OBSTACLE_RISK_GOAL_ALONG_WEIGHT=""
EVAL_RESPECT_INPUT_POSITIONS=${EVAL_RESPECT_INPUT_POSITIONS:-0}

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# 创建结果目录
mkdir -p "$SAVE_PATH"

# 设置环境变量
export SUPPRESS_MA_PROMPT=1

echo "开始模型评估..."
echo "======================================"
echo "评估解释器: $EVAL_PYTHON_BIN"
echo "GPU选择: GPU_ID=${GPU_ID:-<unset>} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "运行环境: CONDA_PREFIX=${CONDA_PREFIX:-<unset>} | LD_LIBRARY_PATH=$([ -n \"${LD_LIBRARY_PATH:-}\" ] && echo set || echo unset)"

# 构造命令参数（关键参数优先由训练配置驱动，确保严格对齐）
# 场景/算法先设占位默认值，读取训练配置后再覆盖
SCENARIO_NAME=${SCENARIO_NAME:-paper3d_terrain_weighted}
export ALGORITHM=${ALGORITHM:-matd3}

# 🔧 新增：XLA加速配置（与训练脚本保持一致）
# 默认启用XLA Global加速，提升评估性能
export XLA_GLOBAL=${XLA_GLOBAL:-1}  # 默认启用XLA加速
export PF_JIT=${PF_JIT:-1}          # 评估默认保留训练已验证的 PF 小核 JIT
if [ "${XLA_GLOBAL}" = "1" ]; then
    echo "✅ XLA加速: 启用（Global JIT模式）"
    # 设置TF_XLA_FLAGS（禁用Auto JIT，使用手动控制）
    export TF_XLA_FLAGS=${TF_XLA_FLAGS:-"--tf_xla_auto_jit=0"}
else
    echo "❌ XLA加速: 禁用"
fi

# 设置默认地形模式
export RANDOM_TERRAIN=${RANDOM_TERRAIN:-1}  # 默认每回合随机新地形

# episode-length 先设默认，读取训练配置后再覆盖
EPISODE_LENGTH=${EPISODE_LENGTH:-2800}
# 🔧 使用双引号包裹路径变量，确保特殊字符（中文、空格等）正确处理
CMD_ARGS="--load-model-path \"$MODEL_PATH\" --eval-episodes $EVAL_EPISODES --save-viz-path \"$SAVE_PATH\" --scenario-name $SCENARIO_NAME --episode-length $EPISODE_LENGTH --algorithm $ALGORITHM"

# 地形模式在训练配置读取完成后再统一决定，避免默认值覆盖训练环境

# 🔧 修复：设置默认物理参数（与训练脚本完全一致）
export GRAVITY=${GRAVITY:-0.0}
export CONTROL_ACCEL_GAIN=${CONTROL_ACCEL_GAIN:-1.0}
export AGENT_SIZE=${AGENT_SIZE:-0.5}
export AGENT_MAX_SPEED=${AGENT_MAX_SPEED:-37.5}  # 🔧 修复：与训练脚本一致（32.5→37.5）
export AGENT_ACCEL=${AGENT_ACCEL:-3.6}  # 🔧 修复：与训练脚本一致（2.6→3.6）

# 通过环境变量传入物理与动作范围参数
CMD_ARGS="$CMD_ARGS --gravity $GRAVITY"
CMD_ARGS="$CMD_ARGS --control-accel-gain $CONTROL_ACCEL_GAIN"
CMD_ARGS="$CMD_ARGS --agent-size $AGENT_SIZE"
CMD_ARGS="$CMD_ARGS --agent-max-speed $AGENT_MAX_SPEED"
CMD_ARGS="$CMD_ARGS --agent-accel $AGENT_ACCEL"
# 🔧 关键修复：设置动作范围参数（优先级：训练配置 > 环境变量 > 默认值）
# 确保评估时使用的动作范围与训练时完全一致
if [ -n "$TRAINING_ACTION_RANGE_X" ]; then
    export ACTION_RANGE_X=$TRAINING_ACTION_RANGE_X
elif [ -z "${ACTION_RANGE_X}" ]; then
export ACTION_RANGE_X=${ACTION_RANGE_X:-2.5}  # 🔧 修复：与训练脚本一致（run_optimized.sh默认2.5）
fi

if [ -n "$TRAINING_ACTION_RANGE_Y" ]; then
    export ACTION_RANGE_Y=$TRAINING_ACTION_RANGE_Y
elif [ -z "${ACTION_RANGE_Y}" ]; then
export ACTION_RANGE_Y=${ACTION_RANGE_Y:-2.5}  # 🔧 修复：与训练脚本一致（run_optimized.sh默认2.5）
fi

if [ -n "$TRAINING_ACTION_RANGE_Z" ]; then
    export ACTION_RANGE_Z=$TRAINING_ACTION_RANGE_Z
elif [ -z "${ACTION_RANGE_Z}" ]; then
export ACTION_RANGE_Z=${ACTION_RANGE_Z:-2.2}  # 🔧 修复：与训练脚本一致（run_optimized.sh默认2.2）
fi

if [ -n "$TRAINING_DAMPING" ]; then
    export DAMPING=$TRAINING_DAMPING
elif [ -z "${DAMPING}" ]; then
export DAMPING=${DAMPING:-0.12}  # 🔧 修复：与训练脚本一致（0.15→0.18）
fi

if [ -n "$TRAINING_SIMULATION_DT" ]; then
    export SIMULATION_DT=$TRAINING_SIMULATION_DT
elif [ -z "${SIMULATION_DT}" ]; then
export SIMULATION_DT=${SIMULATION_DT:-0.08}
fi

if [ -n "$TRAINING_Z_ACTION_BIAS" ]; then
    export Z_ACTION_BIAS=$TRAINING_Z_ACTION_BIAS
elif [ -z "${Z_ACTION_BIAS}" ]; then
export Z_ACTION_BIAS=${Z_ACTION_BIAS:-0.0}
fi

if [ -n "$TRAINING_QUADROTOR_ATTITUDE_RESPONSE_TIME" ]; then
    export QUADROTOR_ATTITUDE_RESPONSE_TIME=$TRAINING_QUADROTOR_ATTITUDE_RESPONSE_TIME
elif [ -z "${QUADROTOR_ATTITUDE_RESPONSE_TIME}" ]; then
export QUADROTOR_ATTITUDE_RESPONSE_TIME=${QUADROTOR_ATTITUDE_RESPONSE_TIME:-0.0}
fi

if [ -n "$TRAINING_QUADROTOR_PSI_CMD" ]; then
    export QUADROTOR_PSI_CMD=$TRAINING_QUADROTOR_PSI_CMD
elif [ -z "${QUADROTOR_PSI_CMD}" ]; then
export QUADROTOR_PSI_CMD=${QUADROTOR_PSI_CMD:-0.0}
fi

if [ -n "$TRAINING_REWARD_POS_SCALE" ]; then
    export REWARD_POS_SCALE=$TRAINING_REWARD_POS_SCALE
elif [ -z "${REWARD_POS_SCALE}" ]; then
export REWARD_POS_SCALE=${REWARD_POS_SCALE:-1.5}  # 🔧 修复：与训练脚本一致（1.3→1.5）
fi

if [ -n "$TRAINING_REWARD_NEG_SCALE" ]; then
    export REWARD_NEG_SCALE=$TRAINING_REWARD_NEG_SCALE
elif [ -z "${REWARD_NEG_SCALE}" ]; then
export REWARD_NEG_SCALE=${REWARD_NEG_SCALE:-2.5}  # 🔧 修复：与训练脚本一致（1.1→2.5）
fi

CMD_ARGS="$CMD_ARGS --action-range-x $ACTION_RANGE_X"
CMD_ARGS="$CMD_ARGS --action-range-y $ACTION_RANGE_Y"
CMD_ARGS="$CMD_ARGS --action-range-z $ACTION_RANGE_Z"
CMD_ARGS="$CMD_ARGS --damping $DAMPING"
CMD_ARGS="$CMD_ARGS --simulation-dt $SIMULATION_DT"
CMD_ARGS="$CMD_ARGS --z-action-bias $Z_ACTION_BIAS"
CMD_ARGS="$CMD_ARGS --quadrotor-attitude-response-time $QUADROTOR_ATTITUDE_RESPONSE_TIME"
CMD_ARGS="$CMD_ARGS --quadrotor-psi-cmd $QUADROTOR_PSI_CMD"
CMD_ARGS="$CMD_ARGS --reward-pos-scale $REWARD_POS_SCALE"
CMD_ARGS="$CMD_ARGS --reward-neg-scale $REWARD_NEG_SCALE"

# 🔧 关键修复：从训练配置（results.json）中读取训练时使用的ACTION_FORCE_RATIO
# 优先使用训练时的FR值，确保评估与训练一致
TRAINING_FR=""
RESULTS_JSON_PATH=""

# 从模型路径推断并严格绑定 results.json（按 exp_name 精确匹配）
MODEL_PARENT_DIR=$(dirname "$MODEL_PATH")
EXPECTED_EXP_NAME=$(basename "$MODEL_PARENT_DIR")

# 先收集候选，再做精确匹配，避免 head -1 误选到其他实验
declare -a RESULTS_JSON_CANDIDATES
if [ -f "$MODEL_PARENT_DIR/results.json" ]; then
    RESULTS_JSON_CANDIDATES+=("$MODEL_PARENT_DIR/results.json")
fi
if [ -d "logs/$EXPECTED_EXP_NAME" ]; then
    while IFS= read -r p; do
        [ -n "$p" ] && RESULTS_JSON_CANDIDATES+=("$p")
    done < <(find "logs/$EXPECTED_EXP_NAME" -name "results.json" -type f 2>/dev/null)
fi
if [[ "$MODEL_PARENT_DIR" =~ ([0-9]{8}_[0-9]{6})$ ]]; then
    TIMESTAMP="${BASH_REMATCH[1]}"
    while IFS= read -r p; do
        [ -n "$p" ] && RESULTS_JSON_CANDIDATES+=("$p")
    done < <(find logs -type f -path "*/${TIMESTAMP}*/results.json" 2>/dev/null)
fi
while IFS= read -r p; do
    [ -n "$p" ] && RESULTS_JSON_CANDIDATES+=("$p")
done < <(find logs -type d -name "*${EXPECTED_EXP_NAME}*" 2>/dev/null -exec find {} -name "results.json" -type f \; 2>/dev/null)

# 候选去重
declare -a UNIQUE_RESULTS_JSON_CANDIDATES
for p in "${RESULTS_JSON_CANDIDATES[@]}"; do
    skip=0
    for q in "${UNIQUE_RESULTS_JSON_CANDIDATES[@]}"; do
        if [ "$p" = "$q" ]; then
            skip=1
            break
        fi
    done
    [ $skip -eq 0 ] && UNIQUE_RESULTS_JSON_CANDIDATES+=("$p")
done

# 优先选择 exp_name 精确匹配的 results.json
declare -a MATCHED_RESULTS_JSON
for p in "${UNIQUE_RESULTS_JSON_CANDIDATES[@]}"; do
    is_match=$("$EVAL_PYTHON_BIN" - <<PYTHON_EOF
import json
import sys
path = r"""$p"""
expected = r"""$EXPECTED_EXP_NAME"""
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    args = data.get("args", {}) if isinstance(data, dict) else {}
    exp_name = None
    if isinstance(args, dict):
        exp_name = args.get("exp_name")
    if exp_name is None and isinstance(data, dict):
        exp_name = data.get("exp_name")
    print("1" if (exp_name is not None and str(exp_name) == expected) else "0")
except Exception:
    print("0")
PYTHON_EOF
)
    if [ "$is_match" = "1" ]; then
        MATCHED_RESULTS_JSON+=("$p")
    fi
done

if [ ${#MATCHED_RESULTS_JSON[@]} -gt 0 ]; then
    RESULTS_JSON_PATH="${MATCHED_RESULTS_JSON[0]}"
    if [ ${#MATCHED_RESULTS_JSON[@]} -gt 1 ] && ([ "$STRICT_EVAL_MATCH" = "1" ] || [ "$STRICT_EVAL_MATCH" = "true" ] || [ "$STRICT_EVAL_MATCH" = "yes" ] || [ "$STRICT_EVAL_MATCH" = "on" ]); then
        MODEL_RESULTS_JSON="$MODEL_PARENT_DIR/results.json"
        if [ -f "$MODEL_RESULTS_JSON" ]; then
            for p in "${MATCHED_RESULTS_JSON[@]}"; do
                if [ "$p" = "$MODEL_RESULTS_JSON" ]; then
                    RESULTS_JSON_PATH="$MODEL_RESULTS_JSON"
                    break
                fi
            done
        fi
        if [ "$RESULTS_JSON_PATH" != "$MODEL_RESULTS_JSON" ]; then
            echo "❌ 严格模式：发现多个与 exp_name=$EXPECTED_EXP_NAME 精确匹配的 results.json，存在歧义："
            for p in "${MATCHED_RESULTS_JSON[@]}"; do
                echo "   - $p"
            done
            echo "   请保留唯一匹配文件后重试。"
            exit 1
        else
            echo "ℹ️  严格模式：发现多个精确匹配的 results.json，优先使用模型目录旁配置: $RESULTS_JSON_PATH"
        fi
    fi
elif [ ${#UNIQUE_RESULTS_JSON_CANDIDATES[@]} -gt 0 ] && ! ([ "$STRICT_EVAL_MATCH" = "1" ] || [ "$STRICT_EVAL_MATCH" = "true" ] || [ "$STRICT_EVAL_MATCH" = "yes" ] || [ "$STRICT_EVAL_MATCH" = "on" ]); then
    # 非严格模式下回退到第一个候选
    RESULTS_JSON_PATH="${UNIQUE_RESULTS_JSON_CANDIDATES[0]}"
fi

if [ "$STRICT_EVAL_MATCH" = "1" ] || [ "$STRICT_EVAL_MATCH" = "true" ] || [ "$STRICT_EVAL_MATCH" = "yes" ] || [ "$STRICT_EVAL_MATCH" = "on" ]; then
    if [ -z "$RESULTS_JSON_PATH" ] || [ ! -f "$RESULTS_JSON_PATH" ]; then
        echo "❌ 严格模式：未找到训练配置 results.json，拒绝评估。"
        echo "   请至少提供以下任一位置的训练配置："
        echo "   - $MODEL_PARENT_DIR/results.json"
        echo "   - logs/<exp_name>/<timestamp>/results.json"
        exit 1
    fi
fi

# 🔧 关键修复：从训练配置（results.json）中读取所有势场参数
# 优先使用训练时的参数值，确保评估与训练完全一致
if [ -n "$RESULTS_JSON_PATH" ] && [ -f "$RESULTS_JSON_PATH" ]; then
    # 使用Python读取results.json中的所有势场参数
    TRAINING_PARAMS=$("$EVAL_PYTHON_BIN" <<PYTHON_EOF
import json
import sys
try:
    with open("$RESULTS_JSON_PATH", 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 优先从args中读取参数
    args = None
    if 'args' in results and isinstance(results['args'], dict):
        args = results['args']
    elif isinstance(results, dict):
        args = results
    training_environment = results.get('training_environment', {}) if isinstance(results, dict) else {}
    if not isinstance(training_environment, dict):
        training_environment = {}

    # Older runs predate structured recording of scenario-level environment
    # variables.  Their resolved ablation manifest is the authoritative source;
    # recover it by exact exp_name/batch identity instead of substituting current
    # defaults.
    from pathlib import Path
    from experiment_runtime_config import (
        find_training_runtime_manifest,
        load_training_runtime_manifest,
        runtime_environment_from_manifest,
    )
    exp_name = str(args.get('exp_name') or Path("$MODEL_PARENT_DIR").name)
    runtime_manifest_path = find_training_runtime_manifest(
        Path("$SCRIPT_DIR"),
        exp_name,
        results.get('training_manifest_path'),
    )
    runtime_manifest_values = {}
    if runtime_manifest_path is not None:
        runtime_manifest = load_training_runtime_manifest(
            runtime_manifest_path,
            exp_name=exp_name,
            expected_content_sha256=results.get('training_manifest_sha256'),
        )
        runtime_manifest_values = runtime_environment_from_manifest(runtime_manifest)
    
    if args is not None:
        params = {}
        # 读取所有势场相关参数和动作范围参数
        param_names = [
            'scenario_name',
            'scenario',
            'algorithm',
            'algo',
            'episode_length',
            'action_force_ratio',
            'gravity',
            'control_accel_gain',
            'agent_size',
            'agent_max_speed',
            'agent_accel',
            'goal_attraction',
            'lambda_1_base',
            'terrain_repulsion',
            'agent_influence_range',
            'delta_k_att',
            'delta_lambda_1',
            'delta_k_rep',
            'delta_radius',
            'max_force_magnitude',
            'action_range_x',
            'action_range_y',
            'action_range_z',
            'damping',
            'simulation_dt',
            'z_action_bias',
            'use_quadrotor_dynamics',
            'start_altitude_offset',
            'goal_altitude',
            'agent_goal_formation_radius',
            'enable_role_shuffle',
            'init_vel_jitter_max',
            'mountain_margin',
            'obstacle_min_clearance_start_goal',
            'terrain_collision_eps',
            'use_legacy_terrain',
            'ring_base_reward',
            'min_start_goal_dist',
            'max_start_goal_dist',
            'start_pos_margin',
            'start_pos_max_trials',
            'quadrotor_attitude_response_time',
            'quadrotor_psi_cmd',
            'reward_pos_scale',
            'reward_neg_scale',
            'distance_weight',
            'exploration_weight',
            'stationary_weight',
            'direction_weight',
            'deviation_weight',
            'start_area_weight',
            'approach_weight',
            'energy_weight',
            'height_weight',
            'height_reward_enabled',
            'height_ideal_min',
            'height_ideal_max',
            'lateral_weight',
            'clearance_weight',
            'clearance_d_max',
            'success_weight',
            'collision_weight',
            'collision_reduction_weight',
            'global_weight',
            'shaping_weight',
            'max_reward',
            'min_reward',
            'success_reward_value',
            'no_collision_reward_value',
            'success_distance_threshold',
            'collision_penalty_value',
            'collision_distance_threshold',
            'global_reward_mode',
            'shaping_gamma',
            'terrain_contact_eps',
            'random_terrain',
            'terrain_seed',
            'per_episode_terrain',
            'per_env_terrain',
            'semi_random_terrain',
            'terrain_base_seed',
            'terrain_complexity_level',
            'map_size',
            'mountain_min_distance',
            'obstacle_observation_mode',
            'obstacle_risk_velocity_forward_weight',
            'obstacle_risk_goal_along_weight',
            'reward_version',
            'reward_terminal_order_fix',
            'goal_ring_individual_scale',
            'goal_ring_team_gated',
            'goal_ring_require_agent_safe',
            'progress_distance_state_scale',
            'progress_reward_scale',
            'team_progress_bottleneck_only',
            'team_progress_non_bottleneck_scale',
            'team_progress_bottleneck_eps',
            'team_success_bonus',
            'unsafe_arrival_penalty',
            'non_success_terminal_guard_enabled',
            'non_success_terminal_penalty_base',
            'non_success_terminal_penalty_per_meter',
            'non_success_terminal_penalty_max',
            'terminal_failure_penalty_base',
            'terminal_failure_penalty_per_meter',
            'terminal_failure_penalty_max',
            'clearance_quality_bonus_weight',
            'efficiency_bonus_weight',
            'team_sync_reward_enabled',
            'team_goal_occupancy_scale',
            'team_bottleneck_progress_scale',
            'team_waiting_scale',
            'team_bottleneck_delta_clip',
            'clearance_dense_positive_scale',
            'height_dense_positive_scale',
            'use_fixed_positions',
            'positions_file'
        ]
        for param_name in param_names:
            # schema>=2 training_environment is the canonical execution record;
            # args is retained as a legacy fallback for fields not recorded there.
            if param_name in training_environment and training_environment[param_name] is not None:
                params[param_name] = training_environment[param_name]
            elif param_name in args:
                params[param_name] = args[param_name]
            elif isinstance(results, dict) and param_name in results:
                params[param_name] = results[param_name]

        for param_name, runtime_value in runtime_manifest_values.items():
            if params.get(param_name) is None:
                params[param_name] = runtime_value
        params['_training_runtime_manifest_path'] = (
            str(runtime_manifest_path) if runtime_manifest_path is not None else ''
        )
        
        # 输出为JSON格式，方便bash解析
        import json
        print(json.dumps(params))
        sys.exit(0)
except Exception as e:
    print(f"训练配置解析失败: {e}", file=sys.stderr)
sys.exit(1)
PYTHON_EOF
)
    
if [ -n "$TRAINING_PARAMS" ]; then
        json_get_from_training_params() {
            local key="$1"
            echo "$TRAINING_PARAMS" | "$EVAL_PYTHON_BIN" -c "import sys, json; d=json.load(sys.stdin); v=d.get('$key', ''); print('' if v is None else v)" 2>/dev/null
        }

        # 解析JSON并设置环境变量
        TRAINING_FR=$(json_get_from_training_params action_force_ratio)
        TRAINING_GOAL_ATTRACTION=$(json_get_from_training_params goal_attraction)
        TRAINING_LAMBDA_1_BASE=$(json_get_from_training_params lambda_1_base)
        TRAINING_TERRAIN_REPULSION=$(json_get_from_training_params terrain_repulsion)
        TRAINING_AGENT_INFLUENCE_RANGE=$(json_get_from_training_params agent_influence_range)
        TRAINING_DELTA_K_ATT=$(json_get_from_training_params delta_k_att)
        TRAINING_DELTA_LAMBDA_1=$(json_get_from_training_params delta_lambda_1)
        TRAINING_DELTA_K_REP=$(json_get_from_training_params delta_k_rep)
        TRAINING_DELTA_RADIUS=$(json_get_from_training_params delta_radius)
        TRAINING_MAX_FORCE_MAGNITUDE=$(json_get_from_training_params max_force_magnitude)
        TRAINING_ACTION_RANGE_X=$(json_get_from_training_params action_range_x)
        TRAINING_ACTION_RANGE_Y=$(json_get_from_training_params action_range_y)
        TRAINING_ACTION_RANGE_Z=$(json_get_from_training_params action_range_z)
        TRAINING_GRAVITY=$(json_get_from_training_params gravity)
        TRAINING_CONTROL_ACCEL_GAIN=$(json_get_from_training_params control_accel_gain)
        TRAINING_AGENT_SIZE=$(json_get_from_training_params agent_size)
        TRAINING_AGENT_MAX_SPEED=$(json_get_from_training_params agent_max_speed)
        TRAINING_AGENT_ACCEL=$(json_get_from_training_params agent_accel)
        TRAINING_DAMPING=$(json_get_from_training_params damping)
        TRAINING_SIMULATION_DT=$(json_get_from_training_params simulation_dt)
        TRAINING_Z_ACTION_BIAS=$(json_get_from_training_params z_action_bias)
        TRAINING_USE_QUADROTOR_DYNAMICS=$(json_get_from_training_params use_quadrotor_dynamics)
        TRAINING_START_ALTITUDE_OFFSET=$(json_get_from_training_params start_altitude_offset)
        TRAINING_GOAL_ALTITUDE=$(json_get_from_training_params goal_altitude)
        TRAINING_AGENT_GOAL_FORMATION_RADIUS=$(json_get_from_training_params agent_goal_formation_radius)
        TRAINING_ENABLE_ROLE_SHUFFLE=$(json_get_from_training_params enable_role_shuffle)
        TRAINING_INIT_VEL_JITTER_MAX=$(json_get_from_training_params init_vel_jitter_max)
        TRAINING_MOUNTAIN_MARGIN=$(json_get_from_training_params mountain_margin)
        TRAINING_OBSTACLE_MIN_CLEARANCE_START_GOAL=$(json_get_from_training_params obstacle_min_clearance_start_goal)
        TRAINING_TERRAIN_COLLISION_EPS=$(json_get_from_training_params terrain_collision_eps)
        TRAINING_USE_LEGACY_TERRAIN=$(json_get_from_training_params use_legacy_terrain)
        TRAINING_RING_BASE_REWARD=$(json_get_from_training_params ring_base_reward)
        TRAINING_MIN_START_GOAL_DIST=$(json_get_from_training_params min_start_goal_dist)
        TRAINING_MAX_START_GOAL_DIST=$(json_get_from_training_params max_start_goal_dist)
        TRAINING_START_POS_MARGIN=$(json_get_from_training_params start_pos_margin)
        TRAINING_START_POS_MAX_TRIALS=$(json_get_from_training_params start_pos_max_trials)
        TRAINING_RUNTIME_MANIFEST_PATH=$(json_get_from_training_params _training_runtime_manifest_path)
        TRAINING_QUADROTOR_ATTITUDE_RESPONSE_TIME=$(json_get_from_training_params quadrotor_attitude_response_time)
        TRAINING_QUADROTOR_PSI_CMD=$(json_get_from_training_params quadrotor_psi_cmd)
        TRAINING_REWARD_POS_SCALE=$(json_get_from_training_params reward_pos_scale)
        TRAINING_REWARD_NEG_SCALE=$(json_get_from_training_params reward_neg_scale)
        TRAINING_SCENARIO_NAME=$(json_get_from_training_params scenario_name)
        TRAINING_ALGORITHM=$(json_get_from_training_params algorithm)
        if [ -z "$TRAINING_SCENARIO_NAME" ]; then
            TRAINING_SCENARIO_NAME=$(json_get_from_training_params scenario)
        fi
        if [ -z "$TRAINING_ALGORITHM" ]; then
            TRAINING_ALGORITHM=$(json_get_from_training_params algo)
        fi
        TRAINING_EPISODE_LENGTH=$(json_get_from_training_params episode_length)
        TRAINING_SUCCESS_DISTANCE_THRESHOLD=$(json_get_from_training_params success_distance_threshold)
        TRAINING_COLLISION_DISTANCE_THRESHOLD=$(json_get_from_training_params collision_distance_threshold)
        TRAINING_TERRAIN_CONTACT_EPS=$(json_get_from_training_params terrain_contact_eps)
        TRAINING_USE_FIXED_POSITIONS=$(json_get_from_training_params use_fixed_positions)
        TRAINING_POSITIONS_FILE=$(json_get_from_training_params positions_file)
        TRAINING_RANDOM_TERRAIN=$(json_get_from_training_params random_terrain)
        TRAINING_TERRAIN_SEED=$(json_get_from_training_params terrain_seed)
        TRAINING_PER_EPISODE_TERRAIN=$(json_get_from_training_params per_episode_terrain)
        TRAINING_PER_ENV_TERRAIN=$(json_get_from_training_params per_env_terrain)
        TRAINING_SEMI_RANDOM_TERRAIN=$(json_get_from_training_params semi_random_terrain)
        TRAINING_TERRAIN_BASE_SEED=$(json_get_from_training_params terrain_base_seed)
        TRAINING_TERRAIN_COMPLEXITY_LEVEL=$(json_get_from_training_params terrain_complexity_level)
        TRAINING_MAP_SIZE=$(json_get_from_training_params map_size)
        TRAINING_MOUNTAIN_MIN_DISTANCE=$(json_get_from_training_params mountain_min_distance)
        TRAINING_OBSTACLE_OBSERVATION_MODE=$(json_get_from_training_params obstacle_observation_mode)
        TRAINING_OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT=$(json_get_from_training_params obstacle_risk_velocity_forward_weight)
        TRAINING_OBSTACLE_RISK_GOAL_ALONG_WEIGHT=$(json_get_from_training_params obstacle_risk_goal_along_weight)

        TRAINING_REWARD_PARAM_KEYS=(
            distance_weight exploration_weight stationary_weight direction_weight deviation_weight
            start_area_weight approach_weight energy_weight height_weight height_reward_enabled
            height_ideal_min height_ideal_max lateral_weight clearance_weight clearance_d_max
            success_weight collision_weight collision_reduction_weight global_weight shaping_weight
            max_reward min_reward success_reward_value no_collision_reward_value
            collision_penalty_value global_reward_mode shaping_gamma
        )
        for reward_key in "${TRAINING_REWARD_PARAM_KEYS[@]}"; do
            reward_val=$(json_get_from_training_params "$reward_key")
            reward_var="TRAINING_$(printf '%s' "$reward_key" | tr '[:lower:]' '[:upper:]')"
            printf -v "$reward_var" '%s' "$reward_val"
            if [ -n "$reward_val" ]; then
                echo "✅ 从训练配置读取$(printf '%s' "$reward_key" | tr '[:lower:]' '[:upper:]'): $reward_val"
            fi
        done

        # 这些参数由场景/向量化奖励器直接从环境变量读取，不能只对齐
        # argparse 中的基础 reward weights。尤其安全奖励消融的这些值与
        # 其他模型不同，遗漏会导致评估奖励口径悄悄回到当前代码默认值。
        TRAINING_RUNTIME_ENV_MAPPINGS=(
            reward_version:REWARD_VERSION
            reward_terminal_order_fix:REWARD_TERMINAL_ORDER_FIX
            goal_ring_individual_scale:GOAL_RING_INDIVIDUAL_SCALE
            goal_ring_team_gated:GOAL_RING_TEAM_GATED
            goal_ring_require_agent_safe:GOAL_RING_REQUIRE_AGENT_SAFE
            progress_distance_state_scale:PROGRESS_DISTANCE_STATE_SCALE
            progress_reward_scale:PROGRESS_REWARD_SCALE
            team_progress_bottleneck_only:TEAM_PROGRESS_BOTTLENECK_ONLY
            team_progress_non_bottleneck_scale:TEAM_PROGRESS_NON_BOTTLENECK_SCALE
            team_progress_bottleneck_eps:TEAM_PROGRESS_BOTTLENECK_EPS
            team_success_bonus:TEAM_SUCCESS_BONUS
            unsafe_arrival_penalty:UNSAFE_ARRIVAL_PENALTY
            non_success_terminal_guard_enabled:NON_SUCCESS_TERMINAL_GUARD_ENABLED
            non_success_terminal_penalty_base:NON_SUCCESS_TERMINAL_PENALTY_BASE
            non_success_terminal_penalty_per_meter:NON_SUCCESS_TERMINAL_PENALTY_PER_METER
            non_success_terminal_penalty_max:NON_SUCCESS_TERMINAL_PENALTY_MAX
            terminal_failure_penalty_base:TERMINAL_FAILURE_PENALTY_BASE
            terminal_failure_penalty_per_meter:TERMINAL_FAILURE_PENALTY_PER_METER
            terminal_failure_penalty_max:TERMINAL_FAILURE_PENALTY_MAX
            clearance_quality_bonus_weight:CLEARANCE_QUALITY_BONUS_WEIGHT
            efficiency_bonus_weight:EFFICIENCY_BONUS_WEIGHT
            team_sync_reward_enabled:TEAM_SYNC_REWARD_ENABLED
            team_goal_occupancy_scale:TEAM_GOAL_OCCUPANCY_SCALE
            team_bottleneck_progress_scale:TEAM_BOTTLENECK_PROGRESS_SCALE
            team_waiting_scale:TEAM_WAITING_SCALE
            team_bottleneck_delta_clip:TEAM_BOTTLENECK_DELTA_CLIP
            clearance_dense_positive_scale:CLEARANCE_DENSE_POSITIVE_SCALE
            height_dense_positive_scale:HEIGHT_DENSE_POSITIVE_SCALE
        )
        for runtime_mapping in "${TRAINING_RUNTIME_ENV_MAPPINGS[@]}"; do
            IFS=':' read -r runtime_key runtime_env_name <<< "$runtime_mapping"
            runtime_value=$(json_get_from_training_params "$runtime_key")
            if [ -n "$runtime_value" ]; then
                export "$runtime_env_name=$runtime_value"
                echo "✅ 从训练配置读取${runtime_env_name}: $runtime_value"
            fi
        done
        
        if [ -n "$TRAINING_FR" ]; then
            echo "✅ 从训练配置读取ACTION_FORCE_RATIO: $TRAINING_FR (来源: $RESULTS_JSON_PATH)"
        fi
        if [ -n "$TRAINING_GOAL_ATTRACTION" ]; then
            echo "✅ 从训练配置读取GOAL_ATTRACTION: $TRAINING_GOAL_ATTRACTION"
        fi
        if [ -n "$TRAINING_LAMBDA_1_BASE" ]; then
            echo "✅ 从训练配置读取LAMBDA_1_BASE: $TRAINING_LAMBDA_1_BASE"
        fi
        if [ -n "$TRAINING_TERRAIN_REPULSION" ]; then
            echo "✅ 从训练配置读取TERRAIN_REPULSION: $TRAINING_TERRAIN_REPULSION"
        fi
        if [ -n "$TRAINING_AGENT_INFLUENCE_RANGE" ]; then
            echo "✅ 从训练配置读取AGENT_INFLUENCE_RANGE: $TRAINING_AGENT_INFLUENCE_RANGE"
        fi
        if [ -n "$TRAINING_DELTA_K_ATT" ]; then
            echo "✅ 从训练配置读取DELTA_K_ATT: $TRAINING_DELTA_K_ATT"
        fi
        if [ -n "$TRAINING_DELTA_LAMBDA_1" ]; then
            echo "✅ 从训练配置读取DELTA_LAMBDA_1: $TRAINING_DELTA_LAMBDA_1"
        fi
        if [ -n "$TRAINING_DELTA_K_REP" ]; then
            echo "✅ 从训练配置读取DELTA_K_REP: $TRAINING_DELTA_K_REP"
        fi
        if [ -n "$TRAINING_DELTA_RADIUS" ]; then
            echo "✅ 从训练配置读取DELTA_RADIUS: $TRAINING_DELTA_RADIUS"
        fi
        if [ -n "$TRAINING_MAX_FORCE_MAGNITUDE" ]; then
            echo "✅ 从训练配置读取MAX_FORCE_MAGNITUDE: $TRAINING_MAX_FORCE_MAGNITUDE"
        fi
        if [ -n "$TRAINING_ACTION_RANGE_X" ]; then
            echo "✅ 从训练配置读取ACTION_RANGE_X: $TRAINING_ACTION_RANGE_X"
        fi
        if [ -n "$TRAINING_ACTION_RANGE_Y" ]; then
            echo "✅ 从训练配置读取ACTION_RANGE_Y: $TRAINING_ACTION_RANGE_Y"
        fi
        if [ -n "$TRAINING_ACTION_RANGE_Z" ]; then
            echo "✅ 从训练配置读取ACTION_RANGE_Z: $TRAINING_ACTION_RANGE_Z"
        fi
        if [ -n "$TRAINING_GRAVITY" ]; then
            echo "✅ 从训练配置读取GRAVITY: $TRAINING_GRAVITY"
        fi
        if [ -n "$TRAINING_CONTROL_ACCEL_GAIN" ]; then
            echo "✅ 从训练配置读取CONTROL_ACCEL_GAIN: $TRAINING_CONTROL_ACCEL_GAIN"
        fi
        if [ -n "$TRAINING_AGENT_MAX_SPEED" ]; then
            echo "✅ 从训练配置读取AGENT_MAX_SPEED: $TRAINING_AGENT_MAX_SPEED"
        fi
        if [ -n "$TRAINING_AGENT_ACCEL" ]; then
            echo "✅ 从训练配置读取AGENT_ACCEL: $TRAINING_AGENT_ACCEL"
        fi
        if [ -n "$TRAINING_SIMULATION_DT" ]; then
            echo "✅ 从训练配置读取SIMULATION_DT: $TRAINING_SIMULATION_DT"
        fi
        if [ -n "$TRAINING_Z_ACTION_BIAS" ]; then
            echo "✅ 从训练配置读取Z_ACTION_BIAS: $TRAINING_Z_ACTION_BIAS"
        fi
        if [ -n "$TRAINING_USE_QUADROTOR_DYNAMICS" ]; then
            echo "✅ 从训练配置读取USE_QUADROTOR_DYNAMICS: $TRAINING_USE_QUADROTOR_DYNAMICS"
        fi
        if [ -n "$TRAINING_QUADROTOR_ATTITUDE_RESPONSE_TIME" ]; then
            echo "✅ 从训练配置读取QUADROTOR_ATTITUDE_RESPONSE_TIME: $TRAINING_QUADROTOR_ATTITUDE_RESPONSE_TIME"
        fi
        if [ -n "$TRAINING_QUADROTOR_PSI_CMD" ]; then
            echo "✅ 从训练配置读取QUADROTOR_PSI_CMD: $TRAINING_QUADROTOR_PSI_CMD"
        fi
        if [ -n "$TRAINING_SCENARIO_NAME" ]; then
            echo "✅ 从训练配置读取SCENARIO_NAME: $TRAINING_SCENARIO_NAME"
        fi
        if [ -n "$TRAINING_ALGORITHM" ]; then
            echo "✅ 从训练配置读取ALGORITHM: $TRAINING_ALGORITHM"
        fi
        if [ -n "$TRAINING_EPISODE_LENGTH" ]; then
            echo "✅ 从训练配置读取EPISODE_LENGTH: $TRAINING_EPISODE_LENGTH"
        fi
        if [ -n "$TRAINING_SUCCESS_DISTANCE_THRESHOLD" ]; then
            echo "✅ 从训练配置读取SUCCESS_DISTANCE_THRESHOLD: $TRAINING_SUCCESS_DISTANCE_THRESHOLD"
        fi
        if [ -n "$TRAINING_COLLISION_DISTANCE_THRESHOLD" ]; then
            echo "✅ 从训练配置读取COLLISION_DISTANCE_THRESHOLD: $TRAINING_COLLISION_DISTANCE_THRESHOLD"
        fi
        if [ -n "$TRAINING_TERRAIN_CONTACT_EPS" ]; then
            echo "✅ 从训练配置读取TERRAIN_CONTACT_EPS: $TRAINING_TERRAIN_CONTACT_EPS"
        fi
        if [ -n "$TRAINING_USE_FIXED_POSITIONS" ]; then
            echo "✅ 从训练配置读取USE_FIXED_POSITIONS: $TRAINING_USE_FIXED_POSITIONS"
        fi
        if [ -n "$TRAINING_POSITIONS_FILE" ]; then
            echo "✅ 从训练配置读取POSITIONS_FILE: $TRAINING_POSITIONS_FILE"
        fi
        if [ -n "$TRAINING_RANDOM_TERRAIN" ]; then
            echo "✅ 从训练配置读取RANDOM_TERRAIN: $TRAINING_RANDOM_TERRAIN"
        fi
        if [ -n "$TRAINING_TERRAIN_SEED" ]; then
            echo "✅ 从训练配置读取TERRAIN_SEED: $TRAINING_TERRAIN_SEED"
        fi
        if [ -n "$TRAINING_PER_EPISODE_TERRAIN" ]; then
            echo "✅ 从训练配置读取PER_EPISODE_TERRAIN: $TRAINING_PER_EPISODE_TERRAIN"
        fi
        if [ -n "$TRAINING_PER_ENV_TERRAIN" ]; then
            echo "✅ 从训练配置读取PER_ENV_TERRAIN: $TRAINING_PER_ENV_TERRAIN"
        fi
        if [ -n "$TRAINING_SEMI_RANDOM_TERRAIN" ]; then
            echo "✅ 从训练配置读取SEMI_RANDOM_TERRAIN: $TRAINING_SEMI_RANDOM_TERRAIN"
        fi
        if [ -n "$TRAINING_TERRAIN_BASE_SEED" ]; then
            echo "✅ 从训练配置读取TERRAIN_BASE_SEED: $TRAINING_TERRAIN_BASE_SEED"
        fi
        if [ -n "$TRAINING_TERRAIN_COMPLEXITY_LEVEL" ]; then
            echo "✅ 从训练配置读取TERRAIN_COMPLEXITY_LEVEL: $TRAINING_TERRAIN_COMPLEXITY_LEVEL"
        fi
        if [ -n "$TRAINING_MAP_SIZE" ]; then
            echo "✅ 从训练配置读取MAP_SIZE: $TRAINING_MAP_SIZE"
        fi
        if [ -n "$TRAINING_MOUNTAIN_MIN_DISTANCE" ]; then
            echo "✅ 从训练配置读取MOUNTAIN_MIN_DISTANCE: $TRAINING_MOUNTAIN_MIN_DISTANCE"
        fi
        if [ -n "$TRAINING_OBSTACLE_OBSERVATION_MODE" ]; then
            echo "✅ 从训练配置读取OBSTACLE_OBSERVATION_MODE: $TRAINING_OBSTACLE_OBSERVATION_MODE"
        fi
        if [ -n "$TRAINING_OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT" ]; then
            echo "✅ 从训练配置读取OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT: $TRAINING_OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT"
        fi
        if [ -n "$TRAINING_OBSTACLE_RISK_GOAL_ALONG_WEIGHT" ]; then
            echo "✅ 从训练配置读取OBSTACLE_RISK_GOAL_ALONG_WEIGHT: $TRAINING_OBSTACLE_RISK_GOAL_ALONG_WEIGHT"
        fi
    fi
fi

apply_training_or_default() {
    local target_var="$1"
    local training_var="$2"
    local default_value="$3"
    local training_value="${!training_var}"
    local current_value="${!target_var}"

    if [ -n "$training_value" ]; then
        printf -v "$target_var" '%s' "$training_value"
    elif [ -z "$current_value" ]; then
        printf -v "$target_var" '%s' "$default_value"
    fi
    export "$target_var"
}

normalize_bool_env_value() {
    local raw="${1:-}"
    raw=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
    case "$raw" in
        1|true|yes|on) printf '1' ;;
        0|false|no|off) printf '0' ;;
        *) printf '%s' "$1" ;;
    esac
}

# ===== 严格对齐训练关键参数（场景/算法/步长/成功与碰撞阈值）=====
if [ -n "$TRAINING_SCENARIO_NAME" ]; then
    SCENARIO_NAME="$TRAINING_SCENARIO_NAME"
fi
if [ -n "$TRAINING_ALGORITHM" ]; then
    export ALGORITHM="$TRAINING_ALGORITHM"
fi
if [ -n "$TRAINING_EPISODE_LENGTH" ]; then
    EPISODE_LENGTH="$TRAINING_EPISODE_LENGTH"
fi
if [ -n "$EVAL_EPISODE_LENGTH_MULTIPLIER" ]; then
    MULTIPLIER_VALID=$(awk -v v="$EVAL_EPISODE_LENGTH_MULTIPLIER" 'BEGIN {print ((v+0)>0) ? 1 : 0}')
    if [ "$MULTIPLIER_VALID" = "1" ]; then
        MULTIPLIER_IS_DEFAULT=$(awk -v v="$EVAL_EPISODE_LENGTH_MULTIPLIER" 'BEGIN {print (v >= 0.999999 && v <= 1.000001) ? 1 : 0}')
        if [ "$MULTIPLIER_IS_DEFAULT" != "1" ]; then
            BASE_EPISODE_LENGTH="$EPISODE_LENGTH"
            EPISODE_LENGTH=$(awk -v base="$BASE_EPISODE_LENGTH" -v mult="$EVAL_EPISODE_LENGTH_MULTIPLIER" 'BEGIN {v=int(base*mult + 0.5); if (v < 1) v = 1; print v}')
            echo "✅ 测试步长倍率生效: ${BASE_EPISODE_LENGTH} x ${EVAL_EPISODE_LENGTH_MULTIPLIER} -> ${EPISODE_LENGTH}"
        fi
    else
        echo "⚠️  无效的 EVAL_EPISODE_LENGTH_MULTIPLIER=$EVAL_EPISODE_LENGTH_MULTIPLIER，忽略步长倍率设置"
    fi
fi
if [ -n "$TRAINING_SUCCESS_DISTANCE_THRESHOLD" ]; then
    export SUCCESS_DISTANCE_THRESHOLD="$TRAINING_SUCCESS_DISTANCE_THRESHOLD"
fi
if [ -n "$TRAINING_COLLISION_DISTANCE_THRESHOLD" ]; then
    export COLLISION_DISTANCE_THRESHOLD="$TRAINING_COLLISION_DISTANCE_THRESHOLD"
fi
if [ -n "$TRAINING_TERRAIN_CONTACT_EPS" ]; then
    export TERRAIN_CONTACT_EPS="$TRAINING_TERRAIN_CONTACT_EPS"
fi

# 对训练配置中常见缺失字段做可追溯回退
# 说明：
# - terrain_contact_eps 缺失时，回退到训练脚本默认 0.2
# - 场景/算法在严格模式下必须从训练配置读取，禁止回退（防止口径漂移）
if [ -z "$TRAINING_TERRAIN_CONTACT_EPS" ]; then
    export TERRAIN_CONTACT_EPS=${TERRAIN_CONTACT_EPS:-0.2}
    echo "ℹ️  训练配置缺少 terrain_contact_eps（旧结果常见），使用统一训练默认: $TERRAIN_CONTACT_EPS"
fi

# ===== 严格对齐训练地形生成参数（随机模式/seed/复杂度/地图尺寸）=====
if [ -n "$TRAINING_RANDOM_TERRAIN" ]; then
    export RANDOM_TERRAIN=$(normalize_bool_env_value "$TRAINING_RANDOM_TERRAIN")
fi
if [ -n "$TRAINING_PER_EPISODE_TERRAIN" ]; then
    export PER_EPISODE_TERRAIN=$(normalize_bool_env_value "$TRAINING_PER_EPISODE_TERRAIN")
fi
if [ -n "$TRAINING_PER_ENV_TERRAIN" ]; then
    export PER_ENV_TERRAIN=$(normalize_bool_env_value "$TRAINING_PER_ENV_TERRAIN")
fi
if [ -n "$TRAINING_SEMI_RANDOM_TERRAIN" ]; then
    export SEMI_RANDOM_TERRAIN=$(normalize_bool_env_value "$TRAINING_SEMI_RANDOM_TERRAIN")
fi
if [ -n "$TRAINING_TERRAIN_SEED" ]; then
    export USE_SCENARIO_SEED=1
    export SCENARIO_SEED="$TRAINING_TERRAIN_SEED"
fi
if [ -n "$TRAINING_TERRAIN_BASE_SEED" ]; then
    export TERRAIN_BASE_SEED="$TRAINING_TERRAIN_BASE_SEED"
elif [ -n "$TRAINING_TERRAIN_SEED" ]; then
    export TERRAIN_BASE_SEED="$TRAINING_TERRAIN_SEED"
fi
if [ -n "$TRAINING_TERRAIN_COMPLEXITY_LEVEL" ]; then
    export TERRAIN_COMPLEXITY_LEVEL="$TRAINING_TERRAIN_COMPLEXITY_LEVEL"
fi
if [ -n "$TRAINING_MAP_SIZE" ]; then
    export MAP_SIZE="$TRAINING_MAP_SIZE"
fi
if [ -n "$TRAINING_MOUNTAIN_MIN_DISTANCE" ]; then
    export MOUNTAIN_MIN_DISTANCE="$TRAINING_MOUNTAIN_MIN_DISTANCE"
fi
apply_training_or_default OBSTACLE_OBSERVATION_MODE TRAINING_OBSTACLE_OBSERVATION_MODE nearest_surface
apply_training_or_default OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT TRAINING_OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT 4.0
apply_training_or_default OBSTACLE_RISK_GOAL_ALONG_WEIGHT TRAINING_OBSTACLE_RISK_GOAL_ALONG_WEIGHT 3.0
export OBSTACLE_OBS_MODE="$OBSTACLE_OBSERVATION_MODE"

CMD_ARGS=$(echo "$CMD_ARGS" | sed -E 's/ --random-terrain//g')
if [ "${RANDOM_TERRAIN,,}" = "1" ] || [ "${RANDOM_TERRAIN,,}" = "true" ] || [ "${RANDOM_TERRAIN,,}" = "yes" ] || [ "${RANDOM_TERRAIN,,}" = "on" ]; then
    CMD_ARGS="$CMD_ARGS --random-terrain"
    echo "🏔️ 使用随机地形模式"
else
    echo "🏔️ 使用固定地形模式（与训练环境一致）"
fi
if [ -n "$TERRAIN_COMPLEXITY_LEVEL" ]; then
    echo "🏔️ 地形复杂度等级: $TERRAIN_COMPLEXITY_LEVEL"
fi
if [ -n "$SCENARIO_SEED" ]; then
    echo "🏔️ 地形种子: $SCENARIO_SEED"
fi
if [ -n "$MAP_SIZE" ]; then
    echo "🗺️ 地图尺寸: $MAP_SIZE"
fi
if [ -n "$MOUNTAIN_MIN_DISTANCE" ]; then
    echo "⛰️ 山峰最小间距: $MOUNTAIN_MIN_DISTANCE"
fi
echo "🧭 障碍观测模式: $OBSTACLE_OBSERVATION_MODE (vel_w=$OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT, goal_w=$OBSTACLE_RISK_GOAL_ALONG_WEIGHT)"

# 关键物理/控制参数最初在脚本前半段就写进了 CMD_ARGS；
# 这里在训练配置读取完成后再次统一回写，确保最终命令真正使用训练时的值。
apply_training_or_default GRAVITY TRAINING_GRAVITY 0.0
apply_training_or_default CONTROL_ACCEL_GAIN TRAINING_CONTROL_ACCEL_GAIN 1.0
apply_training_or_default AGENT_SIZE TRAINING_AGENT_SIZE 0.5
apply_training_or_default AGENT_MAX_SPEED TRAINING_AGENT_MAX_SPEED 37.5
apply_training_or_default AGENT_ACCEL TRAINING_AGENT_ACCEL 3.6
apply_training_or_default ACTION_RANGE_X TRAINING_ACTION_RANGE_X 2.5
apply_training_or_default ACTION_RANGE_Y TRAINING_ACTION_RANGE_Y 2.5
apply_training_or_default ACTION_RANGE_Z TRAINING_ACTION_RANGE_Z 2.2
apply_training_or_default DAMPING TRAINING_DAMPING 0.12
apply_training_or_default SIMULATION_DT TRAINING_SIMULATION_DT 0.08
apply_training_or_default Z_ACTION_BIAS TRAINING_Z_ACTION_BIAS 0.0
apply_training_or_default USE_QUADROTOR_DYNAMICS TRAINING_USE_QUADROTOR_DYNAMICS 0
apply_training_or_default START_ALTITUDE_OFFSET TRAINING_START_ALTITUDE_OFFSET 7.0
apply_training_or_default GOAL_ALTITUDE TRAINING_GOAL_ALTITUDE 12.0
apply_training_or_default AGENT_GOAL_FORMATION_RADIUS TRAINING_AGENT_GOAL_FORMATION_RADIUS 10.0
apply_training_or_default ENABLE_ROLE_SHUFFLE TRAINING_ENABLE_ROLE_SHUFFLE 1
apply_training_or_default INIT_VEL_JITTER_MAX TRAINING_INIT_VEL_JITTER_MAX 0.3
apply_training_or_default MOUNTAIN_MIN_DISTANCE TRAINING_MOUNTAIN_MIN_DISTANCE 55
apply_training_or_default MOUNTAIN_MARGIN TRAINING_MOUNTAIN_MARGIN 20
apply_training_or_default OBSTACLE_MIN_CLEARANCE_START_GOAL TRAINING_OBSTACLE_MIN_CLEARANCE_START_GOAL 25.0
apply_training_or_default TERRAIN_COLLISION_EPS TRAINING_TERRAIN_COLLISION_EPS 0.3
apply_training_or_default USE_LEGACY_TERRAIN TRAINING_USE_LEGACY_TERRAIN 0
apply_training_or_default RING_BASE_REWARD TRAINING_RING_BASE_REWARD 80.0
apply_training_or_default MIN_START_GOAL_DIST TRAINING_MIN_START_GOAL_DIST 40.0
apply_training_or_default MAX_START_GOAL_DIST TRAINING_MAX_START_GOAL_DIST 100.0
apply_training_or_default START_POS_MARGIN TRAINING_START_POS_MARGIN 5.0
apply_training_or_default START_POS_MAX_TRIALS TRAINING_START_POS_MAX_TRIALS 2000
apply_training_or_default QUADROTOR_ATTITUDE_RESPONSE_TIME TRAINING_QUADROTOR_ATTITUDE_RESPONSE_TIME 0.0
apply_training_or_default QUADROTOR_PSI_CMD TRAINING_QUADROTOR_PSI_CMD 0.0
apply_training_or_default REWARD_POS_SCALE TRAINING_REWARD_POS_SCALE 1.0
apply_training_or_default REWARD_NEG_SCALE TRAINING_REWARD_NEG_SCALE 1.0
if [ -n "$TRAINING_RUNTIME_MANIFEST_PATH" ]; then
    export TRAINING_RUNTIME_MANIFEST_PATH
fi

# 固定位置策略：在严格模式下优先对齐训练时的位置文件
if [ "$STRICT_EVAL_MATCH" = "1" ] || [ "$STRICT_EVAL_MATCH" = "true" ] || [ "$STRICT_EVAL_MATCH" = "yes" ] || [ "$STRICT_EVAL_MATCH" = "on" ]; then
    if [ "${TRAINING_USE_FIXED_POSITIONS,,}" = "true" ] || [ "$TRAINING_USE_FIXED_POSITIONS" = "1" ]; then
        USE_FIXED_POSITIONS=true
        if [ "$EVAL_RESPECT_INPUT_POSITIONS" = "1" ] || [ "${EVAL_RESPECT_INPUT_POSITIONS,,}" = "true" ] || [ "${EVAL_RESPECT_INPUT_POSITIONS,,}" = "yes" ] || [ "${EVAL_RESPECT_INPUT_POSITIONS,,}" = "on" ]; then
            echo "ℹ️  严格模式下保留调用方传入的位置文件: $POSITIONS_FILE"
        elif [ -n "$TRAINING_POSITIONS_FILE" ]; then
            POSITIONS_FILE="$TRAINING_POSITIONS_FILE"
        fi
    fi
fi

if [ "$STRICT_EVAL_MATCH" = "1" ] || [ "$STRICT_EVAL_MATCH" = "true" ] || [ "$STRICT_EVAL_MATCH" = "yes" ] || [ "$STRICT_EVAL_MATCH" = "on" ]; then
    missing_keys=()
    [ -z "$TRAINING_SCENARIO_NAME" ] && missing_keys+=("scenario/scenario_name")
    [ -z "$TRAINING_ALGORITHM" ] && missing_keys+=("algo/algorithm")
    [ -z "$TRAINING_EPISODE_LENGTH" ] && missing_keys+=("episode_length")
    [ -z "$TRAINING_SUCCESS_DISTANCE_THRESHOLD" ] && missing_keys+=("success_distance_threshold")
    [ -z "$TRAINING_COLLISION_DISTANCE_THRESHOLD" ] && missing_keys+=("collision_distance_threshold")
    [ -z "$TRAINING_AGENT_SIZE" ] && missing_keys+=("agent_size")
    [ -z "$TRAINING_AGENT_MAX_SPEED" ] && missing_keys+=("agent_max_speed")
    [ -z "$TRAINING_AGENT_ACCEL" ] && missing_keys+=("agent_accel")
    [ -z "$TRAINING_USE_QUADROTOR_DYNAMICS" ] && missing_keys+=("use_quadrotor_dynamics")
    [ -z "$TRAINING_START_ALTITUDE_OFFSET" ] && missing_keys+=("start_altitude_offset")
    [ -z "$TRAINING_GOAL_ALTITUDE" ] && missing_keys+=("goal_altitude")
    [ -z "$TRAINING_AGENT_GOAL_FORMATION_RADIUS" ] && missing_keys+=("agent_goal_formation_radius")
    [ -z "$TRAINING_ENABLE_ROLE_SHUFFLE" ] && missing_keys+=("enable_role_shuffle")
    [ -z "$TRAINING_INIT_VEL_JITTER_MAX" ] && missing_keys+=("init_vel_jitter_max")
    [ -z "$TRAINING_MOUNTAIN_MIN_DISTANCE" ] && missing_keys+=("mountain_min_distance")
    [ -z "$TRAINING_MOUNTAIN_MARGIN" ] && missing_keys+=("mountain_margin")
    [ -z "$TRAINING_OBSTACLE_MIN_CLEARANCE_START_GOAL" ] && missing_keys+=("obstacle_min_clearance_start_goal")
    [ -z "$TRAINING_TERRAIN_COLLISION_EPS" ] && missing_keys+=("terrain_collision_eps")
    [ -z "$TRAINING_USE_LEGACY_TERRAIN" ] && missing_keys+=("use_legacy_terrain")
    [ -z "$TRAINING_RING_BASE_REWARD" ] && missing_keys+=("ring_base_reward")
    [ -z "$TRAINING_MIN_START_GOAL_DIST" ] && missing_keys+=("min_start_goal_dist")
    [ -z "$TRAINING_MAX_START_GOAL_DIST" ] && missing_keys+=("max_start_goal_dist")
    [ -z "$TRAINING_START_POS_MARGIN" ] && missing_keys+=("start_pos_margin")
    [ -z "$TRAINING_START_POS_MAX_TRIALS" ] && missing_keys+=("start_pos_max_trials")
    [ -z "$TRAINING_GOAL_ATTRACTION" ] && missing_keys+=("goal_attraction")
    [ -z "$TRAINING_LAMBDA_1_BASE" ] && missing_keys+=("lambda_1_base")
    [ -z "$TRAINING_TERRAIN_REPULSION" ] && missing_keys+=("terrain_repulsion")
    [ -z "$TRAINING_AGENT_INFLUENCE_RANGE" ] && missing_keys+=("agent_influence_range")
    [ -z "$TRAINING_DELTA_K_ATT" ] && missing_keys+=("delta_k_att")
    [ -z "$TRAINING_DELTA_LAMBDA_1" ] && missing_keys+=("delta_lambda_1")
    [ -z "$TRAINING_DELTA_K_REP" ] && missing_keys+=("delta_k_rep")
    [ -z "$TRAINING_DELTA_RADIUS" ] && missing_keys+=("delta_radius")
    [ -z "$TRAINING_MAX_FORCE_MAGNITUDE" ] && missing_keys+=("max_force_magnitude")
    if [ ${#missing_keys[@]} -gt 0 ]; then
        echo "❌ 严格模式：训练配置缺少关键字段: ${missing_keys[*]}"
        echo "   为避免评估与训练口径不一致，已中止。"
        exit 1
    fi
fi

echo "使用评估场景: $SCENARIO_NAME"
echo "使用算法: $ALGORITHM"
echo "评估步长(episode_length): $EPISODE_LENGTH"
echo "评估步长倍率: $EVAL_EPISODE_LENGTH_MULTIPLIER"

# 更新已构造的基础参数，避免前面默认值遗留到最终命令
CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--scenario-name [^ ]+/--scenario-name ${SCENARIO_NAME}/")
CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--episode-length [^ ]+/--episode-length ${EPISODE_LENGTH}/")
CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--algorithm [^ ]+/--algorithm ${ALGORITHM}/")
CMD_ARGS="$CMD_ARGS --gravity $GRAVITY"
CMD_ARGS="$CMD_ARGS --control-accel-gain $CONTROL_ACCEL_GAIN"
CMD_ARGS="$CMD_ARGS --agent-max-speed $AGENT_MAX_SPEED"
CMD_ARGS="$CMD_ARGS --agent-accel $AGENT_ACCEL"
CMD_ARGS="$CMD_ARGS --action-range-x $ACTION_RANGE_X"
CMD_ARGS="$CMD_ARGS --action-range-y $ACTION_RANGE_Y"
CMD_ARGS="$CMD_ARGS --action-range-z $ACTION_RANGE_Z"
CMD_ARGS="$CMD_ARGS --damping $DAMPING"
CMD_ARGS="$CMD_ARGS --simulation-dt $SIMULATION_DT"
CMD_ARGS="$CMD_ARGS --z-action-bias $Z_ACTION_BIAS"
CMD_ARGS="$CMD_ARGS --use-quadrotor-dynamics $USE_QUADROTOR_DYNAMICS"
CMD_ARGS="$CMD_ARGS --quadrotor-attitude-response-time $QUADROTOR_ATTITUDE_RESPONSE_TIME"
CMD_ARGS="$CMD_ARGS --quadrotor-psi-cmd $QUADROTOR_PSI_CMD"
CMD_ARGS="$CMD_ARGS --reward-pos-scale $REWARD_POS_SCALE"
CMD_ARGS="$CMD_ARGS --reward-neg-scale $REWARD_NEG_SCALE"

# 🔧 关键修复：设置势场参数（优先级：训练配置 > 环境变量 > 默认值）
# 确保评估时使用的势场参数与训练时完全一致

# ACTION_FORCE_RATIO
if [ -n "${FORCE_EVAL_ACTION_FORCE_RATIO}" ]; then
    export ACTION_FORCE_RATIO=$FORCE_EVAL_ACTION_FORCE_RATIO
    echo "⚠️  控制实验：强制使用ACTION_FORCE_RATIO: $ACTION_FORCE_RATIO"
elif [ -n "$TRAINING_FR" ]; then
    export ACTION_FORCE_RATIO=$TRAINING_FR
    echo "✅ 使用训练时的ACTION_FORCE_RATIO: $ACTION_FORCE_RATIO"
elif [ -n "${ACTION_FORCE_RATIO}" ]; then
    echo "ℹ️  使用环境变量中的ACTION_FORCE_RATIO: $ACTION_FORCE_RATIO"
else
    export ACTION_FORCE_RATIO=${ACTION_FORCE_RATIO:-0.15}  # 🔧 修复：与训练脚本一致（0.75→0.50）
    echo "ℹ️  使用默认ACTION_FORCE_RATIO: $ACTION_FORCE_RATIO（未找到训练配置）"
fi

export USE_TF_POTENTIAL_FIELD=${USE_TF_POTENTIAL_FIELD:-1}
export USE_FR_FEATURE=${USE_FR_FEATURE:-1}  # 启用FR特征（与训练保持一致）
export USE_PF_FEATURE=${USE_PF_FEATURE:-1}  # 启用势场特征（与训练保持一致）
export TERRAIN_SENSING_MODE=${TERRAIN_SENSING_MODE:-local}  # 🔧 新增：地形感知模式 (local/oracle_same_probes/oracle_dense)
                                                                      # local: 使用观测中的地形信息（默认）
                                                                      # oracle_same_probes: 使用Oracle接口获取真值，probe布局与local一致
                                                                      # oracle_dense: 使用Oracle接口，提升探测密度

# 注意：FORCE_PARAM_*_RANGE 参数已废弃，改用delta+base模式
export FORCE_PARAM_GOAL_ATTRACTION_RANGE=${FORCE_PARAM_GOAL_ATTRACTION_RANGE:-"0.5 6.0"}
export FORCE_PARAM_LAMBDA_1_RANGE=${FORCE_PARAM_LAMBDA_1_RANGE:-"5.0 12.0"}
export FORCE_PARAM_TERRAIN_REPULSION_RANGE=${FORCE_PARAM_TERRAIN_REPULSION_RANGE:-"5 25.0"}
export FORCE_PARAM_DETECTION_RADIUS_RANGE=${FORCE_PARAM_DETECTION_RADIUS_RANGE:-"10.0 50.0"}

# 🔧 关键修复：势场基准参数（优先级：训练配置 > 环境变量 > 默认值）
# 🚨 问题：评估脚本的默认值与训练脚本不一致，导致评估时势场行为完全不同
# 🚨 修复：改为与训练脚本完全一致的默认值，并优先从训练配置读取
if [ -n "$TRAINING_GOAL_ATTRACTION" ]; then
    export GOAL_ATTRACTION=$TRAINING_GOAL_ATTRACTION
elif [ -z "${GOAL_ATTRACTION}" ]; then
    export GOAL_ATTRACTION=${GOAL_ATTRACTION:-26.0}  # 与run_optimized.sh当前训练默认一致
fi

if [ -n "$TRAINING_LAMBDA_1_BASE" ]; then
    export LAMBDA_1_BASE=$TRAINING_LAMBDA_1_BASE
elif [ -z "${LAMBDA_1_BASE}" ]; then
    export LAMBDA_1_BASE=${LAMBDA_1_BASE:-8.5}  # ✅ 与训练脚本一致
fi

if [ -n "$TRAINING_TERRAIN_REPULSION" ]; then
    export TERRAIN_REPULSION=$TRAINING_TERRAIN_REPULSION
elif [ -z "${TERRAIN_REPULSION}" ]; then
    export TERRAIN_REPULSION=${TERRAIN_REPULSION:-1600.0}  # 与run_optimized.sh当前训练默认一致
fi

if [ -n "$TRAINING_AGENT_INFLUENCE_RANGE" ]; then
    export AGENT_INFLUENCE_RANGE=$TRAINING_AGENT_INFLUENCE_RANGE
elif [ -z "${AGENT_INFLUENCE_RANGE}" ]; then
    export AGENT_INFLUENCE_RANGE=${AGENT_INFLUENCE_RANGE:-150.0}  # 🔧 修复：与训练脚本一致（120.0→150.0）
fi

# 🔧 关键修复：势场delta参数（优先级：训练配置 > 环境变量 > 默认值）
if [ -n "$TRAINING_DELTA_K_ATT" ]; then
    export DELTA_K_ATT=$TRAINING_DELTA_K_ATT
elif [ -z "${DELTA_K_ATT}" ]; then
    export DELTA_K_ATT=${DELTA_K_ATT:-5.0}  # 🔧 修复：与训练脚本一致（8.5→5.0）
fi

if [ -n "$TRAINING_DELTA_LAMBDA_1" ]; then
    export DELTA_LAMBDA_1=$TRAINING_DELTA_LAMBDA_1
elif [ -z "${DELTA_LAMBDA_1}" ]; then
    export DELTA_LAMBDA_1=${DELTA_LAMBDA_1:-2.2}  # ✅ 与训练脚本一致
fi

if [ -n "$TRAINING_DELTA_K_REP" ]; then
    export DELTA_K_REP=$TRAINING_DELTA_K_REP
elif [ -z "${DELTA_K_REP}" ]; then
    export DELTA_K_REP=${DELTA_K_REP:-600.0}  # 与run_optimized.sh当前训练默认一致
fi

if [ -n "$TRAINING_DELTA_RADIUS" ]; then
    export DELTA_RADIUS=$TRAINING_DELTA_RADIUS
elif [ -z "${DELTA_RADIUS}" ]; then
    export DELTA_RADIUS=${DELTA_RADIUS:-80.0}  # 🔧 修复：与训练脚本一致（60.0→80.0）
fi

if [ -n "$TRAINING_MAX_FORCE_MAGNITUDE" ]; then
    export MAX_FORCE_MAGNITUDE=$TRAINING_MAX_FORCE_MAGNITUDE
elif [ -z "${MAX_FORCE_MAGNITUDE}" ]; then
    export MAX_FORCE_MAGNITUDE=${MAX_FORCE_MAGNITUDE:-80.0}  # 与run_optimized.sh当前训练默认一致
fi

CMD_ARGS="$CMD_ARGS --action-force-ratio $ACTION_FORCE_RATIO"
if [ -n "${FORCE_EVAL_ACTION_FORCE_RATIO}" ]; then
    CMD_ARGS="$CMD_ARGS --force-action-force-ratio $FORCE_EVAL_ACTION_FORCE_RATIO"
fi
CMD_ARGS="$CMD_ARGS --use-tf-potential-field $USE_TF_POTENTIAL_FIELD"
CMD_ARGS="$CMD_ARGS --apf-backend $APF_BACKEND"
CMD_ARGS="$CMD_ARGS --use-fr-feature $USE_FR_FEATURE"
CMD_ARGS="$CMD_ARGS --use-pf-feature $USE_PF_FEATURE"
CMD_ARGS="$CMD_ARGS --terrain-sensing-mode $TERRAIN_SENSING_MODE"  # 🔧 新增：地形感知模式
CMD_ARGS="$CMD_ARGS --force-param-goal-attraction-range $FORCE_PARAM_GOAL_ATTRACTION_RANGE"
CMD_ARGS="$CMD_ARGS --force-param-lambda-1-range $FORCE_PARAM_LAMBDA_1_RANGE"
CMD_ARGS="$CMD_ARGS --force-param-terrain-repulsion-range $FORCE_PARAM_TERRAIN_REPULSION_RANGE"
CMD_ARGS="$CMD_ARGS --force-param-detection-radius-range $FORCE_PARAM_DETECTION_RADIUS_RANGE"
# 🔧 新增：传递元优化基准值
CMD_ARGS="$CMD_ARGS --goal-attraction $GOAL_ATTRACTION"
CMD_ARGS="$CMD_ARGS --lambda-1-base $LAMBDA_1_BASE"
CMD_ARGS="$CMD_ARGS --terrain-repulsion $TERRAIN_REPULSION"
CMD_ARGS="$CMD_ARGS --agent-influence-range $AGENT_INFLUENCE_RANGE"
CMD_ARGS="$CMD_ARGS --delta-k-att $DELTA_K_ATT"
CMD_ARGS="$CMD_ARGS --delta-lambda-1 $DELTA_LAMBDA_1"
CMD_ARGS="$CMD_ARGS --delta-k-rep $DELTA_K_REP"
CMD_ARGS="$CMD_ARGS --delta-radius $DELTA_RADIUS"
CMD_ARGS="$CMD_ARGS --max-force-magnitude $MAX_FORCE_MAGNITUDE"

# 势场/动作修正相关参数（可选）
if [ -n "$ENABLE_ACTION_CORRECTION" ]; then
    CMD_ARGS="$CMD_ARGS --enable-action-correction $ENABLE_ACTION_CORRECTION"
fi
if [ -n "$CORRECTION_TYPE" ]; then
    CMD_ARGS="$CMD_ARGS --correction-type $CORRECTION_TYPE"
fi
if [ -n "$INFLUENCE_RANGE" ]; then
    CMD_ARGS="$CMD_ARGS --influence-range $INFLUENCE_RANGE"
fi
if [ -n "$FORCE_PARAM_RATIO" ]; then
    CMD_ARGS="$CMD_ARGS --force-param-ratio $FORCE_PARAM_RATIO"
fi

# 设置默认地形复杂度等级（与训练脚本保持一致）
export TERRAIN_COMPLEXITY_LEVEL=${TERRAIN_COMPLEXITY_LEVEL:-3}  # 🔧 修复：与训练脚本一致的地形复杂度
CMD_ARGS="$CMD_ARGS --terrain-complexity-level $TERRAIN_COMPLEXITY_LEVEL"

# 🔧 修复：奖励参数优先完全回读训练配置，其次才使用外部环境变量/脚本默认值
apply_training_or_default DISTANCE_WEIGHT TRAINING_DISTANCE_WEIGHT 3.5
apply_training_or_default EXPLORATION_WEIGHT TRAINING_EXPLORATION_WEIGHT 0.3
apply_training_or_default STATIONARY_WEIGHT TRAINING_STATIONARY_WEIGHT 0.3
apply_training_or_default DIRECTION_WEIGHT TRAINING_DIRECTION_WEIGHT 0.6
apply_training_or_default DEVIATION_WEIGHT TRAINING_DEVIATION_WEIGHT 0.35
apply_training_or_default START_AREA_WEIGHT TRAINING_START_AREA_WEIGHT 0.3
apply_training_or_default APPROACH_WEIGHT TRAINING_APPROACH_WEIGHT 0.55
apply_training_or_default ENERGY_WEIGHT TRAINING_ENERGY_WEIGHT 0.2
apply_training_or_default HEIGHT_WEIGHT TRAINING_HEIGHT_WEIGHT 0.42
apply_training_or_default HEIGHT_REWARD_ENABLED TRAINING_HEIGHT_REWARD_ENABLED 1
apply_training_or_default HEIGHT_IDEAL_MIN TRAINING_HEIGHT_IDEAL_MIN 10.0
apply_training_or_default HEIGHT_IDEAL_MAX TRAINING_HEIGHT_IDEAL_MAX 60.0
apply_training_or_default LATERAL_WEIGHT TRAINING_LATERAL_WEIGHT 1.0
apply_training_or_default CLEARANCE_WEIGHT TRAINING_CLEARANCE_WEIGHT 0.44
apply_training_or_default CLEARANCE_D_MAX TRAINING_CLEARANCE_D_MAX 60.0
apply_training_or_default SUCCESS_WEIGHT TRAINING_SUCCESS_WEIGHT 1.8
apply_training_or_default COLLISION_WEIGHT TRAINING_COLLISION_WEIGHT 4.5
apply_training_or_default COLLISION_REDUCTION_WEIGHT TRAINING_COLLISION_REDUCTION_WEIGHT 0.0
apply_training_or_default GLOBAL_WEIGHT TRAINING_GLOBAL_WEIGHT 0.4
apply_training_or_default SHAPING_WEIGHT TRAINING_SHAPING_WEIGHT 0.3
apply_training_or_default MAX_REWARD TRAINING_MAX_REWARD 800.0
apply_training_or_default MIN_REWARD TRAINING_MIN_REWARD -800.0
apply_training_or_default SUCCESS_REWARD_VALUE TRAINING_SUCCESS_REWARD_VALUE 8000.0
apply_training_or_default NO_COLLISION_REWARD_VALUE TRAINING_NO_COLLISION_REWARD_VALUE 0.0
apply_training_or_default SUCCESS_DISTANCE_THRESHOLD TRAINING_SUCCESS_DISTANCE_THRESHOLD 5.0
apply_training_or_default TERRAIN_CONTACT_EPS TRAINING_TERRAIN_CONTACT_EPS 0.2
apply_training_or_default COLLISION_DISTANCE_THRESHOLD TRAINING_COLLISION_DISTANCE_THRESHOLD 0.5
apply_training_or_default COLLISION_PENALTY_VALUE TRAINING_COLLISION_PENALTY_VALUE 60.0
apply_training_or_default GLOBAL_REWARD_MODE TRAINING_GLOBAL_REWARD_MODE avg_progress
apply_training_or_default SHAPING_GAMMA TRAINING_SHAPING_GAMMA 0.9

CMD_ARGS="$CMD_ARGS --terrain-contact-eps $TERRAIN_CONTACT_EPS"

# 添加分项加权奖励参数（如果使用加权场景）
CMD_ARGS="$CMD_ARGS --distance-weight $DISTANCE_WEIGHT"
CMD_ARGS="$CMD_ARGS --exploration-weight $EXPLORATION_WEIGHT"
CMD_ARGS="$CMD_ARGS --stationary-weight $STATIONARY_WEIGHT"
CMD_ARGS="$CMD_ARGS --direction-weight $DIRECTION_WEIGHT"
CMD_ARGS="$CMD_ARGS --deviation-weight $DEVIATION_WEIGHT"
CMD_ARGS="$CMD_ARGS --start-area-weight $START_AREA_WEIGHT"
CMD_ARGS="$CMD_ARGS --approach-weight $APPROACH_WEIGHT"
CMD_ARGS="$CMD_ARGS --energy-weight $ENERGY_WEIGHT"
CMD_ARGS="$CMD_ARGS --height-weight $HEIGHT_WEIGHT"
CMD_ARGS="$CMD_ARGS --height-reward-enabled $HEIGHT_REWARD_ENABLED"
CMD_ARGS="$CMD_ARGS --height-ideal-min $HEIGHT_IDEAL_MIN"
CMD_ARGS="$CMD_ARGS --height-ideal-max $HEIGHT_IDEAL_MAX"
CMD_ARGS="$CMD_ARGS --lateral-weight $LATERAL_WEIGHT"
CMD_ARGS="$CMD_ARGS --clearance-weight $CLEARANCE_WEIGHT"
CMD_ARGS="$CMD_ARGS --clearance-d-max $CLEARANCE_D_MAX"
CMD_ARGS="$CMD_ARGS --success-weight $SUCCESS_WEIGHT"
CMD_ARGS="$CMD_ARGS --collision-weight $COLLISION_WEIGHT"
CMD_ARGS="$CMD_ARGS --collision-reduction-weight $COLLISION_REDUCTION_WEIGHT"
CMD_ARGS="$CMD_ARGS --global-weight $GLOBAL_WEIGHT"
CMD_ARGS="$CMD_ARGS --shaping-weight $SHAPING_WEIGHT"
CMD_ARGS="$CMD_ARGS --max-reward $MAX_REWARD"
CMD_ARGS="$CMD_ARGS --min-reward $MIN_REWARD"
CMD_ARGS="$CMD_ARGS --success-reward-value $SUCCESS_REWARD_VALUE"
CMD_ARGS="$CMD_ARGS --no-collision-reward-value $NO_COLLISION_REWARD_VALUE"
CMD_ARGS="$CMD_ARGS --success-distance-threshold $SUCCESS_DISTANCE_THRESHOLD"
CMD_ARGS="$CMD_ARGS --collision-penalty-value $COLLISION_PENALTY_VALUE"
CMD_ARGS="$CMD_ARGS --collision-distance-threshold $COLLISION_DISTANCE_THRESHOLD"
CMD_ARGS="$CMD_ARGS --global-reward-mode $GLOBAL_REWARD_MODE"
CMD_ARGS="$CMD_ARGS --shaping-gamma $SHAPING_GAMMA"

# 仅当设置时，透传隐藏层结构以匹配训练拓扑
if [ -n "$ACTOR_HIDDEN" ]; then
    CMD_ARGS="$CMD_ARGS --actor-hidden $ACTOR_HIDDEN"
fi
if [ -n "$CRITIC_HIDDEN" ]; then
    CMD_ARGS="$CMD_ARGS --critic-hidden $CRITIC_HIDDEN"
fi

# 🔧 关键修复：检查是否使用固定位置（支持"1"、"true"、"yes"、"on"等多种格式）
# 确保与ablation_terrain_sensing.py传递的"1"格式兼容
if [ "$USE_FIXED_POSITIONS" = "true" ] || [ "$USE_FIXED_POSITIONS" = "1" ] || [ "${USE_FIXED_POSITIONS,,}" = "yes" ] || [ "${USE_FIXED_POSITIONS,,}" = "on" ]; then
    if [ -f "$POSITIONS_FILE" ]; then
        echo "✅ 检测到固定位置设置，正在加载位置文件: $POSITIONS_FILE"
        CMD_ARGS="$CMD_ARGS --use-fixed-positions --positions-file \"$POSITIONS_FILE\""
    else
        if [ "$STRICT_EVAL_MATCH" = "1" ] || [ "$STRICT_EVAL_MATCH" = "true" ] || [ "$STRICT_EVAL_MATCH" = "yes" ] || [ "$STRICT_EVAL_MATCH" = "on" ]; then
            echo "❌ 严格模式：要求固定位置评估，但位置文件不存在: $POSITIONS_FILE"
            echo "   请提供训练时一致的位置文件路径，或将第5个参数设为 false。"
            exit 1
        else
            echo "⚠️  警告: 设置了使用固定位置，但位置文件不存在: $POSITIONS_FILE"
            echo "将使用随机初始化位置"
        fi
    fi
else
    echo "使用随机初始化位置（不使用固定位置）"
fi

# 检查是否禁用提前终止（用于生成完整轨迹GIF）
if [ "$DISABLE_EARLY_TERMINATION" = "true" ]; then
    echo ""
    echo "📹 启用完整轨迹模式:"
    echo "  - 禁用提前终止，将运行完整的4000步"
    echo "  - GIF将显示数千帧动画（而非只有~20帧）"
    echo "  - ⚠️  注意：运行时间将显著增加，GIF文件也会非常大"
    echo ""
    CMD_ARGS="$CMD_ARGS --disable-early-termination"
else
    echo ""
    echo "🎯 使用智能终止模式:"
    echo "  - 允许提前终止（智能体完成任务后结束）"
    echo "  - GIF可能只显示较少帧数，但运行快速"
    echo ""
fi

_eval_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# 默认可视化策略：只保留最佳奖励/最佳成功回合结果，避免每个episode都生成PNG/HTML拖慢评估
if _eval_truthy "${FAST_ARTIFACTS:-0}"; then
    : "${SAVE_INTERACTIVE_TRAJ:=0}"
    : "${SAVE_EVAL_ALL_EPISODES:=0}"
    : "${SAVE_BEST_TRAJ:=0}"
    : "${SAVE_TEAM_SUCCESS_HTML:=0}"
    : "${SAVE_EVAL_TRAJECTORY_JSON:=0}"
    : "${SAVE_EVAL_TRAJECTORY_PNG:=0}"
    : "${SAVE_EVAL_ACTOR_SEQUENCE:=0}"
    : "${SAVE_EVAL_CONTROL_DIAGNOSTICS:=0}"
    : "${DISABLE_TRAJECTORY_RECORDING:=1}"
    : "${EVAL_LIGHT_ACTION_PATH:=1}"
    : "${EVAL_LIGHT_INFO:=1}"
    : "${EVAL_ACTOR_ONLY:=1}"
    : "${EVAL_VERIFY_WEIGHT_COVERAGE:=0}"
    : "${TIMING_DETAIL:=0}"
    : "${TIMING_LEVEL:=1}"
    : "${DEBUG_EPISODE_SUMMARY:=0}"
    : "${DEBUG_COLLISION_SUMMARY:=0}"
    echo "⚡ FAST_ARTIFACTS=1：默认关闭评估轨迹/HTML/诊断artifact（显式设置的变量会保留）"
fi
export SAVE_INTERACTIVE_TRAJ=${SAVE_INTERACTIVE_TRAJ:-1}
export SAVE_EVAL_ALL_EPISODES=${SAVE_EVAL_ALL_EPISODES:-0}
export SAVE_BEST_TRAJ=${SAVE_BEST_TRAJ:-1}
export SAVE_TEAM_SUCCESS_HTML=${SAVE_TEAM_SUCCESS_HTML:-1}
export SAVE_EVAL_TRAJECTORY_JSON=${SAVE_EVAL_TRAJECTORY_JSON:-0}
export SAVE_EVAL_TRAJECTORY_PNG=${SAVE_EVAL_TRAJECTORY_PNG:-0}
export SAVE_EVAL_ACTOR_SEQUENCE=${SAVE_EVAL_ACTOR_SEQUENCE:-1}
export SAVE_EVAL_CONTROL_DIAGNOSTICS=${SAVE_EVAL_CONTROL_DIAGNOSTICS:-0}
export DISABLE_TRAJECTORY_RECORDING=${DISABLE_TRAJECTORY_RECORDING:-0}
export EVAL_LIGHT_ACTION_PATH=${EVAL_LIGHT_ACTION_PATH:-0}
export EVAL_LIGHT_INFO=${EVAL_LIGHT_INFO:-0}
# Policy evaluation only needs Actor weights; set EVAL_ACTOR_ONLY=0 only for critic loading audits.
export EVAL_ACTOR_ONLY=${EVAL_ACTOR_ONLY:-1}
export EVAL_VERIFY_WEIGHT_COVERAGE=${EVAL_VERIFY_WEIGHT_COVERAGE:-0}
export TIMING_DETAIL=${TIMING_DETAIL:-0}
export TIMING_LEVEL=${TIMING_LEVEL:-1}
export DEBUG_EPISODE_SUMMARY=${DEBUG_EPISODE_SUMMARY:-1}
export DEBUG_COLLISION_SUMMARY=${DEBUG_COLLISION_SUMMARY:-1}
export QUIET_OUTPUT=${QUIET_OUTPUT:-1}
export TQDM_DISABLE=${TQDM_DISABLE:-1}
export ENABLE_OVERLAY=${ENABLE_OVERLAY:-0}
export DISABLE_GIF=${DISABLE_GIF:-1}

# 启用overlay图片（包含地形信息）
if [ "${ENABLE_OVERLAY}" = "1" ] || [ "${ENABLE_OVERLAY,,}" = "true" ] || [ "${ENABLE_OVERLAY,,}" = "yes" ] || [ "${ENABLE_OVERLAY,,}" = "on" ]; then
    CMD_ARGS="$CMD_ARGS --enable-overlay"
    echo "✅ 启用overlay图片（包含地形信息）"
else
    echo "⏭️  跳过overlay图片生成"
fi

# 默认禁用GIF生成
if [ "${DISABLE_GIF:-0}" = "1" ] || [ "${DISABLE_GIF,,}" = "true" ] || [ "${DISABLE_GIF,,}" = "yes" ] || [ "${DISABLE_GIF,,}" = "on" ]; then
    CMD_ARGS="$CMD_ARGS --disable-gif"
    echo "⏭️  禁用GIF生成"
fi

# 运行评估
# 🔧 关键修复：使用参数数组执行，避免 CMD_ARGS 字符串中的引号/空格导致 shell 收尾误报失败
if [ "${ALGORITHM}" = "mappo" ]; then
    EVAL_SCRIPT="evaluate_mappo.py"
else
    EVAL_SCRIPT="evaluate_optimized.py"
fi

append_split_args() {
    local option_name="$1"
    local raw_value="$2"
    local parsed_values=()
    if [ -n "$raw_value" ]; then
        # 这些参数语义上就是“若干个以空格分隔的数值”，按空白拆分后再安全入数组。
        read -r -a parsed_values <<< "$raw_value"
        if [ ${#parsed_values[@]} -gt 0 ]; then
            EVAL_CMD+=("$option_name" "${parsed_values[@]}")
        fi
    fi
}

EVAL_CMD=(
    "$EVAL_PYTHON_BIN"
    "$EVAL_SCRIPT"
    --load-model-path "$MODEL_PATH"
    --eval-episodes "$EVAL_EPISODES"
    --eval-episode-parallelism "$EVAL_EPISODE_PARALLELISM"
    --eval-env-step-threads "$EVAL_ENV_STEP_THREADS"
    --save-viz-path "$SAVE_PATH"
    --scenario-name "$SCENARIO_NAME"
    --episode-length "$EPISODE_LENGTH"
    --algorithm "$ALGORITHM"
    --eval-noise-scale "$EVAL_NOISE_SCALE"
    --eval-random-action-prob "$EVAL_RANDOM_ACTION_PROB"
    --gravity "$GRAVITY"
    --control-accel-gain "$CONTROL_ACCEL_GAIN"
    --agent-size "$AGENT_SIZE"
    --agent-max-speed "$AGENT_MAX_SPEED"
    --agent-accel "$AGENT_ACCEL"
    --action-range-x "$ACTION_RANGE_X"
    --action-range-y "$ACTION_RANGE_Y"
    --action-range-z "$ACTION_RANGE_Z"
    --damping "$DAMPING"
    --simulation-dt "$SIMULATION_DT"
    --z-action-bias "$Z_ACTION_BIAS"
    --use-quadrotor-dynamics "$USE_QUADROTOR_DYNAMICS"
    --start-altitude-offset "$START_ALTITUDE_OFFSET"
    --goal-altitude "$GOAL_ALTITUDE"
    --agent-goal-formation-radius "$AGENT_GOAL_FORMATION_RADIUS"
    --enable-role-shuffle "$ENABLE_ROLE_SHUFFLE"
    --init-vel-jitter-max "$INIT_VEL_JITTER_MAX"
    --mountain-min-distance "$MOUNTAIN_MIN_DISTANCE"
    --mountain-margin "$MOUNTAIN_MARGIN"
    --obstacle-min-clearance-start-goal "$OBSTACLE_MIN_CLEARANCE_START_GOAL"
    --terrain-collision-eps "$TERRAIN_COLLISION_EPS"
    --use-legacy-terrain "$USE_LEGACY_TERRAIN"
    --ring-base-reward "$RING_BASE_REWARD"
    --min-start-goal-dist "$MIN_START_GOAL_DIST"
    --max-start-goal-dist "$MAX_START_GOAL_DIST"
    --start-pos-margin "$START_POS_MARGIN"
    --start-pos-max-trials "$START_POS_MAX_TRIALS"
    --quadrotor-attitude-response-time "$QUADROTOR_ATTITUDE_RESPONSE_TIME"
    --quadrotor-psi-cmd "$QUADROTOR_PSI_CMD"
    --reward-pos-scale "$REWARD_POS_SCALE"
    --reward-neg-scale "$REWARD_NEG_SCALE"
    --action-force-ratio "$ACTION_FORCE_RATIO"
    --use-tf-potential-field "$USE_TF_POTENTIAL_FIELD"
    --apf-backend "$APF_BACKEND"
    --use-fr-feature "$USE_FR_FEATURE"
    --use-pf-feature "$USE_PF_FEATURE"
    --terrain-sensing-mode "$TERRAIN_SENSING_MODE"
    --goal-attraction "$GOAL_ATTRACTION"
    --lambda-1-base "$LAMBDA_1_BASE"
    --terrain-repulsion "$TERRAIN_REPULSION"
    --agent-influence-range "$AGENT_INFLUENCE_RANGE"
    --delta-k-att "$DELTA_K_ATT"
    --delta-lambda-1 "$DELTA_LAMBDA_1"
    --delta-k-rep "$DELTA_K_REP"
    --delta-radius "$DELTA_RADIUS"
    --max-force-magnitude "$MAX_FORCE_MAGNITUDE"
    --terrain-contact-eps "$TERRAIN_CONTACT_EPS"
    --obstacle-observation-mode "$OBSTACLE_OBSERVATION_MODE"
    --obstacle-risk-velocity-forward-weight "$OBSTACLE_RISK_VELOCITY_FORWARD_WEIGHT"
    --obstacle-risk-goal-along-weight "$OBSTACLE_RISK_GOAL_ALONG_WEIGHT"
    --terrain-complexity-level "$TERRAIN_COMPLEXITY_LEVEL"
    --distance-weight "$DISTANCE_WEIGHT"
    --exploration-weight "$EXPLORATION_WEIGHT"
    --stationary-weight "$STATIONARY_WEIGHT"
    --direction-weight "$DIRECTION_WEIGHT"
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
    --collision-reduction-weight "$COLLISION_REDUCTION_WEIGHT"
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

if [ -n "$EVAL_NOISE_SEED" ]; then
    EVAL_CMD+=(--eval-noise-seed "$EVAL_NOISE_SEED")
fi

if [ -n "${FORCE_EVAL_ACTION_FORCE_RATIO}" ]; then
    EVAL_CMD+=(--force-action-force-ratio "$FORCE_EVAL_ACTION_FORCE_RATIO")
fi

append_split_args --force-param-goal-attraction-range "$FORCE_PARAM_GOAL_ATTRACTION_RANGE"
append_split_args --force-param-lambda-1-range "$FORCE_PARAM_LAMBDA_1_RANGE"
append_split_args --force-param-terrain-repulsion-range "$FORCE_PARAM_TERRAIN_REPULSION_RANGE"
append_split_args --force-param-detection-radius-range "$FORCE_PARAM_DETECTION_RADIUS_RANGE"

if [ "${RANDOM_TERRAIN,,}" = "1" ] || [ "${RANDOM_TERRAIN,,}" = "true" ] || [ "${RANDOM_TERRAIN,,}" = "yes" ] || [ "${RANDOM_TERRAIN,,}" = "on" ]; then
    EVAL_CMD+=(--random-terrain)
fi
if [ -n "$ACTOR_HIDDEN" ]; then
    EVAL_CMD+=(--actor-hidden "$ACTOR_HIDDEN")
fi
if [ -n "$CRITIC_HIDDEN" ]; then
    EVAL_CMD+=(--critic-hidden "$CRITIC_HIDDEN")
fi
if [ "$USE_FIXED_POSITIONS" = "true" ] || [ "$USE_FIXED_POSITIONS" = "1" ] || [ "${USE_FIXED_POSITIONS,,}" = "yes" ] || [ "${USE_FIXED_POSITIONS,,}" = "on" ]; then
    if [ -f "$POSITIONS_FILE" ]; then
        EVAL_CMD+=(--use-fixed-positions --positions-file "$POSITIONS_FILE")
    fi
fi
if [ "$DISABLE_EARLY_TERMINATION" = "true" ]; then
    EVAL_CMD+=(--disable-early-termination)
fi
if [ "${ENABLE_OVERLAY}" = "1" ] || [ "${ENABLE_OVERLAY,,}" = "true" ] || [ "${ENABLE_OVERLAY,,}" = "yes" ] || [ "${ENABLE_OVERLAY,,}" = "on" ]; then
    EVAL_CMD+=(--enable-overlay)
fi
if [ "${DISABLE_GIF:-0}" = "1" ] || [ "${DISABLE_GIF,,}" = "true" ] || [ "${DISABLE_GIF,,}" = "yes" ] || [ "${DISABLE_GIF,,}" = "on" ]; then
    EVAL_CMD+=(--disable-gif)
fi
if [ "${EVAL_CONFIG_RESOLVE_ONLY:-0}" = "1" ] || [ "${EVAL_CONFIG_RESOLVE_ONLY,,}" = "true" ] || [ "${EVAL_CONFIG_RESOLVE_ONLY,,}" = "yes" ] || [ "${EVAL_CONFIG_RESOLVE_ONLY,,}" = "on" ]; then
    EVAL_CMD+=(--config-resolve-only)
fi

printf '正在执行:'
printf ' %q' "${EVAL_CMD[@]}"
printf '\n'
"${EVAL_CMD[@]}"

EVAL_EXIT_CODE=$?

echo ""
echo "======================================"

if [ $EVAL_EXIT_CODE -eq 0 ] && { [ "${EVAL_CONFIG_RESOLVE_ONLY:-0}" = "1" ] || [ "${EVAL_CONFIG_RESOLVE_ONLY,,}" = "true" ] || [ "${EVAL_CONFIG_RESOLVE_ONLY,,}" = "yes" ] || [ "${EVAL_CONFIG_RESOLVE_ONLY,,}" = "on" ]; }; then
    echo "✅ 评估配置解析与训练口径校验完成（未启动环境、未加载权重）"
elif [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ 评估成功完成!"
    echo ""
    echo "结果文件:"
    echo "  📊 评估统计: $SAVE_PATH/evaluation_results.json"
    
    # 列出生成的文件
    if [ -d "$SAVE_PATH" ]; then
        echo "  🖼️  生成的图片:"
        ls -la "$SAVE_PATH"/*.png 2>/dev/null | head -5 | while read -r line; do
            echo "     $(echo $line | awk '{print $9, $5}')"
        done
        
        echo "  🌐 生成的HTML交互图:"
        ls -la "$SAVE_PATH"/*.html 2>/dev/null | head -5 | while read -r line; do
            echo "     $(echo $line | awk '{print $9, $5}')"
        done
    fi
    
    echo ""
    echo "💡 查看结果:"
    echo "   cd $SAVE_PATH && ls -la"
else
    echo "❌ 评估失败，退出码: $EVAL_EXIT_CODE"
    echo ""
    echo "故障排除:"
    echo "  1. 检查模型路径是否正确"
    echo "  2. 检查Python依赖是否完整"
    echo "  3. 检查GPU内存是否充足"
fi

echo "======================================"
exit "$EVAL_EXIT_CODE"

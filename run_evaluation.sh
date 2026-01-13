#!/bin/bash

# MADDPG优化版模型评估一键运行脚本

# 🔧 关键修复：确保从脚本所在目录运行，避免相对路径问题
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置默认参数
MODEL_PATH=${1:-"models/双头q改进测试、无重力、无早停、无预热、固定地图、4环境_exp_20260109_220551"}
EVAL_EPISODES=${2:-3}

# 🔧 关键修复：规范化模型路径，移除多余的斜杠
MODEL_PATH=$(echo "$MODEL_PATH" | sed 's|//|/|g' | sed 's|/$||')
SAVE_PATH=${3:-"evaluation_results/$(basename "$MODEL_PATH")_$(date +%Y%m%d_%H%M%S)"}
POSITIONS_FILE=${4:-"./saved_positions/default_positions.json"}
USE_FIXED_POSITIONS=${5:-false}
DISABLE_EARLY_TERMINATION=${6:-true}  # 新增：是否禁用提前终止

echo ""
echo "评估参数:"
echo "  - 模型路径: $MODEL_PATH"
echo "  - 评估回合数: $EVAL_EPISODES"
echo "  - 结果保存路径: $SAVE_PATH"
echo "  - 固定位置文件: $POSITIONS_FILE"
echo "  - 使用固定位置: $USE_FIXED_POSITIONS"
echo "  - 禁用提前终止: $DISABLE_EARLY_TERMINATION"
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
            echo "  📅 $model_dir/best     - 训练过程中的最佳模型 (时间戳: $timestamp)"
            echo "  📅 $model_dir/final    - 训练完成后的最终模型 (时间戳: $timestamp)"
            echo "  📅 $model_dir/ep500    - 第500轮的模型快照 (时间戳: $timestamp)"
        fi
    done
    
    # 🔧 关键修复：搜索所有包含时间戳的模型目录（包括中文路径）
    for model_dir in models/*_*_*; do
        if [ -d "$model_dir" ] && [[ "$model_dir" =~ [0-9]{8}_[0-9]{6}$ ]]; then
            model_name=$(basename "$model_dir")
            echo "  📁 $model_dir/best     - 最佳模型 ($model_name)"
            echo "  📁 $model_dir/final    - 最终模型 ($model_name)"
        fi
    done
    
    # 搜索不带时间戳的旧模型
    if [ -d "models/optimized_exp" ]; then
        echo "  📁 models/optimized_exp/best     - 训练过程中的最佳模型 (旧版本)"
        echo "  📁 models/optimized_exp/final    - 训练完成后的最终模型 (旧版本)"
        echo "  📁 models/optimized_exp/ep500    - 第500轮的模型快照 (旧版本)"
    fi
    
    echo ""
    echo "使用方法:"
    echo "  $0 [模型路径] [评估回合数] [保存路径] [固定位置文件] [是否使用固定位置] [是否禁用提前终止]"
    echo ""
    echo "示例:"
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
    echo "  # 使用固定地形（与训练环境一致，推荐）"
    echo "  RANDOM_TERRAIN=0 $0 models/optimized_exp/best 3"
    echo "  # 使用随机地形（测试泛化能力）"
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
    echo "    - 0: 固定地形（与训练环境一致，推荐用于性能评估）"
    echo "    - 1: 随机地形（测试泛化能力，每回合生成新地形）"
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
        
        # 优先级：final -> best -> ep*（按数字排序，最新的优先）
        FOUND_PATH=""
        
        # 1. 优先查找 final 目录
        if [ -d "$MODEL_PATH/final" ]; then
            FINAL_WEIGHTS=$(find "$MODEL_PATH/final" -maxdepth 1 -name "actor_*.weights.h5" -type f 2>/dev/null | head -1)
            if [ -n "$FINAL_WEIGHTS" ]; then
                FOUND_PATH="$MODEL_PATH/final"
                echo "✅ 找到权重文件: $FOUND_PATH"
            fi
        fi
        
        # 2. 如果没有找到，查找 best 目录
        if [ -z "$FOUND_PATH" ] && [ -d "$MODEL_PATH/best" ]; then
            BEST_WEIGHTS=$(find "$MODEL_PATH/best" -maxdepth 1 -name "actor_*.weights.h5" -type f 2>/dev/null | head -1)
            if [ -n "$BEST_WEIGHTS" ]; then
                FOUND_PATH="$MODEL_PATH/best"
                echo "✅ 找到权重文件: $FOUND_PATH"
            fi
        fi
        
        # 3. 如果还没找到，查找所有 ep* 目录（按数字排序，最新的优先）
        if [ -z "$FOUND_PATH" ]; then
            EP_DIRS=$(find "$MODEL_PATH" -maxdepth 1 -type d -name "ep*" 2>/dev/null | sort -V -r)
            for ep_dir in $EP_DIRS; do
                EP_WEIGHTS=$(find "$ep_dir" -maxdepth 1 -name "actor_*.weights.h5" -type f 2>/dev/null | head -1)
                if [ -n "$EP_WEIGHTS" ]; then
                    FOUND_PATH="$ep_dir"
                    echo "✅ 找到权重文件: $FOUND_PATH"
                    break
                fi
            done
        fi
        
        # 4. 如果找到了，更新 MODEL_PATH
        if [ -n "$FOUND_PATH" ]; then
            MODEL_PATH="$FOUND_PATH"
            echo "📁 使用模型路径: $MODEL_PATH"
        else
            echo "❌ 错误: 在 $MODEL_PATH 及其子目录中未找到权重文件"
            echo ""
            echo "请检查以下目录:"
            echo "  - $MODEL_PATH/final"
            echo "  - $MODEL_PATH/best"
            echo "  - $MODEL_PATH/ep*"
            exit 1
        fi
    else
        echo "✅ 在指定目录中找到权重文件: $MODEL_PATH"
    fi
fi

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

# 构造命令参数
# 🔧 场景选择说明：
#   - 训练脚本默认使用 paper3d_terrain_weighted（USE_WEIGHTED_REWARD=1, VECTORIZED_SCENARIO=0）
#     或 paper3d_terrain_vectorized（USE_WEIGHTED_REWARD=1, VECTORIZED_SCENARIO=1）
#   - paper3d_terrain_vectorized 继承自 paper3d_terrain_weighted，奖励函数逻辑完全一致
#   - 评估时使用 paper3d_terrain_weighted 即可，确保奖励函数与训练完全一致
#   - 注意：评估场景与训练场景必须一致，否则奖励计算会不同，影响评估结果
SCENARIO_NAME="paper3d_terrain_weighted"
echo "使用评估场景: $SCENARIO_NAME"
echo "✅ 场景与训练一致（paper3d_terrain_weighted，奖励函数逻辑相同）"

# 🔧 新增：设置默认算法（与训练脚本保持一致）
export ALGORITHM=${ALGORITHM:-matd3}
echo "使用算法: $ALGORITHM"

# 设置默认地形模式（与训练脚本保持一致）
export RANDOM_TERRAIN=${RANDOM_TERRAIN:-0}  # 🔧 修复：默认使用固定地形，与训练脚本一致

# 🔧 关键修复：episode-length 与训练脚本一致（2800步）
# 🚨 问题：评估脚本使用2200步，但训练脚本使用2800步，导致评估时步数不足
# 🚨 修复：改为2800步，与训练脚本完全一致
# 🔧 关键修复：使用双引号包裹路径变量，确保特殊字符（中文、空格等）正确处理
CMD_ARGS="--load-model-path \"$MODEL_PATH\" --eval-episodes $EVAL_EPISODES --save-viz-path \"$SAVE_PATH\" --scenario-name $SCENARIO_NAME --episode-length 2800 --algorithm $ALGORITHM"

# 根据RANDOM_TERRAIN参数决定是否使用随机地形
if [ "$RANDOM_TERRAIN" = "1" ] || [ "$RANDOM_TERRAIN" = "true" ] || [ "$RANDOM_TERRAIN" = "yes" ]; then
    CMD_ARGS="$CMD_ARGS --random-terrain"
    echo "🏔️ 使用随机地形模式"
else
    echo "🏔️ 使用固定地形模式（与训练环境一致）"
fi

# 🔧 修复：设置默认物理参数（与训练脚本完全一致）
export GRAVITY=${GRAVITY:-0.0}
export CONTROL_ACCEL_GAIN=${CONTROL_ACCEL_GAIN:-1.0}
export AGENT_MAX_SPEED=${AGENT_MAX_SPEED:-37.5}  # 🔧 修复：与训练脚本一致（32.5→37.5）
export AGENT_ACCEL=${AGENT_ACCEL:-3.6}  # 🔧 修复：与训练脚本一致（2.6→3.6）

# 通过环境变量传入物理与动作范围参数
CMD_ARGS="$CMD_ARGS --gravity $GRAVITY"
CMD_ARGS="$CMD_ARGS --control-accel-gain $CONTROL_ACCEL_GAIN"
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
export DAMPING=${DAMPING:-0.18}  # 🔧 修复：与训练脚本一致（0.15→0.18）
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
CMD_ARGS="$CMD_ARGS --reward-pos-scale $REWARD_POS_SCALE"
CMD_ARGS="$CMD_ARGS --reward-neg-scale $REWARD_NEG_SCALE"

# 🔧 关键修复：从训练配置（results.json）中读取训练时使用的ACTION_FORCE_RATIO
# 优先使用训练时的FR值，确保评估与训练一致
TRAINING_FR=""
RESULTS_JSON_PATH=""

# 从模型路径推断可能的results.json位置
# 1. 模型目录的父目录（models/xxx/results.json）
MODEL_PARENT_DIR=$(dirname "$MODEL_PATH")
if [ -f "$MODEL_PARENT_DIR/results.json" ]; then
    RESULTS_JSON_PATH="$MODEL_PARENT_DIR/results.json"
fi

# 2. 如果没找到，尝试在logs目录中查找（根据模型路径推断实验名称）
if [ -z "$RESULTS_JSON_PATH" ]; then
    # 尝试从模型路径提取实验名称（例如：models/xxx_exp_20260105_223843/best -> xxx_exp_20260105_223843）
    MODEL_BASENAME=$(basename "$MODEL_PARENT_DIR")
    if [ -d "logs/$MODEL_BASENAME" ]; then
        # 在logs目录及其子目录中查找results.json（递归搜索）
        RESULTS_JSON_PATH=$(find "logs/$MODEL_BASENAME" -name "results.json" -type f 2>/dev/null | head -1)
    fi
fi

# 3. 如果还没找到，尝试在整个logs目录中搜索（匹配时间戳）
if [ -z "$RESULTS_JSON_PATH" ]; then
    # 尝试从模型路径提取时间戳（例如：models/xxx_exp_20260105_223843/best -> 20260105_223843）
    if [[ "$MODEL_PARENT_DIR" =~ ([0-9]{8}_[0-9]{6})$ ]]; then
        TIMESTAMP="${BASH_REMATCH[1]}"
        # 搜索包含该时间戳的logs目录下的results.json（递归搜索）
        RESULTS_JSON_PATH=$(find logs -type f -path "*/${TIMESTAMP}*/results.json" 2>/dev/null | head -1)
    fi
fi

# 4. 如果还没找到，尝试在整个logs目录中搜索包含实验名称的目录
if [ -z "$RESULTS_JSON_PATH" ]; then
    MODEL_BASENAME=$(basename "$MODEL_PARENT_DIR")
    # 搜索logs目录中包含实验名称的目录下的results.json（递归搜索）
    RESULTS_JSON_PATH=$(find logs -type d -name "*${MODEL_BASENAME}*" 2>/dev/null -exec find {} -name "results.json" -type f \; 2>/dev/null | head -1)
fi

# 🔧 关键修复：从训练配置（results.json）中读取所有势场参数
# 优先使用训练时的参数值，确保评估与训练完全一致
if [ -n "$RESULTS_JSON_PATH" ] && [ -f "$RESULTS_JSON_PATH" ]; then
    # 使用Python读取results.json中的所有势场参数
    TRAINING_PARAMS=$(python3 <<PYTHON_EOF
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
    
    if args is not None:
        params = {}
        # 读取所有势场相关参数和动作范围参数
        param_names = [
            'action_force_ratio',
            'goal_attraction',
            'lambda_1_base',
            'terrain_repulsion',
            'agent_influence_range',
            'delta_k_att',
            'delta_lambda_1',
            'delta_k_rep',
            'delta_radius',
            'action_range_x',
            'action_range_y',
            'action_range_z',
            'damping',
            'reward_pos_scale',
            'reward_neg_scale'
        ]
        for param_name in param_names:
            if param_name in args:
                params[param_name] = args[param_name]
        
        # 输出为JSON格式，方便bash解析
        import json
        print(json.dumps(params))
        sys.exit(0)
except Exception as e:
    pass
sys.exit(1)
PYTHON_EOF
)
    
    if [ -n "$TRAINING_PARAMS" ]; then
        # 解析JSON并设置环境变量
        TRAINING_FR=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('action_force_ratio', ''))" 2>/dev/null)
        TRAINING_GOAL_ATTRACTION=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('goal_attraction', ''))" 2>/dev/null)
        TRAINING_LAMBDA_1_BASE=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('lambda_1_base', ''))" 2>/dev/null)
        TRAINING_TERRAIN_REPULSION=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('terrain_repulsion', ''))" 2>/dev/null)
        TRAINING_AGENT_INFLUENCE_RANGE=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('agent_influence_range', ''))" 2>/dev/null)
        TRAINING_DELTA_K_ATT=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('delta_k_att', ''))" 2>/dev/null)
        TRAINING_DELTA_LAMBDA_1=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('delta_lambda_1', ''))" 2>/dev/null)
        TRAINING_DELTA_K_REP=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('delta_k_rep', ''))" 2>/dev/null)
        TRAINING_DELTA_RADIUS=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('delta_radius', ''))" 2>/dev/null)
        TRAINING_ACTION_RANGE_X=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('action_range_x', ''))" 2>/dev/null)
        TRAINING_ACTION_RANGE_Y=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('action_range_y', ''))" 2>/dev/null)
        TRAINING_ACTION_RANGE_Z=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('action_range_z', ''))" 2>/dev/null)
        TRAINING_DAMPING=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('damping', ''))" 2>/dev/null)
        TRAINING_REWARD_POS_SCALE=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('reward_pos_scale', ''))" 2>/dev/null)
        TRAINING_REWARD_NEG_SCALE=$(echo "$TRAINING_PARAMS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('reward_neg_scale', ''))" 2>/dev/null)
        
        if [ -n "$TRAINING_FR" ]; then
            echo "✅ 从训练配置读取ACTION_FORCE_RATIO: $TRAINING_FR (来源: $RESULTS_JSON_PATH)"
        fi
        if [ -n "$TRAINING_GOAL_ATTRACTION" ]; then
            echo "✅ 从训练配置读取GOAL_ATTRACTION: $TRAINING_GOAL_ATTRACTION"
        fi
        if [ -n "$TRAINING_TERRAIN_REPULSION" ]; then
            echo "✅ 从训练配置读取TERRAIN_REPULSION: $TRAINING_TERRAIN_REPULSION"
        fi
        if [ -n "$TRAINING_AGENT_INFLUENCE_RANGE" ]; then
            echo "✅ 从训练配置读取AGENT_INFLUENCE_RANGE: $TRAINING_AGENT_INFLUENCE_RANGE"
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
    fi
fi

# 🔧 关键修复：设置势场参数（优先级：训练配置 > 环境变量 > 默认值）
# 确保评估时使用的势场参数与训练时完全一致

# ACTION_FORCE_RATIO
if [ -n "$TRAINING_FR" ]; then
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
    export GOAL_ATTRACTION=${GOAL_ATTRACTION:-6.0}  # 🔧 修复：与训练脚本一致（15.0→6.0）
fi

if [ -n "$TRAINING_LAMBDA_1_BASE" ]; then
    export LAMBDA_1_BASE=$TRAINING_LAMBDA_1_BASE
elif [ -z "${LAMBDA_1_BASE}" ]; then
    export LAMBDA_1_BASE=${LAMBDA_1_BASE:-8.5}  # ✅ 与训练脚本一致
fi

if [ -n "$TRAINING_TERRAIN_REPULSION" ]; then
    export TERRAIN_REPULSION=$TRAINING_TERRAIN_REPULSION
elif [ -z "${TERRAIN_REPULSION}" ]; then
    export TERRAIN_REPULSION=${TERRAIN_REPULSION:-8000.0}  # 🔧 修复：与训练脚本一致（3800.0→8000.0）
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
    export DELTA_K_REP=${DELTA_K_REP:-1200.0}  # 🔧 修复：与训练脚本一致（1000.0→1200.0）
fi

if [ -n "$TRAINING_DELTA_RADIUS" ]; then
    export DELTA_RADIUS=$TRAINING_DELTA_RADIUS
elif [ -z "${DELTA_RADIUS}" ]; then
    export DELTA_RADIUS=${DELTA_RADIUS:-80.0}  # 🔧 修复：与训练脚本一致（60.0→80.0）
fi

CMD_ARGS="$CMD_ARGS --action-force-ratio $ACTION_FORCE_RATIO"
CMD_ARGS="$CMD_ARGS --use-tf-potential-field $USE_TF_POTENTIAL_FIELD"
CMD_ARGS="$CMD_ARGS --use-fr-feature $USE_FR_FEATURE"
CMD_ARGS="$CMD_ARGS --use-pf-feature $USE_PF_FEATURE"
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
export TERRAIN_COMPLEXITY_LEVEL=${TERRAIN_COMPLEXITY_LEVEL:-1}  # 🔧 修复：恢复与训练脚本一致的地形复杂度
CMD_ARGS="$CMD_ARGS --terrain-complexity-level $TERRAIN_COMPLEXITY_LEVEL"

# 🔧 修复：设置默认分项加权奖励参数（与训练脚本完全一致）
export DISTANCE_WEIGHT=${DISTANCE_WEIGHT:-3.5}  # 🔧 修复：与训练脚本一致（1.0→3.5）
export EXPLORATION_WEIGHT=${EXPLORATION_WEIGHT:-0.3}  # 🔧 修复：与训练脚本一致（0.8→0.3）
export STATIONARY_WEIGHT=${STATIONARY_WEIGHT:-0.3}  # 🔧 修复：与训练脚本一致（-1.8→0.3）
export DIRECTION_WEIGHT=${DIRECTION_WEIGHT:-0.6}  # 🔧 修复：与训练脚本一致（0.2→0.6）
export DEVIATION_WEIGHT=${DEVIATION_WEIGHT:-0.35}  # 🔧 修复：与训练脚本一致（0.5→0.35）
export START_AREA_WEIGHT=${START_AREA_WEIGHT:-0.3}  # 🔧 修复：与训练脚本一致（0.5→0.3）
export APPROACH_WEIGHT=${APPROACH_WEIGHT:-0.55}  # 🔧 修复：与训练脚本一致（0.8→0.55）
export ENERGY_WEIGHT=${ENERGY_WEIGHT:-0.2}  # 🔧 修复：与训练脚本一致（0.05→0.2）
export HEIGHT_WEIGHT=${HEIGHT_WEIGHT:-0.75}  # 🔧 修复：与训练脚本一致（0.2→0.75）
export HEIGHT_REWARD_ENABLED=${HEIGHT_REWARD_ENABLED:-1}
export HEIGHT_IDEAL_MIN=${HEIGHT_IDEAL_MIN:-10.0}  # 🔧 修复：与训练脚本一致（8.0→10.0）
export HEIGHT_IDEAL_MAX=${HEIGHT_IDEAL_MAX:-60.0}  # 🔧 修复：与训练脚本一致（25.0→60.0）
export LATERAL_WEIGHT=${LATERAL_WEIGHT:-1.0}  # 🔧 修复：与训练脚本一致（0.8→1.0）
export CLEARANCE_WEIGHT=${CLEARANCE_WEIGHT:-0.75}  # 🔧 修复：与训练脚本一致（0.8→0.75）
export CLEARANCE_D_MAX=${CLEARANCE_D_MAX:-60.0}  # 🔧 修复：与训练脚本一致（78.0→60.0）
export SUCCESS_WEIGHT=${SUCCESS_WEIGHT:-1.8}  # 🔧 修复：与训练脚本一致（5.0→1.8）
export COLLISION_WEIGHT=${COLLISION_WEIGHT:-4.5}  # 🔧 修复：与训练脚本一致（2.5→4.5）
export GLOBAL_WEIGHT=${GLOBAL_WEIGHT:-0.4}  # 🔧 修复：与训练脚本一致（0.2→0.4）
export SHAPING_WEIGHT=${SHAPING_WEIGHT:-0.3}  # 🔧 修复：与训练脚本一致（0.2→0.3）
export MAX_REWARD=${MAX_REWARD:-800.0}  # 🔧 修复：与训练脚本一致（550.0→800.0）
export MIN_REWARD=${MIN_REWARD:--800.0}  # 🔧 修复：与训练脚本一致（-550.0→-800.0）
export SUCCESS_REWARD_VALUE=${SUCCESS_REWARD_VALUE:-8000.0}  # 🔧 修复：与训练脚本一致（500.0→8000.0）
export SUCCESS_DISTANCE_THRESHOLD=${SUCCESS_DISTANCE_THRESHOLD:-5.0}  # 🔧 关键修复：与训练脚本一致（6.0→5.0）
                                                                     # 🚨 问题：评估脚本使用6.0，但训练脚本使用5.0，导致评估时成功阈值更严格（7.2米 vs 6.0米）
                                                                     # 🚨 修复：改为5.0，与训练脚本完全一致
                                                                     # 实际判断阈值 = 1.2 * SUCCESS_DISTANCE_THRESHOLD = 6.0米（与训练一致）
export COLLISION_PENALTY_VALUE=${COLLISION_PENALTY_VALUE:-39.0}  # 🔧 修复：与训练脚本一致（80.0→39.0）
export COLLISION_DISTANCE_THRESHOLD=${COLLISION_DISTANCE_THRESHOLD:-0.8}  # 🔧 修复：与训练脚本一致（0.3→0.8）
export GLOBAL_REWARD_MODE=${GLOBAL_REWARD_MODE:-avg_progress}
export SHAPING_GAMMA=${SHAPING_GAMMA:-0.9}

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
CMD_ARGS="$CMD_ARGS --global-weight $GLOBAL_WEIGHT"
CMD_ARGS="$CMD_ARGS --shaping-weight $SHAPING_WEIGHT"
CMD_ARGS="$CMD_ARGS --max-reward $MAX_REWARD"
CMD_ARGS="$CMD_ARGS --min-reward $MIN_REWARD"
CMD_ARGS="$CMD_ARGS --success-reward-value $SUCCESS_REWARD_VALUE"
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

# 检查是否使用固定位置
if [ "$USE_FIXED_POSITIONS" = "true" ]; then
    if [ -f "$POSITIONS_FILE" ]; then
        echo "检测到固定位置设置，正在加载位置文件: $POSITIONS_FILE"
        CMD_ARGS="$CMD_ARGS --use-fixed-positions --positions-file \"$POSITIONS_FILE\""
    else
        echo "⚠️  警告: 设置了使用固定位置，但位置文件不存在: $POSITIONS_FILE"
        echo "将使用随机初始化位置"
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

# 🔧 启用交互式HTML和overlay图片（与训练时的最佳回合图效果一致）
# 交互式HTML默认已启用（评估脚本默认生成），无需额外设置
# 启用overlay图片（包含地形信息）
if [ "${ENABLE_OVERLAY:-0}" = "1" ] || [ "${ENABLE_OVERLAY,,}" = "true" ] || [ "${ENABLE_OVERLAY,,}" = "yes" ] || [ "${ENABLE_OVERLAY,,}" = "on" ]; then
    CMD_ARGS="$CMD_ARGS --enable-overlay"
    echo "✅ 启用overlay图片（包含地形信息）"
fi

# 运行评估
# 🔧 关键修复：使用eval正确展开CMD_ARGS中的引号，确保路径中的特殊字符被正确处理
echo "正在执行: python3 evaluate_optimized.py $CMD_ARGS"
eval "python3 evaluate_optimized.py $CMD_ARGS"

EVAL_EXIT_CODE=$?

echo ""
echo "======================================"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
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
        
        echo "  🎬 生成的动画:"
        ls -la "$SAVE_PATH"/*.gif 2>/dev/null | head -3 | while read -r line; do
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

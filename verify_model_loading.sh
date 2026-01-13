#!/bin/bash

# 模型加载验证脚本
# 用法: source maddpg_venv/bin/activate && bash verify_model_loading.sh

echo "========================================"
echo "模型加载维度匹配验证"
echo "========================================"
echo ""

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 错误: 未检测到虚拟环境"
    echo "请先激活: source maddpg_venv/bin/activate"
    exit 1
fi

echo "✅ 虚拟环境: $VIRTUAL_ENV"
echo ""

# 模型路径
MODEL_PATH="models/调试分离梯度、无重力、无早停、预热、随机地图、高变FR低高低_exp_20251201_112141/final"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "📦 模型路径: $MODEL_PATH"
echo ""

# 测试1: 检查网络构建维度
echo "【测试1】验证网络构建维度"
python3 << 'PYTHON_EOF'
import sys
import argparse

# 模拟参数
args = argparse.Namespace(
    scenario_name='paper3d_terrain_weighted',
    use_fr_feature=True,
    use_pf_feature=True,
)

# 导入场景
try:
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    scenario = scenarios.load(args.scenario_name + '.py').Scenario()
    world = scenario.make_world(argparse.Namespace(
        n_agents=3,
        use_tf_potential_field=True
    ))
    
    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation,
        info_callback=None, shared_viewer=False
    )
    
    # 基础观测维度
    base_obs_dim = env.observation_space[0].shape[0]
    print(f"  基础观测维度: {base_obs_dim}")
    
    # 如果启用PF特征，需要+3
    if args.use_pf_feature:
        final_obs_dim = base_obs_dim + 3
        print(f"  启用PF特征后: {final_obs_dim} (基础{base_obs_dim} + 势场力3)")
    else:
        final_obs_dim = base_obs_dim
        print(f"  最终观测维度: {final_obs_dim}")
    
    # 验证维度
    expected_dim = 81  # 训练时的观测维度
    if final_obs_dim == expected_dim:
        print(f"  ✅ 维度匹配! 期望={expected_dim}, 实际={final_obs_dim}")
        sys.exit(0)
    else:
        print(f"  ❌ 维度不匹配! 期望={expected_dim}, 实际={final_obs_dim}")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

TEST1_RESULT=$?
if [ $TEST1_RESULT -eq 0 ]; then
    echo "✅ 测试1通过"
else
    echo "❌ 测试1失败"
    exit 1
fi
echo ""

# 测试2: 快速评估1步（检查模型加载率）
echo "【测试2】快速评估测试（1个回合，检查加载率）"
echo "运行评估脚本..."

# 临时禁用tqdm输出
export EVAL_EPISODES=1

timeout 120s bash run_evaluation.sh "$MODEL_PATH" 1 2>&1 | tee /tmp/eval_test.log

# 检查加载率
LOAD_RATE=$(grep "加载比例" /tmp/eval_test.log | tail -1 | grep -oP '\d+\.\d+%' | head -1)
echo ""
echo "📊 模型加载率: $LOAD_RATE"

# 检查是否有维度不匹配警告
MISMATCH_COUNT=$(grep -c "shape of the target variable and the shape of the target value" /tmp/eval_test.log || echo "0")

if [ "$MISMATCH_COUNT" -eq "0" ]; then
    echo "✅ 没有维度不匹配警告"
    echo "✅ 测试2通过"
else
    echo "⚠️  仍有 $MISMATCH_COUNT 处维度不匹配警告"
    echo "❌ 测试2失败"
    
    # 显示警告详情
    echo ""
    echo "警告详情（前3个）:"
    grep -A 2 "shape of the target variable" /tmp/eval_test.log | head -10
    exit 1
fi

echo ""
echo "========================================"
echo "✅ 所有验证测试通过！"
echo "========================================"


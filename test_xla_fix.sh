#!/bin/bash
# XLA修复测试脚本 - 验证初始高度和XLA稳定性

echo "=========================================="
echo "XLA修复测试 - 5回合快速验证"
echo "=========================================="

# 清理旧的固定位置文件（已执行）
# rm -f saved_positions/*.json

# 导出关键参数
export NUM_EPISODES=5
export BATCH_SIZE=1024
export NUM_ENVS=3

# 初始高度配置（关键修复）
export START_ALTITUDE_OFFSET=7.0      # 智能体起始高度：7米
export GOAL_ALTITUDE=12.0              # 目标高度：12米
export HEIGHT_IDEAL_MIN=5.0            # 理想高度下限：5米
export HEIGHT_IDEAL_MAX=35.0           # 理想高度上限：35米

# XLA加速（必须启用）
export XLA_GLOBAL=1
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# 固定位置和地形（便于验证）
export USE_FIXED_POSITIONS=1
export DYNAMIC_FIRST_TIME=1            # 首次动态生成，然后固定
export SCENARIO_SEED=5                 # 固定地形种子
export PER_ENV_TERRAIN=0
export PER_EPISODE_TERRAIN=0
export RANDOM_TERRAIN=0
export UNLOCK_ENV_ON_SUCCESS=0
export UNLOCK_ENV_ON_PLATEAU=0

# Critic Loss修复参数
export Q_CLIP_VALUE=1000.0             # Q值裁剪：2000→1000
export CRITIC_Q_REG=0.01               # Q正则系数：0.05→0.01
export LEARNING_RATE_CRITIC=0.0003     # Critic学习率

# 其他关键参数
export ALGORITHM=matd3
export POLICY_FREQ=2
export ACTOR_UPDATE_DELAY=2

echo "关键配置："
echo "  - 智能体初始高度: ${START_ALTITUDE_OFFSET}米"
echo "  - 目标高度: ${GOAL_ALTITUDE}米"
echo "  - Q裁剪值: ${Q_CLIP_VALUE}"
echo "  - Q正则系数: ${CRITIC_Q_REG}"
echo "  - XLA加速: ${XLA_GLOBAL}"
echo "  - 训练回合: ${NUM_EPISODES}"
echo "=========================================="

# 运行测试
/bin/bash /home/tang/Desktop/run_optimized.sh

echo "=========================================="
echo "测试完成！"
echo "请检查："
echo "  1. 智能体初始位置是否在地形上方7米"
echo "  2. 目标位置是否在地形上方12米"
echo "  3. XLA是否稳定运行（无CUDA错误）"
echo "  4. Critic Loss是否稳定（无NaN）"
echo "=========================================="


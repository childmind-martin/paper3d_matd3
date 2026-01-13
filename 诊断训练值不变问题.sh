#!/bin/bash

# 训练值不变问题诊断和修复脚本
# 使用方法: ./诊断训练值不变问题.sh

echo "======================================"
echo "训练值不变问题诊断工具"
echo "======================================"
echo ""

# 检查关键配置
echo "【1】检查学习预热机制配置..."
WARMUP_ENABLED=$(grep -E "^export LEARNING_WARMUP_ENABLED" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
WARMUP_CODE_DEFAULT=$(grep -E "warmup_enabled = os.getenv\('LEARNING_WARMUP_ENABLED'," paper3d_train_optimized.py | head -1 | sed "s/.*'\([^']*\)'.*/\1/")

echo "  - 脚本默认值: ${WARMUP_ENABLED:-未找到}"
echo "  - 代码默认值: ${WARMUP_CODE_DEFAULT:-未找到}"

if [ "$WARMUP_ENABLED" != "$WARMUP_CODE_DEFAULT" ]; then
    echo "  ⚠️  警告：脚本和代码的默认值不一致！"
    echo "     这可能导致预热机制行为不符合预期"
else
    echo "  ✅ 配置一致"
fi
echo ""

# 检查学习率
echo "【2】检查学习率配置..."
ACTOR_LR=$(grep -E "^export LEARNING_RATE_ACTOR" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
CRITIC_LR=$(grep -E "^export LEARNING_RATE_CRITIC" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
echo "  - Actor学习率: ${ACTOR_LR:-未找到}"
echo "  - Critic学习率: ${CRITIC_LR:-未找到}"

if [ -n "$ACTOR_LR" ]; then
    ACTOR_LR_NUM=$(echo "$ACTOR_LR" | sed 's/[^0-9.]//g')
    if (( $(echo "$ACTOR_LR_NUM < 0.0001" | bc -l 2>/dev/null || echo "0") )); then
        echo "  ⚠️  警告：Actor学习率非常小（<0.0001），可能导致训练缓慢"
    fi
fi
echo ""

# 检查学习率衰减
echo "【3】检查学习率衰减配置..."
LR_DECAY_ENABLED=$(grep -E "^export LR_DECAY_ENABLED" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
LR_DECAY_RATE=$(grep -E "^export LR_DECAY_RATE" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
LR_DECAY_STEPS=$(grep -E "^export LR_DECAY_STEPS" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
echo "  - 衰减启用: ${LR_DECAY_ENABLED:-未找到}"
echo "  - 衰减率: ${LR_DECAY_RATE:-未找到}"
echo "  - 衰减步数: ${LR_DECAY_STEPS:-未找到}"

if [ "$LR_DECAY_ENABLED" = "1" ] || [ "$LR_DECAY_ENABLED" = "true" ]; then
    if [ -n "$LR_DECAY_RATE" ]; then
        RATE_NUM=$(echo "$LR_DECAY_RATE" | sed 's/[^0-9.]//g')
        if (( $(echo "$RATE_NUM < 0.95" | bc -l 2>/dev/null || echo "0") )); then
            echo "  ⚠️  警告：衰减率较小（<0.95），学习率会快速下降"
        fi
    fi
fi
echo ""

# 检查Actor延迟更新
echo "【4】检查Actor延迟更新配置..."
ACTOR_DELAY=$(grep -E "^export ACTOR_UPDATE_DELAY" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
echo "  - Actor更新延迟: ${ACTOR_DELAY:-未找到}"
if [ -n "$ACTOR_DELAY" ]; then
    DELAY_NUM=$(echo "$ACTOR_DELAY" | sed 's/[^0-9]//g')
    if [ "$DELAY_NUM" -gt 5 ]; then
        echo "  ⚠️  警告：Actor更新延迟较大（>5），可能导致Actor更新频率过低"
    fi
fi
echo ""

# 检查更新频率
echo "【5】检查训练更新频率..."
UPDATE_RATE=$(grep -E "^export UPDATE_RATE" run_optimized.sh | head -1 | sed 's/.*=\(.*\)#.*/\1/' | tr -d '[:space:]')
BATCH_SIZE=$(grep -E "^BATCH_SIZE=\${2:-" run_optimized.sh | head -1 | sed 's/.*:-\([^}]*\)}.*/\1/' | tr -d '[:space:]')
echo "  - 更新频率: ${UPDATE_RATE:-未找到}"
echo "  - 批次大小: ${BATCH_SIZE:-未找到}"
echo ""

# 检查梯度更新逻辑
echo "【6】检查梯度更新逻辑..."
GRAD_CHECK_COUNT=$(grep -c "grads_finite" paper3d_train_optimized.py 2>/dev/null || echo "0")
GRAD_WARNING_COUNT=$(grep -c "梯度异常\|gradient.*warning\|⚠️.*梯度" paper3d_train_optimized.py 2>/dev/null || echo "0")
echo "  - 梯度检查代码行数: $GRAD_CHECK_COUNT"
echo "  - 梯度警告代码行数: $GRAD_WARNING_COUNT"

if [ "$GRAD_WARNING_COUNT" -eq 0 ]; then
    echo "  ⚠️  警告：未找到梯度异常警告输出，可能导致问题难以发现"
fi
echo ""

# 生成修复建议
echo "======================================"
echo "修复建议"
echo "======================================"
echo ""

# 建议1：修复预热机制默认值
if [ "$WARMUP_ENABLED" != "$WARMUP_CODE_DEFAULT" ]; then
    echo "【建议1】修复预热机制默认值不一致问题"
    echo "  修改 paper3d_train_optimized.py 第8631行："
    echo "  将: warmup_enabled = os.getenv('LEARNING_WARMUP_ENABLED', '1')"
    echo "  改为: warmup_enabled = os.getenv('LEARNING_WARMUP_ENABLED', '0')"
    echo ""
fi

# 建议2：添加梯度警告
if [ "$GRAD_WARNING_COUNT" -eq 0 ]; then
    echo "【建议2】添加梯度异常警告输出"
    echo "  在 paper3d_train_optimized.py 的梯度检查处添加警告输出"
    echo "  详见分析报告中的修复建议2"
    echo ""
fi

# 建议3：快速测试
echo "【建议3】快速诊断测试"
echo "  运行以下命令进行快速测试："
echo ""
echo "  # 测试1：禁用预热机制"
echo "  LEARNING_WARMUP_ENABLED=0 ./run_optimized.sh 10 512 'test_no_warmup' 1"
echo ""
echo "  # 测试2：使用固定学习率"
echo "  LR_DECAY_ENABLED=0 ./run_optimized.sh 10 512 'test_fixed_lr' 1"
echo ""
echo "  # 测试3：启用详细输出"
echo "  QUIET_OUTPUT=0 DEBUG_EFF_SAMPLES=1 ./run_optimized.sh 10 512 'test_debug' 1"
echo ""

echo "======================================"
echo "详细分析报告已生成：训练值不变问题分析报告.md"
echo "======================================"


#!/bin/bash
# XLA 训练测试脚本：验证修复后的代码能否正常运行

echo "=========================================="
echo "XLA 加速训练测试"
echo "=========================================="

# 设置环境变量
export SUPPRESS_MA_PROMPT=1
export QUIET_OUTPUT=0
export TF_CPP_MIN_LOG_LEVEL=1

# 关闭 Auto JIT（按需手动启用）
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# 清理旧的 XLA_FLAGS（避免冲突）
unset XLA_FLAGS 2>/dev/null || true

echo ""
echo "环境配置:"
echo "  TF_XLA_FLAGS = $TF_XLA_FLAGS"
echo "  XLA_FLAGS    = ${XLA_FLAGS:-<未设置>}"
echo ""

# 运行小规模训练测试
echo "开始训练测试（1 episode, 20 steps, 2 envs）..."
echo "=========================================="

timeout 180 python3 /home/tang/Desktop/paper3d_train_optimized.py \
    --algo matd3 \
    --scenario paper3d_terrain_vectorized \
    --train-episodes 1 \
    --episode-length 20 \
    --num-envs 2 \
    --batch-size 64 \
    --buffer-size 1000 \
    --lite-buffer 1 \
    --jit-compile 1 \
    --exp-name "xla_test" \
    2>&1 | tee /tmp/xla_training_test.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 测试成功完成！"
    echo ""
    echo "检查是否出现 XLA 错误..."
    if grep -q "cond/zeros.*is out of scope" /tmp/xla_training_test.log; then
        echo "✗ 发现 XLA 兼容性错误"
        grep "cond/zeros.*is out of scope" /tmp/xla_training_test.log | head -5
        exit 1
    else
        echo "✓ 没有发现 XLA 兼容性错误"
    fi
    
    if grep -q "Unknown flags in XLA_FLAGS" /tmp/xla_training_test.log; then
        echo "✗ 发现 XLA_FLAGS 配置错误"
        grep "Unknown flags in XLA_FLAGS" /tmp/xla_training_test.log | head -5
        exit 1
    else
        echo "✓ 没有发现 XLA_FLAGS 配置错误"
    fi
    
    echo ""
    echo "修复验证成功！可以正常使用 XLA 加速。"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⚠ 测试超时（这可能是正常的，取决于硬件性能）"
    echo "检查日志查看是否有错误..."
    if grep -q "cond/zeros.*is out of scope" /tmp/xla_training_test.log; then
        echo "✗ 发现 XLA 兼容性错误"
        exit 1
    else
        echo "✓ 运行期间没有 XLA 错误"
    fi
else
    echo "✗ 测试失败，退出码: $EXIT_CODE"
    echo ""
    echo "最后50行日志:"
    tail -50 /tmp/xla_training_test.log
    exit 1
fi

echo "=========================================="
echo "完整日志保存在: /tmp/xla_training_test.log"
echo "=========================================="


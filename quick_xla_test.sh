#!/bin/bash
# 快速XLA测试 - 5分钟验证方案1

cd /home/tang/Desktop

echo "=========================================="
echo "⚡ 快速XLA测试 (5分钟)"
echo "=========================================="
echo ""
echo "测试配置："
echo "  - XLA保守模式 (autotune_level=1)"
echo "  - 确定性执行 (deterministic_ops=true)"
echo "  - 同步执行 (CUDA_LAUNCH_BLOCKING=1)"
echo "  - 混合精度 BF16"
echo "  - 测试回合：10"
echo "  - 批次大小：512（快速测试）"
echo ""
echo "如果这个测试通过，说明XLA可以在您的硬件上稳定运行！"
echo ""
echo "开始测试..."
echo ""

USE_XLA=1 \
XLA_FLAGS="--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true --xla_gpu_force_compilation_parallelism=1" \
CUDA_LAUNCH_BLOCKING=1 \
TF_SYNC_ON_FINISH=1 \
OPTIMIZER_JIT=1 \
JIT_COMPILE=1 \
AMP_MODE=bf16 \
./run_optimized.sh 10 512 "xla_quick_test" 1

STATUS=$?

echo ""
echo "=========================================="
echo "测试结果"
echo "=========================================="
echo ""

if [ $STATUS -eq 0 ]; then
    echo "🎉 测试成功！XLA保守模式运行稳定"
    echo ""
    echo "预期性能提升：+10-15% (每回合约22-25秒)"
    echo ""
    echo "下一步："
    echo "  1. 运行完整测试：./run_optimized.sh 100 1024 \"xla_training\" 1"
    echo "  2. 或运行全面测试：./test_xla_stability.sh"
    echo ""
    echo "如果您想将此配置设为默认，告诉我，我会修改 run_optimized.sh"
else
    echo "❌ 测试失败 (退出码: $STATUS)"
    echo ""
    echo "建议："
    echo "  1. 尝试降级TensorFlow："
    echo "     pip install tensorflow==2.12.0"
    echo ""
    echo "  2. 或保持禁用XLA（当前默认配置）"
    echo "     虽然慢5-10秒，但稳定运行"
fi

echo ""
echo "=========================================="

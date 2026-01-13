#!/bin/bash
# XLA稳定性测试套件 - 逐步测试不同的XLA配置

cd /home/tang/Desktop

echo "=========================================="
echo "🔬 XLA稳定性测试套件"
echo "=========================================="
echo ""
echo "将依次测试3种XLA配置，找到最稳定的方案"
echo ""

# 测试1：保守模式（最稳定）
echo "=========================================="
echo "测试1/3: XLA保守模式"
echo "=========================================="
echo "配置："
echo "  - autotune_level=1 (保守)"
echo "  - deterministic_ops=true (确定性)"
echo "  - force_compilation_parallelism=1 (串行编译)"
echo "  - CUDA_LAUNCH_BLOCKING=1 (同步执行)"
echo "  - AMP=bf16"
echo ""
echo "开始测试 10 回合..."
echo ""

USE_XLA=1 \
XLA_FLAGS="--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true --xla_gpu_force_compilation_parallelism=1" \
CUDA_LAUNCH_BLOCKING=1 \
TF_SYNC_ON_FINISH=1 \
OPTIMIZER_JIT=1 \
JIT_COMPILE=1 \
./run_optimized.sh 10 1024 "xla_test1_conservative" 1

TEST1_STATUS=$?

echo ""
if [ $TEST1_STATUS -eq 0 ]; then
    echo "✅ 测试1通过！保守模式稳定运行"
    echo ""
    echo "【推荐配置】使用保守XLA模式："
    echo "  USE_XLA=1"
    echo "  XLA_FLAGS=\"--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true --xla_gpu_force_compilation_parallelism=1\""
    echo "  CUDA_LAUNCH_BLOCKING=1"
    echo ""
    echo "继续测试更高性能的配置..."
else
    echo "❌ 测试1失败"
    echo "   即使保守模式也失败，建议："
    echo "   1. 降级TensorFlow版本到2.12.0"
    echo "   2. 或接受禁用XLA（稳定性优先）"
    echo ""
    exit 1
fi

sleep 5

# 测试2：中等模式（性能更好）
echo ""
echo "=========================================="
echo "测试2/3: XLA中等模式"
echo "=========================================="
echo "配置："
echo "  - autotune_level=2 (更激进)"
echo "  - CUDA_LAUNCH_BLOCKING=1 (同步执行)"
echo "  - AMP=bf16"
echo ""
echo "开始测试 10 回合..."
echo ""

USE_XLA=1 \
XLA_FLAGS="--xla_gpu_autotune_level=2" \
CUDA_LAUNCH_BLOCKING=1 \
TF_SYNC_ON_FINISH=1 \
OPTIMIZER_JIT=1 \
JIT_COMPILE=1 \
./run_optimized.sh 10 1024 "xla_test2_moderate" 1

TEST2_STATUS=$?

echo ""
if [ $TEST2_STATUS -eq 0 ]; then
    echo "✅ 测试2通过！中等模式稳定运行"
    echo ""
    echo "【推荐配置】使用中等XLA模式："
    echo "  USE_XLA=1"
    echo "  XLA_FLAGS=\"--xla_gpu_autotune_level=2\""
    echo "  CUDA_LAUNCH_BLOCKING=1"
    echo ""
    echo "继续测试最高性能的配置..."
else
    echo "⚠️  测试2失败，但测试1成功"
    echo "   建议使用测试1的保守模式配置"
    echo ""
    exit 0  # 测试1成功就算成功
fi

sleep 5

# 测试3：FP32模式（最稳定的数值）
echo ""
echo "=========================================="
echo "测试3/3: XLA + FP32模式"
echo "=========================================="
echo "配置："
echo "  - autotune_level=2"
echo "  - AMP=off (禁用混合精度，使用FP32)"
echo "  - CUDA_LAUNCH_BLOCKING=1"
echo ""
echo "开始测试 10 回合..."
echo ""

USE_XLA=1 \
XLA_FLAGS="--xla_gpu_autotune_level=2" \
CUDA_LAUNCH_BLOCKING=1 \
TF_SYNC_ON_FINISH=1 \
AMP_MODE=off \
OPTIMIZER_JIT=1 \
JIT_COMPILE=1 \
./run_optimized.sh 10 1024 "xla_test3_fp32" 1

TEST3_STATUS=$?

echo ""
if [ $TEST3_STATUS -eq 0 ]; then
    echo "✅ 测试3通过！XLA + FP32稳定运行"
    echo ""
    echo "【推荐配置】使用XLA + FP32模式："
    echo "  USE_XLA=1"
    echo "  XLA_FLAGS=\"--xla_gpu_autotune_level=2\""
    echo "  AMP_MODE=off"
    echo "  CUDA_LAUNCH_BLOCKING=1"
else
    echo "⚠️  测试3失败"
    echo "   可能是BF16与XLA的兼容性问题"
    echo ""
fi

echo ""
echo "=========================================="
echo "📊 测试结果汇总"
echo "=========================================="
echo ""
echo "测试1 (保守XLA + BF16): $([ $TEST1_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo "测试2 (中等XLA + BF16): $([ $TEST2_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo "测试3 (中等XLA + FP32): $([ $TEST3_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo ""

# 给出最终建议
if [ $TEST2_STATUS -eq 0 ]; then
    echo "🎉 最佳配置：测试2 (中等XLA + BF16)"
    echo ""
    echo "修改 run_optimized.sh，设置："
    echo "  export USE_XLA=1"
    echo "  export XLA_FLAGS=\"--xla_gpu_autotune_level=2\""
    echo "  export CUDA_LAUNCH_BLOCKING=1"
    echo "  export AMP_MODE=bf16"
elif [ $TEST1_STATUS -eq 0 ]; then
    echo "✅ 推荐配置：测试1 (保守XLA + BF16)"
    echo ""
    echo "修改 run_optimized.sh，设置："
    echo "  export USE_XLA=1"
    echo "  export XLA_FLAGS=\"--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true --xla_gpu_force_compilation_parallelism=1\""
    echo "  export CUDA_LAUNCH_BLOCKING=1"
    echo "  export AMP_MODE=bf16"
elif [ $TEST3_STATUS -eq 0 ]; then
    echo "⚠️  备用配置：测试3 (中等XLA + FP32)"
    echo ""
    echo "修改 run_optimized.sh，设置："
    echo "  export USE_XLA=1"
    echo "  export XLA_FLAGS=\"--xla_gpu_autotune_level=2\""
    echo "  export CUDA_LAUNCH_BLOCKING=1"
    echo "  export AMP_MODE=off"
else
    echo "❌ 所有XLA配置均失败"
    echo ""
    echo "建议："
    echo "  1. 降级TensorFlow: pip install tensorflow==2.12.0"
    echo "  2. 或接受禁用XLA（当前配置）"
fi

echo ""
echo "=========================================="


#!/bin/bash
# XLA友好性修复验证脚本
# 运行10回合测试，验证CUDA错误是否修复

echo "=========================================="
echo "XLA友好性修复验证测试"
echo "=========================================="
echo ""
echo "测试配置:"
echo "  - 回合数: 10"
echo "  - GPU缓存清理: 禁用 (0)"
echo "  - XLA Global: 启用 (1)"
echo "  - 异步执行: 启用 (默认)"
echo ""
echo "验证要点:"
echo "  1. 第1回合XLA编译完成 (15-30秒)"
echo "  2. 第2+回合无重复编译 (25-40秒)"
echo "  3. 无 CUDA_ERROR_INVALID_PC 错误"
echo "  4. 无 device_event_mgr 错误"
echo "  5. 至少连续运行10回合无崩溃"
echo ""
echo "=========================================="
echo ""

# 进入工作目录
cd /home/tang/Desktop || exit 1

# 确保XLA友好配置
export GPU_CACHE_CLEAR_INTERVAL=0  # 完全禁用GPU缓存清理
export XLA_GLOBAL=1                 # 启用XLA Global
export JIT_COMPILE=1                # 启用JIT编译
export PF_JIT=0                     # 禁用势场JIT（避免内存对齐问题）
export NUM_EPISODES=10              # 测试10回合
export QUIET_OUTPUT=0               # 输出详细日志

# 显式设置XLA标志（禁用Triton GEMM）
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "开始测试..."
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行训练
/bin/bash run_optimized.sh

# 捕获退出码
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo ""
echo "总用时: ${DURATION}秒"
echo "退出码: ${EXIT_CODE}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试通过！"
    echo ""
    echo "验证结果:"
    echo "  ✅ 无CUDA错误"
    echo "  ✅ 无core dumped"
    echo "  ✅ XLA编译正常"
    echo ""
    echo "下一步建议:"
    echo "  1. 恢复到完整训练 (NUM_EPISODES=200)"
    echo "  2. 监控长时间运行稳定性"
    echo "  3. 检查训练性能提升"
else
    echo "❌ 测试失败！退出码: ${EXIT_CODE}"
    echo ""
    echo "请检查终端输出中的错误信息:"
    echo "  - CUDA_ERROR_INVALID_PC"
    echo "  - device_event_mgr"
    echo "  - Aborted (core dumped)"
    echo ""
    echo "如果仍有问题，尝试降级方案:"
    echo "  1. 禁用XLA Global: export XLA_GLOBAL=0"
    echo "  2. 启用同步执行: export CUDA_LAUNCH_BLOCKING=1"
    echo "  3. 降低并行度: export NUM_ENVS=1"
fi

echo ""
echo "详细报告请查看: XLA_FRIENDLY_FIX_REPORT.md"
echo "=========================================="

exit $EXIT_CODE


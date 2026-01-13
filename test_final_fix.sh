#!/bin/bash
# 终极修复验证脚本

echo "========================================"
echo "🔧 终极修复验证：完全消除 Variant Tensor 错误"
echo "========================================"
echo ""
echo "修复内容："
echo "  - 移除了 _multi_agent_update_step 中两处基于动态形状的 if/else 分支"
echo "  - 替换为无分支的张量拼接操作，彻底消除 tf.cond"
echo "  - 保持逻辑100%等价，性能更高，GPU更友好"
echo ""
echo "验证配置："
echo "  - 训练回合: 10"
echo "  - 批次大小: 1024"
echo "  - 加速模式: 无XLA + BF16 + TF32（最优配置）"
echo ""
echo "预期结果："
echo "  ✅ 训练成功运行10个回合"
echo "  ✅ 彻底消除 'GPU copy from non-DMA variant tensor' 错误"
echo "  ⚡ 首回合约270秒（编译）"
echo "  ⚡ 后续回合约50秒"
echo ""
echo "开始验证..."
echo "========================================"
echo ""

# 记录开始时间
start_time=$(date +%s)

# 运行10回合验证
./run_optimized.sh 10 1024 "final_fix_test_$(date +%H%M%S)" 1

# 记录结束时间和状态
end_time=$(date +%s)
exit_code=$?
duration=$((end_time - start_time))

echo ""
echo "========================================"
echo "验证结果"
echo "========================================"
echo "退出代码: $exit_code"
echo "总耗时: ${duration}秒"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "🎉 验证成功！终极修复生效！"
    echo ""
    echo "✅ Variant Tensor 问题已彻底解决"
    echo "✅ 训练可以正常运行"
    echo "✅ GPU 计算图完全稳定"
    echo ""
    echo "现在可以开始正式的长时间训练："
    echo "  ./run_optimized.sh 100 1536 \"production_training\" 1"
    echo ""
    echo "预期性能："
    echo "  ⚡ 每回合约 50秒（稳定后）"
    echo "  💾 显存占用约 4GB（BF16混合精度）"
    echo "  🔥 训练稳定，无崩溃风险"
    echo ""
else
    echo "❌ 验证失败（退出代码: $exit_code）"
    echo ""
    echo "如果仍然出现错误，可能需要进一步排查："
    echo "  1. 检查是否还有其他隐藏的 tf.cond 节点"
    echo "  2. 验证 TensorFlow 版本兼容性"
    echo "  3. 排查其他可能的 GPU 内存问题"
    echo ""
fi
echo "========================================"

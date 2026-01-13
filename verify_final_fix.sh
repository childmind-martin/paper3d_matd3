#!/bin/bash
# 最终修复验证脚本

echo "========================================"
echo "🔧 最终修复验证：Variant Tensor 问题"
echo "========================================"
echo ""
echo "修复内容："
echo "  - 移除 _multi_agent_update_step 中的 try...except 块"
echo "  - 改用纯 TensorFlow 操作（tf.where + tf.math.is_finite）"
echo "  - 彻底消除 tf.cond 生成的 Variant 张量"
echo ""
echo "验证配置："
echo "  - 训练回合: 10"
echo "  - 批次大小: 1024"
echo "  - 加速模式: 无XLA + BF16 + TF32"
echo ""
echo "预期结果："
echo "  ✅ 训练顺利完成10个回合"
echo "  ⚡ 首回合约270秒（编译）"
echo "  ⚡ 后续回合约50秒"
echo ""
echo "开始验证..."
echo "========================================"
echo ""

# 记录开始时间
start_time=$(date +%s)

# 运行10回合验证
./run_optimized.sh 10 1024 "final_fix_validation_$(date +%H%M%S)" 1

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
    echo "✅ 验证成功！"
    echo ""
    echo "修复已生效，可以开始正式训练："
    echo "  ./run_optimized.sh 100 1536 \"production_training\" 1"
    echo ""
else
    echo "❌ 验证失败（退出代码: $exit_code）"
    echo ""
    echo "请检查日志了解详细错误信息"
    echo ""
fi
echo "========================================"

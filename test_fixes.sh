#!/bin/bash
#
# 快速测试修复效果
#
# 运行10回合，观察：
# 1. 奖励是否在合理范围内（-300k到-100k）
# 2. 是否出现CUDA崩溃
# 3. Critic loss是否下降
# 4. 奖励是否有改善趋势

cd /home/tang/Desktop

echo "=========================================="
echo "🧪 测试修复效果"
echo "=========================================="
echo ""
echo "修复内容："
echo "  ✅ 奖励裁剪：从-2500改为-120"
echo "  ✅ 势场归一化：折中强度（max_force*1.5）"
echo "  ✅ CUDA内存对齐：强制ascontiguousarray"
echo "  ✅ 诊断输出：每500步打印奖励统计"
echo ""
echo "测试参数："
echo "  - 回合数: 10"
echo "  - 批次大小: 1024"
echo "  - 并行环境: 5"
echo "  - 势场修正: TF版本（启用）"
echo ""
echo "预期结果："
echo "  1. 单步奖励应在[-120, 120]范围内"
echo "  2. 回合奖励应在[-300k, -100k]范围内"
echo "  3. Critic loss应逐渐下降（< 200）"
echo "  4. 不应出现CUDA崩溃"
echo "  5. 奖励应有改善趋势"
echo ""
read -p "按Enter开始测试（或Ctrl+C取消）..." 

# 设置安静输出以减少干扰
export QUIET_OUTPUT=0  # 启用诊断输出

# 运行测试
./run_optimized.sh 10 1024 "fix_comprehensive_test" 1

echo ""
echo "=========================================="
echo "测试完成！请检查："
echo "=========================================="
echo "1. 查看日志中的 [诊断] 输出，确认奖励范围"
echo "2. 检查是否出现CUDA错误"
echo "3. 观察Critic loss是否下降"
echo "4. 比较回合1-10的奖励是否改善"
echo ""
echo "如果问题仍然存在，请查看详细分析报告："
echo "  cat TRAINING_ISSUES_ANALYSIS.md"
echo ""


#!/bin/bash
####################################################################################
# 🧪 新地形生成测试脚本
# 用途：快速验证高斯山峰地形生成效果
####################################################################################

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      🧪 测试新的高斯山峰地形生成                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 设置基础参数
EPISODES=1
BATCH_SIZE=1024
QUIET=1

# 测试1：复杂度等级1（简单）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 测试1：复杂度等级1 (简单 - 5个山峰)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TERRAIN_COMPLEXITY_LEVEL=1 \
USE_SCENARIO_SEED=1 \
SCENARIO_SEED=101 \
./run_optimized.sh $EPISODES $BATCH_SIZE "terrain_test_level1" $QUIET

echo ""
echo "✅ 测试1完成，查看地图:"
echo "   xdg-open logs/terrain_test_level1_*/best_episode_*.html"
echo ""
sleep 2

# 测试2：复杂度等级2（中等）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 测试2：复杂度等级2 (中等 - 6个山峰)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TERRAIN_COMPLEXITY_LEVEL=2 \
USE_SCENARIO_SEED=1 \
SCENARIO_SEED=102 \
./run_optimized.sh $EPISODES $BATCH_SIZE "terrain_test_level2" $QUIET

echo ""
echo "✅ 测试2完成，查看地图:"
echo "   xdg-open logs/terrain_test_level2_*/best_episode_*.html"
echo ""
sleep 2

# 测试3：复杂度等级3（困难）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 测试3：复杂度等级3 (困难 - 7个山峰)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TERRAIN_COMPLEXITY_LEVEL=3 \
USE_SCENARIO_SEED=1 \
SCENARIO_SEED=103 \
./run_optimized.sh $EPISODES $BATCH_SIZE "terrain_test_level3" $QUIET

echo ""
echo "✅ 测试3完成，查看地图:"
echo "   xdg-open logs/terrain_test_level3_*/best_episode_*.html"
echo ""
sleep 2

# 测试4：复杂度等级4（极难）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 测试4：复杂度等级4 (极难 - 8个山峰)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TERRAIN_COMPLEXITY_LEVEL=4 \
USE_SCENARIO_SEED=1 \
SCENARIO_SEED=104 \
./run_optimized.sh $EPISODES $BATCH_SIZE "terrain_test_level4" $QUIET

echo ""
echo "✅ 测试4完成，查看地图:"
echo "   xdg-open logs/terrain_test_level4_*/best_episode_*.html"
echo ""

# 总结
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  🎉 所有测试完成！                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 生成的地图位置："
echo ""
echo "  等级1 (简单):   logs/terrain_test_level1_*/best_episode_*.html"
echo "  等级2 (中等):   logs/terrain_test_level2_*/best_episode_*.html"
echo "  等级3 (困难):   logs/terrain_test_level3_*/best_episode_*.html"
echo "  等级4 (极难):   logs/terrain_test_level4_*/best_episode_*.html"
echo ""
echo "🔍 预期效果："
echo "  ✅ 地形使用自然的灰绿色配色"
echo "  ✅ 显示独立的山峰结构"
echo "  ✅ 山峰之间有清晰的通道"
echo "  ✅ 随复杂度增加，山峰数量增多"
echo ""
echo "📖 详细说明："
echo "  cat TERRAIN_GENERATION_UPGRADE.md"
echo ""


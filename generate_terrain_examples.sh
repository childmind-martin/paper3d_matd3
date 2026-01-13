#!/bin/bash
#
# 快速生成常用地形地图示例
#
# 生成4个不同复杂度的地图，使用相同的种子（便于对比）

cd /home/tang/Desktop

echo "=========================================="
echo "🗺️  批量生成地形地图示例"
echo "=========================================="
echo ""

SEED=45  # 使用与训练环境相同的种子

echo "使用随机种子: $SEED"
echo ""

# 生成4个复杂度等级的地图
echo "📊 生成复杂度等级1（简单）..."
python3 visualize_terrain_map.py --complexity 1 --seed $SEED --output terrain_level1_simple.html
echo ""

echo "📊 生成复杂度等级2（中等）..."
python3 visualize_terrain_map.py --complexity 2 --seed $SEED --output terrain_level2_medium.html
echo ""

echo "📊 生成复杂度等级3（困难）..."
python3 visualize_terrain_map.py --complexity 3 --seed $SEED --output terrain_level3_hard.html
echo ""

echo "📊 生成复杂度等级4（极难）..."
python3 visualize_terrain_map.py --complexity 4 --seed $SEED --output terrain_level4_extreme.html
echo ""

echo "=========================================="
echo "✅ 完成！生成了4个地图："
echo "=========================================="
echo "  1. terrain_level1_simple.html   - 简单（5峰，4障碍）"
echo "  2. terrain_level2_medium.html   - 中等（6峰，8障碍）"
echo "  3. terrain_level3_hard.html     - 困难（7峰，12障碍）"
echo "  4. terrain_level4_extreme.html  - 极难（8峰，16障碍）"
echo ""
echo "在浏览器中打开这些文件即可查看！"
echo ""


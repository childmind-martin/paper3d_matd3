# 3D地形地图可视化工具

## 📋 功能介绍

这是一个独立的3D地形可视化工具，可以生成带障碍物的交互式地形地图HTML文件。

**特点**：
- ✅ 纯地形展示（不包含起点/终点）
- ✅ 支持4个复杂度等级
- ✅ 包含障碍物信息
- ✅ 交互式3D可视化（Plotly）
- ✅ 可设置随机种子（生成可复现地图）
- ✅ 支持自定义地图尺寸

---

## 🚀 快速开始

### 基本使用

```bash
# 生成默认地图（复杂度等级2）
python3 visualize_terrain_map.py

# 在浏览器中打开生成的 terrain_map.html
```

### 指定复杂度

```bash
# 简单地图（等级1：5个山峰，4个障碍物）
python3 visualize_terrain_map.py --complexity 1

# 中等地图（等级2：6个山峰，8个障碍物）- 默认
python3 visualize_terrain_map.py --complexity 2

# 困难地图（等级3：7个山峰，12个障碍物）
python3 visualize_terrain_map.py --complexity 3

# 极难地图（等级4：8个山峰，16个障碍物）
python3 visualize_terrain_map.py --complexity 4
```

### 使用固定种子（生成可复现地图）

```bash
# 使用种子45生成地图（与训练环境一致）
python3 visualize_terrain_map.py --seed 45 --complexity 2
```

### 自定义输出文件名

```bash
# 指定输出文件名
python3 visualize_terrain_map.py --output my_terrain.html

# 复杂示例：等级4，种子42，自定义文件名
python3 visualize_terrain_map.py -c 4 -s 42 -o extreme_terrain.html
```

### 自定义地图尺寸

```bash
# 生成300x300的大地图
python3 visualize_terrain_map.py --size 300 --complexity 3
```

---

## 📊 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--complexity` | `-c` | int | 2 | 复杂度等级（1-4） |
| `--size` | `-s` | int | 200 | 地图尺寸（200表示200x200） |
| `--seed` | - | int | None | 随机种子（用于生成可复现地图） |
| `--output` | `-o` | str | terrain_map.html | 输出HTML文件路径 |

---

## 🏔️ 复杂度等级详情

| 等级 | 山峰数 | 障碍物数 | 山峰高度范围 | 山峰宽度范围 | 噪声强度 | 适用场景 |
|------|--------|----------|--------------|--------------|----------|----------|
| 1 | 5 | 4 | 50-70m | 12-18 | 2.0 | 简单训练/测试 |
| 2 | 6 | 8 | 60-80m | 14-20 | 2.5 | 标准训练 |
| 3 | 7 | 12 | 70-90m | 16-22 | 3.0 | 高级训练 |
| 4 | 8 | 16 | 80-100m | 18-24 | 3.5 | 极限挑战 |

---

## 📖 使用示例

### 示例1：生成训练环境的地图

```bash
# 与训练环境使用相同的种子和复杂度
python3 visualize_terrain_map.py --seed 45 --complexity 2 --output training_terrain.html
```

### 示例2：对比不同复杂度

```bash
# 生成4个不同复杂度的地图
python3 visualize_terrain_map.py -c 1 -o terrain_easy.html
python3 visualize_terrain_map.py -c 2 -o terrain_medium.html
python3 visualize_terrain_map.py -c 3 -o terrain_hard.html
python3 visualize_terrain_map.py -c 4 -o terrain_extreme.html
```

### 示例3：生成系列地图（用于文档）

```bash
# 使用循环生成多个地图
for seed in 1 2 3 4 5; do
    python3 visualize_terrain_map.py --seed $seed --complexity 2 --output "terrain_seed_${seed}.html"
done
```

---

## 🎨 HTML输出特性

生成的HTML文件包含：

1. **交互式3D可视化**
   - 鼠标拖动旋转视角
   - 滚轮缩放
   - 悬停显示详细信息

2. **地形特性**
   - 颜色渐变表示高度
   - 等高线显示
   - 光照效果

3. **障碍物显示**
   - 红色标记
   - 悬停显示位置和半径
   - 图例说明

4. **统计信息**
   - 地图尺寸
   - 障碍物数量
   - 最高点高度
   - 平均高度

---

## 🔧 技术细节

### 地形生成算法

1. **山峰生成**：
   - 使用高斯分布创建独立尖峰
   - 保持山峰之间至少45单位距离
   - 避免山峰过于密集

2. **噪声添加**：
   - 低强度随机噪声（2.0-3.5）
   - 保持山峰之间的通道清晰
   - 避免过度平滑

3. **障碍物放置**：
   - 随机位置
   - 障碍物底部接触地面
   - 固定半径（7.0米）

### 性能优化

- 地形采样率：每4个点采样1个（降低数据量）
- 响应式设计：自动调整窗口大小
- 文件大小：通常< 100KB

---

## 📁 输出文件

生成的HTML文件特点：

- **独立文件**：包含所有必要的代码和数据
- **无需服务器**：直接在浏览器中打开
- **支持导出**：可导出为PNG图片（右上角工具栏）
- **跨平台**：Windows/Linux/Mac通用

---

## ⚠️ 注意事项

1. **浏览器要求**：
   - 推荐使用现代浏览器（Chrome, Firefox, Edge）
   - 需要启用JavaScript

2. **文件大小**：
   - 地图尺寸越大，文件越大
   - 建议尺寸≤300（更大可能导致浏览器卡顿）

3. **随机种子**：
   - 使用相同种子生成的地图完全相同
   - 适用于复现训练环境

4. **障碍物位置**：
   - 障碍物会自动放置在地面上
   - Z坐标 = 地形高度 + 障碍物半径

---

## 🆘 常见问题

### Q: 如何生成与训练环境完全相同的地图？

A: 使用相同的种子和复杂度：
```bash
python3 visualize_terrain_map.py --seed 45 --complexity 2
```

### Q: 生成的HTML文件很大怎么办？

A: 减小地图尺寸或降低复杂度：
```bash
python3 visualize_terrain_map.py --size 150 --complexity 1
```

### Q: 如何批量生成地图？

A: 使用shell循环：
```bash
for i in {1..10}; do
    python3 visualize_terrain_map.py --seed $i -o "map_$i.html"
done
```

### Q: 可以自定义颜色吗？

A: 可以编辑脚本中的colorscale数组（第199-206行）

---

## 📞 技术支持

如有问题或建议，请查看：
- `paper3d_train_optimized.py` - 完整训练代码
- `TRAINING_ISSUES_ANALYSIS.md` - 训练问题分析

---

**最后更新**：2025-11-19
**版本**：1.0


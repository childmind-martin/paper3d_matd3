#!/usr/bin/env python3
"""
独立的3D地形可视化工具

功能：
- 生成带障碍物的3D地形地图
- 支持可调节的复杂度等级（1-4）
- 输出交互式HTML文件
- 不包含起点/终点，纯地形展示

使用方法：
    python3 visualize_terrain_map.py --complexity 2 --output terrain.html
    python3 visualize_terrain_map.py --complexity 4 --seed 42 --size 200
"""

import numpy as np
import argparse
import os
import sys

def generate_terrain(map_size=200, complexity_level=2, seed=None):
    """
    生成地形高度图
    
    参数：
        map_size: 地图尺寸（默认200x200）
        complexity_level: 复杂度等级（1-4）
        seed: 随机种子（None表示随机）
    
    返回：
        height_map: 地形高度图 (map_size, map_size)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 根据复杂度等级设置参数
    complexity_config = {
        1: {'n_peaks': 5, 'height_range': (50, 70), 'width_range': (12, 18), 'noise': 2.0},
        2: {'n_peaks': 6, 'height_range': (60, 80), 'width_range': (14, 20), 'noise': 2.5},
        3: {'n_peaks': 7, 'height_range': (70, 90), 'width_range': (16, 22), 'noise': 3.0},
        4: {'n_peaks': 8, 'height_range': (80, 100), 'width_range': (18, 24), 'noise': 3.5},
    }
    
    config = complexity_config.get(complexity_level, complexity_config[2])
    
    # 初始化地形
    height_map = np.zeros((map_size, map_size), dtype=np.float32)
    
    # 生成山峰
    peak_positions = []
    min_distance = 45  # 山峰之间的最小距离
    
    for _ in range(config['n_peaks']):
        attempts = 0
        while attempts < 100:
            x = np.random.randint(20, map_size - 20)
            y = np.random.randint(20, map_size - 20)
            
            # 检查与已有山峰的距离
            too_close = False
            for px, py in peak_positions:
                if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                peak_positions.append((x, y))
                break
            attempts += 1
    
    # 为每个山峰添加高度
    for px, py in peak_positions:
        height = np.random.uniform(*config['height_range'])
        width = np.random.uniform(*config['width_range'])
        
        for i in range(map_size):
            for j in range(map_size):
                dist = np.sqrt((i - px)**2 + (j - py)**2)
                # 使用高斯分布创建尖峰
                contribution = height * np.exp(-(dist**2) / (2 * width**2))
                height_map[i, j] += contribution
    
    # 添加低强度噪声（保持山峰之间的通道清晰）
    noise = np.random.randn(map_size, map_size) * config['noise']
    height_map += noise
    
    # 确保非负
    height_map = np.maximum(height_map, 0)
    
    return height_map, peak_positions


def generate_obstacles(map_size=200, complexity_level=2, height_map=None, seed=None):
    """
    生成障碍物
    
    参数：
        map_size: 地图尺寸
        complexity_level: 复杂度等级（1-4）
        height_map: 地形高度图（用于确定障碍物Z坐标）
        seed: 随机种子
    
    返回：
        obstacles: [(x, y, z, radius), ...]
    """
    if seed is not None:
        np.random.seed(seed + 1000)
    
    # 根据复杂度确定障碍物数量
    n_obstacles_map = {1: 4, 2: 8, 3: 12, 4: 16}
    n_obstacles = n_obstacles_map.get(complexity_level, 8)
    
    obstacles = []
    radius = 7.0  # 固定半径
    
    for i in range(n_obstacles):
        # 随机位置
        x = np.random.randint(10, map_size - 10)
        y = np.random.randint(10, map_size - 10)
        
        # 获取该位置的地形高度
        if height_map is not None:
            terrain_h = height_map[int(x), int(y)]
            z = terrain_h + radius  # 障碍物底部接触地面
        else:
            z = radius
        
        obstacles.append((float(x), float(y), float(z), float(radius)))
    
    return obstacles


def generate_html_visualization(height_map, obstacles, output_file, map_size=200):
    """
    生成交互式3D地形HTML文件
    
    参数：
        height_map: 地形高度图
        obstacles: 障碍物列表
        output_file: 输出HTML文件路径
        map_size: 地图尺寸
    """
    # 采样地形数据（降低密度以提高性能）
    sample_rate = 4  # 每4个点采样1个
    x_samples = np.arange(0, map_size, sample_rate)
    y_samples = np.arange(0, map_size, sample_rate)
    
    # 生成网格数据
    terrain_data = []
    for x in x_samples:
        row = []
        for y in y_samples:
            z = height_map[int(x), int(y)]
            row.append(float(z))
        terrain_data.append(row)
    
    # 生成障碍物数据
    obstacles_data = []
    for x, y, z, r in obstacles:
        obstacles_data.append({
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'radius': float(r)
        })
    
    # HTML模板
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D地形地图 - 复杂度等级 {height_map.shape[0]//50}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }}
        #info {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        #plot {{
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .stat {{
            display: inline-block;
            margin-right: 20px;
        }}
        .label {{
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>3D地形地图可视化</h2>
        <div class="stat">
            <span class="label">地图尺寸:</span> {map_size}×{map_size}
        </div>
        <div class="stat">
            <span class="label">障碍物数量:</span> {len(obstacles)}
        </div>
        <div class="stat">
            <span class="label">最高点:</span> {height_map.max():.1f}m
        </div>
        <div class="stat">
            <span class="label">平均高度:</span> {height_map.mean():.1f}m
        </div>
    </div>
    <div id="plot"></div>
    
    <script>
        // 地形数据
        var terrainData = {terrain_data};
        var xSamples = {list(x_samples)};
        var ySamples = {list(y_samples)};
        
        // 障碍物数据
        var obstacles = {obstacles_data};
        
        // 创建地形表面
        var terrainTrace = {{
            type: 'surface',
            x: xSamples,
            y: ySamples,
            z: terrainData,
            colorscale: [
                [0, 'rgb(220, 220, 180)'],      // 低海拔：浅黄色
                [0.3, 'rgb(180, 200, 120)'],    // 中低：浅绿色
                [0.5, 'rgb(120, 160, 100)'],    // 中等：绿色
                [0.7, 'rgb(100, 120, 80)'],     // 中高：深绿色
                [0.85, 'rgb(139, 137, 137)'],   // 高：灰色
                [1, 'rgb(255, 255, 255)']       // 极高：白色（雪）
            ],
            name: '地形',
            showscale: true,
            colorbar: {{
                title: '高度 (m)',
                titleside: 'right'
            }},
            lighting: {{
                ambient: 0.6,
                diffuse: 0.8,
                specular: 0.2,
                roughness: 0.5
            }},
            contours: {{
                z: {{
                    show: true,
                    usecolormap: true,
                    highlightcolor: "limegreen",
                    project: {{z: false}}
                }}
            }}
        }};
        
        var data = [terrainTrace];
        
        // 添加障碍物（使用散点图表示）
        if (obstacles.length > 0) {{
            var obstacleX = [];
            var obstacleY = [];
            var obstacleZ = [];
            var obstacleText = [];
            
            obstacles.forEach(function(obs, idx) {{
                obstacleX.push(obs.x);
                obstacleY.push(obs.y);
                obstacleZ.push(obs.z);
                obstacleText.push(`障碍物 ${{idx+1}}<br>位置: (${{obs.x.toFixed(1)}}, ${{obs.y.toFixed(1)}}, ${{obs.z.toFixed(1)}})<br>半径: ${{obs.radius.toFixed(1)}}m`);
            }});
            
            var obstacleTrace = {{
                type: 'scatter3d',
                mode: 'markers',
                x: obstacleX,
                y: obstacleY,
                z: obstacleZ,
                marker: {{
                    size: 8,
                    color: 'red',
                    symbol: 'circle',
                    line: {{
                        color: 'darkred',
                        width: 2
                    }}
                }},
                text: obstacleText,
                hoverinfo: 'text',
                name: '障碍物'
            }};
            
            data.push(obstacleTrace);
        }}
        
        // 布局配置
        var layout = {{
            title: {{
                text: '3D地形地图',
                font: {{size: 24}}
            }},
            scene: {{
                xaxis: {{title: 'X (m)', range: [0, {map_size}]}},
                yaxis: {{title: 'Y (m)', range: [0, {map_size}]}},
                zaxis: {{title: 'Z - 高度 (m)', range: [0, {height_map.max() * 1.2:.1f}]}},
                camera: {{
                    eye: {{x: 1.5, y: 1.5, z: 1.2}},
                    center: {{x: 0, y: 0, z: -0.1}}
                }},
                aspectmode: 'manual',
                aspectratio: {{x: 1, y: 1, z: 0.5}}
            }},
            autosize: true,
            width: 1200,
            height: 800,
            margin: {{l: 0, r: 0, b: 0, t: 50}},
            hovermode: 'closest',
            showlegend: true,
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255, 255, 255, 0.8)',
                bordercolor: 'black',
                borderwidth: 1
            }}
        }};
        
        // 配置选项
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['select2d', 'lasso2d'],
            toImageButtonOptions: {{
                format: 'png',
                filename: 'terrain_map',
                height: 1080,
                width: 1920,
                scale: 2
            }}
        }};
        
        // 绘制图形
        Plotly.newPlot('plot', data, layout, config);
        
        // 响应式调整
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('plot');
        }});
    </script>
</body>
</html>"""
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 地图已保存到: {output_file}")
    print(f"   文件大小: {os.path.getsize(output_file) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='生成3D地形地图可视化HTML文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 生成默认复杂度（等级2）的地图
  python3 visualize_terrain_map.py
  
  # 生成简单地图（等级1）
  python3 visualize_terrain_map.py --complexity 1 --output simple_terrain.html
  
  # 生成复杂地图（等级4），使用固定种子
  python3 visualize_terrain_map.py --complexity 4 --seed 42 --output complex_terrain.html
  
  # 生成大地图（300x300）
  python3 visualize_terrain_map.py --size 300 --complexity 3
        """
    )
    
    parser.add_argument(
        '--complexity', '-c',
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help='地形复杂度等级 (1=简单, 2=中等, 3=困难, 4=极难，默认: 2)'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=200,
        help='地图尺寸 (默认: 200x200)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（用于生成可复现的地图，默认: 随机）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='terrain_map.html',
        help='输出HTML文件路径 (默认: terrain_map.html)'
    )
    
    args = parser.parse_args()
    
    # 打印配置
    print("=" * 60)
    print("🗺️  3D地形地图生成器")
    print("=" * 60)
    print(f"地图尺寸: {args.size}×{args.size}")
    print(f"复杂度等级: {args.complexity} ", end='')
    complexity_names = {1: '(简单)', 2: '(中等)', 3: '(困难)', 4: '(极难)'}
    print(complexity_names[args.complexity])
    print(f"随机种子: {args.seed if args.seed is not None else '随机'}")
    print(f"输出文件: {args.output}")
    print()
    
    # 生成地形
    print("🏔️  生成地形...")
    height_map, peak_positions = generate_terrain(
        map_size=args.size,
        complexity_level=args.complexity,
        seed=args.seed
    )
    print(f"   山峰数量: {len(peak_positions)}")
    print(f"   最高点: {height_map.max():.1f}m")
    print(f"   平均高度: {height_map.mean():.1f}m")
    print()
    
    # 生成障碍物
    print("🚧 生成障碍物...")
    obstacles = generate_obstacles(
        map_size=args.size,
        complexity_level=args.complexity,
        height_map=height_map,
        seed=args.seed
    )
    print(f"   障碍物数量: {len(obstacles)}")
    
    # 打印障碍物详细信息
    for i, (x, y, z, r) in enumerate(obstacles, 1):
        terrain_h = height_map[int(x), int(y)]
        print(f"   障碍物 {i}: 位置=({x:.1f}, {y:.1f}, {z:.1f}), "
              f"半径={r:.1f}, 地形高度={terrain_h:.1f}")
    print()
    
    # 生成HTML
    print("📊 生成HTML可视化...")
    generate_html_visualization(
        height_map=height_map,
        obstacles=obstacles,
        output_file=args.output,
        map_size=args.size
    )
    print()
    
    print("=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"在浏览器中打开 {args.output} 查看地图")
    print()


if __name__ == '__main__':
    main()


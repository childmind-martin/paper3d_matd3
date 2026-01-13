# paper3d_test.py
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，确保在Windows上正常工作
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D支持
from matplotlib.lines import Line2D

# 导入TensorFlow
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev  # 添加样条插值所需的导入
from agents.maddpg_runner import MADDPGRunner
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import importlib
import importlib.util  # 添加importlib.util支持
import pygame
import matplotlib as mpl
import time
import signal
# 移除ActionFixer导入
from matplotlib import animation
from matplotlib.lines import Line2D  # 添加Line2D导入
import random
import traceback  # 添加traceback模块导入
from tqdm import tqdm
import io  # 用于内存中的图像处理
from PIL import Image  # 用于处理图像和保存GIF
import threading
import datetime

# 导入MADDPG类 - 支持从新模块或旧模块导入
try:
    from maddpg_agent import MADDPG
    print("成功从maddpg_agent模块导入MADDPG类")
except ImportError:
    try:
        from agents.maddpg import MADDPG
        print("从agents.maddpg模块导入MADDPG类")
    except ImportError:
        print("警告: 无法导入MADDPG类，将依赖MADDPGRunner")

# 导入力场校正器
try:
    from potential_field_corrector import ContinuousPotentialFieldCorrector
except ImportError:
    print("警告: 无法导入ContinuousPotentialFieldCorrector，将禁用动作校正功能")
    ContinuousPotentialFieldCorrector = None

# 调试输出控制开关 - 设为False以禁用大多数调试输出
DEBUG_OUTPUT = False

# 配置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    import matplotlib
    import matplotlib.font_manager as fm
    import platform
    import os
    
    system = platform.system()
    print(f"当前操作系统: {system}")
    
    # 定义需要尝试的字体列表（按优先级排序）
    font_found = False
    
    if system == 'Windows':
        # Windows字体列表
        font_list = [
            'SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS',
            'SimHei Regular', 'Microsoft YaHei UI', 'NSimSun', 'DengXian', 'Source Han Sans CN',
            'Microsoft JhengHei', 'Yu Gothic', 'Meiryo'
        ]
        
        # 移除尝试添加整个字体目录的代码，该操作可能导致权限问题
        # 直接通过rcParams设置字体
        
    elif system == 'Linux':
        font_list = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC',
            'AR PL UKai CN', 'AR PL UMing CN', 'AR PL SungtiL GB', 'Ubuntu'
        ]
    elif system == 'Darwin':  # macOS
        font_list = [
            'PingFang SC', 'Heiti SC', 'STHeiti', 'STSong', 'STFangsong',
            'Hiragino Sans GB', 'Apple LiGothic Medium', 'Apple LiSung Light', 'Arial Unicode MS'
        ]
    else:
        font_list = ['DejaVu Sans', 'Arial Unicode MS', 'FreeSans']
    
    # 尝试多种方法设置字体
    # 1. 直接设置rcParams
    for font in font_list:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams.get('font.sans-serif', [])
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            # 简单测试
            import matplotlib.pyplot as plt
            test_fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试', ha='center', va='center')
            plt.close(test_fig)
            
            font_found = True
            print(f"成功设置中文字体: {font}")
            break
        except Exception as e:
            if debug_mode:
                print(f"字体 {font} 设置失败: {e}")
            continue
    
    # 2. 如果第一种方法失败，尝试查找系统中已安装的字体
    if not font_found:
        try:
            # 获取系统中所有可用字体
            font_paths = fm.findSystemFonts(fontpaths=None)
            # 查找包含中文的字体
            chinese_fonts = []
            for path in font_paths:
                try:
                    if any(keyword in os.path.basename(path).lower() for keyword in 
                           ['simhei', 'yahei', 'simsun', 'kaiti', 'fangsong', 'heiti']):
                        chinese_fonts.append(path)
                except:
                    continue
            
            if chinese_fonts:
                # 使用找到的第一个中文字体
                prop = fm.FontProperties(fname=chinese_fonts[0])
                matplotlib.rcParams['font.sans-serif'] = ['sans-serif']  # 重置sans-serif
                font_found = True
                print(f"找到中文字体: {os.path.basename(chinese_fonts[0])}")
        except Exception as e:
            print(f"查找系统字体失败: {e}")
    
    # 3. 如果前两种方法都失败，使用配置文件方式
    if not font_found:
        try:
            # 尝试在配置文件中设置
            config_dir = matplotlib.get_configdir()
            config_file = os.path.join(config_dir, 'matplotlibrc')
            
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            with open(config_file, 'a') as f:
                f.write("\nfont.family : sans-serif")
                f.write("\nfont.sans-serif : " + ", ".join(font_list))
                f.write("\naxes.unicode_minus : False")
                
            # 重新加载配置
            matplotlib.rc_file(config_file)
            font_found = True
            print(f"通过配置文件设置中文字体")
        except Exception as e:
            print(f"通过配置文件设置字体失败: {e}")
    
    # 4. 如果前三种方法都失败，使用全局表达方式
    if not font_found:
        print("警告: 未找到适合的中文字体，将使用默认字体")
        # 设置为默认字体
        matplotlib.rcParams['font.sans-serif'] = ['sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
        font_found = "default"
        
    return font_found

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser("MADDPG 3D测试")
    
    # 环境相关参数
    parser.add_argument("--scenario", type=str, default="paper3d_terrain", help="环境场景名称")
    parser.add_argument("--episode-length", type=int, default=1000, help="每个episode的最大步数")
    parser.add_argument("--num-episodes", type=int, default=1, help="测试回合数")
    
    # 可视化相关参数
    parser.add_argument("--display", action="store_true", default=True, help="使用pygame可视化")
    parser.add_argument("--save-gif", action="store_true", default=True, help="是否保存为GIF图")
    parser.add_argument("--save-png", action="store_true", default=True, help="是否保存为PNG图")
    parser.add_argument("--render", action="store_true", default=False, help="是否渲染环境")
    
    # GPU相关配置
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="是否禁用GPU")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="是否使用GPU训练")

    # 地形相关参数
    parser.add_argument("--random-terrain", action="store_true", default=False, help="是否使用随机地形")
    parser.add_argument("--terrain-seed", type=int, default=42, help="地形生成的随机种子")
    
    # 固定位置相关参数
    parser.add_argument("--use-fixed-positions", action="store_true", default=True, help="是否使用固定初始位置")
    parser.add_argument("--dynamic-first-time", action="store_true", default=True, help="首次运行是否动态生成初始位置")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/fixed_positions.json", help="固定位置的保存/加载文件路径")
    parser.add_argument("--random-z0-positions", action="store_true", default=True, help="是否随机初始化Z轴位置")
    
    # 策略相关参数
    parser.add_argument("--policy", type=str, default="best_average", 
                        choices=["best_average", "best_overall", "last"], 
                        help="使用的策略类型：best_average(平均最佳)、best_overall(整体最佳)、last(最新)")
    parser.add_argument("--model-dir", type=str, default=None, help="权重保存的目录")
    parser.add_argument("--model-suffix", type=str, default="", help="模型文件后缀")
    parser.add_argument("--load-dir", type=str, default="", help="加载权重的目录")
    
    # 可视化参数
    parser.add_argument("--no-visualization", action="store_true", default=False, help="禁用3D可视化")
    parser.add_argument("--save-viz", action="store_true", default=True, help="保存可视化结果")
    parser.add_argument("--save-path", type=str, default=None, help="可视化结果保存路径")
    parser.add_argument("--record-animation", action="store_true", default=True, help="记录GIF动画")
    parser.add_argument("--animation-file", type=str, default=None, help="动画文件保存路径")
    parser.add_argument("--animation-fps", type=int, default=10, help="动画FPS")
    
    # 动作校正设置
    parser.add_argument("--enable-action-correction", action="store_true", default=True, help="启用力场动作校正")
    parser.add_argument("--correction-type", type=str, default="combined",
                       choices=["terrain_avoidance", "target_guidance", "combined"],
                       help="校正类型：地形避让/目标引导/组合")
    parser.add_argument("--correction-force-ratio", type=float, default=0.3, help="校正力度比例")
    
    # 添加与轨迹处理相关的参数
    parser.add_argument("--load-trajectory", type=str, default=None, help="加载指定的轨迹文件(npy格式)，仅分析不运行测试")
    parser.add_argument("--save-trajectory-png", type=str, default=None, help="保存轨迹分析结果为PNG图像的路径")
    parser.add_argument("--save-trajectory-npy", type=str, default=None, help="指定轨迹数据保存路径(npy格式)")
    parser.add_argument("--process-all-policies", action="store_true", default=False, help="处理所有策略的轨迹(best_average, best_overall, last)")
    parser.add_argument("--vertical-force-suppress", type=float, default=0.2, help="垂直方向力抑制系数")
    parser.add_argument("--min-clearance-factor", type=float, default=1.0, help="最小间隙系数")
    parser.add_argument("--safety-distance-factor", type=float, default=1.5, help="安全距离系数")
    
    # 添加适配训练模型的参数
    parser.add_argument("--expected-obs-dim", type=int, default=None, help="期望的观察空间维度(与训练模型匹配)")
    parser.add_argument("--expected-action-dim", type=int, default=3, help="期望的动作空间维度(与训练模型匹配)")
    parser.add_argument("--hidden-units", type=str, default="384,256,128", 
                       help="Actor网络隐藏层单元数(与训练模型匹配), 逗号分隔")
    parser.add_argument("--critic-hidden-units", type=str, default="128,64,32", 
                       help="Critic网络隐藏层单元数(与训练模型匹配), 逗号分隔")
    parser.add_argument("--continuous-action-space", action="store_true", default=True, 
                       help="是否使用连续动作空间(与训练模型匹配)")
    parser.add_argument("--disable-params-check", action="store_true", default=False, 
                       help="禁用训练参数兼容性检查")
    
    # 添加更多力场参数（兼容训练时使用的参数）
    parser.add_argument("--goal-attraction-weight", type=float, default=1.0, 
                       help="目标吸引力场权重")
    parser.add_argument("--terrain-repulsion-weight", type=float, default=1.0, 
                       help="地形排斥力场权重")
    parser.add_argument("--agent-repulsion-weight", type=float, default=0.5, 
                       help="智能体互斥力场权重")
    parser.add_argument("--influence-range", type=float, default=10.0, 
                       help="力场影响范围")
    parser.add_argument("--min-clearance", type=float, default=2.0, 
                       help="最小地形间隙")
    parser.add_argument("--max-force-magnitude", type=float, default=10.0, 
                       help="最大力场幅值")
    parser.add_argument("--debug-corrector", action="store_true", default=False, 
                       help="启用力场校正器调试模式")
    parser.add_argument("--detection-radius", type=float, default=5.0, 
                       help="地形检测半径")
    parser.add_argument("--detection-height-range", type=float, default=8.0, 
                       help="高度检测范围")
    parser.add_argument("--agent-detection-radius", type=float, default=10.0, 
                       help="智能体检测半径")
    parser.add_argument("--check-count", type=int, default=8, 
                       help="检测点数量")
    parser.add_argument("--check-spacing", type=float, default=2.0, 
                       help="检测点间隔")
    parser.add_argument("--gravity", type=float, default=0.0, help="重力加速度（作用于 -Z 方向）")
    parser.add_argument("--use-dynamic-force-params", action="store_true", default=True,
                       help="使用模型生成的动态力场参数")
    parser.add_argument("--force-param-ratio", type=float, default=0.5,
                       help="力场参数变化比例系数")
    
    return parser.parse_args()

def init_pygame():
    """初始化Pygame，用于处理键盘和鼠标事件"""
    try:
        import pygame
        pygame.init()
        pygame.display.set_mode((100, 100), pygame.RESIZABLE)
        pygame.display.set_caption("3D Terrain Control")
        print("Pygame初始化成功!")
        return True
    except Exception as e:
        print(f"Pygame初始化失败: {e}")
        return False

# 添加更多全局变量用于控制和调试
global_elev = 30  # 初始俯仰角
global_azim = 135  # 初始方位角
show_controls = True  # 是否显示控制提示
debug_mode = True  # 调试模式
fig_initialized = False
viz_fig = None
viz_ax = None
current_colorbar = None

# 添加全局变量用于存储艺术家对象
global terrain_surface
global target_markers
global trajectory_lines
global agent_markers
global projection_lines
global projection_points
global projection_trajectories  # 存储投影轨迹线
global text_annotations
global control_text  # 控制提示文本
global scatter_agents  # 确保声明此全局变量
global scatter_landmarks  # 确保声明此全局变量

# 添加全局变量用于存储轨迹数据
trajectory_history = []  # 主轨迹历史
projection_history = []  # 投影轨迹历史

# 添加全局变量用于存储GIF动画帧
animation_frames = []
record_animation = False  # 是否记录动画

# 全局变量初始化
last_figure_update_time = 0  # 初始化图形更新时间

# 定义更新控制文本的函数
def update_control_text(ax):
    """更新控制说明文本
    
    Args:
        ax: matplotlib图表对象，可能是Axes或Axes3D
    """
    try:
        # 检查ax是否是有效的图表对象
        if ax is None:
            return
            
        # 清除之前的文本信息
        if hasattr(ax, '_control_text') and ax._control_text:
            for text in ax._control_text:
                if text and hasattr(text, 'remove'):
                    text.remove()
            ax._control_text = []
        else:
            ax._control_text = []
            
        # 添加控制说明文本
        control_text = [
            "控制说明:",
            "- 鼠标左键拖动: 旋转视角",
            "- 鼠标右键拖动: 平移视角",
            "- 鼠标滚轮: 缩放视角"
        ]
        
        # 优先使用text2D方法(Axes3D特有)，或者fallback到text方法(普通Axes)
        for i, text in enumerate(control_text):
            try:
                if hasattr(ax, 'text2D'):
                    txt = ax.text2D(0.02, 0.98 - i*0.03, text, transform=ax.transAxes, 
                             color='white', fontsize=9, backgroundcolor='black', alpha=0.7)
                else:
                    # 普通Axes对象使用text方法
                    txt = ax.text(0.02, 0.98 - i*0.03, text, transform=ax.transAxes, 
                           color='white', fontsize=9, backgroundcolor='black', alpha=0.7)
                ax._control_text.append(txt)
            except Exception as e:
                print(f"添加控制文本时出错: {e}, 文本: {text}")
            
    except Exception as e:
        print(f"更新控制文本时出错: {e}")

def visualization_callback(env, maddpg, scenario, step, episode, max_steps, max_episodes, trajectories, episode_reward, args=None):
    """3D可视化回调函数，用于显示3D环境状态"""
    # 直接检查args参数，如果禁用可视化则立即返回
    if args is not None and hasattr(args, 'disable_visualization') and args.disable_visualization:
        return
        
    # 全局变量声明
    global control_text, last_figure_update_time, trajectory_lines, scatter_agents, scatter_landmarks
    global terrain_surface, viz_ax, viz_fig, target_markers, text_annotations, trajectory_history, projection_history
    global agent_markers, projection_trajectories, projection_lines, projection_points, DEBUG_OUTPUT
    
    # 设置调试模式（可在测试时开启）
    DEBUG_OUTPUT = False  # 调试输出开关
    
    # 确保全局变量在第一次调用时被初始化
    if 'scatter_agents' not in globals():
        global scatter_agents
        scatter_agents = None
        
    if 'scatter_landmarks' not in globals():
        global scatter_landmarks
        scatter_landmarks = None
        
    if 'terrain_surface' not in globals():
        global terrain_surface
        terrain_surface = None
        
    if 'target_markers' not in globals():
        global target_markers
        target_markers = []
        
    if 'trajectory_lines' not in globals():
        global trajectory_lines
        trajectory_lines = []
        
    if 'agent_markers' not in globals():
        global agent_markers
        agent_markers = []
        
    if 'projection_lines' not in globals():
        global projection_lines
        projection_lines = []
        
    if 'projection_points' not in globals():
        global projection_points
        projection_points = []
        
    if 'projection_trajectories' not in globals():
        global projection_trajectories
        projection_trajectories = []
        
    if 'text_annotations' not in globals():
        global text_annotations
        text_annotations = []
        
    if 'trajectory_history' not in globals():
        global trajectory_history
        trajectory_history = []
        
    if 'projection_history' not in globals():
        global projection_history
        projection_history = []
    
    # 首次调用时初始化全局变量
    if 'control_text' not in globals():
        global control_text
        control_text = None
    
    if 'last_figure_update_time' not in globals():
        global last_figure_update_time
        last_figure_update_time = 0
        
    # 性能优化 - 限制更新频率
    current_time = time.time()
    if current_time - last_figure_update_time < 0.1:  # 限制帧率约为10fps
        return True
    last_figure_update_time = current_time
    
    # 获取代理位置
    try:
        positions = []
        velocities = []
        
        for i, agent in enumerate(env.world.agents):
            pos = agent.state.p_pos
            vel = agent.state.p_vel
            positions.append(pos)
            velocities.append(vel)
            
        # 获取地标位置
        landmarks = []
        if hasattr(env.world, 'landmarks'):
            landmarks = [landmark.state.p_pos for landmark in env.world.landmarks]
        
        # 设置颜色
        agent_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        landmark_colors = ['gray']
        
        # 获取目标位置（如果有）
        goal_pos = None
        if hasattr(scenario, 'goal') and scenario.goal is not None:
            goal_pos = scenario.goal.state.p_pos
        
        # 确保图形存在
        if not plt.fignum_exists(1):
            viz_fig = plt.figure(figsize=(10, 8))
            viz_ax = plt.axes(projection='3d')
            # 修改坐标轴范围，适应实际地形范围
            viz_ax.set_xlim(0, 100)
            viz_ax.set_ylim(0, 100)
            viz_ax.set_zlim(0, 100)
            viz_ax.set_xlabel('X轴')
            viz_ax.set_ylabel('Y轴')
            viz_ax.set_zlabel('Z轴')
            viz_ax.set_title(f'3D环境状态 - 回合 {episode}/{max_episodes}, 步骤 {step}/{max_steps}')
        else:
            plt.figure(1)
            viz_ax = plt.gca()
            viz_ax.set_title(f'3D环境状态 - 回合 {episode}/{max_episodes}, 步骤 {step}/{max_steps}')
            
            # 确保坐标轴范围适应地形
            viz_ax.set_xlim(0, 100)
            viz_ax.set_ylim(0, 100)
            viz_ax.set_zlim(0, 100)
        
        # 清除之前的轨迹点
        while len(trajectory_lines) < len(positions):
            line, = viz_ax.plot([], [], [], '-', alpha=0.5, color=agent_colors[len(trajectory_lines) % len(agent_colors)])
            trajectory_lines.append(line)
        
        # 更新轨迹
        for i, (line, trajectory) in enumerate(zip(trajectory_lines, trajectories)):
            if i < len(positions) and trajectory:
                xs = [pos[0] for pos in trajectory]
                ys = [pos[1] for pos in trajectory]
                zs = [pos[2] for pos in trajectory]
                line.set_data(xs, ys)
                line.set_3d_properties(zs)
        
        # 更新智能体散点图
        if scatter_agents:
            scatter_agents.remove()
        
        agent_xs = [pos[0] for pos in positions]
        agent_ys = [pos[1] for pos in positions]
        agent_zs = [pos[2] for pos in positions]
        
        # 避免参数重复
        scatter_agents = viz_ax.scatter(
            agent_xs, agent_ys, agent_zs, 
            c=agent_colors[:len(positions)], 
            marker='o',
            s=100  # 单独设置大小参数
        )
        
        # 更新地标散点图
        if landmarks and len(landmarks) > 0:
            if scatter_landmarks:
                scatter_landmarks.remove()
            
            landmark_xs = [pos[0] for pos in landmarks]
            landmark_ys = [pos[1] for pos in landmarks]
            landmark_zs = [pos[2] for pos in landmarks]
            
            # 避免参数重复
            scatter_landmarks = viz_ax.scatter(
                landmark_xs, landmark_ys, landmark_zs, 
                c=landmark_colors * len(landmarks),
                marker='s',
                s=50  # 单独设置大小参数
            )
        
        # 显示目标（如果有）
        if goal_pos is not None:
            viz_ax.scatter(
                [goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                c='gold', 
                marker='*',
                s=200  # 单独设置大小参数
            )
    
    except Exception as e:
        print(f"可视化更新失败: {e}")
        return True
    
    # 一定间隔更新标题信息
    if viz_fig is not None:
        # 更新标题信息
        if episode_reward is not None:
            reward_text = f"奖励: {episode_reward:.2f}" if isinstance(episode_reward, float) else f"奖励: {episode_reward}"
        else:
            reward_text = "奖励: N/A"
            
        viz_fig.suptitle(f"Episode: {episode+1}/{max_episodes}, Step: {step+1}/{max_steps}, {reward_text}",
                     fontsize=12)
    
    # 仅在第一次绘制地形
    if terrain_surface is None and viz_ax is not None and scenario is not None and hasattr(scenario, 'terrain') and scenario.terrain is not None:
        try:
            # 获取地形数据
            terrain = scenario.terrain
            print(f"地形数据形状: {terrain.shape}, 最大高度: {np.max(terrain)}, 最小高度: {np.min(terrain)}")
            
            # 手动创建网格点
            x = np.arange(0, terrain.shape[1])
            y = np.arange(0, terrain.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # 使用采样率减少地形点数，提高渲染性能
            sample_rate = 2  # 减小采样率以增加点数
            X_sampled = X[::sample_rate, ::sample_rate]
            Y_sampled = Y[::sample_rate, ::sample_rate]
            terrain_sampled = terrain[::sample_rate, ::sample_rate]
            
            # 绘制地形 - 增强可见性
            terrain_surface = viz_ax.plot_surface(
                X_sampled, Y_sampled, terrain_sampled, 
                cmap='terrain', 
                alpha=0.7,  # 使用较低的不透明度
                linewidth=0,  # 移除网格线
                antialiased=True,
                rstride=1, cstride=1,
                shade=True,  # 保留阴影效果
                edgecolor=None  # 移除边缘颜色
            )
            print("地形绘制成功!")
        except Exception as e:
            print(f"[可视化] 地形绘制错误: {e}")
    
    # 更新或创建目标点标记
    if viz_ax is not None and scenario is not None and hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
        try:
            goal_pos = scenario.goal_pos
            
            # 如果目标标记不存在，创建它
            if not target_markers:
                # 创建目标点标记
                target = viz_ax.scatter(
                    [goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                    color='yellow', 
                    marker='*', 
                    s=200, 
                    edgecolors='red', 
                    linewidth=2, 
                    label='Target'
                )
                target_markers.append(target)
                
                # 添加地面到目标点的连接线
                if hasattr(scenario, 'get_terrain_height') or hasattr(scenario, 'get_height_at') or hasattr(scenario, 'terrain'):
                    try:
                        # 优先使用最可靠的方法获取地形高度
                        terrain_height = 0
                        if hasattr(scenario, 'get_height_at'):
                            terrain_height = scenario.get_height_at(goal_pos[0], goal_pos[1])
                        elif hasattr(scenario, 'get_terrain_height'):
                            terrain_height = scenario.get_terrain_height(goal_pos[0], goal_pos[1])
                        elif hasattr(scenario, 'terrain') and scenario.terrain is not None:
                            # 直接从地形数据获取高度
                            x = int(max(0, min(goal_pos[0], scenario.terrain.shape[1]-1)))
                            y = int(max(0, min(goal_pos[1], scenario.terrain.shape[0]-1)))
                            terrain_height = scenario.terrain[y, x]
                        
                        # 确保地形高度是有效的数值
                        if terrain_height is None or np.isnan(terrain_height):
                            print(f"警告: 目标点 ({goal_pos[0]}, {goal_pos[1]}) 处的地形高度无效")
                            # 使用地图上的平均高度作为默认值
                            if hasattr(scenario, 'terrain') and scenario.terrain is not None:
                                terrain_height = np.mean(scenario.terrain)
                            else:
                                terrain_height = 0
                        
                        print(f"目标点 ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}) 处的地形高度: {terrain_height:.1f}")
                        
                        # 绘制从地形到目标点的连接线 - 使用粗一点的线增强可见性
                        target_line = viz_ax.plot([goal_pos[0], goal_pos[0]], 
                                              [goal_pos[1], goal_pos[1]],
                                              [terrain_height, goal_pos[2]], 
                                              'r--', linewidth=2.5)[0]
                        target_markers.append(target_line)
                        
                        # 在地形上添加一个额外的标记点以增强可见性
                        terrain_marker = viz_ax.scatter(
                            [goal_pos[0]], [goal_pos[1]], [terrain_height], 
                            color='red', marker='o', s=50
                        )
                        target_markers.append(terrain_marker)
                    except Exception as e:
                        print(f"绘制目标点投影线失败: {e}")
                
                # 添加目标位置标签
                target_text = viz_ax.text(
                    goal_pos[0], goal_pos[1], goal_pos[2]+2, 
                    f'目标 ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f})', 
                    color='red', 
                    fontsize=10
                )
                text_annotations.append(target_text)
        except Exception as e:
            print(f"[可视化] 目标标记错误: {e}")
    
    # 确保列表长度匹配
    while len(trajectory_history) < len(trajectories):
        trajectory_history.append([])
    while len(projection_history) < len(trajectories):
        projection_history.append([])
    
    # 更新轨迹数据
    if viz_ax is None:
        return False
    
    # 更新或创建智能体轨迹和标记
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # 创建足够的轨迹线对象
    while len(trajectory_lines) < len(trajectories):
        line, = viz_ax.plot([], [], [], '-', linewidth=2, 
                        color=colors[len(trajectory_lines) % len(colors)],
                        label=f'agent_{len(trajectory_lines)}')
        trajectory_lines.append(line)
    
    # 创建足够的投影轨迹线对象
    while len(projection_trajectories) < len(trajectories):
        proj_traj, = viz_ax.plot([], [], [], '--', linewidth=1, 
                             color=colors[len(projection_trajectories) % len(colors)],
                             alpha=0.6)
        projection_trajectories.append(proj_traj)
    
    # 创建足够的智能体标记
    while len(agent_markers) < len(trajectories):
        agent_dot = viz_ax.scatter(
            [], [], [], 
            color=colors[len(agent_markers) % len(colors)], 
            s=100, 
            marker='o'
        )
        agent_markers.append(agent_dot)
    
    # 创建足够的投影线
    while len(projection_lines) < len(trajectories):
        proj_line, = viz_ax.plot([], [], [], '--', 
                             color=colors[len(projection_lines) % len(colors)], 
                             alpha=0.7, linewidth=1.5)
        projection_lines.append(proj_line)
    
    # 创建足够的投影点
    while len(projection_points) < len(trajectories):
        proj_point = viz_ax.scatter(
            [], [], [], 
            color=colors[len(projection_points) % len(colors)],
            s=50, 
            marker='x', 
            alpha=0.8
        )
        projection_points.append(proj_point)
    
    # 更新轨迹和投影轨迹数据
    for i, traj in enumerate(trajectories):
        if traj and len(traj) > 1:
            # 过滤掉None值
            valid_points = [p for p in traj if p is not None]
            if len(valid_points) > 0:
                # 更新主轨迹历史
                if len(valid_points) > 0:
                    last_point = valid_points[-1]
                    if len(trajectory_history[i]) == 0 or not np.array_equal(trajectory_history[i][-1], last_point):
                        trajectory_history[i].append(last_point)
                        
                        # 调试输出
                        if DEBUG_OUTPUT:
                            print(f"轨迹{i}更新: {last_point}, 历史长度: {len(trajectory_history[i])}")
                
                # 创建投影点并更新投影轨迹历史
                if hasattr(scenario, 'get_terrain_height') and len(valid_points) > 0:
                    try:
                        last_pos = valid_points[-1]
                        # 确保使用正确的方法获取地形高度
                        terrain_height = 0
                        if hasattr(scenario, 'get_height_at'):
                            terrain_height = scenario.get_height_at(last_pos[0], last_pos[1])
                        elif hasattr(scenario, 'get_terrain_height'):
                            terrain_height = scenario.get_terrain_height(last_pos[0], last_pos[1])
                        elif hasattr(scenario, 'terrain') and scenario.terrain is not None:
                            # 直接从地形数据获取高度
                            x = int(max(0, min(last_pos[0], scenario.terrain.shape[1]-1)))
                            y = int(max(0, min(last_pos[1], scenario.terrain.shape[0]-1)))
                            terrain_height = scenario.terrain[y, x]
                            
                        # 确保地形高度是有效值
                        if terrain_height is None or np.isnan(terrain_height):
                            if hasattr(scenario, 'terrain') and scenario.terrain is not None:
                                terrain_height = np.mean(scenario.terrain)
                            else:
                                terrain_height = 0
                                
                        # 创建投影点 - 使用确定的地形高度
                        projection_point = [last_pos[0], last_pos[1], terrain_height]
                        
                        # 只有当投影点是新的时才添加到历史
                        if len(projection_history[i]) == 0 or not np.array_equal(projection_history[i][-1], projection_point):
                            projection_history[i].append(projection_point)
                            
                            # 调试输出
                            if DEBUG_OUTPUT:
                                print(f"投影{i}更新: {projection_point}, 历史长度: {len(projection_history[i])}")
                    except Exception as e:
                        print(f"[调试] 创建投影点失败: {e}")
                        pass
                
                try:
                    # 更新轨迹
                    if len(trajectory_history[i]) > 1:
                        # 对历史轨迹进行采样以提高性能
                        sample_rate = max(1, len(trajectory_history[i]) // 100) if len(trajectory_history[i]) > 100 else 1
                        traj_points = trajectory_history[i][::sample_rate]
                        
                        # 确保包含最后一个点
                        if len(traj_points) > 0 and len(trajectory_history[i]) > 0:
                            if not np.array_equal(traj_points[-1], trajectory_history[i][-1]):
                                traj_points.append(trajectory_history[i][-1])
                        
                        # 更新轨迹线
                        traj_array = np.array(traj_points)
                        trajectory_lines[i].set_data(traj_array[:, 0], traj_array[:, 1])
                        trajectory_lines[i].set_3d_properties(traj_array[:, 2])
                        
                        # 更新智能体当前位置
                        last_pos = trajectory_history[i][-1]
                        agent_markers[i]._offsets3d = ([last_pos[0]], [last_pos[1]], [last_pos[2]])
                        
                        # 更新投影线
                        if len(trajectory_history[i]) > 0:
                            try:
                                last_pos = trajectory_history[i][-1]
                                # 确保使用正确的方法获取地形高度
                                terrain_height = 0
                                if hasattr(scenario, 'get_height_at'):
                                    terrain_height = scenario.get_height_at(last_pos[0], last_pos[1])
                                elif hasattr(scenario, 'get_terrain_height'):
                                    terrain_height = scenario.get_terrain_height(last_pos[0], last_pos[1])
                                elif hasattr(scenario, 'terrain') and scenario.terrain is not None:
                                    # 直接从地形数据获取高度
                                    x = int(max(0, min(last_pos[0], scenario.terrain.shape[1]-1)))
                                    y = int(max(0, min(last_pos[1], scenario.terrain.shape[0]-1)))
                                    terrain_height = scenario.terrain[y, x]
                                
                                # 确保地形高度是有效值
                                if terrain_height is None or np.isnan(terrain_height):
                                    if hasattr(scenario, 'terrain') and scenario.terrain is not None:
                                        terrain_height = np.mean(scenario.terrain)
                                    else:
                                        terrain_height = 0
                                
                                # 更新投影线，确保线条始终从地面到智能体
                                projection_lines[i].set_data([last_pos[0], last_pos[0]], [last_pos[1], last_pos[1]])
                                projection_lines[i].set_3d_properties([terrain_height, last_pos[2]])
                                
                                # 更新投影点，确保投影点位于地形上
                                projection_points[i]._offsets3d = ([last_pos[0]], [last_pos[1]], [terrain_height])
                                
                                # 调试输出
                                if DEBUG_OUTPUT and i == 0:  # 只输出第一个智能体的信息以减少日志量
                                    print(f"智能体{i}投影线更新: 位置=({last_pos[0]:.1f}, {last_pos[1]:.1f}), " +
                                         f"智能体高度={last_pos[2]:.1f}, 地形高度={terrain_height:.1f}")
                            except Exception as e:
                                if DEBUG_OUTPUT:
                                    print(f"[调试] 更新投影线失败: {e}")
                                pass
                    
                    # 更新投影轨迹线
                    if len(projection_history[i]) > 1:
                        try:
                            sample_rate = max(1, len(projection_history[i]) // 100) if len(projection_history[i]) > 100 else 1
                            proj_points = projection_history[i][::sample_rate]
                            
                            if len(proj_points) > 0 and len(projection_history[i]) > 0:
                                if not np.array_equal(proj_points[-1], projection_history[i][-1]):
                                    proj_points.append(projection_history[i][-1])
                            
                            if len(proj_points) > 1:  # 确保有足够的点来绘制线
                                proj_array = np.array(proj_points)
                                projection_trajectories[i].set_data(proj_array[:, 0], proj_array[:, 1])
                                projection_trajectories[i].set_3d_properties(proj_array[:, 2])
                                
                                if DEBUG_OUTPUT:
                                    print(f"投影轨迹{i}更新: {len(proj_points)}个点")
                        except Exception as e:
                            print(f"[可视化] 更新投影轨迹{i}时出错: {e}")
                except Exception as e:
                    print(f"[可视化] 更新轨迹{i}时出错: {e}")
    
    # 图例更新
    def update_legend(ax, agent_count, has_goal, has_landmarks):
        """更新图例"""
        from matplotlib.lines import Line2D  # 直接导入Line2D
        
        handles = []
        labels = []
        agent_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        
        # 添加智能体
        for i in range(agent_count):
            color = agent_colors[i % len(agent_colors)]
            handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8)
            handles.append(handle)
            labels.append(f'智能体 {i+1}')
        
        # 添加目标
        if has_goal:
            handle = Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=10)
            handles.append(handle)
            labels.append('目标')
        
        # 添加地标
        if has_landmarks:
            handle = Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8)
            handles.append(handle)
            labels.append('地标')
        
        # 设置图例
        if handles:
            leg = ax.legend(handles, labels, loc='upper right', frameon=True, fontsize=10)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_alpha(0.8)
        
        return ax
    
    # 更新绘图区域
    if viz_fig is not None and hasattr(viz_fig, 'canvas'):
        try:
            viz_fig.canvas.draw_idle()
            plt.pause(0.01)  # 短暂暂停以允许图形更新
        except Exception as e:
            print(f"[可视化] 更新图形时出错: {e}")
    
    # 更新控制文本
    update_control_text(viz_ax)
    
    # 更新图例
    update_legend(viz_ax, len(positions), goal_pos is not None, len(landmarks) > 0)
    
    # 更新图像和UI
    try:
        plt.draw()
        plt.pause(0.001)  # 短暂暂停以更新图形
        
        # 记录动画帧
        global record_animation, animation_frames
        if record_animation and viz_fig is not None:
            try:
                # 创建PIL图像对象
                from PIL import Image
                import io
                
                # 将matplotlib图形转换为PIL图像
                buf = io.BytesIO()
                viz_fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img = Image.open(buf)
                
                # 添加到动画帧列表
                animation_frames.append(img.copy())
                
                # 释放资源
                buf.close()
                
                # 输出更新信息（仅在关键帧）
                if step % 100 == 0 or step == max_steps - 1:
                    print(f"已捕获动画帧: {len(animation_frames)}, 当前步骤: {step}/{max_steps}")
            except Exception as e:
                print(f"捕获动画帧时出错: {e}")
    except Exception as e:
        print(f"[可视化] 绘图错误: {e}")
    
    # 在处理轨迹数据后，确保完整更新轨迹线显示
    try:
        for i, history in enumerate(trajectory_history):
            if i < len(trajectory_lines) and len(history) >= 2:  # 至少需要两个点才能画线
                # 转换为列表以方便分离三维坐标
                hist_array = np.array(history)
                xs = hist_array[:, 0]
                ys = hist_array[:, 1]
                zs = hist_array[:, 2]
                
                # 更新轨迹线
                trajectory_lines[i].set_data(xs, ys)
                trajectory_lines[i].set_3d_properties(zs)
                
                # 更新智能体当前位置标记
                if i < len(agent_markers):
                    last_pos = history[-1]
                    agent_markers[i]._offsets3d = ([last_pos[0]], [last_pos[1]], [last_pos[2]])
                
                if DEBUG_OUTPUT:
                    print(f"更新智能体{i}轨迹线: {len(xs)}个点")
    except Exception as e:
        print(f"更新轨迹线时出错: {e}")
        
    # 更新投影轨迹线
    try:
        for i, proj_history in enumerate(projection_history):
            if i < len(projection_trajectories) and len(proj_history) >= 2:
                # 将投影历史转换为数组
                proj_array = np.array(proj_history)
                
                # 应用采样以减少点数（如果点数太多）
                if len(proj_array) > 100:
                    sample_rate = max(1, len(proj_array) // 100)
                    sample_indices = np.arange(0, len(proj_array), sample_rate)
                    # 确保包含最后一个点
                    if sample_indices[-1] != len(proj_array) - 1:
                        sample_indices = np.append(sample_indices, len(proj_array) - 1)
                    proj_array = proj_array[sample_indices]
                
                xs = proj_array[:, 0]
                ys = proj_array[:, 1]
                zs = proj_array[:, 2]
                
                # 更新投影轨迹线
                projection_trajectories[i].set_data(xs, ys)
                projection_trajectories[i].set_3d_properties(zs)
                
                # 更新投影点（当前位置）
                if len(proj_history) > 0:
                    last_proj = proj_history[-1]
                    projection_points[i]._offsets3d = ([last_proj[0]], [last_proj[1]], [last_proj[2]])
                
                if DEBUG_OUTPUT:
                    print(f"更新智能体{i}投影轨迹线: {len(xs)}个点")
    except Exception as e:
        print(f"更新投影轨迹线时出错: {e}")
        
    return True  # 继续动画

def save_visualization_result(save_path=None, dpi=300, episode=None, reward=None):
    """保存当前可视化结果为高质量图像"""
    global viz_fig
    
    if viz_fig is None:
        print("错误：没有可视化图形可保存")
        return False
    
    try:
        # 设置默认保存路径
        if save_path is None:
            # 创建结果目录
            results_dir = os.path.join("results", "visualizations")
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            ep_str = f"_ep{episode}" if episode is not None else ""
            reward_str = f"_r{reward:.2f}" if reward is not None else ""
            filename = f"terrain_trajectory{ep_str}{reward_str}_{timestamp}.png"
            
            save_path = os.path.join(results_dir, filename)
        
        # 保存图像
        viz_fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
        return True
    except Exception as e:
        print(f"保存可视化结果时出错: {e}")
        traceback.print_exc()
        return False

def save_animation_as_gif(filename=None, fps=10, loop=0, optimize=True, duration=None):
    """将捕获的动画帧保存为GIF文件
    
    参数:
        filename: GIF文件名，如果为None则使用默认值
        fps: 每秒帧数
        loop: 循环次数，0表示无限循环
        optimize: 是否优化GIF
        duration: 每帧持续时间(毫秒)，如果为None则根据fps计算
    """
    global animation_frames
    
    if not animation_frames:
        print("错误: 没有动画帧可以保存")
        return False
    
    try:
        # 设置默认文件名
        if filename is None:
            # 创建结果目录
            results_dir = os.path.join("results", "animations")
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            filename = os.path.join(results_dir, f"terrain_trajectory_{timestamp}.gif")
        
        # 计算帧间隔
        if duration is None:
            duration = int(1000 / fps)  # 毫秒
        
        # 保存GIF
        print(f"正在保存GIF动画，共{len(animation_frames)}帧，帧率{fps}fps...")
        # 使用第一帧作为画布大小参考
        first_frame = animation_frames[0]
        first_frame.save(
            filename,
            save_all=True,
            append_images=animation_frames[1:],
            duration=duration,
            loop=loop,
            optimize=optimize
        )
        print(f"GIF动画已保存至: {filename}")
        return True
    except Exception as e:
        print(f"保存GIF动画时出错: {e}")
        traceback.print_exc()
        return False

def regenerate_terrain_if_needed(episode, scenario, terrain_change_episodes, env):
    """根据需要重新生成地形"""
    if episode in terrain_change_episodes:
        print(f"\n===== 回合 {episode}: 重新生成地形 =====")
        
        # 生成新种子
        new_seed = np.random.randint(0, 100000)
        print(f"使用新种子: {new_seed}")
        
        # 重新生成地形
        try:
            scenario.regenerate_terrain(new_seed=new_seed)
            
            # 重新创建世界
            world = scenario.make_world()
            
            # 重新初始化环境
            env.__init__(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None)
            
            print("地形重新生成成功!")
            return True
        except Exception as e:
            print(f"重新生成地形时出错: {e}")
            return False
    return False

# 添加configure_gpu函数在main函数之前
def configure_gpu(use_gpu=True):
    """配置GPU使用"""
    if not use_gpu:
        print("禁用GPU，将使用CPU进行计算")
        try:
            # 尝试设置环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            # 确保TensorFlow使用CPU
            tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            print(f"禁用GPU时出错: {e}")
            print("将尝试使用其他方法禁用GPU")
            try:
                # 如果上面的方法失败，使用另一种方式
                tf.config.experimental.set_visible_devices([], 'GPU')
            except Exception as e2:
                print(f"使用备选方法禁用GPU时也出错: {e2}")
                print("无法完全禁用GPU，性能可能会受影响")
    else:
        print("启用GPU加速")
        # 配置允许内存增长，避免占用全部显存
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"已配置 {len(gpus)} 个GPU，设置允许内存增长")
            except RuntimeError as e:
                print(f"配置GPU内存增长失败: {e}")

def ensure_observation_format(observation, expected_dim=None):
    """确保观察值是numpy数组格式并调整维度
    
    Args:
        observation: 原始观察值
        expected_dim: 期望的观察值维度，None表示保持原始维度不变
    
    Returns:
        numpy数组格式的观察值
    """
    # 如果没有指定维度，保持原始维度
    if expected_dim is None:
        # 让系统自动适配维度，尝试从环境或模型中推断维度
        if hasattr(observation, 'shape'):
            expected_dim = observation.shape[0] if len(observation.shape) > 0 else 24
        else:
            expected_dim = 24  # 默认维度
        
    if observation is None:
        return np.zeros(expected_dim, dtype=np.float32)  # 返回默认值
    
    # 处理字典类型的观察值 - 与 paper3d_train.py 中的 process_single_dict 函数兼容
    if isinstance(observation, dict):
        # 尝试提取观察值中的state信息
        if 'state' in observation:
            obs = np.array(observation['state'], dtype=np.float32)
        elif 'observation' in observation:
            obs = np.array(observation['observation'], dtype=np.float32)
        else:
            # 尝试将字典值拼接成一维数组
            try:
                values = []
                # 按固定顺序处理常见键，确保顺序一致性
                common_keys = ['pos', 'vel', 'goal', 'landmark', 'other_pos', 'other_vel', 
                              'entity_pos', 'entity_vel', 'rel_pos', 'dist', 'state']
                
                # 先处理常见键
                for key in common_keys:
                    if key in observation:
                        if isinstance(observation[key], (list, np.ndarray)):
                            values.extend(observation[key])
                        else:
                            values.append(observation[key])
                
                # 再处理其他键
                for key in sorted(observation.keys()):
                    if key not in common_keys:
                        if isinstance(observation[key], (list, np.ndarray)):
                            values.extend(observation[key])
                        else:
                            values.append(observation[key])
                
                obs = np.array(values, dtype=np.float32)
            except Exception as e:
                print(f"处理字典类型观察值失败: {e}")
                obs = np.zeros(expected_dim, dtype=np.float32)
    
    # 处理元组类型的观察值 - 通常是从env.reset()返回的
    elif isinstance(observation, tuple):
        print(f"处理元组类型观察值，长度: {len(observation)}")
        # 如果是(obs, reward, done, info)格式的元组，提取第一个元素
        if len(observation) >= 1:
            # 检查是否是一个智能体的观察或多个智能体的观察列表
            if isinstance(observation[0], (list, tuple, np.ndarray)) and len(observation[0]) > 0:
                # 将多个智能体的观察转换为数组
                agent_obs = []
                for obs in observation[0]:
                    if obs is not None:
                        try:
                            agent_obs.append(_adjust_observation_dim(np.array(obs, dtype=np.float32), expected_dim))
                        except:
                            agent_obs.append(np.zeros(expected_dim, dtype=np.float32))
                    else:
                        agent_obs.append(np.zeros(expected_dim, dtype=np.float32))
                return agent_obs
            else:
                # 单个智能体观察
                obs = np.array(observation[0], dtype=np.float32) if observation[0] is not None else np.zeros(expected_dim, dtype=np.float32)
        else:
            print(f"警告: 元组观察值格式不正确: {observation}")
            obs = np.zeros(expected_dim, dtype=np.float32)
    
    # 处理列表类型的观察值
    elif isinstance(observation, list):
        # 处理包含多个智能体观察的列表
        if all(isinstance(obs, (list, tuple, np.ndarray)) for obs in observation):
            return [_adjust_observation_dim(np.array(obs, dtype=np.float32) if obs is not None else np.zeros(expected_dim, dtype=np.float32), expected_dim) for obs in observation]
        
        # 处理单个智能体的列表观察值
        obs = np.array(observation, dtype=np.float32)
    
    # 处理numpy数组
    elif isinstance(observation, np.ndarray):
        obs = observation.astype(np.float32)
    
    # 处理TensorFlow张量
    elif hasattr(observation, 'numpy') and callable(getattr(observation, 'numpy')):
        try:
            obs = observation.numpy().astype(np.float32)
        except:
            print(f"无法转换TensorFlow张量为numpy数组: {type(observation)}")
            obs = np.zeros(expected_dim, dtype=np.float32)
    
    # 其他类型
    else:
        try:
            obs = np.array(observation, dtype=np.float32)
        except:
            print(f"无法转换观察值为numpy数组: {type(observation)}")
            obs = np.zeros(expected_dim, dtype=np.float32)
    
    # 调整观察值维度
    return _adjust_observation_dim(obs, expected_dim)

def _adjust_observation_dim(obs, expected_dim):
    """调整观察值维度为期望的维度
    
    Args:
        obs: 原始观察值数组
        expected_dim: 期望的维度，None表示保持原始维度不变
        
    Returns:
        调整后的观察值数组
    """
    # 如果expected_dim为None，保持原始维度
    if expected_dim is None:
        return obs
        
    if obs.shape == ():  # 标量
        return np.zeros(expected_dim, dtype=np.float32)
        
    if len(obs.shape) == 1:  # 一维数组
        actual_dim = obs.shape[0]
        if actual_dim == expected_dim:
            return obs
        elif actual_dim > expected_dim:
            print(f"裁剪观察值：从{actual_dim}维到{expected_dim}维")
            return obs[:expected_dim]
        else:  # actual_dim < expected_dim
            print(f"扩展观察值：从{actual_dim}维到{expected_dim}维")
            padded = np.zeros(expected_dim, dtype=np.float32)
            padded[:actual_dim] = obs
            return padded
    
    elif len(obs.shape) == 2:  # 二维数组，通常是批处理观察
        # 如果是单个样本的二维数组 (1, N)，转换为一维 (N,)
        if obs.shape[0] == 1:
            return _adjust_observation_dim(obs[0], expected_dim)
        else:
            print(f"警告：无法处理多样本观察值 {obs.shape}，尝试转换为一维数组")
            try:
                flattened = obs.flatten()
                return _adjust_observation_dim(flattened, expected_dim)
            except:
                return np.zeros(expected_dim, dtype=np.float32)
    
    else:  # 更高维数组
        print(f"警告：无法处理高维观察值 {obs.shape}，使用零向量")
        return np.zeros(expected_dim, dtype=np.float32)

def build_continuous_action_network(input_shape, action_dim=3, hidden_units=(384, 256, 128)):
    """
    构建连续动作空间的Actor网络 - 与训练代码中的测试兼容网络保持一致
    
    参数:
        input_shape: 输入状态的形状
        action_dim: 动作空间维度 (默认为3，表示XYZ坐标系中的力)
        hidden_units: 隐藏层神经元数量
        
    返回:
        actor_model: 连续动作Actor网络
    """
    # 为每个网络构建设置随机种子，与训练代码一致
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # 构建模型（函数式API）
    state_input = tf.keras.Input(shape=input_shape)
    
    # 第一隐藏层
    x = tf.keras.layers.Dense(
        hidden_units[0], 
        activation='relu',
        kernel_initializer=tf.keras.initializers.glorot_normal()
    )(state_input)
    
    # 第二隐藏层
    x = tf.keras.layers.Dense(
        hidden_units[1], 
        activation='relu',
        kernel_initializer=tf.keras.initializers.glorot_normal()
    )(x)
    
    # 第三隐藏层
    x = tf.keras.layers.Dense(
        hidden_units[2], 
        activation='relu',
        kernel_initializer=tf.keras.initializers.glorot_normal()
    )(x)
    
    # 输出层
    output_layer = tf.keras.layers.Dense(
        action_dim, 
        activation='tanh',
        kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
    )(x)
    
    # 创建模型
    actor_model = tf.keras.Model(inputs=state_input, outputs=output_layer)
    
    return actor_model

def build_continuous_critic_network(state_shape, action_dim=2, hidden_units=(128, 64, 32), n_agents=3):
    """
    构建连续动作空间的Critic网络 - 与训练代码中的测试兼容网络保持一致
    
    参数:
        state_shape: 状态的形状
        action_dim: 单个智能体动作空间维度 (默认为2，表示XY平面上的加速度)
        hidden_units: 隐藏层神经元数量
        n_agents: 智能体数量，默认为3
        
    返回:
        critic_model: Critic网络模型
    """
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # 构建模型（函数式API）
    state_input = tf.keras.Input(shape=state_shape)
    action_input = tf.keras.Input(shape=action_dim)
    
    # 处理状态
    x_state = tf.keras.layers.Dense(
        hidden_units[0], 
        activation='relu'
    )(state_input)
    
    # 合并状态和动作输入
    concat = tf.keras.layers.Concatenate()([x_state, action_input])
    
    # 第一隐藏层
    x = tf.keras.layers.Dense(
        hidden_units[1], 
        activation='relu'
    )(concat)
    
    # 第二隐藏层
    x = tf.keras.layers.Dense(
        hidden_units[2], 
        activation='relu'
    )(x)
    
    # 输出层
    output_layer = tf.keras.layers.Dense(
        1, 
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
    )(x)
    
    # 创建模型
    critic_model = tf.keras.Model(inputs=[state_input, action_input], outputs=output_layer)
    
    return critic_model

def create_force_field_corrector(scenario, args):
    """
    创建具有垂直力抑制功能的力场校正器
    
    参数:
        scenario: 场景对象，包含地形数据
        args: 命令行参数，包含各种力场配置
        
    返回:
        corrector: 配置好的力场校正器对象
    """
    import traceback
    
    if ContinuousPotentialFieldCorrector is None:
        print("警告: 力场校正器模块未导入，无法创建校正器")
        return None
    
    if not args.enable_action_correction:
        print("动作校正功能已禁用，不创建校正器")
        return None
    
    try:
        # 获取地形数据
        terrain_data = None
        X = None
        Y = None
        
        if hasattr(scenario, 'terrain') and scenario.terrain is not None:
            terrain_data = scenario.terrain
            if hasattr(scenario, 'X') and hasattr(scenario, 'Y'):
                X = scenario.X
                Y = scenario.Y
        
        # 打印力场校正器配置
        print("创建力场校正器，使用以下参数:")
        print(f"  最小间隙系数: {args.min_clearance_factor}")
        print(f"  安全距离系数: {args.safety_distance_factor}")
        print(f"  垂直力抑制: {args.vertical_force_suppress}")
        
        # 确保所有必要参数都存在，否则使用默认值
        goal_attraction = getattr(args, 'goal_attraction_weight', 1.0)
        terrain_repulsion = getattr(args, 'terrain_repulsion_weight', 1.0)
        agent_repulsion = getattr(args, 'agent_repulsion_weight', 0.5)
        influence_range = getattr(args, 'influence_range', 10.0)
        min_clearance = getattr(args, 'min_clearance', 2.0)
        force_scale = getattr(args, 'force_scale', 5.0)
        max_force = getattr(args, 'max_force_magnitude', 10.0)
        debug_mode = getattr(args, 'debug_corrector', False)
        vertical_suppress = getattr(args, 'vertical_force_suppress', 0.2)
        gravity_g = getattr(args, 'gravity', 0.0)
        sphere_radius = getattr(args, 'sphere_detection_radius', 5.0)
        detection_points = getattr(args, 'detection_points', 24)
        use_range = getattr(args, 'use_range_detection', True)
        detection_radius = getattr(args, 'detection_radius', 5.0)
        height_range = getattr(args, 'detection_height_range', 8.0)
        agent_radius = getattr(args, 'agent_detection_radius', 10.0)
        check_count = getattr(args, 'check_count', 8)
        check_spacing = getattr(args, 'check_spacing', 2.0)
        
        # 创建校正器
        corrector = ContinuousPotentialFieldCorrector(
            terrain_data=terrain_data,
            X=X,
            Y=Y,
            goal_attraction=goal_attraction,
            terrain_repulsion=terrain_repulsion,
            agent_repulsion=agent_repulsion,
            influence_range=influence_range,
            minimum_clearance=min_clearance,
            force_scale=force_scale,
            max_force_magnitude=max_force,
            debug_mode=debug_mode,
            vertical_force_suppress=vertical_suppress,
            sphere_detection_radius=sphere_radius,
            detection_points=detection_points,
            use_range_detection=use_range,
            # 新增检测参数
            detection_radius=detection_radius,
            detection_height_range=height_range,
            agent_detection_radius=agent_radius,
            check_count=check_count,
            check_spacing=check_spacing,
            gravity=gravity_g
        )
        
        # 使校正器支持动态添加属性
        old_setattr = corrector.__class__.__setattr__
        
        def new_setattr(self, name, value):
            try:
                old_setattr(self, name, value)
            except AttributeError:
                self.__dict__[name] = value
                
        corrector.__class__.__setattr__ = new_setattr
        
        # 将特殊参数存储为属性
        corrector.min_clearance_factor = args.min_clearance_factor
        corrector.safety_distance_factor = args.safety_distance_factor
        
        # 定义一个新的动作校正函数，替代原始的correct_action_continuous方法
        def new_correct_action_continuous(self, action, agent_pos, goal_pos=None, other_agents=None):
            # 首先调用原始方法获取基本的校正结果
            original_method = self.__class__.correct_action_continuous
            corrected_action = original_method(self, action, agent_pos, goal_pos, other_agents)
            
            # 获取当前位置的地形高度
            terrain_height = None
            try:
                if hasattr(self, 'get_terrain_height'):
                    terrain_height = self.get_terrain_height(agent_pos[0], agent_pos[1])
                elif hasattr(scenario, 'get_terrain_height'):
                    terrain_height = scenario.get_terrain_height(agent_pos[0], agent_pos[1])
                elif hasattr(scenario, 'get_height_at'):
                    terrain_height = scenario.get_height_at(agent_pos[0], agent_pos[1])
            except Exception as e:
                print(f"获取地形高度时出错: {e}")
                pass
            
            # 如果无法获取地形高度，直接返回原始校正结果
            if terrain_height is None:
                return corrected_action
            
            # 计算离地高度
            agent_clearance = max(0, agent_pos[2] - terrain_height)
            
            # 确定安全距离
            min_clearance = max(0.5, self.minimum_clearance * self.min_clearance_factor)
            safety_distance = max(1.0, self.minimum_clearance * self.safety_distance_factor)
            
            # 确定垂直抑制系数
            vertical_suppress = 1.0  # 默认不抑制
            
            # 根据高度应用智能抑制
            if agent_clearance < min_clearance:
                # 太接近地面，不抑制垂直力，避免穿透
                vertical_suppress = 1.0
            elif agent_clearance > safety_distance:
                # 足够高，应用设定的抑制系数
                vertical_suppress = self.vertical_force_suppress
            else:
                # 在过渡区域内，线性插值
                t = (agent_clearance - min_clearance) / (safety_distance - min_clearance)
                vertical_suppress = 1.0 - t * (1.0 - self.vertical_force_suppress)
            
            # 应用垂直抑制 - 仅在Z分量为正时抑制（避免减弱向下的力）
            if len(corrected_action) >= 3 and corrected_action[2] > 0:
                corrected_action[2] *= vertical_suppress
                
                if self.debug_mode:
                    print(f"垂直力抑制: {vertical_suppress:.2f}, 离地高度: {agent_clearance:.2f}")
            
            return corrected_action
        
        # 使用猴子补丁替换方法
        import types
        corrector.correct_action_continuous = types.MethodType(new_correct_action_continuous, corrector)
        
        print(f"成功创建力场校正器，垂直力抑制系数: {vertical_suppress}")
        return corrector
    
    except Exception as e:
        print(f"创建力场校正器时出错: {e}")
        traceback.print_exc()
        return None

# 添加act方法，该方法将调用run_episode中的policy方法
def add_act_method_to_maddpg(maddpg_instance):
    """为MADDPGRunner实例添加act方法，兼容测试代码"""
    if not hasattr(maddpg_instance, 'act'):
        def act_method(self, obs_n, add_noise=True):
            """
            从所有智能体的观察生成动作
            
            参数:
                obs_n: 所有智能体的观察列表
                add_noise: 是否添加探索噪声
                
            返回:
                actions: 所有智能体的动作列表
            """
            actions = []
            
            # 确保传入正确维度的观察
            if hasattr(self, 'expected_obs_dim'):
                expected_dim = self.expected_obs_dim
                print(f"使用期望的观察维度: {expected_dim}")
            else:
                # 尝试从环境获取维度
                try:
                    if hasattr(self, 'env') and hasattr(self.env, 'observation_space'):
                        obs_space = self.env.observation_space[0]
                        if hasattr(obs_space, 'shape'):
                            expected_dim = obs_space.shape[0]
                        elif hasattr(obs_space, 'n'):
                            expected_dim = obs_space.n
                        else:
                            expected_dim = 24  # 默认维度
                    else:
                        expected_dim = 24  # 默认维度
                except:
                    expected_dim = 24  # 默认维度
                
            # 修正：检查self.agents的类型，确保可以正确遍历 
            if not hasattr(self, 'agents'):
                print("错误: MADDPG实例没有agents属性")
                # 返回默认动作
                # 获取期望的动作维度
                expected_action_dim = getattr(self, 'expected_action_dim', 3)
                return [np.zeros(expected_action_dim) for _ in range(len(obs_n))]
            
            # 确保agents是可迭代对象
            try:
                # 检查agents的类型并适当处理
                if isinstance(self.agents, list):
                    # 如果是列表，直接使用
                    agents_to_iterate = self.agents
                elif hasattr(self.agents, '__iter__') and not isinstance(self.agents, (str, bytes, bytearray)):
                    # 如果是其他可迭代类型但不是字符串类型，转换为列表
                    agents_to_iterate = list(self.agents)
                else:
                    # 如果是单个对象（如MADDPGAgent实例），创建单元素列表
                    agents_to_iterate = [self.agents]
                    print(f"注意: 智能体不是列表类型，已转换为单元素列表")
                
                n_agents = len(agents_to_iterate)
                
            except Exception as e:
                print(f"处理智能体列表时出错: {str(e)}")
                traceback.print_exc()
                agents_to_iterate = []
            
            # 获取期望的动作维度
            expected_action_dim = getattr(self, 'expected_action_dim', 3)
            
            # 依次获取每个智能体的动作
            for i, agent in enumerate(agents_to_iterate):
                try:
                    # 处理观察值
                    if i < len(obs_n):
                        # 确保观察值是正确的格式和维度
                        obs = ensure_observation_format(obs_n[i], expected_dim)
                        
                        # 关键修改：确保输入到网络的观察值具有batch维度
                        # TensorFlow模型期望输入形状为 [batch_size, feature_dim]
                        # 但在推理时我们只有一个样本，所以需要添加batch维度
                        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
                            # 添加batch维度
                            obs = np.expand_dims(obs, axis=0)
                        
                        # 使用智能体的策略获取动作
                        if hasattr(agent, 'policy'):
                            try:
                                # 修复：尝试不同的参数名称，首先尝试不带噪声参数直接调用
                                try:
                                    action = agent.policy(obs)
                                except TypeError as e:
                                    # 如果失败，尝试使用add_noise参数
                                    try:
                                        action = agent.policy(obs, add_noise=add_noise)
                                    except TypeError:
                                        # 如果还失败，尝试使用noise_scale参数
                                        noise_scale = 1.0 if add_noise else 0.0
                                        action = agent.policy(obs, noise_scale=noise_scale)
                                
                                # 如果返回的动作有batch维度，去掉它
                                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                                    action = action.squeeze(0)  # 去掉batch维度
                                elif hasattr(action, 'numpy') and callable(getattr(action, 'numpy')):  # TensorFlow张量
                                    action = action.numpy().squeeze(0)
                                
                            except Exception as e:
                                print(f"执行policy时出错，返回默认动作: {str(e)}")
                                action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                                
                        elif hasattr(agent, 'act'):
                            # 如果智能体没有policy方法，尝试使用act网络
                            try:
                                # 同样尝试不同的参数名称
                                try:
                                    action = agent.act(obs)
                                except TypeError:
                                    try:
                                        action = agent.act(obs, add_noise=add_noise)
                                    except TypeError:
                                        noise_scale = 1.0 if add_noise else 0.0
                                        action = agent.act(obs, noise_scale=noise_scale)
                                
                                # 如果返回的是张量，转换为numpy数组
                                if hasattr(action, 'numpy') and callable(getattr(action, 'numpy')):
                                    action = action.numpy()
                                
                                # 如果返回的动作有batch维度，去掉它
                                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                                    action = action.squeeze(0)  # 去掉batch维度
                            except Exception as e:
                                print(f"执行act方法时出错，返回默认动作: {str(e)}")
                                action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                                
                        elif isinstance(agent, dict) and 'policy' in agent:
                            try:
                                # 字典方式访问，同样尝试不同参数
                                try:
                                    action = agent['policy'](obs)
                                except TypeError:
                                    try:
                                        action = agent['policy'](obs, add_noise=add_noise)
                                    except TypeError:
                                        noise_scale = 1.0 if add_noise else 0.0
                                        action = agent['policy'](obs, noise_scale=noise_scale)
                                
                                # 处理张量输出
                                if hasattr(action, 'numpy') and callable(getattr(action, 'numpy')):
                                    action = action.numpy()
                                
                                # 处理batch维度
                                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                                    action = action.squeeze(0)
                            except Exception as e:
                                print(f"执行字典policy时出错: {str(e)}")
                                action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                                
                        elif isinstance(agent, dict) and 'actor' in agent:
                            try:
                                # 直接使用actor网络
                                action = agent['actor'](obs)
                                
                                # 处理张量输出
                                if hasattr(action, 'numpy') and callable(getattr(action, 'numpy')):
                                    action = action.numpy()
                                
                                # 处理batch维度
                                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                                    action = action.squeeze(0)
                            except Exception as e:
                                print(f"执行字典actor时出错: {str(e)}")
                                action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                                
                        else:
                            print(f"警告: 智能体{i}没有policy或actor，使用零向量作为动作")
                            action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                    else:
                        # 如果观察列表长度不够，使用零向量
                        action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                        print(f"警告: 智能体{i}的观察不存在，使用零向量作为动作")
                        
                except Exception as e:
                    print(f"智能体{i}生成动作时出错: {str(e)}")
                    traceback.print_exc()
                    action = np.zeros(expected_action_dim)  # 使用期望的动作维度
                
                # 确保动作是numpy数组
                if not isinstance(action, np.ndarray):
                    try:
                        action = np.array(action)
                    except:
                        action = np.zeros(3)
                
                # 如果动作维度不是3，调整为3维
                if action.shape[0] != 3:
                    # 创建3维向量
                    action_3d = np.zeros(3)
                    # 复制可用的维度
                    try:
                        # 打印调试信息
                        print(f"调整动作维度: 原始形状={action.shape}, 类型={type(action)}")
                        
                        # 处理不同的情况
                        if len(action.shape) == 0:  # 标量
                            action_3d[0] = float(action)
                        elif len(action.shape) == 1:  # 向量
                            for j in range(min(action.shape[0], 3)):
                                if isinstance(action[j], (np.ndarray, list, tuple)):
                                    # 如果元素是序列，取第一个值
                                    action_3d[j] = float(action[j][0]) if len(action[j]) > 0 else 0.0
                                else:
                                    action_3d[j] = float(action[j])
                        else:  # 多维数组
                            # 将多维数组展平
                            flat_action = action.flatten()
                            for j in range(min(len(flat_action), 3)):
                                action_3d[j] = float(flat_action[j])
                    except Exception as e:
                        print(f"调整动作维度时出错: {e}")
                        action_3d = np.zeros(3)  # 出错时使用零向量
                    
                    action = action_3d
                
                actions.append(action)
            
            return actions
        
        # 将方法添加到实例
        import types
        maddpg_instance.act = types.MethodType(act_method, maddpg_instance)
        print("已为MADDPGRunner添加act方法")
    
    return maddpg_instance

# 添加load_model方法到MADDPGRunner
def add_load_model_to_maddpg(maddpg_instance):
    """为MADDPGRunner实例添加load_model方法，兼容测试代码"""
    if not hasattr(maddpg_instance, 'load_model'):
        def load_model_method(model_path):
            """从文件加载模型，简单地调用load_agents"""
            # 从model_path提取后缀
            import os
            suffix = ""
            base_path = os.path.basename(model_path)
            if "best_avg" in base_path:
                suffix = "_best_avg"
            elif "best" in base_path:
                suffix = "_best"
            
            print(f"load_model尝试使用后缀 '{suffix}' 加载模型")
            try:
                return maddpg_instance.load_agents(suffix=suffix)
            except Exception as e:
                print(f"load_model方法调用load_agents失败: {e}")
                return False
            
        # 将方法添加到实例
        import types
        maddpg_instance.load_model = types.MethodType(load_model_method, maddpg_instance)
        print("已为MADDPGRunner添加load_model方法")
    
    return maddpg_instance

def check_training_compatibility(args):
    """检查测试参数与训练参数的兼容性，调整不一致的参数
    
    Args:
        args: 命令行参数
        
    Returns:
        更新后的参数
    """
    print("\n===== 检查训练兼容性 =====")
    
    # 检查参数文件（优先查找与场景同名的参数文件）
    train_param_files = [
        f"./train_params_{args.scenario}.json",
        "./train_params.json",
        "./last_train_params.json"
    ]
    
    train_params = None
    for param_file in train_param_files:
        try:
            if os.path.exists(param_file):
                import json
                with open(param_file, 'r') as f:
                    train_params = json.load(f)
                print(f"成功加载训练参数文件: {param_file}")
                break
        except Exception as e:
            print(f"加载参数文件 {param_file} 失败: {e}")
    
    if train_params is None:
        print("未找到训练参数文件，将使用默认测试参数")
        return args
    
    # 提取关键参数并与当前测试参数比较
    important_params = [
        ("scenario", "场景"),
        ("continuous_action_space", "连续动作空间"),
        ("use_fixed_positions", "使用固定位置"),
        ("random_terrain", "随机地形"),
        ("random_z0_positions", "随机Z轴位置")
    ]
    
    # 打印参数比较表格
    print("\n训练参数与测试参数比较:")
    print("=" * 60)
    print(f"{'参数名称':<25} {'训练值':<15} {'测试值':<15} {'一致性'}")
    print("-" * 60)
    
    # 检查并调整参数
    for param_name, display_name in important_params:
        if param_name in train_params:
            train_value = train_params[param_name]
            # 获取测试参数值，如果不存在则使用None
            test_value = getattr(args, param_name, None)
            
            # 判断参数是否一致
            is_consistent = train_value == test_value
            consistency = "✓" if is_consistent else "✗"
            
            # 打印比较结果
            print(f"{display_name:<25} {str(train_value):<15} {str(test_value):<15} {consistency}")
            
            # 如果不一致，修改测试参数
            if not is_consistent and hasattr(args, param_name):
                old_value = getattr(args, param_name)
                setattr(args, param_name, train_value)
                print(f"  调整参数: {param_name} = {train_value}  (原值: {old_value})")
    
    print("-" * 60)
    
    # 检查训练参数中是否有关于观察空间和动作空间的信息
    if "obs_dim" in train_params:
        print(f"训练观察空间维度: {train_params['obs_dim']}")
        args.expected_obs_dim = train_params["obs_dim"]
    
    if "action_dim" in train_params:
        print(f"训练动作空间维度: {train_params['action_dim']}")
        args.expected_action_dim = train_params["action_dim"]
    
    # 检查其他重要参数
    if "hidden_units" in train_params:
        print(f"神经网络隐藏层: {train_params['hidden_units']}")
        args.hidden_units = train_params["hidden_units"]
    
    print("===== 参数检查完成 =====\n")
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查训练参数兼容性，确保测试环境与训练环境一致
    if not args.disable_params_check:
        args = check_training_compatibility(args)
    
    # 处理隐藏层单元数参数
    if isinstance(args.hidden_units, str):
        try:
            args.hidden_units = tuple(map(int, args.hidden_units.split(',')))
        except:
            print(f"警告: 无法解析隐藏层单元数'{args.hidden_units}'，使用默认值(384,256,128)")
            args.hidden_units = (384, 256, 128)
    
    if isinstance(args.critic_hidden_units, str):
        try:
            args.critic_hidden_units = tuple(map(int, args.critic_hidden_units.split(',')))
        except:
            print(f"警告: 无法解析Critic隐藏层单元数'{args.critic_hidden_units}'，使用默认值(128,64,32)")
            args.critic_hidden_units = (128, 64, 32)
    
    # 首先检查是否需要处理轨迹文件
    if process_trajectory_files(args):
        return 0
        
    # 显示基本参数信息
    print(f"测试场景: {args.scenario}")
    print(f"测试策略: {args.policy}")
    print(f"测试episode数: {args.num_episodes}")
    print(f"每个episode步数: {args.episode_length}")
    
    # 固定位置相关设置
    # 确保位置文件目录存在
    positions_dir = os.path.dirname(os.path.abspath(args.positions_file))
    if positions_dir and not os.path.exists(positions_dir):
        os.makedirs(positions_dir, exist_ok=True)
        print(f"已创建固定位置文件目录: {positions_dir}")
    
    # 显示固定位置设置的提示信息
    if args.use_fixed_positions:
        print(f"\n{'*'*70}")
        print(f"* {'固定位置模式已启用':^64} *")
    
    # 以下是现有的main函数代码
    # ...
    
    # 如果GPU配置不一致，则警告
    gpu_config_consistent = not (args.use_gpu and args.disable_gpu)
    if not gpu_config_consistent:
        print("警告: use-gpu和disable-gpu参数设置不一致，将以disable-gpu为准")
        
    # 配置GPU
    if args.disable_gpu:
        configure_gpu(False)
        print("已禁用GPU")
    elif args.use_gpu:
        configure_gpu()
        print("使用GPU")
    else:
        configure_gpu(False)
        print("默认禁用GPU")
    
    # 初始化
    np.random.seed(args.terrain_seed)
    random.seed(args.terrain_seed)
    
    # 如果使用可视化，初始化Pygame
    if not args.no_visualization:
        if not init_pygame():
            print("警告：可视化初始化失败，将继续但不显示可视化")
    
    # 载入场景模块
    try:
        scenario_module = importlib.import_module(f"multiagent.scenarios.{args.scenario}")
        scenario_class = scenario_module.Scenario
        print(f"加载场景模块: {args.scenario}")
    except Exception as e:
        print(f"加载场景模块失败: {e}")
        return
    
    # 创建场景、世界和环境
    scenario = scenario_class(random_terrain=args.random_terrain, 
                              seed=args.terrain_seed,
                              use_fixed_positions=args.use_fixed_positions,
                              fixed_positions_file=args.positions_file,
                              dynamic_first_time=args.dynamic_first_time)
                                                  
    # 打印位置初始化状态
    position_mode = "固定" if args.use_fixed_positions else "随机"
    position_status = f"首次运行动态生成" if args.dynamic_first_time else "使用预定义位置"
    print(f"位置初始化模式: {position_mode} ({position_status})")
    print(f"位置文件: {args.positions_file}")
    
    # 设置中文支持
    setup_chinese_font()
    
    # 创建世界
    world = scenario.make_world()
    
    # 创建环境
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                       scenario.done if hasattr(scenario, 'done') else None,
                       shared_viewer=True)
    
    # 设置verbose属性为False，减少调试输出
    env.verbose = False
    
    # 动作校正设置
    force_field_corrector = None
    if args.enable_action_correction:
        force_field_corrector = create_force_field_corrector(scenario, args)
    
    # 只有当环境对象有set_action_corrector方法时才调用
    if force_field_corrector is not None and hasattr(env, 'set_action_corrector'):
        env.set_action_corrector(force_field_corrector)
        print(f"启用动作校正: {args.correction_type}, 力度比例: {args.correction_force_ratio}")
    else:
        print("注意: 动作校正功能不可用，环境对象不支持该功能")
    
    # 查找模型目录 - 直接使用新生成的模型路径
    if args.model_dir is None:
        # 直接使用新生成的模型路径
        default_model_dir = "./weights"
        if os.path.exists(default_model_dir):
            args.model_dir = default_model_dir
            print(f"使用默认模型目录: {args.model_dir}")
    else:
        print(f"使用指定的模型目录: {args.model_dir}")
    
    # 检查模型目录是否存在
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        return
    
    # 创建MADDPG runner
    try:
        # 尝试直接初始化MADDPG实例，使用来自参数的网络配置
        maddpg = MADDPG(env, hidden_units=args.hidden_units, critic_hidden_units=args.critic_hidden_units, 
                       expected_obs_dim=args.expected_obs_dim, expected_action_dim=args.expected_action_dim)
        print(f"直接使用MADDPG类初始化成功")
    except Exception as e:
        try:
            # 如果使用MADDPG类失败，尝试使用MADDPGRunner，但要确保传递隐藏层参数
            print(f"使用MADDPG类初始化失败，切换到MADDPGRunner: {e}")
            maddpg = MADDPGRunner(env, hidden_units=args.hidden_units, critic_hidden_units=args.critic_hidden_units,
                                expected_obs_dim=args.expected_obs_dim, expected_action_dim=args.expected_action_dim)
            # 为MADDPGRunner添加act方法以兼容测试代码
            maddpg = add_act_method_to_maddpg(maddpg)
            # 为MADDPGRunner添加load_model方法
            maddpg = add_load_model_to_maddpg(maddpg)
            print(f"使用MADDPGRunner初始化成功")
        except Exception as e:
            print(f"创建MADDPG实例时出错: {e}")
            traceback.print_exc()
            return

    # 加载模型 - 使用新的load_models函数
    if not load_models(maddpg, args):
        print("加载模型失败，将使用随机初始化的模型运行")
    
    # 准备用于存储轨迹的数组
    trajectories = [[] for _ in range(env.n)]
    trajectory_file = os.path.join(args.model_dir, f"trajectories_{args.policy}.npy")
    if args.save_trajectory_npy:
        # 使用指定路径
        trajectory_file = args.save_trajectory_npy
        print(f"使用自定义轨迹保存路径: {trajectory_file}")
    
    # 创建结果保存目录
    if args.save_viz and args.save_path is None:
        args.save_path = os.path.join(args.model_dir, "visualizations")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print(f"创建可视化保存目录: {args.save_path}")
    
    # 准备用于动画的设置
    global record_animation
    record_animation = True  # 默认启用动画记录
    if args.animation_file is None:
        args.animation_file = os.path.join(args.model_dir, f"animation_{args.policy}.gif")
        print(f"动画将保存至: {args.animation_file}")
    
    # 测试循环
    total_rewards = []
    for episode in range(args.num_episodes):
        print(f"---------------------------- Episode {episode+1}/{args.num_episodes} ----------------------------")
        
        # 初始化episode轨迹
        episode_trajectories = [[] for _ in range(env.n)]
        
        # 重置环境
        obs_n = env.reset()
        
        # 确保观察是正确的格式
        if not isinstance(obs_n, list):
            print(f"警告: 观察不是列表格式，尝试转换")
            obs_n = ensure_observation_format(obs_n)
        
        # 如果转换后仍然不是列表，创建空的观察列表
        if not isinstance(obs_n, list):
            print(f"警告: 观察转换后仍不是列表，使用默认观察")
            obs_n = [np.zeros(24, dtype=np.float32) for _ in range(env.n)]
        
        # 测试每个智能体的观察维度
        if episode == 0:
            for i, obs in enumerate(obs_n):
                try:
                    print(f"智能体 {i} 观察维度: {obs.shape}")
                except:
                    print(f"智能体 {i} 观察无法获取形状: {type(obs)}")
                    # 为这个智能体创建默认观察
                    obs_n[i] = np.zeros(24, dtype=np.float32)
                    print(f"已为智能体 {i} 创建默认观察，维度: {obs_n[i].shape}")
        
        # 确保观察列表长度与智能体数量匹配
        if len(obs_n) < env.n:
            print(f"警告: 观察数量({len(obs_n)})小于智能体数量({env.n})，扩展观察列表")
            obs_n.extend([np.zeros(24, dtype=np.float32) for _ in range(env.n - len(obs_n))])
        
        # 获取世界中的障碍物和地形数据（如果有）
        obstacles = None
        terrain_data = None
        if hasattr(world, 'obstacles'):
            obstacles = world.obstacles
        if hasattr(world, 'terrain_data'):
            terrain_data = world.terrain_data
        
        episode_reward = 0
        
        # 用于可视化的第一步
        if not args.no_visualization:
            # 存储初始位置
            for i, agent in enumerate(world.agents):
                episode_trajectories[i].append(agent.state.p_pos)
                
            # 调用可视化回调
            visualization_callback(env, maddpg, scenario, 0, episode, args.episode_length, 
                                args.num_episodes, episode_trajectories, 0, args)
            
            # 渲染环境
            if args.render:
                env.render()
        
        # 主循环
        for step in range(args.episode_length):
            # 计算动作
            action_n = maddpg.act(obs_n)
            
            # 确保动作与智能体数量匹配
            if len(action_n) != env.n:
                print(f"警告：动作数量 ({len(action_n)}) 与智能体数量 ({env.n}) 不匹配！")
                # 截断或扩展动作列表
                if len(action_n) > env.n:
                    action_n = action_n[:env.n]
                else:
                    action_n.extend([np.zeros(3) for _ in range(env.n - len(action_n))])  # 确保是3维零向量
            
            # 最终检查所有动作维度，确保3D环境中所有动作都是3维
            if hasattr(env, 'world') and hasattr(env.world, 'dim_p') and env.world.dim_p == 3:
                for i, action in enumerate(action_n):
                    # 检查动作类型和维度
                    if not isinstance(action, np.ndarray):
                        print(f"智能体{i}的动作不是numpy数组，转换为3维数组")
                        try:
                            if isinstance(action, (list, tuple)) and len(action) >= 1:
                                action_3d = np.zeros(3)
                                # 复制尽可能多的维度
                                for j in range(min(len(action), 3)):
                                    action_3d[j] = action[j]
                                action_n[i] = action_3d
                            else:
                                action_n[i] = np.zeros(3)
                        except:
                            action_n[i] = np.zeros(3)
                    elif action.shape[0] != 3:
                        # 如果是numpy数组但维度不是3
                        print(f"智能体{i}的动作是{action.shape[0]}维，转换为3维")
                        action_3d = np.zeros(3)
                        # 如果是一维或二维数组，复制这些值
                        if len(action.shape) == 1:
                            for j in range(min(action.shape[0], 3)):
                                action_3d[j] = action[j]
                        elif len(action.shape) > 1:
                            # 对于更高维度的数组，展平后取前3个值
                            flat_action = action.flatten()
                            for j in range(min(len(flat_action), 3)):
                                action_3d[j] = flat_action[j]
                        action_n[i] = action_3d
            
            # 检查是否是离散动作空间，如果是，需要转换为连续动作
            if hasattr(env, 'action_space') and any(hasattr(space, 'n') for space in env.action_space):
                print("检测到离散动作空间，转换动作")
                for i, action in enumerate(action_n):
                    if hasattr(env.action_space[i], 'n'):  # Discrete
                        # 将离散动作转为one-hot
                        discrete_action = np.zeros(env.action_space[i].n)
                        action_idx = int(action[0]) if isinstance(action, np.ndarray) and action.size > 0 else 0
                        discrete_action[action_idx] = 1.0
                        action_n[i] = discrete_action
                    elif hasattr(env.action_space[i], 'nvec'):  # MultiDiscrete
                        # 确保MultiDiscrete动作正确
                        if isinstance(action, np.ndarray) and action.size >= 2:
                            # 先取前两维，对应MultiDiscrete的两个维度
                            multi_discrete_action = np.zeros_like(env.action_space[i].nvec)
                            multi_discrete_action[:2] = np.clip(action[:2], 0, env.action_space[i].nvec[:2] - 1).astype(int)
                            action_n[i] = multi_discrete_action
                        else:
                            print(f"警告：智能体{i}的动作不是数组或长度不足，使用默认动作")
                            action_n[i] = np.zeros_like(env.action_space[i].nvec)
                    
                    # 确保最终动作维度与3D环境匹配
                    if hasattr(env.world, 'dim_p') and env.world.dim_p == 3:
                        if not isinstance(action_n[i], np.ndarray) or action_n[i].shape[0] != 3:
                            print(f"调整智能体{i}最终动作维度为3：{action_n[i].shape if isinstance(action_n[i], np.ndarray) else type(action_n[i])} -> (3,)")
                            # 创建3维向量
                            force = np.zeros(3)
                            try:
                                # 如果原动作有1或2维，保留这些值
                                if isinstance(action_n[i], np.ndarray) and action_n[i].size >= 1:
                                    for j in range(min(action_n[i].size, 3)):
                                        force[j] = action_n[i].flatten()[j]
                                elif isinstance(action_n[i], (list, tuple)) and len(action_n[i]) >= 1:
                                    for j in range(min(len(action_n[i]), 3)):
                                        force[j] = action_n[i][j]
                            except Exception as e:
                                print(f"转换动作时出错: {e}")
                            action_n[i] = force
            
            # 最终检查所有动作维度
            if hasattr(env, 'verbose') and env.verbose:
                print(f"最终动作维度检查:")
                for i, action in enumerate(action_n):
                    print(f"  智能体{i}: {action.shape if isinstance(action, np.ndarray) else type(action)}")
            # 最后确保所有动作都是3维数组
            for i, action in enumerate(action_n):
                if hasattr(env.world, 'dim_p') and env.world.dim_p == 3:
                    if not isinstance(action, np.ndarray) or action.shape[0] != 3:
                        if hasattr(env, 'verbose') and env.verbose:
                            print(f"  智能体{i}动作维度不匹配，强制设为3维零向量")
                        action_n[i] = np.zeros(3)
            
            # 保存强化前的动作
            original_action_n = action_n.copy() if isinstance(action_n, list) else None
            
            # 执行动作
            new_obs_n, reward_n, done_n, info_n = env.step(action_n)
            
            # 确保新观察是正确的格式
            if not isinstance(new_obs_n, list):
                print(f"警告: 新观察不是列表格式，尝试转换")
                new_obs_n = ensure_observation_format(new_obs_n)
                
            # 如果转换后仍然不是列表，创建空的观察列表
            if not isinstance(new_obs_n, list):
                print(f"警告: 新观察转换后仍不是列表，使用默认观察")
                new_obs_n = [np.zeros(24, dtype=np.float32) for _ in range(env.n)]
                
            # 确保观察列表长度与智能体数量匹配
            if len(new_obs_n) < env.n:
                print(f"警告: 新观察数量({len(new_obs_n)})小于智能体数量({env.n})，扩展观察列表")
                new_obs_n.extend([np.zeros(24, dtype=np.float32) for _ in range(env.n - len(new_obs_n))])
            
            # 记录每个智能体的位置到轨迹中
            for agent_idx, agent in enumerate(env.world.agents):
                if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                    pos = agent.state.p_pos.copy()  # 复制以避免引用问题
                    # 如果是2D位置，转换为3D (添加高度)
                    if len(pos) == 2:
                        pos = np.append(pos, 0)  # 使用默认高度0
                    
                    # 添加到轨迹
                    episode_trajectories[agent_idx].append(pos)
                    
                    # 每50步显示一次轨迹长度，用于调试
                    if step % 50 == 0 and step > 0:
                        print(f"Step {step}: 智能体{agent_idx}轨迹长度: {len(episode_trajectories[agent_idx])}")
            
            # 如果任一智能体完成，提前结束episode
            if any(done_n):
                print(f"Episode提前结束，步数: {step+1}/{args.episode_length}")
                break
            
            # 更新观察
            obs_n = new_obs_n
            
            # 累计奖励
            episode_reward += sum(reward_n)
            
            # 可视化
            if not args.no_visualization and (step % 10 == 0 or step == args.episode_length - 1):
                visualization_callback(env, maddpg, scenario, step, episode, args.episode_length, 
                                    args.num_episodes, episode_trajectories, episode_reward, args)
                
                # 渲染环境
                if args.render:
                    env.render()
        
        # Episode结束后，将本episode的轨迹添加到总轨迹中
        for i, episode_traj in enumerate(episode_trajectories):
            if len(episode_traj) > 0:  # 确保轨迹不为空
                # 转换为numpy数组以便于后续处理
                episode_traj_array = np.array(episode_traj)
                print(f"Episode {episode+1}结束: 智能体{i}轨迹点数={len(episode_traj_array)}")
                
                # 添加到总轨迹
                trajectories[i].append(episode_traj_array)
            else:
                print(f"警告: Episode {episode+1}中智能体{i}没有收集到轨迹数据")
                # 添加一个空数组作为占位符
                trajectories[i].append(np.array([]))
                
        # 添加总奖励
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} 总奖励: {episode_reward:.4f}")
        
        # 调试轨迹数据，查看当前格式
        debug_trajectories(trajectories, f"Episode {episode+1}后的轨迹数据")
        
        # 保存最后一步的可视化
        if args.save_viz and not args.no_visualization:
            save_visualization_result(args.save_path, episode=episode, reward=episode_reward)
    
    # 测试结束后的清理
    env.close()
    
    # 创建固定的结果目录结构，便于查找
    test_results_dir = "./results/test_results"
    viz_dir = os.path.join(test_results_dir, "visualizations")
    anim_dir = os.path.join(test_results_dir, "animations")
    traj_dir = os.path.join(test_results_dir, "trajectories")
    
    # 确保目录存在
    for directory in [test_results_dir, viz_dir, anim_dir, traj_dir]:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")
        except Exception as e:
            print(f"创建目录失败: {directory}, 错误: {e}")
            # 尝试使用相对路径
            alt_directory = os.path.join(".", os.path.basename(directory))
            try:
                os.makedirs(alt_directory, exist_ok=True)
                print(f"改用备用目录: {alt_directory}")
                if directory == viz_dir:
                    viz_dir = alt_directory
                elif directory == anim_dir:
                    anim_dir = alt_directory
                elif directory == traj_dir:
                    traj_dir = alt_directory
            except Exception as e2:
                print(f"创建备用目录也失败: {e2}")
                # 最后尝试使用当前目录
                if directory == viz_dir:
                    viz_dir = "."
                elif directory == anim_dir:
                    anim_dir = "."
                elif directory == traj_dir:
                    traj_dir = "."
    
    # 生成时间戳，确保文件名唯一性
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    scenario_name = args.scenario
    policy_name = args.policy
    base_filename = f"{scenario_name}_{policy_name}_{timestamp}"
    
    # 保存带有地形的轨迹图
    # 设置文件路径
    trajectory_image_path = os.path.join(viz_dir, f"{base_filename}_trajectory.png")
    
    # 使用完整的轨迹数据而不仅仅是最后一个点
    final_trajectories = []
    for agent_idx, agent_traj in enumerate(trajectories):
        if agent_traj and len(agent_traj) > 0:
            # 确保我们获取的是完整的轨迹点序列，不仅仅是最后一个点
            if isinstance(agent_traj[-1], (list, np.ndarray)) and hasattr(agent_traj[-1], '__len__'):
                final_trajectories.append(agent_traj[-1])
                print(f"智能体{agent_idx}的轨迹点数: {len(agent_traj[-1])}")
            else:
                # 如果数据结构不同，尝试直接使用整个轨迹数据
                final_trajectories.append(agent_traj)
                print(f"智能体{agent_idx}的轨迹点数: {len(agent_traj) if hasattr(agent_traj, '__len__') else '未知'}")
        else:
            final_trajectories.append([])
            print(f"警告: 智能体{agent_idx}没有轨迹数据")
    
    # 调试生成的最终轨迹数据
    debug_trajectories(final_trajectories, "最终处理后的轨迹数据")
    
    # 获取智能体最终奖励和状态
    avg_reward = sum(total_rewards) / len(total_rewards)
    episode_info = f"Episode {episode+1}/{args.num_episodes}, Step {args.episode_length}/{args.episode_length}, Reward: {avg_reward:.2f}"
    
    # 暂时禁用额外轨迹图生成
    print(f"\n{'*'*80}")
    print(f"{'提示：已暂时禁用额外轨迹图生成':^80}")
    print(f"{'您可以在主3D渲染窗口中手动调整视角并保存图像':^80}")
    print(f"{'*'*80}\n")
    
    # 找到原来被注释的创建轨迹图代码段，并取消注释
    # 原代码在行2156-2180左右

    # 暂时禁用额外轨迹图生成
    print(f"\n{'*'*80}")
    print(f"{'生成带地形的轨迹图...':^80}")
    print(f"{'*'*80}\n")
    
    # 创建带有地形的轨迹图
    try:
        print(f"保存轨迹图到路径: {trajectory_image_path}")
        # 确保路径存在
        os.makedirs(os.path.dirname(trajectory_image_path), exist_ok=True)
        
        # 创建带有地形的轨迹图
        fig = load_and_plot_trajectories(
            trajectory_file=None,  # 不从文件加载，直接使用已有轨迹
            goal_position=scenario.goal_pos if hasattr(scenario, 'goal_pos') else None,
            terrain_data=scenario,
            output_path=trajectory_image_path,
            direct_trajectory_data=final_trajectories  # 修正参数名称
        )
        
        if fig is not None:
            # 保存完成后再关闭图形
            plt.savefig(trajectory_image_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已成功保存至: {trajectory_image_path}")
            plt.close(fig)  # 关闭图形，释放资源
        else:
            print(f"警告：无法创建轨迹图")
    except Exception as e:
        print(f"保存轨迹图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 注释掉保存轨迹数据（npy文件）的部分
    '''
    # 保存轨迹数据（使用专用目录）
    if args.save_trajectory_npy:
        # 使用指定路径
        trajectory_file = args.save_trajectory_npy
    else:
        # 使用新的默认路径
        trajectory_file = os.path.join(traj_dir, f"{base_filename}_trajectories.npy")
    
    # 保存轨迹数据
    save_result = save_trajectory_data(trajectories, trajectory_file)
    '''
    
    # 添加虚拟轨迹文件路径以避免后续代码出错
    trajectory_file = os.path.join(traj_dir, f"{base_filename}_trajectories.npy")
    print("注意: 已禁用npy轨迹文件生成")
    
    # 保存动画GIF
    animation_path = os.path.join(anim_dir, f"{base_filename}_animation.gif")
    if record_animation and len(animation_frames) > 0:
        save_result = save_animation_as_gif(animation_path, fps=args.animation_fps)
    else:
        # 如果没有记录动画，强制生成一个
        print("没有记录到动画帧，尝试生成新的动画...")
        if not args.no_visualization and viz_fig is not None:
            try:
                # 进行简单的旋转动画生成
                from matplotlib import animation
                
                def rotate(angle):
                    viz_ax.view_init(elev=30, azim=angle)
                    return [viz_ax]
                
                # 创建旋转动画
                ani = animation.FuncAnimation(viz_fig, rotate, frames=range(0, 360, 5), 
                                             interval=100, blit=False)
                
                # 保存为GIF
                ani.save(animation_path, writer='pillow', fps=10)
            except Exception as e:
                print(f"生成旋转动画时出错: {e}")
                traceback.print_exc()
            
    # 更改输出信息，移除对npy文件的引用
    # 输出所有结果文件路径信息，格式统一且醒目
    print(f"\n{'='*80}")
    print(f"{'测试结果文件路径':^80}")
    print(f"{'-'*80}")
    print(f"轨迹图保存路径: {os.path.abspath(trajectory_image_path)}")
    if record_animation and len(animation_frames) > 0:
        print(f"GIF动画保存路径: {os.path.abspath(animation_path)}")
    print(f"{'='*80}")
    
    # 输出最终结果
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n{'='*80}")
    print(f"{'测试性能统计':^80}")
    print(f"{'-'*80}")
    print(f"场景: {args.scenario}")
    print(f"测试策略: {args.policy}")
    print(f"测试episode数: {args.num_episodes}")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"{'='*80}")
    
    # 保持窗口打开直到用户关闭，增加视角控制提示
    if not args.no_visualization:
        plt.ioff()
        if viz_fig is not None:
            # 更新主窗口视图
            if hasattr(viz_ax, 'view_init'):
                # 设置更好的初始视角
                viz_ax.view_init(elev=35, azim=45)
                print("\n视角控制指南:")
                print("- 鼠标拖拽: 旋转视角")
                print("- 鼠标滚轮: 缩放")
                print("- 右键菜单可保存当前视图")
                print("按关闭窗口按钮退出程序\n")
            plt.show()
    
    return avg_reward

def load_and_plot_trajectories(trajectory_file=None, goal_position=None, terrain_data=None, output_path=None, direct_trajectory_data=None):
    """加载并绘制轨迹图"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os
    import traceback
    
    try:
        # 确保中文字体设置
        font_setting = setup_chinese_font()
        
        # 创建图形和3D轴
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 加载轨迹数据
        if direct_trajectory_data is not None:
            trajectories = direct_trajectory_data
            print(f"使用直接提供的轨迹数据")
            if isinstance(trajectories, list) or isinstance(trajectories, np.ndarray):
                print(f"轨迹数据包含 {len(trajectories)} 个智能体的数据")
            else:
                print(f"警告: 无法确定轨迹数据的结构: {type(trajectories)}")
        elif trajectory_file and os.path.exists(trajectory_file):
            try:
                trajectories = np.load(trajectory_file, allow_pickle=True)
                print(f"从文件加载轨迹数据: {trajectory_file}")
                if hasattr(trajectories, 'shape'):
                    print(f"轨迹数据形状: {trajectories.shape}")
            except Exception as e:
                print(f"加载轨迹文件出错: {e}")
                traceback.print_exc()
                return None
        else:
            print(f"轨迹文件不存在: {trajectory_file}")
            return None
        
        # 处理轨迹数据
        if not isinstance(trajectories, list) and not isinstance(trajectories, np.ndarray):
            print(f"轨迹数据类型错误: {type(trajectories)}")
            return None
        
        # 确保目录存在
        if output_path:
            try:
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"创建输出目录: {output_dir}")
            except Exception as e:
                print(f"创建输出目录失败: {e}")
        
        # 设置图表标题
        plt.title('智能体轨迹图', fontsize=18, pad=20)
        
        # 设置坐标轴标签，更清晰的风格
        ax.set_xlabel('X轴', fontsize=14, labelpad=10)
        ax.set_ylabel('Y轴', fontsize=14, labelpad=10)
        ax.set_zlabel('Z轴', fontsize=14, labelpad=10)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 设置背景色为白色
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 如果有地形数据，绘制地形
        if terrain_data is not None:
            try:
                print(f"地形数据类型: {type(terrain_data)}")
                
                # 检查地形数据类型
                if hasattr(terrain_data, 'terrain') and hasattr(terrain_data.terrain, 'shape'):
                    # 如果地形数据是场景对象
                    print(f"地形数据来自场景对象")
                    terrain = terrain_data.terrain
                    grid_size = terrain_data.grid_size if hasattr(terrain_data, 'grid_size') else 100
                    
                    # 创建网格
                    x = np.linspace(0, grid_size, terrain.shape[0])
                    y = np.linspace(0, grid_size, terrain.shape[1])
                    X, Y = np.meshgrid(x, y)
                    
                    # 使用更清晰的地形样式
                    terrain_cmap = plt.cm.terrain  # 使用terrain配色方案
                    terrain_surface = ax.plot_surface(
                        X, Y, terrain, 
                        cmap=terrain_cmap,
                        alpha=0.8,  # 增加不透明度
                        rstride=1,  # 减小步长，增加细节
                        cstride=1,
                        linewidth=0,  # 移除网格线
                        antialiased=True,
                        vmin=np.min(terrain),  # 设置高度范围
                        vmax=np.max(terrain),
                        edgecolor=None  # 移除边缘颜色
                    )
                    # 添加高度色标 - 放在底部，完全避免遮挡图例
                    cbar = fig.colorbar(terrain_surface, ax=ax, 
                                       orientation='horizontal',  # 水平方向
                                       shrink=0.6,                # 宽度
                                       aspect=30,                 # 长宽比
                                       pad=0.1,                   # 与图的间距
                                       location='bottom')         # 底部位置
                    cbar.set_label('高度 (m)', fontsize=10, labelpad=5)
                    
                elif isinstance(terrain_data, tuple) and len(terrain_data) == 3:
                    # 如果地形数据是(X, Y, Z)三元组
                    terrain_x, terrain_y, terrain_z = terrain_data
                    print(f"绘制地形，数据形状: X={terrain_x.shape if hasattr(terrain_x, 'shape') else 'unknown'}, Y={terrain_y.shape if hasattr(terrain_y, 'shape') else 'unknown'}, Z={terrain_z.shape if hasattr(terrain_z, 'shape') else 'unknown'}")
                    
                    # 使用更清晰的地形样式
                    terrain_cmap = plt.cm.terrain  # 使用terrain配色方案
                    terrain_surface = ax.plot_surface(
                        terrain_x, terrain_y, terrain_z, 
                        cmap=terrain_cmap,
                        alpha=0.8,  # 增加不透明度
                        rstride=1,  # 减小步长，增加细节
                        cstride=1,
                        linewidth=0,  # 移除网格线
                        antialiased=True,
                        vmin=np.min(terrain_z),  # 设置高度范围
                        vmax=np.max(terrain_z),
                        edgecolor=None  # 移除边缘颜色
                    )
                    # 添加高度色标 - 放在底部，完全避免遮挡图例
                    cbar = fig.colorbar(terrain_surface, ax=ax, 
                                       orientation='horizontal',  # 水平方向
                                       shrink=0.6,                # 宽度
                                       aspect=30,                 # 长宽比
                                       pad=0.1,                   # 与图的间距
                                       location='bottom')         # 底部位置
                    cbar.set_label('高度 (m)', fontsize=10, labelpad=5)
                else:
                    print(f"未知的地形数据类型: {type(terrain_data)}")
            except Exception as e:
                print(f"绘制地形时出错: {e}")
                traceback.print_exc()
        
        # 设置视角和缩放
        ax.view_init(elev=30, azim=45)  # 设置初始视角
        
        # 定义智能体名称和颜色
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 使用列表确保可以正确索引
        agent_names = ['智能体 1', '智能体 2', '智能体 3']
        
        # 处理可能的数据结构差异
        try:
            # 检查trajectories是否是嵌套列表或数组
            if len(trajectories) > 0:
                # 打印轨迹数据类型，辅助调试
                print(f"原始轨迹数据类型: {type(trajectories)}")
                if isinstance(trajectories[0], list) or isinstance(trajectories[0], np.ndarray):
                    print(f"轨迹[0]类型: {type(trajectories[0])}")
                    if len(trajectories[0]) > 0:
                        print(f"轨迹[0][0]类型: {type(trajectories[0][0]) if isinstance(trajectories[0], list) and len(trajectories[0]) > 0 else 'N/A'}")
                
                # 处理可能的不同轨迹数据结构
                final_trajectories = []
                for agent_idx, agent_traj in enumerate(trajectories):
                    # 确保我们处理的是实际的轨迹数据
                    if isinstance(agent_traj, list) and len(agent_traj) > 0:
                        if isinstance(agent_traj[0], np.ndarray) and len(agent_traj[0]) > 0:
                            # 这是正常的情况，使用完整轨迹
                            final_trajectories.append(agent_traj[0])
                            print(f"添加完整轨迹，智能体{agent_idx}，点数: {len(agent_traj[0])}")
                        else:
                            # 如果agent_traj本身是轨迹，直接使用
                            final_trajectories.append(agent_traj)
                            print(f"添加轨迹列表，智能体{agent_idx}，点数: {len(agent_traj)}")
                    elif isinstance(agent_traj, np.ndarray) and len(agent_traj) > 0:
                        # 如果是numpy数组，直接使用
                        final_trajectories.append(agent_traj)
                        print(f"添加numpy轨迹，智能体{agent_idx}，点数: {len(agent_traj)}")
                    else:
                        print(f"警告: 智能体{agent_idx}轨迹为空或格式不正确")
                        final_trajectories.append([])
            else:
                print("警告: 轨迹数据为空")
                final_trajectories = []
        except Exception as e:
            print(f"处理轨迹数据时出错: {e}")
            traceback.print_exc()
            final_trajectories = trajectories  # 失败时直接使用原始数据
         
        # 创建图例列表
        legend_elements = []
        
        # 绘制每个智能体的完整轨迹
        for i, trajectory in enumerate(final_trajectories):
            # 确保colors列表不为空
            if not colors:
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 设置默认颜色列表
            
            color = colors[i % len(colors)]
            agent_name = agent_names[i] if i < len(agent_names) else f'智能体 {i+1}'
            
            if isinstance(trajectory, np.ndarray) and trajectory.size > 0:
                # 需要检查轨迹点的维度
                if len(trajectory.shape) == 1:
                    print(f"警告: {agent_name}轨迹是一维数组，可能无法绘制")
                    continue
                
                if trajectory.shape[1] >= 3:  # 确保有3D坐标
                    print(f"绘制{agent_name}的轨迹，共{len(trajectory)}个点")
                    
                    # 绘制轨迹线
                    line = ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                                 color=color, linewidth=2, label=agent_name)[0]
                    
                    # 标记起点
                    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                             color=color, marker='o', s=80, edgecolors='black', linewidths=0.5)
                    
                    # 标记终点
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                             color=color, marker='o', s=100, edgecolors='black', linewidths=0.5)
                    
                    # 添加投影线（从最后一个点到地面）
                    if terrain_data is not None:
                        try:
                            last_point = trajectory[-1]
                            # 计算地面高度 - 这里假设是最低点
                            if hasattr(terrain_data, 'get_terrain_height'):
                                ground_height = terrain_data.get_terrain_height(last_point[0], last_point[1])
                            else:
                                # 简单估计地面高度
                                ground_height = 0
                            
                            # 画投影线
                            ax.plot([last_point[0], last_point[0]], 
                                   [last_point[1], last_point[1]],
                                   [ground_height, last_point[2]],
                                   '--', color=color, alpha=0.5, linewidth=1.5)
                            
                            # 画投影点
                            ax.scatter(last_point[0], last_point[1], ground_height,
                                     color=color, alpha=0.5, marker='x', s=50)
                        except Exception as e:
                            print(f"绘制投影线时出错: {e}")
                    
                    # 添加到图例
                    from matplotlib.lines import Line2D
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=agent_name))
                else:
                    print(f"警告: {agent_name}轨迹维度不足3D，实际维度: {trajectory.shape}")
            elif isinstance(trajectory, list) and len(trajectory) > 0:
                # 处理列表类型的轨迹
                try:
                    trajectory_array = np.array(trajectory)
                    if len(trajectory_array.shape) >= 2 and trajectory_array.shape[1] >= 3:
                        print(f"绘制{agent_name}的轨迹(从列表转换)，共{len(trajectory)}个点")
                        
                        # 绘制轨迹线
                        line = ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                                     color=color, linewidth=2, label=agent_name)[0]
                        
                        # 标记起点和终点
                        ax.scatter(trajectory_array[0, 0], trajectory_array[0, 1], trajectory_array[0, 2], 
                                 color=color, marker='o', s=80, edgecolors='black', linewidths=0.5)
                        ax.scatter(trajectory_array[-1, 0], trajectory_array[-1, 1], trajectory_array[-1, 2], 
                                 color=color, marker='o', s=100, edgecolors='black', linewidths=0.5)
                        
                        # 添加投影线（从最后一个点到地面）
                        if terrain_data is not None:
                            try:
                                last_point = trajectory_array[-1]
                                # 计算地面高度 - 这里假设是最低点
                                if hasattr(terrain_data, 'get_terrain_height'):
                                    ground_height = terrain_data.get_terrain_height(last_point[0], last_point[1])
                                else:
                                    # 简单估计地面高度
                                    ground_height = 0
                                
                                # 画投影线
                                ax.plot([last_point[0], last_point[0]], 
                                       [last_point[1], last_point[1]],
                                       [ground_height, last_point[2]],
                                       '--', color=color, alpha=0.5, linewidth=1.5)
                                
                                # 画投影点
                                ax.scatter(last_point[0], last_point[1], ground_height,
                                         color=color, alpha=0.5, marker='x', s=50)
                            except Exception as e:
                                print(f"绘制投影线时出错: {e}")
                        
                        # 添加到图例
                        from matplotlib.lines import Line2D
                        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=agent_name))
                    else:
                        print(f"警告: {agent_name}轨迹(列表)维度不足3D")
                except Exception as e:
                    print(f"处理{agent_name}列表轨迹时出错: {e}")
                    traceback.print_exc()
            else:
                print(f"警告: {agent_name}轨迹数据无效或为空")
        
        # 如果提供了目标位置，则绘制目标点
        if goal_position is not None:
            try:
                print(f"绘制目标点: {goal_position}")
                # 绘制目标点（星形）
                goal_marker = ax.scatter(
                    goal_position[0], goal_position[1], goal_position[2],
                    color='red',
                    marker='*',
                    s=200,  # 大尺寸
                    edgecolors='black',
                    linewidths=0.5,
                    label='目标'
                )
                
                # 添加红色虚线连接到地面
                if terrain_data is not None:
                    try:
                        # 计算地面高度
                        if hasattr(terrain_data, 'get_terrain_height'):
                            ground_height = terrain_data.get_terrain_height(goal_position[0], goal_position[1])
                        else:
                            ground_height = 0
                        
                        # 画投影线（红色虚线）
                        ax.plot([goal_position[0], goal_position[0]], 
                               [goal_position[1], goal_position[1]],
                               [ground_height, goal_position[2]],
                               'r--', alpha=0.8, linewidth=1.5)
                    except Exception as e:
                        print(f"绘制目标投影线时出错: {e}")
                
                # 添加目标点文本
                goal_text = f"Target ({goal_position[0]:.1f}, {goal_position[1]:.1f}, {goal_position[2]:.1f})"
                ax.text(
                    goal_position[0], goal_position[1], goal_position[2] + 2,
                    goal_text,
                    color='red',
                    fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom'
                )
                
                # 添加到图例
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                      markersize=15, label='Target'))
            except Exception as e:
                print(f"绘制目标点时出错: {e}")
                traceback.print_exc()
        
        # 添加图例到图的右上角，colorbar在底部不会遮挡
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='upper right', 
                             frameon=True, framealpha=0.9, 
                             fontsize=10, facecolor='white',
                             edgecolor='black', fancybox=True)
            legend.set_zorder(100)  # 确保图例在最上层
        
        # 添加控制说明（类似用户示例图片）
        control_info = "视角控制:\n→ ← : 水平旋转\n↑ ↓ : 垂直旋转\nR : 重置视角\nH : 显示/隐藏"
        # 在图像左下角添加控制说明
        plt.figtext(0.05, 0.05, control_info, fontsize=10, 
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'),
                   color='white')
        
        # 添加标题，使用episode和step信息
        episode_info = "Episode 1/1, Step 2000/2000, Reward: 0.00"
        plt.figtext(0.5, 0.95, episode_info, fontsize=12, ha='center')
        
        # 调整坐标轴比例，确保图形不变形
        try:
            # 获取当前坐标轴范围
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
            
            # 计算最大范围
            max_range = max(float(x_range), float(y_range), float(z_range))
            
            mid_x = np.mean(ax.get_xlim())
            mid_y = np.mean(ax.get_ylim())
            mid_z = np.mean(ax.get_zlim())
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        except Exception as e:
            print(f"调整坐标轴时出错: {e}")
            traceback.print_exc()
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部标题留出空间
        
        # 保存图像
        if output_path:
            try:
                # 确保有足够时间渲染
                plt.pause(0.5)
                print(f"保存轨迹图到: {output_path}")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"轨迹图保存成功")
            except Exception as e:
                print(f"保存轨迹图出错: {e}")
                traceback.print_exc()
        
        # 返回图形对象以便进一步处理
        return fig
    except Exception as e:
        print(f"绘制轨迹图时发生错误: {e}")
        traceback.print_exc()
        return None

def save_trajectory_data(trajectories, save_path):
    """保存轨迹数据到npy文件"""
    try:
        import numpy as np
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存轨迹数据
        np.save(save_path, trajectories)
        print(f"轨迹数据已保存到: {save_path}")
        return True
    except Exception as e:
        print(f"保存轨迹数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_trajectory_files(args):
    """处理轨迹文件并生成可视化报告"""
    import matplotlib.pyplot as plt
    import importlib
    import os
    import numpy as np
    import traceback
    
    try:
        # 检查是否仅处理单个轨迹文件
        if args.load_trajectory is not None:
            # 检查文件是否存在
            if not os.path.exists(args.load_trajectory):
                print(f"错误: 轨迹文件不存在: {args.load_trajectory}")
                print(f"当前工作目录: {os.getcwd()}")
                return False
            
            print(f"仅处理轨迹文件模式: {args.load_trajectory}")
            
            # 创建场景对象用于地形数据
            try:
                scenario_module = importlib.import_module(f"multiagent.scenarios.{args.scenario}")
                scenario_class = scenario_module.Scenario
                scenario = scenario_class(random_terrain=args.random_terrain, 
                                        seed=args.terrain_seed)
                # 确保地形已生成
                if not hasattr(scenario, 'terrain') or scenario.terrain is None:
                    scenario.generate_terrain(is_random=args.random_terrain, seed=args.terrain_seed)
                
                print(f"加载场景 {args.scenario} 用于地形数据")
            except Exception as e:
                print(f"警告：加载场景模块失败，将不使用地形数据: {e}")
                traceback.print_exc()
                scenario = None
            
            # 设置输出路径
            output_path = args.save_trajectory_png or f"{os.path.splitext(args.load_trajectory)[0]}_analysis.png"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 加载轨迹数据并生成报告
            try:
                fig = load_and_plot_trajectories(
                    trajectory_file=args.load_trajectory, 
                    goal_position=scenario.goal_pos if hasattr(scenario, 'goal_pos') and scenario is not None else None,
                    terrain_data=scenario,
                    output_path=output_path
                )
                
                # 显示图表（除非明确指定不显示）
                if not args.no_visualization and fig is not None:
                    plt.show()
                    
                return True
            except Exception as e:
                print(f"加载和绘制轨迹时出错: {e}")
                traceback.print_exc()
                return False
        
        # 检查是否处理所有策略的轨迹
        if args.process_all_policies:
            print("处理所有策略的轨迹文件")
            
            # 创建场景对象用于地形数据
            try:
                scenario_module = importlib.import_module(f"multiagent.scenarios.{args.scenario}")
                scenario_class = scenario_module.Scenario
                scenario = scenario_class(random_terrain=args.random_terrain, 
                                         seed=args.terrain_seed)
                # 确保地形已生成
                if not hasattr(scenario, 'terrain') or scenario.terrain is None:
                    scenario.generate_terrain(is_random=args.random_terrain, seed=args.terrain_seed)
                
                print(f"加载场景 {args.scenario} 用于地形数据")
            except Exception as e:
                print(f"警告：加载场景模块失败，将不使用地形数据: {e}")
                traceback.print_exc()
                scenario = None
                
            # 找到模型目录
            model_dir = args.model_dir
            if model_dir is None:
                # 尝试使用场景名称作为目录
                default_model_dir = f"./experiments/maddpg/{args.scenario}"
                if os.path.exists(default_model_dir):
                    model_dir = default_model_dir
                    print(f"使用默认模型目录: {model_dir}")
                else:
                    # 寻找最近修改的目录
                    exp_dir = "./experiments/maddpg"
                    if os.path.exists(exp_dir):
                        dirs = [(d, os.path.getmtime(os.path.join(exp_dir, d))) 
                               for d in os.listdir(exp_dir) 
                               if os.path.isdir(os.path.join(exp_dir, d))]
                        
                        if dirs:
                            # 按修改时间排序
                            dirs.sort(key=lambda x: x[1], reverse=True)
                            latest_dir = dirs[0][0]
                            model_dir = os.path.join(exp_dir, latest_dir)
                            print(f"自动选择最近训练的模型目录: {model_dir}")
                        else:
                            print("错误: 未找到任何训练模型目录")
                            return False
                    else:
                        print(f"错误: 实验目录不存在: {exp_dir}")
                        return False
                        
            print(f"使用模型目录: {model_dir}")
            
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                print(f"错误: 模型目录不存在: {model_dir}")
                return False
            
            # 处理每个策略的轨迹
            policies = ["best_average", "best_overall", "last"]
            for policy in policies:
                trajectory_file = os.path.join(model_dir, f"trajectories_{policy}.npy")
                
                if os.path.exists(trajectory_file):
                    print(f"处理 {policy} 策略的轨迹文件: {trajectory_file}")
                    
                    # 设置输出路径
                    output_path = f"{os.path.splitext(trajectory_file)[0]}_analysis.png"
                    
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # 加载轨迹数据并生成报告
                    try:
                        fig = load_and_plot_trajectories(
                            trajectory_file=trajectory_file, 
                            goal_position=scenario.goal_pos if hasattr(scenario, 'goal_pos') and scenario is not None else None,
                            terrain_data=scenario,
                            output_path=output_path
                        )
                        
                        # 显示图表（除非明确指定不显示或处理多个策略）
                        if not args.no_visualization and fig is not None and len(policies) == 1:
                            plt.show()
                        else:
                            plt.close(fig)
                    except Exception as e:
                        print(f"处理 {policy} 策略轨迹时出错: {e}")
                        traceback.print_exc()
                else:
                    print(f"警告: 未找到 {policy} 策略的轨迹文件: {trajectory_file}")
            
            return True
        
        # 没有处理轨迹文件
        return False
    except Exception as e:
        print(f"process_trajectory_files函数执行出错: {e}")
        traceback.print_exc()
        return False

def debug_trajectories(trajectories, title="轨迹数据调试"):
    """调试轨迹数据结构的辅助函数"""
    print(f"\n===== {title} =====")
    print(f"轨迹总数: {len(trajectories)}")
    
    for i, agent_traj in enumerate(trajectories):
        print(f"  智能体{i}:")
        if isinstance(agent_traj, list) or isinstance(agent_traj, np.ndarray):
            print(f"    类型: {type(agent_traj)}")
            print(f"    长度: {len(agent_traj)}")
            
            # 检查第一层嵌套
            if len(agent_traj) > 0:
                first_elem = agent_traj[0]
                print(f"    第一个元素类型: {type(first_elem)}")
                
                # 检查第二层嵌套
                if isinstance(first_elem, list) or isinstance(first_elem, np.ndarray):
                    print(f"    第一个元素长度: {len(first_elem)}")
                    
                    # 检查第三层嵌套
                    if len(first_elem) > 0:
                        second_elem = first_elem[0]
                        print(f"    第一个元素的第一个元素类型: {type(second_elem)}")
                        
                        if isinstance(second_elem, np.ndarray):
                            print(f"    形状: {second_elem.shape}")
                        elif isinstance(second_elem, list):
                            print(f"    长度: {len(second_elem)}")
                            if len(second_elem) > 0:
                                print(f"    前三个值: {second_elem[:3]}")
                        else:
                            print(f"    值: {second_elem}")
        else:
            print(f"    非列表/数组类型: {type(agent_traj)}")
    
    print("=" * (len(title) + 13))

def load_models(maddpg_instance, args):
    """加载模型权重"""
    # 确定模型路径
    model_dir = args.model_dir if args.model_dir else "./weights"
    
    # 确定后缀
    suffix = args.model_suffix if args.model_suffix else ""
    
    # 确定策略类型对应的目录名
    policy_suffix = ""
    if args.policy == "best_overall":
        policy_suffix = "best"
    elif args.policy == "best_average":
        policy_suffix = "best_avg"
    elif args.policy == "last":
        policy_suffix = "final"
    
    # 构建完整的模型路径
    model_path = os.path.join(model_dir, f"model_{policy_suffix}{suffix}")
    print(f"尝试加载模型: {model_path}")
    
    # 尝试使用新的保存格式加载模型
    print(f"尝试使用新的保存格式加载模型，后缀: _{policy_suffix}")
    
    # 加载每个智能体的模型
    success = True
    for i in range(len(maddpg_instance.agents)):
        # 尝试三种不同的目录命名格式
        agent_dir_with_underscore = f"{model_dir}/maddpg_{i}_{policy_suffix}"  # 有下划线的格式
        agent_dir_without_underscore = f"{model_dir}/maddpg_{i}{policy_suffix}"  # 无下划线的格式
        agent_dir_ep = f"{model_dir}/maddpg_{i}{policy_suffix}ep"  # 用于训练期间保存的临时模型
        
        # 先检查带下划线的路径是否存在
        if os.path.exists(agent_dir_with_underscore):
            agent_dir = agent_dir_with_underscore
            print(f"加载智能体 {i} 模型从 {agent_dir}")
        # 如果不存在，检查不带下划线的路径
        elif os.path.exists(agent_dir_without_underscore):
            agent_dir = agent_dir_without_underscore
            print(f"加载智能体 {i} 模型从 {agent_dir}")
        # 最后检查临时模型路径
        elif os.path.exists(agent_dir_ep):
            agent_dir = agent_dir_ep
            print(f"加载智能体 {i} 模型从临时保存目录 {agent_dir}")
        else:
            # 都不存在，报错
            print(f"错误: 智能体 {i} 模型目录不存在: {agent_dir_with_underscore} 或 {agent_dir_without_underscore} 或 {agent_dir_ep}")
            success = False
            continue
        
        # 尝试加载模型
        try:
            # 检查目录内容
            print(f"检查目录内容: {agent_dir}")
            files = os.listdir(agent_dir)
            print(f"目录内容: {files}")
            
            # 打印当前模型结构信息
            actor_model = maddpg_instance.agents[i]['actor']
            print("\n当前Actor模型结构信息:")
            print(f"层数量: {len(actor_model.layers)}")
            layer_names = [layer.name for layer in actor_model.layers]
            print(f"层名称: {layer_names}")
            
            # 打印每层的输出形状
            for j, layer in enumerate(actor_model.layers):
                print(f"  层 {j}: {layer.name}, 输出形状: {layer.output_shape}")
            print()
            
            # 尝试多种文件格式 - 适配paper3d_train.py中的保存格式
            actor_weights_files = [
                os.path.join(agent_dir, "actor.weights.h5"),
                os.path.join(agent_dir, "actor_weights.h5"),
                os.path.join(agent_dir, "actor.h5")
            ]
            
            critic_weights_files = [
                os.path.join(agent_dir, "critic.weights.h5"),
                os.path.join(agent_dir, "critic_weights.h5"),
                os.path.join(agent_dir, "critic.h5")
            ]
            
            target_actor_weights_files = [
                os.path.join(agent_dir, "target_actor.weights.h5"),
                os.path.join(agent_dir, "target_actor_weights.h5"),
                os.path.join(agent_dir, "target_actor.h5")
            ]
            
            target_critic_weights_files = [
                os.path.join(agent_dir, "target_critic.weights.h5"),
                os.path.join(agent_dir, "target_critic_weights.h5"),
                os.path.join(agent_dir, "target_critic.h5")
            ]
            
            # 加载Actor模型
            actor_loaded = False
            for actor_file in actor_weights_files:
                if os.path.exists(actor_file):
                    try:
                        maddpg_instance.agents[i]['actor'].load_weights(actor_file)
                        print(f"成功加载Actor权重: {actor_file}")
                        actor_loaded = True
                        break
                    except Exception as e:
                        print(f"尝试加载{actor_file}失败: {e}")
                        
            if not actor_loaded:
                print(f"无法找到可用的Actor权重文件")
                success = False
            
            # 加载Critic模型
            critic_loaded = False
            for critic_file in critic_weights_files:
                if os.path.exists(critic_file):
                    try:
                        maddpg_instance.agents[i]['critic'].load_weights(critic_file)
                        print(f"成功加载Critic权重: {critic_file}")
                        critic_loaded = True
                        break
                    except Exception as e:
                        print(f"尝试加载{critic_file}失败: {e}")
                        
            if not critic_loaded:
                print(f"无法找到可用的Critic权重文件")
                success = False
            
            # 加载Target Actor模型
            target_actor_loaded = False
            for target_actor_file in target_actor_weights_files:
                if os.path.exists(target_actor_file):
                    try:
                        maddpg_instance.agents[i]['target_actor'].load_weights(target_actor_file)
                        print(f"成功加载Target Actor权重: {target_actor_file}")
                        target_actor_loaded = True
                        break
                    except Exception as e:
                        print(f"尝试加载{target_actor_file}失败: {e}")
            
            # 如果找不到Target Actor权重，从Actor复制
            if not target_actor_loaded:
                print(f"找不到Target Actor权重文件，从Actor复制权重")
                maddpg_instance.agents[i]['target_actor'].set_weights(
                    maddpg_instance.agents[i]['actor'].get_weights()
                )
            
            # 加载Target Critic模型
            target_critic_loaded = False
            for target_critic_file in target_critic_weights_files:
                if os.path.exists(target_critic_file):
                    try:
                        maddpg_instance.agents[i]['target_critic'].load_weights(target_critic_file)
                        print(f"成功加载Target Critic权重: {target_critic_file}")
                        target_critic_loaded = True
                        break
                    except Exception as e:
                        print(f"尝试加载{target_critic_file}失败: {e}")
            
            # 如果找不到Target Critic权重，从Critic复制
            if not target_critic_loaded:
                print(f"找不到Target Critic权重文件，从Critic复制权重")
                maddpg_instance.agents[i]['target_critic'].set_weights(
                    maddpg_instance.agents[i]['critic'].get_weights()
                )
                
        except Exception as e:
            print(f"加载智能体 {i} 模型时发生错误: {str(e)}")
            traceback.print_exc()
            success = False
    
    if success:
        print(f"成功加载所有智能体模型: {policy_suffix}")
        return True
    else:
        print(f"加载模型失败: {policy_suffix}")
        return False

if __name__ == "__main__":
    import traceback  # 确保在这里也导入一次
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()

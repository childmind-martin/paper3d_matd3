"""
轨迹可视化模块
提供3D轨迹可视化、GIF生成等功能
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适合服务器环境
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import datetime
import traceback


class TrajectoryVisualizer:
    """轨迹可视化器，统一处理所有可视化任务"""
    
    def __init__(self, figsize=(12, 10), dpi=300, verbose=False):
        """初始化可视化器
        
        参数:
            figsize: 图形大小
            dpi: 图形分辨率
            verbose: 是否打印详细日志
        """
        self.figsize = figsize
        self.dpi = dpi
        self.verbose = verbose
        self.agent_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
        self._setup_matplotlib()
        
        # 抑制特定的 UserWarning
        import warnings
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    def _log(self, msg):
        """仅在verbose模式下打印日志"""
        if self.verbose:
            print(msg)
    
    def _find_chinese_font(self):
        """查找可用的中文字体"""
        try:
            from matplotlib.font_manager import FontManager
            fm = FontManager()
            # 常见的简体中文字体名称
            font_list = ['SimHei', 'Microsoft YaHei', 'Heiti SC', 'WenQuanYi Zen Hei', 'Source Han Sans SC']
            for font_name in font_list:
                if any(font.name == font_name for font in fm.ttflist):
                    return font_name
        except Exception:
            pass
        return None

    def _setup_matplotlib(self):
        """设置matplotlib参数，优先使用中文字体"""
        chinese_font = self._find_chinese_font()
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial', 'Helvetica']
            self._log(f"✅ 已配置中文字体支持: {chinese_font}")
        else:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
            self._log("⚠️ 未找到中文字体，绘图中的中文可能无法正常显示")
        plt.rcParams['axes.unicode_minus'] = False
    
    def _normalize_plotly_symbol(self, symbol):
        """将输入的marker symbol规范为Plotly Scatter3d支持的集合。
        兼容常见别名/非法值，默认回退为 'diamond'。
        支持集合: {'circle','circle-open','cross','diamond','diamond-open','square','square-open','x'}
        """
        allowed = {
            'circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x'
        }
        try:
            if isinstance(symbol, str):
                s = symbol.lower()
                if s in allowed:
                    return s
                alias_map = {
                    '*': 'diamond',
                    'star': 'diamond',
                    'o': 'circle',
                    'dot': 'circle',
                    '.': 'circle',
                    'x-open': 'x',
                    '^': 'diamond',
                    'triangle': 'x',
                    'triangle-up': 'x',
                    'pentagon': 'square',
                    'hexagon': 'square'
                }
                return alias_map.get(s, 'diamond')
        except Exception:
            pass
        return 'diamond'
    
    def _process_trajectory_data(self, trajectories):
        """处理轨迹数据，支持多种格式
        
        参数:
            trajectories: 轨迹数据，可能的格式:
                        - [agent][timestep][xyz]
                        - [timestep][agent][xyz]
        返回:
            标准化的轨迹数据: [agent][timestep][xyz]
        """
        if not trajectories:
            return []
        
        try:
            # 基于维度的稳健判定：比较外层长度（可能是时间步数）与内层长度（可能是智能体数）
            if isinstance(trajectories, list) and len(trajectories) > 0 and isinstance(trajectories[0], list):
                outer_len = len(trajectories)                # 可能是时间步数
                inner_len = len(trajectories[0])            # 可能是智能体数 或 时间步数
                inner_item = trajectories[0][0] if inner_len > 0 else None
                inner_item_is_point = isinstance(inner_item, (list, np.ndarray)) and len(inner_item) in (3,)
                
                # 情况A：很可能是 [timestep][agent][xyz]
                if inner_item_is_point and outer_len >= max(50, inner_len * 5):
                    print(f"检测到 [timestep][agent][xyz] 格式，正在转置... (time_steps={outer_len}, agents={inner_len})")
                    num_agents = inner_len
                    agent_trajectories = [[] for _ in range(num_agents)]
                    for timestep_data in trajectories:
                        for agent_idx, agent_pos in enumerate(timestep_data):
                            if agent_idx < num_agents and agent_pos is not None:
                                agent_trajectories[agent_idx].append(agent_pos)
                    print(f"转置完成: {len(agent_trajectories)}个智能体; 每条轨迹长度示例: {len(agent_trajectories[0]) if agent_trajectories and agent_trajectories[0] else 0}")
                    return agent_trajectories
                
                # 情况B：很可能是 [agent][timestep][xyz]（无需转置）
                if inner_item_is_point and inner_len >= max(50, outer_len * 5):
                    print(f"检测到 [agent][timestep][xyz] 格式 (agents={outer_len}, time_steps≈{inner_len})")
                    return trajectories
                
                # 情况C：不明显，回退到类型判定
                if inner_item_is_point:
                    # 偏向于 [timestep][agent][xyz]，因为更常见
                    print(f"检测到嵌套点结构，默认按 [timestep][agent][xyz] 处理并转置... (outer={outer_len}, inner={inner_len})")
                    num_agents = inner_len
                    agent_trajectories = [[] for _ in range(num_agents)]
                    for timestep_data in trajectories:
                        for agent_idx, agent_pos in enumerate(timestep_data):
                            if agent_idx < num_agents and agent_pos is not None:
                                agent_trajectories[agent_idx].append(agent_pos)
                    return agent_trajectories
            
            # 默认返回原数据
            return trajectories
        except Exception as e:
            print(f"轨迹数据处理错误: {e}")
            traceback.print_exc()
            return trajectories

    def _trim_effective_traj(self, traj, eps=1e-6):
        """裁剪单条轨迹的有效步：
        - 去除末尾的静止段（尾部重复点）
        - 去除相邻重复点（避免完全重叠导致的视觉“拉长”）
        输入/输出: List[[x,y,z]]
        """
        try:
            if not traj or len(traj) == 0:
                return traj
            # 末尾裁剪：找到最后一次发生位移的位置
            last_idx = len(traj) - 1
            # 基准点：从尾部回溯，第一 个与其前一点不同的位置即为有效截止
            # 先寻找最后一个“发生变化”的索引
            base = np.asarray(traj[last_idx], dtype=np.float32)
            cutoff = last_idx
            for k in range(last_idx - 1, -1, -1):
                p = np.asarray(traj[k], dtype=np.float32)
                if np.linalg.norm(p - base) > eps:
                    cutoff = k + 1
                    break
            trimmed = traj[:max(1, cutoff)]
            # 相邻去重
            deduped = []
            prev = None
            for p in trimmed:
                cur = np.asarray(p, dtype=np.float32)
                if prev is None or np.linalg.norm(cur - prev) > eps:
                    deduped.append(p)
                    prev = cur
            return deduped if len(deduped) > 0 else trimmed
        except Exception:
            return traj
    
    def generate_trajectory_image(self, trajectories, scenario, save_path, 
                                 episode_num, reward, episode_type='current',
                                 correction_type=None, elev=30, azim=45,
                                 goal_positions=None, env_instance=None, actor_outputs_history=None,
                                 env_idx=0, title_step_note=None):
        """生成轨迹静态图像
        
        参数:
            trajectories: 智能体轨迹列表
            scenario: 场景对象
            save_path: 保存路径
            episode_num: 回合编号
            reward: 回合奖励
            episode_type: 回合类型 ('current' 或 'best')
            correction_type: 动作修正类型
            elev: 视角俯仰角
            azim: 视角方位角
            goal_positions: 目标位置信息
            env_instance: 环境实例（用于获取与轨迹一致的地形和目标数据）
            actor_outputs_history: Actor网络7维输出历史数据 [timestep][agent][7_dim]
        """
        try:
            self._log(f"🎨 开始生成轨迹图像: {save_path}")
            # 统计有效步（按各智能体裁剪后的最大长度）
            processed = self._process_trajectory_data(trajectories)
            effective_lengths = [len(self._trim_effective_traj(t)) for t in processed] if processed else []
            eff_steps = max(effective_lengths) if effective_lengths else len(trajectories)
            self._log(f"🎨 轨迹数据: 有效步≈{eff_steps}")
            self._log(f"🎨 场景类型: {type(scenario)}")
            self._log(f"🎨 目标位置: {goal_positions}")
            self._log(f"🎨 环境实例: {type(env_instance) if env_instance else 'None'}")
            
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            
            # 绘制地形 - 优先使用环境实例的地形数据
            if env_instance is not None:
                self._log(f"🎨 使用环境实例的地形数据")
                self._plot_terrain_from_env(ax, env_instance)
            else:
                self._log(f"🎨 使用场景参数的地形数据")
                self._plot_terrain(ax, scenario)
            
            # 绘制障碍（优先实时环境）
            self._plot_obstacles(ax, scenario, env_instance=env_instance, env_idx=env_idx)
            
            # 绘制目标位置 - 优先使用环境实例的目标数据
            if env_instance is not None:
                self._log(f"🎨 使用环境实例的目标数据")
                self._plot_goal_from_env(ax, env_instance, goal_positions)
            else:
                self._log(f"🎨 使用场景参数的目标数据")
                self._plot_goal(ax, scenario, goal_positions=goal_positions)
            
            # 绘制智能体轨迹
            self._plot_trajectories(ax, trajectories)
            
            # 计算总步数和有效步数
            total_steps = len(trajectories) if trajectories else 0
            
            # 设置标签和标题
            self._set_labels_and_title(
                ax,
                episode_type,
                episode_num,
                reward,
                correction_type,
                total_steps=total_steps,
                effective_steps=eff_steps,
                title_step_note=title_step_note,
            )
            
            # 新增：如果有Actor输出历史数据，生成包含Actor输出分析的图像
            if actor_outputs_history is not None and len(actor_outputs_history) > 0:
                try:
                    self._log(f"🎨 检测到Actor输出历史数据，长度: {len(actor_outputs_history)}")
                    # 创建包含3D轨迹和Actor输出分析的复合图像
                    self._generate_trajectory_with_actor_analysis(
                        trajectories, scenario, save_path, episode_num, reward, 
                        episode_type, goal_positions, env_instance, actor_outputs_history,
                        total_steps, eff_steps, elev, azim, env_idx=env_idx
                    )
                    self._log(f"✅ 轨迹图像（含Actor分析）已保存: {save_path}")
                    self._log(f"📊 轨迹统计: 总步数={total_steps}, 有效步数={eff_steps}")
                    return
                except Exception as e:
                    self._log(f"⚠️ Actor分析图像生成失败，回退到标准图像: {e}")
                    import traceback
                    if self.verbose:
                        traceback.print_exc()
                    # 继续执行标准图像生成
            else:
                self._log(f"🎨 未检测到Actor输出历史数据，生成标准图像")
            
            # 保存图像
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            self._log(f"✅ 轨迹图像已保存: {save_path}")
            self._log(f"📊 轨迹统计: 总步数={total_steps}, 有效步数={eff_steps}")
            
        except Exception as e:
            self._log(f"❌ 生成轨迹图像失败: {e}")
            if self.verbose:
                traceback.print_exc()
            if 'fig' in locals():
                plt.close(fig)
    
    def generate_trajectory_gif(self, trajectories, scenario, save_path,
                               episode_num, reward, episode_type='current',
                               fps=3, duration=None, goal_positions=None, gif_max_frames=60):
        """生成轨迹动画GIF（按照paper3d_train.py的逻辑）
        
        参数:
            trajectories: 智能体轨迹列表
            scenario: 场景对象
            save_path: 保存路径
            episode_num: 回合编号
            reward: 回合奖励
            episode_type: 回合类型
            fps: 帧率（默认3fps）
            duration: 保留参数，不使用
        """
        try:
            # 处理轨迹数据格式
            processed_trajectories = self._process_trajectory_data(trajectories)
            # 裁剪有效步
            processed_trajectories = [self._trim_effective_traj(t) for t in processed_trajectories]
            
            if not processed_trajectories or not any(traj for traj in processed_trajectories):
                print("轨迹数据为空，跳过GIF生成")
                return
            
            # 按照paper3d_train.py的逻辑
            max_traj_points = max([len(traj) for traj in processed_trajectories if traj]) if processed_trajectories else 0
            
            if max_traj_points == 0:
                print("轨迹数据为空，跳过GIF生成")
                return
                
            # 增强轨迹密度（插值） - 按照原版逻辑
            enhanced_trajectories = []
            for traj in processed_trajectories:
                if not traj or len(traj) < 2:
                    enhanced_trajectories.append(traj)
                    continue
                    
                if len(traj) < 50:
                    enhanced_traj = []
                    for i in range(len(traj) - 1):
                        p1 = traj[i]
                        p2 = traj[i+1]
                        enhanced_traj.append(p1)
                        
                        # 在两点之间添加插值点
                        for j in range(1, 10):
                            ratio = j / 10
                            interp_point = [
                                p1[0] + (p2[0] - p1[0]) * ratio,
                                p1[1] + (p2[1] - p1[1]) * ratio,
                                p1[2] + (p2[2] - p1[2]) * ratio
                            ]
                            enhanced_traj.append(interp_point)
                    enhanced_traj.append(traj[-1])
                    enhanced_trajectories.append(enhanced_traj)
                else:
                    enhanced_trajectories.append(traj)
            
            # 重新计算最大轨迹点数
            max_traj_points = max([len(traj) for traj in enhanced_trajectories if traj]) if enhanced_trajectories else 0
            
            # 按照原版逻辑：最多60帧，但根据轨迹点数调整
            frame_count = int(gif_max_frames) if gif_max_frames is not None else 60
            step_size = max(1, max_traj_points // frame_count)
            
            # 计算实际会生成的帧数
            actual_frames = len(range(0, max_traj_points, step_size))
            
            print(f"正在生成GIF: 实际{actual_frames}帧 (目标{frame_count}帧), 最大轨迹点数: {max_traj_points}, 步长: {step_size}")
            if actual_frames < frame_count:
                print(f"  注意：由于轨迹点数较少({max_traj_points})，实际生成{actual_frames}帧而非{frame_count}帧")
            
            # 按照原版逻辑生成帧
            gif_frames = []
            
            # 使用固定的智能体颜色
            agent_colors = self.agent_colors  # 🔧 修复：使用统一的颜色方案
            
            for frame in range(0, max_traj_points, step_size):
                try:
                    # 创建图形
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 固定视角（按照原版paper3d_train.py）
                    ax.view_init(elev=30, azim=45)
                    
                    # 绘制地形、障碍和目标（抑制内部print，避免逐帧刷屏）
                    import builtins as _bi
                    _old_print = _bi.print
                    try:
                        _bi.print = lambda *args, **kwargs: None
                        self._plot_terrain(ax, scenario, alpha=0.3)
                        self._plot_obstacles(ax, scenario)
                        self._plot_goal(ax, scenario, goal_positions=goal_positions)
                    finally:
                        _bi.print = _old_print
                    
                    # 绘制轨迹（到当前帧）
                    for i, traj in enumerate(enhanced_trajectories):
                        if i >= 3 or not traj:  # 只显示前3个智能体
                            continue
                        
                        color = agent_colors[i % len(agent_colors)]
                        current_frame_traj = traj[:min(frame + 1, len(traj))]
                        
                        if len(current_frame_traj) > 0:
                            xs = [p[0] for p in current_frame_traj if p is not None and len(p) >= 3]
                            ys = [p[1] for p in current_frame_traj if p is not None and len(p) >= 3]
                            zs = [p[2] for p in current_frame_traj if p is not None and len(p) >= 3]
                            
                            if xs:
                                # 绘制轨迹线
                                if len(xs) > 1:
                                    ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=0.8, label=f'Agent {i}')
                                # 当前位置
                                ax.scatter(xs[-1], ys[-1], zs[-1], color=color, s=120, marker='o', 
                                          edgecolors='black', linewidth=2)
                    
                    # 设置标题
                    progress_percent = int(frame * 100 / max_traj_points)
                    ax.set_title(f'{episode_type.capitalize()} Episode {episode_num} - Step {frame}/{max_traj_points} ({progress_percent}%)', 
                               fontsize=12, fontfamily='sans-serif')
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    
                    # 添加图例
                    legend_handles = [plt.Line2D([0], [0], color=agent_colors[j % len(agent_colors)], lw=4) 
                                     for j in range(min(3, len(enhanced_trajectories))) if enhanced_trajectories[j]]
                    legend_labels = [f'Agent {j}' for j in range(min(3, len(enhanced_trajectories))) 
                                   if enhanced_trajectories[j]]
                    if legend_handles:
                        ax.legend(handles=legend_handles, labels=legend_labels, fontsize=10, 
                                framealpha=0.8, edgecolor='black')
                    
                    plt.tight_layout()
                    
                    # 转换为图像（按照原版逻辑）
                    fig.canvas.draw()
                    width, height = fig.canvas.get_width_height()
                    buf = fig.canvas.renderer.buffer_rgba()
                    image = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
                    image = image[:, :, :3]  # 只保留RGB通道
                    gif_frames.append(image)
                    
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"警告: 帧{frame}生成失败: {e}")
                    continue
                    
                # 显示进度
                if len(gif_frames) % 10 == 0 and len(gif_frames) > 0:
                    print(f"  已生成 {len(gif_frames)} 帧...")
            
            # 保存GIF（按照原版fps=3）
            if gif_frames:
                print(f"正在保存GIF，共{len(gif_frames)}帧，fps={fps}...")
                imageio.mimsave(save_path, gif_frames, fps=fps, loop=0)
                print(f"✅ GIF动画已保存到: {save_path}")
            else:
                print("⚠️ 没有生成有效帧，跳过GIF保存")
            
        except Exception as e:
            print(f"生成GIF失败: {e}")
            traceback.print_exc()
    
    def _plot_terrain(self, ax, scenario, alpha=0.7):
        """绘制地形"""
        if hasattr(scenario, 'terrain') and scenario.terrain is not None:
            terrain_data = np.asarray(scenario.terrain)
            terrain_h, terrain_w = terrain_data.shape[:2]
            map_size = getattr(scenario, 'map_size', None)
            if map_size is not None and map_size > max(terrain_h, terrain_w):
                # 地形被降采样，按照原地图尺寸拉伸坐标
                x = np.linspace(0, map_size - 1, terrain_w)
                y = np.linspace(0, map_size - 1, terrain_h)
            else:
                x = np.arange(terrain_w)
                y = np.arange(terrain_h)
            X, Y = np.meshgrid(x, y, indexing='xy')
            
            self._log(f"[地形绘制] 坐标范围: X=[{np.min(X):.1f}, {np.max(X):.1f}], Y=[{np.min(Y):.1f}, {np.max(Y):.1f}]")
            self._log(f"[地形绘制] 地形形状: {terrain_data.shape}, 高度范围: [{np.min(terrain_data):.1f}, {np.max(terrain_data):.1f}]")
            
            ax.plot_surface(X, Y, terrain_data, cmap='terrain', alpha=alpha, antialiased=True)

    def _plot_terrain_from_env(self, ax, env_instance, alpha=0.7):
        """从环境实例绘制地形，确保与轨迹数据一致"""
        try:
            self._log(f"🎨 从环境实例绘制地形")
            
            # 尝试从环境实例获取地形数据
            terrain_data = None
            map_size = None
            
            # 方法1：从环境实例的world获取
            if hasattr(env_instance, 'world') and hasattr(env_instance.world, 'terrain'):
                terrain_data = env_instance.world.terrain
                if hasattr(env_instance.world, 'map_size'):
                    map_size = env_instance.world.map_size
                self._log(f"🎨 从 env.world 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            
            # 方法2：从环境实例的scenario获取
            elif hasattr(env_instance, 'scenario') and hasattr(env_instance.scenario, 'terrain'):
                terrain_data = env_instance.scenario.terrain
                if hasattr(env_instance.scenario, 'map_size'):
                    map_size = env_instance.scenario.map_size
                self._log(f"🎨 从 env.scenario 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            
            # 方法3：从环境实例直接获取
            elif hasattr(env_instance, 'terrain'):
                terrain_data = env_instance.terrain
                if hasattr(env_instance, 'map_size'):
                    map_size = env_instance.map_size
                self._log(f"🎨 从 env 直接获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            
            # 方法4：从并行环境的子环境获取（ParallelEnv）
            elif hasattr(env_instance, 'envs') and len(env_instance.envs) > 0:
                self._log(f"🎨 检测到并行环境，尝试从子环境获取地形数据")
                sub_env = env_instance.envs[0]  # 使用第一个子环境
                if hasattr(sub_env, 'world') and hasattr(sub_env.world, 'terrain'):
                    terrain_data = sub_env.world.terrain
                    if hasattr(sub_env.world, 'map_size'):
                        map_size = sub_env.world.map_size
                    self._log(f"🎨 从子环境 env.world 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
                elif hasattr(sub_env, 'scenario') and hasattr(sub_env.scenario, 'terrain'):
                    terrain_data = sub_env.scenario.terrain
                    if hasattr(sub_env.scenario, 'map_size'):
                        map_size = sub_env.scenario.map_size
                    self._log(f"🎨 从子环境 env.scenario 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
                elif hasattr(sub_env, 'terrain'):
                    terrain_data = sub_env.terrain
                    if hasattr(sub_env, 'map_size'):
                        map_size = sub_env.map_size
                    self._log(f"🎨 从子环境 env 直接获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            
            # 方法5：通过ParallelEnv的get_terrain_data方法获取
            elif hasattr(env_instance, 'get_terrain_data'):
                self._log(f"🎨 通过ParallelEnv.get_terrain_data获取地形数据")
                terrain_result = env_instance.get_terrain_data(0)  # 使用第一个子环境
                if isinstance(terrain_result, dict) and terrain_result.get('terrain') is not None:
                    terrain_data = terrain_result['terrain']
                    map_size = terrain_result.get('map_size')
                    self._log(f"🎨 通过get_terrain_data获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
                else:
                    self._log(f"🎨 get_terrain_data返回空结果: {terrain_result}")
            
            if terrain_data is not None:
                # 🔧 关键修复：根据地形数据的实际形状创建网格，而不是使用map_size
                # 如果地形已降采样（50×50），网格也应该是50×50，但坐标需要映射到0-200范围
                terrain_h, terrain_w = terrain_data.shape[0], terrain_data.shape[1]
                
                # 如果map_size存在且大于地形尺寸，说明地形已降采样，需要将坐标映射到原始范围
                if map_size is not None and map_size > terrain_w:
                    # 地形已降采样，创建与地形尺寸匹配的网格，但坐标映射到原始范围
                    # 例如：50×50地形，坐标范围0-200，每4个单位对应一个地形点
                    x = np.linspace(0, map_size - 1, terrain_w)
                    y = np.linspace(0, map_size - 1, terrain_h)
                else:
                    # 地形未降采样，直接使用地形尺寸
                    x = np.arange(terrain_w)
                    y = np.arange(terrain_h)
                
                X, Y = np.meshgrid(x, y, indexing='xy')  # 明确指定 indexing='xy'
                
                self._log(f"[环境地形绘制] 坐标范围: X=[{np.min(X):.1f}, {np.max(X):.1f}], Y=[{np.min(Y):.1f}, {np.max(Y):.1f}]")
                self._log(f"[环境地形绘制] 地形形状: {terrain_data.shape}, 高度范围: [{np.min(terrain_data):.1f}, {np.max(terrain_data):.1f}]")
                
                # 找到地形中的最高点并标记
                max_height_idx = np.unravel_index(np.argmax(terrain_data), terrain_data.shape)
                y_idx, x_idx = max_height_idx  # numpy.unravel_index 返回 (row, col) = (y, x)
                max_height = terrain_data[y_idx, x_idx]
                
                self._log(f"🔍 环境地形最高点: terrain[{y_idx}, {x_idx}] = {max_height:.1f}")
                self._log(f"🔍 环境地形最高点坐标: ({x_idx}, {y_idx}, {max_height:.1f})")
                
                # 在地形最高点绘制一个大的红色X标记
                ax.scatter(x_idx, y_idx, max_height, color='red', marker='X', s=2000,
                           edgecolors='black', linewidth=3, zorder=10, label='环境地形最高点')
                
                # 绘制地形
                terrain_plot = ax.plot_surface(X, Y, terrain_data, cmap='terrain',
                                              alpha=alpha, antialiased=True)
                
                self._log(f"✅ 环境地形绘制完成")
            else:
                self._log(f"❌ 无法从环境实例获取地形数据")
                
        except Exception as e:
            self._log(f"❌ 从环境实例绘制地形失败: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

    def _normalize_obstacles(self, raw_obstacles):
        """将障碍物统一规范为 [{'center':[x,y,z], 'radius':r}, ...]。"""
        normalized = []
        try:
            if not raw_obstacles:
                return normalized
            for ob in raw_obstacles:
                try:
                    center = None
                    radius = None
                    if isinstance(ob, dict):
                        center = ob.get('center', ob.get('pos', ob.get('position', None)))
                        radius = ob.get('radius', ob.get('r', ob.get('size', None)))
                    else:
                        state = getattr(ob, 'state', None)
                        p_pos = getattr(state, 'p_pos', None)
                        center = getattr(ob, 'center', getattr(ob, 'pos', getattr(ob, 'position', p_pos)))
                        radius = getattr(ob, 'radius', getattr(ob, 'r', getattr(ob, 'size', None)))
                    if center is None or radius is None:
                        continue
                    c = np.asarray(center, dtype=np.float32).reshape(-1)
                    if c.shape[0] < 3:
                        continue
                    r = float(radius)
                    if r <= 0:
                        continue
                    normalized.append({
                        'center': [float(c[0]), float(c[1]), float(c[2])],
                        'radius': r
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return normalized

    def _get_obstacles_for_plot(self, scenario=None, env_instance=None, env_idx=0):
        """优先从实时环境获取障碍物，失败时回退到场景快照。"""
        # 1) 首选：env_instance.get_vis_bundle（训练/评估时最可靠）
        try:
            if env_instance is not None and hasattr(env_instance, 'get_vis_bundle'):
                vb = env_instance.get_vis_bundle(int(env_idx))
                if isinstance(vb, dict):
                    obs = self._normalize_obstacles(vb.get('obstacles', []))
                    if obs:
                        return obs
        except Exception:
            pass

        # 2) 回退：从env/world对象提取
        try:
            worlds = []
            if env_instance is not None:
                if hasattr(env_instance, 'world'):
                    worlds.append(env_instance.world)
                if hasattr(env_instance, 'env') and hasattr(env_instance.env, 'world'):
                    worlds.append(env_instance.env.world)
                if hasattr(env_instance, 'envs') and isinstance(getattr(env_instance, 'envs', None), list):
                    for sub_env in env_instance.envs:
                        if hasattr(sub_env, 'world'):
                            worlds.append(sub_env.world)
            for w in worlds:
                raw = getattr(w, 'obstacles', None)
                obs = self._normalize_obstacles(raw)
                if obs:
                    return obs
        except Exception:
            pass

        # 3) 最后回退：scenario.obstacles（可能是静态快照）
        try:
            raw = getattr(scenario, 'obstacles', None) if scenario is not None else None
            return self._normalize_obstacles(raw)
        except Exception:
            return []

    def _plot_obstacles(self, ax, scenario, alpha=0.6, env_instance=None, env_idx=0):
        """绘制障碍（优先实时环境，回退场景快照）。"""
        try:
            obstacles = self._get_obstacles_for_plot(scenario=scenario, env_instance=env_instance, env_idx=env_idx)
            if len(obstacles) == 0:
                return
            import numpy as _np
            # 低分辨率网格，避免过重
            u = _np.linspace(0, 2 * _np.pi, 24)
            v = _np.linspace(0, _np.pi, 16)
            for ob in obstacles:
                try:
                    center = _np.asarray(ob.get('center', [0, 0, 0]), dtype=_np.float32)
                    radius = float(ob.get('radius', 0.0))
                    if radius <= 0:
                        continue
                    # 球面参数化
                    uu, vv = _np.meshgrid(u, v)
                    xs = radius * _np.cos(uu) * _np.sin(vv) + center[0]
                    ys = radius * _np.sin(uu) * _np.sin(vv) + center[1]
                    zs = radius * _np.cos(vv) + center[2]
                    ax.plot_surface(xs, ys, zs, color='red', alpha=alpha, linewidth=0, shade=True)
                except Exception:
                    continue
        except Exception:
            pass
    
    def _plot_goal(self, ax, scenario, goal_positions=None):
        """绘制目标位置"""
        try:
            self._log(f"🔍 TrajectoryVisualizer._plot_goal 被调用")
            self._log(f"🔍 goal_positions 类型: {type(goal_positions)}")
            self._log(f"🔍 goal_positions 内容: {goal_positions}")
            self._log(f"🔍 scenario.goal_pos: {getattr(scenario, 'goal_pos', 'None')}")
            
            # 山顶标记已移除（用户不需要）
            
            # 优先使用外部传入的目标信息
            if isinstance(goal_positions, dict):
                self._log(f"✅ 检测到字典格式的目标信息")
                # 中央目标（如果提供）
                if 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    g = goal_positions['goal_pos']
                    self._log(f"✅ 找到中央目标: {g}")
                    try:
                        import numpy as _np
                        g = _np.asarray(g, dtype=_np.float32).reshape(-1)
                        self._log(f"✅ 目标转换后: {g}")
                    except Exception as e:
                        self._log(f"❌ 目标转换失败: {e}")
                    if len(g) >= 3:
                        gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
                    else:
                        gx, gy, gz = g[0], g[1], (g[2] if len(g) > 2 else 0.0)
                    ax.scatter(gx, gy, gz, color='yellow', marker='*', s=1500,
                               edgecolors='red', linewidth=6, zorder=1000, label='Goal', alpha=0.9)
                    ax.text(gx, gy, gz + 10.0, "GOAL", color='red', fontsize=16,
                            fontweight='bold', ha='center', va='bottom', zorder=1000)
                else:
                    print(f"❌ 没有找到中央目标")
                
                # 各智能体目标（如果提供）
                if 'agent_goals' in goal_positions and isinstance(goal_positions['agent_goals'], list):
                    # 🔧 修复：使用与轨迹线相同的颜色方案
                    colors = self.agent_colors  # 使用统一的颜色方案
                    for idx, gp in enumerate(goal_positions['agent_goals']):
                        if gp is None:
                            continue
                        try:
                            import numpy as _np
                            gp_arr = _np.asarray(gp, dtype=_np.float32).reshape(-1)
                            if len(gp_arr) >= 3:
                                gx, gy, gz = float(gp_arr[0]), float(gp_arr[1]), float(gp_arr[2])
                            else:
                                gx, gy, gz = gp_arr[0], gp_arr[1], 0.0
                        except Exception:
                            gx, gy, gz = gp[0], gp[1], (gp[2] if len(gp) > 2 else 0.0)
                        c = colors[idx % len(colors)]
                        ax.scatter(gx, gy, gz, color=c, marker='^', s=200, zorder=900, alpha=0.9)
                        ax.text(gx, gy, gz + 5.0, f"Agent {idx} Target", color=c, fontsize=12,
                                ha='center', va='bottom', zorder=900, fontweight='bold')
            else:
                print(f"ℹ️ goal_positions 不是字典格式，跳过外部目标数据，回退使用场景内目标")
            
            # 回退到从场景/世界对象读取（若仍需要）
            target_pos = None
            agent_goal_list = None
            if hasattr(scenario, 'goal_pos') and getattr(scenario, 'goal_pos') is not None:
                target_pos = getattr(scenario, 'goal_pos')
            elif hasattr(scenario, 'target_landmark') and getattr(scenario, 'target_landmark') is not None:
                target_pos = scenario.target_landmark.state.p_pos
            if target_pos is None and hasattr(scenario, 'world') and hasattr(scenario.world, 'landmarks') and len(scenario.world.landmarks) > 0:
                lm0 = scenario.world.landmarks[0]
                if hasattr(lm0, 'state') and getattr(lm0.state, 'p_pos', None) is not None:
                    target_pos = lm0.state.p_pos
            if hasattr(scenario, 'agent_goals') and isinstance(getattr(scenario, 'agent_goals'), list):
                agent_goal_list = getattr(scenario, 'agent_goals')
            elif hasattr(scenario, 'world') and hasattr(scenario.world, 'agent_goals'):
                agent_goal_list = getattr(scenario.world, 'agent_goals')
            if target_pos is not None:
                ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                          color='yellow', marker='*', s=800, label='Goal',
                          edgecolors='red', linewidth=4, zorder=1000)
                ax.text(target_pos[0], target_pos[1], float(target_pos[2]) + 5.0,
                       "GOAL", color='red', fontsize=14,
                       fontweight='bold', horizontalalignment='center',
                       verticalalignment='bottom', zorder=1000)
            if agent_goal_list:
                # 🔧 修复：使用与轨迹线相同的颜色方案
                colors = self.agent_colors  # 使用统一的颜色方案
                for idx, gp in enumerate(agent_goal_list):
                    if gp is None:
                        continue
                    try:
                        import numpy as _np
                        gpa = _np.asarray(gp, dtype=_np.float32).reshape(-1)
                        gx, gy, gz = float(gpa[0]), float(gpa[1]), float(gpa[2]) if len(gpa) >= 3 else (gpa[0], gpa[1], 0.0)
                    except Exception:
                        gx, gy, gz = gp[0], gp[1], gp[2]
                    c = colors[idx % len(colors)]
                    ax.scatter(gx, gy, gz, color=c, marker='^', s=200, zorder=900, alpha=0.9)
                    ax.text(gx, gy, gz + 5.0, f"Agent {idx} Target", color=c, fontsize=12,
                            ha='center', va='bottom', zorder=900, fontweight='bold')
        except Exception:
            pass
    
    def _plot_goal_from_env(self, ax, env_instance, goal_positions=None):
        """从环境实例绘制目标位置，确保与轨迹数据一致"""
        try:
            print(f"🎨 从环境实例绘制目标位置")
            print(f"🎨 环境实例类型: {type(env_instance)}")
            print(f"🎨 goal_positions: {goal_positions}")
            
            # 首先绘制环境实例地形中的最高点
            terrain_data = None
            
            # 尝试从环境实例获取地形数据
            if hasattr(env_instance, 'world') and hasattr(env_instance.world, 'terrain'):
                terrain_data = env_instance.world.terrain
                print(f"🎨 从 env.world 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            elif hasattr(env_instance, 'scenario') and hasattr(env_instance.scenario, 'terrain'):
                terrain_data = env_instance.scenario.terrain
                print(f"🎨 从 env.scenario 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            elif hasattr(env_instance, 'terrain'):
                terrain_data = env_instance.terrain
                print(f"🎨 从 env 直接获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            # 方法4：从并行环境的子环境获取（ParallelEnv）
            elif hasattr(env_instance, 'envs') and len(env_instance.envs) > 0:
                print(f"🎨 检测到并行环境，尝试从子环境获取地形数据")
                sub_env = env_instance.envs[0]  # 使用第一个子环境
                if hasattr(sub_env, 'world') and hasattr(sub_env.world, 'terrain'):
                    terrain_data = sub_env.world.terrain
                    print(f"🎨 从子环境 env.world 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
                elif hasattr(sub_env, 'scenario') and hasattr(sub_env.scenario, 'terrain'):
                    terrain_data = sub_env.scenario.terrain
                    print(f"🎨 从子环境 env.scenario 获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
                elif hasattr(sub_env, 'terrain'):
                    terrain_data = sub_env.terrain
                    print(f"🎨 从子环境 env 直接获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
            
            # 方法5：通过ParallelEnv的get_terrain_data方法获取
            elif hasattr(env_instance, 'get_terrain_data'):
                self._log(f"🎨 通过ParallelEnv.get_terrain_data获取地形数据")
                terrain_result = env_instance.get_terrain_data(0)  # 使用第一个子环境
                if isinstance(terrain_result, dict) and terrain_result.get('terrain') is not None:
                    terrain_data = terrain_result['terrain']
                    self._log(f"🎨 通过get_terrain_data获取地形数据: {terrain_data.shape if terrain_data is not None else 'None'}")
                else:
                    self._log(f"🎨 get_terrain_data返回空结果: {terrain_result}")
            
            # 环境山顶标记已移除（用户不需要）
            
            # 绘制目标位置 - 优先使用外部传入的目标信息
            if isinstance(goal_positions, dict):
                self._log(f"✅ 检测到字典格式的目标信息")
                # 中央目标
                if 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    g = goal_positions['goal_pos']
                    self._log(f"✅ 找到中央目标: {g}")
                    try:
                        import numpy as _np
                        g = _np.asarray(g, dtype=_np.float32).reshape(-1)
                        self._log(f"✅ 目标转换后: {g}")
                    except Exception as e:
                        self._log(f"❌ 目标转换失败: {e}")
                        pass
                    if len(g) >= 3:
                        gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
                        self._log(f"🎯 开始绘制中央目标: ({gx}, {gy}, {gz})")
                        # 增强中央目标显示效果
                        ax.scatter(gx, gy, gz, color='yellow', marker='*', s=1500,
                                   edgecolors='red', linewidth=6, zorder=1000, label='Goal', alpha=0.9)
                        ax.text(gx, gy, gz + 10.0, "GOAL", color='red', fontsize=16,
                                fontweight='bold', ha='center', va='bottom', zorder=1000)
                        self._log(f"✅ 中央目标绘制完成")
                    else:
                        gx, gy, gz = g[0], g[1], (g[2] if len(g) > 2 else 0.0)
                        self._log(f"🎯 开始绘制中央目标(备用): ({gx}, {gy}, {gz})")
                        ax.scatter(gx, gy, gz, color='yellow', marker='*', s=1500,
                                   edgecolors='red', linewidth=6, zorder=1000, label='Goal', alpha=0.9)
                        ax.text(gx, gy, gz + 10.0, "GOAL", color='red', fontsize=16,
                                fontweight='bold', ha='center', va='bottom', zorder=1000)
                        self._log(f"✅ 中央目标绘制完成(备用)")
                else:
                    self._log(f"❌ 没有找到中央目标")
                    
                # 每个智能体的分配目标
                if 'agent_goals' in goal_positions and isinstance(goal_positions['agent_goals'], list):
                    # 🔧 修复：使用与轨迹线相同的颜色方案
                    colors = self.agent_colors  # 使用统一的颜色方案
                    for idx, gp in enumerate(goal_positions['agent_goals']):
                        if gp is None:
                            continue
                        try:
                            gpa = _np.asarray(gp, dtype=_np.float32).reshape(-1)
                            gx, gy, gz = float(gpa[0]), float(gpa[1]), float(gpa[2]) if len(gpa) >= 3 else (gpa[0], gpa[1], 0.0)
                        except Exception:
                            gx, gy, gz = gp[0], gp[1], gp[2]
                        c = colors[idx % len(colors)]
                        ax.scatter(gx, gy, gz, color=c, marker='^', s=200, zorder=900, alpha=0.9)
                        ax.text(gx, gy, gz + 5.0, f"Agent {idx} Target", color=c, fontsize=12,
                                ha='center', va='bottom', zorder=900, fontweight='bold')
            else:
                print(f"ℹ️ goal_positions 不是字典格式，跳过外部目标数据")
                
        except Exception as e:
            print(f"❌ 从环境实例绘制目标位置失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_trajectories(self, ax, trajectories):
        """绘制智能体轨迹
        
        参数:
            ax: matplotlib 3D轴
            trajectories: 轨迹数据，格式可能为:
                        - [agent][timestep][xyz] 或 
                        - [timestep][agent][xyz]
        """
        # 首先检测轨迹数据格式
        processed_trajectories = self._process_trajectory_data(trajectories)
        # 裁剪每条轨迹的有效步并去重
        processed_trajectories = [self._trim_effective_traj(t) for t in processed_trajectories]
        
        for i, traj in enumerate(processed_trajectories):
            if i >= 3 or not traj:  # 只显示前3个智能体
                continue
            
            color = self.agent_colors[i % len(self.agent_colors)]
            
            # 提取轨迹点
            xs = [p[0] for p in traj if p is not None and len(p) >= 3]
            ys = [p[1] for p in traj if p is not None and len(p) >= 3]
            zs = [p[2] for p in traj if p is not None and len(p) >= 3]
            
            if len(xs) > 1:
                # 绘制轨迹线
                ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=0.8, label=f'Agent {i}')
                
                # 标记起点和终点
                ax.scatter(xs[0], ys[0], zs[0], color=color, marker='o', s=150,
                          edgecolors='black', linewidth=2)
                ax.text(xs[0], ys[0], zs[0]+3, f"Start{i}", color='black', fontsize=10,
                       fontweight='bold')
                
                ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='o', s=150,
                          edgecolors='black', linewidth=2)
                ax.text(xs[-1], ys[-1], zs[-1]+3, f"End{i}", color='black', fontsize=10,
                       fontweight='bold')
    
    def _set_labels_and_title(self, ax, episode_type, episode_num, reward, correction_type,
                              total_steps=None, effective_steps=None, title_step_note=None):
        """设置标签和标题"""
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        title = f'{episode_type.capitalize()} Episode {episode_num} - Reward: {reward:.2f}'
        if title_step_note:
            title += f'\n{title_step_note}'
        elif total_steps is not None and effective_steps is not None:
            title += f'\nSteps: {effective_steps}/{total_steps} (effective/total)'
        elif total_steps is not None:
            title += f'\nTotal Steps: {total_steps}'
        elif effective_steps is not None:
            title += f'\nEffective Steps: {effective_steps}'
        
        if correction_type:
            correction_display = {
                'potential_field': 'Potential Field',
                'hybrid': 'Hybrid',
                'basic': 'Basic',
                'terrain_avoidance': 'Terrain Avoidance',
                'target_guidance': 'Target Guidance',
                'combined': 'Combined'
            }.get(correction_type, correction_type)
            title += f' ({correction_display})'
        
        ax.set_title(title)
        ax.legend(loc='upper right')

    def _lighten_color(self, rgb, factor):
        """将颜色向白色插值以实现更浅的色调。
        factor ∈ [0,1]，越大越接近原色，越小越接近白色。
        """
        try:
            base = np.array(rgb, dtype=np.float32)
            white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            out = white * (1.0 - factor) + base * factor
            return np.clip(out, 0.0, 1.0)
        except Exception:
            return rgb

    def generate_all_episodes_overlay(self, episodes_trajectories, scenario, save_path,
                                      elev=30, azim=45, max_episodes=None):
        """将多个回合的轨迹叠加到一张图中，并用颜色深浅区分早晚回合。

        参数:
            episodes_trajectories: List[episode_traj]，每个元素为一回合的轨迹，
                                   支持 [timestep][agent][xyz] 或 [agent][timestep][xyz]
            scenario: 场景对象（用于绘制地形/目标）
            save_path: 保存路径
            elev/azim: 视角
            max_episodes: 若指定，仅绘制最近的N个回合
        """
        try:
            if not episodes_trajectories:
                return

            # 选择要绘制的回合范围
            all_eps = episodes_trajectories
            if max_episodes is not None and len(all_eps) > int(max_episodes):
                all_eps = all_eps[-int(max_episodes):]

            num_eps = len(all_eps)

            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=elev, azim=azim)

            # 背景：地形/障碍/目标
            self._plot_terrain(ax, scenario, alpha=0.25)
            self._plot_obstacles(ax, scenario)
            self._plot_goal(ax, scenario)

            base_colors = [np.array([0.121, 0.466, 0.705]),   # 蓝 (Agent0)
                           np.array([1.0,   0.498, 0.0549]),  # 橙 (Agent1)
                           np.array([0.172, 0.627, 0.172]),   # 绿 (Agent2)
                           np.array([0.839, 0.152, 0.156]),   # 红
                           np.array([0.580, 0.404, 0.741]),   # 紫
                           np.array([0.549, 0.337, 0.294])]   # 棕

            # 逐回合绘制，越新的回合颜色越深/alpha越高
            for ep_idx, ep_traj in enumerate(all_eps):
                # 归一化深浅因子：旧→浅，新→深
                depth = (ep_idx + 1) / num_eps  # (0,1]
                alpha = 0.15 + 0.75 * depth     # 0.15~0.9

                trajs = self._process_trajectory_data(ep_traj)
                if not trajs:
                    continue
                for ag_idx, traj in enumerate(trajs):
                    if not traj or ag_idx >= len(base_colors):
                        continue
                    xs = [p[0] for p in traj if p is not None and len(p) >= 3]
                    ys = [p[1] for p in traj if p is not None and len(p) >= 3]
                    zs = [p[2] for p in traj if p is not None and len(p) >= 3]
                    if len(xs) < 2:
                        continue
                    base = base_colors[ag_idx % len(base_colors)]
                    color = self._lighten_color(base, factor=alpha)
                    ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=alpha)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'All Episodes Overlay (N={num_eps})')

            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"所有回合叠加图已保存到: {save_path}")

        except Exception as e:
            print(f"生成所有回合叠加图失败: {e}")
            traceback.print_exc()
    
    def save_episode_visualization(self, episode_trajectories, scenario, args,
                                  best_episode, actual_episode, total_reward, best_reward):
        """保存回合可视化（主入口函数）"""
        try:
            print(f"\n正在生成回合({actual_episode})轨迹图和GIF...")
            
            # 创建保存目录
            episode_dir = os.path.join("logs", args.exp_name, f"episode_{actual_episode}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # 处理当前回合
            if actual_episode < len(episode_trajectories) and episode_trajectories[actual_episode]:
                trajectories = self._reorganize_trajectories(episode_trajectories[actual_episode])
                
                # 生成图像
                image_path = os.path.join(episode_dir, f"trajectory_episode_{actual_episode}.png")
                self.generate_trajectory_image(
                    trajectories, scenario, image_path,
                    actual_episode, total_reward, "current",
                    getattr(args, 'correction_type', None)
                )
                
                # 生成GIF
                gif_path = os.path.join(episode_dir, f"episode_{actual_episode}.gif")
                self.generate_trajectory_gif(
                    trajectories, scenario, gif_path,
                    actual_episode, total_reward, "current"
                )
            
            # 处理最佳回合
            if (best_episode < len(episode_trajectories) and 
                episode_trajectories[best_episode] and 
                best_episode != actual_episode):
                
                print(f"\n正在生成最佳回合({best_episode})轨迹图和GIF...")
                
                trajectories = self._reorganize_trajectories(episode_trajectories[best_episode])
                
                # 生成图像
                image_path = os.path.join(episode_dir, f"trajectory_best_episode_{best_episode}.png")
                self.generate_trajectory_image(
                    trajectories, scenario, image_path,
                    best_episode, best_reward, "best",
                    getattr(args, 'correction_type', None)
                )
                
                # 生成GIF
                gif_path = os.path.join(episode_dir, f"best_episode_{best_episode}.gif")
                self.generate_trajectory_gif(
                    trajectories, scenario, gif_path,
                    best_episode, best_reward, "best"
                )
                
        except Exception as e:
            print(f"保存轨迹可视化时出错: {e}")
            traceback.print_exc()
    
    def _reorganize_trajectories(self, episode_trajectories):
        """重组轨迹数据：从[时间步][智能体]转置为[智能体][时间步]"""
        if not episode_trajectories or not episode_trajectories[0]:
            return []
        
        num_agents = len(episode_trajectories[0])
        trajectories = []
        
        for agent_idx in range(num_agents):
            agent_trajectory = []
            for step_data in episode_trajectories:
                if agent_idx < len(step_data) and step_data[agent_idx] is not None:
                    agent_trajectory.append(np.array(step_data[agent_idx]).copy())
            trajectories.append(agent_trajectory)
        
        return trajectories

    def generate_trajectory_interactive(self, trajectories, save_path="trajectory_interactive.html", 
                                      title="Interactive Trajectory", goal_positions=None, scenario=None, env_instance=None,
                                      env_idx=0):
        """生成可交互的3D轨迹HTML文件，支持拖拽视角
        
        参数:
            trajectories: 轨迹数据 [agent][timestep][xyz]
            save_path: 保存路径
            title: 图表标题
            goal_positions: 目标位置列表
            scenario: 场景对象（用于获取地形信息）
            env_instance: 环境实例（用于获取与轨迹一致的地形和目标数据）
        """
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
            from plotly.subplots import make_subplots
        except ImportError:
            print("警告：需要安装 plotly 才能生成可交互HTML: pip install plotly")
            return False
        
        # 处理轨迹数据
        processed_trajectories = self._process_trajectory_data(trajectories)
        processed_trajectories = [self._trim_effective_traj(t) for t in processed_trajectories]
        if not processed_trajectories:
            print("警告：没有有效的轨迹数据生成可交互HTML")
            return False
        
        # 创建3D图表
        fig = go.Figure()
        
        # 安全添加3D散点（带符号规范与回退）
        def _add_marker_safe(x, y, z, size, color, symbol, line_width, line_color, name, text):
            sym = self._normalize_plotly_symbol(symbol)
            try:
                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=color,
                        symbol=sym,
                        line=dict(width=line_width, color=line_color)
                    ),
                    name=name,
                    text=[text],
                    textposition='top center'
                ))
            except Exception as e:
                try:
                    fig.add_trace(go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='markers',
                        marker=dict(
                            size=size,
                            color=color,
                            symbol='diamond',
                            line=dict(width=line_width, color=line_color)
                        ),
                        name=name,
                        text=[text],
                        textposition='top center'
                    ))
                    print(f"⚠️ 兼容性回退: 将无效symbol '{symbol}' 替换为 'diamond'")
                except Exception as e2:
                    print(f"绘制目标点失败(回退也失败): {e2}")
        
        # 添加地形（优先使用环境实例数据）
        terrain_data = None
        map_size = None
        
        # 优先从环境实例获取地形数据（兼容向量化/非向量化环境）
        if env_instance is not None and terrain_data is None:
            try:
                if hasattr(env_instance, 'get_terrain_data'):
                    terrain_result = env_instance.get_terrain_data(0)
                    if isinstance(terrain_result, dict) and terrain_result.get('terrain') is not None:
                        terrain_data = terrain_result['terrain']
                        map_size = terrain_result.get('map_size')
                # 额外回退：直接读取属性
                if terrain_data is None and hasattr(env_instance, 'terrain') and env_instance.terrain is not None:
                    terrain_data = env_instance.terrain
                    map_size = getattr(env_instance, 'map_size', terrain_data.shape[1])
                # 额外回退：并行环境 envs[0]
                if terrain_data is None and hasattr(env_instance, 'envs') and len(getattr(env_instance, 'envs', [])) > 0:
                    sub_env = env_instance.envs[0]
                    if hasattr(sub_env, 'terrain') and sub_env.terrain is not None:
                        terrain_data = sub_env.terrain
                        map_size = getattr(sub_env, 'map_size', terrain_data.shape[1])
            except Exception as _:
                pass
        
        # 回退到场景数据
        if terrain_data is None and scenario:
            if hasattr(scenario, 'terrain') and scenario.terrain is not None:
                terrain_data = scenario.terrain
                map_size = getattr(scenario, 'map_size', terrain_data.shape[1])
            elif hasattr(scenario, 'world') and hasattr(scenario.world, 'terrain'):
                terrain_data = scenario.world.terrain
                map_size = getattr(scenario.world, 'map_size', terrain_data.shape[1])
            # 兼容另一种高度图字段 height_map
            elif hasattr(scenario, 'world') and hasattr(scenario.world, 'height_map') and scenario.world.height_map is not None:
                terrain_data = np.asarray(scenario.world.height_map)
                map_size = terrain_data.shape[1]
        
        terrain_height_sampler = None
        if terrain_data is not None:
            try:
                # 🔧 关键修复：如果地形已经降采样（50×50），直接使用；否则进行降采样
                terrain_data_array = np.asarray(terrain_data)
                terrain_h, terrain_w = terrain_data_array.shape[0], terrain_data_array.shape[1]
                
                # 判断地形是否已降采样：如果地形尺寸约为map_size/4，说明已降采样
                # 或者如果map_size存在且远大于地形尺寸，说明已降采样
                is_downsampled = False
                if map_size is not None:
                    # 如果map_size是地形尺寸的4倍左右，说明已降采样
                    if abs(map_size / terrain_w - 4.0) < 0.5:
                        is_downsampled = True
                
                if is_downsampled:
                    # 地形已降采样，直接使用，但坐标需要映射到0-map_size范围
                    terrain_data_sampled = terrain_data_array
                    # 创建与地形尺寸匹配的网格，坐标映射到原始范围
                    x_samples = np.linspace(0, map_size - 1, terrain_w)
                    y_samples = np.linspace(0, map_size - 1, terrain_h)
                    X_terrain, Y_terrain = np.meshgrid(x_samples, y_samples)
                    
                    def _sample_terrain_height(xs, ys):
                        # 将坐标从0-map_size映射到0-terrain_w范围
                        x_idx = np.clip(np.round(xs * terrain_w / map_size).astype(int), 0, terrain_w - 1)
                        y_idx = np.clip(np.round(ys * terrain_h / map_size).astype(int), 0, terrain_h - 1)
                        return terrain_data_array[y_idx, x_idx]
                    terrain_height_sampler = _sample_terrain_height
                else:
                    # 地形未降采样，进行降采样（从200×200降到50×50）
                    sample_rate = 4  # 每4个点采样1个
                    def _sample_terrain_height(xs, ys):
                        x_idx = np.clip(np.round(xs).astype(int), 0, terrain_data_array.shape[1]-1)
                        y_idx = np.clip(np.round(ys).astype(int), 0, terrain_data_array.shape[0]-1)
                        return terrain_data_array[y_idx, x_idx]
                    terrain_height_sampler = _sample_terrain_height
                    x_samples = np.arange(0, terrain_data_array.shape[1], sample_rate)
                    y_samples = np.arange(0, terrain_data_array.shape[0], sample_rate)
                    
                    # 创建降采样后的地形数据
                    terrain_data_sampled = []
                    for y in y_samples:
                        row = []
                        for x in x_samples:
                            z = terrain_data_array[int(y), int(x)]
                            row.append(float(z))
                        terrain_data_sampled.append(row)
                    terrain_data_sampled = np.array(terrain_data_sampled)
                    
                    # 创建降采样后的网格
                    X_terrain, Y_terrain = np.meshgrid(x_samples, y_samples)
                
                # 添加地形表面 - 使用自然的地形颜色（灰绿色系）
                fig.add_trace(go.Surface(
                    x=X_terrain,
                    y=Y_terrain, 
                    z=terrain_data_sampled,
                    colorscale=[
                        [0, 'rgb(220, 220, 180)'],      # 低海拔：浅黄色
                        [0.3, 'rgb(180, 200, 120)'],    # 中低：浅绿色
                        [0.5, 'rgb(120, 160, 100)'],    # 中等：绿色
                        [0.7, 'rgb(100, 120, 80)'],     # 中高：深绿色
                        [0.85, 'rgb(139, 137, 137)'],   # 高：灰色
                        [1, 'rgb(255, 255, 255)']       # 极高：白色（雪）
                    ],
                    name='地形',
                    opacity=1.0,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="高度 (m)", side='right'),
                        tickmode="linear",
                        tick0=0,
                        dtick=20,
                        len=0.8,
                        thickness=16,
                        x=1.02,           # 放在绘图区右侧
                        xanchor='left',
                        bgcolor='rgba(255,255,255,0.6)'
                    ),
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.8,
                        specular=0.2,
                        roughness=0.5
                    ),
                    contours=dict(
                        z=dict(
                            show=True,
                            usecolormap=True,
                            highlightcolor="limegreen",
                            project=dict(z=False)
                        )
                    )
                ))
            except Exception as e:
                print(f"添加地形失败: {e}")
        
        # 山顶位置标记已移除（用户不需要）

        # 绘制障碍（球体）- 优先使用实时环境中的障碍物快照
        try:
            obstacles = self._get_obstacles_for_plot(scenario=scenario, env_instance=env_instance, env_idx=env_idx)
            for ob in obstacles:
                try:
                    center = ob.get('center', [0, 0, 0])
                    radius = ob.get('radius', 0.0)
                    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
                    r = float(radius)
                    if r <= 0:
                        continue
                    theta = np.linspace(0, 2*np.pi, 24)
                    phi = np.linspace(0, np.pi, 16)
                    th, ph = np.meshgrid(theta, phi)
                    sx = cx + r * np.cos(th) * np.sin(ph)
                    sy = cy + r * np.sin(th) * np.sin(ph)
                    sz = cz + r * np.cos(ph)
                    fig.add_trace(go.Surface(
                        x=sx, y=sy, z=sz,
                        colorscale=[[0, 'rgba(255,0,0,1)'], [1, 'rgba(255,0,0,1)']],
                        opacity=0.25,
                        showscale=False,
                        name='障碍'
                    ))
                except Exception:
                    continue
        except Exception as e:
            print(f"添加障碍失败: {e}")

        # 绘制目标点
        if goal_positions:
            try:
                # 中央目标
                if isinstance(goal_positions, dict) and 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    goal_pos = goal_positions['goal_pos']
                    if len(goal_pos) >= 3:
                        gx, gy, gz = float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2])
                        
                        _add_marker_safe(
                            x=gx,
                            y=gy,
                            z=gz,
                            size=15,
                            color='yellow',
                            symbol='diamond',
                            line_width=4,
                            line_color='red',
                            name='中央目标',
                            text=f'目标 ({gx:.1f}, {gy:.1f}, {gz:.1f})'
                        )
                
                # 智能体目标
                if isinstance(goal_positions, dict) and 'agent_goals' in goal_positions:
                    # 🔧 修复：使用与轨迹线相同的颜色方案
                    colors = self.agent_colors  # 使用统一的颜色方案
                    for idx, agent_goal in enumerate(goal_positions['agent_goals']):
                        if agent_goal is not None and len(agent_goal) >= 3:
                            ax, ay, az = float(agent_goal[0]), float(agent_goal[1]), float(agent_goal[2])
                            color = colors[idx % len(colors)]
                            
                            _add_marker_safe(
                                x=ax,
                                y=ay,
                                z=az,
                                size=12,
                                color=color,
                                symbol='diamond',
                                line_width=2,
                                line_color='black',
                                name=f'智能体{idx}目标',
                                text=f'Agent{idx} ({ax:.1f}, {ay:.1f}, {az:.1f})'
                            )
            except Exception as e:
                print(f"绘制目标点失败: {e}")

        # 添加智能体轨迹 - 使用更鲜明的颜色和更粗的线条
        colors = [
            'rgb(0, 0, 0)',       # 黑色
            'rgb(255, 0, 0)',     # 红色
            'rgb(0, 0, 255)',     # 蓝色
            'rgb(255, 255, 0)',   # 黄色
            'rgb(0, 255, 255)',   # 青色
            'rgb(255, 0, 255)'    # 品红色
        ]
        for agent_idx, agent_traj in enumerate(processed_trajectories):
            if len(agent_traj) > 0:
                agent_traj = np.array(agent_traj)
                plotted_traj = agent_traj.copy()
                if terrain_height_sampler is not None:
                    terrain_heights = terrain_height_sampler(plotted_traj[:, 0], plotted_traj[:, 1])
                    penetration_mask = plotted_traj[:, 2] < terrain_heights
                    # 🚨 关键修复：不要将穿透的点设为NaN，而是调整到地形上方，确保可视化正确
                    # 原因：如果设为NaN，起点可能不显示，导致可视化问题
                    # 解决方案：将穿透的点调整到地形上方（地形高度 + 0.5m），确保可见
                    if np.any(penetration_mask):
                        plotted_traj[penetration_mask, 2] = terrain_heights[penetration_mask] + 0.5
                color = colors[agent_idx % len(colors)]
                
                # 轨迹线 - 更粗的线条
                fig.add_trace(go.Scatter3d(
                    x=plotted_traj[:, 0],
                    y=plotted_traj[:, 1], 
                    z=plotted_traj[:, 2],
                    mode='lines',
                    line=dict(color=color, width=8),
                    name=f'智能体 {agent_idx}',
                    hovertemplate='<b>智能体 %{fullData.name}</b><br>' +
                                'X: %{x:.2f}<br>' +
                                'Y: %{y:.2f}<br>' +
                                'Z: %{z:.2f}<br>' +
                                '<extra></extra>'
                ))
                
                # 起点标记 - 小球标记
                # 🚨 关键修复：确保起点Z坐标在地形上方（如果地形采样器可用）
                start_pos = [agent_traj[0, 0], agent_traj[0, 1], agent_traj[0, 2]]
                if terrain_height_sampler is not None:
                    start_terrain_h = terrain_height_sampler(start_pos[0], start_pos[1])
                    if start_pos[2] < start_terrain_h:
                        # 起点在地形下方，调整到地形上方
                        start_pos[2] = start_terrain_h + 0.5
                
                fig.add_trace(go.Scatter3d(
                    x=[start_pos[0]],
                    y=[start_pos[1]],
                    z=[start_pos[2]],
                    mode='markers+text',
                    marker=dict(
                        size=8,  # 增大起点标记，更明显
                        color=color,
                        symbol='circle',
                        line=dict(color='rgb(0, 0, 0)', width=2)
                    ),
                    text=[f"起点<br>X: {start_pos[0]:.2f}<br>Y: {start_pos[1]:.2f}<br>Z: {start_pos[2]:.2f}"],
                    textposition="top center",
                    textfont=dict(size=10, color='black'),
                    name=f'起点 {agent_idx}',
                    showlegend=False,
                    hovertemplate='<b>起点</b><br>' +
                                'X: %{x:.2f}<br>' +
                                'Y: %{y:.2f}<br>' +
                                'Z: %{z:.2f}<br>' +
                                '<extra></extra>'
                ))
                
                # 终点标记 - 小球标记
                fig.add_trace(go.Scatter3d(
                    x=[agent_traj[-1, 0]],
                    y=[agent_traj[-1, 1]],
                    z=[agent_traj[-1, 2]],
                    mode='markers',
                    marker=dict(
                        size=8,  # 从15改为8，更小更精致
                        color=color,
                        symbol='circle',
                        line=dict(color='rgb(0, 0, 0)', width=1.5)
                    ),
                    name=f'终点 {agent_idx+1}',
                    showlegend=False,
                    hovertemplate='<b>终点</b><br>' +
                                'X: %{x:.2f}<br>' +
                                'Y: %{y:.2f}<br>' +
                                'Z: %{z:.2f}<br>' +
                                '<extra></extra>'
                ))
        
        # 添加目标点
        # 添加目标点标注
        if goal_positions:
            # 处理字典格式的目标点信息
            if isinstance(goal_positions, dict):
                # 中央目标点
                if 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    goal = goal_positions['goal_pos']
                    if len(goal) >= 3:
                        fig.add_trace(go.Scatter3d(
                            x=[goal[0]],
                            y=[goal[1]],
                            z=[goal[2]],
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='diamond', line=dict(color='yellow', width=2)),
                            name='中央目标',
                            showlegend=True,
                            hovertemplate='<b>中央目标</b><br>' +
                                        'X: %{x:.2f}<br>' +
                                        'Y: %{y:.2f}<br>' +
                                        'Z: %{z:.2f}<br>' +
                                        '<extra></extra>'
                        ))
                # 各智能体的目标点
                if 'agent_goals' in goal_positions and isinstance(goal_positions['agent_goals'], list):
                    # 🔧 使用与智能体轨迹相同的颜色方案（与轨迹线颜色一致）
                    agent_colors_rgb = [
                        'rgb(0, 0, 0)',       # 黑色（智能体0）
                        'rgb(255, 0, 0)',     # 红色（智能体1）
                        'rgb(0, 0, 255)',     # 蓝色（智能体2）
                        'rgb(255, 255, 0)',   # 黄色（智能体3）
                        'rgb(0, 255, 255)',   # 青色（智能体4）
                        'rgb(255, 0, 255)'    # 品红色（智能体5）
                    ]
                    for i, goal in enumerate(goal_positions['agent_goals']):
                        if goal is not None and len(goal) >= 3:
                            color = agent_colors_rgb[i % len(agent_colors_rgb)]
                            fig.add_trace(go.Scatter3d(
                                x=[goal[0]],
                                y=[goal[1]],
                                z=[goal[2]],
                                mode='markers',
                                marker=dict(size=12, color=color, symbol='diamond', line=dict(color='white', width=1)),
                                name=f'智能体{i}目标',
                                showlegend=True,
                                hovertemplate=f'<b>智能体{i}目标</b><br>' +
                                            'X: %{x:.2f}<br>' +
                                            'Y: %{y:.2f}<br>' +
                                            'Z: %{z:.2f}<br>' +
                                            '<extra></extra>'
                            ))
            # 处理列表格式的目标点信息（向后兼容）
            elif isinstance(goal_positions, list):
                for i, goal in enumerate(goal_positions):
                    if goal is not None and len(goal) >= 3:
                        fig.add_trace(go.Scatter3d(
                            x=[goal[0]],
                            y=[goal[1]],
                            z=[goal[2]],
                            mode='markers',
                            marker=dict(size=12, color='gold', symbol='diamond'),
                            name=f'目标 {i+1}',
                            showlegend=False,
                            hovertemplate='<b>目标点</b><br>' +
                                        'X: %{x:.2f}<br>' +
                                        'Y: %{y:.2f}<br>' +
                                        'Z: %{z:.2f}<br>' +
                                        '<extra></extra>'
                        ))
        
        # 设置3D轨迹HTML布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color='rgb(0, 0, 0)')
            ),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="高度",
                xaxis=dict(
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                zaxis=dict(
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            width=1200,
            height=900,
            margin=dict(l=10, r=120, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # 保存HTML文件
        pyo.plot(fig, filename=save_path, auto_open=False)
        
        print(f"可交互轨迹HTML已保存: {save_path}")
        return True

    def generate_all_episodes_overlay_interactive(self, all_trajectories, save_path="overlay_interactive.html",
                                                title="All Episodes Overlay", goal_positions=None, scenario=None):
        """生成所有回合叠加的可交互HTML"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
        except ImportError:
            print("警告：需要安装 plotly 才能生成可交互HTML: pip install plotly")
            return False
        
        if not all_trajectories:
            print("警告：没有轨迹数据生成叠加可交互HTML")
            return False
        
        # 创建3D图表
        fig = go.Figure()
        
        # 添加地形（如果有场景信息）
        if scenario and hasattr(scenario, 'world') and hasattr(scenario.world, 'height_map'):
            try:
                height_map = scenario.world.height_map
                extent = getattr(scenario.world, 'extent', [-50, 50, -50, 50])
                
                # 🔧 关键修复：使用与 visualize_terrain_map.py 相同的降采样方式
                sample_rate = 4  # 每4个点采样1个，从200×200降到50×50
                
                # 对地形数据进行降采样
                height_map_array = np.asarray(height_map)
                def _sample_height(xs, ys):
                    x_norm = (xs - extent[0]) / (extent[1] - extent[0]) * (height_map_array.shape[1] - 1)
                    y_norm = (ys - extent[2]) / (extent[3] - extent[2]) * (height_map_array.shape[0] - 1)
                    x_idx = np.clip(np.round(x_norm).astype(int), 0, height_map_array.shape[1]-1)
                    y_idx = np.clip(np.round(y_norm).astype(int), 0, height_map_array.shape[0]-1)
                    return height_map_array[y_idx, x_idx]
                terrain_height_sampler = _sample_height
                x_samples = np.arange(0, height_map_array.shape[1], sample_rate)
                y_samples = np.arange(0, height_map_array.shape[0], sample_rate)
                
                # 创建降采样后的地形数据
                height_map_sampled = []
                for y in y_samples:
                    row = []
                    for x in x_samples:
                        z = height_map_array[int(y), int(x)]
                        row.append(float(z))
                    height_map_sampled.append(row)
                height_map_sampled = np.array(height_map_sampled)
                
                # 创建降采样后的网格（保持extent范围）
                x_terrain = np.linspace(extent[0], extent[1], len(x_samples))
                y_terrain = np.linspace(extent[2], extent[3], len(y_samples))
                X_terrain, Y_terrain = np.meshgrid(x_terrain, y_terrain)
                
                fig.add_trace(go.Surface(
                    x=X_terrain,
                    y=Y_terrain,
                    z=height_map_sampled,
                    colorscale=[
                        [0, 'rgb(220, 220, 180)'],      # 低海拔：浅黄色
                        [0.3, 'rgb(180, 200, 120)'],    # 中低：浅绿色
                        [0.5, 'rgb(120, 160, 100)'],    # 中等：绿色
                        [0.7, 'rgb(100, 120, 80)'],     # 中高：深绿色
                        [0.85, 'rgb(139, 137, 137)'],   # 高：灰色
                        [1, 'rgb(255, 255, 255)']       # 极高：白色（雪）
                    ],
                    name='地形',
                    opacity=0.95,
                    showscale=False,
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.8,
                        specular=0.2,
                        roughness=0.5
                    ),
                    contours=dict(
                        z=dict(
                            show=True,
                            usecolormap=True,
                            highlightcolor="limegreen",
                            project=dict(z=False)
                        )
                    )
                ))
            except Exception as e:
                print(f"添加地形失败: {e}")
        
        # 颜色映射
        colors = self.agent_colors  # 🔧 修复：使用统一的颜色方案
        num_episodes = len(all_trajectories)
        
        # 为每个回合添加轨迹
        for ep_idx, episode_traj in enumerate(all_trajectories):
            processed_traj = self._process_trajectory_data(episode_traj)
            
            # 计算透明度（早期回合更透明）
            alpha = 0.3 + 0.7 * (ep_idx / max(1, num_episodes - 1))
            
            for agent_idx, agent_traj in enumerate(processed_traj):
                if len(agent_traj) > 0:
                    agent_traj = np.array(agent_traj)
                    plotted_traj = agent_traj.copy()
                    if terrain_height_sampler is not None:
                        terrain_heights = terrain_height_sampler(plotted_traj[:, 0], plotted_traj[:, 1])
                        penetration_mask = plotted_traj[:, 2] < terrain_heights
                        plotted_traj[penetration_mask, :] = np.nan
                    base_color = colors[agent_idx % len(colors)]
                    
                    # 轨迹线
                    fig.add_trace(go.Scatter3d(
                        x=plotted_traj[:, 0],
                        y=plotted_traj[:, 1],
                        z=plotted_traj[:, 2],
                        mode='lines',
                        line=dict(color=base_color, width=2),
                        opacity=alpha,
                        name=f'回合{ep_idx+1}-智能体{agent_idx+1}',
                        showlegend=(ep_idx < 5),  # 只显示前5个回合的图例
                        hovertemplate=f'<b>回合 {ep_idx+1} - 智能体 {agent_idx+1}</b><br>' +
                                    'X: %{x:.2f}<br>' +
                                    'Y: %{y:.2f}<br>' +
                                    'Z: %{z:.2f}<br>' +
                                    '<extra></extra>'
                    ))
        
        # 添加目标点
        # 添加目标点标注
        if goal_positions:
            # 处理字典格式的目标点信息
            if isinstance(goal_positions, dict):
                # 中央目标点
                if 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    goal = goal_positions['goal_pos']
                    if len(goal) >= 3:
                        fig.add_trace(go.Scatter3d(
                            x=[goal[0]],
                            y=[goal[1]],
                            z=[goal[2]],
                            mode='markers',
                            marker=dict(size=15, color='gold', symbol='diamond'),
                            name='共同目标点',
                            showlegend=True,
                            hovertemplate='<b>共同目标点</b><br>' +
                                        'X: %{x:.2f}<br>' +
                                        'Y: %{y:.2f}<br>' +
                                        'Z: %{z:.2f}<br>' +
                                        '<extra></extra>'
                        ))
                # 各智能体的目标点
                if 'agent_goals' in goal_positions and isinstance(goal_positions['agent_goals'], list):
                    # 🔧 使用与智能体轨迹相同的颜色方案（与轨迹线颜色一致）
                    agent_colors_rgb = [
                        'rgb(0, 0, 0)',       # 黑色（智能体0）
                        'rgb(255, 0, 0)',     # 红色（智能体1）
                        'rgb(0, 0, 255)',     # 蓝色（智能体2）
                        'rgb(255, 255, 0)',   # 黄色（智能体3）
                        'rgb(0, 255, 255)',   # 青色（智能体4）
                        'rgb(255, 0, 255)'    # 品红色（智能体5）
                    ]
                    for i, goal in enumerate(goal_positions['agent_goals']):
                        if goal is not None and len(goal) >= 3:
                            color = agent_colors_rgb[i % len(agent_colors_rgb)]
                            fig.add_trace(go.Scatter3d(
                                x=[goal[0]],
                                y=[goal[1]],
                                z=[goal[2]],
                                mode='markers',
                                marker=dict(size=12, color=color, symbol='diamond', line=dict(color='white', width=1)),
                                name=f'智能体{i}目标',
                                showlegend=True,
                                hovertemplate=f'<b>智能体{i}目标</b><br>' +
                                            'X: %{x:.2f}<br>' +
                                            'Y: %{y:.2f}<br>' +
                                            'Z: %{z:.2f}<br>' +
                                            '<extra></extra>'
                            ))
            # 处理列表格式的目标点信息（向后兼容）
            elif isinstance(goal_positions, list):
                for i, goal in enumerate(goal_positions):
                    if goal is not None and len(goal) >= 3:
                        fig.add_trace(go.Scatter3d(
                            x=[goal[0]],
                            y=[goal[1]],
                            z=[goal[2]],
                            mode='markers',
                            marker=dict(size=15, color='gold', symbol='diamond'),
                            name=f'目标 {i+1}',
                            showlegend=True,
                            hovertemplate='<b>目标点</b><br>' +
                                        'X: %{x:.2f}<br>' +
                                        'Y: %{y:.2f}<br>' +
                                        'Z: %{z:.2f}<br>' +
                                        '<extra></extra>'
                        ))
        
        # 设置布局
        fig.update_layout(
            title=dict(
                text=f"{title} ({num_episodes} 回合)",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            width=1200,
            height=900,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        # 保存HTML文件
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"叠加可交互轨迹HTML已保存: {save_path}")
        return True
    
    def _check_terrain_goal_consistency(self, scenario, terrain, max_height, peak_x, peak_y):
        """检查地形与目标的一致性，记录警告和来源信息"""
        try:
            # 获取目标位置
            goal_pos = None
            if hasattr(scenario, 'goal_pos') and scenario.goal_pos is not None:
                goal_pos = scenario.goal_pos
            elif isinstance(scenario, dict) and 'goal_pos' in scenario:
                goal_pos = scenario['goal_pos']
            
            if goal_pos is not None:
                goal_x, goal_y, goal_z = goal_pos[0], goal_pos[1], goal_pos[2]
                
                # 计算目标点在地形上的高度
                terrain_height_at_goal = None
                if 0 <= int(goal_x) < terrain.shape[1] and 0 <= int(goal_y) < terrain.shape[0]:
                    terrain_height_at_goal = terrain[int(goal_y), int(goal_x)]
                
                # 检查一致性（仅在VIS_DEBUG模式下输出详细信息）
                import os as _vis_os
                if _vis_os.getenv('VIS_DEBUG', '0') == '1':
                    height_diff = abs(goal_z - max_height) if terrain_height_at_goal is not None else None
                    peak_goal_distance = np.sqrt((goal_x - peak_x)**2 + (goal_y - peak_y)**2)
                    
                    # 获取来源信息
                    terrain_source = getattr(scenario, 'terrain_source', 'unknown')
                    scenario_seed = getattr(scenario, 'scenario_seed', getattr(scenario, 'terrain_seed', 'unknown'))
                    episode_info = getattr(scenario, 'episode', 'unknown')
                    
                    # 记录详细信息
                    print(f"🔍 [一致性检查] 地形最高点: ({peak_x}, {peak_y}, {max_height:.1f})")
                    print(f"🔍 [一致性检查] 中央目标: ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f})")
                    print(f"🔍 [一致性检查] 目标点地形高度: {terrain_height_at_goal:.1f}" if terrain_height_at_goal is not None else "🔍 [一致性检查] 目标点地形高度: 超出范围")
                    print(f"🔍 [一致性检查] 地形来源: {terrain_source}, 种子: {scenario_seed}, 回合: {episode_info}")
                    print(f"🔍 [一致性检查] 高度差: {height_diff:.1f}, 距离: {peak_goal_distance:.1f}")
                    
                    # 警告条件
                    warnings = []
                    if height_diff is not None and height_diff > 10.0:
                        warnings.append(f"目标高度与地形最高点差异过大: {height_diff:.1f}m")
                    if peak_goal_distance > 20.0:
                        warnings.append(f"目标与山顶距离过远: {peak_goal_distance:.1f}m")
                    if terrain_source == 'world':
                        warnings.append("使用world.terrain作为地形来源，可能存在同步问题")
                    
                    if warnings:
                        print(f"⚠️ [一致性警告] {', '.join(warnings)}")
                    else:
                        print(f"✅ [一致性检查] 地形与目标基本一致")
                    
        except Exception as e:
            if _vis_os.getenv('VIS_DEBUG', '0') == '1':
                print(f"🔍 [一致性检查] 检查失败: {e}")
    
    def _generate_trajectory_with_actor_analysis(self, trajectories, scenario, save_path, 
                                               episode_num, reward, episode_type, 
                                               goal_positions, env_instance, actor_outputs_history,
                                               total_steps, effective_steps, elev=30, azim=45,
                                               env_idx=0):
        """生成包含3D轨迹和Actor输出分析的复合图像，并单独保存Actor输出序列图
        
        参数:
            trajectories: 智能体轨迹列表
            scenario: 场景对象
            save_path: 保存路径
            episode_num: 回合编号
            reward: 回合奖励
            episode_type: 回合类型
            goal_positions: 目标位置信息
            env_instance: 环境实例
            actor_outputs_history: Actor网络7维输出历史数据 [timestep][agent][7_dim]
            total_steps: 总步数
            effective_steps: 有效步数
            elev: 视角俯仰角
            azim: 视角方位角
        """
        try:
            # 1. 生成标准的3D轨迹图
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            
            # 绘制地形 - 优先使用环境实例的地形数据
            if env_instance is not None:
                self._plot_terrain_from_env(ax, env_instance)
            else:
                self._plot_terrain(ax, scenario)
            
            # 绘制障碍（优先实时环境）
            self._plot_obstacles(ax, scenario, env_instance=env_instance, env_idx=env_idx)
            
            # 绘制目标位置 - 优先使用环境实例的目标数据
            if env_instance is not None:
                self._plot_goal_from_env(ax, env_instance, goal_positions)
            else:
                self._plot_goal(ax, scenario, goal_positions=goal_positions)
            
            # 绘制智能体轨迹
            self._plot_trajectories(ax, trajectories)
            
            # 设置3D图标签和标题
            self._set_labels_and_title(ax, episode_type, episode_num, reward, None, 
                                     total_steps=total_steps, effective_steps=effective_steps)
            
            # 保存3D轨迹图
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            # 2. 单独生成Actor输出时间序列图
            self._generate_actor_outputs_sequence_image(
                actor_outputs_history, episode_num, reward, episode_type, save_path
            )
            
        except Exception as e:
            print(f"❌ 生成Actor分析图像失败: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals():
                plt.close(fig)
    
    def _generate_actor_outputs_sequence_image(self, actor_outputs_history, episode_num, reward, episode_type, original_save_path):
        """单独生成Actor输出时间序列图
        
        参数:
            actor_outputs_history: Actor网络7维输出历史数据 [timestep][agent][7_dim]
            episode_num: 回合编号
            reward: 回合奖励
            episode_type: 回合类型
            original_save_path: 原始保存路径，用于生成序列图路径
        """
        try:
            # 🔧 修复：正确处理numpy数组的检查，避免"The truth value of an array is ambiguous"错误
            if actor_outputs_history is None:
                print(f"🎨 无Actor输出数据，跳过序列图生成")
                return
            try:
                # 尝试获取长度（如果是数组或列表）
                if hasattr(actor_outputs_history, '__len__'):
                    if len(actor_outputs_history) == 0:
                        print(f"🎨 无Actor输出数据，跳过序列图生成")
                        return
                else:
                    print(f"🎨 Actor输出数据格式无效，跳过序列图生成")
                    return
            except (ValueError, TypeError) as e:
                # 如果是numpy数组且无法直接判断，尝试转换为列表
                try:
                    import numpy as np
                    if isinstance(actor_outputs_history, np.ndarray):
                        if actor_outputs_history.size == 0:
                            print(f"🎨 无Actor输出数据，跳过序列图生成")
                            return
                    else:
                        print(f"🎨 Actor输出数据格式无效，跳过序列图生成")
                        return
                except Exception:
                    print(f"🎨 Actor输出数据格式无效，跳过序列图生成")
                    return
            
            # 生成序列图保存路径
            import os
            base_path = os.path.splitext(original_save_path)[0]
            sequence_save_path = f"{base_path}_actor_sequence.png"
            
            # 创建独立的序列图
            fig = plt.figure(figsize=(16, 12))
            
            # 🔧 修复：将采样索引映射到实际时间步数
            # 从环境变量获取采样间隔，如果没有则默认10（与训练脚本一致）
            try:
                actor_output_interval = int(os.getenv('ACTOR_OUTPUT_SAMPLE_INTERVAL', '10'))
                if actor_output_interval <= 0:
                    actor_output_interval = 10
            except Exception:
                actor_output_interval = 10
            
            # 将采样索引映射到实际时间步数
            # 采样点0对应step=0，采样点1对应step=actor_output_interval，采样点2对应step=2*actor_output_interval，以此类推
            n_samples = len(actor_outputs_history)
            timesteps = [i * actor_output_interval for i in range(n_samples)]
            
            print(f"🎨 Actor输出时序图：采样间隔={actor_output_interval}步，采样点数={n_samples}，时间步范围=[{timesteps[0]}, {timesteps[-1]}]")
            
            n_agents = len(actor_outputs_history[0]) if actor_outputs_history[0] is not None else 0
            
            if n_agents == 0:
                print(f"🎨 无智能体数据，跳过序列图生成")
                return
            
            # 定义7维输出的名称和颜色
            # Actor网络7维输出：[ax, ay, az, k_att, k_rep, d0, mix]
            # 前3维：加速度动作；后4维：势场参数
            # mix参数 = 检测半径(r_safe)，控制地形检测的安全半径范围
            output_names = ['ax', 'ay', 'az', 'k_att', 'k_rep', 'd0', 'radius']
            output_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']
            
            # 定义智能体线条样式和透明度，支持更多智能体
            agent_linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5, 1, 5))]
            agent_alphas = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            
            # 创建2x2的子图布局
            # 子图1：加速度 (ax, ay, az)
            ax1 = fig.add_subplot(2, 2, 1)
            for agent_idx in range(n_agents):  # 显示所有智能体
                current_linestyle = agent_linestyles[agent_idx % len(agent_linestyles)]
                current_alpha = agent_alphas[agent_idx % len(agent_alphas)]
                for dim_idx in range(3):  # ax, ay, az
                    values = []
                    # 🔧 修复：使用采样索引访问数据，但用实际时间步数绘图
                    for sample_idx in range(n_samples):
                        try:
                            v = actor_outputs_history[sample_idx][agent_idx][dim_idx]
                        except Exception:
                            v = 0.0
                        try:
                            import numpy as _np
                            v = float(_np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
                        except Exception:
                            try:
                                v = float(v)
                            except Exception:
                                v = 0.0
                        values.append(v)
                    ax1.plot(timesteps, values, color=output_colors[dim_idx], 
                            linestyle=current_linestyle,
                            alpha=current_alpha,
                            label=f'{output_names[dim_idx]} (Agent {agent_idx})')
            
            ax1.set_title('Actor Output - Acceleration Actions', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Acceleration Values')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax1.set_ylim(-1.1, 1.1)
            
            # 子图2：势场参数 (k_att, k_rep, d0)
            ax2 = fig.add_subplot(2, 2, 2)
            for agent_idx in range(n_agents):  # 显示所有智能体
                current_linestyle = agent_linestyles[agent_idx % len(agent_linestyles)]
                current_alpha = agent_alphas[agent_idx % len(agent_alphas)]
                for dim_idx in range(3, 6):  # k_att, k_rep, d0
                    values = []
                    # 🔧 修复：使用采样索引访问数据，但用实际时间步数绘图
                    for sample_idx in range(n_samples):
                        try:
                            v = actor_outputs_history[sample_idx][agent_idx][dim_idx]
                        except Exception:
                            v = 0.0
                        try:
                            import numpy as _np
                            v = float(_np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
                        except Exception:
                            try:
                                v = float(v)
                            except Exception:
                                v = 0.0
                        values.append(v)
                    ax2.plot(timesteps, values, color=output_colors[dim_idx], 
                            linestyle=current_linestyle,
                            alpha=current_alpha,
                            label=f'{output_names[dim_idx]} (Agent {agent_idx})')
            
            ax2.set_title('Actor Output - Potential Field Parameters (Normalized)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time Steps')
            # 🚨 关键修复：说明这是归一化的pf_u值，不是实际的k_rep值
            # 实际k_rep = base_k_rep + pf_u * delta_k_rep，范围[40, 120]，不会为负
            ax2.set_ylabel('Normalized Parameter Values (pf_u, range [-1, 1])')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax2.set_ylim(-1.1, 1.1)
            
            # 子图3：混合参数 (mix)
            ax3 = fig.add_subplot(2, 2, 3)
            for agent_idx in range(n_agents):  # 显示所有智能体
                current_linestyle = agent_linestyles[agent_idx % len(agent_linestyles)]
                current_alpha = agent_alphas[agent_idx % len(agent_alphas)]
                values = []
                # 🔧 修复：使用采样索引访问数据，但用实际时间步数绘图
                for sample_idx in range(n_samples):
                    try:
                        v = actor_outputs_history[sample_idx][agent_idx][6]
                    except Exception:
                        v = 0.0
                    try:
                        import numpy as _np
                        v = float(_np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
                    except Exception:
                        try:
                            v = float(v)
                        except Exception:
                            v = 0.0
                    values.append(v)
                ax3.plot(timesteps, values, color=output_colors[6], 
                        linestyle=current_linestyle,
                        alpha=current_alpha,
                        label=f'{output_names[6]} (Agent {agent_idx})')
            
            ax3.set_title('Actor Output - Detection Radius', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Radius Values')
            ax3.grid(True, alpha=0.3)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax3.set_ylim(-1.1, 1.1)
            
            # 子图4：所有参数的综合视图
            ax4 = fig.add_subplot(2, 2, 4)
            for agent_idx in range(n_agents):  # 显示所有智能体
                current_linestyle = agent_linestyles[agent_idx % len(agent_linestyles)]
                current_alpha = agent_alphas[agent_idx % len(agent_alphas)]
                for dim_idx in range(7):  # 所有7个参数
                    values = []
                    # 🔧 修复：使用采样索引访问数据，但用实际时间步数绘图
                    for sample_idx in range(n_samples):
                        try:
                            v = actor_outputs_history[sample_idx][agent_idx][dim_idx]
                        except Exception:
                            v = 0.0
                        try:
                            import numpy as _np
                            v = float(_np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
                        except Exception:
                            try:
                                v = float(v)
                            except Exception:
                                v = 0.0
                        values.append(v)
                    ax4.plot(timesteps, values, color=output_colors[dim_idx], 
                            linestyle=current_linestyle,
                            alpha=current_alpha,
                            label=f'{output_names[dim_idx]} (Agent {agent_idx})')
            
            ax4.set_title('Actor Output - All Parameters Overview', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Time Steps')
            ax4.set_ylabel('Parameter Values')
            ax4.grid(True, alpha=0.3)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax4.set_ylim(-1.1, 1.1)
            
            # 添加总标题
            # 🔧 修复：显示实际时间步数而不是采样点数
            max_timestep = timesteps[-1] if timesteps else 0
            fig.suptitle(f'{episode_type.capitalize()} Episode {episode_num} - Reward: {reward:.2f}\n'
                        f'Actor Network 7D Output Time Series Analysis (Sampled Points: {len(timesteps)}, Time Steps: 0-{max_timestep})', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # 保存序列图
            plt.tight_layout()
            plt.savefig(sequence_save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Actor输出时间序列图已保存: {sequence_save_path}")
            
        except Exception as e:
            print(f"❌ 生成Actor输出时间序列图失败: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals():
                plt.close(fig)
    
    def _plot_actor_outputs_analysis(self, fig, actor_outputs_history, episode_num, reward, episode_type):
        """绘制Actor输出分析子图
        
        参数:
            fig: matplotlib图形对象
            actor_outputs_history: Actor网络7维输出历史数据 [timestep][agent][7_dim]
            episode_num: 回合编号
            reward: 回合奖励
            episode_type: 回合类型
        """
        try:
            if not actor_outputs_history or len(actor_outputs_history) == 0:
                return
            
            # 提取数据
            timesteps = list(range(len(actor_outputs_history)))
            n_agents = len(actor_outputs_history[0]) if actor_outputs_history[0] else 0
            
            if n_agents == 0:
                return
            
            # 定义7维输出的名称和颜色
            output_names = ['ax', 'ay', 'az', 'k_att', 'k_rep', 'd0', 'mix']
            output_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
            
            # 创建3个子图：加速度、势场参数、混合参数
            # 子图1：加速度 (ax, ay, az)
            ax1 = fig.add_subplot(2, 2, 3)
            for agent_idx in range(min(n_agents, 3)):  # 最多显示3个智能体
                for dim_idx in range(3):  # ax, ay, az
                    values = [actor_outputs_history[t][agent_idx][dim_idx] for t in timesteps]
                    ax1.plot(timesteps, values, color=output_colors[dim_idx], 
                            linestyle='-' if agent_idx == 0 else '--',
                            alpha=0.8 if agent_idx == 0 else 0.6,
                            label=f'{output_names[dim_idx]} (Agent {agent_idx})' if agent_idx == 0 else f'{output_names[dim_idx]} (Agent {agent_idx})')
            
            ax1.set_title('Actor输出 - 加速度动作', fontsize=12, fontweight='bold')
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('加速度值')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.set_ylim(-1.1, 1.1)
            
            # 子图2：势场参数 (k_att, k_rep, d0)
            ax2 = fig.add_subplot(2, 2, 4)
            for agent_idx in range(min(n_agents, 3)):  # 最多显示3个智能体
                for dim_idx in range(3, 6):  # k_att, k_rep, d0
                    values = [actor_outputs_history[t][agent_idx][dim_idx] for t in timesteps]
                    ax2.plot(timesteps, values, color=output_colors[dim_idx], 
                            linestyle='-' if agent_idx == 0 else '--',
                            alpha=0.8 if agent_idx == 0 else 0.6,
                            label=f'{output_names[dim_idx]} (Agent {agent_idx})' if agent_idx == 0 else f'{output_names[dim_idx]} (Agent {agent_idx})')
            
            ax2.set_title('Actor输出 - 势场参数', fontsize=12, fontweight='bold')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('参数值')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.set_ylim(-1.1, 1.1)
            
            # 添加总标题
            fig.suptitle(f'{episode_type.capitalize()} Episode {episode_num} - Reward: {reward:.2f}\n'
                        f'Actor网络7维输出分析 (总步数: {len(timesteps)})', 
                        fontsize=14, fontweight='bold', y=0.98)
            
        except Exception as e:
            print(f"❌ 绘制Actor输出分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_loss_plot(self, actor_losses, critic_losses, save_path, episode_num, reward):
        """生成单独的Loss曲线PNG图
        
        参数:
            actor_losses: Actor loss历史列表
            critic_losses: Critic loss历史列表
            save_path: 保存路径
            episode_num: 回合编号
            reward: 回合奖励
        """
        try:
            if (not actor_losses or len(actor_losses) == 0) and (not critic_losses or len(critic_losses) == 0):
                print("⚠️ 没有loss数据，跳过loss图生成")
                return
            
            print(f"🎨 开始生成Loss曲线图: {save_path}")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 上图：Critic Loss
            if critic_losses and len(critic_losses) > 0:
                steps = list(range(len(critic_losses)))
                # 过滤掉无效值（NaN、Inf、None）
                valid_critic = [(s, v) for s, v in zip(steps, critic_losses) 
                               if v is not None and np.isfinite(v)]
                if valid_critic:
                    steps_valid, values_valid = zip(*valid_critic)
                    ax1.plot(steps_valid, values_valid, color='red', linewidth=2, label='Critic Loss')
                    ax1.set_ylabel('Critic Loss', fontsize=12, fontweight='bold')
                    # 🔧 修复：只有在数据中有正值时才使用对数刻度
                    if any(v > 0 for v in values_valid):
                        ax1.set_yscale('log')  # 使用对数坐标
                    else:
                        ax1.set_yscale('linear')  # 如果没有正值，使用线性刻度
                    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
                    ax1.legend(loc='upper right', fontsize=10)
                    ax1.set_title(f'Episode {episode_num} - Critic Loss (reward={reward:.1f})', 
                                 fontsize=14, fontweight='bold')
                else:
                    ax1.text(0.5, 0.5, 'No Valid Critic Loss Data', 
                            ha='center', va='center', fontsize=14, transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'No Critic Loss Data', 
                        ha='center', va='center', fontsize=14, transform=ax1.transAxes)
            
            # 下图：Actor Loss
            if actor_losses and len(actor_losses) > 0:
                steps = list(range(len(actor_losses)))
                # 过滤掉无效值（NaN、Inf、None）
                valid_actor = [(s, v) for s, v in zip(steps, actor_losses) 
                              if v is not None and np.isfinite(v)]
                if valid_actor:
                    steps_valid, values_valid = zip(*valid_actor)
                    ax2.plot(steps_valid, values_valid, color='green', linewidth=2, label='Actor Loss')
                    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Actor Loss', fontsize=12, fontweight='bold')
                    # 🔧 修复：只有在数据中有正值时才使用对数刻度
                    if any(v > 0 for v in values_valid):
                        ax2.set_yscale('log')  # 使用对数坐标
                    else:
                        ax2.set_yscale('linear')  # 如果没有正值，使用线性刻度
                    ax2.grid(True, alpha=0.3, which='both', linestyle='--')
                    ax2.legend(loc='upper right', fontsize=10)
                    ax2.set_title(f'Episode {episode_num} - Actor Loss (reward={reward:.1f})', 
                                 fontsize=14, fontweight='bold')
                else:
                    ax2.text(0.5, 0.5, 'No Valid Actor Loss Data', 
                            ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No Actor Loss Data', 
                        ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Loss曲线图已保存: {save_path}")
            print(f"📊 Loss统计: Actor点数={len(actor_losses) if actor_losses else 0}, Critic点数={len(critic_losses) if critic_losses else 0}")
            
        except Exception as e:
            print(f"❌ 生成Loss曲线图失败: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals():
                plt.close(fig)

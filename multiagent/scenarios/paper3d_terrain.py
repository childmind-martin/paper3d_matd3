# 创建multiagent/scenarios/paper3d_terrain.py文件
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.ndimage import gaussian_filter, zoom
import os
import json
from scipy import signal
import time
import os
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

class Scenario(BaseScenario):
    """
    自定义3D地形场景
    特性：
    1. 可生成随机地形
    2. 支持3D移动
    3. 目标设置在山顶或任意高点
    """
    def __init__(self, use_fixed_positions=True, dynamic_first_time=True, fixed_positions=None, fixed_positions_file=None, random_terrain=False, seed=None, random_z0_positions=False):
        """
        场景初始化方法
        参数:
            use_fixed_positions: 是否使用固定位置
            dynamic_first_time: 第一次运行时是否动态生成位置
            fixed_positions: 固定位置数据
            fixed_positions_file: 固定位置数据文件路径
            random_terrain: 是否使用随机地形
            seed: 随机数种子
            random_z0_positions: 是否随机初始化智能体位置在地形上方
        """
        BaseScenario.__init__(self)
        self.seed = seed
        self.fixed_positions = fixed_positions
        self.fixed_positions_file = fixed_positions_file
        self.random_terrain = random_terrain
        self.use_fixed_positions = use_fixed_positions
        self.dynamic_first_time = dynamic_first_time
        self.positions_initialized = False
        self.init_count = 0
        # 地图尺寸：默认100，可通过环境变量 MAP_SIZE 覆盖
        try:
            self.map_size = int(os.getenv('MAP_SIZE', '100'))
        except Exception:
            self.map_size = 100
        self.terrain = None
        self.grid_points = None
        self.obstacles = []
        self.world = None
        self.goal_pos = None
        self.fixed_terrain = None
        self.fixed_terrain_file = None
        self.use_fixed_terrain = False
        
        # 初始化探索奖励相关参数
        self.visited_cells = {}
        self.exploration_reward_scale = 0.01
        self.cell_size = 5

        # 设置随机种子
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # 加载固定位置数据
        if self.fixed_positions is not None:
            print("[初始化] 使用传入的固定位置数据")
        elif fixed_positions_file is not None and os.path.exists(fixed_positions_file) and use_fixed_positions:
            try:
                # 确定文件类型（json或numpy）
                file_ext = os.path.splitext(fixed_positions_file)[1].lower()
                
                if file_ext == '.json':
                    # 从JSON文件加载
                    with open(fixed_positions_file, 'r') as f:
                        self.fixed_positions = json.load(f)
                    print(f"[初始化] 从JSON文件加载固定位置数据: {fixed_positions_file}")
                else:
                    # 从Numpy文件加载
                    from_file = np.load(fixed_positions_file)
                    self.fixed_positions = {
                        'agents': from_file['agents'].tolist() if 'agents' in from_file else [],
                        'goal': from_file['goal'].tolist() if 'goal' in from_file else [50, 50, 15]
                    }
                    print(f"[初始化] 从Numpy文件加载固定位置数据: {fixed_positions_file}")
            except Exception as e:
                print(f"[初始化] 加载固定位置数据失败: {e}")
                self.fixed_positions = None
        
        print(f"[初始化] 使用随机种子: {self.seed}")
        if self.fixed_positions is not None:
            print(f"[初始化] 使用传入的固定位置数据")
        else:
            print(f"[初始化] 将使用默认的固定位置")
        
        # 验证并调整智能体的初始位置
        if self.fixed_positions:
            if self.validate_and_adjust_fixed_positions():
                print(f"[初始化] 已调整智能体位置以满足最小距离要求")

        # 初始化新添加的参数
        self.random_z0_positions = random_z0_positions

    def make_world(self):
        """创建世界并设置环境"""
        world = World()
        
        # 设置世界属性
        world.dim_c = 0  # 通信维度
        world.dim_p = 3  # 位置维度（3D）
        world.collaborative = True  # 协作环境
        
        # 添加智能体
        num_agents = 3
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = False
            agent.size = 0.15
            agent.accel = 3.0
            agent.max_speed = 1.0
            agent.id = i  # 为每个智能体设置唯一ID
        
        # 添加地标（目标点）
        num_landmarks = 1
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        
        # 初始化visited_cells
        self.visited_cells = {i: set() for i in range(len(world.agents))}
        
        # 设置地形尺寸和生成地形（允许环境变量覆盖）
        try:
            self.map_size = int(os.getenv('MAP_SIZE', str(self.map_size)))
        except Exception:
            self.map_size = int(self.map_size)
        
        # 🔧 关键修复：在生成地形前保存原始种子
        if not hasattr(self, '_original_terrain_seed'):
            self._original_terrain_seed = self.seed
        
        if self.random_terrain:
            # 生成随机地形
            self.generate_terrain(seed=self.seed)
        else:
            # 使用预设地形
            self.generate_terrain(is_random=False)
            
        # 生成障碍物（如果还没有）
        if not hasattr(self, 'obstacles') or self.obstacles is None or len(self.obstacles) == 0:
            self.generate_obstacles()
        
        # 重置世界状态
        self.reset_world(world)

        # 同步地图尺寸到 world，并提供越界检测函数
        try:
            world.map_size = int(self.map_size)
            margin = 0.0
            def _is_within_bounds(pos):
                try:
                    x, y = float(pos[0]), float(pos[1])
                    return (margin <= x < world.map_size - margin) and (margin <= y < world.map_size - margin)
                except Exception:
                    return True
            world.is_within_bounds = _is_within_bounds
        except Exception:
            pass
        
        return world
        
    def update_visited_cells(self, world):
        """更新智能体访问过的网格单元"""
        for i, agent in enumerate(world.agents):
            try:
                # 将当前位置转换为网格单元
                cell_size = getattr(self, "cell_size", 5.0)  # 默认单元格大小为5
                cell_x = int(agent.state.p_pos[0] / cell_size)
                cell_y = int(agent.state.p_pos[1] / cell_size)
                current_cell = (cell_x, cell_y)
                
                # 添加到已访问集合
                if i in self.visited_cells:
                    self.visited_cells[i].add(current_cell)
                else:
                    self.visited_cells[i] = {current_cell}
            except Exception as e:
                print(f"更新已访问单元格时出错: {e}")
            # 确保 visited_cells 是字典
            if not isinstance(self.visited_cells, dict):
                self.visited_cells = {j: set() for j in range(len(world.agents))}
            # 添加当前单元格
            if i in self.visited_cells:
                try:
                    self.visited_cells[i].add(current_cell)
                except AttributeError:
                    self.visited_cells[i] = set([current_cell])
            else:
                self.visited_cells[i] = set([current_cell])

    def reset_world(self, world):
        """重置世界状态，初始化智能体和目标位置"""
        self.world = world  # 保存对世界对象的引用，以便后续使用
        
        # 🔧 关键修复：确保numpy随机状态使用原始地形种子，保证确定性
        if hasattr(self, '_original_terrain_seed') and self._original_terrain_seed is not None:
            np.random.seed(self._original_terrain_seed)
        elif self.seed is not None:
            np.random.seed(self.seed)
        
        # 如果没有地形或者每次重置都要重新生成地形，则生成地形
        if self.terrain is None:
            if self.random_terrain:
                self.generate_terrain(is_random=True, seed=self.seed)
            else:
                self.generate_terrain(is_random=False, seed=self.seed)
        
        # 如果没有障碍物，生成障碍物
        if not self.obstacles:
            self.generate_obstacles()
        
        # 更新智能体位置
        self.initialize_agent_positions(world)
        
        # 使用固定区域设置目标位置
        self.set_target_in_fixed_target_area(world)
        
        # 如果使用固定位置并且有固定目标
        if self.use_fixed_positions and self.fixed_positions and 'goal' in self.fixed_positions:
            goal_pos = self.fixed_positions['goal']
            if isinstance(goal_pos, list) and len(goal_pos) == 3:
                self.goal_pos = np.array(goal_pos)
                try:
                    if getattr(world, 'is_main_env', True):
                        print(f"使用固定目标位置: {self.goal_pos}")
                except Exception:
                    pass
        
        # 设置标志物位置（作为目标）
        for i, landmark in enumerate(world.landmarks):
            if i == 0:  # 只使用第一个标志物作为目标
                landmark.state.p_pos = self.goal_pos.copy()
                landmark.state.p_vel = np.zeros(3)
        
        try:
            if getattr(world, 'is_main_env', True):
                print(f"目标设置在山顶位置: {self.goal_pos}, 山顶高度: {self.goal_pos[2] - 10}")
        except Exception:
            pass
        
        # 重置智能体状态
        for agent in world.agents:
            agent.state.c = np.zeros(3)  # 清零通信状态
            agent.state.p_vel = np.zeros(3)  # 清零速度
            
            # 初始化用于奖励计算的智能体距离
            agent.last_goal_dist = np.linalg.norm(agent.state.p_pos - self.goal_pos)
            
            # 初始化调试信息
            if not hasattr(agent, 'debug_info'):
                agent.debug_info = {}
            agent.debug_info['total_penetration_count'] = 0
            agent.debug_info['path_length'] = 0.0
            agent.debug_info['start_pos'] = agent.state.p_pos.copy()
            
            # 保存初始位置，用于出发点停留检测
            agent.init_pos = agent.state.p_pos.copy()
            
            # 注意：不要重置initialized_for_reward标志，确保自适应学习率机制正常运行
            
            # 打印智能体位置信息
            print(f"智能体 {agent.id if hasattr(agent, 'id') else '?'} 位置: {agent.state.p_pos}")
        
        # 重置访问的单元格记录
        self.visited_cells = {}
        for agent in world.agents:
            if hasattr(agent, 'id'):
                self.visited_cells[agent.id] = set()
        
        return True

    def initialize_agent_positions(self, world):
        """初始化智能体位置"""
        # 检查是否使用论文公式生成的地形
        if hasattr(self, 'terrain_params') and self.terrain_params:
            # 使用固定起始区域放置智能体
            self.place_agents_in_fixed_start_area(world)
            return
        
        # 动态首次：第一次运行时随机生成，随后锁定为固定位置
        if getattr(self, 'dynamic_first_time', False) and not getattr(self, 'positions_initialized', False):
            # 1) 首次随机放置
            if hasattr(self, 'random_z0_positions') and self.random_z0_positions:
                self.place_agents_above_terrain(world)
            else:
                self.place_agents_on_horizontal_plane(world)

            # 2) 记录为固定位置，后续复用（确保复现性）
            try:
                agents_pos = [agent.state.p_pos.tolist() for agent in world.agents]
                goal_pos = self.goal_pos.tolist() if self.goal_pos is not None else [float(self.map_size/2), float(self.map_size/2), 15.0]
                self.fixed_positions = { 'agents': agents_pos, 'goal': goal_pos }
                self.use_fixed_positions = True
                self.positions_initialized = True
                print(f"[位置] 动态首次生效：已随机生成并锁定固定起始位置，共 {len(agents_pos)} 个")
            except Exception as e:
                print(f"[位置] 动态首次锁定固定位置失败: {e}")
            return

        # 非动态首次或已初始化：按配置放置
        if self.use_fixed_positions and self.fixed_positions is not None and 'agents' in self.fixed_positions:
            self.place_agents_from_fixed_positions(world)
        else:
            if hasattr(self, 'random_z0_positions') and self.random_z0_positions:
                self.place_agents_above_terrain(world)
            else:
                self.place_agents_on_horizontal_plane(world)
    
    def place_agents_from_fixed_positions(self, world):
        """从固定位置配置中放置智能体"""
        print("[位置] 使用固定位置数据放置智能体")
        
        # 使用固定位置前确保地形已生成
        if self.terrain is None:
            self.generate_terrain(is_random=self.random_terrain, seed=self.seed)
        
        for i, agent in enumerate(world.agents):
            if i < len(self.fixed_positions['agents']):
                pos = self.fixed_positions['agents'][i]
                
                # 确保位置是列表或数组并且有3个元素
                if isinstance(pos, (list, np.ndarray)) and len(pos) == 3:
                    x, y, z = pos
                    
                    # 确保在地图范围内
                    x = max(5, min(x, self.map_size - 5))
                    y = max(5, min(y, self.map_size - 5))
                    
                    # 直接使用配置文件中的z值，不再尝试修改或调整
                    agent.state.p_pos = np.array([float(x), float(y), float(z)])
                    print(f"[位置] 智能体 {i} 放置在固定位置: [{x}, {y}, {z}]")
                    
                    # 记录初始位置到debug_info中
                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    agent.debug_info['start_position'] = agent.state.p_pos.copy()
            else:
                # 如果固定位置数据不足，随机放置在水平面上
                self.place_agent_on_horizontal_plane(agent, world)
                
    def flatten_terrain_around_position(self, center_x, center_y, radius):
        """在指定位置周围创建平坦区域"""
        if self.terrain is None:
            return
            
        # 将浮点坐标转换为整数坐标
        cx = int(center_x)
        cy = int(center_y)
        
        # 确定平坦区域的边界
        x_min = max(0, cx - int(radius))
        x_max = min(self.map_size - 1, cx + int(radius) + 1)
        y_min = max(0, cy - int(radius))
        y_max = min(self.map_size - 1, cy + int(radius) + 1)
        
        # 记录原始高度用于平滑过渡
        original_height = self.terrain[cy, cx]
        
        # 平坦区域中将地形高度设置为零
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # 计算到中心的距离
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist <= radius:
                    # 在半径内将地形设置为接近0
                    self.terrain[y, x] = 0.0
                
        print(f"已在 ({center_x}, {center_y}) 周围创建半径为 {radius} 的平坦区域")
                
    def place_agents_on_horizontal_plane(self, world):
        """随机放置所有智能体在水平面上 (z=0)"""
        print("[位置] 随机初始化智能体位置在水平面上 (z=0)")
        
        # 已放置智能体的位置列表
        placed_positions = []
        min_distance_between_agents = 20.0  # 智能体之间的最小距离
        
        for agent in world.agents:
            self.place_agent_on_horizontal_plane(agent, world, placed_positions, min_distance_between_agents)

    def place_agent_on_horizontal_plane(self, agent, world, placed_positions=None, min_distance=20.0):
        """在水平面上(z=0)随机放置单个智能体"""
        if placed_positions is None:
            placed_positions = []
            
        max_attempts = 30
        attempt = 0
        placed = False
        
        # 根据目标位置确定智能体应该放在哪一侧
        goal_side_x = "left"
        goal_side_y = "top"
        map_center_x = self.map_size / 2
        map_center_y = self.map_size / 2
        
        # 判断目标在哪一侧
        if hasattr(self, 'goal_pos') and self.goal_pos is not None:
            if self.goal_pos[0] < map_center_x:
                goal_side_x = "left"
            else:
                goal_side_x = "right"
                
            if self.goal_pos[1] < map_center_y:
                goal_side_y = "top"
            else:
                goal_side_y = "bottom"
        
        # 设置智能体的初始化范围 - 根据图像确定的右下角蓝色平地区域
        # 当目标在左上方时，定位到更精确的右下角区域
        if goal_side_x == "left" and goal_side_y == "top":
            x_min = self.map_size * 0.7  # 使用70%~90%的区域
            x_max = self.map_size * 0.9
            y_min = self.map_size * 0.7
            y_max = self.map_size * 0.9
        # 当目标在右上方时，定位到左下角区域
        elif goal_side_x == "right" and goal_side_y == "top":
            x_min = self.map_size * 0.1
            x_max = self.map_size * 0.3
            y_min = self.map_size * 0.7
            y_max = self.map_size * 0.9
        # 当目标在左下方时，定位到右上角区域
        elif goal_side_x == "left" and goal_side_y == "bottom":
            x_min = self.map_size * 0.7
            x_max = self.map_size * 0.9
            y_min = self.map_size * 0.1
            y_max = self.map_size * 0.3
        # 当目标在右下方时，定位到左上角区域
        else:
            x_min = self.map_size * 0.1
            x_max = self.map_size * 0.3
            y_min = self.map_size * 0.1
            y_max = self.map_size * 0.3
        
        while attempt < max_attempts and not placed:
            # 在确定的区域内随机选择x,y坐标
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            # Z坐标固定为0（水平面）
            z = 0.0
            
            # 检查与其他已放置智能体的距离
            too_close = False
            for pos in placed_positions:
                dist = np.linalg.norm(np.array([x, y]) - pos[:2])
                if dist < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                # 位置符合要求
                agent.state.p_pos = np.array([float(x), float(y), float(z)])
                placed = True
                placed_positions.append(agent.state.p_pos.copy())
                print(f"[位置] 智能体放置在水平面上: [{x}, {y}, {z}] (尝试次数: {attempt+1})")
            else:
                attempt += 1
        
        if not placed:
            # 如果无法满足距离要求，放宽条件重试
            print("[警告] 放宽条件重试放置智能体")
            
            # 放宽距离限制
            relaxed_min_distance = min_distance * 0.7
            
            # 稍微扩大搜索范围但仍在合理区域内
            x_min_relaxed = max(5, x_min - self.map_size * 0.05)
            x_max_relaxed = min(self.map_size - 5, x_max + self.map_size * 0.05)
            y_min_relaxed = max(5, y_min - self.map_size * 0.05)
            y_max_relaxed = min(self.map_size - 5, y_max + self.map_size * 0.05)
            
            attempt = 0
            while attempt < max_attempts and not placed:
                # 在扩大的区域内随机选择x,y坐标
                x = np.random.uniform(x_min_relaxed, x_max_relaxed)
                y = np.random.uniform(y_min_relaxed, y_max_relaxed)
                z = 0.0  # 水平面
                
                # 检查与其他已放置智能体的距离（放宽后的）
                too_close = False
                for pos in placed_positions:
                    dist = np.linalg.norm(np.array([x, y]) - pos[:2])
                    if dist < relaxed_min_distance:
                        too_close = True
                        break
            
                if not too_close:
                    agent.state.p_pos = np.array([float(x), float(y), float(z)])
                    placed = True
                    placed_positions.append(agent.state.p_pos.copy())
                    print(f"[位置] 智能体放置在水平面上: [{x}, {y}, {z}] (放宽条件后，尝试次数: {attempt+1})")
                else:
                    attempt += 1
            
            if not placed:
                # 如果仍然无法满足，使用随机位置但尽量保持在合理区域内
                # 重新根据目标选择合适的四分之一区域位置
                if goal_side_x == "left" and goal_side_y == "top":
                    # 右下区域
                    x = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                    y = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                elif goal_side_x == "right" and goal_side_y == "top":
                    # 左下区域
                    x = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                    y = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                elif goal_side_x == "left" and goal_side_y == "bottom":
                    # 右上区域
                    x = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                    y = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                else:
                    # 左上区域
                    x = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                    y = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                
                # 确保在地图范围内
                x = max(5, min(x, self.map_size - 5))
                y = max(5, min(y, self.map_size - 5))
                z = 0.0
                
                agent.state.p_pos = np.array([float(x), float(y), float(z)])
                placed_positions.append(agent.state.p_pos.copy())
                print(f"[警告] 无法为智能体找到满足距离要求的位置，使用指定区域随机位置: [{x}, {y}, {z}]")
    
    def place_agents_above_terrain(self, world):
        """随机放置所有智能体在地形上方"""
        print("[位置] 随机初始化智能体位置在地形上方")
        
        # 已放置智能体的位置列表
        placed_positions = []
        min_distance_between_agents = 20.0  # 智能体之间的最小距离
        
        for agent in world.agents:
            self.place_agent_above_terrain(agent, world, placed_positions, min_distance_between_agents)
    
    def place_agent_above_terrain(self, agent, world, placed_positions=None, min_distance=20.0):
        """在地形上方随机放置单个智能体"""
        if placed_positions is None:
            placed_positions = []
            
        max_attempts = 30
        attempt = 0
        placed = False
        
        # 根据目标位置确定智能体应该放在哪一侧
        goal_side_x = "left"
        goal_side_y = "top"
        map_center_x = self.map_size / 2
        map_center_y = self.map_size / 2
        
        # 判断目标在哪一侧
        if hasattr(self, 'goal_pos') and self.goal_pos is not None:
            if self.goal_pos[0] < map_center_x:
                goal_side_x = "left"
            else:
                goal_side_x = "right"
                
            if self.goal_pos[1] < map_center_y:
                goal_side_y = "top"
            else:
                goal_side_y = "bottom"
        
        # 设置智能体的初始化范围 - 根据图像确定的右下角蓝色平地区域
        # 当目标在左上方时，定位到更精确的右下角区域
        if goal_side_x == "left" and goal_side_y == "top":
            x_min = self.map_size * 0.7  # 使用70%~90%的区域
            x_max = self.map_size * 0.9
            y_min = self.map_size * 0.7
            y_max = self.map_size * 0.9
        # 当目标在右上方时，定位到左下角区域
        elif goal_side_x == "right" and goal_side_y == "top":
            x_min = self.map_size * 0.1
            x_max = self.map_size * 0.3
            y_min = self.map_size * 0.7
            y_max = self.map_size * 0.9
        # 当目标在左下方时，定位到右上角区域
        elif goal_side_x == "left" and goal_side_y == "bottom":
            x_min = self.map_size * 0.7
            x_max = self.map_size * 0.9
            y_min = self.map_size * 0.1
            y_max = self.map_size * 0.3
        # 当目标在右下方时，定位到左上角区域
        else:
            x_min = self.map_size * 0.1
            x_max = self.map_size * 0.3
            y_min = self.map_size * 0.1
            y_max = self.map_size * 0.3
        
        while attempt < max_attempts and not placed:
            # 在确定的区域内随机选择x,y坐标
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            # 获取该位置的地形高度
            terrain_height = self.get_height_at(x, y)
            
            # Z坐标设置为地形高度上方5-10单位
            z = terrain_height + np.random.uniform(5.0, 10.0)
            
            # 检查与其他已放置智能体的距离
            too_close = False
            for pos in placed_positions:
                dist = np.linalg.norm(np.array([x, y]) - pos[:2])
                if dist < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                # 位置符合要求
                agent.state.p_pos = np.array([float(x), float(y), float(z)])
                placed = True
                placed_positions.append(agent.state.p_pos.copy())
                print(f"[位置] 智能体放置在地形上方: [{x}, {y}, {z}]，地形高度: {terrain_height} (尝试次数: {attempt+1})")
            else:
                attempt += 1
        
        if not placed:
            # 如果无法满足距离要求，放宽条件重试
            print("[警告] 放宽条件重试放置智能体")
            
            # 放宽距离限制
            relaxed_min_distance = min_distance * 0.7
            
            # 稍微扩大搜索范围但仍在合理区域内
            x_min_relaxed = max(5, x_min - self.map_size * 0.05)
            x_max_relaxed = min(self.map_size - 5, x_max + self.map_size * 0.05)
            y_min_relaxed = max(5, y_min - self.map_size * 0.05)
            y_max_relaxed = min(self.map_size - 5, y_max + self.map_size * 0.05)
            
            attempt = 0
            while attempt < max_attempts and not placed:
                # 在扩大的区域内随机选择x,y坐标
                x = np.random.uniform(x_min_relaxed, x_max_relaxed)
                y = np.random.uniform(y_min_relaxed, y_max_relaxed)
                
                # 获取该位置的地形高度
                terrain_height = self.get_height_at(x, y)
                
                # Z坐标设置为地形高度上方5-10单位
                z = terrain_height + np.random.uniform(5.0, 10.0)
                
                # 检查与其他已放置智能体的距离（放宽后的）
                too_close = False
                for pos in placed_positions:
                    dist = np.linalg.norm(np.array([x, y]) - pos[:2])
                    if dist < relaxed_min_distance:
                        too_close = True
                        break
            
                if not too_close:
                    agent.state.p_pos = np.array([float(x), float(y), float(z)])
                    placed = True
                    placed_positions.append(agent.state.p_pos.copy())
                    print(f"[位置] 智能体放置在地形上方: [{x}, {y}, {z}]，地形高度: {terrain_height} (放宽条件后，尝试次数: {attempt+1})")
                else:
                    attempt += 1
            
            if not placed:
                # 如果仍然无法满足，使用随机位置但尽量保持在合理区域内
                # 重新根据目标选择合适的四分之一区域位置
                if goal_side_x == "left" and goal_side_y == "top":
                    # 右下区域
                    x = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                    y = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                elif goal_side_x == "right" and goal_side_y == "top":
                    # 左下区域
                    x = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                    y = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                elif goal_side_x == "left" and goal_side_y == "bottom":
                    # 右上区域
                    x = np.random.uniform(self.map_size * 0.65, self.map_size * 0.95)
                    y = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                else:
                    # 左上区域
                    x = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                    y = np.random.uniform(self.map_size * 0.05, self.map_size * 0.35)
                
                # 确保在地图范围内
                x = max(5, min(x, self.map_size - 5))
                y = max(5, min(y, self.map_size - 5))
                terrain_height = self.get_height_at(x, y)
                z = terrain_height + 7.0  # 默认高度
                
                agent.state.p_pos = np.array([float(x), float(y), float(z)])
                placed_positions.append(agent.state.p_pos.copy())
                print(f"[警告] 无法为智能体找到满足距离要求的位置，使用指定区域随机位置: [{x}, {y}, {z}]，地形高度: {terrain_height}")
    
    def save_fixed_positions(self, file_path):
        """保存当前的固定位置到文件"""
        try:
            positions = {
                'agents': [agent.state.p_pos.tolist() for agent in self.world.agents],
                'goal': self.goal_pos.tolist() if self.goal_pos is not None else [50.0, 50.0, 15.0]
            }
            
            # 临时设置固定位置数据用于验证
            temp_fixed_positions = self.fixed_positions
            self.fixed_positions = positions
            
            # 验证并调整位置
            if self.validate_and_adjust_fixed_positions():
                print(f"保存前对智能体位置进行了调整以确保最小距离要求")
                # 使用调整后的位置
                positions = self.fixed_positions
            
            # 恢复原来的固定位置数据
            self.fixed_positions = temp_fixed_positions
            
            # 保存到文件
            with open(file_path, 'w') as f:
                json.dump(positions, f, indent=2)
                
            print(f"固定位置已保存到: {file_path}")
            print(f"智能体位置: {positions['agents']}")
            print(f"目标位置: {positions['goal']}")
            return True
        except Exception as e:
            print(f"保存固定位置时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_terrain(self, is_random=True, seed=None):
        """
        生成地形高度图，使用高斯山峰叠加方法创建真实自然的地形
        
        🔧 改进：从三角函数公式改为高斯山峰叠加，更符合真实地形特征
        """
        # 如果有固定地形数据，直接使用
        if self.use_fixed_terrain and self.terrain is not None:
            print(f"使用固定地形数据，跳过地形生成")
            return True
        
        # 🔧 关键修复：保存并使用原始地形种子，防止被episode seed覆盖
        if not hasattr(self, '_original_terrain_seed'):
            self._original_terrain_seed = seed if seed is not None else self.seed
        
        # 优先使用原始种子，确保固定地形模式下地形一致性
        terrain_seed = self._original_terrain_seed
        if terrain_seed is not None:
            np.random.seed(terrain_seed)
            print(f"[地形生成] 使用固定种子: {terrain_seed}")
        
        # 初始化地形数组
        self.terrain = np.zeros((self.map_size, self.map_size))
        
        # 生成网格，将坐标归一化到[-1, 1]范围（保持兼容性）
        x = np.linspace(-1, 1, self.map_size)
        y = np.linspace(-1, 1, self.map_size)
        X, Y = np.meshgrid(x, y)
        
        # 保存网格点以供可视化使用
        self.grid_points = (X, Y)
        
        try:
            # 🔧 新方法：使用高斯山峰叠加生成真实地形
            # 根据地图复杂度确定山峰数量
            terrain_complexity = getattr(self, 'terrain_complexity_level', 2)
            complexity_config = {
                1: {'num_peaks': 5, 'height_range': (50, 70), 'width_range': (15, 25), 'noise': 1.0},
                2: {'num_peaks': 6, 'height_range': (60, 80), 'width_range': (12, 22), 'noise': 1.5},
                3: {'num_peaks': 7, 'height_range': (70, 90), 'width_range': (10, 20), 'noise': 2.0},
                4: {'num_peaks': 8, 'height_range': (80, 100), 'width_range': (8, 18), 'noise': 2.5}
            }
            config = complexity_config.get(terrain_complexity, complexity_config[2])
            
            num_peaks = config['num_peaks']
            height_range = config['height_range']
            width_range = config['width_range']
            noise_scale = config['noise']
            
            # 🔧 确定起点区域（通常在地图角落，比如左下角SW象限）
            # 起点区域大小约为地图的15-20%，确保有足够空间
            map_size_int = int(self.map_size)
            start_area_size = int(map_size_int * 0.15)  # 起点区域大小
            start_area_margin = int(map_size_int * 0.05)  # 起点区域距离边缘的距离
            
            # 随机选择一个角落作为起点区域（保持一定的随机性，但确保是角落）
            corner_choice = np.random.randint(0, 4)  # 0=SW, 1=SE, 2=NW, 3=NE
            if corner_choice == 0:  # 西南角 (左下)
                start_area_x = (start_area_margin, start_area_margin + start_area_size)
                start_area_y = (start_area_margin, start_area_margin + start_area_size)
            elif corner_choice == 1:  # 东南角 (右下)
                start_area_x = (map_size_int - start_area_margin - start_area_size, map_size_int - start_area_margin)
                start_area_y = (start_area_margin, start_area_margin + start_area_size)
            elif corner_choice == 2:  # 西北角 (左上)
                start_area_x = (start_area_margin, start_area_margin + start_area_size)
                start_area_y = (map_size_int - start_area_margin - start_area_size, map_size_int - start_area_margin)
            else:  # 东北角 (右上)
                start_area_x = (map_size_int - start_area_margin - start_area_size, map_size_int - start_area_margin)
                start_area_y = (map_size_int - start_area_margin - start_area_size, map_size_int - start_area_margin)
            
            # 保存起点区域信息，供后续使用
            self.start_area = {
                'x_range': start_area_x,
                'y_range': start_area_y,
                'size': start_area_size
            }
            
            print(f"[地形生成] 起点区域: x=[{start_area_x[0]}, {start_area_x[1]}], y=[{start_area_y[0]}, {start_area_y[1]}]")
            
            # 随机生成山峰位置（避免边缘和重叠）
            peak_positions = []
            min_distance = self.map_size * 0.15  # 山峰间最小距离
            margin = self.map_size * 0.1  # 边缘留白
            
            # 🔧 确保起点区域与山峰保持足够距离
            start_area_center_x = (start_area_x[0] + start_area_x[1]) / 2
            start_area_center_y = (start_area_y[0] + start_area_y[1]) / 2
            min_distance_from_start = start_area_size * 1.5  # 山峰距离起点区域的最小距离
            
            for _ in range(num_peaks):
                attempts = 0
                while attempts < 50:
                    px = np.random.randint(int(margin), int(self.map_size - margin))
                    py = np.random.randint(int(margin), int(self.map_size - margin))
                    
                    # 🔧 检查是否距离起点区域太近
                    dist_from_start = np.sqrt((px - start_area_center_x)**2 + (py - start_area_center_y)**2)
                    if dist_from_start < min_distance_from_start:
                        attempts += 1
                        continue
                    
                    # 检查与已有山峰的距离
                    valid = True
                    for existing_px, existing_py in peak_positions:
                        dist = np.sqrt((px - existing_px)**2 + (py - existing_py)**2)
                        if dist < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        peak_positions.append((px, py))
                        break
                    attempts += 1
            
            # 为每个山峰添加高度（使用高斯分布）
            self.mountain_centers = []  # 保存山峰信息
            for px, py in peak_positions:
                height = np.random.uniform(*height_range)
                width = np.random.uniform(*width_range)
                
                # 使用高斯分布创建山峰
                for i in range(self.map_size):
                    for j in range(self.map_size):
                        dist = np.sqrt((i - px)**2 + (j - py)**2)
                        # 高斯函数：h * exp(-(dist^2) / (2 * width^2))
                        contribution = height * np.exp(-(dist**2) / (2 * width**2))
                        self.terrain[i, j] += contribution
                
                # 保存山峰信息（格式：x, y, z）
                self.mountain_centers.append((px, py, height))
            
            # 添加低强度噪声（保持山峰之间的通道清晰）
            noise = np.random.randn(self.map_size, self.map_size) * noise_scale
            self.terrain += noise
            
            # 🔧 对起点区域进行平坦化处理，确保无人机可以正常起飞
            # 将起点区域的高度设为低值（0-5米），并使用平滑过渡
            start_flat_height = np.random.uniform(0.0, 5.0)  # 起点区域的目标高度（0-5米）
            start_x0, start_x1 = int(start_area_x[0]), int(start_area_x[1])
            start_y0, start_y1 = int(start_area_y[0]), int(start_area_y[1])
            
            # 对起点区域内的所有点进行平坦化
            for i in range(start_y0, min(start_y1, map_size_int)):
                for j in range(start_x0, min(start_x1, map_size_int)):
                    # 使用平滑过渡，避免硬边界
                    # 计算到起点区域边界的距离，用于平滑过渡
                    dist_to_edge_x = min(j - start_x0, start_x1 - j) / (start_area_size / 2.0)
                    dist_to_edge_y = min(i - start_y0, start_y1 - i) / (start_area_size / 2.0)
                    blend_factor = min(dist_to_edge_x, dist_to_edge_y)
                    blend_factor = np.clip(blend_factor, 0.0, 1.0)
                    
                    # 在起点区域中心保持平坦，边缘平滑过渡到原始地形
                    target_height = start_flat_height * blend_factor + self.terrain[i, j] * (1.0 - blend_factor)
                    self.terrain[i, j] = target_height
            
            # 对起点区域进行轻微平滑，确保平坦度
            from scipy.ndimage import gaussian_filter
            smooth_region = self.terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)].copy()
            smooth_region = gaussian_filter(smooth_region, sigma=1.0)
            self.terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)] = smooth_region
            
            start_avg_height = np.mean(self.terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)])
            start_height_std = np.std(self.terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)])
            print(f"[地形生成] 起点区域平坦化完成: 平均高度={start_avg_height:.2f}m, 标准差={start_height_std:.2f}m")
            
            # 确保非负
            self.terrain = np.maximum(self.terrain, 0)
            
            # 保存参数供调试使用
            self.terrain_params = {
                'method': 'gaussian_peaks',
                'num_peaks': len(peak_positions),
                'terrain_complexity': terrain_complexity,
                'height_range': height_range,
                'width_range': width_range,
                'noise_scale': noise_scale
            }
            
            print(f"✅ 使用高斯山峰叠加生成地形 (复杂度: {terrain_complexity})")
            print(f"   山峰数量: {len(peak_positions)}, 高度范围: {height_range}, 宽度范围: {width_range}")
            print(f"   检测到 {len(self.mountain_centers)} 个山峰")
            
        except Exception as e:
            print(f"❌ 使用高斯山峰生成地形失败，切换到基本模式: {e}")
            import traceback
            traceback.print_exc()
            
            # 回退到基本地形生成
            self.terrain = np.random.rand(self.map_size, self.map_size) * 50
            self.mountain_centers = [(self.map_size//2, self.map_size//2, 50)]
        
        finally:
            # 确保地形为正值
            self.terrain = np.maximum(self.terrain, 0)
            
            # 限制地形高度，与paper3D.m一致
            self.terrain = np.minimum(self.terrain, 100)
            
        return True
    
    def _detect_peaks_from_formula(self):
        """从论文公式生成的地形中检测山峰位置"""
        try:
            from scipy.ndimage import maximum_filter
            from scipy.ndimage import label
            
            # 使用局部最大值检测山峰
            # 设置最小山峰高度阈值
            min_height = np.percentile(self.terrain, 70)  # 只考虑前30%的高点
            
            # 创建高度掩码
            height_mask = self.terrain >= min_height
            
            # 使用最大值滤波器检测局部峰值
            local_maxima = maximum_filter(self.terrain, size=10) == self.terrain
            peaks = local_maxima & height_mask
            
            # 标记连通区域
            labeled_peaks, num_peaks = label(peaks)
            
            mountain_centers = []
            
            # 为每个山峰计算中心位置和高度
            for i in range(1, num_peaks + 1):
                peak_mask = labeled_peaks == i
                peak_coords = np.where(peak_mask)
                
                if len(peak_coords[0]) > 0:
                    # 计算山峰中心（加权平均）
                    center_x = np.mean(peak_coords[1])
                    center_y = np.mean(peak_coords[0])
                    peak_height = np.max(self.terrain[peak_mask])
                    
                    # 转换为地图坐标
                    map_x = int(center_x)
                    map_y = int(center_y)
                    
                    # 确保坐标在地图范围内
                    map_x = max(0, min(map_x, self.map_size - 1))
                    map_y = max(0, min(map_y, self.map_size - 1))
                    
                    mountain_centers.append((map_x, map_y, peak_height))
            
            # 按高度排序，返回最高的山峰
            mountain_centers.sort(key=lambda x: x[2], reverse=True)
            
            # 限制山峰数量，避免过多山峰
            max_peaks = 8
            if len(mountain_centers) > max_peaks:
                mountain_centers = mountain_centers[:max_peaks]
            
            return mountain_centers
            
        except Exception as e:
            print(f"山峰检测失败: {e}")
            # 回退到简单方法：在地图中心放置一个山峰
            center = self.map_size // 2
            return [(center, center, np.max(self.terrain))]
    
    def get_fixed_start_area(self):
        """获取固定的起始区域（中央低洼区域）"""
        # 基于论文公式，中央区域通常是低洼的
        # 定义起始区域为地图中央的一个圆形区域
        center = self.map_size // 2
        radius = self.map_size // 6  # 起始区域半径
        
        return {
            'center': (center, center),
            'radius': radius,
            'min_height': 0,
            'max_height': 20  # 起始区域最大高度
        }
    
    def get_fixed_target_area(self):
        """获取固定的目标区域（外环山峰区域）"""
        # 基于论文公式，外环区域通常是山峰
        # 定义目标区域为外环的一个区域
        center = self.map_size // 2
        outer_radius = self.map_size // 3  # 外环半径
        
        return {
            'center': (center, center),
            'inner_radius': outer_radius - 20,
            'outer_radius': outer_radius + 20,
            'min_height': 60  # 目标区域最小高度
        }
    
    def place_agents_in_fixed_start_area(self, world):
        """在固定的起始区域放置智能体"""
        print("[位置] 在固定起始区域放置智能体")
        
        # 确保地形已生成
        if self.terrain is None:
            self.generate_terrain(is_random=self.random_terrain, seed=self.seed)
        
        start_area = self.get_fixed_start_area()
        center_x, center_y = start_area['center']
        radius = start_area['radius']
        max_height = start_area['max_height']
        
        # 已放置智能体的位置列表
        placed_positions = []
        min_distance_between_agents = 15.0
        
        for agent in world.agents:
            self.place_agent_in_start_area(agent, world, start_area, placed_positions, min_distance_between_agents)
    
    def place_agent_in_start_area(self, agent, world, start_area, placed_positions=None, min_distance=15.0):
        """在起始区域内放置单个智能体"""
        if placed_positions is None:
            placed_positions = []
            
        center_x, center_y = start_area['center']
        radius = start_area['radius']
        max_height = start_area['max_height']
        
        max_attempts = 50
        attempt = 0
        placed = False
        
        while attempt < max_attempts and not placed:
            # 在起始区域内随机选择位置
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius)
            
            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            
            # 确保坐标在地图范围内
            x = max(0, min(x, self.map_size - 1))
            y = max(0, min(y, self.map_size - 1))
            
            # 检查地形高度是否合适
            terrain_height = self.get_terrain_height(x, y)
            if terrain_height <= max_height:
                # 检查与已放置智能体的距离
                too_close = False
                for pos in placed_positions:
                    dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    # 设置智能体位置（在地形上方2米）
                    z = terrain_height + 2.0
                    agent.state.p_pos = np.array([float(x), float(y), float(z)])
                    placed_positions.append(agent.state.p_pos.copy())
                    placed = True
                    print(f"智能体 {agent.name} 放置在起始区域: [{x:.1f}, {y:.1f}, {z:.1f}]")
            
            attempt += 1
        
        if not placed:
            # 如果无法找到合适位置，使用起始区域中心
            x, y = center_x, center_y
            terrain_height = self.get_terrain_height(x, y)
            z = terrain_height + 2.0
            agent.state.p_pos = np.array([float(x), float(y), float(z)])
            placed_positions.append(agent.state.p_pos.copy())
            print(f"[警告] 智能体 {agent.name} 使用起始区域中心位置: [{x}, {y}, {z}]")
    
    def set_target_in_fixed_target_area(self, world):
        """在固定的目标区域设置目标点"""
        print("[目标] 在固定目标区域设置目标点")
        
        target_area = self.get_fixed_target_area()
        center_x, center_y = target_area['center']
        inner_radius = target_area['inner_radius']
        outer_radius = target_area['outer_radius']
        min_height = target_area['min_height']
        
        max_attempts = 100
        attempt = 0
        target_found = False
        
        while attempt < max_attempts and not target_found:
            # 在目标区域内随机选择位置
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(inner_radius, outer_radius)
            
            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            
            # 确保坐标在地图范围内
            x = max(0, min(x, self.map_size - 1))
            y = max(0, min(y, self.map_size - 1))
            
            # 检查地形高度是否合适
            terrain_height = self.get_terrain_height(x, y)
            if terrain_height >= min_height:
                # 设置目标位置
                z = terrain_height + 5.0  # 目标点在地形上方5米
                self.goal_pos = np.array([float(x), float(y), float(z)])
                target_found = True
                print(f"目标点设置在目标区域: [{x:.1f}, {y:.1f}, {z:.1f}]")
            
            attempt += 1
        
        if not target_found:
            # 如果无法找到合适位置，使用目标区域中心
            x, y = center_x, center_y
            terrain_height = self.get_terrain_height(x, y)
            z = terrain_height + 5.0
            self.goal_pos = np.array([float(x), float(y), float(z)])
            print(f"[警告] 目标点使用目标区域中心位置: [{x}, {y}, {z}]")
            
            print(f"地形生成完成，尺寸: {self.terrain.shape}, 最高点: {np.max(self.terrain):.2f}")
            
            # 记录地形复杂度（用于调试）
            self.terrain_complexity = {
                'max_height': np.max(self.terrain),
                'min_height': np.min(self.terrain),
                'avg_height': np.mean(self.terrain),
                'mountains': 1
            }
            
            return True
    
    def get_height_at(self, x, y):
        """获取指定位置的地形高度"""
        if self.terrain is None:
            return 0
        
        # 确保坐标在地图范围内
        x = max(0, min(int(x), self.map_size - 1))
        y = max(0, min(int(y), self.map_size - 1))
        
        return self.terrain[y, x]
    
    def generate_obstacles(self):
        """生成障碍物"""
        self.obstacles = []
        
        # 生成随机障碍物
        num_obstacles = 12  # 从5个增加到12个障碍物
        for i in range(num_obstacles):
            obstacle = {
                'center': [np.random.uniform(10, self.map_size - 10), 
                          np.random.uniform(10, self.map_size - 10), 
                          0],  # z坐标将在reset_world中更新
                'radius': np.random.uniform(1.5, 4.0)  # 增加障碍物尺寸
            }
            self.obstacles.append(obstacle)
            
        print(f"已生成 {len(self.obstacles)} 个障碍物")
    
    def reward(self, agent, world):
        """计算奖励值"""
        # 初始化奖励
        reward = 0
        
        # 检查是否需要初始化
        if not hasattr(agent, 'initialized_for_reward') or not agent.initialized_for_reward:
            # 初始化智能体的状态跟踪变量
            current_dist = np.linalg.norm(agent.state.p_pos - self.goal_pos)
            agent.last_goal_dist = current_dist
            # 初始化调试信息
            if not hasattr(agent, 'debug_info'):
                agent.debug_info = {}
            agent.initialized_for_reward = True
            # 第一次计算返回0
            return 0.0
            
        # 计算与目标的距离奖励
        if hasattr(agent, 'last_goal_dist') and agent.last_goal_dist is not None:
            current_dist = np.linalg.norm(agent.state.p_pos - self.goal_pos)
            
            # 与上一步比较，获得距离变化
            dist_change = agent.last_goal_dist - current_dist
            reward += dist_change * 15  # 放大距离变化奖励（从10增加到15）
            
            # 更新最后距离
            agent.last_goal_dist = current_dist
            
            # 增加与目标距离相关的奖励，距离越近奖励越高
            # 使用非线性函数使靠近目标时奖励增长更快
            proximity_reward = 30.0 / (current_dist + 5.0)  # 新增：基于距离的非线性奖励
            reward += proximity_reward
            
            # 如果非常接近目标，给予额外奖励
            if current_dist < 10:  # 扩大范围（从5增加到10）
                reward += (10 - current_dist) * 8  # 越近奖励越高（从5增加到8）
            
            # 添加到目标的各向异性奖励组件
            if hasattr(agent, 'state') and hasattr(agent.state, 'p_vel'):
                vel = agent.state.p_vel
                vel_magnitude = np.linalg.norm(vel)
                
                if vel_magnitude > 0:
                    vel_direction = vel / vel_magnitude
                    to_goal = self.goal_pos - agent.state.p_pos
                    to_goal_magnitude = np.linalg.norm(to_goal)
                    
                    if to_goal_magnitude > 0:
                        to_goal_direction = to_goal / to_goal_magnitude
                        alignment = np.dot(vel_direction, to_goal_direction)
                        
                        # 朝向目标移动得到额外奖励（增加奖励力度）
                        reward += alignment * 3  # 从2增加到3
        
        # 碰撞惩罚
        if agent.collide:
            for other in world.agents:
                if other is agent: 
                    continue
                
                # 计算碰撞 
                if self.is_collision(agent, other):
                    reward -= 10  # 严重惩罚碰撞
        
        # 与地形相关的奖励
        if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
            pos = agent.state.p_pos
            # 获取当前位置的地形高度
            terrain_height = self.get_height_at(pos[0], pos[1])
            
            # 穿透地形的惩罚
            if pos[2] < terrain_height:
                penetration_depth = terrain_height - pos[2]
                reward -= penetration_depth * 5  # 惩罚穿透
                
                # 记录穿透次数（用于调试）
                if hasattr(agent, 'debug_info'):
                    agent.debug_info['total_penetration_count'] = agent.debug_info.get('total_penetration_count', 0) + 1
                
            # 过高的惩罚（防止飞得太高）
            height_above_terrain = pos[2] - terrain_height
            if height_above_terrain > 20:
                reward -= (height_above_terrain - 20) * 0.1
        
        # 探索奖励 - 访问新区域得到奖励
        if hasattr(agent, 'id') and agent.id in self.visited_cells:
            exploration_reward = len(self.visited_cells[agent.id]) * 0.01 * self.exploration_reward_scale
            reward += exploration_reward
            
            # 记录探索单元格数（用于调试）
            if hasattr(agent, 'debug_info'):
                agent.debug_info['exploration_cells'] = len(self.visited_cells[agent.id])
        
        # 保持稳定的奖励 - 惩罚在出发点附近停留
        if hasattr(agent, 'state') and hasattr(agent.state, 'p_vel'):
            vel = agent.state.p_vel
            speed = np.linalg.norm(vel)
            
            # 惩罚在出发点附近停留
            if hasattr(agent, 'init_pos') and hasattr(agent.state, 'p_pos'):
                # 计算与初始位置的距离
                dist_to_start = np.linalg.norm(agent.state.p_pos - agent.init_pos)
                
                # 如果距离初始位置很近且速度很小，给予额外惩罚
                if dist_to_start < 5.0 and speed < 0.3:
                    start_penalty = (5.0 - dist_to_start) * 0.5 * (1.0 - speed/0.3)
                    reward -= start_penalty * 2.0  # 加大惩罚力度
                    
                    # 记录调试信息
                    if hasattr(agent, 'debug_info'):
                        agent.debug_info['start_penalty'] = start_penalty
        
        # 更新智能体的debug_info
        if hasattr(agent, 'debug_info'):
            agent.debug_info['reward'] = reward
            agent.debug_info['position'] = agent.state.p_pos.tolist()
            agent.debug_info['velocity'] = agent.state.p_vel.tolist()
            agent.debug_info['terrain_height'] = self.get_height_at(agent.state.p_pos[0], agent.state.p_pos[1])
            agent.debug_info['dist_to_goal'] = np.linalg.norm(agent.state.p_pos - self.goal_pos) if self.goal_pos is not None else 0
        
        return reward
    
    def observation(self, agent, world):
        """构建智能体的观察值 - 增强版
        
        特别强化对Z轴信息的处理，使动作网络能够更好地学习连续的垂直动作
        """
        # 安全检查：确保智能体有状态属性
        if not hasattr(agent, 'state'):
            print(f"警告: 智能体 {agent.name if hasattr(agent, 'name') else 'unknown'} 没有state属性")
            # 返回零向量作为默认观察
            return np.zeros(61, dtype=np.float32)  # 61是期望的观察维度
            
        # 安全检查：确保state有p_pos和p_vel属性
        if not hasattr(agent.state, 'p_pos') or not hasattr(agent.state, 'p_vel'):
            print(f"警告: 智能体 {agent.name if hasattr(agent, 'name') else 'unknown'} 的state对象不完整")
            return np.zeros(61, dtype=np.float32)
            
        # 智能体自身状态信息
        entity_pos = agent.state.p_pos
        entity_vel = agent.state.p_vel
        
        # 检查位置和速度数据是否有效
        if entity_pos is None or entity_vel is None:
            print(f"警告: 智能体 {agent.name if hasattr(agent, 'name') else 'unknown'} 的位置或速度为None")
            return np.zeros(61, dtype=np.float32)
        
        # Z轴特殊处理 - 将高度信息单独编码并增强权重
        # 这能够帮助网络更好地理解垂直位置信息
        try:
            z_pos = entity_pos[2]
            z_vel = entity_vel[2]
        except IndexError:
            print(f"警告: 智能体 {agent.name if hasattr(agent, 'name') else 'unknown'} 的位置或速度数据不完整")
            return np.zeros(61, dtype=np.float32)
        
        # 当前位置的地形高度和Z轴信息
        terrain_height = self.get_height_at(entity_pos[0], entity_pos[1])
        
        # 计算离地高度 - 重要的Z轴特征
        clearance = max(0, z_pos - terrain_height)
        # 归一化离地高度，增强表达能力
        norm_clearance = np.tanh(clearance / 20.0)  # 将高度差归一化到[-1,1]范围
        
        # 创建更丰富的Z轴特征 - 关键改进点
        z_features = [
            z_pos,                      # 原始高度
            z_vel,                      # 垂直速度
            z_pos - terrain_height,     # 相对地形高度
            norm_clearance,             # 归一化离地高度
            np.sin(z_pos * 0.1),        # 高度的周期特征
            np.cos(z_pos * 0.1),        # 周期特征的另一个分量
            np.tanh(z_vel),             # 归一化垂直速度
        ]
        
        # 计算周围地形梯度（改进版 - 更多采样点）
        dx1 = self.get_height_at(min(entity_pos[0] + 1, self.map_size - 1), entity_pos[1]) - terrain_height
        dy1 = self.get_height_at(entity_pos[0], min(entity_pos[1] + 1, self.map_size - 1)) - terrain_height
        dx2 = self.get_height_at(max(entity_pos[0] - 1, 0), entity_pos[1]) - terrain_height
        dy2 = self.get_height_at(entity_pos[0], max(entity_pos[1] - 1, 0)) - terrain_height
        
        # 计算更丰富的地形特征
        gradient = np.array([dx1, dy1, dx2, dy2])
        
        # 计算周围点的高度
        surround_heights = []
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        for dx, dy in directions:
            nx, ny = entity_pos[0] + dx * 5, entity_pos[1] + dy * 5
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                height = self.get_height_at(nx, ny)
                # 计算与当前高度差的特征
                height_diff = height - terrain_height
                # 归一化高度差
                norm_diff = np.tanh(height_diff / 10.0)
                surround_heights.append(norm_diff)
            else:
                surround_heights.append(0.0)
        
        # 获取前方位置的地形高度（沿移动方向）
        vel_direction = entity_vel[:2]  # 只考虑XY平面的速度方向
        vel_norm = np.linalg.norm(vel_direction)
        if vel_norm > 1e-6:
            vel_direction = vel_direction / vel_norm
        else:
            vel_direction = np.array([1.0, 0.0])  # 默认朝X轴正方向
        
        # 采样前方点的高度 - 增加多样性
        forward_heights = []
        distances = [5, 10, 15, 20, 25]  # 增加采样距离和数量
        for dist in distances:
            future_x = entity_pos[0] + vel_direction[0] * dist
            future_y = entity_pos[1] + vel_direction[1] * dist
            if 0 <= future_x < self.map_size and 0 <= future_y < self.map_size:
                future_height = self.get_height_at(future_x, future_y)
                # 计算高度差并归一化
                future_diff = (future_height - terrain_height) / 10.0
                # 应用tanh避免极端值
                norm_future_diff = np.tanh(future_diff)
                forward_heights.append(norm_future_diff)
                
                # 同时添加该位置的Z轴可能移动范围特征
                z_potential = np.tanh((future_height - z_pos) / 5.0)
                forward_heights.append(z_potential)
            else:
                forward_heights.append(0.0)
                forward_heights.append(0.0)
        
        # 其他智能体的相对位置和速度
        rel_pos = []
        rel_vel = []
        for other in world.agents:
            if other is agent: 
                continue
            # 计算相对位置 - 特别重视Z轴差异
            other_rel_pos = other.state.p_pos - entity_pos
            # 将Z轴相对位置单独处理，加大权重
            z_rel = other_rel_pos[2] * 2.0  # 增加Z轴权重
            other_rel_pos = np.concatenate([other_rel_pos, [z_rel]])  # 增强Z轴差异的表达
            rel_pos.append(other_rel_pos)
            
            # 相对速度 - 同样增强Z轴表达
            other_rel_vel = other.state.p_vel - entity_vel
            z_vel_rel = other_rel_vel[2] * 2.0  # 增加Z轴速度权重
            other_rel_vel = np.concatenate([other_rel_vel, [z_vel_rel]])
            rel_vel.append(other_rel_vel)
        
        # 目标的相对位置
        goal_pos = []
        for entity in world.landmarks:
            # 获取目标相对位置
            goal_rel_pos = entity.state.p_pos - entity_pos
            # 将Z轴相对位置单独处理，加大权重
            goal_z_rel = goal_rel_pos[2] * 2.0  # 增强Z轴方向
            # 计算XY平面距离和总距离
            xy_dist = np.linalg.norm(goal_rel_pos[:2])
            total_dist = np.linalg.norm(goal_rel_pos)
            # 创建更丰富的目标特征
            enhanced_goal_pos = np.concatenate([
                goal_rel_pos,           # 原始相对位置
                [goal_z_rel],           # 增强Z轴差异 
                [xy_dist / 100.0],      # XY平面距离
                [total_dist / 100.0],   # 总距离
                [goal_rel_pos[2] / (total_dist + 1e-6)]  # Z轴方向的归一化比例
            ])
            goal_pos.append(enhanced_goal_pos)
        
        # 增加Z轴可移动性分析
        z_mobility = []
        # 向上移动空间
        up_space = 100.0 - z_pos  # 假设最高高度为100
        norm_up_space = np.tanh(up_space / 20.0)
        # 向下移动空间
        down_space = z_pos - terrain_height
        norm_down_space = np.tanh(down_space / 10.0)
        # 合并移动性特征
        z_mobility.extend([norm_up_space, norm_down_space])
        
        # 构建完整的观察向量
        # 特别注意：将Z轴相关特征放在前面，提高它们的重要性
        try:
            obs_components = [
                # Z轴特征放在最前面以提高重要性
                np.array(z_features),        # 7D: Z轴增强特征
                np.array(z_mobility),        # 2D: Z轴移动性特征
                # 其他特征
                entity_pos,                  # 3D: 自身位置
                entity_vel,                  # 3D: 自身速度
                [terrain_height],            # 1D: 当前地形高度
                gradient,                    # 4D: 地形梯度（四个方向）
                np.array(surround_heights),  # 8D: 周围地形高度
                np.array(forward_heights),   # 10D: 前方地形高度和Z轴潜在移动
            ]
            
            # 安全添加其他智能体相关特征
            if len(rel_pos) > 0:
                obs_components.extend(rel_pos)
            if len(rel_vel) > 0:
                obs_components.extend(rel_vel)
            if len(goal_pos) > 0:
                obs_components.extend(goal_pos)
                
            # 拼接所有组件
            obs = np.concatenate(obs_components)
            
            # 确保观察值维度一致性
            expected_dim = 61  # 预期的观察维度
            if len(obs) < expected_dim:
                # 如果维度不足，填充零值
                padding = np.zeros(expected_dim - len(obs), dtype=np.float32)
                obs = np.concatenate([obs, padding])
            elif len(obs) > expected_dim:
                # 如果维度过多，截断
                obs = obs[:expected_dim]

            # 移除调试输出，减少控制台输出量
            # print(f"观察函数成功为智能体 {agent.name if hasattr(agent, 'name') else 'unknown'} 生成观察数据，维度: {len(obs)}")
            return obs
                
        except Exception as e:
            print(f"构建观察向量时出错: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(61, dtype=np.float32)

    def find_peak_positions(self, min_height=30, neighborhood_size=5, max_peaks=5):
        """
        在地形中查找局部最高点（山顶）
        参数:
            min_height: 最小高度，低于此高度的点不视为山顶
            neighborhood_size: 判断局部最高点的邻域大小
            max_peaks: 最多返回的山顶数量
            
        返回:
            山顶位置列表，按高度降序排列
        """
        if self.terrain is None:
            return []
            
        peaks = []
        # 更密集地搜索以找到真正的峰值
        for y in range(1, self.map_size - 1):
            for x in range(1, self.map_size - 1):
                height = self.terrain[y, x]
                
                # 只考虑超过最小高度的点
                if height < min_height:
                    continue
                
                # 检查是否是局部最高点
                is_peak = True  # 明确初始化is_peak变量
                
                # 检查邻域中是否有更高点
                for dy in range(-neighborhood_size, neighborhood_size + 1):
                    if not is_peak:  # 如果已经确定不是峰值，提前结束外层循环
                        break
                        
                    for dx in range(-neighborhood_size, neighborhood_size + 1):
                        if (dx == 0 and dy == 0):
                            continue
                        
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.map_size and 0 <= nx < self.map_size and 
                            self.terrain[ny, nx] > height):
                            is_peak = False
                            break
                
                # 如果是峰值，添加到列表
                if is_peak:
                    peaks.append({
                        'position': (x, y, height),
                        'height': height
                    })
        
        # 如果没有找到峰值，找到地形中的最高点
        if not peaks:
            max_height = np.max(self.terrain)
            max_indices = np.where(self.terrain == max_height)
            if len(max_indices[0]) > 0:
                idx = np.random.randint(0, len(max_indices[0]))
                y, x = max_indices[0][idx], max_indices[1][idx]
                peaks.append({
                    'position': (x, y, max_height),
                    'height': max_height
                })
        
        # 按高度降序排列，只返回指定数量的山顶
        return sorted(peaks, key=lambda p: p['height'], reverse=True)[:max_peaks]
    
    def is_collision(self, agent, entity):
        """检测智能体之间的碰撞"""
        delta_pos = agent.state.p_pos - entity.state.p_pos
        dist = np.linalg.norm(delta_pos)
        collision_dist = agent.size + entity.size
        return dist < collision_dist
        
    def regenerate_terrain(self, new_seed=None):
        """重新生成地形"""
        # 如果使用固定地形，则不进行重新生成
        if self.use_fixed_terrain:
            print(f"使用固定地形数据，跳过地形重新生成")
            return True
        
        old_seed = self.seed
        
        # 设置新的随机种子
        if new_seed is not None:
            np.random.seed(new_seed)
            self.seed = new_seed
        else:
            # 如果没有提供新种子，使用不同于当前种子的随机种子
            new_seed = np.random.randint(0, 100000)
            while new_seed == old_seed:
                new_seed = np.random.randint(0, 100000)
            np.random.seed(new_seed)
            self.seed = new_seed
            
        print(f"重新生成地形，新种子: {self.seed}")
        
        # 记住当前的智能体和目标位置
        agent_positions = []
        if hasattr(self, 'world') and self.world and hasattr(self.world, 'agents'):
            agent_positions = [agent.state.p_pos.copy() for agent in self.world.agents]
        goal_pos = None
        if hasattr(self, 'goal_pos') and self.goal_pos is not None:
            goal_pos = self.goal_pos.copy()
            
        # 重新生成地形
        self.generate_terrain(is_random=self.random_terrain, seed=self.seed)
        
        # 重新生成障碍物
        self.generate_obstacles()
        
        # 删除平坦化处理的代码，保持原始地形
        # 不再对智能体位置周围进行平坦化处理
            
        return True

    def validate_and_adjust_fixed_positions(self):
        """验证并调整固定位置以确保智能体之间的最小距离"""
        if not self.fixed_positions or 'agents' not in self.fixed_positions:
            return False
            
        min_required_distance = 20.0  # 智能体之间要求的最小距离
        positions = self.fixed_positions['agents']
        adjusted = False
        
        # 检查所有智能体对之间的距离
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1 = np.array(positions[i][:2])  # 只比较x,y坐标
                pos2 = np.array(positions[j][:2])
                
                # 计算当前距离
                dist = np.linalg.norm(pos1 - pos2)
                
                # 如果距离不足，调整位置
                if dist < min_required_distance:
                    print(f"[警告] 智能体 {i} 和 {j} 之间的距离 ({dist:.2f}) 小于要求的最小距离 ({min_required_distance})")
                    
                    # 计算方向向量
                    direction = pos2 - pos1
                    if np.linalg.norm(direction) < 0.001:  # 避免零向量
                        direction = np.array([1.0, 0.0])  # 默认方向
                    
                    # 将方向向量归一化
                    direction = direction / np.linalg.norm(direction)
                    
                    # 计算需要的位移
                    required_move = (min_required_distance - dist) / 2.0
                    
                    # 移动两个智能体，保持相同方向但增加距离
                    if required_move > 0:
                        new_pos1 = pos1 - direction * required_move
                        new_pos2 = pos2 + direction * required_move
                        
                        # 确保位置仍在地图范围内
                        margin = 5.0
                        new_pos1[0] = np.clip(new_pos1[0], margin, self.map_size - margin)
                        new_pos1[1] = np.clip(new_pos1[1], margin, self.map_size - margin)
                        new_pos2[0] = np.clip(new_pos2[0], margin, self.map_size - margin)
                        new_pos2[1] = np.clip(new_pos2[1], margin, self.map_size - margin)
                        
                        # 更新位置
                        positions[i][0] = float(new_pos1[0])
                        positions[i][1] = float(new_pos1[1])
                        positions[j][0] = float(new_pos2[0])
                        positions[j][1] = float(new_pos2[1])
                        
                        print(f"  调整后: 智能体 {i} 位置: [{positions[i][0]:.2f}, {positions[i][1]:.2f}, {positions[i][2]:.2f}]")
                        print(f"  调整后: 智能体 {j} 位置: [{positions[j][0]:.2f}, {positions[j][1]:.2f}, {positions[j][2]:.2f}]")
                        adjusted = True
        
        # 如果进行了调整，再次验证
        if adjusted:
            # 检查调整后是否所有智能体都满足距离要求
            all_valid = True
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    pos1 = np.array(positions[i][:2])
                    pos2 = np.array(positions[j][:2])
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < min_required_distance:
                        print(f"[警告] 调整后智能体 {i} 和 {j} 之间的距离 ({dist:.2f}) 仍小于要求值")
                        all_valid = False
            
            if all_valid:
                print("所有智能体位置现在满足最小距离要求")
            else:
                print("即使经过调整，部分智能体位置仍不满足最小距离要求")
        
        return adjusted

    def save_terrain_data(self, file_path):
        """保存当前地形数据到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 收集需要保存的地形数据
            terrain_data = {
                'terrain': self.terrain.tolist() if self.terrain is not None else None,
                'obstacles': self.obstacles,
                'goal_pos': self.goal_pos.tolist() if self.goal_pos is not None else None,
                'map_size': self.map_size
            }
            
            # 保存到文件
            with open(file_path, 'w') as f:
                json.dump(terrain_data, f, indent=2)
                
            print(f"已将地形数据保存到文件: {file_path}")
            return True
        except Exception as e:
            print(f"保存地形数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_terrain_data(self, file_path):
        """从文件加载地形数据"""
        try:
            if os.path.exists(file_path):
                # 从文件加载数据
                with open(file_path, 'r') as f:
                    terrain_data = json.load(f)
                
                # 更新地形数据
                if 'terrain' in terrain_data and terrain_data['terrain']:
                    self.terrain = np.array(terrain_data['terrain'])
                
                if 'obstacles' in terrain_data:
                    self.obstacles = terrain_data['obstacles']
                
                if 'goal_pos' in terrain_data and terrain_data['goal_pos']:
                    self.goal_pos = np.array(terrain_data['goal_pos'])
                    
                if 'map_size' in terrain_data:
                    self.map_size = terrain_data['map_size']
                    
                print(f"从文件 {file_path} 加载地形数据成功")
                print(f"地形尺寸: {self.terrain.shape}, 最高点: {np.max(self.terrain):.2f}")
                print(f"障碍物数量: {len(self.obstacles)}")
                print(f"目标位置: {self.goal_pos}")
                
                # 标记地形已从文件加载
                self.use_fixed_terrain = True
                self.fixed_terrain_file = file_path
                
                return True
            else:
                print(f"地形数据文件 {file_path} 不存在")
                return False
        except Exception as e:
            print(f"加载地形数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

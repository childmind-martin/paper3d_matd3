# 创建multiagent/scenarios/paper3d_terrain.py文件
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy import signal
import time
import os

class Scenario(BaseScenario):
    """
    自定义3D地形场景
    特性：
    1. 可生成随机地形
    2. 支持3D移动
    3. 目标设置在山顶或任意高点
    """
    def __init__(self, use_fixed_positions=True, dynamic_first_time=True, fixed_positions=None, fixed_positions_file=None, random_terrain=False, seed=None):
        """
        场景初始化方法
        参数:
            use_fixed_positions: 是否使用固定位置
            dynamic_first_time: 第一次运行时是否动态生成位置
            fixed_positions: 固定位置数据
            fixed_positions_file: 固定位置数据文件路径
            random_terrain: 是否使用随机地形
            seed: 随机数种子
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
            print(f"[初始化] 使用随机种子: {seed}")
        else:
            self.seed = np.random.randint(0, 10000)
            print(f"[初始化] 使用自动生成的随机种子: {self.seed}")
            
        # 固定位置相关设置
        self.use_fixed_positions = use_fixed_positions
        self.dynamic_first_time = dynamic_first_time
        
        # 初始化固定位置数据
        if fixed_positions is not None:
            self.fixed_positions = fixed_positions
            print(f"[初始化] 使用传入的固定位置数据")
        elif fixed_positions_file is not None and os.path.exists(fixed_positions_file) and use_fixed_positions:
            # 从文件加载固定位置数据
            try:
                import json
                with open(fixed_positions_file, 'r') as f:
                    self.fixed_positions = json.load(f)
                print(f"[初始化] 从文件加载固定位置数据: {fixed_positions_file}")
                if 'agents' in self.fixed_positions:
                    print(f"[初始化] 加载了{len(self.fixed_positions['agents'])}个智能体位置")
            except Exception as e:
                print(f"[初始化] 从文件加载固定位置数据失败: {e}")
                self.fixed_positions = {}
        else:
            self.fixed_positions = {}
            
        # 打印固定位置设置
        print(f"[初始化] 固定位置设置: use_fixed_positions={use_fixed_positions}, dynamic_first_time={dynamic_first_time}")
        if self.fixed_positions and len(self.fixed_positions) > 0:
            print(f"[初始化] 已加载固定位置数据")
            
            # 打印智能体和目标位置信息
            if 'agents' in self.fixed_positions:
                for i, pos in enumerate(self.fixed_positions['agents']):
                    print(f"[初始化] 智能体{i}固定位置: {pos}")
                    
            if 'goal' in self.fixed_positions:
                print(f"[初始化] 目标固定位置: {self.fixed_positions['goal']}")
        else:
            print("[初始化] 未提供固定位置数据，将在需要时动态生成")
        
        # 地形参数
        self.random_terrain = random_terrain
        self.use_obstacles = False  # 是否使用障碍物
        self.visited_cells = None  # 用于记录探索过的区域
        self.world_dim = 100.0  # 世界尺寸
        self.map_size = self.world_dim  # 添加map_size属性，确保与world_dim一致
        self.z_scale = 15.0  # 地形高度缩放因子
        self.difficulty = 0.8  # 地形难度系数，越高越难
        self.smoothness = 3  # 地形平滑系数，越高越平滑
        self.octaves = 5  # 噪声叠加次数，影响地形复杂度
        
        # 地形生成参数
        self.grid_dim = 100  # 地形网格维度
        self.water_level = 0.3  # 水平面高度
        
        # 创建随机数生成器
        self.rng = np.random.RandomState(self.seed)
        
        # 障碍物参数
        self.num_obstacles = 15  # 障碍物数量
        self.obstacle_size_range = (5, 8)  # 障碍物大小范围
        self.obstacle_height_boost = 2.0  # 障碍物高度提升
        
        # 智能体参数
        self.num_agents = 3  # 默认智能体数量
        
        # 目标相关参数
        self.goal_pos = None  # 目标位置
        self.goal_height_offset = 10.0  # 目标高度偏移
        
        # 探索奖励比例
        self.exploration_reward_scale = 1.0
        
        # 记录初始化次数，用于动态第一次生成
        self.init_count = 0
        
        # 生成地形
        if not random_terrain:
            print(f"[初始化] 开始生成固定地形...")
            self.terrain_seed = self.seed if seed is not None else 42
            print(f"[初始化] 使用地形种子: {self.terrain_seed}")
        else:
            print(f"[初始化] 开始生成随机地形...")
            self.terrain_seed = np.random.randint(0, 10000)
            print(f"[初始化] 使用随机地形种子: {self.terrain_seed}")
        
        # 生成地形
        self.generate_terrain()
        print("[初始化] 地形生成完成")
        
    def perlin_noise_2d(self, x, y):
        """
        生成二维柏林噪声
        这是一个简化的柏林噪声实现，用于地形生成
        
        参数:
            x, y: 坐标点
            
        返回:
            一个-1到1之间的噪声值
        """
        # 整数部分
        xi, yi = int(x), int(y)
        # 小数部分
        xf, yf = x - xi, y - yi
        
        # 平滑插值
        u = self.fade(xf)
        v = self.fade(yf)
        
        # 获取四个角落的哈希值
        n00 = self.gradient(self.hash(xi, yi), xf, yf)
        n01 = self.gradient(self.hash(xi, yi+1), xf, yf-1)
        n10 = self.gradient(self.hash(xi+1, yi), xf-1, yf)
        n11 = self.gradient(self.hash(xi+1, yi+1), xf-1, yf-1)
        
        # 插值
        x1 = self.lerp(n00, n10, u)
        x2 = self.lerp(n01, n11, u)
        
        # 最终噪声值
        return self.lerp(x1, x2, v)
    
    def fade(self, t):
        """平滑过渡函数"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, a, b, t):
        """线性插值"""
        return a + t * (b - a)
    
    def hash(self, x, y):
        """简单哈希函数，基于坐标生成一个伪随机值"""
        seed = self.terrain_seed if hasattr(self, 'terrain_seed') else 42
        return (x * 73856093 ^ y * 19349663 ^ seed) & 0xFFFFFFFF
    
    def gradient(self, hash_val, x, y):
        """生成梯度向量，并计算与(x,y)的点积"""
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y
        
    def regenerate_terrain(self, new_seed=None):
        """
        重新生成地形，可选择使用新的随机种子
        在训练/测试过程中可调用以动态改变环境
        """
        if new_seed is not None:
            self.seed = new_seed
            self.rng = np.random.RandomState(new_seed)
        else:
            # 如果没有提供新种子，使用随机种子
            self.seed = np.random.randint(0, 100000)
            self.rng = np.random.RandomState(self.seed)
            
        # 清空旧数据
        self.terrain = None
        self.obstacles = []
        self.terrain_complexity = {}
        
        # 重新生成地形
        self.generate_terrain()
        print(f"\n[地形重生成] Seed: {self.seed}")
        
        return self.terrain
        
    def generate_terrain(self):
        """
        生成3D地形
        """
        # 设置随机种子确保可重复性
        np.random.seed(self.terrain_seed)
        
        # 创建坐标网格
        x = np.linspace(0, self.world_dim, self.grid_dim)
        y = np.linspace(0, self.world_dim, self.grid_dim)
        xx, yy = np.meshgrid(x, y)
        
        # 初始化地形高度为0
        self.terrain_heights = np.zeros((self.grid_dim, self.grid_dim))
        
        # 使用柏林噪声生成基础地形
        for octave in range(self.octaves):
            scale = self.smoothness * (2 ** octave)
            weight = self.difficulty ** octave
            
            # 生成随机偏移
            offset_x = np.random.rand() * 100
            offset_y = np.random.rand() * 100
            
            # 计算该倍频的噪声值
            noise = np.zeros((self.grid_dim, self.grid_dim))
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    # 柏林噪声模拟
                    nx = (xx[i, j] / self.world_dim + offset_x) * scale
                    ny = (yy[i, j] / self.world_dim + offset_y) * scale
                    noise[i, j] = self.perlin_noise_2d(nx, ny)
            
            # 将该倍频的噪声添加到地形
            self.terrain_heights += noise * weight
        
        # 将噪声结果规范化到[0, 1]范围
        min_height = np.min(self.terrain_heights)
        max_height = np.max(self.terrain_heights)
        self.terrain_heights = (self.terrain_heights - min_height) / (max_height - min_height)
        
        # 应用高度缩放
        self.terrain_heights *= self.z_scale
        
        # 设置水平面
        water_level_height = self.water_level * self.z_scale
        self.terrain_heights = np.maximum(self.terrain_heights, water_level_height)
        
        # 同时设置terrain属性，确保兼容性
        self.terrain = self.terrain_heights
        
        print(f"[地形生成] 地形已生成: 尺寸={self.grid_dim}x{self.grid_dim}, 高度范围=[{np.min(self.terrain_heights):.1f}, {np.max(self.terrain_heights):.1f}]")

    def get_terrain_height(self, x, y):
        """
        获取指定位置的地形高度
        
        参数:
            x, y: 位置坐标
            
        返回:
            该位置的地形高度
        """
        try:
            # 将世界坐标转换为地形网格索引
            grid_x = int(x / self.world_dim * (self.grid_dim - 1))
            grid_y = int(y / self.world_dim * (self.grid_dim - 1))
            
            # 确保索引在有效范围内
            grid_x = max(0, min(grid_x, self.grid_dim - 1))
            grid_y = max(0, min(grid_y, self.grid_dim - 1))
            
            # 获取地形高度
            if hasattr(self, 'terrain_heights'):
                return self.terrain_heights[grid_y, grid_x]
            elif hasattr(self, 'terrain'):
                return self.terrain[grid_y, grid_x]
            else:
                print("[警告] 未找到地形数据，返回默认高度0")
                return 0
        except Exception as e:
            print(f"[错误] 获取地形高度失败: {e}")
            return 0

    def randomly_place_agents(self, world):
        """
        随机放置智能体和目标
        """
        # 初始化世界状态
        for agent in world.agents:
            agent.state.p_vel = np.zeros(3)  # 初始速度为0
            agent.state.c = np.zeros(world.dim_c)  # 初始通信状态为0
        
        # 首先为目标寻找位置
        self.generate_goal_position(world)
        
        # 为每个智能体寻找适当的初始位置
        for i, agent in enumerate(world.agents):
            # 确保智能体有id属性
            if not hasattr(agent, 'id'):
                agent.id = i
                
            # 尝试找到一个有效的位置
            valid_position = False
            attempt = 0
            max_attempts = 20
            
            while not valid_position and attempt < max_attempts:
                attempt += 1
                
                # 随机选择平面位置
                x = np.random.uniform(10, self.world_dim - 10)
                y = np.random.uniform(10, self.world_dim - 10)
                
                # 获取地形高度
                terrain_height = self.get_terrain_height(x, y)
                
                # 在地形之上设置初始高度，增加高度以避免穿透
                z = terrain_height + 5.0  # 从2.0增加到5.0
                
                # 设置智能体位置
                agent.state.p_pos = np.array([x, y, z])
                
                # 检查位置是否离目标太近
                if self.goal_pos is not None:
                    dist_to_goal = np.linalg.norm(agent.state.p_pos[:2] - self.goal_pos[:2])
                    if dist_to_goal < 20.0:  # 如果太近，重新生成
                        continue
                    
                # 检查是否与其他已放置的智能体太近
                too_close = False
                for j in range(i):
                    other_agent = world.agents[j]
                    dist = np.linalg.norm(agent.state.p_pos[:2] - other_agent.state.p_pos[:2])
                    if dist < 10.0:  # 如果太近，重新生成
                        too_close = True
                        break
                
                if not too_close:
                    valid_position = True
            
            print(f"[智能体放置] 智能体{i}位置设置为: {agent.state.p_pos}, 地形高度: {terrain_height:.1f}")
            
        print(f"[智能体放置] 所有智能体放置完成")

    def generate_goal_position(self, world):
        """
        生成目标位置
        """
        # 尝试找到一个合适的目标位置
        max_attempts = 20
        for attempt in range(max_attempts):
            # 随机选择平面位置
            x = np.random.uniform(20, self.world_dim - 20)
            y = np.random.uniform(20, self.world_dim - 20)
            
            # 获取该点的地形高度
            terrain_height = self.get_terrain_height(x, y)
            
            # 设置目标的高度（在地形之上）
            z = terrain_height + self.goal_height_offset
            
            # 设置目标位置
            self.goal_pos = np.array([x, y, z])
            
            # 目标位置有效，结束循环
            break
            
        print(f"[目标生成] 目标位置设置为: {self.goal_pos}, 地形高度: {terrain_height:.1f}")
        return self.goal_pos
        
    def make_world(self):
        world = World()
        world.dim_p = 3  # 3D环境
        
        # 创建智能体
        num_agents = 3
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.id = i  # 添加唯一id属性
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            # 确保地形文件生成之后再进行属性设置
            if hasattr(agent, 'max_speed'):
                agent.max_speed = 1.2  # 原始值是1.0，调整到1.2
            if hasattr(agent, 'color'):
                agent.color = np.array([0.35, 0.35, 0.85])
        
        # 创建目标点 - 放在远处，使得智能体需要穿越地形
        world.landmarks = [Landmark()]
        goal = world.landmarks[0]
        goal.name = 'goal'
        goal.collide = False
        goal.movable = False
        goal.size = 0.1
        if hasattr(goal, 'color'):
            goal.color = np.array([0.85, 0.85, 0.35])
        
        # 创建障碍物 - 使用球形障碍物，模拟雷达系统
        num_obstacles = 3
        for i in range(num_obstacles):
            obstacle = Landmark()
            obstacle.name = f'obstacle_{i}'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.15
            if hasattr(obstacle, 'color'):
                obstacle.color = np.array([0.75, 0.25, 0.25])  # 更红的颜色，更明显
            world.landmarks.append(obstacle)
        
        # 确保地形已经生成，如果还没有，则生成
        if self.terrain is None:
            print("\n==========================================")
            print(f"在make_world中初始化地形（这是预期行为）")
            print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"当前种子: {self.seed}")
            print("==========================================\n")
            self.generate_terrain()
            
        # 生成障碍物（如果还没有）
        if not hasattr(self, 'obstacles') or self.obstacles is None or len(self.obstacles) == 0:
            self.generate_obstacles()
        
        self.reset_world(world)
        return world
        
    def reset_world(self, world):
        """
        重置世界状态，设置智能体和目标位置
        """
        # 增加初始化计数
        self.init_count += 1
        
        # 简化的位置初始化输出信息
        position_mode = "固定" if self.use_fixed_positions else "随机"
        print(f"[位置初始化 #{self.init_count}] 模式: {position_mode}")
            
        # 检查是否需要重新生成地形（仅在随机地形模式下）
        if self.random_terrain and self.init_count > 1:
            print(f"[地形] 重新生成随机地形，种子: {self.terrain_seed}")
            self.terrain_seed = np.random.randint(0, 10000)
            self.generate_terrain()

        # 智能体位置初始化
        if self.use_fixed_positions:
            # 检查是否是第一次运行且需要动态生成
            if self.init_count == 1 and self.dynamic_first_time:
                print("[位置] 首次运行 + dynamic_first_time=True → 动态生成并保存固定位置")
                self.randomly_place_agents(world)
                # 保存生成的位置作为固定位置，确保是字典格式
                self.fixed_positions = {
                    'agents': [agent.state.p_pos.copy() for agent in world.agents],
                    'goal': self.goal_pos.copy() if self.goal_pos is not None else None
                }
                print(f"[位置] 已生成固定位置数据: {len(world.agents)}个智能体 + 1个目标")
            else:
                # 使用已有的固定位置
                if self.fixed_positions and isinstance(self.fixed_positions, dict) and 'agents' in self.fixed_positions and len(self.fixed_positions['agents']) == len(world.agents):
                    for i, agent in enumerate(world.agents):
                        agent.state.p_pos = np.array(self.fixed_positions['agents'][i])
                        agent.state.p_vel = np.zeros(3)
                        agent.state.c = np.zeros(world.dim_c)
                        # 确保agent有id属性
                        if not hasattr(agent, 'id'):
                            agent.id = i
                
                # 设置目标位置
                if 'goal' in self.fixed_positions and self.fixed_positions['goal'] is not None:
                    self.goal_pos = np.array(self.fixed_positions['goal'])
                else:
                    # 如果没有保存目标位置，重新生成
                    print("[位置] 警告: 固定位置数据中没有目标位置，重新生成")
                    self.generate_goal_position(world)
        else:
            # 固定位置数据不完整，发出警告并使用随机位置
            if self.fixed_positions is None:
                missing_reason = "fixed_positions为None"
            elif not isinstance(self.fixed_positions, dict):
                missing_reason = f"fixed_positions类型错误，不是字典而是{type(self.fixed_positions)}"
            elif 'agents' not in self.fixed_positions:
                missing_reason = "fixed_positions字典中缺少'agents'键"
            elif len(self.fixed_positions['agents']) != len(world.agents):
                missing_reason = f"智能体数量不匹配: {len(self.fixed_positions['agents'])}(数据) != {len(world.agents)}(世界)"
            
            print(f"[位置] 警告: 固定位置数据不完整 ({missing_reason})")
            print(f"[位置] 切换到随机位置模式")
            self.randomly_place_agents(world)
            
            # 确保所有agent都有id属性
            for i, agent in enumerate(world.agents):
                if not hasattr(agent, 'id'):
                    agent.id = i
        
        # 重置障碍物位置
        if hasattr(self, 'obstacles') and self.obstacles:
            for i, obstacle in enumerate(self.obstacles):
                if i + 1 < len(world.landmarks):  # 第一个landmark是目标
                    world.landmarks[i + 1].state.p_pos = obstacle['center'].copy()
            
        # 设置目标位置 - 使用最新的goal_pos
        if self.goal_pos is not None and len(world.landmarks) > 0:
            world.landmarks[0].state.p_pos = self.goal_pos.copy()
            
        # 重置已访问区域
        if self.exploration_reward_scale > 0:
            self.visited_cells = {i: set() for i in range(len(world.agents))}
            
        # 验证目标和智能体坐标（调试用）
        if False:  # 仅在需要调试时开启
            self.validate_goal_coordinates(world)
        
        return self.goal_pos, [agent.state.p_pos.copy() for agent in world.agents]

    def generate_obstacles(self):
        """生成更多障碍物，围绕关键路径和目标区域"""
        self.obstacles = []
        
        # 使用设定的障碍物数量
        num_obstacles = self.num_obstacles
        
        # 障碍物分布策略：
        # 1. 在地图四周放置一些障碍物
        # 2. 在地图中央区域放置一些障碍物
        # 3. 在通往山顶的路径上放置一些障碍物
        
        # 地图区域划分
        regions = [
            # 左侧区域
            {"min_x": 10, "max_x": 30, "min_y": 10, "max_y": 90},
            # 右侧区域
            {"min_x": 70, "max_x": 90, "min_y": 10, "max_y": 90},
            # 上方区域
            {"min_x": 30, "max_x": 70, "min_y": 70, "max_y": 90},
            # 下方区域
            {"min_x": 30, "max_x": 70, "min_y": 10, "max_y": 30},
            # 中央区域
            {"min_x": 30, "max_x": 70, "min_y": 30, "max_y": 70},
        ]
        
        # 为每个区域分配障碍物
        obstacles_per_region = max(1, num_obstacles // len(regions))
        remaining_obstacles = num_obstacles % len(regions)
        
        for i, region in enumerate(regions):
            # 确定当前区域的障碍物数量
            region_obstacles = obstacles_per_region
            if i < remaining_obstacles:
                region_obstacles += 1
                
            for j in range(region_obstacles):
                # 在区域内随机选择位置
                center_x = self.rng.randint(region["min_x"], region["max_x"])
                center_y = self.rng.randint(region["min_y"], region["max_y"])
                
                # 随机选择障碍物尺寸
                radius = self.rng.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                
                # 获取地形高度，并增加高度以确保障碍物突出
                terrain_height = self.get_terrain_height(center_x, center_y)
                center_z = terrain_height + self.obstacle_height_boost
                
                # 添加障碍物
                self.obstacles.append({'center': [center_x, center_y, center_z], 'radius': radius})
        
        return self.obstacles
        
    def reward(self, agent, world):
        """
        奖励函数计算：
        1. 靠近目标的奖励
        2. 探索新区域的奖励
        3. 维持智能体在地形上方的约束
        """
        # 初始化调试信息字典
        if not hasattr(agent, 'debug_info'):
            agent.debug_info = {
                'penetration_count': 0,
                'position': agent.state.p_pos.copy(),
                'terrain_height': 0,
                'dist_to_goal': 0,
                'exploration_cells': 0
            }
            
        # 初始化奖励
        rew = 0
        
        # 获取智能体当前位置
        pos = agent.state.p_pos
        agent.debug_info['position'] = pos.copy()
        
        # 地形穿透检测与惩罚
        terrain_height = self.get_terrain_height(pos[0], pos[1])
        agent.debug_info['terrain_height'] = terrain_height
        if pos[2] < terrain_height:
            penetration_depth = terrain_height - pos[2]
            # 地形穿透惩罚，与穿透深度成正比
            terrain_penalty = -5.0 * penetration_depth
            rew += terrain_penalty
            
            # 严重穿透时输出警告
            agent.debug_info['penetration_count'] += 1
            if penetration_depth > 5.0:
                # 确保agent有id属性
                agent_id = getattr(agent, 'id', 0)
                print(f"警告: 智能体{agent_id}严重穿透地形！深度={penetration_depth:.2f}, 位置={pos}, 地形高度={terrain_height:.2f}")
        
        # 障碍物碰撞检测与惩罚
        for obstacle in getattr(self, 'obstacles', []):
            obstacle_center = np.array(obstacle['center'])
            obstacle_radius = obstacle['radius']
            
            # 计算智能体到障碍物中心的距离
            dist_to_obstacle = np.linalg.norm(pos - obstacle_center)
            
            # 如果距离小于障碍物半径，给予惩罚
            if dist_to_obstacle < obstacle_radius:
                obstacle_penalty = -5.0 * (obstacle_radius - dist_to_obstacle)
                rew += obstacle_penalty
        
        # 静止惩罚 - 如果智能体几乎不动，给予小惩罚以鼓励移动
        if np.linalg.norm(agent.state.p_vel) < 0.1:
            rew -= 0.1
        
        # 探索奖励 - 鼓励智能体探索未访问过的区域
        # 确保agent有id属性
        agent_idx = getattr(agent, 'id', 0)  # 如果没有id属性，默认使用0
        cell_x = int(pos[0] / 2)
        cell_y = int(pos[1] / 2)
        cell_key = (cell_x, cell_y)
        
        # 确保visited_cells是字典，并且包含agent_idx键
        if not isinstance(self.visited_cells, dict):
            self.visited_cells = {i: set() for i in range(len(world.agents))
        # 确保 visited_cells 初始化为字典
        if not isinstance(self.visited_cells, dict):}
        if agent_idx not in self.visited_cells:
            self.visited_cells[agent_idx] = set()
            
        # 如果这个区域是新访问的，给予奖励
        try:
            if cell_key not in self.visited_cells[agent_idx]:
                self.visited_cells[agent_idx].add(cell_key)
                exploration_reward = self.exploration_reward_scale
                rew += exploration_reward
                
            # 更新探索单元格数量
            agent.debug_info['exploration_cells'] = len(self.visited_cells[agent_idx])
        except Exception as e:    print(f"智能体 {agent_idx} 奖励计算错误: {e}")
            # 确保 visited_cells 是字典
            if not isinstance(self.visited_cells, dict):
                self.visited_cells = {i: set() for i in range(len(world.agents))}
            # 确保 agent_idx 在字典中
            if agent_idx not in self.visited_cells:
                self.visited_cells[agent_idx] = set()
            # 使用 set 的 add 方法
            self.visited_cells[agent_idx] = set([cell_key])
            print(f"智能体 {agent_idx} 奖励计算错误: {e}")
            # 确保初始化正确
            self.visited_cells[agent_idx] = set([cell_key])
        
        # 距离目标的奖励
        if self.goal_pos is not None:
            # 计算到目标的距离
            dist_to_goal = np.linalg.norm(pos - self.goal_pos)
            agent.debug_info['dist_to_goal'] = dist_to_goal
            
            # 距离惩罚 - 距离越远惩罚越大
            distance_penalty = -0.1 * dist_to_goal
            rew += distance_penalty
            
            # 距离变化奖励 - 如果距离减小，给予奖励
            if hasattr(agent, 'prev_dist_to_goal'):
                dist_change = agent.prev_dist_to_goal - dist_to_goal
                if dist_change > 0:  # 距离减小
                    rew += 0.5 * dist_change
            
            # 更新前一步的距离
            agent.prev_dist_to_goal = dist_to_goal
            
            # 如果非常接近目标，给予额外奖励
            if dist_to_goal < 5.0:
                rew += 5.0 * (5.0 - dist_to_goal)
        
        # 方向一致性奖励 - 鼓励智能体朝着目标的方向移动
        if self.goal_pos is not None and np.linalg.norm(agent.state.p_vel) > 0.5:
            # 计算速度方向
            vel_direction = agent.state.p_vel / np.linalg.norm(agent.state.p_vel)
            
            # 计算到目标的方向
            to_goal = self.goal_pos - pos
            if np.linalg.norm(to_goal) > 0:
                to_goal = to_goal / np.linalg.norm(to_goal)
                
                # 计算方向一致性（点积）
                direction_alignment = np.dot(vel_direction, to_goal)
                if direction_alignment > 0:  # 如果朝着目标方向
                    rew += 0.2 * direction_alignment
        
        # 更新调试信息
        agent.debug_info.update({
            'reward': rew,
            'position': pos.copy(),
            'velocity': agent.state.p_vel.copy(),
            'terrain_height': terrain_height,
            'dist_to_goal': agent.debug_info.get('dist_to_goal', 0),
            'energy': np.sum(np.square(agent.state.p_vel))
        })
        
        return rew
    
    def is_collision(self, agent, entity):
        delta_pos = agent.state.p_pos - entity.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = agent.size + entity.size
        return dist < dist_min
    
    def observation(self, agent, world):
        """确保观察空间始终包含目标方向信息，但显著降低其权重"""
        # 目标方向信息 - 大幅降低权重
        goal_info = []
        if hasattr(self, 'goal_pos') and self.goal_pos is not None:
            # 计算到目标的向量和距离
            goal_rel_pos = self.goal_pos - agent.state.p_pos
            dist_to_goal = np.linalg.norm(goal_rel_pos)
            # 归一化方向向量
            norm_direction = goal_rel_pos / (dist_to_goal + 1e-6)
            
            # 目标信息在前面，但极大降低其权重
            goal_info = np.concatenate([
                norm_direction * 0.3,  # 从0.8进一步降低到0.3，极小化目标方向的权重
                [dist_to_goal / 100.0]  # 归一化距离
            ])
        else:
            # 如果没有目标位置，提供零向量作为目标信息
            goal_info = np.zeros(4)  # 3维方向 + 1维距离
        
        # 基本观察信息（自身位置和速度）- 6维
        base_obs = np.concatenate([
            agent.state.p_pos,                      # 自身位置
            agent.state.p_vel,                      # 自身速度
        ])
        
        # 添加地形高度观察信息，增加更多周围地形信息
        terrain_info = []
        try:
            if hasattr(self, 'get_terrain_height'):
                # 获取当前位置的地形高度
                current_x, current_y = agent.state.p_pos[0], agent.state.p_pos[1]
                current_height = self.get_terrain_height(current_x, current_y)
                
                # 获取前方位置的地形高度（沿移动方向）
                vel_direction = agent.state.p_vel
                vel_norm = np.linalg.norm(vel_direction)
                if vel_norm > 1e-6:
                    vel_direction = vel_direction / vel_norm
                else:
                    vel_direction = np.array([1.0, 0.0, 0.0])  # 默认朝X轴正方向
                
                # 采样多个前方点的高度
                terrain_heights = []
                distances = [3, 6, 9, 12, 15]  # 增加采样点数量
                for dist in distances:
                    future_x = current_x + vel_direction[0] * dist
                    future_y = current_y + vel_direction[1] * dist
                    if 0 <= future_x < self.map_size and 0 <= future_y < self.map_size:
                        future_height = self.get_terrain_height(future_x, future_y)
                        terrain_heights.append((future_height - current_height) / 20.0)  # 归一化高度差
                    else:
                        terrain_heights.append(0.0)  # 超出地图范围，假设平坦
                
                # 增加周围点的地形信息
                surround_heights = []
                directions = [
                    (1, 0), (1, 1), (0, 1), (-1, 1),
                    (-1, 0), (-1, -1), (0, -1), (1, -1)
                ]
                
                for dx, dy in directions:
                    nx, ny = current_x + dx * 10, current_y + dy * 10  # 从5单位改为10单位
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        height = self.get_terrain_height(nx, ny)
                        surround_heights.append((height - current_height) / 20.0)
                    else:
                        surround_heights.append(0.0)
                
                # 添加地形高度信息（占更重要的位置）
                terrain_info = np.array([current_height / 100.0] + terrain_heights + surround_heights)
            else:
                # 如果没有地形高度函数，提供默认值
                terrain_info = np.zeros(1 + 5 + 8)  # 当前高度 + 前方5点 + 周围8点
        except Exception as e:
            print(f"地形观察计算错误: {e}")
            # 出现错误时提供默认值
            terrain_info = np.zeros(14)  # 当前高度 + 前方5点 + 周围8点
        
        # 添加障碍物信息
        obstacle_info = []
        nearest_obstacles = []
        
        try:
            # 查找最近的3个障碍物
            obstacle_distances = []
            for i, obstacle in enumerate(self.obstacles):
                obstacle_center = np.array(obstacle['center'])
                dist = np.linalg.norm(agent.state.p_pos - obstacle_center)
                obstacle_distances.append((i, dist))
            
            # 按距离排序并选取最近的3个
            obstacle_distances.sort(key=lambda x: x[1])
            for i in range(min(3, len(obstacle_distances))):
                idx, dist = obstacle_distances[i]
                obstacle = self.obstacles[idx]
                obstacle_center = np.array(obstacle['center'])
                
                # 相对位置向量
                rel_pos = obstacle_center - agent.state.p_pos
                
                # 归一化相对位置和距离
                if dist > 1e-6:
                    norm_dir = rel_pos / dist
                else:
                    norm_dir = np.zeros(3)
                    
                nearest_obstacles.extend([norm_dir[0], norm_dir[1], norm_dir[2], dist / 100.0, obstacle['radius'] / 20.0])
            
            # 如果障碍物不足3个，用零填充
            while len(nearest_obstacles) < 3 * 5:  # 3个障碍物，每个5个值
                nearest_obstacles.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            obstacle_info = np.array(nearest_obstacles)
        except Exception as e:
            print(f"障碍物观察计算错误: {e}")
            # 出现错误时提供默认值
            obstacle_info = np.zeros(15)  # 3个障碍物，每个5个值
        
        # 添加其他智能体信息
        other_agents_info = []
        try:
            for other in world.agents:
                if other is not agent:
                    other_agents_info.append(other.state.p_pos - agent.state.p_pos)  # 相对位置
                    other_agents_info.append(other.state.p_vel)                      # 速度
            
            # 转换为数组，如果有其他智能体信息
            if other_agents_info:
                other_agents_obs = np.concatenate(other_agents_info)
            else:
                other_agents_obs = np.array([])
        except Exception as e:
            print(f"其他智能体观察计算错误: {e}")
            # 出现错误时提供默认值
            other_agents_obs = np.array([])
        
        # 合并所有观察信息，优先级：地形信息 > 障碍物信息 > 基本信息 > 目标信息 > 其他智能体信息
        # 注意目标信息现在放到后面，减少其重要性
        obs_components = []
        
        if len(terrain_info) > 0:
            obs_components.append(terrain_info)
            
        if len(obstacle_info) > 0:
            obs_components.append(obstacle_info)
            
        obs_components.append(base_obs)
        
        if len(goal_info) > 0:
            obs_components.append(goal_info)
            
        if len(other_agents_obs) > 0:
            obs_components.append(other_agents_obs)
            
        # 尝试合并所有观察分量
        try:
            obs = np.concatenate(obs_components)
        except Exception as e:
            print(f"观察值合并错误: {e}")
            print(f"观察分量大小: 地形={len(terrain_info)}, 障碍物={len(obstacle_info)}, 基本={len(base_obs)}, 目标={len(goal_info)}, 其他智能体={len(other_agents_obs)}")
            # 在合并出错时提供基本观察值
            obs = np.zeros(36)
            # 确保至少基本观察信息不为空
            if len(base_obs) > 0:
                obs[:len(base_obs)] = base_obs
        
        # 确保观察空间维度一致 - 填充到36维
        if len(obs) < 36:
            obs = np.pad(obs, (0, 36 - len(obs)), 'constant')
        elif len(obs) > 36:
            obs = obs[:36]  # 截断到36维
        
        # 确保返回的是numpy数组而不是列表或嵌套结构
        if not isinstance(obs, np.ndarray):
            try:
                obs = np.array(obs)
            except:
                # 最后的后备方案，创建一个零向量
                obs = np.zeros(36)
        
        # 验证形状正确
        if len(obs.shape) > 1:
            print(f"警告: 观察值形状不正确: {obs.shape}，尝试修复")
            # 尝试展平或重塑
            try:
                obs = obs.flatten()[:36]  # 取前36个元素
                if len(obs) < 36:  # 如果还不够，填充
                    obs = np.pad(obs, (0, 36 - len(obs)), 'constant')
            except:
                # 最终后备方案
                obs = np.zeros(36)
        
        # 最后的检查确保返回正确形状和类型的数组
        if not isinstance(obs, np.ndarray) or obs.shape != (36,):
            print(f"严重警告: 观察值仍有问题: 类型={type(obs)}, 形状={getattr(obs, 'shape', 'unknown')}")
            obs = np.zeros(36)
        
        # 最终的安全检查
        try:
            if obs is None or len(obs) != 36:
                print(f"最终安全检查失败: 观察值长度 = {len(obs) if obs is not None else 'None'}")
                obs = np.zeros(36)
        except Exception as e:
            print(f"最终观察值验证出错: {e}")
            obs = np.zeros(36)
            
        return obs

    def find_peak_positions(self, min_height=50, neighborhood_size=10, max_peaks=5):
        """寻找地形中的山顶位置（局部最高点）"""
        # print("\n=== 寻找地形山顶 ===")
        
        # 确保地形已经生成
        if self.terrain is None:
            # 不要自动生成地形，而是返回默认值
            # print("警告: 寻找山顶时地形数据为空，返回默认山顶。这不应该发生于正常训练过程中。")
            # 返回地图中心作为默认山顶
            default_position = [self.map_size // 2, self.map_size // 2, 50]
            default_peak = {
                'position': default_position,
                'height': 50,
                'prominence': 20
            }
            return [default_peak]
            
        # 获取地形的基本统计数据
        terrain_mean = np.mean(self.terrain)
        terrain_std = np.std(self.terrain)
        terrain_max = np.max(self.terrain)
        terrain_min = np.min(self.terrain)
        
        # print(f"地形统计: 平均高度={terrain_mean:.2f}, 标准差={terrain_std:.2f}")
        # print(f"地形范围: 最低点={terrain_min:.2f}, 最高点={terrain_max:.2f}")
        
        # 设置高度阈值 - 只考虑比平均高度高出标准差的点
        height_threshold = terrain_mean + terrain_std
        # print(f"高度阈值: {height_threshold:.2f}")
        
        # 山顶候选点 - 简单地选择最高的若干个点
        highest_points = []
        margin = 5  # 避免边缘
        
        # 第一步：找到所有高于阈值的点
        for x in range(margin, self.map_size - margin):
            for y in range(margin, self.map_size - margin):
                height = self.terrain[y, x]
                if height > height_threshold:
                    highest_points.append((x, y, height))
        
        # 按高度降序排序
        highest_points.sort(key=lambda p: p[2], reverse=True)
        
        # print(f"找到 {len(highest_points)} 个高于阈值的点")
        if len(highest_points) == 0:
            # print("警告: 没有找到任何高点，使用地形最高点")
            # 找到地形最高点
            max_height_idx = np.unravel_index(np.argmax(self.terrain), self.terrain.shape)
            y, x = max_height_idx
            if margin <= x < self.map_size - margin and margin <= y < self.map_size - margin:
                highest_points = [(x, y, self.terrain[y, x])]
            else:
                # 如果最高点在边缘，选择中心点
                x, y = self.map_size // 2, self.map_size // 2
                highest_points = [(x, y, self.terrain[y, x])]
        
        # 第二步：验证这些点是否确实是局部最高点
        confirmed_peaks = []
        small_window = 2  # 小窗口用于局部最高点检测
        
        for x, y, height in highest_points[:min(30, len(highest_points))]:  # 只检查前30个最高点
            is_peak = True
            
            # 检查小窗口内是否为局部最高点
            for dx in range(-small_window, small_window + 1):
                for dy in range(-small_window, small_window + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if self.terrain[ny, nx] > height:
                            is_peak = False
                            break
                
                if not is_peak:
                    break
            
            if is_peak:
                # 确认是否存在足够的高度落差(至少1个标准差)
                large_window = 10
                surroundings = []
                
                for dx in range(-large_window, large_window + 1):
                    for dy in range(-large_window, large_window + 1):
                        if abs(dx) > small_window or abs(dy) > small_window:  # 只考虑大窗口中排除小窗口的区域
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                                surroundings.append(self.terrain[ny, nx])
                
                if surroundings:
                    surrounding_mean = np.mean(surroundings)
                    prominence = height - surrounding_mean
                    
                    if prominence > terrain_std:
                        # 只有当点高出周围区域至少一个标准差时，才认为它是真正的山顶
                        confirmed_peaks.append({
                            'position': [x, y, height],
                            'height': height,
                            'prominence': prominence
                        })
                        # print(f"确认山顶: 位置=({x}, {y}), 高度={height:.2f}, 突出度={prominence:.2f}")
                        
                        if len(confirmed_peaks) >= max_peaks:
                            break
        
        # 如果没有找到符合条件的山顶，直接使用最高点
        if not confirmed_peaks and highest_points:
            x, y, height = highest_points[0]
            # 确保地形最高点高度高于平均值
            if height > terrain_mean:
                # print(f"使用最高点作为山顶: ({x}, {y}), 高度={height:.2f}")
                confirmed_peaks.append({
                    'position': [x, y, height],
                    'height': height,
                    'prominence': height - terrain_mean
                })
        
        # 如果仍然没有找到任何山顶，使用地形中心点
        if not confirmed_peaks:
            x, y = self.map_size // 2, self.map_size // 2
            height = self.terrain[y, x]
            # print(f"未找到任何山顶，使用中心点: ({x}, {y}), 高度={height:.2f}")
            confirmed_peaks.append({
                'position': [x, y, height],
                'height': height,
                'prominence': 0
            })
        
        # 检查第一个山顶的高度，如果低于平均高度，增加Z坐标，让目标漂浮在高处
        if confirmed_peaks and confirmed_peaks[0]['height'] < terrain_mean + terrain_std:
            confirmed_peaks[0]['position'][2] += 10
            new_height = confirmed_peaks[0]['position'][2]
            # print(f"提升目标高度至: {new_height:.2f} (原高度过低)")
        
        # 打印最终选择的山顶
        if confirmed_peaks:
            top_peak = confirmed_peaks[0]
            # print(f"\n最终选择的山顶: 位置=({top_peak['position'][0]}, {top_peak['position'][1]}), "
            #       f"高度={top_peak['position'][2]:.2f}, 突出度={top_peak['prominence']:.2f}")
        
        # print("=== 山顶搜索完成 ===\n")
        return confirmed_peaks

    def debug_coordinates(self):
        """返回坐标系相关调试信息"""
        debug_info = {
            'goal_pos': self.goal_pos.copy() if self.goal_pos is not None else None,
            'goal_pos_id': id(self.goal_pos) if self.goal_pos is not None else None,
            'terrain_shape': self.terrain.shape if self.terrain is not None else None,
            'map_size': self.map_size,
            'coordinate_system': '3D坐标系，X/Y为平面坐标，Z为高度'
        }
        return debug_info

    def dump_coordinate_info(self, world):
        """输出坐标信息用于调试"""
        with open("debug_coordinates.txt", "w") as f:
            f.write("===== 坐标系统信息 =====\n")
            f.write(f"目标位置: {self.goal_pos}\n")
            
            for i, agent in enumerate(world.agents):
                pos = agent.state.p_pos
                goal_rel_pos = self.goal_pos - pos
                dist = np.linalg.norm(goal_rel_pos)
                direction = goal_rel_pos / (dist + 1e-6)
                
                f.write(f"\n智能体 {i}:\n")
                f.write(f"  位置: {pos}\n")
                f.write(f"  到目标向量: {goal_rel_pos}\n") 
                f.write(f"  到目标距离: {dist}\n")
                f.write(f"  到目标方向: {direction}\n")

    def validate_goal_coordinates(self, world):
        """
        验证目标坐标和方向计算是否正确
        此函数打印目标位置、智能体位置、方向向量等关键信息
        """
        # print("\n==== 目标坐标验证 ====")
        if not hasattr(self, 'goal_pos') or self.goal_pos is None:
            # print("警告: 目标位置未设置!")
            return
        
        # print(f"目标位置: {self.goal_pos}")
        
        # 定义坐标轴
        axes = [
            np.array([1, 0, 0]),  # X轴
            np.array([0, 1, 0]),  # Y轴
            np.array([0, 0, 1])   # Z轴
        ]
        axis_names = ["X轴", "Y轴", "Z轴"]
        
        # 检查每个智能体的位置和相对目标的方向
        for i, agent in enumerate(world.agents):
            pos = agent.state.p_pos
            # 计算到目标的向量和距离
            to_goal_vec = self.goal_pos - pos
            dist = np.linalg.norm(to_goal_vec)
            
            # 归一化方向向量
            if dist > 1e-6:
                direction = to_goal_vec / dist
            else:
                direction = np.zeros(3)
            
            # print(f"智能体 {i}:")
            # print(f"  位置: {pos}")
            # print(f"  到目标向量: {to_goal_vec}")
            # print(f"  距离: {dist}")
            # print(f"  方向: {direction}")
            
            # 检查坐标轴方向一致性
            # print("  坐标轴方向一致性:")
            for ax, name in zip(axes, axis_names):
                alignment = np.dot(direction, ax)
                # print(f"    与{name}一致性: {alignment:.4f} ({alignment*100:.1f}%)")
            
            # 测试轻微移动效果
            # print("  模拟沿方向移动:")
            move_dist = 5.0  # 移动5个单位
            new_pos = pos + direction * move_dist
            new_dist = np.linalg.norm(self.goal_pos - new_pos)
            # print(f"    移动后位置: {new_pos}")
            # print(f"    新距离: {new_dist:.2f} (减少: {dist - new_dist:.2f})")
        
        # print("========================\n")
        
        # 将信息写入文件以便后续分析
        # with open("goal_direction_debug.txt", "w") as f:
        #     f.write(f"目标位置: {self.goal_pos}\n\n")
        #     for i, agent in enumerate(world.agents):
        #         pos = agent.state.p_pos
        #         to_goal_vec = self.goal_pos - pos
        #         dist = np.linalg.norm(to_goal_vec)
        #         direction = to_goal_vec / dist if dist > 1e-6 else np.zeros(3)
        #         
        #         f.write(f"智能体 {i}:\n")
        #         f.write(f"  位置: {pos}\n")
        #         f.write(f"  到目标向量: {to_goal_vec}\n")
        #         f.write(f"  归一化方向: {direction}\n")
        #         
        #         # 测试与各轴方向的一致性
        #         for ax, name in zip(axes, axis_names):
        #             alignment = np.dot(direction, ax)
        #             f.write(f"  与{name}一致性: {alignment:.4f}\n")
        #         
        #         f.write("\n")
        
        return

    def update_visited_cells(self, world):
        """更新智能体访问过的网格单元"""
        for i, agent in enumerate(world.agents):
        try:
            # 将当前位置转换为网格单元
            cell_size = getattr(self, 'cell_size', 5.0)  # 默认单元格大小为5
            cell_x = int(agent.state.p_pos[0] / cell_size)
            cell_y = int(agent.state.p_pos[1] / cell_size)
            current_cell = (cell_x, cell_y)
            
            # 添加到已访问集合
            if i in self.visited_cells:
                self.visited_cells[i].add(current_cell)
            else:
                self.visited_cells[i] = {current_cell}
        # 处理异常情况
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
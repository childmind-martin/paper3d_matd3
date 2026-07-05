# 创建multiagent/scenarios/paper3d_terrain.py文件
import hashlib
import json
import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy import signal
import time
import matplotlib.pyplot as plt
import os
import traceback  # 添加traceback模块用于错误跟踪


def _scenario_quiet_output():
    try:
        if os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1', 'true', 'yes', 'on'):
            return True
        return os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        return True


class Scenario(BaseScenario):
    """
    自定义3D地形场景
    特性：
    1. 可生成随机地形
    2. 支持3D移动
    3. 目标设置在山顶或任意高点
    """
    def __init__(self, seed=None, use_fixed_positions=False, fixed_positions=None, dynamic_first_time=False, 
                 fixed_positions_file=None, random_terrain=False, random_z0_positions=False, 
                 terrain_complexity_level=1, **kwargs):
        """初始化3D地形场景
        
        参数:
            seed (int): 随机种子，控制地形生成，默认为None（使用随机种子）
            use_fixed_positions (bool): 是否使用固定位置，True表示使用固定起始位置，False表示每次随机
            fixed_positions (dict): 固定位置数据，格式为{'agents': [...], 'goal': [...]}
            dynamic_first_time (bool): 首次运行是否动态生成位置，后续固定
            fixed_positions_file (str): 固定位置文件路径，如果提供且文件存在则从中加载位置
            random_terrain (bool): 是否随机生成地形，True表示每次重置时重新生成地形
            random_z0_positions (bool): 是否随机生成初始高度，即使使用固定XY坐标也会随机化Z
            terrain_complexity_level (int): 地形复杂度等级，1-4，控制山峰数量、障碍物数量等
            **kwargs: 额外的参数，会被忽略
        """
        super().__init__()
        # 缓存常用环境参数，避免在reward中频繁os.getenv导致瓶颈
        try:
            import os as _os
            self.clearance_d_max = float(_os.getenv('CLEARANCE_D_MAX', '50.0'))
            self.clearance_weight = float(_os.getenv('CLEARANCE_WEIGHT', '0.4'))
            self.reward_clip_min = float(_os.getenv('REWARD_CLIP_MIN', '-500.0'))
            self.reward_clip_max = float(_os.getenv('REWARD_CLIP_MAX', '100.0'))
        except Exception:
            self.clearance_d_max = 50.0
            self.clearance_weight = 0.4
            self.reward_clip_min = -500.0
            self.reward_clip_max = 100.0
        
        # 储存地形数据
        map_size_kw = kwargs.get('map_size', None)
        if map_size_kw is None:
            try:
                map_size_env = os.getenv('MAP_SIZE')
                if map_size_env is not None:
                    map_size_kw = float(map_size_env)
            except Exception:
                map_size_kw = None
        if map_size_kw is None:
            map_size_kw = 200.0
        self.map_size = float(map_size_kw)  # 地图大小（从100扩大到200，可配置）
        self.terrain = None  # 地形高度图，生成后填充
        
        # 储存目标信息
        self.goal_pos = None  # 目标位置，生成后填充
        
        # 记录复杂度
        self.terrain_complexity = {}
        
        # 随机数生成器种子
        self.seed = seed
        self.rng = np.random.RandomState(seed)  # 创建独立的随机数生成器
        
        # 添加探索奖励所需的属性
        self.visited_cells = {}  # 各智能体已访问区域初始化为空字典
        self.cell_size = 5  # 探索区域的网格大小（从3减小到2，使网格更细，提高探索粒度）
        self.exploration_reward_scale = 1.2  # 探索新区域的奖励比例（从1.0提高到2.0）
        
        # 地形参数
        self.mountain_x_range = [40, 60]  # 山脉位置X范围
        self.mountain_y_range = [40, 60]  # 山脉位置Y范围
        self.mountain_height_range = [40, 80]  # 山高度范围
        self.mountain_radius_range = [15, 30]  # 山半径范围
        self.flatten_edges = True  # 是否使边缘平坦
        self.edge_size = 5  # 边缘平坦区域大小
        self.terrain_seed = seed  # 地形生成种子
        self.X = None  # X坐标网格
        self.Y = None  # Y坐标网格
        
        # 地形复杂度控制
        self.terrain_complexity_level = terrain_complexity_level if terrain_complexity_level is not None else 2
        
        # 根据复杂度等级设置参数
        self._setup_complexity_parameters()
        
        # 障碍物设置（由复杂度等级控制）
        self.obstacle_size_range = [5, 15]  # 障碍物尺寸范围
        self.obstacle_height_boost = 10  # 障碍物高度提升
        try:
            agent_size_kw = kwargs.get('agent_size', None)
            if agent_size_kw is None:
                agent_size_kw = os.getenv('AGENT_SIZE', '0.5')
            self.agent_size = float(agent_size_kw)
        except Exception:
            self.agent_size = 0.5

        # 观察构建的静态模板与缓存
        self._obs_centers = np.zeros((0, 3), dtype=np.float32)
        self._obs_radii = np.zeros((0,), dtype=np.float32)
        self._obs_forward_distances = np.asarray([2, 4, 6, 10, 15, 20, 25, 30], dtype=np.float32)
        self._obs_direction_pairs = np.asarray(
            [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
            dtype=np.float32,
        )
        self._obs_gradient_offsets = np.asarray(
            [[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
            dtype=np.float32,
        )
        self._obs_near_distance = 5.0
        self._obs_far_distance = 12.0
        self._obs_env_info = None
        self._obs_env_info_agent_count = None
        self._obs_step_cache_key = None
        self._obs_step_cache = {}
        self.observation_dim = 81
        
        # 🔧 障碍物生成模式：从环境变量读取
        use_dynamic_obstacles_kw = kwargs.get('use_dynamic_obstacles', None)
        try:
            if use_dynamic_obstacles_kw is None:
                use_dynamic_obstacles = os.getenv('USE_DYNAMIC_OBSTACLES', '1').lower() in ('1', 'true', 'yes', 'on')
            else:
                use_dynamic_obstacles = bool(use_dynamic_obstacles_kw)
        except Exception:
            use_dynamic_obstacles = True  # 默认启用动态障碍物
        self.use_dynamic_obstacles = use_dynamic_obstacles  # 是否每次reset时重新生成障碍物
        
        # 位置设置模式
        self.use_fixed_positions = use_fixed_positions  # 是否使用固定位置
        self._initial_use_fixed_positions = use_fixed_positions  # 🚨 保存初始值，用于dynamic_first_time逻辑判断
        self.dynamic_first_time = dynamic_first_time    # 是否首次动态后续固定
        self.positions_initialized = False              # 标记是否已经初始化过位置
        self.random_z0_positions = random_z0_positions  # 是否随机化高度位置
        self.random_terrain = random_terrain            # 是否随机生成地形
        semi_random_kw = kwargs.get('semi_random_terrain', None)
        try:
            if semi_random_kw is None:
                self.use_semi_random_terrain = os.getenv('SEMI_RANDOM_TERRAIN', '0').lower() in ('1', 'true', 'yes', 'on')
            else:
                self.use_semi_random_terrain = bool(semi_random_kw)
        except Exception:
            self.use_semi_random_terrain = False
        try:
            default_base_seed = self.seed if self.seed is not None else 67
            terrain_base_seed_kw = kwargs.get('terrain_base_seed', None)
            if terrain_base_seed_kw is None:
                self.terrain_base_seed = int(os.getenv('TERRAIN_BASE_SEED', str(default_base_seed)))
            else:
                self.terrain_base_seed = int(terrain_base_seed_kw)
        except Exception:
            self.terrain_base_seed = int(self.seed if self.seed is not None else 67)
        try:
            peak_jitter_kw = kwargs.get('peak_jitter_range', None)
            self.peak_jitter_range = float(os.getenv('PEAK_JITTER_RANGE', '15.0') if peak_jitter_kw is None else peak_jitter_kw)
        except Exception:
            self.peak_jitter_range = 15.0
        try:
            peak_center_jitter_kw = kwargs.get('peak_center_jitter_range', None)
            self.peak_center_jitter_range = float(
                os.getenv('PEAK_CENTER_JITTER_RANGE', str(min(float(self.peak_jitter_range), 3.0)))
                if peak_center_jitter_kw is None else peak_center_jitter_kw
            )
        except Exception:
            self.peak_center_jitter_range = float(min(float(self.peak_jitter_range), 3.0))
        try:
            peak_height_jitter_ratio_min_kw = kwargs.get('peak_height_jitter_ratio_min', None)
            self.peak_height_jitter_ratio_min = float(
                os.getenv('PEAK_HEIGHT_JITTER_RATIO_MIN', '0.20')
                if peak_height_jitter_ratio_min_kw is None else peak_height_jitter_ratio_min_kw
            )
        except Exception:
            self.peak_height_jitter_ratio_min = 0.20
        try:
            peak_height_jitter_ratio_max_kw = kwargs.get('peak_height_jitter_ratio_max', None)
            self.peak_height_jitter_ratio_max = float(
                os.getenv('PEAK_HEIGHT_JITTER_RATIO_MAX', '0.40')
                if peak_height_jitter_ratio_max_kw is None else peak_height_jitter_ratio_max_kw
            )
        except Exception:
            self.peak_height_jitter_ratio_max = 0.40
        try:
            peak_height_max_scale_kw = kwargs.get('peak_height_max_scale', None)
            self.peak_height_max_scale = float(
                os.getenv('PEAK_HEIGHT_MAX_SCALE', '1.30')
                if peak_height_max_scale_kw is None else peak_height_max_scale_kw
            )
        except Exception:
            self.peak_height_max_scale = 1.30
        self.peak_height_jitter_ratio_min = max(0.0, float(self.peak_height_jitter_ratio_min))
        self.peak_height_jitter_ratio_max = max(
            float(self.peak_height_jitter_ratio_min),
            float(self.peak_height_jitter_ratio_max),
        )
        self.peak_height_max_scale = max(1.0, float(self.peak_height_max_scale))
        try:
            terrain_variant_noise_ratio_kw = kwargs.get('terrain_variant_noise_ratio', None)
            self.terrain_variant_noise_ratio = float(
                os.getenv('TERRAIN_VARIANT_NOISE_RATIO', '0.15')
                if terrain_variant_noise_ratio_kw is None else terrain_variant_noise_ratio_kw
            )
        except Exception:
            self.terrain_variant_noise_ratio = 0.15
        self.peak_center_jitter_range = max(0.0, float(self.peak_center_jitter_range))
        self.terrain_variant_noise_ratio = max(0.0, float(self.terrain_variant_noise_ratio))
        hold_mode_kw = kwargs.get('semi_random_hold_mode', None)
        hold_mode_raw = str(
            os.getenv('SEMI_RANDOM_TERRAIN_HOLD_MODE', 'episode')
            if hold_mode_kw is None else hold_mode_kw
        ).strip().lower()
        if hold_mode_raw not in ('episode', 'fixed', 'range'):
            hold_mode_raw = 'episode'
        self.semi_random_hold_mode = hold_mode_raw
        try:
            hold_episodes_kw = kwargs.get('semi_random_hold_episodes', None)
            self.semi_random_hold_episodes = max(
                1,
                int(os.getenv('SEMI_RANDOM_TERRAIN_HOLD_EPISODES', '1') if hold_episodes_kw is None else hold_episodes_kw),
            )
        except Exception:
            self.semi_random_hold_episodes = 1
        try:
            hold_min_episodes_kw = kwargs.get('semi_random_hold_min_episodes', None)
            self.semi_random_hold_min_episodes = max(
                1,
                int(
                    os.getenv('SEMI_RANDOM_TERRAIN_HOLD_MIN_EPISODES', str(self.semi_random_hold_episodes))
                    if hold_min_episodes_kw is None else hold_min_episodes_kw
                ),
            )
        except Exception:
            self.semi_random_hold_min_episodes = max(1, int(self.semi_random_hold_episodes))
        try:
            hold_max_episodes_kw = kwargs.get('semi_random_hold_max_episodes', None)
            self.semi_random_hold_max_episodes = max(
                self.semi_random_hold_min_episodes,
                int(
                    os.getenv('SEMI_RANDOM_TERRAIN_HOLD_MAX_EPISODES', str(self.semi_random_hold_min_episodes))
                    if hold_max_episodes_kw is None else hold_max_episodes_kw
                ),
            )
        except Exception:
            self.semi_random_hold_max_episodes = max(1, int(self.semi_random_hold_min_episodes))
        try:
            self.terrain_variant_seed = int(
                os.getenv(
                    'TERRAIN_VARIANT_SEED',
                    str(self.seed if self.seed is not None else self.terrain_base_seed),
                )
            )
        except Exception:
            self.terrain_variant_seed = int(self.seed if self.seed is not None else self.terrain_base_seed)
        try:
            self.curriculum_multi_terrain_enabled = os.getenv('CURRICULUM_MULTI_TERRAIN_MODE', '0').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            self.curriculum_multi_terrain_enabled = False
        try:
            seeds_raw = os.getenv('CURRICULUM_MULTI_TERRAIN_SEEDS', '').strip()
            self.curriculum_multi_terrain_seeds = [
                int(tok.strip()) for tok in seeds_raw.split(',') if tok.strip()
            ]
        except Exception:
            self.curriculum_multi_terrain_seeds = []
        try:
            deterministic_train_env_sequence_kw = kwargs.get('deterministic_train_env_sequence', None)
            if deterministic_train_env_sequence_kw is None:
                self.deterministic_train_env_sequence = os.getenv('DETERMINISTIC_TRAIN_ENV_SEQUENCE', '0').lower() in ('1', 'true', 'yes', 'on')
            else:
                self.deterministic_train_env_sequence = bool(deterministic_train_env_sequence_kw)
        except Exception:
            self.deterministic_train_env_sequence = False
        try:
            training_env_sequence_seed_kw = kwargs.get('training_env_sequence_seed', None)
            self.training_env_sequence_seed = int(
                os.getenv(
                    'TRAIN_ENV_SEQUENCE_SEED',
                    str(self.terrain_base_seed),
                ) if training_env_sequence_seed_kw is None else training_env_sequence_seed_kw
            )
        except Exception:
            self.training_env_sequence_seed = int(self.terrain_base_seed)
        try:
            train_obstacle_sequence_mode_kw = kwargs.get('train_obstacle_sequence_mode', None)
            self.train_obstacle_sequence_mode = str(
                os.getenv('TRAIN_OBSTACLE_SEQUENCE_MODE', 'legacy_linear')
                if train_obstacle_sequence_mode_kw is None else train_obstacle_sequence_mode_kw
            ).strip().lower()
        except Exception:
            self.train_obstacle_sequence_mode = 'legacy_linear'
        if self.train_obstacle_sequence_mode not in ('legacy_linear', 'post_eval_family'):
            self.train_obstacle_sequence_mode = 'legacy_linear'
        try:
            train_obstacle_sequence_namespace_kw = kwargs.get('train_obstacle_sequence_namespace', None)
            self.train_obstacle_sequence_namespace = str(
                os.getenv('TRAIN_OBSTACLE_SEQUENCE_NAMESPACE', 'train_obstacle')
                if train_obstacle_sequence_namespace_kw is None else train_obstacle_sequence_namespace_kw
            ).strip()
        except Exception:
            self.train_obstacle_sequence_namespace = 'train_obstacle'
        if not self.train_obstacle_sequence_namespace:
            self.train_obstacle_sequence_namespace = 'train_obstacle'
        self.current_episode_index = 0
        self.current_episode_env_id = 0
        self.current_episode_rng_seed = None
        self.current_episode_obstacle_seed = None
        self.current_episode_obstacle_seed_override = None
        self._train_obstacle_sequence_cache = {}
        
        # 🔧 关键修复：跟踪当前地形种子，用于检测地形变化
        # 当地形种子变化时，重置位置初始化标记，使每个新地图都动态生成位置
        self.current_terrain_seed = self.seed  # 记录当前地形种子
        self.current_terrain_variant_seed = self.terrain_variant_seed
        self.current_terrain_hold_block_index = 0
        self.current_terrain_hold_block_start_episode = 0
        self.current_terrain_hold_block_length = 1
        
        # 固定位置文件
        self.fixed_positions_file = fixed_positions_file
        self.fixed_positions = None
        # 缓存固定位置验证签名，避免每回合对同一套位置/地形重复校验
        self._fixed_positions_validation_signature = None
        # 障碍物布局签名：仅当起点/目标/地形/seed 对应的布局一致时才复用固定障碍
        self._obstacle_layout_signature = None

        # 🚨 关键修复：DYNAMIC_FIRST_TIME 模式下，在初始化时就删除旧的位置文件
        # 这样可以确保每次新运行都会重新生成初始位置
        if dynamic_first_time and self.fixed_positions_file:
            if os.path.exists(self.fixed_positions_file):
                try:
                    os.remove(self.fixed_positions_file)
                    print(f"[DYNAMIC_FIRST_TIME] 初始化时删除旧的位置文件: {self.fixed_positions_file}")
                except Exception as e:
                    print(f"[DYNAMIC_FIRST_TIME] 删除位置文件失败: {e}")

        # 🚨 关键修复：只有在use_fixed_positions=True时才加载固定位置
        # 如果通过参数直接提供了固定位置，优先采用
        if fixed_positions is not None:
            self.fixed_positions = fixed_positions
            self.validate_and_adjust_fixed_positions()
            self.use_fixed_positions = True
            self.positions_initialized = True
        # 如果提供了固定位置文件且use_fixed_positions=True，尝试加载
        elif self.fixed_positions_file and use_fixed_positions:
            try:
                if self.load_fixed_positions(self.fixed_positions_file):
                    self.use_fixed_positions = True
                    self.positions_initialized = True
            except Exception as _load_err:
                print(f"[固定位置] 加载文件失败({self.fixed_positions_file}): {_load_err}")
        # 🚨 修复：如果use_fixed_positions=False，即使有fixed_positions_file也不加载
        elif self.fixed_positions_file and not use_fixed_positions:
            print(f"[固定位置] 已禁用固定位置，忽略文件: {self.fixed_positions_file}")

    def _refresh_observation_static_cache(self, num_agents=None):
        """刷新 observation 需要的静态障碍物/环境常量缓存。"""
        try:
            if self.obstacles:
                self._obs_centers = np.asarray([ob['center'] for ob in self.obstacles], dtype=np.float32)
                self._obs_radii = np.asarray([ob['radius'] for ob in self.obstacles], dtype=np.float32)
            else:
                self._obs_centers = np.zeros((0, 3), dtype=np.float32)
                self._obs_radii = np.zeros((0,), dtype=np.float32)
        except Exception:
            self._obs_centers = np.zeros((0, 3), dtype=np.float32)
            self._obs_radii = np.zeros((0,), dtype=np.float32)

        if num_agents is not None:
            num_agents = int(num_agents)
            if self._obs_env_info is None or self._obs_env_info_agent_count != num_agents:
                self._obs_env_info = np.array(
                    [num_agents / 10.0, -0.8, -0.4, 0.0, 0.4, 0.8],
                    dtype=np.float32,
                )
                self._obs_env_info_agent_count = num_agents
        
    def _get_start_altitude_offset(self):
        """统一获取起始离地高度配置"""
        try:
            import os
            value = float(os.getenv('START_ALTITUDE_OFFSET', '7.0'))
            # 🔧 临时调试：输出读取到的值
            suppress_output = os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1', 'true', 'yes', 'on')
            if not suppress_output and not hasattr(self, '_altitude_offset_logged'):
                print(f"[调试] _get_start_altitude_offset() 返回: {value}")
                self._altitude_offset_logged = True
            return value
        except Exception as e:
            print(f"[错误] _get_start_altitude_offset() 异常: {e}，使用默认值7.0")
            return 7.0
    
    def _get_goal_altitude(self):
        """统一获取目标离地高度配置"""
        try:
            import os
            return float(os.getenv('GOAL_ALTITUDE', '12.0'))
        except Exception:
            return 12.0

    def _get_fixed_goal_metadata(self):
        """提取固定位置文件中的目标元数据。"""
        fixed_positions = getattr(self, 'fixed_positions', None)
        if not isinstance(fixed_positions, dict):
            return {}
        position_setup = fixed_positions.get('position_setup')
        if not isinstance(position_setup, dict):
            position_setup = fixed_positions.get('heldout_metadata')
        return position_setup if isinstance(position_setup, dict) else {}

    def _estimate_goal_support_height(self, goal_xy, radius=6.0, grid_points=7):
        """估计目标点周边平台可支撑的最高地形高度。"""
        try:
            goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        except Exception:
            return 0.0
        if goal_xy.size < 2:
            return 0.0

        x = float(goal_xy[0])
        y = float(goal_xy[1])
        radius = max(1.0, float(radius))
        sample_count = max(3, int(grid_points))
        offsets = np.linspace(-radius, radius, sample_count)
        max_height = float(self.get_terrain_height(x, y))
        for dx in offsets:
            for dy in offsets:
                try:
                    h = float(self.get_terrain_height(x + dx, y + dy))
                except Exception:
                    continue
                if h > max_height:
                    max_height = h
        return float(max_height)

    def _adjust_fixed_goal_position_for_current_terrain(self, goal_pos, quiet_output=False):
        """
        固定位置模式下保持目标 XY 不变，但确保目标 Z 不低于当前地形/平台。

        旧逻辑完全保留固定位置文件中的绝对 Z，这在固定地形下没有问题；
        但在随机/半随机地形下，同一 XY 处的局部地形会变化，目标可能落到坡面里。
        这里采用“只抬不降”的策略：仅当当前地形/平台更高时，把目标抬到
        support_height + required_clearance。
        """
        goal_pos = np.asarray(goal_pos, dtype=float).copy()
        if goal_pos.size < 3:
            return goal_pos

        goal_pos[0] = np.clip(goal_pos[0], 2, self.map_size - 3)
        goal_pos[1] = np.clip(goal_pos[1], 2, self.map_size - 3)

        metadata = self._get_fixed_goal_metadata()
        try:
            platform_radius = float(metadata.get('goal_platform_radius', 6.0))
        except Exception:
            platform_radius = 6.0
        platform_radius = float(np.clip(platform_radius, 4.0, 12.0))

        support_height = self._estimate_goal_support_height(
            goal_pos[:2],
            radius=platform_radius,
            grid_points=7,
        )

        required_clearance = float(self._get_goal_altitude())
        try:
            stored_support_height = metadata.get('goal_support_terrain_height')
            if stored_support_height is not None:
                stored_support_height = float(stored_support_height)
                required_clearance = max(required_clearance, float(goal_pos[2]) - stored_support_height)
        except Exception:
            pass

        required_goal_height = float(support_height + max(required_clearance, 1.0))
        if float(goal_pos[2]) < required_goal_height - 1e-6:
            old_goal_z = float(goal_pos[2])
            goal_pos[2] = required_goal_height
            if not quiet_output:
                print(
                    f"[目标位置修正] 固定目标Z抬升: {old_goal_z:.2f} -> {goal_pos[2]:.2f} "
                    f"(support_h={support_height:.2f}, clearance={required_clearance:.2f})"
                )
        return goal_pos
        
    def _setup_complexity_parameters(self):
        """根据复杂度等级设置地形参数"""
        complexity_configs = {
            1: {  # 简单
                'num_mountains': 5,
                'num_obstacles': 4,
                'noise_amplitude': 2.0,  # 大幅降低噪声，从8.0降到2.0
                'add_canyon': False,
                'mountain_height_range': [50, 70],  # 增加高度让山峰更明显
                'mountain_width_range': [12, 18]   # 减小宽度让山峰更尖锐
            },
            2: {  # 中等
                'num_mountains': 6,
                'num_obstacles': 8,
                'noise_amplitude': 2.5,  # 大幅降低噪声，从10.0降到2.5
                'add_canyon': False,
                'mountain_height_range': [60, 80],
                'mountain_width_range': [14, 20]
            },
            3: {  # 困难
                'num_mountains': 7,
                'num_obstacles': 12,
                'noise_amplitude': 3.0,  # 大幅降低噪声，从12.0降到3.0
                'add_canyon': False,
                'mountain_height_range': [70, 90],
                'mountain_width_range': [16, 22]
            },
            4: {  # 极难
                'num_mountains': 8,
                'num_obstacles': 16,
                'noise_amplitude': 3.5,  # 大幅降低噪声，从15.0降到3.5
                'add_canyon': False,
                'mountain_height_range': [80, 100],
                'mountain_width_range': [18, 24]
            }
        }
        
        # 获取当前复杂度等级的配置
        config = complexity_configs.get(self.terrain_complexity_level, complexity_configs[1])
        
        # 设置参数
        self.num_mountains = config['num_mountains']
        self.num_obstacles = config['num_obstacles']
        self.noise_amplitude = config['noise_amplitude']
        self.add_canyon = config['add_canyon']
        self.mountain_height_range = config['mountain_height_range']
        self.mountain_width_range = config['mountain_width_range']
        
        quiet_output = os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')
        if not quiet_output and not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
            print(f"[复杂度设置] 等级: {self.terrain_complexity_level}")
            print(f"[复杂度设置] 山峰数量: {self.num_mountains}")
            print(f"[复杂度设置] 障碍物数量: {self.num_obstacles}")
            print(f"[复杂度设置] 噪声强度: {self.noise_amplitude}")
            print(f"[复杂度设置] 峡谷: {'是' if self.add_canyon else '否'}")

    def _use_deterministic_train_env_sequence(self):
        return bool(getattr(self, 'deterministic_train_env_sequence', False))

    def _resolve_episode_context(self, world=None):
        try:
            episode_idx = int(
                getattr(
                    world,
                    'episode_index',
                    getattr(self, 'current_episode_index', 0),
                )
            )
        except Exception:
            episode_idx = int(getattr(self, 'current_episode_index', 0) or 0)
        try:
            env_id = int(
                getattr(
                    world,
                    'env_id',
                    getattr(self, 'current_episode_env_id', 0),
                )
            )
        except Exception:
            env_id = int(getattr(self, 'current_episode_env_id', 0) or 0)
        return max(0, int(episode_idx)), max(0, int(env_id))

    def _make_deterministic_episode_seed(self, namespace, episode_idx, env_id=0):
        base_sequence_seed = int(
            getattr(
                self,
                'training_env_sequence_seed',
                getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67),
            )
        )
        terrain_base_seed = int(getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67))
        payload = (
            f"{namespace}|seq={base_sequence_seed}|terrain={terrain_base_seed}|"
            f"episode={int(episode_idx)}|env={int(env_id)}|"
            f"complexity={int(getattr(self, 'terrain_complexity_level', 0))}|"
            f"map={int(round(float(getattr(self, 'map_size', 0.0))))}"
        )
        digest = hashlib.blake2b(payload.encode('utf-8'), digest_size=8).digest()
        seed = int.from_bytes(digest, 'little') % 2147483647
        return int(seed if seed > 0 else 1)

    def _make_train_obstacle_family_seed(self, episode_idx, env_id=0):
        base_sequence_seed = int(
            getattr(
                self,
                'training_env_sequence_seed',
                getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67),
            )
        )
        namespace = str(
            getattr(self, 'train_obstacle_sequence_namespace', 'train_obstacle')
        ).strip() or 'train_obstacle'
        cache_key = (int(base_sequence_seed), str(namespace), int(env_id))
        cache = getattr(self, '_train_obstacle_sequence_cache', None)
        if cache is None:
            cache = {}
            self._train_obstacle_sequence_cache = cache
        entry = cache.get(cache_key)
        if entry is None:
            entry = {
                'rng': random.Random(f"{int(base_sequence_seed)}::{namespace}::env{int(env_id)}"),
                'values': [],
            }
            cache[cache_key] = entry
        values = entry['values']
        rng = entry['rng']
        target_idx = max(0, int(episode_idx))
        while len(values) <= target_idx:
            values.append(int(rng.randint(1000, 99999)))
        return int(values[target_idx])

    def _make_hold_block_length(self, block_idx, env_id=0):
        hold_mode = str(getattr(self, 'semi_random_hold_mode', 'episode')).strip().lower()
        if hold_mode != 'range':
            return max(1, int(getattr(self, 'semi_random_hold_episodes', 1) or 1))
        min_len = max(1, int(getattr(self, 'semi_random_hold_min_episodes', 1) or 1))
        max_len = max(min_len, int(getattr(self, 'semi_random_hold_max_episodes', min_len) or min_len))
        span = max_len - min_len + 1
        if span <= 1:
            return int(min_len)
        base_sequence_seed = int(
            getattr(
                self,
                'training_env_sequence_seed',
                getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67),
            )
        )
        terrain_base_seed = int(getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67))
        payload = (
            f"semi_random_hold|seq={base_sequence_seed}|terrain={terrain_base_seed}|"
            f"block={int(block_idx)}|env={int(env_id)}|"
            f"complexity={int(getattr(self, 'terrain_complexity_level', 0))}|"
            f"map={int(round(float(getattr(self, 'map_size', 0.0))))}"
        )
        digest = hashlib.blake2b(payload.encode('utf-8'), digest_size=8).digest()
        offset = int.from_bytes(digest, 'little') % span
        return int(min_len + offset)

    def _resolve_semi_random_hold_context(self, episode_idx, env_id=0):
        episode_idx = max(0, int(episode_idx))
        env_id = max(0, int(env_id))
        hold_mode = str(getattr(self, 'semi_random_hold_mode', 'episode')).strip().lower()
        if hold_mode == 'fixed':
            hold_length = max(1, int(getattr(self, 'semi_random_hold_episodes', 1) or 1))
            block_idx = episode_idx // hold_length
            block_start = block_idx * hold_length
            return {
                'mode': 'fixed',
                'block_idx': int(block_idx),
                'block_start_episode': int(block_start),
                'block_length': int(hold_length),
            }
        if hold_mode == 'range':
            block_idx = 0
            block_start = 0
            while True:
                block_length = self._make_hold_block_length(block_idx, env_id)
                block_end = block_start + block_length
                if episode_idx < block_end:
                    return {
                        'mode': 'range',
                        'block_idx': int(block_idx),
                        'block_start_episode': int(block_start),
                        'block_length': int(block_length),
                    }
                block_idx += 1
                block_start = block_end
        return {
            'mode': 'episode',
            'block_idx': int(episode_idx),
            'block_start_episode': int(episode_idx),
            'block_length': 1,
        }

    def _make_deterministic_terrain_variant_seed(self, episode_idx, env_id=0):
        hold_ctx = self._resolve_semi_random_hold_context(episode_idx, env_id)
        block_idx = int(hold_ctx.get('block_idx', max(0, int(episode_idx))))
        seed = self._make_deterministic_episode_seed('terrain_variant', block_idx, env_id)
        self.current_terrain_hold_block_index = block_idx
        self.current_terrain_hold_block_start_episode = int(hold_ctx.get('block_start_episode', 0))
        self.current_terrain_hold_block_length = int(hold_ctx.get('block_length', 1))
        return int(seed)

    def _load_heldout_reference_layout(self):
        position_mode = os.getenv('HELDOUT_POSITION_MODE', '').strip().lower()
        if position_mode != 'same_region':
            return None

        reference_path = os.getenv('HELDOUT_REFERENCE_POSITIONS_FILE', '').strip()
        if not reference_path or not os.path.exists(reference_path):
            return None

        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            agents = np.asarray(data.get('agents', []), dtype=np.float32)
            goal = np.asarray(data.get('goal', []), dtype=np.float32)
            if agents.ndim != 2 or agents.shape[0] == 0 or agents.shape[1] < 2:
                return None
            if goal.ndim != 1 or goal.size < 2:
                return None
            start_center_xy = np.mean(agents[:, :2], axis=0)
            goal_xy = goal[:2].astype(np.float32)
            route_xy = goal_xy - start_center_xy
            route_distance = float(np.linalg.norm(route_xy))
            route_unit = None
            if route_distance > 1e-6:
                route_unit = (route_xy / route_distance).astype(np.float32)
            return {
                'path': reference_path,
                'agents': agents,
                'goal': goal,
                'start_center_xy': start_center_xy.astype(np.float32),
                'goal_xy': goal_xy,
                'route_xy': route_xy.astype(np.float32),
                'route_distance': route_distance,
                'route_unit': route_unit,
            }
        except Exception:
            return None

    def _load_heldout_reference_start_center(self):
        reference_layout = self._load_heldout_reference_layout()
        if reference_layout is None:
            return None
        return reference_layout.get('start_center_xy')

    def _resolve_start_area_bounds(self, map_size):
        start_area_size = int(map_size * 0.15)
        start_area_margin = int(map_size * 0.05)

        reference_center = self._load_heldout_reference_start_center()
        if reference_center is not None:
            half_size = max(1.0, start_area_size / 2.0)
            max_origin = max(start_area_margin, map_size - start_area_margin - start_area_size)
            origin_x = int(np.clip(np.floor(float(reference_center[0]) - half_size), start_area_margin, max_origin))
            origin_y = int(np.clip(np.floor(float(reference_center[1]) - half_size), start_area_margin, max_origin))
            start_area_x = (origin_x, origin_x + start_area_size)
            start_area_y = (origin_y, origin_y + start_area_size)
            return start_area_x, start_area_y, {
                'size': start_area_size,
                'source': 'heldout_reference_same_region',
            }

        use_semi_random = bool(
            getattr(
                self,
                'use_semi_random_terrain',
                os.getenv('SEMI_RANDOM_TERRAIN', '0').lower() in ('1', 'true', 'yes', 'on'),
            )
        )
        if use_semi_random:
            try:
                base_seed = int(getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67))
            except Exception:
                base_seed = int(self.seed if self.seed is not None else 67)
            start_area_rng = np.random.RandomState(base_seed + 911)
            corner_choice = int(start_area_rng.randint(0, 4))
            source_prefix = 'semi_random_base_corner'
        else:
            corner_choice = int(self.rng.randint(0, 4))  # 0=SW, 1=SE, 2=NW, 3=NE
            source_prefix = 'random_corner'

        if corner_choice == 0:  # 西南角 (左下)
            start_area_x = (start_area_margin, start_area_margin + start_area_size)
            start_area_y = (start_area_margin, start_area_margin + start_area_size)
            source = f'{source_prefix}_sw'
        elif corner_choice == 1:  # 东南角 (右下)
            start_area_x = (map_size - start_area_margin - start_area_size, map_size - start_area_margin)
            start_area_y = (start_area_margin, start_area_margin + start_area_size)
            source = f'{source_prefix}_se'
        elif corner_choice == 2:  # 西北角 (左上)
            start_area_x = (start_area_margin, start_area_margin + start_area_size)
            start_area_y = (map_size - start_area_margin - start_area_size, map_size - start_area_margin)
            source = f'{source_prefix}_nw'
        else:  # 东北角 (右上)
            start_area_x = (map_size - start_area_margin - start_area_size, map_size - start_area_margin)
            start_area_y = (map_size - start_area_margin - start_area_size, map_size - start_area_margin)
            source = f'{source_prefix}_ne'

        return start_area_x, start_area_y, {
            'size': start_area_size,
            'source': source,
        }

    def _generate_visualizer_style_terrain(self):
        """
        使用与 visualize_terrain_map.py 相同的逻辑生成地形
        确保训练环境地形与独立地图生成工具完全一致
        
        🔧 改进：确保起点附近区域平坦，便于无人机正常起飞
        """
        quiet = (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on'))
        map_size = int(self.map_size)

        # 初始化高度图，完全复刻 visualize_terrain_map.py 的实现
        height_map = np.zeros((map_size, map_size), dtype=np.float32)

        # 🔧 确定起点区域：
        # - 常规训练沿用随机角落平坦区
        # - heldout same_region 测试时，直接绑定到参考起点区域，避免“同区域位置”却落在另一块随机角落平坦区上
        start_area_x, start_area_y, start_area_meta = self._resolve_start_area_bounds(map_size)
        start_area_size = int(start_area_meta['size'])

        # 保存起点区域信息，供后续使用
        self.start_area = {
            'x_range': start_area_x,
            'y_range': start_area_y,
            'size': start_area_size,
            'source': start_area_meta.get('source', 'random_corner')
        }
        
        if not quiet:
            print(
                f"[地形生成] 起点区域: x=[{start_area_x[0]}, {start_area_x[1]}], "
                f"y=[{start_area_y[0]}, {start_area_y[1]}], "
                f"source={start_area_meta.get('source', 'random_corner')}"
            )

        # 与可视化脚本相同的参数
        num_peaks = int(getattr(self, 'num_mountains', 6))
        height_range = tuple(self.mountain_height_range)
        width_range = tuple(self.mountain_width_range)
        noise_scale = float(self.noise_amplitude)

        min_distance = int(os.getenv('MOUNTAIN_MIN_DISTANCE', '55'))
        margin = int(os.getenv('MOUNTAIN_MARGIN', '20'))
        
        # 🔧 确保起点区域与山峰保持足够距离
        start_area_center_x = (start_area_x[0] + start_area_x[1]) / 2
        start_area_center_y = (start_area_y[0] + start_area_y[1]) / 2
        min_distance_from_start = start_area_size * 1.5  # 山峰距离起点区域的最小距离

        # ========== 🔧 半随机地形生成：固定基准位置，局部波动 ==========
        # 环境变量控制
        use_semi_random = bool(getattr(self, 'use_semi_random_terrain',
            os.getenv('SEMI_RANDOM_TERRAIN', '0').lower() in ('1', 'true', 'yes', 'on')))
        peak_jitter_range = float(getattr(self, 'peak_jitter_range',
            os.getenv('PEAK_JITTER_RANGE', '15.0')))  # 山峰位置波动范围（米）
        peak_center_jitter_range = float(
            getattr(
                self,
                'peak_center_jitter_range',
                os.getenv('PEAK_CENTER_JITTER_RANGE', str(min(float(peak_jitter_range), 3.0))),
            )
        )
        peak_height_jitter_ratio_min = float(
            getattr(self, 'peak_height_jitter_ratio_min', os.getenv('PEAK_HEIGHT_JITTER_RATIO_MIN', '0.20'))
        )
        peak_height_jitter_ratio_max = float(
            getattr(self, 'peak_height_jitter_ratio_max', os.getenv('PEAK_HEIGHT_JITTER_RATIO_MAX', '0.40'))
        )
        peak_height_max_scale = float(
            getattr(self, 'peak_height_max_scale', os.getenv('PEAK_HEIGHT_MAX_SCALE', '1.30'))
        )
        peak_height_jitter_ratio_min = max(0.0, peak_height_jitter_ratio_min)
        peak_height_jitter_ratio_max = max(peak_height_jitter_ratio_min, peak_height_jitter_ratio_max)
        peak_height_max_scale = max(1.0, peak_height_max_scale)
        peak_center_jitter_range = max(0.0, peak_center_jitter_range)
        variant_noise_ratio = max(
            0.0,
            float(
                getattr(self, 'terrain_variant_noise_ratio', os.getenv('TERRAIN_VARIANT_NOISE_RATIO', '0.15'))
            ),
        )
        variant_seed_value = getattr(self, 'terrain_variant_seed', None)
        if variant_seed_value is None:
            variant_seed_value = getattr(self, 'current_terrain_variant_seed', None)
        if variant_seed_value is None:
            variant_seed_value = self.seed if self.seed is not None else getattr(self, 'terrain_base_seed', 67)
        variant_seed = int(variant_seed_value)
        variant_rng = np.random.RandomState(variant_seed)

        heldout_reference_layout = self._load_heldout_reference_layout()
        reserved_peak_regions = [
            {
                'center': np.asarray([start_area_center_x, start_area_center_y], dtype=np.float32),
                'radius': float(min_distance_from_start),
                'label': 'start_area',
            }
        ]
        if heldout_reference_layout is not None and heldout_reference_layout.get('goal_xy') is not None:
            goal_safe_radius = float(np.clip(
                max(
                    18.0,
                    start_area_size * 0.7,
                    min_distance * 0.45,
                    peak_jitter_range + 10.0,
                ),
                18.0,
                map_size * 0.18,
            ))
            reserved_peak_regions.append(
                {
                    'center': np.asarray(heldout_reference_layout['goal_xy'][:2], dtype=np.float32),
                    'radius': goal_safe_radius,
                    'label': 'heldout_goal_region',
                }
            )

        def _peak_region_clearance(cx, cy):
            if not reserved_peak_regions:
                return float('inf')
            best = float('inf')
            for region in reserved_peak_regions:
                center = np.asarray(region['center'], dtype=np.float32)
                clearance = float(np.hypot(float(cx) - float(center[0]), float(cy) - float(center[1]))) - float(region['radius'])
                best = min(best, clearance)
            return best

        peak_spacing_floor = max(18.0, float(min_distance) * 0.82)
        configured_height_span = max(0.0, float(height_range[1]) - float(height_range[0]))

        def _sample_local_peak_offset():
            if peak_center_jitter_range <= 1e-6:
                return 0.0, 0.0
            offset_std = max(0.35, peak_center_jitter_range / 2.8)
            for _ in range(16):
                offset_x = float(variant_rng.normal(0.0, offset_std))
                offset_y = float(variant_rng.normal(0.0, offset_std))
                radius = float(np.hypot(offset_x, offset_y))
                if radius <= peak_center_jitter_range:
                    return offset_x, offset_y
            theta = float(variant_rng.uniform(0.0, 2.0 * np.pi))
            radius = float(variant_rng.uniform(0.35 * peak_center_jitter_range, peak_center_jitter_range))
            return np.cos(theta) * radius, np.sin(theta) * radius
        
        base_seed = None
        if use_semi_random:
            # 使用固定种子生成基准山峰位置（确保每次训练的基准位置相同）
            base_seed = int(getattr(self, 'terrain_base_seed', os.getenv('TERRAIN_BASE_SEED', '67')))
            base_rng = np.random.RandomState(base_seed)
            
            # 生成基准山峰参数（固定不变）
            # 对 similar_unseen 而言，保持峰高/峰宽/噪声基底稳定，仅允许位置局部漂移。
            base_peak_specs = []
            for _ in range(num_peaks):
                attempts = 0
                while attempts < 200:
                    x = base_rng.randint(margin, max(margin + 1, map_size - margin))
                    y = base_rng.randint(margin, max(margin + 1, map_size - margin))
                    
                    if _peak_region_clearance(x, y) < 0.0:
                        attempts += 1
                        continue
                    
                    too_close = False
                    for px, py, _, _ in base_peak_specs:
                        if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        base_peak_specs.append(
                            (
                                x,
                                y,
                                float(base_rng.uniform(*height_range)),
                                float(base_rng.uniform(*width_range)),
                            )
                        )
                        break
                    attempts += 1

            base_peak_heights = [float(spec[2]) for spec in base_peak_specs]
            base_peak_height_cap = max(base_peak_heights) * peak_height_max_scale if base_peak_heights else float(height_range[1]) * peak_height_max_scale
            min_peak_height = max(0.0, float(height_range[0]))
            
            # 在基准位置周围生成实际山峰位置（小范围波动），峰高允许有限振幅变化
            peak_specs = []
            for base_x, base_y, base_height, base_width in base_peak_specs:
                best_candidate = None
                best_cost = None
                sampled_candidate = None
                num_attempts = 48 if peak_center_jitter_range > 1e-6 else 1
                for _ in range(num_attempts):
                    jitter_x, jitter_y = _sample_local_peak_offset()

                    actual_x = int(np.clip(np.rint(base_x + jitter_x), margin, map_size - margin - 1))
                    actual_y = int(np.clip(np.rint(base_y + jitter_y), margin, map_size - margin - 1))

                    candidate_cost = 0.0
                    region_clearance = _peak_region_clearance(actual_x, actual_y)
                    if region_clearance < 0.0:
                        candidate_cost += 1e6 + abs(region_clearance) * 5000.0
                    is_valid = region_clearance >= 0.0
                    for px, py, _, _ in peak_specs:
                        sep = float(np.hypot(actual_x - px, actual_y - py))
                        if sep < peak_spacing_floor:
                            candidate_cost += 1e6 + (peak_spacing_floor - sep) * 5000.0
                            is_valid = False
                    candidate_cost += float(np.hypot(actual_x - base_x, actual_y - base_y)) * 2.0

                    if best_cost is None or candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_candidate = (actual_x, actual_y)
                    if is_valid:
                        sampled_candidate = (actual_x, actual_y)
                        break

                if sampled_candidate is not None:
                    chosen_x, chosen_y = sampled_candidate
                elif best_candidate is not None:
                    chosen_x, chosen_y = best_candidate
                else:
                    chosen_x, chosen_y = (
                        int(np.clip(base_x, margin, map_size - margin - 1)),
                        int(np.clip(base_y, margin, map_size - margin - 1)),
                    )

                height_delta = 0.0
                if configured_height_span > 1e-6 and peak_height_jitter_ratio_max > 1e-9:
                    height_delta_ratio = float(
                        variant_rng.uniform(peak_height_jitter_ratio_min, peak_height_jitter_ratio_max)
                    )
                    height_delta_sign = -1.0 if float(variant_rng.uniform(0.0, 1.0)) < 0.5 else 1.0
                    height_delta = height_delta_sign * height_delta_ratio * configured_height_span

                actual_height = float(
                    np.clip(
                        float(base_height) + height_delta,
                        min_peak_height,
                        base_peak_height_cap,
                    )
                )
                peak_specs.append((chosen_x, chosen_y, actual_height, float(base_width)))
            
            if not quiet:
                avg_jitter = np.mean([
                    np.sqrt((p[0] - bp[0])**2 + (p[1] - bp[1])**2)
                    for p, bp in zip(peak_specs, base_peak_specs)
                ])
                avg_height_delta = np.mean([
                    abs(float(p[2]) - float(bp[2]))
                    for p, bp in zip(peak_specs, base_peak_specs)
                ]) if peak_specs else 0.0
                print(f"[半随机地形] 基准种子: {base_seed}, 山峰数量: {num_peaks}")
                print(
                    f"[半随机地形] 同源扰动: base_seed={base_seed}, variant_seed={variant_seed}, "
                    f"中心扰动≤±{peak_center_jitter_range:.2f}m, 平均偏移={avg_jitter:.2f}m"
                )
                print(
                    f"[半随机地形] 峰高扰动: {peak_height_jitter_ratio_min:.2f}-{peak_height_jitter_ratio_max:.2f} x range, "
                    f"cap={peak_height_max_scale:.2f}x base_max, 平均|Δh|={avg_height_delta:.2f}m"
                )
                if len(reserved_peak_regions) > 1:
                    protected = reserved_peak_regions[1]
                    print(
                        f"[半随机地形] 保护目标区域: center=({protected['center'][0]:.1f}, {protected['center'][1]:.1f}), "
                        f"radius={protected['radius']:.1f}m"
                    )
        else:
            # 原始的完全随机生成方式
            peak_specs = []
            for _ in range(num_peaks):
                attempts = 0
                while attempts < 200:
                    x = self.rng.randint(margin, max(margin + 1, map_size - margin))
                    y = self.rng.randint(margin, max(margin + 1, map_size - margin))

                    if _peak_region_clearance(x, y) < 0.0:
                        attempts += 1
                        continue

                    too_close = False
                    for px, py, _, _ in peak_specs:
                        if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                            too_close = True
                            break

                    if not too_close:
                        peak_specs.append((x, y, None, None))
                        break
                    attempts += 1

        # 純粹依照 visualize_terrain_map.py 的双层循环实现高斯山峰
        for px, py, base_height, base_width in peak_specs:
            if use_semi_random and base_height is not None and base_width is not None:
                height = float(base_height)
                width = float(base_width)
            else:
                height = self.rng.uniform(*height_range)
                width = self.rng.uniform(*width_range)

            for i in range(map_size):
                for j in range(map_size):
                    dist = np.sqrt((i - px)**2 + (j - py)**2)
                    contribution = height * np.exp(-(dist**2) / (2 * width**2))
                    height_map[i, j] += contribution

        if use_semi_random and base_seed is not None:
            noise_rng = np.random.RandomState(base_seed + 104729)
            noise = noise_rng.randn(map_size, map_size) * noise_scale
            if variant_noise_ratio > 1e-9:
                noise += variant_rng.randn(map_size, map_size) * (noise_scale * variant_noise_ratio)
        else:
            noise = self.rng.randn(map_size, map_size) * noise_scale
        height_map += noise.astype(np.float32)
        height_map = np.maximum(height_map, 0.0).astype(np.float32)
        
        # 🔧 对起点区域进行平坦化处理，确保无人机可以正常起飞
        # 将起点区域的高度设为低值（0-5米），并使用平滑过渡
        if use_semi_random and base_seed is not None:
            start_height_rng = np.random.RandomState(base_seed + 2047)
            start_flat_height = start_height_rng.uniform(0.0, 5.0)
        else:
            start_flat_height = self.rng.uniform(0.0, 5.0)  # 起点区域的目标高度（0-5米）
        start_x0, start_x1 = int(start_area_x[0]), int(start_area_x[1])
        start_y0, start_y1 = int(start_area_y[0]), int(start_area_y[1])
        
        # 对起点区域内的所有点进行平坦化
        for i in range(start_y0, min(start_y1, map_size)):
            for j in range(start_x0, min(start_x1, map_size)):
                # 使用平滑过渡，避免硬边界
                # 计算到起点区域边界的距离，用于平滑过渡
                dist_to_edge_x = min(j - start_x0, start_x1 - j) / (start_area_size / 2.0)
                dist_to_edge_y = min(i - start_y0, start_y1 - i) / (start_area_size / 2.0)
                blend_factor = min(dist_to_edge_x, dist_to_edge_y)
                blend_factor = np.clip(blend_factor, 0.0, 1.0)
                
                # 在起点区域中心保持平坦，边缘平滑过渡到原始地形
                target_height = start_flat_height * blend_factor + height_map[i, j] * (1.0 - blend_factor)
                height_map[i, j] = target_height
        
        # 对起点区域进行轻微平滑，确保平坦度
        from scipy.ndimage import gaussian_filter
        smooth_region = height_map[start_y0:min(start_y1, map_size), start_x0:min(start_x1, map_size)].copy()
        smooth_region = gaussian_filter(smooth_region, sigma=1.0)
        height_map[start_y0:min(start_y1, map_size), start_x0:min(start_x1, map_size)] = smooth_region
        
        if not quiet:
            start_avg_height = np.mean(height_map[start_y0:min(start_y1, map_size), start_x0:min(start_x1, map_size)])
            start_height_std = np.std(height_map[start_y0:min(start_y1, map_size), start_x0:min(start_x1, map_size)])
            print(f"[地形生成] 起点区域平坦化完成: 平均高度={start_avg_height:.2f}m, 标准差={start_height_std:.2f}m")

        # 🔧 关键修复：对地形进行降采样，确保训练地图与可视化地图完全一致
        # 使用与可视化代码相同的降采样方式（sample_rate = 4，从200×200降到50×50）
        # 保持坐标系统不变（0-200），但地形数据降采样为50×50，通过插值获取中间值
        sample_rate = 4  # 每4个点采样1个，与可视化代码保持一致
        # 🔧 修复：确保采样覆盖整个map_size范围（0到map_size-1）
        # 如果map_size-1不能被sample_rate整除，需要添加最后一个点
        x_samples = np.arange(0, map_size, sample_rate)
        y_samples = np.arange(0, map_size, sample_rate)
        # 确保包含最后一个点（map_size-1），以覆盖整个坐标范围
        if (map_size - 1) % sample_rate != 0:
            x_samples = np.append(x_samples, map_size - 1)
            y_samples = np.append(y_samples, map_size - 1)
        
        # 创建降采样后的地形数据（与可视化代码完全相同的逻辑）
        terrain_data_sampled = []
        for y in y_samples:
            row = []
            for x in x_samples:
                z = height_map[int(y), int(x)]
                row.append(float(z))
            terrain_data_sampled.append(row)
        terrain_data_sampled = np.array(terrain_data_sampled, dtype=np.float32)
        
        # 保存降采样后的地形数据（50×50）
        # 注意：map_size保持为200（坐标系统不变），但terrain是50×50
        self.terrain = terrain_data_sampled
        self.terrain_downsampled = True  # 标记地形已降采样
        self.terrain_sample_rate = sample_rate  # 保存降采样率，用于get_terrain_height插值
        
        # 保存山峰中心坐标（保持原始坐标，不缩放）
        self.base_mountain_centers = []
        self.actual_mountain_centers = []
        if use_semi_random and 'base_peak_specs' in locals():
            self.base_mountain_centers = [
                (int(px), int(py), float(self.get_terrain_height(px, py))) for px, py, _, _ in base_peak_specs
            ]
        self.actual_mountain_centers = [
            (int(px), int(py), float(self.get_terrain_height(px, py))) for px, py, _, _ in peak_specs
        ]
        self.mountain_centers = list(self.actual_mountain_centers)
        self.grid_points = np.meshgrid(
            np.arange(map_size, dtype=np.float32),
            np.arange(map_size, dtype=np.float32),
            indexing='ij'
        )
        self.terrain_params = {
            'method': 'visualizer_gaussian',
            'terrain_complexity_level': self.terrain_complexity_level,
            'num_peaks': len(peak_specs),
            'height_range': tuple(self.mountain_height_range),
            'width_range': tuple(self.mountain_width_range),
            'noise_scale': noise_scale,
            'min_distance': min_distance,
            'seed': self.seed,
            'terrain_variant_seed': int(variant_seed) if use_semi_random else None,
            'semi_random_terrain': bool(use_semi_random),
            'terrain_base_seed': int(base_seed) if base_seed is not None else None,
            'peak_jitter_range': float(peak_jitter_range),
            'peak_center_jitter_range': float(peak_center_jitter_range),
            'peak_height_jitter_ratio_range': (
                float(peak_height_jitter_ratio_min),
                float(peak_height_jitter_ratio_max),
            ),
            'peak_height_max_scale': float(peak_height_max_scale),
            'terrain_variant_noise_ratio': float(variant_noise_ratio),
            'semi_random_hold_mode': str(getattr(self, 'semi_random_hold_mode', 'episode')),
            'semi_random_hold_episodes': int(getattr(self, 'semi_random_hold_episodes', 1)),
            'semi_random_hold_min_episodes': int(getattr(self, 'semi_random_hold_min_episodes', 1)),
            'semi_random_hold_max_episodes': int(getattr(self, 'semi_random_hold_max_episodes', 1)),
            'terrain_hold_block_index': int(getattr(self, 'current_terrain_hold_block_index', 0)),
            'terrain_hold_block_start_episode': int(getattr(self, 'current_terrain_hold_block_start_episode', 0)),
            'terrain_hold_block_length': int(getattr(self, 'current_terrain_hold_block_length', 1)),
            'deterministic_train_env_sequence': bool(getattr(self, 'deterministic_train_env_sequence', False)),
            'training_env_sequence_seed': int(getattr(self, 'training_env_sequence_seed', 0)),
            'episode_index': int(getattr(self, 'current_episode_index', 0)),
            'env_id': int(getattr(self, 'current_episode_env_id', 0)),
            'episode_rng_seed': int(getattr(self, 'current_episode_rng_seed', 0)) if getattr(self, 'current_episode_rng_seed', None) is not None else None,
        }

        if not quiet:
            print(f"[地形生成] ✅ 使用visualizer风格生成完成")
            print(f"[地形生成] 地图尺寸: {map_size}×{map_size}, 山峰数量: {len(peak_specs)}, 噪声: {noise_scale}")

        return True

    def generate_terrain(self):
        """根据配置选择地形生成方式"""
        use_legacy = os.getenv('USE_LEGACY_TERRAIN', '0').lower() in ('1', 'true', 'yes', 'on', 'legacy')
        if use_legacy:
            return self._generate_terrain_legacy()
        return self._generate_visualizer_style_terrain()
        
    def regenerate_terrain(self, new_seed=None, variant_seed=None):
        """
        重新生成地形，可选择使用新的随机种子
        在训练/测试过程中可调用以动态改变环境
        """
        use_semi_random = bool(
            getattr(
                self,
                'use_semi_random_terrain',
                os.getenv('SEMI_RANDOM_TERRAIN', '0').lower() in ('1', 'true', 'yes', 'on'),
            )
        )

        old_signature = (
            getattr(self, 'current_terrain_seed', self.seed),
            getattr(self, 'current_terrain_variant_seed', getattr(self, 'terrain_variant_seed', self.seed)),
        )

        if use_semi_random:
            try:
                base_seed = int(getattr(self, 'terrain_base_seed', self.seed if self.seed is not None else 67))
            except Exception:
                base_seed = int(self.seed if self.seed is not None else 67)
            if variant_seed is None:
                if new_seed is not None:
                    variant_seed = int(new_seed)
                elif self._use_deterministic_train_env_sequence():
                    episode_idx, env_id = self._resolve_episode_context()
                    variant_seed = self._make_deterministic_terrain_variant_seed(episode_idx, env_id)
                else:
                    variant_seed = int(np.random.randint(0, 100000))
            self.seed = int(base_seed)
            self.terrain_seed = int(base_seed)
            self.terrain_variant_seed = int(variant_seed)
            self.rng = np.random.RandomState(int(variant_seed))
            if not self._use_deterministic_train_env_sequence():
                self.current_terrain_hold_block_index = int(getattr(self, 'current_episode_index', 0))
                self.current_terrain_hold_block_start_episode = int(getattr(self, 'current_episode_index', 0))
                self.current_terrain_hold_block_length = 1
        else:
            if new_seed is not None:
                self.seed = int(new_seed)
                self.rng = np.random.RandomState(int(new_seed))
            elif self._use_deterministic_train_env_sequence():
                episode_idx, env_id = self._resolve_episode_context()
                self.seed = self._make_deterministic_episode_seed('terrain', episode_idx, env_id)
                self.rng = np.random.RandomState(self.seed)
            else:
                self.seed = int(np.random.randint(0, 100000))
                self.rng = np.random.RandomState(self.seed)
            self.terrain_seed = self.seed
            self.terrain_variant_seed = None
            
        # 🔧 关键修复：检测地形种子是否变化
        # 如果地形种子变化了，重置位置初始化标记，使每个新地图都动态生成位置
        new_signature = (
            int(self.seed) if self.seed is not None else None,
            int(self.terrain_variant_seed) if getattr(self, 'terrain_variant_seed', None) is not None else None,
        )
        terrain_changed = (old_signature != new_signature)
        if terrain_changed and hasattr(self, 'dynamic_first_time') and self.dynamic_first_time:
            # 地形变化了，重置位置初始化标记
            self.positions_initialized = False
            # 清空之前保存的固定位置，因为地形已经变化
            if hasattr(self, 'fixed_positions'):
                self.fixed_positions = None
            self.use_fixed_positions = False  # 重置为不使用固定位置
            try:
                # 🔧 关键修复：添加 SUPPRESS_TERRAIN_OUTPUT 检查，减少并行环境输出
                suppress_output = os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1', 'true', 'yes', 'on')
                if not suppress_output:
                    print(f"[位置重置] 检测到地形变化 (旧签名: {old_signature}, 新签名: {new_signature})，重置位置初始化标记")
            except Exception:
                pass
        
        # 更新当前地形种子
        self.current_terrain_seed = self.seed
        self.current_terrain_variant_seed = self.terrain_variant_seed
            
        # 清空旧数据
        self.terrain = None
        self.obstacles = []
        self._obstacle_layout_signature = None
        self.terrain_complexity = {}
        
        # 重新生成地形
        self.generate_terrain()
        # 🔧 关键修复：添加 SUPPRESS_TERRAIN_OUTPUT 检查，减少并行环境输出
        suppress_output = os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1', 'true', 'yes', 'on')
        if not suppress_output:
            print(f"\n************************************************")
            print(f"*                                              *")
            print(f"*        [TERRAIN REGENERATED]                 *")
            if use_semi_random:
                print(f"*  Base Seed: {self.seed:<8} Variant: {self.terrain_variant_seed:<8} *")
            else:
                print(f"*        Seed: {self.seed}                     *")
            print(f"*                                              *")
            print(f"************************************************\n")
        
        return self.terrain

    def build_terrain_snapshot(self):
        """构建可序列化的地形快照，便于训练/评估严格复现与审计。"""
        snapshot = {
            'terrain': np.asarray(self.terrain, dtype=np.float32).copy() if self.terrain is not None else None,
            'map_size': int(self.map_size) if self.map_size is not None else None,
            'goal_pos': np.asarray(self.goal_pos, dtype=np.float32).copy() if self.goal_pos is not None else None,
            'obstacles': list(getattr(self, 'obstacles', []) or []),
            'terrain_seed': int(getattr(self, 'current_terrain_seed', self.seed)) if getattr(self, 'current_terrain_seed', self.seed) is not None else None,
            'terrain_variant_seed': int(getattr(self, 'current_terrain_variant_seed', getattr(self, 'terrain_variant_seed', 0))) if getattr(self, 'current_terrain_variant_seed', getattr(self, 'terrain_variant_seed', None)) is not None else None,
            'terrain_params': dict(getattr(self, 'terrain_params', {}) or {}),
            'terrain_source': 'scenario_snapshot',
            'base_mountain_centers': list(getattr(self, 'base_mountain_centers', []) or []),
            'actual_mountain_centers': list(getattr(self, 'actual_mountain_centers', []) or []),
            'episode_index': int(getattr(self, 'current_episode_index', 0)),
            'env_id': int(getattr(self, 'current_episode_env_id', 0)),
            'episode_rng_seed': int(getattr(self, 'current_episode_rng_seed', 0)) if getattr(self, 'current_episode_rng_seed', None) is not None else None,
            'obstacle_seed': int(getattr(self, 'current_episode_obstacle_seed', 0)) if getattr(self, 'current_episode_obstacle_seed', None) is not None else None,
            'training_env_sequence_seed': int(getattr(self, 'training_env_sequence_seed', 0)),
            'deterministic_train_env_sequence': bool(getattr(self, 'deterministic_train_env_sequence', False)),
            'semi_random_hold_mode': str(getattr(self, 'semi_random_hold_mode', 'episode')),
            'semi_random_hold_episodes': int(getattr(self, 'semi_random_hold_episodes', 1)),
            'semi_random_hold_min_episodes': int(getattr(self, 'semi_random_hold_min_episodes', 1)),
            'semi_random_hold_max_episodes': int(getattr(self, 'semi_random_hold_max_episodes', 1)),
            'terrain_hold_block_index': int(getattr(self, 'current_terrain_hold_block_index', 0)),
            'terrain_hold_block_start_episode': int(getattr(self, 'current_terrain_hold_block_start_episode', 0)),
            'terrain_hold_block_length': int(getattr(self, 'current_terrain_hold_block_length', 1)),
        }
        try:
            snapshot['agent_goals'] = [
                np.asarray(getattr(getattr(agent, 'goal_a', None).state, 'p_pos', None), dtype=np.float32).copy()
                if getattr(agent, 'goal_a', None) is not None and getattr(agent.goal_a, 'state', None) is not None and getattr(agent.goal_a.state, 'p_pos', None) is not None
                else None
                for agent in getattr(self, 'agents', []) or []
            ]
        except Exception:
            snapshot['agent_goals'] = []
        return snapshot
    
    def save_fixed_positions(self, file_path):
        """
        将当前的固定位置保存到文件
        参数:
            file_path (str): 保存位置的文件路径
        """
        import json
        try:
            with open(file_path, 'w') as f:
                json.dump(self.fixed_positions, f, indent=4)
            print(f"固定位置已保存到文件: {file_path}")
            return True
        except Exception as e:
            print(f"保存固定位置到文件失败: {e}")
            return False
    
    def load_fixed_positions(self, file_path):
        """
        从文件加载固定位置
        参数:
            file_path (str): 固定位置文件路径
        返回:
            bool: 是否成功加载
        """
        import json
        try:
            if not os.path.exists(file_path):
                print(f"固定位置文件不存在: {file_path}")
                return False
                
            with open(file_path, 'r') as f:
                positions_data = json.load(f)
                
            # 验证数据格式
            if isinstance(positions_data, dict) and 'agents' in positions_data and 'goal' in positions_data:
                # 检查agents是列表且每个元素是长度为3的列表
                if not isinstance(positions_data['agents'], list):
                    raise ValueError("'agents'应该是列表")
                    
                for i, pos in enumerate(positions_data['agents']):
                    if not isinstance(pos, list) or len(pos) != 3:
                        raise ValueError(f"智能体 {i} 位置格式错误，应为长度为3的列表: {pos}")
                
                # 检查goal是长度为3的列表
                if not isinstance(positions_data['goal'], list) or len(positions_data['goal']) != 3:
                    raise ValueError(f"目标位置格式错误，应为长度为3的列表: {positions_data['goal']}")
                
                self.fixed_positions = positions_data
                print(f"成功从文件 {file_path} 加载固定位置: {len(positions_data['agents'])}个智能体")
                self.validate_and_adjust_fixed_positions()
                return True
            else:
                print(f"固定位置文件格式错误: {file_path}")
                return False
                
        except Exception as e:
            print(f"加载固定位置文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def validate_and_adjust_fixed_positions(self):
        """
        验证和调整固定位置，确保位置有效且在地形范围内
        """
        if not hasattr(self, 'fixed_positions') or self.fixed_positions is None:
            print("警告: 没有固定位置数据可以验证")
            return
            
        try:
            # 确保地形已生成
            if self.terrain is None:
                self.generate_terrain()

            current_signature = self._build_fixed_positions_validation_signature()
            if self._fixed_positions_validation_signature == current_signature:
                return
            
            altitude_offset = self._get_start_altitude_offset()
            min_air_gap = max(1.0, altitude_offset)
            
            # 验证并调整智能体位置
            if 'agents' in self.fixed_positions:
                valid_agents = []
                for i, pos in enumerate(self.fixed_positions['agents']):
                    # 转换为numpy数组以便操作
                    if not isinstance(pos, np.ndarray):
                        pos = np.array(pos, dtype=np.float32)
                    
                    # 确保位置在地图范围内
                    pos[0] = np.clip(pos[0], 2, self.map_size - 3)
                    pos[1] = np.clip(pos[1], 2, self.map_size - 3)
                    
                    # 确保高度合理
                    terrain_height = self.get_terrain_height(pos[0], pos[1])
                    required_height = terrain_height + min_air_gap
                    if pos[2] < required_height:
                        pos[2] = required_height
                    
                    valid_agents.append(pos.tolist())
                
                # 更新智能体位置
                self.fixed_positions['agents'] = valid_agents
            
            # 验证并调整目标位置
            if 'goal' in self.fixed_positions:
                goal_pos = self.fixed_positions['goal']
                if not isinstance(goal_pos, np.ndarray):
                    goal_pos = np.array(goal_pos, dtype=np.float32)
                
                goal_pos = self._adjust_fixed_goal_position_for_current_terrain(
                    goal_pos,
                    quiet_output=True,
                )
                self.fixed_positions['goal'] = goal_pos.tolist()

            self._fixed_positions_validation_signature = self._build_fixed_positions_validation_signature()
            if not _scenario_quiet_output():
                print(f"固定位置已验证并调整: {len(self.fixed_positions['agents'])}个智能体")
        except Exception as e:
            print(f"验证和调整固定位置时出错: {e}")
            import traceback
            traceback.print_exc()

    def _build_fixed_positions_validation_signature(self):
        """构造固定位置验证签名；地形和固定位置未变时可直接跳过重复校验。"""
        terrain = getattr(self, 'terrain', None)
        terrain_token = (
            id(terrain) if terrain is not None else None,
            tuple(getattr(terrain, 'shape', ())) if terrain is not None else (),
            int(getattr(self, 'map_size', 0)),
            getattr(self, 'current_terrain_seed', getattr(self, 'seed', None)),
        )

        fp = getattr(self, 'fixed_positions', None)
        if not isinstance(fp, dict):
            return (terrain_token, None, None)

        def _pos_sig(pos):
            try:
                arr = np.asarray(pos, dtype=np.float32).reshape(-1)
                return tuple(np.round(arr, 4).tolist())
            except Exception:
                return tuple()

        agents_sig = tuple(_pos_sig(pos) for pos in fp.get('agents', []))
        goal_sig = _pos_sig(fp.get('goal', None)) if fp.get('goal', None) is not None else None
        return (terrain_token, agents_sig, goal_sig)

    def _wait_for_fixed_positions(self, timeout_s: float = 5.0, interval_s: float = 0.05) -> bool:
        """
        在多进程并行环境下，非主环境用于等待主环境通过 dynamic_first_time
        生成并保存固定位置文件，然后再加载一次。
        
        设计目标：
        - 所有并行环境和主环境共享同一套固定起点/目标（同一个positions_file）
        - 仅主环境执行动态首次生成逻辑，其它环境只读取，不写入文件
        """
        file_path = getattr(self, 'fixed_positions_file', None)
        if not file_path:
            return False
        try:
            timeout_s = float(timeout_s)
            interval_s = float(interval_s)
        except Exception:
            timeout_s = 5.0
            interval_s = 0.05
        import time
        deadline = time.time() + max(0.5, timeout_s)
        while time.time() < deadline:
            try:
                if os.path.exists(file_path):
                    # load_fixed_positions 内部已经调用 validate_and_adjust_fixed_positions
                    if self.load_fixed_positions(file_path):
                        # 标记为已初始化并启用固定位置
                        self.positions_initialized = True
                        self.use_fixed_positions = True
                        return True
            except Exception:
                # 读取过程中可能遇到JSON尚未写完等问题，短暂休眠后重试
                pass
            time.sleep(interval_s)
        return False
    def _generate_terrain_legacy(self):
        """
        生成随机山脉地形 - 完全基于paper3D.m中的实现
        使用高斯分布生成自然山形
        
        🔧 改进：确保起点附近区域平坦，便于无人机正常起飞
        """
        # 创建网格 - 确保坐标系统一致
        # 使用 indexing='xy' 确保坐标顺序为 (X, Y)，与可视化系统一致
        [X, Y] = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size), indexing='xy')
        terrain = np.zeros_like(X, dtype=float)
        
        if not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
            print(f"[地形生成] 网格形状: X={X.shape}, Y={Y.shape}")
            print(f"[地形生成] 坐标范围: X=[{np.min(X):.1f}, {np.max(X):.1f}], Y=[{np.min(Y):.1f}, {np.max(Y):.1f}]")
            print(f"[地形生成] 地形数组形状: {terrain.shape}")
        
        # 🔧 确定起点区域（通常在地图角落，比如左下角SW象限）
        # 起点区域大小约为地图的15-20%，确保有足够空间
        map_size_int = int(self.map_size)
        start_area_size = int(map_size_int * 0.15)  # 起点区域大小
        start_area_margin = int(map_size_int * 0.05)  # 起点区域距离边缘的距离
        
        # 随机选择一个角落作为起点区域（保持一定的随机性，但确保是角落）
        corner_choice = self.rng.randint(0, 4)  # 0=SW, 1=SE, 2=NW, 3=NE
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
        
        if not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
            print(f"[地形生成] 起点区域: x=[{start_area_x[0]}, {start_area_x[1]}], y=[{start_area_y[0]}, {start_area_y[1]}]")
        
        # 根据复杂度等级添加山脉
        num_mountains = self.num_mountains
        
        # 预先定义mountain_width，以便在循环后仍能访问
        mountain_height = 0
        mountain_width = 0
        
        # 记录已生成的山峰位置，确保它们之间有足够的距离
        mountain_centers = []
        # 山峰之间的最小距离（可通过环境变量 MOUNTAIN_MIN_DISTANCE 调整）
        min_mountain_distance = float(os.getenv('MOUNTAIN_MIN_DISTANCE', '55'))
        max_attempts = 100  # 最大尝试次数，避免无限循环
        
        # 🔧 确保起点区域与山峰保持足够距离
        start_area_center_x = (start_area_x[0] + start_area_x[1]) / 2
        start_area_center_y = (start_area_y[0] + start_area_y[1]) / 2
        min_distance_from_start = start_area_size * 1.5  # 山峰距离起点区域的最小距离
        
        # 添加主要山脉
        for i in range(num_mountains):
            # 尝试找到一个合适的山峰位置
            placed = False
            for attempt in range(max_attempts):
                # 随机选择山脉位置、高度和宽度 - 根据复杂度等级调整
                # 避免在边缘区域生成山峰，留出20个单位的边界
                center_x = self.rng.randint(20, map_size_int - 20)
                center_y = self.rng.randint(20, map_size_int - 20)
                
                # 🔧 检查是否距离起点区域太近
                dist_from_start = np.sqrt((center_x - start_area_center_x)**2 + (center_y - start_area_center_y)**2)
                if dist_from_start < min_distance_from_start:
                    continue
                
                # 检查与已有山峰的距离
                too_close = False
                for existing_center in mountain_centers:
                    distance = np.sqrt((center_x - existing_center[0])**2 + (center_y - existing_center[1])**2)
                    if distance < min_mountain_distance:
                        too_close = True
                        break
                
                if not too_close or len(mountain_centers) == 0:
                    # 位置合适，放置山峰
                    mountain_height = self.rng.randint(self.mountain_height_range[0], self.mountain_height_range[1])
                    mountain_width = self.rng.randint(self.mountain_width_range[0], self.mountain_width_range[1])
                    
                    # 使用高斯函数创建山脉
                    mountain = mountain_height * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * mountain_width**2))
                    terrain += mountain
                    
                    # 记录山峰位置
                    mountain_centers.append((center_x, center_y, mountain_height))
                    placed = True
                    break
            
            if not placed and not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
                print(f"[地形生成] 警告：第{i+1}个山峰放置失败，已尝试{max_attempts}次")
        
        # 保存山峰中心位置供后续使用
        self.mountain_centers = mountain_centers
        
        # 使用低频噪声使地形更自然 - 与paper3D.m一致
        # 生成小尺寸随机噪声
        noise_res = 10  # 与paper3D.m一致
        small_noise = self.rng.randn(noise_res, noise_res)
        
        # 调整尺寸到地形大小
        from scipy.ndimage import zoom
        noise_factor = self.map_size / noise_res
        # 使用order=3(cubic)模拟MATLAB的bicubic插值
        low_freq_noise = zoom(small_noise, noise_factor, order=3)
        
        # 应用平滑滤波器 - 与paper3D.m的15x15均值滤波器一致
        kernel_size = 15  # 与paper3D.m一致
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        low_freq_noise = signal.convolve2d(low_freq_noise, kernel, mode='same', boundary='symm')
        
        # 将噪声添加到地形 - 根据复杂度等级调整噪声强度
        terrain += self.noise_amplitude * low_freq_noise
        
        # 添加峡谷（如果启用）
        if self.add_canyon:
            # 在地形中间位置创建一条峡谷
            canyon_width = self.rng.randint(5, 15)
            canyon_depth = self.rng.randint(20, 40)
            canyon_start = self.rng.randint(20, 40)
            canyon_end = self.rng.randint(60, 80)
            
            # 随机选择峡谷方向（水平或垂直）
            if self.rng.rand() > 0.5:  # 水平峡谷
                canyon_y = self.rng.randint(30, 70)
                map_size_int = int(self.map_size)
                for y in range(max(0, canyon_y - canyon_width), min(map_size_int, canyon_y + canyon_width)):
                    for x in range(canyon_start, canyon_end):
                        # 使用高斯形状创建平滑的峡谷边缘
                        dist_from_center = abs(y - canyon_y)
                        depth_factor = np.exp(-(dist_from_center**2) / (2 * (canyon_width/2)**2))
                        terrain[y, x] = max(0, terrain[y, x] - canyon_depth * depth_factor)
            else:  # 垂直峡谷
                canyon_x = self.rng.randint(30, 70)
                map_size_int = int(self.map_size)
                for x in range(max(0, canyon_x - canyon_width), min(map_size_int, canyon_x + canyon_width)):
                    for y in range(canyon_start, canyon_end):
                        # 使用高斯形状创建平滑的峡谷边缘
                        dist_from_center = abs(x - canyon_x)
                        depth_factor = np.exp(-(dist_from_center**2) / (2 * (canyon_width/2)**2))
                        terrain[y, x] = max(0, terrain[y, x] - canyon_depth * depth_factor)
        
        # 🔧 对起点区域进行平坦化处理，确保无人机可以正常起飞
        # 将起点区域的高度设为低值（0-5米），并使用平滑过渡
        start_flat_height = self.rng.uniform(0.0, 5.0)  # 起点区域的目标高度（0-5米）
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
                target_height = start_flat_height * blend_factor + terrain[i, j] * (1.0 - blend_factor)
                terrain[i, j] = target_height
        
        # 对起点区域进行轻微平滑，确保平坦度
        from scipy.ndimage import gaussian_filter
        smooth_region = terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)].copy()
        smooth_region = gaussian_filter(smooth_region, sigma=1.0)
        terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)] = smooth_region
        
        if not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
            start_avg_height = np.mean(terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)])
            start_height_std = np.std(terrain[start_y0:min(start_y1, map_size_int), start_x0:min(start_x1, map_size_int)])
            print(f"[地形生成] 起点区域平坦化完成: 平均高度={start_avg_height:.2f}m, 标准差={start_height_std:.2f}m")
        
        # 确保地形为正值
        terrain = np.maximum(terrain, 0)
        
        # 🔧 修复：根据复杂度等级动态调整地形高度限制
        # 复杂度等级4的山峰高度范围是80-100米，需要更高的限制
        max_terrain_height = 100 + (self.terrain_complexity_level - 1) * 20  # 等级1=100m, 等级4=160m
        terrain = np.minimum(terrain, max_terrain_height)
        
        if not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
            print(f"[地形生成] 地形高度限制: {max_terrain_height}m (复杂度等级{self.terrain_complexity_level})")
        
        # 🔧 关键修复：对地形进行降采样，确保训练地图与可视化地图完全一致
        # 使用与可视化代码相同的降采样方式（sample_rate = 4，从200×200降到50×50）
        # 保持坐标系统不变（0-200），但地形数据降采样为50×50，通过插值获取中间值
        sample_rate = 4  # 每4个点采样1个，与可视化代码保持一致
        map_size = int(self.map_size)
        # 🔧 修复：确保采样覆盖整个map_size范围（0到map_size-1）
        # 如果map_size-1不能被sample_rate整除，需要添加最后一个点
        x_samples = np.arange(0, map_size, sample_rate)
        y_samples = np.arange(0, map_size, sample_rate)
        # 确保包含最后一个点（map_size-1），以覆盖整个坐标范围
        if (map_size - 1) % sample_rate != 0:
            x_samples = np.append(x_samples, map_size - 1)
            y_samples = np.append(y_samples, map_size - 1)
        
        # 创建降采样后的地形数据（与可视化代码完全相同的逻辑）
        terrain_data_sampled = []
        for y in y_samples:
            row = []
            for x in x_samples:
                z = terrain[int(y), int(x)]
                row.append(float(z))
            terrain_data_sampled.append(row)
        terrain_data_sampled = np.array(terrain_data_sampled, dtype=np.float32)
        
        # 保存降采样后的地形数据（50×50）
        # 注意：map_size保持为200（坐标系统不变），但terrain是50×50
        self.terrain = terrain_data_sampled
        self.terrain_downsampled = True  # 标记地形已降采样
        self.terrain_sample_rate = sample_rate  # 保存降采样率，用于get_terrain_height插值
        self.X, self.Y = X, Y
        
        # 记录地形复杂度参数
        self.terrain_complexity = {
            'num_mountains': num_mountains,
            'mountain_height': mountain_height,
            'mountain_width': mountain_width,
            'noise_amplitude': self.noise_amplitude,
        }
        
        # 障碍物将在 reset_world 阶段依据起点/目标再生成
        
        # 同步地形到world对象（如果存在）
        self._sync_terrain_to_world()
        
        # 每次生成后打印种子，方便跟踪（只在主环境输出）
        suppress_output = os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1', 'true', 'yes', 'on')
        if not suppress_output:
            print(f"\n************************************************")
            print(f"*                                              *")
            print(f"*        [NEW TERRAIN GENERATED]               *")
            print(f"*        Seed: {self.seed}                     *")
            print(f"*        Mountains: {num_mountains}            *")
            print(f"*        Width: {mountain_width}               *")
            print(f"*        Height: {mountain_height}             *")
            print(f"*        Obstacles: {self.num_obstacles}       *")
            print(f"*                                              *")
            print(f"************************************************\n")
        
        return X, Y, terrain
        
    def generate_obstacles(self, start_positions=None, goal_position=None, agent_goal_positions=None):
        """生成障碍物，优先在起点和目标之间的路径上
        
        参数:
            start_positions: 智能体起点位置列表 [[x1, y1, z1], [x2, y2, z2], ...]
            goal_position: 目标位置 [x, y, z]
            agent_goal_positions: 各智能体独立目标位置列表（可选），若提供则沿每条 start→agent_goal 路径生成
        """
        self.obstacles = []
        
        # 使用设定的障碍物数量
        num_obstacles = self.num_obstacles
        
        # 如果提供了起点与目标，则在相关路径上生成障碍
        if start_positions is not None and goal_position is not None and len(start_positions) > 0:
            start_positions = [np.asarray(p, dtype=float) for p in list(start_positions)]
            goal_pos = np.asarray(goal_position, dtype=float)
            # 禁区点：所有起点与目标的 2D 坐标，障碍物中心需与此保持最小距离
            keep_out_points = [(float(p[0]), float(p[1])) for p in start_positions]
            keep_out_points.append((float(goal_pos[0]), float(goal_pos[1])))
            if agent_goal_positions is not None:
                try:
                    for g in list(agent_goal_positions):
                        ag = np.asarray(g, dtype=float)
                        keep_out_points.append((float(ag[0]), float(ag[1])))
                except Exception:
                    pass
            min_clearance = float(os.getenv('OBSTACLE_MIN_CLEARANCE_START_GOAL', '25.0'))

            def _is_clear_of_start_goal(cx, cy):
                for (ox, oy) in keep_out_points:
                    if np.hypot(cx - ox, cy - oy) < min_clearance:
                        return False
                return True

            # 中枢路径：起点均值 → 中央目标
            start_mean = np.mean(np.stack(start_positions, axis=0), axis=0)
            segments = [(start_mean, goal_pos)]
            # 如果提供各自目标，则按每个 agent 的起点→其独立目标追加路径
            if agent_goal_positions is not None:
                try:
                    ag_goals = [np.asarray(g, dtype=float) for g in list(agent_goal_positions)]
                    n = min(len(start_positions), len(ag_goals))
                    for i in range(n):
                        segments.append((start_positions[i], ag_goals[i]))
                except Exception:
                    pass

            # 分配比例：65% 路径障碍，35% 全图随机
            path_obstacles = int(num_obstacles * 0.65)
            random_obstacles = max(0, num_obstacles - path_obstacles)
            k = max(1, len(segments))
            base_per_seg = path_obstacles // k
            extra = path_obstacles - base_per_seg * k

            def _emit_ob(center_x, center_y):
                radius = self.rng.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                terrain_height = self.get_terrain_height(center_x, center_y)
                center_z = terrain_height + radius + self.obstacle_height_boost
                self.obstacles.append({'center': [float(center_x), float(center_y), float(center_z)], 'radius': int(radius)})

            # 沿每条路径投放（仅当与所有起点/目标保持 min_clearance 以上才放置）
            for si, (s, g) in enumerate(segments):
                vec = g - s
                path_len = float(np.linalg.norm(vec[:2]))
                if path_len < 1e-6:
                    continue
                unit_x, unit_y = vec[0]/path_len, vec[1]/path_len
                # 横向偏移上限：随路径长度变化，限制在[5, 20]米
                max_perp = float(np.clip(0.10 * path_len, 5.0, 20.0))
                cnt = base_per_seg + (1 if si < extra else 0)
                
                # 计算起点和终点的安全距离（避免障碍物太靠近起点或目标点）
                start_safe_distance = max(15.0, path_len * 0.15)
                end_safe_distance = max(15.0, path_len * 0.15)
                
                for _ in range(cnt):
                    t_min = start_safe_distance / path_len if path_len > 0 else 0.2
                    t_max = 1.0 - (end_safe_distance / path_len) if path_len > 0 else 0.8
                    t_min = max(0.2, min(0.4, t_min))
                    t_max = min(0.8, max(0.6, t_max))
                    placed = False
                    for _attempt in range(8):
                        t = float(self.rng.uniform(t_min, t_max))
                        cx = s[0] + t * vec[0]
                        cy = s[1] + t * vec[1]
                        off = self.rng.uniform(-max_perp, max_perp)
                        cx += -unit_y * off
                        cy +=  unit_x * off
                        cx = np.clip(cx, 5, self.map_size - 5)
                        cy = np.clip(cy, 5, self.map_size - 5)
                        if _is_clear_of_start_goal(cx, cy):
                            _emit_ob(cx, cy)
                            placed = True
                            break
                    if not placed:
                        pass  # 跳过该障碍，避免压在起点/目标附近

            # 其余随机分布（全图），且与所有起点/目标保持 min_clearance 以上
            low, high = int(self.map_size * 0.05), int(self.map_size * 0.95)
            for _ in range(random_obstacles):
                placed = False
                for _attempt in range(50):
                    cx = self.rng.randint(low, max(low+1, high))
                    cy = self.rng.randint(low, max(low+1, high))
                    if _is_clear_of_start_goal(cx, cy):
                        _emit_ob(cx, cy)
                        placed = True
                        break
                if not placed:
                    pass  # 跳过该随机障碍
        
        else:
            # 回退策略：按 map_size 的百分比区域分布，避免硬编码 10..90 导致偏边
            m = float(self.map_size)
            def _r(lo_x, hi_x, lo_y, hi_y):
                return {
                    "min_x": int(m * lo_x), "max_x": int(m * hi_x),
                    "min_y": int(m * lo_y), "max_y": int(m * hi_y),
                }
            regions = [
                _r(0.05, 0.25, 0.05, 0.95),  # 左侧
                _r(0.75, 0.95, 0.05, 0.95),  # 右侧
                _r(0.35, 0.65, 0.75, 0.95),  # 上方
                _r(0.35, 0.65, 0.05, 0.25),  # 下方
                _r(0.35, 0.65, 0.35, 0.65),  # 中央
            ]
            
            obstacles_per_region = max(1, num_obstacles // len(regions))
            remaining_obstacles = num_obstacles % len(regions)
            
            for i, region in enumerate(regions):
                region_obstacles = obstacles_per_region
                if i < remaining_obstacles:
                    region_obstacles += 1
                    
                for j in range(region_obstacles):
                    center_x = self.rng.randint(region["min_x"], max(region["min_x"]+1, region["max_x"]))
                    center_y = self.rng.randint(region["min_y"], max(region["min_y"]+1, region["max_y"]))
                    
                    radius = self.rng.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                    terrain_height = self.get_terrain_height(center_x, center_y)
                    center_z = terrain_height + radius + self.obstacle_height_boost
                    
                    self.obstacles.append({'center': [center_x, center_y, center_z], 'radius': radius})
                    
                    if len(self.obstacles) <= 3:
                        print(f"🏗️ 生成障碍物 {len(self.obstacles)}: 位置=({center_x},{center_y},{center_z:.1f}), 半径={radius}, 地形高度={terrain_height:.1f}")
        
        self._refresh_observation_static_cache()
        return self.obstacles
        
    def make_world(self):
        world = World()
        world.dim_p = 3  # 3D环境
        
        # 创建智能体
        num_agents = 3
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = float(getattr(self, 'agent_size', 0.5))
            # 确保地形文件生成之后再进行属性设置
            if hasattr(agent, 'max_speed'):
                agent.max_speed = 25  # 原始值是1.0，调整到1.2
            # 设置更大的加速度值，增强智能体移动能力
            agent.accel = 8.5  # 增大加速度系数，原先默认为5.0
            if hasattr(agent, 'color'):
                agent.color = np.array([0.35, 0.35, 0.85])
            
            # 🔧 四旋翼动力学：通过环境变量启用
            # 设置 USE_QUADROTOR_DYNAMICS=1 来启用四旋翼动力学模型
            import os
            use_quadrotor = os.getenv('USE_QUADROTOR_DYNAMICS', '0').lower() in ('1', 'true', 'yes', 'on')
            agent.use_quadrotor_dynamics = use_quadrotor
        
        # 创建一个中央地标（对智能体不可见，仅用于计算），并为每个智能体创建目标
        world.landmarks = [Landmark()]  # Central landmark
        center_goal = world.landmarks[0]
        center_goal.name = 'center_goal'
        center_goal.collide = False
        center_goal.movable = False
        # 为可视化与外部查询提供直接引用占位
        world.goal_pos = None
        world.agent_goals = []

        # 为每个智能体创建可见的目标地标
        for i, agent in enumerate(world.agents):
            goal = Landmark()
            goal.name = f'agent_goal_{i}'
            goal.collide = False
            goal.movable = False
            goal.size = 2.0  # 成功半径
            if hasattr(goal, 'color'):
                goal.color = np.array([0.15, 0.65, 0.15]) # Green
            world.landmarks.append(goal)
            agent.goal_a = goal # 将目标分配给智能体

        # 创建障碍物 - 使用球形障碍物，并将其存放在一个专门的列表中
        world.obstacles = [] # 创建专门的障碍物列表
        # 与复杂度等级保持一致：障碍物对象数量 = 计划生成数量
        num_obstacles = int(getattr(self, 'num_obstacles', 6))
        for i in range(num_obstacles):
            obstacle = Landmark()
            obstacle.name = f'obstacle_{i}'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.15
            if hasattr(obstacle, 'color'):
                obstacle.color = np.array([0.75, 0.25, 0.25])  # 更红的颜色，更明显
            world.landmarks.append(obstacle)
            world.obstacles.append(obstacle) # 添加到专门的列表
        
        # 确保地形已经生成，如果还没有，则生成
        if self.terrain is None:
            if not (os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1','true','yes','on')):
                print("\n==========================================")
                print(f"在make_world中初始化地形（这是预期行为）")
                print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"当前种子: {self.seed}")
                print("==========================================\n")
            
            # 如果是随机地形模式，每次都重新生成，否则只在首次生成
            if hasattr(self, 'random_terrain') and self.random_terrain:
                # 使用当前时间戳作为种子确保每次都不同
                new_seed = int(time.time()) % 100000
                self.seed = new_seed
                self.rng = np.random.RandomState(new_seed)
                print(f"随机地形模式：使用新种子 {new_seed}")
            else:
                # 使用固定种子确保地形一致
                if self.seed is None:
                    self.seed = 42
                    self.rng = np.random.RandomState(self.seed)
                    print(f"固定地形模式：使用默认种子 {self.seed}")
            
            self.generate_terrain()
            
        # 注意：不要在 make_world() 阶段生成障碍物。
        # 正确顺序必须是：
        #   1. reset_world() 先确定起点与目标
        #   2. 再按当前布局生成障碍物
        # 否则固定障碍物模式会把“无起点/目标 keep-out”的旧障碍物错误复用到正式训练中。
        
        # 基本位置初始化，避免None值导致的错误
        # 注意：这里只做最基本的初始化，具体状态由reset_world设置
        for agent in world.agents:
            if not hasattr(agent.state, 'p_pos') or agent.state.p_pos is None:
                agent.state.p_pos = np.zeros(world.dim_p)
            if not hasattr(agent.state, 'p_vel') or agent.state.p_vel is None:
                agent.state.p_vel = np.zeros(world.dim_p)
        
        # 确保所有地标都有一个默认位置，防止None错误
        for landmark in world.landmarks:
            if not hasattr(landmark.state, 'p_pos') or landmark.state.p_pos is None:
                landmark.state.p_pos = np.zeros(world.dim_p)
        
        return world
        
    def _apply_role_randomization(self, world):
        """
        🚀 角色随机化 (Index Shuffling)：
        在分配完起始位置和目标后，随机打乱各智能体的物理状态映射。
        打破网络对特定索引(Agent 2)与特定出生位置/行为的死板映射，防止某个智能体固化为"逃跑/献祭"角色。
        """
        import os
        # 默认开启，可通过环境变量关闭
        enable_role_shuffle = os.getenv('ENABLE_ROLE_SHUFFLE', '1').lower() in ('1', 'true', 'yes', 'on')
        if enable_role_shuffle and len(world.agents) > 1:
            # 临时保存所有智能体当前的物理状态和目标
            states = []
            for agent in world.agents:
                states.append({
                    'p_pos': agent.state.p_pos.copy() if agent.state.p_pos is not None else np.zeros(3),
                    'p_vel': agent.state.p_vel.copy() if agent.state.p_vel is not None else np.zeros(3),
                    'initial_velocity_hint': getattr(agent, 'initial_velocity_hint', np.zeros(3)).copy(),
                    'goal_pos': agent.goal_a.state.p_pos.copy() if hasattr(agent, 'goal_a') and agent.goal_a is not None else None,
                    'last_goal_dist': getattr(agent, 'last_goal_dist', None)
                })
            
            # 使用场景的随机数生成器打乱索引，确保如果种子固定，打乱也是确定的
            shuffle_indices = list(range(len(world.agents)))
            self.rng.shuffle(shuffle_indices)
            
            # 重新分配给各个智能体（网络 0 可能控制了状态 2）
            for i, agent in enumerate(world.agents):
                idx = shuffle_indices[i]
                agent.state.p_pos = states[idx]['p_pos']
                agent.state.p_vel = states[idx]['p_vel']
                agent.initial_velocity_hint = states[idx]['initial_velocity_hint']
                
                # 必须连同目标一起交换，否则智能体的轨迹会交叉并导致碰撞！
                if states[idx]['goal_pos'] is not None and hasattr(agent, 'goal_a') and agent.goal_a is not None:
                    agent.goal_a.state.p_pos = states[idx]['goal_pos']
                
                agent.last_goal_dist = states[idx]['last_goal_dist']
                
                # 同步更新 world 级别的可视化记录
                if hasattr(world, 'agent_goals') and i < len(world.agent_goals):
                    world.agent_goals[i] = states[idx]['goal_pos']
                
                # 如果有观察缓存，也要清空或更新，防止第一步读取到错误的旧位置观察值
                if hasattr(self, 'observation_cache'):
                    try:
                        self.observation_cache[id(agent)] = self.observation(agent, world)
                    except Exception:
                        pass

    def reset_world(self, world):
        """
        重置世界状态
        初始化智能体和目标位置
        """
        self._obs_step_cache_key = None
        self._obs_step_cache = {}
        # 根据环境变量启用软复位：当检测到穿透/落地时，将智能体抬到地表以上，而不提前结束回合
        try:
            import os as _os
            world.enable_collision_autoreset = _os.getenv('ENABLE_COLLISION_AUTORESET', '1').lower() in ('1','true','yes','on')
        except Exception:
            world.enable_collision_autoreset = True
        # 通知子类/组件：回合开始（用于ARW等自适应模块）
        try:
            if hasattr(self, 'on_episode_start') and callable(getattr(self, 'on_episode_start')):
                self.on_episode_start()
        except Exception:
            pass
        self.current_episode_index, self.current_episode_env_id = self._resolve_episode_context(world)
        self.current_episode_obstacle_seed = None
        if self._use_deterministic_train_env_sequence():
            self.current_episode_rng_seed = self._make_deterministic_episode_seed(
                'episode_rng',
                self.current_episode_index,
                self.current_episode_env_id,
            )
            self.rng = np.random.RandomState(int(self.current_episode_rng_seed))
        else:
            self.current_episode_rng_seed = None
        # print("\n=== 重置世界状态 ===")
        # print(f"use_fixed_positions: {self.use_fixed_positions}")
        # print(f"random_z0_positions: {hasattr(self, 'random_z0_positions') and self.random_z0_positions}")
        # print(f"dynamic_first_time: {hasattr(self, 'dynamic_first_time') and self.dynamic_first_time}")
        
        # 清空智能体完成日志标记（避免跨回合残留）
        self._agent_done_logged = {}
        self._refresh_observation_static_cache(num_agents=len(world.agents))
        
        # 重置各智能体已访问区域
        self.visited_cells = {i: set() for i in range(len(world.agents))}
        
        # 为每个智能体初始化能量相关属性
        for agent in world.agents:
            agent.total_energy = 3000.0  # 初始总能量
            agent.energy_consumed = 0.0  # 已消耗能量
            agent.last_velocity = np.zeros(3)  # 上一步速度，用于计算加速度
            
            # 确保加速度值被显式重置为零
            if hasattr(agent.state, 'p_vel'):
                agent.state.p_vel = np.zeros(3)  # 确保速度为零
            if hasattr(agent, 'action') and hasattr(agent.action, 'u'):
                agent.action.u = np.zeros(world.dim_p)  # 显式重置加速度为零
            
            agent.last_goal_dist = None  # 上一步到目标的距离
            agent.debug_info = {}  # 调试信息
            agent.debug_info['total_penetration_count'] = 0
            agent.debug_info['terrain_penetration_count'] = 0  # 地形穿透次数（仅显示用）
            agent.debug_info['obstacle_collision_count'] = 0   # 球形障碍碰撞次数（仅显示用）
            agent.debug_info['last_pos'] = agent.state.p_pos.copy() if hasattr(agent.state, 'p_pos') else None
            agent.debug_info['stationary_count'] = 0
            # 🚨 关键修复：重置碰撞计数相关标志
            agent._last_collision_counted_step = -1
            agent._last_collision_position = None
            # 统一 episode 级碰撞语义：本回合是否发生过任何碰撞/穿透（不受去重计数影响）
            agent._episode_has_collision = False
            # 兼容旧逻辑：回合开始清零碰撞/穿透标志
            agent._had_penetration_or_collision = False
            agent._had_obstacle_collision = False
            agent._had_terrain_contact_or_penetration = False
            # 确保奖励相关状态在每回合重置
            agent.last_position = agent.state.p_pos.copy()
            agent.stationary_count = 0
            agent.initialized_for_reward = False
            
            # 🔧 关键修复：每回合重置成功状态标志
            # 问题：如果不重置，智能体在首次到达目标后，后续回合将无法再获得一次性成功奖励
            if hasattr(agent, '_success_state'):
                agent._success_state = {
                    'success_reward_given': False,
                    'first_success_step': None,
                    'hover_reward_count': 0
                }
            
            # 🔧 同时重置碰撞减少奖励的相关状态
            if hasattr(agent, 'current_episode_collision_count'):
                # 将当前回合碰撞计数存为上一回合的值
                agent.previous_episode_collision_count = agent.current_episode_collision_count
                # 重置当前回合碰撞计数
                agent.current_episode_collision_count = 0
            if hasattr(agent, 'collision_reduction_reward_given'):
                agent.collision_reduction_reward_given = False

        # 重新生成地形（如果设置了随机地形）
        if hasattr(self, 'random_terrain') and self.random_terrain:
            try:
                if getattr(world, 'is_main_env', True) and not _scenario_quiet_output():
                    print("启用随机地形，重新生成")
            except Exception:
                pass
            # 🔧 关键修复：regenerate_terrain() 内部已经处理了地形变化时的位置重置
            # 当地形种子变化时，会自动重置 positions_initialized = False
            terrain_seed_override = None
            terrain_variant_seed_override = None
            if self._use_deterministic_train_env_sequence():
                if bool(getattr(self, 'use_semi_random_terrain', False)):
                    terrain_variant_seed_override = self._make_deterministic_terrain_variant_seed(
                        self.current_episode_index,
                        self.current_episode_env_id,
                    )
                else:
                    terrain_seed_override = self._make_deterministic_episode_seed(
                        'terrain',
                        self.current_episode_index,
                        self.current_episode_env_id,
                    )
            self.regenerate_terrain(
                new_seed=terrain_seed_override,
                variant_seed=terrain_variant_seed_override,
            )
        
        # 🚨 关键修复：只有在use_fixed_positions=True时才加载固定位置文件
        # 原因：即使USE_FIXED_POSITIONS=0，如果fixed_positions_file存在，代码也会加载固定位置
        # 🚨 关键修复：在reset_world时强制确认固定位置配置
        # 问题：dynamic_first_time逻辑在多进程环境中失效，导致每个进程独立重置positions_initialized
        # 解决：始终检查_initial_use_fixed_positions（初始配置），确保从文件加载
        
        if getattr(self, '_initial_use_fixed_positions', False):  # 使用初始配置而不是运行时配置
            # 强制启用固定位置（忽略运行时的use_fixed_positions状态）
            self.use_fixed_positions = True
            # 如果还没加载固定位置，尝试从文件加载
            if self.fixed_positions is None and self.fixed_positions_file:
                if os.path.exists(self.fixed_positions_file):
                    if self.load_fixed_positions(self.fixed_positions_file):
                        self.positions_initialized = True
        else:
            # 如果初始配置就是False，确保不使用固定位置
            self.use_fixed_positions = False
            self.fixed_positions = None
            self.positions_initialized = False

        # 🔧 新增：在多进程并行环境下，非主环境等待主环境生成并保存固定位置文件
        # 仅当用户启用了固定位置、当前环境没有fixed_positions、并且提供了文件路径时才等待
        try:
            if (getattr(self, 'use_fixed_positions', False)
                and self.fixed_positions is None
                and getattr(self, 'fixed_positions_file', None)
                and hasattr(world, 'is_main_env')
                and not getattr(world, 'is_main_env')):
                self._wait_for_fixed_positions()
        except Exception:
            pass
        
        # 处理首次动态后续固定模式（针对当前地图的首次，仅主环境执行生成逻辑）
        # 🔧 关键修复：DYNAMIC_FIRST_TIME 的正确逻辑：
        # 1. 每次运行（新启动训练脚本）时，dynamic_first_time=True 且 positions_initialized=False
        # 2. 第一次 reset_world 时，动态生成位置并保存到文件
        # 3. 同一次运行的后续回合（episode 2, 3, ...）读取固定位置文件
        # 4. 下次运行时，删除或忽略旧的位置文件，重新动态生成新的初始位置
        # 
        # 修复方案：positions_initialized 应该在每次新运行时重置，而不是持久化
        # 因此我们不再检查 positions_initialized，而是检查固定位置文件是否存在
        if hasattr(self, 'dynamic_first_time') and hasattr(self, 'positions_initialized'):
            is_main_env = getattr(world, 'is_main_env', True)
            # 🚨 关键修复：每次运行时都应该重新生成位置（第一个回合）
            # 条件：dynamic_first_time=True 且 固定位置尚未加载（fixed_positions为None）
            should_generate_new_positions = (
                self.dynamic_first_time and 
                not self.positions_initialized and 
                is_main_env and
                self.fixed_positions is None  # 只有当固定位置为空时才生成
            )
            if should_generate_new_positions:
                try:
                    if getattr(world, 'is_main_env', True):
                        print(f"检测到当前地图首次使用动态位置 (地形种子: {getattr(self, 'current_terrain_seed', self.seed)})，后续将使用固定位置")
                except Exception:
                    pass
                # 第一次重置时使用动态位置设置
                self._dynamic_reset_world(world)
                
                # 保存动态生成的位置，用于后续固定使用
                agent_positions = []
                # 保存智能体位置
                for agent in world.agents:
                    agent_positions.append(agent.state.p_pos.tolist())
                
                # 🚨 关键修复：确保目标位置的Z坐标已经根据地形高度正确设置（地形高度 + goal_altitude）
                # 这样保存到文件中的Z坐标就是正确的，后续所有回合都使用这个Z坐标，不再调整
                goal_pos_to_save = self.goal_pos.copy() if self.goal_pos is not None else np.array([50.0, 50.0, 50.0])
                if self.goal_pos is not None:
                    # 确保目标位置在地形上方（地形高度 + goal_altitude）
                    goal_terrain_h = self.get_terrain_height(goal_pos_to_save[0], goal_pos_to_save[1])
                    goal_altitude = self._get_goal_altitude()
                    required_goal_height = goal_terrain_h + goal_altitude
                    if goal_pos_to_save[2] < required_goal_height:
                        goal_pos_to_save[2] = required_goal_height
                        print(f"[目标位置设置] 首次生成时调整目标Z坐标到地形上方: {goal_pos_to_save[2]:.2f} (地形高度={goal_terrain_h:.2f}, goal_altitude={goal_altitude:.2f})")
                    # 更新self.goal_pos，确保后续使用正确的Z坐标
                    self.goal_pos = goal_pos_to_save.copy()
                
                # 使用新格式保存位置（Z坐标已经正确设置）
                self.fixed_positions = {
                    'agents': agent_positions,
                    'goal': goal_pos_to_save.tolist()
                }

                # 🚨 关键修复：只有在初始化时use_fixed_positions=True时才保存位置到文件
                # 原因：即使USE_FIXED_POSITIONS=0，dynamic_first_time也会保存位置文件
                # 这导致后续回合可能加载固定位置
                initial_use_fixed = getattr(self, '_initial_use_fixed_positions', False)
                if initial_use_fixed and hasattr(self, 'fixed_positions_file') and self.fixed_positions_file:
                    try:
                        if getattr(world, 'is_main_env', True):
                            print(f"保存第一次生成的位置到文件: {self.fixed_positions_file}")
                    except Exception:
                        pass
                    self.save_fixed_positions(self.fixed_positions_file)

                # 🚨 关键修复：在保存文件后，不再调用validate_and_adjust_fixed_positions调整目标位置
                # 原因：目标位置的Z坐标已经在上面正确设置（地形高度 + goal_altitude），
                # 如果再次调整，可能会导致不一致（因为地形高度计算可能有微小差异）
                # 解决方案：只验证智能体位置，不调整目标位置
                # self.validate_and_adjust_fixed_positions()  # 不再调用，避免调整目标位置
                
                # 🔧 只验证智能体位置，确保在地形上方（不影响目标位置）
                if 'agents' in self.fixed_positions:
                    altitude_offset = self._get_start_altitude_offset()
                    min_air_gap = max(1.0, altitude_offset)
                    valid_agents = []
                    for i, pos in enumerate(self.fixed_positions['agents']):
                        if not isinstance(pos, np.ndarray):
                            pos = np.array(pos, dtype=np.float32)
                        terrain_height = self.get_terrain_height(pos[0], pos[1])
                        required_height = terrain_height + min_air_gap
                        if pos[2] < required_height:
                            pos[2] = required_height
                        valid_agents.append(pos.tolist())
                    self.fixed_positions['agents'] = valid_agents
                
                # 标记位置已初始化（针对当前地图）
                # 🔧 关键修复：positions_initialized 现在绑定到当前地形种子
                # 当地形变化时，regenerate_terrain() 会重置此标记
                self.positions_initialized = True
                # 🚨 关键修复：只有在初始化时use_fixed_positions=True时，才设置use_fixed_positions=True
                # 原因：即使USE_FIXED_POSITIONS=0，dynamic_first_time也会强制设置use_fixed_positions=True
                # 这导致第一回合后所有回合都使用相同的固定位置，目标位置不会改变
                # 修复：只有在初始化时use_fixed_positions=True时，才在dynamic_first_time模式下设置use_fixed_positions=True
                initial_use_fixed = getattr(self, '_initial_use_fixed_positions', False)
                if initial_use_fixed:
                    self.use_fixed_positions = True
                else:
                    # 如果初始化时use_fixed_positions=False，即使dynamic_first_time=True，也不使用固定位置
                    self.use_fixed_positions = False
                    self.fixed_positions = None  # 清除固定位置，确保后续回合动态生成
                
                # 确保在动态初始化后，各智能体的last_goal_dist也被正确初始化
                for i, agent in enumerate(world.agents):
                    if agent.last_goal_dist is None and self.goal_pos is not None:
                        agent.last_goal_dist = np.linalg.norm(agent.state.p_pos - self.goal_pos)
                
                # 🚀 插入角色随机化 (动态首次生成时)
                self._apply_role_randomization(world)
                
                # 标记reset已完成，允许调试信息打印
                world._reset_completed = True
                
                # 创建观察值缓存
                observation_cache = {}
                for i, agent in enumerate(world.agents):
                    try:
                        obs = self.observation(agent, world)
                        observation_cache[id(agent)] = obs
                    except Exception as e:
                        observation_cache[id(agent)] = np.zeros(self.observation_dim, dtype=np.float32)

                # 存储观察值缓存，供环境的reset函数使用
                self.observation_cache = observation_cache

                # 起飞前保护：记录起始位置并抬升初始Z至地形+阈值，避免首回合即穿透
                try:
                    for agent in world.agents:
                        agent.start_position = agent.state.p_pos.copy()
                        try:
                            airborne_thr = float(getattr(world, 'pre_takeoff_airborne_threshold', 0.5))
                        except Exception:
                            airborne_thr = 0.5
                        try:
                            terrain_h = self.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
                        except Exception:
                            terrain_h = 0.0
                        min_z = float(terrain_h) + float(airborne_thr)
                        if agent.state.p_pos[2] < min_z:
                            agent.state.p_pos[2] = min_z
                except Exception:
                    pass

                # 🚨 修复：检查fixed_positions是否成功创建
                # 注意：如果initial_use_fixed=False，fixed_positions会被设置为None，这是正常行为
                initial_use_fixed = getattr(self, '_initial_use_fixed_positions', False)
                if initial_use_fixed:
                    # 只有在启用固定位置时才检查fixed_positions
                    if self.fixed_positions is not None and 'agents' in self.fixed_positions and 'goal' in self.fixed_positions:
                        if not _scenario_quiet_output():
                            print(f"已保存动态生成的位置: {len(self.fixed_positions['agents'])}个智能体, 目标位置: {self.fixed_positions['goal']}")
                    else:
                        if not _scenario_quiet_output():
                            print(f"⚠️  动态位置生成失败或未创建fixed_positions")
                else:
                    # 如果未启用固定位置，fixed_positions为None是正常的
                    if self.goal_pos is not None:
                        if not _scenario_quiet_output():
                            print(f"动态位置生成完成: {len(world.agents)}个智能体, 目标位置: {self.goal_pos}")
                    else:
                        print(f"⚠️  动态位置生成完成，但目标位置未设置")
                return
        
        # 如果设置了使用固定位置且有已保存的位置，这是优先级最高的设置
        if self.use_fixed_positions and hasattr(self, 'fixed_positions') and self.fixed_positions is not None:
            # print("使用预定义的固定位置")
            if isinstance(self.fixed_positions, dict) and 'agents' in self.fixed_positions:
                # print(f"固定位置格式: 字典, 包含 {len(self.fixed_positions['agents'])} 个智能体")
                # 打印前几个智能体位置作为示例
                # for i, pos in enumerate(self.fixed_positions['agents'][:3]):
                #     print(f"  智能体{i}位置: {pos}")
                # if 'goal' in self.fixed_positions:
                #     print(f"  目标位置: {self.fixed_positions['goal']}")
                pass
            elif isinstance(self.fixed_positions, list):
                # print(f"固定位置格式: 列表, 长度为 {len(self.fixed_positions)}")
                pass
                
            self._apply_fixed_positions(world)
            # 同地形/同起点/同目标基础上，为每个并行环境引入独立的初始朝向/速度微扰
            self._apply_per_env_randomization(world)
            
            # 固定位置模式下也必须按“当前固定起点/目标”刷新障碍物。
            # 这里统一走布局签名判断，避免复用 make_world() 或旧回合残留的错误障碍物。
            try:
                agent_positions_now = [agent.state.p_pos for agent in world.agents]
                agent_goals_now = [agent.goal_a.state.p_pos for agent in world.agents] if hasattr(world, 'agents') else None
                self._refresh_obstacles_for_current_layout(
                    world,
                    start_positions=agent_positions_now,
                    goal_position=self.goal_pos,
                    agent_goal_positions=agent_goals_now,
                )
            except Exception:
                pass
            
            # 🚀 插入角色随机化 (固定位置重置时)
            self._apply_role_randomization(world)
            
            # print("=== 重置完成 ===\n")
            return
        
        # 如果没有启用固定位置或固定位置不可用，使用动态位置设置
        # 根据DEBUG_ENV_OUTPUT环境变量控制输出
        quiet_output = os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')
        debug_mode = int(os.getenv('DEBUG_ENV_OUTPUT', '0'))
        should_output = False
        
        if quiet_output:
            should_output = False
        elif debug_mode == 0:  # 仅主环境输出
            should_output = hasattr(world, 'is_main_env') and world.is_main_env
        elif debug_mode == 1:  # 所有环境都输出
            should_output = True
        elif debug_mode == 2:  # 仅错误时输出
            should_output = False
        
        if should_output:
            print("没有启用固定位置或无可用的固定位置数据，使用动态设置")
        self._dynamic_reset_world(world)
        # 同地形/同起点/同目标基础上，为每个并行环境引入独立的初始朝向/速度微扰
        self._apply_per_env_randomization(world)
        
        # 确保在动态重置后，所有智能体的last_goal_dist都被正确初始化
        for agent in world.agents:
            if self.goal_pos is not None:
                agent.last_goal_dist = np.linalg.norm(agent.state.p_pos - self.goal_pos)
                
        # 🚀 插入角色随机化 (完全动态位置生成时)
        self._apply_role_randomization(world)
        
        # 根据DEBUG_ENV_OUTPUT环境变量控制完成信息输出
        debug_mode = int(os.getenv('DEBUG_ENV_OUTPUT', '0'))
        should_output = False
        
        if debug_mode == 0:  # 仅主环境输出
            should_output = hasattr(world, 'is_main_env') and world.is_main_env
        elif debug_mode == 1:  # 所有环境都输出
            should_output = True
        elif debug_mode == 2:  # 仅错误时输出
            should_output = False
        
        if should_output and not _scenario_quiet_output():
            # 输出各智能体的初始位置坐标信息
            print(f"[智能体位置] 智能体初始位置坐标:")
            for i, agent in enumerate(world.agents):
                pos = agent.state.p_pos
                terrain_h = self.get_terrain_height(pos[0], pos[1])
                height_above_terrain = pos[2] - terrain_h
                print(f"  Agent{i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | terrain_h={terrain_h:.2f} | 离地高度={height_above_terrain:.2f}m")
            print("=== 重置完成 ===\n")

    def _apply_per_env_randomization(self, world):
        """在不改变地形/起点/目标的前提下，为每个并行环境注入独立的初始朝向/速度微扰。
        - 依据 world.env_id 生成独立随机流，保证并行环境去相关
        - 仅对初始速度与轻微水平朝向做扰动，不改变初始位置与目标
        """
        try:
            env_id = int(getattr(world, 'env_id', 0))
        except Exception:
            env_id = 0
        # 组合种子：场景种子 + 环境索引，确保并行环境不同扰动但可复现
        seed_val = getattr(self, 'seed', 42)
        base_seed = int(seed_val) if seed_val is not None else 42
        rng = np.random.RandomState((base_seed + 10007 * env_id) % 2147483647)
        # 速度微扰幅度（可通过环境变量控制），默认较小
        try:
            import os
            vel_jitter_max = float(os.getenv('INIT_VEL_JITTER_MAX', '0.3'))  # m/s
        except Exception:
            vel_jitter_max = 0.3
        for agent in world.agents:
            # 水平面随机方向
            theta = rng.uniform(0.0, 2.0 * np.pi)
            speed = rng.uniform(0.0, vel_jitter_max)
            vx = np.cos(theta) * speed
            vy = np.sin(theta) * speed
            # 轻微竖直分量，避免扎根
            vz = rng.uniform(-0.05, 0.05)
            if hasattr(agent, 'state') and hasattr(agent.state, 'p_vel') and isinstance(agent.state.p_vel, np.ndarray):
                agent.state.p_vel[...] = np.array([vx, vy, vz], dtype=float)
            # 记录初始朝向（用于奖励中的方向一致性等）
            try:
                agent.initial_velocity_hint = np.array([vx, vy, vz], dtype=float)
            except Exception:
                pass
    
    def _apply_fixed_positions(self, world):
        """应用固定位置设置"""
        try:
            # 🔧 修复：确保地形已经生成，否则强制生成
            quiet_output = os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')
            debug_setup_info = os.getenv('DEBUG_SETUP_INFO', '1').lower() in ('1', 'true', 'yes', 'on')
            if self.terrain is None:
                try:
                    suppress_output = os.getenv('SUPPRESS_TERRAIN_OUTPUT', '0').lower() in ('1', 'true', 'yes', 'on')
                    if debug_setup_info and not suppress_output:
                        print("[警告] 应用固定位置时地形未生成，正在生成地形...")
                except Exception:
                    pass
                self.generate_terrain()
            
            if self.fixed_positions is not None:
                self.validate_and_adjust_fixed_positions()
            
            # 🔧 修复：获取离地高度配置（优先使用环境变量，否则使用默认值12.0米）
            altitude_offset = self._get_start_altitude_offset()
            min_air_gap = max(1.0, altitude_offset)
            
            # 新格式 {'agents': [...], 'goal': [...]}
            if isinstance(self.fixed_positions, dict) and 'agents' in self.fixed_positions:
                # 🚨 关键修复：确保所有智能体使用相同的固定位置（评估时不应该有差异）
                # 问题：如果fixed_positions['agents']中的位置数量少于智能体数量，会导致部分智能体使用随机位置
                # 解决：检查并确保所有智能体都使用固定位置，如果位置不足则使用第一个位置
                fixed_agents_pos = self.fixed_positions['agents']
                if len(fixed_agents_pos) == 0:
                    if debug_setup_info:
                        print(f"⚠️  警告: fixed_positions['agents']为空，无法应用固定位置")
                    return
                
                # 设置智能体位置
                for i, agent in enumerate(world.agents):
                    # 🚨 关键修复：如果固定位置数量不足，使用第一个位置（确保所有智能体起点一致）
                    if i < len(fixed_agents_pos):
                        pos_source = fixed_agents_pos[i]
                    else:
                        # 如果位置不足，使用第一个位置（确保所有智能体起点一致）
                        pos_source = fixed_agents_pos[0]
                        if i == len(fixed_agents_pos):  # 只在第一次遇到不足时打印警告
                            if debug_setup_info:
                                print(f"⚠️  警告: 固定位置数量({len(fixed_agents_pos)})少于智能体数量({len(world.agents)})，智能体{i}及之后将使用第一个固定位置")
                    
                    # 确保位置是numpy数组，并复制以避免修改原始数据
                    pos = np.array(pos_source, dtype=float).copy()
                    
                    # 🔧 修复：确保X、Y坐标不被修改（固定起点必须保持X、Y不变）
                    # 只根据当前地形的实际高度重新计算Z坐标，避免地形变化后位置在地形下方
                    current_terrain_h = self.get_terrain_height(pos[0], pos[1])
                    
                    # 🔧 修复：验证地形高度是否有效，如果为0且地形已生成，可能是计算错误
                    if current_terrain_h == 0.0 and self.terrain is not None:
                        # 尝试使用最近的地形点
                        # 🔧 关键修复：使用get_terrain_height方法，自动处理降采样后的坐标映射
                        x_int = int(np.clip(pos[0], 0, self.map_size - 1))
                        y_int = int(np.clip(pos[1], 0, self.map_size - 1))
                        current_terrain_h = float(self.get_terrain_height(x_int, y_int))
                    
                    # 只有当启用随机Z高度时才随机化Z坐标
                    if hasattr(self, 'random_z0_positions') and self.random_z0_positions:
                        random_height = current_terrain_h + 2 + np.random.uniform(0, 5)
                        pos[2] = random_height
                    else:
                        # 🚨 关键修复：必须始终根据当前地形高度调整Z坐标，确保智能体在地形上方
                        # 原因：
                        # 1. 固定位置文件中的Z坐标是基于生成位置时的地形高度
                        # 2. 评估时可能使用不同的地形种子，导致地形高度不同
                        # 3. 如果保留文件中的Z坐标，可能导致智能体在地形下方（如Agent 2的Z=10.97，但地形高度更高）
                        # 解决方案：始终根据当前地形高度调整Z坐标，确保在地形上方（地形高度 + min_air_gap）
                        final_terrain_h = self.get_terrain_height(pos[0], pos[1])
                        required_height = final_terrain_h + min_air_gap
                        old_z = pos[2]
                        if pos[2] < required_height:
                            # Z坐标太低，必须调整到地形上方
                            pos[2] = required_height
                            if abs(old_z - pos[2]) > 1e-6 and i < 3 and debug_setup_info:  # 只打印前3个智能体的调整信息
                                print(f"⚠️  [固定位置调整] Agent{i}: Z坐标从{old_z:.2f}调整到{pos[2]:.2f}（地形高度={final_terrain_h:.2f}, 要求高度={required_height:.2f}）")
                        # 🚨 关键修复：即使Z坐标已经在地形上方，也要确保有足够的安全间隙
                        # 如果Z坐标刚好在地形上方但间隙太小（< min_air_gap），也要调整
                        elif pos[2] < final_terrain_h + min_air_gap * 1.5:
                            # 间隙太小，调整到更安全的高度
                            pos[2] = final_terrain_h + min_air_gap * 1.5
                            if abs(old_z - pos[2]) > 1e-6 and i < 3 and not quiet_output:
                                print(f"🔧 [固定位置优化] Agent{i}: Z坐标从{old_z:.2f}优化到{pos[2]:.2f}（增加安全间隙，地形高度={final_terrain_h:.2f}）")
                    
                    agent.state.p_pos = pos
                    agent.state.p_vel = np.zeros(world.dim_p)
                    agent.state.c = np.zeros(world.dim_c)
                    
                    # 🔧 调试信息：打印前3个智能体的实际位置，验证是否一致
                    if i < 3 and not quiet_output:
                        print(f"[固定位置验证] Agent{i}: 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] (来源: {'固定位置' if i < len(fixed_agents_pos) else '第一个位置'})")
                
                # 设置目标位置
                if 'goal' in self.fixed_positions and self.fixed_positions['goal'] is not None:
                    goal = world.landmarks[0]
                    # 确保目标位置是numpy数组，并复制以避免修改原始数据
                    goal_pos = np.array(self.fixed_positions['goal'], dtype=float).copy()
                    goal_pos = self._adjust_fixed_goal_position_for_current_terrain(
                        goal_pos,
                        quiet_output=quiet_output,
                    )
                    goal.state.p_pos = goal_pos
                    self.goal_pos = goal_pos
                    # 同步到world，便于并行worker/可视化获取
                    if hasattr(world, 'goal_pos'):
                        world.goal_pos = goal_pos.copy()
                    
                    # 🔧 关键修复：调用_set_agent_goals为每个智能体设置独立的、环绕的目标位置
                    # 这样评估器和可视化工具可以正确显示每个智能体的独立目标
                    self._set_agent_goals(world)
                    if len(world.agents) > 0 and not quiet_output:
                        print(f"[目标位置设置] 中央目标: [{self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}, {self.goal_pos[2]:.2f}]")
                        print(f"[目标位置设置] 已为{len(world.agents)}个智能体设置独立的环绕目标")
            # 旧格式：列表 [agent1_pos, agent2_pos, ..., goal_pos]
            elif isinstance(self.fixed_positions, list) and len(self.fixed_positions) >= len(world.agents) + 1:
                # 设置智能体位置
                for i, agent in enumerate(world.agents):
                    if i < len(self.fixed_positions) - 1:  # 最后一个是目标位置
                        # 确保位置是numpy数组，并复制以避免修改原始数据
                        pos = np.array(self.fixed_positions[i], dtype=float).copy()
                        
                        # 🔧 修复：确保X、Y坐标不被修改（固定起点必须保持X、Y不变）
                        # 只根据当前地形的实际高度重新计算Z坐标
                        current_terrain_h = self.get_terrain_height(pos[0], pos[1])
                        
                        # 🔧 修复：验证地形高度是否有效
                        if current_terrain_h == 0.0 and self.terrain is not None:
                            # 🔧 关键修复：使用get_terrain_height方法，自动处理降采样后的坐标映射
                            x_int = int(np.clip(pos[0], 0, self.map_size - 1))
                            y_int = int(np.clip(pos[1], 0, self.map_size - 1))
                            current_terrain_h = float(self.get_terrain_height(x_int, y_int))
                        
                        # 只有当启用随机Z高度时才随机化Z坐标
                        if hasattr(self, 'random_z0_positions') and self.random_z0_positions:
                            random_height = current_terrain_h + 2 + np.random.uniform(0, 5)
                            pos[2] = random_height
                        else:
                            # 🔧 关键修复：保留文件中保存的Z坐标，只在必要时进行安全调整
                            final_terrain_h = self.get_terrain_height(pos[0], pos[1])
                            required_height = final_terrain_h + min_air_gap
                            if pos[2] < required_height:
                                old_z = pos[2]
                                pos[2] = required_height
                                # 🔧 修复：只有在Z坐标实际发生变化时才打印警告
                                if abs(old_z - pos[2]) > 1e-6:
                                    print(f"[固定位置调整] Agent{i}: Z坐标从{old_z:.2f}调整到{pos[2]:.2f}（地形高度={final_terrain_h:.2f}）")
                            
                        agent.state.p_pos = pos
                        agent.state.p_vel = np.zeros(world.dim_p)
                        agent.state.c = np.zeros(world.dim_c)
                
                # 设置目标位置
                goal = world.landmarks[0]
                # 确保目标位置是numpy数组，并复制以避免修改原始数据
                goal_pos = np.array(self.fixed_positions[-1], dtype=float).copy()
                goal_pos = self._adjust_fixed_goal_position_for_current_terrain(
                    goal_pos,
                    quiet_output=False,
                )
                goal.state.p_pos = goal_pos
                self.goal_pos = goal_pos.copy()
                if hasattr(world, 'goal_pos'):
                    world.goal_pos = self.goal_pos.copy()
                
                # 🔧 关键修复：调用_set_agent_goals为每个智能体设置独立的、环绕的目标位置（旧格式）
                self._set_agent_goals(world)
                if len(world.agents) > 0:
                    print(f"[目标位置设置-旧格式] 中央目标: [{self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}, {self.goal_pos[2]:.2f}]")
                    print(f"[目标位置设置-旧格式] 已为{len(world.agents)}个智能体设置独立的环绕目标")
            else:
                print(f"固定位置格式错误或不完整，将使用动态位置代替")
                self._dynamic_reset_world(world)
                return
        except Exception as e:
            print(f"使用固定位置时发生错误: {e}，将使用动态位置代替")
            import traceback
            traceback.print_exc()
            self._dynamic_reset_world(world)
            return
        
        # 🚨 关键修复：使用固定位置时，所有智能体共享同一个中央目标，不需要设置包围目标
        # 问题：_set_agent_goals会将每个智能体的agent.goal_a设置为围绕中央目标的等边三角形位置
        # 但使用固定位置时，所有智能体应该共享同一个中央目标（已经在_apply_fixed_positions中设置）
        # 🔧 关键修复：_set_agent_goals已在_apply_fixed_positions中调用，这里不需要再次调用
        # 旧逻辑（已移除）：
        #   1. 非固定位置模式：调用_set_agent_goals设置独立目标
        #   2. 固定位置模式：验证每个智能体的goal_a是否与中央目标一致，不一致则强制修复
        # 问题：
        #   - _apply_fixed_positions中已调用_set_agent_goals，为每个智能体设置独立的、环绕的目标位置
        #   - 旧的验证逻辑检测到目标"不一致"，就强制覆盖为中央目标，导致独立目标被撤销
        #   - 这会导致交互式轨迹图无法显示各智能体的独立目标
        # 新逻辑：
        #   - 信任_apply_fixed_positions中的设置，不再进行额外的验证和修复
        #   - 仅在完全动态模式（不使用固定位置）时补充调用_set_agent_goals
        if not (getattr(self, 'use_fixed_positions', False) and self.fixed_positions is not None):
            # 完全动态模式：_apply_fixed_positions不会被调用，需要在这里设置独立目标
            self._set_agent_goals(world)
        # 固定位置模式：_apply_fixed_positions中已调用_set_agent_goals，这里无需操作
        # 障碍物刷新统一延后到起点/目标完全确定之后执行，避免复用错误布局。
        try:
            agent_positions_now = [agent.state.p_pos for agent in world.agents]
            agent_goals_now = [agent.goal_a.state.p_pos for agent in world.agents] if hasattr(world, 'agents') else None
            self._refresh_obstacles_for_current_layout(
                world,
                start_positions=agent_positions_now,
                goal_position=self.goal_pos,
                agent_goal_positions=agent_goals_now,
            )
        except Exception:
            pass

        # 初始化或更新到目标距离
        for agent in world.agents:
            if hasattr(agent, 'goal_a') and agent.goal_a.state.p_pos is not None:
                agent.last_goal_dist = np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos)
            else:
                # 如果智能体没有独立目标（理论上不应发生），回退到中央目标
                agent.last_goal_dist = np.linalg.norm(agent.state.p_pos - self.goal_pos) if self.goal_pos is not None else 0.0

        # 标记reset已完成，允许调试信息打印
        world._reset_completed = True
        
        # 创建观察值缓存
        observation_cache = {}
        for i, agent in enumerate(world.agents):
            try:
                obs = self.observation(agent, world)
                observation_cache[id(agent)] = obs
            except Exception as e:
                observation_cache[id(agent)] = np.zeros(self.observation_dim, dtype=np.float32)
        
        # 存储观察值缓存，供环境的reset函数使用
        self.observation_cache = observation_cache
        # 记录每个智能体的起始位置（用于起飞前保护/重力补偿与奖励初始化）
        try:
            for agent in world.agents:
                agent.start_position = agent.state.p_pos.copy()
                # 确保出生时在地形之上至少 airborne_threshold（若world提供），避免初始即穿透
                try:
                    airborne_thr = float(getattr(world, 'pre_takeoff_airborne_threshold', 0.5))
                except Exception:
                    airborne_thr = 0.5
                try:
                    terrain_h = self.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
                except Exception:
                    terrain_h = 0.0
                min_z = float(terrain_h) + float(airborne_thr)
                if agent.state.p_pos[2] < min_z:
                    agent.state.p_pos[2] = min_z
        except Exception:
            pass
        
        # 输出各智能体的初始位置坐标信息（与上游保持一致的开关控制，避免重复打印）
        try:
            import os as _os
            quiet_output = _os.getenv('QUIET_OUTPUT', '1').lower() in ('1', 'true', 'yes', 'on')
            debug_mode = int(_os.getenv('DEBUG_ENV_OUTPUT', '0'))
            should_output = False
            if quiet_output:
                should_output = False
            elif debug_mode == 0:
                should_output = hasattr(world, 'is_main_env') and world.is_main_env
            elif debug_mode == 1:
                should_output = True
            elif debug_mode == 2:
                should_output = False
            if should_output:
                print(f"[智能体位置] 智能体初始位置坐标与最近3个障碍物:")
                # 预取障碍物中心，避免在循环中重复解析
                obstacle_centers = []
                try:
                    import numpy as _np
                    if hasattr(self, 'obstacles') and isinstance(self.obstacles, list):
                        for ob in self.obstacles:
                            try:
                                c = ob.get('center', None)
                                if c is None:
                                    continue
                                c_arr = _np.asarray(c, dtype=_np.float32).reshape(-1)
                                if c_arr.shape[0] >= 3:
                                    obstacle_centers.append(c_arr[:3])
                            except Exception:
                                continue
                except Exception:
                    obstacle_centers = []
                for i, agent in enumerate(world.agents):
                    pos = agent.state.p_pos
                    terrain_h = self.get_terrain_height(pos[0], pos[1])
                    height_above_terrain = pos[2] - terrain_h
                    # 计算与所有障碍物的水平距离，选出最近的3个
                    nearest_str = "None"
                    try:
                        if obstacle_centers:
                            import numpy as _np
                            pos_xy = _np.asarray(pos[:2], dtype=_np.float32)
                            dists = []
                            for c in obstacle_centers:
                                d = float(_np.linalg.norm(pos_xy - c[:2]))
                                dists.append((d, c))
                            dists.sort(key=lambda x: x[0])
                            top3 = dists[:3]
                            parts = []
                            for rank, (d, c) in enumerate(top3, start=1):
                                parts.append(f"{rank}: center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), dist_xy={d:.2f}")
                            nearest_str = " | ".join(parts)
                    except Exception:
                        nearest_str = "计算失败"
                    print(f"  Agent{i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                          f"terrain_h={terrain_h:.2f} | 离地高度={height_above_terrain:.2f}m")
                    print(f"           最近障碍物: {nearest_str}")
        except Exception:
            pass
        # print(f"成功使用固定位置配置：{len(world.agents)}个智能体, 目标位置: {self.goal_pos}")
        
    def _place_obstacles(self, world):
        """
        设置障碍物位置，确保它们在地形表面之上。
        现在直接从 world.obstacles 读取，更加健壮。
        """
        # 直接使用 world.obstacles 列表
        obstacles_to_place = world.obstacles if hasattr(world, 'obstacles') else []
        
        for i, obstacle in enumerate(obstacles_to_place):
            if hasattr(self, 'obstacles') and i < len(self.obstacles):
                obstacle_data = self.obstacles[i]
                # 同步障碍物中心与半径
                obstacle.state.p_pos = np.array(obstacle_data['center'])
                try:
                    obstacle.size = float(obstacle_data.get('radius', getattr(obstacle, 'size', 0.15)))
                except Exception:
                    pass
                
                # 确保障碍物位于地形表面之上，而不是嵌入地形内部
                terrain_height = self.get_terrain_height(obstacle.state.p_pos[0], obstacle.state.p_pos[1])
                min_obstacle_z = terrain_height + float(getattr(obstacle, 'size', 0.15))  # 障碍物底部应该在地形表面
                
                # 如果障碍物的Z坐标太低，调整到合适位置
                if obstacle.state.p_pos[2] < min_obstacle_z:
                    obstacle.state.p_pos[2] = min_obstacle_z
                    print(f"🔧 调整障碍物 {obstacle.name} 位置: 从地形内部提升到地表之上 (Z={min_obstacle_z:.2f})")
                    
            else:
                # 如果障碍物数据不足，放置在安全位置
                obstacle.state.p_pos = self.get_safe_position()

    def _normalize_obstacle_layout_points(self, points):
        """将位置集合归一化为可哈希签名，避免浮点噪声导致的误判。"""
        if points is None:
            return ()
        normalized = []
        try:
            iterable = list(points)
        except Exception:
            iterable = [points]
        for point in iterable:
            try:
                arr = np.asarray(point, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if arr.size == 0:
                continue
            normalized.append(tuple(round(float(x), 4) for x in arr[:3]))
        return tuple(normalized)

    def _build_obstacle_layout_signature(self, start_positions, goal_position, agent_goal_positions, obstacle_seed):
        """构造障碍物布局签名，用于判断固定障碍是否还能安全复用。"""
        try:
            terrain_seed = int(
                getattr(
                    self,
                    'current_terrain_seed',
                    self.seed if self.seed is not None else -1,
                )
            )
        except Exception:
            terrain_seed = -1
        return (
            bool(getattr(self, 'use_dynamic_obstacles', True)),
            int(obstacle_seed),
            int(getattr(self, 'num_obstacles', 0)),
            int(round(float(getattr(self, 'map_size', 0.0)))),
            terrain_seed,
            self._normalize_obstacle_layout_points(start_positions),
            self._normalize_obstacle_layout_points([goal_position] if goal_position is not None else []),
            self._normalize_obstacle_layout_points(agent_goal_positions),
        )

    def _refresh_obstacles_for_current_layout(self, world, start_positions, goal_position, agent_goal_positions=None, force_regenerate=False):
        """
        统一的障碍物刷新入口。

        规则：
        - 动态障碍物：每次 reset 都重生成；
        - 固定障碍物：仅当布局签名变化或当前尚未生成时才重生成；
        - 无论哪种模式，都只在“已知当前起点/目标”之后执行。
        """
        if start_positions is None or goal_position is None:
            return
        try:
            start_positions = [np.asarray(p, dtype=np.float64) for p in list(start_positions)]
        except Exception:
            return
        if len(start_positions) == 0:
            return

        dynamic_obstacles = bool(getattr(self, 'use_dynamic_obstacles', True))
        base_seed = int(self.seed) if self.seed is not None else 42
        obstacle_seed_override = getattr(self, 'current_episode_obstacle_seed_override', None)
        if obstacle_seed_override is not None:
            obstacle_seed = int(obstacle_seed_override)
        elif dynamic_obstacles:
            episode_idx, env_id = self._resolve_episode_context(world)
            obstacle_sequence_mode = str(
                getattr(self, 'train_obstacle_sequence_mode', 'legacy_linear')
            ).strip().lower()
            if obstacle_sequence_mode == 'post_eval_family':
                obstacle_seed = self._make_train_obstacle_family_seed(episode_idx, env_id)
            elif self._use_deterministic_train_env_sequence():
                obstacle_seed = self._make_deterministic_episode_seed('obstacle', episode_idx, env_id)
            else:
                if not hasattr(self, '_obstacle_reset_count'):
                    self._obstacle_reset_count = 0
                self._obstacle_reset_count += 1
                obstacle_seed = base_seed + 10000 + self._obstacle_reset_count * 1000
        else:
            obstacle_seed = base_seed + 10000
        self.current_episode_obstacle_seed = int(obstacle_seed)

        target_signature = self._build_obstacle_layout_signature(
            start_positions=start_positions,
            goal_position=goal_position,
            agent_goal_positions=agent_goal_positions,
            obstacle_seed=obstacle_seed,
        )
        current_signature = getattr(self, '_obstacle_layout_signature', None)
        need_regenerate = bool(force_regenerate or current_signature != target_signature)
        if not need_regenerate:
            if not hasattr(self, 'obstacles') or self.obstacles is None or len(self.obstacles) == 0:
                need_regenerate = True

        if need_regenerate:
            self.rng = np.random.RandomState(obstacle_seed)
            self.generate_obstacles(
                start_positions=start_positions,
                goal_position=goal_position,
                agent_goal_positions=agent_goal_positions,
            )
            self._obstacle_layout_signature = target_signature

        self._place_obstacles(world)
        try:
            self._refresh_observation_static_cache(num_agents=len(world.agents))
        except Exception:
            self._refresh_observation_static_cache()
    
    def _dynamic_reset_world(self, world):
        """动态设置世界位置的原始实现"""
        # 如果设置了随机地形且这是新的回合
        if hasattr(self, 'random_terrain') and self.random_terrain and not hasattr(self, '_terrain_initialized'):
            # 重新生成地形
            self.regenerate_terrain()
            self._terrain_initialized = True
        
        # 确保禁用了固定位置的使用
        # 根据DEBUG_ENV_OUTPUT环境变量控制输出
        debug_mode = int(os.getenv('DEBUG_ENV_OUTPUT', '0'))
        should_output = False
        
        if debug_mode == 0:  # 仅主环境输出
            should_output = hasattr(world, 'is_main_env') and world.is_main_env
        elif debug_mode == 1:  # 所有环境都输出
            should_output = True
        elif debug_mode == 2:  # 仅错误时输出
            should_output = False
        
        if should_output:
            print("使用动态位置设置")
        
        # 使用统一的随机初始化方式（标准方式）
        self._place_agents_standard(world)
            
        # 验证坐标系和目标方向
        self.validate_goal_coordinates(world)
        
        # 新增：根据中央目标点设置每个智能体的包围目标
        self._set_agent_goals(world)

        # 标记reset已完成，允许调试信息打印
        world._reset_completed = True
        
        # 确认所有观察值可以正常获取
        observation_cache = {}
        for i, agent in enumerate(world.agents):
            try:
                obs = self.observation(agent, world)
                observation_cache[id(agent)] = obs
            except Exception as e:
                observation_cache[id(agent)] = np.zeros(self.observation_dim, dtype=np.float32)
        
        # 存储观察值缓存，供环境的reset函数使用
        self.observation_cache = observation_cache
        # 记录每个智能体的起始位置（用于起飞前保护/重力补偿与奖励初始化）
        try:
            for agent in world.agents:
                agent.start_position = agent.state.p_pos.copy()
                # 确保出生时在地形之上至少 airborne_threshold（若world提供），避免初始即穿透
                try:
                    airborne_thr = float(getattr(world, 'pre_takeoff_airborne_threshold', 0.5))
                except Exception:
                    airborne_thr = 0.5
                try:
                    terrain_h = self.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
                except Exception:
                    terrain_h = 0.0
                min_z = float(terrain_h) + float(airborne_thr)
                if agent.state.p_pos[2] < min_z:
                    agent.state.p_pos[2] = min_z
        except Exception:
            pass
    
    def _place_agents_standard(self, world):
        """使用标准方式放置智能体和目标"""
        # 🚨 关键修复：使用场景的随机数生成器self.rng，而不是全局np.random，确保可重复性
        # 如果self.rng不存在，使用全局np.random（向后兼容）
        rng = getattr(self, 'rng', None)
        if rng is None:
            rng = np.random
        
        # 🔧 修复：获取离地高度配置（优先使用环境变量，否则使用默认值12.0米）
        altitude_offset = self._get_start_altitude_offset()
        min_air_gap = max(1.0, altitude_offset)
        
        # 1) 🚨 修复：寻找低海拔平坦区域（地形高度≤8m），用于智能体初始位置
        # 确保智能体从平原/低地出发，而不是山腰或山上
        flat_areas = self.find_flat_area(min_height=0, max_height=8, min_area_size=8)
        if not flat_areas:
            # 如果找不到8m以下的，放宽到12m
            flat_areas = self.find_flat_area(min_height=0, max_height=12, min_area_size=8)
            if not flat_areas:
                # 最后备选：放宽到20m
                flat_areas = self.find_flat_area(min_height=0, max_height=20, min_area_size=8)
        
        if not flat_areas:
            # 极端情况：没有找到平坦区域，使用默认逻辑
            print("警告：未找到任何低海拔平坦区域，使用备用方案")
            return self._place_agents_fallback(world)
        
        flat_areas.sort(key=lambda a: a['height'])
        
        # 2) 选择一个起始区域：在地图的某个角落（距离中心较远的区域）
        # 将地图分为4个象限，选择平坦区域最多的象限作为起始区域
        map_center = self.map_size / 2
        quadrants = {
            'NW': [],  # 西北 (x<center, y>center)
            'NE': [],  # 东北 (x>center, y>center)
            'SW': [],  # 西南 (x<center, y<center)
            'SE': []   # 东南 (x>center, y<center)
        }
        
        for area in flat_areas:
            cx, cy = area['center']
            if cx < map_center:
                if cy > map_center:
                    quadrants['NW'].append(area)
                else:
                    quadrants['SW'].append(area)
            else:
                if cy > map_center:
                    quadrants['NE'].append(area)
                else:
                    quadrants['SE'].append(area)
        
        # 选择平坦区域最多的象限
        best_quadrant = max(quadrants.keys(), key=lambda k: len(quadrants[k]))
        start_areas = quadrants[best_quadrant]
        
        if len(start_areas) < 3:
            # 如果最好的象限也没有足够的区域，从所有区域中选择
            if not _scenario_quiet_output():
                print(f"[智能体放置] 象限{best_quadrant}仅有{len(start_areas)}个区域，从全局选择")
            start_areas = flat_areas[:20]  # 取前20个最平坦的区域
        
        # 3) 在起始区域内集中放置3个智能体
        num_agents = len(world.agents)
        selected_positions = []
        
        # 🚨 关键修复：使用场景的随机数生成器self.rng，而不是全局np.random，确保可重复性
        # 如果self.rng不存在，使用全局np.random（向后兼容）
        rng = getattr(self, 'rng', None)
        if rng is None:
            rng = np.random
        
        # 随机选择区域放置智能体（确保在同一片区域）
        available_areas = list(start_areas)
        for i in range(num_agents):
            if not available_areas:
                available_areas = list(start_areas)
            
            # 随机选择一个区域（使用场景的随机数生成器）
            area = available_areas[rng.randint(0, len(available_areas))]
            cx, cy = area['center']
            h = area['height']
            
            # 在区域内随机偏移（使用场景的随机数生成器）
            jitter = max(3.0, area.get('size', 8.0) / 4.0)
            ax = cx + rng.uniform(-jitter, jitter)
            ay = cy + rng.uniform(-jitter, jitter)
            # 🔧 修复：使用环境变量配置的离地高度，而不是硬编码的2.0米
            terrain_h_at_pos = self.get_terrain_height(ax, ay)
            az = terrain_h_at_pos + altitude_offset
            # 🔧 关键修复：最终验证，确保Z坐标不会低于地形高度
            if az < terrain_h_at_pos + min_air_gap:
                az = terrain_h_at_pos + min_air_gap
            
            selected_positions.append((ax, ay, az))
            
            # 移除这个区域，避免重复使用相同区域
            if area in available_areas:
                available_areas.remove(area)
        
        # 计算智能体集中区域的中心
        agents_center_x = np.mean([pos[0] for pos in selected_positions])
        agents_center_y = np.mean([pos[1] for pos in selected_positions])
        agents_center_z = np.mean([pos[2] for pos in selected_positions])
        agents_center = np.array([agents_center_x, agents_center_y, agents_center_z])
        
        if not _scenario_quiet_output():
            print(f"[智能体放置] 在象限{best_quadrant}集中放置{num_agents}个智能体，区域中心: ({agents_center_x:.1f}, {agents_center_y:.1f}, {agents_center_z:.1f})")
        
        # 4) 找到距离智能体区域最远且最高的山峰
        all_peaks = self.find_peak_positions(neighborhood_size=8, max_peaks=100)
        
        # 获取目标点引用
        goal = world.landmarks[0]
        
        if len(all_peaks) == 0:
            print("警告：找不到合适的山顶，将使用地图对角最远点作为目标")
            # 使用对角点
            goal_x = self.map_size - agents_center_x * 0.3
            goal_y = self.map_size - agents_center_y * 0.3
            goal_x = np.clip(goal_x, self.map_size * 0.5, self.map_size * 0.95)
            goal_y = np.clip(goal_y, self.map_size * 0.5, self.map_size * 0.95)
            terrain_h = self.get_terrain_height(goal_x, goal_y)
            # 🔧 修复：使用配置的目标高度
            goal_z = terrain_h + self._get_goal_altitude()
            goal.state.p_pos = np.array([goal_x, goal_y, goal_z])
            self.goal_pos = goal.state.p_pos.copy()
        else:
            # 🔧 修复：筛选出真正的高峰（高度在前30%）
            peak_heights = [p['position'][2] for p in all_peaks]
            height_threshold = np.percentile(peak_heights, 70)  # 只保留前30%
            high_peaks = [p for p in all_peaks if p['position'][2] >= height_threshold]
            
            # 如果高峰太少，放宽条件
            if len(high_peaks) < 5:
                high_peaks = all_peaks[:min(20, len(all_peaks))]  # 至少选前20个
            
            if not _scenario_quiet_output():
                print(f"[目标选择] 总山峰数={len(all_peaks)}, 筛选高峰={len(high_peaks)}, 高度阈值={height_threshold:.1f}m")
            
            # 🔧 修复：从高峰中选择距离智能体中心最远的（确保距离足够远）
            # 🚨 新增：计算所有高峰到智能体中心的距离，只选择距离足够远的（>80m）
            min_distance_to_start = 80.0  # 最小距离要求
            far_peaks = [
                p for p in high_peaks 
                if np.linalg.norm(np.array([p['position'][0], p['position'][1]]) - agents_center[:2]) > min_distance_to_start
            ]
            
            # 如果没有足够远的高峰，放宽距离要求
            if len(far_peaks) == 0:
                min_distance_to_start = 50.0
                far_peaks = [
                    p for p in high_peaks 
                    if np.linalg.norm(np.array([p['position'][0], p['position'][1]]) - agents_center[:2]) > min_distance_to_start
                ]
            
            # 如果还是没有，直接使用所有高峰
            if len(far_peaks) == 0:
                far_peaks = high_peaks
                if not _scenario_quiet_output():
                    print(f"[目标选择] ⚠️  没有找到距离>{min_distance_to_start}m的高峰，使用所有高峰")
            else:
                if not _scenario_quiet_output():
                    print(f"[目标选择] 找到{len(far_peaks)}个距离>{min_distance_to_start}m的高峰")
            
            # 从远距离高峰中选择最远的
            farthest_peak = max(far_peaks, key=lambda p: np.linalg.norm(
                np.array([p['position'][0], p['position'][1]]) - agents_center[:2]
            ))
            peak_x, peak_y, peak_z = farthest_peak['position']
            
            # 打印最终选择的山峰距离
            final_distance = np.linalg.norm(np.array([peak_x, peak_y]) - agents_center[:2])
            if not _scenario_quiet_output():
                print(f"[目标选择] 选择的遮挡山峰距离智能体: {final_distance:.1f}m")
            
            # 计算到智能体的距离和方向
            direction_to_peak = np.array([peak_x - agents_center_x, peak_y - agents_center_y])
            direction_norm = np.linalg.norm(direction_to_peak)
            
            # 5) 🔧 修复：将目标放在山峰后方更远处（60-100%的额外距离）
            if direction_norm > 1e-6:
                # 归一化方向向量
                direction_unit = direction_to_peak / direction_norm
                
                # 🚨 关键修复：使用场景的随机数生成器，确保目标位置可重复
                # 🔧 关键修复：大幅增加延伸距离，从20-30%提升到60-100%
                extension_distance = direction_norm * rng.uniform(0.6, 1.0)
                goal_x = peak_x + direction_unit[0] * extension_distance
                goal_y = peak_y + direction_unit[1] * extension_distance
                
                # 限制在地图范围内（但允许接近边界）
                goal_x = np.clip(goal_x, 5, self.map_size - 5)
                goal_y = np.clip(goal_y, 5, self.map_size - 5)
                
                # 目标高度：比山峰低一些，确保被遮挡（使用场景的随机数生成器）
                terrain_h_at_goal = self.get_terrain_height(goal_x, goal_y)
                goal_z = terrain_h_at_goal + peak_z * rng.uniform(0.3, 0.5)
                
                # 🔧 修复：使用配置的目标高度（而不是硬编码25米）
                goal_altitude = self._get_goal_altitude()
                goal_z = max(goal_z, terrain_h_at_goal + goal_altitude)
            else:
                # 备用方案：直接放在山峰位置
                goal_x, goal_y = peak_x, peak_y
                terrain_h_at_goal = self.get_terrain_height(goal_x, goal_y)
                # 🔧 修复：使用配置的目标高度
                goal_z = terrain_h_at_goal + self._get_goal_altitude()
            
            goal.state.p_pos = np.array([goal_x, goal_y, goal_z])
            self.goal_pos = goal.state.p_pos.copy()
            
            # 计算并显示距离信息
            dist_agents_to_peak = np.linalg.norm(np.array([peak_x, peak_y, peak_z]) - agents_center)
            dist_agents_to_goal = np.linalg.norm(goal.state.p_pos - agents_center)
            dist_peak_to_goal = np.linalg.norm(goal.state.p_pos - np.array([peak_x, peak_y, peak_z]))
            
            if not _scenario_quiet_output():
                print(f"[目标设置] 遮挡山峰: ({peak_x:.1f}, {peak_y:.1f}, {peak_z:.1f})")
                print(f"[目标设置] 目标位置: ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f})")
                print(f"[目标设置] 距离统计: 智能体→山峰={dist_agents_to_peak:.1f}m, 智能体→目标={dist_agents_to_goal:.1f}m, 山峰→目标={dist_peak_to_goal:.1f}m")
                print(f"[目标设置] ✓ 目标放置在山峰后方，距离增加{(dist_agents_to_goal/dist_agents_to_peak - 1)*100:.1f}%")
        
        if hasattr(world, 'goal_pos'):
            world.goal_pos = self.goal_pos.copy()
        
        # 6) 将智能体放置到预先计算的集中位置
        for i, agent in enumerate(world.agents):
            if i < len(selected_positions):
                ax, ay, az = selected_positions[i]
                agent.state.p_pos = np.array([ax, ay, az])
            else:
                # 备用方案：如果位置不够，使用最后一个位置附近（使用场景的随机数生成器）
                ax, ay, az = selected_positions[-1]
                ax += rng.uniform(-5, 5)
                ay += rng.uniform(-5, 5)
                # 🔧 修复：使用环境变量配置的离地高度
                terrain_h_at_pos = self.get_terrain_height(ax, ay)
                az = terrain_h_at_pos + altitude_offset
                agent.state.p_pos = np.array([ax, ay, az])
            
            # 🔧 关键修复：最终验证，确保智能体不会在地形下方
            final_terrain_h = self.get_terrain_height(agent.state.p_pos[0], agent.state.p_pos[1])
            if agent.state.p_pos[2] < final_terrain_h + min_air_gap:
                agent.state.p_pos[2] = final_terrain_h + min_air_gap
                if not _scenario_quiet_output():
                    print(f"[智能体放置] 警告：智能体{i}位置在地形下方，已调整到地形高度+{min_air_gap:.1f}m")
            
            agent.state.p_vel = np.zeros(3)
            if hasattr(agent, 'action') and hasattr(agent.action, 'u'):
                agent.action.u = np.zeros(world.dim_p)
        
        # 在生成障碍前，先设置每个智能体的独立目标
        self._set_agent_goals(world)
        agent_positions = [agent.state.p_pos for agent in world.agents]
        agent_goals_now = [agent.goal_a.state.p_pos for agent in world.agents]
        self._refresh_obstacles_for_current_layout(
            world,
            start_positions=agent_positions,
            goal_position=self.goal_pos,
            agent_goal_positions=agent_goals_now,
            force_regenerate=True,
        )
        
        # 输出各智能体的初始位置坐标信息
        if not _scenario_quiet_output():
            print(f"[智能体位置] 智能体初始位置坐标:")
            for i, agent in enumerate(world.agents):
                pos = agent.state.p_pos
                terrain_h = self.get_terrain_height(pos[0], pos[1])
                height_above_terrain = pos[2] - terrain_h
                print(f"  Agent{i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | terrain_h={terrain_h:.2f} | 离地高度={height_above_terrain:.2f}m")
    
    def _place_agents_fallback(self, world):
        """备用智能体放置方案（当找不到平坦区域时）"""
        if not _scenario_quiet_output():
            print("[智能体放置] 使用备用方案：在低海拔安全位置放置")
        
        # 🔧 修复：获取离地高度配置（使用统一方法）
        altitude_offset = self._get_start_altitude_offset()
        goal_altitude = self._get_goal_altitude()
        
        # 获取目标点引用
        goal = world.landmarks[0]
        
        # 设置目标为地图对角
        goal_x = self.map_size * 0.8
        goal_y = self.map_size * 0.8
        terrain_h = self.get_terrain_height(goal_x, goal_y)
        # 🔧 修复：使用配置的目标高度
        goal_z = terrain_h + goal_altitude
        goal.state.p_pos = np.array([goal_x, goal_y, goal_z])
        self.goal_pos = goal.state.p_pos.copy()
        if hasattr(world, 'goal_pos'):
            world.goal_pos = self.goal_pos.copy()
        
        # 将智能体放置在地图另一角
        num_agents = len(world.agents)
        start_x = self.map_size * 0.2
        start_y = self.map_size * 0.2
        
        for i, agent in enumerate(world.agents):
            # 围绕起点做环形分散
            angle = 2 * np.pi * i / max(1, num_agents)
            offset_x = 10.0 * np.cos(angle)
            offset_y = 10.0 * np.sin(angle)
            
            ax = start_x + offset_x
            ay = start_y + offset_y
            # 🔧 修复：使用环境变量配置的离地高度
            az = self.get_terrain_height(ax, ay) + altitude_offset
            
            agent.state.p_pos = np.array([ax, ay, az])
            agent.state.p_vel = np.zeros(3)
            if hasattr(agent, 'action') and hasattr(agent.action, 'u'):
                agent.action.u = np.zeros(world.dim_p)
        
        # 先设置包围目标，再按当前布局刷新障碍物
        self._set_agent_goals(world)
        agent_positions = [agent.state.p_pos for agent in world.agents]
        agent_goals_now = [agent.goal_a.state.p_pos for agent in world.agents]
        self._refresh_obstacles_for_current_layout(
            world,
            start_positions=agent_positions,
            goal_position=self.goal_pos,
            agent_goal_positions=agent_goals_now,
            force_regenerate=True,
        )
    
    def find_flat_area(self, min_height=0, max_height=20, min_area_size=5):
        """寻找地形中的平坦区域（高度在指定范围内，特别是0-20cm区域）"""
        flat_areas = []
        
        # 确保地形已经生成
        if self.terrain is None:
            print("警告: 查找平坦区域时地形数据为空，返回空列表")
            return flat_areas
        
        # 获取地形统计信息
        terrain_mean = np.mean(self.terrain)
        terrain_std = np.std(self.terrain)
        terrain_min = np.min(self.terrain)
        terrain_max = np.max(self.terrain)
        
        if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
            print(f"[平坦区域检测] 地形统计: 平均高度={terrain_mean:.2f}, 标准差={terrain_std:.2f}")
            print(f"[平坦区域检测] 地形范围: 最低点={terrain_min:.2f}, 最高点={terrain_max:.2f}")
            print(f"[平坦区域检测] 寻找高度范围: {min_height}-{max_height}cm")
        
        # 扫描整个地形寻找平坦区域
        # 🔧 修复：将 map_size 和 min_area_size 转换为整数，避免 range() 类型错误
        map_size_int = int(self.map_size)
        min_area_size_int = int(min_area_size)
        for x in range(0, map_size_int - min_area_size_int, min_area_size_int):
            for y in range(0, map_size_int - min_area_size_int, min_area_size_int):
                # 检查当前区域
                # 🔧 关键修复：使用get_terrain_height方法，自动处理降采样后的坐标映射
                heights = []
                for dx in range(min_area_size_int):
                    for dy in range(min_area_size_int):
                        heights.append(self.get_terrain_height(x+dx, y+dy))
                
                # 计算区域高度统计
                avg_height = np.mean(heights)
                height_variance = np.var(heights)
                height_range = max(heights) - min(heights)
                
                # 更严格的平坦度判断：
                # 1. 高度方差小于2.0（更严格）
                # 2. 高度范围小于5.0（区域内高度差不超过5cm）
                # 3. 平均高度在指定范围内
                is_flat = (height_variance < 2.0 and 
                          height_range < 5.0 and 
                          min_height <= avg_height <= max_height)
                
                if is_flat:
                    # 添加该区域中心点
                    center_x = x + min_area_size // 2
                    center_y = y + min_area_size // 2
                    flat_areas.append({
                        'center': (center_x, center_y),
                        'height': avg_height,
                        'size': min_area_size,
                        'variance': height_variance,
                        'range': height_range
                    })
        
        # 按高度排序，优先选择较低的区域
        flat_areas.sort(key=lambda a: a['height'])
        
        if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
            print(f"[平坦区域检测] 找到 {len(flat_areas)} 个平坦区域:")
            for i, area in enumerate(flat_areas[:3]):  # 降低打印量，最多3条
                center = area['center']
                print(f"  区域{i+1}: 中心=({center[0]:.1f}, {center[1]:.1f}), 高度={area['height']:.1f}cm, 方差={area['variance']:.2f}, 范围={area['range']:.1f}")
            if len(flat_areas) > 3:
                print(f"  ... 还有 {len(flat_areas) - 3} 个区域")
        
        return flat_areas

    def get_safe_position(self, min_x=0, max_x=100, min_y=0, max_y=100, max_height=40, min_dist_from_obstacles=10, safety_height=2):
        """获取一个安全的位置（不在山脉内部，远离障碍物）"""
        # 尝试最多30次找到合适的位置
        for _ in range(30):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            
            # 获取此处地形高度
            terrain_height = self.get_terrain_height(x, y)
            
            # 检查是否低于最大高度
            if terrain_height > max_height:
                continue
            
            # 检查是否远离所有障碍物
            far_from_obstacles = True
            for obstacle in self.obstacles:
                obstacle_x, obstacle_y = obstacle['center'][0], obstacle['center'][1]
                dist = np.sqrt((x - obstacle_x)**2 + (y - obstacle_y)**2)
                if dist < min_dist_from_obstacles:
                    far_from_obstacles = False
                    break
            
            if far_from_obstacles:
                # 返回安全位置，高度为地形高度加上安全高度
                return np.array([x, y, terrain_height + safety_height])
        
        # 如果找不到理想位置，返回备用位置
        backup_x = np.random.uniform(min_x, max_x)
        backup_y = np.random.uniform(min_y, max_y)
        backup_height = self.get_terrain_height(backup_x, backup_y) + safety_height * 2
        # print(f"警告：无法找到理想的安全位置，使用备用位置: [{backup_x}, {backup_y}, {backup_height}]")
        return np.array([backup_x, backup_y, backup_height])
        
    def get_terrain_height(self, x, y):
        """获取地形高度，使用双线性插值获得平滑值"""
        # 确保地形已经生成
        if self.terrain is None:
            return 0.0
        
        # 🔧 关键修复：如果地形已降采样，需要将坐标从0-200范围映射到降采样后的索引范围
        if getattr(self, 'terrain_downsampled', False):
            sample_rate = getattr(self, 'terrain_sample_rate', 4)
            terrain_w = self.terrain.shape[1]
            terrain_h = self.terrain.shape[0]
            
            # 将坐标从原始范围(0-map_size-1)映射到降采样后的索引范围(0-terrain_w-1)
            # 使用线性映射：x_scaled = x * (terrain_w - 1) / (map_size - 1)
            # 这样可以确保map_size-1映射到terrain_w-1
            x_scaled = x * (terrain_w - 1) / (self.map_size - 1) if self.map_size > 1 else 0
            y_scaled = y * (terrain_h - 1) / (self.map_size - 1) if self.map_size > 1 else 0
            
            # 确保坐标在降采样后的地形范围内
            x_scaled = max(0, min(x_scaled, terrain_w - 1))
            y_scaled = max(0, min(y_scaled, terrain_h - 1))
            
            x_low = int(np.floor(x_scaled))
            y_low = int(np.floor(y_scaled))
            x_high = int(np.ceil(x_scaled))
            y_high = int(np.ceil(y_scaled))
            
            # 确保索引在有效范围内
            x_low = max(0, min(x_low, terrain_w - 1))
            x_high = max(0, min(x_high, terrain_w - 1))
            y_low = max(0, min(y_low, terrain_h - 1))
            y_high = max(0, min(y_high, terrain_h - 1))
        else:
            # 原始逻辑：未降采样，直接使用原始坐标
            # 确保坐标在有效范围内
            x = max(0, min(x, self.map_size-1))
            y = max(0, min(y, self.map_size-1))
            
            x_low = int(np.floor(x))
            y_low = int(np.floor(y))
            x_high = int(np.ceil(x))
            y_high = int(np.ceil(y))
            
            # 确保索引在有效范围内
            x_low = max(0, min(x_low, self.map_size-1))
            x_high = max(0, min(x_high, self.map_size-1))
            y_low = max(0, min(y_low, self.map_size-1))
            y_high = max(0, min(y_high, self.map_size-1))
        
        # 如果索引相同（整数坐标），直接返回
        if x_low == x_high and y_low == y_high:
            return self.terrain[y_low, x_low]
        
        # 计算插值权重
        # 🔧 关键修复：如果地形已降采样，使用缩放后的坐标计算权重
        if getattr(self, 'terrain_downsampled', False):
            # x_scaled和y_scaled已经在上面计算过了，直接使用
            x_weight = x_scaled - x_low
            y_weight = y_scaled - y_low
        else:
            x_weight = x - x_low
            y_weight = y - y_low
        
        # 双线性插值
        val1 = self.terrain[y_low, x_low]
        val2 = self.terrain[y_low, x_high] if x_low != x_high else val1
        val3 = self.terrain[y_high, x_low] if y_low != y_high else val1
        val4 = self.terrain[y_high, x_high] if x_low != x_high and y_low != y_high else val3
        
        # 加权平均
        height = (1-x_weight)*(1-y_weight)*val1 + \
                 x_weight*(1-y_weight)*val2 + \
                 (1-x_weight)*y_weight*val3 + \
                 x_weight*y_weight*val4
                 
        return height
    
    def batch_get_terrain_height(self, coords):
        """
        🚀 批量获取地形高度，使用向量化双线性插值
        
        Args:
            coords: (N, 2) array of (x, y) coordinates
            
        Returns:
            heights: (N,) array of terrain heights
            
        性能优化：
        - 使用NumPy向量化操作，避免Python循环
        - 对XLA加速友好（纯NumPy操作）
        - 与单点版本数值完全一致
        """
        # 确保地形已经生成
        if self.terrain is None:
            return np.zeros(len(coords), dtype=np.float32)
        
        # 转换为NumPy数组并确保形状正确
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        N = coords.shape[0]
        x_coords = coords[:, 0]  # (N,)
        y_coords = coords[:, 1]  # (N,)
        
        # 🔧 关键修复：支持降采样地形
        if getattr(self, 'terrain_downsampled', False):
            terrain_w = self.terrain.shape[1]
            terrain_h = self.terrain.shape[0]
            
            # 批量缩放坐标
            if self.map_size > 1:
                x_scaled = x_coords * (terrain_w - 1) / (self.map_size - 1)
                y_scaled = y_coords * (terrain_h - 1) / (self.map_size - 1)
            else:
                x_scaled = np.zeros(N, dtype=np.float32)
                y_scaled = np.zeros(N, dtype=np.float32)
            
            # 批量裁剪到有效范围
            x_scaled = np.clip(x_scaled, 0, terrain_w - 1)
            y_scaled = np.clip(y_scaled, 0, terrain_h - 1)
            
            # 批量计算索引
            x_low = np.floor(x_scaled).astype(np.int32)
            y_low = np.floor(y_scaled).astype(np.int32)
            x_high = np.ceil(x_scaled).astype(np.int32)
            y_high = np.ceil(y_scaled).astype(np.int32)
            
            # 批量裁剪索引
            x_low = np.clip(x_low, 0, terrain_w - 1)
            x_high = np.clip(x_high, 0, terrain_w - 1)
            y_low = np.clip(y_low, 0, terrain_h - 1)
            y_high = np.clip(y_high, 0, terrain_h - 1)
            
            # 批量计算插值权重
            x_weight = x_scaled - x_low
            y_weight = y_scaled - y_low
        else:
            # 未降采样：直接使用原始坐标
            x_clipped = np.clip(x_coords, 0, self.map_size - 1)
            y_clipped = np.clip(y_coords, 0, self.map_size - 1)
            
            # 批量计算索引
            x_low = np.floor(x_clipped).astype(np.int32)
            y_low = np.floor(y_clipped).astype(np.int32)
            x_high = np.ceil(x_clipped).astype(np.int32)
            y_high = np.ceil(y_clipped).astype(np.int32)
            
            # 批量裁剪索引
            x_low = np.clip(x_low, 0, self.map_size - 1)
            x_high = np.clip(x_high, 0, self.map_size - 1)
            y_low = np.clip(y_low, 0, self.map_size - 1)
            y_high = np.clip(y_high, 0, self.map_size - 1)
            
            # 批量计算插值权重
            x_weight = x_clipped - x_low
            y_weight = y_clipped - y_low
        
        # 批量获取四个角点的地形高度
        val1 = self.terrain[y_low, x_low]  # (N,)
        val2 = self.terrain[y_low, x_high]  # (N,)
        val3 = self.terrain[y_high, x_low]  # (N,)
        val4 = self.terrain[y_high, x_high]  # (N,)
        
        # 批量双线性插值（向量化）
        heights = ((1 - x_weight) * (1 - y_weight) * val1 +
                   x_weight * (1 - y_weight) * val2 +
                   (1 - x_weight) * y_weight * val3 +
                   x_weight * y_weight * val4)
        
        return heights.astype(np.float32)
    
    def get_terrain_grad(self, x, y, dx=1.0, dy=1.0):
        """
        🔧 新增：获取地形梯度（Oracle接口）
        
        使用有限差分计算地形梯度：
        - grad_x = (h(x+dx, y) - h(x-dx, y)) / (2*dx)
        - grad_y = (h(x, y+dy) - h(x, y-dy)) / (2*dy)
        
        Args:
            x: X坐标
            y: Y坐标
            dx: X方向差分步长（默认1.0米）
            dy: Y方向差分步长（默认1.0米）
        
        Returns:
            grad: (grad_x, grad_y) 地形梯度向量
        """
        try:
            # 获取周围点高度
            h_x_plus = self.get_terrain_height(x + dx, y)
            h_x_minus = self.get_terrain_height(x - dx, y)
            h_y_plus = self.get_terrain_height(x, y + dy)
            h_y_minus = self.get_terrain_height(x, y - dy)
            
            # 计算梯度 (dz/dx, dz/dy) - 中心差分法
            grad_x = (h_x_plus - h_x_minus) / (2.0 * max(dx, 1e-6))
            grad_y = (h_y_plus - h_y_minus) / (2.0 * max(dy, 1e-6))
            
            return np.array([grad_x, grad_y], dtype=np.float32)
        except Exception as e:
            # 如果计算失败，返回零梯度
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def get_terrain_heights_batch(self, x_coords, y_coords, heights_out=None):
        """
        🔧 备份：批量查询地形高度（向量化实现，用于后续优化）
        
        Args:
            x_coords: x坐标数组 (N,) 或单个值
            y_coords: y坐标数组 (N,) 或单个值
            heights_out: 输出数组 (N,)，如果为None则创建新数组
            
        Returns:
            heights_out: 高度数组 (N,)
        """
        """
        批量查询地形高度（向量化实现，性能优化）
        
        Args:
            x_coords: x坐标数组 (N,) 或单个值
            y_coords: y坐标数组 (N,) 或单个值
            heights_out: 输出数组 (N,)，如果为None则创建新数组
            
        Returns:
            heights_out: 高度数组 (N,)
        """
        # 确保地形已经生成
        if self.terrain is None:
            if heights_out is None:
                if np.isscalar(x_coords):
                    return np.array([0.0], dtype=np.float32)[0]
                return np.zeros(len(x_coords), dtype=np.float32)
            heights_out[:] = 0.0
            return heights_out
        
        # 处理单个值的情况
        if np.isscalar(x_coords) or np.isscalar(y_coords):
            x_coords = np.array([x_coords], dtype=np.float32)
            y_coords = np.array([y_coords], dtype=np.float32)
            single_value = True
        else:
            x_coords = np.asarray(x_coords, dtype=np.float32)
            y_coords = np.asarray(y_coords, dtype=np.float32)
            single_value = False
        
        num_coords = len(x_coords)
        if heights_out is None:
            heights_out = np.zeros(num_coords, dtype=np.float32)
        else:
            heights_out = np.asarray(heights_out, dtype=np.float32)
            if len(heights_out) < num_coords:
                heights_out = np.zeros(num_coords, dtype=np.float32)
        
        # 检查地形是否降采样
        terrain_downsampled = getattr(self, 'terrain_downsampled', False)
        terrain_h, terrain_w = self.terrain.shape[0], self.terrain.shape[1]
        
        if terrain_downsampled:
            # 地形已降采样，需要缩放坐标
            if self.map_size > 1:
                x_scaled = x_coords * (terrain_w - 1) / (self.map_size - 1)
                y_scaled = y_coords * (terrain_h - 1) / (self.map_size - 1)
            else:
                x_scaled = np.zeros_like(x_coords)
                y_scaled = np.zeros_like(y_coords)
            
            # 确保坐标在降采样后的地形范围内
            x_coords_clipped = np.clip(x_scaled, 0.0, float(terrain_w - 1))
            y_coords_clipped = np.clip(y_scaled, 0.0, float(terrain_h - 1))
        else:
            # 未降采样，直接使用原始坐标
            x_coords_clipped = np.clip(x_coords, 0.0, float(self.map_size - 1))
            y_coords_clipped = np.clip(y_coords, 0.0, float(self.map_size - 1))
        
        # 计算四个角点的索引（向量化）
        x_low = np.floor(x_coords_clipped).astype(np.int32)
        y_low = np.floor(y_coords_clipped).astype(np.int32)
        x_high = np.minimum(x_low + 1, terrain_w - 1)
        y_high = np.minimum(y_low + 1, terrain_h - 1)
        
        # 确保索引在有效范围内
        x_low = np.clip(x_low, 0, terrain_w - 1)
        y_low = np.clip(y_low, 0, terrain_h - 1)
        x_high = np.clip(x_high, 0, terrain_w - 1)
        y_high = np.clip(y_high, 0, terrain_h - 1)
        
        # 计算插值权重（向量化）
        x_weight = x_coords_clipped - x_low.astype(np.float32)
        y_weight = y_coords_clipped - y_low.astype(np.float32)
        
        # 获取四个角点的高度值（向量化）
        h00 = self.terrain[y_low, x_low]  # 左下
        h10 = self.terrain[y_low, x_high]  # 右下
        h01 = self.terrain[y_high, x_low]  # 左上
        h11 = self.terrain[y_high, x_high]  # 右上
        
        # 处理NaN和Inf值
        valid_mask = np.isfinite(h00) & np.isfinite(h10) & np.isfinite(h01) & np.isfinite(h11)
        if not np.all(valid_mask):
            invalid_mask = ~valid_mask
            # 使用最近的有效值（优先使用h00）
            h00[invalid_mask] = np.where(
                np.isfinite(h00[invalid_mask]), h00[invalid_mask],
                np.where(np.isfinite(h10[invalid_mask]), h10[invalid_mask],
                        np.where(np.isfinite(h01[invalid_mask]), h01[invalid_mask], h11[invalid_mask]))
            )
            h10[invalid_mask] = np.where(np.isfinite(h10[invalid_mask]), h10[invalid_mask], h00[invalid_mask])
            h01[invalid_mask] = np.where(np.isfinite(h01[invalid_mask]), h01[invalid_mask], h00[invalid_mask])
            h11[invalid_mask] = np.where(np.isfinite(h11[invalid_mask]), h11[invalid_mask], h00[invalid_mask])
        
        # 双线性插值（向量化）
        h0 = h00 * (1.0 - x_weight) + h10 * x_weight  # 下边插值
        h1 = h01 * (1.0 - x_weight) + h11 * x_weight  # 上边插值
        heights_out[:num_coords] = h0 * (1.0 - y_weight) + h1 * y_weight  # 最终插值
        
        # 处理无效输出
        invalid_output = ~np.isfinite(heights_out[:num_coords])
        if np.any(invalid_output):
            x_nearest = np.clip(x_low[invalid_output], 0, terrain_w - 1)
            y_nearest = np.clip(y_low[invalid_output], 0, terrain_h - 1)
            heights_out[invalid_output] = self.terrain[y_nearest, x_nearest]
        
        # 如果是单个值，返回标量
        if single_value:
            return heights_out[0]
        return heights_out[:num_coords]
    
        
    def reward(self, agent, world):
        import os  # 确保os模块可用
        """
        分项加权求和奖励函数
        """
        # 🔧 修复：健壮性检查 - 确保输入有效
        if agent.state.p_pos is None:
            return 0.0
        if np.any(np.isnan(agent.state.p_pos)) or np.any(np.isinf(agent.state.p_pos)):
            # 如果位置无效，给予巨大惩罚并尝试不崩溃（虽然此时物理引擎可能已经坏了）
            return -1000.0
            
        # 平滑基元，避免硬阈值跳变
        def _smoothstep(x, c=0.0, w=1.0):
            w = max(w, 1e-6)
            return 1.0 / (1.0 + np.exp(-(x - c) / w))

        def _softplus(x, beta=5.0):
            beta = max(beta, 1e-6)
            return (1.0 / beta) * np.log1p(np.exp(beta * x))
        
        try:
            # 获取智能体的独立目标位置
            if not hasattr(agent, 'goal_a') or agent.goal_a.state.p_pos is None:
                return 0.0 # Agent has no goal, no reward
            goal_pos = agent.goal_a.state.p_pos
            
            # 计算到目标的距离
            agent_pos = agent.state.p_pos
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            
            # 初始化智能体需要的所有属性
            if not hasattr(agent, 'initialized_for_reward') or not agent.initialized_for_reward:
                agent.last_goal_dist = dist_to_goal
                agent.stationary_count = 0
                agent.last_position = agent.state.p_pos.copy()
                agent.last_velocity = np.zeros(3)
                agent.visited_cells = set()  # 用于记录已访问区域
                agent.debug_info = {}  # 添加调试信息字典
                agent.initialized_for_reward = True
                agent.start_position = agent.state.p_pos.copy()  # 记录起始位置
                
                # 计算从起始点到目标点的初始距离和方向（优先每智能体目标）
                agent.initial_distance_to_goal = dist_to_goal  # 记录初始距离
                agent.start_to_goal_dir = None
                _goal_vec = None
                try:
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None:
                        _goal_vec = agent.goal_a.state.p_pos
                    elif self.goal_pos is not None:
                        _goal_vec = self.goal_pos
                except Exception:
                    _goal_vec = self.goal_pos if self.goal_pos is not None else None
                if _goal_vec is not None:
                    start_to_goal = _goal_vec - agent.state.p_pos
                    _d0 = np.linalg.norm(start_to_goal)
                    if _d0 > 1e-6:
                        agent.start_to_goal_dir = start_to_goal / _d0
                
                # 第一次计算默认值
                return 0.0  # 第一帧没有奖励
            
            # 基础变量初始化
            rew = 0.0
            energy_consumption = 0.0
            energy_reward = 0.0
            energy_penalty = 0.0
            exploration_reward = 0.0
            dist_penalty = 0.0
            height_diff = 0.0
            is_stationary = False
            
            # 确保 last_goal_dist 存在，即使初始化失败也能正常工作
            if not hasattr(agent, 'last_goal_dist'):
                agent.last_goal_dist = dist_to_goal
            
            # 确保 initial_distance_to_goal 存在
            if not hasattr(agent, 'initial_distance_to_goal'):
                agent.initial_distance_to_goal = dist_to_goal
            
            # 计算距离变化（正值表示接近目标，负值表示远离）
            dist_change = agent.last_goal_dist - dist_to_goal
            
            # 更新上一次距离
            agent.last_goal_dist = dist_to_goal
            
            # === 距离惩罚（软化：连续饱和） ===
            distance_ratio = dist_to_goal / agent.initial_distance_to_goal
            k_dev = 15.0
            dev_penalty = k_dev * _softplus(distance_ratio - 1.0, beta=4.0)
            rew -= dev_penalty
            if hasattr(agent, 'debug_info'):
                agent.debug_info['deviation_penalty_soft'] = float(dev_penalty)
                agent.debug_info['distance_ratio'] = float(distance_ratio)
            
            agent_radius = float(max(0.0, getattr(agent, 'size', getattr(self, 'agent_size', 0.0))))

            # 计算当前位置和对应的地形高度，用于记录
            terrain_height = 0.0
            height_diff = 0.0
            if hasattr(self, 'get_terrain_height'):
                current_pos = agent.state.p_pos
                terrain_height = self.get_terrain_height(current_pos[0], current_pos[1])
                body_bottom_z = current_pos[2] - agent_radius
                height_diff = body_bottom_z - terrain_height
                
                # === 地形穿透惩罚（软化） ===
                depth = max(0.0, terrain_height - body_bottom_z)
                k_pen = 500.0  # 🚨 修复穿透问题：从100.0提高到500.0，大幅增强穿透惩罚
                k_deep = 1500.0  # 🚨 修复穿透问题：从300.0提高到1500.0，严惩深度穿透
                pen_main = k_pen * _softplus(depth, beta=6.0)
                pen_deep = k_deep * _softplus(depth - 2.0, beta=6.0)
                rew -= (pen_main + pen_deep)
                if hasattr(agent, 'debug_info') and depth > 0.0:
                    agent.debug_info['penetration_penalty_soft'] = float(pen_main + pen_deep)
                    agent.debug_info['penetration_depth'] = float(depth)
            
            # === 障碍物穿透/近邻惩罚（软化） ===
            obstacle_penalty = 0.0
            if getattr(self, '_obs_centers', None) is not None and self._obs_centers.shape[0] > 0:
                current_pos = agent.state.p_pos.astype(np.float32)
                diff = self._obs_centers - current_pos
                d = np.sqrt(np.sum(diff * diff, axis=1))
                r = self._obs_radii
                gap = d - (r + agent_radius)
                k_near = 50.0   # 🚨 修复穿透问题：从20.0提高到50.0，进一步增强接近障碍物的惩罚
                k_coll = 1500.0 # 🚨 修复穿透问题：从600.0提高到1500.0，更严厉惩罚穿透障碍物
                near_term = k_near * np.sum(_softplus(-(gap - 1.0), beta=5.0))
                pen_term = k_coll * np.sum(_softplus(-gap, beta=6.0))
                rew -= (near_term + pen_term)
                obstacle_penalty += -(near_term + pen_term)
            
                        # === 净空/最小距离增益奖励机制 ===
            # 🔧 关键修复：最小净空应该记录地形和障碍物的最小距离，允许负值（穿透）
            # 鼓励智能体增加与最近障碍物和地形的距离，防止被势阱卡住
            clearance_distances = []  # 收集所有净空距离（包括地形和障碍物）
            
            # 1. 计算地形净空（允许负值表示穿透）
            if hasattr(self, 'get_terrain_height'):
                current_pos = agent.state.p_pos
                terrain_height = self.get_terrain_height(current_pos[0], current_pos[1])
                terrain_clearance = current_pos[2] - agent_radius - terrain_height  # 正值=机体底部在地形上方，负值=穿透地形
                clearance_distances.append(terrain_clearance)
            
            # 2. 计算障碍物净空（允许负值表示穿透）
            if getattr(self, '_obs_centers', None) is not None and self._obs_centers.shape[0] > 0:
                current_pos_obs = agent.state.p_pos.astype(np.float32)
                diff = self._obs_centers - current_pos_obs
                d = np.sqrt(np.sum(diff * diff, axis=1))
                obstacle_clearances = d - (self._obs_radii + agent_radius)  # 正值=机体在障碍物外，负值=穿透障碍物
                clearance_distances.extend(obstacle_clearances.tolist())
            
            # 3. 取最小净空值（最负值表示最大穿透深度）
            if clearance_distances:
                d_min_current = float(np.min(clearance_distances))  # 允许负值
            else:
                d_min_current = 0.0  # 默认值（如果没有地形和障碍物）
            
            # 🚨 关键修复：检测碰撞并更新total_penetration_count
            # 使用与vectorized_reward_calculator相同的碰撞检测逻辑
            # 从环境变量或类属性获取碰撞阈值
            try:
                import os
                collision_distance_threshold = float(os.getenv('COLLISION_DISTANCE_THRESHOLD', '0.5'))
            except Exception:
                collision_distance_threshold = float(getattr(self, 'collision_distance_threshold', 0.5))
            
            # 检测碰撞：d_min_current < collision_distance_threshold 或 < 0（穿透）
            has_collision = (d_min_current < collision_distance_threshold) or (d_min_current < 0.0)
            
            # 如果检测到碰撞，更新碰撞计数
            if has_collision:
                # 只要本步检测到碰撞/穿透，即认为本回合已发生危险行为（Safe_i 应为 False）
                # 计数可按去重策略增量，但 episode 标志必须无条件置 True
                try:
                    agent._episode_has_collision = True
                except Exception:
                    pass
                # 确保debug_info已初始化
                if not hasattr(agent, 'debug_info'):
                    agent.debug_info = {}
                if not isinstance(agent.debug_info, dict):
                    agent.debug_info = {}
                
                # 🚨 关键修复：改进防重复计数机制
                # 如果current_step不可用，使用位置变化来防止重复计数
                current_step = int(getattr(world, 'current_step', -1))
                last_counted_step = getattr(agent, '_last_collision_counted_step', -1)
                
                # 如果current_step不可用（-1），使用位置变化来检测是否是新碰撞
                should_count = False
                if current_step != -1 and current_step >= 0:
                    # 正常情况：使用步数判断
                    should_count = (current_step != last_counted_step)
                else:
                    # 回退方案：使用位置变化判断（如果位置变化超过阈值，认为是新碰撞）
                    # 这样可以避免在评估时current_step不可用导致计数失效
                    current_pos = agent.state.p_pos
                    last_collision_pos = getattr(agent, '_last_collision_position', None)
                    if last_collision_pos is None:
                        should_count = True  # 第一次碰撞
                    else:
                        pos_change = np.linalg.norm(current_pos - last_collision_pos)
                        # 位置变化超过0.1m认为是新碰撞（避免同一位置重复计数）
                        should_count = (pos_change > 0.1)
                
                if should_count:
                    # 更新碰撞计数（逻辑不变）
                    old_count = agent.debug_info.get('total_penetration_count', 0)
                    try:
                        old_count_int = int(old_count) if np.isfinite(old_count) else 0
                    except (ValueError, TypeError, OverflowError):
                        old_count_int = 0
                    
                    new_count = old_count_int + 1
                    if new_count > 1000000:  # 防止溢出
                        new_count = 1000000
                    
                    agent.debug_info['total_penetration_count'] = int(new_count)
                    agent._last_collision_counted_step = current_step if current_step != -1 else -1
                    agent._last_collision_position = agent.state.p_pos.copy()  # 保存碰撞位置
                    
                    # 分项统计（仅显示用）：判断本次碰撞来自地形还是球形障碍
                    n_terrain = 1 if (hasattr(self, 'get_terrain_height') and clearance_distances) else 0
                    arr = np.array(clearance_distances, dtype=np.float64)
                    min_idx = int(np.argmin(arr))
                    is_terrain_collision = (n_terrain == 1 and min_idx == 0)
                    if is_terrain_collision:
                        agent.debug_info['terrain_penetration_count'] = agent.debug_info.get('terrain_penetration_count', 0) + 1
                    else:
                        agent.debug_info['obstacle_collision_count'] = agent.debug_info.get('obstacle_collision_count', 0) + 1
                    
                    # 同时更新current_episode_collision_count（保持兼容性）
                    if not hasattr(agent, 'current_episode_collision_count'):
                        agent.current_episode_collision_count = 0
                    agent.current_episode_collision_count += 1
                    
                    # 标记碰撞状态
                    agent._had_penetration_or_collision = True
                    if d_min_current < 0.0:  # 穿透地形
                        agent._had_terrain_contact_or_penetration = True
                    
            # 获取上一时刻的最小距离
            if not hasattr(agent, 'last_min_distance'):
                agent.last_min_distance = d_min_current
                clearance_reward = 0.0
            else:
                d_min_previous = agent.last_min_distance
                distance_change = d_min_current - d_min_previous
                normalized_change = distance_change / max(1e-6, self.clearance_d_max)
                clipped_change = np.clip(normalized_change, -1.0, 1.0)
                clearance_reward = self.clearance_weight * clipped_change
                if hasattr(agent, 'debug_info'):
                    agent.debug_info['clearance_reward'] = clearance_reward
                    agent.debug_info['d_min_current'] = d_min_current
                    agent.debug_info['d_min_previous'] = d_min_previous
                    agent.debug_info['distance_change'] = distance_change
                    agent.debug_info['normalized_change'] = normalized_change
                    agent.debug_info['clearance_weight'] = self.clearance_weight
                # 🔧 新增：记录地形和障碍物的净空值，便于调试
                if hasattr(self, 'get_terrain_height'):
                    terrain_clearance = current_pos[2] - agent_radius - self.get_terrain_height(current_pos[0], current_pos[1])
                    agent.debug_info['terrain_clearance'] = float(terrain_clearance)
                if getattr(self, '_obs_centers', None) is not None and self._obs_centers.shape[0] > 0:
                    current_pos_obs = agent.state.p_pos.astype(np.float32)
                    diff = self._obs_centers - current_pos_obs
                    d = np.sqrt(np.sum(diff * diff, axis=1))
                    obstacle_clearances = d - (self._obs_radii + agent_radius)
                    agent.debug_info['obstacle_clearances'] = obstacle_clearances.tolist()
            rew += clearance_reward
            agent.last_min_distance = d_min_current
            
# === 停滞惩罚/奖励 ===
            # 计算位置变化（仅考虑XY平面，避免Z轴动作导致“快速离开”）
            try:
                delta_pos_xy = (agent.state.p_pos - agent.last_position)[:2]
            except Exception:
                delta_pos_xy = np.zeros(2)
            pos_change = np.linalg.norm(delta_pos_xy)
            # 更新上次位置
            agent.last_position = agent.state.p_pos.copy()
            
            # 连续停滞惩罚（软化，按速度与与起点距离权重）
            v_mag = np.linalg.norm(agent.state.p_vel)
            try:
                dist_to_start = np.linalg.norm((agent.state.p_pos - agent.start_position)[:2])
            except Exception:
                dist_to_start = np.linalg.norm(agent.state.p_pos - agent.start_position)
            stall = _smoothstep(0.1 - v_mag, c=0.0, w=0.05)
            locality = _smoothstep(20.0 - dist_to_start, c=0.0, w=5.0)
            stationary_penalty = 1.5 * stall * (0.5 + 0.5 * locality)  # 🚨 修复Critic Loss：从5.0降到0.5，减少每步惩罚累积
            rew -= stationary_penalty
            if hasattr(agent, 'debug_info'):
                agent.debug_info['stationary_penalty_soft'] = float(stationary_penalty)
            
            # === 起始区域强制推动 === (新增)
            # 如果在起始区域附近，添加额外的离开起始点的强制性奖励
            dist_to_start = np.linalg.norm(agent.state.p_pos - agent.start_position)
            if dist_to_start < 20:
                # 构建一个从起始点指向智能体当前速度方向的奖励
                if np.linalg.norm(agent.state.p_vel) > 0.1:  # 有明确的移动方向
                    leave_start_reward = (20 - dist_to_start) * 0.5  # 越接近起点，奖励越大
                    rew += leave_start_reward
                    if hasattr(agent, 'debug_info'):
                        agent.debug_info['leave_start_reward'] = leave_start_reward
                
                # 弱化方向一致性奖励 - 不再强制要求方向一致，只要有移动即可获得一定奖励
                if hasattr(agent, 'start_to_goal_dir') and agent.start_to_goal_dir is not None:
                    if np.linalg.norm(agent.state.p_vel) > 0.1:  # 有明确的移动
                        vel_dir = agent.state.p_vel / np.linalg.norm(agent.state.p_vel)
                        # 计算与初始方向的一致性
                        init_alignment = np.dot(vel_dir, agent.start_to_goal_dir)
                        # 无论方向如何，只要有移动就给予基础奖励，方向一致则额外奖励
                        base_movement_reward = 2.0  # 基础移动奖励
                        alignment_bonus = init_alignment * 4.0 * (20 - dist_to_start) / 20.0  # 方向一致性额外奖励，从10.0减少到4.0
                        direction_bonus = base_movement_reward + (alignment_bonus if init_alignment > 0 else 0)
                        rew += direction_bonus
                        if hasattr(agent, 'debug_info'):
                            agent.debug_info['init_direction_bonus'] = direction_bonus
            
            # === 探索奖励 ===
            # 将连续空间离散化为网格（减小网格尺寸，提高精度）
            cell_size = 3.0  # 从5.0降低到3.0，增加网格数量
            x_cell = int(agent.state.p_pos[0] / cell_size)
            y_cell = int(agent.state.p_pos[1] / cell_size)
            z_cell = int(agent.state.p_pos[2] / cell_size)
            current_cell = (x_cell, y_cell, z_cell)
            
            # 如果访问了新的区域，给予更高奖励
            if current_cell not in agent.visited_cells:
                agent.visited_cells.add(current_cell)
                exploration_reward = 0.1  # 🚨 修复"刷分"问题：从0.5降到0.1，大幅降低绕圈探索的回报
                rew += exploration_reward
            
            # === 智能距离奖励系统 ===
            # 修复：移除每步的距离惩罚，改为基于距离变化的智能奖励
            # 区分"有意义的探索"和"无意义的远离"
            
            # 1. 接近目标的奖励（标准化后）
            if dist_change > 0:  # 接近目标
                approach_reward = dist_change * 10.0  # 降低权重从50.0到10.0，减少波动
                rew += approach_reward
                
                # 记录这次接近，用于后续路径评估
                if not hasattr(agent, 'recent_approach_history'):
                    agent.recent_approach_history = []
                agent.recent_approach_history.append(dist_change)
                # 只保留最近10次的接近记录
                if len(agent.recent_approach_history) > 10:
                    agent.recent_approach_history = agent.recent_approach_history[-10:]
            
            # 2. 远离目标的智能评估（软化）
            elif dist_change < -0.1:
                retreat_distance = abs(dist_change)
                v_norm = np.linalg.norm(agent.state.p_vel)
                vel_direction = agent.state.p_vel / (v_norm + 1e-6)
                goal_direction = (goal_pos - agent.state.p_pos) / (dist_to_goal + 1e-6)
                align = float(np.dot(vel_direction, goal_direction)) if v_norm > 1e-6 else 0.0
                away = _smoothstep(retreat_distance - 0.0, c=0.0, w=0.05)
                angle_w = _smoothstep(0.3 - align, c=0.0, w=0.2)
                retreat_penalty = 2.0 * away * (0.5 + 0.5 * angle_w)
                rew -= retreat_penalty
                if hasattr(agent, 'debug_info'):
                    agent.debug_info['retreat_penalty_soft'] = float(retreat_penalty)
            
            # 接近目标奖励（软化）
            k_close = 0.5
            k_vclose = 5.0
            proximity_bonus = k_close * _smoothstep(15.0 - dist_to_goal, c=0.0, w=5.0)
            proximity_bonus += k_vclose * _smoothstep(5.0 - dist_to_goal, c=0.0, w=1.5)
            rew += proximity_bonus
            if hasattr(agent, 'debug_info'):
                agent.debug_info['proximity_bonus_soft'] = float(proximity_bonus)
            
            # === 同心层级奖励（软化：Sigmoid 差分） ===
            try:
                ring_percentages = [0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.1]
                initial_d = max(1e-6, float(agent.initial_distance_to_goal))
                ring_thresholds = [p * initial_d for p in ring_percentages]
                ring_base_reward = float(os.getenv('RING_BASE_REWARD', '80.0'))
                soft_rings = 0.0
                for idx, thr in enumerate(ring_thresholds):
                    w = (len(ring_thresholds) - idx)
                    soft_rings += w * _smoothstep(thr - dist_to_goal, c=0.0, w=initial_d * 0.02)
                rew += soft_rings * (ring_base_reward / max(len(ring_thresholds), 1))
            except Exception:
                pass

            # === 高级路径寻找奖励机制 ===
            # 基于历史行为评估探索的有效性
            if hasattr(agent, 'recent_approach_history') and len(agent.recent_approach_history) >= 3:
                # 计算最近几次接近的平均效果
                recent_avg_approach = np.mean(agent.recent_approach_history[-3:])
                
                # 如果智能体最近有有效的接近行为，当前的远离可能是策略性的
                if recent_avg_approach > 0.5:  # 最近有显著接近
                    # 检查当前是否在探索新区域
                    if current_cell not in agent.visited_cells:
                        # 在有效接近后探索新区域，给予额外奖励
                        strategic_exploration_reward = 3.0
                        rew += strategic_exploration_reward
                        
                        if hasattr(agent, 'debug_info'):
                            agent.debug_info['strategic_exploration_reward'] = strategic_exploration_reward
            
            # === 长期路径效率奖励 ===
            # 如果智能体能够通过绕行最终更接近目标，给予奖励
            if not hasattr(agent, 'path_efficiency_history'):
                agent.path_efficiency_history = []
            
            # 记录当前距离
            agent.path_efficiency_history.append(dist_to_goal)
            # 只保留最近50步的历史
            if len(agent.path_efficiency_history) > 50:
                agent.path_efficiency_history = agent.path_efficiency_history[-50:]
            
            # 如果最近10步内距离有显著改善，给予路径效率奖励
            if len(agent.path_efficiency_history) >= 10:
                recent_distances = agent.path_efficiency_history[-10:]
                if len(recent_distances) >= 2:
                    distance_improvement = recent_distances[0] - recent_distances[-1]
                    if distance_improvement > 5.0:  # 显著改善
                        path_efficiency_reward = distance_improvement * 0.5
                        rew += path_efficiency_reward
                        
                        if hasattr(agent, 'debug_info'):
                            agent.debug_info['path_efficiency_reward'] = path_efficiency_reward
            
            # === 能量消耗计算 ===
            # 检查智能体是否有上一步速度记录
            if not hasattr(agent, 'last_velocity'):
                agent.last_velocity = np.zeros(3)
            
            # 计算速度变化（加速度）
            velocity_change = agent.state.p_vel - agent.last_velocity
            
            # 能量消耗与加速度平方成正比
            acceleration_magnitude = np.linalg.norm(velocity_change)
            energy_consumption = acceleration_magnitude ** 2 * 0.1
            
            # 根据距离变化确定能量消耗的效果
            # 仅在“明显远离目标”时施加惩罚；接近时且高度不高于理想上界，给予能量效率奖励
            # 理想高度上界（若未提供则降级为仅按接近判定）
            try:
                import os as _os_env
                ideal_max = float(_os_env.getenv('HEIGHT_IDEAL_MAX', '5.0'))
            except Exception:
                ideal_max = 5.0

            is_height_ok = True
            try:
                is_height_ok = (height_diff <= ideal_max)
            except Exception:
                pass

            if dist_change > 0 and is_height_ok:
                # 根据距离变化幅度调整能量奖励（效率奖励）
                distance_improvement_ratio = dist_change / (agent.last_goal_dist + 1e-6)  # 避免除零
                energy_reward = energy_consumption * 8.0 * distance_improvement_ratio
                rew += energy_reward
            elif dist_change < -0.1:
                # 明显远离目标：按远离比例惩罚
                distance_worsening_ratio = abs(dist_change) / (dist_to_goal + 1e-6)
                energy_penalty = energy_consumption * 8.0 * distance_worsening_ratio
                rew -= energy_penalty
            
            # 移除基于加速度的额外惩罚，改为基于距离的静止行为奖励/惩罚
            if hasattr(agent, 'debug_info'):
                agent.debug_info['acceleration_penalty'] = 0.0  # 保留字段但值为0
            
            # 记录当前速度作为下一步的上一步速度
            agent.last_velocity = agent.state.p_vel.copy()
            
            # === 方向一致性奖励（标准化后） ===
            if np.linalg.norm(agent.state.p_vel) > 0.3:  # 降低速度阈值，使更多的移动获得方向奖励
                # 速度方向
                vel_direction = agent.state.p_vel / np.linalg.norm(agent.state.p_vel)
                # 目标方向
                goal_direction = (goal_pos - agent.state.p_pos) / (dist_to_goal + 1e-6)
                # 一致性计算 (点积)
                alignment = np.dot(vel_direction, goal_direction)
                
                # 方向一致性奖励 - 降低权重，减少波动
                direction_reward = alignment * 1.0  # 从4.0降低到1.0
                rew += direction_reward
                
                # 减弱速度奖励 - 进一步降低
                speed_bonus = np.linalg.norm(agent.state.p_vel) * 0.2  # 从0.8降低到0.2
                rew += speed_bonus
                if hasattr(agent, 'debug_info'):
                    agent.debug_info['speed_bonus'] = speed_bonus
            
            # 为每个智能体记录更多信息，用于调试
            vel_magnitude = np.linalg.norm(agent.state.p_vel)
            if hasattr(agent, 'debug_info'):
                agent.debug_info.update({
                    'position': agent.state.p_pos.tolist(),
                    'velocity': agent.state.p_vel.tolist(),
                    'vel_magnitude': vel_magnitude,
                    'distance': dist_to_goal,
                    'distance_change': dist_change,
                    'initial_distance': agent.initial_distance_to_goal,
                    'distance_ratio': distance_ratio,
                    'terrain_height': terrain_height,
                    'height_diff': height_diff,
                    'reward': rew,
                    'energy_reward': energy_reward,
                    'energy_penalty': energy_penalty,
                    'exploration_reward': exploration_reward,
                    'dist_penalty': dist_penalty,
                    'obstacle_penalty': obstacle_penalty,
                })
                
            # === 简单奖励裁剪（无平滑） ===
            # 从环境变量获取参数
            # 直接裁剪奖励（使用缓存参数）
            rew_final = np.clip(rew, self.reward_clip_min, self.reward_clip_max)
            
            # 统计信息
            self.total_rewards = getattr(self, 'total_rewards', 0) + rew_final
            
            return rew_final
            
        except Exception as e:
            print(f"奖励计算异常: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # 出错时返回零奖励
    
    def is_collision(self, agent, entity):
        if (not getattr(agent, 'collide', True)) or (not getattr(entity, 'collide', True)):
            return False
        delta_pos = agent.state.p_pos - entity.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = agent.size + entity.size
        return dist < dist_min
    
    def observation(self, agent, world):
        """带 step 级缓存的 observation 包装，避免同一步对所有 agent 重复重算。"""
        try:
            cache_key = (id(world), int(getattr(world, 'current_step', -1)))
        except Exception:
            cache_key = None

        if cache_key is not None and self._obs_step_cache_key == cache_key:
            cached_obs = self._obs_step_cache.get(id(agent))
            if cached_obs is not None:
                return cached_obs

        try:
            agents = getattr(world, 'agents', None)
            if cache_key is not None and agents:
                cache = {}
                batch_obs = self._compute_observations_batch_uncached(world)
                if batch_obs:
                    cache.update(batch_obs)
                else:
                    for ag in agents:
                        cache[id(ag)] = self._compute_observation_uncached(ag, world)
                self._obs_step_cache_key = cache_key
                self._obs_step_cache = cache
                cached_obs = cache.get(id(agent))
                if cached_obs is not None:
                    return cached_obs
        except Exception:
            self._obs_step_cache_key = None
            self._obs_step_cache = {}

        return self._compute_observation_uncached(agent, world)

    def _compute_observations_batch_uncached(self, world):
        """批量构建当前 world 所有智能体的 observation，减少重复地形/障碍查询。"""
        agents = getattr(world, 'agents', None)
        if not agents:
            return {}

        num_agents = len(agents)
        if num_agents <= 0:
            return {}

        try:
            if self._obs_env_info is None or self._obs_env_info_agent_count != num_agents:
                self._refresh_observation_static_cache(num_agents=num_agents)

            positions = np.asarray([ag.state.p_pos for ag in agents], dtype=np.float32)
            velocities = np.asarray([ag.state.p_vel for ag in agents], dtype=np.float32)
            accelerations = np.asarray(
                [getattr(ag.state, 'p_acc', np.zeros(3, dtype=np.float32)) for ag in agents],
                dtype=np.float32,
            )
            max_speeds = np.asarray(
                [max(float(getattr(ag, 'max_speed', 22.5)), 1e-6) for ag in agents],
                dtype=np.float32,
            )

            # 1. 状态信息 (9维)
            map_half = max(self.map_size * 0.5, 1e-6)
            normalized_pos = positions / map_half - 1.0
            normalized_vel = velocities / max_speeds[:, None]
            normalized_acc = np.clip(accelerations / 10.0, -1.0, 1.0)
            state_info = np.concatenate([normalized_pos, normalized_vel, normalized_acc], axis=1).astype(np.float32)

            # 2. 目标信息 (7维)
            goal_info = np.zeros((num_agents, 7), dtype=np.float32)
            for idx, ag in enumerate(agents):
                try:
                    if hasattr(ag, 'goal_a') and ag.goal_a.state.p_pos is not None:
                        goal_pos = np.asarray(ag.goal_a.state.p_pos, dtype=np.float32)
                        goal_rel_pos = goal_pos - positions[idx]
                        dist_to_goal = float(np.linalg.norm(goal_rel_pos))
                        norm_direction = goal_rel_pos / (dist_to_goal + 1e-6)
                        normalized_goal_pos = goal_pos / map_half - 1.0
                        goal_info[idx, :3] = norm_direction
                        goal_info[idx, 3] = dist_to_goal / max(self.map_size, 1e-6)
                        goal_info[idx, 4:] = normalized_goal_pos
                except Exception:
                    continue

            # 3. 地形信息 (32维)
            terrain_info = np.zeros((num_agents, 32), dtype=np.float32)
            try:
                if hasattr(self, 'batch_get_terrain_height'):
                    current_xy = positions[:, :2].astype(np.float32, copy=False)
                    map_max = np.float32(self.map_size - 1)

                    sample_coords = np.zeros((num_agents, 29, 2), dtype=np.float32)
                    sample_coords[:, 0, :] = current_xy
                    sample_coords[:, 1:5, :] = np.clip(
                        current_xy[:, None, :] + self._obs_gradient_offsets[None, :, :],
                        0.0,
                        map_max,
                    )

                    vel_norm = np.linalg.norm(velocities, axis=1, keepdims=True)
                    vel_dir3 = np.zeros_like(velocities, dtype=np.float32)
                    vel_dir3[:, 0] = 1.0
                    moving_mask = vel_norm[:, 0] > 1e-6
                    if np.any(moving_mask):
                        vel_dir3[moving_mask] = velocities[moving_mask] / vel_norm[moving_mask]

                    forward_raw = current_xy[:, None, :] + (
                        self._obs_forward_distances[None, :, None] * vel_dir3[:, None, :2]
                    )
                    forward_valid = np.logical_and.reduce((
                        forward_raw[:, :, 0] >= 0.0,
                        forward_raw[:, :, 0] < self.map_size,
                        forward_raw[:, :, 1] >= 0.0,
                        forward_raw[:, :, 1] < self.map_size,
                    ))
                    sample_coords[:, 5:13, :] = np.where(forward_valid[:, :, None], forward_raw, 0.0)

                    near_raw = current_xy[:, None, :] + self._obs_direction_pairs[None, :, :] * self._obs_near_distance
                    far_raw = current_xy[:, None, :] + self._obs_direction_pairs[None, :, :] * self._obs_far_distance
                    near_valid = np.logical_and.reduce((
                        near_raw[:, :, 0] >= 0.0,
                        near_raw[:, :, 0] < self.map_size,
                        near_raw[:, :, 1] >= 0.0,
                        near_raw[:, :, 1] < self.map_size,
                    ))
                    far_valid = np.logical_and.reduce((
                        far_raw[:, :, 0] >= 0.0,
                        far_raw[:, :, 0] < self.map_size,
                        far_raw[:, :, 1] >= 0.0,
                        far_raw[:, :, 1] < self.map_size,
                    ))
                    near_coords = np.where(near_valid[:, :, None], near_raw, 0.0)
                    far_coords = np.where(far_valid[:, :, None], far_raw, 0.0)
                    sample_coords[:, 13:29:2, :] = near_coords
                    sample_coords[:, 14:29:2, :] = far_coords

                    all_heights = self.batch_get_terrain_height(sample_coords.reshape(-1, 2)).reshape(num_agents, 29)
                    current_height = all_heights[:, 0]
                    grad_heights = all_heights[:, 1:5]
                    forward_heights = all_heights[:, 5:13]
                    surround_heights = all_heights[:, 13:29]

                    terrain_info[:, 0] = (positions[:, 2] - current_height) / 20.0
                    terrain_info[:, 1] = current_height / 100.0
                    terrain_info[:, 2] = (grad_heights[:, 0] - current_height) / 10.0
                    terrain_info[:, 3] = (grad_heights[:, 1] - current_height) / 10.0
                    terrain_info[:, 4] = (grad_heights[:, 2] - current_height) / 10.0
                    terrain_info[:, 5] = (grad_heights[:, 3] - current_height) / 10.0
                    terrain_info[:, 6:14] = np.where(forward_valid, forward_heights / 100.0, 0.0).astype(np.float32)

                    surround_valid = np.empty((num_agents, 16), dtype=bool)
                    surround_valid[:, 0::2] = near_valid
                    surround_valid[:, 1::2] = far_valid
                    terrain_info[:, 14:30] = np.where(
                        surround_valid,
                        surround_heights / 100.0,
                        0.0,
                    ).astype(np.float32)

                    complexity_valid_counts = np.sum(near_valid, axis=1)
                    terrain_complexity = np.zeros((num_agents,), dtype=np.float32)
                    valid_complexity_rows = complexity_valid_counts > 1
                    if np.any(valid_complexity_rows):
                        near_heights = surround_heights[:, 0::2].astype(np.float32, copy=False)
                        valid_heights = np.where(near_valid, near_heights, 0.0)
                        valid_counts_f = np.maximum(complexity_valid_counts.astype(np.float32), 1.0)
                        mean_heights = np.sum(valid_heights, axis=1) / valid_counts_f
                        centered = np.where(near_valid, near_heights - mean_heights[:, None], 0.0)
                        terrain_complexity[valid_complexity_rows] = (
                            np.sqrt(np.sum(centered[valid_complexity_rows] * centered[valid_complexity_rows], axis=1)
                                    / valid_counts_f[valid_complexity_rows]) / 20.0
                        ).astype(np.float32)
                    terrain_info[:, 30] = terrain_complexity

                    h_right = grad_heights[:, 0]
                    h_left = grad_heights[:, 2]
                    h_forward = grad_heights[:, 1]
                    h_back = grad_heights[:, 3]
                    px = (h_right - h_left) / 6.0
                    py = (h_forward - h_back) / 6.0
                    dz = positions[:, 2] - current_height
                    d_n = dz / np.sqrt(1.0 + px * px + py * py)
                    terrain_info[:, 31] = np.clip(d_n / 20.0, -1.0, 1.0).astype(np.float32)
            except Exception as e:
                print(f"地形信息批量计算错误: {e}")
                terrain_info.fill(0.0)

            # 4. 障碍物信息 (15维)
            obstacle_info = np.zeros((num_agents, 15), dtype=np.float32)
            obstacle_info[:, 3::5] = 1.0
            try:
                max_dist = max(self.map_size, 1e-6)
                if getattr(self, '_obs_centers', None) is not None and self._obs_centers.shape[0] > 0:
                    rel_positions = self._obs_centers[None, :, :] - positions[:, None, :]
                    dists_to_center = np.linalg.norm(rel_positions, axis=2)
                    dists_to_surface = np.maximum(0.0, dists_to_center - self._obs_radii[None, :])
                    obstacle_count = dists_to_surface.shape[1]
                    top_k = min(3, obstacle_count)
                    if obstacle_count > top_k:
                        nearest_indices = np.argpartition(dists_to_surface, kth=top_k - 1, axis=1)[:, :top_k]
                        nearest_dists = np.take_along_axis(dists_to_surface, nearest_indices, axis=1)
                        sorted_indices = np.take_along_axis(
                            nearest_indices,
                            np.argsort(nearest_dists, axis=1),
                            axis=1,
                        )
                    else:
                        sorted_indices = np.argsort(dists_to_surface, axis=1)
                    row_idx = np.arange(num_agents)
                    for k in range(top_k):
                        idx = sorted_indices[:, k]
                        offset = k * 5
                        obstacle_info[:, offset:offset + 3] = rel_positions[row_idx, idx] / max_dist
                        obstacle_info[:, offset + 3] = np.minimum(dists_to_surface[row_idx, idx] / max_dist, 1.0)
                        obstacle_info[:, offset + 4] = np.minimum(self._obs_radii[idx] / max_dist, 1.0)
            except Exception as e:
                print(f"障碍物信息批量计算错误: {e}")
                obstacle_info.fill(0.0)
                obstacle_info[:, 3::5] = 1.0

            # 5. 其他智能体信息 (12维)
            other_agents_obs = np.zeros((num_agents, 12), dtype=np.float32)
            try:
                for idx in range(num_agents):
                    other_indices = [j for j in range(num_agents) if j != idx]
                    if len(other_indices) >= 2:
                        for slot, other_idx in enumerate(other_indices[:2]):
                            base = slot * 6
                            other_agents_obs[idx, base:base + 3] = (
                                (positions[other_idx] - positions[idx]) / 100.0
                            )
                            other_agents_obs[idx, base + 3:base + 6] = velocities[other_idx] / max_speeds[idx]
                    elif len(other_indices) == 1:
                        other_idx = other_indices[0]
                        rel_pos = (positions[other_idx] - positions[idx]) / 100.0
                        rel_vel = velocities[other_idx] / max_speeds[idx]
                        other_agents_obs[idx, 0:3] = rel_pos
                        other_agents_obs[idx, 3:6] = rel_vel
                        other_agents_obs[idx, 6:9] = rel_pos
                        other_agents_obs[idx, 9:12] = rel_vel
            except Exception as e:
                print(f"其他智能体信息批量计算错误: {e}")
                other_agents_obs.fill(0.0)

            env_info = self._obs_env_info.astype(np.float32, copy=False)
            obs = np.concatenate(
                [
                    state_info,
                    terrain_info,
                    obstacle_info,
                    goal_info,
                    other_agents_obs,
                    np.broadcast_to(env_info[None, :], (num_agents, env_info.shape[0])),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            return {id(ag): obs[idx] for idx, ag in enumerate(agents)}
        except Exception:
            return {}

    def _compute_observation_uncached(self, agent, world):
        """构建81维观察值 - 🚨 关键修复：目标信息从4维增加到7维（添加绝对位置3维），总维度从75→81"""
        
        # 1. 完整状态信息 (9维) - 位置、速度、加速度 [🔧 已添加归一化]
        state_info = []
        try:
            # 位置信息 (3维) - 🔧 归一化到[-1, 1]
            position = agent.state.p_pos
            map_half = max(self.map_size * 0.5, 1e-6)
            normalized_pos = position / map_half - 1.0  # 地图尺寸映射到[-1, 1]
            
            # 速度信息 (3维) - 🔧 归一化到[-1, 1]
            # 🔧 关键修复：使用实际的agent.max_speed而不是硬编码值，确保训练和评估一致
            velocity = agent.state.p_vel
            max_speed = getattr(agent, 'max_speed', 22.5)  # 默认值22.5用于向后兼容
            if max_speed <= 0:
                max_speed = 22.5  # 防止除零错误
            normalized_vel = velocity / max_speed  # 使用实际的max_speed进行归一化
            
            # 加速度信息 (3维) - 🔧 归一化并裁剪到[-1, 1]
            if hasattr(agent.state, 'p_acc'):
                acceleration = agent.state.p_acc
                normalized_acc = np.clip(acceleration / 10.0, -1.0, 1.0)  # 假设最大加速度10
            else:
                normalized_acc = np.zeros(3)
            
            # 🔧 使用归一化后的值
            state_info = np.concatenate([normalized_pos, normalized_vel, normalized_acc])
        except Exception as e:
            print(f"状态信息获取错误: {e}")
            state_info = np.zeros(9)  # 9维：位置3 + 速度3 + 加速度3
        
        # 2. 目标信息 (7维) - 🚨 关键修复：添加目标绝对位置（3维），从4维扩展到7维
        # 原因：势场代码从obs[57:60]读取目标绝对位置，但之前只有4维目标信息，导致读取到错误数据
        goal_info = []
        try:
            if hasattr(agent, 'goal_a') and agent.goal_a.state.p_pos is not None:
                goal_pos = agent.goal_a.state.p_pos
                # 计算到目标的向量和距离
                goal_rel_pos = goal_pos - agent.state.p_pos
                dist_to_goal = np.linalg.norm(goal_rel_pos)
                # 归一化方向向量
                norm_direction = goal_rel_pos / (dist_to_goal + 1e-6)
                
                # 🚨 关键：归一化目标绝对位置（与智能体位置归一化方式一致）
                map_half = max(self.map_size * 0.5, 1e-6)
                normalized_goal_pos = goal_pos / map_half - 1.0  # 归一化到[-1, 1]
                
                # 目标信息 (7维)
                goal_info = np.concatenate([
                    norm_direction * 1.0,  # 3维：方向向量
                    [dist_to_goal / max(self.map_size, 1e-6)],  # 1维：归一化距离
                    normalized_goal_pos  # 🚨 新增：3维目标绝对位置（归一化）
                ])
                
                # 减少调试信息输出，避免并行环境信息过多
                # if hasattr(agent, '_debug_printed') == False:
                #     print(f"DEBUG: 智能体观察值 - 目标位置: {goal_pos}")
                #     print(f"DEBUG: 智能体位置: {agent.state.p_pos}")
                #     print(f"DEBUG: 到目标距离: {dist_to_goal:.2f}")
                #     print(f"DEBUG: 目标方向向量: {norm_direction}")
                #     print(f"DEBUG: 目标信息(前4维): {goal_info[:4]}")
                #     agent._debug_printed = True
            else:
                # 如果没有目标位置，提供零向量作为目标信息
                goal_info = np.zeros(7)  # 🚨 修复：3维方向 + 1维距离 + 3维绝对位置 = 7维
        except Exception as e:
            # 只在出现错误时输出信息，并显示环境ID
            env_id = getattr(world, 'env_id', 'unknown') if 'world' in locals() else 'unknown'
            print(f"[环境{env_id}] 目标信息计算错误: {e}")
            goal_info = np.zeros(7)  # 🚨 修复：7维
        
        # 3. 扩展地形信息 (32维) - 🚀 使用批量地形采样优化性能
        terrain_info = np.zeros(32, dtype=np.float32)
        try:
            if hasattr(self, 'batch_get_terrain_height'):
                current_x, current_y = agent.state.p_pos[0], agent.state.p_pos[1]
                current_xy = np.asarray([current_x, current_y], dtype=np.float32)
                map_max = np.float32(self.map_size - 1)
                
                # 🚀 预分配采样坐标，避免 list append + np.array 二次构造
                sample_coords = np.zeros((29, 2), dtype=np.float32)
                sample_coords[0] = current_xy
                idx_current = 0
                idx_gradient = slice(1, 5)
                idx_forward = slice(5, 13)
                idx_surround = slice(13, 29)

                grad_coords = np.clip(current_xy + self._obs_gradient_offsets, 0.0, map_max)
                sample_coords[idx_gradient] = grad_coords
                
                # 计算速度方向（用于前方探测）
                vel_dir3 = agent.state.p_vel
                vel_norm = np.linalg.norm(vel_dir3)
                if vel_norm > 1e-6:
                    vel_dir3 = vel_dir3 / vel_norm
                else:
                    vel_dir3 = np.array([1.0, 0.0, 0.0])
                
                # 索引5-12：前方8个探测点
                forward_offsets = self._obs_forward_distances[:, None] * np.asarray(vel_dir3[:2], dtype=np.float32)[None, :]
                forward_raw = current_xy[None, :] + forward_offsets
                forward_valid = np.logical_and.reduce((
                    forward_raw[:, 0] >= 0.0,
                        forward_raw[:, 0] < self.map_size,
                        forward_raw[:, 1] >= 0.0,
                        forward_raw[:, 1] < self.map_size,
                ))
                forward_coords = np.where(forward_valid[:, None], forward_raw, 0.0)
                sample_coords[idx_forward] = forward_coords
                
                # 索引13-28：周围16个探测点（8方向×2距离层）
                near_raw = current_xy[None, :] + self._obs_direction_pairs * self._obs_near_distance
                far_raw = current_xy[None, :] + self._obs_direction_pairs * self._obs_far_distance
                near_valid = np.logical_and.reduce((
                    near_raw[:, 0] >= 0.0,
                    near_raw[:, 0] < self.map_size,
                    near_raw[:, 1] >= 0.0,
                    near_raw[:, 1] < self.map_size,
                ))
                far_valid = np.logical_and.reduce((
                    far_raw[:, 0] >= 0.0,
                    far_raw[:, 0] < self.map_size,
                    far_raw[:, 1] >= 0.0,
                    far_raw[:, 1] < self.map_size,
                ))
                near_coords = np.where(near_valid[:, None], near_raw, 0.0)
                far_coords = np.where(far_valid[:, None], far_raw, 0.0)
                surround_coords = np.empty((16, 2), dtype=np.float32)
                surround_coords[0::2] = near_coords
                surround_coords[1::2] = far_coords
                surround_valid = np.empty(16, dtype=bool)
                surround_valid[0::2] = near_valid
                surround_valid[1::2] = far_valid
                sample_coords[idx_surround] = surround_coords
                
                # 🚀 批量查询：一次性获取所有地形高度
                all_heights = self.batch_get_terrain_height(sample_coords)
                
                # 提取各部分高度
                current_height = all_heights[idx_current]
                grad_heights = all_heights[idx_gradient]
                forward_heights = all_heights[idx_forward]
                surround_heights = all_heights[idx_surround]
                complexity_heights = surround_heights[0::2]
                complexity_valid = near_valid
                normal_heights = np.asarray(
                    [grad_heights[0], grad_heights[2], grad_heights[1], grad_heights[3]],
                    dtype=np.float32,
                )
                
                # 构建地形信息特征（与原逻辑完全一致）
                # 1. 相对高度 (1维)
                relative_height = agent.state.p_pos[2] - current_height
                terrain_info[0] = relative_height / 20.0
                
                # 2. 当前地形高度 (1维)
                terrain_info[1] = current_height / 100.0
                
                # 3. 地形梯度信息 (4维)
                dx1 = grad_heights[0] - current_height
                dy1 = grad_heights[1] - current_height
                dx2 = grad_heights[2] - current_height
                dy2 = grad_heights[3] - current_height
                terrain_info[2:6] = np.asarray([dx1 / 10.0, dy1 / 10.0, dx2 / 10.0, dy2 / 10.0], dtype=np.float32)
                
                # 4. 前方地形信息 (8维)
                terrain_info[6:14] = np.where(forward_valid, forward_heights / 100.0, 0.0).astype(np.float32)
                
                # 5. 周围地形信息 (16维)
                terrain_info[14:30] = np.where(surround_valid, surround_heights / 100.0, 0.0).astype(np.float32)
                
                # 6. 地形复杂度信息 (1维)
                valid_complexity_heights = complexity_heights[complexity_valid]
                if valid_complexity_heights.size > 1:
                    terrain_complexity = np.std(valid_complexity_heights) / 20.0
                else:
                    terrain_complexity = 0.0
                terrain_info[30] = terrain_complexity
                
                # 7. 法向净空 (1维)
                agent_z = agent.state.p_pos[2]
                h_right, h_left, h_forward, h_back = normal_heights
                px = (h_right - h_left) / (2 * 3.0)  # X方向坡度
                py = (h_forward - h_back) / (2 * 3.0)  # Y方向坡度
                dz = agent_z - current_height  # 高度差
                d_n = dz / np.sqrt(1 + px*px + py*py)  # 法向净空
                normal_clearance = np.clip(d_n / 20.0, -1.0, 1.0)
                terrain_info[31] = normal_clearance
                
            else:
                terrain_info = np.zeros(32)
        except Exception as e:
            print(f"地形信息计算错误: {e}")
            import traceback
            traceback.print_exc()
            terrain_info = np.zeros(32)
        
        # 4. 障碍物信息 (15维) - Top-3最近障碍物编码
        # 编码结构：3个最近障碍物 × 5维(归一化相对位置3 + 归一化表面距离1 + 归一化半径1) = 15维
        # 每个障碍物槽位(5维)：
        #   [0] rel_x / max_dist : 归一化X方向相对位置（智能体→障碍物中心）
        #   [1] rel_y / max_dist : 归一化Y方向相对位置
        #   [2] rel_z / max_dist : 归一化Z方向相对位置
        #   [3] surface_dist / max_dist : 归一化表面距离（到障碍物表面的最短距离）
        #   [4] radius / max_dist : 归一化障碍物半径
        # 按表面距离升序排列（最近→最远），不足3个时用sentinel (0,0,0,1,0)填充
        # 优点：信息密度高（15维全部携带有效障碍物信息），PF修正可直接使用无需重建
        obstacle_info = np.zeros(15, dtype=np.float32)
        try:
            agent_pos = agent.state.p_pos
            max_dist = max(self.map_size, 1e-6)
            
            if getattr(self, '_obs_centers', None) is not None and self._obs_centers.shape[0] > 0:
                rel_positions = self._obs_centers - agent_pos          # (N, 3)
                dists_to_center = np.linalg.norm(rel_positions, axis=1)  # (N,)
                dists_to_surface = np.maximum(0.0, dists_to_center - self._obs_radii)  # (N,)
                
                sorted_indices = np.argsort(dists_to_surface)
                top_k = min(3, len(sorted_indices))
                
                for k in range(top_k):
                    idx = sorted_indices[k]
                    offset = k * 5
                    obstacle_info[offset:offset+3] = rel_positions[idx] / max_dist
                    obstacle_info[offset+3] = min(dists_to_surface[idx] / max_dist, 1.0)
                    obstacle_info[offset+4] = min(self._obs_radii[idx] / max_dist, 1.0)
                
                for k in range(top_k, 3):
                    offset = k * 5
                    obstacle_info[offset:offset+3] = 0.0
                    obstacle_info[offset+3] = 1.0   # sentinel: 最大距离 = 无障碍物
                    obstacle_info[offset+4] = 0.0
            else:
                for k in range(3):
                    offset = k * 5
                    obstacle_info[offset+3] = 1.0   # sentinel
            
        except Exception as e:
            print(f"障碍物信息计算错误: {e}")
            obstacle_info = np.zeros(15, dtype=np.float32)
        
        # 5. 其他智能体信息 (12维) - 修复：确保固定12维输出
        other_agents_obs = np.zeros(12, dtype=np.float32)
        try:
            # 固定处理3个智能体的情况
            other_agents = [a for a in world.agents if a is not agent]
            
            # 确保总是有2个其他智能体（3个智能体总数）
            if len(other_agents) == 2:
                for idx, other in enumerate(other_agents):
                    # 🔧 归一化相对位置到约[-2, 2]
                    rel_pos = other.state.p_pos - agent.state.p_pos
                    normalized_rel_pos = rel_pos / 100.0
                    base = idx * 6
                    other_agents_obs[base:base + 3] = normalized_rel_pos
                    
                    # 🔧 归一化速度到[-1, 1]
                    # 🔧 关键修复：使用实际的agent.max_speed而不是硬编码值
                    max_speed = getattr(agent, 'max_speed', 22.5)
                    if max_speed <= 0:
                        max_speed = 22.5
                    normalized_vel = other.state.p_vel / max_speed
                    other_agents_obs[base + 3:base + 6] = normalized_vel
            elif len(other_agents) == 1:
                # 只有一个其他智能体，重复一次
                other = other_agents[0]
                # 🔧 归一化相对位置和速度
                # 🔧 关键修复：使用实际的agent.max_speed而不是硬编码值
                rel_pos = other.state.p_pos - agent.state.p_pos
                normalized_rel_pos = rel_pos / 100.0
                max_speed = getattr(agent, 'max_speed', 22.5)
                if max_speed <= 0:
                    max_speed = 22.5
                normalized_vel = other.state.p_vel / max_speed
                other_agents_obs[0:3] = normalized_rel_pos
                other_agents_obs[3:6] = normalized_vel
                other_agents_obs[6:9] = normalized_rel_pos
                other_agents_obs[9:12] = normalized_vel
        except Exception as e:
            print(f"其他智能体信息计算错误: {e}")
            other_agents_obs = np.zeros(12)  # 固定12维
        
        # 6. 环境状态信息 (6维) - 新增
        if self._obs_env_info is None or self._obs_env_info_agent_count != len(world.agents):
            self._refresh_observation_static_cache(num_agents=len(world.agents))
        env_info = self._obs_env_info  # 段落标记位（固定常量，方便解析）
        
        # 合并所有观察信息（目标结构：9 + 32 + 15 + 7 + 12 + 6 = 81维）
        obs_components = [
            state_info,       # 9维：位置、速度、加速度
            terrain_info,     # 32维：扩展地形信息
            obstacle_info,    # 15维：障碍物信息（Top-3最近障碍物 × 5维：相对位置3 + 表面距离1 + 半径1）
            goal_info,        # 7维：目标信息（方向3 + 距离1 + 绝对位置3）
            other_agents_obs, # 12维：其他智能体信息
            env_info,         # 6维：环境信息 / 段落标记
        ]
        
        # 🔧 调试：验证各组件维度和内容（仅在reset_world完成后的首次调用时打印）
        if not hasattr(agent, '_obs_dim_checked') and hasattr(world, '_reset_completed') and world._reset_completed:
            actual_dims = [len(comp) if hasattr(comp, '__len__') else 1 for comp in obs_components]
            expected_dims = [9, 32, 15, 7, 12, 6]
            if actual_dims != expected_dims:
                print(f"⚠️ 观测维度不匹配！期望{expected_dims}，实际{actual_dims}")
                print(f"  state_info: {len(state_info)}, terrain_info: {len(terrain_info)}, obstacle_info: {len(obstacle_info)}")
                print(f"  goal_info: {len(goal_info)}, other_agents_obs: {len(other_agents_obs)}, env_info: {len(env_info)}")
            # 打印目标信息内容用于调试
            print(f"🎯 [{agent.name}] 目标信息调试:")
            print(f"  智能体位置: {agent.state.p_pos}")
            print(f"  目标位置: {agent.goal_a.state.p_pos if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and agent.goal_a.state.p_pos is not None else 'None'}")
            print(f"  goal_info内容: {goal_info}")
            print(f"  state_info前3维(归一化位置): {state_info[:3]}")
            agent._obs_dim_checked = True
        
        # 拼接所有组件
        try:
            obs = np.concatenate(obs_components)
        except Exception as e:
            print(f"观察值合并错误: {e}")
            # 🚨 修复：观察维度从75维增加到81维（目标信息从4维增加到7维，地形从21维增加到32维）
            obs = np.zeros(getattr(self, 'observation_dim', 81), dtype=np.float32)
        
        # 确保观察空间维度一致 - 填充到81维
        # 🚨 关键修复：目标信息从4维增加到7维（添加绝对位置3维），总维度从75→81
        expected_dim = int(getattr(self, 'observation_dim', 81))
        if len(obs) < expected_dim:
            obs = np.pad(obs, (0, expected_dim - len(obs)), 'constant')
        elif len(obs) > expected_dim:
            obs = obs[:expected_dim]
        
        # 确保返回的是numpy数组而不是列表或嵌套结构
        if not isinstance(obs, np.ndarray):
            try:
                obs = np.array(obs)
            except:
                # 最后的后备方案，创建一个零向量
                # 🔧 修复：观察维度从75维增加到81维（目标信息+3维绝对位置，地形信息从21维增加到32维）
                obs = np.zeros(expected_dim, dtype=np.float32)
        
        # 验证形状正确
        if len(obs.shape) > 1:
            print(f"警告: 观察值形状不正确: {obs.shape}，尝试修复")
            # 尝试展平或重塑
            try:
                obs = obs.flatten()[:expected_dim]
                if len(obs) < expected_dim:
                    obs = np.pad(obs, (0, expected_dim - len(obs)), 'constant')
            except:
                # 最终后备方案
                obs = np.zeros(expected_dim, dtype=np.float32)
        
        # 最后的检查确保返回正确形状和类型的数组
        if not isinstance(obs, np.ndarray) or obs.shape != (expected_dim,):
            print(f"严重警告: 观察值仍有问题: 类型={type(obs)}, 形状={getattr(obs, 'shape', 'unknown')}")
            obs = np.zeros(expected_dim, dtype=np.float32)
        
        # 最终的安全检查
        try:
            if obs is None or len(obs) != expected_dim:
                print(f"最终安全检查失败: 观察值长度 = {len(obs) if obs is not None else 'None'}")
                obs = np.zeros(expected_dim, dtype=np.float32)
        except Exception as e:
            print(f"最终观察值验证出错: {e}")
            obs = np.zeros(expected_dim, dtype=np.float32)
        
        # 确保返回正确的numpy数组
        if not isinstance(obs, np.ndarray) or obs.shape != (expected_dim,):
            obs = np.zeros(expected_dim, dtype=np.float32)
        
        # 🔧 最终安全裁剪：防止极端值影响训练
        obs = np.clip(obs, -10.0, 10.0)
            
        return obs

    def is_done(self, agent, world):
        """
        判断回合是否结束。
        根据用户定义的需求：
        1. 失败条件（立即终止）:
           - 任何智能体飞出边界 (is_within_bounds)
           - 任何智能体发生碰撞 (agent.collide)
        2. 成功条件（立即终止）:
           - 所有智能体都到达了各自的目标附近
        """
        # 初始化完成标记字典（避免重复打印）
        if not hasattr(self, '_agent_done_logged'):
            self._agent_done_logged = {}
        
        # 获取智能体标识符
        agent_key = getattr(agent, 'name', f'agent_{id(agent)}')
        
        # 失败条件1: 飞出边界
        if not world.is_within_bounds(agent.state.p_pos):
            if not self._agent_done_logged.get(agent_key, False):
                self._agent_done_logged[agent_key] = True
                # 🔧 关键修复：设置终止原因，供_get_done使用
                if not hasattr(world, '_termination_reasons'):
                    world._termination_reasons = {}
                agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
                world._termination_reasons[agent_name] = ["越界"]
                # 打印一次边界终止信息（可选）
            return True

        # 失败条件2: 非目标附近地形落地/穿透 或 近距离碰撞（统一逻辑，优先于穿透）
        try:
            pos = agent.state.p_pos
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            terrain_h = self.get_terrain_height(x, y)
            agent_radius = float(max(0.0, getattr(agent, 'size', getattr(self, 'agent_size', 0.0))))
            body_bottom_z = z - agent_radius
            # 目标距离
            if hasattr(self, 'goal_pos') and self.goal_pos is not None:
                gx, gy, gz = float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2])
                dist_to_goal = ((x-gx)**2 + (y-gy)**2 + (z-gz)**2) ** 0.5
            else:
                dist_to_goal = 1e9
            # 触地/穿透的提前终止缓冲（可通过环境变量覆盖）
            try:
                import os
                eps = float(os.getenv('TERRAIN_CONTACT_EPS', '0.3'))  # 🔧 统一阈值：从0.03提高到0.3米，与shell脚本保持一致
            except Exception:
                eps = 0.3  # 🔧 修复：失败时也使用0.3米，与shell脚本默认值保持一致
            thr_succ = getattr(self, 'success_distance_threshold', 2.0)
            
            # 🚨 修复侧面穿透：改进为3D碰撞检测，而非只检查Z轴触底
            # 旧逻辑：只检查 z <= terrain_h（只能检测从上往下的穿透）
            # 新逻辑：检查 z < terrain_h（真正在地形下方）
            if body_bottom_z < terrain_h - eps:  # 机体底部低于地形高度 → 真正穿透
                # 只在首次判定时打印
                if not self._agent_done_logged.get(agent_key, False):
                    self._agent_done_logged[agent_key] = True
                    # 🔧 关键修复：设置终止原因，供_get_done使用
                    if not hasattr(world, '_termination_reasons'):
                        world._termination_reasons = {}
                    agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
                    world._termination_reasons[agent_name] = ["地形穿透"]
                    try:
                        step_idx = int(getattr(world, 'current_step', -1))
                        ep_len = int(getattr(world, 'episode_length', -1))
                        depth = terrain_h - body_bottom_z
                        print(f"[终止] 地形穿透 | step={step_idx}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({x:.2f},{y:.2f},{z:.2f}) | body_bottom_z={body_bottom_z:.2f} | terrain_h={terrain_h:.2f} | 穿透深度={depth:.2f}m")
                    except Exception:
                        pass
                return True
            # 近距离接触障碍/实体
            try:
                dmin = None
                if hasattr(world, 'landmarks'):
                    for landmark in world.landmarks:
                        if not getattr(landmark, 'collide', False):
                            continue
                        lp = getattr(getattr(landmark, 'state', None), 'p_pos', None)
                        if lp is None:
                            continue
                        r = float(getattr(landmark, 'size', 0.0)) + float(getattr(agent, 'size', 0.0))
                        d = float(np.linalg.norm(pos - lp) - r)
                        dmin = d if dmin is None else min(dmin, d)
                thr_coll = float(getattr(self, 'collision_distance_threshold', 0.5))
                if dmin is not None and dmin <= thr_coll:
                    # 只在首次判定时打印
                    if not self._agent_done_logged.get(agent_key, False):
                        self._agent_done_logged[agent_key] = True
                        # 🔧 关键修复：设置终止原因，供_get_done使用
                        if not hasattr(world, '_termination_reasons'):
                            world._termination_reasons = {}
                        agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
                        world._termination_reasons[agent_name] = ["实体碰撞"]
                        try:
                            step_idx = int(getattr(world, 'current_step', -1))
                            ep_len = int(getattr(world, 'episode_length', -1))
                            print(f"[终止] 实体碰撞 | step={step_idx}/{ep_len} | agent={getattr(agent,'name','?')} | pos=({x:.2f},{y:.2f},{z:.2f}) | min_dist={dmin:.3f}")
                        except Exception:
                            pass
                    return True
            except Exception:
                pass
        except Exception:
            pass

        # 成功条件: 所有智能体都到达目标
        # 这个检查应该在所有智能体都评估过之后进行，这里只检查当前智能体
        # 实际的 "all agents done" 逻辑需要在 MultiAgentEnv 中处理
        return False

    def find_peak_positions(self, min_height=None, neighborhood_size=10, max_peaks=5):
        """寻找地形中的真正山顶位置（局部最高点）"""
        # 确保地形已经生成
        if self.terrain is None:
            # 返回地图中心作为默认山顶
            default_position = [self.map_size // 2, self.map_size // 2, 50]
            default_peak = {
                'position': default_position,
                'height': 50,
                'prominence': 20
            }
            return [default_peak]
        
        # 🔧 修复：优先使用已知的山峰中心位置
        if hasattr(self, 'mountain_centers') and len(self.mountain_centers) > 0:
            peaks = []
            for center in self.mountain_centers:
                # mountain_centers 格式: (x, y, height)
                x, y, height = center
                
                # 直接使用已知的山峰信息，不需要验证局部最高点
                # 因为这些都是我们生成的山峰中心
                peaks.append({
                    'position': [x, y, height],
                    'height': height,
                    'prominence': height - np.mean(self.terrain)  # 简单的突出度计算
                })
            
            # 按高度降序排序
            peaks.sort(key=lambda p: p['height'], reverse=True)
            
            if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
                print(f"[山顶检测] 使用已知山峰中心: 找到 {len(peaks)} 个山顶")
                for i, peak in enumerate(peaks[:3]):
                    pos = peak['position']
                    print(f"  山顶{i+1}: 位置=({pos[0]:.1f}, {pos[1]:.1f}), 高度={pos[2]:.1f}, 突出度={peak['prominence']:.1f}")
            
            return peaks[:max_peaks]
        
        # 如果没有已知的山峰中心，使用原来的检测算法
        peaks = []
        margin = max(neighborhood_size, 5)  # 避免边缘位置
        
        # 获取地形统计信息
        terrain_mean = np.mean(self.terrain)
        terrain_std = np.std(self.terrain)
        terrain_max = np.max(self.terrain)
        
        # 设置最小高度阈值
        if min_height is None:
            min_height = terrain_mean + terrain_std * 0.5  # 高于平均值0.5个标准差
        
        if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
            print(f"[山顶检测] 地形统计: 平均高度={terrain_mean:.2f}, 标准差={terrain_std:.2f}, 最高点={terrain_max:.2f}")
            print(f"[山顶检测] 最小高度阈值: {min_height:.2f}")
        
        # 扫描整个地形寻找局部最高点
        # 🔧 修复：将 map_size 转换为整数，避免 range() 类型错误
        map_size_int = int(self.map_size)
        for x in range(margin, map_size_int - margin):
            for y in range(margin, map_size_int - margin):
                # 🔧 关键修复：使用get_terrain_height方法，自动处理降采样后的坐标映射
                current_height = self.get_terrain_height(x, y)
                
                # 检查是否高于最小高度阈值
                if current_height < min_height:
                    continue
                
                # 检查是否为局部最高点
                is_local_max = True
                max_neighbor_height = 0  # 改为0，不包括当前点
                
                # 检查邻域内的所有点
                for dx in range(-neighborhood_size, neighborhood_size + 1):
                    for dy in range(-neighborhood_size, neighborhood_size + 1):
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            # 🔧 关键修复：使用get_terrain_height方法，自动处理降采样后的坐标映射
                            neighbor_height = self.get_terrain_height(nx, ny)
                            max_neighbor_height = max(max_neighbor_height, neighbor_height)
                            
                            # 如果邻域内有更高的点，则不是局部最高点
                            if neighbor_height > current_height:
                                is_local_max = False
                                break
                    
                    if not is_local_max:
                        break
                
                # 如果是局部最高点，计算其突出度
                if is_local_max:
                    prominence = current_height - max_neighbor_height
                    
                    # 降低突出度要求，使其更容易找到山峰
                    # 对于低噪声地形，山峰更独立，突出度自然更高
                    if prominence > min(terrain_std * 0.2, 10.0):  # 至少0.2个标准差或10米
                        peaks.append({
                            'position': [x, y, current_height],  # 坐标顺序：x, y, z
                            'height': current_height,
                            'prominence': prominence
                        })
        
        # 按高度降序排序
        peaks.sort(key=lambda p: p['height'], reverse=True)
        
        # 限制返回的山峰数量
        peaks = peaks[:max_peaks]
        
        if not (os.getenv('QUIET_OUTPUT', '1').lower() in ('1','true','yes','on')):
            print(f"[山顶检测] 找到 {len(peaks)} 个山顶:")
            for i, peak in enumerate(peaks[:3]):
                pos = peak['position']
                print(f"  山顶{i+1}: 位置=({pos[0]:.1f}, {pos[1]:.1f}), 高度={pos[2]:.1f}, 突出度={peak['prominence']:.1f}")
        
        # 如果没有找到合适的山顶，返回全局最高点
        if not peaks:
            # 🔧 关键修复：对于降采样后的地形，需要遍历所有坐标找到最高点
            max_height = -1
            max_x, max_y = 0, 0
            map_size_int = int(self.map_size)
            for x in range(map_size_int):
                for y in range(map_size_int):
                    h = self.get_terrain_height(x, y)
                    if h > max_height:
                        max_height = h
                        max_x, max_y = x, y
            y, x = max_y, max_x
            # 仅在详细调试模式下输出（VIS_DEBUG=1）
            if os.getenv('VIS_DEBUG', '0') == '1':
                print("[山顶检测] 未找到合适的局部最高点，使用全局最高点")
                print(f"[山顶检测] 全局最高点: terrain[{y}, {x}] = {max_height:.1f}")
            
            peaks = [{
                'position': [x, y, max_height],  # 坐标顺序：x, y, z
                'height': max_height,
                'prominence': max_height - terrain_mean
            }]
        
        return peaks

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
    
    def _sync_terrain_to_world(self):
        """同步地形数据到world对象，确保可视化数据一致性"""
        try:
            # 尝试获取world对象
            world = None
            if hasattr(self, 'world') and self.world is not None:
                world = self.world
            elif hasattr(self, 'env') and hasattr(self.env, 'world'):
                world = self.env.world
            
            if world is not None:
                # 同步地形数据
                world.terrain = self.terrain.copy() if self.terrain is not None else None
                world.map_size = self.map_size
                world.terrain_seed = getattr(self, 'seed', None)
                
                # 可选：输出同步信息（仅在调试模式下）
                import os
                debug_mode = os.getenv('TERRAIN_SYNC_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
                if debug_mode:
                    print(f"[TERRAIN_SYNC] 已同步地形到world: shape={self.terrain.shape if self.terrain is not None else None}, seed={getattr(self, 'seed', None)}")
        except Exception as e:
            # 静默处理同步失败，不影响主要功能
            import os
            debug_mode = os.getenv('TERRAIN_SYNC_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
            if debug_mode:
                print(f"[TERRAIN_SYNC] 同步失败: {e}")

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

    def save_terrain_data(self, file_path):
        """
        保存地形数据到文件
        参数:
            file_path (str): 保存地形的文件路径
        """
        try:
            import pickle
            if self.terrain is None:
                print("警告: 没有地形数据可以保存")
                return False
                
            terrain_data = {
                'terrain': self.terrain,
                'seed': self.seed,
                'complexity': self.terrain_complexity,
                'obstacles': self.obstacles if hasattr(self, 'obstacles') else [],
                'map_size': self.map_size
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(terrain_data, f)
            print(f"地形数据已保存到文件: {file_path}")
            return True
        except Exception as e:
            print(f"保存地形数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_terrain_data(self, file_path):
        """
        从文件加载地形数据
        参数:
            file_path (str): 地形数据文件路径
        返回:
            bool: 是否成功加载
        """
        try:
            import pickle
            if not os.path.exists(file_path):
                print(f"地形数据文件不存在: {file_path}")
                return False
                
            with open(file_path, 'rb') as f:
                terrain_data = pickle.load(f)
                
            # 验证数据格式
            if isinstance(terrain_data, dict) and 'terrain' in terrain_data:
                self.terrain = terrain_data['terrain']
                self.seed = terrain_data.get('seed', self.seed)
                self.terrain_complexity = terrain_data.get('complexity', {})
                self.obstacles = terrain_data.get('obstacles', [])
                self._obstacle_layout_signature = None
                self.map_size = terrain_data.get('map_size', self.map_size)
                self._refresh_observation_static_cache()
                
                print(f"成功从文件 {file_path} 加载地形数据")
                
                # 如果有固定位置，验证与地形的兼容性
                if hasattr(self, 'fixed_positions') and self.fixed_positions is not None:
                    self.validate_and_adjust_fixed_positions()
                    
                return True
            else:
                print(f"地形数据文件格式错误: {file_path}")
                return False
                
        except Exception as e:
            print(f"加载地形数据文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def flatten_terrain_around_position(self, center_x, center_y, radius=5.0, height=None):
        """
        平坦化地形上的一片区域，可用于创建起始区域
        
        参数:
            center_x (float): 中心点X坐标
            center_y (float): 中心点Y坐标 
            radius (float): 平坦化半径
            height (float): 平坦化高度，None表示使用中心点高度
        """
        if self.terrain is None:
            self.generate_terrain()
            
        # 确保坐标在有效范围内
        center_x = max(0, min(center_x, self.map_size-1))
        center_y = max(0, min(center_y, self.map_size-1))
        
        # 如果没有指定高度，使用中心点高度
        # 🔧 关键修复：使用get_terrain_height方法，自动处理降采样后的坐标映射
        if height is None:
            height = self.get_terrain_height(center_x, center_y)
            
        # 平坦化区域
        # 🔧 关键修复：如果地形已降采样，需要将坐标映射到降采样后的范围
        map_size_int = int(self.map_size)
        if getattr(self, 'terrain_downsampled', False):
            sample_rate = getattr(self, 'terrain_sample_rate', 4)
            # 将中心坐标和半径映射到降采样后的范围
            center_x_scaled = center_x / sample_rate
            center_y_scaled = center_y / sample_rate
            radius_scaled = radius / sample_rate
            terrain_w = self.terrain.shape[1]
            terrain_h = self.terrain.shape[0]
            for x_scaled in range(max(0, int(center_x_scaled - radius_scaled)), min(terrain_w, int(center_x_scaled + radius_scaled + 1))):
                for y_scaled in range(max(0, int(center_y_scaled - radius_scaled)), min(terrain_h, int(center_y_scaled + radius_scaled + 1))):
                    # 计算到中心点的距离（在降采样后的坐标系中）
                    dist_scaled = np.sqrt((x_scaled - center_x_scaled)**2 + (y_scaled - center_y_scaled)**2)
                    if dist_scaled <= radius_scaled:
                        # 使用高斯平滑过渡
                        weight = np.exp(-(dist_scaled**2) / (2 * (radius_scaled/2)**2))
                        self.terrain[y_scaled, x_scaled] = height * weight + self.terrain[y_scaled, x_scaled] * (1 - weight)
        else:
            # 原始逻辑：未降采样
            for x in range(max(0, int(center_x - radius)), min(map_size_int, int(center_x + radius + 1))):
                for y in range(max(0, int(center_y - radius)), min(map_size_int, int(center_y + radius + 1))):
                    # 计算到中心点的距离
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist <= radius:
                        # 使用高斯平滑过渡
                        weight = np.exp(-(dist**2) / (2 * (radius/2)**2))
                        self.terrain[y, x] = height * weight + self.terrain[y, x] * (1 - weight)
                    
        return True

    def _set_agent_goals(self, world):
        """根据中央目标点，为每个智能体计算并设置其独立的、环绕的最终目标位置。
        
        🔧 关键修复：每个智能体目标的Z坐标必须根据其自身XY位置的地形高度动态设置，
        而不是简单地复制中央目标的Z坐标。否则会导致部分智能体目标在地形内部。
        """
        if self.goal_pos is None:
            print("警告: 中央目标点 (self.goal_pos) 未设置，无法设定智能体目标。")
            return

        # 包围圈半径：各智能体目标绕中央目标分布，适当拉开可减少到终点时的组间斥力
        formation_radius = float(os.getenv('AGENT_GOAL_FORMATION_RADIUS', '10.0'))
        num_agents = len(world.agents)
        # 维护world级的可视化读取容器
        if hasattr(world, 'agent_goals'):
            world.agent_goals = []
        
        # 🔧 获取目标高度偏移量（中央目标相对于其地形的高度）
        central_terrain_h = self.get_terrain_height(self.goal_pos[0], self.goal_pos[1])
        goal_altitude = self.goal_pos[2] - central_terrain_h  # 目标点应在地形上方的高度
        goal_altitude = max(goal_altitude, 5.0)  # 至少5米高度，确保目标在地形上方
        
        for i, agent in enumerate(world.agents):
            # 为3个智能体创建一个等边三角形
            angle = (2 * np.pi * i / num_agents) + (np.pi / 2) # 旋转90度，让一个点朝上
            offset_x = formation_radius * np.cos(angle)
            offset_y = formation_radius * np.sin(angle)
            
            goal_pos = self.goal_pos.copy()
            goal_pos[0] += offset_x
            goal_pos[1] += offset_y
            # 保证目标XY仍在地图范围内（避免靠边时越界）
            try:
                goal_pos[0] = float(np.clip(goal_pos[0], 0, self.map_size - 1))
                goal_pos[1] = float(np.clip(goal_pos[1], 0, self.map_size - 1))
            except Exception:
                pass
            
            # 🔧 关键修复：根据该目标XY位置的地形高度动态设置Z坐标
            # 原设计：goal_pos[2] = self.goal_pos[2]  ❌ 错误！会导致目标在地形内部
            # 新设计：goal_pos[2] = 该位置地形高度 + goal_altitude  ✅ 正确！
            agent_goal_terrain_h = self.get_terrain_height(goal_pos[0], goal_pos[1])
            goal_pos[2] = agent_goal_terrain_h + goal_altitude
            
            # 确保agent.goal_a存在
            if hasattr(agent, 'goal_a') and agent.goal_a is not None:
                agent.goal_a.state.p_pos = goal_pos
            # 同步到world容器
            if hasattr(world, 'agent_goals'):
                world.agent_goals.append(goal_pos.copy())
                # 减少调试信息输出，避免并行环境信息过多
                # if i == 0:  # 只为第一个智能体打印，避免输出过多
                #     print(f"DEBUG: 智能体{i}目标位置设置为: {goal_pos}")
                #     print(f"DEBUG: 中央目标位置: {self.goal_pos}")
                #     print(f"DEBUG: 偏移量: ({offset_x:.2f}, {offset_y:.2f})")

    def _calculate_collision_penalty(self, agent, world):
        """重写/补充：对穿透和接触给予惩罚，并在穿透时按剩余步数放大惩罚"""
        try:
            pos = agent.state.p_pos
            terrain_h = self.get_terrain_height(pos[0], pos[1])
            agent_radius = float(max(0.0, getattr(agent, 'size', getattr(self, 'agent_size', 0.0))))
            body_bottom_z = float(pos[2]) - agent_radius
            try:
                eps = float(os.getenv('TERRAIN_COLLISION_EPS', '0.3'))
            except Exception:
                eps = 0.3
            if body_bottom_z <= terrain_h + eps:
                base = float(getattr(self, 'collision_penalty_value', 50.0))
                ep_len = int(getattr(world, 'episode_length', 1000))
                cur_step = int(getattr(world, 'current_step', ep_len))
                remaining = max(ep_len - cur_step, 0)
                scale = max(1, remaining)
                return -base * scale
            # 近距离接触按固定惩罚
            dmin = None
            if hasattr(world, 'landmarks'):
                for landmark in world.landmarks:
                    if not getattr(landmark, 'collide', False):
                        continue
                    lp = getattr(getattr(landmark, 'state', None), 'p_pos', None)
                    if lp is None:
                        continue
                    r = float(getattr(landmark, 'size', 0.0)) + float(getattr(agent, 'size', 0.0))
                    d = float(np.linalg.norm(pos - lp) - r)
                    dmin = d if dmin is None else min(dmin, d)
            thr_coll = float(getattr(self, 'collision_distance_threshold', 0.5))
            if dmin is not None and dmin <= thr_coll:
                base = float(getattr(self, 'collision_penalty_value', 50.0))
                return -base
        except Exception:
            pass
        return 0.0

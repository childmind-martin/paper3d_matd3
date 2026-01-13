"""
球形探索区域的势场动作修正器
"""

import numpy as np
import traceback

class ContinuousPotentialFieldCorrector:
    """连续势场修正器，对连续动作空间进行修正
    
    使用人工势场法，结合引力场和斥力场来修正智能体行为，针对加速度控制
    """
    
    def __init__(self, terrain_data=None, X=None, Y=None, 
                 goal_attraction=5.0, lambda_1_base=6.0,
                 terrain_repulsion=40.0, agent_repulsion=1.0,
                 influence_range=10.0,
                 terrain_gradient_threshold=0.1,
                 minimum_clearance=2.0,
                 force_scale=5.0,
                 max_force_magnitude=15.0,
                 sphere_detection_radius=5.0,  # 安全检测半径（对应TF版本radius）
                 detection_points=24,
                 use_range_detection=True,
                 detection_radius=15.0,
                 detection_height_range=30.0,
                 agent_detection_radius=20.0,
                 check_count=5,
                 check_spacing=2.5,
                 previous_velocity_weight=0.4,
                 min_height_above_terrain=1.0,
                 terrain_safety_margin=1.5,
                 gravity=0.0,
                 debug_mode=False):
        """初始化连续势场修正器
        
        参数:
            terrain_data: 地形高度数据二维数组
            X, Y: 地形的网格坐标
            goal_attraction: 目标吸引力系数
            terrain_repulsion: 地形排斥力系数
            agent_repulsion: 智能体间排斥力系数
            minimum_clearance: 最小安全间距
            min_height_above_terrain: 智能体与地形的最小高度差
            terrain_safety_margin: 地形安全边界
        """
        # 存储参数
        self.terrain_data = terrain_data
        self.X = X
        self.Y = Y
        self.goal_attraction = goal_attraction
        self.terrain_repulsion = terrain_repulsion
        self.lambda_1_base = lambda_1_base
        self.agent_repulsion = agent_repulsion
        self.influence_range = influence_range
        self.terrain_gradient_threshold = terrain_gradient_threshold
        self.minimum_clearance = minimum_clearance
        self.force_scale = force_scale
        self.max_force_magnitude = max_force_magnitude
        # 垂直力抑制参数已移除
        self.sphere_detection_radius = sphere_detection_radius
        self.detection_points = detection_points
        self.use_range_detection = use_range_detection
        self.detection_radius = detection_radius
        self.detection_height_range = detection_height_range
        self.agent_detection_radius = agent_detection_radius
        self.check_count = check_count
        self.check_spacing = check_spacing
        self.previous_velocity_weight = previous_velocity_weight
        self.min_height_above_terrain = min_height_above_terrain  # 新增：最小高度差
        self.terrain_safety_margin = terrain_safety_margin        # 新增：安全边界
        self.gravity = float(gravity) if gravity is not None else 0.0
        self.debug_mode = debug_mode
        
        # 初始化状态变量
        self.was_emergency = False
        self.emergency_recovery_steps = 0
        self.previous_velocities = {}
        
        # 生成球形探测方向
        self.sphere_directions = None  # 先设置为None
        try:
            self.sphere_directions = self._generate_sphere_directions()
        except Exception as e:
            print(f"球形探测方向生成失败: {e}")
            import traceback
            traceback.print_exc()
            
        # 确保不会返回None，提供默认方向
        if self.sphere_directions is None or len(self.sphere_directions) == 0:
            self.sphere_directions = [np.array([1,0,0]), np.array([-1,0,0]), 
                                     np.array([0,1,0]), np.array([0,-1,0]), 
                                     np.array([0,0,1]), np.array([0,0,-1])]
        
    def _generate_sphere_directions(self):
        """生成球面方向向量
        
        生成一组均匀分布在球面上的方向向量，用于地形碰撞检测
        固定生成24个方向（或更少），确保索引永远不会越界
        """
        try:
            # 固定点数量为24，避免动态计算导致的问题
            count = min(24, self.detection_points)
            
            # 使用黄金螺旋算法生成均匀分布的球面点
            indices = np.arange(0, count, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / count)
            theta = np.pi * (1 + 5**0.5) * indices
            
            # 转换为笛卡尔坐标
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            
            # 组合为方向向量列表
            directions = []
            for i in range(count):
                if i < len(x) and i < len(y) and i < len(z):  # 防御性检查
                    direction = np.array([x[i], y[i], z[i]])
                    # 标准化方向向量
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        directions.append(direction)
            
            # 如果生成的方向数量不足，添加6个主轴方向
            if len(directions) < 6:
                primary_directions = [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([-1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, -1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.0, 0.0, -1.0])
                ]
                directions.extend(primary_directions)
                
                # 去重
                unique_directions = []
                for d in directions:
                    is_duplicate = False
                    for ud in unique_directions:
                        if np.allclose(d, ud, atol=1e-6):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_directions.append(d)
                
                directions = unique_directions
            
            # 确保方向数量不会超过限制
            max_directions = 30  # 设置一个安全的上限
            if len(directions) > max_directions:
                directions = directions[:max_directions]
            
            # 确保directions是numpy数组的列表，而不是单个二维数组
            self.sphere_directions = directions
            
            if self.debug_mode:
                print(f"生成了{len(self.sphere_directions)}个球面方向向量")
                
        except Exception as e:
            # 出错时使用简单的6方向探测（上下左右前后）
            self.sphere_directions = [
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0])
            ]
            
            if self.debug_mode:
                print(f"方向向量生成失败: {e}，使用默认6方向探测")
    
    def set_terrain(self, terrain_data, X=None, Y=None):
        """设置地形数据"""
        self.terrain_data = terrain_data
        self.X = X
        self.Y = Y
    
    def get_terrain_height(self, x, y):
        """获取指定位置的地形高度（行=Y，列=X，与奖励/可视化索引保持一致）。"""
        if self.terrain_data is None:
            return 0.0

        # numpy 高度图通常 shape 为 (rows=y, cols=x)
        max_y, max_x = self.terrain_data.shape
        x_idx = int(np.clip(round(x), 0, max_x - 1))
        y_idx = int(np.clip(round(y), 0, max_y - 1))

        return float(self.terrain_data[y_idx, x_idx])
    
    def _get_goal_force_tf_style(self, agent_pos, goal_pos):
        if goal_pos is None:
            return np.zeros(3, dtype=np.float32)
        vec = goal_pos - agent_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float32)
        direction = vec / dist
        lambda_1 = max(0.1, float(self.lambda_1_base))
        k_att = float(self.goal_attraction)
        if dist > lambda_1:
            strength = 2.0 * k_att * dist
        else:
            strength = k_att * lambda_1
        strength = min(strength, self.max_force_magnitude)
        return direction * strength

    def _sample_height(self, x, y):
        return self.get_terrain_height(x, y)

    def _height_gradient(self, x, y, eps=1.0):
        hx1 = self._sample_height(x + eps, y)
        hx0 = self._sample_height(x - eps, y)
        hy1 = self._sample_height(x, y + eps)
        hy0 = self._sample_height(x, y - eps)
        gx = 0.5 * (hx1 - hx0) / max(eps, 1e-3)
        gy = 0.5 * (hy1 - hy0) / max(eps, 1e-3)
        return gx, gy

    def calculate_terrain_forces_sphere(self, agent_pos, goal_pos=None):
        if self.terrain_data is None:
            return np.zeros(3, dtype=np.float32)

        agent_pos = np.asarray(agent_pos, dtype=np.float32)
        ax, ay, az = agent_pos
        terrain_height = self._sample_height(ax, ay)
        gx, gy = self._height_gradient(ax, ay)
        normal_vec = np.array([-gx, -gy, 1.0], dtype=np.float32)
        normal_vec /= (np.linalg.norm(normal_vec) + 1e-6)
        q = np.array([ax, ay, terrain_height], dtype=np.float32)
        r_min = float(np.dot(agent_pos - q, normal_vec))

        penetration = r_min < 0.0
        if penetration:
            r_eff = max(0.05, abs(r_min) * 0.1)
            normal_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            r_eff = max(0.1, r_min)

        R_safe = max(self.sphere_detection_radius, self.influence_range)
        kappa = 1.0
        if goal_pos is not None:
            d_goal = np.linalg.norm(goal_pos - agent_pos)
            kappa = np.exp(-d_goal / 50.0)

        lambda_r = float(self.terrain_repulsion)
        inv_r = 1.0 / (r_eff + 1e-6)
        inv_R = 1.0 / (R_safe + 1e-6)
        base = lambda_r * (inv_r - inv_R) * (inv_r ** 2) * kappa
        if penetration:
            base *= 5.0
        strength = np.clip(base, 0.0, 50.0 if penetration else 10.0)

        obstacles_force = np.zeros(3, dtype=np.float32)
        obstacles = getattr(self, 'obstacles', None)
        if obstacles:
            for ob in obstacles:
                try:
                    center = np.asarray(ob.get('center'), dtype=np.float32)
                    radius = float(ob.get('radius', 0.0))
                    if center.shape[0] < 3 or radius <= 0.0:
                        continue
                    vec = agent_pos - center
                    dist = np.linalg.norm(vec)
                    if dist <= radius:
                        continue
                    r_obs = dist - radius
                    if r_obs > R_safe:
                        continue
                    inv_r_obs = 1.0 / (r_obs + 1e-6)
                    core = lambda_r * (inv_r_obs - inv_R) * (inv_r_obs ** 2)
                    core = np.clip(core, 0.0, 3.0)
                    obstacles_force += (vec / (dist + 1e-6)) * core
                except Exception:
                    continue

        terrain_force = normal_vec * strength + obstacles_force
        if not np.all(np.isfinite(terrain_force)):
            return np.zeros(3, dtype=np.float32)
        return terrain_force.astype(np.float32, copy=False)
    
    def calculate_terrain_forces_grid(self, agent_pos):
        """已禁用的网格法地形力：返回零（不再产生上向排斥）。"""
        return np.zeros(3, dtype=np.float32)
    
    def calculate_agent_repulsion_forces(self, agent_pos, other_agents):
        """计算智能体间斥力
        
        参数:
            agent_pos: 当前智能体位置 [x, y, z]
            other_agents: 其他智能体位置列表
            
        返回:
            force: 智能体间排斥力向量
        """
        if other_agents is None or len(other_agents) == 0:
            return np.zeros(3)
            
        agent_pos = np.asarray(agent_pos, dtype=np.float32)
        total_force = np.zeros(3, dtype=np.float32)
        R_det = max(self.agent_detection_radius, self.influence_range)
        for other_pos in other_agents:
            if other_pos is None:
                continue
            other = np.asarray(other_pos, dtype=np.float32)
            if other.shape[0] < 3:
                continue
            vec = agent_pos - other
            dist = np.linalg.norm(vec)
            if dist < 1e-3 or dist > R_det:
                continue
            inv_d = 1.0 / (dist + 1e-6)
            rep_strength = self.agent_repulsion * (inv_d - 1.0 / R_det) * (inv_d ** 2)
            rep_strength = np.clip(rep_strength, 0.0, 3.0)
            total_force += (vec / (dist + 1e-6)) * rep_strength

        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_force_magnitude:
            total_force = total_force * self.max_force_magnitude / (force_mag + 1e-8)
        return total_force
    
    def calculate_goal_attraction_force(self, agent_pos, goal_pos):
        """计算目标吸引力
        
        使用新的基于势函数的吸引力计算公式：
        U_att = {
            λa * d²(p, p_target),                    if d(p, p_target) > λ1
            λa * (λ1 * d(p, p_target) - R²_target),  if d(p, p_target) ≤ λ1
        }
        F_att = -∂U_att/∂p
        
        参数:
            agent_pos: 智能体位置
            goal_pos: 目标位置
            
        返回:
            force: 目标吸引力向量
        """
        if goal_pos is None:
            return np.zeros(3)
        agent_pos = np.asarray(agent_pos, dtype=np.float32)
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
            
        # 计算到目标的向量
        vec_to_goal = goal_pos - agent_pos
        dist = np.linalg.norm(vec_to_goal)
        
        if dist < 0.001:  # 已到达目标
            return np.zeros(3)
            
        # 归一化方向向量
        dir_to_goal = vec_to_goal / dist
        
        # 参数定义 - 从网络输出或默认值获取
        lambda_a = self.goal_attraction  # 吸引势场因子
        
        # 使用网络训练出的lambda_1或默认值
        lambda_1 = getattr(self, 'lambda_1_base', 10.0)
        
        R_target = 1.0   # 目标半径（固定值）
        
        # 计算吸引力
        if dist > lambda_1:
            # 远距离区域：二次势函数
            # U_att = λa * d²
            # F_att = -∂U_att/∂p = -2*λa*d*dir_to_goal
            attraction_strength = 2 * lambda_a * dist
        else:
            # 近距离区域：锥形势函数
            # U_att = λa * (λ1 * d - R²_target)
            # F_att = -∂U_att/∂p = -λa*λ1*dir_to_goal
            attraction_strength = lambda_a * lambda_1
        
        # 限制最大吸引力
        attraction_strength = min(attraction_strength, self.max_force_magnitude)
        
        # 计算吸引力向量
        attraction_force = dir_to_goal * attraction_strength
        
        # 不再基于地形高度对垂直分量做上向校正（移除硬性抬升）
            
        return attraction_force
    
    def correct_action_continuous(self, action, agent_pos, goal_pos=None, other_agents=None, force_ratio=1.0):
        """根据势场力修正连续动作
        
        参数:
            action: 原始动作向量 [x, y, z]
            agent_pos: 智能体位置 [x, y, z]
            goal_pos: 目标位置 [x, y, z]（可选）
            other_agents: 其他智能体位置列表（可选）
            force_ratio: 势场力调整系数，默认为1.0
            
        返回:
            corrected_action: 修正后的动作向量
        """
        try:
            # 确保action是numpy数组，并且至少有2维
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            # 扩展dimensions到至少3维
            if len(action) < 3:
                action = np.pad(action, (0, 3 - len(action)), 'constant')
            
            agent_pos = np.asarray(agent_pos, dtype=np.float32)
            goal_force = self.calculate_goal_attraction_force(agent_pos, goal_pos)
            terrain_force = self.calculate_terrain_forces_sphere(agent_pos, goal_pos)

            agent_force = np.zeros(3, dtype=np.float32)
            if other_agents is not None and len(other_agents) > 0:
                agent_force = self.calculate_agent_repulsion_forces(agent_pos, other_agents)

            total_force = goal_force + terrain_force + agent_force
            
            # 🔧 修复：与TF版本保持完全一致的势场力缩放逻辑
            # 直接将势场力缩放到动作空间[-1,1]，不进行过度归一化
            # 这样可以保留势场力的相对强度信息
            
            # 将势场力直接缩放到动作空间[-1,1]
            # 如果total_force幅度=10，max_force=8.8，缩放后幅度=10/8.8≈1.14，clip后=1.0
            # 如果total_force幅度=4，max_force=8.8，缩放后幅度=4/8.8≈0.45（保留）
            a_pf_raw = total_force / self.max_force_magnitude
            a_pf = np.clip(a_pf_raw, -1.0, 1.0)
            
            # 混合网络动作和势场动作
            corrected_action = (1.0 - force_ratio) * action + force_ratio * a_pf
            
            # 裁剪动作值到有效范围
            corrected_action = np.clip(corrected_action, -1.0, 1.0)
            
            return corrected_action
            
        except Exception as e:
            # 如果出现任何错误，返回原始动作
            print(f"修正连续动作时出错: {e}")
            traceback.print_exc()
            return action

    def new_correct_action_continuous(self, action, agent_pos, goal_pos=None, other_agents=None, force_ratio=1.0):
        """使用自适应参数修正连续动作
        
        参数:
            action: 原始动作向量 [x, y, z]
            agent_pos: 智能体位置 [x, y, z]
            goal_pos: 目标位置 [x, y, z]（可选）
            other_agents: 其他智能体位置列表（可选）
            force_ratio: 势场力调整系数，默认为1.0
            
        返回:
            corrected_action: 修正后的动作向量
        """
        try:
            # 确保action是numpy数组，并且至少有2维
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            # 扩展dimensions到至少3维
            if len(action) < 3:
                action = np.pad(action, (0, 3 - len(action)), 'constant')
                
            # 后续处理...与原方法相同，但不直接使用对象属性
            # 获取目标吸引力
            goal_force = np.zeros(3)
            if goal_pos is not None:
                goal_force = self.calculate_goal_attraction_force(agent_pos, goal_pos)
            
            # 计算地形斥力（方向性半球采样）
            terrain_force = self.calculate_terrain_forces_sphere(agent_pos, goal_pos)
            
            # 计算智能体间斥力
            agent_force = np.zeros(3)
            if other_agents is not None and len(other_agents) > 0:
                agent_force = self.calculate_agent_repulsion_forces(agent_pos, other_agents)
            
            # 计算总力（未应用任何权重）
            total_force = goal_force + terrain_force + agent_force
            
            # 使用传入的权重系数或默认值
            goal_att = self.goal_attraction * force_ratio
            terrain_rep = self.terrain_repulsion * force_ratio
            agent_rep = self.agent_repulsion * force_ratio
            force_scale = self.force_scale
            
            # 根据参数调整总力
            total_force = (goal_force * goal_att + 
                          terrain_force * terrain_rep + 
                          agent_force * agent_rep)
            
            # 限制力的最大幅度
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > self.max_force_magnitude:
                total_force = total_force * (self.max_force_magnitude / force_magnitude)
            
            # 计算原始动作与“势场力”的加权组合
            force_weight = min(np.linalg.norm(total_force) / force_scale, 1.0)
            corrected_action = (1.0 - force_weight) * action + force_weight * (total_force * force_scale)
            
            # 裁剪动作值到有效范围
            corrected_action = np.clip(corrected_action, -1.0, 1.0)
            
            return corrected_action
            
        except Exception as e:
            # 如果出现任何错误，返回原始动作
            print(f"自适应修正连续动作时出错: {e}")
            return action
    
    def fix_actions_continuous(self, actions=None, agent_positions=None, goal_pos=None, force_ratio=1.0, maddpg_runner=None, observations=None):
        """修正一组连续动作空间的动作向量
        
        参数:
            actions: 动作列表，每个元素是一个智能体的动作
            agent_positions: 智能体位置列表
            goal_pos: 目标位置
            force_ratio: 修正系数，控制势场力的比例
            maddpg_runner: MADDPG运行器实例，用于获取力场参数
            observations: 原始观察数据，用于智能体适应性调整
            
        返回:
            fixed_actions: 修正后的动作列表
        """
        try:
            # 基本输入验证
            if actions is None:
                return []
                
            if agent_positions is None or len(agent_positions) == 0:
                return actions
                
            # 确定智能体数量和动作列表长度
            agent_count = len(agent_positions)
            action_count = len(actions) if isinstance(actions, (list, tuple)) else 0
            
            # 尝试确定动作维度
            action_dim = 3  # 默认三维动作
            
            if action_count > 0:
                first_action = actions[0]
                if isinstance(first_action, np.ndarray) and first_action.size > 0:
                    action_dim = first_action.shape[0] if len(first_action.shape) > 0 else first_action.size
                elif isinstance(first_action, (list, tuple)):
                    action_dim = len(first_action)
            
            # 创建标准化的动作列表，保证长度和类型
            standardized_actions = []
            
            # 精简调整逻辑，避免重复检查
            for i in range(agent_count):
                if i < action_count:
                    # 处理现有动作
                    action = actions[i]
                    # 确保是numpy数组
                    if not isinstance(action, np.ndarray):
                        try:
                            action = np.array(action, dtype=np.float32)
                        except:
                            # 转换失败时使用零向量
                            action = np.zeros(action_dim, dtype=np.float32)
                    
                    # 确保维度正确
                    if len(action.shape) == 0:  # 标量
                        action = np.zeros(action_dim, dtype=np.float32)
                    elif action.shape[0] != action_dim:
                        # 不匹配则调整大小
                        try:
                            if action.shape[0] < action_dim:
                                # 扩展数组
                                new_action = np.zeros(action_dim, dtype=np.float32)
                                new_action[:action.shape[0]] = action
                                action = new_action
                            else:
                                # 截断数组
                                action = action[:action_dim]
                        except:
                            # 处理失败使用零向量
                            action = np.zeros(action_dim, dtype=np.float32)
                else:
                    # 对于超出原始动作列表的索引，创建零向量
                    action = np.zeros(action_dim, dtype=np.float32)
                
                standardized_actions.append(action)
            
            # 应用势场修正
            fixed_actions = []
            
            # 处理每个智能体的动作
            for i, (action, pos) in enumerate(zip(standardized_actions, agent_positions)):
                # 获取智能体的邻居（其他智能体）
                neighbors = [p for j, p in enumerate(agent_positions) if j != i]
                
                # 确定修正系数
                agent_force_ratio = force_ratio
                
                # 尝试获取智能体的自适应势场参数
                try:
                    if maddpg_runner is not None and hasattr(maddpg_runner, 'agents') and i < len(maddpg_runner.agents):
                        agent = maddpg_runner.agents[i]
                        if hasattr(agent, 'get_force_params'):
                            params = agent.get_force_params()
                            if params is not None and len(params) >= 1:
                                agent_force_ratio *= float(params[0])  # 使用第一个参数作为额外比例
                except Exception:
                    # 如果获取参数失败，使用默认值
                    pass
                            
                # 应用动作修正
                try:
                    fixed_action = self.correct_action_continuous(
                        action, 
                        pos, 
                        goal_pos, 
                        neighbors, 
                        force_ratio=agent_force_ratio
                    )
                    fixed_actions.append(fixed_action)
                except Exception:
                    # 如果修正失败，使用原始动作
                    fixed_actions.append(action)
            
            # 最终验证：确保返回的动作列表长度正确
            if len(fixed_actions) != agent_count:
                # 长度不匹配，调整
                if len(fixed_actions) < agent_count:
                    # 不足则补充
                    for _ in range(agent_count - len(fixed_actions)):
                        fixed_actions.append(np.zeros(action_dim, dtype=np.float32))
                else:
                    # 超出则截断
                    fixed_actions = fixed_actions[:agent_count]
            
            return fixed_actions
            
        except Exception as e:
            # 捕获所有异常，确保函数不会崩溃
            if self.debug_mode:
                print(f"动作修正函数出错: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # 尝试返回原始动作，如果失败则返回空列表
            try:
                return actions
            except:
                return []

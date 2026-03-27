import os

import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        # 添加id属性
        self.id = None

        super(AgentState, self).__init__()
        # communication utterance
        self.p_pos = np.zeros(3)  # 3D位置
        self.p_vel = np.zeros(3)  # 3D速度
        self.c = None
        
        # 🔧 四旋翼动力学状态（可选，仅在use_quadrotor_dynamics时使用）
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # 姿态四元数 [w, x, y, z]
        self.angular_vel = np.zeros(3)  # 机体系角速度 [wx, wy, wz] (rad/s)
        self.p_acc = np.zeros(3)  # 线加速度（用于观测）
        self.motor_speeds = np.zeros(4)  # 电机转速 [omega1, omega2, omega3, omega4] (rad/s)

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:实体的默认尺寸
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 3
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = float(os.getenv('SIMULATION_DT', '0.08'))  # 支持通过环境变量覆盖，默认0.08秒
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # gravity (m/s^2) acting along -Z; 0.0 means disabled
        self.gravity = 0.0
        # control acceleration gain: scales agent action ([-1,1]) to physical acceleration units
        self.control_accel_gain = 12.0
        # per-axis action range scaling (network output ∈ [-1,1] → [-range, range])
        self.action_range = [1.0, 1.0, 1.0]
        # reward scaling for positive/negative parts
        self.reward_pos_scale = 1.0
        self.reward_neg_scale = 1.0
        # optional map info for bounds checking
        self.map_size = None
        # 是否在检测到穿透/碰撞时自动将位置复位到地面上方（默认关闭）
        self.enable_collision_autoreset = False
        # 热路径常量缓存：避免在每个 world.step 内重复读取环境变量
        self.z_action_bias = float(os.getenv('Z_ACTION_BIAS', '0.0'))
        self.quadrotor_attitude_response_time = float(os.getenv('QUADROTOR_ATTITUDE_RESPONSE_TIME', '0.0'))
        self.quadrotor_psi_cmd = float(os.getenv('QUADROTOR_PSI_CMD', '0.0'))
        # 实体/碰撞 pair 缓存：world 构成稳定时可长期复用，减少热循环中的重复过滤。
        self._entities_cache_signature = None
        self._entities_cache = []
        self._collision_pair_indices = ()

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    def _refresh_entity_runtime_cache(self):
        """刷新实体列表与有效碰撞 pair 缓存。"""
        entities = list(self.agents) + list(self.landmarks)
        signature = tuple(
            (id(entity), bool(getattr(entity, 'collide', False)), bool(getattr(entity, 'movable', False)))
            for entity in entities
        )
        if signature != self._entities_cache_signature:
            self._entities_cache_signature = signature
            self._entities_cache = entities
            self._collision_pair_indices = tuple(
                (a, b)
                for a, entity_a in enumerate(entities)
                for b in range(a + 1, len(entities))
                if getattr(entity_a, 'collide', False)
                and getattr(entities[b], 'collide', False)
                and (getattr(entity_a, 'movable', False) or getattr(entities[b], 'movable', False))
            )
        return self._entities_cache, self._collision_pair_indices

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        entities, collision_pairs = self._refresh_entity_runtime_cache()
        # 确保所有实体的位置是numpy数组
        for entity in entities:
            if hasattr(entity.state, 'p_pos') and isinstance(entity.state.p_pos, list):
                entity.state.p_pos = np.array(entity.state.p_pos)
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force, entities=entities, collision_pairs=collision_pairs)
        # integrate physical state
        self.integrate_state(p_force, entities=entities)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        
        # 🚨 关键修复：更新world.current_step，确保碰撞计数逻辑能正确工作
        # 问题：如果current_step不被更新，碰撞计数逻辑中的防重复计数机制会失效
        # 因为所有碰撞都会被判定为"同一歩"，导致只计数一次或根本不计数
        if hasattr(self, 'current_step'):
            self.current_step += 1
        else:
            self.current_step = 1

    def is_within_bounds(self, pos):
        """通用越界检测：检查XYZ是否在合理范围内。
        X,Y: [0, map_size)
        Z: [-50, 150] （允许一定的地下和高空范围）
        """
        try:
            if self.map_size is None:
                return True
            x = float(pos[0])
            y = float(pos[1])
            z = float(pos[2]) if len(pos) > 2 else 0.0
            size = int(self.map_size)
            
            # XY边界检查
            xy_in_bounds = (0.0 <= x < size) and (0.0 <= y < size)
            
            # Z边界检查：允许-50到150的范围
            # -50: 允许轻微穿透地形（碰撞检测会处理）
            # 150: 允许高空飞行但不能无限高
            z_min = -50.0
            z_max = 150.0
            z_in_bounds = (z_min <= z <= z_max)
            
            return xy_in_bounds and z_in_bounds
        except Exception:
            return True

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                # 将七维动作转换为3维力向量
                action = agent.action.u
                if isinstance(action, np.ndarray):
                    # 更灵活地处理动作数组
                    force = np.zeros(3)
                    action_dim = len(action)
                    
                    if action_dim == 7:  # 处理七维输入：前3维是连续3D力向量，后4维是势场参数
                        # 直接使用前3维作为连续的3D力向量
                        force = action[:3].copy()
                    elif action_dim == 3:  # 直接使用3维动作
                        force = action.copy()
                    else:
                        # 其他维度，尝试使用前3维
                        force = action[:3].copy() if action_dim >= 3 else np.zeros(3)
                    
                    # 将网络动作按轴映射到可配置范围
                    try:
                        ar = getattr(self, 'action_range', [1.0, 1.0, 1.0])
                        if isinstance(ar, (list, tuple)) and len(ar) >= 3:
                            force[0] = force[0] * float(ar[0])
                            force[1] = force[1] * float(ar[1])
                            # Z轴动作映射：零中心[-1,1]，再加上可配置偏置后按比例缩放到物理量级
                            # 为保持训练/执行一致，这里的偏置应与训练侧（Actor映射、回放入库）一致
                            # 使用缓存的 z_bias，避免每步/每 agent 读取环境变量
                            force[2] = (force[2] + self.z_action_bias) * float(ar[2])
                    except Exception:
                        pass

                    # 仅对Z轴施加控制加速度增益，X/Y不放大
                    try:
                        gain = float(getattr(self, 'control_accel_gain', 1.0))
                    except Exception:
                        gain = 1.0
                    if force.shape[0] >= 3:
                        force[2] = force[2] * gain
                    
                    # 直接使用力，不添加噪声
                    p_force[i] = force
                else:
                    # 直接使用动作，不添加噪声
                    p_force[i] = action
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force, entities=None, collision_pairs=None):
        # simple collision response with pair filtering
        if entities is None or collision_pairs is None:
            entities, collision_pairs = self._refresh_entity_runtime_cache()
        for a, b in collision_pairs:
            entity_a = entities[a]
            entity_b = entities[b]
            [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
            if(f_a is not None):
                if(p_force[a] is None): p_force[a] = 0.0
                p_force[a] = f_a + p_force[a]
            if(f_b is not None):
                if(p_force[b] is None): p_force[b] = 0.0
                p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force, entities=None):
        if entities is None:
            entities, _ = self._refresh_entity_runtime_cache()
        for i,entity in enumerate(entities):
            if not entity.movable: continue
            
            # 🔧 检查是否使用四旋翼动力学模型
            use_quadrotor = getattr(entity, 'use_quadrotor_dynamics', False)
            
            if use_quadrotor and p_force[i] is not None:
                # 使用四旋翼动力学模型
                try:
                    # 初始化四旋翼动力学对象（如果不存在）
                    if not hasattr(entity, 'quadrotor_dynamics'):
                        from multiagent.quadrotor_dynamics import QuadrotorDynamics
                        # 🔧 修复：直接使用 self.gravity 的值，不使用默认值9.81
                        # self.gravity 已经在 World.__init__ 中初始化为 0.0，然后由训练脚本根据 --gravity 参数设置
                        g = float(self.gravity)
                        entity.quadrotor_dynamics = QuadrotorDynamics(
                            mass=entity.mass,
                            g=g,
                            attitude_response_time=self.quadrotor_attitude_response_time
                        )
                    
                    # 初始化状态（如果不存在）
                    if not hasattr(entity.state, 'orientation'):
                        entity.state.orientation = np.array([1.0, 0.0, 0.0, 0.0])
                    if not hasattr(entity.state, 'angular_vel'):
                        entity.state.angular_vel = np.zeros(3)
                    if not hasattr(entity.state, 'p_acc'):
                        entity.state.p_acc = np.zeros(3)
                    if not hasattr(entity.state, 'motor_speeds'):
                        entity.state.motor_speeds = np.zeros(4)
                    
                    # 当前状态
                    current_state = {
                        'p_pos': entity.state.p_pos,
                        'p_vel': entity.state.p_vel,
                        'orientation': entity.state.orientation,
                        'angular_vel': entity.state.angular_vel
                    }
                    
                    # p_force[i] 是期望加速度（已通过 apply_action_force 处理）
                    # 注意：apply_action_force 输出的 force 已经是加速度单位（m/s^2），
                    # 因为在质点模型中会除以质量，所以这里直接使用
                    a_cmd = p_force[i].copy()
                    
                    # 🚨 关键修复：在四旋翼动力学路径中也要应用 agent.accel 放大
                    # 与质点模型保持一致，确保 agent.accel 参数生效
                    try:
                        accel_gain = getattr(entity, 'accel', None)
                        if accel_gain is not None:
                            accel_gain = float(accel_gain)
                            a_cmd = a_cmd * accel_gain
                    except Exception:
                        pass
                    
                    # 执行一步动力学积分（传入阻尼参数）
                    damping = getattr(self, 'damping', 0.25)
                    new_state, motor_speeds = entity.quadrotor_dynamics.integrate_step(
                        current_state, a_cmd, self.dt, self.quadrotor_psi_cmd, damping
                    )
                    
                    # 更新状态
                    entity.state.p_pos = new_state['p_pos']
                    entity.state.p_vel = new_state['p_vel']
                    entity.state.orientation = new_state['orientation']
                    entity.state.angular_vel = new_state['angular_vel']
                    entity.state.p_acc = new_state['p_acc']
                    entity.state.motor_speeds = motor_speeds
                    
                    # 速度限制
                    if entity.max_speed is not None:
                        speed = np.sqrt(np.sum(np.square(entity.state.p_vel)))
                        if speed > entity.max_speed:
                            entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
                    
                except Exception as e:
                    # 如果四旋翼动力学失败，回退到质点模型
                    import traceback
                    print(f"⚠️ 四旋翼动力学计算失败，回退到质点模型: {e}")
                    if hasattr(self, '_debug_quadrotor_errors'):
                        traceback.print_exc()
                    use_quadrotor = False
            
            if not use_quadrotor:
                # 原有的质点模型（向后兼容）
                entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
                if (p_force[i] is not None):
                    # 在速度积分前，对三轴合力统一乘以agent基础加速度系数（若提供）
                    try:
                        accel_gain = getattr(entity, 'accel', None)
                        if accel_gain is not None:
                            accel_gain = float(accel_gain)
                            p_force[i] = p_force[i] * accel_gain
                    except Exception:
                        pass
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                # apply gravity as acceleration along -Z if enabled and in 3D,
                # with optional pre-takeoff ground support:
                # 在起飞前（仍在起始XY半径且未离地）若合力z加速度≤0，禁止向下（将z向下动作置0）且不施加重力；
                # 仅当合力z加速度>0时撤销支持力，恢复重力与动作的正常作用。
                try:
                    if self.dim_p >= 3 and getattr(self, 'gravity', 0.0) != 0.0:
                        g = float(self.gravity)
                        apply_gravity = True
                        try:
                            # 仅对Agent启用该逻辑（具有 start_position 的实体）
                            if hasattr(entity, 'state') and hasattr(entity, 'movable') and entity.movable:
                                start_pos = getattr(entity, 'start_position', None)
                                if start_pos is None and hasattr(entity.state, 'p_pos'):
                                    # 回退：若场景未设置 start_position，使用当前作为起始参考（仅第一次会等于当前值）
                                    entity.start_position = entity.state.p_pos.copy()
                                    start_pos = entity.start_position
                                if start_pos is not None:
                                    # 判定是否仍在起始区域（XY半径阈值）
                                    dx = float(entity.state.p_pos[0] - start_pos[0])
                                    dy = float(entity.state.p_pos[1] - start_pos[1])
                                    dist_xy = (dx*dx + dy*dy) ** 0.5
                                    # 从world获取可配置阈值
                                    try:
                                        start_radius = float(getattr(self, 'pre_takeoff_start_radius', 1.0))
                                    except Exception:
                                        start_radius = 1.0
                                    # 判定是否已离地（与地形高度差超过阈值）
                                    try:
                                        airborne_threshold = float(getattr(self, 'pre_takeoff_airborne_threshold', 0.5))
                                    except Exception:
                                        airborne_threshold = 0.5
                                    terrain_h = self._get_terrain_height(entity.state.p_pos[0], entity.state.p_pos[1])
                                    if terrain_h is None:
                                        terrain_h = 0.0
                                    height_diff = float(entity.state.p_pos[2]) - float(terrain_h)
                                    still_in_start = dist_xy <= start_radius
                                    not_airborne = height_diff <= airborne_threshold
                                    if still_in_start and not_airborne:
                                        # 计算动作导致的z向加速度（尚未加上重力）
                                        action_acc_z = 0.0
                                        try:
                                            if (p_force[i] is not None) and hasattr(entity, 'mass') and entity.mass != 0:
                                                action_acc_z = float(p_force[i][2] / entity.mass)
                                        except Exception:
                                            action_acc_z = 0.0
                                        # 若包含重力后的合力z加速度≤0，则模拟地面支持力：禁止向下并不施加重力
                                        if (action_acc_z - g) <= 0.0:
                                            try:
                                                if p_force[i] is not None and len(p_force[i]) >= 3 and p_force[i][2] < 0.0:
                                                    p_force[i][2] = 0.0
                                            except Exception:
                                                pass
                                            apply_gravity = False
                                        else:
                                            # 有足够上升推力：允许施加重力，进入正常飞行
                                            apply_gravity = True
                                    else:
                                        pass
                        except Exception:
                            pass
                        if apply_gravity:
                            entity.state.p_vel[2] += (-g) * self.dt
                except Exception:
                    pass
                if entity.max_speed is not None:
                    # 计算三维速度的模长
                    speed = np.sqrt(np.sum(np.square(entity.state.p_vel)))
                    if speed > entity.max_speed:
                        # 按比例缩放所有三个维度的速度
                        entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
                entity.state.p_pos += entity.state.p_vel * self.dt
                
                # 🔧 已彻底删除：智能体位置修正机制（复位机制）
                # 原因：干扰重力模拟和真实物理行为学习
                # 地形碰撞应该通过势场排斥力和奖励函数来处理，而不是强制复位
                pass

    def _get_terrain_height(self, x, y):
        """获取指定位置的地形高度"""
        try:
            # 尝试从scenario获取地形高度
            if hasattr(self, 'scenario'):
                if hasattr(self.scenario, 'get_terrain_height'):
                    return self.scenario.get_terrain_height(x, y)
                elif hasattr(self.scenario, 'get_height_at'):
                    return self.scenario.get_height_at(x, y)
            
            # 尝试从terrain数据直接获取
            if hasattr(self, 'terrain') and self.terrain is not None:
                map_size = self.terrain.shape[0]
                x_idx = max(0, min(int(x), map_size - 1))
                y_idx = max(0, min(int(y), map_size - 1))
                return float(self.terrain[y_idx, x_idx])
                
            return None
        except Exception:
            return None

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        pos_a = entity_a.state.p_pos
        if not isinstance(pos_a, np.ndarray):
            pos_a = np.asarray(pos_a, dtype=np.float32)
            entity_a.state.p_pos = pos_a
        pos_b = entity_b.state.p_pos
        if not isinstance(pos_b, np.ndarray):
            pos_b = np.asarray(pos_b, dtype=np.float32)
            entity_b.state.p_pos = pos_b
        # compute actual distance between entities
        delta_pos = pos_a - pos_b
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        
        # 🔧 修复：防止除零错误 (RuntimeWarning: invalid value encountered in divide)
        # 当两个实体完全重叠时，dist为0，导致NaN
        if dist < 1e-6:
            # 如果重叠，给一个随机方向或固定方向的排斥力
            force_mag = self.contact_force * penetration
            # 使用随机微小扰动作为方向，避免NaN
            force = np.random.uniform(-1, 1, size=delta_pos.shape)
            force = force / (np.linalg.norm(force) + 1e-6) * force_mag
        else:
            force = self.contact_force * delta_pos / dist * penetration
            
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

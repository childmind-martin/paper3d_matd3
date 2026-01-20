"""
四旋翼无人机动力学模块
实现从世界坐标系期望加速度到电机转速的逆解
保持高层接口不变：动作仍然是世界坐标系加速度 a_cmd = (ax, ay, az)
"""
import numpy as np


class QuadrotorDynamics:
    """
    四旋翼动力学模型
    
    输入：世界坐标系期望加速度 a_cmd = (ax, ay, az)
    输出：电机转速 omega_i (i=1,2,3,4)
    
    假设：理想姿态跟踪（姿态能瞬时达到期望值）
    """
    
    def __init__(self, 
                 mass=1.0,
                 arm_length=0.25,  # 螺旋桨臂长 l (m)
                 thrust_coeff=1.0e-6,  # 推力系数 kf (N/(rad/s)^2)
                 torque_coeff=1.0e-7,  # 反扭矩系数 km (N·m/(rad/s)^2)
                 max_motor_speed=1000.0,  # 最大电机转速 (rad/s)
                 g=9.81,  # 重力加速度 (m/s^2)
                 inertia=np.array([0.01, 0.01, 0.02]),  # 惯性矩 [Ixx, Iyy, Izz] (kg·m^2)
                 attitude_response_time=0.0):  # 姿态响应时间常数 (s)，0表示理想跟踪
        """
        初始化四旋翼参数
        
        参数:
            mass: 无人机质量 (kg)
            arm_length: 螺旋桨臂长 (m)
            thrust_coeff: 推力系数 kf (N/(rad/s)^2)
            torque_coeff: 反扭矩系数 km (N·m/(rad/s)^2)
            max_motor_speed: 最大电机转速 (rad/s)
            g: 重力加速度 (m/s^2)
            inertia: 惯性矩 [Ixx, Iyy, Izz] (kg·m^2)
        """
        self.mass = mass
        self.l = arm_length
        self.kf = thrust_coeff
        self.km = torque_coeff
        self.omega_max = max_motor_speed
        self.g = g
        self.I = inertia
        self.attitude_response_time = attitude_response_time  # 姿态响应时间常数
        
        # 构建分配矩阵 B (4x4)
        # [f]         [1   1   1   1  ] [f1]
        # [tau_roll]   [0  -l   0   l  ] [f2]
        # [tau_pitch] =[l   0  -l   0  ] [f3]
        # [tau_yaw]    [km -km  km -km] [f4]
        # 其中 fi = kf * omega_i^2
        self.B = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.l, 0.0, self.l],
            [self.l, 0.0, -self.l, 0.0],
            [self.km, -self.km, self.km, -self.km]
        ])
        
        # 计算分配矩阵的逆（用于从总推力和力矩反推电机推力）
        try:
            self.B_inv = np.linalg.inv(self.B)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            self.B_inv = np.linalg.pinv(self.B)
    
    def acceleration_to_attitude_and_thrust(self, a_cmd, psi_cmd=0.0):
        """
        从期望加速度反推期望姿态和总推力
        
        参数:
            a_cmd: 世界坐标系期望加速度 [ax, ay, az] (m/s^2)
            psi_cmd: 期望偏航角 (rad)，默认0
            
        返回:
            R_des: 期望旋转矩阵 (3x3)
            f: 总推力 (N)
        """
        # 1. 计算期望升力方向
        # 动力学方程: m * a = m * g * e3 - f * R * e3
        # 其中 e3 = [0, 0, 1]^T (世界坐标系Z轴)
        # 整理得: f * R * e3 = m * (g * e3 - a_cmd)
        # 令 h = g * e3 - a_cmd，则期望升力方向 zB_des = h / ||h||
        
        e3 = np.array([0.0, 0.0, 1.0])
        
        # 🔧 修复：当重力为0时，升力方向应该与期望加速度同向
        # 动力学方程分析：
        #   - 有重力：m * a = m * g * e3 - f * R * e3  =>  f * R * e3 = m * (g * e3 - a_cmd)
        #   - 无重力：m * a = f * R * e3  =>  f * R * e3 = m * a_cmd
        # 因此无重力时，升力方向应该与 a_cmd 同向（不是反向）
        if abs(self.g) < 1e-6:
            # 无重力情况：升力方向 = a_cmd 的归一化方向
            # 如果 a_cmd = [0, 0, 0]，使用垂直向上作为默认方向（悬停）
            if np.linalg.norm(a_cmd) < 1e-6:
                zB_des = e3
                f = 0.0  # 无推力（悬停）
            else:
                zB_des = a_cmd / np.linalg.norm(a_cmd)  # 升力方向与加速度同向
                f = self.mass * np.linalg.norm(a_cmd)  # 推力 = m * |a_cmd|
        else:
            # 有重力情况：使用原始公式 h = g * e3 - a_cmd
            h = self.g * e3 - a_cmd
            h_norm = np.linalg.norm(h)
            h_min = 0.1  # 最小值保护，避免除零
            
            if h_norm < h_min:
                # 如果期望加速度接近重力，使用垂直向上
                zB_des = e3
                f = self.mass * self.g  # 悬停推力
            else:
                zB_des = h / h_norm
                f = self.mass * h_norm
        
        # 2. 构造期望旋转矩阵 R_des
        # 使用期望偏航角 psi_cmd 和 zB_des 构造完整的旋转矩阵
        # 方法：先定义水平面参考方向 xC = [cos(psi), sin(psi), 0]
        # 然后 yB_des = normalize(cross(zB_des, xC))
        # xB_des = cross(yB_des, zB_des)
        
        xC = np.array([np.cos(psi_cmd), np.sin(psi_cmd), 0.0])
        
        # 计算 yB_des = normalize(cross(zB_des, xC))
        yB_des_cross = np.cross(zB_des, xC)
        yB_des_norm = np.linalg.norm(yB_des_cross)
        if yB_des_norm < 1e-6:
            # 如果 zB_des 与 xC 平行，使用另一个参考方向
            yC = np.array([-np.sin(psi_cmd), np.cos(psi_cmd), 0.0])
            yB_des_cross = np.cross(zB_des, yC)
            yB_des_norm = np.linalg.norm(yB_des_cross)
        
        if yB_des_norm < 1e-6:
            # 如果仍然平行（zB_des 垂直），使用默认方向
            yB_des = np.array([0.0, 1.0, 0.0])
        else:
            yB_des = yB_des_cross / yB_des_norm
        
        # 计算 xB_des = cross(yB_des, zB_des)
        xB_des = np.cross(yB_des, zB_des)
        xB_des = xB_des / np.linalg.norm(xB_des)  # 归一化
        
        # 构造旋转矩阵 R_des = [xB_des, yB_des, zB_des]^T
        R_des = np.array([xB_des, yB_des, zB_des]).T
        
        return R_des, f
    
    def rotation_matrix_to_quaternion(self, R):
        """
        旋转矩阵转四元数
        
        参数:
            R: 旋转矩阵 (3x3)
            
        返回:
            q: 四元数 [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        q = np.array([w, x, y, z])
        # 归一化
        q = q / np.linalg.norm(q)
        return q
    
    def quaternion_to_rotation_matrix(self, q):
        """
        四元数转旋转矩阵
        
        参数:
            q: 四元数 [w, x, y, z]
            
        返回:
            R: 旋转矩阵 (3x3)
        """
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R
    
    def quaternion_multiply(self, q1, q2):
        """
        四元数乘法
        
        参数:
            q1, q2: 四元数 [w, x, y, z]
            
        返回:
            q: 四元数乘积 [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])
    
    def quaternion_conjugate(self, q):
        """
        四元数共轭
        
        参数:
            q: 四元数 [w, x, y, z]
            
        返回:
            q_conj: 共轭四元数 [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def thrust_and_torque_to_motor_speeds(self, f, tau=np.zeros(3)):
        """
        从总推力和力矩计算电机转速
        
        参数:
            f: 总推力 (N)
            tau: 力矩 [tau_roll, tau_pitch, tau_yaw] (N·m)，默认[0,0,0]
            
        返回:
            omega: 电机转速 [omega1, omega2, omega3, omega4] (rad/s)
        """
        # 构建控制向量 [f, tau_roll, tau_pitch, tau_yaw]^T
        u = np.array([f, tau[0], tau[1], tau[2]])
        
        # 通过分配矩阵逆解得到电机推力 fi
        # u = B * [f1, f2, f3, f4]^T
        # [f1, f2, f3, f4]^T = B_inv * u
        f_motors = self.B_inv @ u
        
        # 从推力计算转速平方: fi = kf * omega_i^2
        # omega_i^2 = fi / kf
        omega_squared = f_motors / self.kf
        
        # 裁剪到有效范围 [0, omega_max^2]
        omega_squared = np.clip(omega_squared, 0.0, self.omega_max**2)
        
        # 开方得到转速
        omega = np.sqrt(omega_squared)
        
        return omega
    
    def integrate_step(self, state, a_cmd, dt, psi_cmd=0.0, damping=0.25):
        """
        执行一步动力学积分
        
        参数:
            state: 状态字典，包含:
                - p_pos: 位置 [x, y, z]
                - p_vel: 速度 [vx, vy, vz]
                - orientation: 姿态四元数 [w, x, y, z]
                - angular_vel: 角速度 [wx, wy, wz] (机体系)
            a_cmd: 世界坐标系期望加速度 [ax, ay, az] (m/s^2)
            dt: 时间步长 (s)
            psi_cmd: 期望偏航角 (rad)，默认0
            damping: 速度阻尼系数，默认0.25（对应原来的 damping 参数）
            
        返回:
            new_state: 更新后的状态字典
            motor_speeds: 电机转速 [omega1, omega2, omega3, omega4] (rad/s)
        """
        # 1. 从期望加速度反推期望姿态和总推力
        R_des, f_desired = self.acceleration_to_attitude_and_thrust(a_cmd, psi_cmd)
        
        # 2. 姿态跟踪（支持理想跟踪和响应延迟两种模式）
        q_des = self.rotation_matrix_to_quaternion(R_des)
        
        if self.attitude_response_time <= 0.0:
            # 理想姿态跟踪：直接设置姿态为期望姿态，角速度为0
            q_actual = q_des
            omega_body = np.zeros(3)  # 理想跟踪假设：角速度为0
        else:
            # 🔧 真实姿态响应：使用一阶低通滤波器模拟姿态响应延迟
            # 当前姿态
            q_current = state.get('orientation', np.array([1.0, 0.0, 0.0, 0.0]))
            
            # 计算姿态误差（四元数差值）
            # 使用四元数插值：q_actual = slerp(q_current, q_des, alpha)
            # alpha = dt / (attitude_response_time + dt) 是一阶低通滤波器的系数
            alpha = dt / (self.attitude_response_time + dt)
            alpha = np.clip(alpha, 0.0, 1.0)  # 限制在[0,1]范围内
            
            # 四元数球面线性插值（slerp）
            # 简化版本：使用线性插值后归一化（对于小角度误差足够准确）
            q_actual = (1 - alpha) * q_current + alpha * q_des
            # 🔧 性能优化：使用平方和开方替代np.linalg.norm（避免函数调用开销）
            q_norm = np.sqrt(np.sum(q_actual**2))
            if q_norm > 1e-6:
                q_actual = q_actual / q_norm
            else:
                q_actual = q_des  # 如果归一化失败，使用期望姿态
            
            # 计算角速度（从姿态变化率估算）
            # 🔧 性能优化：简化四元数误差计算，避免函数调用开销
            # 四元数共轭：q_conj = [w, -x, -y, -z]
            q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
            # 四元数乘法：q_error = q_actual * q_current_conj
            # 简化计算（避免函数调用）
            w_a, x_a, y_a, z_a = q_actual
            w_c, x_c, y_c, z_c = q_current_conj
            q_error = np.array([
                w_a*w_c - x_a*x_c - y_a*y_c - z_a*z_c,  # w
                w_a*x_c + x_a*w_c + y_a*z_c - z_a*y_c,  # x
                w_a*y_c - x_a*z_c + y_a*w_c + z_a*x_c,  # y
                w_a*z_c + x_a*y_c - y_a*x_c + z_a*w_c   # z
            ])
            # 确保最短路径
            if q_error[0] < 0:
                q_error = -q_error
            # 从四元数误差提取角速度（简化：假设小角度）
            omega_body = 2.0 * q_error[1:4] / (dt + 1e-6)  # 提取虚部并缩放
            omega_body = np.clip(omega_body, -10.0, 10.0)  # 限制角速度范围（rad/s）
        
        # 3. 计算力矩（理想跟踪下，力矩为0，因为姿态已瞬时达到）
        tau = np.zeros(3)
        
        # 4. 从总推力和力矩计算电机转速（可能被限制）
        motor_speeds = self.thrust_and_torque_to_motor_speeds(f_desired, tau)
        
        # 🔧 关键修复：从受限的电机转速反推实际推力，让电机转速限制真正影响推力
        # 如果电机转速被限制，实际推力应该减小
        # 从电机转速计算实际推力：f_actual = sum(kf * omega_i^2)
        motor_thrusts = self.kf * (motor_speeds ** 2)  # 每个电机的推力
        f_actual = np.sum(motor_thrusts)  # 实际总推力（可能小于f_desired）
        
        # 5. 更新平动状态
        # 使用实际姿态（可能因响应延迟而不同于期望姿态）和实际推力计算升力
        R_actual = self.quaternion_to_rotation_matrix(q_actual)
        e3 = np.array([0.0, 0.0, 1.0])
        
        # 升力在世界坐标系中的方向（使用实际姿态和实际推力）
        thrust_force_world = f_actual * (R_actual @ e3)
        
        # 重力（仅在重力非零时应用）
        if abs(self.g) > 1e-6:
            gravity_force = np.array([0.0, 0.0, -self.g * self.mass])
        else:
            gravity_force = np.zeros(3)
        
        # 总力
        total_force = thrust_force_world + gravity_force
        
        # 线加速度
        linear_acc = total_force / self.mass
        
        # 更新速度（考虑阻尼）
        p_vel = state['p_vel'].copy()
        p_vel = p_vel * (1 - damping)  # 使用传入的阻尼参数
        p_vel += linear_acc * dt
        
        # 更新位置
        p_pos = state['p_pos'].copy()
        p_pos += p_vel * dt
        
        # 构造新状态
        new_state = {
            'p_pos': p_pos,
            'p_vel': p_vel,
            'orientation': q_actual,  # 🔧 使用实际姿态（可能因响应延迟而不同于期望姿态）
            'angular_vel': omega_body,
            'p_acc': linear_acc  # 保存加速度用于观测
        }
        
        return new_state, motor_speeds


import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        # 确保有一个存储目标位置的属性
        self.goal_pos = None
        # 用于存储viewer的属性
        self.viewer = None
        # 添加碰撞惩罚属性
        self.collision_penalty = True  # 默认启用碰撞惩罚
        # 实体索引字典
        self.entity_indices = {}
    
    def make_world(self):
        world = World()
        # 设置为3D
        world.dim_p = 3
        # 增加通信维度
        world.dim_c = 2
        # 增加智能体数量（可选）
        num_agents = 3
        # 障碍物数量设为原来的10个
        num_landmarks = 10  # 修改为10个以简化环境
        
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15  # 智能体尺寸
            # 这些属性在Agent类中可能并不存在，但我们可以添加
            if hasattr(agent, 'accel'):
                agent.accel = 5.0  # 加速度
            if hasattr(agent, 'max_speed'):
                agent.max_speed = 2.0  # 最大速度
        
        # 添加障碍物和目标点
        world.landmarks = []
        
        # 创建障碍物 - 使用固定大小
        # 基准大小
        base_size = 0.1
        for i in range(num_landmarks - 1):
            landmark = Landmark()
            landmark.name = f'obstacle_{i}'
            landmark.collide = True
            landmark.movable = False
            # 使用固定大小
            landmark.size = base_size
            # 设置颜色
            landmark.color = np.array([0.8, 0.1, 0.1]) if landmark.color is None else landmark.color
            world.landmarks.append(landmark)
        
        # 创建目标点
        goal = Landmark()
        goal.name = 'goal'
        goal.collide = False
        goal.movable = False
        goal.size = 0.1
        # 设置颜色
        goal.color = np.array([0.1, 0.65, 0.15]) if goal.color is None else goal.color
        world.landmarks.append(goal)
    
        # 设置初始状态
        self.reset_world(world)
        return world
        
    def reset_world(self, world):
        # 随机设置智能体位置和速度
        for agent in world.agents:
            # 在3D空间中更广泛地分布
            agent.state.p_pos = np.random.uniform(-2, +2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        # 随机设置障碍物位置
        for i, landmark in enumerate(world.landmarks[:-1]):  # 除了最后一个（目标点）
            # 3D空间中更广泛地分布障碍物
            landmark.state.p_pos = np.random.uniform(-2, +2, world.dim_p)
            # 确保障碍物在z轴上也有变化
            landmark.state.p_pos[2] = np.random.uniform(-1, +1)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        # 设置目标点位置
        goal = world.landmarks[-1]
        # 保存目标位置引用
        self.goal_pos = np.random.uniform(-2, +2, world.dim_p)
        self.goal_pos[2] = np.random.uniform(-1, +1)
        goal.state.p_pos = self.goal_pos.copy()  # 确保使用副本
        goal.state.p_vel = np.zeros(world.dim_p)
        
        # 设置智能体颜色
        for agent in world.agents:
            if agent.color is None:
                agent.color = np.array([0.35, 0.35, 0.85])
        
        # 设置障碍物颜色 - 设置略微不同的颜色以便区分
        for i, landmark in enumerate(world.landmarks[:-1]):
            r = 0.85
            g = 0.35 + (i % 3) * 0.1  # 轻微变化
            b = 0.35 + (i % 2) * 0.1  # 轻微变化
            if landmark.color is None:
                landmark.color = np.array([r, g, b])
        
        # 设置目标点颜色
        if goal.color is None:
            goal.color = np.array([0.15, 0.65, 0.15])
        
        # 确保初始位置没有碰撞
        for agent in world.agents:
            collision_count = 0
            while self.is_collision(agent, world) and collision_count < 10:
                agent.state.p_pos = np.random.uniform(-2, +2, world.dim_p)
                collision_count += 1
    
    def is_collision(self, agent, world):
        """
        3D版本的碰撞检测机制
        检测智能体与其他实体（智能体、障碍物）之间的碰撞
        """
        # 检查与障碍物的碰撞
        for obstacle in world.landmarks[:-1]:  # 除了最后一个（目标点）
            # 计算3D空间中实体间的实际距离
            delta_pos = obstacle.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # 最小允许距离
            dist_min = obstacle.size + agent.size
            if dist < dist_min:
                return True
                
        # 检查与其他智能体的碰撞
        for other in world.agents:
            if other is agent: continue
            # 计算3D空间中实体间的实际距离
            delta_pos = other.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # 最小允许距离
            dist_min = other.size + agent.size
            if dist < dist_min:
                return True
                
        return False
    
    def rel_pos_cost(self, pos1, pos2):
        """
        计算两个位置之间的相对位置代价
        基于2D版本的rel_pos_cost函数但适用于3D环境
        """
        # 计算两点之间的欧式距离
        dist = np.linalg.norm(pos1 - pos2)
        goal_dist = 0.15  # 目标距离
        
        if 0.05 < dist < goal_dist:
            cost = -abs(goal_dist - dist) * 5
        elif dist <= 0.05:
            cost = -abs(1 - dist)
        else:
            cost = 0
        return cost
        
    def reward(self, agent, world):
        # 基础奖励
        rew = 0
        
        # 到目标距离
        goal = [landmark for landmark in world.landmarks if landmark.name == 'goal'][0]
        dist = np.linalg.norm(agent.state.p_pos - goal.state.p_pos)
        
        # 距离奖励 - 越近奖励越高
        rew -= dist * 2.0
        
        # 额外的方向性奖励 - 鼓励朝向目标移动
        direction_to_goal = goal.state.p_pos - agent.state.p_pos
        direction_to_goal = direction_to_goal / (np.linalg.norm(direction_to_goal) + 1e-6)  # 归一化
        vel_direction = agent.state.p_vel / (np.linalg.norm(agent.state.p_vel) + 1e-6)  # 归一化
        
        # 计算智能体速度与目标方向的点积（相似度）
        alignment = np.dot(direction_to_goal, vel_direction)
        rew += alignment * 0.5  # 朝向目标移动时增加奖励
        
        # 阶段性奖励 - 当接近目标时给予更高奖励
        if dist < 1.0:
            rew += 10.0
        elif dist < 3.0:
            rew += 2.0
        
        # 添加碰撞惩罚
        for a in world.agents:
            if a is agent: continue
            dist = np.linalg.norm(a.state.p_pos - agent.state.p_pos)
            if dist < 1.0:  # 假设1.0是碰撞阈值
                rew -= 2.0
        
        return rew
        
    def observation(self, agent, world):
        # 基础观察
        obs = []
        
        # 添加自身位置和速度（确保在3D环境中正确维度）
        obs.append(agent.state.p_pos)
        obs.append(agent.state.p_vel)
        
        # 目标相对位置（保持一致的坐标系）
        goal = [landmark for landmark in world.landmarks if landmark.name == 'goal'][0]
        goal_rel_pos = goal.state.p_pos - agent.state.p_pos
        
        # 归一化目标相对位置 - 增强方向信息
        goal_dist = np.linalg.norm(goal_rel_pos)
        goal_dir = goal_rel_pos / (goal_dist + 1e-6)
        
        # 分别添加目标方向和距离，使网络更容易学习
        obs.append(goal_dir)  # 归一化方向
        obs.append([goal_dist])  # 距离
        
        # 添加障碍物信息（如果有）
        for landmark in world.landmarks:
            if 'obstacle' in landmark.name:
                rel_pos = landmark.state.p_pos - agent.state.p_pos
                obs.append(rel_pos)
        
        # 确保所有观察都是浮点数组
        obs = np.concatenate(obs)
        return obs
        
    def info(self, agent, world):
        """返回额外信息用于可视化和调试"""
        return [self.goal_pos, world.landmarks[:-1]]
    
    def done(self, agent, world):
        """判断是否完成任务的条件"""
        return False

    # 移除与渲染相关的方法，简化为基本渲染支持
    def render(self, mode='human'):
        """简化的渲染函数"""
        return None
    
    def close(self):
        """关闭资源的函数"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer = None
            except:
                pass
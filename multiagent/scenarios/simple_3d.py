import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # 设置为3D环境
        world.dim_p = 3  # 3D空间
        world.collaborative = True
        
        # 添加智能体
        num_agents = 3
        num_landmarks = 3
        num_obstacles = 6  # 从2个增加到6个障碍物
        world.agents = [Agent() for i in range(num_agents)]
        
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.max_speed = 1.0  # 限制最大速度
            # 设置颜色：RGB值在[0,1]范围内
            if i == 0:
                agent.color = np.array([0.85, 0.35, 0.35])  # 红色
            elif i == 1:
                agent.color = np.array([0.35, 0.35, 0.85])  # 蓝色
            else:
                agent.color = np.array([0.35, 0.85, 0.35])  # 绿色
        
        # 添加目标点
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
            landmark.color = np.array([0.85, 0.85, 0.35])  # 黄色目标点
        
        # 添加障碍物
        for i in range(num_obstacles):
            obstacle = Landmark()
            obstacle.name = f'obstacle_{i}'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.20
            obstacle.color = np.array([0.25, 0.25, 0.25])  # 灰色障碍物
            world.landmarks.append(obstacle)
        
        # 初始化状态
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 随机初始化智能体位置
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        # 随机放置目标点，确保在视野范围内
        targets = world.landmarks[:3]  # 前3个是目标点
        for i, landmark in enumerate(targets):
            # 在x-y平面的一定范围内随机放置
            landmark.state.p_pos = np.random.uniform(-0.9, 0.9, world.dim_p)
            # 确保目标点有不同的高度
            landmark.state.p_pos[2] = 0.1 * (i + 1)  # z坐标
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        # 放置障碍物
        obstacles = world.landmarks[3:]  # 后面的是障碍物
        for i, obstacle in enumerate(obstacles):
            # 在智能体和目标之间放置障碍物
            obstacle.state.p_pos = np.random.uniform(-0.5, 0.5, world.dim_p)
            # 障碍物高度随机
            obstacle.state.p_pos[2] = np.random.uniform(0.2, 0.5)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # 简单的奖励函数：智能体与目标的距离越近，奖励越高
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # 获取其他智能体的相对位置
        other_agents = [a.state.p_pos - agent.state.p_pos for a in world.agents if a is not agent]
        other_agents_vel = [a.state.p_vel for a in world.agents if a is not agent]
        
        # 获取所有目标点的相对位置
        landmarks_pos = [l.state.p_pos - agent.state.p_pos for l in world.landmarks]
        
        # 组合观测
        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + other_agents + other_agents_vel + landmarks_pos)

    def render(self, world, mode='human'):
        # 渲染函数，在3D环境中默认是空的，渲染由environment.py处理
        pass 
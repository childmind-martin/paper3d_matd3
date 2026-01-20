import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_agents = 3
        num_obstacles = 8

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collidable = True
            agent.collide = False
            agent.silent = True
            agent.max_speed = 1.0  # None/1.0
            agent.accel = 1.0  # standard is none but later used as 5
            # agent.size = 0.025  # 智能体的大小2.5m/大小0.0125

        # Add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.obstacles):
            landmark.name = 'obstacle %d' % i
            landmark.collidable = True
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1

        world.goal = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.goal):
            landmark.name = 'goal landmark %d' % i
            landmark.collide = False
            landmark.collidable = False
            landmark.movable = False
            # landmark.size = 0.05

        world.landmarks = world.goal.copy()   #TODO: copy goal to landmark instead of by reference (make sure it doesnt cause any bugs)
        world.landmarks += world.obstacles.copy()
        
        # make initial conditions
        self.goal_dist = 0.25   # 1.0
        self.reset_world(world)
        return world

    def reset_world(self, world):

        for i, landmark in enumerate(world.obstacles):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[0] += 0.8
            landmark.index = i
            landmark.state.p_pos = np.random.uniform(-2.5, +2.5, world.dim_p)
            # if i == 0:
            #     landmark.state.p_pos = np.array([1.5, 2.5])
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Set a random goal position
        self.goal_pos = np.random.uniform(+1, +2.5, world.dim_p)
        # self.goal_pos = np.array([+2.5, +2.5])

        # Update our goal landmark for visualization
        world.goal[0].color = np.array([0.0, 0.0, 1.0])
        world.goal[0].state.p_pos = self.goal_pos
        world.goal[0].state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for i, agent in enumerate(world.agents):     # for agent in world.agents:
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.p_pos = np.random.uniform(-2.5, +2.5, world.dim_p)
            # if i == 0:
            #     agent.state.p_pos = np.array([-1.8, -2.2])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # After we set all the inital agent and landmark positions, 
        # make sure we aren't in any collisions already
        for agent in world.agents:
            while(self.is_collision(agent, world)):
                agent.state.p_pos = np.random.uniform(-2.5, +2.5, world.dim_p)


    # def rel_pos_cost(self, pos1, pos2):
    #     # Distance is the l1 Norm
    #     dist = np.linalg.norm(pos1 - pos2)
    #     # Cost is 0 at specified distance, larger otherwise
    #     cost = -abs(dist - self.goal_dist)
    #     return cost
    def rel_pos_cost(self, pos1, pos2):
        # 计算两点之间的欧式距离
        dist = np.linalg.norm(pos1 - pos2)
        if 0.05 < dist < 0.15:
            cost = -abs(0.15 - dist) * 5  # 0.25
        elif dist <= 0.05:
            cost = -abs(1 - dist)
        # 代价在目标距离处为0，其他情况下为0
        else:
            cost = 0
        return cost

    def is_collision(self, agent, world):
        for entity in (world.agents + world.obstacles + world.goal):
            # compute actual distance between entities
            delta_pos = entity.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity.size + agent.size
            if dist < dist_min and entity != agent and entity.collidable:
                return True
        return False

    def reward(self, agent, world):
        total_cost = 0.0
        formation_weight = 10
        # Accumulate costs from each agent to each other agent
        for other in world.agents:
            if agent != other:
                total_cost += formation_weight*self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)

        # distance for cost from goal pos
        dist_from_goal = 10*np.linalg.norm(agent.state.p_pos - self.goal_pos)

        total_cost -= dist_from_goal

        # Chosen kind of arbitrarily, collision cost
        total_cost -= 20 if self.is_collision(agent, world) else 0.0  # total_cost -= 10 if self.is_collision(agent, world) else 0.0

        return total_cost

    # def reward(self, agent, world):
    #     total_cost = 0.0
    #     coord = []
    #     formation_weight = 10  # 调整智能体之间相对位置的权重系数10
    #     for other in world.agents:
    #         if agent != other:
    #             # 调用rel_p os_cost函数计算智能体与其他智能体之间的碰撞惩罚
    #             total_cost += formation_weight * self.rel_pos_cost(agent.state.p_pos, other.state.p_pos)
    #
    #     dist_from_goal = 0  # 设置默认值
    #     # 当前智能体与目标的距离奖励
    #     dist_agent_goal = np.linalg.norm(agent.state.p_pos - self.goal_pos)
    #     if dist_agent_goal > 0.5:
    #         dist_from_goal -= 10 * dist_agent_goal
    #     elif 0.25 < dist_agent_goal <= 0.5:
    #         dist_from_goal -= 5 * dist_agent_goal   # += (1 - dist_agent_goal)
    #     elif dist_agent_goal <= 0.25:
    #         dist_from_goal -= dist_agent_goal                     # 5 * (1 - dist_agent_goal) + 5
    #     # 将智能体与目标位置的距离从total_cost中减去
    #     # total_cost -= dist_from_goal
    #     # 根据智能体是否发生碰撞，决定是否再从total_cost中减去一个collision cost（10）
    #     total_cost -= 10 if self.is_collision(agent, world) else 0.0     # 初始为10
    #
    #     return total_cost

    # Our observation include every agents velocity and our current positions
    def observation(self, agent, world):
        obs = agent.state.p_vel
        obs = np.append(obs, self.goal_dist)
        for other in world.agents:
            if agent != other:
                rel_pos = (agent.state.p_pos - other.state.p_pos)
                obs = np.concatenate([obs, rel_pos, other.state.p_vel])

        for obst in world.obstacles:
            rel_pos = (agent.state.p_pos - obst.state.p_pos)
            obs = np.concatenate([obs, rel_pos])
        rel_goal = agent.state.p_pos - self.goal_pos
        obs = np.concatenate([obs, rel_goal])
        return obs

    def info(self, agent, world):
        return [self.goal_pos, world.obstacles.copy()]

    def done(self, agent, world):
        return False
        


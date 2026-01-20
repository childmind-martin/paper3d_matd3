#!/usr/bin/env python

"""
山脉场景：
使用连续的山脉替代离散的球型障碍物
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
import warnings
import time
import random
from mpl_toolkits.mplot3d import Axes3D
import pyglet
from pyglet.gl import *  # 移至模块级别
from multiagent.core import World, Agent, Landmark, EntityState
from multiagent.scenario import BaseScenario
from multiagent.rendering import Viewer, Transform, Line, FilledPolygon
import pyglet.gl as gl

class MountainLandmark(Landmark):
    """
    山脉地标类，用于创建3D山脉地形
    """
    def __init__(self):
        # 初始化父类
        super(MountainLandmark, self).__init__()
        
        # 设置基本属性
        self.name = 'mountain'
        self.collide = True
        self.movable = False
        self.size = 0.0
        
        # 山脉形状参数
        self.mountain_range = 4.0  # 山脉范围大小
        self.peak_positions = []  # 山峰位置
        self.peak_heights = []    # 山峰高度
        self.valley_points = []   # 山谷点
        
        # 添加初始颜色
        self.color = np.array([0.7, 0.5, 0.3])  # 山脉棕色
        
        # 确保状态不为None
        if not hasattr(self, 'state') or self.state is None:
            self.state = EntityState()
        if hasattr(self.state, 'p_pos') and self.state.p_pos is None:
            self.state.p_pos = np.zeros(3)
        if hasattr(self.state, 'p_vel') and self.state.p_vel is None:
            self.state.p_vel = np.zeros(3)

    def setup_mountain(self, num_peaks=8, height_range=(0.5, 2.5), spread=0.5):
        """设置山脉参数 - 生成连绵的山脉效果"""
        self.peak_positions = []
        self.peak_heights = []
        self.valley_points = []
        
        # 生成主要山脊线 - 创建蜿蜒的路径
        num_ridge_points = 12  # 山脊线上的点数
        ridge_points = []
        
        # 生成一条主山脊线 - 使用随机游走算法
        x, y = 0, 0  # 从中心开始
        for i in range(num_ridge_points):
            # 添加一些随机噪声来创建蜿蜒的山脊
            delta_x = np.random.uniform(-0.5, 0.5)
            delta_y = np.random.uniform(-0.5, 0.5)
            
            # 确保山脊不会离中心太远
            max_dist = self.mountain_range / 2 * 0.8
            curr_dist = np.sqrt(x**2 + y**2)
            if curr_dist > max_dist:
                # 施加一个向中心的力
                center_force = 0.5
                x = x * (1 - center_force) 
                y = y * (1 - center_force)
            
            # 更新位置
            x += delta_x
            y += delta_y
            
            # 添加到山脊点
            ridge_points.append([x, y])
            
        # 在山脊线上生成山峰
        for i, point in enumerate(ridge_points):
            # 主山脊上的点有较高概率成为山峰
            if np.random.random() < 0.6:
                # 生成山峰高度 - 靠近中心的点高度更高
                dist_to_center = np.sqrt(point[0]**2 + point[1]**2)
                max_height = height_range[1] * (1 - 0.3 * dist_to_center / max_dist)
                min_height = height_range[0]
                
                # 随机高度，但有一定概率生成特别高的山峰
                if np.random.random() < 0.2:  # 20%概率生成高峰
                    height = np.random.uniform(max_height * 0.7, max_height)
                else:
                    height = np.random.uniform(min_height, max_height * 0.7)
                
                # 添加一些随机偏移，使山峰不完全位于山脊线上
                peak_x = point[0] + np.random.uniform(-0.3, 0.3)
                peak_y = point[1] + np.random.uniform(-0.3, 0.3)
                
                self.peak_positions.append([peak_x, peak_y])
                self.peak_heights.append(height)
        
        # 生成额外的独立山峰
        for _ in range(num_peaks - len(self.peak_positions)):
            # 随机位置，但确保不会太接近现有山峰
            while True:
                pos_x = np.random.uniform(-self.mountain_range/2 * 0.8, self.mountain_range/2 * 0.8)
                pos_y = np.random.uniform(-self.mountain_range/2 * 0.8, self.mountain_range/2 * 0.8)
                
                # 检查是否离现有山峰太近
                too_close = False
                for existing_pos in self.peak_positions:
                    dist = np.sqrt((pos_x - existing_pos[0])**2 + (pos_y - existing_pos[1])**2)
                    if dist < 0.5:  # 最小山峰间距
                        too_close = True
                        break
                        
                if not too_close:
                    break
            
            # 生成山峰高度
            dist_to_center = np.sqrt(pos_x**2 + pos_y**2)
            height_factor = 1 - 0.5 * dist_to_center / (self.mountain_range/2)
            height = np.random.uniform(height_range[0], height_range[1] * height_factor)
            
            self.peak_positions.append([pos_x, pos_y])
            self.peak_heights.append(height)
        
        # 添加山谷 - 在山峰之间低洼的区域
        num_valleys = int(num_peaks * 0.7)  # 山谷数量少于山峰
        for _ in range(num_valleys):
            # 在两个山峰之间随机选择位置
            if len(self.peak_positions) >= 2:
                idx1, idx2 = np.random.choice(len(self.peak_positions), size=2, replace=False)
                peak1 = self.peak_positions[idx1]
                peak2 = self.peak_positions[idx2]
                
                # 山谷位置 - 在两山峰间随机点，稍微偏移
                t = np.random.uniform(0.3, 0.7)  # 混合因子
                valley_x = peak1[0] * t + peak2[0] * (1 - t) + np.random.uniform(-0.3, 0.3)
                valley_y = peak1[1] * t + peak2[1] * (1 - t) + np.random.uniform(-0.3, 0.3)
                
                # 山谷深度 - 比周围山峰低
                peak1_height = self.peak_heights[idx1]
                peak2_height = self.peak_heights[idx2]
                avg_height = (peak1_height + peak2_height) / 2
                valley_depth = avg_height * np.random.uniform(0.1, 0.4)  # 山谷深度是平均高度的10-40%
                
                self.valley_points.append([valley_x, valley_y])
        
        # 创建所有点的统一数组用于RBF插值
        all_positions = []
        all_heights = []
        
        # 添加山峰
        for pos, height in zip(self.peak_positions, self.peak_heights):
            all_positions.append(pos)
            all_heights.append(height)
        
        # 添加山谷（负高度点）
        for pos in self.valley_points:
            all_positions.append(pos)
            all_heights.append(-0.1)  # 负值表示凹下去
        
        # 创建山脊线 - 在山峰之间插入额外的点
        if len(self.peak_positions) >= 2:
            # 根据距离对山峰排序，使相近的山峰连接在一起
            for i in range(len(self.peak_positions)):
                for j in range(i+1, len(self.peak_positions)):
                    pos1 = self.peak_positions[i]
                    pos2 = self.peak_positions[j]
                    height1 = self.peak_heights[i]
                    height2 = self.peak_heights[j]
                    
                    # 计算两山峰间距离
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # 如果足够近，添加连接点
                    if dist < 1.5:  # 最大连接距离
                        # 创建山脊线
                        ridge_steps = max(3, int(dist * 5))  # 基于距离的点数
                        for step in range(1, ridge_steps):
                            t = step / ridge_steps
                            ridge_x = pos1[0] * (1-t) + pos2[0] * t
                            ridge_y = pos1[1] * (1-t) + pos2[1] * t
                            
                            # 山脊高度 - 在两山峰间平滑过渡，添加一些起伏
                            base_height = height1 * (1-t) + height2 * t
                            # 添加随机扰动使山脊不完全平滑
                            ridge_height = base_height * np.random.uniform(0.7, 0.9)
                            
                            # 添加山脊点
                            all_positions.append([ridge_x, ridge_y])
                            all_heights.append(ridge_height)
        
        # 添加边界点以确保山脉在边缘平滑过渡到0
        num_boundary_points = 16
        boundary_radius = self.mountain_range / 2
        for i in range(num_boundary_points):
            angle = 2 * np.pi * i / num_boundary_points
            bx = boundary_radius * np.cos(angle)
            by = boundary_radius * np.sin(angle)
            all_positions.append([bx, by])
            all_heights.append(0.0)  # 边界点高度为0
        
        # 创建径向基函数插值器
        positions = np.array(all_positions)
        heights = np.array(all_heights)
        
        # 使用multiquadric函数并调整epsilon来控制平滑度
        self.rbf = Rbf(positions[:, 0], positions[:, 1], heights, 
                      function='multiquadric', epsilon=spread)
        
    def get_height(self, x, y):
        """获取位置(x,y)处的山脉高度"""
        if self.rbf is None:
            return 0.0
        
        # 计算与山脉中心的距离
        dist_from_center = np.sqrt(x**2 + y**2)
        
        # 如果超出山脉范围，返回0高度
        if dist_from_center > self.mountain_range / 2:
            # 平滑过渡区
            transition_zone = 0.5  # 过渡区宽度
            if dist_from_center < (self.mountain_range / 2 + transition_zone):
                # 计算在过渡区内的插值因子 (1 -> 0)
                factor = 1.0 - (dist_from_center - self.mountain_range / 2) / transition_zone
                # 平滑函数
                factor = np.cos((1.0 - factor) * np.pi / 2)
                # 使用RBF获取基本高度
                height = self.rbf(x, y)
                # 应用平滑因子
                return max(0, height * factor)
            else:
                return 0.0
            
        # 使用RBF获取高度
        height = self.rbf(x, y)
        
        # 确保山谷不会是负值（转换负值为低点但仍是正值）
        if height < 0:
            # 将负高度转换为低点（约为最高点的5-20%）
            max_height = max(self.peak_heights) if self.peak_heights else 1.0
            valley_factor = 0.05 + (abs(height) / max_height) * 0.15
            height = max_height * valley_factor
        
        return max(0, height)  # 确保不返回负值
    
    def is_collision(self, agent_pos):
        """检查智能体是否与山脉碰撞"""
        # 获取当前位置的山脉高度
        mountain_height = self.get_height(agent_pos[0], agent_pos[1])
        
        # 如果智能体z坐标小于山脉高度，则发生碰撞
        return agent_pos[2] < mountain_height

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
        # 山脉对象
        self.mountain = None
    
    def make_world(self):
        world = World()
        # 添加场景引用到世界对象
        world.scenario = self

        # 设置世界属性
        num_agents = 3
        num_landmarks = 1 + 0  # 1个目标点 + 0个障碍物
        world.dim_c = 0  # 无通信
        world.dim_p = 3  # 3D环境

        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.accel = 3.0  # 提高加速度
            agent.max_speed = 1.0  # 限制最大速度
            agent.is_3d = True

        # 添加地标（目标点和障碍物）
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1

        # 设置颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # 设置目标点为红色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.85, 0.35, 0.35])

        # 创建山脉障碍物
        if not hasattr(self, 'mountain') or self.mountain is None:
            # 为更复杂的连续山脉创建一个特殊的地标
            self.mountain = MountainLandmark()
            # 确保此山脉在world对象中可访问
            world.mountain = self.mountain
            # 设置山脉参数，创建更复杂的连续山脉结构
            self.mountain.setup_mountain(num_peaks=15, height_range=(0.5, 2.5), spread=0.7)
            print("已创建山脉障碍物环境")
        
        self.reset_world(world)
        return world
        
    def reset_world(self, world):
        # 设置山脉属性
        if not hasattr(self, 'mountain') or self.mountain is None:
            self.mountain = MountainLandmark()
            self.mountain.setup_mountain(num_peaks=12, height_range=(0.8, 2.0), spread=1.2)
            if not hasattr(world, 'landmarks_by_name'):
                world.landmarks_by_name = {}
            world.landmarks_by_name['mountain'] = self.mountain
            world.landmarks.append(self.mountain)
        
        # 初始化目标位置（确保它不为None）
        self.goal_pos = None
        
        # 设置目标点属性
        goal_exists = False
        for l in world.landmarks:
            if hasattr(l, 'name') and l.name == 'goal':
                goal_exists = True
                goal = l
                break
                
        if not goal_exists:
            # 创建目标点
            goal = Landmark()
            goal.name = 'goal'
            goal.collide = False
            goal.movable = False
            goal.size = 0.1
            # 使用类型兼容的方式设置颜色
            if not hasattr(goal, 'color') or goal.color is None:
                goal.color = np.array([0.1, 0.65, 0.15])  # 绿色
            else:
                goal.color[:] = np.array([0.1, 0.65, 0.15])
            
            # 确保目标点有有效的状态
            if not hasattr(goal, 'state') or goal.state is None:
                goal.state = EntityState()
            if hasattr(goal.state, 'p_pos') and goal.state.p_pos is None:
                goal.state.p_pos = np.zeros(3)
            if hasattr(goal.state, 'p_vel') and goal.state.p_vel is None:
                goal.state.p_vel = np.zeros(3)
                
            world.landmarks.append(goal)
        
        # 初始化智能体位置
        for i, agent in enumerate(world.agents):
            # 随机位置，但避开高山区域
            for _ in range(100):  # 最多尝试100次找到合适位置
                agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)
                agent.state.p_pos[2] = 1.0 + np.random.uniform(-0.1, +0.1)  # 初始高度轻微随机
                
                # 如果位置与山脉冲突，重试
                if self.mountain and hasattr(self.mountain, 'is_collision'):
                    if not self.mountain.is_collision(agent.state.p_pos):
                        break
            
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        # 设置随机高度的目标位置
        if hasattr(goal, 'state') and goal.state is not None:
            goal.state.p_pos = np.random.uniform(-2, +2, world.dim_p)
            goal.state.p_pos[2] = np.random.uniform(2.5, 3.0)  # Z轴高度
            goal.state.p_vel = np.zeros(world.dim_p)
            
            # 更新目标位置属性，确保不为None
            self.goal_pos = goal.state.p_pos.copy()
        else:
            print("警告：目标点状态未初始化")
            self.goal_pos = np.array([0.0, 0.0, 3.0])  # 默认目标位置

        # 设置随机高度的目标位置
        if hasattr(goal, 'state') and goal.state is not None:
            goal.state.p_pos = np.random.uniform(-2, +2, world.dim_p)
            goal.state.p_pos[2] = np.random.uniform(2.5, 3.0)  # Z轴高度
            
            # 更新目标位置属性，确保不为None
            self.goal_pos = goal.state.p_pos.copy()
            
            # 打印目标位置确认
            print(f"目标位置已设置: {self.goal_pos}")
        else:
            print("警告：目标点状态未初始化")
            self.goal_pos = np.array([0.0, 0.0, 3.0])  # 默认目标位置
    
    def is_collision(self, agent_pos):
        # 使用山脉的碰撞检测方法
        if hasattr(self, 'mountain') and self.mountain:
            # 调用山脉的碰撞检测方法
            return self.mountain.is_collision(agent_pos)
        return False
    
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
        
        # 山脉碰撞惩罚
        if self.mountain.is_collision(agent.state.p_pos):
            rew -= 5.0
            
        # 高度惩罚 - 防止飞得太高
        if agent.state.p_pos[2] > 5.0:
            rew -= (agent.state.p_pos[2] - 5.0) * 0.5
        
        return rew
        
    def observation(self, agent, world):
        # 基础观察
        obs = []
        
        # 添加自身位置和速度
        obs.append(agent.state.p_pos)
        obs.append(agent.state.p_vel)
        
        # 目标相对位置
        goal = [landmark for landmark in world.landmarks if landmark.name == 'goal'][0]
        goal_rel_pos = goal.state.p_pos - agent.state.p_pos
        
        # 归一化目标相对位置 - 增强方向信息
        goal_dist = np.linalg.norm(goal_rel_pos)
        goal_dir = goal_rel_pos / (goal_dist + 1e-6)
        
        # 分别添加目标方向和距离
        obs.append(goal_dir)  # 归一化方向
        obs.append([goal_dist])  # 距离
        
        # 添加当前位置的山脉高度信息
        mountain_height = self.mountain.get_height(agent.state.p_pos[0], agent.state.p_pos[1])
        obs.append([mountain_height])
        
        # 添加周围山脉高度采样 - 提供地形感知
        num_samples = 8  # 采样点数量
        sample_radius = 0.5  # 采样半径
        
        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            sample_x = agent.state.p_pos[0] + sample_radius * np.cos(angle)
            sample_y = agent.state.p_pos[1] + sample_radius * np.sin(angle)
            height = self.mountain.get_height(sample_x, sample_y)
            obs.append([height])
        
        # 确保所有观察都是浮点数组
        obs = np.concatenate(obs)
        return obs
        
    def info(self, agent, world):
        """返回额外信息用于可视化和调试"""
        # 为了可视化，返回山峰位置和高度
        info = {
            'goal_pos': self.goal_pos,
            'mountain_peaks': self.mountain.peak_positions,
            'mountain_heights': self.mountain.peak_heights,
            'mountain_range': self.mountain.mountain_range
        }
        return info
    
    def done(self, agent, world):
        """判断是否完成任务的条件"""
        return False

    def render(self, mode='human'):
        """实现渲染函数，显示山脉地形"""
        from multiagent.rendering import Viewer
        
        if self.viewer is None:
            # 创建固定大小的窗口，确保一致性
            self.viewer = Viewer(width=800, height=600)
            
            # 创建山脉网格
            try:
                # 添加网格渲染方法
                self.render_mountain()
            except Exception as e:
                print(f"无法渲染山脉：{e}")
                import traceback
                traceback.print_exc()
        
        # 使用默认渲染更新视图
        return self.viewer.render(return_rgb_array=mode=='rgb_array')
    
    def render_mountain(self):
        """渲染山脉地形"""
        if not self.viewer or not self.mountain:
            return
            
        # 创建地形网格 (简化版)
        try:
            # 设置网格参数 - 减少网格复杂度以避免性能问题
            grid_size = 12  # 减少网格大小
            mountain_range = self.mountain.mountain_range / 2
            x_min, x_max = -mountain_range, mountain_range
            y_min, y_max = -mountain_range, mountain_range
            
            # 使用更小的值来表示山脉，避免坐标过大
            scale = 1.0  # 缩放值
            
            # 添加基底面
            base_vertices = [
                (-mountain_range * scale, -mountain_range * scale),
                (-mountain_range * scale, mountain_range * scale),
                (mountain_range * scale, mountain_range * scale),
                (mountain_range * scale, -mountain_range * scale)
            ]
            base = FilledPolygon(base_vertices)
            base.set_color(0.7, 0.7, 0.7, 1)  # 浅灰色底板
            self.viewer.add_geom(base)
            
            # 预先计算所有高度值，找出最大高度用于颜色映射
            heights = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = x_min + (x_max - x_min) * i / (grid_size - 1)
                    y = y_min + (y_max - y_min) * j / (grid_size - 1)
                    heights.append(self.mountain.get_height(x, y))
            
            max_height = max(heights) if heights else 1.0
            
            # 使用简化的方法渲染山脉 - 仅渲染一组四边形
            for i in range(grid_size-1):
                for j in range(grid_size-1):
                    # 计算四个顶点位置 (物理坐标)
                    x1 = x_min + (x_max - x_min) * i / (grid_size - 1)
                    y1 = y_min + (y_max - y_min) * j / (grid_size - 1)
                    x2 = x_min + (x_max - x_min) * (i+1) / (grid_size - 1)
                    y2 = y_min + (y_max - y_min) * (j+1) / (grid_size - 1)
                    
                    # 获取高度值
                    z11 = self.mountain.get_height(x1, y1)
                    z12 = self.mountain.get_height(x1, y2)
                    z21 = self.mountain.get_height(x2, y1)
                    z22 = self.mountain.get_height(x2, y2)
                    
                    # 仅渲染有足够高度的区域 - 提高性能
                    if max(z11, z12, z21, z22) < 0.05:
                        continue
                    
                    # 计算颜色 (基于高度) - 使用整数值而非浮点数
                    c11 = [0.4 + 0.4*(z11/max_height), 0.3 + 0.2*(z11/max_height), 0.1, 1]
                    c12 = [0.4 + 0.4*(z12/max_height), 0.3 + 0.2*(z12/max_height), 0.1, 1]
                    c21 = [0.4 + 0.4*(z21/max_height), 0.3 + 0.2*(z21/max_height), 0.1, 1]
                    c22 = [0.4 + 0.4*(z22/max_height), 0.3 + 0.2*(z22/max_height), 0.1, 1]
                    
                    # 取平均颜色
                    avg_r = int((c11[0] + c12[0] + c21[0] + c22[0]) / 4 * 255)
                    avg_g = int((c11[1] + c12[1] + c21[1] + c22[1]) / 4 * 255)
                    avg_b = int((c11[2] + c12[2] + c21[2] + c22[2]) / 4 * 255)
                    
                    # 为有高度的区域创建填充多边形 (单个四边形)
                    vertices = [
                        (x1 * scale, y1 * scale),
                        (x1 * scale, y2 * scale),
                        (x2 * scale, y2 * scale),
                        (x2 * scale, y1 * scale)
                    ]
                    
                    # 创建颜色渐变多边形
                    quad = FilledPolygon(vertices)
                    quad.set_color(avg_r/255, avg_g/255, avg_b/255, 1)
                    self.viewer.add_geom(quad)
            
            # 添加主要山峰标记
            if hasattr(self.mountain, 'peak_positions') and hasattr(self.mountain, 'peak_heights'):
                for i, (pos, height) in enumerate(zip(self.mountain.peak_positions, self.mountain.peak_heights)):
                    # 仅标记较高的山峰
                    if height > max_height * 0.6:
                        # 创建山峰标记 - 使用三角形
                        size = 0.1 + 0.05 * (height / max_height)  # 根据高度调整大小
                        peak_mark = FilledPolygon([
                            (pos[0] * scale, pos[1] * scale),
                            ((pos[0]-size) * scale, (pos[1]-size) * scale),
                            ((pos[0]+size) * scale, (pos[1]-size) * scale)
                        ])
                        # 使用整数值设置颜色
                        peak_mark.set_color(1, 0, 0, 1)  # 使用整数值
                        self.viewer.add_geom(peak_mark)
        
        except Exception as e:
            print(f"渲染山脉网格失败: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """关闭资源的函数"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()
                self.viewer = None
            except:
                pass 
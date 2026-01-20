"""
统一的奖励函数模块
包含所有场景共用的奖励机制，确保一致性
"""

import numpy as np
import os


class UnifiedRewardCalculator:
    """
    统一的奖励计算器
    包含所有场景共用的奖励机制
    """
    
    def __init__(self, scenario):
        """
        初始化奖励计算器
        
        Args:
            scenario: 场景实例，用于访问地形和障碍物信息
        """
        self.scenario = scenario
        
        # 从环境变量获取奖励权重
        self.lateral_weight = float(os.getenv('LATERAL_WEIGHT', '0.3'))
        self.clearance_weight = float(os.getenv('CLEARANCE_WEIGHT', '0.4'))
        
        # 奖励裁剪范围
        self.reward_clip_min = float(os.getenv('REWARD_CLIP_MIN', '-50.0'))
        self.reward_clip_max = float(os.getenv('REWARD_CLIP_MAX', '50.0'))
    
    def calculate_obstacle_penalties(self, agent, world):
        """
        计算障碍物相关惩罚和奖励
        
        Returns:
            tuple: (total_penalty, debug_info)
        """
        total_penalty = 0.0
        debug_info = {}
        
        if not (hasattr(self.scenario, 'obstacles') and self.scenario.obstacles):
            return total_penalty, debug_info
        
        current_pos = agent.state.p_pos
        
        for i, obstacle_data in enumerate(self.scenario.obstacles):
            if 'center' not in obstacle_data or 'radius' not in obstacle_data:
                continue
                
            obstacle_center = np.array(obstacle_data['center'])
            obstacle_radius = obstacle_data['radius']
            
            # 计算到障碍物中心的距离
            dist_to_obstacle = np.linalg.norm(current_pos - obstacle_center)
            
            # 1. 障碍物穿透惩罚
            if dist_to_obstacle < obstacle_radius:
                penetration_depth = obstacle_radius - dist_to_obstacle
                obstacle_penalty = -penetration_depth * 30.0
                total_penalty += obstacle_penalty
                
                debug_info[f'obstacle_{i}_penetration_penalty'] = obstacle_penalty
                
                # 严重穿透惩罚
                if penetration_depth > 1.0:
                    severe_penalty = -(penetration_depth - 1.0) * 60.0
                    total_penalty += severe_penalty
                    debug_info[f'obstacle_{i}_severe_penalty'] = severe_penalty
            
            # 2. 接近障碍物的警告惩罚
            elif dist_to_obstacle < obstacle_radius + 2.0:
                proximity_penalty = -(2.0 - (dist_to_obstacle - obstacle_radius)) * 5.0
                total_penalty += proximity_penalty
                debug_info[f'obstacle_{i}_proximity_penalty'] = proximity_penalty
                
                # 3. 侧向/绕行奖励机制
                if dist_to_obstacle < obstacle_radius + 1.5:
                    lateral_reward = self._calculate_lateral_reward(
                        agent, obstacle_center, obstacle_radius, i
                    )
                    total_penalty += lateral_reward  # 注意：这里可能是负值
                    debug_info[f'obstacle_{i}_lateral_reward'] = lateral_reward
        
        return total_penalty, debug_info
    
    def _calculate_lateral_reward(self, agent, obstacle_center, obstacle_radius, obstacle_idx):
        """
        计算侧向/绕行奖励
        
        Args:
            agent: 智能体
            obstacle_center: 障碍物中心
            obstacle_radius: 障碍物半径
            obstacle_idx: 障碍物索引
            
        Returns:
            float: 侧向奖励值
        """
        current_pos = agent.state.p_pos
        
        # 计算从障碍物到智能体的法线向量
        obstacle_to_agent = current_pos - obstacle_center
        obstacle_to_agent[2] = 0  # 忽略Z轴，只考虑水平方向
        obstacle_to_agent_norm = obstacle_to_agent / (np.linalg.norm(obstacle_to_agent) + 1e-6)
        
        # 计算智能体的移动方向
        if np.linalg.norm(agent.state.p_vel) <= 0.1:
            return 0.0
            
        vel_direction = agent.state.p_vel.copy()
        vel_direction[2] = 0  # 忽略Z轴，只考虑水平方向
        vel_direction_norm = vel_direction / (np.linalg.norm(vel_direction) + 1e-6)
        
        # 计算侧向移动成分
        lateral_component = np.dot(vel_direction_norm, obstacle_to_agent_norm)
        
        # 如果智能体在侧向移动
        if abs(lateral_component) < 0.5:
            lateral_progress = np.linalg.norm(vel_direction) * (1 - abs(lateral_component))
            lateral_reward = self.lateral_weight * lateral_progress
            return lateral_reward
        
        return 0.0
    
    def calculate_clearance_reward(self, agent):
        """
        计算净空/最小距离增益奖励
        
        Args:
            agent: 智能体
            
        Returns:
            tuple: (clearance_reward, debug_info)
        """
        debug_info = {}
        
        if not (hasattr(self.scenario, 'obstacles') and self.scenario.obstacles):
            return 0.0, debug_info
        
        current_pos = agent.state.p_pos
        
        # 计算当前时刻到所有障碍物的最小距离
        min_distances = []
        for obstacle_data in self.scenario.obstacles:
            if 'center' in obstacle_data and 'radius' in obstacle_data:
                obstacle_center = np.array(obstacle_data['center'])
                obstacle_radius = obstacle_data['radius']
                dist_to_obstacle = np.linalg.norm(current_pos - obstacle_center)
                min_distances.append(dist_to_obstacle - obstacle_radius)
        
        if not min_distances:
            return 0.0, debug_info
        
        d_min_current = min(min_distances)
        
        # 获取上一时刻的最小距离
        if not hasattr(agent, 'last_min_distance'):
            agent.last_min_distance = d_min_current
        
        d_min_previous = agent.last_min_distance
        
        # 计算距离变化
        distance_change = d_min_current - d_min_previous
        
        # 归一化因子
        D_max = 50.0
        normalized_change = distance_change / D_max
        clipped_change = np.clip(normalized_change, -1.0, 1.0)
        
        # 计算净空奖励
        clearance_reward = self.clearance_weight * clipped_change
        
        # 更新上一时刻的最小距离
        agent.last_min_distance = d_min_current
        
        # 调试信息
        debug_info.update({
            'clearance_reward': clearance_reward,
            'd_min_current': d_min_current,
            'd_min_previous': d_min_previous,
            'distance_change': distance_change,
            'normalized_change': normalized_change
        })
        
        return clearance_reward, debug_info
    
    def calculate_terrain_penalty(self, agent):
        """
        计算地形穿透惩罚
        
        Args:
            agent: 智能体
            
        Returns:
            tuple: (terrain_penalty, debug_info)
        """
        debug_info = {}
        
        if not hasattr(self.scenario, 'get_terrain_height'):
            return 0.0, debug_info
        
        current_pos = agent.state.p_pos
        terrain_height = self.scenario.get_terrain_height(current_pos[0], current_pos[1])
        
        if current_pos[2] < terrain_height:
            penetration_depth = terrain_height - current_pos[2]
            penetration_penalty = -penetration_depth * 20.0
            debug_info['penetration_penalty'] = penetration_penalty
            
            # 严重穿透惩罚
            if penetration_depth > 2.0:
                severe_penalty = -(penetration_depth - 2.0) * 50.0
                debug_info['severe_penalty'] = severe_penalty
                return penetration_penalty + severe_penalty, debug_info
            
            return penetration_penalty, debug_info
        
        return 0.0, debug_info
    
    def clip_reward(self, reward):
        """
        裁剪奖励值到指定范围
        
        Args:
            reward: 原始奖励值
            
        Returns:
            float: 裁剪后的奖励值
        """
        return np.clip(reward, self.reward_clip_min, self.reward_clip_max)


def create_unified_reward_calculator(scenario):
    """
    创建统一的奖励计算器实例
    
    Args:
        scenario: 场景实例
        
    Returns:
        UnifiedRewardCalculator: 奖励计算器实例
    """
    return UnifiedRewardCalculator(scenario)

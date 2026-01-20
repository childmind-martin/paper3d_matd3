"""
扩展的环境类，将动作空间从5维扩展到7维，以支持完整的势场力参数
"""

import numpy as np
from gym import spaces
from .environment import MultiAgentEnv

class ExtendedMultiAgentEnv(MultiAgentEnv):
    """
    扩展的多智能体环境，支持7维动作空间
    """
    
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=False):
        """
        初始化环境
        """
        # 调用父类初始化方法
        super(ExtendedMultiAgentEnv, self).__init__(world, 
                                                   reset_callback, 
                                                   reward_callback,
                                                   observation_callback, 
                                                   info_callback,
                                                   done_callback, 
                                                   post_step_callback,
                                                   shared_viewer, 
                                                   discrete_action)
        
        # 重新配置动作空间为7维
        self._configure_extended_action_space()
        
        print("初始化扩展环境，动作空间扩展为7维")
    
    def _configure_extended_action_space(self):
        """
        重新配置动作空间为7维
        - 3维用于加速度控制(x,y,z)
        - 4维用于势场力参数控制(λa, λ1, λr, R_safe)
        """
        # 保存原始动作空间尺寸，以备将来参考
        self.original_action_spaces = self.action_space.copy() if hasattr(self, 'action_space') else []
        
        # 重新创建7维动作空间
        self.action_space = []
        
        for agent in self.agents:
            # 创建7维的Box空间
            # 所有维度的范围都是[-1,1]，网络将在使用时映射到适当的范围
            extended_action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(7,),  # 7维动作空间
                dtype=np.float32
            )
            self.action_space.append(extended_action_space)
            
            # 设置智能体的动作空间
            if hasattr(agent, 'action_space'):
                agent.action_space = extended_action_space
                
        print(f"成功将{len(self.action_space)}个智能体的动作空间扩展为7维")
    
    def _set_action(self, action, agent, action_space, time=None):
        """
        重写动作设置方法，处理7维动作
        
        参数:
            action: 7维动作向量 [x,y,z,λa,λ1,λr,R_safe]
            agent: 执行动作的智能体
            action_space: 动作空间
            time: 时间步
        """
        # 检查动作维度
        if isinstance(action, np.ndarray) and action.shape[0] >= 7:
            # 使用前3维作为物理动作
            physical_action = action[:3].copy()
            # 剩余4维是势场力参数，在其他地方处理
            
            # 调用原始方法处理物理动作部分
            super(ExtendedMultiAgentEnv, self)._set_action(physical_action, agent, action_space, time)
            
            # 将完整7维动作保存到智能体，以便势场修正器使用
            if not hasattr(agent, 'full_action'):
                agent.full_action = action.copy()
            else:
                agent.full_action[:] = action
                
            # 如果需要，也可以分别保存势场参数部分
            if not hasattr(agent, 'force_params'):
                agent.force_params = action[3:].copy()
            else:
                agent.force_params[:] = action[3:]
        else:
            # 如果维度不匹配，使用原始方法处理
            super(ExtendedMultiAgentEnv, self)._set_action(action, agent, action_space, time) 
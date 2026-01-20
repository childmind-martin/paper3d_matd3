#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MADDPG优化版模型评估与可视化脚本
仿照1.0版本功能，支持模型加载、评估和可视化生成
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import traceback
import json
import time
import math

# 设置环境变量抑制多智能体环境警告
os.environ['SUPPRESS_MA_PROMPT'] = '1'
# 🔧 抑制 TensorFlow/XLA 警告（如 cuFFT 注册警告）
# TF_CPP_MIN_LOG_LEVEL: 0=全部日志, 1=INFO及以上, 2=WARNING及以上, 3=ERROR及以上
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 抑制警告，保留错误信息

# 可视化依赖（非交互后端）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入优化的模块
from paper3d_train_optimized import (
    OptimizedMADDPG, 
    OptimizedMATD3,
    load_scenario_module, 
    configure_gpu, 
    try_apply_scenario_params, 
    build_continuous_action_network, 
    build_continuous_critic_network,
    build_continuous_critic_network_matd3
)
from visualization.trajectory_visualizer import TrajectoryVisualizer
from utils.observation_processor import ObservationProcessor

# 导入环境
from multiagent.environment import MultiAgentEnv

class ModelEvaluator:
    """模型评估器，仿照1.0版本的评估逻辑"""
    
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.setup_visualizer()
        
    def setup_environment(self):
        """初始化环境"""
        print("初始化评估环境...")
        
        # 配置GPU
        configure_gpu()
        
        # 根据场景名称选择场景，支持新的场景选择逻辑
        scenario_name = self.args.scenario_name
        print(f"使用场景: {scenario_name}")
        
        # 加载场景
        self.scenario = load_scenario_module(scenario_name, self.args)
        if self.scenario is None:
            raise RuntimeError(f"无法加载场景: {scenario_name}")
        self.world = self.scenario.make_world()
        # 应用重力、控制增益与奖励缩放（仅在显式提供时覆盖）
        try:
            if hasattr(self.world, 'gravity') and getattr(self.args, 'gravity', None) is not None:
                self.world.gravity = float(self.args.gravity)
                print(f"已设置评估环境重力: gravity={self.world.gravity}")
            if hasattr(self.world, 'control_accel_gain') and getattr(self.args, 'control_accel_gain', None) is not None:
                self.world.control_accel_gain = float(self.args.control_accel_gain)
                print(f"已设置控制加速度增益: control_accel_gain={self.world.control_accel_gain}")
            if hasattr(self.world, 'reward_pos_scale') and getattr(self.args, 'reward_pos_scale', None) is not None:
                self.world.reward_pos_scale = float(self.args.reward_pos_scale)
            if hasattr(self.world, 'reward_neg_scale') and getattr(self.args, 'reward_neg_scale', None) is not None:
                self.world.reward_neg_scale = float(self.args.reward_neg_scale)
        except Exception as _e:
            print(f"评估环境设置物理/奖励缩放失败: {_e}")

        # 应用智能体速度/加速度（若提供）
        try:
            if getattr(self.args, 'agent_max_speed', None) is not None or getattr(self.args, 'agent_accel', None) is not None:
                for ag in getattr(self.world, 'agents', []):
                    if getattr(self.args, 'agent_max_speed', None) is not None and hasattr(ag, 'max_speed'):
                        ag.max_speed = float(self.args.agent_max_speed)
                    if getattr(self.args, 'agent_accel', None) is not None and hasattr(ag, 'accel'):
                        ag.accel = float(self.args.agent_accel)
        except Exception as _e:
            print(f"评估环境应用速度/加速度失败: {_e}")

        # 将可能影响避障/检测的参数尽量下发到场景/世界（若存在对应属性）
        try:
            try_apply_scenario_params(self.scenario, self.world, self.args, tqdm_file=None)
        except Exception:
            pass
        
        # 🚨 关键修复：确保碰撞检测参数被正确设置到场景对象
        # 即使try_apply_scenario_params没有处理这些参数，也要手动设置
        try:
            if hasattr(self.args, 'collision_distance_threshold') and self.args.collision_distance_threshold is not None:
                if hasattr(self.scenario, 'collision_distance_threshold'):
                    self.scenario.collision_distance_threshold = float(self.args.collision_distance_threshold)
                    print(f"✅ 已设置碰撞距离阈值: {self.scenario.collision_distance_threshold}")
            if hasattr(self.args, 'collision_penalty_value') and self.args.collision_penalty_value is not None:
                if hasattr(self.scenario, 'collision_penalty_value'):
                    self.scenario.collision_penalty_value = float(self.args.collision_penalty_value)
                    print(f"✅ 已设置碰撞惩罚值: {self.scenario.collision_penalty_value}")
        except Exception as e:
            print(f"⚠️  设置碰撞检测参数失败: {e}")
        
        # 创建环境
        self.env = MultiAgentEnv(
            self.world,
            self.scenario.reset_world,
            self.scenario.reward,
            self.scenario.observation,
            info_callback=None,
            shared_viewer=False
        )
        # 应用动作范围映射（仅在显式提供任一轴时覆盖）
        try:
            ax = getattr(self.args, 'action_range_x', None)
            ay = getattr(self.args, 'action_range_y', None)
            az = getattr(self.args, 'action_range_z', None)
            if any(v is not None for v in (ax, ay, az)) and hasattr(self.env, 'world'):
                current = getattr(self.env.world, 'action_range', None)
                if isinstance(current, (list, tuple)) and len(current) >= 3:
                    new_range = [float(current[0]), float(current[1]), float(current[2])]
                else:
                    new_range = [1.0, 1.0, 1.0]
                if ax is not None:
                    new_range[0] = float(ax)
                if ay is not None:
                    new_range[1] = float(ay)
                if az is not None:
                    new_range[2] = float(az)
                self.env.world.action_range = new_range
        except Exception:
            pass
        
        # 获取环境信息
        self.n_agents = self.env.n
        base_obs_shapes = [self.env.observation_space[i].shape[0] for i in range(self.n_agents)]
        
        # 🔧 关键修复：从训练配置（results.json）中读取训练时使用的观测维度
        # 优先从results.json读取，确保与训练时完全一致
        training_obs_shapes = None
        training_use_pf = None
        if hasattr(self.args, 'load_model_path') and self.args.load_model_path:
            try:
                # 尝试从模型路径找到results.json
                model_path = self.args.load_model_path
                # 移除 final/best/epXXX 等子目录
                if model_path.endswith(('final', 'best')) or '/ep' in os.path.basename(model_path):
                    model_base_dir = os.path.dirname(model_path)
                else:
                    model_base_dir = model_path
                
                exp_name = os.path.basename(model_base_dir)
                potential_log_dirs = [
                    os.path.join("logs", exp_name),
                    model_base_dir,
                    os.path.dirname(model_base_dir),
                ]
                
                for log_dir in potential_log_dirs:
                    # 查找results.json（可能在子目录中）
                    results_files = []
                    if os.path.isdir(log_dir):
                        # 在log_dir及其子目录中查找results.json
                        for root, dirs, files in os.walk(log_dir):
                            if 'results.json' in files:
                                results_files.append(os.path.join(root, 'results.json'))
                    
                    for results_file in results_files:
                        try:
                            with open(results_file, 'r', encoding='utf-8') as f:
                                results = json.load(f)
                            
                            # 🔧 关键修复：从args字典中读取配置（results.json格式：{'args': {...}}）
                            training_args = None
                            if 'args' in results and isinstance(results['args'], dict):
                                training_args = results['args']
                            elif isinstance(results, dict) and 'base_obs_shapes' in results:
                                # 向后兼容：如果args不在顶层，尝试从顶层读取
                                training_args = results
                            
                            if training_args is not None:
                                # 读取训练时的配置
                                if 'base_obs_shapes' in training_args:
                                    training_obs_shapes = training_args['base_obs_shapes']
                                    print(f"✅ 从训练配置读取观测维度: {training_obs_shapes}")
                                
                                if 'use_pf_feature' in training_args:
                                    training_use_pf = bool(training_args['use_pf_feature'])
                                    print(f"✅ 从训练配置读取PF特征标志: {training_use_pf}")
                            
                            if training_obs_shapes is not None:
                                break
                        except Exception as e:
                            continue
                    
                    if training_obs_shapes is not None:
                        break
            except Exception as e:
                print(f"⚠️  读取训练配置失败: {e}")
        
        # 🚨 关键修复：观测维度应该与训练时完全一致
        # 训练时PF特征是作为独立输入传递的，不是追加到观测中
        # 因此obs_shapes应该保持基础观测维度（81维），而不是84维
        if training_obs_shapes is not None:
            # 直接使用训练时的观测维度（训练时PF特征是独立输入，观测维度不包含PF特征）
            self.obs_shapes = training_obs_shapes
            print(f"✅ 使用训练时的观测维度: {self.obs_shapes} (PF特征作为独立输入，不包含在观测维度中)")
        else:
            # 回退到当前配置：使用基础观测维度（不含PF特征）
            # 因为PF特征是作为独立输入传递的，不应该追加到观测维度中
            self.obs_shapes = base_obs_shapes
            print(f"ℹ️  未找到训练配置，使用基础观测维度: {self.obs_shapes} (PF特征将作为独立输入传递)")
        
        self.action_dims = [7] * self.n_agents
        
        print(f"环境初始化完成:")
        print(f"  - 智能体数量: {self.n_agents}")
        print(f"  - 观察空间维度: {self.obs_shapes}")
        print(f"  - 动作空间维度: {self.action_dims}")
        
    def setup_visualizer(self):
        """初始化可视化器"""
        self.visualizer = TrajectoryVisualizer() if not self.args.disable_visualization else None
    
    def select_actions_eval(self, processed_obs, use_fr=False, use_pf=False):
        """
        评估时的动作选择，兼容MADDPG和MATD3
        
        Args:
            processed_obs: (n_agents, obs_dim) 或 (batch_size, n_agents, obs_dim)
            use_fr: 是否使用填充率特征（作为单独输入）
            use_pf: 是否使用势场特征（作为单独输入，如果启用）
        
        Returns:
            actions: (n_agents, action_dim) 或 (batch_size, n_agents, action_dim) - TensorFlow tensor
        """
        # 🔧 性能优化：使用tf.function装饰器加速（与训练代码一致）
        return self._select_actions_eval_tf(processed_obs, use_fr, use_pf)
    
    @tf.function(reduce_retracing=True)
    def _select_actions_eval_tf(self, processed_obs, use_fr, use_pf):
        """
        评估时的动作选择（TensorFlow图模式，性能优化）
        
        Args:
            processed_obs: TensorFlow tensor (n_agents, obs_dim) 或 (batch_size, n_agents, obs_dim)
            use_fr: bool 是否使用填充率特征
            use_pf: bool 是否使用势场特征
        
        Returns:
            actions: TensorFlow tensor (n_agents, action_dim) 或 (batch_size, n_agents, action_dim)
        """
        # 🔧 修复：Actor网络期望的输入结构
        # 如果 use_pf_feature=True，Actor期望3个输入：[obs, fr_input, pf_input]
        # 如果 use_fr_feature=True 但 use_pf_feature=False，Actor期望2个输入：[obs, fr_input]
        # 如果两者都False，Actor期望1个输入：[obs]
        
        # 确保是tensor
        processed_obs = tf.convert_to_tensor(processed_obs, dtype=tf.float32)
        
        # 确保有批次维度
        if len(processed_obs.shape) == 2:  # (n_agents, obs_dim)
            processed_obs = tf.expand_dims(processed_obs, axis=0)  # (1, n_agents, obs_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = tf.shape(processed_obs)[0]
        actions_list = []
        
        # 🔧 获取PF特征维度（从训练配置或网络结构推断）
        pf_feature_dim = 3  # 固定为3（PF力是3维向量）
        
        # 🔧 关键修复：计算真实的FR值（action_force_ratio），而不是使用零向量
        # 训练时FR值会随时间变化（schedule），评估时应该使用固定的FR值
        # 🔧 性能优化：在tf.function外部缓存FR值，避免每次调用时获取
        if not hasattr(self, '_cached_fr_value'):
            action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
            self._cached_fr_value = float(action_force_ratio)
        fr_value = tf.cast(self._cached_fr_value, tf.float32)
        
        # 🔧 性能优化：批量处理所有智能体（与训练代码一致）
        for i in range(self.n_agents):
            # 提取当前智能体的观测（基础观测，不包含PF特征）
            obs_dim = self.obs_shapes[i] if i < len(self.obs_shapes) else processed_obs.shape[2]
            agent_obs = processed_obs[:, i, :obs_dim]  # (batch_size, obs_dim)
            
            # 构建输入：obs + fr_input（如果启用）+ pf_input（如果启用）
            actor_inputs = [agent_obs]
            if use_fr:
                # 🔧 关键修复：使用真实的FR值，而不是零向量
                fr_input = tf.fill([batch_size, 1], fr_value)  # (batch_size, 1)
                actor_inputs.append(fr_input)
            if use_pf:
                # 🔧 关键修复：计算真实的PF特征（从势场力中提取）
                # 注意：这里我们需要先计算势场力，然后提取PF特征
                # 为了简化，我们先使用零向量，但应该从实际的势场力计算中提取
                # TODO: 从实际的势场力计算中提取PF特征
                # 临时方案：使用零向量（与训练时PF特征的计算方式一致）
                pf_placeholder = tf.zeros([batch_size, pf_feature_dim], dtype=tf.float32)
                actor_inputs.append(pf_placeholder)
            
            # 调用actor
            if len(actor_inputs) == 1:
                agent_actions = self.maddpg.agents[i]['actor'](actor_inputs[0], training=False)
            else:
                agent_actions = self.maddpg.agents[i]['actor'](actor_inputs, training=False)
            
            actions_list.append(agent_actions)
        
        # 堆叠为 (batch_size, n_agents, action_dim)
        actions = tf.stack(actions_list, axis=1)
        
        # 如果输入没有批次维度，移除输出的批次维度
        if squeeze_output:
            actions = actions[0]  # (n_agents, action_dim)
        
        # 🔧 性能优化：返回tensor而不是numpy，延迟转换到env.step时
        return actions
        
    def load_model(self):
        """加载训练好的模型"""
        # 🔧 新增：检测并加载元优化基准配置
        meta_baseline_config = None
        # 从模型路径提取实验名称
        # 例如：models/调试分离梯度、无重力、无早停、预热、随机地图、高变FR低高低_exp_20251201_112141/final
        # 提取：调试分离梯度、无重力、无早停、预热、随机地图、高变FR低高低_exp_20251201_112141
        model_path = self.args.load_model_path
        # 移除 final/best/epXXX 等子目录
        if model_path.endswith(('final', 'best')) or '/ep' in os.path.basename(model_path):
            model_base_dir = os.path.dirname(model_path)
        else:
            model_base_dir = model_path
        
        exp_name = os.path.basename(model_base_dir)
        
        # 尝试查找元优化配置文件（在logs目录下）
        potential_log_dirs = [
            os.path.join("logs", exp_name),  # logs/{exp_name}
            model_base_dir,  # models/{exp_name}
            os.path.dirname(model_base_dir),  # models/{exp_name}/..
        ]
        
        for log_dir in potential_log_dirs:
            meta_baseline_file = os.path.join(log_dir, "pf_meta_baseline.json")
            if os.path.isfile(meta_baseline_file):
                try:
                    with open(meta_baseline_file, 'r', encoding='utf-8') as f:
                        meta_baseline_config = json.load(f)
                    print(f"✅ 成功加载元优化配置: {meta_baseline_file}")
                    # 将元优化配置应用到args
                    if 'goal_attraction' in meta_baseline_config:
                        self.args.goal_attraction = float(meta_baseline_config['goal_attraction'])
                    if 'lambda_1_base' in meta_baseline_config:
                        self.args.lambda_1_base = float(meta_baseline_config['lambda_1_base'])
                    if 'terrain_repulsion' in meta_baseline_config:
                        self.args.terrain_repulsion = float(meta_baseline_config['terrain_repulsion'])
                    if 'agent_influence_range' in meta_baseline_config:
                        self.args.agent_influence_range = float(meta_baseline_config['agent_influence_range'])
                    if 'delta_k_att' in meta_baseline_config:
                        self.args.delta_k_att = float(meta_baseline_config['delta_k_att'])
                    if 'delta_lambda_1' in meta_baseline_config:
                        self.args.delta_lambda_1 = float(meta_baseline_config['delta_lambda_1'])
                    if 'delta_k_rep' in meta_baseline_config:
                        self.args.delta_k_rep = float(meta_baseline_config['delta_k_rep'])
                    if 'delta_radius' in meta_baseline_config:
                        self.args.delta_radius = float(meta_baseline_config['delta_radius'])
                    if 'action_force_ratio' in meta_baseline_config:
                        self.args.action_force_ratio = float(meta_baseline_config['action_force_ratio'])
                    break
                except Exception as e:
                    print(f"⚠️  元优化配置加载失败: {e}")
        
        if meta_baseline_config is None:
            print(f"ℹ️  未找到元优化配置文件: {os.path.join(potential_log_dirs[0] if potential_log_dirs else 'logs', 'pf_meta_baseline.json')}")
            print(f"   使用默认势场参数")
        
        # 🔧 关键修复：从训练配置（results.json）中读取训练时使用的特征标志、ACTION_FORCE_RATIO和动作范围
        # 优先从results.json读取，确保与训练时完全一致
        training_use_fr = None
        training_use_pf = None
        training_pf_feature_dim = None
        best_episode_force_ratio = None  # 🔧 关键：只读取最佳回合的FR值，不进行任何回退
        best_episode_num = None  # 🔧 新增：保存最佳回合编号
        training_action_range_x = None
        training_action_range_y = None
        training_action_range_z = None
        training_episode_length = None  # 🚨 新增：读取训练时的episode_length
        training_actor_hidden = None  # 🚨 新增：读取训练时的actor_hidden
        training_critic_hidden = None  # 🚨 新增：读取训练时的critic_hidden
        training_gravity = None  # 🔧 新增：读取训练时的gravity
        training_damping = None  # 🔧 新增：读取训练时的damping
        training_agent_max_speed = None  # 🔧 新增：读取训练时的agent_max_speed
        training_agent_accel = None  # 🔧 新增：读取训练时的agent_accel
        training_control_accel_gain = None  # 🔧 新增：读取训练时的control_accel_gain
        for log_dir in potential_log_dirs:
            # 查找results.json（可能在子目录中）
            results_files = []
            if os.path.isdir(log_dir):
                # 在log_dir及其子目录中查找results.json
                for root, dirs, files in os.walk(log_dir):
                    if 'results.json' in files:
                        results_files.append(os.path.join(root, 'results.json'))
            
            for results_file in results_files:
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    # 🔧 关键修复：优先从顶层读取最佳回合的FR值（无论是否有args，都应该先检查顶层）
                    # best_episode_force_ratio 保存在 results.json 的顶层，与 best_episode 同级
                    if 'best_episode_force_ratio' in results:
                        best_episode_force_ratio = float(results['best_episode_force_ratio'])
                        if 'best_episode' in results:
                            best_episode_num = int(results['best_episode']) + 1
                        print(f"✅ 从训练配置读取最佳回合的FR值: {best_episode_force_ratio} (回合 {best_episode_num if best_episode_num is not None else '?'})")
                    
                    # 读取训练时的特征标志
                    if 'args' in results and isinstance(results['args'], dict):
                        # 从args字典中读取
                        if 'use_fr_feature' in results['args']:
                            training_use_fr = bool(results['args']['use_fr_feature'])
                        if 'use_pf_feature' in results['args']:
                            training_use_pf = bool(results['args']['use_pf_feature'])
                        # 🔧 关键修复：读取训练时的pf_feature_dim
                        if 'pf_feature_dim' in results['args']:
                            training_pf_feature_dim = int(results['args']['pf_feature_dim'])
                        # 🔧 关键修复：读取训练时的动作范围参数
                        if 'action_range_x' in results['args']:
                            training_action_range_x = float(results['args']['action_range_x'])
                        if 'action_range_y' in results['args']:
                            training_action_range_y = float(results['args']['action_range_y'])
                        if 'action_range_z' in results['args']:
                            training_action_range_z = float(results['args']['action_range_z'])
                        # 🚨 新增：读取训练时的episode_length
                        if 'episode_length' in results['args']:
                            training_episode_length = int(results['args']['episode_length'])
                        # 🚨 新增：读取训练时的网络结构配置
                        if 'actor_hidden' in results['args']:
                            training_actor_hidden = str(results['args']['actor_hidden'])
                        if 'critic_hidden' in results['args']:
                            training_critic_hidden = str(results['args']['critic_hidden'])
                        # 🔧 新增：读取训练时的物理参数（如果值为None，保持None以便后续报错）
                        if 'gravity' in results['args']:
                            val = results['args']['gravity']
                            training_gravity = float(val) if val is not None else None
                        if 'damping' in results['args']:
                            val = results['args']['damping']
                            training_damping = float(val) if val is not None else None
                        if 'agent_max_speed' in results['args']:
                            val = results['args']['agent_max_speed']
                            training_agent_max_speed = float(val) if val is not None else None
                        if 'agent_accel' in results['args']:
                            val = results['args']['agent_accel']
                            training_agent_accel = float(val) if val is not None else None
                        if 'control_accel_gain' in results['args']:
                            val = results['args']['control_accel_gain']
                            training_control_accel_gain = float(val) if val is not None else None
                    else:
                        # 从顶层读取（向后兼容）
                        if 'use_fr_feature' in results:
                            training_use_fr = bool(results['use_fr_feature'])
                        if 'use_pf_feature' in results:
                            training_use_pf = bool(results['use_pf_feature'])
                        if 'pf_feature_dim' in results:
                            training_pf_feature_dim = int(results['pf_feature_dim'])
                        # 🔧 关键修复：从顶层读取动作范围参数（向后兼容）
                        if 'action_range_x' in results:
                            training_action_range_x = float(results['action_range_x'])
                        if 'action_range_y' in results:
                            training_action_range_y = float(results['action_range_y'])
                        if 'action_range_z' in results:
                            training_action_range_z = float(results['action_range_z'])
                        # 🚨 新增：从顶层读取网络结构配置（向后兼容）
                        if 'actor_hidden' in results:
                            training_actor_hidden = str(results['actor_hidden'])
                        if 'critic_hidden' in results:
                            training_critic_hidden = str(results['critic_hidden'])
                        # 🔧 新增：从顶层读取物理参数（向后兼容，如果值为None，保持None以便后续报错）
                        if 'gravity' in results:
                            val = results['gravity']
                            training_gravity = float(val) if val is not None else None
                        if 'damping' in results:
                            val = results['damping']
                            training_damping = float(val) if val is not None else None
                        if 'agent_max_speed' in results:
                            val = results['agent_max_speed']
                            training_agent_max_speed = float(val) if val is not None else None
                        if 'agent_accel' in results:
                            val = results['agent_accel']
                            training_agent_accel = float(val) if val is not None else None
                        if 'control_accel_gain' in results:
                            val = results['control_accel_gain']
                            training_control_accel_gain = float(val) if val is not None else None
                    
                    if training_use_fr is not None and training_use_pf is not None:
                        print(f"✅ 从训练配置读取特征标志: use_fr_feature={training_use_fr}, use_pf_feature={training_use_pf}")
                        if training_pf_feature_dim is not None:
                            print(f"✅ 从训练配置读取PF特征维度: {training_pf_feature_dim}")
                        # 🔧 关键修复：打印动作范围参数
                        if training_action_range_x is not None or training_action_range_y is not None or training_action_range_z is not None:
                            print(f"✅ 从训练配置读取动作范围: X={training_action_range_x}, Y={training_action_range_y}, Z={training_action_range_z}")
                        break
                except Exception as e:
                    continue
            
            if training_use_fr is not None and training_use_pf is not None:
                break
        
        # 使用训练时的配置（如果找到），否则使用当前配置
        use_fr_feature = training_use_fr if training_use_fr is not None else int(os.getenv('USE_FR_FEATURE', getattr(self.args, 'use_fr_feature', 1))) > 0
        use_pf_feature = training_use_pf if training_use_pf is not None else int(os.getenv('USE_PF_FEATURE', getattr(self.args, 'use_pf_feature', 1))) > 0
        
        # 🔧 关键修复：只使用最佳回合的FR值，不进行任何回退
        # 测试时应该使用对应训练留下的最佳回合的FR值，而不是使用默认值（即使能运行也毫无意义）
        if best_episode_force_ratio is not None:
            # 使用最佳回合的FR值
            eval_action_force_ratio = best_episode_force_ratio
            ep_str = f"回合 {best_episode_num}" if best_episode_num is not None else "最佳回合"
            print(f"✅ 使用最佳回合的ACTION_FORCE_RATIO: {eval_action_force_ratio} ({ep_str})")
        else:
            # 🚨 关键修复：如果找不到最佳回合的FR值，报错并退出
            # 因为测试应该使用对应训练留下的最佳回合的FR值，使用默认值即使能运行也毫无意义
            error_msg = (
                "❌ 错误：无法找到训练配置中的最佳回合FR值 (best_episode_force_ratio)\n"
                "   测试时必须使用对应训练留下的最佳回合的FR值，而不是使用默认值\n"
                "   请检查训练配置文件 (results.json) 是否包含 best_episode_force_ratio 字段\n"
                "   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError("无法找到最佳回合的FR值，评估无法继续。请确保训练配置文件包含 best_episode_force_ratio 字段。")
        
        # 更新args中的action_force_ratio，确保后续使用
        self.args.action_force_ratio = eval_action_force_ratio
        
        # 🔧 关键修复：必须使用训练时的动作范围参数实际值，不允许使用默认值
        # 如果找不到训练配置中的参数，报错并退出
        missing_action_ranges = []
        
        if training_action_range_x is None:
            missing_action_ranges.append("action_range_x")
        else:
            self.args.action_range_x = training_action_range_x
            print(f"✅ 使用训练时的ACTION_RANGE_X: {training_action_range_x}")
        
        if training_action_range_y is None:
            missing_action_ranges.append("action_range_y")
        else:
            self.args.action_range_y = training_action_range_y
            print(f"✅ 使用训练时的ACTION_RANGE_Y: {training_action_range_y}")
        
        if training_action_range_z is None:
            missing_action_ranges.append("action_range_z")
        else:
            self.args.action_range_z = training_action_range_z
            print(f"✅ 使用训练时的ACTION_RANGE_Z: {training_action_range_z}")
        
        # 🚨 如果缺少任何必需的动作范围参数，报错并退出
        if missing_action_ranges:
            error_msg = (
                f"❌ 错误：无法找到训练配置中的以下动作范围参数: {', '.join(missing_action_ranges)}\n"
                f"   测试时必须使用训练时实际使用的参数值，而不是使用默认值\n"
                f"   请检查训练配置文件 (results.json) 是否包含这些参数\n"
                f"   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError(f"无法找到训练配置中的动作范围参数: {', '.join(missing_action_ranges)}，评估无法继续。")
        
        # 🔧 关键修复：优先使用命令行参数，如果未指定则使用训练配置中的值
        # 这样支持消融实验使用不同的episode_length（如4000步）进行评估
        # 保存命令行参数的值（在覆盖之前）
        cmd_episode_length = getattr(self.args, 'episode_length', None)
        
        # 检查命令行参数是否明确指定（不是默认值2200）
        if cmd_episode_length is not None and cmd_episode_length != 2200:
            # 命令行参数明确指定了episode_length（不是默认值2200），使用命令行参数
            print(f"✅ 使用命令行指定的EPISODE_LENGTH: {cmd_episode_length} (覆盖训练配置中的 {training_episode_length if training_episode_length is not None else 'N/A'})")
            # 保持使用命令行参数，不覆盖
        elif training_episode_length is not None:
            # 使用训练配置中的episode_length
            self.args.episode_length = training_episode_length
            print(f"✅ 使用训练时的EPISODE_LENGTH: {training_episode_length}")
        else:
            # 既没有命令行参数，也没有训练配置，使用默认值
            error_msg = (
                "❌ 错误：无法找到训练配置中的episode_length参数\n"
                "   测试时必须使用训练时实际使用的episode_length值，而不是使用默认值\n"
                "   请检查训练配置文件 (results.json) 是否包含 episode_length 字段\n"
                "   或者通过 --episode-length 参数明确指定\n"
                "   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError("无法找到训练配置中的episode_length，评估无法继续。请确保训练配置文件包含 episode_length 字段，或通过 --episode-length 参数指定。")
        
        # 🔧 关键修复：必须使用训练时的物理参数实际值，不允许使用默认值
        # 如果找不到训练配置中的参数，报错并退出（因为使用默认值会导致评估不准确）
        missing_params = []
        
        if training_gravity is None:
            missing_params.append("gravity")
        else:
            self.args.gravity = training_gravity
            print(f"✅ 使用训练时的GRAVITY: {training_gravity}")
        
        if training_damping is None:
            missing_params.append("damping")
        else:
            self.args.damping = training_damping
            print(f"✅ 使用训练时的DAMPING: {training_damping}")
        
        if training_agent_max_speed is None:
            missing_params.append("agent_max_speed")
        else:
            self.args.agent_max_speed = training_agent_max_speed
            print(f"✅ 使用训练时的AGENT_MAX_SPEED: {training_agent_max_speed}")
        
        if training_agent_accel is None:
            missing_params.append("agent_accel")
        else:
            self.args.agent_accel = training_agent_accel
            print(f"✅ 使用训练时的AGENT_ACCEL: {training_agent_accel}")
        
        if training_control_accel_gain is None:
            missing_params.append("control_accel_gain")
        else:
            self.args.control_accel_gain = training_control_accel_gain
            print(f"✅ 使用训练时的CONTROL_ACCEL_GAIN: {training_control_accel_gain}")
        
        # 🚨 如果缺少任何必需的物理参数，报错并退出
        if missing_params:
            error_msg = (
                f"❌ 错误：无法找到训练配置中的以下物理参数: {', '.join(missing_params)}\n"
                f"   测试时必须使用训练时实际使用的参数值，而不是使用默认值\n"
                f"   请检查训练配置文件 (results.json) 是否包含这些参数\n"
                f"   如果训练配置不完整，评估结果将不准确，因此拒绝继续运行"
            )
            print(error_msg)
            raise ValueError(f"无法找到训练配置中的物理参数: {', '.join(missing_params)}，评估无法继续。")
        
        if training_use_fr is not None or training_use_pf is not None:
            print(f"✅ 使用特征标志: use_fr_feature={use_fr_feature}, use_pf_feature={use_pf_feature}")
        else:
            print(f"ℹ️  未找到训练配置，使用当前配置: use_fr_feature={use_fr_feature}, use_pf_feature={use_pf_feature}")
        
        # 🚨 关键修复：优先使用训练时的网络结构配置，确保与训练时完全一致
        # 如果找到训练配置，使用训练时的配置；否则使用命令行参数；最后才使用默认值
        actor_hidden = (
            training_actor_hidden if training_actor_hidden is not None else
            (getattr(self.args, 'actor_hidden', None) if getattr(self.args, 'actor_hidden', None) else
             '256,256,256')  # 默认值：与训练脚本一致
        )
        critic_hidden = (
            training_critic_hidden if training_critic_hidden is not None else
            (getattr(self.args, 'critic_hidden', None) if getattr(self.args, 'critic_hidden', None) else
             '256,256,256')  # 默认值：与训练脚本一致（3层×256）
        )
        
        # 打印使用的网络配置
        if training_actor_hidden is not None:
            print(f"✅ 使用训练时的Actor隐藏层配置: {actor_hidden}")
        else:
            print(f"ℹ️  未找到训练配置中的actor_hidden，使用: {actor_hidden}")
        if training_critic_hidden is not None:
            print(f"✅ 使用训练时的Critic隐藏层配置: {critic_hidden}")
        else:
            print(f"ℹ️  未找到训练配置中的critic_hidden，使用: {critic_hidden}")
        
        # 创建临时args用于MADDPG初始化
        maddpg_args = argparse.Namespace(
            learning_rate_actor=1e-4,
            learning_rate_critic=3e-4,
            gamma=0.95,
            tau=0.005,
            grad_clip_norm=10.0,
            huber_delta=1.0,
            noise_scale=0.0,  # 🔧 关键修复：评估时禁用噪声，确保使用纯策略
            noise_decay=1.0,  # 评估时不需要衰减
            noise_min=0.0,  # 评估时不需要最小噪声
            random_action_prob=0.0,  # 评估时禁用随机动作
            per_enabled=False,
            # 🚨 关键修复：使用从训练配置读取的网络结构
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            # 🔧 新增：FR和PF特征标志
            use_fr_feature=use_fr_feature,
            use_pf_feature=use_pf_feature,
            # 🔧 关键修复：使用训练时的pf_feature_dim（确保Critic输入维度一致）
            pf_feature_dim=training_pf_feature_dim if training_pf_feature_dim is not None else getattr(self.args, 'pf_feature_dim', 3),
            # 🔧 关键修复：使用训练时的action_force_ratio（如果是apf_learnable，FR=1.0）
            action_force_ratio=eval_action_force_ratio,
            use_tf_potential_field=getattr(self.args, 'use_tf_potential_field', True),
            goal_attraction=getattr(self.args, 'goal_attraction', 1.0),
            lambda_1_base=getattr(self.args, 'lambda_1_base', 5.0),
            terrain_repulsion=getattr(self.args, 'terrain_repulsion', 80.0),
            agent_influence_range=getattr(self.args, 'agent_influence_range', 10.0),
            delta_k_att=getattr(self.args, 'delta_k_att', 0.5),
            delta_lambda_1=getattr(self.args, 'delta_lambda_1', 2.5),
            delta_k_rep=getattr(self.args, 'delta_k_rep', 40.0),
            delta_radius=getattr(self.args, 'delta_radius', 5.0),
        )
        
        # 🔧 新增：创建环境对象以支持Oracle模式
        # 注意：评估时使用SingleEnvWrapper，需要获取实际的env对象
        eval_env = MultiAgentEnv(
            self.world,
            self.scenario.reset_world,
            self.scenario.reward,
            self.scenario.observation,
            info_callback=None,
            shared_viewer=False
        )
        # 将scenario引用挂到world（与训练路径保持一致）
        try:
            self.world.scenario = self.scenario
            eval_env.scenario = self.scenario
        except Exception:
            pass
        
        # 🔧 新增：设置terrain_sensing_mode（评估时使用，不影响Actor/Critic输入）
        terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
        maddpg_args.terrain_sensing_mode = terrain_sensing_mode
        print(f"[地形感知模式] {terrain_sensing_mode}")
        if terrain_sensing_mode.startswith('oracle'):
            print(f"  ⚠️  Oracle模式：仅用于APF地形力计算，Actor/Critic输入保持不变（使用观测）")
        
        # 初始化MADDPG或MATD3（根据算法选择）
        # 🔧 修复：移除env参数，因为训练脚本中已经移除了env参数支持
        # Oracle模式通过其他方式实现（在_calculate_terrain_forces_sphere_tf中直接访问scenario）
        algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
        if algorithm == 'matd3':
            self.maddpg = OptimizedMATD3(self.n_agents, self.obs_shapes, self.action_dims, maddpg_args)
        else:
            self.maddpg = OptimizedMADDPG(self.n_agents, self.obs_shapes, self.action_dims, maddpg_args)
        
        # 🔧 新增：为Oracle模式设置scenario引用（如果需要）
        # 注意：Oracle模式在评估时通过evaluate_optimized.py中的scenario对象访问
        # 训练脚本中的_calculate_terrain_forces_sphere_tf会通过其他方式获取scenario（如果需要）
        terrain_sensing_mode = getattr(self.args, 'terrain_sensing_mode', 'local')
        if terrain_sensing_mode.startswith('oracle'):
            # 将scenario引用存储到maddpg对象中，以便在APF计算时使用
            try:
                self.maddpg.scenario_ref = self.scenario
                self.maddpg.world_ref = self.world
                print(f"  ✅ Oracle模式：已设置scenario引用，scenario类型: {type(self.scenario).__name__}")
                # 验证scenario是否有get_terrain_height方法
                if hasattr(self.scenario, 'get_terrain_height'):
                    print(f"  ✅ Oracle模式：scenario具有get_terrain_height方法")
                else:
                    print(f"  ⚠️  Oracle模式：scenario缺少get_terrain_height方法，Oracle模式可能无法正常工作")
            except Exception as e:
                print(f"  ❌ Oracle模式：设置scenario引用失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 加载模型权重（带有效性检查与回退策略）
        def _is_valid_weights_dir(dir_path: str, n_agents: int) -> bool:
            """检查权重目录是否有效（支持MATD3 Twin Critic）"""
            try:
                if not os.path.isdir(dir_path):
                    return False
                algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
                for i in range(n_agents):
                    ap = os.path.join(dir_path, f"actor_{i}.weights.h5")
                    if not os.path.isfile(ap) or os.path.getsize(ap) <= 0:
                        return False
                    # 🚨 标准MATD3：检查两个独立的Critic网络文件
                    if algorithm == 'matd3':
                        cp1 = os.path.join(dir_path, f"critic1_{i}.weights.h5")
                        cp2 = os.path.join(dir_path, f"critic2_{i}.weights.h5")
                        # 如果新格式文件不存在，尝试旧格式（兼容性）
                        if not (os.path.isfile(cp1) and os.path.getsize(cp1) > 0) and \
                           not (os.path.isfile(cp2) and os.path.getsize(cp2) > 0):
                            # 回退到旧格式检查
                            cp_old = os.path.join(dir_path, f"critic_{i}.weights.h5")
                            if not os.path.isfile(cp_old) or os.path.getsize(cp_old) <= 0:
                                return False
                    else:
                        # MADDPG：单个Critic网络
                        cp = os.path.join(dir_path, f"critic_{i}.weights.h5")
                        if not os.path.isfile(cp) or os.path.getsize(cp) <= 0:
                            return False
                return True
            except Exception:
                return False

        def _find_fallback_dir(preferred_dir: str, n_agents: int) -> str:
            # 🔧 关键修复：支持中文路径，在传入路径下查找子目录
            # 首先尝试在传入路径下查找 final -> best -> 最新 ep*
            candidates = []
            for name in ("final", "best"):
                candidates.append(os.path.join(preferred_dir, name))
            try:
                if os.path.isdir(preferred_dir):
                    eps = [d for d in os.listdir(preferred_dir) if d.startswith("ep") and os.path.isdir(os.path.join(preferred_dir, d))]
                    # 按数字部分降序排序
                    def _ep_key(s):
                        import re
                        m = re.search(r"\d+", s)
                        return int(m.group(0)) if m else -1
                    eps_sorted = sorted(eps, key=_ep_key, reverse=True)
                    candidates.extend([os.path.join(preferred_dir, d) for d in eps_sorted])
            except Exception:
                pass
            
            # 🔧 关键修复：如果传入路径下找不到，再尝试在父目录下查找（兼容旧格式）
            if not any(_is_valid_weights_dir(c, n_agents) for c in candidates):
                parent = os.path.dirname(preferred_dir)
                for name in ("final", "best"):
                    candidates.append(os.path.join(parent, name))
                try:
                    if os.path.isdir(parent):
                        eps = [d for d in os.listdir(parent) if d.startswith("ep") and os.path.isdir(os.path.join(parent, d))]
                        def _ep_key(s):
                            import re
                            m = re.search(r"\d+", s)
                            return int(m.group(0)) if m else -1
                        eps_sorted = sorted(eps, key=_ep_key, reverse=True)
                        candidates.extend([os.path.join(parent, d) for d in eps_sorted])
                except Exception:
                    pass
            
            for c in candidates:
                if _is_valid_weights_dir(c, n_agents):
                    return c
            return None

        model_dir = self.args.load_model_path
        if not _is_valid_weights_dir(model_dir, self.n_agents):
            fb = _find_fallback_dir(model_dir, self.n_agents)
            if fb is not None:
                print(f"⚠️  检测到模型目录不完整，回退到: {fb}")
                model_dir = fb
            else:
                raise FileNotFoundError(f"找不到可用的权重文件，请检查目录: {self.args.load_model_path}")

        # 先以虚拟输入构建网络，确保变量已创建（为每个智能体的 actor/critic 都建图）
        try:
            critic_state_dim = sum(self.obs_shapes)
            use_fr = getattr(maddpg_args, 'use_fr_feature', False)
            use_pf = getattr(maddpg_args, 'use_pf_feature', False)
            pf_feature_dim = getattr(maddpg_args, 'pf_feature_dim', 3)
            
            for i in range(self.n_agents):
                # 🔧 构建 actor：根据训练配置构建正确的输入结构
                # 如果 use_pf_feature=True，Actor期望3个输入：[obs, fr_input, pf_input]
                # 如果 use_fr_feature=True 但 use_pf_feature=False，Actor期望2个输入：[obs, fr_input]
                # 如果两者都False，Actor期望1个输入：[obs]
                dummy_obs = tf.zeros((1, self.obs_shapes[i]), dtype=tf.float32)
                actor_inputs = [dummy_obs]
                if use_fr:
                    actor_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
                if use_pf:
                    # 🔧 关键修复：如果训练时启用了PF特征，Actor需要PF特征作为单独输入
                    actor_inputs.append(tf.zeros((1, pf_feature_dim), dtype=tf.float32))
                
                if len(actor_inputs) == 1:
                    _ = self.maddpg.agents[i]['actor'](actor_inputs[0], training=False)
                else:
                    _ = self.maddpg.agents[i]['actor'](actor_inputs, training=False)
                
                # 🚨 标准MATD3：构建两个独立的Critic网络（Twin Critic）
                # Critic需要PF特征作为单独输入（与Actor不同）
                dummy_state = tf.zeros((1, critic_state_dim), dtype=tf.float32)
                dummy_actions = tf.zeros((1, self.action_dims[i] * self.n_agents), dtype=tf.float32)
                critic_inputs = [dummy_state, dummy_actions]
                if use_fr:
                    critic_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
                if use_pf:
                    critic_inputs.append(tf.zeros((1, pf_feature_dim * self.n_agents), dtype=tf.float32))
                
                if algorithm == 'matd3':
                    # 🚨 标准MATD3：构建critic1和critic2
                    critic1_output = self.maddpg.agents[i]['critic1'](critic_inputs, training=False)
                    critic2_output = self.maddpg.agents[i]['critic2'](critic_inputs, training=False)
                    # 每个critic输出两个Q值（用于梯度分离）
                    assert isinstance(critic1_output, (list, tuple)) and len(critic1_output) == 2, \
                        f"MATD3 Critic1应该输出两个Q值，实际输出: {type(critic1_output)}"
                    assert isinstance(critic2_output, (list, tuple)) and len(critic2_output) == 2, \
                        f"MATD3 Critic2应该输出两个Q值，实际输出: {type(critic2_output)}"
                else:
                    # MADDPG：单个Critic网络
                    critic_output = self.maddpg.agents[i]['critic'](critic_inputs, training=False)
        except Exception as e:
            print(f"⚠️ 网络构建警告: {e}")
            pass

        # 🔧 关键修复：手动加载权重，支持新格式权重文件
        def _manual_load_weights(model, weight_file: str):
            """手动从 HDF5 文件加载权重，支持新格式（layers/*/vars/*）"""
            try:
                import h5py
                with h5py.File(weight_file, 'r') as f:
                    # 检查是否是新格式（有 layers 组）
                    if 'layers' in f:
                        # 新格式：layers/dense/vars/0, layers/dense/vars/1
                        layer_weights = {}
                        def collect_weights(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                # 提取层名（例如：layers/dense/vars/0 -> dense）
                                parts = name.split('/')
                                if len(parts) >= 3 and parts[0] == 'layers' and parts[2] == 'vars':
                                    layer_name = parts[1]
                                    var_idx = int(parts[3]) if len(parts) > 3 else 0
                                    if layer_name not in layer_weights:
                                        layer_weights[layer_name] = {}
                                    layer_weights[layer_name][var_idx] = obj[:]
                        f.visititems(collect_weights)
                        
                        # 将权重设置到模型中
                        loaded_count = 0
                        for layer in model.layers:
                            if layer.name in layer_weights:
                                weights_data = layer_weights[layer.name]
                                # 按索引排序（0=kernel, 1=bias 或 gamma, beta）
                                sorted_vars = [weights_data[i] for i in sorted(weights_data.keys())]
                                try:
                                    layer.set_weights(sorted_vars)
                                    loaded_count += 1
                                except Exception as e:
                                    # 如果形状不匹配，跳过
                                    pass
                        return loaded_count > 0
                    else:
                        # 旧格式，使用标准加载
                        model.load_weights(weight_file)
                        return True
            except Exception as e:
                return False
        
        # 安全加载：先常规加载，失败则尝试手动加载，最后 skip_mismatch 兜底
        def _safe_load(agent, path: str, kind: str):
            try:
                agent.load_weights(path)
                return True
            except Exception as e:
                # 🔧 关键修复：尝试手动加载（支持新格式权重文件）
                try:
                    if _manual_load_weights(agent, path):
                        print(f"✅ {kind} 使用手动加载方式成功加载: {os.path.basename(path)}")
                        return True
                except Exception as e_manual:
                    pass
                try:
                    # 兼容可能的细微层名差异，只使用skip_mismatch
                    agent.load_weights(path, skip_mismatch=True)
                    print(f"⚠️  {kind} 使用skip_mismatch方式加载: {os.path.basename(path)} | {e}")
                    return True
                except Exception as e2:
                    print(f"❌ 加载{kind}失败: {path} | {e2}")
                    return False

        print(f"正在从 {model_dir} 加载模型...")
        ok = True
        total_loaded_vars = 0
        total_vars = 0
        
        for i in range(self.n_agents):
            a_path = os.path.join(model_dir, f"actor_{i}.weights.h5")
            # 🚨 标准MATD3：对于MATD3，c_path仅用于MADDPG兼容性检查
            algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
            c_path = os.path.join(model_dir, f"critic_{i}.weights.h5") if algorithm != 'matd3' else None
            # 加载前后变量快照用于统计覆盖比例
            def _snapshot_vars(model):
                return [v.numpy().copy() for v in model.trainable_variables]
            def _count_changed(before, after):
                import numpy as _np
                changed = 0
                total = min(len(before), len(after))
                for bi, ai in zip(before, after):
                    if bi.shape != ai.shape:
                        continue
                    if not _np.array_equal(bi, ai):
                        changed += 1
                return changed, total

            # actor
            a_before = _snapshot_vars(self.maddpg.agents[i]['actor'])
            ok = _safe_load(self.maddpg.agents[i]['actor'], a_path, f"actor[{i}]") and ok
            a_after = _snapshot_vars(self.maddpg.agents[i]['actor'])
            chg, tot = _count_changed(a_before, a_after)
            total_loaded_vars += chg
            total_vars += tot
            if tot > 0:
                ratio = (chg / tot) * 100.0
                print(f"actor[{i}] 覆盖变量: {chg}/{tot} ({ratio:.1f}%)")
                if ratio < 60.0:
                    print(f"⚠️  actor[{i}] 覆盖比例偏低，可能与训练结构不一致")

            # 🚨 标准MATD3：加载两个独立的Critic网络
            algorithm = getattr(self.args, 'algorithm', 'matd3').lower()
            if algorithm == 'matd3':
                # 加载critic1
                c1_path = os.path.join(model_dir, f"critic1_{i}.weights.h5")
                # 如果新格式文件不存在，尝试旧格式（兼容性）
                if not os.path.exists(c1_path):
                    c1_path = os.path.join(model_dir, f"critic_{i}.weights.h5")
                    if os.path.exists(c1_path):
                        print(f"⚠️  检测到旧格式critic文件，将同时加载到critic1和critic2...")
                
                c1_before = _snapshot_vars(self.maddpg.agents[i]['critic1'])
                ok = _safe_load(self.maddpg.agents[i]['critic1'], c1_path, f"critic1[{i}]") and ok
                c1_after = _snapshot_vars(self.maddpg.agents[i]['critic1'])
                chg1, tot1 = _count_changed(c1_before, c1_after)
                total_loaded_vars += chg1
                total_vars += tot1
                if tot1 > 0:
                    ratio1 = (chg1 / tot1) * 100.0
                    print(f"critic1[{i}] 覆盖变量: {chg1}/{tot1} ({ratio1:.1f}%)")
                    if ratio1 < 60.0:
                        print(f"⚠️  critic1[{i}] 覆盖比例偏低，可能与训练结构不一致")
                
                # 加载critic2（如果旧格式，使用相同文件；否则使用critic2文件）
                c2_path = os.path.join(model_dir, f"critic2_{i}.weights.h5")
                if not os.path.exists(c2_path):
                    c2_path = c1_path  # 使用旧格式文件
                
                c2_before = _snapshot_vars(self.maddpg.agents[i]['critic2'])
                ok = _safe_load(self.maddpg.agents[i]['critic2'], c2_path, f"critic2[{i}]") and ok
                c2_after = _snapshot_vars(self.maddpg.agents[i]['critic2'])
                chg2, tot2 = _count_changed(c2_before, c2_after)
                total_loaded_vars += chg2
                total_vars += tot2
                if tot2 > 0:
                    ratio2 = (chg2 / tot2) * 100.0
                    print(f"critic2[{i}] 覆盖变量: {chg2}/{tot2} ({ratio2:.1f}%)")
                    if ratio2 < 60.0:
                        print(f"⚠️  critic2[{i}] 覆盖比例偏低，可能与训练结构不一致")
                
                # 同步到目标网络
                try:
                    self.maddpg.agents[i]['target_actor'].set_weights(self.maddpg.agents[i]['actor'].get_weights())
                    self.maddpg.agents[i]['target_critic1'].set_weights(self.maddpg.agents[i]['critic1'].get_weights())
                    self.maddpg.agents[i]['target_critic2'].set_weights(self.maddpg.agents[i]['critic2'].get_weights())
                except Exception:
                    pass
            else:
                # MADDPG：单个Critic网络
                c_before = _snapshot_vars(self.maddpg.agents[i]['critic'])
                ok = _safe_load(self.maddpg.agents[i]['critic'], c_path, f"critic[{i}]") and ok
                c_after = _snapshot_vars(self.maddpg.agents[i]['critic'])
                chg, tot = _count_changed(c_before, c_after)
                total_loaded_vars += chg
                total_vars += tot
                if tot > 0:
                    ratio = (chg / tot) * 100.0
                    print(f"critic[{i}] 覆盖变量: {chg}/{tot} ({ratio:.1f}%)")
                    if ratio < 60.0:
                        print(f"⚠️  critic[{i}] 覆盖比例偏低，可能与训练结构不一致")
                # 同步到目标网络
                try:
                    self.maddpg.agents[i]['target_actor'].set_weights(self.maddpg.agents[i]['actor'].get_weights())
                    self.maddpg.agents[i]['target_critic'].set_weights(self.maddpg.agents[i]['critic'].get_weights())
                except Exception:
                    pass

        # 总体加载统计
        if total_vars > 0:
            overall_ratio = (total_loaded_vars / total_vars) * 100.0
            print(f"\n📊 总体模型加载统计:")
            print(f"   - 总变量数: {total_vars}")
            print(f"   - 成功加载: {total_loaded_vars}")
            print(f"   - 加载比例: {overall_ratio:.1f}%")
            
            if overall_ratio < 50.0:
                print(f"❌ 警告: 模型加载比例过低 ({overall_ratio:.1f}%)，可能使用的是随机权重!")
                print(f"   建议重新训练模型或检查模型文件完整性")
            elif overall_ratio < 80.0:
                print(f"⚠️  注意: 模型加载比例较低 ({overall_ratio:.1f}%)，部分权重可能未正确加载")
            else:
                print(f"✅ 模型加载比例良好 ({overall_ratio:.1f}%)")

        if not ok:
            raise RuntimeError("无法成功加载全部模型权重，请检查权重文件是否完整匹配。")

        print("✅ 模型加载完成!")
        
    def evaluate_single_episode(self, episode_idx):
        """评估单个回合，仿照1.0版本的逻辑"""
        print(f"\n🚀 开始评估回合 {episode_idx + 1}")
        
        # 🔧 关键修复：在reset前记录固定位置信息，确保所有评估模式使用相同的起点
        if hasattr(self.scenario, 'use_fixed_positions') and self.scenario.use_fixed_positions:
            if hasattr(self.scenario, 'fixed_positions') and self.scenario.fixed_positions:
                agents_pos = self.scenario.fixed_positions.get('agents', [])
                goal_pos = self.scenario.fixed_positions.get('goal', None)
                if episode_idx == 0:  # 只在第一个回合打印
                    print(f"🔧 固定位置验证: 使用固定位置，{len(agents_pos)}个智能体")
                    if len(agents_pos) > 0:
                        print(f"   智能体0起点: [{agents_pos[0][0]:.2f}, {agents_pos[0][1]:.2f}, {agents_pos[0][2]:.2f}]")
                    if goal_pos:
                        print(f"   目标位置: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
            else:
                if episode_idx == 0:
                    print(f"⚠️  警告: use_fixed_positions=True但fixed_positions为None，可能未正确加载位置文件")
        
        # 环境重置
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs_n, _ = reset_result
        else:
            obs_n = reset_result
        
        # 🔧 关键修复：在reset后验证实际起点位置，确保所有评估模式使用相同的起点
        if episode_idx == 0:  # 只在第一个回合打印
            actual_start_positions = []
            for i, agent in enumerate(self.world.agents):
                actual_start_positions.append(agent.state.p_pos.copy())
                if i < 3:  # 只打印前3个智能体
                    print(f"   实际智能体{i}起点: [{agent.state.p_pos[0]:.2f}, {agent.state.p_pos[1]:.2f}, {agent.state.p_pos[2]:.2f}]")
            
        episode_reward = 0
        try:
            light_mode = os.getenv("EVAL_LIGHT_MODE", "0").lower() in ("1", "true", "yes", "on")
        except Exception:
            light_mode = False
        # 🔧 修复：默认记录轨迹和动作历史（用于生成交互图）
        # 评估时默认需要记录轨迹，因为交互图生成需要轨迹数据
        # 可以通过EVAL_LIGHT_MODE=1禁用（如果不需要可视化）
        record_trajectory = not light_mode  # 默认记录轨迹（用于可视化）
        record_actions = not light_mode  # 默认记录动作历史（用于可视化）
        episode_trajectory = []
        episode_actions_history = []  # 🔧 新增：记录动作历史（用于生成动作时序图）
        step_count = 0
        
        # 处理观察数据
        processed_obs = self.maddpg.obs_processor.batch_process_observations(obs_n)
        
        # 🔧 关键修复：如果启用PF特征，计算真实的PF力而不是使用零向量占位符
        # 🔧 修复：使用训练配置的特征标志，确保与训练时一致
        use_pf = getattr(self.maddpg.args, 'use_pf_feature', False) if hasattr(self, 'maddpg') and hasattr(self.maddpg, 'args') else getattr(self.args, 'use_pf_feature', False)
        use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
        action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
        
        if use_pf and use_tf_potential_field and action_force_ratio > 0.0 and len(processed_obs.shape) == 2:
            # 🔧 关键修复：计算真实的PF力（与训练时一致）- 批量化以提升速度
            n_agents_eval = processed_obs.shape[0]
            dummy_actions_tf = tf.zeros((n_agents_eval, 7), dtype=tf.float32)  # (n_agents, action_dim)
            base_obs_tf = tf.convert_to_tensor(processed_obs, dtype=tf.float32)  # (n_agents, obs_dim)
            _, pf_forces_tf = self.maddpg._apply_potential_field_correction(
                dummy_actions_tf, base_obs_tf, action_force_ratio
            )
            pf_forces = pf_forces_tf.numpy()
            # 将PF力追加到观测中
            processed_obs = np.concatenate([processed_obs, pf_forces], axis=1)
        elif use_pf and len(processed_obs.shape) == 2:
            # 如果启用PF特征但不满足计算条件，使用零向量占位符
            n_agents_eval = processed_obs.shape[0]
            pf_placeholder = np.zeros((n_agents_eval, 3), dtype=np.float32)
            processed_obs = np.concatenate([processed_obs, pf_placeholder], axis=1)
        
        # 记录开始时间
        start_time = time.time()
        
        # 🚨 关键修复：确保episode_length正确设置
        episode_length = getattr(self.args, 'episode_length', 2200)
        if episode_length <= 0:
            episode_length = 2200
            print(f"⚠️  警告: episode_length无效，使用默认值: {episode_length}")
        
        # 🔧 新增：打印关键评估参数，便于调试
        action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
        use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
        use_fr_feature = getattr(self.args, 'use_fr_feature', False)
        use_pf_feature = getattr(self.args, 'use_pf_feature', False)
        action_range_x = getattr(self.args, 'action_range_x', None)
        action_range_y = getattr(self.args, 'action_range_y', None)
        action_range_z = getattr(self.args, 'action_range_z', None)
        
        print(f"📊 评估配置:")
        print(f"   - episode_length={episode_length}")
        print(f"   - disable_early_termination={getattr(self.args, 'disable_early_termination', False)}")
        print(f"   - ACTION_FORCE_RATIO={action_force_ratio} {'⚠️ 为0，势场修正将不生效！' if action_force_ratio == 0.0 else ''}")
        print(f"   - USE_TF_POTENTIAL_FIELD={use_tf_potential_field}")
        print(f"   - use_fr_feature={use_fr_feature}")
        print(f"   - use_pf_feature={use_pf_feature}")
        print(f"   - action_range: X={action_range_x}, Y={action_range_y}, Z={action_range_z}")
        if use_tf_potential_field and action_force_ratio > 0.0:
            print(f"   ✅ 势场修正将生效 (FR={action_force_ratio})")
        else:
            print(f"   ⚠️  势场修正将不生效 (use_tf_potential_field={use_tf_potential_field}, FR={action_force_ratio})")
        
        # 🔧 新增：添加进度条显示评估进度
        # 🔧 修复：确保进度条正确显示，设置合适的更新频率
        try:
            tqdm_to_stdout = os.getenv("TQDM_TO_STDOUT", "1").lower() in ("1", "true", "yes", "on")
            tqdm_file = sys.stdout if tqdm_to_stdout else sys.stderr
        except Exception:
            tqdm_file = sys.stdout
        try:
            tqdm_disable = os.getenv("TQDM_DISABLE", "0").lower() in ("1", "true", "yes", "on")
        except Exception:
            tqdm_disable = False
        try:
            debug_action_steps = int(os.getenv("EVAL_DEBUG_ACTION_STEPS", "3"))
        except Exception:
            debug_action_steps = 3
        # 🚨 关键修复：设置position参数，避免多个进度条重叠显示
        # 问题：多个回合的进度条同时显示时，如果没有设置position，会导致显示混乱
        # 解决方案：每个回合使用position=0（单行显示），或者使用leave=False（完成后清除）
        # 注意：如果使用position，需要确保所有进度条使用相同的position，否则会显示多行
        # 这里使用leave=False，让进度条完成后自动清除，避免累积
        pbar = tqdm(range(int(episode_length)), desc=f"回合 {episode_idx + 1}", unit="步",
                   ncols=120, leave=False, mininterval=0.1, miniters=1,
                   file=tqdm_file, dynamic_ncols=False, ascii=False, disable=tqdm_disable,
                   position=0)  # 🔧 修复：设置position=0，确保所有进度条在同一行显示
        
        for step in pbar:
            # 🔧 关键修复：如果启用PF特征，计算真实的PF力而不是使用零向量占位符
            # 与训练时保持一致：先计算PF力，然后作为特征传递给Actor
            use_pf = getattr(self.args, 'use_pf_feature', False)
            use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
            action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
            
            # 🔧 性能优化：大幅减少进度条更新频率（每200步更新一次，与训练脚本一致）
            # 训练脚本通常不使用步级进度条，评估时也减少更新频率以提升性能
            if step % 200 == 0 or step < 5:
                pbar.set_postfix({
                    '奖励': f'{episode_reward:.1f}',
                    '步数': f'{step + 1}/{episode_length}'
                })
                # 🔧 性能优化：移除pbar.refresh()，让tqdm自动管理刷新频率
            
            if use_pf and use_tf_potential_field and action_force_ratio > 0.0:
                # 🔧 关键修复：计算真实的PF力（与训练时一致）
                # 使用dummy_actions来计算PF力，就像训练时那样
                if len(processed_obs.shape) == 2:  # (n_agents, obs_dim)
                    n_agents_eval = processed_obs.shape[0]
                    # 获取基础观测维度（去除PF特征后的维度）
                    base_obs_dim = processed_obs.shape[1]
                    if base_obs_dim > 78:  # 如果已经包含PF特征，先去除
                        base_obs_dim = 78
                    
                    # 提取基础观测（去除可能存在的PF特征）
                    base_obs = processed_obs[:, :base_obs_dim] if processed_obs.shape[1] > base_obs_dim else processed_obs
                    
                    # 🔧 性能优化：批量计算PF力，直接使用批量势场修正
                    base_obs_tf = tf.convert_to_tensor(base_obs, dtype=tf.float32)  # (n_agents, obs_dim)
                    dummy_actions_tf = tf.zeros((n_agents_eval, 7), dtype=tf.float32)  # (n_agents, action_dim)
                    _, pf_forces_tf = self.maddpg._apply_potential_field_correction(
                        dummy_actions_tf, base_obs_tf, action_force_ratio
                    )
                    pf_forces = pf_forces_tf.numpy()  # 一次性转换
                    
                    # 将PF力追加到观测中
                    processed_obs = np.concatenate([base_obs, pf_forces], axis=1)
                else:
                    # 如果形状不对，使用零向量作为后备
                    n_agents_eval = processed_obs.shape[0] if len(processed_obs.shape) >= 2 else 1
                    pf_placeholder = np.zeros((n_agents_eval, 3), dtype=np.float32)
                    processed_obs = np.concatenate([processed_obs, pf_placeholder], axis=1)
            elif use_pf:
                # 如果启用PF特征但不满足计算条件，使用零向量占位符
                if len(processed_obs.shape) == 2:  # (n_agents, obs_dim)
                    n_agents_eval = processed_obs.shape[0]
                    pf_placeholder = np.zeros((n_agents_eval, 3), dtype=np.float32)
                    processed_obs = np.concatenate([processed_obs, pf_placeholder], axis=1)
            
            # 选择动作（评估时不加噪声）
            # 🔧 关键修复：使用训练配置的特征标志，而不是命令行参数
            # 确保与训练时的网络输入结构完全一致
            use_fr = getattr(self.maddpg.args, 'use_fr_feature', False)
            use_pf = getattr(self.maddpg.args, 'use_pf_feature', False)
            raw_actions_tf = self.select_actions_eval(processed_obs, use_fr=use_fr, use_pf=use_pf)
            # 🔧 性能优化：延迟numpy转换，尽量在tensor空间内操作
            raw_actions = raw_actions_tf.numpy() if isinstance(raw_actions_tf, tf.Tensor) else raw_actions_tf
            
            # 🔧 新增：记录Actor原始输出（用于生成动作时序图）
            # 🔧 性能优化：只在需要时记录，避免不必要的copy操作
            if record_actions:
                # 🔧 性能优化：使用列表推导式，避免循环中的多次copy
                episode_actions_history.append([action.copy() for action in raw_actions])
            
            # 🔧 关键修复：应用势场修正（与训练时一致）
            # 势场修正生效条件：USE_TF_POTENTIAL_FIELD=1 AND ACTION_FORCE_RATIO > 0.0
            use_tf_potential_field = getattr(self.args, 'use_tf_potential_field', True)
            action_force_ratio = getattr(self.args, 'action_force_ratio', 0.0)
            
            if use_tf_potential_field and action_force_ratio > 0.0:
                # 🔧 性能优化：批量应用势场修正，尽量在tensor空间内操作
                # 🔧 关键优化：如果raw_actions已经是tensor，避免重复转换
                if isinstance(raw_actions_tf, tf.Tensor):
                    raw_actions_tf_for_correction = raw_actions_tf
                else:
                    raw_actions_tf_for_correction = tf.convert_to_tensor(raw_actions, dtype=tf.float32)  # (n_agents, action_dim)
                
                # 🔧 性能优化：如果processed_obs已经包含PF特征，直接使用；否则需要提取基础观测
                # 注意：processed_obs此时是numpy数组（从obs_processor返回），需要转换为tensor
                base_obs_dim = processed_obs.shape[1]
                if base_obs_dim > 78:  # 如果已经包含PF特征，去除PF特征部分
                    base_obs_dim = 78
                base_obs_for_correction = processed_obs[:, :base_obs_dim] if processed_obs.shape[1] > base_obs_dim else processed_obs
                processed_obs_tf = tf.convert_to_tensor(base_obs_for_correction, dtype=tf.float32)  # (n_agents, obs_dim)
                    
                # 🔧 性能优化：使用训练代码的@tf.function装饰的势场修正函数
                corrected_head_tf, _ = self.maddpg._apply_potential_field_correction(
                    raw_actions_tf_for_correction, processed_obs_tf, action_force_ratio
                    )
                    
                # 获取action_dim（从tensor形状推断）
                action_dim = tf.shape(raw_actions_tf_for_correction)[1]
                if action_dim > 3:
                    corrected_actions_tf = tf.concat([corrected_head_tf, raw_actions_tf_for_correction[:, 3:]], axis=1)
                else:
                    corrected_actions_tf = corrected_head_tf
                
                # 🔧 性能优化：延迟numpy转换，只在env.step需要时转换
                actions = corrected_actions_tf.numpy()  # env.step需要numpy数组
            else:
                # 不使用势场修正，直接使用原始动作
                # 🔧 性能优化：如果raw_actions_tf是tensor，转换为numpy；否则直接使用
                if isinstance(raw_actions_tf, tf.Tensor):
                    actions = raw_actions_tf.numpy()
                else:
                    actions = raw_actions
            
            # 🔧 性能优化：记录轨迹（只在需要可视化时记录）
            # 注意：环境中的agent._trajectory记录无法禁用（在environment.py的step方法中），但这里的额外记录可以禁用
            if record_trajectory:
                try:
                    # 🔧 性能优化：使用列表推导式，一次性完成所有操作
                    positions = [agent.state.p_pos.copy() if hasattr(agent.state, 'p_pos') else [0, 0, 0] 
                                for agent in self.env.agents]
                    episode_trajectory.append(positions)
                except Exception as e:
                    # 🔧 性能优化：只在非quiet模式下输出警告
                    if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                        tqdm.write(f"轨迹记录警告: {e}", file=tqdm_file)
                
            # 🔧 新增：在前几步打印动作值，便于调试（使用tqdm.write避免干扰进度条）
            if step < debug_action_steps:
                tqdm.write(f"   Step {step}: 动作值 (前3个智能体):", file=tqdm_file)
                for i in range(min(3, len(actions))):
                    tqdm.write(
                        f"      Agent {i}: {actions[i][:3]} (原始动作: {raw_actions[i][:3] if 'raw_actions' in locals() else 'N/A'})",
                        file=tqdm_file
                    )
                
            # 执行动作
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                next_obs_n, rew_n, done_n, info_n = step_result
            elif len(step_result) == 5:
                next_obs_n, rew_n, terminated, truncated, info_n = step_result
                done_n = [t or tr for t, tr in zip(terminated, truncated)]
            else:
                raise ValueError(f"意外的环境step返回值: {len(step_result)}")

            # 累计奖励
            episode_reward += sum(rew_n)
            step_count += 1
            
            # 更新观察
            processed_obs = self.maddpg.obs_processor.batch_process_observations(next_obs_n)
            # 注意：势场力追加会在下一轮循环开始时进行
            
            # 检查结束条件（支持禁用提前终止）
            if all(done_n) and (not getattr(self.args, 'disable_early_termination', False)):
                # 🔧 性能优化：只在进度条启用时更新
                if not tqdm_disable:
                    pbar.set_postfix({
                        '奖励': f'{episode_reward:.1f}',
                        '状态': '提前结束',
                        '步数': f'{step + 1}/{episode_length}'
                    })
                    pbar.close()
                if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                    tqdm.write(f"📍 回合在第 {step + 1}/{episode_length} 步自然结束（所有智能体done）", file=tqdm_file)
                break
                
        # 🔧 修复：在循环结束后更新最终状态
        # 🔧 性能优化：只在进度条启用时更新
        if not tqdm_disable:
            pbar.set_postfix({
                '奖励': f'{episode_reward:.1f}',
                '步数': f'{step_count}/{episode_length}',
                '状态': '完成'
            })
            pbar.close()
        
        # 计算回合统计
        episode_duration = time.time() - start_time
        avg_step_time = episode_duration / step_count if step_count > 0 else 0
        
        # 🔧 新增：收集碰撞次数和最小净空距离（与训练脚本一致）
        # 🚨 关键修复：确保使用与训练时相同的统计方式
        # 训练时使用 agent.current_episode_collision_count 和 agent.debug_info['total_penetration_count']
        # 两者应该同步更新，但优先使用 debug_info['total_penetration_count']（与训练脚本一致）
        episode_collision_counts = []
        episode_min_distances = []
        try:
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                for agent in self.env.world.agents:
                    # 🚨 关键修复：确保debug_info已初始化
                    if not hasattr(agent, 'debug_info'):
                        agent.debug_info = {}
                    if not isinstance(agent.debug_info, dict):
                        agent.debug_info = {}
                    
                    # 🚨 关键修复：优先从debug_info读取，如果没有则从current_episode_collision_count读取
                    # 确保与训练脚本的统计方式完全一致
                    penetration_count = 0
                    if 'total_penetration_count' in agent.debug_info:
                        penetration_count = agent.debug_info.get('total_penetration_count', 0)
                    elif hasattr(agent, 'current_episode_collision_count'):
                        # 回退：如果debug_info中没有，使用current_episode_collision_count
                        penetration_count = agent.current_episode_collision_count
                    
                    try:
                        penetration_count = int(penetration_count) if np.isfinite(penetration_count) else 0
                    except (ValueError, TypeError, OverflowError):
                        penetration_count = 0
                    
                    episode_collision_counts.append(penetration_count)
                    
                    # 收集min_distance_to_obstacle
                    min_dist = None
                    if hasattr(agent, 'debug_info') and isinstance(agent.debug_info, dict):
                        min_dist = agent.debug_info.get('d_min_current', None)
                        if min_dist is not None:
                            try:
                                if isinstance(min_dist, np.ndarray):
                                    if min_dist.size > 0:
                                        min_dist = float(min_dist[-1] if min_dist.ndim > 0 else min_dist.item())
                                    else:
                                        min_dist = None
                                else:
                                    min_dist = float(min_dist)
                            except (ValueError, TypeError, AttributeError):
                                min_dist = None
                    
                    # 如果debug_info中没有，尝试从last_min_distance获取
                    if min_dist is None and hasattr(agent, 'last_min_distance') and agent.last_min_distance is not None:
                        try:
                            last_min_dist = agent.last_min_distance
                            if isinstance(last_min_dist, np.ndarray):
                                if last_min_dist.size > 0:
                                    min_dist = float(last_min_dist[-1] if last_min_dist.ndim > 0 else last_min_dist.item())
                                else:
                                    min_dist = None
                            else:
                                min_dist = float(last_min_dist)
                        except (ValueError, TypeError, AttributeError):
                            min_dist = None
                    
                    if min_dist is not None and np.isfinite(min_dist):
                        episode_min_distances.append(min_dist)
        except Exception as e:
            if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                print(f"⚠️  收集碰撞统计信息时出错: {e}")
        
        # 🔧 新增：计算成功标志（与训练脚本一致）
        # 定义：Reach_i := ∃t ≤ T_max, ||p_t^i - g^i|| ≤ r_goal
        #       Safe_i := ∀t ≤ T_max, no-collision for agent i
        #       Succ_i := Reach_i ∧ Safe_i
        #       Succ_team := Λ_{i=1}^{M} Succ_i
        agent_success_flags = []
        team_success_flag = 1
        try:
            thr_success = float(getattr(self.args, 'success_distance_threshold', 4.0))
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                scn = getattr(self.env, 'scenario', None)
                goal_pos = None
                if scn is not None:
                    goal_pos = getattr(scn, 'goal_pos', None)
                    if goal_pos is None and hasattr(scn, 'get_goal_pos'):
                        try:
                            goal_pos = scn.get_goal_pos()
                        except Exception:
                            goal_pos = None
                
                for agent_idx, agent in enumerate(self.env.world.agents):
                    # 检查是否到达目标（Reach_i）
                    pos = getattr(getattr(agent, 'state', None), 'p_pos', None)
                    if pos is None or len(pos) < 3:
                        agent_success_flags.append(0)
                        team_success_flag = 0
                        continue
                    
                    # 获取智能体的目标位置
                    ag_goal = None
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state') and getattr(agent.goal_a.state, 'p_pos', None) is not None:
                        ag_goal = agent.goal_a.state.p_pos
                    if ag_goal is None:
                        ag_goal = goal_pos
                    
                    if ag_goal is None:
                        agent_success_flags.append(0)
                        team_success_flag = 0
                        continue
                    
                    # 计算到目标的距离
                    dx = pos[0] - ag_goal[0]
                    dy = pos[1] - ag_goal[1]
                    dz = pos[2] - ag_goal[2]
                    dist_goal_3d = (dx*dx + dy*dy + dz*dz) ** 0.5
                    reach_i = (dist_goal_3d <= (1.2 * thr_success))
                    
                    # 检查是否无碰撞（Safe_i）
                    safe_i = (episode_collision_counts[agent_idx] == 0) if agent_idx < len(episode_collision_counts) else False
                    
                    # 单智能体成功 = 到达目标 AND 无碰撞
                    succ_i = 1 if (reach_i and safe_i) else 0
                    agent_success_flags.append(succ_i)
                    
                    if succ_i == 0:
                        team_success_flag = 0
        except Exception as e:
            if not os.getenv("QUIET_OUTPUT", "1").lower() in ("1", "true", "yes", "on"):
                print(f"⚠️  计算成功标志时出错: {e}")
            agent_success_flags = [0] * len(episode_collision_counts) if episode_collision_counts else []
            team_success_flag = 0
        
        # 计算汇总统计
        total_collisions = sum(episode_collision_counts) if episode_collision_counts else 0
        try:
            total_collisions = int(total_collisions) if np.isfinite(total_collisions) else 0
        except (ValueError, TypeError, OverflowError):
            total_collisions = 0
        
        # 计算最小净空距离的统计（与训练脚本格式一致）
        min_distance_stat = None
        if episode_min_distances:
            try:
                min_distance_stat = {
                    'mean': float(np.mean(episode_min_distances)),
                    'min': float(np.min(episode_min_distances))
                }
            except Exception:
                min_distance_stat = None
        
        episode_success = (team_success_flag == 1)
        success_flag = 1 if episode_success else 0
        
        # 🔧 新增：计算到达时间/步数（如果成功）
        arrival_step = step_count if success_flag == 1 else None
        arrival_time = episode_duration if success_flag == 1 else None
        
        # 🔧 新增：计算穿透深度统计（从min_distance中的负值提取）
        penetration_depths = []
        penetration_count_episodes = 0
        if episode_min_distances:
            for min_dist in episode_min_distances:
                if min_dist is not None and np.isfinite(min_dist):
                    if min_dist < 0:  # 负值表示穿透
                        penetration_depths.append(abs(min_dist))
                        penetration_count_episodes += 1
        
        penetration_stat = None
        if penetration_depths:
            penetration_stat = {
                'count': penetration_count_episodes,
                'max_depth': float(np.max(penetration_depths)),
                'mean_depth': float(np.mean(penetration_depths)),
                'min_depth': float(np.min(penetration_depths))
            }
        
        print(f"✅ 回合 {episode_idx + 1} 完成:")
        print(f"   - 奖励: {episode_reward:.2f}")
        print(f"   - 步数: {step_count}/{episode_length} (完成度: {step_count/episode_length*100:.1f}%)")
        print(f"   - 用时: {episode_duration:.2f}秒")
        print(f"   - 平均步时: {avg_step_time:.4f}秒/步")
        print(f"   - 碰撞次数: {total_collisions} (智能体: {episode_collision_counts})")
        print(f"   - 成功率: 团队={success_flag}, 智能体={agent_success_flags}")
        if arrival_step is not None:
            print(f"   - 到达步数: {arrival_step}")
        if min_distance_stat is not None:
            print(f"   - 最小净空距离: 均值={min_distance_stat['mean']:.2f}, 最小值={min_distance_stat['min']:.2f}")
        if penetration_stat is not None:
            print(f"   - 穿透统计: 次数={penetration_stat['count']}, 最大深度={penetration_stat['max_depth']:.2f}, 平均深度={penetration_stat['mean_depth']:.2f}")
        if step_count < episode_length:
            print(f"   ⚠️  注意: 回合提前结束（可能由于done=True或提前终止）")
        
        return {
            'episode': episode_idx,
            'reward': episode_reward,
            'steps': step_count,
            'trajectory': episode_trajectory,
            'actions_history': episode_actions_history,  # 🔧 新增：返回动作历史
            'duration': episode_duration,
            # 🔧 新增：返回碰撞和成功指标（与训练脚本一致）
            'collision_count': total_collisions,
            'agent_collision_counts': episode_collision_counts,
            'min_distance': min_distance_stat,
            'success': success_flag,
            'agent_success_flags': agent_success_flags,
            'team_success': team_success_flag,
            # 🔧 新增：到达时间/步数
            'arrival_step': arrival_step,
            'arrival_time': arrival_time,
            # 🔧 新增：穿透深度统计
            'penetration_stat': penetration_stat
        }
        
    def generate_visualization(self, episode_data, is_best=False):
        """生成可视化结果，仿照1.0版本
        
        Args:
            episode_data: 回合数据
            is_best: 是否为最佳回合（用于文件名标识）
        """
        if not self.visualizer or not episode_data['trajectory']:
            return
            
        print("🎨 正在生成可视化结果...")
        
        # 创建保存目录
        os.makedirs(self.args.save_viz_path, exist_ok=True)
        
        # 仿照1.0版本的可视化参数
        viz_args = argparse.Namespace(
            save_gifs=True,
            save_trajectory_images=True,
            exp_name=os.path.basename(self.args.save_viz_path),
            save_dir=self.args.save_viz_path
        )
        
        try:
            # 生成轨迹图像（同时传入场景提取的目标信息，避免空字典提示）
            terrain_level = episode_data.get('terrain_complexity_level', 'unknown')
            # 🔧 修改：如果是最佳回合，使用best前缀（与训练脚本一致）
            if is_best:
                image_path = os.path.join(
                    self.args.save_viz_path, 
                    f"best_trajectory_ep{episode_data['episode']+1}_r{episode_data['reward']:.0f}.png"
                )
            else:
                image_path = os.path.join(
                    self.args.save_viz_path, 
                    f"trajectory_ep{episode_data['episode']}_level{terrain_level}_r{episode_data['reward']:.0f}.png"
                )
            goal_positions_img = None
            try:
                goal_positions_img = self._get_goal_positions_from_scenario()
            except Exception:
                goal_positions_img = None
            
            # 原有轨迹图（如果记录了动作历史，会自动生成动作时序图）
            actor_outputs_history = None
            if 'actions_history' in episode_data and episode_data['actions_history']:
                try:
                    # 将动作历史转换为numpy数组格式（与训练时一致）
                    # 格式: (steps, n_agents, action_dim)
                    actor_outputs_history = np.array(episode_data['actions_history'])
                    print(f"✅ 检测到动作历史数据，长度: {len(actor_outputs_history)} 步")
                    
                    # 🔧 关键修复：评估时每步都记录，但时序图期望采样数据（每10步一个点）
                    # 为了与训练时一致，需要按10步间隔采样
                    # 注意：os已在文件开头导入，不需要重复导入
                    actor_output_interval = int(os.getenv('ACTOR_OUTPUT_SAMPLE_INTERVAL', '10'))
                    if actor_output_interval <= 0:
                        actor_output_interval = 10
                    
                    # 按间隔采样，与训练时保持一致
                    if len(actor_outputs_history) > actor_output_interval:
                        sampled_indices = list(range(0, len(actor_outputs_history), actor_output_interval))
                        actor_outputs_history = actor_outputs_history[sampled_indices]
                        print(f"✅ 动作历史已采样：原始{len(episode_data['actions_history'])}步 → 采样后{len(actor_outputs_history)}步（间隔={actor_output_interval}）")
                except Exception as e:
                    print(f"⚠️ 动作历史数据转换失败: {e}")
                    import traceback
                    traceback.print_exc()
                    actor_outputs_history = None
            
            # 🔧 关键修复：传递env_instance参数，确保能正确绘制地形和目标位置
            # 如果没有env_instance，visualizer无法获取正确的地形数据，导致图片为空
            # 🚨 关键修复：确保goal_positions不为None，如果获取失败则使用场景中的目标位置
            if goal_positions_img is None:
                try:
                    goal_positions_img = self._get_goal_positions_from_scenario()
                    if goal_positions_img and goal_positions_img.get('goal_pos') is None:
                        # 如果仍然没有目标位置，尝试从world.landmarks获取
                        if hasattr(self.world, 'landmarks') and len(self.world.landmarks) > 0:
                            landmark = self.world.landmarks[0]
                            if hasattr(landmark, 'state') and hasattr(landmark.state, 'p_pos'):
                                goal_positions_img = {'goal_pos': landmark.state.p_pos.tolist(), 'agent_goals': []}
                                print(f"✅ 从world.landmarks获取目标位置: {goal_positions_img['goal_pos']}")
                except Exception as e:
                    print(f"⚠️  获取目标位置失败: {e}")
            
            self.visualizer.generate_trajectory_image(
                    trajectories=episode_data['trajectory'],
                    scenario=self.scenario,
                    save_path=image_path,
                    episode_num=episode_data['episode'],
                    reward=episode_data['reward'],
                    episode_type='evaluation',
                    goal_positions=goal_positions_img,
                    env_instance=self.env,  # 🔧 关键修复：传递环境实例，用于获取地形和目标数据
                    actor_outputs_history=actor_outputs_history  # 🔧 传入动作历史（如果存在）
            )

            # 叠加障碍/地形等信息的增强版图（默认禁用）
            enable_overlay = getattr(self.args, 'enable_overlay', False) and not getattr(self.args, 'disable_overlay', False)
            if enable_overlay:
                overlay_path = os.path.join(
                    self.args.save_viz_path,
                    f"trajectory_ep{episode_data['episode']}_level{terrain_level}_overlay.png"
                )
                self._generate_overlay_image(episode_data, overlay_path)
                print(f"✅ Overlay图片已保存: {overlay_path}")
            else:
                print(f"⏭️ 跳过overlay图片生成（已禁用）")
            
            # 🔧 修改：只生成最佳回合的交互式HTML（与训练脚本一致）
            # 通过环境变量SAVE_INTERACTIVE_TRAJ控制，且只在最佳回合时生成
            enable_interactive_traj = os.getenv('SAVE_INTERACTIVE_TRAJ', '0').lower() in ('1','true','yes','on')
            enable_html = enable_interactive_traj and is_best
            if enable_html:
                # 🔧 修改：使用best前缀（与训练脚本一致）
                html_path = os.path.join(
                    self.args.save_viz_path,
                    f"best_trajectory_ep{episode_data['episode']+1}_interactive.html"
                )
                # 🚨 关键修复：确保交互式HTML中显示目标点
                goal_positions_html = None
                try:
                    # 优先使用之前获取的目标位置
                    if goal_positions_img is not None:
                        goal_positions_html = goal_positions_img
                    else:
                        # 尝试从环境获取
                        if hasattr(self.env, 'get_goal_positions'):
                            goal_positions_html = self.env.get_goal_positions(0)
                        if not isinstance(goal_positions_html, dict) or goal_positions_html.get('goal_pos') is None:
                            goal_positions_html = self._get_goal_positions_from_scenario()
                        # 如果仍然没有，尝试从world.landmarks获取
                        if (not goal_positions_html or goal_positions_html.get('goal_pos') is None) and hasattr(self.world, 'landmarks') and len(self.world.landmarks) > 0:
                            landmark = self.world.landmarks[0]
                            if hasattr(landmark, 'state') and hasattr(landmark.state, 'p_pos'):
                                goal_positions_html = {'goal_pos': landmark.state.p_pos.tolist(), 'agent_goals': []}
                                print(f"✅ 交互式HTML: 从world.landmarks获取目标位置: {goal_positions_html['goal_pos']}")
                except Exception as e:
                    print(f"⚠️  获取交互式HTML目标位置失败: {e}")
                    goal_positions_html = None
                
                self.visualizer.generate_trajectory_interactive(
                    trajectories=episode_data['trajectory'],
                    save_path=html_path,
                    title=f"Evaluation Episode {episode_data['episode']} (reward={episode_data['reward']:.1f})",
                    goal_positions=goal_positions_html,
                    scenario=self.scenario,
                    env_instance=self.env  # 🔧 关键修复：传递环境实例，确保能获取目标位置
                )

            # 🔧 修改：只生成最佳回合的GIF（与训练脚本一致）
            if is_best and len(episode_data['trajectory']) > 10 and not getattr(self.args, 'disable_gif', False):
                gif_path = os.path.join(
                    self.args.save_viz_path,
                    f"best_trajectory_ep{episode_data['episode']+1}_animation.gif"
                )
                self.visualizer.generate_trajectory_gif(
                    trajectories=episode_data['trajectory'],
                    scenario=self.scenario,
                    save_path=gif_path,
                    episode_num=episode_data['episode'],
                    reward=episode_data['reward'],
                    goal_positions=goal_positions_img,
                    gif_max_frames=getattr(self.args, 'gif_max_frames', 60)
                )
            
            # 如禁用HTML
            if not enable_html:
                print(f"⏭️ 跳过HTML轨迹图生成（已禁用）")
            
            print(f"✅ 可视化结果已保存到: {self.args.save_viz_path}")
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
            traceback.print_exc()
            
    def run_evaluation(self):
        """运行完整评估流程"""
        print("="*60)
        print("🔬 MADDPG模型评估开始")
        print("="*60)
        
        # 加载模型
        self.load_model()
        
        # 评估统计
        all_rewards = []
        all_episodes_data = []
        
        # 🔧 新增：跟踪最佳回合（与训练脚本逻辑一致）
        best_reward = -np.inf
        best_episode = 0
        best_episode_data = None
        best_trajectory = None
        best_actor_outputs_history = None
        
        # 🔧 新增：检查是否保存最佳回合可视化（与训练脚本一致）
        enable_best_traj = os.getenv('SAVE_BEST_TRAJ', '1').lower() in ('1','true','yes','on')
        enable_interactive = os.getenv('SAVE_INTERACTIVE_TRAJ', '0').lower() in ('1','true','yes','on')
        
        print(f"\n📊 开始评估 {self.args.eval_episodes} 个回合")
        print(f"🏔️ 地形模式: {'随机地形' if self.args.random_terrain else '固定地形'}")
        if self.args.terrain_complexity_level is not None:
            print(f"🏔️ 地形复杂度等级: {self.args.terrain_complexity_level}")
        else:
            print(f"🏔️ 地形复杂度等级: 随机选择 (1-4)")
        print(f"📸 保存最佳回合可视化: {'是' if enable_best_traj else '否'}")
        print(f"📊 保存交互式HTML: {'是' if enable_interactive else '否'}")
        
        # 🔧 关键修复：从环境变量读取地形种子序列，确保所有评估模式使用相同的地图顺序
        terrain_seed_sequence = None
        terrain_seed_str = os.getenv('TERRAIN_SEED_SEQUENCE', '')
        if terrain_seed_str:
            try:
                terrain_seed_sequence = [int(s.strip()) for s in terrain_seed_str.split(',') if s.strip()]
                print(f"🔧 使用预定义地形种子序列（共{len(terrain_seed_sequence)}个）: {terrain_seed_sequence[:5]}... (前5个)")
            except Exception as e:
                print(f"⚠️  解析地形种子序列失败: {e}，将使用随机地形")
                terrain_seed_sequence = None
        
        for episode in range(self.args.eval_episodes):
            print(f"\n🚀 开始评估回合 {episode + 1}/{self.args.eval_episodes}")
            
            # 🔧 关键修复：如果提供了地形种子序列，使用序列中的种子重新生成地形
            if terrain_seed_sequence and episode < len(terrain_seed_sequence):
                terrain_seed = terrain_seed_sequence[episode]
                print(f"🔧 使用预定义地形种子: {terrain_seed} (回合 {episode + 1})")
                # 重新生成地形（使用指定的种子）
                if hasattr(self.scenario, 'regenerate_terrain'):
                    self.scenario.regenerate_terrain(new_seed=terrain_seed)
                    # 重新创建world（因为地形已变化）
                    self.world = self.scenario.make_world()
                    # 重新创建环境（因为world已变化）
                    self.env = MultiAgentEnv(
                        self.world,
                        reset_callback=self.scenario.reset_world,
                        reward_callback=self.scenario.reward,
                        observation_callback=self.scenario.observation,
                        done_callback=self.scenario.is_done,
                        info_callback=None
                    )
            
            # 为每个回合随机选择地形复杂度等级（如果未指定）
            if self.args.terrain_complexity_level is None:
                terrain_level = np.random.randint(1, 5)  # 1-4
                print(f"🎲 随机选择地形复杂度等级: {terrain_level}")
                # 临时设置地形复杂度等级
                original_level = getattr(self.scenario, 'terrain_complexity_level', None)
                self.scenario.terrain_complexity_level = terrain_level
            else:
                terrain_level = self.args.terrain_complexity_level
                print(f"🏔️ 使用指定地形复杂度等级: {terrain_level}")
                # 确保场景的terrain_complexity_level也被设置
                self.scenario.terrain_complexity_level = terrain_level
            
            # 🔧 关键修复：添加异常处理，确保单个回合失败不会导致整个评估失败
            try:
                episode_data = self.evaluate_single_episode(episode)
                if episode_data is None:
                    print(f"⚠️  回合 {episode + 1} 评估返回空数据，跳过")
                    continue
                
                # 验证episode_data是否包含必要字段
                if 'reward' not in episode_data or 'trajectory' not in episode_data:
                    print(f"⚠️  回合 {episode + 1} 评估数据不完整，跳过")
                    print(f"    episode_data keys: {list(episode_data.keys())}")
                    continue
                
                episode_data['terrain_complexity_level'] = terrain_level
                all_rewards.append(episode_data['reward'])
                all_episodes_data.append(episode_data)
                
                # 🔧 修改：跟踪最佳回合（与训练脚本逻辑一致）
                if episode_data['reward'] > best_reward:
                    best_reward = episode_data['reward']
                    best_episode = episode
                    best_episode_data = episode_data.copy()
                    best_trajectory = episode_data.get('trajectory', [])
                    best_actor_outputs_history = episode_data.get('actions_history', None)
                    print(f"✅ 更新最佳回合: Episode {episode + 1}, Reward = {best_reward:.2f}")
                
                # 🔧 修改：不再为每个回合生成可视化，只在最后生成最佳回合的可视化
                # 与训练脚本保持一致，避免生成大量文件
            except Exception as ep_e:
                print(f"❌ 回合 {episode + 1} 评估失败: {ep_e}")
                import traceback
                traceback.print_exc()
                # 继续下一个回合，不中断整个评估流程
                continue
                
        # 评估总结
        print("\n" + "="*60)
        print("📈 评估结果总结")
        print("="*60)
        
        # 🔧 关键修复：检查是否有有效的评估结果
        if len(all_rewards) == 0:
            print("❌ 警告: 没有成功完成任何评估回合！")
            print("   可能的原因:")
            print("   1. 模型加载失败（维度不匹配）")
            print("   2. 环境初始化失败")
            print("   3. 势场修正配置不匹配")
            print("   4. 评估过程中出现异常")
            print("")
            print("   建议检查:")
            print("   - 模型路径是否正确")
            print("   - 训练配置（use_pf_feature, use_fr_feature）是否与评估时一致")
            print("   - DELTA_*参数是否与训练时一致（传统APF需要DELTA_*=0.0）")
            print("   - 查看上方的错误信息")
            # 即使没有结果，也保存一个空的评估结果文件，便于调试
            results = {
                'model_path': self.args.load_model_path,
                'scenario': self.args.scenario_name,
                'episodes': 0,
                'avg_reward': None,
                'std_reward': None,
                'max_reward': None,
                'min_reward': None,
                'all_rewards': [],
                'episode_details': [],
                'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': 'No episodes completed successfully'
            }
            results_path = os.path.join(self.args.save_viz_path, 'evaluation_results.json')
            os.makedirs(self.args.save_viz_path, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"⚠️  已保存空的评估结果文件: {results_path}")
            return results
        
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        max_reward = np.max(all_rewards)
        min_reward = np.min(all_rewards)
        
        print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"最高奖励: {max_reward:.2f}")
        print(f"最低奖励: {min_reward:.2f}")
        print(f"总回合数: {len(all_rewards)}")
        
        # 保存评估结果
        results = {
            'model_path': self.args.load_model_path,
            'scenario': self.args.scenario_name,
            'episodes': len(all_rewards),
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
            'all_rewards': [float(r) for r in all_rewards],
            'episode_details': [
                {
                    'episode': ep['episode'],
                    'reward': float(ep['reward']),
                    'steps': ep['steps'],
                    'terrain_complexity_level': ep.get('terrain_complexity_level', 'unknown'),
                    'duration': ep['duration'],
                    # 🔧 新增：保存轨迹数据（用于生成对比交互图）
                    'trajectory': ep.get('trajectory', []),  # 保存轨迹数据
                    # 🔧 新增：保存碰撞和成功指标（与训练脚本一致）
                    'collision_count': ep.get('collision_count', 0),
                    'agent_collision_counts': ep.get('agent_collision_counts', []),
                    'min_distance': ep.get('min_distance', None),
                    'success': ep.get('success', 0),
                    'agent_success_flags': ep.get('agent_success_flags', []),
                    'team_success': ep.get('team_success', 0),
                    # 🔧 新增：到达时间/步数
                    'arrival_step': ep.get('arrival_step', None),
                    'arrival_time': ep.get('arrival_time', None),
                    # 🔧 新增：穿透深度统计
                    'penetration_stat': ep.get('penetration_stat', None)
                } for ep in all_episodes_data
            ],
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_path = os.path.join(self.args.save_viz_path, 'evaluation_results.json')
        
        # 🔧 修复：将numpy数组转换为列表，确保JSON可序列化
        def convert_to_json_serializable(obj):
            """递归地将numpy数组和其他不可序列化对象转换为JSON可序列化的格式"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, (bool, int, float, str)):
                return obj
            else:
                # 对于其他类型，尝试转换为字符串
                try:
                    return str(obj)
                except Exception:
                    return None
        
        # 转换结果数据
        results_serializable = convert_to_json_serializable(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 评估结果已保存: {results_path}")
        
        # 🔧 新增：生成最佳回合的可视化（与训练脚本逻辑一致）
        if enable_best_traj and best_episode_data is not None and not self.args.disable_visualization:
            print(f"\n🎨 正在生成最佳回合可视化 (Episode {best_episode + 1}, Reward = {best_reward:.2f})...")
            try:
                # 准备最佳回合数据
                best_episode_data_for_viz = best_episode_data.copy()
                best_episode_data_for_viz['trajectory'] = best_trajectory
                if best_actor_outputs_history is not None:
                    best_episode_data_for_viz['actions_history'] = best_actor_outputs_history
                
                # 生成可视化（只生成最佳回合）
                self.generate_visualization(best_episode_data_for_viz, is_best=True)
                print(f"✅ 最佳回合可视化已生成")
            except Exception as viz_e:
                print(f"⚠️  最佳回合可视化生成失败: {viz_e}")
                import traceback
                traceback.print_exc()
        elif not enable_best_traj:
            print(f"⏭️  跳过最佳回合可视化生成（SAVE_BEST_TRAJ=0）")
        elif self.args.disable_visualization:
            print(f"⏭️  跳过最佳回合可视化生成（--disable-visualization）")
        
        # 显示生成的文件信息
        print(f"\n📁 生成的文件:")
        print(f"  📊 评估统计: {results_path}")
        if enable_best_traj and best_episode_data is not None:
            print(f"  🏆 最佳回合: Episode {best_episode + 1}, Reward = {best_reward:.2f}")
        
        # 列出生成的图片和HTML文件
        if os.path.exists(self.args.save_viz_path):
            print(f"  🖼️  生成的图片:")
            png_files = [f for f in os.listdir(self.args.save_viz_path) if f.endswith('.png')]
            for png_file in png_files[:5]:  # 只显示前5个
                file_path = os.path.join(self.args.save_viz_path, png_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"     {png_file} ({file_size:.1f}KB)")
            
            print(f"  🎬 生成的动画:")
            gif_files = [f for f in os.listdir(self.args.save_viz_path) if f.endswith('.gif')]
            for gif_file in gif_files[:3]:  # 只显示前3个
                file_path = os.path.join(self.args.save_viz_path, gif_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"     {gif_file} ({file_size:.1f}KB)")
            
            print(f"  🌐 生成的HTML交互图:")
            html_files = [f for f in os.listdir(self.args.save_viz_path) if f.endswith('.html')]
            for html_file in html_files[:5]:  # 只显示前5个
                file_path = os.path.join(self.args.save_viz_path, html_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"     {html_file} ({file_size:.1f}KB)")
        
        print(f"\n💡 查看结果:")
        print(f"   cd {self.args.save_viz_path} && ls -la")
        print(f"   python -m http.server 8000  # 启动HTTP服务器查看HTML文件")
        
        return results

    # ============= 可视化增强：障碍/地形/安全区叠加 ============= #
    def _get_extent_from_world(self):
        """尽量从world/terrain获取绘图区间；失败则返回None"""
        try:
            terrain = getattr(self.world, 'terrain', None)
            if terrain is not None and hasattr(terrain, 'extent'):
                return terrain.extent  # (xmin, xmax, ymin, ymax)
        except Exception:
            pass
        return None

    def _derive_extent_from_trajectory(self, traj):
        try:
            xs, ys = [], []
            for step_pos in traj:
                for p in step_pos:
                    if len(p) >= 2:
                        xs.append(float(p[0])); ys.append(float(p[1]))
            if not xs:
                return None
            pad = 5.0
            return (min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad)
        except Exception:
            return None

    def _plot_terrain_and_obstacles(self, ax, extent):
        """叠加地形等高线/障碍掩膜/圆形障碍"""
        # 地形高度图/等高线
        try:
            terrain = getattr(self.world, 'terrain', None)
            if terrain is not None and hasattr(terrain, 'height_map') and not self.args.no_plot_terrain:
                hmap = np.asarray(terrain.height_map)
                if hmap.ndim == 2 and hmap.size > 0:
                    ax.contourf(hmap, levels=20, cmap='Greys', alpha=0.25, extent=extent)
            if terrain is not None and hasattr(terrain, 'obstacle_mask') and not self.args.no_plot_obstacles:
                mask = np.asarray(terrain.obstacle_mask).astype(float)
                if mask.ndim == 2 and mask.size > 0:
                    ax.imshow(mask, cmap='Reds', alpha=0.25, extent=extent, origin='lower')
        except Exception:
            pass

        # 圆形/实体障碍
        try:
            if not self.args.no_plot_obstacles:
                obs_list = getattr(self.world, 'obstacles', None)
                if isinstance(obs_list, (list, tuple)):
                    for ob in obs_list:
                        pos = getattr(getattr(ob, 'state', None), 'p_pos', None)
                        r = getattr(ob, 'radius', None) or getattr(ob, 'size', None)
                        if pos is None or r is None:
                            continue
                        x, y = float(pos[0]), float(pos[1])
                        circ = plt.Circle((x, y), float(r), color='red', alpha=0.25, lw=1.0)
                        ax.add_patch(circ)
        except Exception:
            pass

    def _plot_trajectories(self, ax, traj):
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
        n_agents = self.n_agents
        steps = len(traj)
        for i_agent in range(n_agents):
            xs, ys = [], []
            for s in range(steps):
                p = traj[s][i_agent]
                xs.append(float(p[0])); ys.append(float(p[1]))
            ax.plot(xs, ys, '-', lw=2, color=colors[i_agent % len(colors)], label=f'agent{i_agent}')
            # 起点/终点
            ax.plot(xs[0], ys[0], 'o', color=colors[i_agent % len(colors)], ms=5, alpha=0.9)
            ax.plot(xs[-1], ys[-1], 's', color=colors[i_agent % len(colors)], ms=5, alpha=0.9)

    def _compute_min_inter_agent_distance(self, traj):
        min_d = math.inf
        cnt_below = 0
        thr = getattr(self.args, 'minimum_clearance', None)
        for step_pos in traj:
            for i in range(len(step_pos)):
                for j in range(i+1, len(step_pos)):
                    dx = float(step_pos[i][0]) - float(step_pos[j][0])
                    dy = float(step_pos[i][1]) - float(step_pos[j][1])
                    d = math.hypot(dx, dy)
                    if d < min_d:
                        min_d = d
                    if thr is not None and d < float(thr):
                        cnt_below += 1
        return (min_d if min_d != math.inf else None), cnt_below

    def _generate_overlay_image(self, episode_data, save_path):
        """生成overlay图片，包含地形、障碍、目标点和轨迹"""
        traj = episode_data['trajectory']
        if not traj:
            return
        extent = self._get_extent_from_world()
        if extent is None:
            extent = self._derive_extent_from_trajectory(traj)
        fig, ax = plt.subplots(figsize=(10, 8))
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # 地形/障碍叠加
        self._plot_terrain_and_obstacles(ax, extent)
        
        # 🔧 关键修复：绘制目标点（中央目标和各智能体目标）
        try:
            goal_positions = self._get_goal_positions_from_scenario()
            if goal_positions:
                # 绘制中央目标
                if 'goal_pos' in goal_positions and goal_positions['goal_pos'] is not None:
                    g = goal_positions['goal_pos']
                    try:
                        import numpy as _np
                        g = _np.asarray(g, dtype=_np.float32).reshape(-1)
                        if len(g) >= 2:
                            gx, gy = float(g[0]), float(g[1])
                            ax.scatter(gx, gy, color='yellow', marker='*', s=500, 
                                      edgecolors='red', linewidth=2, zorder=1000, 
                                      label='Goal', alpha=0.9)
                            ax.text(gx, gy + 5.0, "GOAL", color='red', fontsize=14,
                                   fontweight='bold', ha='center', va='bottom', zorder=1000)
                    except Exception as e:
                        print(f"⚠️ 绘制中央目标失败: {e}")
                
                # 绘制各智能体目标
                if 'agent_goals' in goal_positions and isinstance(goal_positions['agent_goals'], list):
                    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
                    for idx, gp in enumerate(goal_positions['agent_goals']):
                        if gp is None:
                            continue
                        try:
                            import numpy as _np
                            gpa = _np.asarray(gp, dtype=_np.float32).reshape(-1)
                            if len(gpa) >= 2:
                                gx, gy = float(gpa[0]), float(gpa[1])
                                c = colors[idx % len(colors)]
                                ax.scatter(gx, gy, color=c, marker='^', s=200, zorder=900, 
                                          alpha=0.9, label=f'Agent {idx} Target')
                                ax.text(gx, gy + 3.0, f"Agent {idx}", color=c, fontsize=10,
                                       ha='center', va='bottom', zorder=900, fontweight='bold')
                        except Exception as e:
                            print(f"⚠️ 绘制智能体{idx}目标失败: {e}")
        except Exception as e:
            print(f"⚠️ 获取目标位置失败: {e}")

        # 轨迹
        self._plot_trajectories(ax, traj)

        # 安全距离统计
        min_d, cnt_below = self._compute_min_inter_agent_distance(traj)
        subtitle = f"min inter-agent d: {min_d:.2f}" if min_d is not None else "min inter-agent d: N/A"
        if getattr(self.args, 'minimum_clearance', None) is not None:
            subtitle += f", <thr count: {cnt_below} (thr={self.args.minimum_clearance})"

        ax.set_title(f"Trajectory Overlay - ep {episode_data['episode']} | reward {episode_data['reward']:.1f}\n{subtitle}")
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.grid(True, ls='--', alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _get_goal_positions_from_scenario(self):
        """从场景中获取目标位置信息"""
        try:
            result = {'goal_pos': None, 'agent_goals': []}
            
            # 获取中央目标位置
            if hasattr(self.scenario, 'goal_pos') and self.scenario.goal_pos is not None:
                result['goal_pos'] = np.asarray(self.scenario.goal_pos, dtype=np.float32)
                print(f"✅ 找到中央目标位置: {result['goal_pos']}")
            else:
                print("⚠️ 场景中没有找到中央目标位置")
            
            # 获取每个智能体的目标位置
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                agents = self.env.world.agents
                for i, agent in enumerate(agents):
                    if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state'):
                        agent_goal = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
                        result['agent_goals'].append(agent_goal)
                        print(f"✅ 找到智能体{i}目标位置: {agent_goal}")
                    else:
                        result['agent_goals'].append(None)
                        print(f"⚠️ 智能体{i}没有独立目标位置")
            
            return result
            
        except Exception as e:
            print(f"⚠️ 从场景获取目标信息失败: {e}")
            return {'goal_pos': None, 'agent_goals': []}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MADDPG优化版模型评估脚本",
        epilog="""
示例用法:
  # 基本评估（生成PNG、GIF和HTML文件）
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --eval-episodes 5
  
  # 禁用HTML生成
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --disable-html
  
  # 禁用所有可视化
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --disable-visualization
  
  # 使用固定位置
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --use-fixed-positions --positions-file ./saved_positions/my_positions.json
  
  # 调整势场参数
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --action-force-ratio 0.8 --influence-range 3.0
  
  # 禁用势场修正
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --enable-action-correction false
  
  # 启用overlay图片（包含地形信息）
  python3 evaluate_optimized.py --load-model-path models/optimized_exp/best --enable-overlay

生成的文件:
  - trajectory_ep{episode}_r{reward}.png: 静态轨迹图
  - trajectory_ep{episode}_overlay.png: 带地形信息的轨迹图（需要--enable-overlay）
  - trajectory_ep{episode}_animation.gif: 轨迹动画
  - trajectory_ep{episode}_interactive.html: 可交互3D轨迹图（需要plotly）
  - evaluation_results.json: 评估统计结果

HTML交互式轨迹图功能:
  - 支持3D视角拖拽和缩放
  - 显示智能体轨迹、目标位置和地形信息
  - 每个评估回合都会生成独立的HTML文件
  - 需要安装plotly: pip install plotly
  - 可通过--disable-html参数禁用
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 环境参数
    parser.add_argument("--scenario-name", type=str, default="paper3d_terrain_weighted", 
                       help="场景名称 (paper3d_terrain_weighted, paper3d_terrain_vectorized, paper3d_terrain_energy)")
    parser.add_argument("--episode-length", type=int, default=2200, 
                       help="每回合最大步数（默认2200，与训练脚本一致）")
    parser.add_argument("--eval-episodes", type=int, default=3, 
                       help="评估回合数（将随机生成不同复杂度的地图）")
    parser.add_argument("--terrain-complexity-level", type=int, default=None, 
                       help="地形复杂度等级 (1-4)，None表示随机选择")
    parser.add_argument("--random-terrain", action="store_true", default=False,
                       help="使用随机地形（默认启用）")
    # 🔧 修复：与训练脚本保持一致的默认参数
    parser.add_argument("--gravity", type=float, default=0.0, help="环境重力加速度（作用于 -Z 方向），默认0.0（无重力）")
    parser.add_argument("--control-accel-gain", type=float, default=1.0, help="动作到物理加速度的控制增益，默认1.0")
    parser.add_argument("--reward-pos-scale", type=float, default=1.5, help="正向奖励缩放系数，默认1.5")
    parser.add_argument("--reward-neg-scale", type=float, default=2.5, help="负向奖励缩放系数，默认2.5")
    parser.add_argument("--agent-max-speed", type=float, default=37.5, help="智能体最大速度，默认37.5")
    parser.add_argument("--agent-accel", type=float, default=3.6, help="智能体加速度，默认3.6")
    parser.add_argument("--action-range-x", type=float, default=3.5, help="动作X轴映射范围系数（将网络输出乘以该系数），默认3.5")
    parser.add_argument("--action-range-y", type=float, default=3.5, help="动作Y轴映射范围系数（将网络输出乘以该系数），默认3.5")
    parser.add_argument("--action-range-z", type=float, default=3.0, help="动作Z轴映射范围系数（将网络输出乘以该系数），默认3.0")
    parser.add_argument("--damping", type=float, default=0.18, help="速度阻尼系数，默认0.18")
    
    # 势场/动作修正相关参数
    parser.add_argument("--enable-action-correction", type=lambda x: (str(x).lower() == 'true'), default=True, 
                       help="启用势场/混合动作修正（如集成时生效）")
    parser.add_argument("--correction-type", type=str, default="potential_field", 
                       choices=["potential_field", "hybrid", "none"], help="修正类型")
    parser.add_argument("--influence-range", type=float, default=2.5, help="Potential field influence range")
    parser.add_argument("--force-param-ratio", type=float, default=0.8, help="Potential field parameter adjustment base coefficient")
    
    # 势场力参数范围映射
    parser.add_argument("--force-param-goal-attraction-range", type=float, nargs=2, default=[0.5, 3.0], 
                       help="势场力参数：目标吸引力范围 [min, max]，网络输出p[0]映射到此范围")
    parser.add_argument("--force-param-lambda-1-range", type=float, nargs=2, default=[0.1, 2.0], 
                       help="势场力参数：lambda_1范围 [min, max]，网络输出p[1]映射到此范围")
    parser.add_argument("--force-param-terrain-repulsion-range", type=float, nargs=2, default=[0.1, 1.5], 
                       help="势场力参数：地形排斥力范围 [min, max]，网络输出p[2]映射到此范围")
    parser.add_argument("--force-param-detection-radius-range", type=float, nargs=2, default=[2.0, 10.0], 
                       help="势场力参数：检测半径范围 [min, max]，网络输出p[3]映射到此范围")
    
    # 网络动作和势场动作混合比例
    parser.add_argument("--action-force-ratio", type=float, default=0.75, 
                       help="网络动作和势场动作的混合比例 (0.0=完全网络动作, 1.0=完全势场动作, 默认0.75=75%势场+25%网络)")
    
    # 势场修正版本选择
    parser.add_argument("--use-tf-potential-field", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="是否使用TensorFlow版本的势场修正 (1=TF版本, 0=原版)")
    
    # 🔧 新增：地形感知模式参数（仅用于评估时APF地形力计算）
    parser.add_argument("--terrain-sensing-mode", type=str, default="local",
                        choices=["local", "oracle_same_probes", "oracle_dense"],
                        help="地形感知模式: local=使用观测中的地形信息, oracle_same_probes=Oracle真值(相同probe布局), oracle_dense=Oracle真值(密集探测)")
    
    # 🔧 新增：FR和PF特征标志
    parser.add_argument("--use-fr-feature", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="Enable FR feature (Force Ratio as separate input)")
    parser.add_argument("--use-pf-feature", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=True,
                       help="Enable PF feature (Potential field force appended to obs)")
    
    # 🔧 修复：与训练脚本保持一致的势场参数默认值
    parser.add_argument("--goal-attraction", type=float, default=15.0, help="Goal attraction force，默认15.0")
    parser.add_argument("--lambda-1-base", type=float, default=8.5, help="Lambda_1 base value，默认8.5")
    parser.add_argument("--terrain-repulsion", type=float, default=3800.0, help="Terrain repulsion force，默认3800.0")
    parser.add_argument("--agent-influence-range", type=float, default=10.0, help="Agent influence range，默认10.0")
    parser.add_argument("--delta-k-att", type=float, default=0.5, help="Delta K_att，默认0.5")
    parser.add_argument("--delta-lambda-1", type=float, default=2.5, help="Delta Lambda_1，默认2.5")
    parser.add_argument("--delta-k-rep", type=float, default=40.0, help="Delta K_rep，默认40.0")
    parser.add_argument("--delta-radius", type=float, default=5.0, help="Delta Radius，默认5.0")
    
    # 🔧 新增：算法选择
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3"],
                       help="Training algorithm selection (maddpg or matd3)")
    
    # 分项加权奖励参数（如果使用加权场景）
    parser.add_argument("--distance-weight", type=float, default=None, help="距离奖励权重")
    parser.add_argument("--exploration-weight", type=float, default=None, help="探索奖励权重")
    parser.add_argument("--stationary-weight", type=float, default=None, help="停滞惩罚权重")
    parser.add_argument("--direction-weight", type=float, default=None, help="方向一致性奖励权重")
    parser.add_argument("--deviation-weight", type=float, default=None, help="偏离奖励权重")
    parser.add_argument("--start-area-weight", type=float, default=None, help="起始区域奖励权重")
    parser.add_argument("--approach-weight", type=float, default=None, help="接近目标奖励权重")
    parser.add_argument("--energy-weight", type=float, default=None, help="能量效率奖励权重")
    parser.add_argument("--height-weight", type=float, default=None, help="高度适应性奖励权重")
    parser.add_argument("--height-reward-enabled", type=lambda x: (str(x).lower() in ('1','true','yes','on')), default=None, help="是否启用高度奖励")
    parser.add_argument("--height-ideal-min", type=float, default=None, help="理想高度下限")
    parser.add_argument("--height-ideal-max", type=float, default=None, help="理想高度上限")
    parser.add_argument("--lateral-weight", type=float, default=None, help="侧向/绕行奖励权重")
    parser.add_argument("--clearance-weight", type=float, default=None, help="净空/最小距离增益奖励权重")
    parser.add_argument("--clearance-d-max", type=float, default=None, help="净空奖励归一化因子")
    parser.add_argument("--success-weight", type=float, default=None, help="成功奖励权重")
    parser.add_argument("--collision-weight", type=float, default=None, help="碰撞惩罚权重")
    parser.add_argument("--global-weight", type=float, default=None, help="全局奖励权重")
    parser.add_argument("--shaping-weight", type=float, default=None, help="潜势函数 shaping 权重")
    parser.add_argument("--max-reward", type=float, default=None, help="最大奖励值")
    parser.add_argument("--min-reward", type=float, default=None, help="最小奖励值")
    parser.add_argument("--success-reward-value", type=float, default=None, help="成功一次性奖励值")
    parser.add_argument("--success-distance-threshold", type=float, default=None, help="成功判定距离阈值")
    parser.add_argument("--collision-penalty-value", type=float, default=None, help="碰撞惩罚绝对值")
    parser.add_argument("--collision-distance-threshold", type=float, default=None, help="碰撞/接触距离阈值")
    parser.add_argument("--global-reward-mode", type=str, default=None, help="全局奖励模式")
    parser.add_argument("--shaping-gamma", type=float, default=None, help="潜势函数 gamma")
    
    # 模型和保存路径
    default_model_path = os.getenv('MODEL_PATH', 'models/optimized_exp/best')
    parser.add_argument("--load-model-path", type=str, default=default_model_path,
                       help="要加载的模型权重文件夹路径（默认从环境变量MODEL_PATH或models/optimized_exp/best）")
    parser.add_argument("--save-viz-path", type=str, default="evaluation_results", 
                       help="可视化结果保存路径")
    
    # 可视化控制
    parser.add_argument("--disable-visualization", action="store_true", 
                       help="禁用可视化生成")
    parser.add_argument("--enable-overlay", action="store_true", default=False,
                       help="启用overlay图片生成（包含地形和障碍物信息）")
    parser.add_argument("--disable-overlay", action="store_true",
                       help="禁用overlay图片生成（默认禁用）")
    parser.add_argument("--disable-gif", action="store_true",
                       help="禁用GIF生成（避免长时间阻塞或大文件）")
    parser.add_argument("--gif-max-frames", type=int, default=60,
                       help="限制GIF的最大帧数（默认60帧）")
    
    # 场景兼容性参数（为了与原版兼容）
    parser.add_argument("--terrain-seed", type=int, default=None, help="地形种子")
    parser.add_argument("--use-fixed-positions", action="store_true", help="使用固定位置")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/default_positions.json", 
                       help="固定位置文件路径")
    parser.add_argument("--dynamic-first-time", action="store_true", help="动态首次运行")
    parser.add_argument("--disable-early-termination", action="store_true", 
                       help="禁用提前终止，强制运行完整的episode_length步数")
    # 与训练一致的隐藏层配置（可选），用于构建相同拓扑以加载权重
    parser.add_argument("--actor-hidden", type=str, default=None, help="Actor隐藏层，例如: 384,256,128,64")
    parser.add_argument("--critic-hidden", type=str, default=None, help="Critic隐藏层，例如: 512,256,128,64")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    try:
        disable_viz_env = os.getenv("EVAL_DISABLE_VISUALIZATION", "0").lower() in ("1", "true", "yes", "on")
    except Exception:
        disable_viz_env = False
    try:
        light_mode_env = os.getenv("EVAL_LIGHT_MODE", "0").lower() in ("1", "true", "yes", "on")
    except Exception:
        light_mode_env = False
    if disable_viz_env or light_mode_env:
        args.disable_visualization = True
        args.disable_gif = True
        setattr(args, 'disable_html', True)
    # 在任何 TensorFlow 操作之前优先配置GPU，避免已初始化后再设内存增长
    try:
        configure_gpu()
    except Exception:
        pass
    
    # 🔧 新增：启用XLA加速（如果环境变量XLA_GLOBAL=1）
    # 与训练脚本保持一致，使用XLA Global模式加速评估
    # 注意：必须在TensorFlow图构建之前启用XLA
    xla_global = os.getenv('XLA_GLOBAL', '1').lower() in ('1', 'true', 'yes', 'on')  # 默认启用
    if xla_global:
        try:
            # 🔧 关键修复：在启用XLA前设置稳定的XLA配置，避免CUDA错误
            # 与训练脚本保持一致，使用稳定的XLA配置
            existing_xla_flags = os.environ.get('XLA_FLAGS', '')
            # 移除所有可能导致问题的flag
            flags_to_remove = [
                '--xla_gpu_enable_triton_gemm=false',
                '--xla_gpu_enable_triton_gemm=true',
                '--xla_gpu_force_compilation_parallelism=1',
                '--xla_gpu_force_compilation_parallelism=0',
            ]
            cleaned_flags = existing_xla_flags
            for flag in flags_to_remove:
                cleaned_flags = cleaned_flags.replace(flag, '')
            cleaned_flags = ' '.join(cleaned_flags.split())  # 清理多余空格
            
            # 设置稳定的XLA配置
            stable_xla_flags = [
                '--xla_gpu_autotune_level=1',  # 降低autotune级别，减少kernel搜索空间
                '--xla_gpu_deterministic_ops=true',  # 强制确定性操作
            ]
            
            # 合并配置
            if cleaned_flags:
                stable_xla_flags_str = cleaned_flags + ' ' + ' '.join(stable_xla_flags)
            else:
                stable_xla_flags_str = ' '.join(stable_xla_flags)
            os.environ['XLA_FLAGS'] = stable_xla_flags_str
            
            # 启用XLA Global JIT编译（与训练脚本一致）
            tf.config.optimizer.set_jit(True)
            print("✅ XLA加速已启用（Global JIT模式）")
            print("   💡 提示：XLA首次编译可能需要一些时间，后续运行会更快")
        except Exception as e:
            print(f"⚠️  XLA加速启用失败: {e}")
            print("   💡 提示：如果遇到问题，可以设置XLA_GLOBAL=0禁用XLA")
    else:
        print("ℹ️  XLA加速未启用（设置XLA_GLOBAL=1以启用）")

    # 设置随机种子以确保每次评估都有不同的随机性
    import time
    import random
    current_time = int(time.time() * 1000000) % 2**32
    random.seed(current_time)
    np.random.seed(current_time)
    tf.random.set_seed(current_time)
    print(f"🎲 设置随机种子: {current_time} (确保每次评估的随机性)")
    
    # 显示HTML生成状态
    enable_html = getattr(args, 'enable_html', True) and not getattr(args, 'disable_html', False)
    if enable_html:
        print("🌐 HTML交互式轨迹图生成: 启用")
        print("💡 提示: 如果HTML生成失败，请安装plotly: pip install plotly")
    else:
        print("🌐 HTML交互式轨迹图生成: 禁用")
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator(args)
        
        # 运行评估
        results = evaluator.run_evaluation()
        
        print("\n🎉 评估完成!")
        
        # 显示HTML文件查看提示
        if enable_html:
            print(f"\n🌐 查看HTML交互式轨迹图:")
            print(f"   cd {args.save_viz_path}")
            print(f"   python -m http.server 8000")
            print(f"   然后在浏览器中打开: http://localhost:8000")
        
    except KeyboardInterrupt:
        print("\n⚠️ 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

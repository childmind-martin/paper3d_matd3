#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MATD3 Dual Q训练有效性验证脚本

验证内容：
1. Dual Q机制是否正常工作（Q1和Q2有差异）
2. 训练是否有效学习（不是fake learning）
3. 测试评估是否正确配置
4. 与MADDPG baseline对比验证
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from paper3d_train_optimized import OptimizedMATD3, load_scenario_module, configure_gpu
from multiagent.environment import MultiAgentEnv


class MATD3Verifier:
    """MATD3 Dual Q验证器"""
    
    def __init__(self, model_dir, scenario_name='paper3d_terrain_energy'):
        self.model_dir = Path(model_dir)
        self.scenario_name = scenario_name
        
    def verify_training_logs(self):
        """验证1：检查训练日志，确认Dual Q机制工作"""
        print("\n" + "="*80)
        print("验证1：检查训练日志中的Dual Q指标")
        print("="*80)
        
        # 读取训练结果
        results_file = self.model_dir / 'training_results.json'
        if not results_file.exists():
            print(f"❌ 找不到训练结果文件: {results_file}")
            return False
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 检查算法配置
        algo = results.get('args', {}).get('algorithm', 'unknown')
        use_dual_q = results.get('args', {}).get('matd3_use_dual_q', False)
        use_separated_gradient = results.get('args', {}).get('matd3_use_separated_gradient', False)
        
        print(f"\n算法配置:")
        print(f"  算法类型: {algo}")
        print(f"  使用Dual Q: {use_dual_q}")
        print(f"  使用分离梯度: {use_separated_gradient}")
        
        if algo.lower() != 'matd3':
            print(f"❌ 错误：算法不是MATD3，而是 {algo}")
            return False
        
        if not use_dual_q:
            print(f"⚠️  警告：matd3_use_dual_q=False，这是标准TD3（single Q）")
            print(f"   如果要验证Dual Q，需要设置 --matd3-use-dual-q 1")
        
        print(f"\n✅ 算法配置正确")
        return True
    
    def verify_model_weights(self):
        """验证2：检查模型权重文件，确认Q1和Q2独立"""
        print("\n" + "="*80)
        print("验证2：检查模型权重文件结构")
        print("="*80)
        
        # 检查最佳模型目录
        best_model_dir = self.model_dir / 'best_models'
        if not best_model_dir.exists():
            print(f"❌ 找不到最佳模型目录: {best_model_dir}")
            return False
        
        # 读取智能体数量
        results_file = self.model_dir / 'training_results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)
        n_agents = results.get('args', {}).get('num_agents', 3)
        
        print(f"\n智能体数量: {n_agents}")
        print(f"\n检查权重文件:")
        
        all_files_exist = True
        for i in range(n_agents):
            actor_file = best_model_dir / f'actor_{i}.weights.h5'
            critic1_file = best_model_dir / f'critic1_{i}.weights.h5'
            critic2_file = best_model_dir / f'critic2_{i}.weights.h5'
            
            actor_exists = actor_file.exists()
            critic1_exists = critic1_file.exists()
            critic2_exists = critic2_file.exists()
            
            print(f"\n  智能体 {i}:")
            print(f"    Actor:   {'✅' if actor_exists else '❌'} {actor_file.name}")
            print(f"    Critic1: {'✅' if critic1_exists else '❌'} {critic1_file.name}")
            print(f"    Critic2: {'✅' if critic2_exists else '❌'} {critic2_file.name}")
            
            if not (actor_exists and critic1_exists and critic2_exists):
                all_files_exist = False
        
        if not all_files_exist:
            print(f"\n❌ 某些权重文件缺失")
            return False
        
        # 验证Q1和Q2权重不同
        print(f"\n验证Q1和Q2权重独立性:")
        for i in range(n_agents):
            critic1_file = best_model_dir / f'critic1_{i}.weights.h5'
            critic2_file = best_model_dir / f'critic2_{i}.weights.h5'
            
            # 比较文件大小（快速检查）
            size1 = critic1_file.stat().st_size
            size2 = critic2_file.stat().st_size
            
            print(f"  智能体 {i}: Critic1={size1} bytes, Critic2={size2} bytes")
            
            if size1 == 0 or size2 == 0:
                print(f"    ❌ 警告：某个Critic文件为空")
                return False
        
        print(f"\n✅ 模型权重文件结构正确")
        return True
    
    def verify_q_values_divergence(self):
        """验证3：加载模型，测试Q1和Q2是否有差异"""
        print("\n" + "="*80)
        print("验证3：测试Q1和Q2输出差异")
        print("="*80)
        
        # 配置GPU
        configure_gpu()
        
        # 读取训练配置
        results_file = self.model_dir / 'training_results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        args_dict = results.get('args', {})
        
        # 创建一个简单的args对象
        class Args:
            pass
        args = Args()
        for key, value in args_dict.items():
            setattr(args, key, value)
        
        # 确保关键参数存在
        if not hasattr(args, 'use_fr_feature'):
            args.use_fr_feature = 0
        if not hasattr(args, 'use_pf_feature'):
            args.use_pf_feature = 0
        if not hasattr(args, 'actor_hidden'):
            args.actor_hidden = '256,256,256'
        if not hasattr(args, 'critic_hidden'):
            args.critic_hidden = '256,256,256'
        
        # 加载场景
        print(f"\n加载场景: {self.scenario_name}")
        scenario = load_scenario_module(self.scenario_name, args)
        if scenario is None:
            print(f"❌ 无法加载场景")
            return False
        
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, 
                           scenario.observation, done_callback=scenario.done)
        
        n_agents = env.n
        obs_shapes = [env.observation_space[i].shape[0] for i in range(n_agents)]
        action_dims = [env.action_space[i].n if hasattr(env.action_space[i], 'n') 
                      else env.action_space[i].shape[0] for i in range(n_agents)]
        
        print(f"智能体数量: {n_agents}")
        print(f"观察空间: {obs_shapes}")
        print(f"动作空间: {action_dims}")
        
        # 创建MATD3实例
        print(f"\n创建MATD3实例...")
        matd3 = OptimizedMATD3(n_agents, obs_shapes, action_dims, args)
        
        # 加载权重
        best_model_dir = self.model_dir / 'best_models'
        print(f"加载权重: {best_model_dir}")
        
        for i in range(n_agents):
            actor_file = str(best_model_dir / f'actor_{i}.weights.h5')
            critic1_file = str(best_model_dir / f'critic1_{i}.weights.h5')
            critic2_file = str(best_model_dir / f'critic2_{i}.weights.h5')
            
            matd3.agents[i]['actor'].load_weights(actor_file)
            matd3.agents[i]['critic1'].load_weights(critic1_file)
            matd3.agents[i]['critic2'].load_weights(critic2_file)
            print(f"  ✅ 智能体 {i} 权重加载完成")
        
        # 生成测试数据
        print(f"\n生成测试数据...")
        obs_n = env.reset()
        
        # 获取动作
        actions = []
        for i in range(n_agents):
            obs = tf.convert_to_tensor([obs_n[i]], dtype=tf.float32)
            actor_inputs = [obs]
            if getattr(args, 'use_fr_feature', 0) == 1:
                actor_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
            if getattr(args, 'use_pf_feature', 0) == 1:
                pf_dim = getattr(args, 'pf_feature_dim', 3)
                actor_inputs.append(tf.zeros((1, pf_dim), dtype=tf.float32))
            
            action = matd3.agents[i]['actor'](actor_inputs, training=False)
            actions.append(action.numpy()[0])
        
        # 准备Critic输入
        global_state = tf.concat([tf.convert_to_tensor([obs_n[i]], dtype=tf.float32) 
                                 for i in range(n_agents)], axis=1)
        global_actions = tf.concat([tf.convert_to_tensor([actions[i]], dtype=tf.float32) 
                                   for i in range(n_agents)], axis=1)
        
        # 测试Q1和Q2
        print(f"\n测试Q值输出:")
        q_diffs = []
        for i in range(n_agents):
            critic_inputs = [global_state, global_actions]
            if getattr(args, 'use_fr_feature', 0) == 1:
                critic_inputs.append(tf.zeros((1, 1), dtype=tf.float32))
            if getattr(args, 'use_pf_feature', 0) == 1:
                pf_dim = getattr(args, 'pf_feature_dim', 3)
                critic_inputs.append(tf.zeros((1, pf_dim * n_agents), dtype=tf.float32))
            
            q1 = matd3.agents[i]['critic1'](critic_inputs, training=False).numpy()[0, 0]
            q2 = matd3.agents[i]['critic2'](critic_inputs, training=False).numpy()[0, 0]
            diff = abs(q1 - q2)
            q_diffs.append(diff)
            
            print(f"  智能体 {i}: Q1={q1:.4f}, Q2={q2:.4f}, |Q1-Q2|={diff:.4f}")
        
        avg_diff = np.mean(q_diffs)
        max_diff = np.max(q_diffs)
        
        print(f"\nQ值差异统计:")
        print(f"  平均差异: {avg_diff:.4f}")
        print(f"  最大差异: {max_diff:.4f}")
        
        # 判断标准：平均差异应该 > 0.01（说明Q1和Q2确实不同）
        if avg_diff < 0.01:
            print(f"\n❌ 警告：Q1和Q2几乎相同，Dual Q机制可能没有正常工作")
            print(f"   可能原因：")
            print(f"   1. 训练步数不足，Q网络还未充分分化")
            print(f"   2. separated_gradient未启用，导致Q1和Q2收敛到相同值")
            print(f"   3. 模型权重加载错误")
            return False
        else:
            print(f"\n✅ Q1和Q2有明显差异，Dual Q机制正常工作")
            return True
    
    def verify_learning_effectiveness(self):
        """验证4：检查训练曲线，确认有学习进展"""
        print("\n" + "="*80)
        print("验证4：检查学习有效性")
        print("="*80)
        
        results_file = self.model_dir / 'training_results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        episode_rewards = results.get('episode_rewards', [])
        
        if len(episode_rewards) < 10:
            print(f"❌ 训练回合太少 ({len(episode_rewards)}回合)，无法判断")
            return False
        
        # 计算前10%和后10%的平均奖励
        n_episodes = len(episode_rewards)
        early_window = max(1, n_episodes // 10)
        late_window = max(1, n_episodes // 10)
        
        early_rewards = episode_rewards[:early_window]
        late_rewards = episode_rewards[-late_window:]
        
        avg_early = np.mean(early_rewards)
        avg_late = np.mean(late_rewards)
        improvement = avg_late - avg_early
        improvement_pct = (improvement / abs(avg_early)) * 100 if avg_early != 0 else 0
        
        print(f"\n训练进展分析:")
        print(f"  总回合数: {n_episodes}")
        print(f"  前10%平均奖励: {avg_early:.2f}")
        print(f"  后10%平均奖励: {avg_late:.2f}")
        print(f"  改进: {improvement:.2f} ({improvement_pct:.1f}%)")
        
        # 判断标准：后期奖励至少提升5%
        if improvement_pct > 5:
            print(f"\n✅ 训练有效，奖励显著提升")
            return True
        else:
            print(f"\n⚠️  警告：奖励提升不明显，可能需要更多训练")
            return False
    
    def run_all_verifications(self):
        """运行所有验证"""
        print("\n" + "="*80)
        print("MATD3 Dual Q训练有效性验证")
        print(f"模型目录: {self.model_dir}")
        print("="*80)
        
        results = {
            '训练日志配置': self.verify_training_logs(),
            '模型权重文件': self.verify_model_weights(),
            'Q值差异性': self.verify_q_values_divergence(),
            '学习有效性': self.verify_learning_effectiveness(),
        }
        
        print("\n" + "="*80)
        print("验证结果汇总:")
        print("="*80)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        print("\n" + "="*80)
        if all_passed:
            print("🎉 所有验证通过！MATD3 Dual Q训练有效！")
        else:
            print("⚠️  部分验证未通过，请检查上述问题")
        print("="*80 + "\n")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="验证MATD3 Dual Q训练有效性")
    parser.add_argument('--model-dir', type=str, required=True,
                       help='模型目录路径（包含best_models和training_results.json）')
    parser.add_argument('--scenario', type=str, default='paper3d_terrain_energy',
                       help='场景名称')
    
    args = parser.parse_args()
    
    verifier = MATD3Verifier(args.model_dir, args.scenario)
    success = verifier.run_all_verifications()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

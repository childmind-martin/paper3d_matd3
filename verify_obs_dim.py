#!/usr/bin/env python3
"""
快速验证观测维度修改是否成功

用法:
    python verify_obs_dim.py

预期输出:
    ✅ 观测维度: 32 (正确)
    ❌ 观测维度: 29 (需要重新加载代码)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("="*60)
    print("🔍 验证观测维度修改")
    print("="*60)
    
    try:
        # 导入必要模块
        print("\n1️⃣  导入模块...")
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
        print("   ✅ 导入成功")
        
        # 加载场景
        print("\n2️⃣  加载场景...")
        scenario_name = 'paper3d_terrain_energy'
        scenario = scenarios.load(f"{scenario_name}.py").Scenario()
        print(f"   ✅ 场景加载: {scenario_name}")
        
        # 创建环境
        print("\n3️⃣  创建环境...")
        world = scenario.make_world()
        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            done_callback=scenario.is_done
        )
        print("   ✅ 环境创建成功")
        
        # 重置环境并获取观测
        print("\n4️⃣  获取观测维度...")
        obs_n = env.reset()
        
        # 检查维度
        print("\n5️⃣  检查结果:")
        print(f"   - 智能体数量: {len(obs_n)}")
        
        for i, obs in enumerate(obs_n):
            obs_dim = len(obs)
            print(f"   - Agent {i} 观测维度: {obs_dim}")
            
            if obs_dim == 32:
                print(f"     ✅ 正确！(预期: 32)")
            elif obs_dim == 29:
                print(f"     ❌ 错误！维度仍是29，需要重新加载代码")
                print(f"     💡 建议:")
                print(f"        1. 清理Python缓存: find . -name '*.pyc' -delete")
                print(f"        2. 重新加载终端或重启Python")
            else:
                print(f"     ⚠️  意外维度: {obs_dim} (预期: 32)")
        
        # 检查observation_space
        print("\n6️⃣  检查observation_space:")
        for i in range(len(env.observation_space)):
            space_dim = env.observation_space[i].shape[0]
            print(f"   - Agent {i} observation_space维度: {space_dim}")
            
            if space_dim == 32:
                print(f"     ✅ 正确！")
            else:
                print(f"     ❌ 错误！")
        
        # 验证前方探测点
        print("\n7️⃣  验证前方探测点数量:")
        print(f"   📝 检查scenario代码中的distances数组...")
        import inspect
        source_file = inspect.getsourcefile(scenario.__class__)
        print(f"   - 场景文件: {source_file}")
        
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'distances = [2, 4, 6, 10, 15, 20, 25, 30]' in content:
                print(f"     ✅ 前方探测点已更新为8个 [2, 4, 6, 10, 15, 20, 25, 30]")
            elif 'distances = [5, 10, 15, 20, 25]' in content:
                print(f"     ❌ 前方探测点仍是5个 [5, 10, 15, 20, 25]")
                print(f"     💡 建议: 检查文件是否保存")
            else:
                print(f"     ⚠️  未找到distances数组定义")
        
        # 总结
        print("\n" + "="*60)
        all_correct = all(len(obs) == 32 for obs in obs_n)
        if all_correct:
            print("🎉 验证通过！观测维度修改成功！")
            print("\n下一步:")
            print("  1. 删除旧的replay buffer")
            print("     rm -rf logs/*/replay_buffer_*.pkl")
            print("  2. 开始训练")
            print("     ./run_optimized.sh 100 1024 'terrain_fix'")
        else:
            print("❌ 验证失败！观测维度不正确")
            print("\n排查步骤:")
            print("  1. 确认代码已保存")
            print("  2. 清理Python缓存:")
            print("     find . -name '*.pyc' -delete")
            print("     find . -name '__pycache__' -type d -exec rm -rf {} +")
            print("  3. 重新运行此脚本")
        print("="*60)
        
        env.close()
        return 0 if all_correct else 1
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())


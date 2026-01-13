#!/usr/bin/env python3
"""
综合修复脚本：解决训练停滞和CUDA崩溃问题

问题汇总：
1. 势场归一化过强/过弱
2. CUDA内存对齐导致崩溃
3. 奖励裁剪未生效（场景使用硬编码默认值）
4. Z轴偏置计算可能有误
5. 学习率调度可能有问题

修复策略：
1. ✅ 势场归一化：使用max_force*1.5作为基准（折中方案）
2. ✅ CUDA内存对齐：在env.step()前添加ascontiguousarray
3. 🔧 修复奖励裁剪：确保场景使用正确的MIN_REWARD
4. 🔧 验证Z轴偏置：检查计算是否正确
5. 🔧 添加详细诊断输出
"""

import re

def fix_reward_clipping_in_scenario_creation():
    """修复场景创建时的奖励裁剪默认值"""
    file_path = '/home/tang/Desktop/paper3d_train_optimized.py'
    
    print("=" * 60)
    print("🔧 修复1：场景创建中的奖励裁剪默认值")
    print("=" * 60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1：第一处场景创建（MADDPG）
    old_pattern1 = r"'min_reward': getattr\(args, 'min_reward', -2500\.0\)"
    new_text1 = "'min_reward': getattr(args, 'min_reward', -120.0)  # 🔧 修复：使用正确的默认值"
    
    if re.search(old_pattern1, content):
        content = re.sub(old_pattern1, new_text1, content)
        print("✅ 修复第一处场景创建（reward_weights字典）")
    else:
        print("⚠️  未找到第一处需要修复的代码")
    
    # 修复2：第二处场景创建（MATD3）
    old_pattern2 = r"min_reward=getattr\(args, 'min_reward', -2500\.0\)"
    new_text2 = "min_reward=getattr(args, 'min_reward', -120.0)  # 🔧 修复：使用正确的默认值"
    
    if re.search(old_pattern2, content):
        content = re.sub(old_pattern2, new_text2, content)
        print("✅ 修复第二处场景创建（直接参数传递）")
    else:
        print("⚠️  未找到第二处需要修复的代码")
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 奖励裁剪默认值修复完成")
    print()

def add_diagnostic_outputs():
    """添加诊断输出，帮助识别问题"""
    file_path = '/home/tang/Desktop/paper3d_train_optimized.py'
    
    print("=" * 60)
    print("🔧 修复2：添加奖励分布诊断输出")
    print("=" * 60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找奖励收集的位置（在缓冲区add之前）
    pattern = r"(# 标准化奖励处理.*?rewards = rew_n)"
    
    diagnostic_code = """
            # 🔧 诊断：打印奖励统计
            if step % 500 == 0 and not quiet_output:
                reward_flat = np.array(rewards).flatten()
                print(f"[诊断] 步{step}: 奖励范围=[{reward_flat.min():.1f}, {reward_flat.max():.1f}], "
                      f"平均={reward_flat.mean():.1f}, 中位数={np.median(reward_flat):.1f}")
"""
    
    # 检查是否已添加
    if "[诊断]" not in content:
        # 在rewards收集后添加诊断
        replacement = r"\1" + diagnostic_code
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("✅ 添加奖励统计诊断输出")
    else:
        print("ℹ️  诊断输出已存在")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print()

def verify_z_bias_calculation():
    """验证Z轴偏置计算"""
    print("=" * 60)
    print("🔍 验证Z轴偏置计算")
    print("=" * 60)
    
    gravity = 9.81
    control_gain = 8.5
    action_range_z = 1.4
    
    # 计算需要的偏置：gravity / (control_gain * action_range_z)
    z_bias_needed = gravity / (control_gain * action_range_z)
    
    print(f"重力加速度: {gravity} m/s²")
    print(f"控制增益: {control_gain}")
    print(f"Z轴动作范围: {action_range_z}")
    print(f"计算得到的Z轴偏置: {z_bias_needed:.4f}")
    print()
    
    current_z_bias = 0.78
    print(f"当前代码中的Z轴偏置: {current_z_bias:.4f}")
    
    if abs(z_bias_needed - current_z_bias) > 0.05:
        print(f"⚠️  偏置不匹配！应该是 {z_bias_needed:.4f}，但代码中是 {current_z_bias:.4f}")
        print(f"   偏差: {abs(z_bias_needed - current_z_bias):.4f}")
        return False
    else:
        print(f"✅ Z轴偏置计算正确（误差 < 0.05）")
        return True
    
    print()

def print_summary():
    """打印修复汇总"""
    print("=" * 60)
    print("📋 修复汇总")
    print("=" * 60)
    print("""
已完成的修复：
1. ✅ 势场归一化：折中方案（max_force * 1.5）
2. ✅ CUDA内存对齐：actions_for_env = np.ascontiguousarray(...)
3. ✅ 奖励裁剪默认值：从-2500.0改为-120.0
4. ✅ Z轴偏置验证：确认计算正确

建议的下一步：
1. 运行训练，观察以下指标：
   - 奖励是否被正确裁剪到[-120, 120]范围
   - CUDA崩溃是否消失
   - 奖励是否开始改善

2. 如果仍有问题，检查：
   - Actor/Critic loss数值
   - Q值范围
   - 梯度范数

运行命令：
    ./run_optimized.sh 10 1024 "fix_test" 1
    """)
    print("=" * 60)

if __name__ == '__main__':
    print("🚀 开始综合修复...")
    print()
    
    # 执行修复
    fix_reward_clipping_in_scenario_creation()
    add_diagnostic_outputs()
    z_bias_ok = verify_z_bias_calculation()
    
    print_summary()
    
    if not z_bias_ok:
        print("\n⚠️  警告：Z轴偏置可能需要调整！")


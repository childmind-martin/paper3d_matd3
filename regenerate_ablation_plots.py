#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成消融实验的对比图表

基于已保存的训练数据，重新生成所有可视化图表（使用英文标签）
"""

import sys
import subprocess
from pathlib import Path

def regenerate_action_pf_comparison():
    """重新生成 Action vs APF 对比实验的图表"""
    print("="*70)
    print("重新生成 Action vs APF 对比实验图表")
    print("="*70)
    
    script = Path("ablation_action_pf_comparison.py")
    if not script.exists():
        print(f"错误：找不到脚本 {script}")
        return False
    
    # 使用 --reuse 选项，复用已有的训练数据，只重新生成图表
    cmd = [
        "python3",
        str(script),
        "--reuse",  # 关键：复用已有数据，不重新训练
        "--quick-comparison",  # 快速对比模式（3个实验）
        "--output-dir", "ablation_action_pf_outputs",
        "--logs-root", "logs"
    ]
    
    print(f"\n运行命令: {' '.join(cmd)}")
    print("\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✓ Action vs APF 对比图表生成完成")
        print("输出目录: ablation_action_pf_outputs/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 生成失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 异常: {e}")
        return False


def regenerate_fixed_fr_comparison():
    """重新生成 Fixed FR 对比实验的图表"""
    print("\n" + "="*70)
    print("重新生成 Fixed FR 对比实验图表")
    print("="*70)
    
    script = Path("ablation_fixed_fr_comparison.py")
    if not script.exists():
        print(f"错误：找不到脚本 {script}")
        return False
    
    # 使用 --reuse 选项，复用已有的训练数据，只重新生成图表
    cmd = [
        "python3",
        str(script),
        "--reuse",  # 关键：复用已有数据，不重新训练
        "--output-dir", "ablation_fixed_fr_outputs",
        "--logs-root", "logs"
    ]
    
    print(f"\n运行命令: {' '.join(cmd)}")
    print("\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✓ Fixed FR 对比图表生成完成")
        print("输出目录: ablation_fixed_fr_outputs/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 生成失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 异常: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("重新生成所有消融实验图表（基于已保存数据）")
    print("="*70)
    print("\n说明：")
    print("- 使用 --reuse 选项，复用已有训练数据")
    print("- 只重新生成图表，不重新训练")
    print("- 所有标签使用英文，不显示方框")
    print("\n")
    
    success_count = 0
    total_count = 2
    
    # 1. 重新生成 Action vs APF 对比图表
    if regenerate_action_pf_comparison():
        success_count += 1
    
    # 2. 重新生成 Fixed FR 对比图表
    if regenerate_fixed_fr_comparison():
        success_count += 1
    
    # 总结
    print("\n" + "="*70)
    print(f"完成：{success_count}/{total_count} 个实验图表生成成功")
    print("="*70)
    
    if success_count == total_count:
        print("\n✓ 所有图表已重新生成（使用英文标签）")
        return 0
    else:
        print("\n⚠ 部分图表生成失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())

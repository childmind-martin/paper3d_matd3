#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看地形感知模式评估结果对比

用法:
    python3 view_terrain_sensing_results.py [评估结果目录]
    
示例:
    # 查看最新评估结果
    python3 view_terrain_sensing_results.py
    
    # 查看指定目录的结果
    python3 view_terrain_sensing_results.py evaluation_results/apf_traditional_local_local
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_evaluation_results(eval_dir: Path) -> Optional[Dict]:
    """加载评估结果"""
    results_file = eval_dir / "evaluation_results.json"
    if not results_file.exists():
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(results: Dict) -> Dict:
    """提取关键指标"""
    metrics = {
        "avg_reward": results.get("avg_reward", 0.0),
        "std_reward": results.get("std_reward", 0.0),
        "max_reward": results.get("max_reward", 0.0),
        "min_reward": results.get("min_reward", 0.0),
        "episodes": results.get("episodes", 0),
    }
    
    # 计算成功率
    all_rewards = results.get("all_rewards", [])
    episode_details = results.get("episode_details", [])
    
    if episode_details:
        successes = [ep.get("success", False) for ep in episode_details]
        metrics["success_rate"] = sum(successes) / len(successes) if successes else 0.0
        metrics["avg_steps"] = np.mean([ep.get("steps", 0) for ep in episode_details])
    else:
        metrics["success_rate"] = 0.0
        metrics["avg_steps"] = 0.0
    
    return metrics


def print_comparison_table(results_dict: Dict[str, Dict]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("地形感知模式评估结果对比")
    print("="*80)
    
    # 表头
    print(f"\n{'模式':<30} {'平均奖励':<15} {'成功率':<12} {'平均步数':<12} {'回合数':<8}")
    print("-"*80)
    
    # 按模式排序
    mode_order = ["local", "oracle_same_probes", "oracle_dense"]
    for mode in mode_order:
        for label, metrics in results_dict.items():
            if mode in label.lower():
                print(f"{metrics['name']:<30} "
                      f"{metrics['avg_reward']:>12.2f} ± {metrics['std_reward']:<6.2f} "
                      f"{metrics['success_rate']:>10.1%} "
                      f"{metrics['avg_steps']:>10.1f} "
                      f"{metrics['episodes']:>6}")
                break
    
    print("="*80)


def main():
    # 确定要查看的目录
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        # 查找最新的评估结果目录
        eval_base = Path("evaluation_results")
        if not eval_base.exists():
            print("❌ 错误: 找不到 evaluation_results 目录")
            print("\n使用方法:")
            print("  1. 先运行评估: python3 ablation_terrain_sensing.py --eval-only --trained-model-path <模型路径>")
            print("  2. 然后查看结果: python3 view_terrain_sensing_results.py")
            sys.exit(1)
        
        # 查找所有评估结果目录
        eval_dirs = [d for d in eval_base.iterdir() if d.is_dir()]
        if not eval_dirs:
            print("❌ 错误: evaluation_results 目录为空")
            print("\n请先运行评估:")
            print("  python3 ablation_terrain_sensing.py --eval-only --trained-model-path <模型路径>")
            sys.exit(1)
        
        # 使用最新的目录
        base_dir = eval_base
    
    # 查找所有评估结果
    results_dict = {}
    
    eval_base = Path("evaluation_results")
    if eval_base.exists():
        for eval_dir in eval_base.iterdir():
            if not eval_dir.is_dir():
                continue
            
            # 从目录名推断模式
            dir_name = eval_dir.name
            if "local" in dir_name and "oracle" not in dir_name:
                mode = "local"
                name = "可学习APF (Local感知)"
            elif "oracle_same" in dir_name:
                mode = "oracle_same_probes"
                name = "可学习APF (Oracle相同探测)"
            elif "oracle_dense" in dir_name:
                mode = "oracle_dense"
                name = "可学习APF (Oracle密集探测)"
            else:
                continue
            
            # 加载结果
            results = load_evaluation_results(eval_dir)
            if results:
                metrics = extract_metrics(results)
                metrics["name"] = name
                metrics["mode"] = mode
                metrics["dir"] = str(eval_dir)
                results_dict[dir_name] = metrics
    
    if not results_dict:
        print("❌ 错误: 未找到有效的评估结果")
        print(f"\n检查目录: {eval_base}")
        print("\n请先运行评估:")
        print("  python3 ablation_terrain_sensing.py --eval-only --trained-model-path <模型路径>")
        sys.exit(1)
    
    # 打印对比表格
    print_comparison_table(results_dict)
    
    # 打印详细结果
    print("\n详细结果:")
    print("-"*80)
    for label, metrics in results_dict.items():
        print(f"\n{metrics['name']} ({metrics['mode']}):")
        print(f"  目录: {metrics['dir']}")
        print(f"  平均奖励: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  最高奖励: {metrics['max_reward']:.2f}")
        print(f"  最低奖励: {metrics['min_reward']:.2f}")
        print(f"  成功率: {metrics['success_rate']:.1%}")
        print(f"  平均步数: {metrics['avg_steps']:.1f}")
        print(f"  评估回合数: {metrics['episodes']}")
    
    print("\n" + "="*80)
    print("提示: 查看详细结果文件:")
    for label, metrics in results_dict.items():
        print(f"  {metrics['name']}: {metrics['dir']}/evaluation_results.json")


if __name__ == "__main__":
    main()

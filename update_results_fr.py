#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新results.json文件，添加最佳回合和最后回合的FR值
根据schedule反推当时回合的FR值
"""

import os
import json
import sys
import argparse
from pathlib import Path


def calculate_fr_from_schedule(episode, total_episodes, schedule_str):
    """
    根据schedule字符串计算指定回合的FR值
    
    Args:
        episode: 回合编号（从0开始）
        total_episodes: 总回合数
        schedule_str: schedule字符串，格式如 "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10"
    
    Returns:
        FR值（float）
    """
    if not schedule_str or schedule_str.strip().upper() == 'DISABLED':
        # 如果没有schedule，返回固定值（需要从args中获取，这里先返回None）
        return None
    
    # 解析schedule字符串
    pairs = [p.strip() for p in schedule_str.split(',') if p.strip()]
    schedule_points = []
    for kv in pairs:
        if ':' not in kv:
            continue
        k, v = kv.split(':', 1)
        ks = k.strip().rstrip('%')
        try:
            kp = float(ks)
            if '%' in k:
                kp = kp / 100.0
            schedule_points.append((kp, float(v)))
        except Exception:
            pass
    
    if not schedule_points:
        return None
    
    # 排序
    schedule_points.sort(key=lambda x: x[0])
    
    # 计算当前回合的进度
    progress = (episode + 1) / max(1, total_episodes)
    progress = max(0.0, min(1.0, progress))
    
    # 根据进度计算FR值
    if progress <= schedule_points[0][0]:
        fr_value = schedule_points[0][1]
    elif progress >= schedule_points[-1][0]:
        fr_value = schedule_points[-1][1]
    else:
        # 线性插值
        left = schedule_points[0]
        right = schedule_points[-1]
        for idx in range(1, len(schedule_points)):
            if schedule_points[idx][0] >= progress:
                left = schedule_points[idx - 1]
                right = schedule_points[idx]
                break
        span = max(right[0] - left[0], 1e-6)
        t = (progress - left[0]) / span
        fr_value = left[1] + t * (right[1] - left[1])
    
    return float(fr_value)


def update_results_json(results_file, schedule_str=None, base_fr=None):
    """
    更新results.json文件，添加最佳回合和最后回合的FR值
    
    Args:
        results_file: results.json文件路径
        schedule_str: schedule字符串（如果results.json中没有）
        base_fr: 基础FR值（如果schedule被禁用）
    """
    if not os.path.exists(results_file):
        print(f"❌ 文件不存在: {results_file}")
        return False
    
    # 读取现有的results.json
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    # 检查是否已经有best_episode_force_ratio
    if 'best_episode_force_ratio' in results and results['best_episode_force_ratio'] is not None:
        print(f"✅ 文件已包含 best_episode_force_ratio: {results['best_episode_force_ratio']}")
        if 'last_episode_force_ratio' in results and results['last_episode_force_ratio'] is not None:
            print(f"✅ 文件已包含 last_episode_force_ratio: {results['last_episode_force_ratio']}")
            return True
    
    # 获取必要信息
    episodes = results.get('episodes', 0)
    best_episode = results.get('best_episode', 0)
    last_episode = episodes - 1
    
    if episodes == 0:
        print(f"❌ 无法确定回合数")
        return False
    
    # 获取schedule字符串
    if schedule_str is None:
        # 尝试从args中获取
        args = results.get('args', {})
        schedule_str = args.get('action_force_ratio_schedule_pct', None)
        if schedule_str is None or schedule_str == '':
            # 尝试从环境变量获取（默认schedule）
            schedule_str = os.getenv('ACTION_FORCE_RATIO_SCHEDULE_PCT', 
                                    '0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10')
    
    # 获取基础FR值
    if base_fr is None:
        args = results.get('args', {})
        base_fr = args.get('action_force_ratio', 0.0)
    
    # 计算最佳回合的FR值
    if schedule_str and schedule_str.strip().upper() != 'DISABLED':
        best_fr = calculate_fr_from_schedule(best_episode, episodes, schedule_str)
        last_fr = calculate_fr_from_schedule(last_episode, episodes, schedule_str)
    else:
        # 如果没有schedule，使用固定值
        best_fr = base_fr
        last_fr = base_fr
    
    if best_fr is None:
        best_fr = base_fr
    if last_fr is None:
        last_fr = base_fr
    
    # 更新results
    results['best_episode_force_ratio'] = best_fr
    results['last_episode_force_ratio'] = last_fr
    
    # 保存更新后的文件
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功更新 {results_file}")
        print(f"   - 最佳回合 ({best_episode + 1}/{episodes}): FR = {best_fr:.4f}")
        print(f"   - 最后回合 ({last_episode + 1}/{episodes}): FR = {last_fr:.4f}")
        return True
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return False


def find_results_files(logs_dir):
    """查找所有results.json文件"""
    results_files = []
    for root, dirs, files in os.walk(logs_dir):
        if 'results.json' in files:
            results_files.append(os.path.join(root, 'results.json'))
    return results_files


def main():
    parser = argparse.ArgumentParser(description='更新results.json文件，添加最佳回合和最后回合的FR值')
    parser.add_argument('results_file', type=str, nargs='?', 
                       help='results.json文件路径（如果未指定，将在logs目录中查找所有results.json）')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='logs目录路径（默认: logs）')
    parser.add_argument('--schedule', type=str, default=None,
                       help='FR schedule字符串（如果results.json中没有）')
    parser.add_argument('--base-fr', type=float, default=None,
                       help='基础FR值（如果schedule被禁用）')
    parser.add_argument('--all', action='store_true',
                       help='更新logs目录中的所有results.json文件')
    
    args = parser.parse_args()
    
    if args.all:
        # 更新所有results.json文件
        if not os.path.exists(args.logs_dir):
            print(f"❌ logs目录不存在: {args.logs_dir}")
            return 1
        
        results_files = find_results_files(args.logs_dir)
        if not results_files:
            print(f"❌ 在 {args.logs_dir} 中未找到results.json文件")
            return 1
        
        print(f"📁 找到 {len(results_files)} 个results.json文件")
        success_count = 0
        for results_file in results_files:
            print(f"\n处理: {results_file}")
            if update_results_json(results_file, args.schedule, args.base_fr):
                success_count += 1
        
        print(f"\n✅ 成功更新 {success_count}/{len(results_files)} 个文件")
        return 0 if success_count == len(results_files) else 1
    else:
        # 更新单个文件
        if not args.results_file:
            print("❌ 请指定results.json文件路径，或使用 --all 更新所有文件")
            return 1
        
        if update_results_json(args.results_file, args.schedule, args.base_fr):
            return 0
        else:
            return 1


if __name__ == '__main__':
    sys.exit(main())

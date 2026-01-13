#!/usr/bin/env python3
"""
诊断脚本：检查环境变量在多进程中的传递
"""
import os
import multiprocessing as mp

def check_env_var(env_id):
    """检查单个进程中的环境变量"""
    start_alt = os.getenv('START_ALTITUDE_OFFSET', 'NOT_SET')
    goal_alt = os.getenv('GOAL_ALTITUDE', 'NOT_SET')
    print(f"Process {env_id}: START_ALTITUDE_OFFSET={start_alt}, GOAL_ALTITUDE={goal_alt}")
    return (env_id, start_alt, goal_alt)

if __name__ == '__main__':
    print("=== 主进程环境变量 ===")
    print(f"START_ALTITUDE_OFFSET = {os.getenv('START_ALTITUDE_OFFSET', 'NOT_SET')}")
    print(f"GOAL_ALTITUDE = {os.getenv('GOAL_ALTITUDE', 'NOT_SET')}")
    
    print("\n=== 子进程环境变量（模拟并行环境） ===")
    with mp.Pool(3) as pool:
        results = pool.map(check_env_var, [0, 1, 2])
    
    print("\n=== 结果汇总 ===")
    for env_id, start_alt, goal_alt in results:
        print(f"Env {env_id}: START_ALTITUDE_OFFSET={start_alt}, GOAL_ALTITUDE={goal_alt}")


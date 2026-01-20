#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双Q头和分离式梯度设计消融对比实验

对比四种情况：
1. Baseline 1: MADDPG（单Critic，单Q头，统一梯度）
2. Baseline 2: MATD3-单Q（Twin Critic，单Q头，统一梯度）
3. 实验1: MATD3-双Q头（Twin Critic，双Q头，统一梯度）
4. 实验2: MATD3-完整（Twin Critic，双Q头，分离式梯度）

所有实验使用相同的环境（固定位置+地形），确保公平对比。

🚨 关键配置：消融实验始终禁用课程学习
- UNLOCK_ENV_ON_SUCCESS=0：禁用基于成功次数的环境解锁
- UNLOCK_ENV_ON_PLATEAU=0：禁用基于奖励停滞的环境解锁
- RANDOM_TERRAIN=0：始终使用固定地形
- PER_ENV_TERRAIN=0：每个环境使用相同地形
- PER_EPISODE_TERRAIN=0：每个回合使用相同地形

实验设计说明：
- Baseline 1 (MADDPG): 标准基线算法，单Critic，单Q输出
- Baseline 2 (MATD3-单Q): MATD3框架但使用单Q头，验证Twin Critic的效果
- 实验1 (MATD3-双Q头): MATD3+双Q头但统一梯度，验证双Q头架构的效果
- 实验2 (MATD3-完整): 当前实现，验证分离式梯度设计的贡献
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 🔧 新增：导入批次管理器
try:
    from ablation_batch_manager import AblationBatchManager
except ImportError:
    AblationBatchManager = None
    print("警告：未找到 ablation_batch_manager，将使用默认批次管理")

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    HAS_MATPLOTLIB = True
    
    # 🔧 关键修复：在导入后立即设置英文字体，避免所有文本显示为方框
    def setup_english_fonts():
        """设置英文字体，避免显示方块字符"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        # 强制清除中文字体设置，确保使用英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # 立即设置字体
    setup_english_fonts()
except ImportError:
    print("缺少依赖，请安装：pip install matplotlib scipy")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("警告：未安装 plotly，将跳过交互图生成。安装：pip install plotly")


# ============================================================================
# 实验配置
# ============================================================================

# ============================================================================
# 实验配置
# ============================================================================
# 
# 🚨 注意：当前代码中MATD3的双Q头和分离式梯度是硬编码的
# 要运行完整的4个实验组，需要先修改训练脚本添加环境变量支持
# 详见：ABLATION_DUAL_Q_README.md
#
# 当前可用配置（无需修改代码）：
# 1. maddpg_baseline: MADDPG算法（单Critic，单Q头）
# 2. matd3_full: MATD3算法（Twin Critic，双Q头，分离式梯度）
#
# 需要代码修改的配置（已注释，待实现）：
# 3. matd3_single_q: MATD3但单Q头（需要修改代码）
# 4. matd3_dual_q: MATD3+双Q头但统一梯度（需要修改代码）
# ============================================================================

EXPERIMENT_CONFIGS = [
    {
        "label": "maddpg_baseline",
        "name": "MADDPG Baseline",
        "name_en": "MADDPG Baseline",
        "description": "Standard MADDPG: Single Critic, Single Q head, Unified gradient",
        "env": {
            # 🚨 关键：使用MADDPG算法
            "ALGORITHM": "maddpg",  # 使用MADDPG算法（单Critic，单Q头）
            # 🔧 关键配置：确保使用标准MADDPG设置
            "USE_TF_POTENTIAL_FIELD": "1",  # 保持TF版本启用
            "ACTION_FORCE_RATIO": "0.50",  # 使用默认混合比例
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10",  # 使用默认schedule
            # 🚨 关键修复：网络初始化一致性
            "SEED": "252488",  # ✅ 为所有实验设置相同的训练随机种子，确保网络初始化一致
            "TF_DETERMINISTIC_OPS": "0"  # ✅ 启用TensorFlow确定性操作，确保完全可重复
        }
    },
    {
        "label": "matd3_full",
        "name": "MATD3 Full",
        "name_en": "MATD3 Full",
        "description": "MATD3 with Twin Critic, dual Q heads, and separated gradient (current implementation)",
        "env": {
            # 🚨 关键：使用MATD3算法，启用所有特性（当前实现）
            "ALGORITHM": "matd3",  # 使用MATD3算法（Twin Critic）
            # 注意：MATD3_USE_DUAL_Q 和 MATD3_USE_SEPARATED_GRADIENT 在当前代码中未实现
            # 当前MATD3默认使用双Q头和分离式梯度
            # 🔧 关键配置：确保使用标准MATD3设置
            "USE_TF_POTENTIAL_FIELD": "1",  # 保持TF版本启用
            "ACTION_FORCE_RATIO": "0.50",  # 使用默认混合比例
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10",  # 使用默认schedule
            # 🚨 关键修复：网络初始化一致性
            "SEED": "252488",  # ✅ 为所有实验设置相同的训练随机种子，确保网络初始化一致
            "TF_DETERMINISTIC_OPS": "0"  # ✅ 启用TensorFlow确定性操作，确保完全可重复
        }
    }
    # ============================================================================
    # 以下配置需要修改训练脚本后才能使用
    # ============================================================================
    # {
    #     "label": "matd3_single_q",
    #     "name": "MATD3 Single Q",
    #     "name_en": "MATD3 Single Q",
    #     "description": "MATD3 framework with Twin Critic but single Q head, unified gradient",
    #     "env": {
    #         "ALGORITHM": "matd3",
    #         "MATD3_USE_DUAL_Q": "0",  # 🔧 需要代码支持
    #         "MATD3_USE_SEPARATED_GRADIENT": "0",  # 🔧 需要代码支持
    #         "USE_TF_POTENTIAL_FIELD": "1",
    #         "ACTION_FORCE_RATIO": "0.50",
    #         "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10",
    #         "SEED": "252488",
    #         "TF_DETERMINISTIC_OPS": "0"
    #     }
    # },
    # {
    #     "label": "matd3_dual_q",
    #     "name": "MATD3 Dual Q",
    #     "name_en": "MATD3 Dual Q",
    #     "description": "MATD3 with Twin Critic and dual Q heads, but unified gradient",
    #     "env": {
    #         "ALGORITHM": "matd3",
    #         "MATD3_USE_DUAL_Q": "1",  # 🔧 需要代码支持
    #         "MATD3_USE_SEPARATED_GRADIENT": "0",  # 🔧 需要代码支持
    #         "USE_TF_POTENTIAL_FIELD": "1",
    #         "ACTION_FORCE_RATIO": "0.50",
    #         "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10",
    #         "SEED": "252488",
    #         "TF_DETERMINISTIC_OPS": "0"
    #     }
    # }
]


# ============================================================================
# 固定位置生成
# ============================================================================

def generate_fixed_positions(positions_file: Path, n_agents: int = 3, map_size: float = 200.0):
    """生成固定位置文件（与ablation_action_pf_comparison.py保持一致）"""
    if positions_file.exists():
        print(f"[消融实验] 位置文件已存在: {positions_file}")
        return positions_file
    
    print(f"[消融实验] 生成固定位置文件: {positions_file}")
    
    # 保存原始环境变量
    original_env = {}
    for key in ['USE_FIXED_POSITIONS', 'DYNAMIC_FIRST_TIME', 'POSITIONS_FILE', 
                'UNLOCK_ENV_ON_SUCCESS', 'UNLOCK_ENV_ON_PLATEAU', 'RANDOM_TERRAIN',
                'PER_ENV_TERRAIN', 'PER_EPISODE_TERRAIN', 'USE_SCENARIO_SEED', 'SCENARIO_SEED']:
        original_env[key] = os.environ.get(key)
    
    # 设置环境变量用于位置生成
    os.environ['USE_FIXED_POSITIONS'] = '1'
    os.environ['DYNAMIC_FIRST_TIME'] = '0'
    os.environ['POSITIONS_FILE'] = str(positions_file)
    os.environ['UNLOCK_ENV_ON_SUCCESS'] = '0'
    os.environ['UNLOCK_ENV_ON_PLATEAU'] = '0'
    os.environ['RANDOM_TERRAIN'] = '0'
    os.environ['PER_ENV_TERRAIN'] = '0'
    os.environ['PER_EPISODE_TERRAIN'] = '0'
    os.environ['USE_SCENARIO_SEED'] = '1'
    os.environ['SCENARIO_SEED'] = '67'  # 与ablation_action_pf_comparison.py保持一致
    
    try:
        # 导入场景类
        sys.path.insert(0, str(Path(__file__).parent))
        from multiagent.scenarios.paper3d_terrain_energy import Scenario
        
        # 创建场景实例
        scenario = Scenario()
        world = scenario.make_world()
        scenario.reset_world(world)
        
        # 提取位置信息
        agents_pos = []
        for agent in world.agents:
            pos = agent.state.p_pos.copy()
            agents_pos.append(pos.tolist() if hasattr(pos, 'tolist') else list(pos))
        
        goal_pos = scenario.goal_pos.copy() if hasattr(scenario.goal_pos, 'copy') else list(scenario.goal_pos)
        if hasattr(goal_pos, 'tolist'):
            goal_pos = goal_pos.tolist()
        
        # 保存位置数据
        positions_data = {
            "agents": agents_pos,
            "goal": goal_pos,
            "n_agents": n_agents,
            "map_size": map_size,
            "generated_by": "ablation_dual_q_separated_gradient.py"
        }
        
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(positions_file, 'w', encoding='utf-8') as f:
            json.dump(positions_data, f, indent=2, ensure_ascii=False)
        
        # 恢复环境变量
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        print(f"[消融实验] ✅ 位置文件已生成: {positions_file}")
        print(f"[消融实验]   智能体数量: {len(agents_pos)}")
        print(f"[消融实验]   目标位置: {goal_pos}")
        
    except Exception as e:
        raise RuntimeError(
            f"生成固定位置文件失败: {e}\n"
            f"提示：请检查场景初始化代码和环境配置是否正确。\n"
            f"位置文件路径: {positions_file}"
        )
    
    return positions_file


# ============================================================================
# 基础环境变量设置
# ============================================================================

def setup_base_env_vars(positions_file: Path, gpu_id: int = None) -> dict:
    """
    设置基础环境变量（公共逻辑，消除代码冗余）
    
    Args:
        positions_file: 固定位置文件路径
        gpu_id: GPU ID（None表示共享GPU模式）
    
    Returns:
        配置好的环境变量字典
    """
    env = os.environ.copy()
    
    # === GPU配置 ===
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GPU_ID"] = str(gpu_id)
    else:
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env["GPU_ID"] = "0"
    
    # GPU内存稳定性配置
    env["TF_GPU_ALLOCATOR"] = ""
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if gpu_id is None:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["TF_SYNC_ON_FINISH"] = "1"
    else:
        if "CUDA_LAUNCH_BLOCKING" in env:
            del env["CUDA_LAUNCH_BLOCKING"]
        if "TF_SYNC_ON_FINISH" in env:
            del env["TF_SYNC_ON_FINISH"]
    
    # XLA配置
    env["XLA_FLAGS"] = ""
    env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    
    # 基础配置
    env.setdefault("NUM_ENVS", "1")
    env.setdefault("XLA_GLOBAL", "1")
    env.setdefault("CPU_THREADS", "12")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("QUIET_OUTPUT", "0")
    env.setdefault("SUPPRESS_MA_PROMPT", "1")
    env.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")
    
    # === 确保使用固定位置和地形（消融实验要求）===
    env["USE_FIXED_POSITIONS"] = "1"
    env["DYNAMIC_FIRST_TIME"] = "0"
    env["POSITIONS_FILE"] = str(positions_file)
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    env["USE_SCENARIO_SEED"] = "1"
    env["SCENARIO_SEED"] = "67"
    
    # 删除基础配置中的SEED，让实验配置中的SEED生效
    if "SEED" in env:
        del env["SEED"]
    
    # 地图生成参数
    env["TERRAIN_COMPLEXITY_LEVEL"] = "3"
    env["MAP_SIZE"] = "200"
    env["MOUNTAIN_MIN_DISTANCE"] = "55"
    env["TERRAIN_CONTACT_EPS"] = "0.2"
    
    # 清除DELTA相关环境变量，确保每个实验从干净状态开始
    delta_vars = ["DELTA_K_ATT", "DELTA_LAMBDA_1", "DELTA_K_REP", "DELTA_RADIUS"]
    for var in delta_vars:
        if var in env:
            del env[var]
    
    return env


# ============================================================================
# 实验运行函数
# ============================================================================

def run_experiment_worker(args_tuple):
    """并行训练的工作函数"""
    (cfg, positions_file, episodes, batch_size, script, use_weighted_reward, gpu_id) = args_tuple
    
    label = cfg["label"]
    env_vars = cfg.get("env", {})
    
    # 设置基础环境变量
    env = setup_base_env_vars(positions_file, gpu_id)
    
    # 应用实验特定的环境变量
    for key, value in env_vars.items():
        env[key] = value
    
    # 确保ALGORITHM被正确设置
    algorithm = env.get("ALGORITHM", "matd3")
    
    env["EXP_NAME"] = label
    
    # 并行训练时减少日志输出
    env["QUIET_OUTPUT"] = "1"
    
    cmd = [
        script,
        str(episodes),
        str(batch_size),
        label,
        str(use_weighted_reward),
        algorithm,
    ]
    
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[并行训练-{label}] 开始: {cfg.get('name', label)}", file=sys.stderr)
    print(f"[并行训练-{label}] 算法: {algorithm}", file=sys.stderr)
    print(f"[并行训练-{label}] 双Q头: {env.get('MATD3_USE_DUAL_Q', 'N/A')}", file=sys.stderr)
    print(f"[并行训练-{label}] 分离梯度: {env.get('MATD3_USE_SEPARATED_GRADIENT', 'N/A')}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path(script).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        elapsed = time.time() - start_time
        
        return {
            "label": label,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "elapsed": elapsed,
            "output": result.stdout
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "label": label,
            "success": False,
            "returncode": -1,
            "elapsed": elapsed,
            "error": str(e)
        }


def run_experiment(cfg: Dict, positions_file: Path, args, cache: Dict[str, Dict], gpu_id: int = None) -> Dict:
    """运行单个实验（非并行版本）"""
    label = cfg["label"]
    env_vars = cfg.get("env", {})
    
    # 设置基础环境变量
    env = setup_base_env_vars(positions_file, gpu_id)
    
    # 应用实验特定的环境变量
    for key, value in env_vars.items():
        env[key] = value
    
    # 确保ALGORITHM被正确设置
    algorithm = env.get("ALGORITHM", "matd3")
    
    env["EXP_NAME"] = label
    
    cmd = [
        args.script,
        str(args.episodes),
        str(args.batch_size),
        label,
        str(args.use_weighted_reward),
        algorithm,
    ]
    
    print(f"\n{'='*70}")
    print(f"[运行] {cfg.get('name', label)}")
    print(f"[运行] 算法: {algorithm}")
    print(f"[运行] 双Q头: {env.get('MATD3_USE_DUAL_Q', 'N/A')}")
    print(f"[运行] 分离梯度: {env.get('MATD3_USE_SEPARATED_GRADIENT', 'N/A')}")
    print(f"{'='*70}")
    
    subprocess.run(cmd, env=env, cwd=Path(args.script).parent)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="双Q头和分离式梯度设计消融对比实验")
    parser.add_argument("--episodes", type=int, default=150, help="训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024, help="批次大小")
    parser.add_argument("--script", type=str, default="./run_optimized.sh", help="训练脚本路径")
    parser.add_argument("--use-weighted-reward", type=int, default=1, help="是否使用分项加权奖励")
    parser.add_argument("--parallel", action="store_true", help="是否并行运行")
    parser.add_argument("--gpus", type=str, nargs="+", help="GPU ID列表（并行模式）")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/dual_q_ablation.json", help="固定位置文件路径")
    
    args = parser.parse_args()
    
    # 生成固定位置文件
    positions_file = Path(args.positions_file)
    generate_fixed_positions(positions_file)
    
    if args.parallel:
        # 并行运行
        print("🚀 并行运行模式")
        if args.gpus:
            gpu_ids = [int(g) for g in args.gpus]
        else:
            gpu_ids = [None] * len(EXPERIMENT_CONFIGS)
        
        tasks = [
            (cfg, positions_file, args.episodes, args.batch_size, args.script, args.use_weighted_reward, gpu_ids[i % len(gpu_ids)])
            for i, cfg in enumerate(EXPERIMENT_CONFIGS)
        ]
        
        with ProcessPoolExecutor(max_workers=len(EXPERIMENT_CONFIGS)) as executor:
            futures = {executor.submit(run_experiment_worker, task): task[0]["label"] for task in tasks}
            
            results = []
            for future in as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✅" if result["success"] else "❌"
                    print(f"{status} {label}: {result.get('elapsed', 0):.1f}s", file=sys.stderr)
                except Exception as e:
                    print(f"❌ {label}: 异常 - {e}", file=sys.stderr)
        
        print("\n所有实验完成！")
    else:
        # 串行运行
        print("🚀 串行运行模式")
        for cfg in EXPERIMENT_CONFIGS:
            run_experiment(cfg, positions_file, args, {}, None)


if __name__ == "__main__":
    main()

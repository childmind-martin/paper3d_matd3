#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形感知模式消融对比实验

对比不同地形感知模式对APF性能的影响：
1. apf_traditional/local: 使用观测中的地形信息（当前实现）
2. apf_traditional/oracle_same_probes: 使用Oracle接口获取真值，probe布局与local一致
3. apf_traditional/oracle_dense: 使用Oracle接口，提升探测密度（可选）
4. action_apf_fusion: 网络动作+可学习势场修正（作为对比基线）

关键特性：
- terrain_sensing_mode ∈ {local, oracle_same_probes, oracle_dense}
- Oracle模式只用于APF地形力计算，不注入到RL训练观测中
- 所有实验使用相同的随机种子和地图集合
- 记录指标：SR_team、穿透/碰撞次数、d_min分位数、P(d_min≤δ)、回报曲线
"""

import argparse
import json
import os
import subprocess
import sys
import re  # 🚨 关键修复：在模块级别导入re，确保所有函数都可以访问
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 🔧 抑制 TensorFlow/XLA 警告（如 cuFFT 注册警告）
# 必须在导入 TensorFlow 之前设置
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 抑制警告，保留错误信息

# 导入基础消融实验模块
from ablation_action_pf_comparison import (
    ensure_fixed_positions,
    setup_base_env_vars,
    find_latest_log_dir,
    load_metrics,
    AblationBatchManager,
    plot_comparison_rewards,
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance,
    plot_comparison_losses,
    generate_interactive_comparison,
    setup_english_fonts,
    TERRAIN_COMPLEXITY_LEVEL,
    MAP_SIZE,
    MOUNTAIN_MIN_DISTANCE,
    SCENARIO_SEED
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    HAS_MATPLOTLIB = True
except ImportError:
    print("缺少依赖，请安装：pip install matplotlib scipy")
    sys.exit(1)


# ============================================================================
# 🔧 配置区域：在这里修改模型路径和其他参数
# ============================================================================

# 🔧 默认模型路径（如果使用 --eval-only 模式且未指定 --trained-model-path，将使用此路径）
# 可以直接在这里修改为你的模型路径
DEFAULT_MODEL_PATH = "models/apf_learnable_20260111_222207/best"  # 修改这里设置默认模型路径

# 支持的格式：
# - "models/apf_learnable/best"           # 最佳模型（推荐）
# - "models/apf_learnable/final"          # 最终模型
# - "models/apf_learnable_20260114_165908/best"  # 带时间戳的模型
# - "models/apf_learnable/checkpoint"     # 检查点目录

# ============================================================================
# 训练配置
# ============================================================================

# 🔧 修正：只训练一次（使用local模式），然后在评估时分别用local和oracle模式评估
# 训练配置（只训练一次）- 使用apf_learnable，因为需要训练网络学习势场参数
TRAINING_CONFIG = {
    "label": "apf_learnable",
    "name": "可学习APF（训练）",
    "name_en": "Learnable APF (Training)",
    "description": "使用local模式训练可学习APF，然后在评估时分别用local和oracle模式评估",
    "env": {
        "ACTION_FORCE_RATIO": "1.0",
        "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",  # 🔧 修复：使用DISABLED而不是空字符串
        "USE_TF_POTENTIAL_FIELD": "1",
        "TERRAIN_SENSING_MODE": "local",  # 训练时使用local模式
        
        # 🔧 新增：显式设置DELTA参数，与run_optimized.sh完全一致
        # 这些参数允许网络学习势场参数，会在训练过程中通过Actor网络学习优化
        "DELTA_K_ATT": "5.0",           # ✅ 与run_optimized.sh一致（默认5.0）
        "DELTA_LAMBDA_1": "2.2",        # ✅ 与run_optimized.sh一致（默认2.2）
        "DELTA_K_REP": "1200.0",        # ✅ 修复：1200.0（不是1000.0），与run_optimized.sh一致
        "DELTA_RADIUS": "80.0",         # ✅ 与run_optimized.sh一致（默认80.0）
        
        # 🔧 新增：显式设置BASE参数，确保与run_optimized.sh一致
        "GOAL_ATTRACTION": "6.0",                    # ✅ 与run_optimized.sh一致（默认6.0）
        "LAMBDA_1_BASE": "8.5",                      # ✅ 与run_optimized.sh一致（默认8.5）
        "TERRAIN_REPULSION": "8000.0",               # ✅ 与run_optimized.sh一致（默认8000.0）
        "AGENT_INFLUENCE_RANGE": "150.0",           # ✅ 与run_optimized.sh一致（默认150.0）
        
        # 🔧 新增：显式设置地形参数，确保与run_optimized.sh一致
        "TERRAIN_COMPLEXITY_LEVEL": str(TERRAIN_COMPLEXITY_LEVEL),  # ✅ 与run_optimized.sh一致
        "MAP_SIZE": str(MAP_SIZE),                                 # ✅ 与run_optimized.sh一致
        "MOUNTAIN_MIN_DISTANCE": str(MOUNTAIN_MIN_DISTANCE),       # ✅ 与run_optimized.sh一致
    }
}

# 评估配置（评估时使用不同的terrain_sensing_mode）
EVALUATION_CONFIGS = [
    {
        "label": "apf_learnable_local",
        "name": "可学习APF (Local感知)",
        "name_en": "Learnable APF (Local Sensing)",
        "description": "使用观测中的地形信息计算APF地形斥力",
        "terrain_sensing_mode": "local",
    },
    {
        "label": "apf_learnable_oracle_same",
        "name": "可学习APF (Oracle相同探测)",
        "name_en": "Learnable APF (Oracle Same Probes)",
        "description": "使用Oracle接口获取真值，probe布局与local一致",
        "terrain_sensing_mode": "oracle_same_probes",
    },
    {
        "label": "apf_learnable_oracle_dense",
        "name": "可学习APF (Oracle密集探测)",
        "name_en": "Learnable APF (Oracle Dense Probes)",
        "description": "使用Oracle接口，提升探测密度作为上界",
        "terrain_sensing_mode": "oracle_dense",
    },
]

def parse_args():
    parser = argparse.ArgumentParser(description="地形感知模式消融对比实验、测试步长积分对于apf是否穿透的影响")
    parser.add_argument("--script", type=str, default="./run_optimized.sh",
                        help="训练启动脚本路径 (默认 ./run_optimized.sh)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="每个实验的训练回合数（默认120）")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="训练批次大小")
    parser.add_argument("--use-weighted-reward", type=int, default=1, choices=[0, 1],
                        help="是否使用分项加权奖励")
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3"],
                        help="训练算法选择")
    parser.add_argument("--output-dir", type=str, default="terrain_sensing_outputs",
                        help="图表输出目录")
    parser.add_argument("--logs-root", type=str, default="logs",
                        help="训练日志根目录")
    parser.add_argument("--positions-file", type=str, default=None,
                        help="固定位置文件路径")
    parser.add_argument("--reuse", action="store_true",
                        help="若检测到同名实验已存在，则跳过重新训练，直接复用最新日志")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        choices=["apf_traditional_local", "apf_traditional_oracle_same", 
                                "apf_traditional_oracle_dense", "action_apf_fusion"],
                        help="选择要运行的实验")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅运行评估，不进行训练")
    parser.add_argument("--trained-model-path", type=str, default=None,
                        help=f"训练好的模型路径（用于--eval-only模式，默认: {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="评估回合数（默认20）")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="评估随机种子（默认42）")
    parser.add_argument("--eval-episode-length", type=int, default=2800,
                        help="评估回合长度（步数，默认2800，应与训练时一致）")
    return parser.parse_args()


def run_experiment(cfg: Dict, positions_file: Path, args, gpu_id: Optional[int] = None) -> Dict:
    """运行单个实验配置"""
    label = cfg["label"]
    
    # 设置基础环境变量
    env = setup_base_env_vars(positions_file, gpu_id if gpu_id is not None else 0)
    
    # 应用实验特定配置
    env.update(cfg.get("env", {}))
    
    # 🚨 关键：消融实验始终禁用课程学习
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    # 🔧 地形生成模式设置
    # 如果希望每个评估回合使用不同的地图，设置 PER_EPISODE_TERRAIN=1
    # 如果希望所有回合使用同一个地图（当前默认），保持 PER_EPISODE_TERRAIN=0
    env["RANDOM_TERRAIN"] = "0"  # 不使用完全随机地形
    env["PER_ENV_TERRAIN"] = "0"  # 每个环境使用相同地形（评估时只有一个环境）
    env["PER_EPISODE_TERRAIN"] = "1"  # 🔧 修改：每个回合使用不同的地图（更全面的评估）
    
    env["EXP_NAME"] = label
    
    # 日志控制
    env["QUIET_OUTPUT"] = "0"  # 启用详细输出
    env["TQDM_DISABLE"] = "0"
    env["TQDM_TO_STDOUT"] = "1"
    
    cmd = [
        args.script,
        str(args.episodes),
        str(args.batch_size),
        label,
        str(args.use_weighted_reward),
        args.algorithm,
    ]
    
    print(f"\n{'='*70}")
    print(f"[实验-{label}] 开始: {cfg.get('name', label)}")
    print(f"[实验-{label}] 地形感知模式: {env.get('TERRAIN_SENSING_MODE', 'local')}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True, env=env)
        log_dir = find_latest_log_dir(label, args.logs_root)
        metrics = load_metrics(log_dir)
        print(f"[实验-{label}] 完成: {cfg.get('name', label)}")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": log_dir,
            "metrics": metrics,
            "success": True
        }
    except subprocess.CalledProcessError as e:
        print(f"[实验-{label}] 训练失败: {cfg.get('name', label)}, 错误码: {e.returncode}")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": None,
            "metrics": {},
            "success": False
        }


def run_training(cfg: Dict, positions_file: Path, args, gpu_id: Optional[int] = None) -> Dict:
    """训练入口（保持与历史调用兼容）"""
    return run_experiment(cfg, positions_file, args, gpu_id=gpu_id)


def _run_single_evaluation_worker(args_tuple):
    """运行单个评估配置的worker函数（用于多进程）"""
    (eval_cfg, trained_model_path, positions_file, episodes, seed, 
     episode_length, episode_positions_dir, terrain_seeds, batch_dir) = args_tuple
    
    return _run_single_evaluation(
        eval_cfg, trained_model_path, positions_file, episodes, seed,
        episode_length, episode_positions_dir, terrain_seeds, batch_dir
    )


def _run_single_evaluation(eval_cfg, trained_model_path, positions_file, episodes, seed, 
                          episode_length, episode_positions_dir, terrain_seeds, batch_dir):
    """运行单个评估配置（用于并行执行）"""
    # 转换字符串参数为Path对象
    trained_model_path = Path(trained_model_path)
    positions_file = Path(positions_file)
    episode_positions_dir = Path(episode_positions_dir)
    batch_dir = Path(batch_dir) if batch_dir is not None else None
    
    label = eval_cfg["label"]
    terrain_sensing_mode = eval_cfg["terrain_sensing_mode"]
    
    print(f"\n{'='*70}")
    print(f"[评估-{label}] 开始评估: {eval_cfg.get('name', label)}")
    print(f"[评估-{label}] 地形感知模式: {terrain_sensing_mode}")
    print(f"[评估-{label}] 模型路径: {trained_model_path}")
    print(f"[评估-{label}] 评估回合数: {episodes}")
    print(f"{'='*70}\n")
    
    # 设置评估环境变量
    env = os.environ.copy()
    env.update(setup_base_env_vars(positions_file, 0))  # 使用默认GPU ID
    # 🚨 关键修复：使用episode位置文件目录，确保所有评估模式使用相同的初始条件
    env["EPISODE_POSITIONS_DIR"] = str(episode_positions_dir)
    env["USE_FIXED_POSITIONS"] = "1"
    env["SEED"] = str(seed)
    env["QUIET_OUTPUT"] = "0"  # 🔧 修复：改为0，显示输出，方便调试
    env["TQDM_DISABLE"] = "0"
    env["TQDM_TO_STDOUT"] = "1"
    # 🚨 关键优化：大幅降低进度条更新频率，避免输出混乱
    env["TQDM_MININTERVAL"] = "5.0"  # 至少5秒更新一次进度条（大幅降低）
    env["TQDM_MINITERS"] = "200"  # 至少200步更新一次（大幅降低）
    env["EVAL_DEBUG_ACTION_STEPS"] = "0"
    env["EVAL_DISABLE_VISUALIZATION"] = "1"
    env["DISABLE_TRAJECTORY_RECORDING"] = "1"
    env["NOISE_SCALE"] = "0.0"
    env["RANDOM_ACTION_PROB"] = "0.0"
    env["RANDOM_ACTION_PROB_TRAINING"] = "0.0"
    env["TERRAIN_SENSING_MODE"] = terrain_sensing_mode
    env["TERRAIN_COMPLEXITY_LEVEL"] = str(TERRAIN_COMPLEXITY_LEVEL)
    env["MAP_SIZE"] = str(MAP_SIZE)
    env["MOUNTAIN_MIN_DISTANCE"] = str(MOUNTAIN_MIN_DISTANCE)
    
    # 势场参数
    env["GOAL_ATTRACTION"] = "6.0"
    env["LAMBDA_1_BASE"] = "8.5"
    env["TERRAIN_REPULSION"] = "8000.0"
    env["AGENT_INFLUENCE_RANGE"] = "150.0"
    env["DELTA_K_ATT"] = "5.0"
    env["DELTA_LAMBDA_1"] = "2.2"
    env["DELTA_K_REP"] = "1200.0"
    env["DELTA_RADIUS"] = "80.0"
    
    env["EPISODE_LENGTH"] = str(episode_length)
    env["DISABLE_GIF"] = "1"
    env["TERRAIN_SEED_SEQUENCE"] = ",".join(map(str, terrain_seeds))
    env["SAVE_BEST_TRAJ"] = "1"
    env["SAVE_INTERACTIVE_TRAJ"] = "0"
    env["DISABLE_TRAJECTORY_RECORDING"] = "0"
    
    env["TERRAIN_CONTACT_EPS"] = "0.2"
    env["COLLISION_DISTANCE_THRESHOLD"] = "0.5"
    env["COLLISION_PENALTY_VALUE"] = "60.0"
    env["SIMULATION_DT"] = "0.08"
    
    env["ACTION_FORCE_RATIO"] = "1.0"
    env["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = "DISABLED"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # 评估结果保存路径
    if batch_dir is not None:
        batch_path = Path(batch_dir)
        eval_save_path = str(batch_path / "evaluation_results" / f"{label}_{terrain_sensing_mode}")
        Path(eval_save_path).mkdir(parents=True, exist_ok=True)
    else:
        eval_save_path = f"evaluation_results/{label}_{terrain_sensing_mode}"
    
    # 运行评估
    eval_cmd = [
        "./run_evaluation.sh",
        str(trained_model_path),
        str(episodes),
        eval_save_path,
        str(positions_file),
        "1",  # use_fixed_positions
        "true",  # disable_early_termination
    ]
    
    # 🚨 关键修复：为每个评估进程创建独立的日志文件，支持并行执行 + 实时查看
    log_dir = Path(eval_save_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{label}_{terrain_sensing_mode}.log"
    
    try:
        print(f"[评估-{label}] 🚀 开始执行评估命令: {' '.join(eval_cmd)}")
        print(f"[评估-{label}] 📝 日志文件: {log_file}")
        
        # 使用Popen启动进程，将输出重定向到日志文件
        with open(log_file, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(
                eval_cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,  # 将stderr也重定向到stdout
                text=True,
                bufsize=1  # 行缓冲，确保实时写入
            )
            
            # 等待进程完成
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, eval_cmd)
        
        # 验证评估结果
        eval_results_json = Path(eval_save_path) / "evaluation_results.json"
        if eval_results_json.exists():
            with open(eval_results_json, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            actual_episodes = eval_data.get('episodes', 0)
            if actual_episodes != episodes:
                print(f"[评估-{label}] ⚠️  警告: 评估结果回合数不一致！期望: {episodes}, 实际: {actual_episodes}")
            else:
                print(f"[评估-{label}] ✅ 验证通过: 评估结果回合数 = {actual_episodes}")
        
        return {
            "label": label,
            "success": True,
            "model_path": str(trained_model_path),
            "terrain_sensing_mode": terrain_sensing_mode,
            "eval_save_path": eval_save_path,
            "episodes": episodes,
        }
    except subprocess.CalledProcessError as e:
        print(f"[评估-{label}] ❌ 评估失败: {e}")
        # 注意：由于移除了capture_output，无法直接获取stderr
        # 错误信息会直接输出到stderr
        return {
            "label": label,
            "success": False,
            "error": str(e),
            "terrain_sensing_mode": terrain_sensing_mode,
        }
    except Exception as e:
        print(f"[评估-{label}] ❌ 评估异常: {e}")
        return {
            "label": label,
            "success": False,
            "error": str(e),
            "terrain_sensing_mode": terrain_sensing_mode,
        }


def evaluate_terrain_sensing(trained_model_path: Path, positions_file: Path, 
                            episodes: int = 20, seed: int = 42, episode_length: int = 4000, batch_dir: Optional[Path] = None) -> Dict:
    """
    🔧 修正：使用同一个训练好的模型，分别用local和oracle模式评估
    
    Args:
        trained_model_path: 训练好的模型路径（例如：models/apf_traditional/best）
        positions_file: 固定位置文件
        episodes: 评估回合数
        seed: 评估随机种子
    
    Returns:
        评估结果字典
    """
    results = {}
    
    # 🔧 关键：使用同一个训练好的模型，分别用不同的terrain_sensing_mode评估
    if not trained_model_path.exists():
        print(f"❌ 错误: 训练模型路径不存在: {trained_model_path}")
        return results
    
    # 🔧 关键修复：生成固定的地形种子序列，确保所有评估模式使用相同的地图顺序
    # 使用固定的随机数生成器，基于评估种子生成地形种子序列
    import random
    rng = random.Random(seed)  # 使用评估种子初始化随机数生成器
    terrain_seeds = [rng.randint(1000, 99999) for _ in range(episodes)]
    print(f"\n{'='*70}")
    print(f"🔧 评估配置一致性检查")
    print(f"{'='*70}")
    print(f"评估回合数: {episodes} (所有评估模式将使用相同的回合数)")
    print(f"评估随机种子: {seed} (所有评估模式将使用相同的种子)")
    print(f"地形种子序列: {terrain_seeds[:5]}... (共{episodes}个，前5个)")
    print(f"   所有评估模式将使用相同的地图顺序，确保公平对比")
    print(f"{'='*70}\n")
    
    # 🚨 关键修复：为每个episode生成固定的位置文件，确保所有评估模式使用相同的初始条件
    # 原因：每个episode使用不同地形，需要为每个地形生成对应的固定位置
    # 这样所有评估模式（Oracle Same、Oracle Dense、Local）都会使用相同的地形和初始位置
    if batch_dir is not None:
        episode_positions_dir: Path = batch_dir / "episode_positions"
    else:
        episode_positions_dir: Path = Path("evaluation_results") / "episode_positions"
    episode_positions_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已经生成了位置文件
    existing_positions = list(episode_positions_dir.glob("episode_*.json"))
    if len(existing_positions) < episodes:
        print(f"\n{'='*70}")
        print(f"🔧 为每个episode生成固定的位置文件")
        print(f"{'='*70}")
        print(f"输出目录: {episode_positions_dir}")
        print(f"需要生成: {episodes} 个位置文件")
        print(f"{'='*70}\n")
        
        # 导入位置生成函数
        from generate_episode_positions import generate_episode_positions
        
        # 基础环境变量（与训练配置一致）
        base_env_vars = {
            "MAP_SIZE": str(MAP_SIZE),
            "TERRAIN_COMPLEXITY_LEVEL": str(TERRAIN_COMPLEXITY_LEVEL),
            "MOUNTAIN_MIN_DISTANCE": str(MOUNTAIN_MIN_DISTANCE),
        }
        
        # 为每个episode生成位置文件
        for episode_idx, terrain_seed in enumerate(terrain_seeds):
            episode_positions_file: Optional[Path] = generate_episode_positions(
                terrain_seed, episode_idx, episode_positions_dir, base_env_vars
            )
            if episode_positions_file is None:
                print(f"⚠️  Episode {episode_idx}位置文件生成失败，将使用动态生成")
    else:
        print(f"✅ 发现已存在的位置文件: {len(existing_positions)} 个")
    
    # 选择一个默认位置文件（用于评估脚本初始化阶段）
    # 优先使用第一个episode的位置文件，避免与地形种子不匹配
    default_positions_file = positions_file
    if terrain_seeds:
        candidate = episode_positions_dir / f"episode_000_seed_{terrain_seeds[0]}.json"
        if not candidate.exists():
            candidate = episode_positions_dir / "episode_000.json"
        if candidate.exists():
            default_positions_file = candidate

    # 🔧 关键修复：验证所有评估配置使用相同的回合数
    # 确保对比的一致性
    expected_episodes = episodes
    print(f"✅ 验证：所有评估配置将使用 {expected_episodes} 个回合进行评估")
    
    # 🚀 关键优化：并行执行三个评估模式，大幅提升评估速度
    # 原因：所有评估模式使用相同的初始条件（地形和位置），可以安全并行
    print(f"\n{'='*70}")
    print(f"🚀 并行评估模式")
    print(f"{'='*70}")
    print(f"评估配置数量: {len(EVALUATION_CONFIGS)}")
    print(f"并行进程数: {min(len(EVALUATION_CONFIGS), multiprocessing.cpu_count())}")
    print(f"{'='*70}\n")
    
    # 并行执行评估
    results = {}
    max_workers = min(len(EVALUATION_CONFIGS), multiprocessing.cpu_count())
    
    # 准备任务参数（转换为可序列化的格式）
    tasks = [
        (
            eval_cfg,
            str(trained_model_path),
            str(default_positions_file),
            episodes,
            seed,
            episode_length,
            str(episode_positions_dir),
            terrain_seeds,
            str(batch_dir) if batch_dir is not None else None
        )
        for eval_cfg in EVALUATION_CONFIGS
    ]
    
    # 🚀 并行执行 + 实时日志监控
    # 方案：每个进程输出到独立日志文件，主进程实时读取并显示
    print(f"🚀 启动并行评估（{max_workers}个进程）\n")
    print(f"📝 每个评估模式的日志将保存到独立的日志文件中")
    print(f"💡 提示：可以使用 'tail -f <log_file>' 实时查看日志\n")
    
    import threading
    import time
    
    # 预先创建日志文件路径（用于监控）
    log_files = {}
    for task in tasks:
        label = task[0]["label"]
        terrain_sensing_mode = task[0]["terrain_sensing_mode"]
        if batch_dir is not None:
            eval_save_path = str(Path(batch_dir) / "evaluation_results" / f"{label}_{terrain_sensing_mode}")
        else:
            eval_save_path = f"evaluation_results/{label}_{terrain_sensing_mode}"
        log_dir = Path(eval_save_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{label}_{terrain_sensing_mode}.log"
        log_files[label] = log_file
        print(f"📝 [{label}] 日志文件: {log_file}")
    
    print()  # 空行
    
    # 启动日志监控线程（实时显示所有进程的输出）
    def monitor_logs():
        """实时监控所有日志文件并显示"""
        import re  # 🚨 关键修复：在函数内部导入re模块
        
        # 等待日志文件创建
        time.sleep(1)
        
        # 实时读取并显示日志
        file_handles = {}
        
        for label, log_file in log_files.items():
            if log_file.exists():
                try:
                    file_handles[label] = open(log_file, 'r', encoding='utf-8')
                    # 移动到文件末尾（只显示新内容）
                    file_handles[label].seek(0, 2)
                except Exception:
                    pass
        
        # 如果文件还没创建，等待一下
        if not file_handles:
            time.sleep(2)
            for label, log_file in log_files.items():
                if log_file.exists():
                    try:
                        file_handles[label] = open(log_file, 'r', encoding='utf-8')
                        file_handles[label].seek(0, 2)
                    except Exception:
                        pass
        
        # 持续监控直到所有进程完成
        last_progress = {}  # 记录每个进程的最后进度信息
        progress_update_interval = 5.0  # 进度条更新间隔（秒）- 增加到5秒，大幅减少刷新频率
        last_progress_time = time.time()
        last_display_time = {}  # 记录每个进程最后显示的时间
        
        # 🚨 关键修复：使用更智能的进度条检测和显示逻辑
        # 问题：tqdm会频繁刷新进度条，导致输出混乱
        # 解决方案：完全过滤原始进度条，只定期显示简化版本
        
        while file_handles:
            current_time = time.time()
            # 不再使用全局的should_update_progress，而是每个进程独立检查
            
            for label, fh in list(file_handles.items()):
                if fh is None:
                    continue
                try:
                    line = fh.readline()
                    if line:
                        line_stripped = line.rstrip()
                        
                        # 过滤和优化显示
                        # 1. 跳过空行
                        if not line_stripped:
                            continue
                        
                        # 2. 进度条信息：完全过滤原始进度条，只显示简化版本
                        # 🚨 关键：使用更严格的检测条件，确保完全过滤所有进度条输出
                        # 检测tqdm进度条的特征：包含"回合"、"%"、"|"、数字/步数等
                        # 注意：re模块已在monitor_logs函数开头导入
                        has_episode = "回合" in line_stripped
                        has_percent = "%" in line_stripped
                        has_bar = "|" in line_stripped
                        has_step_info = "步/s" in line_stripped or "步数=" in line_stripped
                        # 检查是否包含"数字/数字"格式（如"2146/2800"）
                        has_slash_number = bool(re.search(r'\d+/\d+', line_stripped))
                        
                        # 进度条必须同时满足：回合 + 百分比 + 进度条符号 + 步数信息
                        is_progress_bar = (
                            has_episode and 
                            has_percent and 
                            has_bar and 
                            (has_step_info or has_slash_number)
                        )
                        
                        if is_progress_bar:
                            # 🚨 关键修复：完全过滤原始进度条，只定期显示简化版本
                            # 提取关键信息：回合号、进度百分比、奖励
                            # 简化进度条显示
                            # 匹配格式：回合 1:  77%|... 或 回合 1: 77%
                            match = re.search(r'回合\s*(\d+):\s*(\d+)%', line_stripped)
                            if match:
                                ep_num = match.group(1)
                                pct = match.group(2)
                                # 提取奖励（可能格式：奖励=46238.5 或 奖励: 46238.5）
                                reward_match = re.search(r'奖励[=:]?\s*([\d.]+)', line_stripped)
                                reward = reward_match.group(1) if reward_match else "N/A"
                                
                                # 检查是否与上次显示的内容相同，且距离上次显示时间足够长（避免重复显示）
                                current_progress = (ep_num, pct, reward)
                                current_display_time = current_time
                                
                                # 只有进度发生变化，或者距离上次显示超过更新间隔，才显示
                                should_display = False
                                if label not in last_progress:
                                    should_display = True
                                elif last_progress[label] != current_progress:
                                    # 进度变化了，检查时间间隔
                                    if label not in last_display_time:
                                        should_display = True
                                    elif (current_display_time - last_display_time[label]) >= progress_update_interval:
                                        should_display = True
                                
                                if should_display:
                                    # 🚨 关键优化：使用固定位置显示，每个进程一行，保持稳定
                                    # 格式：[label] 回合X: Y% | 奖励=Z
                                    # 不使用\r覆盖，而是每次新行，但更新频率低，看起来稳定
                                    print(f"[{label}] 回合{ep_num}: {pct}% | 奖励={reward}", flush=True)
                                    last_progress[label] = current_progress
                                    last_display_time[label] = current_display_time
                                    last_progress_time = current_display_time
                            
                            # 🚨 关键：完全跳过原始进度条输出，无论是否显示简化版本
                            continue
                        
                        # 3. 过滤掉tqdm的刷新字符和ANSI转义序列
                        if '\r' in line_stripped or line_stripped.startswith('\x1b') or '\x1b[' in line_stripped:
                            continue
                        
                        # 4. 过滤掉只包含进度条字符的行（如只有████等）
                        if all(c in ' █▉▊▋▌▍▎▏|' for c in line_stripped.strip()):
                            continue
                        
                        # 5. 过滤掉任何包含进度条特征的行（确保完全过滤）
                        # 即使检测逻辑没完全匹配，如果包含关键特征也跳过
                        if "回合" in line_stripped and "%" in line_stripped and "|" in line_stripped:
                            # 这可能是进度条的一部分，完全跳过
                            continue
                        
                        # 6. 过滤掉包含tqdm特有格式的行
                        if "步/s" in line_stripped and ("[" in line_stripped or "]" in line_stripped):
                            # tqdm的格式，跳过
                            continue
                        
                        # 6. 显示重要信息（非进度条）
                        # 只显示有实际内容的信息
                        if len(line_stripped.strip()) > 0:
                            print(f"[{label}] {line_stripped}", flush=True)
                    else:
                        # 检查文件是否被关闭（进程完成）
                        if fh.closed:
                            file_handles[label] = None
                except (ValueError, OSError):
                    # 文件可能被关闭
                    file_handles[label] = None
            
            # 清理None值
            file_handles = {k: v for k, v in file_handles.items() if v is not None}
            
            if file_handles:
                time.sleep(0.1)  # 稍微增加延迟，减少CPU占用
            else:
                break
        
        # 关闭文件句柄
        for fh in file_handles.values():
            if fh and not fh.closed:
                fh.close()
    
    # 启动日志监控线程
    monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
    monitor_thread.start()
    
    # 并行执行评估
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_label = {
            executor.submit(_run_single_evaluation_worker, task): task[0]["label"]
            for task in tasks
        }
        
        # 收集结果
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result = future.result()
                results[label] = result
                if result.get("success", False):
                    print(f"\n✅ [评估-{label}] 完成")
                else:
                    print(f"\n❌ [评估-{label}] 失败: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"\n❌ [评估-{label}] 异常: {e}")
                import traceback
                traceback.print_exc()
                results[label] = {
                    "label": label,
                    "success": False,
                    "error": str(e),
                }
    
    # 等待日志监控线程完成
    monitor_thread.join(timeout=5)
    
    return results


def main():
    args = parse_args()
    
    # 检查训练脚本
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"[错误] 找不到训练脚本: {script_path}")
        sys.exit(1)
    
    # 生成固定位置文件
    positions_file = ensure_fixed_positions(args.positions_file, args, "terrain_sensing")
    
    # 创建批次目录
    manager = AblationBatchManager(root_dir="terrain_sensing_experiments")
    batch_config = {
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "algorithm": args.algorithm,
        "use_weighted_reward": args.use_weighted_reward,
        "scenario_seed": SCENARIO_SEED,
        "terrain_complexity": TERRAIN_COMPLEXITY_LEVEL,
        "map_size": MAP_SIZE,
        "mountain_min_distance": MOUNTAIN_MIN_DISTANCE,
        "positions_file": str(positions_file),
        "notes": "地形感知模式消融实验：对比local vs oracle感知对APF性能的影响"
    }
    # 🔧 修复：不创建不必要的实验子目录（terrain_sensing实验不使用这些目录）
    # 只创建plots和results目录即可
    batch_dir = manager.create_batch(config=batch_config, experiments=[])
    
    print(f"{'='*70}")
    print(f"✅ 批次目录已创建: {batch_dir}")
    print(f"{'='*70}\n")
    
    # 🔧 修正：只训练一次，然后在评估时分别用local和oracle模式评估
    if args.eval_only:
        # 仅评估模式：使用指定的模型路径进行评估
        # 🔧 如果未指定模型路径，使用文件中的默认路径
        if not args.trained_model_path:
            trained_model_path = Path(DEFAULT_MODEL_PATH)
            print(f"ℹ️  使用默认模型路径: {trained_model_path}")
            print(f"   如需修改，请在脚本中修改 DEFAULT_MODEL_PATH 或使用 --trained-model-path 参数")
        else:
            trained_model_path = Path(args.trained_model_path)
        
        # 验证模型路径是否存在
        if not trained_model_path.exists():
            print(f"❌ 错误: 模型路径不存在: {trained_model_path}")
            print(f"\n请检查以下位置:")
            print(f"  1. 脚本中的 DEFAULT_MODEL_PATH = '{DEFAULT_MODEL_PATH}'")
            print(f"  2. 命令行参数 --trained-model-path")
            print(f"\n可用的模型路径:")
            # 查找可用的模型
            models_dir = Path("models")
            if models_dir.exists():
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir() and "apf_learnable" in model_dir.name:
                        for subdir in ["best", "final", "checkpoint"]:
                            if (model_dir / subdir).exists():
                                print(f"  - {model_dir / subdir}")
            sys.exit(1)
        results = evaluate_terrain_sensing(trained_model_path, positions_file, args.eval_episodes, args.eval_seed, args.eval_episode_length, batch_dir)
        
        # 🔧 关键修复：在eval_only模式下也生成对比图
        print(f"\n{'='*70}")
        print(f"📊 开始生成对比图...")
        print(f"{'='*70}\n")
        
        # 保存结果
        output_dir = batch_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({"evaluation": results}, f, indent=2, ensure_ascii=False)
        
        # 生成对比图（复用相同的逻辑）
        _generate_comparison_plots(output_dir, results_file, batch_dir)
        
        print("\n评估完成！")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    # 🔧 修正：只训练一次（使用local模式）
    print(f"\n{'='*70}")
    print(f"地形感知模式消融对比实验")
    print(f"训练模式: 只训练一次（使用local模式）")
    print(f"评估模式: 分别用local和oracle模式评估同一个模型")
    print(f"{'='*70}\n")
    
    # 运行训练（只训练一次）
    training_result = run_training(TRAINING_CONFIG, positions_file, args)
    
    if not training_result["success"]:
        print("❌ 训练失败，无法进行评估")
        return
    
    # 训练完成后，使用训练好的模型进行评估
    # 🔧 修复：查找训练好的模型路径（支持带时间戳的模型目录）
    training_label = TRAINING_CONFIG["label"]  # apf_learnable
    models_dir = Path("models")
    
    # 方法1：尝试直接路径（不带时间戳）
    trained_model_path = models_dir / training_label / "best"
    
    if not trained_model_path.exists():
        # 方法2：尝试final目录（不带时间戳）
        trained_model_path = models_dir / training_label / "final"
        
        if not trained_model_path.exists():
            # 方法3：查找带时间戳的最新模型目录
            print(f"🔍 查找带时间戳的模型目录（{training_label}）...")
            matching_dirs = []
            if models_dir.exists():
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith(training_label + "_"):
                        # 检查是否包含时间戳格式（YYYYMMDD_HHMMSS）
                        import re
                        if re.match(rf"^{re.escape(training_label)}_\d{{8}}_\d{{6}}$", model_dir.name):
                            matching_dirs.append(model_dir)
            
            if matching_dirs:
                # 按目录名排序（时间戳越大越新）
                matching_dirs.sort(key=lambda x: x.name, reverse=True)
                latest_model_dir = matching_dirs[0]
                print(f"✅ 找到最新模型目录: {latest_model_dir.name}")
                
                # 尝试best目录
                trained_model_path = latest_model_dir / "best"
                if not trained_model_path.exists():
                    # 尝试final目录
                    trained_model_path = latest_model_dir / "final"
                    if not trained_model_path.exists():
                        print(f"⚠️  警告: 模型目录 {latest_model_dir} 中找不到 best 或 final 子目录")
                        print(f"   请检查模型目录结构")
                        print(f"   可用的模型路径:")
                        for subdir in ["best", "final", "checkpoint"]:
                            if (latest_model_dir / subdir).exists():
                                print(f"     - {latest_model_dir / subdir}")
                        return
            else:
                print(f"⚠️  警告: 找不到训练好的模型")
                print(f"   查找路径:")
                print(f"     1. {models_dir / training_label / 'best'}")
                print(f"     2. {models_dir / training_label / 'final'}")
                print(f"     3. {models_dir / training_label}_*_*/best 或 final")
                print(f"\n   请手动指定模型路径进行评估:")
                print(f"     python ablation_terrain_sensing.py --eval-only --trained-model-path <模型路径>")
                print(f"\n   可用的模型目录:")
                if models_dir.exists():
                    for model_dir in sorted(models_dir.iterdir()):
                        if model_dir.is_dir() and training_label in model_dir.name:
                            for subdir in ["best", "final"]:
                                if (model_dir / subdir).exists():
                                    print(f"     - {model_dir / subdir}")
                return
    
    print(f"\n{'='*70}")
    print(f"✅ 训练完成，开始评估")
    print(f"模型路径: {trained_model_path}")
    print(f"{'='*70}\n")
    
    # 使用训练好的模型进行评估（分别用local和oracle模式）
    eval_results = evaluate_terrain_sensing(trained_model_path, positions_file, args.eval_episodes, args.eval_seed, args.eval_episode_length)
    
    # 合并训练和评估结果
    results = {
        "training": training_result,
        "evaluation": eval_results
    }
    
    # 保存结果
    output_dir = batch_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 🔧 新增：从评估结果加载指标并生成对比图
    _generate_comparison_plots(output_dir, results_file, batch_dir)


def _plot_arrival_penetration_comparison(series: List[Dict], title: str, output_path: Path):
    """
    绘制到达时间/步数和穿透率对比图
    
    Args:
        series: 评估结果序列
        title: 图表标题
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        print("[跳过] matplotlib未安装，跳过绘图")
        return
    
    setup_english_fonts()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    has_data = False
    
    # 使用高对比度颜色
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']  # 深蓝、深红、深绿、深紫
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        # 如果name_en仍然是中文，使用label作为回退
        if name_en and any('\u4e00' <= char <= '\u9fff' for char in str(name_en)):
            name_en = item.get('label', 'Unknown')
        color = colors[idx % len(colors)]
        
        # 1. 到达步数（仅显示成功的回合）
        arrival_steps = metrics.get("arrival_steps", [])
        if arrival_steps:
            has_data = True
            # 过滤掉None值，只保留成功的回合
            valid_steps = [(i+1, step) for i, step in enumerate(arrival_steps) if step is not None]
            if valid_steps:
                episodes, steps = zip(*valid_steps)
                axes[0].scatter(episodes, steps, label=name_en, color=color, alpha=0.7, s=50)
                # 计算平均值
                mean_step = np.mean(steps)
                axes[0].axhline(y=mean_step, color=color, linestyle='--', alpha=0.5, 
                               label=f"{name_en} (Mean: {mean_step:.1f})")
        
        # 2. 到达时间（仅显示成功的回合）
        arrival_times = metrics.get("arrival_times", [])
        if arrival_times:
            has_data = True
            valid_times = [(i+1, time) for i, time in enumerate(arrival_times) if time is not None]
            if valid_times:
                episodes, times = zip(*valid_times)
                axes[1].scatter(episodes, times, label=name_en, color=color, alpha=0.7, s=50)
                # 计算平均值
                mean_time = np.mean(times)
                axes[1].axhline(y=mean_time, color=color, linestyle='--', alpha=0.5,
                               label=f"{name_en} (Mean: {mean_time:.2f}s)")
        
        # 3. 穿透率（每回合的穿透次数）
        penetration_rates = metrics.get("penetration_rates", [])
        if penetration_rates:
            has_data = True
            episodes = range(1, len(penetration_rates) + 1)
            axes[2].plot(episodes, penetration_rates, label=name_en, color=color, 
                        linewidth=2.5, alpha=0.9, linestyle='-')
            # 计算平均值
            mean_rate = np.mean(penetration_rates)
            axes[2].axhline(y=mean_rate, color=color, linestyle='--', alpha=0.5,
                           label=f"{name_en} (Mean: {mean_rate:.1f})")
        
        # 4. 穿透深度（最大深度和平均深度）
        penetration_max_depths = metrics.get("penetration_max_depths", [])
        penetration_mean_depths = metrics.get("penetration_mean_depths", [])
        if penetration_max_depths or penetration_mean_depths:
            has_data = True
            episodes = range(1, max(len(penetration_max_depths), len(penetration_mean_depths)) + 1)
            if penetration_max_depths:
                axes[3].plot(episodes[:len(penetration_max_depths)], penetration_max_depths, 
                            label=f"{name_en} (Max Depth)", color=color, linewidth=2.5, 
                            alpha=0.9, linestyle='-')
            if penetration_mean_depths:
                axes[3].plot(episodes[:len(penetration_mean_depths)], penetration_mean_depths,
                            label=f"{name_en} (Mean Depth)", color=color, linewidth=2.5,
                            alpha=0.7, linestyle='--')
    
    if has_data:
        axes[0].set_title(f"{title} - Arrival Steps", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Arrival Step", fontsize=12)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        axes[1].set_title(f"{title} - Arrival Time", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Arrival Time (s)", fontsize=12)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        axes[2].set_title(f"{title} - Penetration Rate", fontsize=14, fontweight='bold')
        axes[2].set_ylabel("Penetration Count", fontsize=12)
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        axes[3].set_title(f"{title} - Penetration Depth", fontsize=14, fontweight='bold')
        axes[3].set_xlabel("Episode", fontsize=12)
        axes[3].set_ylabel("Penetration Depth (m)", fontsize=12)
        axes[3].legend(loc='upper right', fontsize=10)
        axes[3].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    else:
        print(f"⚠️  没有可用的到达时间/穿透率数据，跳过绘图: {output_path}")
    
    plt.close(fig)


def _generate_comparison_interactive_trajectory(series: List[Dict], output_dir: Path, timestamp: str):
    """
    🔧 新增：生成三个APF方法在同一地图上的独立交互图
    
    找到差距最大的地图（episode），然后为每个方法生成独立的交互图，显示在该地图上的轨迹
    
    Args:
        series: 评估结果序列（每个方法一个）
        output_dir: 输出目录
        timestamp: 时间戳（用于文件名）
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        print("⚠️  需要安装 plotly 才能生成交互对比图: pip install plotly")
        return
    
    print(f"\n{'='*70}")
    print(f"🎨 开始生成三个方法在同一地图上的独立交互图...")
    print(f"{'='*70}\n")
    
    # 1. 找到差距最大的地图（episode）
    # 计算每个episode的奖励差异（最大值-最小值）
    if len(series) < 2:
        print("⚠️  需要至少2个方法才能生成对比图")
        return
    
    # 获取所有方法的奖励数据
    all_rewards_by_method = {}
    episode_details_by_method = {}
    
    for item in series:
        label = item["label"]
        metrics = item["metrics"]
        rewards = metrics.get("episode_rewards", [])
        all_rewards_by_method[label] = rewards
        
        # 加载episode_details（包含轨迹数据）
        eval_save_path = Path(item.get("log_dir", ""))
        eval_results_json = eval_save_path / "evaluation_results.json"
        if eval_results_json.exists():
            try:
                with open(eval_results_json, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                episode_details_by_method[label] = eval_data.get('episode_details', [])
            except Exception as e:
                print(f"⚠️  加载 {label} 的episode_details失败: {e}")
                episode_details_by_method[label] = []
        else:
            episode_details_by_method[label] = []
    
    # 找到所有方法都有数据的episode范围
    min_episodes = min([len(rewards) for rewards in all_rewards_by_method.values()])
    if min_episodes == 0:
        print("⚠️  没有可用的episode数据")
        return
    
    # 计算每个episode的奖励差异
    reward_differences = []
    for ep_idx in range(min_episodes):
        ep_rewards = [all_rewards_by_method[label][ep_idx] for label in all_rewards_by_method.keys()]
        if all(r is not None and np.isfinite(r) for r in ep_rewards):
            reward_diff = max(ep_rewards) - min(ep_rewards)
            reward_differences.append((ep_idx, reward_diff))
    
    if not reward_differences:
        print("⚠️  没有找到有效的episode数据")
        return
    
    # 找到差距最大的episode
    max_diff_episode = max(reward_differences, key=lambda x: x[1])
    max_diff_ep_idx = max_diff_episode[0]
    max_diff_value = max_diff_episode[1]
    
    print(f"📊 找到差距最大的地图: Episode {max_diff_ep_idx + 1}")
    print(f"   奖励差异: {max_diff_value:.2f}")
    for label in all_rewards_by_method.keys():
        reward = all_rewards_by_method[label][max_diff_ep_idx]
        print(f"   - {label}: {reward:.2f}")
    
    # 2. 从每个方法的评估结果中提取该episode的轨迹数据
    trajectories_by_method = {}
    rewards_by_method = {}
    success_flags_by_method = {}
    collision_counts_by_method = {}
    name_en_by_method = {}
    
    for item in series:
        label = item["label"]
        episode_details = episode_details_by_method.get(label, [])
        name_en_by_method[label] = item.get('name_en', label)
        
        if max_diff_ep_idx < len(episode_details):
            ep_data = episode_details[max_diff_ep_idx]
            trajectory = ep_data.get('trajectory', [])
            reward = ep_data.get('reward', 0.0)
            success = ep_data.get('success', 0)
            collision_count = ep_data.get('collision_count', 0)
            
            if trajectory:
                trajectories_by_method[label] = trajectory
                rewards_by_method[label] = reward
                success_flags_by_method[label] = success
                collision_counts_by_method[label] = collision_count
                print(f"✅ {label}: 找到轨迹数据（{len(trajectory)}步）")
            else:
                print(f"⚠️  {label}: 该episode没有轨迹数据")
        else:
            print(f"⚠️  {label}: episode索引超出范围")
    
    if len(trajectories_by_method) == 0:
        print("⚠️  没有找到任何轨迹数据")
        return
    
    # 3. 为每个方法生成独立的交互图
    # 🔧 关键修复：需要获取场景信息以绘制地形和障碍物
    # 从第一个评估结果中获取场景名称
    scenario_name = None
    scenario = None
    env_instance = None
    
    # 尝试从评估结果中获取场景信息
    for item in series:
        eval_save_path = Path(item.get("log_dir", ""))
        eval_results_json = eval_save_path / "evaluation_results.json"
        if eval_results_json.exists():
            try:
                with open(eval_results_json, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                scenario_name = eval_data.get('scenario', 'paper3d_terrain_energy')
                print(f"🔧 从评估结果获取场景名称: {scenario_name}")
                break
            except Exception:
                continue
    
    # 如果没有找到场景名称，使用默认值
    if scenario_name is None:
        scenario_name = 'paper3d_terrain_energy'
        print(f"⚠️  未找到场景名称，使用默认值: {scenario_name}")
    
    # 🔧 关键修复：初始化场景对象以获取地形和障碍物信息
    goal_positions_by_method = {}  # 存储每个方法的目标点信息
    
    # 🔧 关键修复：尝试从评估结果中获取固定位置文件路径
    positions_file = None
    for item in series:
        eval_save_path = Path(item.get("log_dir", ""))
        eval_results_json = eval_save_path / "evaluation_results.json"
        if eval_results_json.exists():
            try:
                with open(eval_results_json, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                # 尝试从评估结果中获取固定位置文件路径
                if 'positions_file' in eval_data:
                    positions_file = eval_data['positions_file']
                    print(f"🔧 从评估结果获取固定位置文件: {positions_file}")
                    break
            except Exception:
                continue
    
    # 如果没有找到，尝试使用默认路径
    if positions_file is None:
        default_positions_file = Path("./saved_positions/5.json")
        if default_positions_file.exists():
            positions_file = str(default_positions_file)
            print(f"🔧 使用默认固定位置文件: {positions_file}")
    
    try:
        from multiagent.scenarios import load as load_scenario
        scenario_module = load_scenario(scenario_name)
        
        # 🔧 关键修复：初始化场景时加载固定位置文件（如果存在）
        scenario_kwargs = {}
        if positions_file and Path(positions_file).exists():
            scenario_kwargs['use_fixed_positions'] = True
            scenario_kwargs['fixed_positions_file'] = positions_file
            print(f"🔧 场景初始化时将加载固定位置文件: {positions_file}")
        
        scenario = scenario_module.Scenario(**scenario_kwargs)
        scenario.make_world()
        print(f"✅ 场景对象已初始化: {scenario_name}")
        
        # 🔧 关键修复：从场景中获取目标点信息
        # 尝试从场景的固定位置中获取目标点
        goal_pos = None
        agent_goals = []
        
        # 方法1：从fixed_positions获取
        if hasattr(scenario, 'fixed_positions') and scenario.fixed_positions:
            goal_pos = scenario.fixed_positions.get('goal', None)
            if goal_pos:
                print(f"✅ 从场景fixed_positions获取中央目标位置: {goal_pos}")
        
        # 方法2：从world.landmarks获取
        if goal_pos is None and hasattr(scenario, 'world') and hasattr(scenario.world, 'landmarks'):
            if len(scenario.world.landmarks) > 0:
                landmark = scenario.world.landmarks[0]
                if hasattr(landmark, 'state') and hasattr(landmark.state, 'p_pos'):
                    goal_pos = landmark.state.p_pos.tolist()
                    print(f"✅ 从world.landmarks获取中央目标位置: {goal_pos}")
        
        # 方法3：从world.agents的goal_a获取各智能体目标
        if hasattr(scenario, 'world') and hasattr(scenario.world, 'agents'):
            for i, agent in enumerate(scenario.world.agents):
                if hasattr(agent, 'goal_a') and hasattr(agent.goal_a, 'state'):
                    agent_goal = agent.goal_a.state.p_pos.tolist()
                    agent_goals.append(agent_goal)
                    print(f"✅ 从world.agents获取智能体{i}目标位置: {agent_goal}")
                else:
                    agent_goals.append(None)
        
        # 为所有方法设置相同的目标点信息（因为使用相同的地图）
        goal_positions_dict = {
            'goal_pos': goal_pos,
            'agent_goals': agent_goals
        }
        for label in trajectories_by_method.keys():
            goal_positions_by_method[label] = goal_positions_dict
        
        if goal_pos is None and len(agent_goals) == 0:
            print(f"⚠️  警告: 无法从场景中获取目标点信息，交互图中将不显示目标点")
    except Exception as e:
        print(f"⚠️  初始化场景失败: {e}")
        print(f"   将只显示轨迹，不显示地形和障碍物")
        scenario = None
    
    colors = {
        'apf_learnable_local': '#0066CC',  # 深蓝
        'apf_learnable_oracle_same': '#CC0000',  # 深红
        'apf_learnable_oracle_dense': '#00AA00',  # 深绿
    }
    
    generated_files = []
    
    for label, trajectory in trajectories_by_method.items():
        if not trajectory:
            continue
        
        # 🔧 关键修复：使用TrajectoryVisualizer生成交互图，确保包含地形和障碍物
        from visualization.trajectory_visualizer import TrajectoryVisualizer
        visualizer = TrajectoryVisualizer()
        
        # 转换轨迹格式：从 [step][agent] 转为 [agent][step]（TrajectoryVisualizer期望的格式）
        n_agents = len(trajectory[0]) if trajectory else 0
        agent_trajectories = []
        for agent_idx in range(n_agents):
            agent_traj = []
            for step_data in trajectory:
                if agent_idx < len(step_data) and step_data[agent_idx] is not None:
                    pos = step_data[agent_idx]
                    if len(pos) >= 3:
                        agent_traj.append([pos[0], pos[1], pos[2]])
            if agent_traj:
                agent_trajectories.append(np.array(agent_traj))
        
        # 获取方法信息
        name_en = name_en_by_method.get(label, label)
        reward = rewards_by_method.get(label, 0.0)
        success = success_flags_by_method.get(label, 0)
        collision = collision_counts_by_method.get(label, 0)
        
        # 保存HTML文件（每个方法一个独立的文件）
        safe_label = label.replace(' ', '_').replace('/', '_')
        html_path = output_dir / f"comparison_interactive_{safe_label}_ep{max_diff_ep_idx + 1}_{timestamp}.html"
        
        try:
            # 🔧 关键修复：使用TrajectoryVisualizer生成交互图，自动包含地形和障碍物
            title = f"{name_en} - Episode {max_diff_ep_idx + 1} (Reward: {reward:.2f}, Success: {success}, Collisions: {collision})"
            # 🔧 关键修复：获取该方法的目标点信息
            goal_positions = goal_positions_by_method.get(label, None)
            visualizer.generate_trajectory_interactive(
                trajectories=agent_trajectories,
                save_path=str(html_path),
                title=title,
                goal_positions=goal_positions,  # 🔧 传递目标点信息
                scenario=scenario,  # 🔧 传递场景对象，用于绘制地形和障碍物
                env_instance=env_instance  # 如果有环境实例，也可以传递
            )
            generated_files.append(html_path)
            print(f"✅ {name_en} 交互图已生成（包含地形和障碍物）: {html_path.name}")
        except Exception as e:
            print(f"⚠️  保存 {name_en} 交互图失败: {e}")
            import traceback
            traceback.print_exc()
    
    if generated_files:
        print(f"\n✅ 共生成 {len(generated_files)} 张独立交互图（同一地图 Episode {max_diff_ep_idx + 1}）")
        print(f"   奖励差异: {max_diff_value:.2f}")
        for html_path in generated_files:
            print(f"   - {html_path.name}")


def _generate_evaluation_summary(series: List[Dict], output_path: Path):
    """
    生成评估汇总统计表（文本格式）
    
    Args:
        series: 评估结果序列
        output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        # 🔧 修复：使用实际的回合数，而不是硬编码的20
        actual_episodes = len(series[0]['metrics']['episode_rewards']) if series else 20
        f.write(f"Terrain Sensing Mode Evaluation Summary ({actual_episodes} Episodes)\n")
        f.write("="*80 + "\n\n")
        
        for item in series:
            metrics = item["metrics"]
            name_en = item.get('name_en') or item.get('label', 'Unknown')
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Algorithm: {name_en}\n")
            f.write(f"{'='*80}\n\n")
            
            # 1. 成功率统计
            success_flags = metrics.get("success_flags", [])
            team_success_flags = metrics.get("team_success_flags", [])
            if success_flags:
                success_rate = np.mean(success_flags) * 100
                f.write(f"Success Rate (Individual): {success_rate:.2f}% ({sum(success_flags)}/{len(success_flags)})\n")
            if team_success_flags:
                team_success_rate = np.mean(team_success_flags) * 100
                f.write(f"Success Rate (Team): {team_success_rate:.2f}% ({sum(team_success_flags)}/{len(team_success_flags)})\n")
            
            # 2. 到达时间/步数统计
            arrival_steps = metrics.get("arrival_steps", [])
            arrival_times = metrics.get("arrival_times", [])
            if arrival_steps:
                valid_steps = [s for s in arrival_steps if s is not None]
                if valid_steps:
                    f.write(f"Arrival Steps: Mean={np.mean(valid_steps):.1f}, Min={np.min(valid_steps):.1f}, Max={np.max(valid_steps):.1f}\n")
            if arrival_times:
                valid_times = [t for t in arrival_times if t is not None]
                if valid_times:
                    f.write(f"Arrival Time: Mean={np.mean(valid_times):.2f}s, Min={np.min(valid_times):.2f}s, Max={np.max(valid_times):.2f}s\n")
            
            # 3. 碰撞统计
            collision_counts = metrics.get("collision_counts", [])
            if collision_counts:
                total_collisions = sum(collision_counts)
                mean_collisions = np.mean(collision_counts)
                f.write(f"Collision Count: Total={total_collisions}, Mean={mean_collisions:.2f}, Max={np.max(collision_counts)}\n")
            
            # 4. 穿透统计
            penetration_rates = metrics.get("penetration_rates", [])
            penetration_max_depths = metrics.get("penetration_max_depths", [])
            penetration_mean_depths = metrics.get("penetration_mean_depths", [])
            if penetration_rates:
                total_penetrations = sum(penetration_rates)
                mean_penetration_rate = np.mean(penetration_rates)
                f.write(f"Penetration Rate: Total={total_penetrations}, Mean={mean_penetration_rate:.2f}\n")
            if penetration_max_depths:
                valid_max_depths = [d for d in penetration_max_depths if d > 0]
                if valid_max_depths:
                    f.write(f"Penetration Max Depth: Mean={np.mean(valid_max_depths):.2f}m, Max={np.max(valid_max_depths):.2f}m\n")
            if penetration_mean_depths:
                valid_mean_depths = [d for d in penetration_mean_depths if d > 0]
                if valid_mean_depths:
                    f.write(f"Penetration Mean Depth: Mean={np.mean(valid_mean_depths):.2f}m\n")
            
            # 5. 最小净空距离统计
            min_distances = metrics.get("min_distances_to_obstacle", [])
            if min_distances:
                min_values = []
                mean_values = []
                for md in min_distances:
                    if isinstance(md, dict):
                        min_values.append(md.get('min', 0.0))
                        mean_values.append(md.get('mean', 0.0))
                    elif isinstance(md, (int, float)):
                        min_values.append(float(md))
                        mean_values.append(float(md))
                
                if min_values:
                    f.write(f"Min Clearance (min): Mean={np.mean(min_values):.2f}m, Min={np.min(min_values):.2f}m\n")
                    # 计算P(d_min ≤ δ) for δ = 1.5m
                    delta = 1.5
                    violation_count = sum(1 for v in min_values if v <= delta)
                    violation_prob = violation_count / len(min_values) * 100
                    f.write(f"P(d_min ≤ {delta}m): {violation_prob:.2f}% ({violation_count}/{len(min_values)})\n")
                if mean_values:
                    f.write(f"Min Clearance (mean): Mean={np.mean(mean_values):.2f}m\n")
            
            # 6. 奖励统计
            episode_rewards = metrics.get("episode_rewards", [])
            if episode_rewards:
                f.write(f"Reward: Mean={np.mean(episode_rewards):.2f}, Std={np.std(episode_rewards):.2f}, Max={np.max(episode_rewards):.2f}, Min={np.min(episode_rewards):.2f}\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("End of Summary\n")
        f.write("="*80 + "\n")


def _generate_comparison_plots(output_dir: Path, results_file: Path, batch_dir: Optional[Path] = None):
    """
    生成对比图的通用函数，可在训练+评估模式和仅评估模式下使用
    
    Args:
        output_dir: 输出目录
        results_file: 结果文件路径（用于保存）
    """
    print(f"\n{'='*70}")
    print(f"📊 开始生成对比图...")
    print(f"{'='*70}\n")
    
    # 加载评估结果并转换为series格式
    series = []
    episodes_count = None  # 🔧 新增：记录所有评估配置的回合数，确保一致性
    
    for eval_cfg in EVALUATION_CONFIGS:
        label = eval_cfg["label"]
        # 🔧 修复：从批次目录中读取评估结果，而不是从根目录下的evaluation_results/
        if batch_dir is not None:
            eval_save_path = batch_dir / "evaluation_results" / f"{label}_{eval_cfg['terrain_sensing_mode']}"
        else:
            eval_save_path = Path(f"evaluation_results/{label}_{eval_cfg['terrain_sensing_mode']}")
        eval_results_json = eval_save_path / "evaluation_results.json"
        
        if not eval_results_json.exists():
            print(f"⚠️  警告: 评估结果文件不存在: {eval_results_json}")
            continue
        
        # 🔧 关键修复：验证所有评估配置的回合数一致
        try:
            with open(eval_results_json, 'r', encoding='utf-8') as f:
                eval_data_preview = json.load(f)
            current_episodes = eval_data_preview.get('episodes', 0)
            if episodes_count is None:
                episodes_count = current_episodes
                print(f"🔧 基准回合数: {episodes_count} (从第一个评估结果获取)")
            elif current_episodes != episodes_count:
                print(f"⚠️  警告: 评估配置 {label} 的回合数 ({current_episodes}) 与基准 ({episodes_count}) 不一致！")
                print(f"   这可能导致对比图不准确，建议重新运行评估以确保一致性")
        except Exception as e:
            print(f"⚠️  警告: 无法验证评估结果回合数: {e}")
        
        try:
            with open(eval_results_json, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # 提取指标
            all_rewards = eval_data.get('all_rewards', [])
            episode_details = eval_data.get('episode_details', [])
            
            # 转换为metrics格式（与训练结果格式一致）
            # 🔧 注意：评估结果可能不包含所有指标，使用默认值
            # 🔧 修复：处理min_distance可能是字典的情况（包含mean和min）
            min_distances_list = []
            if episode_details:
                for ep in episode_details:
                    min_dist = ep.get('min_distance', None)
                    if min_dist is None:
                        min_distances_list.append({'mean': 0.0, 'min': 0.0})
                    elif isinstance(min_dist, dict):
                        # 已经是字典格式（与训练脚本一致）
                        min_distances_list.append(min_dist)
                    else:
                        # 如果是单个数值，转换为字典格式
                        try:
                            min_dist_float = float(min_dist)
                            min_distances_list.append({'mean': min_dist_float, 'min': min_dist_float})
                        except (ValueError, TypeError):
                            min_distances_list.append({'mean': 0.0, 'min': 0.0})
            
            # 🔧 新增：提取到达时间/步数
            arrival_steps = [ep.get('arrival_step') for ep in episode_details] if episode_details else []
            arrival_times = [ep.get('arrival_time') for ep in episode_details] if episode_details else []
            
            # 🔧 新增：提取穿透深度统计
            penetration_stats = [ep.get('penetration_stat') for ep in episode_details] if episode_details else []
            penetration_rates = []
            penetration_max_depths = []
            penetration_mean_depths = []
            for pstat in penetration_stats:
                if pstat is not None and isinstance(pstat, dict):
                    penetration_rates.append(pstat.get('count', 0))
                    penetration_max_depths.append(pstat.get('max_depth', 0.0))
                    penetration_mean_depths.append(pstat.get('mean_depth', 0.0))
                else:
                    penetration_rates.append(0)
                    penetration_max_depths.append(0.0)
                    penetration_mean_depths.append(0.0)
            
            metrics = {
                "episode_rewards": all_rewards,
                "success_flags": [ep.get('success', 0) for ep in episode_details] if episode_details else [],
                "collision_counts": [ep.get('collision_count', 0) for ep in episode_details] if episode_details else [],
                "min_distances_to_obstacle": min_distances_list,
                "team_success_flags": [ep.get('team_success', 0) for ep in episode_details] if episode_details else [],
                # 🔧 新增：智能体级别的成功标志和碰撞次数
                "agent_success_flags": [ep.get('agent_success_flags', []) for ep in episode_details] if episode_details else [],
                "agent_collision_counts": [ep.get('agent_collision_counts', []) for ep in episode_details] if episode_details else [],
                # 🔧 新增：到达时间/步数
                "arrival_steps": arrival_steps,
                "arrival_times": arrival_times,
                # 🔧 新增：穿透深度统计
                "penetration_rates": penetration_rates,
                "penetration_max_depths": penetration_max_depths,
                "penetration_mean_depths": penetration_mean_depths,
            }
            
            # 🔧 如果评估结果中没有这些指标，尝试从其他来源获取或使用空列表
            # 注意：评估脚本可能没有保存这些指标，所以可能为空列表
            
            series.append({
                "label": label,
                "name": eval_cfg.get("name", label),
                "name_en": eval_cfg.get("name_en", label),
                "description": eval_cfg.get("description", ""),
                "metrics": metrics,
                "log_dir": str(eval_save_path),
            })
            
            print(f"✅ 已加载评估结果: {label} ({len(all_rewards)} 回合)")
        except Exception as e:
            print(f"⚠️  警告: 加载评估结果失败 {label}: {e}")
            continue
    
    # 🔧 关键修复：生成评估对比图（三个APF算法在20个回合评估中的性能指标对比）
    if series:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 🔧 修复：使用实际的回合数，而不是硬编码的20
        actual_episodes = len(series[0]['metrics']['episode_rewards']) if series else episodes_count or 20
        title = f"Terrain Sensing Mode Evaluation Comparison ({actual_episodes} Episodes)"
        
        print(f"\n{'='*70}")
        print(f"📊 开始生成评估对比图（三个APF算法）...")
        print(f"   对比算法: {[item['name_en'] for item in series]}")
        print(f"   评估回合数: {len(series[0]['metrics']['episode_rewards']) if series else 0}")
        print(f"{'='*70}\n")
        
        # 1. 奖励对比图
        reward_png = output_dir / f"evaluation_reward_comparison_{timestamp}.png"
        try:
            plot_comparison_rewards(series, title, reward_png, smooth_window=10, fit_method="moving_average")
            print(f"✅ [评估对比] 奖励对比图已生成: {reward_png}")
        except Exception as e:
            print(f"⚠️  [评估对比] 生成奖励对比图失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 成功率、碰撞次数、净空距离对比图（评估核心指标）
        has_metrics = False
        for item in series:
            if (item["metrics"].get("success_flags") or 
                item["metrics"].get("collision_counts") or 
                item["metrics"].get("min_distances_to_obstacle")):
                has_metrics = True
                break
        
        if has_metrics:
            success_collision_png = output_dir / f"evaluation_success_collision_clearance_comparison_{timestamp}.png"
            try:
                plot_comparison_success_collision_clearance(series, title, success_collision_png, smooth_window=10, fit_method="moving_average")
                print(f"✅ [评估对比] 成功率/碰撞/净空对比图已生成: {success_collision_png}")
            except Exception as e:
                print(f"⚠️  [评估对比] 生成成功率/碰撞/净空对比图失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. 团队成功率和最小净空距离分布对比图（包含P(d_min ≤ δ)）
            success_clearance_png = output_dir / f"evaluation_success_rate_and_clearance_comparison_{timestamp}.png"
            try:
                plot_comparison_success_rate_and_clearance(series, title, success_clearance_png, smooth_window=10, fit_method="moving_average")
                print(f"✅ [评估对比] 团队成功率/净空分布对比图已生成（包含P(d_min≤δ)）: {success_clearance_png}")
            except Exception as e:
                print(f"⚠️  [评估对比] 生成团队成功率/净空分布对比图失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  [评估对比] 评估结果中缺少成功率/碰撞/净空指标，跳过相关对比图")
            print(f"   提示：请检查评估脚本是否正确保存了这些指标")
        
        # 4. 到达时间/步数和穿透率对比图（评估关键指标）
        try:
            arrival_penetration_png = output_dir / f"evaluation_arrival_penetration_comparison_{timestamp}.png"
            _plot_arrival_penetration_comparison(series, title, arrival_penetration_png)
            print(f"✅ [评估对比] 到达时间/步数和穿透率对比图已生成: {arrival_penetration_png}")
        except Exception as e:
            print(f"⚠️  [评估对比] 生成到达时间/步数和穿透率对比图失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. 🔧 新增：生成评估汇总统计表（文本格式）
        try:
            summary_txt = output_dir / f"evaluation_summary_{timestamp}.txt"
            _generate_evaluation_summary(series, summary_txt)
            print(f"✅ [评估对比] 评估汇总统计表已生成: {summary_txt}")
        except Exception as e:
            print(f"⚠️  [评估对比] 生成评估汇总统计表失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 6. 🔧 新增：生成三个方法在同一地图上的最佳回合交互对比图
        try:
            _generate_comparison_interactive_trajectory(series, output_dir, timestamp)
        except Exception as e:
            print(f"⚠️  [评估对比] 生成交互对比图失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"✅ [评估对比] 所有对比图已生成！")
        print(f"输出目录: {output_dir}")
        print(f"生成的对比图:")
        print(f"  1. 奖励对比图: evaluation_reward_comparison_{timestamp}.png")
        print(f"  2. 成功率/碰撞/净空对比图: evaluation_success_collision_clearance_comparison_{timestamp}.png")
        print(f"  3. 团队成功率/净空分布对比图（含P(d_min≤δ)）: evaluation_success_rate_and_clearance_comparison_{timestamp}.png")
        print(f"  4. 到达时间/步数和穿透率对比图: evaluation_arrival_penetration_comparison_{timestamp}.png")
        print(f"  5. 评估汇总统计表: evaluation_summary_{timestamp}.txt")
        print(f"  6. 三个方法在同一地图上的独立交互图: comparison_interactive_*_ep*_{timestamp}.html (3张)")
        print(f"{'='*70}\n")
    else:
        print(f"⚠️  警告: 没有可用的评估结果，无法生成对比图")
        print(f"   请检查评估结果文件是否存在:")
        for eval_cfg in EVALUATION_CONFIGS:
            label = eval_cfg["label"]
            # 🔧 修复：从批次目录中读取评估结果
            if batch_dir is not None:
                eval_save_path = batch_dir / "evaluation_results" / f"{label}_{eval_cfg['terrain_sensing_mode']}"
            else:
                eval_save_path = Path(f"evaluation_results/{label}_{eval_cfg['terrain_sensing_mode']}")
            eval_results_json = eval_save_path / "evaluation_results.json"
            print(f"     - {eval_results_json} ({'存在' if eval_results_json.exists() else '不存在'})")
    
    print(f"\n{'='*70}")
    print(f"✅ 实验完成！")
    print(f"结果保存在: {results_file}")
    print(f"对比图保存在: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

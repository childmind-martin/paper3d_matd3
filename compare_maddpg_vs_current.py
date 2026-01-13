#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MADDPG vs 当前算法（MATD3+APF融合）对比实验脚本

对比两种算法：
1. MADDPG: 原始MADDPG算法，无势场修正
2. 当前算法: MATD3 + 可学习APF融合（run_optimized.sh默认配置）

所有实验使用相同的环境（固定位置+地形），确保公平对比。

使用方法：
    python compare_maddpg_vs_current.py --episodes 500 --batch-size 1024
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import time

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    HAS_MATPLOTLIB = True
    
    def setup_english_fonts():
        """设置英文字体"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
    
    setup_english_fonts()
except ImportError:
    print("缺少依赖，请安装：pip install matplotlib scipy")
    HAS_MATPLOTLIB = False


# 实验配置
EXPERIMENT_CONFIGS = [
    {
        "label": "maddpg_baseline",
        "name": "MADDPG Baseline",
        "name_en": "MADDPG Baseline",
        "description": "Original MADDPG algorithm without APF correction",
        "algorithm": "maddpg",
        "env": {
            # 禁用势场修正，使用纯MADDPG
            "ACTION_FORCE_RATIO": "0.0",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",
            "USE_TF_POTENTIAL_FIELD": "1",
            # 使用相同的随机种子确保公平对比
            "SEED": "252488",
            "TF_DETERMINISTIC_OPS": "1"
        }
    },
    {
        "label": "matd3_apf_fusion",
        "name": "MATD3+APF Fusion (Current)",
        "name_en": "MATD3+APF Fusion (Current)",
        "description": "MATD3 with learnable APF correction (run_optimized.sh default)",
        "algorithm": "matd3",
        "env": {
            # 使用run_optimized.sh的默认FR schedule
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10",
            "USE_TF_POTENTIAL_FIELD": "1",
            # 使用相同的随机种子确保公平对比
            "SEED": "252488",
            "TF_DETERMINISTIC_OPS": "1"
        }
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="MADDPG vs 当前算法对比实验")
    parser.add_argument("--script", type=str, default="./run_optimized.sh",
                        help="训练启动脚本路径")
    parser.add_argument("--episodes", type=int, default=150,
                        help="每个实验的训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="训练批次大小")
    parser.add_argument("--use-weighted-reward", type=int, default=1,
                        help="是否使用分项加权奖励")
    parser.add_argument("--output-dir", type=str, default="comparison_maddpg_outputs",
                        help="图表输出目录")
    parser.add_argument("--logs-root", type=str, default="logs",
                        help="训练日志根目录")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/5.json",
                        help="固定位置文件路径")
    parser.add_argument("--reuse", action="store_true",
                        help="复用已存在的实验结果")
    parser.add_argument("--smooth-window", type=int, default=20,
                        help="曲线平滑窗口大小")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="只运行指定的实验标签（例如: --only maddpg_baseline）")
    parser.add_argument("--skip", type=str, nargs="+", default=None,
                        help="跳过指定的实验标签（例如: --skip matd3_apf_fusion）")
    return parser.parse_args()


def setup_base_env_vars(positions_file: Path, args) -> dict:
    """设置基础环境变量"""
    env = os.environ.copy()
    
    # GPU配置
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["GPU_ID"] = "0"
    env["TF_GPU_ALLOCATOR"] = ""
    
    # XLA配置
    env["XLA_FLAGS"] = ""
    env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    
    # 基础训练配置
    env.setdefault("NUM_ENVS", "1")
    env.setdefault("XLA_GLOBAL", "1")
    env.setdefault("CPU_THREADS", "12")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("QUIET_OUTPUT", "0")
    env.setdefault("SUPPRESS_MA_PROMPT", "1")
    env.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")
    
    # 固定位置和地形配置
    env["USE_FIXED_POSITIONS"] = "1"
    env["DYNAMIC_FIRST_TIME"] = "0"
    env["POSITIONS_FILE"] = str(positions_file)
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    env["USE_SCENARIO_SEED"] = "1"
    env["SCENARIO_SEED"] = "88"
    
    # 地形配置
    env["TERRAIN_COMPLEXITY_LEVEL"] = "2"
    env["MAP_SIZE"] = "200"
    env["MOUNTAIN_MIN_DISTANCE"] = "55"
    env["TERRAIN_CONTACT_EPS"] = "0.2"
    
    return env


def find_latest_log_dir(exp_name: str, logs_root: str) -> str:
    """查找最新的日志目录"""
    logs_path = Path(logs_root)
    if not logs_path.exists():
        raise FileNotFoundError(f"日志根目录不存在: {logs_path}")
    
    matching_dirs = []
    for item in logs_path.iterdir():
        if item.is_dir() and item.name.startswith(exp_name + "_"):
            suffix = item.name[len(exp_name) + 1:]
            if len(suffix) >= 15 and suffix[8] == '_' and suffix[:8].isdigit():
                subdirs = sorted([d for d in item.iterdir() if d.is_dir() and d.name != 'evaluation'])
                timestamp_subdirs = [d for d in subdirs if len(d.name) >= 15 and d.name[8] == '_']
                if timestamp_subdirs:
                    matching_dirs.append((item.name, timestamp_subdirs[-1]))
                elif subdirs:
                    matching_dirs.append((item.name, subdirs[-1]))
                else:
                    matching_dirs.append((item.name, item))
    
    if not matching_dirs:
        raise FileNotFoundError(f"未找到以 '{exp_name}' 开头的日志目录")
    
    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    return str(matching_dirs[0][1])


def load_metrics(log_dir: str) -> Dict:
    """加载训练指标"""
    metrics = {}
    
    ep_path = Path(log_dir) / "episode_rewards.json"
    if ep_path.exists():
        with open(ep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metrics["episode_rewards"] = data.get("episode_rewards", [])
            metrics["success_flags"] = data.get("success_flags", [])
            metrics["collision_counts"] = data.get("collision_counts", [])
            metrics["min_distances_to_obstacle"] = data.get("min_distances_to_obstacle", [])
            metrics["team_success_flags"] = data.get("team_success_flags", [])
        else:
            metrics["episode_rewards"] = data
            metrics["success_flags"] = []
            metrics["collision_counts"] = []
            metrics["min_distances_to_obstacle"] = []
            metrics["team_success_flags"] = []
    
    loss_path = Path(log_dir) / "loss_history.json"
    if loss_path.exists():
        with open(loss_path, "r", encoding="utf-8") as f:
            metrics["loss_history"] = json.load(f)
    else:
        metrics["loss_history"] = []
    
    return metrics


def run_experiment(cfg: Dict, positions_file: Path, args) -> Dict:
    """运行单个实验"""
    label = cfg["label"]
    algorithm = cfg.get("algorithm", "matd3")
    
    env = setup_base_env_vars(positions_file, args)
    env.update(cfg.get("env", {}))
    
    # 强制禁用课程学习
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    
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
    print(f"  算法: {algorithm.upper()}")
    print(f"  描述: {cfg.get('description', '')}")
    print(f"  ACTION_FORCE_RATIO: {env.get('ACTION_FORCE_RATIO', '使用schedule')}")
    print(f"  ACTION_FORCE_RATIO_SCHEDULE_PCT: {env.get('ACTION_FORCE_RATIO_SCHEDULE_PCT', '默认')}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True, env=env)
        log_dir = find_latest_log_dir(label, args.logs_root)
        metrics = load_metrics(log_dir)
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", label),
            "description": cfg.get("description", ""),
            "algorithm": algorithm,
            "log_dir": log_dir,
            "metrics": metrics,
            "success": True
        }
    except Exception as e:
        print(f"[错误] 实验失败: {e}")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", label),
            "description": cfg.get("description", ""),
            "algorithm": algorithm,
            "log_dir": None,
            "metrics": {},
            "success": False
        }


def smooth_curve(data: np.ndarray, window: int = 10) -> np.ndarray:
    """平滑曲线"""
    if len(data) < 2:
        return data
    return uniform_filter1d(data.astype(float), size=window, mode='nearest')


def plot_comparison(series: List[Dict], output_dir: Path, smooth_window: int = 20):
    """绘制对比图表"""
    if not HAS_MATPLOTLIB:
        print("[跳过] matplotlib未安装，跳过绘图")
        return
    
    setup_english_fonts()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 颜色配置
    colors = {'maddpg_baseline': '#CC0000', 'matd3_apf_fusion': '#0066CC'}
    
    # 1. 奖励对比图
    fig, ax = plt.subplots(figsize=(14, 8))
    has_data = False
    
    for item in series:
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        episodes = range(1, len(rewards) + 1)
        rewards_array = np.array(rewards)
        color = colors.get(item["label"], '#666666')
        name = item.get("name_en", item["label"])
        
        ax.plot(episodes, rewards, label=f"{name} (Raw)", 
                color=color, alpha=0.3, linewidth=1)
        smoothed = smooth_curve(rewards_array, window=smooth_window)
        ax.plot(episodes, smoothed, label=f"{name} (Smoothed)", 
                color=color, alpha=0.9, linewidth=2.5)
    
    if has_data:
        ax.set_title("MADDPG vs MATD3+APF Fusion: Reward Comparison", 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Reward", fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        reward_path = output_dir / f"reward_comparison_{timestamp}.png"
        plt.savefig(reward_path, dpi=200, bbox_inches='tight')
        print(f"[完成] 奖励对比图: {reward_path}")
    plt.close(fig)
    
    # 2. 成功率和碰撞对比图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    has_data = False
    
    for item in series:
        metrics = item["metrics"]
        color = colors.get(item["label"], '#666666')
        name = item.get("name_en", item["label"])
        
        # 成功率
        success_flags = metrics.get("team_success_flags", metrics.get("success_flags", []))
        if success_flags:
            has_data = True
            episodes = range(1, len(success_flags) + 1)
            success_array = np.array(success_flags, dtype=float)
            window_size = 50
            success_rate = []
            for i in range(len(success_array)):
                start_idx = max(0, i - window_size + 1)
                rate = np.mean(success_array[start_idx:i+1])
                success_rate.append(rate)
            axes[0].plot(episodes, success_rate, label=name, 
                        color=color, linewidth=2.5, alpha=0.9)
        
        # 碰撞次数
        collision_counts = metrics.get("collision_counts", [])
        if collision_counts:
            episodes = range(1, len(collision_counts) + 1)
            collisions_array = np.array(collision_counts, dtype=float)
            smoothed = smooth_curve(collisions_array, window=smooth_window)
            axes[1].plot(episodes, smoothed, label=name, 
                        color=color, linewidth=2.5, alpha=0.9)
    
    if has_data:
        axes[0].set_title("Team Success Rate (Moving Average, Window=50)", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Success Rate", fontsize=12)
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper right', fontsize=11)
        
        axes[1].set_title("Collision Counts (Smoothed)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Episode", fontsize=12)
        axes[1].set_ylabel("Collision Count", fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        success_path = output_dir / f"success_collision_comparison_{timestamp}.png"
        plt.savefig(success_path, dpi=200, bbox_inches='tight')
        print(f"[完成] 成功率/碰撞对比图: {success_path}")
    plt.close(fig)
    
    # 3. Loss对比图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    has_data = False
    
    for item in series:
        history = item["metrics"].get("loss_history", [])
        if not history:
            continue
        has_data = True
        # 🔧 修复：正确处理null值，过滤掉null而不是替换为0
        # 这样可以看到真实的loss曲线，而不是被0值拉平的曲线
        critic_steps = []
        critic_values = []
        actor_steps = []
        actor_values = []
        
        for idx, entry in enumerate(history):
            step = entry.get("step", idx)
            c_loss = entry.get("critic_loss")
            a_loss = entry.get("actor_loss")
            # 只添加非null且有效的值
            if c_loss is not None:
                try:
                    c_val = float(c_loss)
                    if not (np.isnan(c_val) or np.isinf(c_val)):
                        critic_steps.append(step)
                        critic_values.append(c_val)
                except (ValueError, TypeError):
                    pass
            if a_loss is not None:
                try:
                    a_val = float(a_loss)
                    if not (np.isnan(a_val) or np.isinf(a_val)):
                        actor_steps.append(step)
                        actor_values.append(a_val)
                except (ValueError, TypeError):
                    pass
        
        # 绘制有效的loss数据
        color = colors.get(item["label"], '#666666')
        name = item.get("name_en", item["label"])
        if len(critic_steps) > 0:
            axes[0].plot(critic_steps, critic_values, label=f"{name}", color=color, linewidth=2, alpha=0.9)
        if len(actor_steps) > 0:
            axes[1].plot(actor_steps, actor_values, label=f"{name}", color=color, linewidth=2, alpha=0.9)
    
    if has_data:
        axes[0].set_title("Critic Loss", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper right', fontsize=11)
        
        axes[1].set_title("Actor Loss", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Update Step", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        loss_path = output_dir / f"loss_comparison_{timestamp}.png"
        plt.savefig(loss_path, dpi=200, bbox_inches='tight')
        print(f"[完成] Loss对比图: {loss_path}")
    plt.close(fig)
    
    return timestamp


def main():
    args = parse_args()
    
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"[错误] 找不到训练脚本: {script_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    positions_file = Path(args.positions_file).resolve()
    if not positions_file.exists():
        print(f"[警告] 固定位置文件不存在: {positions_file}")
        print(f"[提示] 请先运行一次训练生成位置文件，或指定已存在的位置文件")
    
    print(f"\n{'='*70}")
    print(f"MADDPG vs 当前算法（MATD3+APF融合）对比实验")
    print(f"实验数量: {len(EXPERIMENT_CONFIGS)}")
    for cfg in EXPERIMENT_CONFIGS:
        print(f"  - {cfg['name']}: {cfg['description']}")
    print(f"训练回合数: {args.episodes}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    # 🔧 新增：支持只运行指定实验或跳过指定实验
    configs_to_run = EXPERIMENT_CONFIGS.copy()
    configs_to_compare = EXPERIMENT_CONFIGS.copy()  # 用于对比的所有配置
    
    if args.only:
        configs_to_run = [cfg for cfg in configs_to_run if cfg["label"] in args.only]
        print(f"[信息] 只运行以下实验: {', '.join(args.only)}")
        # 🔧 修复：如果使用--reuse，自动加载其他已有实验结果进行对比
        if args.reuse:
            configs_to_compare = EXPERIMENT_CONFIGS.copy()  # 对比时包含所有配置
            print(f"[信息] 使用--reuse，将自动加载其他已有实验结果进行对比")
    if args.skip:
        configs_to_run = [cfg for cfg in configs_to_run if cfg["label"] not in args.skip]
        configs_to_compare = [cfg for cfg in configs_to_compare if cfg["label"] not in args.skip]
        print(f"[信息] 跳过以下实验: {', '.join(args.skip)}")
    
    if not configs_to_run:
        print("[错误] 没有可运行的实验配置")
        return
    
    series = []
    # 🔧 修复：先加载所有需要对比的已有实验结果（如果使用--reuse）
    if args.reuse:
        for cfg in configs_to_compare:
            if cfg["label"] not in [c["label"] for c in configs_to_run]:  # 跳过需要重新运行的
                try:
                    log_dir = find_latest_log_dir(cfg["label"], args.logs_root)
                    metrics = load_metrics(log_dir)
                    print(f"[复用] 使用已有实验结果: {cfg['label']} (来自 {log_dir})")
                    series.append({
                        "label": cfg["label"],
                        "name": cfg.get("name", cfg["label"]),
                        "name_en": cfg.get("name_en", cfg["label"]),
                        "description": cfg.get("description", ""),
                        "algorithm": cfg.get("algorithm", "matd3"),
                        "log_dir": log_dir,
                        "metrics": metrics,
                        "success": True
                    })
                except (FileNotFoundError, ValueError):
                    print(f"[警告] 未找到已有实验结果: {cfg['label']}，将不会包含在对比中")
    
    # 🔧 修复：运行需要重新运行的实验
    # 如果使用--only，强制重新运行，不复用已有结果
    # 如果只使用--reuse（没有--only），则尝试复用已有结果
    for cfg in configs_to_run:
        # 如果使用--only，强制重新运行，不检查已有结果
        if args.only:
            print(f"[运行] 强制重新运行实验: {cfg['label']} (使用--only，不复用已有结果)")
        elif args.reuse:
            # 如果只使用--reuse（没有--only），尝试复用已有结果
            try:
                log_dir = find_latest_log_dir(cfg["label"], args.logs_root)
                metrics = load_metrics(log_dir)
                print(f"[复用] 使用已有实验结果: {cfg['label']} (来自 {log_dir})")
                series.append({
                    "label": cfg["label"],
                    "name": cfg.get("name", cfg["label"]),
                    "name_en": cfg.get("name_en", cfg["label"]),
                    "description": cfg.get("description", ""),
                    "algorithm": cfg.get("algorithm", "matd3"),
                    "log_dir": log_dir,
                    "metrics": metrics,
                    "success": True
                })
                continue
            except (FileNotFoundError, ValueError):
                print(f"[信息] 未找到已有实验结果，将运行新实验: {cfg['label']}")
        
        # 运行新实验
        print(f"[运行] 开始运行新实验: {cfg['label']}")
        result = run_experiment(cfg, positions_file, args)
        if result["success"]:
            series.append(result)
        else:
            print(f"[警告] 实验 {cfg['label']} 失败，跳过")
    
    if series:
        timestamp = plot_comparison(series, output_dir, args.smooth_window)
        
        # 保存汇总
        summary = {
            "timestamp": timestamp,
            "experiments": [
                {
                    "label": item["label"],
                    "name": item["name"],
                    "algorithm": item["algorithm"],
                    "log_dir": item.get("log_dir", ""),
                    "final_reward": item["metrics"].get("episode_rewards", [])[-1] if item["metrics"].get("episode_rewards") else None,
                    "avg_reward": float(np.mean(item["metrics"].get("episode_rewards", []))) if item["metrics"].get("episode_rewards") else None,
                    "max_reward": float(np.max(item["metrics"].get("episode_rewards", []))) if item["metrics"].get("episode_rewards") else None,
                }
                for item in series
            ]
        }
        
        summary_path = output_dir / f"summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*70}")
        print(f"对比实验完成!")
        print(f"输出目录: {output_dir}")
        print(f"\n实验结果汇总:")
        for exp in summary["experiments"]:
            final = f"{exp['final_reward']:.2f}" if exp['final_reward'] else "N/A"
            avg = f"{exp['avg_reward']:.2f}" if exp['avg_reward'] else "N/A"
            max_r = f"{exp['max_reward']:.2f}" if exp['max_reward'] else "N/A"
            print(f"  - {exp['name']} ({exp['algorithm'].upper()}):")
            print(f"      Final={final}, Avg={avg}, Max={max_r}")
        print(f"{'='*70}")
    else:
        print("[错误] 所有实验都失败了")
        sys.exit(1)


if __name__ == "__main__":
    main()

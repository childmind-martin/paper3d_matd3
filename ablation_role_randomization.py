#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色洗牌 (Role Randomization) 消融对比实验

对比情况：
1. 禁用角色洗牌 (baseline_no_shuffle): 网络索引与物理初始位置/目标强制绑定（可能会出现单一智能体高空挂机）。
2. 启用角色洗牌 (role_shuffle_enabled): 每回合随机打乱网络索引与物理属性的映射关系。

调用底层的 run_optimized.sh 进行训练测试。
"""

import argparse
import sys
import os
import time
import subprocess
from pathlib import Path

# 导入基础消融实验模块，复用绘图与指标加载功能
from ablation_action_pf_comparison import (
    AblationBatchManager,
    plot_comparison_rewards,
    plot_comparison_success_collision_clearance,
    plot_comparison_losses,
    setup_english_fonts,
    find_latest_log_dir,
    SCENARIO_SEED,
    TRAINING_SEED,
    TERRAIN_COMPLEXITY_LEVEL,
    MAP_SIZE,
    MOUNTAIN_MIN_DISTANCE,
    ensure_fixed_positions,
    load_metrics
)

def _extract_exp_name_with_timestamp(log_path: Path):
    try:
        if not log_path.exists():
            return None
        exp_name = None
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "带时间戳的实验名称" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        candidate = parts[1].strip()
                        if candidate:
                            exp_name = candidate
        return exp_name
    except Exception:
        return None

def _resolve_run_log_dir(project_logs_root: Path, exp_name_with_ts: str):
    try:
        base_dir = project_logs_root / exp_name_with_ts
        if not base_dir.exists():
            return None
        subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name != "evaluation"]
        if not subdirs:
            return str(base_dir)
        # 优先使用时间戳格式子目录
        timestamp_subdirs = [
            d for d in subdirs
            if len(d.name) >= 15 and d.name[8] == '_' and d.name[:8].isdigit() and d.name[9:15].isdigit()
        ]
        if timestamp_subdirs:
            timestamp_subdirs.sort(key=lambda d: d.name, reverse=True)
            return str(timestamp_subdirs[0])
        # 否则按修改时间选择
        subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return str(subdirs[0])
    except Exception:
        return None

# ============================================================================
# 角色洗牌消融配置
# ============================================================================
EXPERIMENT_CONFIGS = [
    {
        "label": "baseline_no_shuffle",
        "name": "No Role Shuffle",
        "name_en": "No Role Shuffle",
        "description": "禁用角色随机化（可能导致某个智能体陷入高空挂机的局部最优）",
        "env": {
            "ENABLE_ROLE_SHUFFLE": "0"  # 关闭洗牌
        }
    },
    {
        "label": "role_shuffle_enabled",
        "name": "With Role Shuffle",
        "name_en": "With Role Shuffle",
        "description": "启用角色随机化（每回合随机打乱智能体和位置/目标的映射）",
        "env": {
            "ENABLE_ROLE_SHUFFLE": "1"  # 开启洗牌
        }
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="角色洗牌 (Role Randomization) 消融对比实验")
    parser.add_argument("--script", type=str, default="./run_optimized.sh",
                        help="训练启动脚本路径 (默认 ./run_optimized.sh)")
    parser.add_argument("--positions-file", type=str, default=None,
                        help="固定位置文件路径（None则使用默认值 ./saved_positions/5.json，与 run_optimized.sh 一致）")
    parser.add_argument("--episodes", type=int, default=120,
                        help="每个实验的训练回合数（默认400）")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="训练批次大小")
    parser.add_argument("--use-weighted-reward", type=int, default=1, choices=[0, 1],
                        help="是否使用分项加权奖励")
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3"],
                        help="训练算法选择")
    parser.add_argument("--output-dir", type=str, default="ablation_role_shuffle_outputs",
                        help="图表输出目录")
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="拟合曲线平滑窗口大小")
    parser.add_argument("--fit-method", type=str, default="moving_average",
                        choices=["moving_average", "spline", "poly"],
                        help="拟合方法")
    return parser.parse_args()

def run_experiment(cfg, args, batch_dir, positions_file):
    label = cfg["label"]
    print(f"\n{'='*70}")
    print(f"🚀 开始运行实验: {cfg['name']}")
    print(f"{'='*70}")

    env = os.environ.copy()
    
    # 强制固定环境和种子以便公平对比
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    env["USE_SCENARIO_SEED"] = "1"
    env["SCENARIO_SEED"] = str(SCENARIO_SEED)
    env["SEED"] = str(TRAINING_SEED)
    env["TF_DETERMINISTIC_OPS"] = "1"
    env["TQDM_DISABLE"] = "0"      # 强制开启进度条
    env["TQDM_TO_STDOUT"] = "1"    # 强制进度条输出到标准输出
    
    # === 关键：启用严格的单一变量位置控制 ===
    env["USE_FIXED_POSITIONS"] = "1"
    env["DYNAMIC_FIRST_TIME"] = "1"
    env["POSITIONS_FILE"] = str(positions_file)
    
    # 注入该配置特定的环境变量
    for k, v in cfg["env"].items():
        env[k] = v

    exp_dir = Path(batch_dir) / label
    exp_dir.mkdir(parents=True, exist_ok=True)
    env["EXP_NAME"] = f"../{Path(batch_dir).name}/{label}"
    env["QUIET_OUTPUT"] = "0"
    
    cmd = [
        args.script,
        str(args.episodes),
        str(args.batch_size),
        label,
        str(args.use_weighted_reward),
        args.algorithm
    ]
    
    print(f"[INFO] 训练开始，输出将直接显示在控制台...")
    
    start_time = time.time()
    try:
        # 直接运行，不重定向输出，让它直接显示在控制台（解决 tqdm 的 \\r 刷新卡死管道问题）
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path(args.script).parent,
            capture_output=False,
            text=True,
            check=False
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0
        
        project_logs_root = Path(args.script).resolve().parent / "logs"
        
        # 解析日志目录：训练脚本写入路径为 logs/{exp_name}/{timestamp}/episode_rewards.json，
        # 必须解析到包含 episode_rewards.json 的内层目录，load_metrics 才能读到数据。
        def find_latest_log_dir_fallback(label_name, root_dir):
            try:
                base_dir = Path(root_dir)
                if not base_dir.exists():
                    return None
                prefix = f"{label_name}_"
                candidates = []
                for d in base_dir.iterdir():
                    if d.is_dir() and d.name.startswith(prefix):
                        candidates.append(d)
                if not candidates:
                    return None
                candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(candidates[0])
            except Exception:
                return None

        try:
            # 优先使用与 action_pf 消融一致的解析逻辑，返回含 episode_rewards.json 的内层目录
            log_dir_found = find_latest_log_dir(label, str(project_logs_root))
        except FileNotFoundError:
            # 无完整结果目录时回退到顶层目录，再尝试解析到含 episode_rewards.json 的子目录
            log_dir_found = find_latest_log_dir_fallback(label, project_logs_root)
            if log_dir_found:
                base = Path(log_dir_found)
                if not (base / "episode_rewards.json").exists():
                    for sub in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                        if sub.is_dir() and (sub / "episode_rewards.json").exists():
                            log_dir_found = str(sub)
                            break
        if not log_dir_found:
            log_dir_found = str(exp_dir)

        print(f"[INFO] 从目录加载指标: {log_dir_found}")
        metrics = load_metrics(log_dir_found)
        
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": str(log_dir_found),
            "metrics": metrics,
            "success": success,
            "elapsed": elapsed
        }

    except Exception as e:
        print(f"[ERROR] 实验 {label} 失败: {e}")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": str(exp_dir),
            "metrics": {},
            "success": False,
            "elapsed": time.time() - start_time
        }


def main():
    # 设置字体避免图表出现方块
    try:
        import matplotlib
        matplotlib.use('Agg')
        setup_english_fonts()
    except ImportError:
        print("缺少依赖 matplotlib")
        sys.exit(1)

    args = parse_args()
    
    # 🔧 新增：生成或获取固定位置文件
    exp_name_prefix = "ablation_role_randomization"
    positions_file = ensure_fixed_positions(args.positions_file, args, exp_name_prefix)
    
    manager = AblationBatchManager()
    batch_config = {
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "algorithm": args.algorithm,
        "use_weighted_reward": args.use_weighted_reward,
        "seed": TRAINING_SEED,
        "scenario_seed": SCENARIO_SEED,
        "terrain_complexity": TERRAIN_COMPLEXITY_LEVEL,
        "map_size": MAP_SIZE,
        "mountain_min_distance": MOUNTAIN_MIN_DISTANCE,
        "positions_file": str(positions_file),
        "notes": "角色洗牌 (Role Randomization) 消融实验"
    }
    batch_dir = manager.create_batch(config=batch_config, experiments=[cfg["label"] for cfg in EXPERIMENT_CONFIGS])
    
    print("\n" + "="*60)
    print("🚀 启动 角色洗牌 (Role Randomization) 消融对比实验")
    print(f"✅ 严格单变量控制：统一使用固定位置文件 {positions_file}")
    print("="*60)
    
    results = []
    for cfg in EXPERIMENT_CONFIGS:
        res = run_experiment(cfg, args, batch_dir, positions_file)
        if res.get("success", False) or res.get("metrics", {}).get("episode_rewards"):
            results.append(res)
    
    print("\n" + "="*60)
    print("📊 正在生成对比图表...")
    print("="*60)
    
    output_dir = Path(batch_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        title = "Role Randomization Ablation Comparison"
        plot_comparison_rewards(results, title, output_dir / "rewards_comparison.png", smooth_window=args.smooth_window, fit_method=args.fit_method)
        plot_comparison_success_collision_clearance(results, title, output_dir / "metrics_comparison.png", smooth_window=args.smooth_window, fit_method=args.fit_method)
        plot_comparison_losses(results, title, output_dir / "losses_comparison.png")
        print(f"\n✅ 所有图表已保存至: {output_dir}/")
        
        # 为了兼容性，也可以在 args.output_dir 下拷贝一份
        os.makedirs(args.output_dir, exist_ok=True)
        import shutil
        for f in output_dir.glob("*.png"):
            shutil.copy(f, Path(args.output_dir) / f.name)
            
    except Exception as e:
        print(f"\n❌ 图表生成失败: {e}")

if __name__ == "__main__":
    main()
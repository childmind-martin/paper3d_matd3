#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定FR比例消融对比实验

对比不同固定FR比例（0.55, 0.65, 0.75, 0.85, 0.95）以及动态FR schedule的实验效果。
所有实验完全使用run_optimized.sh的默认配置（包括地图和位置），确保公平对比。

实验配置：
1. 固定FR比例实验（5个）：
   - FR = 0.55: 网络动作45%，势场动作55%
   - FR = 0.65: 网络动作35%，势场动作65%
   - FR = 0.75: 网络动作25%，势场动作75%
   - FR = 0.85: 网络动作15%，势场动作85%
   - FR = 0.95: 网络动作5%，势场动作95%

2. 动态FR schedule实验（1个）：
   - FR schedule: 0%:0.75 → 20%:0.70 → 40%:0.65 → 60%:0.55 → 80%:0.45 → 95%:0.30
   - FR从0.75逐渐降低到0.30，让网络逐渐减少对势场的依赖

重要说明：
- 所有实验完全使用run_optimized.sh的默认配置，包括：
  * 地图配置（TERRAIN_COMPLEXITY_LEVEL=1, MAP_SIZE=200, MOUNTAIN_MIN_DISTANCE=55）
  * 位置配置（USE_FIXED_POSITIONS=1, POSITIONS_FILE=./saved_positions/5.json）
  * 场景随机种子（USE_SCENARIO_SEED=1, SCENARIO_SEED=41）
- 所有实验使用相同的训练随机种子（默认1337，可通过--seed参数覆盖）
  * 这确保了Python、NumPy、TensorFlow的随机数生成器使用相同的初始状态
  * 消除了随机性对实验结果的影响，确保公平对比
- 不单独生成位置文件，确保所有实验使用相同的地图和位置
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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


# 实验配置：固定FR比例
FR_RATIOS = [0.25, 0.35, 0.45, 0.65, 0.85, 0.95]

EXPERIMENT_CONFIGS = []
for fr_ratio in FR_RATIOS:
    network_pct = (1.0 - fr_ratio) * 100
    pf_pct = fr_ratio * 100
    EXPERIMENT_CONFIGS.append({
        "label": f"fr_{fr_ratio:.2f}",
        "name": f"FR={fr_ratio:.2f}",
        "name_en": f"FR={fr_ratio:.2f}",
        "description": f"固定FR比例：网络动作{network_pct:.0f}%，势场动作{pf_pct:.0f}%",
        "env": {
            "ACTION_FORCE_RATIO": str(fr_ratio),  # 固定FR比例
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",  # 🚨 关键：禁用schedule，确保FR固定
            "USE_TF_POTENTIAL_FIELD": "1"  # 启用TF版本势场修正
        }
    })

# 添加使用schedule方法的实验
# 默认schedule: 0%:0.75,20%:0.70,40%:0.65,60%:0.55,80%:0.45,95%:0.30
# FR从0.75逐渐降低到0.30，让网络逐渐减少对势场的依赖
EXPERIMENT_CONFIGS.append({
    "label": "fr_schedule",
    "name": "FR Schedule",
    "name_en": "FR Schedule",
    "description": "动态FR schedule：0%:0.75→20%:0.70→40%:0.65→60%:0.55→80%:0.45→95%:0.30",
    "env": {
        "ACTION_FORCE_RATIO": "0.75",  # 初始FR值（schedule的起始值）
        # 🚨 关键：不设置ACTION_FORCE_RATIO_SCHEDULE_PCT为"DISABLED"
        # 而是使用默认schedule或显式设置，让run_optimized.sh使用默认schedule
        # 如果环境变量未设置，run_optimized.sh会使用默认值："0%:0.75,20%:0.70,40%:0.65,60%:0.55,80%:0.45,95%:0.30"
        "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.75,20%:0.70,40%:0.65,60%:0.55,80%:0.45,95%:0.30",  # 显式设置schedule
        "USE_TF_POTENTIAL_FIELD": "1"  # 启用TF版本势场修正
    }
})


def find_latest_log_dir(exp_name: str, logs_root: str) -> str:
    """查找最新的日志目录"""
    logs_path = Path(logs_root)
    if not logs_path.exists() or not logs_path.is_dir():
        raise FileNotFoundError(f"日志根目录不存在: {logs_path}")
    
    # 🔧 修复：训练脚本会创建带时间戳的目录
    matching_dirs = []
    for item in logs_path.iterdir():
        if item.is_dir() and item.name.startswith(exp_name):
            subdirs = sorted([d for d in item.iterdir() if d.is_dir()])
            if subdirs:
                matching_dirs.append((item.name, subdirs[-1]))
            else:
                matching_dirs.append((item.name, item))
    
    if not matching_dirs:
        raise FileNotFoundError(f"未找到以 '{exp_name}' 开头的日志目录: {logs_path}")
    
    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_dir = matching_dirs[0][1]
    return str(latest_dir)


def load_metrics(log_dir: str) -> Dict:
    """加载训练指标"""
    metrics = {}
    
    # 加载奖励数据
    ep_path = Path(log_dir) / "episode_rewards.json"
    if ep_path.exists():
        with open(ep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metrics["episode_rewards"] = data.get("episode_rewards", [])
        else:
            metrics["episode_rewards"] = data
    else:
        metrics["episode_rewards"] = []
    
    # 加载损失数据
    loss_path = Path(log_dir) / "losses.json"
    if loss_path.exists():
        with open(loss_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metrics["actor_loss"] = data.get("actor_loss", [])
            metrics["critic_loss"] = data.get("critic_loss", [])
        else:
            metrics["actor_loss"] = []
            metrics["critic_loss"] = []
    else:
        metrics["actor_loss"] = []
        metrics["critic_loss"] = []
    
    return metrics


def run_experiment(config: Dict, args) -> Dict:
    """运行单个实验"""
    label = config["label"]
    name = config["name"]
    env_vars = config["env"]
    
    print(f"\n{'='*60}")
    print(f"[并行训练-{label}] 开始: {name}")
    if 'schedule' in label.lower():
        schedule_value = env_vars.get('ACTION_FORCE_RATIO_SCHEDULE_PCT', 'N/A')
        print(f"[并行训练-{label}] 🔧 Schedule配置: {schedule_value}")
    print(f"{'='*60}")
    
    # 准备环境变量（继承当前环境，然后覆盖）
    # 🔧 关键：完全使用run_optimized.sh的默认配置，不覆盖地图和位置相关参数
    # run_optimized.sh的默认配置：
    #   - USE_FIXED_POSITIONS=1 (默认启用固定位置)
    #   - POSITIONS_FILE=./saved_positions/5.json (默认位置文件)
    #   - USE_SCENARIO_SEED=1 (使用固定随机种子)
    #   - SCENARIO_SEED=41 (地图随机种子)
    #   - TERRAIN_COMPLEXITY_LEVEL=1 (地形复杂度)
    #   - MAP_SIZE=200 (地图大小)
    #   - MOUNTAIN_MIN_DISTANCE=55 (山峰最小距离)
    #   - RANDOM_TERRAIN=0 (不使用随机地形)
    # 这些参数会从run_optimized.sh继承，确保所有实验使用相同的地图和位置
    env = os.environ.copy()
    
    # 🔧 关键：设置统一的随机种子，确保所有实验的可重复性和公平对比
    # 使用固定的随机种子，消除随机性对实验结果的影响
    # 注意：这个种子会传递给训练脚本，用于初始化Python、NumPy、TensorFlow的随机数生成器
    # 🔧 修复：确保使用args.seed（默认1337），与命令行参数一致
    env["SEED"] = str(args.seed)  # 默认使用1337，可通过命令行参数覆盖
    # 🔧 关键：同时设置场景随机种子，确保地形生成一致
    env["USE_SCENARIO_SEED"] = "1"
    env["SCENARIO_SEED"] = "41"  # 场景生成种子，确保所有实验使用相同的地形
    
    # 🔧 关键：不设置地图和位置相关参数，让run_optimized.sh使用其默认值
    # 这样所有实验都会使用相同的地图和位置配置
    
    # 应用实验特定的环境变量
    for key, value in env_vars.items():
        env[key] = value
    
    # 构建训练命令
    exp_name = f"fixed_fr_{label}"
    cmd = [
        args.script,
        str(args.episodes),
        str(args.batch_size),
        exp_name,
        str(args.use_weighted_reward),
        args.algorithm,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # 显示实时输出
            text=True
        )
        
        # 查找日志目录
        log_dir = find_latest_log_dir(exp_name, args.logs_root)
        metrics = load_metrics(log_dir)
        
        print(f"[并行训练-{label}] ✓ 完成: {name}")
        return {
            "label": label,
            "name": name,
            "description": config.get("description", ""),
            "log_dir": log_dir,
            "metrics": metrics,
            "success": True
        }
    except subprocess.CalledProcessError as e:
        print(f"[并行训练-{label}] ✗ 失败: {name}, 错误码: {e.returncode}")
        return {
            "label": label,
            "name": name,
            "description": config.get("description", ""),
            "log_dir": None,
            "metrics": {},
            "success": False
        }
    except Exception as e:
        print(f"[并行训练-{label}] ✗ 异常: {name}, 错误: {e}")
        return {
            "label": label,
            "name": name,
            "description": config.get("description", ""),
            "log_dir": None,
            "metrics": {},
            "success": False
        }


def plot_comparison_rewards(series: List[Dict], title: str, output_path: Path, smooth_window: int = 10, fit_method: str = "poly"):
    """绘制奖励对比图"""
    if not HAS_MATPLOTLIB:
        print("警告：matplotlib不可用，跳过绘图")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
    
    for i, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        
        episodes = np.arange(1, len(rewards) + 1)
        label = item["name"]
        
        # 原始数据（半透明）
        ax.plot(episodes, rewards, alpha=0.2, color=colors[i], linewidth=0.5)
        
        # 平滑数据
        if len(rewards) >= smooth_window:
            if fit_method == "poly":
                # 多项式拟合
                z = np.polyfit(episodes, rewards, min(3, len(rewards) - 1))
                p = np.poly1d(z)
                smoothed = p(episodes)
            else:
                # 移动平均
                smoothed = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
            ax.plot(episodes, smoothed, label=label, color=colors[i], linewidth=2)
        else:
            ax.plot(episodes, rewards, label=label, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Episode", fontsize=12, fontfamily='DejaVu Sans')
    ax.set_ylabel("Episode Reward", fontsize=12, fontfamily='DejaVu Sans')
    ax.set_title(title, fontsize=14, fontweight="bold", fontfamily='DejaVu Sans')
    # 只在有数据时才显示图例
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(loc="best", fontsize=10, prop={'family': 'DejaVu Sans'})
        # Ensure all legend text uses English font
        if legend:
            for text in legend.get_texts():
                text.set_fontfamily('DejaVu Sans')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"奖励对比图已保存: {output_path}")


def plot_comparison_losses(series: List[Dict], title: str, output_path: Path):
    """绘制损失对比图"""
    if not HAS_MATPLOTLIB:
        print("警告：matplotlib不可用，跳过绘图")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
    
    for i, item in enumerate(series):
        label = item["name"]
        actor_loss = item["metrics"].get("actor_loss", [])
        critic_loss = item["metrics"].get("critic_loss", [])
        
        if actor_loss:
            steps = np.arange(1, len(actor_loss) + 1)
            ax1.plot(steps, actor_loss, label=label, color=colors[i], linewidth=1.5, alpha=0.7)
        
        if critic_loss:
            steps = np.arange(1, len(critic_loss) + 1)
            ax2.plot(steps, critic_loss, label=label, color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel("Training Step", fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_ylabel("Actor Loss", fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_title("Actor Loss Comparison", fontsize=14, fontweight="bold", fontfamily='DejaVu Sans')
    # 只在有数据时才显示图例
    if ax1.get_legend_handles_labels()[0]:
        legend1 = ax1.legend(loc="best", fontsize=9, prop={'family': 'DejaVu Sans'})
        # Ensure all legend text uses English font
        if legend1:
            for text in legend1.get_texts():
                text.set_fontfamily('DejaVu Sans')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Training Step", fontsize=12, fontfamily='DejaVu Sans')
    ax2.set_ylabel("Critic Loss", fontsize=12, fontfamily='DejaVu Sans')
    ax2.set_title("Critic Loss Comparison", fontsize=14, fontweight="bold", fontfamily='DejaVu Sans')
    # 只在有数据时才显示图例
    if ax2.get_legend_handles_labels()[0]:
        legend2 = ax2.legend(loc="best", fontsize=9, prop={'family': 'DejaVu Sans'})
        # Ensure all legend text uses English font
        if legend2:
            for text in legend2.get_texts():
                text.set_fontfamily('DejaVu Sans')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"损失对比图已保存: {output_path}")


def generate_interactive_comparison(series: List[Dict], title: str, output_path: Path, smooth_window: int = 10, fit_method: str = "poly"):
    """生成交互式对比图"""
    if not HAS_PLOTLY:
        print("警告：plotly不可用，跳过交互图生成")
        return
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        
        episodes = np.arange(1, len(rewards) + 1)
        label = item["name"]
        
        # 平滑数据
        if len(rewards) >= smooth_window:
            if fit_method == "poly":
                z = np.polyfit(episodes, rewards, min(3, len(rewards) - 1))
                p = np.poly1d(z)
                smoothed = p(episodes)
            else:
                smoothed = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
        else:
            smoothed = rewards
        
        fig.add_trace(go.Scatter(
            x=episodes.tolist(),
            y=smoothed.tolist(),
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Episode Reward",
        hovermode='closest',
        width=1200,
        height=600
    )
    
    pyo.plot(fig, filename=str(output_path), auto_open=False)
    print(f"交互式对比图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="固定FR比例消融对比实验")
    parser.add_argument("--script", type=str, default="./run_optimized.sh", help="训练脚本路径")
    parser.add_argument("--episodes", type=int, default=20, help="训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024, help="批次大小")
    parser.add_argument("--use-weighted-reward", type=int, default=1, help="是否使用加权奖励")
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3"], help="训练算法")
    parser.add_argument("--logs-root", type=str, default="./logs", help="日志根目录")
    parser.add_argument("--output-dir", type=str, default="./ablation_fixed_fr_outputs", help="输出目录")
    parser.add_argument("--positions-file", type=Path, default=None, help="固定位置文件路径（可选）")
    parser.add_argument("--smooth-window", type=int, default=10, help="平滑窗口大小")
    parser.add_argument("--fit-method", type=str, default="poly", choices=["poly", "moving"], help="拟合方法")
    parser.add_argument("--generate-interactive", action="store_true", help="生成交互式图表")
    parser.add_argument("--parallel", action="store_true", default=True, help="并行运行实验")
    parser.add_argument("--max-workers", type=int, default=None, help="最大并行worker数（默认=CPU核心数）")
    parser.add_argument("--seed", type=int, default=1337, help="随机种子（默认1337，所有实验使用相同种子确保可重复性）")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("固定FR比例消融对比实验")
    print("="*60)
    print(f"实验配置数量: {len(EXPERIMENT_CONFIGS)}")
    print("实验列表:")
    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        label = config['label']
        name = config['name']
        desc = config.get('description', '')
        if 'schedule' in label.lower():
            print(f"  {i}. {name}: {desc}")
        else:
            fr_ratio = config['env'].get('ACTION_FORCE_RATIO', 'N/A')
            print(f"  {i}. {name}: {desc} (固定FR={fr_ratio})")
    print(f"训练回合数: {args.episodes}")
    print(f"批次大小: {args.batch_size}")
    print(f"算法: {args.algorithm}")
    print(f"随机种子: {args.seed} (所有实验使用相同种子，确保可重复性)")
    print("="*60)
    
    # 🔧 关键：不生成位置文件，完全使用run_optimized.sh的默认配置
    # run_optimized.sh会使用其默认的位置和地图配置，确保所有实验一致
    
    # 运行实验
    results = []
    if args.parallel:
        max_workers = args.max_workers or min(len(EXPERIMENT_CONFIGS), mp.cpu_count())
        print(f"\n[并行训练] 使用 {max_workers} 个worker并行运行 {len(EXPERIMENT_CONFIGS)} 个实验\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_experiment, config, args): config
                for config in EXPERIMENT_CONFIGS
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        print(f"\n[串行训练] 依次运行 {len(EXPERIMENT_CONFIGS)} 个实验\n")
        for config in EXPERIMENT_CONFIGS:
            result = run_experiment(config, args)
            results.append(result)
    
    # 过滤成功的结果
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    print(f"\n{'='*60}")
    print(f"实验完成: {len(successful_results)} 成功, {len(failed_results)} 失败")
    print(f"{'='*60}\n")
    
    if failed_results:
        print("失败的实验:")
        for r in failed_results:
            print(f"  - {r['name']}: {r.get('log_dir', 'N/A')}")
        print()
    
    if not successful_results:
        print("没有成功的实验，无法生成对比图")
        return
    
    # 生成对比图
    title = f"Fixed FR Ratio Comparison ({args.algorithm.upper()})"
    series = successful_results
    
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    reward_png_name = f"reward_comparison_{timestamp}.png"
    reward_png = output_dir / reward_png_name
    plot_comparison_rewards(
        series, 
        title, 
        reward_png,
        smooth_window=args.smooth_window,
        fit_method=args.fit_method
    )
    
    loss_png_name = f"loss_comparison_{timestamp}.png"
    loss_png = output_dir / loss_png_name
    plot_comparison_losses(series, title, loss_png)
    
    interactive_html_name = f"interactive_comparison_{timestamp}.html"
    interactive_html = output_dir / interactive_html_name
    if args.generate_interactive:
        generate_interactive_comparison(
            series,
            title,
            interactive_html,
            smooth_window=args.smooth_window,
            fit_method=args.fit_method
        )
    
    # 保存汇总JSON
    summary_filename = f"summary_{timestamp}.json"
    summary_path = output_dir / summary_filename
    
    output_files = {
        "reward_plot": str(reward_png.name),
        "loss_plot": str(loss_png.name),
        "interactive_plot": str(interactive_html.name) if args.generate_interactive else None,
        "summary_file": str(summary_filename)
    }
    
    full_summary = {
        "timestamp": timestamp,
        "output_files": output_files,
        "experiments": [
            {
                "label": item["label"],
                "name": item["name"],
                "description": item["description"],
                "log_dir": item["log_dir"],
                "final_reward": item["metrics"].get("episode_rewards", [])[-1] if item["metrics"].get("episode_rewards") else None,
                "avg_reward": np.mean(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
                "max_reward": np.max(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
            }
            for item in series
        ]
    }
    
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(full_summary, f_summary, ensure_ascii=False, indent=2)
    
    # 创建一个指向最新summary的软链接
    latest_summary_path = output_dir / "latest_summary.json"
    if latest_summary_path.exists():
        latest_summary_path.unlink()
    latest_summary_path.symlink_to(summary_filename)
    
    print(f"\n{'='*60}")
    print("实验汇总已保存:")
    print(f"  - 奖励对比图: {reward_png}")
    print(f"  - 损失对比图: {loss_png}")
    if args.generate_interactive:
        print(f"  - 交互式图表: {interactive_html}")
    print(f"  - 汇总JSON: {summary_path}")
    print(f"  - 最新汇总: {latest_summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


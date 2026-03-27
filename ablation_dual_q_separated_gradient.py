#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
围绕 MATD3 主线方法的模块消融实验。

默认严格实验只运行 3 组 MATD3 主线相关配置：
1. MATD3 Mainline (Separated)
2. MATD3 Ablation - Unified Actor Loss
3. MATD3 Ablation - Single Q

可选参考实验默认不参与主线模块归因：
1. MADDPG Separated Gradient
2. MADDPG Dual Q
3. MADDPG Baseline

严格版默认策略：
- 保持主线 MATD3 separated-gradient 不动
- 其他 MATD3 实验共享同一个外层 update skeleton，只关闭少数模块
- 默认禁用课程学习（UNLOCK_ENV_ON_SUCCESS=0, UNLOCK_ENV_ON_PLATEAU=0）
- 默认预生成固定位置文件，不依赖 dynamic_first_time
- 默认固定地形、固定位置、固定障碍物，确保跨实验环境完全一致
- 当前脚本仅支持稳定串行运行，不再提供并行训练入口

关键配置说明（由 --config-mode 控制）：
- strict_ablation（默认）：严格固定环境 + run_optimized 默认超参 + 课程学习关闭
- run_optimized_default：兼容旧逻辑，尽量贴近 run_optimized.sh 默认入口
- legacy_ablation：沿用历史消融口径（固定地形/位置 + 显式固定超参）
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import numpy as np
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

# 复用APF消融脚本的可视化与指标加载逻辑，确保输出一致
from ablation_action_pf_comparison import (
    find_latest_log_dir,
    load_metrics,
    generate_interactive_comparison,
)

# 固定颜色映射，避免实验顺序变化时图例颜色漂移。
EXPERIMENT_COLOR_MAP = {
    "matd3_separated_gradient": "#9900CC",  # 深紫 - MATD3 Mainline
    "matd3_dual_q": "#00CCCC",              # 青色 - Unified-Loss Ablation
    "matd3_single_q": "#FF6600",            # 橙色 - Single-Q Ablation
    "maddpg_separated_gradient": "#CC0000", # 深红 - MADDPG Separated
    "maddpg_dual_q": "#00AA00",             # 深绿 - MADDPG Dual Q
    "maddpg_baseline": "#0066CC",           # 深蓝 - MADDPG Baseline
}

STRICT_CORE_EXPERIMENT_LABELS = [
    "matd3_separated_gradient",
    "matd3_dual_q",
    "matd3_single_q",
]

OPTIONAL_REFERENCE_EXPERIMENT_LABELS = [
    "maddpg_separated_gradient",
    "maddpg_dual_q",
    "maddpg_baseline",
]
# 兼容旧变量名，避免影响现有命令行和批次配置读取逻辑。
EXPLORATORY_EXPERIMENT_LABELS = OPTIONAL_REFERENCE_EXPERIMENT_LABELS

EXPERIMENT_DISPLAY_ORDER = (
    STRICT_CORE_EXPERIMENT_LABELS + OPTIONAL_REFERENCE_EXPERIMENT_LABELS
)

AUTO_POSITIONS_FILE_SENTINEL = "__AUTO_STRICT_POSITIONS__"

# 严格环境隔离：仅保留子进程运行所需的基础环境变量，避免继承父进程训练参数
RUNTIME_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "PWD",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "PYTHONPATH",
    "PYTHONHOME",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "XDG_RUNTIME_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
}

RUNTIME_ENV_PREFIX_ALLOWLIST = (
    "CONDA_",
    "VIRTUAL_ENV",
    "PYENV_",
)

ALLOWED_MANIFEST_ENV_DIFF_KEYS = {
    "ALGORITHM",
    "MATD3_USE_DUAL_Q",
    "MATD3_USE_SEPARATED_GRADIENT",
    "MADDPG_USE_DUAL_Q",
    "MADDPG_USE_SEPARATED_GRADIENT",
}

ALLOWED_MANIFEST_ARG_DIFF_KEYS = {
    "--algo",
    "--exp-name",
    "--matd3-use-dual-q",
    "--matd3-use-separated-gradient",
    "--maddpg-use-dual-q",
    "--maddpg-use-separated-gradient",
}


def _build_process_env(env_isolation: str) -> Dict[str, str]:
    """
    构建训练子进程环境：
    - inherit: 完整继承父进程环境（兼容旧行为）
    - strict: 仅保留运行时必要变量，训练配置完全交给 run_optimized 默认值
    """
    if env_isolation not in ("strict", "inherit"):
        raise ValueError(f"未知环境隔离模式: {env_isolation}")

    if env_isolation == "inherit":
        return os.environ.copy()

    env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key in RUNTIME_ENV_ALLOWLIST or any(key.startswith(prefix) for prefix in RUNTIME_ENV_PREFIX_ALLOWLIST):
            env[key] = value

    # 确保最小可运行环境可用
    env.setdefault("PATH", os.defpath)
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("LANG", "C.UTF-8")
    return env

def _extract_exp_name_with_timestamp(log_path: Path) -> Optional[str]:
    """从run_optimized.sh输出日志中解析带时间戳的实验名"""
    try:
        if not log_path.exists():
            return None
        exp_name = None
        # 该行通常出现在日志开头，但为稳妥读取全量
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


def _resolve_run_log_dir(project_logs_root: Path, exp_name_with_ts: str) -> Optional[str]:
    """根据带时间戳的实验名定位本次运行的日志目录"""
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


def _list_label_run_dirs(project_logs_root: Path, label: str) -> List[Path]:
    """列出指定实验标签的根日志目录（label_YYYYMMDD_HHMMSS）"""
    run_dirs: List[Path] = []
    try:
        if not project_logs_root.exists():
            return run_dirs
        prefix = f"{label}_"
        for item in project_logs_root.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if len(suffix) >= 15 and suffix[8] == "_" and suffix[:8].isdigit() and suffix[9:15].isdigit():
                run_dirs.append(item)
        run_dirs.sort(key=lambda d: d.name)
        return run_dirs
    except Exception:
        return run_dirs


def _resolve_current_run_log_dir(project_logs_root: Path, label: str, before_names: set) -> Optional[str]:
    """优先定位本次新创建的日志目录；找不到时返回None供上层回退。"""
    try:
        after_dirs = _list_label_run_dirs(project_logs_root, label)
        new_dirs = [d for d in after_dirs if d.name not in before_names]
        if not new_dirs:
            return None
        new_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for run_root in new_dirs:
            resolved = _resolve_run_log_dir(project_logs_root, run_root.name)
            if resolved:
                return resolved
        return str(new_dirs[0])
    except Exception:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"配置清单格式错误: {path}")
    argv = data.get("argv")
    exec_env = data.get("exec_env")
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise RuntimeError(f"配置清单缺少有效 argv: {path}")
    if not isinstance(exec_env, dict):
        raise RuntimeError(f"配置清单缺少有效 exec_env: {path}")
    return data


def _normalize_cli_args(argv: List[str]) -> Dict[str, List[Any]]:
    normalized: Dict[str, List[Any]] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                normalized.setdefault(token, []).append(argv[i + 1])
                i += 2
            else:
                normalized.setdefault(token, []).append(True)
                i += 1
        else:
            normalized.setdefault("__positional__", []).append(token)
            i += 1
    return normalized


def _normalize_manifest_for_audit(manifest: Dict[str, Any]) -> Dict[str, Any]:
    argv_norm = _normalize_cli_args(manifest.get("argv", []))
    for key in ALLOWED_MANIFEST_ARG_DIFF_KEYS:
        argv_norm.pop(key, None)
    env_norm = dict(manifest.get("audit_env", {}) or {})
    for key in ALLOWED_MANIFEST_ENV_DIFF_KEYS:
        env_norm.pop(key, None)
    return {
        "python_script": manifest.get("python_script"),
        "cwd": manifest.get("cwd"),
        "argv": argv_norm,
        "audit_env": env_norm,
    }


def _build_manifest_diff(reference: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    ref_norm = _normalize_manifest_for_audit(reference)
    cur_norm = _normalize_manifest_for_audit(current)

    argv_ref = ref_norm.get("argv", {})
    argv_cur = cur_norm.get("argv", {})
    env_ref = ref_norm.get("audit_env", {})
    env_cur = cur_norm.get("audit_env", {})

    argv_only_in_ref = sorted(key for key in argv_ref.keys() if key not in argv_cur)
    argv_only_in_cur = sorted(key for key in argv_cur.keys() if key not in argv_ref)
    argv_changed = sorted(
        key for key in (argv_ref.keys() & argv_cur.keys())
        if argv_ref.get(key) != argv_cur.get(key)
    )

    env_only_in_ref = sorted(key for key in env_ref.keys() if key not in env_cur)
    env_only_in_cur = sorted(key for key in env_cur.keys() if key not in env_ref)
    env_changed = sorted(
        key for key in (env_ref.keys() & env_cur.keys())
        if env_ref.get(key) != env_cur.get(key)
    )

    compatible = (
        ref_norm.get("python_script") == cur_norm.get("python_script")
        and ref_norm.get("cwd") == cur_norm.get("cwd")
        and not argv_only_in_ref
        and not argv_only_in_cur
        and not argv_changed
        and not env_only_in_ref
        and not env_only_in_cur
        and not env_changed
    )

    return {
        "compatible": compatible,
        "python_script_ref": ref_norm.get("python_script"),
        "python_script_cur": cur_norm.get("python_script"),
        "cwd_ref": ref_norm.get("cwd"),
        "cwd_cur": cur_norm.get("cwd"),
        "argv_only_in_ref": argv_only_in_ref,
        "argv_only_in_cur": argv_only_in_cur,
        "argv_changed": {key: {"ref": argv_ref.get(key), "cur": argv_cur.get(key)} for key in argv_changed},
        "env_only_in_ref": env_only_in_ref,
        "env_only_in_cur": env_only_in_cur,
        "env_changed": {key: {"ref": env_ref.get(key), "cur": env_cur.get(key)} for key in env_changed},
    }


def _python_supports_tensorflow(python_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [python_bin, "-c", "import tensorflow as tf; print(tf.__version__)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _resolve_training_python() -> Optional[str]:
    candidates: List[str] = []

    def _add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        candidate = str(candidate).strip()
        if not candidate or candidate in candidates:
            return
        candidates.append(candidate)

    _add(os.environ.get("TRAIN_PYTHON_BIN"))
    _add(os.environ.get("CONDA_PYTHON_EXE"))
    _add(sys.executable)
    _add(shutil.which("python3"))
    _add(shutil.which("python"))

    home = Path.home()
    _add(home / "miniconda3" / "envs" / "maddpg_env" / "bin" / "python")
    _add(home / "miniconda3" / "envs" / "maddpg_env" / "bin" / "python3")
    _add(home / "anaconda3" / "envs" / "maddpg_env" / "bin" / "python")
    _add(home / "anaconda3" / "envs" / "maddpg_env" / "bin" / "python3")

    for candidate in candidates:
        if _python_supports_tensorflow(candidate):
            return candidate
    return None


def _expected_algo_flags(cfg: Dict[str, Any], batch_seed: int = None) -> Dict[str, Any]:
    env_cfg = cfg.get("env", {})
    algo = str(env_cfg.get("ALGORITHM", "matd3")).strip().lower()
    if algo == "matd3":
        dual_q = _to_bool(env_cfg.get("MATD3_USE_DUAL_Q", "1"))
        sep_grad = _to_bool(env_cfg.get("MATD3_USE_SEPARATED_GRADIENT", "1"))
    else:
        dual_q = _to_bool(env_cfg.get("MADDPG_USE_DUAL_Q", "0"))
        sep_grad = _to_bool(env_cfg.get("MADDPG_USE_SEPARATED_GRADIENT", "1"))
    expected_seed = batch_seed if batch_seed is not None else int(env_cfg.get("SEED", "0"))
    return {
        "algo": algo,
        "dual_q": dual_q,
        "separated_gradient": sep_grad,
        "seed": expected_seed,
    }


def _load_results_args(log_dir: str) -> Optional[Dict[str, Any]]:
    results_path = Path(log_dir) / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        args_obj = data.get("args", {})
        return args_obj if isinstance(args_obj, dict) else None
    except Exception:
        return None


def _validate_loaded_result(
    cfg: Dict[str, Any],
    log_dir: str,
    metrics: Dict[str, Any],
    expected_episodes: int,
    positions_file: Path,
    expected_terrain_seed: Optional[int] = None,
    batch_seed: int = None,
) -> List[str]:
    """验证单次实验结果是否完整且与消融配置一致。"""
    errors: List[str] = []
    rewards = metrics.get("episode_rewards", [])
    if not isinstance(rewards, list) or len(rewards) == 0:
        errors.append("episode_rewards 为空")
        return errors
    if len(rewards) != int(expected_episodes):
        errors.append(f"episode_rewards 长度不匹配: got={len(rewards)}, expected={expected_episodes}")
    try:
        rewards_np = np.asarray(rewards, dtype=np.float64)
        if not np.all(np.isfinite(rewards_np)):
            errors.append("episode_rewards 含 NaN/Inf")
    except Exception:
        errors.append("episode_rewards 不是可解析的数值数组")

    run_args = _load_results_args(log_dir)
    if run_args is None:
        errors.append("缺少 results.json 或 args 字段")
        return errors

    expected = _expected_algo_flags(cfg, batch_seed=batch_seed)
    algo = str(run_args.get("algo", "")).strip().lower()
    if algo != expected["algo"]:
        errors.append(f"算法不匹配: got={algo}, expected={expected['algo']}")

    if expected["algo"] == "matd3":
        got_dual_q = _to_bool(run_args.get("matd3_use_dual_q", False))
        got_sep = _to_bool(run_args.get("matd3_use_separated_gradient", False))
    else:
        got_dual_q = _to_bool(run_args.get("maddpg_use_dual_q", False))
        got_sep = _to_bool(run_args.get("maddpg_use_separated_gradient", False))
    if got_dual_q != expected["dual_q"]:
        errors.append(f"Dual-Q 开关不匹配: got={got_dual_q}, expected={expected['dual_q']}")
    if got_sep != expected["separated_gradient"]:
        errors.append(
            f"Separated-Gradient 开关不匹配: got={got_sep}, expected={expected['separated_gradient']}"
        )

    try:
        got_seed = int(run_args.get("seed"))
        if got_seed != expected["seed"]:
            errors.append(f"seed 不匹配: got={got_seed}, expected={expected['seed']}")
    except Exception:
        errors.append("results.json 中缺少有效 seed")

    try:
        got_train_ep = int(run_args.get("train_episodes"))
        if got_train_ep != int(expected_episodes):
            errors.append(f"train_episodes 不匹配: got={got_train_ep}, expected={expected_episodes}")
    except Exception:
        errors.append("results.json 中缺少有效 train_episodes")

    if not _to_bool(run_args.get("use_fixed_positions", False)):
        errors.append("use_fixed_positions=False（消融要求固定位置）")
    if expected_terrain_seed is not None:
        try:
            terrain_seed = int(run_args.get("terrain_seed"))
            if terrain_seed != int(expected_terrain_seed):
                errors.append(
                    f"terrain_seed 不匹配: got={terrain_seed}, expected={expected_terrain_seed}"
                )
        except Exception:
            errors.append("results.json 中缺少有效 terrain_seed")

    recorded_positions = str(run_args.get("positions_file", "")).strip()
    if not recorded_positions:
        errors.append("results.json 中缺少 positions_file")
    else:
        if Path(recorded_positions).name != Path(str(positions_file)).name:
            errors.append(
                f"positions_file 不匹配: got={recorded_positions}, expected={positions_file}"
            )

    return errors


def _evaluate_claims(series: List[Dict[str, Any]], selected_labels: set) -> Dict[str, Any]:
    """生成消融有效性声明检查（只对可严格解释的比较给出通过）。"""
    by_label = {item.get("label"): item for item in series}

    def _tail_mean(label: str, tail: int = 100) -> Optional[float]:
        item = by_label.get(label, {})
        rewards = item.get("metrics", {}).get("episode_rewards", [])
        if not rewards:
            return None
        n = min(len(rewards), tail)
        return float(np.mean(np.asarray(rewards[-n:], dtype=np.float64)))

    claims = [
        {
            "name": "matd3_mainline_vs_unified_actor_loss",
            "lhs": "matd3_dual_q",
            "rhs": "matd3_separated_gradient",
            "required": True,
            "confounded": False,
            "description": "MATD3 主线中 separated actor loss 模块增益",
        },
        {
            "name": "matd3_dual_head_gain",
            "lhs": "matd3_single_q",
            "rhs": "matd3_dual_q",
            "required": True,
            "confounded": False,
            "description": "MATD3 主线中 dual-Q head 模块增益",
        },
        {
            "name": "mainline_vs_maddpg_separated_reference",
            "lhs": "maddpg_separated_gradient",
            "rhs": "matd3_separated_gradient",
            "required": False,
            "confounded": False,
            "description": "跨家族参考：MATD3 主线 vs MADDPG separated",
        },
        {
            "name": "mainline_vs_maddpg_dual_reference",
            "lhs": "maddpg_dual_q",
            "rhs": "matd3_separated_gradient",
            "required": False,
            "confounded": False,
            "description": "跨家族参考：MATD3 主线 vs MADDPG dual-Q",
        },
        {
            "name": "mainline_vs_maddpg_baseline_reference",
            "lhs": "maddpg_baseline",
            "rhs": "matd3_separated_gradient",
            "required": False,
            "confounded": False,
            "description": "跨家族参考：MATD3 主线 vs MADDPG baseline",
        },
    ]

    evaluated = []
    required_failed = []
    for c in claims:
        lhs = c["lhs"]
        rhs = c["rhs"]
        if lhs not in selected_labels or rhs not in selected_labels:
            evaluated.append({**c, "status": "skipped", "tail100_delta_rhs_minus_lhs": None})
            continue
        lhs_mean = _tail_mean(lhs)
        rhs_mean = _tail_mean(rhs)
        if lhs_mean is None or rhs_mean is None:
            status = "invalid_data"
            delta = None
        elif c.get("confounded", False):
            status = "confounded"
            delta = rhs_mean - lhs_mean
        else:
            status = "valid"
            delta = rhs_mean - lhs_mean

        row = {
            **c,
            "status": status,
            "tail100_mean_lhs": lhs_mean,
            "tail100_mean_rhs": rhs_mean,
            "tail100_delta_rhs_minus_lhs": delta,
        }
        evaluated.append(row)

        if c.get("required", False) and status != "valid":
            required_failed.append(
                f"{c['name']} status={status}"
            )

    return {
        "claims": evaluated,
        "required_failed": required_failed,
        "required_pass": len(required_failed) == 0,
    }


def _sort_experiment_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order_map = {label: idx for idx, label in enumerate(EXPERIMENT_DISPLAY_ORDER)}
    return sorted(configs, key=lambda cfg: order_map.get(cfg["label"], len(order_map)))

def plot_comparison_rewards_dualq(series, title, output_path, smooth_window=10, fit_method="moving_average"):
    """绘制奖励对比图（6种不同颜色）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    def smooth_curve(data, method='moving_average', window=10):
        if method == 'moving_average' and len(data) >= window:
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode='valid')
        return data
    
    fig, ax = plt.subplots(figsize=(14, 8))
    has_data = False
    
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        episodes = range(1, len(rewards) + 1)
        rewards_array = np.array(rewards)
        
        color = EXPERIMENT_COLOR_MAP.get(item.get("label"), '#444444')
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        
        # 原始曲线
        ax.plot(episodes, rewards, label=f"{name_en} (Raw)", 
                color=color, alpha=0.3, linewidth=1, linestyle='-')
        
        # 拟合曲线
        smoothed = smooth_curve(rewards_array, method=fit_method, window=smooth_window)
        if len(smoothed) < len(rewards):
            offset = (len(rewards) - len(smoothed)) // 2
            smooth_episodes = range(1 + offset, 1 + offset + len(smoothed))
            ax.plot(smooth_episodes, smoothed, label=f"{name_en} (Fitted)", 
                    color=color, alpha=0.9, linewidth=2.5, linestyle='-')
        else:
            ax.plot(episodes, smoothed, label=f"{name_en} (Fitted)", 
                    color=color, alpha=0.9, linewidth=2.5, linestyle='-')
    
    if has_data:
        ax.set_title(f"{title}\n(Fit Method: {fit_method}, Window: {smooth_window})", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Reward", fontsize=14)
        ax.legend(
            loc='lower right',
            fontsize=10,
            framealpha=0.9,
            ncol=2,
            borderaxespad=0.6,
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Complete] Reward comparison plot: {output_path}")
    plt.close(fig)

def plot_comparison_losses_dualq(series, title, output_path):
    """绘制Loss对比图（6种不同颜色）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    has_data = False
    
    for idx, item in enumerate(series):
        history = item["metrics"].get("loss_history", [])
        if not history:
            continue
        
        steps = [entry.get("step", i) for i, entry in enumerate(history)]
        critic = [entry.get("critic_loss", 0) for entry in history]
        actor = [entry.get("actor_loss", 0) for entry in history]
        
        # 过滤None和NaN
        valid_critic = [(s, c) for s, c in zip(steps, critic) 
                       if c is not None and not (isinstance(c, float) and np.isnan(c)) and c != 0]
        valid_actor = [(s, a) for s, a in zip(steps, actor) 
                      if a is not None and not (isinstance(a, float) and np.isnan(a)) and abs(a) < 1000]
        
        color = EXPERIMENT_COLOR_MAP.get(item.get("label"), '#444444')
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        
        if valid_critic:
            has_data = True
            steps_c, values_c = zip(*valid_critic)
            axes[0].plot(steps_c, values_c, label=f"{name_en} (Critic)", 
                        color=color, linewidth=2.5, alpha=0.9)
        
        if valid_actor:
            steps_a, values_a = zip(*valid_actor)
            axes[1].plot(steps_a, values_a, label=f"{name_en} (Actor)", 
                        color=color, linewidth=2.5, alpha=0.9)
    
    if has_data:
        axes[0].set_title("Critic Loss", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=10, ncol=2)
        axes[0].set_facecolor('#fafafa')
        
        axes[1].set_title("Actor Loss", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Update Step", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=10, ncol=2)
        axes[1].set_facecolor('#fafafa')
        
        fig.suptitle(f"{title} - Loss Curves", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Complete] Loss comparison plot: {output_path}")
    plt.close(fig)

# 导入其他绘图函数（保持原有功能）
from ablation_action_pf_comparison import (
    plot_comparison_success_collision_clearance,
    plot_comparison_success_rate_and_clearance,
)


# ============================================================================
# 实验配置
# ============================================================================

# ============================================================================
# 实验配置
# 主线：MATD3 separated-gradient（run_optimized 当前默认主方法）
# 默认严格实验：围绕主线做模块消融
# 可选参考实验：MADDPG 家族对照，不参与主线模块归因
# ============================================================================

EXPERIMENT_CONFIGS = [
    {
        "label": "matd3_separated_gradient",
        "name": "MATD3 Mainline (Separated)",
        "name_en": "MATD3 Mainline (Separated)",
        "description": "Current mainline method: twin critic, dual Q heads, separated actor loss, shared MATD3 update skeleton",
        "env": {
            "ALGORITHM": "matd3",
            "MATD3_USE_DUAL_Q": "1",
            "MATD3_USE_SEPARATED_GRADIENT": "1",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "matd3_dual_q",
        "name": "MATD3 Ablation - Unified Actor Loss",
        "name_en": "MATD3 Ablation - Unified Actor Loss",
        "description": "Ablation on mainline skeleton: keep dual-Q head, replace separated actor loss with unified total-Q actor loss",
        "env": {
            "ALGORITHM": "matd3",
            "MATD3_USE_DUAL_Q": "1",
            "MATD3_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "matd3_single_q",
        "name": "MATD3 Ablation - Single Q",
        "name_en": "MATD3 Ablation - Single Q",
        "description": "Ablation on mainline skeleton: remove dual-Q head, keep shared MATD3 update skeleton",
        "env": {
            "ALGORITHM": "matd3",
            "MATD3_USE_DUAL_Q": "0",
            "MATD3_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "maddpg_separated_gradient",
        "name": "MADDPG Separated Gradient",
        "name_en": "MADDPG Separated Gradient",
        "description": "Reference only: single critic, dual Q heads, separated gradient",
        "env": {
            "ALGORITHM": "maddpg",
            "MADDPG_USE_DUAL_Q": "1",
            "MADDPG_USE_SEPARATED_GRADIENT": "1",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "maddpg_dual_q",
        "name": "MADDPG Dual Q",
        "name_en": "MADDPG Dual Q",
        "description": "Reference only: single critic, dual Q heads, unified gradient",
        "env": {
            "ALGORITHM": "maddpg",
            "MADDPG_USE_DUAL_Q": "1",
            "MADDPG_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "maddpg_baseline",
        "name": "MADDPG Baseline",
        "name_en": "MADDPG Baseline",
        "description": "Reference only: single critic, single Q head, unified gradient",
        "env": {
            "ALGORITHM": "maddpg",
            "MADDPG_USE_DUAL_Q": "0",
            "MADDPG_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="围绕 MATD3 主线方法的模块消融实验")
    parser.add_argument("--episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024, help="训练批次大小")
    parser.add_argument("--script", type=str, default="./run_optimized.sh", help="训练脚本路径")
    parser.add_argument("--use-weighted-reward", type=int, default=1, help="是否使用分项加权奖励")
    parser.add_argument(
        "--positions-file",
        type=str,
        default=AUTO_POSITIONS_FILE_SENTINEL,
        help="固定位置文件路径；默认按 strict_ablation 的场景seed自动生成独立文件名",
    )
    parser.add_argument(
        "--env-isolation",
        type=str,
        default="strict",
        choices=["strict", "inherit"],
        help="环境隔离模式：strict=只保留运行时必需变量；inherit=继承父进程全部环境（旧行为）",
    )
    parser.add_argument(
        "--config-mode",
        type=str,
        default="strict_ablation",
        choices=["strict_ablation", "run_optimized_default", "legacy_ablation"],
        help="配置模式：strict_ablation=严格固定环境；run_optimized_default=兼容旧逻辑；legacy_ablation=旧固定超参方案",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=None,
        help="场景seed（默认随config-mode选择：strict_ablation/run_optimized_default=88，legacy_ablation=67）",
    )
    parser.add_argument(
        "--unlock-env-on-success",
        type=int,
        default=None,
        help="覆盖课程解锁阈值：连续成功回合数（0表示禁用解锁，None表示沿用配置模式默认值）",
    )
    parser.add_argument(
        "--unlock-env-on-plateau",
        type=int,
        default=None,
        help="覆盖课程解锁阈值：奖励停滞回合数（0表示禁用解锁，None表示沿用配置模式默认值）",
    )
    parser.add_argument("--output-dir", type=str, default="ablation_dual_q_outputs", help="图表输出目录")
    parser.add_argument("--logs-root", type=str, default="logs", help="训练日志根目录")
    parser.add_argument("--reuse", action="store_true", help="若检测到同名实验已存在，则跳过重新训练")
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="仅复用历史结果并绘图；若复用失败则直接报错，不触发训练",
    )
    parser.add_argument("--smooth-window", type=int, default=10, help="拟合曲线平滑窗口大小")
    parser.add_argument("--fit-method", type=str, default="moving_average",
                        choices=["moving_average", "spline", "poly"], help="拟合方法")
    parser.add_argument("--generate-interactive", action="store_true", help="生成交互式轨迹图（需要plotly）")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        choices=[cfg["label"] for cfg in EXPERIMENT_CONFIGS],
                        help="选择要运行的实验；默认只运行 MATD3 主线模块消融")
    parser.add_argument(
        "--include-reference-experiments",
        "--include-exploratory-experiments",
        dest="include_reference_experiments",
        action="store_true",
        help="默认只运行 MATD3 主线模块消融；显式开启后再纳入 MADDPG 参考实验",
    )
    parser.add_argument(
        "--disable-strict-validity",
        action="store_true",
        help="关闭严格有效性校验（默认开启，建议保持开启）",
    )
    parser.add_argument(
        "--experiment-group",
        type=str,
        default="A",
        choices=["A", "B"],
        help="实验组：A=纯固定地图（严格默认）；B=从一开始使用随机障碍物（仍禁用课程学习）",
    )
    args = parser.parse_args()
    # 兼容旧字段名，避免主流程或外部脚本仍访问 include_exploratory_experiments。
    setattr(args, "include_exploratory_experiments", bool(getattr(args, "include_reference_experiments", False)))
    return args


def _resolve_scenario_seed(config_mode: str, scenario_seed_arg: Optional[int]) -> int:
    if scenario_seed_arg is not None:
        return int(scenario_seed_arg)
    if config_mode in ("strict_ablation", "run_optimized_default"):
        return 88
    return 67


def _resolve_unlock_thresholds(
    config_mode: str,
    unlock_env_on_success: Optional[int],
    unlock_env_on_plateau: Optional[int],
) -> tuple:
    """解析课程解锁阈值（支持命令行覆盖）。"""
    if config_mode == "strict_ablation":
        return 0, 0

    if unlock_env_on_success is None:
        success_v = 25 if config_mode == "run_optimized_default" else 0
    else:
        success_v = int(unlock_env_on_success)

    if unlock_env_on_plateau is None:
        plateau_v = 100 if config_mode == "run_optimized_default" else 0
    else:
        plateau_v = int(unlock_env_on_plateau)

    return success_v, plateau_v


def _resolve_positions_file(
    raw_positions_file: str,
    scenario_seed: int,
    experiment_group: str,
) -> Path:
    """为严格版自动分配带seed的固定位置文件，避免复用旧seed残留。"""
    if raw_positions_file and raw_positions_file != AUTO_POSITIONS_FILE_SENTINEL:
        return Path(raw_positions_file)
    return Path(
        f"./saved_positions/strict_ablation_seed{int(scenario_seed)}_group{experiment_group}.json"
    )


# ============================================================================
# 固定位置生成
# ============================================================================

def generate_fixed_positions(
    positions_file: Path,
    n_agents: int = 3,
    map_size: float = 200.0,
    scenario_seed: int = 67,
    scenario_name: str = "paper3d_terrain_weighted",
    terrain_complexity_level: int = 3,
):
    """生成严格消融使用的固定位置文件。"""
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
    os.environ['SCENARIO_SEED'] = str(scenario_seed)
    os.environ['TERRAIN_COMPLEXITY_LEVEL'] = str(int(terrain_complexity_level))
    os.environ['MAP_SIZE'] = str(int(map_size))
    
    try:
        scenario_module = importlib.import_module(f"multiagent.scenarios.{scenario_name}")
        
        # 创建场景实例
        scenario = scenario_module.Scenario(
            seed=int(scenario_seed),
            use_fixed_positions=True,
            dynamic_first_time=False,
            fixed_positions_file=str(positions_file),
            random_terrain=False,
            terrain_complexity_level=int(terrain_complexity_level),
            map_size=float(map_size),
        )
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
            f"提示：请检查场景初始化代码和环境配置是否正确（scenario={scenario_name}）。\n"
            f"位置文件路径: {positions_file}"
        )
    
    return positions_file


# ============================================================================
# 基础环境变量设置
# ============================================================================

def setup_base_env_vars(
    positions_file: Path,
    env_isolation: str = "strict",
    config_mode: str = "run_optimized_default",
    scenario_seed: int = 88,
) -> dict:
    """
    构建消融实验的请求环境。

    这里不再手工重建训练超参，也不再大规模 pop/重写环境变量。
    真正的完整训练配置统一交给 run_optimized.sh 解析；当前函数只负责：
    1. 提供最小可运行的运行时环境
    2. 注入消融实验显式要求的环境控制（固定位置/种子/课程学习开关等）
    3. 避免并行/GPU 相关的额外不稳定因素
    """
    env = _build_process_env(env_isolation)

    # === 串行训练的稳定 GPU 配置（不改变训练语义） ===
    env["GPU_ID"] = "0"
    env["TF_GPU_ALLOCATOR"] = ""
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env.pop("TF_SYNC_ON_FINISH", None)
    env.pop("XLA_FLAGS", None)
    env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

    # 仅保留运行时必要的非训练语义开关
    env.setdefault("SUPPRESS_MA_PROMPT", "1")
    env.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")

    if config_mode == "legacy_ablation":
        # 历史消融模式：固定地形/位置 + 显式超参数
        env["USE_FIXED_POSITIONS"] = "1"
        env["DYNAMIC_FIRST_TIME"] = "0"
        env["POSITIONS_FILE"] = str(positions_file)
        env["UNLOCK_ENV_ON_SUCCESS"] = "0"
        env["UNLOCK_ENV_ON_PLATEAU"] = "0"
        env["RANDOM_TERRAIN"] = "0"
        env["PER_ENV_TERRAIN"] = "0"
        env["PER_EPISODE_TERRAIN"] = "0"
        env["USE_SCENARIO_SEED"] = "1"
        env["SCENARIO_SEED"] = str(scenario_seed)

        env["TERRAIN_COMPLEXITY_LEVEL"] = "3"
        env["MAP_SIZE"] = "200"
        env["MOUNTAIN_MIN_DISTANCE"] = "55"
        env["TERRAIN_CONTACT_EPS"] = "0.2"

        env["LEARNING_RATE_ACTOR"] = "0.0005"
        env["LEARNING_RATE_CRITIC"] = "0.002"
        env["LR_DECAY_ENABLED"] = "1"
        env["LR_DECAY_STEPS"] = "20000"
        env["LR_DECAY_RATE"] = "0.9996"
        env["LR_STAIRCASE"] = "1"
        env["LR_MIN_ACTOR"] = "0.00030"
        env["LR_MIN_CRITIC"] = "0.00040"

        env["ACTOR_HIDDEN"] = "256,256,256"
        env["CRITIC_HIDDEN"] = "256,256,256"

        env["BUFFER_SIZE"] = "500000"
        env["UPDATE_RATE"] = "40"
        env["ACTOR_UPDATE_DELAY"] = "2"
        env["HUBER_DELTA"] = "2.0"

        env["POLICY_NOISE"] = "0.28"
        env["NOISE_CLIP"] = "0.32"
        env["POLICY_FREQ"] = "1"

        env["NOISE_SCALE"] = "0.35"
        env["NOISE_DECAY"] = "0.9995"
        env["NOISE_MIN"] = "0.05"

        env["PER_ENABLED"] = "1"
        env["USE_LITE_BUFFER"] = "1"
        env["PER_UNIFORM_MIX"] = "0.45"
        env["PER_TD_WEIGHT"] = "0.80"
        env["PER_REWARD_WEIGHT"] = "0.12"
        env["PER_AGE_DECAY"] = "0.95"

        env["USE_QUADROTOR_DYNAMICS"] = "1"
        env["QUADROTOR_ATTITUDE_RESPONSE_TIME"] = "0.05"
        env["ACTION_RANGE_X"] = "2.5"
        env["ACTION_RANGE_Y"] = "2.5"
        env["ACTION_RANGE_Z"] = "2.2"
        env["CONTROL_ACCEL_GAIN"] = "1.0"
        env["DAMPING"] = "0.12"
        env["GRAVITY"] = "0.0"

        env["REWARD_POS_SCALE"] = "1.0"
        env["REWARD_NEG_SCALE"] = "1.0"

        env["MAX_WEIGHT_THRESHOLD"] = "0.999"
        env["WEIGHT_SCALING_FACTOR"] = "0.999"
    elif config_mode == "strict_ablation":
        # 严格默认：固定地形 + 固定位置 + 禁用课程学习，只把环境基线钉住
        env["TF_DETERMINISTIC_OPS"] = "0"
        env["USE_FIXED_POSITIONS"] = "1"
        env["DYNAMIC_FIRST_TIME"] = "0"
        env["POSITIONS_FILE"] = str(positions_file)
        env["UNLOCK_ENV_ON_SUCCESS"] = "0"
        env["UNLOCK_ENV_ON_PLATEAU"] = "0"
        env["RANDOM_TERRAIN"] = "0"
        env["PER_ENV_TERRAIN"] = "0"
        env["PER_EPISODE_TERRAIN"] = "0"
        env["USE_SCENARIO_SEED"] = "1"
        env["SCENARIO_SEED"] = str(scenario_seed)

        env["TERRAIN_COMPLEXITY_LEVEL"] = "3"
        env["MAP_SIZE"] = "200"
        env["MOUNTAIN_MIN_DISTANCE"] = "55"
        env["TERRAIN_CONTACT_EPS"] = "0.2"
        env["USE_DYNAMIC_OBSTACLES"] = "0"
    elif config_mode == "run_optimized_default":
        # 尽量贴近当前 run_optimized 默认入口，只固定用于公平对比的环境相关项
        env["TF_DETERMINISTIC_OPS"] = "0"
        env["USE_FIXED_POSITIONS"] = "1"
        env["DYNAMIC_FIRST_TIME"] = "1"
        env["POSITIONS_FILE"] = str(positions_file)
        env["UNLOCK_ENV_ON_SUCCESS"] = "25"
        env["UNLOCK_ENV_ON_PLATEAU"] = "100"
        env["RANDOM_TERRAIN"] = "0"
        env["PER_ENV_TERRAIN"] = "0"
        env["PER_EPISODE_TERRAIN"] = "0"
        env["USE_SCENARIO_SEED"] = "1"
        env["SCENARIO_SEED"] = str(scenario_seed)

        env["TERRAIN_COMPLEXITY_LEVEL"] = "3"
        env["MAP_SIZE"] = "200"
        env["MOUNTAIN_MIN_DISTANCE"] = "55"
        env["TERRAIN_CONTACT_EPS"] = "0.2"
    else:
        raise ValueError(f"未知配置模式: {config_mode}")

    return env


def _resolve_experiment_manifest(
    cfg: Dict[str, Any],
    positions_file: Path,
    args,
    manifest_dir: Path,
) -> tuple[Dict[str, Any], Path]:
    label = cfg["label"]
    env_vars = cfg.get("env", {})
    env = setup_base_env_vars(
        positions_file,
        env_isolation=args.env_isolation,
        config_mode=args.config_mode,
        scenario_seed=int(args.resolved_scenario_seed),
    )

    for key, value in env_vars.items():
        env[key] = value

    env["SEED"] = str(getattr(args, "batch_seed", random.randint(100000, 999999)))
    env["UNLOCK_ENV_ON_SUCCESS"] = str(args.resolved_unlock_env_on_success)
    env["UNLOCK_ENV_ON_PLATEAU"] = str(args.resolved_unlock_env_on_plateau)
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    env["USE_DYNAMIC_OBSTACLES"] = "1" if getattr(args, "use_dynamic_obstacles", False) else "0"

    algorithm = env.get("ALGORITHM", "matd3")
    manifest_path = manifest_dir / f"{label}_resolved_manifest.json"

    resolve_env = dict(env)
    training_python = _resolve_training_python()
    if not training_python:
        raise RuntimeError(
            "未找到可用的 TensorFlow 训练解释器。"
            " 请确认已安装并可访问包含 TensorFlow 的 Python 环境，"
            " 或通过 TRAIN_PYTHON_BIN 显式指定训练解释器。"
        )
    resolve_env["TRAIN_PYTHON_BIN"] = training_python
    resolve_env["ABLATION_RESOLVE_ONLY"] = "1"
    resolve_env["ABLATION_MANIFEST_PATH"] = str(manifest_path)

    cmd = [
        "/bin/bash",
        args.script,
        str(args.episodes),
        str(args.batch_size),
        label,
        str(args.use_weighted_reward),
        algorithm,
    ]

    subprocess.run(cmd, env=resolve_env, cwd=Path(args.script).resolve().parent, check=True)
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("meta", {})
    manifest["meta"].update({
        "label": label,
        "algorithm": algorithm,
        "seed": env["SEED"],
        "experiment_group": args.experiment_group,
        "training_python": training_python,
    })
    _save_json(manifest_path, manifest)
    return manifest, manifest_path


def run_experiment(cfg: Dict, positions_file: Path, args, cache: Dict[str, Dict]) -> Dict:
    """运行单个实验（串行版本）"""
    label = cfg["label"]
    project_logs_root = Path(args.script).resolve().parent / "logs"

    if args.reuse:
        try:
            reuse_logs_root = project_logs_root if args.logs_root == "logs" else Path(args.logs_root).resolve()
            log_dir = find_latest_log_dir(label, str(reuse_logs_root))
            metrics = load_metrics(log_dir)
            validation_errors = _validate_loaded_result(
                cfg=cfg,
                log_dir=log_dir,
                metrics=metrics,
                expected_episodes=int(args.episodes),
                positions_file=Path(positions_file),
                expected_terrain_seed=int(args.resolved_scenario_seed),
                batch_seed=getattr(args, 'batch_seed', None),
            )
            if validation_errors:
                raise RuntimeError(
                    "[复用-{}] 历史结果有效性校验失败: {}".format(label, " | ".join(validation_errors))
                )
            return {
                "label": label,
                "name": cfg.get("name", label),
                "name_en": cfg.get("name_en", cfg.get("name", label)),
                "description": cfg.get("description", ""),
                "log_dir": log_dir,
                "metrics": metrics,
                "success": True
            }
        except Exception as e:
            if getattr(args, "reuse_only", False):
                raise RuntimeError(
                    f"[复用失败-{label}] 仅复用模式已开启，停止训练。原因: {e}"
                )
            pass

    manifest_dir = Path(args.manifest_dir)
    manifest, manifest_path = _resolve_experiment_manifest(cfg, positions_file, args, manifest_dir)
    algorithm = str(manifest.get("meta", {}).get("algorithm", "matd3")).strip().lower()

    exec_env = dict(manifest.get("exec_env", {}))
    dual_q_flag = exec_env.get("MATD3_USE_DUAL_Q") if algorithm == "matd3" else exec_env.get("MADDPG_USE_DUAL_Q")
    sep_grad_flag = exec_env.get("MATD3_USE_SEPARATED_GRADIENT") if algorithm == "matd3" else exec_env.get("MADDPG_USE_SEPARATED_GRADIENT")

    if getattr(args, "reference_manifest", None) is None:
        args.reference_manifest = manifest
        args.reference_manifest_label = label
        reference_path = manifest_dir / "_reference_manifest.json"
        _save_json(reference_path, manifest)
    else:
        diff = _build_manifest_diff(args.reference_manifest, manifest)
        diff.update(
            {
                "reference_label": getattr(args, "reference_manifest_label", "reference"),
                "current_label": label,
            }
        )
        diff_path = manifest_dir / f"{label}_manifest_diff.json"
        _save_json(diff_path, diff)
        if not diff.get("compatible", False):
            raise RuntimeError(
                f"[配置审计失败-{label}] 解析后的训练配置超出了算法消融白名单差异。"
                f" 详情见: {diff_path}"
            )

    python_bin = (
        manifest.get("python_executable")
        or exec_env.get("TRAIN_PYTHON_BIN")
        or _resolve_training_python()
        or "python3"
    )
    cmd = [
        str(python_bin),
        str(manifest.get("python_script")),
        *list(manifest.get("argv", [])),
    ]

    print(f"\n{'='*70}")
    print(f"[运行] {cfg.get('name', label)}")
    print(f"[运行] 算法: {algorithm}")
    print(f"[运行] 双Q头: {dual_q_flag}")
    print(f"[运行] 分离梯度: {sep_grad_flag}")
    print(f"[运行] 种子: {exec_env.get('SEED')}")
    print(f"[运行] 训练解释器: {python_bin}")
    print(f"[运行] 冻结配置: {manifest_path}")
    print(f"{'='*70}")

    before_run_dirs = {d.name for d in _list_label_run_dirs(project_logs_root, label)}
    subprocess.run(cmd, env=exec_env, cwd=manifest.get("cwd", str(Path(args.script).resolve().parent)), check=True)

    # 🚨 严格定位本次运行目录：避免“本次失败却回退历史数据”污染消融结论
    log_dir = _resolve_current_run_log_dir(project_logs_root, label, before_run_dirs)
    if not log_dir:
        log_dir = find_latest_log_dir(label, str(project_logs_root))
    metrics = load_metrics(log_dir)
    n_ep = len(metrics.get("episode_rewards", []))
    print(f"[运行-{label}] 加载指标: {log_dir!r}, episode数={n_ep}", file=sys.stderr)
    if n_ep == 0:
        raise RuntimeError(
            f"[运行-{label}] 本次运行未生成有效 episode_rewards.json: {log_dir}. "
            f"为避免使用历史数据污染消融结论，已中止。"
        )
    validation_errors = _validate_loaded_result(
        cfg=cfg,
        log_dir=log_dir,
        metrics=metrics,
        expected_episodes=int(args.episodes),
        positions_file=Path(positions_file),
        expected_terrain_seed=int(args.resolved_scenario_seed),
        batch_seed=getattr(args, 'batch_seed', None),
    )
    if validation_errors:
        raise RuntimeError(
            "[运行-{}] 结果有效性校验失败: {}".format(label, " | ".join(validation_errors))
        )
    result = {
        "label": label,
        "name": cfg.get("name", label),
        "name_en": cfg.get("name_en", cfg.get("name", label)),
        "description": cfg.get("description", ""),
        "log_dir": log_dir,
        "manifest_path": str(manifest_path),
        "metrics": metrics,
        "success": True
    }
    cache[label] = result
    return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    args.resolved_scenario_seed = _resolve_scenario_seed(args.config_mode, args.scenario_seed)
    (
        args.resolved_unlock_env_on_success,
        args.resolved_unlock_env_on_plateau,
    ) = _resolve_unlock_thresholds(
        config_mode=args.config_mode,
        unlock_env_on_success=args.unlock_env_on_success,
        unlock_env_on_plateau=args.unlock_env_on_plateau,
    )

    # === 实验组逻辑：A=纯固定地图, B=纯随机障碍物 ===
    # 两组都禁用课程学习，区别在于是否启用随机障碍物
    experiment_group = args.experiment_group
    if experiment_group == "A":
        args.resolved_unlock_env_on_success = 0
        args.resolved_unlock_env_on_plateau = 0
        args.use_dynamic_obstacles = False
        print("[Group A] Fixed-map mode: curriculum disabled, USE_DYNAMIC_OBSTACLES=0")
    elif experiment_group == "B":
        args.resolved_unlock_env_on_success = 0
        args.resolved_unlock_env_on_plateau = 0
        args.use_dynamic_obstacles = True
        print("[Group B] Random-obstacle mode: curriculum disabled, USE_DYNAMIC_OBSTACLES=1")
        print("[Group B] Note: still cross-algorithm consistent, but not the default pure fixed-environment setup.")

    # === 种子管理：批次内所有实验共享同一随机种子 ===
    batch_seed = random.randint(100000, 999999)
    print(f"[种子管理] 本批次随机种子: {batch_seed}")
    args.batch_seed = batch_seed

    # 位置文件处理
    positions_file = _resolve_positions_file(
        raw_positions_file=args.positions_file,
        scenario_seed=int(args.resolved_scenario_seed),
        experiment_group=args.experiment_group,
    )
    args.resolved_positions_file = str(positions_file)
    if args.config_mode in ("strict_ablation", "legacy_ablation"):
        scenario_name = "paper3d_terrain_weighted" if int(args.use_weighted_reward) == 1 else "paper3d_terrain_energy"
        generate_fixed_positions(
            positions_file,
            scenario_seed=int(args.resolved_scenario_seed),
            scenario_name=scenario_name,
            terrain_complexity_level=3,
        )
    else:
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        print(
            "[Info] run_optimized_default 模式：不预生成固定位置文件，"
            "沿用 run_optimized 的 DYNAMIC_FIRST_TIME=1 默认行为。"
        )
    print(f"[位置文件] {positions_file}")

    # 选择实验
    if args.experiments:
        configs_to_run = [cfg for cfg in EXPERIMENT_CONFIGS if cfg["label"] in args.experiments]
        configs_to_run = _sort_experiment_configs(configs_to_run)
        if not configs_to_run:
            print(f"[错误] 未找到指定实验: {args.experiments}")
            sys.exit(1)
        reference_selected = sorted(
            label for label in args.experiments if label in OPTIONAL_REFERENCE_EXPERIMENT_LABELS
        )
        if reference_selected:
            print(
                f"[提示] 已显式纳入参考实验（不参与主线模块归因）: {reference_selected}"
            )
    else:
        default_labels = list(STRICT_CORE_EXPERIMENT_LABELS)
        if args.include_reference_experiments:
            default_labels.extend(OPTIONAL_REFERENCE_EXPERIMENT_LABELS)
        configs_to_run = [cfg for cfg in EXPERIMENT_CONFIGS if cfg["label"] in default_labels]
        configs_to_run = _sort_experiment_configs(configs_to_run)

    # 创建批次目录（含实验组标识）
    group_label = f"group{experiment_group}"
    group_desc = "Fixed Map" if experiment_group == "A" else "Random Obstacles"
    if AblationBatchManager is not None:
        manager = AblationBatchManager()
        batch_config = {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "use_weighted_reward": args.use_weighted_reward,
            "seed": batch_seed,
            "scenario_seed": int(args.resolved_scenario_seed),
            "terrain_complexity": 3,
            "map_size": 200,
            "mountain_min_distance": 55,
            "env_isolation": args.env_isolation,
            "config_mode": args.config_mode,
            "experiment_group": experiment_group,
            "experiment_group_desc": group_desc,
            "use_dynamic_obstacles": getattr(args, 'use_dynamic_obstacles', False),
            "unlock_env_on_success": int(args.resolved_unlock_env_on_success),
            "unlock_env_on_plateau": int(args.resolved_unlock_env_on_plateau),
            "positions_file": str(positions_file),
            "notes": (
                f"MATD3 mainline module ablation: "
                f"default core experiments={STRICT_CORE_EXPERIMENT_LABELS}, "
                f"optional reference experiments={OPTIONAL_REFERENCE_EXPERIMENT_LABELS}, "
                f"Group {experiment_group} ({group_desc}), mode={args.config_mode}"
            ),
        }
        from datetime import datetime as _dt
        batch_id = f"batch_{group_label}_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir = manager.create_batch(
            batch_id=batch_id,
            config=batch_config,
            experiments=[c["label"] for c in configs_to_run],
        )
        output_dir = batch_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_file = batch_dir / "shared_seed.json"
        with open(seed_file, 'w', encoding='utf-8') as f:
            json.dump({"seed": batch_seed, "source": "random", "experiment_group": experiment_group}, f, indent=2)
        print(f"{'='*70}")
        print(f"✅ 批次目录已创建: {batch_dir}")
        print(f"  Group: {experiment_group} ({group_desc})")
        print(f"  共享种子: {batch_seed} (已保存至 {seed_file})")
        print(f"{'='*70}\n")
    else:
        output_dir = Path(f"{args.output_dir}_{group_label}")
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_dir = output_dir

    manifest_dir = batch_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir = str(manifest_dir)
    args.reference_manifest = None
    args.reference_manifest_label = None

    series = []
    cache: Dict[str, Dict] = {}

    print(f"\n{'='*70}")
    print("MATD3 Mainline Module Ablation")
    print(f"Group: {experiment_group} ({group_desc})")
    print("  Group A = Fixed Map (curriculum disabled, USE_DYNAMIC_OBSTACLES=0)")
    print("  Group B = Random Obstacles (curriculum disabled, USE_DYNAMIC_OBSTACLES=1)")
    print(f"实验数量: {len(configs_to_run)}")
    for cfg in configs_to_run:
        print(f"  - {cfg['name']}: {cfg['description']}")
    print(f"主线模块消融: {STRICT_CORE_EXPERIMENT_LABELS}")
    if args.include_reference_experiments:
        print(f"可选参考实验: {OPTIONAL_REFERENCE_EXPERIMENT_LABELS}")
    print("模式: 串行")
    print(f"配置模式: {args.config_mode}")
    print(f"环境隔离: {args.env_isolation}")
    print(f"动态障碍物: {'启用' if getattr(args, 'use_dynamic_obstacles', False) else '禁用'}")
    print(f"课程学习: 已禁用 (success=0, plateau=0)")
    print(f"固定位置文件: {positions_file}")
    print(f"复用策略: reuse={args.reuse}, reuse_only={args.reuse_only}")
    print(f"训练种子: {batch_seed} (随机生成，批次内共享)")
    print(f"场景Seed: {args.resolved_scenario_seed}")
    print(f"配置清单目录: {manifest_dir}")
    print(f"严格有效性校验: {'关闭' if args.disable_strict_validity else '开启'}")
    print(f"{'='*70}\n")

    for cfg in configs_to_run:
        result = run_experiment(cfg, positions_file, args, cache)
        series.append(result)

    if not series:
        print("⚠️  没有可用的实验数据，无法生成对比图")
        sys.exit(1)

    selected_labels = {cfg["label"] for cfg in configs_to_run}
    available_labels = {item.get("label") for item in series}
    missing_labels = sorted(label for label in selected_labels if label not in available_labels)
    strict_validity_enabled = not bool(args.disable_strict_validity)
    if missing_labels:
        msg = f"缺少实验结果: {missing_labels}"
        if strict_validity_enabled:
            print(f"[严格校验失败] {msg}")
            sys.exit(1)
        print(f"[警告] {msg}")

    claims_report = _evaluate_claims(series, selected_labels)
    print("[有效性检查] 对比声明状态：")
    for row in claims_report["claims"]:
        delta = row.get("tail100_delta_rhs_minus_lhs")
        delta_str = f"{delta:.2f}" if isinstance(delta, (int, float)) else "N/A"
        print(f"  - {row['name']}: {row['status']} (tail100 Δ={delta_str})")
    if claims_report["required_pass"]:
        print("[有效性检查] 主线模块声明检查通过（Unified Actor Loss / Single-Q 消融）。")
    else:
        print("[有效性检查] 核心声明检查失败：")
        for item in claims_report["required_failed"]:
            print(f"  - {item}")
        if strict_validity_enabled:
            print("[严格校验失败] 请先修复上述问题后再生成消融结论。")
            sys.exit(1)
        print("[警告] 严格校验已关闭，继续生成图表。")

    print(f"\n{'='*70}")
    print(f"✅ 找到 {len(series)} 个实验的数据，开始生成对比图...")
    print(f"{'='*70}\n")

    title = f"MATD3 Mainline Module Ablation - Group {experiment_group} ({group_desc})"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    reward_png = output_dir / f"reward_comparison_{timestamp}.png"
    plot_comparison_rewards_dualq(series, title, reward_png, smooth_window=args.smooth_window, fit_method=args.fit_method)

    loss_png = output_dir / f"loss_comparison_{timestamp}.png"
    plot_comparison_losses_dualq(series, title, loss_png)

    success_collision_png = output_dir / f"success_collision_clearance_comparison_{timestamp}.png"
    plot_comparison_success_collision_clearance(
        series, title, success_collision_png, smooth_window=args.smooth_window, fit_method=args.fit_method
    )

    success_clearance_png = output_dir / f"success_rate_and_clearance_comparison_{timestamp}.png"
    plot_comparison_success_rate_and_clearance(
        series, title, success_clearance_png, smooth_window=args.smooth_window, fit_method=args.fit_method
    )

    if args.generate_interactive:
        interactive_html = output_dir / f"interactive_comparison_{timestamp}.html"
        generate_interactive_comparison(
            series, title, interactive_html, smooth_window=args.smooth_window, fit_method=args.fit_method
        )

    summary = {
        "timestamp": timestamp,
        "experiment_group": experiment_group,
        "experiment_group_desc": group_desc,
        "use_dynamic_obstacles": getattr(args, 'use_dynamic_obstacles', False),
        "strict_validity_enabled": strict_validity_enabled,
        "claims_report": claims_report,
        "experiments": [
            {
                "label": item["label"],
                "name": item.get("name", item["label"]),
                "name_en": item.get("name_en", item.get("name", item.get("label", "Unknown"))),
                "description": item.get("description", ""),
                "log_dir": item.get("log_dir", ""),
                "manifest_path": item.get("manifest_path", ""),
                "final_reward": item["metrics"].get("episode_rewards", [])[-1] if item["metrics"].get("episode_rewards") else None,
                "avg_reward": np.mean(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
                "max_reward": np.max(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
            }
            for item in series
        ],
        "output_files": {
            "reward_comparison": str(reward_png.name),
            "loss_comparison": str(loss_png.name),
            "success_collision_clearance_comparison": str(success_collision_png.name),
            "success_rate_and_clearance_comparison": str(success_clearance_png.name),
        }
    }
    if args.generate_interactive:
        summary["output_files"]["interactive_comparison"] = str(interactive_html.name)

    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=False, indent=2)

    latest_summary_path = output_dir / "latest_summary.json"
    with open(latest_summary_path, "w", encoding="utf-8") as f_latest:
        json.dump(summary, f_latest, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Ablation run complete - Group {experiment_group} ({group_desc})")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    print(f"Reward comparison plot: {reward_png.name}")
    print(f"Loss comparison plot: {loss_png.name}")
    print(f"Success/collision/clearance plot: {success_collision_png.name}")
    print(f"Success rate & clearance plot: {success_clearance_png.name}")
    if args.generate_interactive:
        print(f"Interactive plot: {interactive_html.name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

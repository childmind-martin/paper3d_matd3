#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
围绕 MATD3 当前主线骨架的 separated-skeleton / actor-objective 消融实验。

默认严格实验只运行 3 组 MATD3 主线相关配置：
1. MATD3 Mainline - Separated Skeleton + Separated Actor Objective
2. MATD3 Ablation - Separated Skeleton + Unified Actor Objective
3. MATD3 Baseline

可选参考实验默认不参与主线模块归因：
1. MADDPG Reference - Separated Skeleton + Separated Actor Objective
2. MADDPG Reference - Separated Skeleton + Unified Actor Objective
3. MADDPG Baseline

严格版默认策略：
- 保持主线 MATD3 separated skeleton + separated actor objective 不动
- 其他 MATD3 实验共享同一个外层 update skeleton，只关闭少数模块
- 默认禁用课程学习（UNLOCK_ENV_ON_SUCCESS=0, UNLOCK_ENV_ON_PLATEAU=0）
- 默认预生成固定位置文件，不依赖 dynamic_first_time
- 默认固定地形、固定位置、固定障碍物，确保跨实验环境完全一致
- 当前脚本支持单 seed 串行运行，也支持多 seed 父进程并发调度并统一汇总

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random
import numpy as np
import time

from algorithm_ablation_colors import (
    ALGORITHM_ABLATION_COLOR_BY_LABEL,
    get_algorithm_ablation_color,
)

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
    resolve_metric_file,
)

# Shared palette: keep the same algorithm label on the same color in every figure.
EXPERIMENT_COLOR_MAP = dict(ALGORITHM_ABLATION_COLOR_BY_LABEL)

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

DEFAULT_POST_EVAL_EPISODES = 50
POST_EVAL_SUMMARY_SPECS = [
    ("team_success_rate", "Team Success Rate", True, False),
    ("avg_reward", "Average Reward", False, False),
    ("avg_collision_count", "Average Collisions", False, True),
    ("avg_min_clearance_min", "Average Min Clearance (m)", False, False),
    ("avg_team_total_path_length", "Average Team Path Length (m)", False, True),
    ("avg_arrival_step_success_only", "Success Arrival Step", False, True),
]
POST_EVAL_FALLBACK_KEYS = {
    "avg_arrival_step_success_only": "avg_first_reach_step",
}

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


def _load_json_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON 文件格式错误: {path}")
    return data


def _parse_seed_list(seed_text: Optional[str]) -> List[int]:
    seeds: List[int] = []
    if not seed_text:
        return seeds
    for token in str(seed_text).split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"随机种子必须唯一，当前收到: {seeds}")
    return seeds


def _moving_average_binary(flags: Sequence[float], window: int = 50) -> np.ndarray:
    arr = np.asarray(flags, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    out = np.zeros_like(arr, dtype=np.float64)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        out[idx] = float(np.mean(arr[start:idx + 1]))
    return out


def _smooth_curve(values: Sequence[float], method: str = "moving_average", window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if method == "moving_average" and window > 1:
        if uniform_filter1d is not None:
            return uniform_filter1d(arr, size=window, mode="nearest")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(arr, kernel, mode="same")
    return arr


def _extract_clearance_series(raw_items: Sequence[Any], key: str = "mean") -> np.ndarray:
    values: List[float] = []
    for item in raw_items:
        value: Optional[float] = None
        if isinstance(item, dict):
            raw_value = item.get(key, None)
            if raw_value is not None:
                try:
                    value = float(raw_value)
                except Exception:
                    value = None
        elif item is not None:
            try:
                value = float(item)
            except Exception:
                value = None
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=np.float64)


def _pad_series(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    max_len = max((len(series) for series in series_list), default=0)
    if max_len == 0:
        return np.empty((0, 0), dtype=np.float64)
    padded = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for idx, series in enumerate(series_list):
        arr = np.asarray(series, dtype=np.float64)
        padded[idx, :arr.size] = arr
    return padded


def _nanmean_std(series_list: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    padded = _pad_series(series_list)
    if padded.size == 0:
        return np.array([]), np.array([])
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0, ddof=0)


def _tail_array(values: Sequence[float], tail: int = 100) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    n = min(len(values), int(tail))
    return np.asarray(values[-n:], dtype=np.float64)


def _resolve_log_dir_from_manifest(manifest_path: Path, project_logs_root: Path) -> Optional[str]:
    try:
        manifest = _load_json_file(manifest_path)
    except Exception:
        return None

    meta = manifest.get("meta", {}) if isinstance(manifest.get("meta"), dict) else {}
    exp_name_with_timestamp = meta.get("exp_name_with_timestamp")
    if exp_name_with_timestamp:
        resolved = _resolve_run_log_dir(project_logs_root, str(exp_name_with_timestamp))
        if resolved:
            return resolved
        candidate_root = project_logs_root / str(exp_name_with_timestamp)
        metric_file = (
            resolve_metric_file(str(candidate_root), "episode_rewards.json")
            or resolve_metric_file(str(candidate_root), "results.json")
        )
        if metric_file is not None:
            return str(metric_file.parent)
    label = meta.get("label") or manifest_path.stem.replace("_resolved_manifest", "")
    return find_latest_log_dir(str(label), str(project_logs_root))


def _build_seeded_exp_name_base(label: str, args) -> str:
    if not bool(getattr(args, "seed_worker", False)):
        return label
    seed = getattr(args, "batch_seed", None)
    child_tag = str(getattr(args, "child_batch_tag", "") or "").strip()
    parts = [label]
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    if child_tag:
        parts.append(child_tag)
    return "__".join(parts)


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


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _post_eval_summary_value(summary: Dict[str, Any], key: str) -> Optional[float]:
    value = _safe_float(summary.get(key))
    if value is not None:
        return value
    fallback_key = POST_EVAL_FALLBACK_KEYS.get(key)
    if fallback_key:
        return _safe_float(summary.get(fallback_key))
    return None


def _post_eval_enabled(args) -> bool:
    return not bool(getattr(args, "disable_post_eval", False))


def _resolve_post_eval_seed(args) -> int:
    if getattr(args, "post_eval_seed", None) is not None:
        return int(args.post_eval_seed)
    return int(getattr(args, "resolved_scenario_seed", 88)) + 10000


def _generate_post_eval_terrain_seeds(seed: int, episodes: int) -> List[int]:
    rng = random.Random(int(seed))
    return [int(rng.randint(1000, 99999)) for _ in range(int(episodes))]


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


def _ensure_post_eval_episode_positions(spec: Dict[str, Any]) -> Dict[str, Any]:
    if spec.get("mode") != "heldout_shared":
        return spec

    episode_positions_dir = Path(spec["episode_positions_dir"])
    terrain_seed_sequence = [int(seed) for seed in spec.get("terrain_seed_sequence", [])]
    episode_positions_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = [
        episode_positions_dir / f"episode_{idx:03d}_seed_{terrain_seed}.json"
        for idx, terrain_seed in enumerate(terrain_seed_sequence)
    ]
    needs_generation = any(not candidate.exists() for candidate in candidate_files)

    if needs_generation:
        os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")
        from generate_episode_positions import generate_all_episode_positions

        base_env_vars = {
            "MAP_SIZE": str(spec["map_size"]),
            "TERRAIN_COMPLEXITY_LEVEL": str(spec["terrain_complexity"]),
            "MOUNTAIN_MIN_DISTANCE": str(spec["mountain_min_distance"]),
        }
        generate_all_episode_positions(
            terrain_seeds=terrain_seed_sequence,
            output_dir=episode_positions_dir,
            base_env_vars=base_env_vars,
        )

    for idx, terrain_seed in enumerate(terrain_seed_sequence):
        candidate = episode_positions_dir / f"episode_{idx:03d}_seed_{terrain_seed}.json"
        if candidate.exists():
            spec["default_positions_file"] = str(candidate)
            break

    return spec


def _build_post_eval_spec(args, batch_dir: Path, positions_file: Path) -> Optional[Dict[str, Any]]:
    if not _post_eval_enabled(args):
        return None

    base_env = setup_base_env_vars(
        positions_file=positions_file,
        env_isolation="strict",
        config_mode=args.config_mode,
        scenario_seed=int(args.resolved_scenario_seed),
    )
    terrain_complexity = _safe_int(base_env.get("TERRAIN_COMPLEXITY_LEVEL")) or 3
    map_size = _safe_int(float(base_env.get("MAP_SIZE", 200))) or 200
    mountain_min_distance = _safe_int(float(base_env.get("MOUNTAIN_MIN_DISTANCE", 55))) or 55
    post_eval_seed = _resolve_post_eval_seed(args)
    spec = {
        "version": 1,
        "enabled": True,
        "mode": str(getattr(args, "post_eval_mode", "heldout_shared")),
        "episodes": int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)),
        "seed": int(post_eval_seed),
        "model_variant": str(getattr(args, "post_eval_model_variant", "final")),
        "scenario_seed": int(args.resolved_scenario_seed),
        "terrain_complexity": int(terrain_complexity),
        "map_size": int(map_size),
        "mountain_min_distance": int(mountain_min_distance),
        "use_dynamic_obstacles": bool(getattr(args, "use_dynamic_obstacles", False)),
        "use_fixed_positions": True,
        "shared_positions_file": str(positions_file),
        "default_positions_file": str(positions_file),
        "terrain_seed_sequence": [],
        "episode_positions_dir": "",
    }

    if spec["mode"] == "heldout_shared":
        spec["terrain_seed_sequence"] = _generate_post_eval_terrain_seeds(spec["seed"], spec["episodes"])
        episode_positions_dir = batch_dir / "results" / "post_eval_testset" / f"seed_{spec['seed']}" / "episode_positions"
        spec["episode_positions_dir"] = str(episode_positions_dir)
        spec = _ensure_post_eval_episode_positions(spec)
    elif spec["mode"] == "match_train_env":
        pass
    else:
        raise ValueError(f"未知 post-eval 模式: {spec['mode']}")

    spec_path = batch_dir / "results" / "post_eval_shared_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec["spec_path"] = str(spec_path)
    _save_json(spec_path, spec)
    return spec


def _resolve_post_eval_model_root(result: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []

    manifest_path = str(result.get("manifest_path", "")).strip()
    if manifest_path:
        try:
            manifest = _load_manifest(Path(manifest_path))
            meta = manifest.get("meta", {}) if isinstance(manifest.get("meta"), dict) else {}
            for key in ("exp_name_with_timestamp", "exp_name_base", "exp_name"):
                value = str(meta.get(key, "")).strip()
                if value:
                    candidates.append((Path(__file__).resolve().parent / "models" / value).resolve())
        except Exception:
            pass

    run_args = _load_results_args(result.get("log_dir", ""))
    if isinstance(run_args, dict):
        for key in ("exp_name", "exp_name_with_timestamp"):
            value = str(run_args.get(key, "")).strip()
            if value:
                candidates.append((Path(__file__).resolve().parent / "models" / value).resolve())

    log_dir = str(result.get("log_dir", "")).strip()
    if log_dir:
        log_path = Path(log_dir).resolve()
        parent_name = log_path.parent.name
        current_name = log_path.name
        candidates.append((Path(__file__).resolve().parent / "models" / parent_name).resolve())
        candidates.append((Path(__file__).resolve().parent / "models" / current_name).resolve())

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate
    return None


def _resolve_post_eval_python(result: Dict[str, Any]) -> str:
    manifest_path = str(result.get("manifest_path", "")).strip()
    if manifest_path:
        try:
            manifest = _load_manifest(Path(manifest_path))
            for candidate in (
                manifest.get("python_executable"),
                manifest.get("meta", {}).get("training_python") if isinstance(manifest.get("meta"), dict) else None,
            ):
                if candidate and Path(str(candidate)).exists():
                    return str(candidate)
        except Exception:
            pass
    training_python = _resolve_training_python()
    if training_python:
        return str(training_python)
    return sys.executable or "python3"


def _validate_post_eval_results(eval_results_json: Path, spec: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not eval_results_json.exists():
        return [f"缺少评估结果文件: {eval_results_json}"]
    data = _load_json_file(eval_results_json)

    episodes = _safe_int(data.get("episodes"))
    if episodes != int(spec["episodes"]):
        errors.append(f"episodes 不匹配: got={episodes}, expected={spec['episodes']}")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("缺少 summary 字段")
    else:
        for key in ("team_success_rate", "avg_collision_count", "avg_team_total_path_length"):
            if key not in summary:
                errors.append(f"summary 缺少关键字段: {key}")

    expected_seeds = [int(seed) for seed in spec.get("terrain_seed_sequence", [])]
    if expected_seeds:
        actual_seeds = data.get("terrain_seed_sequence", [])
        if [int(seed) for seed in actual_seeds] != expected_seeds:
            errors.append("terrain_seed_sequence 不匹配")

    return errors


def _extract_post_eval_payload(
    eval_data: Dict[str, Any],
    fallback_collision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    episode_details = eval_data.get("episode_details", []) if isinstance(eval_data.get("episode_details", []), list) else []
    summary = dict(eval_data.get("summary", {})) if isinstance(eval_data.get("summary"), dict) else {}
    if summary.get("clearance_violation_threshold") is None and fallback_collision_threshold is not None:
        summary["clearance_violation_threshold"] = float(fallback_collision_threshold)

    min_distances = []
    penetration_rates = []
    penetration_max_depths = []
    penetration_mean_depths = []
    for ep in episode_details:
        min_distance = ep.get("min_distance", None)
        if isinstance(min_distance, dict):
            min_distances.append(min_distance)
        elif min_distance is None:
            min_distances.append({"mean": 0.0, "min": 0.0})
        else:
            parsed = _safe_float(min_distance)
            min_distances.append({"mean": parsed or 0.0, "min": parsed or 0.0})

        penetration_stat = ep.get("penetration_stat")
        if isinstance(penetration_stat, dict):
            penetration_rates.append(int(penetration_stat.get("count", 0)))
            penetration_max_depths.append(float(penetration_stat.get("max_depth", 0.0)))
            penetration_mean_depths.append(float(penetration_stat.get("mean_depth", 0.0)))
        else:
            penetration_rates.append(0)
            penetration_max_depths.append(0.0)
            penetration_mean_depths.append(0.0)

    metrics = {
        "episode_rewards": eval_data.get("all_rewards", []),
        "success_flags": [ep.get("success", 0) for ep in episode_details],
        "collision_counts": [ep.get("collision_count", 0) for ep in episode_details],
        "min_distances_to_obstacle": min_distances,
        "team_success_flags": [ep.get("team_success", ep.get("success", 0)) for ep in episode_details],
        "agent_success_flags": [ep.get("agent_success_flags", []) for ep in episode_details],
        "agent_collision_counts": [ep.get("agent_collision_counts", []) for ep in episode_details],
        "arrival_steps": [ep.get("arrival_step") for ep in episode_details],
        "arrival_times": [ep.get("arrival_time") for ep in episode_details],
        "first_reach_steps": [ep.get("first_reach_step") for ep in episode_details],
        "first_reach_times": [ep.get("first_reach_time") for ep in episode_details],
        "path_lengths": [ep.get("path_length") for ep in episode_details],
        "path_efficiencies": [ep.get("path_efficiency") for ep in episode_details],
        "steps": [ep.get("steps") for ep in episode_details],
        "penetration_rates": penetration_rates,
        "penetration_max_depths": penetration_max_depths,
        "penetration_mean_depths": penetration_mean_depths,
        "collision_distance_threshold": (
            _safe_float(summary.get("clearance_violation_threshold")) or fallback_collision_threshold
        ),
    }

    return {
        "summary": summary,
        "metrics": metrics,
        "episode_details": episode_details,
    }


def _run_post_training_evaluation(
    result: Dict[str, Any],
    cfg: Dict[str, Any],
    positions_file: Path,
    args,
    batch_dir: Path,
    post_eval_spec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if post_eval_spec is None:
        return result

    label = cfg["label"]
    model_variant = str(post_eval_spec["model_variant"])
    eval_dir = batch_dir / "results" / "post_eval" / label / model_variant
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_results_json = eval_dir / "evaluation_results.json"
    eval_log_path = eval_dir / "post_eval.log"
    eval_spec_path = eval_dir / "post_eval_spec.json"
    _save_json(eval_spec_path, post_eval_spec)

    reuse_existing = False
    existing_payload = None
    if eval_results_json.exists():
        validation_errors = _validate_post_eval_results(eval_results_json, post_eval_spec)
        if not validation_errors:
            eval_data = _load_json_file(eval_results_json)
            existing_payload = _extract_post_eval_payload(
                eval_data,
                fallback_collision_threshold=result.get("metrics", {}).get("collision_distance_threshold"),
            )
            reuse_existing = True

    if not reuse_existing:
        model_root = _resolve_post_eval_model_root(result)
        if model_root is None:
            raise RuntimeError(f"[后评估-{label}] 无法定位模型目录，log_dir={result.get('log_dir')}")

        env = _build_process_env(args.env_isolation)
        env["SUPPRESS_MA_PROMPT"] = "1"
        env["STRICT_EVAL_MATCH"] = "1"
        env["EVAL_RESPECT_INPUT_POSITIONS"] = "1"
        env["EVAL_PYTHON_BIN"] = _resolve_post_eval_python(result)
        env["MODEL_VARIANT"] = model_variant
        env["QUIET_OUTPUT"] = "1"
        env["TQDM_DISABLE"] = "1"
        env["TQDM_TO_STDOUT"] = "0"
        env["EVAL_DEBUG_ACTION_STEPS"] = "0"
        env["EVAL_DISABLE_VISUALIZATION"] = "1"
        env["EVAL_LIGHT_MODE"] = "1"
        env["DISABLE_GIF"] = "1"
        env["SAVE_INTERACTIVE_TRAJ"] = "0"
        env["SAVE_EVAL_ALL_EPISODES"] = "0"
        env["SAVE_BEST_TRAJ"] = "0"
        env["NOISE_SCALE"] = "0.0"
        env["RANDOM_ACTION_PROB"] = "0.0"
        env["RANDOM_ACTION_PROB_TRAINING"] = "0.0"
        env["USE_SCENARIO_SEED"] = "1"
        env["SCENARIO_SEED"] = str(post_eval_spec["scenario_seed"])
        env["TERRAIN_COMPLEXITY_LEVEL"] = str(post_eval_spec["terrain_complexity"])
        env["MAP_SIZE"] = str(post_eval_spec["map_size"])
        env["MOUNTAIN_MIN_DISTANCE"] = str(post_eval_spec["mountain_min_distance"])
        env["USE_DYNAMIC_OBSTACLES"] = "1" if post_eval_spec.get("use_dynamic_obstacles") else "0"
        env["USE_FIXED_POSITIONS"] = "1"
        env["POSITIONS_FILE"] = str(post_eval_spec["default_positions_file"])
        if post_eval_spec.get("mode") == "heldout_shared":
            env["RANDOM_TERRAIN"] = "1"
            env["TERRAIN_SEED_SEQUENCE"] = ",".join(map(str, post_eval_spec.get("terrain_seed_sequence", [])))
            if post_eval_spec.get("episode_positions_dir"):
                env["EPISODE_POSITIONS_DIR"] = str(post_eval_spec["episode_positions_dir"])
        else:
            env["RANDOM_TERRAIN"] = "0"
            env.pop("EPISODE_POSITIONS_DIR", None)
            env.pop("TERRAIN_SEED_SEQUENCE", None)

        eval_command = [
            "/bin/bash",
            str((Path(__file__).resolve().parent / "run_evaluation.sh").resolve()),
            str(model_root),
            str(post_eval_spec["episodes"]),
            str(eval_dir),
            str(post_eval_spec["default_positions_file"]),
            "1",
            "false",
        ]

        print(f"\n{'='*70}")
        print(f"[后评估-{label}] 开始共享测试集评估")
        print(f"[后评估-{label}] 模型目录: {model_root}")
        print(f"[后评估-{label}] 测试模式: {post_eval_spec['mode']}")
        print(f"[后评估-{label}] 测试回合数: {post_eval_spec['episodes']}")
        print(f"[后评估-{label}] 结果目录: {eval_dir}")
        print(f"{'='*70}")

        with open(eval_log_path, "w", encoding="utf-8") as log_f:
            subprocess.run(
                eval_command,
                env=env,
                cwd=Path(__file__).resolve().parent,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=True,
            )

        validation_errors = _validate_post_eval_results(eval_results_json, post_eval_spec)
        if validation_errors:
            raise RuntimeError(
                f"[后评估-{label}] 评估结果校验失败: {' | '.join(validation_errors)} | 日志: {eval_log_path}"
            )
        eval_data = _load_json_file(eval_results_json)
        existing_payload = _extract_post_eval_payload(
            eval_data,
            fallback_collision_threshold=result.get("metrics", {}).get("collision_distance_threshold"),
        )

    result["post_eval_dir"] = str(eval_dir)
    result["post_eval_log_path"] = str(eval_log_path)
    result["post_eval_results_path"] = str(eval_results_json)
    result["post_eval_spec_path"] = str(eval_spec_path)
    result["post_eval_spec"] = dict(post_eval_spec)
    result["post_eval_metrics"] = existing_payload["metrics"]
    result["post_eval_summary"] = existing_payload["summary"]
    result["post_eval_episode_count"] = int(post_eval_spec["episodes"])
    return result


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
            "description": "在相同 separated update skeleton 下，MATD3 的 separated actor objective 相对 unified actor objective 的增益",
        },
        {
            "name": "matd3_dual_head_gain",
            "lhs": "matd3_single_q",
            "rhs": "matd3_dual_q",
            "required": True,
            "confounded": False,
            "description": "MATD3 主线中的 separated skeleton 相对 MATD3 baseline 的增益",
        },
        {
            "name": "mainline_vs_maddpg_separated_reference",
            "lhs": "maddpg_separated_gradient",
            "rhs": "matd3_separated_gradient",
            "required": False,
            "confounded": False,
            "description": "跨家族参考：MATD3 mainline vs MADDPG 对应的 separated skeleton + separated actor objective 版本",
        },
        {
            "name": "mainline_vs_maddpg_dual_reference",
            "lhs": "maddpg_dual_q",
            "rhs": "matd3_separated_gradient",
            "required": False,
            "confounded": False,
            "description": "跨家族参考：MATD3 主线 vs MADDPG separated skeleton + unified actor objective",
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
        
        color = get_algorithm_ablation_color(item.get("label"), idx=idx)
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
        
        color = get_algorithm_ablation_color(item.get("label"), idx=idx)
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


def _collect_post_eval_series(series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    collected = []
    for item in series:
        metrics = item.get("post_eval_metrics")
        summary = item.get("post_eval_summary")
        if not isinstance(metrics, dict) or not isinstance(summary, dict):
            continue
        collected.append(
            {
                "label": item["label"],
                "name": item.get("name", item["label"]),
                "name_en": item.get("name_en", item.get("name", item["label"])),
                "description": item.get("description", ""),
                "metrics": metrics,
                "summary": summary,
                "log_dir": item.get("post_eval_dir", ""),
            }
        )
    return collected


def _format_post_eval_value(value: Optional[float], is_percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if is_percent:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _plot_post_eval_summary_dashboard(series: List[Dict[str, Any]], title: str, output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    if not series:
        return

    setup_english_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    labels = [item.get("name_en", item["label"]) for item in series]
    colors = [get_algorithm_ablation_color(item.get("label"), idx=i) for i, item in enumerate(series)]

    for ax, (metric_key, metric_title, is_percent, lower_better) in zip(axes, POST_EVAL_SUMMARY_SPECS):
        values = [_post_eval_summary_value(item.get("summary", {}), metric_key) for item in series]
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            ax.set_visible(False)
            continue
        x = np.arange(len(series))
        bar_values = [value if value is not None else 0.0 for value in values]
        bars = ax.bar(x, bar_values, color=colors, alpha=0.9)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        if is_percent:
            ax.set_ylabel("Rate")
            ax.set_ylim(bottom=0.0, top=max(1.0, max(valid_values) * 1.15))
        else:
            ymin = min(0.0, min(valid_values) * (0.9 if not lower_better else 1.1))
            ymax = max(valid_values) * 1.15 if max(valid_values) > 0 else 1.0
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                ax.set_ylim(ymin, ymax)
        for idx, bar in enumerate(bars):
            value = values[idx]
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                _format_post_eval_value(value, is_percent=is_percent),
                ha='center',
                va='bottom',
                fontsize=9,
            )

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_post_eval_arrival_path_comparison(series: List[Dict[str, Any]], title: str, output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    if not series:
        return

    setup_english_fonts()
    fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
    has_data = False

    for idx, item in enumerate(series):
        metrics = item["metrics"]
        color = get_algorithm_ablation_color(item.get("label"), idx=idx)
        name_en = item.get("name_en", item["label"])

        first_reach_steps = metrics.get("first_reach_steps") or metrics.get("arrival_steps", [])
        path_lengths = metrics.get("path_lengths", [])
        path_efficiencies = metrics.get("path_efficiencies", [])
        episodes = np.arange(1, max(len(first_reach_steps), len(path_lengths), len(path_efficiencies)) + 1)
        if len(episodes) == 0:
            continue
        has_data = True

        if first_reach_steps:
            y = [np.nan if value is None else float(value) for value in first_reach_steps]
            axes[0].plot(range(1, len(y) + 1), y, label=name_en, color=color, linewidth=2.2, alpha=0.9)
        if path_lengths:
            y = [np.nan if value is None else float(value) for value in path_lengths]
            axes[1].plot(range(1, len(y) + 1), y, label=name_en, color=color, linewidth=2.2, alpha=0.9)
        if path_efficiencies:
            y = [np.nan if value is None else float(value) for value in path_efficiencies]
            axes[2].plot(range(1, len(y) + 1), y, label=name_en, color=color, linewidth=2.2, alpha=0.9)

    if not has_data:
        plt.close(fig)
        return

    axes[0].set_title(f"{title} - First Reach Step", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Step")
    axes[1].set_title(f"{title} - Team Path Length", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Length (m)")
    axes[2].set_title(f"{title} - Team Path Efficiency", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Efficiency")
    axes[2].set_xlabel("Evaluation Episode")

    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _write_post_eval_summary_text(series: List[Dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write("Post-training Evaluation Summary\n")
        f.write("=" * 88 + "\n\n")
        for item in series:
            summary = item.get("summary", {})
            f.write(f"[{item.get('name_en', item['label'])}]\n")
            for metric_key, metric_title, is_percent, _ in POST_EVAL_SUMMARY_SPECS:
                value = _post_eval_summary_value(summary, metric_key)
                f.write(f"  - {metric_title}: {_format_post_eval_value(value, is_percent=is_percent)}\n")
            f.write(
                f"  - Average First Reach Step: {_format_post_eval_value(_safe_float(summary.get('avg_first_reach_step')))}\n"
            )
            f.write(
                f"  - Average Team Path Efficiency: {_format_post_eval_value(_safe_float(summary.get('avg_team_path_efficiency')))}\n"
            )
            f.write(
                f"  - Average Agent Path Length: {_format_post_eval_value(_safe_float(summary.get('avg_agent_path_length')))}\n"
            )
            f.write(
                f"  - Clearance Violation Rate: {_format_post_eval_value(_safe_float(summary.get('clearance_violation_rate')), is_percent=True)}\n"
            )
            f.write("\n")


def _plot_multi_seed_post_eval_dashboard(aggregates: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    items = [
        aggregates[label]
        for label in EXPERIMENT_DISPLAY_ORDER
        if label in aggregates and isinstance(aggregates[label].get("post_eval_metric_stats"), dict)
    ]
    if not items:
        return

    setup_english_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    labels = [item.get("name_en", item.get("label")) for item in items]
    colors = [get_algorithm_ablation_color(item.get("label"), idx=i) for i, item in enumerate(items)]

    for ax, (metric_key, metric_title, is_percent, lower_better) in zip(axes, POST_EVAL_SUMMARY_SPECS):
        metric_rows = [item.get("post_eval_metric_stats", {}).get(metric_key, {}) for item in items]
        values = [_safe_float(row.get("mean")) for row in metric_rows]
        errors = [_safe_float(row.get("std")) or 0.0 for row in metric_rows]
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            ax.set_visible(False)
            continue
        x = np.arange(len(items))
        bar_values = [value if value is not None else 0.0 for value in values]
        bars = ax.bar(x, bar_values, yerr=errors, color=colors, alpha=0.9, capsize=4)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        if is_percent:
            ax.set_ylim(bottom=0.0, top=max(1.0, max(valid_values) * 1.2))
        else:
            ymin = min(0.0, min(valid_values) * (0.9 if not lower_better else 1.1))
            ymax = max(valid_values) * 1.2 if max(valid_values) > 0 else 1.0
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                ax.set_ylim(ymin, ymax)
        for idx, bar in enumerate(bars):
            if values[idx] is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                _format_post_eval_value(values[idx], is_percent=is_percent),
                ha='center',
                va='bottom',
                fontsize=9,
            )

    fig.suptitle("Multi-seed Post-training Evaluation Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _write_single_seed_outputs(
    series: List[Dict[str, Any]],
    args,
    batch_dir: Path,
    output_dir: Path,
    positions_file: Path,
    batch_seed: int,
    experiment_group: str,
    group_desc: str,
    strict_validity_enabled: bool,
    claims_report: Dict[str, Any],
) -> Dict[str, Any]:
    title = f"MATD3 Separated-Skeleton / Actor-Objective Ablation - Group {experiment_group} ({group_desc})"
    eval_title = f"Post-training Evaluation - Group {experiment_group} ({group_desc})"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_files: Dict[str, str] = {}

    if not args.skip_local_plots:
        reward_png = output_dir / f"reward_comparison_{timestamp}.png"
        plot_comparison_rewards_dualq(
            series,
            title,
            reward_png,
            smooth_window=args.smooth_window,
            fit_method=args.fit_method,
        )
        output_files["reward_comparison"] = str(reward_png.name)

        loss_png = output_dir / f"loss_comparison_{timestamp}.png"
        plot_comparison_losses_dualq(series, title, loss_png)
        output_files["loss_comparison"] = str(loss_png.name)

        success_collision_png = output_dir / f"success_collision_clearance_comparison_{timestamp}.png"
        plot_comparison_success_collision_clearance(
            series,
            title,
            success_collision_png,
            smooth_window=args.smooth_window,
            fit_method=args.fit_method,
        )
        output_files["success_collision_clearance_comparison"] = str(success_collision_png.name)

        success_clearance_png = output_dir / f"success_rate_and_clearance_comparison_{timestamp}.png"
        plot_comparison_success_rate_and_clearance(
            series,
            title,
            success_clearance_png,
            smooth_window=args.smooth_window,
            fit_method=args.fit_method,
        )
        output_files["success_rate_and_clearance_comparison"] = str(success_clearance_png.name)

        if args.generate_interactive:
            interactive_html = output_dir / f"interactive_comparison_{timestamp}.html"
            generate_interactive_comparison(
                series,
                title,
                interactive_html,
                smooth_window=args.smooth_window,
                fit_method=args.fit_method,
            )
            output_files["interactive_comparison"] = str(interactive_html.name)

    post_eval_series = _collect_post_eval_series(series)
    if post_eval_series and not args.skip_local_plots:
        eval_reward_png = output_dir / f"post_eval_reward_comparison_{timestamp}.png"
        plot_comparison_rewards_dualq(
            post_eval_series,
            eval_title,
            eval_reward_png,
            smooth_window=max(1, min(args.smooth_window, 5)),
            fit_method=args.fit_method,
        )
        output_files["post_eval_reward_comparison"] = str(eval_reward_png.name)

        eval_success_png = output_dir / f"post_eval_success_collision_clearance_comparison_{timestamp}.png"
        plot_comparison_success_collision_clearance(
            post_eval_series,
            eval_title,
            eval_success_png,
            smooth_window=max(1, min(args.smooth_window, 5)),
            fit_method=args.fit_method,
        )
        output_files["post_eval_success_collision_clearance_comparison"] = str(eval_success_png.name)

        eval_clearance_png = output_dir / f"post_eval_success_rate_and_clearance_comparison_{timestamp}.png"
        plot_comparison_success_rate_and_clearance(
            post_eval_series,
            eval_title,
            eval_clearance_png,
            smooth_window=max(1, min(args.smooth_window, 5)),
            fit_method=args.fit_method,
        )
        output_files["post_eval_success_rate_and_clearance_comparison"] = str(eval_clearance_png.name)

        eval_arrival_path_png = output_dir / f"post_eval_arrival_path_comparison_{timestamp}.png"
        _plot_post_eval_arrival_path_comparison(post_eval_series, eval_title, eval_arrival_path_png)
        output_files["post_eval_arrival_path_comparison"] = str(eval_arrival_path_png.name)

        eval_dashboard_png = output_dir / f"post_eval_summary_dashboard_{timestamp}.png"
        _plot_post_eval_summary_dashboard(post_eval_series, eval_title, eval_dashboard_png)
        output_files["post_eval_summary_dashboard"] = str(eval_dashboard_png.name)

        eval_summary_txt = output_dir / f"post_eval_summary_{timestamp}.txt"
        _write_post_eval_summary_text(post_eval_series, eval_summary_txt)
        output_files["post_eval_summary_text"] = str(eval_summary_txt.name)

    summary = {
        "summary_mode": "single_seed",
        "timestamp": timestamp,
        "seed": int(batch_seed),
        "experiment_group": experiment_group,
        "experiment_group_desc": group_desc,
        "batch_dir": str(batch_dir),
        "positions_file": str(positions_file),
        "use_dynamic_obstacles": getattr(args, "use_dynamic_obstacles", False),
        "strict_validity_enabled": strict_validity_enabled,
        "skip_local_plots": bool(args.skip_local_plots),
        "post_eval_enabled": _post_eval_enabled(args),
        "post_eval_mode": getattr(args, "post_eval_mode", "heldout_shared"),
        "post_eval_episodes": int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)),
        "post_eval_seed": int(getattr(args, "resolved_post_eval_seed", _resolve_post_eval_seed(args))),
        "post_eval_model_variant": getattr(args, "post_eval_model_variant", "final"),
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
                "avg_reward": float(np.mean(item["metrics"].get("episode_rewards", []))) if item["metrics"].get("episode_rewards") else None,
                "max_reward": float(np.max(item["metrics"].get("episode_rewards", []))) if item["metrics"].get("episode_rewards") else None,
                "collision_distance_threshold": item["metrics"].get("collision_distance_threshold"),
                "collision_threshold_source": item["metrics"].get("collision_threshold_source", "unknown"),
                "post_eval": {
                    "enabled": item.get("post_eval_summary") is not None,
                    "mode": item.get("post_eval_spec", {}).get("mode") if isinstance(item.get("post_eval_spec"), dict) else None,
                    "episodes": item.get("post_eval_episode_count"),
                    "seed": item.get("post_eval_spec", {}).get("seed") if isinstance(item.get("post_eval_spec"), dict) else None,
                    "model_variant": item.get("post_eval_spec", {}).get("model_variant") if isinstance(item.get("post_eval_spec"), dict) else None,
                    "eval_dir": item.get("post_eval_dir", ""),
                    "results_path": item.get("post_eval_results_path", ""),
                    "log_path": item.get("post_eval_log_path", ""),
                    "spec_path": item.get("post_eval_spec_path", ""),
                    "summary": item.get("post_eval_summary"),
                },
            }
            for item in series
        ],
        "output_files": output_files,
    }

    summary_path = output_dir / f"summary_{timestamp}.json"
    _save_json(summary_path, summary)
    latest_summary_path = output_dir / "latest_summary.json"
    _save_json(latest_summary_path, summary)

    print(f"\n{'='*70}")
    print(f"Ablation run complete - Group {experiment_group} ({group_desc})")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    for key, filename in output_files.items():
        print(f"{key}: {filename}")
    print(f"{'='*70}")
    return summary


def _aggregate_multi_seed_runs(child_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}

    for child_summary in child_summaries:
        seed = child_summary.get("seed")
        child_batch_dir = child_summary.get("batch_dir", "")
        child_summary_path = child_summary.get("summary_path", "")
        for exp in child_summary.get("experiments", []):
            label = str(exp.get("label", "")).strip()
            log_dir = str(exp.get("log_dir", "")).strip()
            if not label or not log_dir:
                continue
            metrics = load_metrics(log_dir)
            row = aggregated.setdefault(
                label,
                {
                    "label": label,
                    "name": exp.get("name", label),
                    "name_en": exp.get("name_en", exp.get("name", label)),
                    "description": exp.get("description", ""),
                    "runs": [],
                },
            )
            reward_curve = np.asarray(metrics.get("episode_rewards", []), dtype=np.float64)
            success_curve = _moving_average_binary(metrics.get("team_success_flags", []), window=50)
            collision_curve = np.asarray(metrics.get("collision_counts", []), dtype=np.float64)
            clearance_curve = _extract_clearance_series(metrics.get("min_distances_to_obstacle", []), key="mean")

            tail_reward = _tail_array(metrics.get("episode_rewards", []), tail=100)
            tail_success = _tail_array(metrics.get("team_success_flags", []), tail=100)
            tail_collision = _tail_array(metrics.get("collision_counts", []), tail=100)
            tail_clearance = _tail_array(clearance_curve.tolist(), tail=100)

            row["runs"].append(
                {
                    "seed": seed,
                    "log_dir": log_dir,
                    "manifest_path": exp.get("manifest_path", ""),
                    "summary_path": child_summary_path,
                    "batch_dir": child_batch_dir,
                    "metrics": metrics,
                    "reward_curve": reward_curve,
                    "success_curve": success_curve,
                    "collision_curve": collision_curve,
                    "clearance_curve": clearance_curve,
                    "tail100_reward_mean": float(np.nanmean(tail_reward)) if tail_reward.size else None,
                    "tail100_success_mean": float(np.nanmean(tail_success)) if tail_success.size else None,
                    "tail100_collision_mean": float(np.nanmean(tail_collision)) if tail_collision.size else None,
                    "tail100_clearance_mean": float(np.nanmean(tail_clearance)) if tail_clearance.size else None,
                    "post_eval_summary": (
                        exp.get("post_eval", {}).get("summary")
                        if isinstance(exp.get("post_eval"), dict)
                        else None
                    ),
                }
            )

    for item in aggregated.values():
        reward_mean, reward_std = _nanmean_std([run["reward_curve"] for run in item["runs"]])
        success_mean, success_std = _nanmean_std([run["success_curve"] for run in item["runs"]])
        collision_mean, collision_std = _nanmean_std([run["collision_curve"] for run in item["runs"]])
        clearance_mean, clearance_std = _nanmean_std([run["clearance_curve"] for run in item["runs"]])
        reward_tail_values = [run["tail100_reward_mean"] for run in item["runs"] if run["tail100_reward_mean"] is not None]
        success_tail_values = [run["tail100_success_mean"] for run in item["runs"] if run["tail100_success_mean"] is not None]
        collision_tail_values = [run["tail100_collision_mean"] for run in item["runs"] if run["tail100_collision_mean"] is not None]
        clearance_tail_values = [run["tail100_clearance_mean"] for run in item["runs"] if run["tail100_clearance_mean"] is not None]
        item["curve_stats"] = {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "success_mean": success_mean,
            "success_std": success_std,
            "collision_mean": collision_mean,
            "collision_std": collision_std,
            "clearance_mean": clearance_mean,
            "clearance_std": clearance_std,
        }
        item["seed_count"] = len(item["runs"])
        item["tail100_reward_mean"] = float(np.nanmean(reward_tail_values)) if reward_tail_values else None
        item["tail100_reward_std"] = float(np.nanstd(reward_tail_values, ddof=0)) if reward_tail_values else None
        item["tail100_success_mean"] = float(np.nanmean(success_tail_values)) if success_tail_values else None
        item["tail100_success_std"] = float(np.nanstd(success_tail_values, ddof=0)) if success_tail_values else None
        item["tail100_collision_mean"] = float(np.nanmean(collision_tail_values)) if collision_tail_values else None
        item["tail100_collision_std"] = float(np.nanstd(collision_tail_values, ddof=0)) if collision_tail_values else None
        item["tail100_clearance_mean"] = float(np.nanmean(clearance_tail_values)) if clearance_tail_values else None
        item["tail100_clearance_std"] = float(np.nanstd(clearance_tail_values, ddof=0)) if clearance_tail_values else None

        post_eval_metric_stats: Dict[str, Dict[str, Optional[float]]] = {}
        for metric_key, _, _, _ in POST_EVAL_SUMMARY_SPECS:
            metric_values = [
                _post_eval_summary_value(run.get("post_eval_summary", {}) or {}, metric_key)
                for run in item["runs"]
            ]
            metric_values = [value for value in metric_values if value is not None]
            if not metric_values:
                continue
            post_eval_metric_stats[metric_key] = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values, ddof=0)),
                "count": float(len(metric_values)),
            }
        item["post_eval_metric_stats"] = post_eval_metric_stats

    return aggregated


def _write_multi_seed_audit_report(batch_dir: Path, audit_report: Dict[str, Any]) -> Path:
    results_dir = batch_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    audit_path = results_dir / f"multi_seed_audit_{timestamp}.json"
    _save_json(audit_path, audit_report)
    latest_audit_path = results_dir / "latest_multi_seed_audit.json"
    _save_json(latest_audit_path, audit_report)
    return audit_path


def _audit_multi_seed_children(
    child_summaries: Sequence[Dict[str, Any]],
    expected_labels: Sequence[str],
    expected_seeds: Sequence[int],
    expected_positions_file: Path,
    expected_post_eval: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_label_set = {str(label) for label in expected_labels}
    expected_seed_set = {int(seed) for seed in expected_seeds}
    expected_positions_resolved = str(expected_positions_file.resolve())

    errors: List[str] = []
    warnings: List[str] = []
    children: List[Dict[str, Any]] = []
    seen_seeds: set[int] = set()

    for child_summary in sorted(child_summaries, key=lambda row: (_safe_int(row.get("seed")) is None, _safe_int(row.get("seed")))):
        child_errors: List[str] = []
        child_warnings: List[str] = []

        seed = _safe_int(child_summary.get("seed"))
        if seed is None:
            child_errors.append("summary 缺少 seed")
        elif seed not in expected_seed_set:
            child_errors.append(f"seed={seed} 不在请求列表中: {sorted(expected_seed_set)}")
        elif seed in seen_seeds:
            child_errors.append(f"seed={seed} 出现重复 summary")
        else:
            seen_seeds.add(seed)

        if child_summary.get("summary_mode") != "single_seed":
            child_errors.append(f"summary_mode 异常: {child_summary.get('summary_mode')}")

        child_positions_raw = str(child_summary.get("positions_file", "")).strip()
        if not child_positions_raw:
            child_errors.append("summary 缺少 positions_file")
        else:
            child_positions_resolved = str(Path(child_positions_raw).resolve())
            if child_positions_resolved != expected_positions_resolved:
                child_errors.append(
                    "positions_file 不一致: "
                    f"got={child_positions_resolved}, expected={expected_positions_resolved}"
                )

        experiments = child_summary.get("experiments", [])
        if not isinstance(experiments, list):
            child_errors.append("experiments 字段缺失或类型错误")
            experiments = []

        label_to_exp: Dict[str, Dict[str, Any]] = {}
        duplicate_labels: List[str] = []
        for exp in experiments:
            label = str(exp.get("label", "")).strip()
            if not label:
                continue
            if label in label_to_exp:
                duplicate_labels.append(label)
            label_to_exp[label] = exp
        if duplicate_labels:
            child_errors.append(f"存在重复实验标签: {sorted(set(duplicate_labels))}")

        available_labels = set(label_to_exp.keys())
        missing_labels = sorted(expected_label_set - available_labels)
        extra_labels = sorted(available_labels - expected_label_set)
        if missing_labels:
            child_errors.append(f"缺少实验结果: {missing_labels}")
        if extra_labels:
            child_warnings.append(f"存在未请求的额外实验: {extra_labels}")

        if expected_post_eval is not None:
            summary_post_eval = {
                "enabled": bool(child_summary.get("post_eval_enabled")),
                "mode": child_summary.get("post_eval_mode"),
                "episodes": _safe_int(child_summary.get("post_eval_episodes")),
                "seed": _safe_int(child_summary.get("post_eval_seed")),
                "model_variant": child_summary.get("post_eval_model_variant"),
            }
            if summary_post_eval != expected_post_eval:
                child_errors.append(
                    "child summary 的后评估配置与父批次不一致: "
                    f"got={summary_post_eval}, expected={expected_post_eval}"
                )

            for label in sorted(expected_label_set):
                exp = label_to_exp.get(label)
                if exp is None:
                    continue
                post_eval = exp.get("post_eval")
                if not isinstance(post_eval, dict):
                    child_errors.append(f"{label}: 缺少 post_eval 字段")
                    continue
                if not post_eval.get("enabled"):
                    child_errors.append(f"{label}: 后评估未完成")
                    continue

                if post_eval.get("mode") != expected_post_eval["mode"]:
                    child_errors.append(
                        f"{label}: post_eval.mode 不一致 got={post_eval.get('mode')} expected={expected_post_eval['mode']}"
                    )
                if _safe_int(post_eval.get("episodes")) != expected_post_eval["episodes"]:
                    child_errors.append(
                        f"{label}: post_eval.episodes 不一致 got={post_eval.get('episodes')} expected={expected_post_eval['episodes']}"
                    )
                if _safe_int(post_eval.get("seed")) != expected_post_eval["seed"]:
                    child_errors.append(
                        f"{label}: post_eval.seed 不一致 got={post_eval.get('seed')} expected={expected_post_eval['seed']}"
                    )
                if post_eval.get("model_variant") != expected_post_eval["model_variant"]:
                    child_errors.append(
                        f"{label}: post_eval.model_variant 不一致 got={post_eval.get('model_variant')} expected={expected_post_eval['model_variant']}"
                    )

                results_path_raw = str(post_eval.get("results_path", "")).strip()
                if not results_path_raw:
                    child_errors.append(f"{label}: 缺少 post_eval results_path")
                elif not Path(results_path_raw).exists():
                    child_errors.append(f"{label}: post_eval results 不存在: {results_path_raw}")

                summary_payload = post_eval.get("summary")
                if not isinstance(summary_payload, dict):
                    child_errors.append(f"{label}: post_eval summary 缺失或类型错误")

        children.append(
            {
                "seed": seed,
                "batch_dir": child_summary.get("batch_dir", ""),
                "summary_path": child_summary.get("summary_path", ""),
                "errors": child_errors,
                "warnings": child_warnings,
                "status": "passed" if not child_errors else "failed",
            }
        )
        for message in child_errors:
            errors.append(f"seed={seed}: {message}")
        for message in child_warnings:
            warnings.append(f"seed={seed}: {message}")

    missing_seed_summaries = sorted(expected_seed_set - seen_seeds)
    if missing_seed_summaries:
        errors.append(f"缺少 seed summary: {missing_seed_summaries}")

    return {
        "passed": len(errors) == 0,
        "expected_labels": sorted(expected_label_set),
        "expected_seeds": sorted(expected_seed_set),
        "expected_positions_file": expected_positions_resolved,
        "expected_post_eval": expected_post_eval,
        "children": children,
        "errors": errors,
        "warnings": warnings,
    }


def _evaluate_multi_seed_claims(aggregated: Dict[str, Dict[str, Any]], selected_labels: set) -> Dict[str, Any]:
    claims = [
        {
            "name": "matd3_mainline_vs_unified_actor_loss",
            "lhs": "matd3_dual_q",
            "rhs": "matd3_separated_gradient",
            "required": True,
        },
        {
            "name": "matd3_dual_head_gain",
            "lhs": "matd3_single_q",
            "rhs": "matd3_dual_q",
            "required": True,
        },
    ]

    evaluated = []
    required_failed = []
    for claim in claims:
        lhs = claim["lhs"]
        rhs = claim["rhs"]
        if lhs not in selected_labels or rhs not in selected_labels:
            evaluated.append({**claim, "status": "skipped", "paired_seed_count": 0, "delta_mean": None, "delta_std": None})
            continue

        lhs_runs = {
            run.get("seed"): run.get("tail100_reward_mean")
            for run in aggregated.get(lhs, {}).get("runs", [])
            if run.get("seed") is not None and run.get("tail100_reward_mean") is not None
        }
        rhs_runs = {
            run.get("seed"): run.get("tail100_reward_mean")
            for run in aggregated.get(rhs, {}).get("runs", [])
            if run.get("seed") is not None and run.get("tail100_reward_mean") is not None
        }
        paired_seeds = sorted(seed for seed in lhs_runs.keys() if seed in rhs_runs)
        deltas = [float(rhs_runs[seed] - lhs_runs[seed]) for seed in paired_seeds]
        if not deltas:
            status = "invalid_data"
            delta_mean = None
            delta_std = None
        else:
            status = "valid"
            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas, ddof=0))

        row = {
            **claim,
            "status": status,
            "paired_seed_count": len(paired_seeds),
            "paired_seeds": paired_seeds,
            "paired_deltas_rhs_minus_lhs": deltas,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        evaluated.append(row)
        if claim.get("required", False) and status != "valid":
            required_failed.append(f"{claim['name']} status={status}")

    return {
        "claims": evaluated,
        "required_failed": required_failed,
        "required_pass": len(required_failed) == 0,
    }


def plot_seed_overlay_by_experiment(
    aggregates: Dict[str, Dict[str, Any]],
    output_path: Path,
    smooth_window: int = 10,
    fit_method: str = "moving_average",
) -> None:
    labels = [label for label in EXPERIMENT_DISPLAY_ORDER if label in aggregates]
    if not labels:
        return

    setup_english_fonts()
    fig, axes = plt.subplots(2, len(labels), figsize=(6 * len(labels), 9), squeeze=False)
    for col_idx, label in enumerate(labels):
        item = aggregates[label]
        reward_ax = axes[0, col_idx]
        success_ax = axes[1, col_idx]
        runs = sorted(item.get("runs", []), key=lambda row: (row.get("seed") is None, row.get("seed")))
        palette = plt.cm.get_cmap("tab10", max(1, len(runs)))

        for idx, run in enumerate(runs):
            seed = run.get("seed")
            reward_curve = np.asarray(run.get("reward_curve", []), dtype=np.float64)
            success_curve = np.asarray(run.get("success_curve", []), dtype=np.float64)
            color = palette(idx % max(1, len(runs)))
            label_text = f"seed={seed}" if seed is not None else f"run{idx + 1}"
            if reward_curve.size:
                xs = np.arange(1, reward_curve.size + 1)
                reward_ax.plot(xs, _smooth_curve(reward_curve, method=fit_method, window=smooth_window), color=color, linewidth=2.0, alpha=0.9, label=label_text)
            if success_curve.size:
                xs = np.arange(1, success_curve.size + 1)
                success_ax.plot(xs, success_curve, color=color, linewidth=2.0, alpha=0.9, label=label_text)

        reward_ax.set_title(item.get("name_en", label), fontsize=12, fontweight="bold")
        reward_ax.set_ylabel("Reward")
        reward_ax.grid(True, alpha=0.3, linestyle="--")
        reward_ax.legend(loc="best", fontsize=8)

        success_ax.set_title("Team Success Rate", fontsize=12, fontweight="bold")
        success_ax.set_xlabel("Episode")
        success_ax.set_ylabel("Success Rate")
        success_ax.set_ylim([0.0, 1.05])
        success_ax.grid(True, alpha=0.3, linestyle="--")
        success_ax.legend(loc="best", fontsize=8)

    fig.suptitle("Multi-seed Overlay by Experiment", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_multi_seed_mean_ablation_comparison(
    aggregates: Dict[str, Dict[str, Any]],
    output_path: Path,
    smooth_window: int = 10,
    fit_method: str = "moving_average",
) -> None:
    labels = [label for label in EXPERIMENT_DISPLAY_ORDER if label in aggregates]
    if not labels:
        return

    setup_english_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False)
    specs = [
        ("reward", "Reward", axes[0, 0]),
        ("success", "Team Success Rate", axes[0, 1]),
        ("collision", "Collision Count", axes[1, 0]),
        ("clearance", "Average Clearance (m)", axes[1, 1]),
    ]

    for label in labels:
        item = aggregates[label]
        color = get_algorithm_ablation_color(label)
        name_en = item.get("name_en", label)
        stats = item.get("curve_stats", {})
        for metric_name, title, ax in specs:
            mean_curve = np.asarray(stats.get(f"{metric_name}_mean", []), dtype=np.float64)
            std_curve = np.asarray(stats.get(f"{metric_name}_std", []), dtype=np.float64)
            if mean_curve.size == 0:
                continue
            if metric_name != "success":
                mean_curve = _smooth_curve(mean_curve, method=fit_method, window=smooth_window)
                std_curve = _smooth_curve(std_curve, method=fit_method, window=smooth_window)
            xs = np.arange(1, mean_curve.size + 1)
            ax.plot(xs, mean_curve, color=color, linewidth=2.5, alpha=0.95, label=name_en)
            ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3, linestyle="--")
            if metric_name == "success":
                ax.set_ylim([0.0, 1.05])

    axes[0, 0].set_ylabel("Reward")
    axes[0, 1].set_ylabel("Success Rate")
    axes[1, 0].set_ylabel("Collisions")
    axes[1, 1].set_ylabel("Clearance (m)")
    for ax in axes.flat:
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Multi-seed Mean Ablation Comparison (mean ± std)", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



# ============================================================================
# 实验配置
# ============================================================================

# ============================================================================
# 实验配置
# 主线：MATD3 separated skeleton + separated actor objective（run_optimized 当前默认主方法）
# 默认严格实验：围绕 separated skeleton / actor-objective 做模块消融
# 可选参考实验：MADDPG 家族对照，不参与主线模块归因
# ============================================================================

EXPERIMENT_CONFIGS = [
    {
        "label": "matd3_separated_gradient",
        "name": "MATD3 Mainline - Separated Skeleton + Separated Actor Objective",
        "name_en": "MATD3 Mainline - Separated Skeleton + Separated Actor Objective",
        "description": "Current mainline method: separated update skeleton with raw/corrected split action semantics and separated actor objective",
        "env": {
            "ALGORITHM": "matd3",
            "MATD3_USE_DUAL_Q": "1",
            "MATD3_USE_SEPARATED_GRADIENT": "1",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "matd3_dual_q",
        "name": "MATD3 Ablation - Separated Skeleton + Unified Actor Objective",
        "name_en": "MATD3 Ablation - Separated Skeleton + Unified Actor Objective",
        "description": "Ablation on the same separated update skeleton: replace the separated actor objective with a unified total-Q actor objective",
        "env": {
            "ALGORITHM": "matd3",
            "MATD3_USE_DUAL_Q": "1",
            "MATD3_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "matd3_single_q",
        "name": "MATD3 Baseline",
        "name_en": "MATD3 Baseline",
        "description": "Baseline experiment: standard MATD3-style twin critics with a single total-Q output per critic and a joint actor update on executed corrected actions",
        "env": {
            "ALGORITHM": "matd3",
            "MATD3_USE_DUAL_Q": "0",
            "MATD3_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "maddpg_separated_gradient",
        "name": "MADDPG Reference - Separated Skeleton + Separated Actor Objective",
        "name_en": "MADDPG Reference - Separated Skeleton + Separated Actor Objective",
        "description": "Reference only: MADDPG framework with the same separated update skeleton and separated actor objective for 7D actions",
        "env": {
            "ALGORITHM": "maddpg",
            "MADDPG_USE_DUAL_Q": "1",
            "MADDPG_USE_SEPARATED_GRADIENT": "1",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    },
    {
        "label": "maddpg_dual_q",
        "name": "MADDPG Reference - Separated Skeleton + Unified Actor Objective",
        "name_en": "MADDPG Reference - Separated Skeleton + Unified Actor Objective",
        "description": "Reference only: MADDPG framework with the same separated update skeleton and unified total-Q actor objective for 7D actions",
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
        "description": "Baseline reference: MADDPG framework with a single total-Q critic output and a joint actor update on executed corrected actions",
        "env": {
            "ALGORITHM": "maddpg",
            "MADDPG_USE_DUAL_Q": "0",
            "MADDPG_USE_SEPARATED_GRADIENT": "0",
            "USE_TF_POTENTIAL_FIELD": "1",
        }
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="围绕 MATD3 separated-skeleton / actor-objective 主线的模块消融实验")
    parser.add_argument("--episodes", type=int, default=200, help="训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024, help="训练批次大小")
    parser.add_argument("--multi-seed", action="store_true", help="开启多随机种子模式，由当前脚本并发调度多个单-seed子批次")
    parser.add_argument("--seeds", type=str, default=None, help="多seed列表，逗号分隔，例如 101,202,303")
    parser.add_argument("--max-parallel", type=int, default=2, help="多seed模式下最多并发多少个子批次；0表示全部同时启动")
    parser.add_argument("--batch-seed", type=int, default=None, help="单seed worker模式下显式指定训练seed")
    parser.add_argument("--seed-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-batch-root", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-batch-tag", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--positions-prepared", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-local-plots", action="store_true", help="跳过单seed子批次本地图表，只保留summary，适合多seed模式提速")
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
    parser.add_argument("--disable-post-eval", action="store_true", help="关闭训练完成后的共享测试集评估")
    parser.add_argument("--post-eval-episodes", type=int, default=DEFAULT_POST_EVAL_EPISODES, help="训练完成后共享测试集评估的回合数")
    parser.add_argument("--post-eval-seed", type=int, default=None, help="共享测试集评估的固定随机种子；默认由 scenario-seed 派生")
    parser.add_argument(
        "--post-eval-mode",
        type=str,
        default="heldout_shared",
        choices=["heldout_shared", "match_train_env"],
        help="后评估模式：heldout_shared=共享留出测试集；match_train_env=严格复现训练环境",
    )
    parser.add_argument(
        "--post-eval-model-variant",
        type=str,
        default="final",
        choices=["auto", "final", "best", "best_by_team_sr", "latest_ep"],
        help="后评估使用的模型检查点变体",
    )
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        choices=[cfg["label"] for cfg in EXPERIMENT_CONFIGS],
                        help="选择要运行的实验；默认只运行 MATD3 核心 separated-skeleton / actor-objective 消融")
    parser.add_argument(
        "--include-reference-experiments",
        "--include-exploratory-experiments",
        dest="include_reference_experiments",
        action="store_true",
        help="默认只运行 MATD3 核心 separated-skeleton / actor-objective 消融；显式开启后再纳入 MADDPG 参考实验",
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


def _apply_experiment_group_overrides(args) -> Tuple[str, str]:
    experiment_group = args.experiment_group
    group_label = f"group{experiment_group}"
    group_desc = "Fixed Map" if experiment_group == "A" else "Random Obstacles"
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
    return group_label, group_desc


def _select_experiment_configs(args) -> List[Dict[str, Any]]:
    if args.experiments:
        configs_to_run = [cfg for cfg in EXPERIMENT_CONFIGS if cfg["label"] in args.experiments]
        configs_to_run = _sort_experiment_configs(configs_to_run)
        if not configs_to_run:
            raise RuntimeError(f"未找到指定实验: {args.experiments}")
        reference_selected = sorted(
            label for label in args.experiments if label in OPTIONAL_REFERENCE_EXPERIMENT_LABELS
        )
        if reference_selected:
            print(f"[提示] 已显式纳入参考实验（不参与主线模块归因）: {reference_selected}")
        return configs_to_run

    default_labels = list(STRICT_CORE_EXPERIMENT_LABELS)
    if args.include_reference_experiments:
        default_labels.extend(OPTIONAL_REFERENCE_EXPERIMENT_LABELS)
    configs_to_run = [cfg for cfg in EXPERIMENT_CONFIGS if cfg["label"] in default_labels]
    return _sort_experiment_configs(configs_to_run)


def _prepare_positions_for_batch(args, positions_file: Path) -> None:
    args.resolved_positions_file = str(positions_file)
    if args.positions_prepared:
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[位置文件] 复用父调度器准备好的共享位置文件: {positions_file}")
        return

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


def _create_batch_dir(
    args,
    batch_seed: int,
    positions_file: Path,
    configs_to_run: List[Dict[str, Any]],
    group_label: str,
    group_desc: str,
    batch_mode: str,
) -> Tuple[Path, Path]:
    if getattr(args, "parent_batch_root", None):
        batch_root = Path(args.parent_batch_root)
    elif AblationBatchManager is None:
        batch_root = Path(f"{args.output_dir}_{group_label}")
    else:
        batch_root = Path("ablation_experiments")
    batch_root.mkdir(parents=True, exist_ok=True)
    explicit_batch_id = str(getattr(args, "child_batch_tag", "") or "").strip()
    if explicit_batch_id:
        batch_id = explicit_batch_id
    else:
        batch_id = f"batch_{group_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
        "experiment_group": args.experiment_group,
        "experiment_group_desc": group_desc,
        "use_dynamic_obstacles": getattr(args, "use_dynamic_obstacles", False),
        "unlock_env_on_success": int(args.resolved_unlock_env_on_success),
        "unlock_env_on_plateau": int(args.resolved_unlock_env_on_plateau),
        "positions_file": str(positions_file),
        "post_eval_enabled": _post_eval_enabled(args),
        "post_eval_mode": getattr(args, "post_eval_mode", "heldout_shared"),
        "post_eval_episodes": int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)),
        "post_eval_seed": int(getattr(args, "resolved_post_eval_seed", _resolve_post_eval_seed(args))),
        "post_eval_model_variant": getattr(args, "post_eval_model_variant", "final"),
        "batch_mode": batch_mode,
        "notes": (
            f"MATD3 separated-skeleton / actor-objective ablation: "
            f"default core experiments={STRICT_CORE_EXPERIMENT_LABELS}, "
            f"optional reference experiments={OPTIONAL_REFERENCE_EXPERIMENT_LABELS}, "
            f"Group {args.experiment_group} ({group_desc}), mode={args.config_mode}, batch_mode={batch_mode}"
        ),
    }

    if AblationBatchManager is not None:
        manager = AblationBatchManager(root_dir=str(batch_root))
        batch_dir = manager.create_batch(
            batch_id=batch_id,
            config=batch_config,
            experiments=[c["label"] for c in configs_to_run],
        )
    else:
        batch_dir = batch_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "plots").mkdir(parents=True, exist_ok=True)
        (batch_dir / "results").mkdir(parents=True, exist_ok=True)
        for cfg in configs_to_run:
            (batch_dir / cfg["label"]).mkdir(parents=True, exist_ok=True)
        _save_json(batch_dir / "config.json", {**batch_config, "experiments": [c["label"] for c in configs_to_run]})

    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_file = batch_dir / "shared_seed.json"
    _save_json(
        seed_file,
        {
            "seed": batch_seed,
            "source": "cli" if getattr(args, "batch_seed", None) is not None else "random",
            "experiment_group": args.experiment_group,
            "batch_mode": batch_mode,
        },
    )
    print(f"{'='*70}")
    print(f"✅ 批次目录已创建: {batch_dir}")
    print(f"  Group: {args.experiment_group} ({group_desc})")
    print(f"  共享种子: {batch_seed} (已保存至 {seed_file})")
    print(f"  模式: {batch_mode}")
    print(f"{'='*70}\n")
    return batch_dir, output_dir


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
    env["AUTO_EVAL"] = "0"

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
    exp_name_base = _build_seeded_exp_name_base(label, args)
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
        exp_name_base,
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
        "exp_name_base": exp_name_base,
    })
    _save_json(manifest_path, manifest)
    return manifest, manifest_path


def run_experiment(
    cfg: Dict,
    positions_file: Path,
    args,
    cache: Dict[str, Dict],
    batch_dir: Path,
    post_eval_spec: Optional[Dict[str, Any]] = None,
) -> Dict:
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
            reused = {
                "label": label,
                "name": cfg.get("name", label),
                "name_en": cfg.get("name_en", cfg.get("name", label)),
                "description": cfg.get("description", ""),
                "log_dir": log_dir,
                "metrics": metrics,
                "success": True
            }
            reused = _run_post_training_evaluation(
                reused, cfg, positions_file, args, batch_dir, post_eval_spec
            )
            cache[label] = reused
            return reused
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
    log_dir = _resolve_log_dir_from_manifest(Path(manifest_path), project_logs_root)
    if not log_dir:
        log_dir = _resolve_current_run_log_dir(project_logs_root, label, before_run_dirs)
    if not log_dir:
        log_dir = find_latest_log_dir(label, str(project_logs_root))
    metrics = load_metrics(log_dir)
    n_ep = len(metrics.get("episode_rewards", []))
    print(f"[运行-{label}] 加载指标: {log_dir!r}, episode数={n_ep}", file=sys.stderr)
    if n_ep == 0:
        raise RuntimeError(
            f"[运行-{label}] 本次运行未生成可恢复的 episode reward 数据: {log_dir}. "
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
    result = _run_post_training_evaluation(
        result, cfg, positions_file, args, batch_dir, post_eval_spec
    )
    cache[label] = result
    return result


# ============================================================================
# 主函数
# ============================================================================

def _launch_worker_job(job: Dict[str, Any]) -> None:
    job["launcher_log"].parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(job["launcher_log"], "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            job["command"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
        )
    except Exception:
        log_handle.close()
        raise
    job["process"] = process
    job["log_handle"] = log_handle
    job["started_at"] = time.time()


def _close_worker_job_log(job: Dict[str, Any]) -> None:
    log_handle = job.pop("log_handle", None)
    if log_handle is not None:
        log_handle.close()


def _terminate_active_worker_jobs(active_jobs: Sequence[Dict[str, Any]]) -> None:
    for job in active_jobs:
        process = job.get("process")
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
    time.sleep(1.0)
    for job in active_jobs:
        process = job.get("process")
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass
        _close_worker_job_log(job)


def _write_multi_seed_outputs(
    batch_dir: Path,
    args,
    positions_file: Path,
    seeds: List[int],
    child_runs: List[Dict[str, Any]],
    aggregates: Dict[str, Dict[str, Any]],
    claims_report_multi_seed: Dict[str, Any],
    group_desc: str,
    audit_report: Optional[Dict[str, Any]] = None,
    audit_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_files: Dict[str, str] = {}
    if audit_report_path is not None:
        output_files["multi_seed_audit_report"] = str(audit_report_path)

    overlay_png = output_dir / f"seed_overlay_by_experiment_{timestamp}.png"
    plot_seed_overlay_by_experiment(
        aggregates,
        overlay_png,
        smooth_window=args.smooth_window,
        fit_method=args.fit_method,
    )
    output_files["seed_overlay_by_experiment"] = overlay_png.name

    mean_png = output_dir / f"multi_seed_mean_ablation_comparison_{timestamp}.png"
    plot_multi_seed_mean_ablation_comparison(
        aggregates,
        mean_png,
        smooth_window=args.smooth_window,
        fit_method=args.fit_method,
    )
    output_files["multi_seed_mean_ablation_comparison"] = mean_png.name

    has_post_eval = any(item.get("post_eval_metric_stats") for item in aggregates.values())
    if has_post_eval:
        post_eval_png = output_dir / f"multi_seed_post_eval_summary_{timestamp}.png"
        _plot_multi_seed_post_eval_dashboard(aggregates, post_eval_png)
        output_files["multi_seed_post_eval_summary"] = post_eval_png.name

    aggregated_rows = []
    for label in EXPERIMENT_DISPLAY_ORDER:
        if label not in aggregates:
            continue
        item = aggregates[label]
        stats = item.get("curve_stats", {})
        aggregated_rows.append(
            {
                "label": label,
                "name": item.get("name", label),
                "name_en": item.get("name_en", label),
                "description": item.get("description", ""),
                "seed_count": item.get("seed_count", 0),
                "seed_values": [
                    {
                        "seed": run.get("seed"),
                        "log_dir": run.get("log_dir"),
                        "manifest_path": run.get("manifest_path"),
                        "summary_path": run.get("summary_path"),
                        "batch_dir": run.get("batch_dir"),
                        "tail100_reward_mean": run.get("tail100_reward_mean"),
                        "tail100_success_mean": run.get("tail100_success_mean"),
                        "tail100_collision_mean": run.get("tail100_collision_mean"),
                        "tail100_clearance_mean": run.get("tail100_clearance_mean"),
                        "post_eval_summary": run.get("post_eval_summary"),
                    }
                    for run in sorted(item.get("runs", []), key=lambda row: (row.get("seed") is None, row.get("seed")))
                ],
                "reward_mean_curve": np.asarray(stats.get("reward_mean", []), dtype=np.float64).tolist(),
                "reward_std_curve": np.asarray(stats.get("reward_std", []), dtype=np.float64).tolist(),
                "success_mean_curve": np.asarray(stats.get("success_mean", []), dtype=np.float64).tolist(),
                "success_std_curve": np.asarray(stats.get("success_std", []), dtype=np.float64).tolist(),
                "collision_mean_curve": np.asarray(stats.get("collision_mean", []), dtype=np.float64).tolist(),
                "collision_std_curve": np.asarray(stats.get("collision_std", []), dtype=np.float64).tolist(),
                "clearance_mean_curve": np.asarray(stats.get("clearance_mean", []), dtype=np.float64).tolist(),
                "clearance_std_curve": np.asarray(stats.get("clearance_std", []), dtype=np.float64).tolist(),
                "tail100_reward_mean": item.get("tail100_reward_mean"),
                "tail100_reward_std": item.get("tail100_reward_std"),
                "tail100_success_mean": item.get("tail100_success_mean"),
                "tail100_success_std": item.get("tail100_success_std"),
                "tail100_collision_mean": item.get("tail100_collision_mean"),
                "tail100_collision_std": item.get("tail100_collision_std"),
                "tail100_clearance_mean": item.get("tail100_clearance_mean"),
                "tail100_clearance_std": item.get("tail100_clearance_std"),
                "post_eval_metric_stats": item.get("post_eval_metric_stats", {}),
            }
        )

    summary = {
        "summary_mode": "multi_seed_parent",
        "timestamp": timestamp,
        "seeds": seeds,
        "max_parallel": args.max_parallel if args.max_parallel > 0 else len(seeds),
        "experiment_group": args.experiment_group,
        "experiment_group_desc": group_desc,
        "batch_dir": str(batch_dir),
        "positions_file": str(positions_file),
        "skip_local_plots_for_children": bool(args.skip_local_plots),
        "post_eval_enabled": _post_eval_enabled(args),
        "post_eval_mode": getattr(args, "post_eval_mode", "heldout_shared"),
        "post_eval_episodes": int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)),
        "post_eval_seed": int(getattr(args, "resolved_post_eval_seed", _resolve_post_eval_seed(args))),
        "post_eval_model_variant": getattr(args, "post_eval_model_variant", "final"),
        "child_runs": child_runs,
        "aggregated_experiments": aggregated_rows,
        "claims_report_multi_seed": claims_report_multi_seed,
        "audit_report": audit_report,
        "audit_report_path": str(audit_report_path) if audit_report_path is not None else "",
        "output_files": output_files,
    }

    summary_path = output_dir / f"summary_{timestamp}.json"
    _save_json(summary_path, summary)
    latest_summary_path = output_dir / "latest_summary.json"
    _save_json(latest_summary_path, summary)

    print(f"\n{'='*70}")
    print(f"Multi-seed ablation complete - Group {args.experiment_group} ({group_desc})")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    for key, filename in output_files.items():
        print(f"{key}: {filename}")
    print(f"{'='*70}")
    return summary


def run_single_seed_batch(args) -> int:
    group_label, group_desc = _apply_experiment_group_overrides(args)
    parsed_seeds = getattr(args, "parsed_seeds", [])
    if getattr(args, "batch_seed", None) is not None:
        batch_seed = int(args.batch_seed)
        seed_source = "cli"
    elif not args.seed_worker and len(parsed_seeds) == 1 and not args.multi_seed:
        batch_seed = int(parsed_seeds[0])
        seed_source = "seeds"
    else:
        batch_seed = random.randint(100000, 999999)
        seed_source = "random"
    print(f"[种子管理] 本批次训练种子: {batch_seed} (source={seed_source})")
    args.batch_seed = batch_seed

    positions_file = _resolve_positions_file(
        raw_positions_file=args.positions_file,
        scenario_seed=int(args.resolved_scenario_seed),
        experiment_group=args.experiment_group,
    ).resolve()
    _prepare_positions_for_batch(args, positions_file)

    configs_to_run = _select_experiment_configs(args)
    batch_mode = "seed_worker" if args.seed_worker else "single_seed"
    batch_dir, output_dir = _create_batch_dir(
        args=args,
        batch_seed=batch_seed,
        positions_file=positions_file,
        configs_to_run=configs_to_run,
        group_label=group_label,
        group_desc=group_desc,
        batch_mode=batch_mode,
    )

    manifest_dir = batch_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir = str(manifest_dir)
    args.reference_manifest = None
    args.reference_manifest_label = None
    post_eval_spec = _build_post_eval_spec(args, batch_dir, positions_file)

    print(f"\n{'='*70}")
    print("MATD3 Separated-Skeleton / Actor-Objective Ablation")
    print(f"Group: {args.experiment_group} ({group_desc})")
    print(f"实验数量: {len(configs_to_run)}")
    for cfg in configs_to_run:
        print(f"  - {cfg['name']}: {cfg['description']}")
    print(f"模式: {'单seed worker' if args.seed_worker else '单seed'}")
    print(f"配置模式: {args.config_mode}")
    print(f"环境隔离: {args.env_isolation}")
    print(f"动态障碍物: {'启用' if getattr(args, 'use_dynamic_obstacles', False) else '禁用'}")
    print(f"课程学习: 已禁用 (success=0, plateau=0)")
    print(f"固定位置文件: {positions_file}")
    print(f"复用策略: reuse={args.reuse}, reuse_only={args.reuse_only}")
    print(f"训练种子: {batch_seed}")
    print(f"场景Seed: {args.resolved_scenario_seed}")
    print(f"配置清单目录: {manifest_dir}")
    print(f"严格有效性校验: {'关闭' if args.disable_strict_validity else '开启'}")
    print(f"跳过本地图表: {'是' if args.skip_local_plots else '否'}")
    if post_eval_spec is not None:
        print(
            f"后评估: 开启 | mode={post_eval_spec['mode']} | episodes={post_eval_spec['episodes']} | "
            f"seed={post_eval_spec['seed']} | model_variant={post_eval_spec['model_variant']}"
        )
        if post_eval_spec.get("episode_positions_dir"):
            print(f"后评估共享测试集目录: {post_eval_spec['episode_positions_dir']}")
    else:
        print("后评估: 关闭")
    print(f"{'='*70}\n")

    series: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, Any]] = {}
    for cfg in configs_to_run:
        result = run_experiment(cfg, positions_file, args, cache, batch_dir, post_eval_spec)
        series.append(result)

    if not series:
        raise RuntimeError("没有可用的实验数据，无法生成对比图")

    selected_labels = {cfg["label"] for cfg in configs_to_run}
    available_labels = {item.get("label") for item in series}
    missing_labels = sorted(label for label in selected_labels if label not in available_labels)
    strict_validity_enabled = not bool(args.disable_strict_validity)
    if missing_labels:
        msg = f"缺少实验结果: {missing_labels}"
        if strict_validity_enabled:
            raise RuntimeError(f"[严格校验失败] {msg}")
        print(f"[警告] {msg}")

    claims_report = _evaluate_claims(series, selected_labels)
    print("[有效性检查] 对比声明状态：")
    for row in claims_report["claims"]:
        delta = row.get("tail100_delta_rhs_minus_lhs")
        delta_str = f"{delta:.2f}" if isinstance(delta, (int, float)) else "N/A"
        print(f"  - {row['name']}: {row['status']} (tail100 Δ={delta_str})")
    if not claims_report["required_pass"]:
        print("[有效性检查] 核心声明检查失败：")
        for item in claims_report["required_failed"]:
            print(f"  - {item}")
        if strict_validity_enabled:
            raise RuntimeError("[严格校验失败] 请先修复上述问题后再生成消融结论。")
        print("[警告] 严格校验已关闭，继续生成图表。")

    _write_single_seed_outputs(
        series=series,
        args=args,
        batch_dir=batch_dir,
        output_dir=output_dir,
        positions_file=positions_file,
        batch_seed=batch_seed,
        experiment_group=args.experiment_group,
        group_desc=group_desc,
        strict_validity_enabled=strict_validity_enabled,
        claims_report=claims_report,
    )
    return 0


def run_multi_seed_parent(args) -> int:
    seeds = list(getattr(args, "parsed_seeds", []))
    if not seeds:
        raise RuntimeError("多seed模式需要通过 --seeds 提供至少一个随机种子")

    group_label, group_desc = _apply_experiment_group_overrides(args)
    configs_to_run = _select_experiment_configs(args)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_batch_id = f"multi_seed_{group_label}_{run_stamp}"

    batch_root = Path("ablation_experiments")
    batch_root.mkdir(parents=True, exist_ok=True)
    parent_config = {
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "use_weighted_reward": args.use_weighted_reward,
        "scenario_seed": int(args.resolved_scenario_seed),
        "env_isolation": args.env_isolation,
        "config_mode": args.config_mode,
        "experiment_group": args.experiment_group,
        "experiment_group_desc": group_desc,
        "use_dynamic_obstacles": getattr(args, "use_dynamic_obstacles", False),
        "positions_file": str(
            _resolve_positions_file(
                raw_positions_file=args.positions_file,
                scenario_seed=int(args.resolved_scenario_seed),
                experiment_group=args.experiment_group,
            ).resolve()
        ),
        "seeds": seeds,
        "max_parallel": args.max_parallel if args.max_parallel > 0 else len(seeds),
        "post_eval_enabled": _post_eval_enabled(args),
        "post_eval_mode": getattr(args, "post_eval_mode", "heldout_shared"),
        "post_eval_episodes": int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)),
        "post_eval_seed": int(getattr(args, "resolved_post_eval_seed", _resolve_post_eval_seed(args))),
        "post_eval_model_variant": getattr(args, "post_eval_model_variant", "final"),
        "batch_mode": "multi_seed_parent",
    }
    if AblationBatchManager is not None:
        manager = AblationBatchManager(root_dir=str(batch_root))
        batch_dir = manager.create_batch(batch_id=parent_batch_id, config=parent_config, experiments=[])
    else:
        batch_dir = batch_root / parent_batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "plots").mkdir(parents=True, exist_ok=True)
        (batch_dir / "results").mkdir(parents=True, exist_ok=True)
        _save_json(batch_dir / "config.json", {**parent_config, "experiments": []})

    launcher_logs_dir = batch_dir / "launcher_logs"
    launcher_logs_dir.mkdir(parents=True, exist_ok=True)
    seed_batches_root = batch_dir / "seed_batches"
    seed_batches_root.mkdir(parents=True, exist_ok=True)

    positions_file = Path(parent_config["positions_file"]).resolve()
    args.resolved_positions_file = str(positions_file)
    positions_prepared = False
    bootstrap_seed: Optional[int] = None
    if args.config_mode in ("strict_ablation", "legacy_ablation"):
        args.positions_prepared = False
        _prepare_positions_for_batch(args, positions_file)
        positions_prepared = True
    else:
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        positions_prepared = positions_file.exists()
        if not positions_prepared:
            bootstrap_seed = int(seeds[0])
            print(
                "[多seed] run_optimized_default 且共享位置文件尚未存在，"
                f"将先由 seed={bootstrap_seed} 启动并写出 positions 文件，然后再释放其他 seed。"
            )

    print(f"\n{'='*70}")
    print("MATD3 Multi-seed Ablation Parent")
    print(f"Group: {args.experiment_group} ({group_desc})")
    print(f"实验数量: {len(configs_to_run)}")
    print(f"随机种子列表: {seeds}")
    print(f"最大并发: {args.max_parallel if args.max_parallel > 0 else len(seeds)}")
    print(f"父批次目录: {batch_dir}")
    print(f"子批次根目录: {seed_batches_root}")
    print(f"共享位置文件: {positions_file}")
    print(f"{'='*70}\n")

    experiment_labels = [cfg["label"] for cfg in configs_to_run]
    jobs: List[Dict[str, Any]] = []
    for seed in seeds:
        child_tag = f"batch_{group_label}_seed{int(seed)}_{run_stamp}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--episodes",
            str(args.episodes),
            "--batch-size",
            str(args.batch_size),
            "--script",
            str(args.script),
            "--use-weighted-reward",
            str(args.use_weighted_reward),
            "--positions-file",
            str(positions_file),
            "--env-isolation",
            str(args.env_isolation),
            "--config-mode",
            str(args.config_mode),
            "--scenario-seed",
            str(args.resolved_scenario_seed),
            "--smooth-window",
            str(args.smooth_window),
            "--fit-method",
            str(args.fit_method),
            "--experiment-group",
            str(args.experiment_group),
            "--logs-root",
            str(args.logs_root),
            "--post-eval-episodes",
            str(args.post_eval_episodes),
            "--post-eval-seed",
            str(getattr(args, "resolved_post_eval_seed", _resolve_post_eval_seed(args))),
            "--post-eval-mode",
            str(args.post_eval_mode),
            "--post-eval-model-variant",
            str(args.post_eval_model_variant),
            "--batch-seed",
            str(seed),
            "--seed-worker",
            "--parent-batch-root",
            str(seed_batches_root),
            "--child-batch-tag",
            child_tag,
            "--experiments",
            *experiment_labels,
        ]
        if args.reuse:
            command.append("--reuse")
        if args.reuse_only:
            command.append("--reuse-only")
        if args.disable_strict_validity:
            command.append("--disable-strict-validity")
        if args.disable_post_eval:
            command.append("--disable-post-eval")
        if args.skip_local_plots:
            command.append("--skip-local-plots")
        if args.generate_interactive and not args.skip_local_plots:
            command.append("--generate-interactive")

        jobs.append(
            {
                "seed": int(seed),
                "child_tag": child_tag,
                "command": command,
                "launcher_log": launcher_logs_dir / f"seed_{int(seed)}.log",
                "summary_path": seed_batches_root / child_tag / "plots" / "latest_summary.json",
                "batch_dir": seed_batches_root / child_tag,
                "positions_prepared": bool(positions_prepared and seed != bootstrap_seed),
            }
        )

    pending_jobs = list(jobs)
    active_jobs: List[Dict[str, Any]] = []
    completed_jobs: List[Dict[str, Any]] = []
    max_parallel = args.max_parallel if args.max_parallel > 0 else len(jobs)

    if bootstrap_seed is not None:
        bootstrap_job = next(job for job in pending_jobs if job["seed"] == bootstrap_seed)
        pending_jobs = [job for job in pending_jobs if job["seed"] != bootstrap_seed]
        if positions_prepared:
            bootstrap_job["command"].append("--positions-prepared")
        _launch_worker_job(bootstrap_job)
        active_jobs.append(bootstrap_job)
        bootstrap_released = positions_prepared
    else:
        bootstrap_job = None
        bootstrap_released = True

    try:
        while pending_jobs or active_jobs:
            if bootstrap_job is not None and not bootstrap_released:
                if positions_file.exists():
                    bootstrap_released = True
                    print(f"[多seed] 检测到共享位置文件已生成: {positions_file}")
                elif bootstrap_job.get("process") is not None and bootstrap_job["process"].poll() is not None:
                    if not positions_file.exists():
                        raise RuntimeError(
                            f"bootstrap seed={bootstrap_job['seed']} 退出后仍未生成共享 positions 文件: {positions_file}"
                        )
                    bootstrap_released = True

            while pending_jobs and len(active_jobs) < max_parallel and bootstrap_released:
                job = pending_jobs.pop(0)
                if job["positions_prepared"] or bootstrap_released:
                    job["command"].append("--positions-prepared")
                _launch_worker_job(job)
                active_jobs.append(job)
                print(f"[多seed] 已启动 seed={job['seed']} -> {job['launcher_log']}")

            time.sleep(1.0)
            for job in list(active_jobs):
                process = job.get("process")
                if process is None:
                    continue
                returncode = process.poll()
                if returncode is None:
                    continue
                job["returncode"] = int(returncode)
                job["elapsed_sec"] = float(time.time() - job.get("started_at", time.time()))
                _close_worker_job_log(job)
                active_jobs.remove(job)
                completed_jobs.append(job)
                print(f"[多seed] seed={job['seed']} 已结束，returncode={returncode}")
    except KeyboardInterrupt:
        print("[多seed] 收到中断信号，正在终止仍在运行的子批次...")
        _terminate_active_worker_jobs(active_jobs)
        raise

    child_runs: List[Dict[str, Any]] = []
    child_summaries: List[Dict[str, Any]] = []
    for job in sorted(completed_jobs, key=lambda row: row["seed"]):
        row = {
            "seed": job["seed"],
            "child_tag": job["child_tag"],
            "launcher_log": str(job["launcher_log"]),
            "batch_dir": str(job["batch_dir"]),
            "summary_path": str(job["summary_path"]),
            "returncode": job.get("returncode"),
            "elapsed_sec": job.get("elapsed_sec"),
            "status": "completed" if job.get("returncode") == 0 else "failed",
        }
        if job.get("returncode") == 0 and job["summary_path"].exists():
            summary_data = _load_json_file(job["summary_path"])
            summary_data["summary_path"] = str(job["summary_path"])
            child_summaries.append(summary_data)
        else:
            row["status"] = "missing_summary" if job.get("returncode") == 0 else "failed"
        child_runs.append(row)

    if not child_summaries:
        raise RuntimeError("所有多seed子批次都失败了，无法生成汇总图")

    expected_post_eval = None
    if _post_eval_enabled(args):
        expected_post_eval = {
            "enabled": True,
            "mode": getattr(args, "post_eval_mode", "heldout_shared"),
            "episodes": int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)),
            "seed": int(getattr(args, "resolved_post_eval_seed", _resolve_post_eval_seed(args))),
            "model_variant": getattr(args, "post_eval_model_variant", "final"),
        }

    audit_report = _audit_multi_seed_children(
        child_summaries=child_summaries,
        expected_labels=experiment_labels,
        expected_seeds=seeds,
        expected_positions_file=positions_file,
        expected_post_eval=expected_post_eval,
    )
    audit_report_path = _write_multi_seed_audit_report(batch_dir, audit_report)
    if not audit_report.get("passed", False):
        raise RuntimeError(f"多seed子批次完整性/后评估审计失败，详情见: {audit_report_path}")

    aggregates = _aggregate_multi_seed_runs(child_summaries)
    selected_labels = {cfg["label"] for cfg in configs_to_run}
    claims_report_multi_seed = _evaluate_multi_seed_claims(aggregates, selected_labels)
    _write_multi_seed_outputs(
        batch_dir=batch_dir,
        args=args,
        positions_file=positions_file,
        seeds=seeds,
        child_runs=child_runs,
        aggregates=aggregates,
        claims_report_multi_seed=claims_report_multi_seed,
        group_desc=group_desc,
        audit_report=audit_report,
        audit_report_path=audit_report_path,
    )

    failed_children = [row for row in child_runs if row["status"] != "completed"]
    if failed_children:
        raise RuntimeError(f"存在失败的多seed子批次: {[row['seed'] for row in failed_children]}")
    return 0


def main():
    args = parse_args()
    args.parsed_seeds = _parse_seed_list(args.seeds)
    args.resolved_scenario_seed = _resolve_scenario_seed(args.config_mode, args.scenario_seed)
    args.resolved_post_eval_seed = _resolve_post_eval_seed(args)
    (
        args.resolved_unlock_env_on_success,
        args.resolved_unlock_env_on_plateau,
    ) = _resolve_unlock_thresholds(
        config_mode=args.config_mode,
        unlock_env_on_success=args.unlock_env_on_success,
        unlock_env_on_plateau=args.unlock_env_on_plateau,
    )

    if int(getattr(args, "post_eval_episodes", DEFAULT_POST_EVAL_EPISODES)) <= 0:
        raise RuntimeError("--post-eval-episodes 必须为正整数")

    if args.seed_worker and args.batch_seed is None:
        raise RuntimeError("--seed-worker 模式必须显式提供 --batch-seed")

    should_run_multi_seed = (not args.seed_worker) and (bool(args.multi_seed) or len(args.parsed_seeds) > 1)
    if should_run_multi_seed:
        return run_multi_seed_parent(args)
    return run_single_seed_batch(args)


if __name__ == "__main__":
    sys.exit(main())

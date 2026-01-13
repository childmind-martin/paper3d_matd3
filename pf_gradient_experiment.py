#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pf_gradient_experiment.py
=========================

（已弃用）势场梯度解耦实验脚本。

说明
----
- 训练主代码中的势场梯度解耦逻辑（PF_DECOUPLE_GRAD）已经移除，当前版本中
  无论 PF_DECOUPLE_GRAD 取值为何，势场参数始终参与梯度更新。
- 因此，本脚本中的 baseline / decoupled / both 三种模式在当前实现下**实际等价**，
  仅实验名称不同，不再建议继续使用本脚本做对比实验。
- 之所以保留本文件，仅作为历史记录，方便日后查阅实验配置。

原始设计目标（仅供参考）
----------------------
1. 不直接修改 `run_optimized.sh` 和 `paper3d_train_optimized.py` 的主流程。
2. 通过环境变量切换「是否对势场参数使用 stop_gradient」的模式：
   - PF_DECOUPLE_GRAD=0  → 原始模式（势场参数参与梯度）
   - PF_DECOUPLE_GRAD=1  → 解耦实验模式（势场参数仅用于前向，不参与梯度）
3. 在同一套固定起点/目标下，对比两种模式的训练表现。
"""

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Literal


ROOT = pathlib.Path(__file__).resolve().parent


def _run_shell(cmd, env):
    """小工具：以子进程方式调用命令，并将输出直接透传到终端。"""
    print("\n[pf_experiment] 运行命令:", " ".join(cmd))
    sys.stdout.flush()
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"命令执行失败，退出码={completed.returncode}")


def ensure_fixed_positions(positions_file: pathlib.Path,
                           episodes: int,
                           batch_size: int,
                           exp_name_prefix: str):
    """
    若指定的固定位置文件不存在，则运行一次带 DYNAMIC_FIRST_TIME 的训练，
    仅用于生成固定起点/终点文件。
    """
    if positions_file.exists():
        print(f"[pf_experiment] 检测到已存在的固定位置文件: {positions_file}")
        return

    print(f"[pf_experiment] 固定位置文件不存在，开始一次性生成: {positions_file}")

    env = os.environ.copy()
    # 生成固定位置：仅第一次动态，其后全部固定；这里强制一个极简、可复现实验环境
    env["USE_FIXED_POSITIONS"] = "1"
    env["DYNAMIC_FIRST_TIME"] = "1"
    env["SAVE_POSITIONS"] = "1"
    env["POSITIONS_FILE"] = str(positions_file)
    # 固定地形与课程制：完全关闭地图解锁与随机地形，确保只生成一套起点/目标
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    # 使用固定场景种子，保证地形+起点可复现
    env.setdefault("USE_SCENARIO_SEED", "1")
    env.setdefault("SCENARIO_SEED", "43")
    # 为避免多个并行环境写入不同起点，这里强制 NUM_ENVS=1
    env["NUM_ENVS"] = "4"
    # 为了让位置生成阶段更稳定，这里显式关闭全局XLA，仅使用常规GPU执行
    # 说明：生成固定位置只跑1个回合，对性能几乎无影响，但可以避免XLA偶发的 CUDA_ERROR_ILLEGAL_ADDRESS
    env["XLA_GLOBAL"] = "1"
    env.setdefault("PF_JIT", "0")
    env.setdefault("PF_DECOUPLE_GRAD", "0")  # 生成固定位置时使用原始梯度模式

    exp_name = f"{exp_name_prefix}_init_positions"
    cmd = [
        "bash",
        str(ROOT / "run_optimized.sh"),
        "1",               # 只需要1个回合完成动态生成
        str(batch_size),
        exp_name,
        "1",               # USE_WEIGHTED_REWARD
        "matd3",           # ALGORITHM
    ]
    _run_shell(cmd, env)

    if not positions_file.exists():
        raise RuntimeError(f"[pf_experiment] 生成固定位置失败，文件仍不存在: {positions_file}")


def run_single_mode(mode: Literal["baseline", "decoupled"],
                    episodes: int,
                    batch_size: int,
                    exp_name_prefix: str,
                    positions_file: pathlib.Path):
    """
    运行单次实验：
    - baseline  : PF_DECOUPLE_GRAD=0
    - decoupled : PF_DECOUPLE_GRAD=1
    """
    env = os.environ.copy()
    env["USE_FIXED_POSITIONS"] = "1"
    env["DYNAMIC_FIRST_TIME"] = "0"  # 固定位置已存在，只使用固定位置
    env["POSITIONS_FILE"] = str(positions_file)
    # 冻结地形与课程制，保证baseline/decoupled完全在同一套起点/地形上比较
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    env.setdefault("USE_SCENARIO_SEED", "1")
    env.setdefault("SCENARIO_SEED", "43")

    # 为了避免 XLA Global + 复杂图在本实验中触发 CUDA_ERROR_ILLEGAL_ADDRESS，
    # 这里只关闭全局XLA，加速仍然来自GPU本身，对20回合的小实验影响有限。
    env["XLA_GLOBAL"] = "0"
    # 势场JIT保持关闭，避免额外的XLA路径
    env.setdefault("PF_JIT", "0")

    if mode == "decoupled":
        env["PF_DECOUPLE_GRAD"] = "1"
    else:
        env["PF_DECOUPLE_GRAD"] = "0"

    exp_name = f"{exp_name_prefix}_{mode}"

    cmd = [
        "bash",
        str(ROOT / "run_optimized.sh"),
        str(episodes),
        str(batch_size),
        exp_name,
        "1",          # USE_WEIGHTED_REWARD=true
        "matd3",      # 使用当前主训练算法
    ]
    _run_shell(cmd, env)


def parse_args():
    parser = argparse.ArgumentParser(description="势场梯度解耦实验调度脚本")
    parser.add_argument("--mode",
                        type=str,
                        default="decoupled",
                        choices=["baseline", "decoupled", "both"],
                        help="实验模式：baseline / decoupled / both(先baseline再decoupled)")
    parser.add_argument("--episodes", type=int, default=20, help="每个实验的训练回合数")
    parser.add_argument("--batch-size", type=int, default=1024, help="训练批次大小（传给 run_optimized.sh）")
    parser.add_argument("--exp-name-prefix", type=str, default="PF_grad_experiment",
                        help="实验名称前缀，会自动附加 _baseline / _decoupled")
    parser.add_argument("--positions-file", type=str, default="./saved_positions/pf_grad_exp.json",
                        help="固定起点/终点位置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    positions_path = pathlib.Path(args.positions_file).resolve()
    positions_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 确保存在固定位置文件
    ensure_fixed_positions(positions_path, args.episodes, args.batch_size, args.exp_name_prefix)

    # 2) 按模式运行实验
    if args.mode in ("baseline", "both"):
        print("\n[pf_experiment] 运行 baseline 模式 (PF_DECOUPLE_GRAD=0)")
        run_single_mode("baseline", args.episodes, args.batch_size, args.exp_name_prefix, positions_path)

    if args.mode in ("decoupled", "both"):
        print("\n[pf_experiment] 运行 decoupled 模式 (PF_DECOUPLE_GRAD=1)")
        run_single_mode("decoupled", args.episodes, args.batch_size, args.exp_name_prefix, positions_path)

    print("\n[pf_experiment] 实验完成。可以在 logs/ 下对比 baseline 与 decoupled 两个目录的奖励和轨迹。")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
从日志中的 init_ep*.json 生成可用于固定训练的 positions.json，
并输出一条一键命令示例：

用法：
  python3 tools/make_fixed_from_init.py \
      --init logs/optimized_exp/episode_3/init_ep3.json \
      --out saved_positions/from_ep3.json \
      [--print-run]

可选：
  若 init_json 中包含 seed，则会顺带打印 USE_SCENARIO_SEED/SCENARIO_SEED 的用法。
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True, help="init_ep*.json 路径")
    parser.add_argument("--out", required=True, help="输出 positions.json 路径")
    parser.add_argument("--exp", default="fixed_retrain", help="建议的新的实验名")
    parser.add_argument("--print-run", action="store_true", help="打印一键运行命令示例")
    args = parser.parse_args()

    init_path = Path(args.init)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(init_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agents = data.get("agents")
    goal = data.get("goal")
    if not agents or goal is None:
        raise SystemExit("init json 缺少 agents/goal 字段")

    fixed = {"agents": agents, "goal": goal}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2, ensure_ascii=False)

    print(f"已生成固定位置文件: {out_path}")

    if args.print_run:
        seed = data.get("seed")
        use_seed_env = ""
        if seed is not None:
            use_seed_env = f" USE_SCENARIO_SEED=1 SCENARIO_SEED={seed}"

        cmd = (
            f"USE_FIXED_POSITIONS=1 POSITIONS_FILE=\"{out_path}\"{use_seed_env} "
            f"bash /home/tang/Desktop/run_optimized.sh 5 1024 {args.exp}"
        )
        print("\n一键运行示例：")
        print(cmd)


if __name__ == "__main__":
    main()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
势场参数快速元优化脚本（最小侵入版）

设计目标：
    - 完全复用现有场景与势场实现：
        * 使用 paper3d_train_optimized.load_scenario_module 加载场景
        * 使用 multiagent.environment.MultiAgentEnv 驱动环境
        * 使用 ContinuousPotentialFieldCorrector 计算势场修正动作
    - 不修改现有训练主循环与 XLA / 多进程配置，仅新增脚本
    - 在「任务分布」（多地形 seed、多复杂度等级）上评估一组势场参数的表现
    - 评估时采用与训练接近的物理配置（gravity>0、FR≈0.5），
      但评价函数只关注与势场职责强相关的指标：
        * 到目标的最终距离
        * 成功率（距离阈值内）
        * 地形/障碍碰撞率（近似）

用法示例（在 Desktop 目录下）：

    # 使用默认搜索范围随机搜索 20 组参数，每组在 4 个 seed 上各跑 1 个 episode
    # 默认复杂度等级 2，最大 400 步，FR=0.5
    python3 pf_meta_optimize.py --num-samples 20

    # 调整搜索范围与任务配置
    python3 pf_meta_optimize.py \\
        --num-samples 30 \\
        --seeds 1001 1002 1003 1004 \\
        --terrain-levels 2 3 \\
        --episodes-per-task 2 \\
        --max-steps 500 \\
        --fr 0.5

运行结束后，会在终端打印一组推荐的环境变量配置：

    export GOAL_ATTRACTION=...
    export LAMBDA_1_BASE=...
    export TERRAIN_REPULSION=...
    export AGENT_INFLUENCE_RANGE=...
    export DELTA_K_ATT=...
    export DELTA_LAMBDA_1=...
    export DELTA_K_REP=...
    export DELTA_RADIUS=...

你可以将这些值复制到 run_optimized.sh 中对应的默认值位置，作为新的势场基准参数。
"""

import argparse
import os
import sys
import math
from types import SimpleNamespace

import numpy as np

try:
    # 复用训练脚本中的场景加载逻辑
    from paper3d_train_optimized import load_scenario_module
except ImportError as e:
    print(f"[ERROR] 无法导入 paper3d_train_optimized.load_scenario_module: {e}")
    print("请在与 paper3d_train_optimized.py 相同的目录下运行本脚本。")
    sys.exit(1)

try:
    from multiagent.environment import MultiAgentEnv
except ImportError as e:
    print(f"[ERROR] 无法导入 multiagent.environment.MultiAgentEnv: {e}")
    print("请确认 multiagent 包在 PYTHONPATH 中。")
    sys.exit(1)

try:
    from potential_field_corrector import ContinuousPotentialFieldCorrector
except ImportError as e:
    print(f"[ERROR] 无法导入 ContinuousPotentialFieldCorrector: {e}")
    sys.exit(1)


def _build_env(seed: int,
               terrain_level: int,
               scenario_name: str = "paper3d_terrain_vectorized",
               quiet: bool = True):
    """
    构建单环境实例（不使用并行），返回 (env, scenario, world)。

    注意：
        - 仅用于元优化评估，不修改全局训练配置。
        - gravity / 物理参数在此阶段仍由场景与 core.py 默认值控制。
    """
    # 构造与训练脚本兼容的 args namespace
    args = SimpleNamespace()
    # 地形与位置配置：随机地形 + 动态位置，模拟真实训练时的随机性
    args.terrain_seed = seed
    args.use_fixed_positions = False
    args.dynamic_first_time = True
    args.random_terrain = True
    args.random_z0_positions = True
    args.positions_file = None
    args.terrain_complexity_level = int(terrain_level)

    if quiet:
        os.environ.setdefault("QUIET_OUTPUT", "1")

    scenario = load_scenario_module(scenario_name, args)
    if scenario is None:
        raise RuntimeError(f"无法加载场景 {scenario_name}")

    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        shared_viewer=False,
        done_callback=getattr(scenario, "is_done", None),
    )

    # 将 scenario 引用挂到 world（与训练路径保持一致）
    try:
        world.scenario = scenario
    except Exception:
        pass

    return env, scenario, world


def _setup_pf_correctors(world,
                         scenario,
                         goal_attraction: float,
                         lambda_1_base: float,
                         terrain_repulsion: float,
                         agent_influence_range: float,
                         max_force_magnitude: float,
                         detection_radius: float,
                         gravity: float,
                         force_ratio: float):
    """
    为每个智能体创建一个 ContinuousPotentialFieldCorrector，配置与当前 world / scenario 对齐。
    返回 correctors 列表，与 world.agents 一一对应。
    """
    # 尝试从 world / scenario 获取地形高度与网格
    terrain = getattr(world, "terrain", None)
    X = getattr(scenario, "X", None)
    Y = getattr(scenario, "Y", None)
    # 障碍物（若存在）
    obstacles = getattr(world, "obstacles", getattr(scenario, "obstacles", []))

    correctors = []
    for _ in world.agents:
        corr = ContinuousPotentialFieldCorrector(
            terrain_data=terrain,
            X=X,
            Y=Y,
            goal_attraction=goal_attraction,
            lambda_1_base=lambda_1_base,
            terrain_repulsion=terrain_repulsion,
            influence_range=agent_influence_range,
            sphere_detection_radius=detection_radius,
            max_force_magnitude=max_force_magnitude,
            gravity=gravity,
            debug_mode=False,
        )
        try:
            corr.obstacles = obstacles
        except Exception:
            pass
        correctors.append(corr)

    # 把 force_ratio 挂到 corrector 上，便于后续使用（虽然函数参数里也会传）
    for corr in correctors:
        setattr(corr, "_force_ratio", float(force_ratio))

    return correctors


def _compute_terrain_height(world, scenario, x: float, y: float) -> float:
    """
    尽量复用现有逻辑获取地形高度，用于碰撞/穿透近似判定。
    """
    # 优先用 scenario 的 get_terrain_height
    if hasattr(scenario, "get_terrain_height"):
        try:
            return float(scenario.get_terrain_height(x, y))
        except Exception:
            pass

    # 再尝试 world._get_terrain_height（core.World 中的封装）
    if hasattr(world, "_get_terrain_height"):
        try:
            return float(world._get_terrain_height(x, y))
        except Exception:
            pass

    # 最后直接访问 world.terrain
    terrain = getattr(world, "terrain", None)
    if terrain is not None:
        try:
            h, w = terrain.shape
            xi = max(0, min(int(x), w - 1))
            yi = max(0, min(int(y), h - 1))
            return float(terrain[yi, xi])
        except Exception:
            pass

    return 0.0


def evaluate_pf_params(params,
                       tasks,
                       max_steps: int = 400,
                       episodes_per_task: int = 1,
                       fr: float = 0.5,
                       gravity: float = 7.9,
                       scenario_name: str = "paper3d_terrain_vectorized",
                       quiet: bool = True):
    """
    在给定的任务分布上评估一组势场参数。

    params: dict，包含以下键（均为 float）：
        - goal_attraction
        - lambda_1_base
        - terrain_repulsion
        - agent_influence_range
        - delta_k_att
        - delta_lambda_1
        - delta_k_rep
        - delta_radius
      注：delta_* 当前仅用于约束范围验证，未进入评估动力学，
          真实训练时会由TF势场内部使用。

    tasks: List[(seed, terrain_level)]

    返回:
        fitness: float
        metrics: dict，包含汇总的成功率/碰撞率/平均最终距离等
    """
    goal_attraction = float(params["goal_attraction"])
    lambda_1_base = float(params["lambda_1_base"])
    terrain_repulsion = float(params["terrain_repulsion"])
    agent_influence_range = float(params["agent_influence_range"])

    # 为势场检测半径提供一个简单的从 agent_influence_range 派生的值
    detection_radius = max(5.0, min(50.0, agent_influence_range))
    max_force_magnitude = float(params.get("max_force_magnitude", 8.0))

    # 汇总指标
    total_episodes = 0
    succ_episodes = 0
    coll_episodes = 0
    sum_final_dist = 0.0

    # success 距离阈值：优先从环境变量/场景读取，其次用 run_optimized.sh 默认（8.0）
    try:
        succ_dist_env = float(os.getenv("SUCCESS_DISTANCE_THRESHOLD", "8.0"))
    except Exception:
        succ_dist_env = 8.0

    for (seed, level) in tasks:
        env, scenario, world = _build_env(seed, level, scenario_name=scenario_name, quiet=quiet)
        n_agents = len(world.agents)

        # 若场景定义了自己的 success_distance_threshold，优先使用
        succ_dist = succ_dist_env
        if hasattr(scenario, "success_distance_threshold"):
            try:
                succ_dist = float(scenario.success_distance_threshold)
            except Exception:
                pass

        correctors = _setup_pf_correctors(
            world,
            scenario,
            goal_attraction=goal_attraction,
            lambda_1_base=lambda_1_base,
            terrain_repulsion=terrain_repulsion,
            agent_influence_range=agent_influence_range,
            max_force_magnitude=max_force_magnitude,
            detection_radius=detection_radius,
            gravity=gravity,
            force_ratio=fr,
        )

        for _ in range(episodes_per_task):
            total_episodes += 1
            obs_n = env.reset()
            # 轨迹内指标
            episode_done = False
            episode_success = False
            episode_collision = False
            final_dists = np.zeros(n_agents, dtype=np.float32)

            for _step in range(max_steps):
                # 从 world 读取当前状态和目标
                actions = []
                for i, agent in enumerate(world.agents):
                    pos = np.asarray(agent.state.p_pos, dtype=np.float32)
                    # 目标：优先每智能体自己的 goal_a，其次场景全局 goal_pos
                    goal = None
                    try:
                        if hasattr(agent, "goal_a") and getattr(agent.goal_a.state, "p_pos", None) is not None:
                            goal = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
                        elif getattr(scenario, "goal_pos", None) is not None:
                            goal = np.asarray(scenario.goal_pos, dtype=np.float32)
                    except Exception:
                        pass

                    # 若无有效目标，则给零动作
                    base_action = np.zeros(3, dtype=np.float32)
                    if goal is not None:
                        corr = correctors[i]
                        a_corr = corr.correct_action_continuous(
                            base_action,
                            pos,
                            goal_pos=goal,
                            other_agents=None,
                            force_ratio=fr,
                        )
                        head = np.asarray(a_corr, dtype=np.float32)
                    else:
                        head = base_action

                    full_act = np.zeros(7, dtype=np.float32)
                    full_act[:3] = head
                    actions.append(full_act)

                next_obs_n, rew_n, done_n, info_n = env.step(actions)

                # 更新最终距离
                for i, agent in enumerate(world.agents):
                    pos = np.asarray(agent.state.p_pos, dtype=np.float32)
                    goal = None
                    try:
                        if hasattr(agent, "goal_a") and getattr(agent.goal_a.state, "p_pos", None) is not None:
                            goal = np.asarray(agent.goal_a.state.p_pos, dtype=np.float32)
                        elif getattr(scenario, "goal_pos", None) is not None:
                            goal = np.asarray(scenario.goal_pos, dtype=np.float32)
                    except Exception:
                        pass
                    if goal is not None:
                        final_dists[i] = float(np.linalg.norm(pos - goal))

                # success 判定：任一智能体距离小于阈值
                if np.any(final_dists <= succ_dist):
                    episode_success = True

                # 粗略碰撞/穿透判定：z 低于地形高度 + eps 且未达成功距离
                try:
                    for i, agent in enumerate(world.agents):
                        pos = np.asarray(agent.state.p_pos, dtype=np.float32)
                        h_terrain = _compute_terrain_height(world, scenario, float(pos[0]), float(pos[1]))
                        if pos[2] <= h_terrain + 0.05 and final_dists[i] > succ_dist:
                            episode_collision = True
                            break
                except Exception:
                    pass

                # env.step 已经根据 scenario.is_done 做了终止判断
                if isinstance(done_n, (list, tuple, np.ndarray)) and any(done_n):
                    episode_done = True
                    break

                obs_n = next_obs_n

            sum_final_dist += float(np.mean(final_dists))
            if episode_success:
                succ_episodes += 1
            if episode_collision:
                coll_episodes += 1

        try:
            env.close()
        except Exception:
            pass

    if total_episodes == 0:
        return -1e9, {
            "success_rate": 0.0,
            "collision_rate": 1.0,
            "mean_final_distance": 1e9,
        }

    success_rate = succ_episodes / float(total_episodes)
    collision_rate = coll_episodes / float(total_episodes)
    mean_final_distance = sum_final_dist / float(total_episodes)

    # 归一化距离以避免尺度过大（使用大致 map_size=200）
    dist_norm = mean_final_distance / 200.0

    # 适应度函数：成功越多越好，距离越小越好，碰撞越少越好
    fitness = (
        +1.0 * success_rate
        -1.0 * dist_norm
        -0.5 * collision_rate
    )

    metrics = {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "mean_final_distance": mean_final_distance,
    }
    return fitness, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="在随机地形任务分布上对势场参数进行快速元优化（随机搜索版）"
    )
    parser.add_argument("--num-samples", type=int, default=20,
                        help="随机搜索的样本数量（每个样本是一组势场参数）")
    parser.add_argument("--episodes-per-task", type=int, default=1,
                        help="每个 (seed, terrain_level) 上评估的 episode 数量")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="每个 episode 的最大步数")
    parser.add_argument("--fr", type=float, default=0.5,
                        help="势场混合比例 FR（0=仅网络, 1=仅势场，建议与训练默认保持一致，如0.5）")
    parser.add_argument("--gravity", type=float, default=7.9,
                        help="评估时使用的重力加速度（用于 ContinuousPotentialFieldCorrector，仅起到幅度参考作用）")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1001, 1002, 1003, 1004],
                        help="用于评估的 terrain_seed 列表")
    parser.add_argument("--terrain-levels", type=int, nargs="*", default=[2],
                        help="用于评估的地形复杂度等级列表（1-4）")
    parser.add_argument("--scenario", type=str, default="paper3d_terrain_vectorized",
                        choices=["paper3d_terrain_vectorized", "paper3d_terrain_weighted"],
                        help="用于评估的场景名称")

    # 势场参数搜索范围（与 run_optimized.sh 默认相近）
    parser.add_argument("--goal-attraction-range", type=float, nargs=2, default=[0.5, 3.0],
                        help="GOAL_ATTRACTION 搜索范围 [min, max]")
    parser.add_argument("--lambda-1-base-range", type=float, nargs=2, default=[3.0, 10.0],
                        help="LAMBDA_1_BASE 搜索范围 [min, max]")
    parser.add_argument("--terrain-repulsion-range", type=float, nargs=2, default=[40.0, 150.0],
                        help="TERRAIN_REPULSION 搜索范围 [min, max]")
    parser.add_argument("--agent-influence-range-range", type=float, nargs=2, default=[5.0, 40.0],
                        help="AGENT_INFLUENCE_RANGE 搜索范围 [min, max]")
    parser.add_argument("--max-force-magnitude-range", type=float, nargs=2, default=[6.0, 12.0],
                        help="MAX_FORCE_MAGNITUDE 搜索范围 [min, max]")

    # delta 参数搜索范围（主要影响 TF 势场内部参数调节幅度）
    parser.add_argument("--delta-k-att-range", type=float, nargs=2, default=[0.3, 1.0],
                        help="DELTA_K_ATT 搜索范围 [min, max]")
    parser.add_argument("--delta-lambda-1-range", type=float, nargs=2, default=[2.0, 5.0],
                        help="DELTA_LAMBDA_1 搜索范围 [min, max]")
    parser.add_argument("--delta-k-rep-range", type=float, nargs=2, default=[30.0, 60.0],
                        help="DELTA_K_REP 搜索范围 [min, max]")
    parser.add_argument("--delta-radius-range", type=float, nargs=2, default=[4.0, 8.0],
                        help="DELTA_RADIUS 搜索范围 [min, max]")

    return parser.parse_args()


def sample_params(args: argparse.Namespace, rng: np.random.Generator):
    """从给定范围中随机采样一组势场参数。"""
    def _u(a, b):
        return float(rng.uniform(a, b))

    goal_attraction = _u(*args.goal_attraction_range)
    lambda_1_base = _u(*args.lambda_1_base_range)
    terrain_repulsion = _u(*args.terrain_repulsion_range)
    agent_influence_range = _u(*args.agent_influence_range_range)
    max_force_magnitude = _u(*args.max_force_magnitude_range)

    delta_k_att = _u(*args.delta_k_att_range)
    delta_lambda_1 = _u(*args.delta_lambda_1_range)
    delta_k_rep = _u(*args.delta_k_rep_range)
    delta_radius = _u(*args.delta_radius_range)

    return {
        "goal_attraction": goal_attraction,
        "lambda_1_base": lambda_1_base,
        "terrain_repulsion": terrain_repulsion,
        "agent_influence_range": agent_influence_range,
        "max_force_magnitude": max_force_magnitude,
        "delta_k_att": delta_k_att,
        "delta_lambda_1": delta_lambda_1,
        "delta_k_rep": delta_k_rep,
        "delta_radius": delta_radius,
    }


def main():
    args = parse_args()

    # 构造任务分布：所有 (seed, terrain_level) 组合
    tasks = []
    for s in args.seeds:
        for lvl in args.terrain_levels:
            lvl_int = int(max(1, min(4, lvl)))
            tasks.append((int(s), lvl_int))

    if not tasks:
        print("[WARN] 未提供有效的任务组合（seed, terrain_level），使用默认 (1001, 2)")
        tasks = [(1001, 2)]

    print("==========================================")
    print("势场参数元优化（随机搜索）")
    print("==========================================")
    print(f"样本数量        : {args.num_samples}")
    print(f"每任务episodes : {args.episodes_per_task}")
    print(f"每episode步数  : {args.max_steps}")
    print(f"FR (混合比例)  : {args.fr}")
    print(f"重力           : {args.gravity}")
    print(f"任务数         : {len(tasks)} (seeds={args.seeds}, levels={args.terrain_levels})")
    print(f"场景           : {args.scenario}")
    print("==========================================")

    rng = np.random.default_rng(seed=12345)

    best_params = None
    best_fitness = -math.inf
    best_metrics = None

    for i in range(args.num_samples):
        params = sample_params(args, rng)
        print(f"\n---- 样本 {i+1}/{args.num_samples} ----")
        print("候选参数:")
        for k in sorted(params.keys()):
            print(f"  {k}: {params[k]:.4f}")

        fitness, metrics = evaluate_pf_params(
            params,
            tasks,
            max_steps=args.max_steps,
            episodes_per_task=args.episodes_per_task,
            fr=args.fr,
            gravity=args.gravity,
            scenario_name=args.scenario,
            quiet=True,
        )

        print(f"评估结果: fitness={fitness:.4f} | "
              f"success_rate={metrics['success_rate']:.3f} | "
              f"collision_rate={metrics['collision_rate']:.3f} | "
              f"mean_final_distance={metrics['mean_final_distance']:.2f}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_params = params
            best_metrics = metrics
            print("  -> 当前为最优参数，已更新。")

    if best_params is None:
        print("[ERROR] 未找到有效的参数样本。")
        sys.exit(1)

    print("\n==========================================")
    print("元优化完成，最优参数如下（可作为 run_optimized.sh 默认值）：")
    print("==========================================")
    for k in sorted(best_params.keys()):
        print(f"{k}: {best_params[k]:.6f}")
    print("\n对应表现：")
    print(f"  fitness           : {best_fitness:.4f}")
    print(f"  success_rate      : {best_metrics['success_rate']:.3f}")
    print(f"  collision_rate    : {best_metrics['collision_rate']:.3f}")
    print(f"  mean_final_dist   : {best_metrics['mean_final_distance']:.3f}")

    print("\n建议在 shell 中设置的环境变量（可直接拷贝到 run_optimized.sh 对应位置）：\n")
    print(f"export GOAL_ATTRACTION={best_params['goal_attraction']:.6f}")
    print(f"export LAMBDA_1_BASE={best_params['lambda_1_base']:.6f}")
    print(f"export TERRAIN_REPULSION={best_params['terrain_repulsion']:.6f}")
    print(f"export AGENT_INFLUENCE_RANGE={best_params['agent_influence_range']:.6f}")
    print(f"export MAX_FORCE_MAGNITUDE={best_params['max_force_magnitude']:.6f}")
    print(f"export DELTA_K_ATT={best_params['delta_k_att']:.6f}")
    print(f"export DELTA_LAMBDA_1={best_params['delta_lambda_1']:.6f}")
    print(f"export DELTA_K_REP={best_params['delta_k_rep']:.6f}")
    print(f"export DELTA_RADIUS={best_params['delta_radius']:.6f}")
    print("\n==========================================")
    print("说明：")
    print("  - 这些参数仅调节势场的基准值与可调幅度，不会改变训练主循环或网络结构。")
    print("  - 评估时已在多地形 seed / 复杂度等级下计算平均表现，")
    print("    更偏向寻找对随机地形分布鲁棒的势场配置。")
    print("==========================================")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作与势场修正消融对比实验

对比四种情况：
1. 仅有动作（网络动作，无势场修正）
2. 传统APF（势场动作，参数固定）
3. 可学习APF（势场动作，参数可学习）
4. 两者融合（网络动作+可学习势场修正）

所有实验使用相同的环境（固定位置+地形），确保公平对比。

🚨 关键配置：消融实验始终禁用课程学习
- UNLOCK_ENV_ON_SUCCESS=0：禁用基于成功次数的环境解锁
- UNLOCK_ENV_ON_PLATEAU=0：禁用基于奖励停滞的环境解锁
- RANDOM_TERRAIN=0：始终使用固定地形
- PER_ENV_TERRAIN=0：每个环境使用相同地形
- PER_EPISODE_TERRAIN=0：每个回合使用相同地形
这些设置会在应用实验配置后强制设置，确保不会被实验配置覆盖。

注意：辅助自监督+PER增强实验（action_apf_fusion_aux_per）已临时注释，待功能实现后再启用。
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
from ablation_batch_manager import AblationBatchManager

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


# 实验配置
# 
# 🔧 关键说明：势场修正控制机制
# 
# TF版本的势场修正控制条件（当前代码实际使用的版本）：
#   - 条件：use_tf_potential_field AND action_force_ratio > 0.0
#   - use_tf_potential_field: 来自 args.use_tf_potential_field（默认True）
#   - action_force_ratio: 来自 args.action_force_ratio
# 
# ⚠️ 重要发现：
#   - ENABLE_ACTION_CORRECTION 参数只控制NumPy版本的势场修正器（maddpg.pf_correctors）
#   - ENABLE_ACTION_CORRECTION 对TF版本的势场修正无效！
#   - 真正控制TF版本的是：ACTION_FORCE_RATIO 和 USE_TF_POTENTIAL_FIELD
# 
# 🚨 关键问题：ACTION_FORCE_RATIO_SCHEDULE_PCT 会动态覆盖固定值！
#   - run_optimized.sh 默认设置了 ACTION_FORCE_RATIO_SCHEDULE_PCT
#   - 这个 schedule 会在每个训练回合根据进度动态调整 action_force_ratio
#   - 即使设置了 ACTION_FORCE_RATIO=0.0，如果 schedule 存在，它会被动态调整
#   - 因此，消融实验必须显式禁用 schedule：ACTION_FORCE_RATIO_SCHEDULE_PCT=""
# 
# 因此，消融实验应该：
#   1. 使用 ACTION_FORCE_RATIO 控制势场修正（0.0=禁用，1.0=完全势场，0.7=混合）
#   2. 🚨 必须设置 ACTION_FORCE_RATIO_SCHEDULE_PCT="" 禁用动态调整
#   3. 可选：使用 USE_TF_POTENTIAL_FIELD 作为双重保险（0=禁用TF版本，1=启用）
#   4. ENABLE_ACTION_CORRECTION 参数可以移除（对TF版本无效）
#
# 说明：
# 1. action_only: 仅使用网络动作，无势场修正
# 2. apf_traditional: 传统APF，势场参数固定（使用base值，不通过网络学习）
# 3. apf_learnable: 基于网络优化的APF，势场参数通过Actor网络输出并学习
# 4. action_apf_fusion: 两者融合，网络动作+可学习势场修正（当前默认）
EXPERIMENT_CONFIGS = [
    {
        "label": "action_only",
        "name": "Action Only",
        "name_en": "Action Only",
        "description": "Network action only, no APF correction",
        "env": {
            # 🔧 关键：ACTION_FORCE_RATIO=0.0 是真正有效的控制参数
            # ENABLE_ACTION_CORRECTION 对TF版本无效，但保留以兼容旧代码
            "ACTION_FORCE_RATIO": "0.0",  # ✅ 真正有效的控制：禁用TF势场修正
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",  # 🚨 关键修复：显式禁用schedule，确保FR始终为0.0
            "USE_TF_POTENTIAL_FIELD": "1",  # 保持TF版本启用，但force_ratio=0会阻止修正
            "SEED": "252488",  # ✅ 修复：为所有实验设置相同的训练随机种子，确保公平对比（消除随机性影响）
            "TF_DETERMINISTIC_OPS": "0"  # ✅ 修复：启用TensorFlow确定性操作，确保完全可重复
        }
    },
    {
        "label": "apf_traditional",
        "name": "APF Traditional",
        "name_en": "APF Traditional",
        "description": "APF action only, fixed parameters (base values, no learning)",
        "env": {
            "ACTION_FORCE_RATIO": "1.0",  # ✅ 真正有效的控制：100%势场动作（前3维网络输出不影响最终动作）
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",  # 🚨 关键修复：使用特殊值标记禁用，确保FR始终为1.0
            "USE_TF_POTENTIAL_FIELD": "1", # 确保使用TF版本
            # 🔧 关键：通过设置delta为0，使势场参数固定为base值
            # 这样 Actor 网络输出的后4维参数不会影响势场参数
            # 注意：保留探索机制（随机动作、OU噪声）以保持与其他实验的公平对比
            "DELTA_K_ATT": "0.0",
            "DELTA_LAMBDA_1": "0.0",
            "DELTA_K_REP": "0.0",
            "DELTA_RADIUS": "0.0",
            # 🚨 关键修复：显式设置base值，确保Traditional APF使用正确的参数（而不是随机初始化的网络参数）
            # 这些base值必须与run_optimized.sh的默认值完全一致
            # run_optimized.sh默认base值（第685-708行）：
            "GOAL_ATTRACTION": "6.0",                    # ✅ 与run_optimized.sh一致（默认6.0）
            "LAMBDA_1_BASE": "8.5",                      # ✅ 与run_optimized.sh一致（默认8.5）
            "TERRAIN_REPULSION": "8000.0",               # ✅ 与run_optimized.sh一致（默认8000.0）
            "AGENT_INFLUENCE_RANGE": "150.0",           # ✅ 与run_optimized.sh一致（默认150.0）
            # 实际使用的势场参数（因为DELTA=0.0，所以参数=base值）：
            #   k_att = 6.0（固定）
            #   lambda_1 = 8.5（固定）
            #   k_rep = 8000.0（固定）
            #   radius = 150.0（固定）
            # 🚨 关键修复：网络初始化一致性
            # 所有算法（apf_traditional、apf_learnable、action_apf_fusion）都使用相同的SEED
            # 训练脚本会在创建网络之前设置随机种子（paper3d_train_optimized.py 第12069-12072行）
            # 这确保了所有算法的网络初始化完全一致，即使Traditional APF不使用网络输出的后4维参数
            # 网络初始化包括：Actor和Critic网络的权重初始化（使用相同的随机种子）
            # 这样对比才公平：Traditional APF使用的是"固定参数的APF"，而不是"随机初始化的网络+固定参数"
            "SEED": "252488",  # ✅ 修复：为所有实验设置相同的训练随机种子，确保网络初始化一致（与apf_learnable、action_apf_fusion相同）
            "TF_DETERMINISTIC_OPS": "0"  # ✅ 修复：启用TensorFlow确定性操作，确保完全可重复
        }
    },
    {
        "label": "apf_learnable",
        "name": "APF Learnable",
        "name_en": "APF Learnable",
        "description": "APF action only, parameters learned via Actor network",
        "env": {
            "ACTION_FORCE_RATIO": "1.0",  # ✅ 真正有效的控制：100%势场动作
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",  # 🚨 关键修复：使用特殊值标记禁用，确保FR始终为1.0
            "USE_TF_POTENTIAL_FIELD": "1",  # 确保使用TF版本
            # 🔧 性能优化：与其他算法保持一致，确保公平对比
            # 注意：移除NOISE_SCALE设置，使用默认值0.35，与其他算法保持一致
            # "NOISE_SCALE": "0.35",  # ✅ 修复：与其他算法保持一致（默认0.35），确保公平对比
            # 🔧 显式禁用随机动作，确保纯APF训练不受随机动作干扰
            "RANDOM_ACTION_PROB_TRAINING": "0.0",  # 禁用训练阶段的随机动作，确保动作完全由APF控制
            # 🚨 关键修复：显式设置delta值为run_optimized.sh的默认值，确保可学习APF真正学习
            # 注意：这些值必须与run_optimized.sh中的默认值完全一致
            # run_optimized.sh默认值（第720-736行）：DELTA_K_ATT=5.0, DELTA_LAMBDA_1=2.2, DELTA_K_REP=1200.0, DELTA_RADIUS=80.0
            # run_optimized.sh base值（第685-708行）：GOAL_ATTRACTION=6.0, LAMBDA_1_BASE=8.5, TERRAIN_REPULSION=8000.0, AGENT_INFLUENCE_RANGE=150.0
            # 实际参数范围（基于base值）：
            #   k_att = 6.0 ± 5.0 = [1.0, 11.0]
            #   lambda_1 = 8.5 ± 2.2 = [6.3, 10.7]
            #   k_rep = 8000.0 ± 1200.0 = [6800.0, 9200.0]
            #   radius = 150.0 ± 80.0 = [70.0, 230.0]
            "DELTA_K_ATT": "5.0",      # ✅ 与run_optimized.sh一致（默认5.0）
            "DELTA_LAMBDA_1": "2.2",   # ✅ 与run_optimized.sh一致（默认2.2）
            "DELTA_K_REP": "1200.0",   # 🚨 修复：与run_optimized.sh一致（默认1200.0，不是1000.0）
            "DELTA_RADIUS": "80.0",    # ✅ 与run_optimized.sh一致（默认80.0）
            # 🚨 关键修复：显式设置base值，确保与apf_traditional和action_apf_fusion使用相同的base值
            # 原因：虽然run_optimized.sh有默认值，但显式设置可以确保所有实验组使用完全相同的base值
            # 这样对比才公平：apf_learnable和action_apf_fusion使用相同的base值，只是DELTA不同
            "GOAL_ATTRACTION": "6.0",                    # ✅ 与run_optimized.sh一致（默认6.0）
            "LAMBDA_1_BASE": "8.5",                      # ✅ 与run_optimized.sh一致（默认8.5）
            "TERRAIN_REPULSION": "8000.0",               # ✅ 与run_optimized.sh一致（默认8000.0）
            "AGENT_INFLUENCE_RANGE": "150.0",           # ✅ 与run_optimized.sh一致（默认150.0）
            # 🚨 关键修复：网络初始化一致性
            # 所有算法（apf_traditional、apf_learnable、action_apf_fusion）都使用相同的SEED
            # 训练脚本会在创建网络之前设置随机种子（paper3d_train_optimized.py 第12069-12072行）
            # 这确保了所有算法的网络初始化完全一致，包括Actor和Critic网络的权重初始化
            "SEED": "252488",  # ✅ 修复：为所有实验设置相同的训练随机种子，确保网络初始化一致（与apf_traditional、action_apf_fusion相同）
            "TF_DETERMINISTIC_OPS": "0"  # ✅ 修复：启用TensorFlow确定性操作，确保完全可重复
        }
    },
    {
        "label": "action_apf_fusion",
        "name": "Action+APF Fusion",
        "name_en": "Action+APF Fusion",
        "description": "Network action + learnable APF correction (default config)",
        "env": {
            # 🔧 关键修复：action_apf_fusion 使用完整的 run_optimized.sh 默认配置
            # 包括动态 FR schedule，让智能体逐渐学会减少对势场的依赖
            # 注意：不设置 ACTION_FORCE_RATIO，让它使用 run_optimized.sh 的默认值 0.50
            # 🚨 关键修复：使用与 run_optimized.sh 完全一致的 FR schedule，确保融合APF完全执行 run_optimized.sh 的要求
            # run_optimized.sh 默认 schedule（第684行）："0%:0.55,10%:0.5,20%:0.5,35%:0.5,50%:0.5,70%:0.5,85%:0.5,100%:0.5"
            # 原因：action_apf_fusion 应该完全使用 run_optimized.sh 的默认配置，包括 FR schedule
            # 这样消融实验才能真正反映"融合APF"在标准训练配置下的表现
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "0%:0.50,10%:0.40,20%:0.30,40%:0.20,60%:0.15,100%:0.10",  # ✅ 修复：与 run_optimized.sh 默认值完全一致
            "USE_TF_POTENTIAL_FIELD": "1",  # 确保使用TF版本
            # 🚨 关键修复：网络初始化一致性
            # 所有算法（apf_traditional、apf_learnable、action_apf_fusion）都使用相同的SEED
            # 训练脚本会在创建网络之前设置随机种子（paper3d_train_optimized.py 第12069-12072行）
            # 这确保了所有算法的网络初始化完全一致，包括Actor和Critic网络的权重初始化
            "SEED": "252488",  # ✅ 修复：为所有实验设置相同的训练随机种子，确保网络初始化一致（与apf_traditional、apf_learnable相同）
            # 🚨 关键修复：启用 TensorFlow 确定性操作，确保完全可重复
            # 原因：即使设置了随机种子，TensorFlow 的某些操作（如卷积、矩阵乘法）可能仍然有非确定性
            # 启用 TF_DETERMINISTIC_OPS 可以确保所有操作都是确定性的
            "TF_DETERMINISTIC_OPS": "0"  # 启用 TensorFlow 确定性操作，确保完全可重复
            # 其他所有配置（学习率、网络架构、奖励权重、噪声参数等）都使用 run_optimized.sh 的默认值
        }
    },
    # 🚨 临时注释：辅助自监督+PER增强实验（功能尚未实现，先注释掉避免运行失败）
    # {
    #     "label": "action_apf_fusion_aux_per",
    #     "name": "融合APF+辅助自监督+PER增强",
    #     "name_en": "Action+APF Fusion + Auxiliary + PER",
    #     "description": "网络动作+可学习势场修正+辅助自监督任务+增强PER（完整增强版本）",
    #     "env": {
    #         # 🔧 基于 action_apf_fusion，添加辅助自监督和PER增强功能
    #         "USE_TF_POTENTIAL_FIELD": "1",  # 确保使用TF版本
    #         # 辅助自监督任务开关和参数
    #         "AUX_ENABLED": "1",  # 启用辅助自监督任务
    #         "AUX_HORIZON_STEPS": "8",  # 辅助任务的前瞻步数（H=8步）
    #         "AUX_CLEARANCE_WEIGHT": "0.2",  # 净空预测损失权重（λ_clear）
    #         "AUX_PROGRESS_WEIGHT": "0.1",  # 进展预测损失权重（λ_prog）
    #         "AUX_COLLISION_WEIGHT": "0.15",  # 碰撞预测损失权重（λ_coll）
    #         # PER增强参数
    #         "PER_AUX_ENABLED": "1",  # 启用PER中的辅助误差项
    #         "PER_AUX_CLEAR_WEIGHT": "0.3",  # PER中净空误差权重（w_clear）
    #         "PER_AUX_PROG_WEIGHT": "0.2",  # PER中进展误差权重（w_prog）
    #         "PER_AUX_COLL_WEIGHT": "0.25",  # PER中碰撞误差权重（w_coll）
    #         "PER_SAFE_BOOST": "0.3",  # 安全样本加成系数（gamma_safe）
    #         "PER_AUX_WEIGHT": "0.5",  # PER中辅助误差总权重（w_aux）
    #         "PER_TD_WEIGHT": "1.0",  # PER中TD误差权重（w_td，保持原值）
    #         "PER_REWARD_WEIGHT": "0.12"  # PER中奖励幅值权重（w_r，保持原值）
    #         # 注意：ACTION_FORCE_RATIO 和 ACTION_FORCE_RATIO_SCHEDULE_PCT 不设置，
    #         # 使用 run_optimized.sh 的默认值（与 action_apf_fusion 一致）
    #     }
    # }
]


def parse_args():
    parser = argparse.ArgumentParser(description="动作与势场修正消融对比实验、地形复杂度3、无随机动作判断穿透原因是否是随机动作导致的")
    parser.add_argument("--script", type=str, default="./run_optimized.sh",
                        help="训练启动脚本路径 (默认 ./run_optimized.sh)")
    parser.add_argument("--episodes", type=int, default=400,
                        help="每个实验的训练回合数（默认5）")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="训练批次大小")
    parser.add_argument("--use-weighted-reward", type=int, default=1, choices=[0, 1],
                        help="是否使用分项加权奖励")
    parser.add_argument("--algorithm", type=str, default="matd3", choices=["maddpg", "matd3"],
                        help="训练算法选择")
    parser.add_argument("--output-dir", type=str, default="ablation_action_pf_outputs",
                        help="图表输出目录")
    parser.add_argument("--logs-root", type=str, default="logs",
                        help="训练日志根目录")
    parser.add_argument("--positions-file", type=str, default=None,
                        help="固定位置文件路径（None则使用默认值 ./saved_positions/5.json，与 run_optimized.sh 一致）")
    parser.add_argument("--reuse", action="store_true",
                        help="若检测到同名实验已存在，则跳过重新训练，直接复用最新日志")
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="拟合曲线平滑窗口大小（用于减少振幅）")
    parser.add_argument("--fit-method", type=str, default="moving_average",
                        choices=["moving_average", "spline", "poly"],
                        help="拟合方法：moving_average(移动平均), spline(样条插值), poly(多项式拟合)")
    parser.add_argument("--generate-interactive", action="store_true",
                        help="生成交互式轨迹图（需要plotly）")
    # 🚨 关键修复：默认禁用并行模式，改为串行运行，确保数据记录正确
    parser.add_argument("--parallel", action="store_true",
                        help="并行训练（默认禁用，串行运行以确保数据记录正确）")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="指定GPU ID列表（逗号分隔，如'0,1,2'），None则自动分配或共享GPU")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        choices=["action_only", "apf_traditional", "apf_learnable", "action_apf_fusion"],
                        help="选择要运行的实验（默认运行所有实验）。可选: action_only, apf_traditional, apf_learnable, action_apf_fusion")
    parser.add_argument("--quick-comparison", action="store_true",
                        help="快速对比模式：运行 apf_traditional、apf_learnable 和 action_apf_fusion 三个实验")
    return parser.parse_args()


def ensure_fixed_positions(positions_file: Path, args, exp_name_prefix: str) -> Path:
    """
    确保固定位置文件存在，不存在则生成（轻量级方法，不运行完整训练）
    
    🔧 优化：直接初始化场景获取位置信息，避免运行完整训练流程产生不必要的日志和模型目录
    """
    if positions_file is None:
        # 🚨 关键修复：使用与 run_optimized.sh 相同的默认位置文件路径
        # run_optimized.sh 默认: ./saved_positions/5.json
        positions_file = Path("./saved_positions") / "5.json"
    
    positions_file = Path(positions_file).resolve()
    positions_file.parent.mkdir(parents=True, exist_ok=True)
    
    if positions_file.exists():
        print(f"[消融实验] 使用已存在的固定位置文件: {positions_file}")
        return positions_file
    
    print(f"[消融实验] 固定位置文件不存在，开始生成: {positions_file}")
    print(f"[消融实验] 使用轻量级方法：直接初始化场景获取位置信息（不运行训练）")
    
    # 🔧 优化：直接初始化场景，获取位置信息并保存，避免运行完整训练
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        # 设置环境变量（用于场景初始化）
        import os
        original_env = {}
        # 🔧 关键修复：使用与 run_optimized.sh 完全一致的默认配置
        # 参考 run_optimized.sh 的默认值：
        #   - TERRAIN_COMPLEXITY_LEVEL=3 (默认等级3)
        #   - MAP_SIZE=200
        #   - MOUNTAIN_MIN_DISTANCE=55
        #   - SCENARIO_SEED=88 (但位置生成使用67，与消融实验保持一致)
        env_vars = {
            "USE_SCENARIO_SEED": "1",
            "SCENARIO_SEED": "67",  # 位置生成使用67，与消融实验保持一致
            "TERRAIN_COMPLEXITY_LEVEL": "3",  # 🔧 修复：与 run_optimized.sh 默认值一致（3）
            "MAP_SIZE": "200",  # 与 run_optimized.sh 默认值一致
            "MOUNTAIN_MIN_DISTANCE": "55",  # 与 run_optimized.sh 默认值一致
            "RANDOM_TERRAIN": "0",
            "PER_ENV_TERRAIN": "0",
            "PER_EPISODE_TERRAIN": "0",
        }
        
        # 临时设置环境变量
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        # 导入场景
        from multiagent.environment import MultiAgentEnv
        from multiagent.scenarios import load
        
        # 🔧 修复：根据奖励模式和向量化场景设置选择场景（与run_optimized.sh逻辑一致）
        # run_optimized.sh 逻辑：
        #   - USE_WEIGHTED_REWARD=1 且 VECTORIZED_SCENARIO=1 → paper3d_terrain_vectorized
        #   - USE_WEIGHTED_REWARD=1 且 VECTORIZED_SCENARIO=0（或未设置）→ paper3d_terrain_weighted
        #   - USE_WEIGHTED_REWARD=0 → paper3d_terrain_energy
        if args.use_weighted_reward:
            # 检查是否启用向量化场景（默认不启用，使用paper3d_terrain_weighted）
            vectorized_scenario = os.environ.get("VECTORIZED_SCENARIO", "0").lower() in ("1", "true", "yes", "on")
            if vectorized_scenario:
                scenario_name = "paper3d_terrain_vectorized"
            else:
                scenario_name = "paper3d_terrain_weighted"  # ✅ 修复：使用weighted而不是vectorized（默认）
        else:
            scenario_name = "paper3d_terrain_energy"
        
        # 加载场景
        scenario = load(scenario_name).Scenario(
            seed=67,
            use_fixed_positions=False,
            dynamic_first_time=True,
            fixed_positions_file=str(positions_file),
            random_terrain=False,
            terrain_complexity_level=3,  # 🔧 修复：与 run_optimized.sh 默认值一致（3）
        )
        
        # 创建环境并初始化
        world = scenario.make_world()
        scenario.reset_world(world)
        
        # 获取智能体位置和目标位置
        agents_pos = [agent.state.p_pos.tolist() for agent in world.agents]
        goal_pos = scenario.goal_pos.tolist() if scenario.goal_pos is not None else None
        
        if goal_pos is None:
            # 如果没有目标位置，使用默认值
            goal_pos = [float(scenario.map_size / 2), float(scenario.map_size / 2), 25.0]
        
        # 保存位置文件
        positions_data = {
            "agents": agents_pos,
            "goal": goal_pos
        }
        
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
        # 🚨 关键修复：位置文件生成失败时，直接抛出错误，不运行完整训练
        # 原因：位置文件生成应该只使用轻量级方法，运行完整训练会产生不必要的日志和模型目录
        raise RuntimeError(
            f"生成固定位置文件失败: {e}\n"
            f"提示：请检查场景初始化代码和环境配置是否正确。\n"
            f"位置文件路径: {positions_file}"
        )
    
    return positions_file


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
        # 共享GPU模式：使用内存增长，允许多个进程共享同一GPU
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env["GPU_ID"] = "0"
    
    # 🔧 修复：添加GPU内存稳定性配置，防止CUDA_ERROR_ILLEGAL_ADDRESS
    # 🚨 关键：确保使用默认BFC分配器，不使用异步分配器（避免内存对齐问题）
    env["TF_GPU_ALLOCATOR"] = ""  # 强制使用默认BFC分配器，不使用cuda_malloc_async
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 内存按需增长，避免一次性占用全部显存
    # 🚨 关键：添加GPU内存碎片整理配置，防止长时间运行后内存碎片化
    # 🚨 关键：禁用CUDA异步执行，避免并行实验时的内存访问冲突
    # 注意：虽然会降低性能，但能显著提高稳定性，特别是在并行运行时
    if gpu_id is None:
        # 共享GPU模式：强制同步执行，避免多个进程同时访问GPU导致冲突
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["TF_SYNC_ON_FINISH"] = "1"
    else:
        # 独立GPU模式：可以使用异步执行
        if "CUDA_LAUNCH_BLOCKING" in env:
            del env["CUDA_LAUNCH_BLOCKING"]
        if "TF_SYNC_ON_FINISH" in env:
            del env["TF_SYNC_ON_FINISH"]
    
    # 🚨 关键修复：完全复制 run_optimized.sh 的 XLA 配置
    env["XLA_FLAGS"] = ""  # 必须设置为空字符串而非 unset，避免继承父shell的值
    env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # 禁用 Auto JIT
    
    # 🔧 GPU执行模式已在上面根据gpu_id设置：
    # - 共享GPU模式（gpu_id=None）：同步执行，提高稳定性
    # - 独立GPU模式（gpu_id!=None）：异步执行，提高性能
    
    # 🚨 关键修复：添加所有 run_optimized.sh 的关键默认配置
    # 确保消融实验完全使用 run_optimized.sh 的默认值
    env.setdefault("NUM_ENVS", "1")  # 🔧 关键：单环境训练，与 run_optimized.sh 一致（这是导致不一致的主要原因！）
    env.setdefault("XLA_GLOBAL", "1")  # 默认启用 XLA Global
    env.setdefault("CPU_THREADS", "12")  # 默认12线程
    env.setdefault("TQDM_DISABLE", "1")  # 完全禁用tqdm进度条
    env.setdefault("QUIET_OUTPUT", "0")  # 🔧 关键：启用详细输出，确保数据记录正确（0=详细，1=安静）
    env.setdefault("SUPPRESS_MA_PROMPT", "1")  # 抑制多智能体环境交互式提示
    env.setdefault("SUPPRESS_TERRAIN_OUTPUT", "1")  # 抑制地形生成冗余输出
    
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
    # 🔧 关键修复：使用与 run_optimized.sh 一致的默认值
    # run_optimized.sh 默认: SCENARIO_SEED=${SCENARIO_SEED:-88}
    # 但位置文件生成使用67，为了保持一致，这里也使用67
    env["SCENARIO_SEED"] = "67"  # 🔧 修复：与位置文件生成保持一致，确保地形一致
    # 🚨 关键修复：不在这里设置SEED，让实验配置中的SEED生效
    # 这样既能保证地形一致（SCENARIO_SEED相同），又能保证训练过程一致（SEED相同，确保公平对比）
    # 如果实验配置中没有设置SEED，训练脚本会使用默认值1337
    # 注意：删除可能从父进程继承的SEED，确保实验配置中的SEED生效
    if "SEED" in env:
        del env["SEED"]  # 删除基础配置中的SEED，让实验配置中的SEED生效
    # 🔧 关键修复：确保地图生成参数与 run_optimized.sh 完全一致（显式设置，覆盖父进程环境变量）
    env["TERRAIN_COMPLEXITY_LEVEL"] = "3"  # ✅ 修复：与run_optimized.sh保持一致（默认3）
    env["MAP_SIZE"] = "200"  # 地图大小
    env["MOUNTAIN_MIN_DISTANCE"] = "55"  # 山峰之间的最小距离
    
    # 🔧 关键修复：确保碰撞检测阈值与run_optimized.sh一致
    # ✅ 修复：使用与run_optimized.sh相同的默认值（0.2米），确保消融实验的碰撞检测与正常训练一致
    # 原问题：之前设置为3.5米，远大于run_optimized.sh的0.2米，可能导致误报碰撞
    # 影响：过大的阈值会在智能体接近地形（但未真正碰撞）时就触发碰撞计数，导致碰撞次数虚高
    env["TERRAIN_CONTACT_EPS"] = "0.2"  # ✅ 与run_optimized.sh保持一致（默认0.2米）
    
    # 🚨 关键修复：清除所有DELTA相关环境变量，确保每个实验从干净状态开始
    # 这样可以避免父进程环境变量影响实验结果，确保传统APF和可学习APF的真正区别
    # 传统APF会在配置中显式设置DELTA_*=0.0，可学习APF会使用run_optimized.sh的默认值
    delta_vars = ["DELTA_K_ATT", "DELTA_LAMBDA_1", "DELTA_K_REP", "DELTA_RADIUS"]
    for var in delta_vars:
        if var in env:
            del env[var]
    
    return env


def find_latest_log_dir(exp_name: str, logs_root: str) -> str:
    """查找最新的日志目录（需要在worker函数之前定义）"""
    logs_path = Path(logs_root)
    if not logs_path.exists() or not logs_path.is_dir():
        raise FileNotFoundError(f"日志根目录不存在: {logs_path}")
    
    # 🔧 修复：训练脚本会创建带时间戳的目录（如 action_only_20251205_203355）
    # 🚨 关键修复：使用精确匹配模式，避免匹配到包含该名称的其他实验（如 action_apf_fusion_aux_per）
    # 匹配规则：目录名必须以 exp_name 开头，且紧跟的是下划线和时间戳（格式：exp_name_YYYYMMDD_HHMMSS）
    # 这样可以避免 action_apf_fusion_aux_per_20251211_213311 匹配到 action_apf_fusion
    matching_dirs = []
    for item in logs_path.iterdir():
        if item.is_dir():
            # 精确匹配：目录名必须以 exp_name 开头，且紧跟的是下划线和时间戳格式（8位日期_6位时间）
            # 例如：action_apf_fusion_20251211_214041 ✓
            #       action_apf_fusion_aux_per_20251211_213311 ✗（包含额外的_aux_per部分）
            if item.name == exp_name:
                # 完全匹配的情况（理论上不应该出现，但保留兼容性）
                matching_dirs.append((item.name, item))
            elif item.name.startswith(exp_name + "_"):
                # 检查是否紧跟时间戳格式（YYYYMMDD_HHMMSS，共15个字符：8位日期+下划线+6位时间）
                suffix = item.name[len(exp_name) + 1:]  # 去掉 "exp_name_"
                # 时间戳格式：8位数字_6位数字（例如：20251211_214041）
                # 使用字符串方法检查，避免导入re模块（如果文件顶部没有导入）
                if len(suffix) >= 15 and suffix[8] == '_' and suffix[:8].isdigit() and suffix[9:15].isdigit():
                    # 检查是否有子目录（训练脚本会在 exp_name_timestamp 下创建 timestamp 子目录）
                    # 🔧 关键修复：排除evaluation目录，只查找训练日志目录
                    subdirs = sorted([d for d in item.iterdir() if d.is_dir() and d.name != 'evaluation'])
                    # 进一步筛选：优先选择时间戳格式的子目录（如 20251220_170523）
                    timestamp_subdirs = [d for d in subdirs if len(d.name) >= 15 and d.name[8] == '_' and d.name[:8].isdigit() and d.name[9:15].isdigit()]
                    if timestamp_subdirs:
                        # 如果有时间戳格式的子目录，使用最新的
                        matching_dirs.append((item.name, timestamp_subdirs[-1]))
                    elif subdirs:
                        # 如果有其他子目录（非evaluation），使用最新的
                        matching_dirs.append((item.name, subdirs[-1]))
                    else:
                        # 如果没有子目录，直接使用该目录（兼容旧格式）
                        matching_dirs.append((item.name, item))
    
    if not matching_dirs:
        raise FileNotFoundError(f"未找到以 '{exp_name}' 开头的日志目录: {logs_path}")
    
    # 按目录名排序（时间戳在名称中），取最新的
    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_dir = matching_dirs[0][1]
    return str(latest_dir)


def load_metrics(log_dir: str) -> Dict:
    """加载训练指标（需要在worker函数之前定义）"""
    metrics = {}
    
    # 加载奖励数据
    ep_path = Path(log_dir) / "episode_rewards.json"
    if ep_path.exists():
        with open(ep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metrics["episode_rewards"] = data.get("episode_rewards", [])
            # 🚨 关键修复：加载所有指标，包括成功率、碰撞次数、平均净空
            metrics["success_flags"] = data.get("success_flags", [])
            metrics["collision_counts"] = data.get("collision_counts", [])
            metrics["min_distances_to_obstacle"] = data.get("min_distances_to_obstacle", [])
            # 🚨 新增：加载单智能体成功率、团队成功率等数据
            metrics["agent_success_flags"] = data.get("agent_success_flags", [])
            metrics["team_success_flags"] = data.get("team_success_flags", [])
            metrics["agent_success_rates"] = data.get("agent_success_rates", [])
            metrics["team_success_rate"] = data.get("team_success_rate", 0.0)
        else:
            metrics["episode_rewards"] = data
            metrics["success_flags"] = []
            metrics["collision_counts"] = []
            metrics["min_distances_to_obstacle"] = []
            metrics["agent_success_flags"] = []
            metrics["team_success_flags"] = []
            metrics["agent_success_rates"] = []
            metrics["team_success_rate"] = 0.0
    else:
        metrics["episode_rewards"] = []
        metrics["success_flags"] = []
        metrics["collision_counts"] = []
        metrics["min_distances_to_obstacle"] = []
        metrics["agent_success_flags"] = []
        metrics["team_success_flags"] = []
        metrics["agent_success_rates"] = []
        metrics["team_success_rate"] = 0.0
    
    # 加载损失数据
    loss_path = Path(log_dir) / "loss_history.json"
    if loss_path.exists():
        with open(loss_path, "r", encoding="utf-8") as f:
            metrics["loss_history"] = json.load(f)
    else:
        metrics["loss_history"] = []
    
    return metrics


def run_experiment_worker(args_tuple):
    """并行训练的工作函数（需要在顶层定义以便pickle）"""
    cfg, positions_file, script, episodes, batch_size, use_weighted_reward, algorithm, logs_root, gpu_id, batch_dir = args_tuple
    
    label = cfg["label"]
    # 🔧 关键修复：使用公共函数设置基础环境变量，消除代码冗余
    env = setup_base_env_vars(positions_file, gpu_id)
    
    # 🔧 新增：设置实验输出目录为批次目录下的实验子目录
    exp_dir = Path(batch_dir) / label
    exp_dir.mkdir(parents=True, exist_ok=True)
    env["EXP_NAME"] = f"../{batch_dir.name}/{label}"  # 相对路径，让训练脚本保存到批次目录
    
    if gpu_id is not None:
        print(f"[并行训练-{label}] 使用GPU {gpu_id}")
    else:
        print(f"[并行训练-{label}] 共享GPU（内存增长模式）")
    
    # === 应用实验特定配置（必须在基础配置之后，确保覆盖）===
    env.update(cfg.get("env", {}))
    
    # 🚨 关键修复：消融实验始终禁用课程学习，确保所有实验使用固定地形和位置
    # 必须在应用实验配置后强制设置，防止实验配置覆盖基础设置
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    
    # 🔧 关键修复：处理 ACTION_FORCE_RATIO_SCHEDULE_PCT，确保行为一致
    # 对于 action_apf_fusion，如果配置中未设置 ACTION_FORCE_RATIO_SCHEDULE_PCT，
    # 则完全删除该环境变量（包括从父进程继承的），让 run_optimized.sh 检测到变量未设置并使用默认 schedule
    if label == "action_apf_fusion":
        if "ACTION_FORCE_RATIO_SCHEDULE_PCT" not in cfg.get("env", {}):
            # 删除可能从父进程继承的值，确保 run_optimized.sh 检测到变量未设置并使用默认 schedule
            if "ACTION_FORCE_RATIO_SCHEDULE_PCT" in env:
                del env["ACTION_FORCE_RATIO_SCHEDULE_PCT"]
        elif cfg.get("env", {}).get("ACTION_FORCE_RATIO_SCHEDULE_PCT") == "DISABLED":
            # 如果配置中明确设置为 "DISABLED"，确保环境变量也设置为 "DISABLED"
            # run_optimized.sh 会将其转换为空字符串，禁用 schedule
            env["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = "DISABLED"
    
    # 🚨 关键修复：如果配置中设置了 ACTION_FORCE_RATIO_SCHEDULE_PCT 为 "DISABLED" 或空字符串，
    # run_optimized.sh 会检查并设置为空字符串，不使用默认值
    # 训练脚本会检查 "DISABLED" 或空字符串，跳过schedule，保持固定的 ACTION_FORCE_RATIO 值
    
    # 🔧 关键优化：检测传统APF配置，启用评估模式（跳过网络训练）
    # 传统APF特点：ACTION_FORCE_RATIO=1.0 且 DELTA_*=0.0
    # 因为网络动作被忽略，网络参数不影响势场，所以网络训练是无效的
    is_traditional_apf = (
        label == "apf_traditional" or
        (env.get("ACTION_FORCE_RATIO") == "1.0" and
         env.get("DELTA_K_ATT") == "0.0" and
         env.get("DELTA_LAMBDA_1") == "0.0" and
         env.get("DELTA_K_REP") == "0.0" and
         env.get("DELTA_RADIUS") == "0.0")
    )
    
    if is_traditional_apf:
        # 🔧 传统APF评估模式：禁用网络训练相关功能，只进行环境交互
        env["PER_ENABLED"] = "0"  # 禁用经验回放（不需要存储经验）
        env["LEARNING_WARMUP_ENABLED"] = "0"  # 禁用预热（不需要预热）
        env["SAVE_MODEL"] = "0"  # 禁用模型保存（网络不会被使用）
        env["ADAPTIVE_PATIENCE"] = "999999"  # 禁用自适应学习（设置极大值）
        env["NOISE_SCALE"] = "0.0"  # 禁用OU噪声（评估模式）
        env["RANDOM_ACTION_PROB_TRAINING"] = "0.0"  # 禁用随机动作（评估模式）
        # 🔧 关键：设置标志，让训练脚本跳过网络更新
        env["SKIP_NETWORK_UPDATE"] = "1"  # 新增标志，提示训练脚本跳过网络更新
        print(f"[并行训练-{label}] 🔧 传统APF模式：启用评估模式，跳过网络训练（节省计算资源）", file=sys.stderr)
    
    env["EXP_NAME"] = label
    
    # === 日志控制 ===
    env["QUIET_OUTPUT"] = "1"  # 并行训练时减少日志输出
    
    # === 参数一致性说明 ===
    # 1. 消融脚本通过环境变量覆盖特定配置（如 ENABLE_ACTION_CORRECTION, ACTION_FORCE_RATIO, DELTA_* 等）
    # 2. 其他所有参数使用 run_optimized.sh 的默认值机制（${VAR:-default}）
    # 3. run_optimized.sh 会将所有参数传递给 paper3d_train_optimized.py
    # 4. 因此，只要 run_optimized.sh 的默认值与正常训练时一致，消融实验的参数就一致
    # 5. ⚠️ 注意：如果父进程环境中有训练相关变量，可能会影响结果，建议在干净环境中运行
    
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
    print(f"[并行训练-{label}] 🔧 关键环境变量配置:", file=sys.stderr)
    print(f"  ACTION_FORCE_RATIO={env.get('ACTION_FORCE_RATIO')}", file=sys.stderr)
    print(f"  ACTION_FORCE_RATIO_SCHEDULE_PCT={env.get('ACTION_FORCE_RATIO_SCHEDULE_PCT', '未设置（将使用默认schedule）')}", file=sys.stderr)
    print(f"  DELTA_K_ATT={env.get('DELTA_K_ATT')}", file=sys.stderr)
    print(f"  DELTA_LAMBDA_1={env.get('DELTA_LAMBDA_1')}", file=sys.stderr)
    print(f"  DELTA_K_REP={env.get('DELTA_K_REP')}", file=sys.stderr)
    print(f"  DELTA_RADIUS={env.get('DELTA_RADIUS')}", file=sys.stderr)
    print(f"  USE_TF_POTENTIAL_FIELD={env.get('USE_TF_POTENTIAL_FIELD')}", file=sys.stderr)
    print(f"  UNLOCK_ENV_ON_SUCCESS={env.get('UNLOCK_ENV_ON_SUCCESS')} (消融实验：始终禁用课程学习)", file=sys.stderr)
    print(f"  UNLOCK_ENV_ON_PLATEAU={env.get('UNLOCK_ENV_ON_PLATEAU')} (消融实验：始终禁用课程学习)", file=sys.stderr)
    print(f"  RANDOM_TERRAIN={env.get('RANDOM_TERRAIN')} (消融实验：始终使用固定地形)", file=sys.stderr)
    print(f"  SCENARIO_SEED={env.get('SCENARIO_SEED')}", file=sys.stderr)
    print(f"  SEED={env.get('SEED', '未设置（将使用默认1337）')}", file=sys.stderr)
    print(f"  TERRAIN_COMPLEXITY_LEVEL={env.get('TERRAIN_COMPLEXITY_LEVEL')}", file=sys.stderr)
    print(f"  MAP_SIZE={env.get('MAP_SIZE')}", file=sys.stderr)
    print(f"  MOUNTAIN_MIN_DISTANCE={env.get('MOUNTAIN_MIN_DISTANCE')}", file=sys.stderr)
    print(f"  TERRAIN_CONTACT_EPS={env.get('TERRAIN_CONTACT_EPS', '未设置（将使用默认1.5）')}", file=sys.stderr)  # 🔧 新增：显示碰撞检测阈值
    print(f"{'='*70}\n", file=sys.stderr)
    
    try:
        # 🚨 关键修复：添加超时机制，防止进程挂起
        # 估算超时时间：每个回合约60-80秒，加上启动时间，200回合约4-5小时
        # 设置超时为估算时间的1.5倍，确保有足够缓冲
        estimated_timeout = max(3600, episodes * 100)  # 至少1小时，或每回合100秒
        result = subprocess.run(cmd, check=True, env=env, capture_output=False, timeout=estimated_timeout)
        log_dir = find_latest_log_dir(label, logs_root)
        metrics = load_metrics(log_dir)
        print(f"[并行训练-{label}] 完成: {cfg.get('name', label)}")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": log_dir,
            "metrics": metrics,
            "success": True
        }
    except subprocess.TimeoutExpired as e:
        print(f"[并行训练-{label}] 训练超时: {cfg.get('name', label)}, 超时时间: {estimated_timeout}秒")
        print(f"[并行训练-{label}] 提示：训练可能仍在进行，但超过了预期时间。可以检查日志目录确认。")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": None,
            "metrics": {},
            "success": False
        }
    except subprocess.CalledProcessError as e:
        print(f"[并行训练-{label}] 训练失败: {cfg.get('name', label)}, 错误码: {e.returncode}")
        print(f"[并行训练-{label}] 提示：可能是GPU内存不足、XLA编译错误或数值不稳定导致")
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": None,
            "metrics": {},
            "success": False
        }
    except (FileNotFoundError, Exception) as e:
        print(f"[并行训练-{label}] 查找日志或加载指标失败: {cfg.get('name', label)}, 错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            "label": label,
            "name": cfg.get("name", label),
            "name_en": cfg.get("name_en", cfg.get("name", label)),
            "description": cfg.get("description", ""),
            "log_dir": None,
            "metrics": {},
            "success": False
        }


def run_experiment(cfg: Dict, positions_file: Path, args, cache: Dict[str, Dict], gpu_id: int = None) -> Dict:
    """运行单个实验配置（串行模式）"""
    label = cfg["label"]
    
    if args.reuse and label in cache:
        print(f"[复用] {label}")
        return cache[label]
    
    # 🔧 关键修复：使用公共函数设置基础环境变量，消除代码冗余
    env = setup_base_env_vars(positions_file, gpu_id)
    
    # === 应用实验特定配置（必须在基础配置之后，确保覆盖）===
    env.update(cfg.get("env", {}))
    
    # 🚨 关键修复：消融实验始终禁用课程学习，确保所有实验使用固定地形和位置
    # 必须在应用实验配置后强制设置，防止实验配置覆盖基础设置
    env["UNLOCK_ENV_ON_SUCCESS"] = "0"
    env["UNLOCK_ENV_ON_PLATEAU"] = "0"
    env["RANDOM_TERRAIN"] = "0"
    env["PER_ENV_TERRAIN"] = "0"
    env["PER_EPISODE_TERRAIN"] = "0"
    
    # 🔧 关键修复：处理 ACTION_FORCE_RATIO_SCHEDULE_PCT，确保行为一致
    # 对于 action_apf_fusion，如果配置中未设置 ACTION_FORCE_RATIO_SCHEDULE_PCT，
    # 则完全删除该环境变量（包括从父进程继承的），让 run_optimized.sh 检测到变量未设置并使用默认 schedule
    if label == "action_apf_fusion":
        if "ACTION_FORCE_RATIO_SCHEDULE_PCT" not in cfg.get("env", {}):
            # 删除可能从父进程继承的值，确保 run_optimized.sh 检测到变量未设置并使用默认 schedule
            if "ACTION_FORCE_RATIO_SCHEDULE_PCT" in env:
                del env["ACTION_FORCE_RATIO_SCHEDULE_PCT"]
        elif cfg.get("env", {}).get("ACTION_FORCE_RATIO_SCHEDULE_PCT") == "DISABLED":
            # 如果配置中明确设置为 "DISABLED"，确保环境变量也设置为 "DISABLED"
            # run_optimized.sh 会将其转换为空字符串，禁用 schedule
            env["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = "DISABLED"
    
    # 🚨 关键修复：如果配置中设置了 ACTION_FORCE_RATIO_SCHEDULE_PCT 为 "DISABLED" 或空字符串，
    # run_optimized.sh 会检查并设置为空字符串，不使用默认值
    # 训练脚本会检查 "DISABLED" 或空字符串，跳过schedule，保持固定的 ACTION_FORCE_RATIO 值
    
    # 🔧 关键优化：检测传统APF配置，启用评估模式（跳过网络训练）
    # 传统APF特点：ACTION_FORCE_RATIO=1.0 且 DELTA_*=0.0
    # 因为网络动作被忽略，网络参数不影响势场，所以网络训练是无效的
    is_traditional_apf = (
        label == "apf_traditional" or
        (env.get("ACTION_FORCE_RATIO") == "1.0" and
         env.get("DELTA_K_ATT") == "0.0" and
         env.get("DELTA_LAMBDA_1") == "0.0" and
         env.get("DELTA_K_REP") == "0.0" and
         env.get("DELTA_RADIUS") == "0.0")
    )
    
    if is_traditional_apf:
        # 🔧 传统APF评估模式：禁用网络训练相关功能，只进行环境交互
        env["PER_ENABLED"] = "0"  # 禁用经验回放（不需要存储经验）
        env["LEARNING_WARMUP_ENABLED"] = "0"  # 禁用预热（不需要预热）
        env["SAVE_MODEL"] = "0"  # 禁用模型保存（网络不会被使用）
        env["ADAPTIVE_PATIENCE"] = "999999"  # 禁用自适应学习（设置极大值）
        env["NOISE_SCALE"] = "0.0"  # 禁用OU噪声（评估模式）
        env["RANDOM_ACTION_PROB_TRAINING"] = "0.0"  # 禁用随机动作（评估模式）
        # 🔧 关键：设置标志，让训练脚本跳过网络更新
        env["SKIP_NETWORK_UPDATE"] = "1"  # 新增标志，提示训练脚本跳过网络更新
        print(f"[运行] {label}: 🔧 传统APF模式：启用评估模式，跳过网络训练（节省计算资源）")
    
    env["EXP_NAME"] = label
    
    cmd = [
        args.script,
        str(args.episodes),
        str(args.batch_size),
        label,
        str(args.use_weighted_reward),
        args.algorithm,
    ]
    
    print(f"\n{'='*70}")
    print(f"[运行] {cfg.get('name', label)}")
    print(f"  描述: {cfg.get('description', '')}")
    print(f"  🔧 关键环境变量配置:")
    print(f"    ACTION_FORCE_RATIO={env.get('ACTION_FORCE_RATIO')}")
    print(f"    ACTION_FORCE_RATIO_SCHEDULE_PCT={env.get('ACTION_FORCE_RATIO_SCHEDULE_PCT', '未设置（将使用默认schedule）')}")
    print(f"    DELTA_K_ATT={env.get('DELTA_K_ATT')}")
    print(f"    DELTA_LAMBDA_1={env.get('DELTA_LAMBDA_1')}")
    print(f"    DELTA_K_REP={env.get('DELTA_K_REP')}")
    print(f"    DELTA_RADIUS={env.get('DELTA_RADIUS')}")
    print(f"    USE_TF_POTENTIAL_FIELD={env.get('USE_TF_POTENTIAL_FIELD')}")
    print(f"    SCENARIO_SEED={env.get('SCENARIO_SEED')}")
    print(f"    SEED={env.get('SEED', '未设置（将使用默认1337）')}")
    print(f"    TERRAIN_COMPLEXITY_LEVEL={env.get('TERRAIN_COMPLEXITY_LEVEL')}")
    print(f"    MAP_SIZE={env.get('MAP_SIZE')}")
    print(f"    MOUNTAIN_MIN_DISTANCE={env.get('MOUNTAIN_MIN_DISTANCE')}")
    print(f"    UNLOCK_ENV_ON_SUCCESS={env.get('UNLOCK_ENV_ON_SUCCESS')} (消融实验：始终禁用课程学习)")
    print(f"    UNLOCK_ENV_ON_PLATEAU={env.get('UNLOCK_ENV_ON_PLATEAU')} (消融实验：始终禁用课程学习)")
    print(f"    RANDOM_TERRAIN={env.get('RANDOM_TERRAIN')} (消融实验：始终使用固定地形)")
    print(f"    NOISE_SCALE={env.get('NOISE_SCALE', '未设置（将使用默认0.35）')} (OU噪声幅度)")
    print(f"    RANDOM_ACTION_PROB_TRAINING={env.get('RANDOM_ACTION_PROB_TRAINING', '未设置（将使用默认0.00）')} (训练阶段随机动作概率)")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True, env=env)
        log_dir = find_latest_log_dir(label, args.logs_root)
        metrics = load_metrics(log_dir)
        result = {
            "label": label,
            "name": cfg.get("name", label),
            "description": cfg.get("description", ""),
            "log_dir": log_dir,
            "metrics": metrics,
            "success": True
        }
        cache[label] = result
        return result
    except subprocess.CalledProcessError as e:
        print(f"[错误] 训练失败: {cfg.get('name', label)}, 错误码: {e.returncode}")
        print(f"[错误] 提示：如果是GPU内存或XLA编译问题，可以尝试：")
        print(f"  1. 降低批次大小（--batch-size）")
        print(f"  2. 禁用XLA（XLA_GLOBAL=0）")
        print(f"  3. 使用串行模式（--no-parallel）")
        raise
    except (FileNotFoundError, Exception) as e:
        print(f"[错误] 查找日志或加载指标失败: {cfg.get('name', label)}, 错误: {e}")
        raise




def smooth_curve(data: np.ndarray, method: str = "moving_average", window: int = 10) -> np.ndarray:
    """平滑曲线以减少振幅"""
    if len(data) < 2:
        return data
    
    if method == "moving_average":
        # 移动平均
        return uniform_filter1d(data.astype(float), size=window, mode='nearest')
    elif method == "spline":
        # 样条插值
        try:
            from scipy.interpolate import make_interp_spline
            if len(data) < 4:
                return data
            x = np.arange(len(data))
            x_smooth = np.linspace(0, len(data)-1, len(data) * 2)
            spline = make_interp_spline(x, data, k=min(3, len(data)-1))
            y_smooth = spline(x_smooth)
            # 下采样回原长度
            indices = np.linspace(0, len(y_smooth)-1, len(data), dtype=int)
            return y_smooth[indices]
        except ImportError:
            return data
    elif method == "poly":
        # 多项式拟合
        if len(data) < 3:
            return data
        x = np.arange(len(data))
        degree = min(5, len(data) - 1)
        coeffs = np.polyfit(x, data, degree)
        poly = np.poly1d(coeffs)
        return poly(x)
    else:
        return data


def plot_comparison_rewards(series: List[Dict], title: str, output_path: Path, 
                            smooth_window: int = 10, fit_method: str = "moving_average"):
    """Plot comparison reward curves (raw + fitted, on same plot) (English only)"""
    # 🔧 关键修复：确保使用英文字体
    setup_english_fonts()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 🔧 使用高对比度颜色，所有拟合曲线使用实线，确保对比明显
    # 颜色方案：深蓝、深红、深绿、深紫（高对比度，与损失图保持一致）
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']  # 深蓝、深红、深绿、深紫
    
    has_data = False
    
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        episodes = range(1, len(rewards) + 1)
        rewards_array = np.array(rewards)
        
        # 原始曲线（半透明，细线）
        color = colors[idx % len(colors)]
        # 🔧 关键修复：确保使用英文标签，避免回退到中文name
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        # 如果name_en仍然是中文（包含中文字符），使用label作为回退
        if name_en and any('\u4e00' <= char <= '\u9fff' for char in str(name_en)):
            name_en = item.get('label', 'Unknown')
        ax.plot(episodes, rewards, 
                label=f"{name_en} (Raw)", 
                color=color, 
                alpha=0.3, 
                linewidth=1,
                linestyle='-')
        
        # 拟合曲线（实线，粗线，确保对比明显）
        smoothed = smooth_curve(rewards_array, method=fit_method, window=smooth_window)
        ax.plot(episodes, smoothed, 
                label=f"{name_en} (Fitted)", 
                color=color, 
                alpha=0.9, 
                linewidth=2.5,
                linestyle='-')  # 🔧 所有拟合曲线使用实线
    
    if has_data:
        # 🔧 关键修复：显式指定字体族，确保图例文本正确显示
        ax.set_title(f"{title}\n(Fit Method: {fit_method}, Window: {smooth_window})", 
                     fontsize=16, fontweight='bold', pad=20, fontfamily='DejaVu Sans')
        ax.set_xlabel("Episode", fontsize=14, fontfamily='DejaVu Sans')
        ax.set_ylabel("Reward", fontsize=14, fontfamily='DejaVu Sans')
        # 🔧 关键修复：图例必须显式设置字体，否则会显示方框
        # 🔧 统一图例位置：右上角
        legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.9, prop={'family': 'DejaVu Sans'})
        # 确保图例文本使用正确字体
        for text in legend.get_texts():
            text.set_fontfamily('DejaVu Sans')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Complete] Reward comparison plot: {output_path}")
    else:
        print(f"[Warning] {title} has no available reward data, skipping plot: {output_path}")
    plt.close(fig)


def plot_comparison_success_collision_clearance(series: List[Dict], title: str, output_path: Path, 
                                               smooth_window: int = 10, fit_method: str = "moving_average"):
    """Plot success rate, collision counts, and average clearance comparison (English only)"""
    # 🔧 关键修复：确保使用英文字体
    setup_english_fonts()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    has_data = False
    
    # 🔧 使用高对比度颜色，所有曲线使用实线，确保对比明显
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']  # 深蓝、深红、深绿、深紫
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        # 🔧 关键修复：确保使用英文标签，避免回退到中文name
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        # 如果name_en仍然是中文（包含中文字符），使用label作为回退
        if name_en and any('\u4e00' <= char <= '\u9fff' for char in str(name_en)):
            name_en = item.get('label', 'Unknown')
        color = colors[idx % len(colors)]
        linestyle = '-'  # 🔧 所有曲线使用实线
        
        # 1. 成功率（滑动窗口平均）
        success_flags = metrics.get("success_flags", [])
        if success_flags:
            has_data = True
            episodes = range(1, len(success_flags) + 1)
            success_array = np.array(success_flags, dtype=float)
            
            # Calculate sliding window success rate (window size=50)
            window_size = 50
            success_rate = []
            for i in range(len(success_array)):
                start_idx = max(0, i - window_size + 1)
                window_data = success_array[start_idx:i+1]
                rate = np.mean(window_data) if len(window_data) > 0 else 0.0
                success_rate.append(rate)
            
            axes[0].plot(episodes, success_rate, 
                       label=name_en, 
                       color=color, 
                       linewidth=2.5, 
                       alpha=0.9, 
                       linestyle=linestyle)
        
        # 2. 碰撞次数
        collision_counts = metrics.get("collision_counts", [])
        if collision_counts:
            has_data = True
            episodes = range(1, len(collision_counts) + 1)
            collisions_array = np.array(collision_counts, dtype=float)
            
            # Smooth processing
            smoothed = smooth_curve(collisions_array, method=fit_method, window=smooth_window)
            axes[1].plot(episodes, smoothed, 
                       label=name_en, 
                       color=color, 
                       linewidth=2.5, 
                       alpha=0.9, 
                       linestyle=linestyle)
        
        # 3. 平均净空（平均值，更有统计意义）
        min_distances = metrics.get("min_distances_to_obstacle", [])
        if min_distances:
            has_data = True
            episodes = range(1, len(min_distances) + 1)
            # 提取mean值（平均值），更有统计意义
            # 🔧 改进：正确处理None值，跳过无效数据点
            if isinstance(min_distances[0], dict):
                min_dist_values = []
                valid_episodes = []
                for ep_idx, d in enumerate(min_distances):
                    mean_val = d.get('mean', None) if isinstance(d, dict) else None
                    if mean_val is not None and np.isfinite(mean_val):
                        min_dist_values.append(float(mean_val))
                        valid_episodes.append(ep_idx + 1)
            else:
                # 如果不是字典，直接使用（向后兼容）
                min_dist_values = [float(d) if d is not None and np.isfinite(d) else None 
                                  for d in min_distances]
                valid_episodes = [ep_idx + 1 for ep_idx, d in enumerate(min_distances) 
                                if d is not None and np.isfinite(d)]
                min_dist_values = [d for d in min_dist_values if d is not None]
            
            if min_dist_values:
                min_distances_array = np.array(min_dist_values, dtype=float)
                # 过滤无效值（允许负值表示穿透）
                valid_mask = np.isfinite(min_distances_array) & (min_distances_array > -1000)
                if np.any(valid_mask):
                    # Smooth processing
                    smoothed = smooth_curve(min_distances_array, method=fit_method, window=smooth_window)
                    # 使用有效episode索引
                    plot_episodes = valid_episodes if valid_episodes else episodes[:len(smoothed)]
                    axes[2].plot(plot_episodes, smoothed, 
                               label=name_en, 
                               color=color, 
                               linewidth=2.5, 
                               alpha=0.9, 
                               linestyle=linestyle)
    
    if has_data:
        # 设置标题和标签
        axes[0].set_title("Success Rate (Moving Average, Window=50)", 
                         fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[0].set_ylabel("Success Rate", fontsize=12, fontfamily='DejaVu Sans')
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3, linestyle='--')
        # 🔧 统一图例位置：右上角
        legend0 = axes[0].legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
        if legend0:
            for text in legend0.get_texts():
                text.set_fontfamily('DejaVu Sans')
        
        axes[1].set_title("Collision Counts (Smoothed)", 
                         fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[1].set_ylabel("Collision Count", fontsize=12, fontfamily='DejaVu Sans')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        # 🔧 统一图例位置：右上角
        legend1 = axes[1].legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
        if legend1:
            for text in legend1.get_texts():
                text.set_fontfamily('DejaVu Sans')
        
        axes[2].set_title("Average Clearance (Average Distance to Obstacle, Smoothed)", 
                         fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[2].set_xlabel("Episode", fontsize=12, fontfamily='DejaVu Sans')
        axes[2].set_ylabel("Average Distance (m)", fontsize=12, fontfamily='DejaVu Sans')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        # 🔧 统一图例位置：右上角
        legend2 = axes[2].legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
        if legend2:
            for text in legend2.get_texts():
                text.set_fontfamily('DejaVu Sans')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Complete] Success rate/Collision/Clearance comparison plot: {output_path}")
    else:
        print(f"[Warning] {title} has no available data, skipping plot: {output_path}")
    plt.close(fig)


def plot_comparison_success_rate_and_clearance(series: List[Dict], title: str, output_path: Path, 
                                               smooth_window: int = 10, fit_method: str = "moving_average"):
    """Plot team success rate and minimum clearance distribution comparison (English only)"""
    # 🔧 Key fix: Ensure English fonts are used
    setup_english_fonts()
    
    fig = plt.figure(figsize=(18, 14))
    # 🔧 Modified layout: Remove single agent success rate subplot, only keep team success rate
    # Layout: Row 0 = Team Success Rate (full width), Row 1 = Min Clearance Time Series (full width),
    #         Row 2 = Min Clearance Distribution (left) + CDF (right), Row 3 = Quantiles (full width)
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 🔧 关键修复：在循环外创建所有子图，避免每次循环都覆盖
    ax1 = fig.add_subplot(gs[0, :])  # Team Success Rate
    ax2 = fig.add_subplot(gs[1, :])  # Minimum Clearance Time Series
    ax3 = fig.add_subplot(gs[2, 0])  # Minimum Clearance Distribution
    ax4 = fig.add_subplot(gs[2, 1])  # CDF
    ax5 = fig.add_subplot(gs[3, :])  # Quantiles
    
    has_data = False
    
    # 🔧 Use high contrast colors
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']  # Deep blue, deep red, deep green, deep purple
    
    # Get collision threshold (for violation probability calculation)
    collision_threshold = 1.5  # Default value, can be read from environment variables or config
    
    # 🔧 修复：在循环外定义quantiles，避免UnboundLocalError
    quantiles = [5, 10, 25, 50, 75, 90, 95]  # 默认分位数列表
    
    # 🔧 标记是否已经设置了阈值线（避免重复绘制）
    threshold_line_added_ax2 = False
    threshold_line_added_ax4 = False
    
    for idx, item in enumerate(series):
        metrics = item["metrics"]
        # 🔧 关键修复：确保使用英文标签，避免回退到中文name
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        # 如果name_en仍然是中文（包含中文字符），使用label作为回退
        if name_en and any('\u4e00' <= char <= '\u9fff' for char in str(name_en)):
            name_en = item.get('label', 'Unknown')
        color = colors[idx % len(colors)]
        linestyle = '-'
        
        # === 1. Team Success Rate (SR_team) only ===
        team_success_flags = metrics.get("team_success_flags", [])
        if team_success_flags:
            has_data = True
            episodes = range(1, len(team_success_flags) + 1)
            team_success_array = np.array(team_success_flags, dtype=float)
            
            # Calculate sliding window success rate
            window_size = 50
            success_rate = []
            for i in range(len(team_success_array)):
                start_idx = max(0, i - window_size + 1)
                window_data = team_success_array[start_idx:i+1]
                rate = np.mean(window_data) if len(window_data) > 0 else 0.0
                success_rate.append(rate)
            
            ax1.plot(episodes, success_rate, 
                   label=name_en, 
                   color=color, 
                   linewidth=2.5, 
                   alpha=0.9, 
                   linestyle=linestyle)
        
        # === 2. Minimum Clearance Distribution (Episode-level minimum clearance d_min^{i,(k)}) ===
        min_distances = metrics.get("min_distances_to_obstacle", [])
        if min_distances:
            has_data = True
            # Extract all episode minimum clearances (using min value, i.e., maximum penetration depth)
            all_min_clearances = []
            for d in min_distances:
                if isinstance(d, dict):
                    min_val = d.get('min', None)
                    if min_val is not None and np.isfinite(min_val):
                        all_min_clearances.append(float(min_val))
                elif d is not None and np.isfinite(d):
                    all_min_clearances.append(float(d))
            
            if all_min_clearances:
                all_min_clearances = np.array(all_min_clearances)
                
                # 2.1 Minimum clearance time series (smoothed)
                episodes = range(1, len(min_distances) + 1)
                # Extract min value sequence
                min_values = []
                valid_episodes = []
                for ep_idx, d in enumerate(min_distances):
                    if isinstance(d, dict):
                        min_val = d.get('min', None)
                    else:
                        min_val = d if d is not None and np.isfinite(d) else None
                    if min_val is not None and np.isfinite(min_val):
                        min_values.append(float(min_val))
                        valid_episodes.append(ep_idx + 1)
                
                if min_values:
                    min_values_array = np.array(min_values, dtype=float)
                    # Smooth processing
                    smoothed = smooth_curve(min_values_array, method=fit_method, window=smooth_window)
                    ax2.plot(valid_episodes, smoothed, 
                           label=name_en, 
                           color=color, 
                           linewidth=2.5, 
                           alpha=0.9, 
                           linestyle=linestyle)
                    
                    # Add collision threshold line (only once)
                    if not threshold_line_added_ax2:
                        ax2.axhline(y=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision Threshold')
                        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                        threshold_line_added_ax2 = True
                
                # 2.2 Minimum clearance distribution (histogram)
                # Filter outliers
                valid_clearances = all_min_clearances[np.isfinite(all_min_clearances) & (all_min_clearances > -1000) & (all_min_clearances < 1000)]
                if len(valid_clearances) > 0:
                    ax3.hist(valid_clearances, bins=50, alpha=0.6, color=color, label=name_en, edgecolor='black', linewidth=0.5)
                    # Add statistical information
                    mean_val = np.mean(valid_clearances)
                    median_val = np.median(valid_clearances)
                    ax3.axvline(x=mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'{name_en} Mean')
                    ax3.axvline(x=median_val, color=color, linestyle=':', linewidth=2, alpha=0.8, label=f'{name_en} Median')
                    # Add threshold lines (only once, after first histogram)
                    if idx == 0:
                        ax3.axvline(x=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                
                # 2.3 CDF curve (according to image definition: Pr(D_min ≤ δ))
                if len(valid_clearances) > 0:
                    # Calculate CDF: Pr(D_min ≤ δ)
                    sorted_clearances = np.sort(valid_clearances)
                    cdf_values = np.arange(1, len(sorted_clearances) + 1) / len(sorted_clearances)
                    
                    ax4.plot(sorted_clearances, cdf_values, 
                           color=color, linewidth=2.5, alpha=0.9, 
                           label=f'{name_en} CDF', linestyle=linestyle)
                    
                    # Add threshold lines (only once)
                    if not threshold_line_added_ax4:
                        ax4.axvline(x=collision_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision Threshold')
                        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                        threshold_line_added_ax4 = True
                
                # 2.4 Quantiles and violation probability
                if len(valid_clearances) > 0:
                    # Calculate quantiles (according to image definition: Q5%, Q50%, Q95%)
                    # quantiles已在循环外定义
                    quantile_values = [np.percentile(valid_clearances, q) for q in quantiles]
                    
                    # Calculate violation probability (P_viol^i(δ) = (1/N) * Σ 1{d_min < δ})
                    violation_probs = []
                    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
                    for threshold in thresholds:
                        prob = np.mean(valid_clearances < threshold)
                        violation_probs.append(prob)
                    
                    # Plot quantiles (bar chart) - use offset for multiple experiments
                    ax5_twin = ax5.twinx()
                    # 🔧 修复：为每个实验使用不同的x位置，避免条形图重叠
                    x_positions = np.arange(len(quantiles))
                    width = 0.25  # 条形宽度
                    offset = (idx - len(series) / 2 + 0.5) * width
                    bars1 = ax5.bar(x_positions + offset, quantile_values, 
                                   width=width, alpha=0.6, color=color, label=f'{name_en} Quantiles', 
                                   edgecolor='black', linewidth=0.5)
                    line1 = ax5_twin.plot(thresholds, violation_probs, 
                                         color=color, marker='o', linewidth=2.5, markersize=6, 
                                         label=f'{name_en} Violation Prob', linestyle=linestyle)
    
    # 🔧 设置所有子图的标题和标签（在循环外，只设置一次）
    ax1.set_title("Team Success Rate (SR_team)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax1.set_ylabel("Success Rate", fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_xlabel("Episode", fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--')
    legend1 = ax1.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    if legend1:
        for text in legend1.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    ax2.set_title("Episode-Level Minimum Clearance (d_min^{i,(k)})", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax2.set_xlabel("Episode", fontsize=12, fontfamily='DejaVu Sans')
    ax2.set_ylabel("Minimum Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax2.grid(True, alpha=0.3, linestyle='--')
    legend2 = ax2.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    if legend2:
        for text in legend2.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    ax3.set_title("Minimum Clearance Distribution", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax3.set_xlabel("Minimum Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax3.set_ylabel("Frequency", fontsize=12, fontfamily='DejaVu Sans')
    ax3.grid(True, alpha=0.3, linestyle='--')
    legend3 = ax3.legend(loc='upper right', fontsize=9, prop={'family': 'DejaVu Sans'})
    if legend3:
        for text in legend3.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    ax4.set_title("CDF: Pr(D_min ≤ δ)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax4.set_xlabel("Minimum Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax4.set_ylabel("Cumulative Probability", fontsize=12, fontfamily='DejaVu Sans')
    ax4.grid(True, alpha=0.3, linestyle='--')
    legend4 = ax4.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    if legend4:
        for text in legend4.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    ax5.set_title("Quantiles (Q5%, Q50%, Q95%) & Violation Probability P_viol(δ)", 
                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    ax5.set_xlabel("Quantile / Threshold (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax5.set_ylabel("Clearance (m)", fontsize=12, fontfamily='DejaVu Sans')
    ax5_twin = ax5.twinx()
    ax5_twin.set_ylabel("Violation Probability", fontsize=12, fontfamily='DejaVu Sans')
    # 🔧 修复：设置分位数x轴标签（quantiles已在循环外定义）
    if has_data:
        ax5.set_xticks(np.arange(len(quantiles)))
        ax5.set_xticklabels([f'Q{q}%' for q in quantiles])
    ax5.grid(True, alpha=0.3, linestyle='--')
    legend5 = ax5.legend(loc='upper left', fontsize=9, prop={'family': 'DejaVu Sans'})
    legend5_twin = ax5_twin.legend(loc='upper right', fontsize=9, prop={'family': 'DejaVu Sans'})
    if legend5:
        for text in legend5.get_texts():
            text.set_fontfamily('DejaVu Sans')
    if legend5_twin:
        for text in legend5_twin.get_texts():
            text.set_fontfamily('DejaVu Sans')
    
    if has_data:
        plt.suptitle(title, fontsize=16, fontweight='bold', fontfamily='DejaVu Sans', y=0.995)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Complete] Success rate and clearance comparison plot: {output_path}")
    else:
        print(f"[Warning] {title} has no available data, skipping plot: {output_path}")
    plt.close(fig)


def plot_comparison_losses(series: List[Dict], title: str, output_path: Path):
    """Plot comparison loss curves (English only)"""
    # 🔧 关键修复：确保使用英文字体
    setup_english_fonts()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    has_data = False
    
    # 🔧 使用高对比度颜色，所有曲线使用实线，确保对比明显
    # 颜色方案：蓝色、红色、绿色、紫色（高对比度）
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']  # 深蓝、深红、深绿、深紫
    markers = ['', '', '', '']  # 不使用标记点，保持线条清晰
    
    for idx, item in enumerate(series):
        history = item["metrics"].get("loss_history", [])
        if not history:
            continue
        has_data = True
        steps = [entry.get("step", idx) for idx, entry in enumerate(history)]
        critic = [entry.get("critic_loss", 0) for entry in history]
        actor = [entry.get("actor_loss", 0) for entry in history]
        color = colors[idx % len(colors)]
        # 🔧 关键修复：确保使用英文标签，避免回退到中文name
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        # 如果name_en仍然是中文（包含中文字符），使用label作为回退
        if name_en and any('\u4e00' <= char <= '\u9fff' for char in str(name_en)):
            name_en = item.get('label', 'Unknown')
        linestyle = '-'  # 🔧 所有曲线使用实线
        
        axes[0].plot(steps, critic, label=f"{name_en} (Critic)", 
                    color=color, linewidth=2.5, alpha=0.9, linestyle=linestyle)
        axes[1].plot(steps, actor, label=f"{name_en} (Actor)", 
                    color=color, linewidth=2.5, alpha=0.9, linestyle=linestyle)
    
    if has_data:
        # 🔧 关键修复：所有文本显式指定字体
        axes[0].set_title("Critic Loss", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[0].set_ylabel("Loss", fontsize=12, fontfamily='DejaVu Sans')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        # 🔧 统一图例位置：右上角
        legend0 = axes[0].legend(loc='upper right', fontsize=11, prop={'family': 'DejaVu Sans'})
        for text in legend0.get_texts():
            text.set_fontfamily('DejaVu Sans')
        axes[0].set_facecolor('#fafafa')
        
        axes[1].set_title("Actor Loss", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        axes[1].set_xlabel("Update Step", fontsize=12, fontfamily='DejaVu Sans')
        axes[1].set_ylabel("Loss", fontsize=12, fontfamily='DejaVu Sans')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        # 🔧 统一图例位置：右上角
        legend1 = axes[1].legend(loc='upper right', fontsize=11, prop={'family': 'DejaVu Sans'})
        for text in legend1.get_texts():
            text.set_fontfamily('DejaVu Sans')
        axes[1].set_facecolor('#fafafa')
        
        fig.suptitle(f"{title} - Loss Curves", fontsize=16, fontweight='bold', y=0.995, fontfamily='DejaVu Sans')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Complete] Loss comparison plot: {output_path}")
    else:
        print(f"[Warning] {title} has no available loss data, skipping plot: {output_path}")
    plt.close(fig)


def generate_interactive_comparison(series: List[Dict], title: str, output_path: Path,
                                   smooth_window: int = 10, fit_method: str = "moving_average"):
    """生成交互式对比图表（使用plotly）"""
    if not HAS_PLOTLY:
        print(f"[跳过] 未安装 plotly，跳过交互图生成: {output_path}")
        return
    
    fig = go.Figure()
    
    # 🔧 使用高对比度颜色，与静态图保持一致
    colors = ['#0066CC', '#CC0000', '#00AA00', '#9900CC']  # 深蓝、深红、深绿、深紫
    has_data = False
    
    for idx, item in enumerate(series):
        rewards = item["metrics"].get("episode_rewards", [])
        if not rewards:
            continue
        has_data = True
        episodes = list(range(1, len(rewards) + 1))
        rewards_array = np.array(rewards)
        color = colors[idx % len(colors)]
        
        # 原始数据（半透明）
        # 🔧 关键修复：确保使用英文标签，避免回退到中文name
        name_en = item.get('name_en') or item.get('label', 'Unknown')
        # 如果name_en仍然是中文（包含中文字符），使用label作为回退
        if name_en and any('\u4e00' <= char <= '\u9fff' for char in str(name_en)):
            name_en = item.get('label', 'Unknown')
        fig.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name=f"{name_en} (Raw)",
            line=dict(width=1, color=color),
            opacity=0.3,
            showlegend=True
        ))
        
        # 拟合数据（实线）
        smoothed = smooth_curve(rewards_array, method=fit_method, window=smooth_window)
        fig.add_trace(go.Scatter(
            x=episodes,
            y=smoothed,
            mode='lines',
            name=f"{name_en} (Fitted)",
            line=dict(width=3, color=color),
            showlegend=True
        ))
    
    if has_data:
        fig.update_layout(
            title=f"{title} - Interactive Reward Comparison",
            xaxis_title="Episode",
            yaxis_title="Reward",
            hovermode='closest',
            template='plotly_white',
            width=1400,
            height=700,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', font=dict(size=12)),
            plot_bgcolor='#fafafa'
        )
        pyo.plot(fig, filename=str(output_path), auto_open=False)
        print(f"[Complete] Interactive comparison plot: {output_path}")
    else:
        print(f"[Warning] {title} has no available reward data, skipping interactive plot: {output_path}")


def main():
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"[错误] 找不到训练脚本: {script_path}")
        sys.exit(1)
    
    # 🔧 优化：先生成固定位置文件，再创建批次目录（避免不必要的目录创建）
    # 生成固定位置文件（所有实验共享）
    exp_name_prefix = "ablation_action_pf"
    positions_file = ensure_fixed_positions(args.positions_file, args, exp_name_prefix)
    
    # 🔧 新增：创建批次管理器和批次目录（在位置文件生成之后）
    # 批次目录用于组织消融实验的结果，确保所有实验使用相同的位置文件
    manager = AblationBatchManager()
    batch_config = {
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "algorithm": args.algorithm,
        "use_weighted_reward": args.use_weighted_reward,
        "seed": 252488,  # 训练随机种子（与 run_optimized.sh 默认值一致）
        "scenario_seed": 67,  # 🔧 地形种子（与位置生成保持一致，确保地形一致）
        "terrain_complexity": 3,  # 🔧 修复：与 run_optimized.sh 默认值一致（3）
        "map_size": 200,  # 与 run_optimized.sh 默认值一致
        "mountain_min_distance": 55,  # 与 run_optimized.sh 默认值一致
        "positions_file": str(positions_file),
        "notes": "消融实验：对比动作与势场修正的效果（使用 run_optimized.sh 运行完整训练）"
    }
    batch_dir = manager.create_batch(config=batch_config)
    
    print(f"{'='*70}")
    print(f"✅ 批次目录已创建: {batch_dir}")
    print(f"   用途：组织消融实验的结果，所有实验共享位置文件: {positions_file}")
    print(f"{'='*70}\n")
    
    # 🔧 修改：输出目录使用批次目录下的plots子目录
    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache: Dict[str, Dict] = {}
    series = []
    
    # 根据参数过滤要运行的实验
    if args.quick_comparison:
        # 快速对比模式：运行 apf_traditional、apf_learnable 和 action_apf_fusion（三者对比）
        # 🚨 关键修复：确保可学习APF先运行，避免传统APF的DELTA_*=0.0覆盖环境变量
        # 虽然每个实验是独立进程，但为了保险起见，将可学习APF放在前面
        labels_order = ["apf_learnable", "apf_traditional", "action_apf_fusion"]
        configs_to_run = [cfg for cfg in EXPERIMENT_CONFIGS if cfg["label"] in labels_order]
        # 按照指定顺序排序
        configs_to_run = sorted(configs_to_run, key=lambda x: labels_order.index(x["label"]))
        print(f"[Info] Quick comparison mode: Running apf_learnable, apf_traditional, and action_apf_fusion (three-way comparison, learnable APF prioritized)")
    elif args.experiments:
        configs_to_run = [cfg for cfg in EXPERIMENT_CONFIGS if cfg["label"] in args.experiments]
        if not configs_to_run:
            print(f"[错误] 未找到指定的实验: {args.experiments}")
            sys.exit(1)
        print(f"[信息] 仅运行指定实验: {[cfg['name'] for cfg in configs_to_run]}")
    else:
        # 🚨 关键修复：确保可学习APF先运行，避免传统APF的DELTA_*=0.0覆盖环境变量
        # 虽然每个实验是独立进程，但为了保险起见，将可学习APF放在前面
        configs_to_run = sorted(EXPERIMENT_CONFIGS, key=lambda x: (x["label"] != "apf_learnable", x["label"]))
        print(f"[Info] Experiment execution order (learnable APF prioritized): {[cfg['label'] for cfg in configs_to_run]}")
    
    print(f"\n{'='*70}")
    print(f"动作与势场修正消融对比实验")
    print(f"实验数量: {len(configs_to_run)}")
    for cfg in configs_to_run:
        print(f"  - {cfg['name']}: {cfg['description']}")
    if args.parallel:
        print(f"模式: 并行训练（{len(configs_to_run)}个实验同时运行，充分利用GPU）")
    else:
        print(f"模式: 串行训练（依次运行）")
    print(f"{'='*70}")
     
    # 运行选定的实验
    if args.parallel:
        # 并行训练模式
        print(f"\n[并行训练] 准备同时运行 {len(configs_to_run)} 个实验...")
        
        # 解析GPU ID列表
        gpu_ids_list = None
        if args.gpu_ids:
            gpu_ids_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
            if len(gpu_ids_list) != len(EXPERIMENT_CONFIGS):
                print(f"[Info] GPU ID count ({len(gpu_ids_list)}) does not match experiment count ({len(EXPERIMENT_CONFIGS)}), will cycle through GPUs")
        else:
            # 检测可用GPU数量
            try:
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    num_gpus = len([line for line in result.stdout.split('\n') if 'GPU' in line])
                    if num_gpus > 0:
                        gpu_ids_list = list(range(num_gpus))
                        print(f"[Info] Detected {num_gpus} GPUs, will assign to experiments")
            except Exception:
                pass  # 如果检测失败，使用None（共享GPU模式）
        
        # 准备参数元组列表
        tasks = []
        for idx, cfg in enumerate(configs_to_run):
            gpu_id = gpu_ids_list[idx % len(gpu_ids_list)] if gpu_ids_list else None
            task = (
                cfg,
                positions_file,
                str(Path(args.script).resolve()),
                args.episodes,
                args.batch_size,
                args.use_weighted_reward,
                args.algorithm,
                args.logs_root,
                gpu_id
            )
            tasks.append(task)
        
        # 使用进程池并行执行
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=len(configs_to_run)) as executor:
            futures = {executor.submit(run_experiment_worker, task): task[0]["label"] for task in tasks}
            
            results_dict = {}
            # 🚨 关键修复：添加超时机制，防止future.result()无限等待
            # 估算每个实验的最大等待时间（与subprocess超时一致）
            estimated_timeout = max(3600, args.episodes * 100)
            total_timeout = estimated_timeout * len(configs_to_run) + 3600  # 总超时时间，加上缓冲
            
            start_wait_time = time.time()
            for future in as_completed(futures):
                # 检查总超时时间
                if time.time() - start_wait_time > total_timeout:
                    print(f"[并行训练] ⚠️  总等待时间超过 {total_timeout} 秒，停止等待剩余任务")
                    break
                    
                label = futures[future]
                try:
                    # 设置单个future的超时时间（Python 3.2+支持）
                    try:
                        result = future.result(timeout=estimated_timeout)
                    except TypeError:
                        # 如果future.result()不支持timeout参数（Python < 3.2），直接调用
                        result = future.result()
                    results_dict[label] = result
                    if result.get("success", False):
                        print(f"[并行训练] ✓ {result['name']} 完成")
                    else:
                        print(f"[并行训练] ✗ {result['name']} 失败")
                except TimeoutError:
                    print(f"[并行训练] ✗ {label} 超时: future.result()等待超时")
                    cfg_failed = next((c for c in configs_to_run if c["label"] == label), None)
                    if cfg_failed:
                        results_dict[label] = {
                            "label": label,
                            "name": cfg_failed.get("name", label),
                            "name_en": cfg_failed.get("name_en", cfg_failed.get("name", label)),
                            "log_dir": None,
                            "metrics": {},
                            "success": False
                        }
                except Exception as e:
                    print(f"[并行训练] ✗ {label} 异常: {e}")
                    import traceback
                    traceback.print_exc()
                    cfg_failed = next((c for c in configs_to_run if c["label"] == label), None)
                    if cfg_failed:
                        results_dict[label] = {
                            "label": label,
                            "name": cfg_failed.get("name", label),
                            "name_en": cfg_failed.get("name_en", cfg_failed.get("name", label)),
                            "log_dir": None,
                            "metrics": {},
                            "success": False
                        }
        
        elapsed_time = time.time() - start_time
        print(f"\n[并行训练] 所有实验完成，总耗时: {elapsed_time:.1f}秒")
        
        # 按原始顺序整理结果
        for cfg in configs_to_run:
            label = cfg["label"]
            if label in results_dict:
                result = results_dict[label]
                # 🔧 关键修复：只添加成功完成的实验，或者有数据的实验
                if result.get("success", False) or result.get("metrics", {}).get("episode_rewards"):
                    series.append(result)
                else:
                    print(f"[Warning] Experiment {label} failed or has no data, skipping")
            else:
                print(f"[Warning] Experiment {label} not completed, skipping")
    else:
        # 串行训练模式
        for cfg in configs_to_run:
            result = run_experiment(cfg, positions_file, args, cache)
            series.append(result)
    
    # 🔧 修复：检查series是否为空，如果为空则提示用户
    if not series:
        print(f"\n{'='*70}")
        print(f"⚠️  警告: 没有可用的实验数据，无法生成对比图")
        print(f"{'='*70}")
        print(f"\n可能的原因:")
        print(f"  1. 所有实验都失败了")
        print(f"  2. 所有实验都没有episode_rewards数据")
        print(f"  3. 训练过程中出现了错误")
        print(f"\n建议:")
        print(f"  1. 检查日志目录: logs/")
        print(f"  2. 检查实验结果文件: logs/*/results.json")
        print(f"  3. 使用 quick_regenerate_plots.py 手动重新生成图表")
        print(f"{'='*70}\n")
        sys.exit(1)
    
    # 🔧 修复：检查series中是否有有效数据
    has_valid_data = False
    for item in series:
        if item.get("metrics", {}).get("episode_rewards"):
            has_valid_data = True
            break
    
    if not has_valid_data:
        print(f"\n{'='*70}")
        print(f"⚠️  警告: 所有实验都没有episode_rewards数据，无法生成对比图")
        print(f"{'='*70}")
        print(f"\n建议:")
        print(f"  1. 检查实验结果文件: logs/*/results.json")
        print(f"  2. 确认训练是否成功完成")
        print(f"  3. 使用 quick_regenerate_plots.py 手动重新生成图表")
        print(f"{'='*70}\n")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"✅ 找到 {len(series)} 个实验的数据，开始生成对比图...")
    print(f"{'='*70}\n")
    
    # 生成图表
    title = "Action vs APF Correction Ablation Comparison"
    
    # 🔧 修复：添加时间戳，避免覆盖之前的实验结果
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    # 奖励对比图（原始+拟合在同一图上）
    reward_png = output_dir / f"reward_comparison_{timestamp}.png"
    plot_comparison_rewards(
        series, 
        title, 
        reward_png,
        smooth_window=args.smooth_window,
        fit_method=args.fit_method
    )
    
    # Loss comparison plot
    loss_png = output_dir / f"loss_comparison_{timestamp}.png"
    plot_comparison_losses(series, title, loss_png)
    
    # 🚨 New: Success rate, collision counts, and average clearance comparison plot
    success_collision_png = output_dir / f"success_collision_clearance_comparison_{timestamp}.png"
    plot_comparison_success_collision_clearance(
        series, title, success_collision_png,
        smooth_window=args.smooth_window,
        fit_method=args.fit_method
    )
    
    # 🚨 New: Team success rate and minimum clearance distribution comparison plot (English only)
    success_clearance_png = output_dir / f"success_rate_and_clearance_comparison_{timestamp}.png"
    plot_comparison_success_rate_and_clearance(
        series, title, success_clearance_png,
        smooth_window=args.smooth_window,
        fit_method=args.fit_method
    )
    
    # 交互式对比图
    if args.generate_interactive:
        interactive_html = output_dir / f"interactive_comparison_{timestamp}.html"
        generate_interactive_comparison(
            series,
            title,
            interactive_html,
            smooth_window=args.smooth_window,
            fit_method=args.fit_method
        )
    
    # 保存汇总信息
    summary = {
        "timestamp": timestamp,  # 🔧 新增：记录时间戳
        "experiments": [
            {
                "label": item["label"],
                "name": item["name"],
                "name_en": item.get("name_en", item.get("label", "Unknown")),  # 🔧 关键修复：确保包含name_en字段
                "description": item.get("description", ""),
                "log_dir": item.get("log_dir", ""),
                "final_reward": item["metrics"].get("episode_rewards", [])[-1] if item["metrics"].get("episode_rewards") else None,
                "avg_reward": np.mean(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
                "max_reward": np.max(item["metrics"].get("episode_rewards", [])) if item["metrics"].get("episode_rewards") else None,
            }
            for item in series
        ],
        "output_files": {  # 🔧 新增：记录输出文件路径
            "reward_comparison": str(reward_png.name),
            "loss_comparison": str(loss_png.name),
        }
    }
    if args.generate_interactive:
        summary["output_files"]["interactive_comparison"] = str(interactive_html.name)
    
    # 🔧 修复：保存带时间戳的summary文件，避免覆盖
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=False, indent=2)
    
    # 🔧 同时保存一个latest_summary.json作为最新结果的快捷方式
    latest_summary_path = output_dir / "latest_summary.json"
    with open(latest_summary_path, "w", encoding="utf-8") as f_latest:
        json.dump(summary, f_latest, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"\nAblation comparison experiment completed!")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    print(f"Reward comparison plot: {reward_png.name}")
    print(f"Loss comparison plot: {loss_png.name}")
    if args.generate_interactive:
        print(f"Interactive plot: {interactive_html.name}")
    print(f"\nExperiment Results Summary:")
    for exp in summary["experiments"]:
        exp_name = exp.get('name_en', exp.get('name', 'Unknown'))
        # 🔧 关键修复：检查实验是否有数据
        has_data = exp.get('final_reward') is not None or exp.get('avg_reward') is not None or exp.get('max_reward') is not None
        
        if not has_data:
            # 实验失败或没有数据，显示明确的警告信息
            log_dir = exp.get('log_dir', '')
            if log_dir:
                print(f"  - {exp_name}: ⚠️  No data available (training may have failed or was interrupted)")
                print(f"    Log directory: {log_dir}")
            else:
                print(f"  - {exp_name}: ⚠️  No data available (training did not complete)")
        else:
            # 有数据，正常显示
            final_reward_str = f"{exp['final_reward']:.2f}" if exp['final_reward'] is not None else "N/A"
            avg_reward_str = f"{exp['avg_reward']:.2f}" if exp['avg_reward'] is not None else "N/A"
            max_reward_str = f"{exp['max_reward']:.2f}" if exp['max_reward'] is not None else "N/A"
            print(f"  - {exp_name}: Final Reward={final_reward_str}, "
                  f"Average Reward={avg_reward_str}, "
                  f"Max Reward={max_reward_str}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从结果文件复现实验并生成消融实验配置

功能：
1. 从 results.json 加载实验配置
2. 提取环境变量和训练参数
3. 生成消融实验配置（符合 ablation_action_pf_comparison.py 格式）
4. 保存配置供消融实验使用

使用方法：
    python reproduce_from_results.py <results_dir> [--output-dir <dir>] [--config-only]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

def load_results(results_dir: Path) -> Dict[str, Any]:
    """从结果目录加载实验配置"""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        # 尝试查找子目录中的 results.json
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                sub_results = subdir / "results.json"
                if sub_results.exists():
                    results_file = sub_results
                    break
        else:
            raise FileNotFoundError(f"未找到 results.json: {results_dir}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def extract_env_vars_from_args(args: Dict[str, Any]) -> Dict[str, str]:
    """从训练参数中提取环境变量"""
    env_vars = {}
    
    # 参数映射：训练参数名 -> 环境变量名
    param_mapping = {
        # 基础配置
        'exp_name': 'EXP_NAME',
        'seed': 'SEED',
        'train_episodes': 'TRAIN_EPISODES',
        'batch_size': 'BATCH_SIZE',
        'buffer_size': 'BUFFER_SIZE',
        
        # 算法配置
        'algorithm': 'ALGORITHM',
        'use_weighted_reward': 'USE_WEIGHTED_REWARD',
        
        # 势场配置
        'use_tf_potential_field': 'USE_TF_POTENTIAL_FIELD',
        'action_force_ratio': 'ACTION_FORCE_RATIO',
        
        # 地形配置
        'terrain_complexity_level': 'TERRAIN_COMPLEXITY_LEVEL',
        'map_size': 'MAP_SIZE',
        'mountain_min_distance': 'MOUNTAIN_MIN_DISTANCE',
        'scenario_seed': 'SCENARIO_SEED',
        'use_scenario_seed': 'USE_SCENARIO_SEED',
        
        # 位置配置
        'use_fixed_positions': 'USE_FIXED_POSITIONS',
        'positions_file': 'POSITIONS_FILE',
        
        # 课程学习配置
        'unlock_env_on_success': 'UNLOCK_ENV_ON_SUCCESS',
        'unlock_env_on_plateau': 'UNLOCK_ENV_ON_PLATEAU',
        'random_terrain': 'RANDOM_TERRAIN',
        'per_env_terrain': 'PER_ENV_TERRAIN',
        'per_episode_terrain': 'PER_EPISODE_TERRAIN',
        
        # 训练配置
        'per_enabled': 'PER_ENABLED',
        'noise_scale': 'NOISE_SCALE',
        'random_action_prob': 'RANDOM_ACTION_PROB',
        'random_action_prob_training': 'RANDOM_ACTION_PROB_TRAINING',
        
        # 势场参数
        'goal_attraction': 'GOAL_ATTRACTION',
        'lambda_1_base': 'LAMBDA_1_BASE',
        'terrain_repulsion': 'TERRAIN_REPULSION',
        'agent_influence_range': 'AGENT_INFLUENCE_RANGE',
        'delta_k_att': 'DELTA_K_ATT',
        'delta_lambda_1': 'DELTA_LAMBDA_1',
        'delta_k_rep': 'DELTA_K_REP',
        'delta_radius': 'DELTA_RADIUS',
        
        # 其他配置
        'num_envs': 'NUM_ENVS',
        'xla_global': 'XLA_GLOBAL',
        'cpu_threads': 'CPU_THREADS',
    }
    
    for param_key, env_key in param_mapping.items():
        if param_key in args:
            value = args[param_key]
            # 转换为字符串
            if isinstance(value, bool):
                env_vars[env_key] = "1" if value else "0"
            elif value is not None:
                env_vars[env_key] = str(value)
    
    return env_vars

def create_ablation_configs(base_env_vars: Dict[str, str], 
                            base_args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """基于基础配置创建消融实验配置（符合 ablation_action_pf_comparison.py 格式）"""
    ablation_configs = []
    
    # 获取基础配置值
    seed = base_env_vars.get("SEED", "1337")
    tf_deterministic = base_env_vars.get("TF_DETERMINISTIC_OPS", "1")
    
    # 获取势场参数（如果存在）
    goal_attraction = base_env_vars.get("GOAL_ATTRACTION", "6.0")
    lambda_1_base = base_env_vars.get("LAMBDA_1_BASE", "8.5")
    terrain_repulsion = base_env_vars.get("TERRAIN_REPULSION", "8000.0")
    agent_influence_range = base_env_vars.get("AGENT_INFLUENCE_RANGE", "150.0")
    delta_k_att = base_env_vars.get("DELTA_K_ATT", "10.0")
    delta_lambda_1 = base_env_vars.get("DELTA_LAMBDA_1", "5.0")
    delta_k_rep = base_env_vars.get("DELTA_K_REP", "4000.0")
    delta_radius = base_env_vars.get("DELTA_RADIUS", "120.0")
    
    # 获取FR schedule（如果存在）
    fr_schedule = base_env_vars.get("ACTION_FORCE_RATIO_SCHEDULE_PCT", None)
    if fr_schedule and fr_schedule.upper() not in ["DISABLED", ""]:
        # 保留原始schedule
        pass
    else:
        fr_schedule = None
    
    # 1. Action Only（纯网络动作，无势场修正）
    ablation_configs.append({
        "label": "action_only_reproduced",
        "name": "Action Only (Reproduced)",
        "name_en": "Action Only (Reproduced)",
        "description": f"Network action only, no APF correction (reproduced from {base_args.get('exp_name', 'base experiment')})",
        "env": {
            **{k: v for k, v in base_env_vars.items() if k not in [
                "ACTION_FORCE_RATIO", "ACTION_FORCE_RATIO_SCHEDULE_PCT", 
                "USE_TF_POTENTIAL_FIELD", "DELTA_K_ATT", "DELTA_LAMBDA_1", 
                "DELTA_K_REP", "DELTA_RADIUS"
            ]},
            "ACTION_FORCE_RATIO": "0.0",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",
            "USE_TF_POTENTIAL_FIELD": "1",
            "SEED": seed,
            "TF_DETERMINISTIC_OPS": tf_deterministic,
        }
    })
    
    # 2. APF Traditional（传统固定参数APF）
    ablation_configs.append({
        "label": "apf_traditional_reproduced",
        "name": "APF Traditional (Reproduced)",
        "name_en": "APF Traditional (Reproduced)",
        "description": f"APF action only, fixed parameters (reproduced from {base_args.get('exp_name', 'base experiment')})",
        "env": {
            **{k: v for k, v in base_env_vars.items() if k not in [
                "ACTION_FORCE_RATIO", "ACTION_FORCE_RATIO_SCHEDULE_PCT",
                "DELTA_K_ATT", "DELTA_LAMBDA_1", "DELTA_K_REP", "DELTA_RADIUS"
            ]},
            "ACTION_FORCE_RATIO": "1.0",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",
            "USE_TF_POTENTIAL_FIELD": "1",
            "DELTA_K_ATT": "0.0",
            "DELTA_LAMBDA_1": "0.0",
            "DELTA_K_REP": "0.0",
            "DELTA_RADIUS": "0.0",
            "GOAL_ATTRACTION": goal_attraction,
            "LAMBDA_1_BASE": lambda_1_base,
            "TERRAIN_REPULSION": terrain_repulsion,
            "AGENT_INFLUENCE_RANGE": agent_influence_range,
            "SEED": seed,
            "TF_DETERMINISTIC_OPS": tf_deterministic,
        }
    })
    
    # 3. APF Learnable（可学习APF）
    ablation_configs.append({
        "label": "apf_learnable_reproduced",
        "name": "APF Learnable (Reproduced)",
        "name_en": "APF Learnable (Reproduced)",
        "description": f"APF action only, parameters learned (reproduced from {base_args.get('exp_name', 'base experiment')})",
        "env": {
            **{k: v for k, v in base_env_vars.items() if k not in [
                "ACTION_FORCE_RATIO", "ACTION_FORCE_RATIO_SCHEDULE_PCT"
            ]},
            "ACTION_FORCE_RATIO": "1.0",
            "ACTION_FORCE_RATIO_SCHEDULE_PCT": "DISABLED",
            "USE_TF_POTENTIAL_FIELD": "1",
            "DELTA_K_ATT": delta_k_att,
            "DELTA_LAMBDA_1": delta_lambda_1,
            "DELTA_K_REP": delta_k_rep,
            "DELTA_RADIUS": delta_radius,
            "GOAL_ATTRACTION": goal_attraction,
            "LAMBDA_1_BASE": lambda_1_base,
            "TERRAIN_REPULSION": terrain_repulsion,
            "AGENT_INFLUENCE_RANGE": agent_influence_range,
            "SEED": seed,
            "TF_DETERMINISTIC_OPS": tf_deterministic,
        }
    })
    
    # 4. Action+APF Fusion（动作+APF融合）
    fusion_env = {
        **{k: v for k, v in base_env_vars.items() if k not in [
            "ACTION_FORCE_RATIO_SCHEDULE_PCT"
        ]},
        "USE_TF_POTENTIAL_FIELD": "1",
        "DELTA_K_ATT": delta_k_att,
        "DELTA_LAMBDA_1": delta_lambda_1,
        "DELTA_K_REP": delta_k_rep,
        "DELTA_RADIUS": delta_radius,
        "GOAL_ATTRACTION": goal_attraction,
        "LAMBDA_1_BASE": lambda_1_base,
        "TERRAIN_REPULSION": terrain_repulsion,
        "AGENT_INFLUENCE_RANGE": agent_influence_range,
        "SEED": seed,
        "TF_DETERMINISTIC_OPS": tf_deterministic,
    }
    
    # 如果原始配置有FR schedule，使用它；否则使用默认schedule
    if fr_schedule:
        fusion_env["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = fr_schedule
    else:
        # 使用默认schedule
        fusion_env["ACTION_FORCE_RATIO_SCHEDULE_PCT"] = "0%:0.50,10%:0.45,20%:0.40,40%:0.35,60%:0.30,80%:0.25,100%:0.20"
    
    ablation_configs.append({
        "label": "action_apf_fusion_reproduced",
        "name": "Action+APF Fusion (Reproduced)",
        "name_en": "Action+APF Fusion (Reproduced)",
        "description": f"Network action + learnable APF correction (reproduced from {base_args.get('exp_name', 'base experiment')})",
        "env": fusion_env
    })
    
    return ablation_configs

def save_reproduction_config(results_dir: Path, 
                             base_env_vars: Dict[str, str],
                             base_args: Dict[str, Any],
                             ablation_configs: List[Dict[str, Any]],
                             output_dir: Path):
    """保存复现配置到文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存基础配置
    base_config = {
        "source_results_dir": str(results_dir),
        "source_exp_name": base_args.get('exp_name', 'unknown'),
        "base_args": base_args,
        "base_env_vars": base_env_vars,
        "timestamp": results_dir.name if results_dir.name.startswith("exp_") else None,
        "extraction_info": {
            "seed": base_env_vars.get("SEED", "N/A"),
            "scenario_seed": base_env_vars.get("SCENARIO_SEED", "N/A"),
            "positions_file": base_env_vars.get("POSITIONS_FILE", "N/A"),
            "algorithm": base_args.get('algorithm', 'N/A'),
            "train_episodes": base_args.get('train_episodes', 'N/A'),
            "batch_size": base_args.get('batch_size', 'N/A'),
        }
    }
    
    with open(output_dir / "base_config.json", 'w', encoding='utf-8') as f:
        json.dump(base_config, f, indent=2, ensure_ascii=False)
    
    # 保存消融实验配置
    ablation_config_file = output_dir / "ablation_configs.json"
    with open(ablation_config_file, 'w', encoding='utf-8') as f:
        json.dump(ablation_configs, f, indent=2, ensure_ascii=False)
    
    # 保存使用说明
    readme_content = f"""# 复现实验配置

## 来源
- 原始实验目录: {results_dir}
- 实验名称: {base_args.get('exp_name', 'unknown')}
- 提取时间: {Path(__file__).stat().st_mtime if Path(__file__).exists() else 'N/A'}

## 关键配置
- 训练随机种子 (SEED): {base_env_vars.get("SEED", "N/A")}
- 地形种子 (SCENARIO_SEED): {base_env_vars.get("SCENARIO_SEED", "N/A")}
- 固定位置文件: {base_env_vars.get("POSITIONS_FILE", "N/A")}
- 算法: {base_args.get('algorithm', 'N/A')}
- 训练回合数: {base_args.get('train_episodes', 'N/A')}
- 批次大小: {base_args.get('batch_size', 'N/A')}

## 使用方法

### 方法1：修改消融实验脚本
在 `ablation_action_pf_comparison.py` 中，在 `EXPERIMENT_CONFIGS` 定义之后添加：

```python
# 从复现配置加载
import json
from pathlib import Path

reproduction_config_file = Path("{ablation_config_file}")
if reproduction_config_file.exists():
    with open(reproduction_config_file, 'r', encoding='utf-8') as f:
        reproduced_configs = json.load(f)
    # 替换或合并到 EXPERIMENT_CONFIGS
    EXPERIMENT_CONFIGS = reproduced_configs
```

### 方法2：使用命令行参数（如果支持）
```bash
python ablation_action_pf_comparison.py --config-file {ablation_config_file}
```

## 配置文件
- 基础配置: `base_config.json`
- 消融实验配置: `ablation_configs.json`
"""
    
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ 配置已保存到: {output_dir}")
    print(f"  - 基础配置: {output_dir / 'base_config.json'}")
    print(f"  - 消融实验配置: {ablation_config_file}")
    print(f"  - 使用说明: {output_dir / 'README.md'}")

def main():
    parser = argparse.ArgumentParser(
        description="从结果文件复现实验并生成消融实验配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从结果目录生成消融实验配置
  python reproduce_from_results.py models/exp_20260109_220551
  
  # 指定输出目录
  python reproduce_from_results.py models/exp_20260109_220551 --output-dir reproduction_configs
  
  # 只生成配置，不运行实验
  python reproduce_from_results.py models/exp_20260109_220551 --config-only
        """
    )
    
    parser.add_argument("results_dir", type=str, help="结果目录路径（包含 results.json）")
    parser.add_argument("--config-only", action="store_true", help="只生成配置，不运行实验")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认：reproduction_<timestamp>）")
    
    args = parser.parse_args()
    
    # 解析结果目录
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ 错误：结果目录不存在: {results_dir}")
        sys.exit(1)
    
    # 加载结果
    print(f"📂 加载结果: {results_dir}")
    try:
        results = load_results(results_dir)
    except Exception as e:
        print(f"❌ 错误：无法加载结果文件: {e}")
        sys.exit(1)
    
    # 提取配置
    print("🔧 提取实验配置...")
    base_args = results.get("args", {})
    if not base_args:
        print("❌ 错误：results.json 中未找到 'args' 字段")
        sys.exit(1)
    
    base_env_vars = extract_env_vars_from_args(base_args)
    
    # 显示关键配置
    print("\n📊 关键配置:")
    print(f"  实验名称: {base_args.get('exp_name', 'N/A')}")
    print(f"  算法: {base_args.get('algorithm', 'N/A')}")
    print(f"  训练回合数: {base_args.get('train_episodes', 'N/A')}")
    print(f"  批次大小: {base_args.get('batch_size', 'N/A')}")
    print(f"  随机种子: {base_args.get('seed', 'N/A')}")
    print(f"  地形种子: {base_args.get('scenario_seed', 'N/A')}")
    print(f"  固定位置文件: {base_args.get('positions_file', 'N/A')}")
    print(f"  势场使用: {base_env_vars.get('USE_TF_POTENTIAL_FIELD', 'N/A')}")
    print(f"  动作力比例: {base_env_vars.get('ACTION_FORCE_RATIO', 'N/A')}")
    
    # 创建消融实验配置
    print("\n🔬 创建消融实验配置...")
    ablation_configs = create_ablation_configs(base_env_vars, base_args)
    print(f"  ✅ 生成了 {len(ablation_configs)} 个消融实验配置:")
    for cfg in ablation_configs:
        print(f"    - {cfg['label']}: {cfg['name']}")
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = results_dir.name if results_dir.name.startswith("exp_") else "reproduction"
        output_dir = Path(f"reproduction_{timestamp}")
    
    # 保存配置
    save_reproduction_config(results_dir, base_env_vars, base_args, ablation_configs, output_dir)
    
    if args.config_only:
        print("\n✅ 配置生成完成（未运行实验）")
        print(f"\n📝 下一步：")
        print(f"   1. 查看生成的配置: {output_dir / 'ablation_configs.json'}")
        print(f"   2. 修改 ablation_action_pf_comparison.py 加载配置")
        print(f"   3. 运行消融实验")
        return
    
    print("\n📝 下一步：")
    print(f"   1. 查看生成的配置: {output_dir / 'ablation_configs.json'}")
    print(f"   2. 使用 ablation_action_pf_comparison.py 运行消融实验")
    print(f"   3. 将生成的配置作为输入")

if __name__ == "__main__":
    main()

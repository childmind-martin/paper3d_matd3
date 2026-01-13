#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验数据管理改进版本
使用统一批次目录结构，确保数据一致性和可追溯性
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class AblationBatchManager:
    """
    消融实验批次管理器
    
    功能：
    1. 创建统一的批次目录
    2. 管理批次内的实验数据
    3. 提供数据查找和复现支持
    """
    
    def __init__(self, root_dir: str = "ablation_experiments"):
        """
        初始化批次管理器
        
        Args:
            root_dir: 批次根目录（默认：ablation_experiments）
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def create_batch(self, batch_id: Optional[str] = None, config: Optional[Dict] = None) -> Path:
        """
        创建新的批次目录
        
        Args:
            batch_id: 批次ID（默认：batch_YYYYMMDD_HHMMSS）
            config: 批次配置信息（实验参数、种子等）
        
        Returns:
            批次目录路径
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        batch_dir = self.root_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建实验子目录
        experiments = ["action_only", "apf_traditional", "apf_learnable", "action_apf_fusion"]
        for exp in experiments:
            (batch_dir / exp).mkdir(exist_ok=True)
        
        # 创建图表输出目录
        (batch_dir / "plots").mkdir(exist_ok=True)
        
        # 保存配置信息
        if config is None:
            config = {
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(),
                "experiments": experiments
            }
        
        config_file = batch_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 更新 latest 符号链接
        latest_link = self.root_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        # 创建符号链接（Windows可能需要管理员权限，失败时静默忽略）
        try:
            latest_link.symlink_to(batch_dir.name, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows上可能失败，创建一个指示文件
            with open(self.root_dir / "latest.txt", 'w') as f:
                f.write(batch_id)
        
        print(f"✅ 创建批次目录: {batch_dir}")
        return batch_dir
    
    def get_batch_dir(self, batch_id: Optional[str] = None) -> Path:
        """
        获取批次目录
        
        Args:
            batch_id: 批次ID（None表示使用最新批次）
        
        Returns:
            批次目录路径
        """
        if batch_id is None:
            # 使用最新批次
            latest_link = self.root_dir / "latest"
            if latest_link.is_symlink():
                return latest_link.resolve()
            elif (self.root_dir / "latest.txt").exists():
                with open(self.root_dir / "latest.txt", 'r') as f:
                    batch_id = f.read().strip()
            else:
                # 查找最新的批次目录
                batch_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
                if not batch_dirs:
                    raise FileNotFoundError(f"未找到任何批次目录: {self.root_dir}")
                return batch_dirs[-1]
        
        batch_dir = self.root_dir / batch_id
        if not batch_dir.exists():
            raise FileNotFoundError(f"批次目录不存在: {batch_dir}")
        
        return batch_dir
    
    def get_experiment_dir(self, experiment_name: str, batch_id: Optional[str] = None) -> Path:
        """
        获取指定实验的目录
        
        Args:
            experiment_name: 实验名称（action_only, apf_traditional等）
            batch_id: 批次ID（None表示使用最新批次）
        
        Returns:
            实验目录路径
        """
        batch_dir = self.get_batch_dir(batch_id)
        exp_dir = batch_dir / experiment_name
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
        
        return exp_dir
    
    def save_experiment_data(self, 
                            experiment_name: str, 
                            data: Dict, 
                            batch_id: Optional[str] = None,
                            filename: str = "episode_rewards.json"):
        """
        保存实验数据到批次目录
        
        Args:
            experiment_name: 实验名称
            data: 要保存的数据
            batch_id: 批次ID（None表示使用最新批次）
            filename: 文件名
        """
        exp_dir = self.get_experiment_dir(experiment_name, batch_id)
        output_file = exp_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 保存数据: {output_file.relative_to(self.root_dir)}")
    
    def load_experiment_data(self, 
                            experiment_name: str, 
                            batch_id: Optional[str] = None,
                            filename: str = "episode_rewards.json") -> Dict:
        """
        从批次目录加载实验数据（单个文件）
        
        Args:
            experiment_name: 实验名称
            batch_id: 批次ID（None表示使用最新批次）
            filename: 文件名
        
        Returns:
            加载的数据
        """
        exp_dir = self.get_experiment_dir(experiment_name, batch_id)
        data_file = exp_dir / filename
        
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_all_experiment_metrics(self, 
                                    experiment_name: str, 
                                    batch_id: Optional[str] = None) -> Dict:
        """
        加载实验的所有指标数据（episode_rewards.json, loss_history.json, results.json）
        
        Args:
            experiment_name: 实验名称
            batch_id: 批次ID（None表示使用最新批次）
        
        Returns:
            包含所有指标的字典
        """
        exp_dir = self.get_experiment_dir(experiment_name, batch_id)
        
        metrics = {
            "episode_rewards": [],
            "success_flags": [],
            "collision_counts": [],
            "min_distances_to_obstacle": [],
            "agent_success_flags": [],
            "team_success_flags": [],
            "agent_success_rates": [],
            "team_success_rate": 0.0,
            "loss_history": []
        }
        
        # 加载 episode_rewards.json
        episode_rewards_file = exp_dir / "episode_rewards.json"
        if episode_rewards_file.exists():
            with open(episode_rewards_file, 'r', encoding='utf-8') as f:
                reward_data = json.load(f)
                metrics["episode_rewards"] = reward_data.get("episode_rewards", [])
                metrics["success_flags"] = reward_data.get("success_flags", [])
                metrics["collision_counts"] = reward_data.get("collision_counts", [])
                metrics["min_distances_to_obstacle"] = reward_data.get("min_distances_to_obstacle", [])
                metrics["agent_success_flags"] = reward_data.get("agent_success_flags", [])
                metrics["team_success_flags"] = reward_data.get("team_success_flags", [])
        
        # 加载 loss_history.json
        loss_history_file = exp_dir / "loss_history.json"
        if loss_history_file.exists():
            with open(loss_history_file, 'r', encoding='utf-8') as f:
                metrics["loss_history"] = json.load(f)
        
        # 加载 results.json（如果有）
        results_file = exp_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                metrics["agent_success_rates"] = results.get("agent_success_rates", [])
                metrics["team_success_rate"] = results.get("team_success_rate", 0.0)
        
        return metrics
    
    def list_batches(self) -> List[Dict]:
        """
        列出所有批次
        
        Returns:
            批次列表（包含ID、创建时间、实验列表等信息）
        """
        batches = []
        
        for batch_dir in sorted(self.root_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
                continue
            
            config_file = batch_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {
                    "batch_id": batch_dir.name,
                    "created_at": "unknown"
                }
            
            # 检查实验完成情况
            experiments = {}
            for exp_name in ["action_only", "apf_traditional", "apf_learnable", "action_apf_fusion"]:
                exp_dir = batch_dir / exp_name
                if exp_dir.exists():
                    data_file = exp_dir / "episode_rewards.json"
                    experiments[exp_name] = {
                        "exists": data_file.exists(),
                        "path": str(exp_dir)
                    }
            
            batches.append({
                "batch_id": batch_dir.name,
                "path": str(batch_dir),
                "config": config,
                "experiments": experiments
            })
        
        return batches
    
    def copy_from_logs(self, 
                      log_paths: Dict[str, str], 
                      batch_id: Optional[str] = None):
        """
        从旧的logs目录复制数据到批次目录
        
        Args:
            log_paths: 实验名称到日志路径的映射
                      例如: {"action_only": "logs/action_only_20251227_133708/20251227_133712"}
            batch_id: 目标批次ID（None表示创建新批次）
        """
        if batch_id is None:
            batch_dir = self.create_batch()
        else:
            batch_dir = self.get_batch_dir(batch_id)
        
        print(f"\n复制数据到批次目录: {batch_dir.name}")
        
        for exp_name, log_path in log_paths.items():
            src_dir = Path(log_path)
            dst_dir = batch_dir / exp_name
            
            if not src_dir.exists():
                print(f"  ⚠️  源目录不存在: {src_dir}")
                continue
            
            # 复制关键文件
            files_to_copy = [
                "episode_rewards.json",
                "loss_history.json",
                "results.json",
                "training.log"
            ]
            
            for filename in files_to_copy:
                src_file = src_dir / filename
                if src_file.exists():
                    dst_file = dst_dir / filename
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✓ {exp_name}/{filename}")
            
            # 可选：复制轨迹图（如果需要）
            # for trajectory_dir in src_dir.glob("episode_*"):
            #     if trajectory_dir.is_dir():
            #         dst_traj = dst_dir / trajectory_dir.name
            #         shutil.copytree(trajectory_dir, dst_traj, dirs_exist_ok=True)
        
        print(f"✅ 数据复制完成")


def migrate_existing_data():
    """
    迁移现有的消融实验数据到新的批次结构
    
    这个函数会：
    1. 查找logs目录中最新的4个实验
    2. 创建新的批次目录
    3. 复制数据到批次目录
    """
    manager = AblationBatchManager()
    
    print("="*70)
    print("迁移现有消融实验数据到批次结构")
    print("="*70)
    
    # 查找最新的4个实验
    logs_root = Path("logs")
    if not logs_root.exists():
        print("❌ logs目录不存在")
        return
    
    experiments = {
        "action_only": None,
        "apf_traditional": None,
        "apf_learnable": None,
        "action_apf_fusion": None
    }
    
    for exp_name in experiments.keys():
        # 查找最新的实验目录（精确匹配，避免匹配到包含该名称的其他实验）
        matching_dirs = []
        for item in logs_root.iterdir():
            if item.is_dir() and item.name.startswith(exp_name + "_"):
                # 🔧 关键修复：精确匹配，避免 action_apf_fusion 匹配到 action_apf_fusion_aux_per
                # 检查是否紧跟时间戳格式（YYYYMMDD_HHMMSS）
                suffix = item.name[len(exp_name) + 1:]
                if len(suffix) >= 15 and suffix[8] == '_' and suffix[:8].isdigit() and suffix[9:15].isdigit():
                    # 查找子目录
                    subdirs = sorted([d for d in item.iterdir() if d.is_dir() and d.name != 'evaluation'])
                    if subdirs:
                        matching_dirs.append((item.name, subdirs[-1]))
        
        if matching_dirs:
            matching_dirs.sort(key=lambda x: x[0], reverse=True)
            experiments[exp_name] = str(matching_dirs[0][1])
            print(f"  ✓ {exp_name}: {matching_dirs[0][1]}")
        else:
            print(f"  ⚠️  {exp_name}: 未找到")
    
    # 检查是否所有实验都找到了
    missing = [k for k, v in experiments.items() if v is None]
    if missing:
        print(f"\n⚠️  以下实验未找到数据: {', '.join(missing)}")
        print("是否继续迁移已找到的实验？(y/n)")
        # 注意：在脚本中自动处理，不需要用户交互
        # 在实际使用时可以添加交互
    
    # 创建新批次并复制数据
    log_paths = {k: v for k, v in experiments.items() if v is not None}
    manager.copy_from_logs(log_paths)
    
    print("\n" + "="*70)
    print("✅ 迁移完成")
    print("="*70)


if __name__ == "__main__":
    # 示例：迁移现有数据
    migrate_existing_data()
    
    # 示例：列出所有批次
    manager = AblationBatchManager()
    print("\n所有批次:")
    for batch in manager.list_batches():
        print(f"  - {batch['batch_id']}")
        print(f"    创建时间: {batch['config'].get('created_at', 'unknown')}")
        print(f"    实验数量: {len([e for e in batch['experiments'].values() if e['exists']])}/4")

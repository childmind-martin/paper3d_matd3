#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自适应奖励加权（ARW, Adaptive Reward Weighting）

设计目标：
- 在不侵入训练主循环的前提下，提供按回合演进的奖励权重自适应能力；
- 若上层愿意提供成功率/损失等指标，也提供更新接口以改进自适应效果；
- 与现有 weighted/vectorized 场景即插即用：
  * weighted 场景在 reward() 内调用 apply_to_rewards 调整部分项；
  * vectorized 场景可在每回合开始时按比例调整 reward_weights 向量。
"""

from __future__ import annotations

import math
from typing import Dict, Optional


class AdaptiveRewardWeighting:
    """自适应奖励加权器。

    参数约定：
    - base_penalty/base_c：碰撞惩罚系数与距离类系数的“基准值”；
    - alpha/beta：随训练进度与成功率的调节强度；
    - max_episodes/warmup_episodes：线性/余量调度的时间尺度；
    - enable_adaptive：关闭时退化为恒等（不做调整）。
    """

    def __init__(
        self,
        base_penalty: float = -10.0,
        alpha: float = 0.5,
        base_c: float = 10.0,
        beta: float = 0.3,
        max_episodes: int = 20000,
        warmup_episodes: int = 200,
        enable_adaptive: bool = True,
    ) -> None:
        self.base_penalty = float(base_penalty)
        self.alpha = float(alpha)
        self.base_c = float(base_c)
        self.beta = float(beta)
        self.max_episodes = int(max_episodes)
        self.warmup_episodes = int(warmup_episodes)
        self.enable = bool(enable_adaptive)

        # 运行时状态
        self.episode_idx: int = 0
        self.success_rate: float = 0.0
        self._success_hist: list[float] = []
        self._critic_loss_hist: list[float] = []

        # 当前缩放系数（>0）
        self._collision_scale: float = 1.0
        self._distance_scale: float = 1.0

    # ------------------------ 外部可选输入（改进自适应效果） ------------------------
    def update_success_rate(self, success_flag: bool, window_size: int = 100) -> None:
        """更新最近窗口的成功率估计。"""
        self._success_hist.append(1.0 if success_flag else 0.0)
        if len(self._success_hist) > max(1, int(window_size)):
            self._success_hist.pop(0)
        if self._success_hist:
            self.success_rate = sum(self._success_hist) / float(len(self._success_hist))

    def update_critic_loss(self, loss_value: float, window_size: int = 50) -> None:
        """记录 Critic 损失（用于检测波动并平滑距离项）。"""
        try:
            lv = float(loss_value)
        except Exception:
            return
        self._critic_loss_hist.append(lv)
        if len(self._critic_loss_hist) > max(1, int(window_size)):
            self._critic_loss_hist.pop(0)

    # ------------------------ 回合起点：推进自适应状态 ------------------------
    def on_episode_start(self, episode_idx: Optional[int] = None) -> None:
        """在每回合开始时调用，用于推进内部调度与系数计算。"""
        if episode_idx is None:
            self.episode_idx += 1
        else:
            self.episode_idx = int(episode_idx)

        if not self.enable:
            self._collision_scale = 1.0
            self._distance_scale = 1.0
            return

        # 预热阶段：不做自适应，保持基准
        if self.episode_idx < self.warmup_episodes:
            self._collision_scale = 1.0
            self._distance_scale = 1.0
            return

        # 进度比例（0→1）
        progress = min(1.0, max(0.0, float(self.episode_idx) / max(1, self.max_episodes)))

        # ① 碰撞惩罚增强：随训练进度线性增强
        #    scale_penalty = 1 + alpha * progress
        self._collision_scale = max(0.0, 1.0 + self.alpha * progress)

        # ② 距离类奖励增强：与成功率负相关，成功率低→增强距离信号
        #    scale_distance = 1 + beta * (1 - success_rate)
        self._distance_scale = max(0.0, 1.0 + self.beta * (1.0 - float(self.success_rate)))

        # ③ Critic 波动保护：若波动较大，适度降低距离权重以平滑训练
        if len(self._critic_loss_hist) >= 10:
            try:
                import numpy as _np
                std = float(_np.std(self._critic_loss_hist))
            except Exception:
                std = 0.0
            if std > 0.5:  # 经验阈值，可按需调参
                self._distance_scale *= 0.8

        # 最终钳制，避免极端值
        self._collision_scale = min(self._collision_scale, 3.0)
        self._distance_scale = min(self._distance_scale, 2.5)

    # ------------------------ 应用于分项（weighted 场景） ------------------------
    def apply_to_rewards(self, reward_terms: Dict[str, float]) -> Dict[str, float]:
        """对局部分项进行缩放（不改变键集合）。

        使用场景：paper3d_terrain_weighted.reward() 内对 'collision' 与 'distance'（含 approach）进行调整。
        """
        if not self.enable:
            return reward_terms

        out = dict(reward_terms)
        if 'collision' in out:
            out['collision'] *= float(self._collision_scale)
        if 'distance' in out:
            out['distance'] *= float(self._distance_scale)
        return out

    # ------------------------ 读数/调试 ------------------------
    @property
    def collision_scale(self) -> float:
        return float(self._collision_scale)

    @property
    def distance_scale(self) -> float:
        return float(self._distance_scale)

    def get_current_scales(self) -> Dict[str, float]:
        return {
            'collision_scale': self.collision_scale,
            'distance_scale': self.distance_scale,
            'episode': int(self.episode_idx),
            'success_rate': float(self.success_rate),
        }
    
    def get_current_weights(self) -> Dict[str, float]:
        """返回当前的权重（与get_current_scales相同，兼容场景文件调用）"""
        return self.get_current_scales()


# Critic发散修复 - 快速参考卡片
## 2025-11-30

---

## 🎯 核心问题
- **Critic Loss发散**：从200持续上升到2200+
- **Actor Loss异常**：≈0（说明Q值完全不准）
- **策略崩溃**：Episode 104最佳，Episode 120退化

---

## ✅ 已修复内容

### 配置参数（run_optimized.sh）
```bash
LEARNING_RATE_CRITIC: 0.0002 → 0.0001  # 降低50%
HUBER_DELTA: 8.0 → 5.0                 # 对TD误差更敏感
TAU: 0.012 → 0.005                     # 稳定目标网络
GRAD_CLIP_NORM: 10.0 → 5.0             # 更强梯度裁剪
```

### 代码增强（paper3d_train_optimized.py）
- **8处**增加两级梯度裁剪
- 逐层裁剪：每层范数<1.0
- 全局裁剪：总范数<5.0

---

## 📊 预期效果

| 指标 | 修复前 | 修复后目标 |
|------|--------|------------|
| Critic Loss | 200→2200+ | 100-500稳定 |
| Actor Loss | ≈0 | -50到-200 |
| 训练稳定性 | 104后崩溃 | 持续改进 |
| 成功率 | <10% | >20% |

---

## 🚀 启动训练

```bash
cd /home/tang/Desktop
./run_optimized.sh
```

---

## 🔍 监控要点

### ✅ 成功标志
- Critic Loss < 500
- Actor Loss在-50到-200之间
- 奖励曲线平稳上升
- Actor输出保持平滑

### ❌ 失败标志
- Critic Loss > 1000
- Actor Loss接近0或NaN
- 奖励曲线突然暴跌
- Actor输出剧烈震荡

---

## 🔧 如果仍发散

### 进一步降低参数
```bash
LEARNING_RATE_CRITIC=0.00005  # 再降50%
HUBER_DELTA=3.0               # 再降40%
```

### 检查奖励设计
- 探索奖励5.0 → 0.5
- 目标奖励100 → 500
- 详见：REWARD_TUNING_20251129.md

---

## 📚 完整文档

- `CRITIC_FIX_SUMMARY_20251130.md`：完整修复总结
- `CRITIC_DIVERGENCE_FIX_20251130.md`：技术方案详解
- `REWARD_TUNING_20251129.md`：奖励设计调优

---

**修复完成时间**：2025-11-30 21:00  
**修改文件数**：2  
**修改位置数**：12


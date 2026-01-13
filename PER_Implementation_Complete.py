"""
Enhanced Prioritized Experience Replay (PER) Implementation
用于论文描述的完整代码实现

核心创新点：
1. 多信号优先级计算（TD误差 + 奖励幅值 + 年龄衰减）
2. PER与均匀采样的混合分布
3. 向量化批量采样优化
4. 数值稳定性保护机制
5. 迭代式SumTree实现（避免递归开销）
"""

import numpy as np


# ============================================================================
# 1. SumTree数据结构（高效优先级存储与检索）
# ============================================================================

class SumTree:
    """
    高效的SumTree实现，用于O(log n)复杂度的优先级更新和采样
    
    结构：
    - 完全二叉树，叶子节点存储优先级值
    - 内部节点存储子节点优先级之和
    - 总节点数 = 2 * capacity - 1
    """
    
    def __init__(self, capacity):
        self.capacity = int(capacity)
        # 树的内部节点数量 = capacity - 1
        # 总大小 = 内部节点 + 叶子节点 = 2*capacity - 1
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        
        # 性能优化：缓存常用值，避免重复计算
        self.leaf_start = self.capacity - 1
        self.max_depth = int(np.ceil(np.log2(max(2, self.capacity)))) + 2

    def _propagate(self, idx, change):
        """
        从叶子节点向上更新父节点（迭代版本，避免递归开销）
        
        Args:
            idx: 叶子节点索引
            change: 优先级变化量
        """
        # 性能优化：快速检查
        if not np.isfinite(change) or change == 0:
            return
        
        # 迭代实现，避免递归开销和栈溢出风险
        current_idx = idx
        while current_idx > 0:  # 当 current_idx == 0 时，已经是根节点，停止
            parent = (current_idx - 1) // 2
            self.tree[parent] += change
            
            # 数值安全：检查父节点值是否有效
            if not np.isfinite(self.tree[parent]):
                self.tree[parent] = 0.0
                break  # 如果父节点无效，停止传播
            
            current_idx = parent

    def _retrieve(self, idx, s):
        """
        根据采样值s查找叶子节点索引（迭代版本）
        
        Args:
            idx: 当前节点索引（从根节点0开始）
            s: 采样值（0到total之间）
            
        Returns:
            叶子节点索引
        """
        current_idx = idx
        current_s = s
        tree_size = len(self.tree)
        leaf_start = self.leaf_start
        
        while True:
            left = 2 * current_idx + 1
            right = left + 1
            
            # 边界检查，防止索引越界
            if left >= tree_size:
                return np.clip(current_idx, leaf_start, tree_size - 1)
            
            left = min(left, tree_size - 1)
            left_val = self.tree[left]
            
            if current_s <= left_val:
                current_idx = left
            else:
                current_idx = min(right, tree_size - 1)
                current_s -= left_val

    def total(self):
        """返回根节点值，即所有优先级的总和"""
        total_val = self.tree[0]
        # 数值安全：检查根节点是否有效
        if not np.isfinite(total_val):
            # 如果根节点是 NaN/Inf，尝试重建树
            print(f"[严重警告] SumTree根节点异常: {total_val}，尝试修复...")
            self._rebuild_tree()
            total_val = self.tree[0]
            if not np.isfinite(total_val):
                return 1e-8
        return total_val
    
    def _rebuild_tree(self):
        """重建整个树（从叶子节点向上）"""
        leaf_start = self.leaf_start
        tree_size = len(self.tree)
        
        # 先清理所有叶子节点的 NaN/Inf
        for i in range(leaf_start, tree_size):
            if not np.isfinite(self.tree[i]):
                self.tree[i] = 1.0
        # 重建内部节点
        for i in range(leaf_start - 1, -1, -1):
            left = 2 * i + 1
            right = left + 1
            left_val = self.tree[left] if left < tree_size else 0.0
            right_val = self.tree[right] if right < tree_size else 0.0
            self.tree[i] = left_val + right_val
            if not np.isfinite(self.tree[i]):
                self.tree[i] = 0.0

    def add(self, p):
        """在树中添加新的优先级，并更新数据指针"""
        idx = self.data_pointer + self.leaf_start
        self.update(idx, p)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, idx, p):
        """更新指定叶子节点的优先级"""
        p_safe = float(p)
        
        # 性能优化：快速路径（值正常时跳过完整检查）
        if 0 < p_safe < 1e10:
            old_value = self.tree[idx]
            if np.isfinite(old_value):
                change = np.float32(p_safe) - old_value
                self.tree[idx] = np.float32(p_safe)
                self._propagate(idx, change)
                return
            if not np.isfinite(old_value):
                old_value = 0.0
                self.tree[idx] = 0.0
        
        # 慢速路径：值可能异常，进行完整检查
        if not np.isfinite(p_safe):
            p_safe = 1.0
        
        old_value = self.tree[idx] if np.isfinite(self.tree[idx]) else 0.0
        if not np.isfinite(old_value):
            old_value = 0.0
            self.tree[idx] = 0.0
        
        change = np.float32(p_safe) - old_value
        self.tree[idx] = np.float32(p_safe)
        self._propagate(idx, change)

    def get(self, s):
        """
        根据采样值s获取 (叶子索引, 优先级, 缓冲区中的实际索引)
        
        Args:
            s: 采样值（0到total之间）
            
        Returns:
            (tree_idx, priority, data_idx)
        """
        idx = self._retrieve(0, s)
        tree_size = len(self.tree)
        idx = np.clip(idx, self.leaf_start, tree_size - 1)
        data_idx = idx - self.leaf_start
        return (idx, self.tree[idx], data_idx)

    def get_batch(self, s_batch):
        """
        批量检索叶子节点（向量化优化版本）
        
        保持与 get(s) 完全一致的检索逻辑，但一次性处理多个 s。
        
        Args:
            s_batch: 采样值数组
            
        Returns:
            (leaf_indices, priorities, data_indices)
        """
        s = np.asarray(s_batch, dtype=np.float64).copy()
        batch = s.shape[0]
        idx = np.zeros((batch,), dtype=np.int32)
        leaf_start = self.leaf_start
        tree_size = len(self.tree)
        
        # 使用缓存的 max_depth，避免重复计算
        for _ in range(self.max_depth):
            not_leaf_mask = idx < leaf_start
            if not not_leaf_mask.any():
                break
            
            # 向量化计算左右子节点
            left = 2 * idx + 1
            right = left + 1
            
            # 边界检查，防止索引越界
            valid_mask = not_leaf_mask & (left < tree_size)
            if not valid_mask.any():
                break
            
            # 只对有效的非叶子节点获取左子树值
            left_val = np.zeros(batch, dtype=np.float64)
            nl_idx = np.where(valid_mask)[0]
            if nl_idx.size > 0:
                left_idx = left[nl_idx]
                left_idx = np.clip(left_idx, 0, tree_size - 1)
                left_val[nl_idx] = self.tree[left_idx]
            
            # 向量化决策（走左还是右）
            go_right = valid_mask & (s > left_val)
            
            # 向量化更新
            s = np.where(go_right, s - left_val, s)
            right_clipped = np.clip(right, 0, tree_size - 1)
            left_clipped = np.clip(left, 0, tree_size - 1)
            idx = np.where(go_right, right_clipped, left_clipped)

        leaf_indices = np.clip(idx, leaf_start, tree_size - 1)
        priorities = self.tree[leaf_indices]
        data_indices = leaf_indices - leaf_start
        return (leaf_indices.astype(np.int32), priorities.astype(np.float32), data_indices.astype(np.int32))


# ============================================================================
# 2. 增强的优先级计算（多信号融合）
# ============================================================================

def compute_priority(td_error, reward_abs, age, 
                    priority_td_weight=1.0, 
                    priority_reward_weight=0.0, 
                    priority_age_decay=1.0,
                    negative_slope=0.6,
                    epsilon=0.01):
    """
    计算样本优先级（多信号融合）
    
    优先级公式：
        priority = (signal^negative_slope) × age_weight
        其中：
            signal = priority_td_weight × TD_error + priority_reward_weight × reward_abs
            age_weight = priority_age_decay^age
    
    Args:
        td_error: TD误差（绝对值）
        reward_abs: 奖励绝对值
        age: 样本年龄（插入步数差）
        priority_td_weight: TD误差权重（默认1.0）
        priority_reward_weight: 奖励幅值权重（默认0.0）
        priority_age_decay: 年龄衰减系数（默认1.0，即不衰减）
        negative_slope: 优先级幂次（默认0.6）
        epsilon: TD误差最小值（默认0.01）
        
    Returns:
        优先级值
    """
    # 1. TD误差处理
    td_error = np.abs(td_error) + epsilon
    MAX_TD_ERROR = 10000.0
    td_error = np.clip(td_error, epsilon, MAX_TD_ERROR)
    
    # 2. 奖励幅值限制
    reward_abs = min(reward_abs, 1000.0)
    
    # 3. 年龄权重计算
    age = max(0, min(age, 100000))
    if priority_age_decay < 1.0 and priority_age_decay > 0:
        age_weight = priority_age_decay ** age
        if not np.isfinite(age_weight) or age_weight <= 0:
            age_weight = 1.0
    else:
        age_weight = 1.0
    
    # 4. 组合信号
    signal = priority_td_weight * td_error + priority_reward_weight * reward_abs
    signal = np.clip(signal, 1e-6, MAX_TD_ERROR)
    
    # 5. 计算最终优先级
    try:
        if not np.isfinite(signal) or not np.isfinite(negative_slope) or not np.isfinite(age_weight):
            p = 1.0
        else:
            p = (signal ** negative_slope) * age_weight
            if not np.isfinite(p) or p <= 0:
                p = 1.0
    except (ValueError, OverflowError, FloatingPointError):
        p = 1.0
    
    # 6. 限制优先级范围
    MAX_PRIORITY_VALUE = 50000.0
    p = np.clip(p, 1e-8, MAX_PRIORITY_VALUE)
    
    if not np.isfinite(p) or p <= 0:
        p = 1.0
    
    return float(p)


# ============================================================================
# 3. PER采样方法（支持混合分布）
# ============================================================================

def per_sample(sum_tree, buffer_size, batch_size, 
               beta=0.4, 
               per_uniform_mix=0.0,
               per_replace=True):
    """
    优先经验回放采样（支持PER与均匀采样混合）
    
    采样策略：
    1. 纯PER采样（per_uniform_mix=0）：完全基于优先级的采样
    2. 混合采样（per_uniform_mix>0）：P_mix = (1-m)×P_per + m×P_uniform
    
    Args:
        sum_tree: SumTree实例
        buffer_size: 缓冲区当前大小
        batch_size: 批次大小
        beta: 重要性采样权重参数（0到1之间）
        per_uniform_mix: PER与均匀采样混合比例（0到1之间，0=纯PER，1=纯均匀）
        per_replace: 是否允许重复采样
        
    Returns:
        (indices, weights, tree_indices)
    """
    total_p = sum_tree.total()
    
    # 数值安全检查
    if not np.isfinite(total_p) or total_p <= 0:
        # 退化为均匀采样
        indices = np.random.randint(0, buffer_size, size=batch_size, dtype=np.int64)
        weights = np.ones((batch_size,), dtype=np.float32)
        return indices, weights, None
    
    if per_uniform_mix <= 1e-8:
        # ========== 纯PER采样 ==========
        indices = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # 分段采样（确保均匀覆盖优先级空间）
        segment = total_p / batch_size
        segment_starts = np.arange(batch_size, dtype=np.float64) * segment
        segment_rand = np.random.uniform(0, segment, size=batch_size)
        random_samples = segment_starts + segment_rand
        
        # 向量化 SumTree 检索
        _tidx, _p, _didx = sum_tree.get_batch(random_samples)
        tmp_indices = list(zip(_tidx.tolist(), _p.tolist(), _didx.tolist()))
        
        # 去重处理（如果per_replace=False）
        if not per_replace:
            seen = set()
            unique = []
            for (tidx, p, didx) in tmp_indices:
                if didx not in seen:
                    seen.add(didx)
                    unique.append((tidx, p, didx))
            while len(unique) < batch_size:
                unique.append(tmp_indices[len(unique) % len(tmp_indices)])
            tmp_indices = unique[:batch_size]
        
        # 计算重要性采样权重
        for i, (tree_idx, p, data_idx) in enumerate(tmp_indices):
            indices[i] = data_idx
            tree_indices[i] = tree_idx
            if p > 1e-8 and total_p > 1e-8:
                prob = np.clip(buffer_size * p / total_p, 1e-8, 1e8)
                weights[i] = prob ** (-beta)
            else:
                weights[i] = 1.0
        
        # 归一化权重
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        max_w = np.max(weights)
        if max_w > 1e-8:
            weights = weights / max_w
        else:
            weights = np.ones_like(weights)
        
        return indices, weights, tree_indices
    
    else:
        # ========== 混合采样：P_mix = (1-m)×P_per + m×P_uniform ==========
        m = float(per_uniform_mix)
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        chosen = set() if not per_replace else None
        
        # 计算PER和均匀采样的数量
        n_per = int(max(0, round((1.0 - m) * batch_size)))
        n_uni = batch_size - n_per
        
        # PER采样部分
        per_indices = []
        if n_per > 0:
            seg = total_p / n_per if n_per > 0 else 0.0
            seg_starts = np.arange(n_per, dtype=np.float64) * seg
            seg_rand = np.random.uniform(0, seg, size=n_per) if n_per > 0 else np.zeros((0,), dtype=np.float64)
            rs = seg_starts + seg_rand
            _tidx, _p, _didx = sum_tree.get_batch(rs)
            tmp = list(zip(_tidx.tolist(), _p.tolist(), _didx.tolist()))
            
            if not per_replace:
                seen = set()
                unique = []
                for (tidx, p, didx) in tmp:
                    if didx not in seen and (chosen is None or didx not in chosen):
                        seen.add(didx)
                        unique.append((tidx, p, didx))
                while len(unique) < n_per:
                    unique.append(tmp[len(unique) % len(tmp)])
                tmp = unique[:n_per]
                for _, _, didx in tmp:
                    chosen.add(int(didx))
            per_indices = tmp
        
        # 均匀采样部分
        uni_indices = []
        if n_uni > 0:
            if per_replace:
                rand_idx = np.random.randint(0, buffer_size, size=n_uni, dtype=np.int64)
            else:
                if chosen is not None and len(chosen) < buffer_size:
                    pool = np.setdiff1d(np.arange(buffer_size, dtype=np.int64), 
                                       np.fromiter(chosen, dtype=np.int64), 
                                       assume_unique=True)
                    if pool.size >= n_uni:
                        rand_idx = np.random.choice(pool, size=n_uni, replace=False)
                    else:
                        extra = np.random.choice(pool, size=pool.size, replace=False) if pool.size > 0 else np.array([], dtype=np.int64)
                        rem = n_uni - extra.size
                        rand_idx = np.concatenate([extra, np.random.randint(0, buffer_size, size=rem, dtype=np.int64)], axis=0)
                else:
                    if buffer_size >= n_uni:
                        rand_idx = np.random.choice(buffer_size, size=n_uni, replace=False).astype(np.int64)
                    else:
                        rand_idx = np.random.randint(0, buffer_size, size=n_uni, dtype=np.int64)
            
            # 为均匀样本获取其当前树优先级（用于计算混合分布下的重要性采样权重）
            for didx in rand_idx:
                tree_idx = int(didx) + sum_tree.capacity - 1
                p = float(sum_tree.tree[tree_idx])
                uni_indices.append((tree_idx, p, int(didx)))
                if not per_replace and chosen is not None:
                    chosen.add(int(didx))
        
        # 合并并计算混合分布下的重要性采样权重
        merged = (per_indices or []) + (uni_indices or [])
        while len(merged) < batch_size:
            didx = int(np.random.randint(0, buffer_size))
            tree_idx = didx + sum_tree.capacity - 1
            p = float(sum_tree.tree[tree_idx])
            merged.append((tree_idx, p, didx))
        merged = merged[:batch_size]
        
        for i, (tree_idx, p, data_idx) in enumerate(merged):
            indices[i] = int(data_idx)
            # 混合分布概率：q_i = (1-m) × P_per(i) + m × P_uniform(i)
            if total_p > 1e-12 and p > 0:
                per_prob = np.clip(p / total_p, 1e-10, 1.0)
            else:
                per_prob = 1.0 / buffer_size
            q_i = (1.0 - m) * per_prob + m * (1.0 / buffer_size)
            q_i = np.clip(q_i, 1e-10, 1.0)
            # 重要性采样权重：w_i = (N × q_i)^(-beta)
            weights[i] = (buffer_size * q_i) ** (-beta)
        
        # 归一化权重
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        max_w = np.max(weights)
        if max_w > 1e-8:
            weights = weights / max_w
        else:
            weights = np.ones_like(weights)
        
        return indices, weights, None


# ============================================================================
# 4. 优先级更新方法
# ============================================================================

def update_priorities(sum_tree, indices, td_errors, rewards, insert_steps,
                     global_insert_counter,
                     priority_td_weight=1.0,
                     priority_reward_weight=0.0,
                     priority_age_decay=1.0,
                     negative_slope=0.6,
                     epsilon=0.01,
                     capacity=None):
    """
    更新样本优先级
    
    Args:
        sum_tree: SumTree实例
        indices: 样本索引数组
        td_errors: TD误差数组
        rewards: 奖励数组（用于计算奖励幅值）
        insert_steps: 样本插入步数数组
        global_insert_counter: 全局插入计数器
        priority_td_weight: TD误差权重
        priority_reward_weight: 奖励幅值权重
        priority_age_decay: 年龄衰减系数
        negative_slope: 优先级幂次
        epsilon: TD误差最小值
        capacity: 缓冲区容量（用于计算树索引）
    """
    # 输入验证
    td_errors = np.asarray(td_errors, dtype=np.float32)
    td_errors = np.nan_to_num(td_errors, nan=1.0, posinf=10000.0, neginf=1.0)
    
    if td_errors.size == 0:
        return
    
    # TD误差处理
    td_errors = np.abs(td_errors) + epsilon
    MAX_TD_ERROR = 10000.0
    td_errors = np.clip(td_errors, epsilon, MAX_TD_ERROR)
    clipped_errors = np.clip(td_errors, 1e-6, MAX_TD_ERROR)
    
    # 更新每个样本的优先级
    for i, idx in enumerate(indices):
        # 奖励幅值
        try:
            rew_abs = float(np.mean(np.abs(rewards[int(idx)])))
            rew_abs = min(rew_abs, 1000.0)
        except Exception:
            rew_abs = 0.0
        
        # 年龄权重
        try:
            age = int(global_insert_counter - insert_steps[int(idx)])
            age = max(0, min(age, 100000))
            if priority_age_decay < 1.0 and priority_age_decay > 0:
                age_w = priority_age_decay ** age
                if not np.isfinite(age_w) or age_w <= 0:
                    age_w = 1.0
            else:
                age_w = 1.0
        except Exception:
            age_w = 1.0
        
        # 计算优先级
        signal = priority_td_weight * float(clipped_errors[i]) + priority_reward_weight * rew_abs
        signal = np.clip(signal, 1e-6, MAX_TD_ERROR)
        
        try:
            if not np.isfinite(signal) or not np.isfinite(negative_slope) or not np.isfinite(age_w):
                p = 1.0
            else:
                p = (signal ** float(negative_slope)) * float(age_w)
                if not np.isfinite(p) or p <= 0:
                    p = 1.0
        except (ValueError, OverflowError, FloatingPointError):
            p = 1.0
        
        MAX_PRIORITY_VALUE = 50000.0
        p = np.clip(p, 1e-8, MAX_PRIORITY_VALUE)
        
        if not np.isfinite(p) or p <= 0:
            p = 1.0
        
        # 更新SumTree
        tree_idx = int(idx) + capacity - 1
        sum_tree.update(tree_idx, float(p))


# ============================================================================
# 5. 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：创建SumTree并测试
    capacity = 1000
    sum_tree = SumTree(capacity)
    
    # 添加一些优先级
    for i in range(100):
        priority = np.random.uniform(0.1, 10.0)
        sum_tree.add(priority)
    
    # 测试采样
    total = sum_tree.total()
    print(f"总优先级: {total}")
    
    # 单个采样
    s = np.random.uniform(0, total)
    tree_idx, priority, data_idx = sum_tree.get(s)
    print(f"采样值 {s:.2f} -> 数据索引 {data_idx}, 优先级 {priority:.4f}")
    
    # 批量采样
    batch_size = 10
    random_samples = np.random.uniform(0, total, size=batch_size)
    leaf_indices, priorities, data_indices = sum_tree.get_batch(random_samples)
    print(f"批量采样结果: {data_indices}")
    
    print("\nPER实现测试完成！")



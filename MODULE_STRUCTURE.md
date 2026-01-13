# 模块化结构说明

## 完整的模块化架构

所有模块都采用**接口层模式**，直接调用 `paper3d_train_optimized.py` 中的实现。

## 模块列表

### 1. utils_refactored/ - 工具函数模块
**位置**: `utils_refactored/`

**包含**:
- `helpers.py` - 基础工具函数
- `gpu_config.py` - GPU配置
- `scenario_loader.py` - 场景加载
- `networks.py` - 网络构建
- `visualization_utils.py` - 可视化工具
- `memory_utils.py` - 内存管理
- `training_utils.py` - 训练辅助

**使用示例**:
```python
from utils_refactored import configure_gpu, load_scenario_module
from utils_refactored.networks import build_continuous_action_network
```

### 2. algorithms/ - 算法模块
**位置**: `algorithms/`

**包含**:
- `OptimizedMADDPG` - MADDPG算法实现
- `OptimizedMATD3` - MATD3算法实现

**使用示例**:
```python
from algorithms import OptimizedMADDPG, OptimizedMATD3

# 创建MADDPG智能体
agent = OptimizedMADDPG(n_agents=3, obs_shapes=[78, 78, 78], action_dims=[7, 7, 7], args=args)
```

### 3. training/ - 训练模块
**位置**: `training/`

**包含**:
- `train()` - 主训练函数
- `parse_args()` - 参数解析函数

**使用示例**:
```python
from training import train, parse_args

args = parse_args()
train(args)
```

### 4. environment/ - 环境管理模块
**位置**: `environment/`

**包含**:
- `ParallelEnv` - 并行环境类
- `SingleEnvWrapper` - 单环境包装器
- `make_env_init()` - 环境初始化函数

**使用示例**:
```python
from environment import ParallelEnv, make_env_init

env = ParallelEnv(env_fns=[make_env_init(0, args_dict)])
```

### 5. replay_buffer/ - 回放缓冲区模块
**位置**: `replay_buffer/`

**包含**:
- `ReplayBuffer` - 基础回放缓冲区
- `SumTree` - 优先经验回放的SumTree
- `LiteReplayBuffer` - 轻量级回放缓冲区
- `TFReplayBuffer` - TensorFlow回放缓冲区

**使用示例**:
```python
from replay_buffer import LiteReplayBuffer

buffer = LiteReplayBuffer(
    capacity=150000,
    n_agents=3,
    obs_dims=[78, 78, 78],
    act_dims=[7, 7, 7]
)
```

### 6. potential_field/ - 势场修正模块
**位置**: `potential_field/`

**说明**: 势场修正的主要方法在算法类中，如果需要独立使用，可以从 `potential_field_corrector` 导入。

## 完整使用示例

```python
# 1. 导入工具函数
from utils_refactored import configure_gpu, load_scenario_module
from utils_refactored.networks import build_continuous_action_network

# 2. 配置GPU
configure_gpu()

# 3. 加载场景
scenario = load_scenario_module('paper3d_terrain_energy', args)

# 4. 导入算法
from algorithms import OptimizedMADDPG

# 5. 创建智能体
agent = OptimizedMADDPG(n_agents=3, obs_shapes=[78, 78, 78], action_dims=[7, 7, 7], args=args)

# 6. 导入回放缓冲区
from replay_buffer import LiteReplayBuffer
buffer = LiteReplayBuffer(capacity=150000, n_agents=3, obs_dims=[78, 78, 78], act_dims=[7, 7, 7])

# 7. 开始训练
from training import train
train(args)
```

## 向后兼容

所有原有代码继续工作：

```python
# 原有方式仍然有效
from paper3d_train_optimized import OptimizedMADDPG, train, configure_gpu
```

## 优势

1. **清晰的模块结构** - 代码按功能分类
2. **零风险** - 所有实现都在原文件中
3. **完全兼容** - 原有代码继续工作
4. **易于使用** - 新代码可以使用清晰的模块导入


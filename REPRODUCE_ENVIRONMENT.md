# 运行环境复现指南 (Environment Requirements)

*本文档由 gemini-3.1-pro-preview 模型生成。*

本文档详细总结了在全新电脑上重新运行 `run_optimized.sh` 与 `paper3d_train_optimized.py` 训练脚本所需的各种硬件配置、系统驱动、依赖包以及项目结构内容。

## 1. 硬件配置要求 (Hardware)

*   **GPU（显卡）**：强烈建议配备 NVIDIA GPU。
    *   当前运行环境使用的是显存 8GB 的 RTX 4060 Laptop，能够正常支持当前负载。
    *   脚本默认开启了全局 XLA 编译（`XLA_GLOBAL=1`）和混合精度计算。为了满足 `BATCH_SIZE=1024`（甚至可扩展至 4096）的并行计算需求，**显存建议至少配备 8GB 及以上**。
    *   *备注意：如果在没有显卡的电脑上运行，程序会触发“强制CPU模式”，但由于多智能体与并行环境的计算量极大，运行速度将会极其缓慢，不建议使用纯 CPU 训练。*
*   **CPU（处理器）**：建议使用多核 CPU。脚本中针对 CPU 的多线程并行做了优化，默认配置了 `CPU_THREADS=12` 来提升经验回放（Replay Buffer）和向量化环境计算的效率。

## 2. 系统与底层驱动 (OS & Drivers)

*   **操作系统**：推荐使用 **Linux 环境**（如当前测试成功的 Ubuntu on WSL2）。如果要在 Windows 原生环境运行，部分底层系统路径操作或多进程生成方式（`spawn`）可能会有差异，需要根据报错手动调整。
*   **CUDA Toolkit 与 NVIDIA 驱动**：
    *   当前的 NVIDIA 驱动版本为 595.71。
    *   当前的 CUDA 运行版本为 13.2，底层编译器 `nvcc` 为 12.8。
    *   **⚠️ 关键注意**：由于项目深度依赖 `tensorflow==2.12.0`，新电脑上最好配置能与之完全兼容的 **CUDA 11.8**，或者确保系统安装了支持 TF 2.12 的特定 NVIDIA 运行时库（如 `nvidia-cudnn-cu12`、`nvidia-cublas-cu12` 等 pip 包）。否则极易发生 GPU 无法被正确识别和调用的情况。

## 3. Python 环境与核心依赖包 (Python Packages)

虽然项目根目录可能存在 `requirements.txt`，但为避免冗余和版本冲突，基于真实终端环境输出的 `pip freeze` 以及代码 `import` 引用，**必须且仅需**安装以下特定版本的包：

*   **Python 版本**：**3.10.x**（当前开发环境稳定在 3.10.12，请勿使用 Python 3.11 或以上版本，可能会导致 TensorFlow 2.12 无法安装）。
*   **深度学习框架**：
    *   `tensorflow==2.12.0` （项目的核心计算框架，代码中涉及到大量 `@tf.function` 的 `jit_compile` 以及 XLA 加速操作，必须是此版本）。
    *   `tensorboard==2.12.0` （用于训练日志和 Loss 监控记录）。
*   **强化学习与环境交互**：
    *   `gym==0.26.2` （⚠️ 注意版本：Gym 在 0.26 版本前后有巨大的 API 破坏性更新，必须指定为 0.26.2）。
*   **数值计算与数据处理**：
    *   `numpy==1.23.5` （⚠️ 极其重要：TensorFlow 2.12 与 NumPy 1.23.5 兼容性最好。切勿使用 NumPy 2.0+ 系列，否则会引发大规模类型不匹配报错）。
    *   `scipy==1.15.2`
    *   `pandas==2.3.0`
*   **可视化与图表记录**：
    *   `matplotlib==3.10.1`
    *   `plotly==5.22.0` （由于脚本启用了可交互 HTML 轨迹图 `SAVE_INTERACTIVE_TRAJ=1`，必须安装 plotly）。
    *   `tqdm==4.67.1` （用于进度条输出）。

## 4. 项目自身的代码与结构依赖 (Project Directory Structure)

仅仅移动 `run_optimized.sh` 和 `paper3d_train_optimized.py` 两个文件是无法运行的。在克隆或转移项目时，必须保证新电脑的工作目录完整包含以下结构：

1.  **启动脚本与执行入口**：
    *   `run_optimized.sh` （主启动脚本，包含全部的环境变量与运行配置）
    *   `paper3d_train_optimized.py` （主训练入口的 Python 脚本）
    *   `run_evaluation.sh` （如果需要运行自动评估测试，这是评估入口脚本，`run_optimized.sh` 末尾会调用它）

2.  **根目录模块文件**：
    *   `potential_field_corrector.py` （势场修正器模块，极其核心）

3.  **核心环境与场景层 (Environment & Scenarios)**：
    必须完整复制 `multiagent/` 文件夹。包括但不限于以下关键内容：
    *   `multiagent/environment.py` （环境引擎主入口）
    *   `multiagent/core.py` （物理引擎，定义了刚体碰撞、运动和动力学）
    *   `multiagent/quadrotor_dynamics.py` （四旋翼动力学模型，如果在脚本中启用此模式，强依赖此文件）
    *   `multiagent/scenarios/` 目录下的场景文件，最关键的是：
        *   `paper3d_terrain_energy.py`（原始奖励机制场景）
        *   `paper3d_terrain_weighted.py`（分项加权求和奖励场景）
        *   `paper3d_terrain_vectorized.py`（向量化分项加权求和场景）

4.  **向量化工具与处理层 (Utils)**：
    必须完整复制 `utils/` 文件夹。包括：
    *   `utils/observation_processor.py` （普通观察处理器）
    *   `utils/vectorized_observation_processor.py` （向量化优化的观察处理器）
    *   `utils/vectorized_reward_calculator.py` （极其重要的向量化计算核心）

5.  **智能体与网络算法层 (Agents & Networks)**：
    必须完整复制 `agents/` 和可能使用的网络基类文件夹。包括：
    *   `agents/maddpg.py`、`agents/ddpg.py` （算法本体实现）
    *   `agents/noise.py` 和 `agents/vectorized_ou_noise.py` （OU探索噪声实现）
    *   `agents/nets/` 或其他存放 Actor 和 Critic 网络定义文件的子目录
    *   (如果有) `core/` 目录（如包含 `replay_buffer.py` 和网络构建模块）

6.  **可视化与后处理层 (Visualization)**：
    必须复制 `visualization/` 文件夹：
    *   `visualization/trajectory_visualizer.py` （用于保存轨迹图及 HTML 交互图谱）

7.  **运行配置与位置数据**：
    *   `saved_positions/` 文件夹及其内部的内容（特别是 `saved_positions/5.json`，这是 `run_optimized.sh` 配置中指定的预设起点和终点坐标信息文件）。如果缺失这个文件，使用固定位置复现会直接失败。

## 5. 快速部署命令指南 (Setup Guide)

在全新电脑上，你可以按照以下流程命令创建一个完全一致的运行环境（前提是系统已正确安装了对应显卡的 NVIDIA 驱动）：

```bash
# 1. 使用 Conda 创建 Python 3.10 的虚拟环境
conda create -n maddpg_env python=3.10.12
conda activate maddpg_env

# 2. 安装核心深度学习库与强化学习库
pip install tensorflow==2.12.0 tensorboard==2.12.0 gym==0.26.2

# 3. 安装科学计算库并降级 NumPy (解决与 TF 2.12 的兼容性)
pip install numpy==1.23.5 scipy==1.15.2 pandas==2.3.0

# 4. 安装图形及其他辅助库
pip install matplotlib==3.10.1 plotly==5.22.0 tqdm==4.67.1

# 5. 传输项目代码
# 请确保将原电脑下的 run_optimized.sh, paper3d_train_optimized.py 
# 以及 multiagent, utils, saved_positions, agents, visualization 等所有文件夹原样拷贝至新目录。

# 6. 赋予脚本执行权限并运行
chmod +x run_optimized.sh
./run_optimized.sh
```
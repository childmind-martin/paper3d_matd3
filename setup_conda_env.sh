#!/bin/bash
# 在当前工作空间安装 conda 环境以运行 run_optimized.sh
# 用法: bash setup_conda_env.sh  或  ./setup_conda_env.sh  （不要用 sh 运行）
# 建议在 WSL 终端中运行（需网络下载 TensorFlow 等包，约 600MB+）
# 若 pip 下载 TensorFlow 时出现 SSL 错误或超时，可在本机终端重试此脚本，或使用国内 PyPI 镜像：
#   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.12.0 ...

# 若被 sh 调用会报 Bad substitution / source not found，此处用 bash 重新执行
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-maddpg_env}"

if [ ! -d "$CONDA_ROOT" ]; then
    echo "未找到 Miniconda: $CONDA_ROOT"
    echo "若已安装在其他路径，请设置: export CONDA_ROOT=/path/to/miniconda3"
    exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "创建 conda 环境: $ENV_NAME (python=3.10.12)"
    conda create -n "$ENV_NAME" python=3.10.12 -y
fi

echo "激活环境: $ENV_NAME"
conda activate "$ENV_NAME"

echo "安装核心依赖 (tensorflow, gym, numpy 等)..."
pip install -U pip
pip install tensorflow==2.12.0 tensorboard==2.12.0 gym==0.26.2
pip install numpy==1.23.5 scipy==1.15.2
pip install matplotlib==3.10.1 plotly==5.22.0 tqdm==4.67.1
echo "安装当前代码实际使用的运行依赖 (opencv, pygame, imageio, psutil, OpenGL/pyglet)..."
pip install imageio imageio-ffmpeg opencv-python pygame psutil PyOpenGL pyglet

echo "安装项目 multiagent 包 (可编辑)..."
pip install -e "$SCRIPT_DIR/src/multiagent"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "检测到 NVIDIA GPU，安装 TensorFlow 2.12 对应的 CUDA 11.8 / cuDNN 8.x 运行库..."
    conda install -y -n "$ENV_NAME" --override-channels -c conda-forge -c nvidia \
        cudatoolkit=11.8 "cudnn>=8.6,<9"
    if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    if [ -f "$SCRIPT_DIR/tools/setup_conda_ld_path.sh" ]; then
        bash "$SCRIPT_DIR/tools/setup_conda_ld_path.sh"
    fi
else
    echo "未检测到 NVIDIA GPU，跳过 CUDA/cuDNN 运行库安装。"
fi

echo "验证 TensorFlow 与 GPU..."
python "$SCRIPT_DIR/tools/check_tf_env.py" --label "TensorFlow Environment Check"

echo ""
echo "环境已就绪。运行训练:"
echo "  conda activate $ENV_NAME"
echo "  cd $SCRIPT_DIR"
echo "  ./run_optimized.sh"
echo ""
echo "或使用: ./run_with_conda.sh [参数...]  # 自动激活 $ENV_NAME 并执行 run_optimized.sh"

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
pip install numpy==1.23.5 scipy==1.15.2 pandas==2.3.0
pip install matplotlib==3.10.1 plotly==5.22.0 tqdm==4.67.1

echo "安装项目 multiagent 包 (可编辑)..."
pip install -e "$SCRIPT_DIR/src/multiagent"

echo "验证 TensorFlow 与 GPU..."
python -c "
import tensorflow as tf
print('TensorFlow', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpus)
"

echo ""
echo "环境已就绪。运行训练:"
echo "  conda activate $ENV_NAME"
echo "  cd $SCRIPT_DIR"
echo "  ./run_optimized.sh"
echo ""
echo "或使用: ./run_with_conda.sh [参数...]  # 自动激活 $ENV_NAME 并执行 run_optimized.sh"

#!/bin/bash
# Repair the conda runtime required by run_optimized.sh.

[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"

set -euo pipefail

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
    echo "未找到 conda 环境: $ENV_NAME"
    echo "请先运行: bash setup_conda_env.sh"
    exit 1
fi

conda activate "$ENV_NAME"

echo "修复 conda 环境: $ENV_NAME"
echo "Python: $(python --version 2>&1)"
echo "解释器: $(command -v python)"
echo ""

echo "移除已知冲突的 TensorFlow 变体..."
pip uninstall -y \
    tensorflow-directml-plugin \
    tensorflow-cpu \
    tensorflow-intel \
    tensorflow-macos || true

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [ -d "$SITE_PACKAGES" ]; then
    find "$SITE_PACKAGES" -maxdepth 1 \
        \( -name "tensorflow-plugins" \
        -o -name "tensorflow-directml-plugin" \
        -o -name "tensorflow_directml_plugin-*.dist-info" \
        -o -name "tensorflow_cpu-*.dist-info" \) \
        -exec rm -rf {} +
fi

echo ""
echo "重装项目要求版本..."
pip install -U pip
pip install --upgrade --force-reinstall \
    tensorflow==2.12.0 \
    tensorboard==2.12.0 \
    gym==0.26.2 \
    numpy==1.23.5 \
    scipy==1.15.2 \
    pandas==2.3.0 \
    matplotlib==3.10.1 \
    plotly==5.22.0 \
    tqdm==4.67.1
pip install --upgrade --force-reinstall \
    imageio \
    imageio-ffmpeg \
    opencv-python \
    pygame \
    psutil \
    PyOpenGL \
    pyglet

echo ""
echo "安装 TensorFlow 2.12 所需 GPU 运行库 (CUDA 11.8 + cuDNN 8.x)..."
conda install -y -n "$ENV_NAME" --override-channels -c conda-forge -c nvidia \
    cudatoolkit=11.8 \
    "cudnn>=8.6,<9"

echo ""
echo "安装项目 multiagent 包..."
pip install -e "$SCRIPT_DIR/src/multiagent"

echo ""
echo "验证 TensorFlow 运行时..."
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -f "$SCRIPT_DIR/tools/setup_conda_ld_path.sh" ]; then
    bash "$SCRIPT_DIR/tools/setup_conda_ld_path.sh"
fi
python "$SCRIPT_DIR/tools/check_tf_env.py" --label "TensorFlow Environment Repair Check"

echo ""
echo "环境修复完成。可尝试运行:"
echo "  ./run_with_conda.sh"

#!/bin/bash
# 在 conda 环境 maddpg_env 下运行 run_optimized.sh
# 用法: ./run_with_conda.sh [run_optimized.sh 的参数，如 200 1024 "实验名" 1 matd3]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-maddpg_env}"

if [ ! -d "$CONDA_ROOT" ]; then
    echo "未找到 Miniconda: $CONDA_ROOT"
    echo "请先运行: bash setup_conda_env.sh"
    exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "检查 TensorFlow 运行时..."
if ! python "$SCRIPT_DIR/tools/check_tf_env.py" --label "Preflight Check"; then
    echo ""
    echo "TensorFlow 环境检查失败。请先运行:"
    echo "  bash $SCRIPT_DIR/repair_conda_env.sh"
    exit 1
fi

exec ./run_optimized.sh "$@"

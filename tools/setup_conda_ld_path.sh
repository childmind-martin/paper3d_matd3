#!/usr/bin/env bash
# 一次性脚本：为 maddpg_env 设置 conda activate 时自动导出 LD_LIBRARY_PATH，
# 这样每次 conda activate maddpg_env 后无需再手动 export，TensorFlow 即可找到 cu11 库。
# 用法: bash tools/setup_conda_ld_path.sh

set -e
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-maddpg_env}"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
ACTIVATE_D="$ENV_DIR/etc/conda/activate.d"

if [ ! -d "$ENV_DIR" ]; then
    echo "未找到环境: $ENV_DIR"
    exit 1
fi

mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/ld_library_path.sh" << 'EOF'
# 将 conda 环境的 lib（含 CUDA 11 / cuDNN 8）加入 LD_LIBRARY_PATH，供 TensorFlow 使用 GPU
if [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
EOF
echo "已创建: $ACTIVATE_D/ld_library_path.sh"
echo "之后每次执行 conda activate $ENV_NAME 时会自动设置 LD_LIBRARY_PATH。"
echo "请在新终端中测试: conda activate $ENV_NAME && python tools/check_tf_env.py"

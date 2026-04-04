#!/usr/bin/env bash
# 收集服务器软硬件与 Python/Conda/TensorFlow 运行时信息，便于部署当前仓库。
# 用法：
#   bash tools/check_server_env.sh
# 或在 conda 环境激活后执行：
#   conda activate maddpg_env && bash tools/check_server_env.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

section() {
    echo ""
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

show_cmd() {
    echo "\$ $*"
    "$@" 2>&1 || true
}

show_file() {
    local path="$1"
    if [ -f "$path" ]; then
        echo "\$ cat $path"
        cat "$path" 2>&1 || true
    fi
}

PY_BIN=""
if have_cmd python; then
    PY_BIN="$(command -v python)"
elif have_cmd python3; then
    PY_BIN="$(command -v python3)"
fi

section "Basic"
show_cmd date
show_cmd uname -a
show_file /etc/os-release
if have_cmd hostnamectl; then
    show_cmd hostnamectl
fi
if have_cmd lsb_release; then
    show_cmd lsb_release -a
fi

section "CPU / Memory / Disk"
if have_cmd lscpu; then
    show_cmd lscpu
else
    show_file /proc/cpuinfo
fi
if have_cmd free; then
    show_cmd free -h
fi
if have_cmd df; then
    show_cmd df -h
fi
if have_cmd lsblk; then
    show_cmd lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
fi

section "GPU / CUDA"
if have_cmd nvidia-smi; then
    show_cmd nvidia-smi -L
    show_cmd nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader
else
    echo "nvidia-smi: not found"
fi
if have_cmd nvcc; then
    show_cmd nvcc --version
else
    echo "nvcc: not found"
fi
show_file /usr/local/cuda/version.txt
show_file /usr/local/cuda/version.json
if have_cmd ldconfig; then
    echo "\$ ldconfig -p | egrep 'libcuda|libcudnn|libcublas|libcudart' | head -n 50"
    ldconfig -p 2>/dev/null | egrep 'libcuda|libcudnn|libcublas|libcudart' | head -n 50 || true
fi

section "Python / Conda / Pip"
if have_cmd which; then
    if have_cmd python; then
        show_cmd which python
    fi
    if have_cmd python3; then
        show_cmd which python3
    fi
    if have_cmd pip; then
        show_cmd which pip
    fi
    if have_cmd pip3; then
        show_cmd which pip3
    fi
fi
if have_cmd python; then
    show_cmd python --version
fi
if have_cmd python3; then
    show_cmd python3 --version
fi
if have_cmd pip; then
    show_cmd pip --version
fi
if have_cmd pip3; then
    show_cmd pip3 --version
fi
if have_cmd conda; then
    show_cmd conda --version
    show_cmd conda info --envs
else
    echo "conda: not found"
fi
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

section "Compiler / Build Tools"
if have_cmd gcc; then
    show_cmd gcc --version
fi
if have_cmd g++; then
    show_cmd g++ --version
fi
if have_cmd make; then
    show_cmd make --version
fi

section "Project"
show_cmd pwd
echo "\$ ls $REPO_ROOT"
ls "$REPO_ROOT" 2>&1 || true
show_file "$REPO_ROOT/requirements.txt"

section "Key Python Packages"
if [ -n "$PY_BIN" ]; then
    "$PY_BIN" - <<'PY'
import importlib
import sys

mods = [
    "tensorflow",
    "tensorboard",
    "numpy",
    "scipy",
    "gym",
    "gymnasium",
    "matplotlib",
    "plotly",
    "cv2",
    "pygame",
    "imageio",
    "psutil",
    "OpenGL",
    "pyglet",
]

print(f"Python executable: {sys.executable}")
for name in mods:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "<no __version__>")
        print(f"{name}: OK ({version})")
    except Exception as exc:
        print(f"{name}: FAIL ({type(exc).__name__}: {exc})")
PY
else
    echo "未找到 python/python3，跳过 Python 包检查。"
fi

section "TensorFlow Project Check"
if [ -n "$PY_BIN" ] && [ -f "$REPO_ROOT/tools/check_tf_env.py" ]; then
    echo "\$ $PY_BIN $REPO_ROOT/tools/check_tf_env.py --label 'Server TensorFlow Check'"
    "$PY_BIN" "$REPO_ROOT/tools/check_tf_env.py" --label "Server TensorFlow Check" 2>&1 || true
else
    echo "未找到 Python 或 tools/check_tf_env.py，跳过 TensorFlow 专项检查。"
fi

section "Done"
echo "请将以上输出返回给我，我会据此给出下一步安装/修复命令。"

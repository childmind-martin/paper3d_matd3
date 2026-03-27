#!/usr/bin/env python3
"""Validate the TensorFlow runtime used by this project."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import sys


PINNED_PACKAGES = {
    "tensorflow": "2.12.0",
    "tensorboard": "2.12.0",
    "numpy": "1.23.5",
    "gym": "0.26.2",
}

SUSPECT_PACKAGES = (
    "tensorflow-directml-plugin",
    "tensorflow-cpu",
    "tensorflow-intel",
    "tensorflow-macos",
)

EXPECTED_GPU_LIBS = (
    "libcudnn.so.8",
    "libcublas.so.11",
    "libcublasLt.so.11",
    "libcudart.so.11.0",
    "libcusolver.so.11",
    "libcusparse.so.11",
    "libcufft.so.10",
    "libcurand.so.10",
)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def installed_suspects() -> dict[str, str]:
    return {
        name: version
        for name in SUSPECT_PACKAGES
        if (version := package_version(name)) is not None
    }


def print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_pinned_versions() -> None:
    print("Pinned runtime:")
    for name, expected in PINNED_PACKAGES.items():
        actual = package_version(name)
        state = actual if actual is not None else "missing"
        print(f"  - {name}: {state} (expected {expected})")


def print_remediation(env_name: str) -> None:
    print("")
    print("Suggested repair:")
    print(f"  - bash repair_conda_env.sh")
    print("Manual alternative:")
    print(f"  - conda activate {env_name}")
    print("  - conda install -y -c conda-forge cudatoolkit=11.8 cudnn=8.6")
    print("  - export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"")
    print(
        "  - pip uninstall -y tensorflow-directml-plugin tensorflow-cpu "
        "tensorflow-intel tensorflow-macos"
    )
    print(
        "  - pip install --upgrade --force-reinstall "
        "tensorflow==2.12.0 tensorboard==2.12.0 gym==0.26.2 "
        "numpy==1.23.5 scipy==1.15.2 pandas==2.3.0 "
        "matplotlib==3.10.1 plotly==5.22.0 tqdm==4.67.1"
    )


def cuda_driver_init_result() -> int | None:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None

    cu_init = lib.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    return int(cu_init(0))


def missing_gpu_runtime_libs() -> list[str]:
    missing: list[str] = []
    for lib_name in EXPECTED_GPU_LIBS:
        try:
            ctypes.CDLL(lib_name)
        except OSError:
            missing.append(lib_name)
    return missing


def running_in_wsl() -> bool:
    if Path("/dev/dxg").exists():
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default="TensorFlow Environment Check",
        help="Title printed at the top of the report.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "maddpg_env")

    print_header(args.label)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print_pinned_versions()

    suspects = installed_suspects()
    if suspects:
        print("Suspicious packages:")
        for name, version in suspects.items():
            print(f"  - {name}: {version}")

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        text = str(exc)
        print("")
        print("TensorFlow import: FAILED")
        print(f"  - {type(exc).__name__}: {text}")

        if "libtfdml_plugin.so" in text or "tensorflow-directml-plugin" in suspects:
            print("")
            print("Diagnosis:")
            print("  - A DirectML plugin is installed in this Linux conda environment.")
            print("  - That plugin is incompatible with the TensorFlow 2.12 runtime used here.")
            print("  - The environment also contains extra TensorFlow variants that should be removed.")
        elif "Could not find cuda drivers" in text:
            print("")
            print("Diagnosis:")
            print("  - TensorFlow imported the CPU runtime but could not see the NVIDIA driver.")
            print("  - Check the WSL / CUDA driver libraries after repairing the Python packages.")

        print_remediation(env_name)
        return 1

    print("")
    print(f"TensorFlow import: OK ({tf.__version__})")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Visible GPUs: {gpus}")
    else:
        print("Visible GPUs: []")
        cu_init_result = cuda_driver_init_result()
        if cu_init_result is not None:
            print(f"CUDA driver init result: {cu_init_result}")
        missing_libs = missing_gpu_runtime_libs()
        if missing_libs:
            print("Missing GPU runtime libs:")
            for lib_name in missing_libs:
                print(f"  - {lib_name}")
        if shutil.which("nvidia-smi"):
            print("Warning:")
            print("  - `nvidia-smi` exists, but TensorFlow still does not see a GPU.")
            print("  - If training runs on CPU only, repair the env first and then re-check.")
        if os.environ.get("CONDA_PREFIX"):
            print(f"  - Active conda env: {os.environ['CONDA_PREFIX']}")
            print("  - If CUDA/cuDNN was installed into the env, ensure `$CONDA_PREFIX/lib` is on `LD_LIBRARY_PATH`.")
        if running_in_wsl() and cu_init_result not in (None, 0):
            print("  - WSL detected: CUDA driver initialization is already failing below TensorFlow.")
            if Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1").exists():
                print("  - `/usr/lib/x86_64-linux-gnu/libcuda.so.1` exists in WSL.")
                print("  - That usually means a Linux-side NVIDIA driver package was installed.")
                print("  - In WSL, CUDA should come from the Windows driver stub under `/usr/lib/wsl/lib`.")

    print("")
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation-path GPU smoke test.

Purpose:
1. Reuse the project's `configure_gpu()` path.
2. Verify whether TensorFlow can actually see a GPU.
3. Run a small matmul + MLP inference and report the real device placement.
4. Optionally hold the process for a few seconds so `nvidia-smi` can observe it.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return int(proc.returncode), proc.stdout.strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _safe_env(name: str, default: str = "<unset>") -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _format_device_name(device_text: str) -> str:
    if not device_text:
        return "<empty>"
    marker = "device:"
    idx = device_text.find(marker)
    if idx >= 0:
        return device_text[idx + len(marker):]
    return device_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test whether evaluation can really use GPU.")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix size for matmul smoke test.")
    parser.add_argument("--iterations", type=int, default=6, help="Matmul iterations after warmup.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for MLP inference smoke test.")
    parser.add_argument("--obs-dim", type=int, default=81, help="Input dimension for the MLP smoke test.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension for the MLP smoke test.")
    parser.add_argument("--hold-seconds", type=float, default=5.0, help="Seconds to keep the process alive after success.")
    parser.add_argument("--disable-project-configure-gpu", action="store_true", help="Skip project configure_gpu() and use raw TensorFlow only.")
    parser.add_argument("--jit-compile", type=int, choices=[0, 1], default=0, help="Use tf.function(jit_compile=True) for the MLP smoke test.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _print_header("Evaluation GPU Smoke Test")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")
    print(f"repo_root: {repo_root}")
    print(f"CONDA_PREFIX: {_safe_env('CONDA_PREFIX')}")
    print(f"CUDA_VISIBLE_DEVICES: {_safe_env('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU_ID: {_safe_env('GPU_ID')}")
    print(f"XLA_GLOBAL: {_safe_env('XLA_GLOBAL')}")
    print(f"PF_JIT: {_safe_env('PF_JIT')}")
    print(f"LD_LIBRARY_PATH: {_safe_env('LD_LIBRARY_PATH')}")

    rc, out = _run_cmd(["nvidia-smi", "-L"])
    print(f"nvidia-smi -L rc={rc}")
    print(out if out else "<no output>")

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        _print_header("Result")
        print(f"FAIL: TensorFlow import failed: {type(exc).__name__}: {exc}")
        return 1

    _print_header("TensorFlow")
    print(f"tensorflow: {tf.__version__}")
    try:
        print(f"is_built_with_cuda: {bool(tf.test.is_built_with_cuda())}")
    except Exception:
        pass
    try:
        build_info = tf.sysconfig.get_build_info()
        cuda_ver = build_info.get("cuda_version", "<unknown>")
        cudnn_ver = build_info.get("cudnn_version", "<unknown>")
        print(f"build_cuda_version: {cuda_ver}")
        print(f"build_cudnn_version: {cudnn_ver}")
    except Exception:
        pass

    configure_gpu_ok: Optional[bool] = None
    if not args.disable_project_configure_gpu:
        try:
            from paper3d_train_optimized import configure_gpu  # type: ignore

            configure_gpu_ok = bool(configure_gpu())
            print(f"project_configure_gpu: {configure_gpu_ok}")
        except Exception as exc:
            configure_gpu_ok = False
            print(f"project_configure_gpu_error: {type(exc).__name__}: {exc}")

    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"physical_gpus: {physical_gpus}")
    print(f"logical_gpus: {logical_gpus}")

    if not physical_gpus:
        _print_header("Result")
        print("FAIL: TensorFlow sees 0 physical GPUs.")
        print("This means evaluation will fall back to CPU.")
        return 2

    try:
        tf.config.set_soft_device_placement(True)
    except Exception:
        pass

    _print_header("MatMul Smoke")
    size = max(512, int(args.matrix_size))
    with tf.device("/GPU:0"):
        a = tf.random.normal((size, size), dtype=tf.float32)
        b = tf.random.normal((size, size), dtype=tf.float32)
        warm = tf.matmul(a, b)
        _ = warm.numpy()
        t0 = time.perf_counter()
        last = warm
        for _ in range(max(1, int(args.iterations))):
            last = tf.matmul(a, b)
        _ = last.numpy()
        t1 = time.perf_counter()
    matmul_device = _format_device_name(getattr(last, "device", ""))
    print(f"matmul_device: {matmul_device}")
    print(f"matmul_elapsed_sec: {t1 - t0:.4f}")

    _print_header("MLP Smoke")
    hidden = max(8, int(args.hidden))
    obs_dim = max(4, int(args.obs_dim))
    batch_size = max(1, int(args.batch_size))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(obs_dim,)),
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dense(7, activation="tanh"),
        ]
    )
    x = tf.random.normal((batch_size, obs_dim), dtype=tf.float32)

    @tf.function(jit_compile=bool(args.jit_compile))
    def _infer(tensor):
        return model(tensor, training=False)

    with tf.device("/GPU:0"):
        y = _infer(x)
        _ = y.numpy()
    infer_device = _format_device_name(getattr(y, "device", ""))
    print(f"mlp_output_device: {infer_device}")

    try:
        gpu_mem = tf.config.experimental.get_memory_info("GPU:0")
        print(f"gpu_memory_info: {gpu_mem}")
    except Exception as exc:
        print(f"gpu_memory_info_unavailable: {type(exc).__name__}: {exc}")

    rc, out = _run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    print(f"nvidia-smi compute-apps rc={rc}")
    print(out if out else "<no compute apps output>")

    gpu_used = "GPU:" in matmul_device or "GPU:" in infer_device
    _print_header("Result")
    if gpu_used:
        print("PASS: TensorFlow saw a GPU and the smoke operators executed on GPU.")
        if args.hold_seconds > 0:
            print(f"Holding process for {args.hold_seconds:.1f}s so you can observe nvidia-smi ...")
            time.sleep(float(args.hold_seconds))
        return 0

    print("FAIL: TensorFlow saw a GPU, but the smoke operators did not land on GPU.")
    print("This usually means soft placement fell back to CPU or the runtime is still misconfigured.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

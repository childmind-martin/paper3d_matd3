#!/usr/bin/env python3
"""Build a native Gazebo Transport dynamic replay player for MATD3 exports."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


MAGIC = b"MATD3DR1"


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _resolve_npz(dynamic_json: Optional[Path], npz: Optional[Path]) -> Path:
    if npz is not None:
        return Path(npz).resolve()
    if dynamic_json is None:
        raise ValueError("--dynamic-json or --npz is required")
    data = _load_json(dynamic_json)
    candidate = data.get("dynamic_npz_path") or data.get("export_paths", {}).get("dynamic_replay_npz")
    if not candidate:
        raise ValueError(f"dynamic npz path missing in {dynamic_json}")
    return Path(candidate).resolve()


def _default_world(dynamic_json: Optional[Path], fallback: str) -> str:
    if dynamic_json is None:
        return fallback
    try:
        data = _load_json(dynamic_json)
        return str(data.get("gazebo_dynamic_replay", {}).get("world_name") or fallback)
    except Exception:
        return fallback


def write_binary_replay(npz_path: Path, output_path: Path) -> Dict[str, Any]:
    with np.load(npz_path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        orientations = np.asarray(data.get("orientations_wxyz"), dtype=np.float32)
        if orientations.size == 0:
            orientations = np.zeros((positions.shape[0], positions.shape[1], 4), dtype=np.float32)
            orientations[:, :, 0] = 1.0
        times = np.asarray(data.get("times"), dtype=np.float64)
        if times.size != positions.shape[0]:
            times = np.arange(positions.shape[0], dtype=np.float64) * 0.08
    if positions.ndim != 3 or positions.shape[2] < 3:
        raise ValueError("positions must have shape [frame][agent][xyz]")
    if orientations.shape[:2] != positions.shape[:2] or orientations.shape[2] < 4:
        raise ValueError("orientations_wxyz must have shape [frame][agent][wxyz]")
    frame_count = int(positions.shape[0])
    agent_count = int(positions.shape[1])
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", frame_count, agent_count))
        for frame_idx in range(frame_count):
            f.write(struct.pack("<d", float(times[frame_idx])))
            for agent_idx in range(agent_count):
                f.write(struct.pack("<3f", *[float(v) for v in positions[frame_idx, agent_idx, :3]]))
                f.write(struct.pack("<4f", *[float(v) for v in orientations[frame_idx, agent_idx, :4]]))
    return {
        "binary_path": str(output_path),
        "frame_count": frame_count,
        "agent_count": agent_count,
        "bytes": int(output_path.stat().st_size),
    }


CPP_SOURCE = r'''
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/transport/Node.hh>

struct Frame {
  double time{0.0};
  std::vector<float> values;
};

static uint32_t readU32(std::ifstream &in) {
  uint32_t v = 0;
  in.read(reinterpret_cast<char *>(&v), sizeof(v));
  if (!in) throw std::runtime_error("failed to read uint32");
  return v;
}

static double readDouble(std::ifstream &in) {
  double v = 0.0;
  in.read(reinterpret_cast<char *>(&v), sizeof(v));
  if (!in) throw std::runtime_error("failed to read double");
  return v;
}

static float readFloat(std::ifstream &in) {
  float v = 0.0f;
  in.read(reinterpret_cast<char *>(&v), sizeof(v));
  if (!in) throw std::runtime_error("failed to read float");
  return v;
}

static std::vector<Frame> loadReplay(const std::string &path, uint32_t &agentCount) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open replay binary: " + path);
  char magic[8];
  in.read(magic, 8);
  if (!in || std::string(magic, 8) != "MATD3DR1") {
    throw std::runtime_error("invalid replay binary magic");
  }
  const uint32_t frameCount = readU32(in);
  agentCount = readU32(in);
  std::vector<Frame> frames;
  frames.reserve(frameCount);
  for (uint32_t frameIdx = 0; frameIdx < frameCount; ++frameIdx) {
    Frame frame;
    frame.time = readDouble(in);
    frame.values.resize(static_cast<size_t>(agentCount) * 7u);
    for (uint32_t agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
      const size_t base = static_cast<size_t>(agentIdx) * 7u;
      frame.values[base + 0] = readFloat(in);
      frame.values[base + 1] = readFloat(in);
      frame.values[base + 2] = readFloat(in);
      frame.values[base + 3] = readFloat(in);
      frame.values[base + 4] = readFloat(in);
      frame.values[base + 5] = readFloat(in);
      frame.values[base + 6] = readFloat(in);
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

static double medianDt(const std::vector<Frame> &frames) {
  std::vector<double> diffs;
  for (size_t i = 1; i < frames.size(); ++i) {
    const double dt = frames[i].time - frames[i - 1].time;
    if (std::isfinite(dt) && dt > 0.0) diffs.push_back(dt);
  }
  if (diffs.empty()) return 0.08;
  std::sort(diffs.begin(), diffs.end());
  return diffs[diffs.size() / 2u];
}

static std::vector<size_t> frameIndices(
    size_t frameCount, const std::vector<Frame> &frames, size_t stride,
    double speed, bool autoStride, double minCallPeriod) {
  stride = std::max<size_t>(1u, stride);
  if (autoStride) {
    const double dt = std::max(1e-6, medianDt(frames));
    const size_t computed = static_cast<size_t>(
        std::ceil(std::max(1e-6, speed) * std::max(0.0, minCallPeriod) / dt));
    stride = std::max<size_t>(stride, std::max<size_t>(1u, computed));
  }
  std::vector<size_t> indices;
  for (size_t i = 0; i < frameCount; i += stride) indices.push_back(i);
  if (indices.empty() || indices.back() != frameCount - 1u) indices.push_back(frameCount - 1u);
  return indices;
}

int main(int argc, char **argv) {
  std::string replayPath = "episode_001_dynamic_replay.bin";
  std::string world = "matd3_static_scene";
  std::string agentPrefix = "dynamic_agent_";
  double speed = 1.0;
  size_t stride = 1u;
  bool loop = false;
  bool autoStride = true;
  double minCallPeriod = 0.01;
  unsigned int timeoutMs = 1000u;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto needValue = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (arg == "--replay") replayPath = needValue(arg);
    else if (arg == "--world") world = needValue(arg);
    else if (arg == "--agent-prefix") agentPrefix = needValue(arg);
    else if (arg == "--speed") speed = std::stod(needValue(arg));
    else if (arg == "--stride") stride = static_cast<size_t>(std::stoul(needValue(arg)));
    else if (arg == "--timeout-ms") timeoutMs = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--min-call-period") minCallPeriod = std::stod(needValue(arg));
    else if (arg == "--loop") loop = true;
    else if (arg == "--no-auto-stride") autoStride = false;
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: fast_dynamic_replay --replay FILE --world NAME [--speed N] [--stride N] [--loop]\n";
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  speed = std::max(1e-6, speed);
  uint32_t agentCount = 0;
  const auto frames = loadReplay(replayPath, agentCount);
  if (frames.empty() || agentCount == 0u) {
    throw std::runtime_error("empty replay");
  }
  const auto indices = frameIndices(frames.size(), frames, stride, speed, autoStride, minCallPeriod);
  gz::transport::Node node;
  const std::string service = "/world/" + world + "/set_pose_vector";
  std::vector<std::string> names;
  names.reserve(agentCount);
  for (uint32_t agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
    names.push_back(agentPrefix + std::to_string(agentIdx));
  }

  std::cout << "Playing " << indices.size() << "/" << frames.size()
            << " frames, agents=" << agentCount
            << ", world=" << world
            << ", speed=" << speed
            << "x, mode=native_set_pose_vector" << std::endl;

  do {
    double previousTime = frames[indices.front()].time;
    for (const size_t frameIdx : indices) {
      const auto start = std::chrono::steady_clock::now();
      gz::msgs::Pose_V req;
      const Frame &frame = frames[frameIdx];
      for (uint32_t agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
        const size_t base = static_cast<size_t>(agentIdx) * 7u;
        auto *pose = req.add_pose();
        pose->set_name(names[agentIdx]);
        auto *pos = pose->mutable_position();
        pos->set_x(frame.values[base + 0]);
        pos->set_y(frame.values[base + 1]);
        pos->set_z(frame.values[base + 2]);
        auto *quat = pose->mutable_orientation();
        quat->set_w(frame.values[base + 3]);
        quat->set_x(frame.values[base + 4]);
        quat->set_y(frame.values[base + 5]);
        quat->set_z(frame.values[base + 6]);
      }
      gz::msgs::Boolean rep;
      bool result = false;
      const bool executed = node.Request(service, req, timeoutMs, rep, result);
      if (!executed || !result || !rep.data()) {
        std::cerr << "set_pose_vector failed at frame " << frameIdx
                  << " executed=" << executed << " result=" << result
                  << " reply=" << rep.data() << std::endl;
        return 2;
      }
      const double targetSleep = std::max(0.0, (frame.time - previousTime) / speed);
      previousTime = frame.time;
      const auto elapsed = std::chrono::steady_clock::now() - start;
      const double elapsedSec = std::chrono::duration<double>(elapsed).count();
      if (targetSleep > elapsedSec) {
        std::this_thread::sleep_for(std::chrono::duration<double>(targetSleep - elapsedSec));
      }
    }
  } while (loop);

  return 0;
}
'''


def write_cpp_project(output_dir: Path) -> Dict[str, str]:
    src = output_dir / "fast_dynamic_replay.cpp"
    cmake = output_dir / "CMakeLists.txt"
    src.write_text(CPP_SOURCE.strip() + "\n", encoding="utf-8")
    cmake.write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.16)",
                "project(matd3_fast_dynamic_replay LANGUAGES CXX)",
                "set(CMAKE_CXX_STANDARD 17)",
                "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
                "find_package(gz-transport13 REQUIRED)",
                "find_package(gz-msgs10 REQUIRED)",
                "add_executable(fast_dynamic_replay fast_dynamic_replay.cpp)",
                "target_link_libraries(fast_dynamic_replay PRIVATE gz-transport13::gz-transport13 gz-msgs10::gz-msgs10)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"source": str(src), "cmake": str(cmake)}


def build_cpp(output_dir: Path) -> Path:
    build_dir = output_dir / "fast_dynamic_replay_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    vendor_prefixes = ["/opt/ros/jazzy"]
    vendor_root = Path("/opt/ros/jazzy/opt")
    if vendor_root.exists():
        vendor_prefixes.extend(str(path) for path in sorted(vendor_root.glob("gz_*_vendor")) if path.is_dir())
    prefix = ":".join(vendor_prefixes)
    if env.get("CMAKE_PREFIX_PATH"):
        env["CMAKE_PREFIX_PATH"] = prefix + ":" + env["CMAKE_PREFIX_PATH"]
    else:
        env["CMAKE_PREFIX_PATH"] = prefix
    subprocess.run(["cmake", "-S", str(output_dir), "-B", str(build_dir)], check=True, env=env)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", "2"], check=True, env=env)
    exe = build_dir / "fast_dynamic_replay"
    if not exe.exists():
        raise FileNotFoundError(f"build completed but executable missing: {exe}")
    return exe


def create_fast_replay_project(
    dynamic_json: Optional[Path] = None,
    npz: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    world: Optional[str] = None,
    compile_player: bool = False,
) -> Dict[str, Any]:
    dynamic_json_path = Path(dynamic_json).resolve() if dynamic_json else None
    npz_path = _resolve_npz(dynamic_json_path, Path(npz).resolve() if npz else None)
    output_path = Path(output_dir).resolve() if output_dir else npz_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    binary_meta = write_binary_replay(npz_path, output_path / f"{npz_path.stem}.bin")
    project_meta = write_cpp_project(output_path)
    world_name = str(world or _default_world(dynamic_json_path, "matd3_static_scene"))
    exe = build_cpp(output_path) if compile_player else None
    executable_path = exe if exe else output_path / "fast_dynamic_replay_build" / "fast_dynamic_replay"
    meta = {
        "npz": str(npz_path),
        "world": world_name,
        "binary": binary_meta,
        "project": project_meta,
        "executable": str(exe) if exe else None,
        "run_command": (
            f"{executable_path} "
            f"--replay {binary_meta['binary_path']} --world {world_name} --speed 10 --stride 1"
        ),
    }
    meta_path = output_path / "fast_dynamic_replay_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    meta["meta_path"] = str(meta_path)
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and optionally compile a native Gazebo Transport replay player.")
    parser.add_argument("--dynamic-json", default=None)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--world", default=None)
    parser.add_argument("--compile", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dynamic_json = Path(args.dynamic_json).resolve() if args.dynamic_json else None
    meta = create_fast_replay_project(
        dynamic_json=dynamic_json,
        npz=Path(args.npz).resolve() if args.npz else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        world=args.world,
        compile_player=bool(args.compile),
    )
    meta_path = Path(meta["meta_path"])
    print(f"[gazebo_fast_replay_builder] meta={meta_path}")
    print(f"[gazebo_fast_replay_builder] binary={meta['binary']['binary_path']}")
    print(f"[gazebo_fast_replay_builder] source={meta['project']['source']}")
    if meta.get("executable"):
        print(f"[gazebo_fast_replay_builder] executable={meta['executable']}")
        print(f"[gazebo_fast_replay_builder] run={meta['run_command']}")
    else:
        print("[gazebo_fast_replay_builder] compile skipped; rerun with --compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

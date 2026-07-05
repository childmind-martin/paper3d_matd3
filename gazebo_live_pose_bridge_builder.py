#!/usr/bin/env python3
"""Build a native Gazebo Transport pose streaming bridge for live MATD3 runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


CPP_SOURCE = r'''
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/transport/Node.hh>

int main(int argc, char **argv) {
  std::string world = "matd3_static_scene";
  std::string agentPrefix = "dynamic_agent_";
  unsigned int timeoutMs = 1000u;
  unsigned int agentCount = 3u;
  unsigned int stepAfterTwist = 0u;
  unsigned int preStepSleepMs = 0u;
  unsigned int postStepSleepMs = 0u;
  unsigned int wallTimeStepMs = 0u;
  bool ack = false;
  bool pauseForStep = false;
  bool watchContacts = true;
  std::string contactFlagFile;
  std::string stateFile;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto needValue = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (arg == "--world") world = needValue(arg);
    else if (arg == "--agent-prefix") agentPrefix = needValue(arg);
    else if (arg == "--agent-count") agentCount = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--timeout-ms") timeoutMs = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--step-after-twist") stepAfterTwist = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--pre-step-sleep-ms") preStepSleepMs = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--post-step-sleep-ms") postStepSleepMs = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--wall-time-step-ms") wallTimeStepMs = static_cast<unsigned int>(std::stoul(needValue(arg)));
    else if (arg == "--ack") ack = true;
    else if (arg == "--pause-for-step") pauseForStep = true;
    else if (arg == "--contact-flag-file") contactFlagFile = needValue(arg);
    else if (arg == "--state-file") stateFile = needValue(arg);
    else if (arg == "--no-contact-watch") watchContacts = false;
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: pose_stream_bridge --world NAME --agent-count N [--agent-prefix dynamic_agent_]\n";
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (agentCount == 0u) {
    std::cerr << "agent-count must be positive" << std::endl;
    return 2;
  }

  gz::transport::Node node;
  const std::string service = "/world/" + world + "/set_pose_vector";
  const std::string controlService = "/world/" + world + "/control";
  std::atomic<bool> contactDetected{false};
  auto requestWorldControl = [&](gz::msgs::WorldControl &req, const std::string &label) -> bool {
    gz::msgs::Boolean rep;
    bool result = false;
    const bool executed = node.Request(controlService, req, timeoutMs, rep, result);
    if (!executed || !result || !rep.data()) {
      std::cerr << "world control " << label
                << " failed: executed=" << executed
                << " result=" << result
                << " reply=" << rep.data() << std::endl;
      return false;
    }
    return true;
  };
  std::vector<std::string> names;
  names.reserve(agentCount);
  for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
    names.push_back(agentPrefix + std::to_string(agentIdx));
  }

  std::mutex stateMutex;
  std::vector<std::array<double, 7>> latestPoses(agentCount);
  std::vector<bool> poseSeen(agentCount, false);
  unsigned long long stateFrame = 0u;
  for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
    latestPoses[agentIdx] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  }
  auto writeStateFileLocked = [&]() {
    if (stateFile.empty()) return;
    std::ofstream out(stateFile);
    if (!out) return;
    out << "{\"frame\":" << stateFrame << ",\"agents\":[";
    for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
      if (agentIdx > 0u) out << ",";
      const auto &p = latestPoses[agentIdx];
      out << "{\"name\":\"" << names[agentIdx] << "\",\"seen\":"
          << (poseSeen[agentIdx] ? "true" : "false")
          << ",\"position\":[" << p[0] << "," << p[1] << "," << p[2]
          << "],\"orientation_wxyz\":[" << p[3] << "," << p[4] << "," << p[5] << "," << p[6]
          << "]}";
    }
    out << "]}\n";
  };
  if (!stateFile.empty()) {
    const std::string poseTopic = "/world/" + world + "/pose/info";
    std::function<void(const gz::msgs::Pose_V &)> poseCb =
        [&](const gz::msgs::Pose_V &msg) {
      bool changed = false;
      std::lock_guard<std::mutex> lock(stateMutex);
      for (int poseIdx = 0; poseIdx < msg.pose_size(); ++poseIdx) {
        const auto &pose = msg.pose(poseIdx);
        for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
          if (pose.name() != names[agentIdx]) continue;
          auto &p = latestPoses[agentIdx];
          if (pose.has_position()) {
            p[0] = pose.position().x();
            p[1] = pose.position().y();
            p[2] = pose.position().z();
          }
          if (pose.has_orientation()) {
            p[3] = pose.orientation().w();
            p[4] = pose.orientation().x();
            p[5] = pose.orientation().y();
            p[6] = pose.orientation().z();
          }
          poseSeen[agentIdx] = true;
          changed = true;
        }
      }
      if (changed) {
        ++stateFrame;
        writeStateFileLocked();
      }
    };
    node.Subscribe<gz::msgs::Pose_V>(poseTopic, poseCb);
  }

	  std::vector<gz::transport::Node::Publisher> twistPubs;
  twistPubs.reserve(agentCount);
  for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
    const std::string topic = "/model/" + names[agentIdx] + "/cmd_vel";
    auto pub = node.Advertise<gz::msgs::Twist>(topic);
    if (!pub) {
      std::cerr << "failed to advertise twist topic: " << topic << std::endl;
      return 5;
    }
    twistPubs.push_back(pub);
	  }
	  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (pauseForStep || wallTimeStepMs > 0u) {
    gz::msgs::WorldControl pauseReq;
    pauseReq.set_pause(true);
    if (!requestWorldControl(pauseReq, "pause")) {
      return 9;
    }
  }
	  if (watchContacts) {
    for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
      const std::string topic = "/world/" + world + "/model/" + names[agentIdx] +
          "/link/uav_marker_link/sensor/python_collision_contact/contact";
      std::function<void(const gz::msgs::Contacts &)> cb =
          [&, topic, agentIdx](const gz::msgs::Contacts &msg) {
        if (msg.contact_size() <= 0) return;
        contactDetected.store(true);
        if (!contactFlagFile.empty()) {
          std::ofstream out(contactFlagFile, std::ios::app);
          if (out) {
            out << "topic " << topic << "\n";
            out << "contacts " << msg.contact_size() << "\n";
            for (int contactIdx = 0; contactIdx < msg.contact_size(); ++contactIdx) {
              const auto &contact = msg.contact(contactIdx);
              out << "contact " << names[agentIdx] << " "
                  << contact.collision1().name() << " "
                  << contact.collision2().name() << "\n";
            }
          }
        }
      };
      node.Subscribe<gz::msgs::Contacts>(topic, cb);
    }
  }

  std::string line;
  unsigned long long frame = 0u;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (line == "quit" || line == "QUIT" || line == "exit") break;

    std::istringstream ss(line);
    std::string mode = "pose";
    if (!line.empty() && (std::isalpha(static_cast<unsigned char>(line[0])) != 0)) {
      ss >> mode;
    }
    std::vector<double> values;
    values.reserve(static_cast<size_t>(agentCount) * 7u);
    double value = 0.0;
    while (ss >> value) values.push_back(value);

    if (mode == "twist" || mode == "cmd_vel" || mode == "velocity") {
      const size_t expectedTwist = static_cast<size_t>(agentCount) * 6u;
      if (values.size() != expectedTwist) {
        std::cerr << "invalid twist line at frame " << frame
                  << ": got " << values.size() << " values, expected " << expectedTwist << std::endl;
        return 6;
      }
      const bool wallTimeStep = wallTimeStepMs > 0u;
      if (pauseForStep || wallTimeStep) {
        gz::msgs::WorldControl pauseReq;
        pauseReq.set_pause(true);
        if (!requestWorldControl(pauseReq, "pause-before-twist")) {
          return 9;
        }
      }
      for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
        const size_t base = static_cast<size_t>(agentIdx) * 6u;
        gz::msgs::Twist msg;
        auto *linear = msg.mutable_linear();
        linear->set_x(values[base + 0]);
        linear->set_y(values[base + 1]);
        linear->set_z(values[base + 2]);
        auto *angular = msg.mutable_angular();
        angular->set_x(values[base + 3]);
        angular->set_y(values[base + 4]);
        angular->set_z(values[base + 5]);
        twistPubs[agentIdx].Publish(msg);
      }
      if (preStepSleepMs > 0u) {
        std::this_thread::sleep_for(std::chrono::milliseconds(preStepSleepMs));
      }
      if (wallTimeStep) {
        gz::msgs::WorldControl runReq;
        runReq.set_pause(false);
        if (!requestWorldControl(runReq, "run-wall-time-step")) {
          return 9;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(wallTimeStepMs));
        gz::msgs::WorldControl pauseReq;
        pauseReq.set_pause(true);
        if (!requestWorldControl(pauseReq, "pause-after-wall-time-step")) {
          return 9;
        }
      } else if (stepAfterTwist > 0u) {
        gz::msgs::WorldControl req;
        req.set_multi_step(stepAfterTwist);
        if (!requestWorldControl(req, "multi_step")) {
          std::cerr << "world control multi_step failed at frame " << frame
                    << " steps=" << stepAfterTwist << std::endl;
          return 8;
        }
      }
      if (pauseForStep && !wallTimeStep) {
        gz::msgs::WorldControl pauseReq;
        pauseReq.set_pause(true);
        if (!requestWorldControl(pauseReq, "pause-after-step")) {
          return 9;
        }
      }
      if (postStepSleepMs > 0u) {
        std::this_thread::sleep_for(std::chrono::milliseconds(postStepSleepMs));
      }
      unsigned long long ackStateFrame = 0u;
      if (!stateFile.empty()) {
        std::lock_guard<std::mutex> lock(stateMutex);
        ackStateFrame = stateFrame;
        writeStateFileLocked();
      }
      if (ack) {
        std::cout << "ok " << frame << " twist contact=" << (contactDetected.load() ? 1 : 0)
                  << " state_frame=" << ackStateFrame << std::endl;
      }
      ++frame;
      continue;
    }

    if (mode != "pose") {
      std::cerr << "unknown stream line mode at frame " << frame << ": " << mode << std::endl;
      return 7;
    }

    const size_t expected = static_cast<size_t>(agentCount) * 7u;
    if (values.size() != expected) {
      std::cerr << "invalid pose line at frame " << frame
                << ": got " << values.size() << " values, expected " << expected << std::endl;
      return 3;
    }

    gz::msgs::Pose_V req;
    for (unsigned int agentIdx = 0; agentIdx < agentCount; ++agentIdx) {
      const size_t base = static_cast<size_t>(agentIdx) * 7u;
      auto *pose = req.add_pose();
      pose->set_name(names[agentIdx]);
      auto *pos = pose->mutable_position();
      pos->set_x(values[base + 0]);
      pos->set_y(values[base + 1]);
      pos->set_z(values[base + 2]);
      auto *quat = pose->mutable_orientation();
      quat->set_w(values[base + 3]);
      quat->set_x(values[base + 4]);
      quat->set_y(values[base + 5]);
      quat->set_z(values[base + 6]);
    }

    gz::msgs::Boolean rep;
    bool result = false;
    const bool executed = node.Request(service, req, timeoutMs, rep, result);
    if (!executed || !result || !rep.data()) {
      std::cerr << "set_pose_vector failed at frame " << frame
                << " executed=" << executed << " result=" << result
                << " reply=" << rep.data() << std::endl;
      return 4;
    }
    unsigned long long ackStateFrame = 0u;
    if (!stateFile.empty()) {
      std::lock_guard<std::mutex> lock(stateMutex);
      ackStateFrame = stateFrame;
      writeStateFileLocked();
    }
    if (ack) {
      std::cout << "ok " << frame << " pose contact=" << (contactDetected.load() ? 1 : 0)
                << " state_frame=" << ackStateFrame << std::endl;
    }
    ++frame;
  }

  return 0;
}
'''


def _vendor_cmake_prefix_path() -> str:
    vendor_prefixes = ["/opt/ros/jazzy"]
    vendor_root = Path("/opt/ros/jazzy/opt")
    if vendor_root.exists():
        vendor_prefixes.extend(str(path) for path in sorted(vendor_root.glob("gz_*_vendor")) if path.is_dir())
    return ":".join(vendor_prefixes)


def write_bridge_project(output_dir: Path) -> Dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    src = output_dir / "pose_stream_bridge.cpp"
    cmake = output_dir / "CMakeLists.txt"
    src.write_text(CPP_SOURCE.strip() + "\n", encoding="utf-8")
    cmake.write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.16)",
                "project(matd3_pose_stream_bridge LANGUAGES CXX)",
                "set(CMAKE_CXX_STANDARD 17)",
                "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
                "find_package(gz-transport13 REQUIRED)",
                "find_package(gz-msgs10 REQUIRED)",
                "add_executable(pose_stream_bridge pose_stream_bridge.cpp)",
                "target_link_libraries(pose_stream_bridge PRIVATE gz-transport13::gz-transport13 gz-msgs10::gz-msgs10)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"source": str(src), "cmake": str(cmake)}


def build_bridge(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    build_dir = output_dir / "pose_stream_bridge_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prefix = _vendor_cmake_prefix_path()
    if env.get("CMAKE_PREFIX_PATH"):
        env["CMAKE_PREFIX_PATH"] = prefix + ":" + env["CMAKE_PREFIX_PATH"]
    else:
        env["CMAKE_PREFIX_PATH"] = prefix
    subprocess.run(["cmake", "-S", str(output_dir), "-B", str(build_dir)], check=True, env=env)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", "2"], check=True, env=env)
    exe = build_dir / "pose_stream_bridge"
    if not exe.exists():
        raise FileNotFoundError(f"build completed but executable missing: {exe}")
    return exe


def create_live_pose_bridge_project(output_dir: Path, compile_bridge: bool = False) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    project = write_bridge_project(output_dir)
    exe = build_bridge(output_dir) if compile_bridge else None
    executable_path = exe if exe else output_dir / "pose_stream_bridge_build" / "pose_stream_bridge"
    meta = {
        "project": project,
        "executable": str(exe) if exe else None,
        "run_command": f"{executable_path} --world matd3_static_scene --agent-count 3",
    }
    meta_path = output_dir / "pose_stream_bridge_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    meta["meta_path"] = str(meta_path)
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and optionally compile the live Gazebo pose stream bridge.")
    parser.add_argument("--output-dir", default="gazebo_live_pose_bridge_runtime")
    parser.add_argument("--compile", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    meta = create_live_pose_bridge_project(Path(args.output_dir), compile_bridge=bool(args.compile))
    print(f"[gazebo_live_pose_bridge_builder] meta={meta['meta_path']}")
    print(f"[gazebo_live_pose_bridge_builder] source={meta['project']['source']}")
    if meta.get("executable"):
        print(f"[gazebo_live_pose_bridge_builder] executable={meta['executable']}")
        print(f"[gazebo_live_pose_bridge_builder] run={meta['run_command']}")
    else:
        print("[gazebo_live_pose_bridge_builder] compile skipped; rerun with --compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

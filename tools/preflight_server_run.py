#!/usr/bin/env python3
"""Preflight checks for running this repo from a GitHub clone or model bundle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Finding:
    level: str
    message: str


class Preflight:
    def __init__(self, repo_root: Path, skip_git_checks: bool) -> None:
        self.repo_root = repo_root
        self.skip_git_checks = skip_git_checks
        self.findings: list[Finding] = []
        self.git_available = (repo_root / ".git").exists() and shutil_which("git") is not None

    def add(self, level: str, message: str) -> None:
        self.findings.append(Finding(level=level, message=message))

    def fail(self, message: str) -> None:
        self.add("FAIL", message)

    def warn(self, message: str) -> None:
        self.add("WARN", message)

    def ok(self, message: str) -> None:
        self.add("PASS", message)

    def repo_rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def run_git(self, *args: str) -> subprocess.CompletedProcess[str] | None:
        if self.skip_git_checks or not self.git_available:
            return None
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def is_tracked(self, rel_path: str, is_dir: bool) -> bool:
        result = self.run_git("ls-files", rel_path)
        if result is None:
            return False
        output = result.stdout.strip().splitlines()
        if is_dir:
            return any(line == rel_path or line.startswith(f"{rel_path}/") for line in output)
        return any(line == rel_path for line in output)

    def git_status_entries(self, rel_paths: Iterable[str]) -> list[str]:
        paths = list(dict.fromkeys(rel_paths))
        if not paths:
            return []
        result = self.run_git("status", "--short", "--", *paths)
        if result is None or result.returncode != 0:
            return []
        return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]

    def check_path(self, rel_path: str, description: str, *, is_dir: bool = False) -> None:
        abs_path = self.repo_root / rel_path
        exists = abs_path.is_dir() if is_dir else abs_path.exists()
        if not exists:
            self.fail(f"缺少{description}: {rel_path}")
            return
        self.ok(f"存在{description}: {rel_path}")

        if self.skip_git_checks:
            return
        if not self.git_available:
            self.warn(f"未检测到 .git 或 git 命令，无法验证是否已上传 GitHub: {rel_path}")
            return
        if self.is_tracked(rel_path, is_dir=is_dir):
            self.ok(f"已被 Git 跟踪: {rel_path}")
        else:
            self.fail(f"尚未被 Git 跟踪，GitHub clone 后拿不到: {rel_path}")

    def check_clean_git(self, rel_paths: Iterable[str], label: str) -> None:
        if self.skip_git_checks or not self.git_available:
            return
        entries = self.git_status_entries(rel_paths)
        if not entries:
            self.ok(f"{label} 对应路径工作区已干净")
            return
        preview = "; ".join(entries[:6])
        if len(entries) > 6:
            preview += f"; ... 共 {len(entries)} 项"
        self.fail(
            f"{label} 对应路径仍有未提交/未跟踪改动，当前 GitHub 版本未必可运行: {preview}"
        )

    def summarize(self) -> int:
        fail_count = sum(1 for item in self.findings if item.level == "FAIL")
        warn_count = sum(1 for item in self.findings if item.level == "WARN")
        pass_count = sum(1 for item in self.findings if item.level == "PASS")

        for item in self.findings:
            prefix = {
                "PASS": "[PASS]",
                "WARN": "[WARN]",
                "FAIL": "[FAIL]",
            }[item.level]
            print(f"{prefix} {item.message}")

        print("")
        print(
            f"Summary: pass={pass_count} warn={warn_count} fail={fail_count}"
        )
        return 1 if fail_count else 0


def shutil_which(cmd: str) -> str | None:
    result = subprocess.run(
        ["bash", "-lc", f"command -v {cmd} >/dev/null 2>&1 && command -v {cmd}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def resolve_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def detect_algorithm(model_dir: Path | None, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if model_dir is None:
        return "matd3"
    if (model_dir / "mappo_meta.json").exists() or (model_dir / "value_critic.weights.h5").exists():
        return "mappo"
    if list(model_dir.glob("critic1_*.weights.h5")):
        return "matd3"
    return "maddpg"


def check_code_mode(preflight: Preflight, algorithm: str) -> list[str]:
    required_files = [
        ("run_optimized.sh", "训练入口脚本"),
        ("run_evaluation.sh", "评估入口脚本"),
        ("run_with_conda.sh", "conda 启动脚本"),
        ("paper3d_train_optimized.py", "主训练脚本"),
        ("evaluate_optimized.py", "主评估脚本"),
        ("potential_field_corrector.py", "势场修正器"),
        ("requirements.txt", "运行依赖清单"),
        ("setup_conda_env.sh", "环境安装脚本"),
        ("repair_conda_env.sh", "环境修复脚本"),
        ("REPRODUCE_ENVIRONMENT.md", "复现说明文档"),
        ("GITHUB_REPRO_CHECKLIST.md", "GitHub 最小清单"),
        ("tools/check_tf_env.py", "TensorFlow 运行时检查脚本"),
        ("tools/check_server_env.sh", "服务器环境检查脚本"),
        ("saved_positions/5.json", "默认固定位置文件"),
    ]
    required_dirs = [
        ("multiagent", "环境实现目录"),
        ("agents", "算法代理目录"),
        ("core", "核心模块目录"),
        ("utils", "工具目录"),
        ("visualization", "可视化目录"),
        ("src/multiagent", "可编辑安装包目录"),
    ]

    rel_paths: list[str] = []
    for rel_path, description in required_files:
        preflight.check_path(rel_path, description)
        rel_paths.append(rel_path)
    for rel_path, description in required_dirs:
        preflight.check_path(rel_path, description, is_dir=True)
        rel_paths.append(rel_path)

    if algorithm == "mappo":
        mappo_required = [
            ("algorithms", "算法接口目录", True),
            ("algorithms/mappo", "MAPPO 算法目录", True),
            ("train_mappo_strict.py", "MAPPO 训练脚本", False),
            ("evaluate_mappo.py", "MAPPO 评估脚本", False),
        ]
        for rel_path, description, is_dir in mappo_required:
            preflight.check_path(rel_path, description, is_dir=is_dir)
            rel_paths.append(rel_path)
    else:
        preflight.ok(f"算法模式为 {algorithm}，不强制要求 MAPPO 文件")

    return rel_paths


def find_results_candidates(repo_root: Path, model_dir: Path) -> tuple[Path | None, list[Path]]:
    parent = model_dir.parent
    direct = parent / "results.json"
    matches: list[Path] = []
    if direct.exists():
        matches.append(direct)

    logs_dir = repo_root / "logs"
    exp_name = parent.name
    if logs_dir.exists():
        matches.extend(logs_dir.glob(f"{exp_name}/**/results.json"))
        matches.extend(logs_dir.glob(f"**/*{exp_name}*/**/results.json"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in matches:
        resolved = str(item.resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(item.resolve())

    direct_resolved = direct.resolve()
    if direct.exists():
        return direct_resolved, deduped
    return None, deduped


def check_eval_mode(
    preflight: Preflight,
    algorithm: str,
    model_dir: Path | None,
    positions_file: Path | None,
    resume: bool,
) -> list[str]:
    rel_paths: list[str] = []
    if model_dir is None:
        preflight.fail("eval-existing 模式必须提供 --model-dir")
        return rel_paths

    if not model_dir.exists() or not model_dir.is_dir():
        preflight.fail(f"模型目录不存在: {model_dir}")
        return rel_paths
    preflight.ok(f"存在模型目录: {preflight.repo_rel(model_dir)}")

    actor_files = sorted(model_dir.glob("actor_*.weights.h5"))
    if not actor_files:
        preflight.fail(f"模型目录缺少 actor 权重: {preflight.repo_rel(model_dir)}")
    else:
        preflight.ok(f"检测到 actor 权重 {len(actor_files)} 个: {preflight.repo_rel(model_dir)}")

    if algorithm == "matd3":
        critic1 = sorted(model_dir.glob("critic1_*.weights.h5"))
        critic2 = sorted(model_dir.glob("critic2_*.weights.h5"))
        if not critic1 or not critic2:
            preflight.fail(
                f"MATD3 评估建议至少包含 critic1/critic2 权重: {preflight.repo_rel(model_dir)}"
            )
        else:
            preflight.ok(
                f"检测到 MATD3 critic 权重 critic1={len(critic1)} critic2={len(critic2)}"
            )
    elif algorithm == "maddpg":
        critics = sorted(model_dir.glob("critic_*.weights.h5"))
        if not critics:
            preflight.fail(f"MADDPG 评估缺少 critic 权重: {preflight.repo_rel(model_dir)}")
        else:
            preflight.ok(f"检测到 MADDPG critic 权重 {len(critics)} 个")
    elif algorithm == "mappo":
        if not (model_dir / "actor_0.weights.h5").exists():
            preflight.fail(f"MAPPO 评估缺少共享 actor 权重: {preflight.repo_rel(model_dir / 'actor_0.weights.h5')}")
        else:
            preflight.ok("检测到 MAPPO 共享 actor 权重")
        optional_paths = [
            model_dir / "value_critic.weights.h5",
            model_dir / "mappo_meta.json",
            model_dir / "actor_log_std.npy",
        ]
        for optional_path in optional_paths:
            if optional_path.exists():
                preflight.ok(f"检测到 MAPPO 附加文件: {preflight.repo_rel(optional_path)}")
            else:
                preflight.warn(f"MAPPO 附加文件缺失（通常建议一并提供）: {preflight.repo_rel(optional_path)}")

    if resume:
        resume_required = [model_dir / "checkpoint_state.json"]
        if algorithm == "matd3":
            grouped = {
                "target_actor_*.weights.h5": sorted(model_dir.glob("target_actor_*.weights.h5")),
                "target_critic1_*.weights.h5": sorted(model_dir.glob("target_critic1_*.weights.h5")),
                "target_critic2_*.weights.h5": sorted(model_dir.glob("target_critic2_*.weights.h5")),
            }
        elif algorithm == "maddpg":
            grouped = {
                "target_actor_*.weights.h5": sorted(model_dir.glob("target_actor_*.weights.h5")),
                "target_critic_*.weights.h5": sorted(model_dir.glob("target_critic_*.weights.h5")),
            }
        else:
            grouped = {}

        for path in resume_required:
            if not path.exists():
                preflight.fail(f"续训缺少文件: {preflight.repo_rel(path)}")
            else:
                preflight.ok(f"续训文件存在: {preflight.repo_rel(path)}")
        for pattern, matches in grouped.items():
            if not matches:
                preflight.fail(
                    f"续训缺少匹配文件: {preflight.repo_rel(model_dir / pattern)}"
                )
            else:
                preflight.ok(
                    f"续训检测到 {pattern} 共 {len(matches)} 个"
                )

    direct_results, all_results = find_results_candidates(preflight.repo_root, model_dir)
    if direct_results is not None and direct_results.exists():
        preflight.ok(f"模型目录旁已带训练配置: {preflight.repo_rel(direct_results)}")
    else:
        if all_results:
            preview = ", ".join(preflight.repo_rel(path) for path in all_results[:3])
            extra = "" if len(all_results) <= 3 else f", ... 共 {len(all_results)} 个候选"
            preflight.fail(
                "模型目录旁缺少 results.json。当前只能依赖 logs 中候选，服务器若不带 logs 会失败: "
                f"{preview}{extra}"
            )
        else:
            preflight.fail(
                f"未找到训练配置 results.json，严格评估将失败: {preflight.repo_rel(model_dir.parent / 'results.json')}"
            )

    if positions_file is None:
        positions_file = preflight.repo_root / "saved_positions/5.json"
    if positions_file.exists():
        preflight.ok(f"固定位置文件存在: {preflight.repo_rel(positions_file)}")
    else:
        preflight.fail(f"固定位置文件不存在: {positions_file}")

    if not preflight.skip_git_checks and preflight.git_available:
        parent_rel = preflight.repo_rel(model_dir.parent)
        rel_paths.extend([parent_rel, preflight.repo_rel(positions_file)])
    return rel_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="检查当前仓库是否已准备好在服务器上运行。")
    parser.add_argument(
        "--mode",
        choices=("code", "eval-existing"),
        default="code",
        help="code=GitHub clone 后重新训练/评估；eval-existing=直接评估已有模型包。",
    )
    parser.add_argument(
        "--algorithm",
        choices=("auto", "matd3", "maddpg", "mappo"),
        default="auto",
        help="算法类型；code 模式下 auto 会按 matd3 检查。",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="eval-existing 模式使用的模型目录，例如 models/<exp>/best_by_team_sr",
    )
    parser.add_argument(
        "--positions-file",
        default="saved_positions/5.json",
        help="固定位置文件路径（相对仓库根目录或绝对路径）。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="如果要继续续训，而不是只做评估，则额外检查 checkpoint/target 网络文件。",
    )
    parser.add_argument(
        "--skip-git-checks",
        action="store_true",
        help="跳过 Git 跟踪与工作区干净度检查；适合离线目录或打包目录自检。",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = resolve_path(repo_root, args.model_dir)
    positions_file = resolve_path(repo_root, args.positions_file)
    algorithm = detect_algorithm(model_dir, args.algorithm)

    preflight = Preflight(repo_root=repo_root, skip_git_checks=args.skip_git_checks)

    print(f"Repo root: {repo_root}")
    print(f"Mode: {args.mode}")
    print(f"Algorithm: {algorithm}")
    if model_dir is not None:
        print(f"Model dir: {model_dir}")
    print(f"Positions file: {positions_file}")
    print("")

    rel_paths = check_code_mode(preflight, algorithm)
    if args.mode == "eval-existing":
        rel_paths.extend(
            check_eval_mode(
                preflight=preflight,
                algorithm=algorithm,
                model_dir=model_dir,
                positions_file=positions_file,
                resume=args.resume,
            )
        )

    preflight.check_clean_git(rel_paths, "服务器运行关键路径")
    return preflight.summarize()


if __name__ == "__main__":
    raise SystemExit(main())

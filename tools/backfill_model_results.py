#!/usr/bin/env python3
"""Mirror matching logs/*/results.json into models/<exp>/results.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_exp_name(results_path: Path) -> str | None:
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        args = data.get("args")
        if isinstance(args, dict) and args.get("exp_name") is not None:
            return str(args.get("exp_name"))
        if data.get("exp_name") is not None:
            return str(data.get("exp_name"))
    return None


def find_candidates(repo_root: Path, exp_name: str) -> list[Path]:
    logs_root = repo_root / "logs" / exp_name
    if not logs_root.exists():
        return []

    candidates = []
    for path in logs_root.rglob("results.json"):
        loaded = load_exp_name(path)
        if loaded == exp_name:
            candidates.append(path.resolve())
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def backfill_one(model_exp_dir: Path, overwrite: bool, dry_run: bool) -> tuple[str, str]:
    target = model_exp_dir / "results.json"
    exp_name = model_exp_dir.name

    if target.exists() and not overwrite:
        return ("skip", f"已存在，跳过: {target}")

    repo_root = model_exp_dir.parents[1]
    candidates = find_candidates(repo_root, exp_name)
    if not candidates:
        return ("fail", f"未找到匹配日志 results.json: {exp_name}")

    source = candidates[0]
    if dry_run:
        return ("plan", f"{source} -> {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    extra = "" if len(candidates) == 1 else f" (候选 {len(candidates)} 个，取最新)"
    return ("ok", f"已回填: {source} -> {target}{extra}")


def main() -> int:
    parser = argparse.ArgumentParser(description="为已有模型目录回填 results.json。")
    parser.add_argument(
        "--models-root",
        default="models",
        help="模型根目录，默认 models/",
    )
    parser.add_argument(
        "--exp-name",
        action="append",
        default=[],
        help="只处理指定实验名；可重复传入。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 models/<exp>/results.json 已存在，允许覆盖。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要回填的映射，不实际复制。",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_root = (repo_root / args.models_root).resolve()
    if not models_root.exists():
        print(f"未找到模型根目录: {models_root}")
        return 1

    exp_filter = set(args.exp_name)
    model_dirs = [
        path for path in sorted(models_root.iterdir())
        if path.is_dir() and (not exp_filter or path.name in exp_filter)
    ]

    if not model_dirs:
        print("没有找到需要处理的模型实验目录。")
        return 1

    counts = {"ok": 0, "plan": 0, "skip": 0, "fail": 0}
    for model_dir in model_dirs:
        status, message = backfill_one(model_dir, overwrite=args.overwrite, dry_run=args.dry_run)
        counts[status] += 1
        prefix = {
            "ok": "[OK]",
            "plan": "[PLAN]",
            "skip": "[SKIP]",
            "fail": "[FAIL]",
        }[status]
        print(f"{prefix} {message}")

    print("")
    print(
        f"Summary: ok={counts['ok']} plan={counts['plan']} skip={counts['skip']} fail={counts['fail']}"
    )
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

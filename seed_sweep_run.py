#!/usr/bin/env python3
"""
Launch a fixed-config multi-seed training sweep for run_optimized.sh.

This runner keeps all configuration identical except SEED, launches multiple
training jobs in parallel, and writes a manifest.json that the summary script
can consume later.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a multi-seed sweep for run_optimized.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated training seeds, e.g. 101,202,303,404,505",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="seed_sweep_default",
        help="Base experiment name shared by all seeds",
    )
    parser.add_argument("--use-weighted-reward", type=int, default=1)
    parser.add_argument("--algorithm", type=str, default="matd3")
    parser.add_argument("--resume-model", type=str, default="")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="0 means launch all seeds at once",
    )
    parser.add_argument("--output-root", type=str, default="seed_sweeps")
    parser.add_argument("--run-script", type=str, default="./run_optimized.sh")
    parser.add_argument(
        "--conda-env",
        type=str,
        default=os.getenv("CONDA_ENV_NAME", "maddpg_env"),
        help="Conda env name used for each launched job (via conda run).",
    )
    parser.add_argument(
        "--no-conda-run",
        action="store_true",
        help="Do not wrap jobs with `conda run -n <env>`.",
    )
    parser.add_argument("--scenario-seed", type=int, default=None)
    parser.add_argument("--positions-file", type=str, default=None)
    parser.add_argument("--use-fixed-positions", type=int, default=None)
    parser.add_argument("--use-scenario-seed", type=int, default=None)
    parser.add_argument("--auto-eval", type=int, default=0)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional environment override in KEY=VALUE form. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifest, but do not launch jobs")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    seeds: List[int] = []
    for token in seed_text.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds were provided.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got: {seeds}")
    return seeds


def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value {item!r}, expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid environment key in {item!r}")
        overrides[key] = value
    return overrides


def make_batch_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_exp_name(base_exp_name: str, batch_id: str, seed: int) -> str:
    return f"{base_exp_name}__seed{seed}__sweep_{batch_id}"


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def find_log_root(logs_root: Path, expected_prefix: str) -> Optional[Path]:
    candidates = [
        p for p in logs_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{expected_prefix}_")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_run_dir(log_root: Path) -> Optional[Path]:
    metrics_files = sorted(log_root.rglob("episode_rewards.json"))
    if not metrics_files:
        return None
    metrics_files = [p for p in metrics_files if "evaluation" not in p.parts]
    if not metrics_files:
        return None
    metrics_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return metrics_files[0].parent


@dataclass
class Job:
    seed: int
    exp_name: str
    launcher_log: Path
    command: List[str]
    env: Dict[str, str]
    process: Optional[subprocess.Popen] = None
    started_at: Optional[float] = None


def build_job(
    args: argparse.Namespace,
    batch_id: str,
    batch_dir: Path,
    seed: int,
    extra_env: Dict[str, str],
) -> Job:
    exp_name = safe_exp_name(args.exp_name, batch_id, seed)
    base_cmd = [
        "bash",
        str(Path(args.run_script)),
        str(args.episodes),
        str(args.batch_size),
        exp_name,
        str(args.use_weighted_reward),
        str(args.algorithm),
    ]
    if getattr(args, "no_conda_run", False):
        command = base_cmd
    else:
        command = ["conda", "run", "-n", str(args.conda_env), *base_cmd]
    if args.resume_model:
        command.append(str(args.resume_model))

    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["AUTO_EVAL"] = str(args.auto_eval)
    if args.scenario_seed is not None:
        env["USE_SCENARIO_SEED"] = "1"
        env["SCENARIO_SEED"] = str(args.scenario_seed)
    if args.use_scenario_seed is not None:
        env["USE_SCENARIO_SEED"] = str(args.use_scenario_seed)
    if args.positions_file is not None:
        env["POSITIONS_FILE"] = str(args.positions_file)
    if args.use_fixed_positions is not None:
        env["USE_FIXED_POSITIONS"] = str(args.use_fixed_positions)
    env.update(extra_env)

    launcher_log = batch_dir / "launcher_logs" / f"seed_{seed}.log"
    return Job(
        seed=seed,
        exp_name=exp_name,
        launcher_log=launcher_log,
        command=command,
        env=env,
    )


def serialize_job(job: Job) -> Dict:
    return {
        "seed": job.seed,
        "exp_name": job.exp_name,
        "launcher_log": str(job.launcher_log),
        "command": job.command,
        "command_pretty": " ".join(shlex.quote(part) for part in job.command),
    }


def update_run_record(
    manifest: Dict,
    job: Job,
    status: str,
    returncode: Optional[int] = None,
    elapsed_sec: Optional[float] = None,
    logs_root: Optional[Path] = None,
    log_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> None:
    run_rows = manifest["runs"]
    row = next(item for item in run_rows if item["seed"] == job.seed)
    row["status"] = status
    if returncode is not None:
        row["returncode"] = int(returncode)
    if elapsed_sec is not None:
        row["elapsed_sec"] = float(elapsed_sec)
    if log_root is not None:
        row["log_root"] = str(log_root)
    if run_dir is not None:
        row["run_dir"] = str(run_dir)
        row["episode_rewards_json"] = str(run_dir / "episode_rewards.json")
        row["results_json"] = str(run_dir / "results.json")
    elif log_root is not None and logs_root is not None:
        row["log_root"] = str(log_root)


def launch_job(job: Job) -> None:
    job.launcher_log.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(job.launcher_log, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            job.command,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=job.env,
        )
    except Exception:
        log_f.close()
        raise
    job.process = process
    job.started_at = time.time()
    job._log_handle = log_f  # type: ignore[attr-defined]


def close_job_log(job: Job) -> None:
    handle = getattr(job, "_log_handle", None)
    if handle is not None:
        handle.close()
        job._log_handle = None  # type: ignore[attr-defined]


def terminate_active_jobs(active_jobs: List[Job]) -> None:
    for job in active_jobs:
        if job.process is None:
            continue
        if job.process.poll() is None:
            try:
                job.process.terminate()
            except Exception:
                pass
    time.sleep(2.0)
    for job in active_jobs:
        if job.process is None:
            continue
        if job.process.poll() is None:
            try:
                job.process.kill()
            except Exception:
                pass
        close_job_log(job)


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    extra_env = parse_env_overrides(args.env)
    max_parallel = args.max_parallel if args.max_parallel and args.max_parallel > 0 else len(seeds)

    batch_id = make_batch_id()
    batch_dir = Path(args.output_root) / f"seed_sweep_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    logs_root = Path("logs")
    logs_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd()),
        "config": {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "exp_name": args.exp_name,
            "use_weighted_reward": args.use_weighted_reward,
            "algorithm": args.algorithm,
            "resume_model": args.resume_model,
            "max_parallel": max_parallel,
            "auto_eval": args.auto_eval,
            "scenario_seed": args.scenario_seed,
            "positions_file": args.positions_file,
            "use_fixed_positions": args.use_fixed_positions,
            "use_scenario_seed": args.use_scenario_seed,
            "extra_env": extra_env,
            "seeds": seeds,
        },
        "runs": [],
    }

    jobs = [build_job(args, batch_id, batch_dir, seed, extra_env) for seed in seeds]
    for job in jobs:
        manifest["runs"].append({
            **serialize_job(job),
            "status": "pending",
        })

    manifest_path = batch_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"Batch directory: {batch_dir}")
    print(f"Manifest      : {manifest_path}")
    print(f"Seeds         : {seeds}")
    print(f"Max parallel  : {max_parallel}")
    print("")

    if args.dry_run:
        for job in jobs:
            print(f"[dry-run] seed={job.seed}")
            print(f"  cmd : {' '.join(shlex.quote(part) for part in job.command)}")
            print(f"  log : {job.launcher_log}")
        return 0

    pending_jobs = jobs[:]
    active_jobs: List[Job] = []

    def persist() -> None:
        write_json(manifest_path, manifest)

    try:
        while pending_jobs or active_jobs:
            while pending_jobs and len(active_jobs) < max_parallel:
                job = pending_jobs.pop(0)
                launch_job(job)
                update_run_record(manifest, job, status="running")
                persist()
                active_jobs.append(job)
                print(f"[launch] seed={job.seed} pid={job.process.pid} log={job.launcher_log}")

            finished: List[Job] = []
            for job in active_jobs:
                assert job.process is not None
                rc = job.process.poll()
                if rc is None:
                    continue
                close_job_log(job)
                elapsed_sec = time.time() - float(job.started_at or time.time())
                log_root = find_log_root(logs_root, job.exp_name)
                run_dir = find_run_dir(log_root) if log_root is not None else None
                status = "completed" if rc == 0 and run_dir is not None else "failed"
                update_run_record(
                    manifest,
                    job,
                    status=status,
                    returncode=rc,
                    elapsed_sec=elapsed_sec,
                    logs_root=logs_root,
                    log_root=log_root,
                    run_dir=run_dir,
                )
                persist()
                summary = f"[done] seed={job.seed} rc={rc} status={status} elapsed={elapsed_sec:.1f}s"
                if run_dir is not None:
                    summary += f" run_dir={run_dir}"
                print(summary)
                finished.append(job)

            if finished:
                active_jobs = [job for job in active_jobs if job not in finished]
            else:
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active jobs...")
        terminate_active_jobs(active_jobs)
        for job in active_jobs:
            update_run_record(manifest, job, status="terminated")
        persist()
        return 130

    succeeded = sum(1 for row in manifest["runs"] if row["status"] == "completed")
    failed = sum(1 for row in manifest["runs"] if row["status"] != "completed")
    print("")
    print(f"Completed runs: {succeeded}")
    print(f"Failed runs   : {failed}")
    print(f"Manifest path : {manifest_path}")
    print("Next step     : run seed_sweep_summary.py on this batch directory.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

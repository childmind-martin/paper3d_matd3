#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from isolated_level2_diagnosis import (
    _collect_contexts,
    _generate_trainlike_obstacle_seeds,
    _load_json,
    _materialize_spec,
    _run_isolated_eval,
)


REPO_ROOT = Path("/home/tang/matd3").resolve()
STREAM_LOCK = threading.Lock()


@dataclass(frozen=True)
class ResumeJob:
    label: str
    seed: int
    python_executable: str
    python_script: str
    argv: List[str]
    exec_env: Dict[str, str]
    current_model_root: Path
    resume_source_dir: Path
    current_log_dir: Path
    child_batch_dir: Path
    eval_python_bin: Optional[str]


def _extract_flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    items = list(argv)
    if flag not in items:
        return None
    idx = items.index(flag)
    if idx + 1 >= len(items):
        return None
    return str(items[idx + 1])


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _default_experiments(parent_batch_dir: Path) -> List[str]:
    summary = _load_json(parent_batch_dir / "plots" / "latest_summary.json")
    labels: List[str] = []
    for row in summary.get("aggregated_experiments", []) or []:
        label = str(row.get("label", "")).strip()
        if label:
            labels.append(label)
    if not labels:
        raise RuntimeError(f"No experiment labels found in: {parent_batch_dir}")
    return labels


def _replace_arg(argv: Sequence[str], flag: str, value: str) -> List[str]:
    items = list(argv)
    if flag in items:
        idx = items.index(flag)
        if idx + 1 >= len(items):
            items.append(str(value))
        else:
            items[idx + 1] = str(value)
        return items
    return items + [flag, str(value)]


def _remove_flag_with_value(argv: Sequence[str], flag: str) -> List[str]:
    items = list(argv)
    while flag in items:
        idx = items.index(flag)
        del items[idx : idx + 2]
    return items


def _remove_flag(argv: Sequence[str], flag: str) -> List[str]:
    return [item for item in argv if item != flag]


def _load_manifest_job(ctx: Any) -> ResumeJob:
    manifest = _load_json(ctx.manifest_path)
    python_executable = str(manifest.get("python_executable", "")).strip()
    python_script = str(manifest.get("python_script", "")).strip()
    argv = [str(v) for v in list(manifest.get("argv", []) or [])]
    exec_env = {str(k): str(v) for k, v in dict(manifest.get("exec_env", {}) or {}).items()}
    if not python_executable or not python_script or not argv:
        raise RuntimeError(f"Incomplete manifest: {ctx.manifest_path}")
    current_model_root = Path(ctx.model_root).resolve()
    resume_source_dir = _resolve_resume_source_dir(current_model_root)
    return ResumeJob(
        label=str(ctx.label),
        seed=int(ctx.seed),
        python_executable=python_executable,
        python_script=python_script,
        argv=argv,
        exec_env=exec_env,
        current_model_root=current_model_root,
        resume_source_dir=resume_source_dir,
        current_log_dir=Path(ctx.log_dir).resolve(),
        child_batch_dir=Path(ctx.child_batch_dir).resolve(),
        eval_python_bin=str(ctx.eval_python_bin).strip() or None,
    )


def _resolve_resume_source_dir(model_root: Path) -> Path:
    candidates = [
        model_root / "checkpoint",
        model_root / "best_by_team_sr",
        model_root / "final",
        model_root / "best",
        model_root,
    ]
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        has_state = (candidate / "checkpoint_state.json").exists()
        has_weights = (candidate / "actor_0.weights.h5").exists()
        if has_state and has_weights:
            return candidate.resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(f"No valid resume source directory found under: {model_root}")


def _resume_family(job: ResumeJob) -> str:
    script_name = Path(job.python_script).name
    if script_name == "train_mappo_strict.py":
        return "mappo"
    algo = str(_extract_flag_value(job.argv, "--algo") or "").strip().lower()
    if algo == "matd3":
        return "matd3"
    return "maddpg"


def _validate_resume_source_dir(job: ResumeJob) -> List[str]:
    source = job.resume_source_dir
    missing: List[str] = []
    if not source.exists() or not source.is_dir():
        return [f"resume source dir missing: {source}"]
    family = _resume_family(job)
    required = ["checkpoint_state.json", "actor_0.weights.h5"]
    if family == "mappo":
        required += ["value_critic.weights.h5", "actor_log_std.npy"]
    elif family == "matd3":
        required += ["critic1_0.weights.h5", "critic2_0.weights.h5"]
    else:
        required += ["critic_0.weights.h5"]
    for name in required:
        if not (source / name).exists():
            missing.append(name)
    return missing


def _discover_results_dir(log_root: Path) -> Optional[Path]:
    if not log_root.exists():
        return None
    direct = log_root / "results.json"
    if direct.exists():
        return log_root
    matches = sorted(log_root.glob("**/results.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1].parent


def _compute_tail100_success_mean(results_dir: Optional[Path]) -> Optional[float]:
    if results_dir is None:
        return None
    payload_path = results_dir / "episode_rewards.json"
    if not payload_path.exists():
        return None
    try:
        payload = _load_json(payload_path)
    except Exception:
        return None
    flags = list(payload.get("team_success_flags", []) or [])
    if not flags:
        return None
    tail = flags[-100:]
    if not tail:
        return None
    try:
        return float(sum(float(v) for v in tail) / len(tail))
    except Exception:
        return None


def _list_matching_dirs(root: Path, prefix: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )


def _resolve_new_dir(root: Path, prefix: str, existing: Sequence[Path]) -> Optional[Path]:
    existing_set = {p.resolve() for p in existing}
    matches = _list_matching_dirs(root, prefix)
    new_matches = [p for p in matches if p.resolve() not in existing_set]
    if new_matches:
        return new_matches[-1]
    return matches[-1] if matches else None


def _flatten_eval_summary(prefix: str, summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = dict(summary or {})
    keys = [
        "team_success_rate",
        "collision_free_rate",
        "avg_reward",
        "avg_collision_count",
        "avg_agent_final_goal_distance",
        "avg_first_reach_step",
    ]
    return {f"{prefix}_{key}": summary.get(key) for key in keys}


def _flatten_train_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = dict(summary or {})
    keys = [
        "episodes",
        "best_reward",
        "best_episode",
        "team_success_rate",
        "best_team_success_rate",
        "best_team_sr_episode",
    ]
    return {f"train_{key}": summary.get(key) for key in keys}


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)
    if not all_keys:
        all_keys = ["empty"]
        rows = [{"empty": ""}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _preview_command(cmd: Sequence[str], env: Dict[str, str]) -> str:
    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items()))
    command = shlex.join([str(part) for part in cmd])
    return f"{env_prefix} {command}".strip()


def _build_train_distribution_spec(base_spec: Dict[str, Any], positions_root: Path) -> Dict[str, Any]:
    terrain_seed = int(base_spec.get("terrain_seed", base_spec.get("scenario_seed", 0)) or 0)
    return _materialize_spec(
        base_spec,
        output_positions_dir=positions_root,
        episodes=50,
        seed=terrain_seed,
        episode_length_multiplier=1.0,
        obstacle_mode="trainlike",
    )


def _stream_lines(prefix: str, process: subprocess.Popen[str], log_file: Any) -> int:
    assert process.stdout is not None
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        with STREAM_LOCK:
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    process.stdout.close()
    return int(process.wait())


def _run_subprocess(
    *,
    cmd: Sequence[str],
    env: Dict[str, str],
    cwd: Path,
    log_path: Path,
    stream_output: bool,
    stream_prefix: str,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        if not stream_output:
            proc = subprocess.run(
                [str(x) for x in cmd],
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return int(proc.returncode)

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        return _stream_lines(stream_prefix, proc, log_file)


def _build_train_command(job: ResumeJob, target_episodes: int, new_exp_name: str) -> List[str]:
    argv = list(job.argv)
    argv = _remove_flag_with_value(argv, "--checkpoint")
    argv = _remove_flag(argv, "--resume")
    argv = _replace_arg(argv, "--train-episodes", str(int(target_episodes)))
    argv = _replace_arg(argv, "--exp-name", str(new_exp_name))
    argv += ["--checkpoint", str(job.resume_source_dir), "--resume"]
    return [job.python_executable, job.python_script] + argv


def _train_env(job: ResumeJob, stream_train_output: bool) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(job.exec_env)
    if stream_train_output:
        env["PYTHONUNBUFFERED"] = "1"
    return env


def _build_eval_jobs(
    *,
    job: ResumeJob,
    new_model_root: Path,
    output_root: Path,
    model_variant: str,
    quiet_eval: bool,
    include_train_distribution_eval: bool,
) -> List[Tuple[str, Dict[str, Any], Path]]:
    shared_spec_path = job.child_batch_dir / "results" / "post_eval_shared_spec.json"
    base_spec = _load_json(shared_spec_path)

    official_spec = json.loads(json.dumps(base_spec))
    official_dir = output_root / "evals" / job.label / f"seed_{job.seed}" / "official_level2_test"
    jobs: List[Tuple[str, Dict[str, Any], Path]] = [("official", official_spec, official_dir)]

    if include_train_distribution_eval:
        replay_positions_dir = (
            output_root / "generated_testsets" / job.label / f"seed_{job.seed}" / "train_distribution_replay" / "episode_positions"
        )
        train_distribution_spec = _build_train_distribution_spec(base_spec, replay_positions_dir)
        train_distribution_dir = output_root / "evals" / job.label / f"seed_{job.seed}" / "train_distribution_replay"
        jobs.append(("train_distribution", train_distribution_spec, train_distribution_dir))

    return jobs


def _run_job(
    job: ResumeJob,
    *,
    output_root: Path,
    target_episodes: int,
    run_stamp: str,
    run_eval: bool,
    model_variant: str,
    quiet_eval: bool,
    include_train_distribution_eval: bool,
    stream_train_output: bool,
    stream_eval_output: bool,
    print_only: bool,
) -> Dict[str, Any]:
    job_root = output_root / "job_artifacts" / job.label / f"seed_{job.seed}"
    job_root.mkdir(parents=True, exist_ok=True)

    new_exp_name = f"{job.current_model_root.name}__continue{int(target_episodes)}_{run_stamp}"
    train_cmd = _build_train_command(job, target_episodes=target_episodes, new_exp_name=new_exp_name)
    train_env = _train_env(job, stream_train_output=stream_train_output)
    train_log_path = job_root / "train_resume.log"

    existing_model_dirs = _list_matching_dirs(REPO_ROOT / "models", new_exp_name)
    existing_log_dirs = _list_matching_dirs(REPO_ROOT / "logs", new_exp_name)

    preview: Dict[str, Any] = {
        "label": job.label,
        "seed": job.seed,
        "train_command": train_cmd,
        "train_env_keys": sorted(train_env.keys()),
        "train_log_path": str(train_log_path),
        "current_model_root": str(job.current_model_root),
        "resume_source_dir": str(job.resume_source_dir),
        "current_log_dir": str(job.current_log_dir),
        "new_exp_name": new_exp_name,
    }
    preflight_missing = _validate_resume_source_dir(job)
    preview["resume_preflight_missing"] = list(preflight_missing)
    (job_root / "job_preview.json").write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")

    row: Dict[str, Any] = {
        "label": job.label,
        "seed": job.seed,
        "status": "pending",
        "current_model_root": str(job.current_model_root),
        "resume_source_dir": str(job.resume_source_dir),
        "current_log_dir": str(job.current_log_dir),
        "train_log_path": str(train_log_path),
        "new_exp_name": new_exp_name,
    }
    row["resume_preflight_missing"] = ",".join(preflight_missing) if preflight_missing else ""
    row["resume_family"] = _resume_family(job)

    if preflight_missing:
        row["status"] = "resume_source_invalid"
        row["error"] = f"Resume source missing required files: {preflight_missing}"
        return row

    if print_only:
        row["status"] = "print_only"
        row["train_command_preview"] = _preview_command(train_cmd, train_env)
        if run_eval:
            shared_spec = _load_json(job.child_batch_dir / "results" / "post_eval_shared_spec.json")
            row["official_eval_save_dir"] = str(
                output_root / "evals" / job.label / f"seed_{job.seed}" / "official_level2_test"
            )
            row["official_obstacle_seed_count"] = len(list(shared_spec.get("obstacle_seed_sequence", []) or []))
            if include_train_distribution_eval:
                replay_positions_dir = (
                    output_root
                    / "generated_testsets"
                    / job.label
                    / f"seed_{job.seed}"
                    / "train_distribution_replay"
                    / "episode_positions"
                )
                replay_spec = _build_train_distribution_spec(shared_spec, replay_positions_dir)
                row["train_distribution_eval_save_dir"] = str(
                    output_root / "evals" / job.label / f"seed_{job.seed}" / "train_distribution_replay"
                )
                row["train_distribution_obstacle_seed_head"] = ",".join(
                    str(x) for x in list(replay_spec.get("obstacle_seed_sequence", []) or [])[:5]
                )
        return row

    train_returncode = _run_subprocess(
        cmd=train_cmd,
        env=train_env,
        cwd=REPO_ROOT,
        log_path=train_log_path,
        stream_output=stream_train_output,
        stream_prefix=f"{job.label}/seed{job.seed} train",
    )
    row["train_returncode"] = train_returncode
    if train_returncode != 0:
        row["status"] = "train_failed"
        row["error"] = f"Training resume failed with rc={train_returncode}"
        return row

    new_model_root = _resolve_new_dir(REPO_ROOT / "models", new_exp_name, existing_model_dirs)
    new_log_root = _resolve_new_dir(REPO_ROOT / "logs", new_exp_name, existing_log_dirs)
    if new_model_root is None or new_log_root is None:
        row["status"] = "artifact_missing"
        row["error"] = "Could not resolve new model/log directories after resume."
        row["new_model_root"] = str(new_model_root) if new_model_root else ""
        row["new_log_root"] = str(new_log_root) if new_log_root else ""
        return row

    train_results_dir = _discover_results_dir(new_log_root)
    train_results_path = train_results_dir / "results.json" if train_results_dir else None
    train_summary = _load_json(train_results_path) if train_results_path and train_results_path.exists() else {}
    tail100_success = _compute_tail100_success_mean(train_results_dir)

    row.update(
        {
            "status": "train_done",
            "new_model_root": str(new_model_root),
            "new_log_root": str(new_log_root),
            "train_results_dir": str(train_results_dir) if train_results_dir else "",
            "train_results_path": str(train_results_path) if train_results_path else "",
            "train_tail100_success_mean": tail100_success,
        }
    )
    row.update(_flatten_train_summary(train_summary))

    if not run_eval:
        row["status"] = "done"
        return row

    eval_records: Dict[str, Dict[str, Any]] = {}
    try:
        for eval_name, spec, save_dir in _build_eval_jobs(
            job=job,
            new_model_root=new_model_root,
            output_root=output_root,
            model_variant=model_variant,
            quiet_eval=quiet_eval,
            include_train_distribution_eval=include_train_distribution_eval,
        ):
            record = _run_isolated_eval(
                model_dir=new_model_root,
                variant=model_variant,
                spec=spec,
                save_dir=save_dir,
                eval_python_bin=job.eval_python_bin,
                quiet_eval=quiet_eval,
                dry_run=False,
                stream_output=stream_eval_output,
                stream_prefix=f"{job.label}/seed{job.seed} {eval_name}",
            )
            eval_records[eval_name] = record
    except Exception as exc:
        row["status"] = "eval_failed"
        row["error"] = str(exc)
        return row

    official_record = eval_records.get("official", {})
    replay_record = eval_records.get("train_distribution", {})
    row["status"] = "done"
    row["official_eval_dir"] = str(official_record.get("save_dir", ""))
    row["train_distribution_eval_dir"] = str(replay_record.get("save_dir", ""))
    row.update(_flatten_eval_summary("official", official_record.get("summary")))
    row.update(_flatten_eval_summary("train_distribution", replay_record.get("summary")))
    return row


def _execute_jobs(
    jobs: Sequence[ResumeJob],
    *,
    output_root: Path,
    target_episodes: int,
    run_stamp: str,
    run_eval: bool,
    model_variant: str,
    quiet_eval: bool,
    include_train_distribution_eval: bool,
    stream_train_output: bool,
    stream_eval_output: bool,
    max_parallel: int,
    print_only: bool,
) -> List[Dict[str, Any]]:
    queued = list(jobs)
    if not queued:
        return []

    total = len(queued)
    workers = max(1, int(max_parallel))
    rows: List[Dict[str, Any]] = []

    if print_only or workers <= 1 or total <= 1:
        for idx, job in enumerate(queued, start=1):
            print(f"[job {idx}/{total}] {job.label}/seed{job.seed}", flush=True)
            rows.append(
                _run_job(
                    job,
                    output_root=output_root,
                    target_episodes=target_episodes,
                    run_stamp=run_stamp,
                    run_eval=run_eval,
                    model_variant=model_variant,
                    quiet_eval=quiet_eval,
                    include_train_distribution_eval=include_train_distribution_eval,
                    stream_train_output=stream_train_output,
                    stream_eval_output=stream_eval_output,
                    print_only=print_only,
                )
            )
        return rows

    print(f"Launching {total} resume jobs with max_parallel={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                output_root=output_root,
                target_episodes=target_episodes,
                run_stamp=run_stamp,
                run_eval=run_eval,
                model_variant=model_variant,
                quiet_eval=quiet_eval,
                include_train_distribution_eval=include_train_distribution_eval,
                stream_train_output=stream_train_output,
                stream_eval_output=stream_eval_output,
                print_only=False,
            ): job
            for job in queued
        }
        completed = 0
        for future in as_completed(futures):
            job = futures[future]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "label": job.label,
                    "seed": job.seed,
                    "status": "crashed",
                    "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                }
            rows.append(row)
            print(
                f"[job {completed}/{total}] {job.label}/seed{job.seed} -> {row.get('status', 'unknown')}",
                flush=True,
            )
    return rows


def _summarize_status(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume the full Level2 ablation in parallel and batch-summarize resumed training/evaluation results."
    )
    parser.add_argument("parent_batch_dir", type=Path)
    parser.add_argument("--target-episodes", type=int, required=True)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument(
        "--skip-train-distribution-eval",
        action="store_true",
        help="When --run-eval is enabled, only run the official Level2 evaluation and skip train-distribution replay evaluation.",
    )
    parser.add_argument("--model-variant", default="best_by_team_sr")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--quiet-eval", action="store_true")
    parser.add_argument("--stream-train-output", action="store_true")
    parser.add_argument("--stream-eval-output", action="store_true")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream both resumed training output and evaluation output to the terminal.",
    )
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    parent_batch_dir = args.parent_batch_dir.resolve()
    experiments = list(args.experiments or _default_experiments(parent_batch_dir))
    stream_train_output = bool(args.stream_train_output or args.stream_output)
    stream_eval_output = bool(args.stream_eval_output or args.stream_output)

    _, _, contexts = _collect_contexts(parent_batch_dir, experiments, args.seeds)
    jobs = [_load_manifest_job(ctx) for ctx in contexts]
    jobs = sorted(jobs, key=lambda item: (item.label, item.seed))

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = REPO_ROOT / "diagnostics" / f"level2_resume_{run_stamp}"
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "summaries").mkdir(parents=True, exist_ok=True)

    metadata = {
        "parent_batch_dir": str(parent_batch_dir),
        "target_episodes": int(args.target_episodes),
        "run_eval": bool(args.run_eval),
        "include_train_distribution_eval": bool(not args.skip_train_distribution_eval),
        "model_variant": str(args.model_variant),
        "max_parallel": int(args.max_parallel),
        "quiet_eval": bool(args.quiet_eval),
        "stream_train_output": stream_train_output,
        "stream_eval_output": stream_eval_output,
        "stream_output": bool(args.stream_output),
        "print_only": bool(args.print_only),
        "jobs": [asdict(job) for job in jobs],
    }
    (output_root / "metadata" / "batch_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    rows = _execute_jobs(
        jobs,
        output_root=output_root,
        target_episodes=args.target_episodes,
        run_stamp=run_stamp,
        run_eval=bool(args.run_eval),
        model_variant=str(args.model_variant),
        quiet_eval=bool(args.quiet_eval),
        include_train_distribution_eval=bool(not args.skip_train_distribution_eval),
        stream_train_output=stream_train_output,
        stream_eval_output=stream_eval_output,
        max_parallel=int(args.max_parallel),
        print_only=bool(args.print_only),
    )

    rows = sorted(rows, key=lambda row: (str(row.get("label", "")), int(row.get("seed", 0))))
    status_counts = _summarize_status(rows)

    summary_payload = {
        "output_root": str(output_root),
        "status_counts": status_counts,
        "rows": rows,
    }
    (output_root / "summaries" / "resume_eval_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_rows_csv(output_root / "summaries" / "resume_eval_summary.csv", rows)

    print(f"Output root: {output_root}", flush=True)
    print(f"Status counts: {json.dumps(status_counts, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

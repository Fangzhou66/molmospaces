#!/usr/bin/env python3
"""MolmoSpaces-only Filament reset probe.

This intentionally bypasses Alice/Ray and measures the MolmoSpaces side of an
episode rebuild: benchmark JSON -> JsonEvalTaskSampler -> CPUMujocoEnv ->
Filament renderer/MjrContext -> task.reset().

It is useful for render/sim optimization and K-ladder checks. It cannot validate
Alice-only behavior such as offload, env reuse, router affinity, or GRPO batch
composition.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import queue
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK = (
    "/home/zekaili/.cache/molmo-spaces-resources/benchmarks/"
    "molmospaces-bench-v2/20260415/procthor-objaverse/"
    "FrankaPickandPlaceHardBench/"
    "FrankaPickandPlaceHardBench_20260206_json_benchmark"
)


@dataclass(frozen=True)
class ProbeJob:
    worker_id: int
    local_index: int
    episode_index: int
    gpu: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _parse_gpus(raw: str) -> list[str]:
    gpus = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must name at least one GPU")
    return gpus


def _configure_process_env(gpu: str, args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("EGL_PLATFORM", "device")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("ALICE_MS_FIL_LOCK_SCOPE", args.lock_scope)
    os.environ.setdefault("ALICE_MS_FIL_RESET_CONCURRENCY", str(args.k))
    os.environ.setdefault("ALICE_MS_FIL_LOCK_TIMEOUT_S", str(args.lock_timeout_s))
    os.environ.setdefault("ALICE_MS_FIL_LOCK_SHARD", "1")
    os.environ.setdefault("ALICE_MS_FIL_TEXTURE_CACHE_LOG", "1" if args.texture_log else "0")


def _build_exp_config(episode: Any, task_horizon: int, seed: int) -> Any:
    """Build the same minimal JSON eval config Alice uses for MolmoSpaces."""
    from molmo_spaces.configs.policy_configs import DummyPolicyConfig
    from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
    from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
    from molmo_spaces.evaluation.eval_main import EvalRuntimeParams

    class _ProbeEvalConfig(JsonBenchmarkEvalConfig):
        robot_config: FrankaRobotConfig = FrankaRobotConfig()
        policy_config: DummyPolicyConfig = DummyPolicyConfig()
        policy_dt_ms: float = 66.0
        ctrl_dt_ms: float = 2.0
        sim_dt_ms: float = 2.0
        task_horizon: int = task_horizon
        seed: int = seed
        filter_for_successful_trajectories: bool = False
        terminate_upon_success: bool = False

        def model_post_init(self, __context) -> None:
            super().model_post_init(__context)
            self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)

    exp_config = _ProbeEvalConfig()

    task = getattr(episode, "task", None) or {}
    is_openable = (
        "opening_tasks" in str(task.get("task_cls", ""))
        or task.get("task_type") in ("open", "close")
    )
    if is_openable:
        from molmo_spaces.configs.task_sampler_configs import OpenTaskSamplerConfig

        exp_config.task_sampler_config = OpenTaskSamplerConfig()

    exp_config.eval_runtime_params = EvalRuntimeParams()
    exp_config.num_workers = 1
    return exp_config


def _episode_horizon(episode: Any, default_horizon: int) -> int:
    task = getattr(episode, "task", None) or {}
    sec = task.get("task_horizon_sec")
    if sec is None:
        return default_horizon
    return round(float(sec) * 1000 / 66.0)


def _run_one_episode(
    episode: Any,
    episode_index: int,
    task_horizon: int,
    seed: int,
) -> dict[str, Any]:
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    row: dict[str, Any] = {
        "episode_index": episode_index,
        "house_index": getattr(episode, "house_index", ""),
        "scene_dataset": getattr(episode, "scene_dataset", ""),
        "data_split": getattr(episode, "data_split", ""),
        "task_type": (getattr(episode, "task", None) or {}).get("task_type", ""),
        "ok": 0,
        "error": "",
    }

    sampler = None
    task = None
    total_t0 = time.monotonic()
    try:
        exp_config = _build_exp_config(episode, task_horizon, seed)
        sampler_t0 = time.monotonic()
        sampler = JsonEvalTaskSampler(exp_config, episode)
        row["sampler_init_s"] = time.monotonic() - sampler_t0

        sample_t0 = time.monotonic()
        task = sampler.sample_task(house_index=episode.house_index)
        row["sample_task_s"] = time.monotonic() - sample_t0
        if task is None:
            raise RuntimeError("sample_task returned None")

        reset_t0 = time.monotonic()
        task.reset()
        row["task_reset_s"] = time.monotonic() - reset_t0
        row["ok"] = 1
    except Exception as exc:  # noqa: BLE001 - probe records failures as rows
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=8)
    finally:
        close_t0 = time.monotonic()
        try:
            if task is not None:
                task.close()
        finally:
            if sampler is not None:
                sampler.close()
        row["close_s"] = time.monotonic() - close_t0
        row["total_s"] = time.monotonic() - total_t0
    return row


def _worker_main(
    args: argparse.Namespace,
    jobs: list[ProbeJob],
    result_queue: mp.Queue,
) -> None:
    gpu = jobs[0].gpu if jobs else "0"
    _configure_process_env(gpu, args)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(processName)s pid=%(process)d "
            "%(name)s %(levelname)s: %(message)s"
        ),
        datefmt="%H:%M:%S",
        force=True,
    )

    # Import MolmoSpaces only after CUDA_VISIBLE_DEVICES and EGL env are set.
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

    benchmark_dir = Path(args.benchmark_dir)
    episodes = load_all_episodes(benchmark_dir)
    for job in jobs:
        episode = episodes[job.episode_index % len(episodes)]
        horizon = args.task_horizon or _episode_horizon(episode, args.default_task_horizon)
        row = _run_one_episode(
            episode=episode,
            episode_index=job.episode_index,
            task_horizon=horizon,
            seed=args.seed + job.episode_index,
        )
        row.update(
            {
                "worker_id": job.worker_id,
                "local_index": job.local_index,
                "gpu": gpu,
                "k": args.k,
                "pid": os.getpid(),
            }
        )
        result_queue.put(row)


def _build_jobs(args: argparse.Namespace) -> list[list[ProbeJob]]:
    gpus = _parse_gpus(args.gpus)
    worker_count = args.workers or len(gpus)
    worker_jobs: list[list[ProbeJob]] = [[] for _ in range(worker_count)]
    for i in range(args.episodes):
        worker_id = i % worker_count
        gpu = gpus[worker_id % len(gpus)]
        worker_jobs[worker_id].append(
            ProbeJob(
                worker_id=worker_id,
                local_index=len(worker_jobs[worker_id]),
                episode_index=args.start_index + i,
                gpu=gpu,
            )
        )
    return worker_jobs


def _write_summary(rows: list[dict[str, Any]], output_csv: Path) -> Path:
    summary_path = output_csv.with_suffix(".summary.json")
    ok_rows = [row for row in rows if int(row.get("ok") or 0) == 1]

    def stats(key: str) -> dict[str, float] | None:
        values = [float(row[key]) for row in ok_rows if row.get(key) not in (None, "")]
        if not values:
            return None
        values_sorted = sorted(values)
        p95_idx = min(len(values_sorted) - 1, round(0.95 * (len(values_sorted) - 1)))
        return {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": values_sorted[p95_idx],
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "failed_rows": len(rows) - len(ok_rows),
        "output_csv": str(output_csv),
        "stats": {
            key: stats(key)
            for key in (
                "total_s",
                "sampler_init_s",
                "sample_task_s",
                "task_reset_s",
                "close_s",
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe MolmoSpaces Filament reset/render timings without Alice.",
    )
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=Path("filament_reset_probe.csv"))
    parser.add_argument("--episodes", type=_positive_int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=_positive_int, default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--k", type=_positive_int, default=2)
    parser.add_argument("--lock-scope", default="context")
    parser.add_argument("--lock-timeout-s", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-horizon", type=int, default=None)
    parser.add_argument("--default-task-horizon", type=int, default=600)
    parser.add_argument("--texture-log", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    worker_jobs = _build_jobs(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    expected_rows = sum(len(jobs) for jobs in worker_jobs)

    for worker_id, jobs in enumerate(worker_jobs):
        if not jobs:
            continue
        proc = ctx.Process(
            target=_worker_main,
            args=(args, jobs, result_queue),
            name=f"probe-worker-{worker_id}",
        )
        proc.start()
        processes.append(proc)

    rows: list[dict[str, Any]] = []
    while len(rows) < expected_rows:
        try:
            row = result_queue.get(timeout=5)
        except queue.Empty:
            if all(not proc.is_alive() for proc in processes):
                break
            continue
        rows.append(row)
        status = "OK" if row.get("ok") else "FAIL"
        print(
            f"{status} worker={row.get('worker_id')} gpu={row.get('gpu')} "
            f"ep={row.get('episode_index')} total_s={float(row.get('total_s') or 0):.3f} "
            f"sample_task_s={float(row.get('sample_task_s') or 0):.3f} "
            f"task_reset_s={float(row.get('task_reset_s') or 0):.3f}",
            flush=True,
        )

    for proc in processes:
        proc.join()
    bad_exit = {proc.name: proc.exitcode for proc in processes if proc.exitcode}

    fieldnames = [
        "ok",
        "error",
        "worker_id",
        "local_index",
        "pid",
        "gpu",
        "k",
        "episode_index",
        "house_index",
        "scene_dataset",
        "data_split",
        "task_type",
        "total_s",
        "sampler_init_s",
        "sample_task_s",
        "task_reset_s",
        "close_s",
        "traceback",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = _write_summary(rows, args.output)
    print(f"wrote_csv={args.output}")
    print(f"wrote_summary={summary_path}")
    if bad_exit:
        print(f"worker_exit_errors={bad_exit}", file=sys.stderr)
        return 2
    if len(rows) != expected_rows:
        print(f"missing_rows expected={expected_rows} got={len(rows)}", file=sys.stderr)
        return 3
    if any(int(row.get("ok") or 0) != 1 for row in rows):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run four collision experiments and plot comparison curves.

Experiments:
1) sync_shared
2) sync_unshared
3) async_shared
4) async_unshared

Run from case_studies/power:
    python -m collision_case.train_four_and_plot --stop-iters 120
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExpCfg:
    name: str
    share_reward: bool
    enable_async: bool


EXPERIMENTS: List[ExpCfg] = [
    ExpCfg("sync_shared", True, False),
    ExpCfg("sync_unshared", False, False),
    ExpCfg("async_shared", True, True),
    ExpCfg("async_unshared", False, True),
]


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("collision_case") / f"batch_runs_{ts}"


def _default_dataset() -> Path:
    return Path("collision_case") / "data2024.pkl"


def _run_one(
    cfg: ExpCfg,
    args,
    output_dir: Path,
    exp_idx: int,
    total_exps: int,
    progress_csv: Path,
    global_state: dict,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{cfg.name}.csv"
    log_path = output_dir / f"{cfg.name}.log"

    cmd = [
        sys.executable,
        "-m",
        "collision_case.train_collision",
        "--algo",
        args.algo,
        "--num-agents",
        "3",
        "--dataset-path",
        str(args.dataset_path),
        "--log-path",
        str(csv_path),
        "--stop-iters",
        str(args.stop_iters),
        "--num-env-runners",
        str(args.num_env_runners),
        "--field-tick-s",
        str(args.field_tick_s),
        "--coord-tick-s",
        str(args.coord_tick_s),
        "--system-tick-s",
        str(args.system_tick_s),
        "--jitter-ratio",
        str(args.jitter_ratio),
    ]
    if cfg.share_reward:
        cmd.append("--share-reward")
    if cfg.enable_async:
        cmd.append("--enable-async")

    print(f"\n=== Running {cfg.name} ===")
    print(" ".join(cmd))
    if args.dry_run:
        return csv_path, log_path

    iter_re = re.compile(r"iter\s+(\d+)/(\d+)\s+\|")
    exp_start = time.time()
    exp_iter_times: List[float] = []
    last_iter_ts = exp_start

    def _append_progress(
        *,
        iter_idx: int,
        iter_total: int,
        eta_exp_s: float,
        eta_all_s: float,
    ) -> None:
        progress_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = progress_csv.exists()
        with progress_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "experiment_index",
                    "experiment_name",
                    "iter_idx",
                    "iter_total",
                    "global_iter_done",
                    "global_iter_total",
                    "eta_experiment_s",
                    "eta_all_s",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment_index": exp_idx,
                    "experiment_name": cfg.name,
                    "iter_idx": iter_idx,
                    "iter_total": iter_total,
                    "global_iter_done": global_state["done_iters"],
                    "global_iter_total": global_state["total_iters"],
                    "eta_experiment_s": round(max(0.0, eta_exp_s), 1),
                    "eta_all_s": round(max(0.0, eta_all_s), 1),
                }
            )

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            m = iter_re.search(line)
            if m:
                cur = int(m.group(1))
                tot = int(m.group(2))
                now = time.time()
                iter_dt = max(0.001, now - last_iter_ts)
                last_iter_ts = now
                exp_iter_times.append(iter_dt)
                avg_iter = sum(exp_iter_times) / len(exp_iter_times)
                eta_exp = avg_iter * max(0, tot - cur)

                global_done = global_state["done_iters_prefix"] + cur
                global_left = max(0, global_state["total_iters"] - global_done)
                eta_all = avg_iter * global_left
                global_state["done_iters"] = global_done

                msg = (
                    f"[Progress] Exp {exp_idx}/{total_exps} ({cfg.name}) "
                    f"iter {cur}/{tot} | global {global_done}/{global_state['total_iters']} "
                    f"| ETA exp {eta_exp/60:.1f} min | ETA all {eta_all/3600:.2f} h"
                )
                print(msg)
                f.write(msg + "\n")
                _append_progress(
                    iter_idx=cur,
                    iter_total=tot,
                    eta_exp_s=eta_exp,
                    eta_all_s=eta_all,
                )
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{cfg.name} failed with exit code {rc}. See {log_path}")

    global_state["done_iters_prefix"] += args.stop_iters
    global_state["done_iters"] = global_state["done_iters_prefix"]
    elapsed = time.time() - exp_start
    print(
        f"=== Completed {cfg.name} in {elapsed/60:.1f} min "
        f"({exp_idx}/{total_exps}) ==="
    )

    return csv_path, log_path


def _compute_curve(csv_path: Path, smooth_window: int):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "safety_sum" not in df.columns:
        raise ValueError(f"{csv_path} missing safety_sum column")

    # IMPORTANT:
    # In multi-env-runner training logs, rows from different workers/episodes are
    # interleaved. A global groupby("timestep") collapses unrelated trajectories and
    # can produce misleading identical curves. Use row-level signal for robust
    # comparison, and smooth with rolling window.
    series = df["safety_sum"].astype(float)
    collision = (series > 0.0).astype(float)
    rolling = collision.rolling(window=smooth_window, min_periods=1).mean()
    x = np.arange(len(rolling))
    return x, rolling.to_numpy(), float(collision.mean()), int(len(collision))


def _plot_results(output_dir: Path, smooth_window: int) -> Path:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    summary_rows = []

    for cfg in EXPERIMENTS:
        csv_path = output_dir / f"{cfg.name}.csv"
        if not csv_path.exists():
            continue
        x, y, mean_rate, nsteps = _compute_curve(csv_path, smooth_window)
        plt.plot(x, y, label=f"{cfg.name} (mean={mean_rate:.3f})")
        summary_rows.append(
            {
                "experiment": cfg.name,
                "mean_collision_rate": mean_rate,
                "num_steps_after_dedup": nsteps,
            }
        )

    plt.title("Collision Frequency Comparison (smoothed)")
    plt.xlabel("Deduplicated timestep index")
    plt.ylabel(f"Collision frequency (rolling window={smooth_window})")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = output_dir / "collision_four_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    summary_path = output_dir / "summary_collision_rates.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "mean_collision_rate", "num_steps_after_dedup"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 4 collision experiments and generate comparison plot."
    )
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--stop-iters", type=int, default=120)
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--dataset-path", type=Path, default=_default_dataset())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--smooth-window", type=int, default=24)
    parser.add_argument("--field-tick-s", type=float, default=5.0)
    parser.add_argument("--coord-tick-s", type=float, default=10.0)
    parser.add_argument("--system-tick-s", type=float, default=30.0)
    parser.add_argument("--jitter-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        default=False,
        help="Skip training and only regenerate plot/summary from existing CSV files.",
    )
    args = parser.parse_args()

    args.dataset_path = args.dataset_path.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_csv = output_dir / "progress.csv"

    print(f"Output dir: {output_dir}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Iterations per experiment: {args.stop_iters}")
    print(f"Num env runners: {args.num_env_runners}")

    total_iters = args.stop_iters * len(EXPERIMENTS)
    global_state = {
        "total_iters": total_iters,
        "done_iters_prefix": 0,
        "done_iters": 0,
    }

    if not args.postprocess_only:
        for idx, cfg in enumerate(EXPERIMENTS, start=1):
            _run_one(
                cfg,
                args,
                output_dir,
                exp_idx=idx,
                total_exps=len(EXPERIMENTS),
                progress_csv=progress_csv,
                global_state=global_state,
            )

    if args.dry_run:
        print("Dry run complete.")
        return

    fig_path = _plot_results(output_dir, args.smooth_window)
    print(f"\nDone. Figure saved to: {fig_path}")
    print(f"Per-experiment logs/csv are under: {output_dir}")
    print(f"Progress timeline CSV: {progress_csv}")


if __name__ == "__main__":
    main()


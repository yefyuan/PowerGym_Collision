"""Train collision experiments with RLlib on HERON + PowerGym (MG1–MG3 agents).

Baseline matches the original three-agent setup: separate policies per MG,
shared vs independent reward via ``CollisionEnv``, optional async construction
via ``create_collision_system`` (used for schedules / future event-driven work).

Run from ``case_studies/power``::

    python -m collision_case.train_collision --share-reward --log-path results/shared.csv

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from collections import deque
from pathlib import Path

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.examples.utils import add_rllib_example_script_args
from ray.tune.registry import get_trainable_cls, register_env
import ray

from collision_case.collision_rllib_env import CollisionRLlibMultiAgentEnv

_POWER_ROOT = Path(__file__).resolve().parent.parent
if str(_POWER_ROOT) not in sys.path:
    sys.path.insert(0, str(_POWER_ROOT))


def _default_dataset_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    power_root = here.parent
    workspace_root = power_root.parent.parent.parent
    return [
        here / "data2024.pkl",
        workspace_root / "Collision" / "data" / "data2024.pkl",
        power_root / "powergrid" / "data.pkl",
    ]


def resolve_dataset_path(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if p.is_file():
            return p
    for p in _default_dataset_candidates():
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "No dataset pickle found. Place data2024.pkl under collision_case/, "
        "or Collision/data/, or powergrid/data.pkl, or pass --dataset-path."
    )


parser = add_rllib_example_script_args(
    default_iters=300,
    default_timesteps=3000000,
    default_reward=0.0,
)
parser.add_argument(
    "--share-reward",
    action="store_true",
    default=False,
    help="Average reward across MG1–MG3 (same as original shared setup).",
)
parser.add_argument(
    "--log-path",
    type=str,
    default=None,
    help=(
        "Optional CSV path for collision metrics (CollisionEnv). "
        "For parallel env runners, prefer a per-env path template like "
        "'collision_case/sync_shared_{pid}_{env_instance}.csv' or pass a directory "
        "to write one file per env."
    ),
)
parser.add_argument(
    "--dataset-path",
    type=Path,
    default=None,
    help="Pickle with load/solar/wind/price (else search default locations).",
)
parser.add_argument(
    "--episode-steps",
    type=int,
    default=24,
    help="Steps per episode (default 24 matches prior collision_case settings).",
)
parser.add_argument(
    "--penalty",
    type=float,
    default=10.0,
    help="Collision penalty weight on coordinators.",
)
parser.add_argument(
    "--enable-async",
    action="store_true",
    default=False,
    help="Build agents with async ScheduleConfig (HERON ticks).",
)
parser.add_argument(
    "--field-tick-s",
    type=float,
    default=5.0,
)
parser.add_argument(
    "--coord-tick-s",
    type=float,
    default=10.0,
)
parser.add_argument(
    "--system-tick-s",
    type=float,
    default=30.0,
)
parser.add_argument(
    "--jitter-ratio",
    type=float,
    default=0.1,
)


class CollisionCallbacks(DefaultCallbacks):
    """Collect per-step collision metrics from env infos and expose as RLlib custom metrics."""

    def on_episode_start(self, *, episode, **kwargs):
        # Nothing to initialize; we push per-timestep values via `add_temporary_timestep_data`.
        return

    def on_episode_step(self, *, episode, metrics_logger=None, **kwargs):
        # New API stack: use episode.get_infos() to retrieve last step's info dicts.
        try:
            infos = episode.get_infos(-1, agent_ids=("MG1", "MG2", "MG3"), env_steps=True)
        except Exception:
            return
        info = None
        if isinstance(infos, dict):
            for aid in ("MG1", "MG2", "MG3"):
                if infos.get(aid):
                    info = infos[aid]
                    break
        if not info:
            return
        col = (info or {}).get("collision")
        if not col:
            return

        flag = float(col.get("collision_flag", 0.0))
        safety_sum = float(col.get("safety_sum", 0.0))
        if metrics_logger is not None:
            # Rolling-window means (roughly "last 100 steps" style).
            metrics_logger.log_value(("collision", "freq_100"), flag, reduce="mean", window=100)
            metrics_logger.log_value(("collision", "safety_sum_100"), safety_sum, reduce="mean", window=100)
            # Also overall means (window=None).
            metrics_logger.log_value(("collision", "freq"), flag, reduce="mean")
            metrics_logger.log_value(("collision", "safety_sum"), safety_sum, reduce="mean")

        mgs = col.get("microgrids", {}) or {}
        for aid in ("MG1", "MG2", "MG3"):
            if metrics_logger is not None:
                metrics_logger.log_value(
                    ("collision", f"{aid}_safety_sum_100"),
                    float((mgs.get(aid) or {}).get("safety_total", 0.0)),
                    reduce="mean",
                    window=100,
                )

    def on_episode_end(self, *, episode, **kwargs):
        return


def main() -> None:
    # RLlib vector multi-agent path re-wraps envs; Gymnasium may warn about duplicate
    # checks / dtype bookkeeping even when our env is consistent (verified standalone).
    warnings.filterwarnings(
        "ignore",
        message=".*precision lowered by casting to float32.*",
        category=UserWarning,
        module=r"gymnasium\.spaces\.box",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    # Reduce extremely noisy pandapower warnings in parallel rollouts.
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    logging.getLogger("pandapower.auxiliary").setLevel(logging.ERROR)

    args = parser.parse_args()

    if not hasattr(args, "num_agents") or args.num_agents == 0:
        args.num_agents = 3
    assert args.num_agents == 3, "This experiment uses exactly three microgrids (MG1–MG3)."

    dataset_path = resolve_dataset_path(args.dataset_path)

    env_config = {
        "dataset_path": str(dataset_path),
        "episode_steps": args.episode_steps,
        "dt": 1.0,
        "share_reward": args.share_reward,
        "penalty": args.penalty,
        "log_path": args.log_path,
        "enable_async": args.enable_async,
        "field_tick_s": args.field_tick_s,
        "coord_tick_s": args.coord_tick_s,
        "system_tick_s": args.system_tick_s,
        "jitter_ratio": args.jitter_ratio,
    }

    register_env(
        "collision_heron_env",
        lambda cfg: CollisionRLlibMultiAgentEnv(cfg),
    )

    policies = {f"MG{i}" for i in (1, 2, 3)}
    rl_module_specs = {
        "MG1": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(
                fcnet_hiddens=[128, 128],
                log_std_clip_param=20.0,
            ),
            catalog_class=PPOCatalog,
        ),
        "MG2": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(
                fcnet_hiddens=[64, 64],
                log_std_clip_param=20.0,
            ),
            catalog_class=PPOCatalog,
        ),
        "MG3": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(
                fcnet_hiddens=[64, 128],
                log_std_clip_param=20.0,
            ),
            catalog_class=PPOCatalog,
        ),
    }

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            "collision_heron_env",
            env_config=env_config,
            disable_env_checking=True,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda aid, *_a, **_k: aid),
        )
        .training(train_batch_size=8760)
        .callbacks(CollisionCallbacks)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs),
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
        )
        .resources(num_gpus=0, num_cpus_for_main_process=2)
    )

    print("Dataset:", dataset_path)
    print("Share reward:", args.share_reward)
    print("Enable async (agent schedules):", args.enable_async)
    print("Num env runners:", args.num_env_runners)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo = base_config.build_algo()
    start = time.time()

    roll = {
        "collision_freq_mean": deque(maxlen=20),
        "safety_mean_mean": deque(maxlen=20),
    }

    for iteration in range(args.stop_iters):
        t0 = time.time()
        result = algo.train()
        env_runner = result.get("env_runners", {})
        ep_ret = env_runner.get("episode_return_mean", 0.0) or 0.0
        ep_len = env_runner.get("episode_len_mean", 0.0) or 0.0
        # New API stack: collision metrics come from MetricsLogger.
        custom = env_runner.get("custom_metrics", {}) or {}
        if iteration == 0 and custom:
            print("custom_metrics keys:", sorted(map(str, custom.keys()))[:50])

        # MetricsLogger keys may appear as tuples or as "/"-joined strings depending on RLlib version.
        def _m(*keys):
            for k in keys:
                if k in custom:
                    return custom.get(k)
            return None

        coll = _m(("collision", "freq"), "collision/freq", "collision.freq")
        safe = _m(("collision", "safety_sum"), "collision/safety_sum", "collision.safety_sum")
        coll_100 = _m(("collision", "freq_100"), "collision/freq_100", "collision.freq_100")
        safe_100 = _m(("collision", "safety_sum_100"), "collision/safety_sum_100", "collision.safety_sum_100")
        if coll is not None:
            roll["collision_freq_mean"].append(float(coll))
        if safe is not None:
            roll["safety_mean_mean"].append(float(safe))

        def _roll_mean(xs):
            return (sum(xs) / len(xs)) if xs else None

        coll_roll = _roll_mean(roll["collision_freq_mean"])
        safe_roll = _roll_mean(roll["safety_mean_mean"])
        print(
            f"iter {iteration + 1}/{args.stop_iters} | "
            f"episode_return_mean={ep_ret:.4f} episode_len_mean={ep_len:.2f} | "
            f"collision_freq_mean={coll if coll is not None else float('nan'):.3f} "
            f"safety_sum_mean={safe if safe is not None else float('nan'):.4f} | "
            f"collision_freq_100={coll_100 if coll_100 is not None else float('nan'):.3f} "
            f"safety_sum_100={safe_100 if safe_100 is not None else float('nan'):.4f} | "
            f"collision_freq_roll={coll_roll if coll_roll is not None else float('nan'):.3f} "
            f"safety_mean_roll={safe_roll if safe_roll is not None else float('nan'):.4f} | "
            f"iter_s={time.time() - t0:.1f} total_s={time.time() - start:.1f}"
        )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

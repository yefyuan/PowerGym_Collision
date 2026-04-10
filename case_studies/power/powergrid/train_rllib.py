"""Power Grid MAPPO Training with Ray RLlib + Event-Driven Evaluation.

Uses HERON's full RLlib adapter stack:
- RLlibBasedHeronEnv: wraps HierarchicalMicrogridEnv as RLlib MultiAgentEnv
- HeronEnvRunner: custom runner with automatic event-driven evaluation
- RLlibModuleBridge: bridges trained RLModules back to HERON policies

Usage:
    python -m case_studies.power.powergrid.train_rllib
"""

import logging
import os
from typing import Any, Dict, List

import numpy as np

from heron.protocols.vertical import VerticalProtocol
from heron.scheduling.schedule_config import JitterType, ScheduleConfig

from powergrid.agents import ESS, Generator, PowerGridAgent
from powergrid.envs import HierarchicalMicrogridEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.pkl"))
EPISODE_STEPS = 24
DT_H = 1.0

MICROGRID_SPECS = [
    {
        "mg_id": "MG1",
        "generators": [
            {"agent_id": "MG1_Gen1", "bus": "MG1_bus", "p_min_MW": 0.1,
             "p_max_MW": 1.0, "cost_curve_coefs": (0.01, 0.4, 0.0)},
        ],
        "ess": [
            {"agent_id": "MG1_ESS1", "bus": "MG1_bus", "capacity_MWh": 2.0,
             "p_min_MW": -0.5, "p_max_MW": 0.5, "degr_cost_per_MWh": 0.1},
        ],
    },
    {
        "mg_id": "MG2",
        "generators": [
            {"agent_id": "MG2_Gen1", "bus": "MG2_bus", "p_min_MW": 0.1,
             "p_max_MW": 1.0, "cost_curve_coefs": (0.01, 0.4, 0.0)},
            {"agent_id": "MG2_Gen2", "bus": "MG2_bus", "p_min_MW": 0.1,
             "p_max_MW": 1.5, "cost_curve_coefs": (0.015, 0.5, 0.0)},
        ],
        "ess": [
            {"agent_id": "MG2_ESS1", "bus": "MG2_bus", "capacity_MWh": 3.0,
             "p_min_MW": -0.6, "p_max_MW": 0.6, "degr_cost_per_MWh": 0.1},
        ],
    },
    {
        "mg_id": "MG3",
        "generators": [
            {"agent_id": "MG3_Gen1", "bus": "MG3_bus", "p_min_MW": 0.1,
             "p_max_MW": 1.0, "cost_curve_coefs": (0.01, 0.4, 0.0)},
        ],
        "ess": [
            {"agent_id": "MG3_ESS1", "bus": "MG3_bus", "capacity_MWh": 2.0,
             "p_min_MW": -0.5, "p_max_MW": 0.5, "degr_cost_per_MWh": 0.1},
            {"agent_id": "MG3_ESS2", "bus": "MG3_bus", "capacity_MWh": 3.0,
             "p_min_MW": -0.6, "p_max_MW": 0.6, "degr_cost_per_MWh": 0.1},
        ],
    },
]

# ScheduleConfigs for event-driven evaluation (embedded in agent specs)
_FIELD_SCHED = ScheduleConfig.with_jitter(
    tick_interval=5.0, obs_delay=0.1, act_delay=0.2, msg_delay=0.1,
    jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=42,
)
_COORD_SCHED = ScheduleConfig.with_jitter(
    tick_interval=10.0, obs_delay=0.2, act_delay=0.3, msg_delay=0.15,
    reward_delay=0.6, jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=43,
)
_SYSTEM_SCHED = ScheduleConfig.with_jitter(
    tick_interval=30.0, obs_delay=0.3, act_delay=0.5, msg_delay=0.2,
    reward_delay=1.0, jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=41,
)


# ============================================================================
# Env config (shared by training and event-driven eval)
# ============================================================================
def build_env_config() -> Dict[str, Any]:
    """Build env_config dict consumed by RLlibBasedHeronEnv + HeronEnvRunner."""
    agents: List[Dict[str, Any]] = []
    coordinators: List[Dict[str, Any]] = []

    for mg in MICROGRID_SPECS:
        mg_id = mg["mg_id"]
        for spec in mg["generators"]:
            agents.append({
                "agent_id": spec["agent_id"], "agent_cls": Generator,
                "coordinator": mg_id, "schedule_config": _FIELD_SCHED,
                **{k: v for k, v in spec.items() if k != "agent_id"},
            })
        for spec in mg["ess"]:
            agents.append({
                "agent_id": spec["agent_id"], "agent_cls": ESS,
                "coordinator": mg_id, "schedule_config": _FIELD_SCHED,
                **{k: v for k, v in spec.items() if k != "agent_id"},
            })
        coordinators.append({
            "coordinator_id": mg_id, "agent_cls": PowerGridAgent,
            "protocol": VerticalProtocol(), "schedule_config": _COORD_SCHED,
        })

    return {
        "agents": agents,
        "coordinators": coordinators,
        "system": {"schedule_config": _SYSTEM_SCHED},
        "env_class": HierarchicalMicrogridEnv,
        "env_kwargs": {"dataset_path": DATASET_PATH, "episode_steps": EPISODE_STEPS, "dt": DT_H},
        "max_steps": EPISODE_STEPS,
    }


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """Map device agents to shared policies by type."""
    return "gen_policy" if "Gen" in agent_id else "ess_policy"


# ============================================================================
# Train + Eval
# ============================================================================
def train_mappo(num_iterations: int = 20):
    """Train device-level MAPPO via RLlib PPO with HeronEnvRunner.

    Training: step-based PPO (fast).
    Evaluation: event-driven via HeronEnvRunner (realistic async timing).
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from heron.adaptors.rllib import RLlibBasedHeronEnv
    from heron.adaptors.rllib_runner import HeronEnvRunner

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    config = (
        PPOConfig()
        .environment(env=RLlibBasedHeronEnv, env_config=build_env_config())
        .framework("torch")
        .training(
            lr=3e-4, gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.01, vf_clip_param=10.0,
            train_batch_size=2000, num_epochs=10, minibatch_size=256,
        )
        .env_runners(
            env_runner_cls=HeronEnvRunner,
            num_env_runners=2,
            num_cpus_per_env_runner=1,
        )
        .evaluation(
            evaluation_interval=num_iterations,  # eval after final iteration
            evaluation_num_env_runners=0,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config=HeronEnvRunner.evaluation_config(t_end=100.0),
        )
        .resources(num_gpus=0, num_cpus_for_main_process=1)
        .multi_agent(
            policies={"gen_policy", "ess_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    logger.info("Building PPO algorithm for powergrid MAPPO...")
    algo = config.build()

    reward_history = []
    try:
        for i in range(num_iterations):
            result = algo.train()
            reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0) or 0.0
            reward_history.append(reward_mean)
            logger.info(f"Iter {i:3d} | Reward mean: {reward_mean:8.3f}")
    except KeyboardInterrupt:
        logger.info("Training interrupted")

    logger.info(f"Training complete. {len(reward_history)} iterations.")
    if len(reward_history) >= 4:
        logger.info(
            f"  First 2 avg: {np.mean(reward_history[:2]):.3f} | "
            f"Last 2 avg: {np.mean(reward_history[-2:]):.3f}"
        )

    # Event-driven evaluation (HeronEnvRunner handles env build + policy bridge)
    logger.info("Running event-driven evaluation via HeronEnvRunner...")
    eval_result = algo.evaluate()
    heron_metrics = eval_result.get("env_runners", {}).get("heron", {})
    if heron_metrics:
        logger.info(f"  Event-driven reward: {heron_metrics.get('episode_reward', 0):.3f}")
        logger.info(f"  Events: {heron_metrics.get('num_events', 0)}")
        logger.info(f"  Duration: {heron_metrics.get('duration', 0):.1f}s")
    else:
        logger.info(f"  Eval metrics: {list(eval_result.get('env_runners', {}).keys())}")

    algo.stop()
    ray.shutdown()
    logger.info("Done.")
    return reward_history


if __name__ == "__main__":
    train_mappo(num_iterations=20)

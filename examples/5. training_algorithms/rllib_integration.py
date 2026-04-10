"""RLlib Integration -- training HERON environments with RLlib.

HERON provides a full RLlib adapter stack:
  - RLlibBasedHeronEnv: wraps HERON env as RLlib MultiAgentEnv
  - HeronEnvRunner: custom runner with event-driven evaluation
  - RLlibModuleBridge: bridges trained RLModules back to HERON policies

This script demonstrates:
1. RLlibBasedHeronEnv setup -- env_config structure and agent specs
2. IPPO configuration -- independent policies per agent
3. MAPPO configuration -- shared policy across agents
4. HeronEnvRunner -- event-driven evaluation setup

Domain: Two thermostats controlling room temperature (from Level 3).
  - Each thermostat adjusts heating power [-1, 1].
  - Reward: negative distance to setpoint (target=22C).

NOTE: This example requires Ray and RLlib.
    pip install "ray[rllib]" torch

Usage:
    cd "examples/5. training_algorithms"
    python rllib_integration.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature


# ---------------------------------------------------------------------------
# 1. Domain: thermostats (same as policy_and_training.py)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RoomTempFeature(Feature):
    """Room temperature (public so coordinator can observe it)."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    temp: float = 20.0
    target: float = 22.0


class Thermostat(FieldAgent):
    """Thermostat that adjusts heating power."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        if "temp" in kwargs:
            self.state.features["RoomTempFeature"].set_values(temp=kwargs["temp"])

    def apply_action(self) -> None:
        feat = self.state.features["RoomTempFeature"]
        new_temp = feat.temp + self.action.c[0] * 2.0
        feat.set_values(temp=float(np.clip(new_temp, 10.0, 35.0)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "RoomTempFeature" not in local_state:
            return 0.0
        vec = local_state["RoomTempFeature"]
        temp, target = float(vec[0]), float(vec[1])
        return -abs(temp - target)


def thermostat_simulation(agent_states: dict) -> dict:
    """Natural cooling toward 15C ambient + cross-room leakage."""
    rooms = {}
    for aid, features in agent_states.items():
        if "RoomTempFeature" in features:
            rooms[aid] = features["RoomTempFeature"]["temp"]

    ambient, cooling_rate, leakage_rate = 15.0, 0.1, 0.05
    for aid in rooms:
        rooms[aid] += cooling_rate * (ambient - rooms[aid])

    if len(rooms) >= 2:
        avg_temp = np.mean(list(rooms.values()))
        for aid in rooms:
            rooms[aid] += leakage_rate * (avg_temp - rooms[aid])

    for aid, temp in rooms.items():
        agent_states[aid]["RoomTempFeature"]["temp"] = float(np.clip(temp, 10.0, 35.0))
    return agent_states


# ---------------------------------------------------------------------------
# 2. Show env_config structure (no RLlib needed)
# ---------------------------------------------------------------------------

def demo_env_config():
    """Show how RLlibBasedHeronEnv is configured."""
    print("=" * 60)
    print("Part 1: RLlibBasedHeronEnv Configuration")
    print("=" * 60)
    print("""
  RLlibBasedHeronEnv wraps a HERON env for RLlib training.
  Configuration is done via env_config dict:

    config = PPOConfig().environment(
        env=RLlibBasedHeronEnv,
        env_config={
            # Agent specs -- each becomes a HERON FieldAgent
            "agents": [
                {
                    "agent_id": "room_a",
                    "agent_cls": Thermostat,
                    "features": [RoomTempFeature()],
                    "coordinator": "building",  # optional
                },
                ...
            ],

            # Coordinator specs -- groups agents
            "coordinators": [
                {
                    "coordinator_id": "building",
                    "agent_cls": CoordinatorAgent,  # optional
                    "protocol": VerticalProtocol(),  # optional
                },
            ],

            # Simulation function (or "env_class" for custom envs)
            "simulation": thermostat_simulation,

            # Episode length
            "max_steps": 50,

            # Which agent IDs RLlib sees (default: all with action_space)
            "agent_ids": ["room_a", "room_b"],
        },
    )

  The adapter:
  - Builds the HERON env via EnvBuilder internally
  - Flattens Observations to float32 numpy vectors
  - Exposes agent action spaces to RLlib automatically
  - Handles reset/step translation
""")


# ---------------------------------------------------------------------------
# 3. IPPO configuration
# ---------------------------------------------------------------------------

def demo_ippo_config():
    """Show IPPO configuration -- independent policies per agent."""
    print("=" * 60)
    print("Part 2: IPPO Configuration")
    print("=" * 60)
    print("""
  IPPO (Independent PPO): each agent has its own policy network.

    config = (
        PPOConfig()
        .environment(env=RLlibBasedHeronEnv, env_config={...})
        .multi_agent(
            policies={
                "room_a_policy": PolicySpec(),   # independent
                "room_b_policy": PolicySpec(),   # independent
            },
            policy_mapping_fn=lambda aid, *a, **kw: f"{aid}_policy",
        )
    )

  Each PolicySpec() creates a separate neural network.
  policy_mapping_fn maps agent_id -> policy_id.

  When to use IPPO:
    - Agents have different roles or capabilities
    - Agents operate in different contexts
    - You want maximal specialization
""")


# ---------------------------------------------------------------------------
# 4. MAPPO configuration
# ---------------------------------------------------------------------------

def demo_mappo_config():
    """Show MAPPO configuration -- shared policy across agents."""
    print("=" * 60)
    print("Part 3: MAPPO Configuration")
    print("=" * 60)
    print("""
  MAPPO (Multi-Agent PPO): agents share one policy network.

    config = (
        PPOConfig()
        .environment(env=RLlibBasedHeronEnv, env_config={...})
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(),   # one network
            },
            policy_mapping_fn=lambda aid, *a, **kw: "shared_policy",
        )
    )

  All agents map to "shared_policy" -> same network, 2x data.

  When to use MAPPO:
    - Agents are homogeneous (same type, same dynamics)
    - You want sample efficiency (N agents = N× data)
    - Agents should behave identically
""")


# ---------------------------------------------------------------------------
# 5. HeronEnvRunner for event-driven evaluation
# ---------------------------------------------------------------------------

def demo_runner_config():
    """Show HeronEnvRunner setup for event-driven evaluation."""
    print("=" * 60)
    print("Part 4: HeronEnvRunner + Event-Driven Evaluation")
    print("=" * 60)
    print("""
  HeronEnvRunner extends RLlib's MultiAgentEnvRunner with:
  1. Access to underlying HERON env
  2. Event-driven evaluation via trained policies

  Training: standard step-based (PPO collects transitions)
  Evaluation: event-driven (HERON scheduler drives timing)

    from heron.adaptors.rllib_runner import HeronEnvRunner

    config = (
        PPOConfig()
        .environment(env=RLlibBasedHeronEnv, env_config={...})
        .env_runners(
            env_runner_cls=HeronEnvRunner,   # custom runner
            num_env_runners=2,
        )
        .evaluation(
            evaluation_interval=5,           # eval every 5 iters
            evaluation_num_env_runners=0,    # eval on local runner
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            # Event-driven evaluation config:
            evaluation_config=HeronEnvRunner.evaluation_config(
                t_end=100.0,   # simulation end time
            ),
        )
    )

  During evaluation, the runner:
  1. Creates a fresh HERON env with ScheduleConfig timing
  2. Bridges trained RLModules -> HERON policies (RLlibModuleBridge)
  3. Runs heron_env.run_event_driven(analyzer, t_end=...)
  4. Returns MultiAgentEpisode with reward trajectories

  This lets you train with fast step-based PPO, then evaluate
  with realistic asynchronous event-driven timing.
""")


# ---------------------------------------------------------------------------
# 6. Full working example (if Ray is available)
# ---------------------------------------------------------------------------

def demo_full_training():
    """Run a minimal RLlib training loop (requires Ray)."""
    print("=" * 60)
    print("Part 5: Full Training Example")
    print("=" * 60)

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.policy.policy import PolicySpec
        from heron.adaptors.rllib import RLlibBasedHeronEnv
        from heron.adaptors.rllib_runner import HeronEnvRunner
    except ImportError:
        print("\n  Ray/RLlib not installed. Skipping live training demo.")
        print("  Install with: pip install 'ray[rllib]' torch")
        print("\n  The configuration patterns above work with any Ray install.")
        return

    ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=0)

    try:
        agent_ids = ["room_a", "room_b"]

        config = (
            PPOConfig()
            .environment(
                env=RLlibBasedHeronEnv,
                env_config={
                    "env_id": "thermostat_rllib",
                    "agents": [
                        {
                            "agent_id": "room_a",
                            "agent_cls": Thermostat,
                            "features": [RoomTempFeature()],
                            "coordinator": "building",
                        },
                        {
                            "agent_id": "room_b",
                            "agent_cls": Thermostat,
                            "features": [RoomTempFeature()],
                            "coordinator": "building",
                        },
                    ],
                    "coordinators": [
                        {
                            "coordinator_id": "building",
                            "agent_cls": CoordinatorAgent,
                        },
                    ],
                    "simulation": thermostat_simulation,
                    "max_steps": 30,
                    "agent_ids": agent_ids,
                },
            )
            .multi_agent(
                policies={
                    "shared_policy": PolicySpec(),
                },
                policy_mapping_fn=lambda aid, *a, **kw: "shared_policy",
            )
            .env_runners(
                env_runner_cls=HeronEnvRunner,
                num_env_runners=0,
                num_envs_per_env_runner=1,
            )
            .training(
                lr=5e-4,
                gamma=0.99,
                train_batch_size=120,
                minibatch_size=32,
                num_epochs=3,
            )
            .framework("torch")
        )

        algo = config.build()
        print("\n  Training MAPPO for 5 iterations...\n")

        for i in range(5):
            result = algo.train()
            reward = result["env_runners"]["episode_return_mean"]
            print(f"    Iter {i + 1}/5: mean_return={reward:.1f}")

        algo.stop()
        print("\n  Training complete.")

    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    demo_env_config()
    demo_ippo_config()
    demo_mappo_config()
    demo_runner_config()
    demo_full_training()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  RLlib Integration Stack:

    RLlibBasedHeronEnv   -- wraps HERON env as MultiAgentEnv
    HeronEnvRunner       -- custom runner with event-driven eval
    RLlibModuleBridge    -- bridges RLModule -> HERON Policy

  Training patterns:

    IPPO: policy_mapping_fn -> unique policy per agent
    MAPPO: policy_mapping_fn -> "shared_policy" for all agents

  Dual-mode execution:

    Training:   step-based (PPO/MAPPO via RLlib)
    Evaluation: event-driven (HERON scheduler + ScheduleConfig)

  See examples/1. starter/ for full production examples
  with ScheduleConfig, jitter, and event-driven evaluation.
""")
    print("Done.")


if __name__ == "__main__":
    main()

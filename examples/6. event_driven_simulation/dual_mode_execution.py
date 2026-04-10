"""Dual-Mode Execution -- step-based training vs event-driven evaluation.

HERON environments support two execution modes:
  1. Step-based (training): env.step() loop, fast, deterministic timing
  2. Event-driven (evaluation): env.run_event_driven(), realistic timing

This script demonstrates:
1. Building an env with ScheduleConfig for event-driven evaluation
2. Step-based execution (training mode)
3. Event-driven execution with EpisodeAnalyzer
4. Comparing behavior across modes
5. EpisodeStats analysis -- event counts, agent activity, timing

Domain: Two thermostats controlling room temperature.
  - Same agents, same physics, different execution modes.

Usage:
    cd "examples/6. event_driven_simulation"
    python dual_mode_execution.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.simple import SimpleEnv
from heron.scheduling import (
    ScheduleConfig,
    JitterType,
    EpisodeAnalyzer,
)


# ---------------------------------------------------------------------------
# 1. Domain: thermostats (reused from Level 5)
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
# 2. Simple policy for event-driven mode
# ---------------------------------------------------------------------------

class SimpleThermostatPolicy(Policy):
    """Rule-based policy: heat if below target, cool if above."""
    observation_mode = "local"

    def __init__(self):
        self.obs_dim = 2         # temp, target
        self.action_dim = 1
        self.action_range = (-1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        temp, target = obs_vec[0], obs_vec[1]
        if temp < target - 0.5:
            return np.array([0.8])    # heat
        elif temp > target + 0.5:
            return np.array([-0.5])   # cool
        return np.array([0.0])        # maintain


# ---------------------------------------------------------------------------
# 3. Build env (shared between both modes)
# ---------------------------------------------------------------------------

def build_env_step_based():
    """Build env for step-based mode (no tick config needed)."""
    thermo_a = Thermostat(agent_id="room_a", features=[RoomTempFeature()])
    thermo_b = Thermostat(agent_id="room_b", features=[RoomTempFeature()])
    coordinator = CoordinatorAgent(
        agent_id="building",
        subordinates={"room_a": thermo_a, "room_b": thermo_b},
    )
    return SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=thermostat_simulation,
        env_id="thermostat_step",
    )


def build_env_event_driven():
    """Build env for event-driven mode with explicit ScheduleConfig."""
    # Field agents: tick every 1s with observation and action delays
    # Each agent gets its own ScheduleConfig instance (independent jitter RNG)
    field_tick_a = ScheduleConfig.with_jitter(
        tick_interval=1.0,
        obs_delay=0.05,
        act_delay=0.1,
        msg_delay=0.02,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=42,
    )
    field_tick_b = ScheduleConfig.with_jitter(
        tick_interval=1.0,
        obs_delay=0.05,
        act_delay=0.1,
        msg_delay=0.02,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=43,
    )

    # Coordinator: ticks every 5s (slower decision-making)
    coord_tick = ScheduleConfig.with_jitter(
        tick_interval=5.0,
        obs_delay=0.1,
        act_delay=0.2,
        msg_delay=0.05,
        reward_delay=0.5,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=44,
    )

    # System agent: ticks every 10s
    system_tick = ScheduleConfig.with_jitter(
        tick_interval=10.0,
        obs_delay=0.1,
        act_delay=0.2,
        msg_delay=0.05,
        reward_delay=1.0,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=45,
    )

    thermo_a = Thermostat(
        agent_id="room_a",
        features=[RoomTempFeature()],
        schedule_config=field_tick_a,
    )
    thermo_b = Thermostat(
        agent_id="room_b",
        features=[RoomTempFeature()],
        schedule_config=field_tick_b,
    )
    coordinator = CoordinatorAgent(
        agent_id="building",
        subordinates={"room_a": thermo_a, "room_b": thermo_b},
        schedule_config=coord_tick,
    )
    return SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=thermostat_simulation,
        env_id="thermostat_event",
        system_agent_schedule_config=system_tick,
    )


# ---------------------------------------------------------------------------
# 4. Demonstrations
# ---------------------------------------------------------------------------

def demo_step_based():
    """Run env in step-based mode (training)."""
    print("=" * 60)
    print("Part 1: Step-Based Execution (Training Mode)")
    print("=" * 60)
    print("  env.step() loop -- fast, deterministic, no timing delays\n")

    env = build_env_step_based()
    obs, _ = env.reset(seed=0)

    print(f"  Running 10 steps with simple heating policy:")
    print(f"  {'Step':<6} {'room_a temp':>12} {'room_b temp':>12} {'reward':>10}")
    print(f"  {'-' * 42}")

    total_reward = 0.0
    for step in range(10):
        # Simple rule: heat if below target
        actions = {}
        for aid in ["room_a", "room_b"]:
            obs_vec = obs[aid].vector() if hasattr(obs[aid], "vector") else np.asarray(obs[aid])
            temp, target = obs_vec[0], obs_vec[1]
            heat = np.clip((target - temp) / 5.0, -1.0, 1.0)

            action = Action()
            action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
            action.set_values(c=[heat])
            actions[aid] = action

        obs, rewards, terminated, _, _ = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward

        if step == 0 or (step + 1) % 3 == 0:
            obs_a = obs["room_a"].vector() if hasattr(obs["room_a"], "vector") else np.asarray(obs["room_a"])
            obs_b = obs["room_b"].vector() if hasattr(obs["room_b"], "vector") else np.asarray(obs["room_b"])
            print(f"  {step + 1:<6} {obs_a[0]:>12.2f} {obs_b[0]:>12.2f} {step_reward:>10.2f}")

    print(f"\n  Total reward over 10 steps: {total_reward:.2f}")
    print(f"  Each step is instantaneous -- no timing simulation.")


def demo_event_driven():
    """Run env in event-driven mode (evaluation)."""
    print("\n" + "=" * 60)
    print("Part 2: Event-Driven Execution (Evaluation Mode)")
    print("=" * 60)
    print("  env.run_event_driven() -- realistic timing with jitter\n")

    env = build_env_event_driven()
    obs, _ = env.reset(seed=0, jitter_seed=42)

    # Assign policies for event-driven execution
    policy = SimpleThermostatPolicy()
    env.set_agent_policies({
        "room_a": policy,
        "room_b": policy,
    })

    # Run event-driven simulation
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    t_end = 30.0
    result = env.run_event_driven(analyzer, t_end=t_end)

    summary = result.summary()
    print(f"  Simulation ran for {summary['duration']:.2f}s (target: {t_end}s)")
    print(f"  Total events processed: {summary['num_events']}")

    # Event type breakdown
    print(f"\n  Event counts by type:")
    for etype, count in sorted(summary["event_counts"].items()):
        print(f"    {etype:<25} {count:>5}")

    # Message type breakdown
    if summary["message_type_counts"]:
        print(f"\n  Message types:")
        for mtype, count in sorted(summary["message_type_counts"].items()):
            print(f"    {mtype:<30} {count:>5}")

    # Per-agent activity
    print(f"\n  Events per agent:")
    for agent_id, count in sorted(summary["agent_event_counts"].items()):
        print(f"    {agent_id or 'None':<25} {count:>5}")

    # Reward history
    reward_history = analyzer.get_reward_history()
    print(f"\n  Reward history:")
    for agent_id, rewards in sorted(reward_history.items()):
        if rewards:
            total = sum(r for _, r in rewards)
            print(f"    {agent_id}: {len(rewards)} rewards, total={total:.2f}")

    return result


def demo_event_analysis(result):
    """Analyze the event-driven episode result."""
    print("\n" + "=" * 60)
    print("Part 3: EpisodeStats Analysis")
    print("=" * 60)
    print("""
  EpisodeStats collects all event analyses from run_event_driven().
  It provides summary statistics and per-agent breakdowns.
""")

    summary = result.summary()

    print(f"  Episode statistics:")
    print(f"    Duration:         {summary['duration']:.2f}s")
    print(f"    Total events:     {summary['num_events']}")
    print(f"    Observations:     {summary['observations']}")
    print(f"    State updates:    {summary['state_updates']}")
    print(f"    Action results:   {summary['action_results']}")

    # Show event timeline (first 10 events)
    print(f"\n  Event timeline (first 10 events):")
    print(f"  {'#':<4} {'Time':>8} {'Type':<22} {'Agent':<15} {'Message':<20}")
    print(f"  {'-' * 72}")
    for i, analysis in enumerate(result.event_analyses[:10]):
        msg = analysis.message_type or ""
        agent = analysis.agent_id or ""
        print(f"  {i + 1:<4} {analysis.timestamp:>8.3f} {analysis.event_type.value:<22} "
              f"{agent:<15} {msg:<20}")

    if result.num_events > 10:
        print(f"  ... ({result.num_events - 10} more events)")


def demo_mode_comparison():
    """Compare step-based vs event-driven side by side."""
    print("\n" + "=" * 60)
    print("Part 4: Mode Comparison")
    print("=" * 60)
    print("""
  Step-Based (Training)          Event-Driven (Evaluation)
  ========================       ============================
  env.step(actions)              env.run_event_driven(analyzer, t_end)
  Instant: obs -> act -> next    Timed: obs_delay, act_delay, msg_delay
  Deterministic timing           Jittered timing (configurable)
  Fast (no scheduling)           Realistic (priority queue)
  No policies needed             Policies attached to agents
  Returns (obs, rew, ...)        Returns EpisodeStats

  The key insight: train fast with step-based mode,
  then validate with realistic event-driven timing.

  This is exactly what HeronEnvRunner does in RLlib:
    - Training:   step-based PPO (fast gradient collection)
    - Evaluation: event-driven with jitter (realistic validation)
""")

    # Quick comparison table
    print(f"  {'Property':<30} {'Step-Based':>15} {'Event-Driven':>15}")
    print(f"  {'-' * 62}")
    print(f"  {'Execution':.<30} {'env.step()':>15} {'run_event_driven':>15}")
    print(f"  {'Timing':.<30} {'instant':>15} {'realistic':>15}")
    print(f"  {'Jitter':.<30} {'none':>15} {'configurable':>15}")
    print(f"  {'Policy needed':.<30} {'no (manual)':>15} {'yes (attached)':>15}")
    print(f"  {'Use case':.<30} {'RL training':>15} {'validation':>15}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    demo_step_based()
    result = demo_event_driven()
    demo_event_analysis(result)
    demo_mode_comparison()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  Dual-mode execution:
    1. Step-based: env.step() for fast training
    2. Event-driven: run_event_driven() for realistic evaluation

  Event-driven setup:
    - Assign ScheduleConfig with delays to each agent
    - Attach Policy objects via env.set_agent_policies()
    - Create EpisodeAnalyzer to collect statistics
    - Call env.run_event_driven(analyzer, t_end=...)

  EpisodeStats provides:
    - Event counts by type and agent
    - Message type breakdown
    - Reward history per agent
    - Full event timeline
""")
    print("Done.")


if __name__ == "__main__":
    main()

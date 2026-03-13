"""EnvBuilder Patterns -- the fluent API for constructing HERON environments.

EnvBuilder eliminates manual agent-hierarchy wiring.  Instead of creating
agents, coordinators, and a system agent by hand, you declare what you want
and the builder resolves the hierarchy for you.

This script demonstrates:
1. add_agents with count + prefix (batch registration)
2. add_coordinator with explicit IDs and coordinator= assignment
3. Auto-coordinator for unassigned agents
4. add_system_agent for custom system-level config
5. Running a step loop on the built environment

Domain: A sensor network with two zones, each containing sensors that
report noisy readings. A simulation step adds measurement noise.

Usage:
    cd "examples/3. building_environments"
    python env_builder_patterns.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.builder import EnvBuilder
from heron.scheduling.tick_config import ScheduleConfig


# ---------------------------------------------------------------------------
# 1. Define feature and agent
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SensorReading(Feature):
    """A sensor's current reading and calibration offset."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0
    offset: float = 0.0


class SensorAgent(FieldAgent):
    """Sensor that adjusts its calibration offset."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        # Action: calibration adjustment in [-0.5, 0.5]
        action.set_specs(dim_c=1, range=(np.array([-0.5]), np.array([0.5])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        for key in ("value", "offset"):
            if key in kwargs:
                self.state.features["SensorReading"].set_values(**{key: kwargs[key]})

    def apply_action(self) -> None:
        feat = self.state.features["SensorReading"]
        # Action adjusts the calibration offset
        new_offset = feat.offset + self.action.c[0] * 0.1
        feat.set_values(offset=float(np.clip(new_offset, -1.0, 1.0)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "SensorReading" not in local_state:
            return 0.0
        vec = local_state["SensorReading"]
        # Reward: negative absolute offset (want offset close to 0)
        return -abs(float(vec[1]))


# ---------------------------------------------------------------------------
# 2. Simulation function
# ---------------------------------------------------------------------------

# Module-level RNG so noise varies across steps (not recreated each call)
_sim_rng = np.random.default_rng(42)


def sensor_simulation(agent_states: dict) -> dict:
    """Add measurement noise to sensor readings."""
    for aid, features in agent_states.items():
        if "SensorReading" in features:
            # True signal + noise + calibration offset
            true_signal = 10.0
            noise = _sim_rng.normal(0, 0.5)
            offset = features["SensorReading"]["offset"]
            features["SensorReading"]["value"] = true_signal + noise + offset
    return agent_states


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main():
    # ------------------------------------------------------------------
    # Pattern 1: add_agents with count + coordinator= assignment
    # ------------------------------------------------------------------
    section("Pattern 1: Batch registration + coordinator assignment")

    env = (
        EnvBuilder("sensor_net")
        .add_agents(
            "sensor_a",                         # prefix -> sensor_a_0, sensor_a_1
            SensorAgent,
            count=2,
            features=[SensorReading()],
            coordinator="zone_a",               # assign to zone_a via coordinator= param
        )
        .add_agents(
            "sensor_b",                         # prefix -> sensor_b_0, sensor_b_1
            SensorAgent,
            count=2,
            features=[SensorReading()],
            coordinator="zone_b",               # assign to zone_b
        )
        .add_coordinator("zone_a")              # auto-populated from coordinator= assignments
        .add_coordinator("zone_b")
        .simulation(sensor_simulation)
        .build()
    )

    obs, _ = env.reset(seed=0)
    print(f"Agents with observations: {list(obs.keys())}")
    print(f"Registered agents: {list(env.registered_agents.keys())}")

    # Run a few steps
    for step in range(3):
        actions = {}
        for aid, agent in env.registered_agents.items():
            if agent.action is not None and agent.action.is_valid():
                agent.action.sample(seed=step)
                actions[aid] = agent.action
        obs, rewards, _, _, _ = env.step(actions)

    print(f"After 3 steps, rewards: { {k: round(v, 3) for k, v in rewards.items()} }")

    # ------------------------------------------------------------------
    # Pattern 2: Auto-coordinator (no coordinator declared)
    # ------------------------------------------------------------------
    section("Pattern 2: Auto-coordinator")

    env2 = (
        EnvBuilder("auto_coord")
        .add_agents("node", SensorAgent, count=3, features=[SensorReading()])
        # No add_coordinator call -- builder auto-creates one
        .simulation(sensor_simulation)
        .build()
    )

    obs2, _ = env2.reset(seed=1)
    print(f"Agents: {list(obs2.keys())}")
    print(f"All registered (includes auto_coordinator): {list(env2.registered_agents.keys())}")

    # ------------------------------------------------------------------
    # Pattern 3: Single named agent via add_agent
    # ------------------------------------------------------------------
    section("Pattern 3: Single named agents with coordinator assignment")

    env3 = (
        EnvBuilder("named_agents")
        .add_agent("temp_sensor",    SensorAgent, features=[SensorReading(value=20.0)], coordinator="monitors")
        .add_agent("humidity_sensor", SensorAgent, features=[SensorReading(value=60.0)], coordinator="monitors")
        .add_agent("pressure_sensor", SensorAgent, features=[SensorReading(value=1013.0)])
        # "monitors" coordinator is auto-populated from the coordinator= assignments
        .add_coordinator("monitors")
        # pressure_sensor has no coordinator -> auto-assigned
        .simulation(sensor_simulation)
        .build()
    )

    obs3, _ = env3.reset(seed=2)
    print(f"Agents: {list(obs3.keys())}")
    print(f"All registered: {list(env3.registered_agents.keys())}")

    # ------------------------------------------------------------------
    # Pattern 4: Custom system agent config
    # ------------------------------------------------------------------
    section("Pattern 4: Custom system agent tick config")

    env4 = (
        EnvBuilder("custom_system")
        .add_agents("s", SensorAgent, count=2, features=[SensorReading()])
        .add_system_agent(schedule_config=ScheduleConfig.deterministic(tick_interval=5.0))
        .simulation(sensor_simulation)
        .build()
    )

    obs4, _ = env4.reset(seed=3)
    system_agent = env4.registered_agents.get("system_agent")
    print(f"System agent tick interval: {system_agent.schedule_config.tick_interval}s")
    print(f"Agents: {list(obs4.keys())}")

    print("\nDone.")


if __name__ == "__main__":
    main()

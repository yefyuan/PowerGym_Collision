"""SimpleEnv Quickstart -- the fastest way to build a HERON environment.

SimpleEnv eliminates boilerplate by auto-bridging between HERON's internal
state format and a flat dict that your simulation function can read/write:

    {agent_id: {feature_name: {field: value, ...}, ...}, ...}

You provide:
  1. Agent classes (FieldAgent subclasses)
  2. A simulation function  (dict -> dict)
  3. A coordinator to group agents

SimpleEnv handles the rest: state serialization, proxy wiring, scheduling.

Domain: Two thermostats controlling room temperature.
  - Each thermostat adjusts heating power [-1, 1].
  - The simulation applies heating, natural cooling, and cross-room leakage.

Usage:
    cd "examples/3. building_environments"
    python simple_env_quickstart.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.simple import SimpleEnv


# ---------------------------------------------------------------------------
# 1. Define a feature
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RoomTempFeature(Feature):
    """Room temperature in Celsius (public so the coordinator can observe it)."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    temp: float = 20.0          # current temperature
    target: float = 22.0        # setpoint


# ---------------------------------------------------------------------------
# 2. Define a field agent
# ---------------------------------------------------------------------------

class Thermostat(FieldAgent):
    """Adjusts heating power to bring room temperature toward the setpoint."""

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
        # Heating effect: action * 2 degrees per step
        new_temp = feat.temp + self.action.c[0] * 2.0
        feat.set_values(temp=float(np.clip(new_temp, 10.0, 35.0)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "RoomTempFeature" not in local_state:
            return 0.0
        vec = local_state["RoomTempFeature"]
        temp, target = float(vec[0]), float(vec[1])
        # Negative absolute error: closer to setpoint = higher reward
        return -abs(temp - target)


# ---------------------------------------------------------------------------
# 3. Define the simulation function
# ---------------------------------------------------------------------------

def thermostat_simulation(agent_states: dict) -> dict:
    """Physics step: natural cooling + cross-room thermal leakage.

    Receives and returns: {agent_id: {feature_name: {field: value}}}
    """
    # Collect room temperatures
    rooms = {}
    for aid, features in agent_states.items():
        if "RoomTempFeature" in features:
            rooms[aid] = features["RoomTempFeature"]["temp"]

    # Natural cooling toward 15C ambient
    ambient = 15.0
    cooling_rate = 0.1
    for aid in rooms:
        rooms[aid] += cooling_rate * (ambient - rooms[aid])

    # Cross-room leakage: rooms equalize slightly
    if len(rooms) >= 2:
        ids = list(rooms.keys())
        avg_temp = np.mean(list(rooms.values()))
        leakage_rate = 0.05
        for aid in ids:
            rooms[aid] += leakage_rate * (avg_temp - rooms[aid])

    # Write back
    for aid, temp in rooms.items():
        agent_states[aid]["RoomTempFeature"]["temp"] = float(np.clip(temp, 10.0, 35.0))

    return agent_states


# ---------------------------------------------------------------------------
# 4. Assemble and run
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SimpleEnv Quickstart: Two-Room Thermostat Control")
    print("=" * 60)

    # Create agents
    # Note: env.reset() resets features to their dataclass defaults,
    # so both rooms start at temp=20.0 regardless of what we pass here.
    thermo_a = Thermostat(
        agent_id="room_a",
        features=[RoomTempFeature()],
    )
    thermo_b = Thermostat(
        agent_id="room_b",
        features=[RoomTempFeature()],
    )

    # Group under a coordinator
    coordinator = CoordinatorAgent(
        agent_id="building",
        subordinates={"room_a": thermo_a, "room_b": thermo_b},
    )

    # Build the environment -- no custom HeronEnv subclass needed
    env = SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=thermostat_simulation,
        env_id="thermostat_demo",
    )

    # Run a few steps with random actions
    obs, info = env.reset(seed=0)
    print(f"\nInitial observations:")
    for aid in ["room_a", "room_b"]:
        if aid in obs:
            vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
            print(f"  {aid}: {vec}")

    field_agent_ids = ["room_a", "room_b"]

    print(f"\nRunning 10 steps with random actions...")
    for step in range(10):
        actions = {}
        for aid in field_agent_ids:
            agent = env.registered_agents[aid]
            agent.action.sample(seed=step * 10 + hash(aid) % 100)
            actions[aid] = agent.action

        obs, rewards, terminated, truncated, infos = env.step(actions)

        if (step + 1) % 5 == 0:
            print(f"\n  Step {step + 1}:")
            for aid in ["room_a", "room_b"]:
                if aid in obs:
                    vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
                    print(f"    {aid}: temp={vec[0]:.1f}C, reward={rewards.get(aid, 0):.2f}")

    print(f"\n  Terminated: {terminated.get('__all__', False)}")
    print("\nDone.")


if __name__ == "__main__":
    main()

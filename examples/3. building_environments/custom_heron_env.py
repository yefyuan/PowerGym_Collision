"""Custom HeronEnv -- full control over the simulation bridge.

When SimpleEnv's auto-bridge isn't enough, subclass HeronEnv directly and
implement three methods:

  1. run_simulation(env_state) -> env_state
     Your physics / domain logic.

  2. global_state_to_env_state(global_state) -> env_state
     Extract what your simulation needs from HERON's internal state dict.

  3. env_state_to_global_state(env_state) -> global_state
     Pack simulation results back into HERON's format.

This two-phase bridge gives you full control over what the simulation sees
and how results are mapped back to agent features.

Domain: Water tank system.
  - Two tanks connected by a pipe.
  - Each tank has a pump agent that controls inflow rate.
  - The simulation handles fluid dynamics (inflow, outflow, pipe transfer).
  - The environment tracks a global "total_water" metric.

Usage:
    cd "examples/3. building_environments"
    python custom_heron_env.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.base import HeronEnv
from heron.agents.constants import FIELD_LEVEL


# ---------------------------------------------------------------------------
# 1. Features and agents
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WaterLevelFeature(Feature):
    """Tank water level (public so coordinator can monitor all tanks)."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    level: float = 50.0         # current level (0-100)
    capacity: float = 100.0     # max capacity


class PumpAgent(FieldAgent):
    """Pump that controls water inflow rate into a tank."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        # Pump rate: -1 (drain) to +1 (fill)
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        if "level" in kwargs:
            self.state.features["WaterLevelFeature"].set_values(level=kwargs["level"])

    def apply_action(self) -> None:
        feat = self.state.features["WaterLevelFeature"]
        # Pump changes water level by action * 5 units per step
        new_level = feat.level + self.action.c[0] * 5.0
        feat.set_values(level=float(np.clip(new_level, 0.0, feat.capacity)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "WaterLevelFeature" not in local_state:
            return 0.0
        vec = local_state["WaterLevelFeature"]
        level, capacity = float(vec[0]), float(vec[1])
        # Reward: keep tank at 70% capacity
        target = 0.7 * capacity
        return -abs(level - target) / capacity


# ---------------------------------------------------------------------------
# 2. Custom environment state
# ---------------------------------------------------------------------------

@dataclass
class WaterSystemState:
    """Custom state object for the water system simulation.

    This is NOT a HERON class -- it's whatever your simulation needs.
    """
    tank_levels: Dict[str, float]
    total_water: float = 0.0
    step_count: int = 0


# ---------------------------------------------------------------------------
# 3. Custom HeronEnv subclass
# ---------------------------------------------------------------------------

class WaterTankEnv(HeronEnv):
    """Water tank environment with custom simulation bridge.

    Demonstrates the three abstract methods:
      - global_state_to_env_state: HERON dict -> WaterSystemState
      - run_simulation: WaterSystemState -> WaterSystemState (physics)
      - env_state_to_global_state: WaterSystemState -> HERON dict
    """

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> WaterSystemState:
        """Extract tank levels from HERON's internal state."""
        agent_states = global_state.get("agent_states", {})
        tank_levels = {}
        for aid, state_dict in agent_states.items():
            features = state_dict.get("features", {})
            if "WaterLevelFeature" in features:
                tank_levels[aid] = features["WaterLevelFeature"]["level"]

        total = sum(tank_levels.values())
        return WaterSystemState(tank_levels=tank_levels, total_water=total)

    def run_simulation(self, env_state: WaterSystemState, *args, **kwargs) -> WaterSystemState:
        """Fluid dynamics: natural drainage + pipe transfer between tanks."""
        levels = env_state.tank_levels

        # Natural drainage: each tank loses 2% per step
        for aid in levels:
            levels[aid] *= 0.98

        # Pipe transfer: water flows from higher to lower tank
        if len(levels) >= 2:
            ids = list(levels.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    diff = levels[ids[i]] - levels[ids[j]]
                    transfer = 0.1 * diff  # 10% of difference transfers
                    levels[ids[i]] -= transfer
                    levels[ids[j]] += transfer

        # Clip levels
        for aid in levels:
            levels[aid] = float(np.clip(levels[aid], 0.0, 100.0))

        env_state.total_water = sum(levels.values())
        env_state.step_count += 1
        return env_state

    def env_state_to_global_state(self, env_state: WaterSystemState) -> Dict[str, Any]:
        """Pack simulation results back into HERON format."""
        agent_states = {}
        for aid, level in env_state.tank_levels.items():
            agent = self.registered_agents.get(aid)
            if agent is None:
                continue
            # Read capacity from the agent's actual feature (don't hardcode)
            feat = agent.state.features["WaterLevelFeature"]
            agent_states[aid] = {
                "_owner_id": aid,
                "_owner_level": FIELD_LEVEL,
                "_state_type": "FieldAgentState",
                "features": {
                    "WaterLevelFeature": {
                        "level": level,
                        "capacity": feat.capacity,
                    }
                },
            }
        return {"agent_states": agent_states}


# ---------------------------------------------------------------------------
# 4. Assemble and run
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Custom HeronEnv: Water Tank System")
    print("=" * 60)

    # Create agents manually (full control)
    # Note: env.reset() resets features to their dataclass defaults,
    # so both tanks start at level=50.0 regardless of what we pass here.
    pump_a = PumpAgent(
        agent_id="tank_a",
        features=[WaterLevelFeature()],
    )
    pump_b = PumpAgent(
        agent_id="tank_b",
        features=[WaterLevelFeature()],
    )

    coordinator = CoordinatorAgent(
        agent_id="plant",
        subordinates={"tank_a": pump_a, "tank_b": pump_b},
    )

    # Build with the custom env class
    env = WaterTankEnv(
        coordinator_agents=[coordinator],
        env_id="water_tank_demo",
    )

    obs, _ = env.reset(seed=0)

    print(f"\nInitial state:")
    for aid in ["tank_a", "tank_b"]:
        if aid in obs:
            vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
            print(f"  {aid}: level={vec[0]:.1f}, capacity={vec[1]:.1f}")

    print(f"\nRunning 20 steps (pump_a fills, pump_b drains)...")

    # Create actions once, reuse each step
    action_a = Action()
    action_a.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
    action_b = action_a.copy()

    for step in range(20):
        action_a.set_values(c=[0.8])   # fill
        action_b.set_values(c=[-0.5])  # drain
        obs, rewards, terminated, truncated, infos = env.step(
            {"tank_a": action_a, "tank_b": action_b}
        )

        if (step + 1) % 5 == 0:
            print(f"\n  Step {step + 1}:")
            for aid in ["tank_a", "tank_b"]:
                if aid in obs:
                    vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
                    level = vec[0]
                    print(f"    {aid}: level={level:.1f}, reward={rewards.get(aid, 0):.3f}")

    # Show the data flow
    print(f"\n{'=' * 60}")
    print("Data Flow Summary")
    print("=" * 60)
    print("""
  env.step(actions)
    |
    v
  SystemAgent.execute()
    ├── Agents apply actions      (apply_action: pump changes level)
    ├── global_state_to_env_state (HERON dict -> WaterSystemState)
    ├── run_simulation            (drainage + pipe transfer)
    ├── env_state_to_global_state (WaterSystemState -> HERON dict)
    └── Agents observe new state  (proxy builds filtered observations)
""")
    print("Done.")


if __name__ == "__main__":
    main()

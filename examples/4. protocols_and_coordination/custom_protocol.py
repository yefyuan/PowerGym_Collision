"""Custom Protocol -- building your own coordination logic.

HERON protocols are composable: mix any CommunicationProtocol with any
ActionProtocol. When the built-in protocols don't fit your domain, you
can build custom ones by subclassing the base abstractions.

This script demonstrates:
1. Custom ActionProtocol -- weighted resource allocation
2. Custom CommunicationProtocol -- threshold-based alerts
3. Composing them into a full Protocol
4. Running the custom protocol through env.step()

Domain: Water distribution network.
  - A dispatcher (coordinator) allocates water flow to districts.
  - Districts have different demands (weights); allocation is proportional.
  - When a district's level drops below a threshold, an alert is sent.

Usage:
    cd "examples/4. protocols_and_coordination"
    python custom_protocol.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.simple import SimpleEnv
from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    NoCommunication,
)
from heron.utils.typing import AgentID


# ---------------------------------------------------------------------------
# 1. Custom ActionProtocol: weighted allocation
# ---------------------------------------------------------------------------

class WeightedAllocationActionProtocol(ActionProtocol):
    """Distributes coordinator action proportionally based on demand weights.

    Instead of splitting by vector dimension (VectorDecomposition) or
    broadcasting the same value (Broadcast), this protocol allocates
    a total resource budget proportionally to subordinate weights.

    Example:
        Total flow = 100
        Weights: district_a=0.5, district_b=0.3, district_c=0.2
        Result:  district_a=50,  district_b=30,  district_c=20
    """

    def __init__(self, weights: Optional[Dict[AgentID, float]] = None):
        self.weights = weights or {}

    def register_subordinates(self, subordinates: Dict[AgentID, Any]) -> None:
        """Auto-assign equal weights for subordinates without explicit weights."""
        for sub_id in subordinates:
            if sub_id not in self.weights:
                self.weights[sub_id] = 1.0

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        if coordinator_action is None or info_for_subordinates is None:
            return {sub_id: None for sub_id in (info_for_subordinates or {})}

        # Extract total budget from coordinator action
        if hasattr(coordinator_action, 'c'):
            total = float(coordinator_action.c[0]) if len(coordinator_action.c) > 0 else 0.0
        elif isinstance(coordinator_action, np.ndarray):
            total = float(coordinator_action[0]) if len(coordinator_action) > 0 else 0.0
        else:
            total = float(coordinator_action)

        # Compute proportional allocation
        sub_ids = list(info_for_subordinates.keys())
        total_weight = sum(self.weights.get(sid, 1.0) for sid in sub_ids)
        if total_weight == 0:
            total_weight = len(sub_ids)  # fallback to equal

        actions = {}
        for sub_id in sub_ids:
            w = self.weights.get(sub_id, 1.0)
            share = total * (w / total_weight)
            actions[sub_id] = np.array([share])
        return actions


# ---------------------------------------------------------------------------
# 2. Custom CommunicationProtocol: threshold alerts
# ---------------------------------------------------------------------------

class ThresholdAlertProtocol(CommunicationProtocol):
    """Sends alerts when subordinate states cross a threshold.

    The coordinator monitors subordinate states and sends "low_level"
    alerts to any subordinate whose level is below the threshold.
    This can be used for prioritized control or attention mechanisms.
    """

    def __init__(self, threshold: float = 30.0, field_name: str = "level"):
        self.threshold = threshold
        self.field_name = field_name

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_infos: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        messages = {}
        for agent_id, state in receiver_infos.items():
            # Extract the monitored field
            level = None
            if isinstance(state, dict) and self.field_name in state:
                level = state[self.field_name]

            if level is not None and level < self.threshold:
                messages[agent_id] = {
                    "type": "alert",
                    "alert": "low_level",
                    "current": level,
                    "threshold": self.threshold,
                }
            else:
                messages[agent_id] = {}  # No alert
        return messages


# ---------------------------------------------------------------------------
# 3. Compose into a full Protocol
# ---------------------------------------------------------------------------

class WaterDispatchProtocol(Protocol):
    """Custom protocol: weighted allocation + threshold alerts.

    Combines:
    - WeightedAllocationActionProtocol: proportional resource distribution
    - ThresholdAlertProtocol: warn when levels are critically low

    Usage:
        protocol = WaterDispatchProtocol(
            weights={"district_a": 0.5, "district_b": 0.3, "district_c": 0.2},
            alert_threshold=30.0,
        )
    """

    def __init__(
        self,
        weights: Optional[Dict[AgentID, float]] = None,
        alert_threshold: float = 30.0,
    ):
        self._action_protocol = WeightedAllocationActionProtocol(weights)
        super().__init__(
            communication_protocol=ThresholdAlertProtocol(
                threshold=alert_threshold,
                field_name="level",
            ),
            action_protocol=self._action_protocol,
        )

    def register_subordinates(self, subordinates: Dict[AgentID, Any]) -> None:
        super().register_subordinates(subordinates)
        self._action_protocol.register_subordinates(subordinates)


# ---------------------------------------------------------------------------
# 4. Domain: agents and simulation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WaterLevelFeature(Feature):
    """District water level."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    level: float = 50.0         # current level (0-100)
    demand: float = 5.0         # consumption per step


class DistrictAgent(FieldAgent):
    """District that consumes water and receives supply from dispatcher."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        # Inflow rate [-1, 1], mapped to [0, 10] units
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        if "level" in kwargs:
            self.state.features["WaterLevelFeature"].set_values(level=kwargs["level"])

    def apply_action(self) -> None:
        feat = self.state.features["WaterLevelFeature"]
        # Map action [-1, 1] to inflow [0, 10]
        inflow = (self.action.c[0] + 1.0) / 2.0 * 10.0
        new_level = feat.level + inflow - feat.demand
        feat.set_values(level=float(np.clip(new_level, 0.0, 100.0)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "WaterLevelFeature" not in local_state:
            return 0.0
        vec = local_state["WaterLevelFeature"]
        level = float(vec[0])
        # Penalize being below 30 or above 80
        if level < 30.0:
            return -(30.0 - level) / 30.0
        if level > 80.0:
            return -(level - 80.0) / 20.0
        return 0.0


def water_simulation(agent_states: dict) -> dict:
    """Water network: natural evaporation (2% per step)."""
    for aid, features in agent_states.items():
        if "WaterLevelFeature" in features:
            level = features["WaterLevelFeature"]["level"]
            features["WaterLevelFeature"]["level"] = level * 0.98
    return agent_states


# ---------------------------------------------------------------------------
# 5. Demonstrations
# ---------------------------------------------------------------------------

def demo_protocol_mechanics():
    """Show the custom protocol mechanics directly."""
    print("=" * 60)
    print("Part 1: Custom Protocol Mechanics")
    print("=" * 60)

    protocol = WaterDispatchProtocol(
        weights={"district_a": 0.5, "district_b": 0.3, "district_c": 0.2},
        alert_threshold=30.0,
    )

    # Simulate coordinator sending total_flow = 0.8
    coordinator_action = np.array([0.8])

    # Simulate subordinate states (for alert checking)
    sub_states = {
        "district_a": {"level": 45.0},  # OK
        "district_b": {"level": 25.0},  # Below threshold!
        "district_c": {"level": 60.0},  # OK
    }

    messages, actions = protocol.coordinate(
        coordinator_state=None,
        coordinator_action=coordinator_action,
        info_for_subordinates=sub_states,
    )

    print("\n  Action allocation (total_flow=0.8, weights=[0.5, 0.3, 0.2]):")
    for sub_id in sorted(actions):
        print(f"    {sub_id}: flow={actions[sub_id][0]:.3f}")

    print(f"\n  Alert messages (threshold=30.0):")
    for sub_id in sorted(messages):
        if messages[sub_id]:
            print(f"    {sub_id}: {messages[sub_id]}")
        else:
            print(f"    {sub_id}: (no alert)")


def demo_through_env():
    """Run the custom protocol through env.step()."""
    print("\n" + "=" * 60)
    print("Part 2: Custom Protocol through env.step()")
    print("=" * 60)

    district_a = DistrictAgent(
        agent_id="district_a",
        features=[WaterLevelFeature(level=50.0, demand=3.0)],
    )
    district_b = DistrictAgent(
        agent_id="district_b",
        features=[WaterLevelFeature(level=50.0, demand=7.0)],  # High demand
    )
    district_c = DistrictAgent(
        agent_id="district_c",
        features=[WaterLevelFeature(level=50.0, demand=5.0)],
    )

    protocol = WaterDispatchProtocol(
        weights={"district_a": 0.3, "district_b": 0.5, "district_c": 0.2},
        alert_threshold=30.0,
    )

    coordinator = CoordinatorAgent(
        agent_id="dispatcher",
        subordinates={
            "district_a": district_a,
            "district_b": district_b,
            "district_c": district_c,
        },
        protocol=protocol,
    )

    env = SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=water_simulation,
        env_id="water_dispatch_demo",
    )

    obs, _ = env.reset(seed=0)

    # Dispatcher sends total flow budget
    dispatch_action = Action()
    dispatch_action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))

    print(f"\n  Running 10 steps (dispatcher sends flow=0.6, weights=[0.3, 0.5, 0.2]):")
    for step in range(10):
        dispatch_action.set_values(c=[0.6])
        obs, rewards, _, _, _ = env.step({"dispatcher": dispatch_action})

        if step == 0 or (step + 1) % 3 == 0:
            print(f"\n    Step {step + 1}:")
            for aid in ["district_a", "district_b", "district_c"]:
                if aid in obs:
                    vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
                    level = vec[0]
                    status = "LOW!" if level < 30.0 else "ok"
                    print(f"      {aid}: level={level:5.1f}, reward={rewards.get(aid, 0):.3f} [{status}]")

    print("\n  Note: district_b has highest demand (7.0) but gets most supply (50% weight)")


def demo_building_blocks():
    """Show how to build protocols from individual components."""
    print("\n" + "=" * 60)
    print("Part 3: Protocol Composition Pattern")
    print("=" * 60)
    print("""
  HERON protocols are composed from two independent pieces:

  Protocol = CommunicationProtocol + ActionProtocol

  Built-in options:
    Communication:
      NoCommunication         -- no messages
      StateShareProtocol      -- share state with neighbors

    Action:
      NoActionCoordination    -- agents act independently
      VectorDecomposition     -- split joint action by dims
      BroadcastAction         -- same action to all

  Custom (this example):
    ThresholdAlertProtocol    -- send alerts on low levels
    WeightedAllocation        -- proportional resource split

  Mix and match:
    VerticalProtocol(action_protocol=BroadcastActionProtocol())
    HorizontalProtocol(state_fields=["temp"], topology=ring)
    WaterDispatchProtocol(weights={...}, alert_threshold=30)

  The Protocol.coordinate() method orchestrates both:
    1. communication_protocol.compute_coordination_messages()
    2. action_protocol.compute_action_coordination()
    -> returns (messages, actions)
""")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    demo_protocol_mechanics()
    demo_through_env()
    demo_building_blocks()
    print("Done.")


if __name__ == "__main__":
    main()

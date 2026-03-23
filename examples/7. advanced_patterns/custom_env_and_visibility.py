"""Custom HeronEnv and Feature Visibility.

HERON supports:
  - Feature visibility: "public", "owner", "system", "upper_level"
  - Custom HeronEnv subclasses with domain-specific simulation logic
  - State.observed_by() for visibility-filtered observation

This script demonstrates:
1. Visibility modes -- what each level can see
2. State.observed_by() -- automatic filtering
3. Custom HeronEnv -- subclassing with abstract method overrides
4. pre_step() hook -- per-step environment setup
5. End-to-end run -- custom env with visibility in action

Domain: Microgrid with solar farm (public power, owner-only degradation)
        and battery (public power, owner-only SoC).

Usage:
    cd "examples/7. advanced_patterns"
    python custom_env_and_visibility.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.state import FieldAgentState
from heron.envs.base import HeronEnv
from heron.agents.constants import FIELD_LEVEL, COORDINATOR_LEVEL, SYSTEM_LEVEL


# ---------------------------------------------------------------------------
# 1. Feature visibility modes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PublicFeature(Feature):
    """Visible to ALL agents (peers, coordinators, system)."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    power_output: float = 0.0


@dataclass(slots=True)
class OwnerOnlyFeature(Feature):
    """Visible ONLY to the owning agent."""
    visibility: ClassVar[Sequence[str]] = ["owner"]
    degradation: float = 0.0


@dataclass(slots=True)
class SystemFeature(Feature):
    """Visible to system-level agents (level >= 3) only."""
    visibility: ClassVar[Sequence[str]] = ["system"]
    maintenance_flag: float = 0.0


@dataclass(slots=True)
class UpperLevelFeature(Feature):
    """Visible to the agent one level above the owner."""
    visibility: ClassVar[Sequence[str]] = ["upper_level"]
    efficiency: float = 0.95


@dataclass(slots=True)
class MixedVisibilityFeature(Feature):
    """Visible to owner AND upper level (but not peers)."""
    visibility: ClassVar[Sequence[str]] = ["owner", "upper_level"]
    internal_temp: float = 25.0


def demo_visibility_modes():
    """Show how is_observable_by() works for each visibility type."""
    print("=" * 60)
    print("Part 1: Feature Visibility Modes")
    print("=" * 60)
    print("""
  Each Feature has a visibility class variable:
    "public"      -- visible to everyone
    "owner"       -- visible only to the owning agent
    "system"      -- visible to system-level agents (level >= 3)
    "upper_level" -- visible to the agent one level above owner
  Multiple modes can be combined: ["owner", "upper_level"]
""")

    features = {
        "PublicFeature": PublicFeature(),
        "OwnerOnlyFeature": OwnerOnlyFeature(),
        "SystemFeature": SystemFeature(),
        "UpperLevelFeature": UpperLevelFeature(),
        "MixedVisibilityFeature": MixedVisibilityFeature(),
    }

    # Define requestors at different levels
    requestors = [
        ("battery_1", FIELD_LEVEL, "peer field agent"),
        ("solar_1", FIELD_LEVEL, "owner (self)"),        # same as owner_id below
        ("grid_op", COORDINATOR_LEVEL, "coordinator (L2)"),
        ("sys", SYSTEM_LEVEL, "system agent (L3)"),
    ]
    owner_id = "solar_1"
    owner_level = FIELD_LEVEL

    header = f"  {'Feature':<28}"
    for _, _, label in requestors:
        header += f" {label:>16}"
    print(header)
    print(f"  {'-' * (28 + 17 * len(requestors))}")

    for fname, feat in features.items():
        row = f"  {fname:<28}"
        for req_id, req_level, _ in requestors:
            visible = feat.is_observable_by(req_id, req_level, owner_id, owner_level)
            row += f" {'yes':>16}" if visible else f" {'--':>16}"
        print(row)


# ---------------------------------------------------------------------------
# 2. State.observed_by() filtering
# ---------------------------------------------------------------------------

def demo_observed_by():
    """State.observed_by() returns only visible features as numpy arrays."""
    print("\n" + "=" * 60)
    print("Part 2: State.observed_by() Filtering")
    print("=" * 60)
    print("""
  State.observed_by(requestor_id, requestor_level) returns a dict
  of {feature_name: numpy_array} containing only visible features.
""")

    # Build a state with mixed visibility features
    state = FieldAgentState(owner_id="solar_1", owner_level=FIELD_LEVEL)
    state.features = {
        "PublicFeature": PublicFeature(power_output=42.0),
        "OwnerOnlyFeature": OwnerOnlyFeature(degradation=0.05),
        "UpperLevelFeature": UpperLevelFeature(efficiency=0.92),
    }

    print(f"  State has {len(state.features)} features: {list(state.features.keys())}")
    print(f"  Full state vector: {state.vector()}\n")

    # Owner sees everything marked "public" or "owner"
    owner_obs = state.observed_by("solar_1", FIELD_LEVEL)
    print(f"  Owner (solar_1, L{FIELD_LEVEL}) sees:")
    for name, vec in owner_obs.items():
        print(f"    {name}: {vec}")

    # Peer sees only "public"
    peer_obs = state.observed_by("battery_1", FIELD_LEVEL)
    print(f"\n  Peer (battery_1, L{FIELD_LEVEL}) sees:")
    for name, vec in peer_obs.items():
        print(f"    {name}: {vec}")
    if not peer_obs:
        print("    (nothing)")

    # Coordinator sees "public" + "upper_level"
    coord_obs = state.observed_by("grid_op", COORDINATOR_LEVEL)
    print(f"\n  Coordinator (grid_op, L{COORDINATOR_LEVEL}) sees:")
    for name, vec in coord_obs.items():
        print(f"    {name}: {vec}")

    # System sees "public" + "system"
    sys_obs = state.observed_by("sys", SYSTEM_LEVEL)
    print(f"\n  System (sys, L{SYSTEM_LEVEL}) sees:")
    for name, vec in sys_obs.items():
        print(f"    {name}: {vec}")


# ---------------------------------------------------------------------------
# 3. Custom HeronEnv subclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SolarFeature(Feature):
    """Solar farm: public output and curtailment setting."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    power: float = 0.0       # actual output after irradiance (set by simulation)
    curtailment: float = 1.0  # agent's curtailment factor 0-1 (set by apply_action)


@dataclass(slots=True)
class SolarPrivateFeature(Feature):
    """Private metrics only the solar agent itself sees."""
    visibility: ClassVar[Sequence[str]] = ["owner"]
    panel_degradation: float = 0.0


@dataclass(slots=True)
class BatteryFeature(Feature):
    """Battery: public power flow, owner-only state of charge."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    power: float = 0.0
    soc: float = 0.5


class SolarAgent(FieldAgent):
    """Solar farm with mixed-visibility features."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(np.array([0.0]), np.array([1.0])),  # curtailment factor
        )
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        curtailment = self.action.c[0]
        feat = self.state.features["SolarFeature"]
        # Store curtailment factor; run_simulation computes actual power
        feat.set_values(curtailment=curtailment)

    def compute_local_reward(self, local_state: dict) -> float:
        if "SolarFeature" not in local_state:
            return 0.0
        vec = local_state["SolarFeature"]  # numpy array [power, curtailment]
        power = float(vec[0])
        return power / 100.0  # reward for generation


class BatteryAgent(FieldAgent):
    """Battery with charge/discharge control."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(np.array([-50.0]), np.array([50.0])),  # MW charge/discharge
        )
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        power = self.action.c[0]
        feat = self.state.features["BatteryFeature"]
        soc = feat.soc + power * 0.01  # simplified SoC update
        soc = float(np.clip(soc, 0.0, 1.0))
        feat.set_values(power=power, soc=soc)

    def compute_local_reward(self, local_state: dict) -> float:
        if "BatteryFeature" not in local_state:
            return 0.0
        vec = local_state["BatteryFeature"]  # numpy array [power, soc]
        soc = float(vec[1])
        return -abs(soc - 0.5)  # keep SoC near 50%


class MicrogridEnv(HeronEnv):
    """Custom HeronEnv for a microgrid with irradiance profiles.

    Demonstrates the three abstract methods every HeronEnv must implement:
      - run_simulation(env_state)
      - global_state_to_env_state(global_state)
      - env_state_to_global_state(env_state)

    Also demonstrates pre_step() for time-varying inputs.
    """

    def __init__(
        self,
        irradiance_profile: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        self._irradiance_profile = irradiance_profile or [0.8, 0.9, 1.0, 0.7, 0.5]
        self._step_count = 0
        self._current_irradiance = 1.0
        super().__init__(**kwargs)

    def pre_step(self) -> None:
        """Update irradiance from profile before each step.

        NOTE: pre_step stores env-level data (e.g., profiles, prices) as
        instance variables rather than modifying agent state directly.
        Agent state gets overwritten by sync_state_from_observed() during
        the execute() cycle, so env-level inputs should flow through
        run_simulation() instead.
        """
        idx = self._step_count % len(self._irradiance_profile)
        self._current_irradiance = self._irradiance_profile[idx]
        self._step_count += 1

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract flat agent states from HERON's global state."""
        agent_states_raw = global_state.get("agent_states", {})
        flat: Dict[str, Dict] = {}
        for agent_id, state_dict in agent_states_raw.items():
            features = state_dict.get("features", state_dict)
            flat[agent_id] = {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in features.items()
                if not k.startswith("_")
            }
        return flat

    def env_state_to_global_state(self, env_state: Dict[str, Dict]) -> Dict[str, Any]:
        """Pack flat states back into HERON format."""
        agent_states: Dict[str, Any] = {}
        for agent_id, features_dict in env_state.items():
            agent = self.registered_agents.get(agent_id)
            if agent is None:
                continue
            level = getattr(agent, "level", FIELD_LEVEL)
            level_to_type = {
                FIELD_LEVEL: "FieldAgentState",
                COORDINATOR_LEVEL: "CoordinatorAgentState",
                SYSTEM_LEVEL: "SystemAgentState",
            }
            agent_states[agent_id] = {
                "_owner_id": agent_id,
                "_owner_level": level,
                "_state_type": level_to_type.get(level, "FieldAgentState"),
                "features": features_dict,
            }
        return {"agent_states": agent_states}

    def run_simulation(self, env_state: Any, *args: Any, **kwargs: Any) -> Any:
        """Microgrid physics: apply irradiance, compute net power, update battery."""
        # Compute solar power: curtailment * irradiance * capacity
        for aid, features in env_state.items():
            if "SolarFeature" in features:
                curtailment = features["SolarFeature"]["curtailment"]
                power = curtailment * self._current_irradiance * 100.0  # 100 MW capacity
                features["SolarFeature"]["power"] = power

        # Compute net generation
        total_generation = 0.0
        demand = 60.0  # fixed demand in MW
        for aid, features in env_state.items():
            if "SolarFeature" in features:
                total_generation += features["SolarFeature"]["power"]

        # Net surplus/deficit flows to battery
        net = total_generation - demand
        for aid, features in env_state.items():
            if "BatteryFeature" in features:
                battery_power = float(np.clip(net, -50.0, 50.0))
                soc = features["BatteryFeature"]["soc"]
                new_soc = float(np.clip(soc + battery_power * 0.01, 0.0, 1.0))
                features["BatteryFeature"]["power"] = battery_power
                features["BatteryFeature"]["soc"] = new_soc

            # Degrade solar panels slightly each step
            if "SolarPrivateFeature" in features:
                deg = features["SolarPrivateFeature"]["panel_degradation"]
                features["SolarPrivateFeature"]["panel_degradation"] = deg + 0.001

        return env_state


def demo_custom_env():
    """Build and run the custom MicrogridEnv."""
    print("\n" + "=" * 60)
    print("Part 3: Custom HeronEnv Subclass")
    print("=" * 60)
    print("""
  HeronEnv requires three abstract methods:
    run_simulation(env_state)          -- domain physics
    global_state_to_env_state(gs)      -- HERON state -> your format
    env_state_to_global_state(es)      -- your format -> HERON state

  Optional: pre_step() -- hook called before each step.
""")

    solar = SolarAgent(
        agent_id="solar_1",
        features=[SolarFeature(), SolarPrivateFeature()],
    )
    battery = BatteryAgent(
        agent_id="battery_1",
        features=[BatteryFeature()],
    )

    coordinator = CoordinatorAgent(
        agent_id="grid_op",
        subordinates={"solar_1": solar, "battery_1": battery},
    )

    env = MicrogridEnv(
        coordinator_agents=[coordinator],
        irradiance_profile=[0.8, 0.9, 1.0, 0.7, 0.5],
        env_id="microgrid_demo",
    )

    obs, _ = env.reset(seed=0)

    print(f"  Registered agents: {[aid for aid in env.registered_agents if not aid.startswith(('system', 'proxy'))]}")
    print(f"  Solar features: {list(solar.state.features.keys())}")
    print(f"  Battery features: {list(battery.state.features.keys())}")

    # Run steps
    print(f"\n  Running 5 steps with varying irradiance [0.8, 0.9, 1.0, 0.7, 0.5]:")
    print(f"  {'Step':<6} {'Irrad':>7} {'Solar MW':>10} {'Batt MW':>10} {'SoC':>8}")
    print(f"  {'-' * 43}")

    irradiance_profile = [0.8, 0.9, 1.0, 0.7, 0.5]
    for step in range(5):
        solar_action = Action()
        solar_action.set_specs(dim_c=1, range=(np.array([0.0]), np.array([1.0])))
        solar_action.set_values(c=[0.9])  # 90% curtailment factor

        battery_action = Action()
        battery_action.set_specs(dim_c=1, range=(np.array([-50.0]), np.array([50.0])))
        battery_action.set_values(c=[0.0])  # let simulation handle battery

        obs, rewards, _, _, _ = env.step({
            "solar_1": solar_action,
            "battery_1": battery_action,
        })

        solar_obs = obs["solar_1"].vector() if hasattr(obs["solar_1"], "vector") else np.asarray(obs["solar_1"])
        batt_obs = obs["battery_1"].vector() if hasattr(obs["battery_1"], "vector") else np.asarray(obs["battery_1"])
        print(f"  {step + 1:<6} {irradiance_profile[step]:>7.1f} {solar_obs[0]:>10.1f} {batt_obs[0]:>10.1f} {batt_obs[1]:>8.3f}")


# ---------------------------------------------------------------------------
# 4. Visibility in action -- what each agent observes
# ---------------------------------------------------------------------------

def demo_visibility_in_env():
    """Show how visibility affects observations in a running environment."""
    print("\n" + "=" * 60)
    print("Part 4: Visibility in a Running Environment")
    print("=" * 60)
    print("""
  In the microgrid:
    SolarFeature     (public)  -- all agents see solar power
    SolarPrivateFeature (owner)  -- only solar_1 sees degradation
    BatteryFeature   (public)  -- all agents see battery state

  solar_1 sees: SolarFeature + SolarPrivateFeature + BatteryFeature (peer public)
  battery_1 sees: BatteryFeature + SolarFeature (peer public)
  Neither peer sees the other's owner-only features.
""")

    solar = SolarAgent(
        agent_id="solar_1",
        features=[SolarFeature(power=80.0, curtailment=0.9), SolarPrivateFeature(panel_degradation=0.02)],
    )
    battery = BatteryAgent(
        agent_id="battery_1",
        features=[BatteryFeature(power=10.0, soc=0.6)],
    )

    # Demonstrate State.observed_by() for each agent
    print(f"  solar_1 state features: {list(solar.state.features.keys())}")
    print(f"  battery_1 state features: {list(battery.state.features.keys())}")

    # What solar_1 sees of its own state
    solar_self_obs = solar.state.observed_by("solar_1", FIELD_LEVEL)
    print(f"\n  solar_1 observes own state:")
    for name, vec in solar_self_obs.items():
        print(f"    {name}: {vec}")

    # What battery_1 sees of solar_1's state
    battery_sees_solar = solar.state.observed_by("battery_1", FIELD_LEVEL)
    print(f"\n  battery_1 observes solar_1's state (peer, only public):")
    for name, vec in battery_sees_solar.items():
        print(f"    {name}: {vec}")

    # What coordinator sees of solar_1's state
    coord_sees_solar = solar.state.observed_by("grid_op", COORDINATOR_LEVEL)
    print(f"\n  grid_op (coordinator) observes solar_1's state:")
    for name, vec in coord_sees_solar.items():
        print(f"    {name}: {vec}")

    # Dimension differences
    print(f"\n  Observation dimensions differ by viewer:")
    print(f"    solar_1 sees own state: {sum(v.size for v in solar_self_obs.values())} dims")
    print(f"    battery_1 sees solar:   {sum(v.size for v in battery_sees_solar.values())} dims")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    demo_visibility_modes()
    demo_observed_by()
    demo_custom_env()
    demo_visibility_in_env()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  Visibility Modes:

    "public"       visible to all agents at any level
    "owner"        visible only to the agent that owns the feature
    "system"       visible to system-level agents (level >= 3)
    "upper_level"  visible to agents one level above the owner
    Combined:      ["owner", "upper_level"] -- visible to owner + supervisor

  Custom HeronEnv:

    class MyEnv(HeronEnv):
        def run_simulation(self, env_state):         # domain physics
        def global_state_to_env_state(self, gs):     # HERON -> your format
        def env_state_to_global_state(self, es):     # your format -> HERON
        def pre_step(self):                          # optional per-step hook
""")
    print("Done.")


if __name__ == "__main__":
    main()

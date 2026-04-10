"""Observations & State -- how HERON composes and filters agent state.

State is composed of multiple Features, each with its own visibility.
When an agent *observes* another agent's state, HERON automatically filters
features based on visibility rules, producing an Observation with:
  - `local`       : the observing agent's own state
  - `global_info` : other agents' states, filtered by visibility

This script demonstrates:
1. Building State from features
2. Visibility-filtered observation via `state.observed_by()`
3. Observation structure (local vs global, vectorization)
4. Serialization round-trip (for async message passing)

Usage:
    cd "examples/2. core_abstractions"
    python observations_and_state.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np

from heron.core.feature import Feature
from heron.core.state import FieldAgentState, CoordinatorAgentState
from heron.core.observation import Observation


# ---------------------------------------------------------------------------
# 1. Define features with different visibility
# ---------------------------------------------------------------------------

# NOTE: We define features here for self-containedness.
# In real projects, share them via a features.py module (see examples/1. starter/).


@dataclass(slots=True)
class BatteryCharge(Feature):
    """State of charge -- visible to owner and direct supervisor."""
    visibility: ClassVar[Sequence[str]] = ["owner", "upper_level"]
    soc: float = 0.5
    capacity_kwh: float = 100.0


@dataclass(slots=True)
class TemperatureSensor(Feature):
    """Internal temperature -- private to the owning agent."""
    visibility: ClassVar[Sequence[str]] = ["owner"]
    temp_celsius: float = 25.0


@dataclass(slots=True)
class GridVoltage(Feature):
    """Bus voltage -- public information."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    voltage_pu: float = 1.0


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main():
    # ------------------------------------------------------------------
    # 1. Build a FieldAgentState from features
    # ------------------------------------------------------------------
    section("1. Building State from Features")

    features = [BatteryCharge(soc=0.8), TemperatureSensor(temp_celsius=32.0), GridVoltage()]

    state = FieldAgentState(
        owner_id="battery_0",
        owner_level=1,
        features={f.feature_name: f for f in features},
    )

    print(f"Owner       : {state.owner_id} (level {state.owner_level})")
    print(f"Features    : {list(state.features.keys())}")
    print(f"Full vector : {state.vector()}")

    # ------------------------------------------------------------------
    # 2. Visibility-filtered observation
    # ------------------------------------------------------------------
    section("2. Visibility-Filtered Observations")

    observers = [
        ("battery_0", 1, "Owner (self)"),
        ("battery_1", 1, "Peer (same level)"),
        ("zone_0",    2, "Coordinator (level 2, direct supervisor)"),
        ("system",    3, "System agent (level 3)"),
    ]

    for obs_id, obs_level, label in observers:
        visible = state.observed_by(obs_id, obs_level)
        visible_names = list(visible.keys())
        vec = np.concatenate(list(visible.values())) if visible else np.array([])
        print(f"  {label}")
        print(f"    visible: {visible_names}")
        print(f"    vec    : {vec}")

    # ------------------------------------------------------------------
    # 3. Observation structure (local + global_info)
    # ------------------------------------------------------------------
    section("3. Observation Structure")

    # Simulate what the proxy would build for battery_0
    local_obs = state.observed_by("battery_0", 1)  # sees everything (owner)

    # Simulate a peer's public state as global_info
    peer_state = FieldAgentState(owner_id="battery_1", owner_level=1)
    peer_state.features["GridVoltage"] = GridVoltage(voltage_pu=0.98)
    peer_state.features["BatteryCharge"] = BatteryCharge(soc=0.3)

    global_obs = peer_state.observed_by("battery_0", 1)  # only public features

    obs = Observation(
        local=local_obs,
        global_info={"battery_1": global_obs},
        timestamp=1.0,
    )

    print(f"Local keys   : {list(obs.local.keys())}")
    print(f"Global keys  : {list(obs.global_info.keys())}")
    print(f"Local vector : {obs.local_vector()}")
    print(f"Global vector: {obs.global_vector()}")
    print(f"Full vector  : {obs.vector()}")
    print(f"Shape        : {obs.shape}")
    print(f"Timestamp    : {obs.timestamp}")

    # Array-like access
    print(f"obs[0]       : {obs[0]}")
    print(f"len(obs)     : {len(obs)}")

    # ------------------------------------------------------------------
    # 4. State update operations
    # ------------------------------------------------------------------
    section("4. State Updates")

    print(f"Before       : soc={state.features['BatteryCharge'].soc}")

    # Single feature update
    state.update_feature("BatteryCharge", soc=0.6)
    print(f"After update : soc={state.features['BatteryCharge'].soc}")

    # Batch update
    state.update({
        "BatteryCharge": {"soc": 0.9, "capacity_kwh": 120.0},
        "TemperatureSensor": {"temp_celsius": 28.0},
    })
    print(f"After batch  : soc={state.features['BatteryCharge'].soc}, "
          f"cap={state.features['BatteryCharge'].capacity_kwh}, "
          f"temp={state.features['TemperatureSensor'].temp_celsius}")

    # Reset to defaults
    state.reset()
    print(f"After reset  : soc={state.features['BatteryCharge'].soc}, "
          f"temp={state.features['TemperatureSensor'].temp_celsius}")

    # ------------------------------------------------------------------
    # 5. Serialization round-trip
    # ------------------------------------------------------------------
    section("5. Serialization (for async message passing)")

    obs2 = Observation(
        local={"BatteryCharge": np.array([0.8, 100.0], dtype=np.float32)},
        global_info={"battery_1": np.array([0.98], dtype=np.float32)},
        timestamp=2.5,
    )

    serialized = obs2.to_dict()
    print(f"Serialized   : {serialized}")

    restored = Observation.from_dict(serialized)
    print(f"Restored     : local={restored.local}, global={restored.global_info}")
    print(f"Vectors match: {np.allclose(obs2.vector(), restored.vector())}")

    # State serialization
    state.update_feature("BatteryCharge", soc=0.75)
    state_dict = state.to_dict(include_metadata=True)
    print(f"\nState dict   : {state_dict}")

    restored_state = FieldAgentState.from_dict(state_dict)
    print(f"Restored SOC : {restored_state.features['BatteryCharge'].soc}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Features & Visibility -- HERON's declarative observability model.

HERON's core differentiator is that information structure is *declared*, not
coded.  Each Feature subclass carries a `visibility` class variable
that controls which agents in the hierarchy can observe it.

Visibility levels
-----------------
- "public"      : visible to every agent in the environment
- "owner"       : visible only to the agent that owns the feature
- "upper_level" : visible to the agent one hierarchy level above the owner
- "system"      : visible only to system-level agents (level >= 3)

This script defines four features (one per visibility level) and shows
exactly who can see what across a three-level hierarchy:

    SystemAgent  (level 3)
    └── Coordinator (level 2)
        └── FieldAgent (level 1)   <-- owns all four features

Usage:
    cd "examples/2. core_abstractions"
    python features_and_visibility.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np

from heron.core.feature import Feature


# ---------------------------------------------------------------------------
# 1. Define features with different visibility levels
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PublicMetric(Feature):
    """Visible to everyone -- e.g. a published price signal."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 1.0


@dataclass(slots=True)
class PrivateSensor(Feature):
    """Visible only to the owning agent -- e.g. internal diagnostics."""
    visibility: ClassVar[Sequence[str]] = ["owner"]
    reading: float = 42.0


@dataclass(slots=True)
class SupervisorReport(Feature):
    """Visible to the owner AND its direct supervisor (one level up)."""
    visibility: ClassVar[Sequence[str]] = ["owner", "upper_level"]
    status: float = 0.8


@dataclass(slots=True)
class SystemTelemetry(Feature):
    """Visible to the owner AND system-level agents (level >= 3)."""
    visibility: ClassVar[Sequence[str]] = ["owner", "system"]
    health: float = 0.95


# ---------------------------------------------------------------------------
# 2. Demonstrate the visibility rules
# ---------------------------------------------------------------------------

def main():
    # Instantiate features (all owned by a field agent at level 1)
    features = [PublicMetric(), PrivateSensor(), SupervisorReport(), SystemTelemetry()]

    owner_id, owner_level = "field_0", 1

    # Agents that might want to observe these features
    requestors = [
        ("field_0",      1, "Owner (field agent, L1)"),
        ("field_1",      1, "Peer (another field agent, L1)"),
        ("coordinator_0", 2, "Direct supervisor (coordinator, L2)"),
        ("system",       3, "System agent (L3)"),
    ]

    print("=" * 68)
    print("Feature Visibility Matrix")
    print(f"All features owned by '{owner_id}' at level {owner_level}")
    print("=" * 68)

    # Header
    feat_names = [type(f).__name__ for f in features]
    header = f"{'Requestor':<38}" + "".join(f"{n:<20}" for n in feat_names)
    print(header)
    print("-" * len(header))

    for req_id, req_level, label in requestors:
        row = f"{label:<38}"
        for feat in features:
            visible = feat.is_observable_by(req_id, req_level, owner_id, owner_level)
            row += f"{'YES':<20}" if visible else f"{'--':<20}"
        print(row)

    # ---------------------------------------------------------------------------
    # 3. Demonstrate feature data operations
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Feature Data Operations")
    print("=" * 68)

    sensor = PrivateSensor(reading=3.14)

    print(f"\nFeature name : {sensor.feature_name}")
    print(f"Field names  : {sensor.names()}")
    print(f"Vector       : {sensor.vector()}")
    print(f"To dict      : {sensor.to_dict()}")

    # set_values updates fields
    sensor.set_values(reading=99.0)
    print(f"After update : {sensor.vector()}")

    # reset restores defaults, then applies overrides
    sensor.reset(reading=7.0)
    print(f"After reset  : {sensor.vector()}")

    # from_dict round-trip
    restored = PrivateSensor.from_dict({"reading": 2.71})
    print(f"From dict    : {restored.vector()}")

    # ---------------------------------------------------------------------------
    # 4. Custom feature name (instance-level override)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Custom Feature Names")
    print("=" * 68)

    a = PublicMetric(value=1.0)
    b = PublicMetric(value=2.0).set_feature_name("PublicMetric_zone_b")

    print(f"Default name : {a.feature_name}")
    print(f"Custom name  : {b.feature_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

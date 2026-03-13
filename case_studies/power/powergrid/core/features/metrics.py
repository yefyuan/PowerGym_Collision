"""Cost and safety metrics feature provider."""

from dataclasses import dataclass
from typing import ClassVar, List, Sequence

import numpy as np

from heron.core.feature import Feature


@dataclass(slots=True)
class CostSafetyMetrics(Feature):
    """Provider for cost and safety tracking.

    Used by device agents to track operating costs and safety violations.
    The proxy agent aggregates these for reward computation.

    Attributes:
        cost: Operating cost for current timestep (e.g., fuel cost, degradation)
        safety: Safety penalty/violation metric (e.g., constraint violations)
    """
    visibility: ClassVar[Sequence[str]] = ["public"]

    cost: float = 0.0
    safety: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.cost, self.safety], dtype=np.float32)

    def names(self) -> List[str]:
        return ["cost", "safety"]

    def clamp_(self) -> None:
        # Cost and safety should be non-negative
        self.cost = max(0.0, self.cost)
        self.safety = max(0.0, self.safety)

    def to_dict(self) -> dict:
        return {
            "cost": self.cost,
            "safety": self.safety,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CostSafetyMetrics":
        return cls(
            cost=d.get("cost", 0.0),
            safety=d.get("safety", 0.0),
        )

    def set_values(self, **kwargs) -> None:
        """Update cost/safety fields.

        Args:
            **kwargs: Field names and values to update (cost, safety)
        """
        allowed_keys = {"cost", "safety"}

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"CostSafetyMetrics.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.clamp_()

    def reset(self) -> None:
        """Reset metrics to zero."""
        self.cost = 0.0
        self.safety = 0.0

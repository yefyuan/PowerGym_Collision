"""Feature providers shared across HMARL-CBF case studies."""

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np

from heron.core.feature import Feature


@dataclass(slots=True)
class DronePositionFeature(Feature):
    """Drone position on the warehouse floor (normalized to [0,1])."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    x_pos: float = 0.5
    y_pos: float = 0.5

    def set_values(self, **kwargs: Any) -> None:
        if "x_pos" in kwargs:
            self.x_pos = float(np.clip(kwargs["x_pos"], 0.0, 1.0))
        if "y_pos" in kwargs:
            self.y_pos = float(np.clip(kwargs["y_pos"], 0.0, 1.0))


@dataclass(slots=True)
class FleetSafetyFeature(Feature):
    """Fleet-level aggregate: separation and delivery progress."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    mean_separation: float = 0.5
    payload_progress: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "mean_separation" in kwargs:
            self.mean_separation = float(np.clip(kwargs["mean_separation"], 0.0, 1.0))
        if "payload_progress" in kwargs:
            self.payload_progress = float(np.clip(kwargs["payload_progress"], 0.0, 1.0))

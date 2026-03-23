"""Step-based discrete state feature provider."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from heron.core.feature import Feature
from powergrid.utils.phase import PhaseModel, PhaseSpec


@dataclass(slots=True)
class StepState(Feature):
    """Provider for discrete step state (e.g., shunt capacitor banks)."""
    max_step: int = 0
    step: Optional[np.ndarray] = None  # One-hot encoded current step

    def vector(self) -> np.ndarray:
        if self.step is not None:
            return self.step.astype(np.float32, copy=False)
        return np.zeros(self.max_step + 1, dtype=np.float32)

    def names(self) -> List[str]:
        return [f"step_{i}" for i in range(self.max_step + 1)]

    def clamp_(self) -> None:
        if self.step is not None:
            # Ensure one-hot encoding
            self.step = self.step.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "max_step": self.max_step,
            "step": None if self.step is None else self.step.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepState":
        step_data = d.get("step")
        return cls(
            max_step=d.get("max_step", 0),
            step=None if step_data is None else np.array(step_data, dtype=np.float32),
        )

    def to_phase_model(self, model: PhaseModel, spec: PhaseSpec, policy=None) -> "StepState":
        return self

    def set_values(self, **kwargs) -> None:
        """Update step state fields.

        Args:
            **kwargs: Field names and values to update
        """
        allowed_keys = {"max_step", "step"}

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"StepState.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.clamp_()
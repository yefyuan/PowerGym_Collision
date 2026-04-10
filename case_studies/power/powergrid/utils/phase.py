"""Phase model utilities for power systems.

This module provides phase configuration and manipulation for
single-phase and three-phase power systems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple

import numpy as np


def remove_duplicate_chars_keep_order(input_string):
    """Remove duplicate characters while preserving order."""
    seen_chars = set()
    result_chars = []
    for char in input_string.upper():
        if char not in seen_chars and char in "ABC":
            seen_chars.add(char)
            result_chars.append(char)
    return "".join(result_chars)


class PhaseModel(Enum):
    """Phase model enumeration."""
    BALANCED_1PH = "balanced_1ph"
    THREE_PHASE = "three_phase"


@dataclass(slots=True)
class PhaseSpec:
    """Phase specification for electrical connections.

    Attributes:
        phases: Phase string (e.g., "A", "AB", "ABC")
        has_neutral: Whether neutral conductor is present
        earth_bond: Whether neutral is earthed
    """
    phases: str = ""  # e.g. "A", "AB", "ABC" (order matters in names/arrays)
    has_neutral: bool = False
    earth_bond: bool = False

    def __post_init__(self):
        # sanitize phases: uppercase, keep subset of ABC, canonical ABC order
        self.phases = remove_duplicate_chars_keep_order(self.phases)
        # if no neutral, cannot have earth bond
        if not self.has_neutral and self.earth_bond:
            self.earth_bond = False

    @property
    def nph(self) -> int:
        """Number of phases."""
        return len(self.phases)

    def index(self, ph: str) -> int:
        """Get index of phase in phases string."""
        return self.phases.index(ph)

    @classmethod
    def from_dict(cls, d: Dict) -> "PhaseSpec":
        """Create PhaseSpec from dictionary."""
        return cls(
            d.get("phases", ""),
            d.get("has_neutral", False),
            d.get("earth_bond", False)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "phases": self.phases,
            "has_neutral": self.has_neutral,
            "earth_bond": self.earth_bond,
        }

    def index_map_to(self, other: "PhaseSpec") -> Tuple[np.ndarray, np.ndarray]:
        """Return (src_idx, dst_idx) for reordering arrays between phase specs."""
        src_pos = {ph: i for i, ph in enumerate(self.phases)}
        pairs = [(src_pos[ph], j) for j, ph in enumerate(other.phases) if ph in src_pos]
        if not pairs:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)

        src_idx, dst_idx = zip(*pairs)
        return np.asarray(src_idx, np.int32), np.asarray(dst_idx, np.int32)

    def align_with(
            self,
            other: "PhaseSpec",
            arr: Iterable[float],
            fill: float = 0.0,
            dtype=np.float32
    ) -> np.ndarray:
        """Align a per-phase array from self to other phase ordering."""
        a = np.asarray(arr, dtype=dtype).ravel()
        if a.size != self.nph:
            raise ValueError(
                f"align_array: expected shape ({self.nph},), got {a.shape}"
            )

        out = np.full(other.nph, fill, dtype)
        si, di = self.index_map_to(other)
        if si.size:
            out[di] = a[si]

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(phases={self.phases!r}, has_neutral={self.has_neutral}, earth_bond={self.earth_bond})"
        )


def check_phase_model_consistency(
    model: PhaseModel,
    spec: PhaseSpec,
) -> None:
    """Validate that a PhaseSpec is consistent with the model."""
    if model is None or spec is None:
        raise ValueError("phase_model and phase_spec cannot be None")

    if isinstance(model, str):
        model = PhaseModel(model)

    # Balanced must be 1φ
    if model == PhaseModel.BALANCED_1PH:
        if spec.nph:
            raise ValueError(
                f"BALANCED_1PH requires nph = 0, got '{spec.nph}'"
            )
        return

    # THREE_PHASE: allow A, AB, ABC (1–3 conductors out of ABC)
    s = "".join([p for p in "ABC" if p in spec.phases.upper()])
    if not s:
        raise ValueError("THREE_PHASE requires at least one of A/B/C")

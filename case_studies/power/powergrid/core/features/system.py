"""System-level feature providers for power grid.

This module provides features that represent system-wide state for
power grid coordination at the SystemAgent level.

Features:
    SystemFrequency: System frequency and deviation from nominal
    AggregateGeneration: Total generation across all areas/grids
    AggregateLoad: Total load across all areas/grids
    InterAreaFlows: Power flows between interconnected areas
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from heron.core.feature import Feature


@dataclass(slots=True)
class SystemFrequency(Feature):
    """System frequency feature for power grid.

    Tracks system frequency and deviation from nominal.
    Important for frequency regulation and stability monitoring.

    Attributes:
        frequency_hz: Current system frequency in Hz
        nominal_hz: Nominal frequency (typically 50 or 60 Hz)

    Example:
        >>> freq = SystemFrequency(frequency_hz=59.95, nominal_hz=60.0)
        >>> freq.vector()
        array([59.95, -0.05], dtype=float32)
    """

    visibility: List[str] = field(default_factory=lambda: ["system", "upper_level"])

    frequency_hz: float = 60.0
    nominal_hz: float = 60.0

    def vector(self) -> np.ndarray:
        """Get frequency vector [frequency, deviation]."""
        deviation = self.frequency_hz - self.nominal_hz
        return np.array([self.frequency_hz, deviation], dtype=np.float32)

    def names(self) -> List[str]:
        """Get feature names."""
        return ["frequency_hz", "frequency_deviation"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "frequency_hz": self.frequency_hz,
            "nominal_hz": self.nominal_hz,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemFrequency":
        """Deserialize from dict."""
        return cls(
            frequency_hz=d.get("frequency_hz", 60.0),
            nominal_hz=d.get("nominal_hz", 60.0),
        )

    def set_values(self, **kwargs) -> None:
        """Update feature values."""
        if "frequency_hz" in kwargs:
            self.frequency_hz = kwargs["frequency_hz"]
        if "nominal_hz" in kwargs:
            self.nominal_hz = kwargs["nominal_hz"]

    def reset(self, **overrides) -> "SystemFrequency":
        """Reset to nominal frequency."""
        self.frequency_hz = overrides.get("frequency_hz", self.nominal_hz)
        return self


@dataclass(slots=True)
class AggregateGeneration(Feature):
    """Aggregate generation across all areas/grids.

    Tracks total generation for system-wide balancing.

    Attributes:
        total_mw: Total active power generation in MW
        total_mvar: Total reactive power generation in MVAr
        reserve_mw: Available spinning reserve in MW

    Example:
        >>> gen = AggregateGeneration(total_mw=1500.0, total_mvar=300.0)
        >>> gen.vector()
        array([1500., 300., 0.], dtype=float32)
    """

    visibility: List[str] = field(default_factory=lambda: ["system"])

    total_mw: float = 0.0
    total_mvar: float = 0.0
    reserve_mw: float = 0.0

    def vector(self) -> np.ndarray:
        """Get generation vector [P, Q, reserve]."""
        return np.array(
            [self.total_mw, self.total_mvar, self.reserve_mw],
            dtype=np.float32
        )

    def names(self) -> List[str]:
        """Get feature names."""
        return ["total_generation_mw", "total_generation_mvar", "reserve_mw"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "total_mw": self.total_mw,
            "total_mvar": self.total_mvar,
            "reserve_mw": self.reserve_mw,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AggregateGeneration":
        """Deserialize from dict."""
        return cls(
            total_mw=d.get("total_mw", 0.0),
            total_mvar=d.get("total_mvar", 0.0),
            reserve_mw=d.get("reserve_mw", 0.0),
        )

    def set_values(self, **kwargs) -> None:
        """Update feature values."""
        if "total_mw" in kwargs:
            self.total_mw = kwargs["total_mw"]
        if "total_mvar" in kwargs:
            self.total_mvar = kwargs["total_mvar"]
        if "reserve_mw" in kwargs:
            self.reserve_mw = kwargs["reserve_mw"]

    def reset(self, **overrides) -> "AggregateGeneration":
        """Reset to zero generation."""
        self.total_mw = overrides.get("total_mw", 0.0)
        self.total_mvar = overrides.get("total_mvar", 0.0)
        self.reserve_mw = overrides.get("reserve_mw", 0.0)
        return self


@dataclass(slots=True)
class AggregateLoad(Feature):
    """Aggregate load across all areas/grids.

    Tracks total load for system-wide balancing.

    Attributes:
        total_mw: Total active power load in MW
        total_mvar: Total reactive power load in MVAr
        controllable_mw: Controllable/flexible load in MW

    Example:
        >>> load = AggregateLoad(total_mw=1400.0, total_mvar=280.0)
        >>> load.vector()
        array([1400., 280., 0.], dtype=float32)
    """

    visibility: List[str] = field(default_factory=lambda: ["system"])

    total_mw: float = 0.0
    total_mvar: float = 0.0
    controllable_mw: float = 0.0

    def vector(self) -> np.ndarray:
        """Get load vector [P, Q, controllable]."""
        return np.array(
            [self.total_mw, self.total_mvar, self.controllable_mw],
            dtype=np.float32
        )

    def names(self) -> List[str]:
        """Get feature names."""
        return ["total_load_mw", "total_load_mvar", "controllable_load_mw"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "total_mw": self.total_mw,
            "total_mvar": self.total_mvar,
            "controllable_mw": self.controllable_mw,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AggregateLoad":
        """Deserialize from dict."""
        return cls(
            total_mw=d.get("total_mw", 0.0),
            total_mvar=d.get("total_mvar", 0.0),
            controllable_mw=d.get("controllable_mw", 0.0),
        )

    def set_values(self, **kwargs) -> None:
        """Update feature values."""
        if "total_mw" in kwargs:
            self.total_mw = kwargs["total_mw"]
        if "total_mvar" in kwargs:
            self.total_mvar = kwargs["total_mvar"]
        if "controllable_mw" in kwargs:
            self.controllable_mw = kwargs["controllable_mw"]

    def reset(self, **overrides) -> "AggregateLoad":
        """Reset to zero load."""
        self.total_mw = overrides.get("total_mw", 0.0)
        self.total_mvar = overrides.get("total_mvar", 0.0)
        self.controllable_mw = overrides.get("controllable_mw", 0.0)
        return self


@dataclass(slots=True)
class InterAreaFlows(Feature):
    """Power flows between interconnected areas.

    Tracks tie-line flows for inter-area coordination.

    Attributes:
        flows_mw: Active power flows on tie lines [MW]
        limits_mw: Flow limits on tie lines [MW]
        area_names: Names of interconnected areas

    Example:
        >>> flows = InterAreaFlows(
        ...     flows_mw=np.array([100.0, -50.0]),
        ...     limits_mw=np.array([200.0, 200.0]),
        ...     area_names=["area_1", "area_2"]
        ... )
        >>> flows.vector()
        array([100., -50., 200., 200.], dtype=float32)
    """

    visibility: List[str] = field(default_factory=lambda: ["system"])

    flows_mw: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    limits_mw: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    area_names: List[str] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        """Get flow vector [flows, limits]."""
        flows = np.asarray(self.flows_mw, dtype=np.float32)
        limits = np.asarray(self.limits_mw, dtype=np.float32)
        if len(flows) == 0 and len(limits) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate([flows, limits])

    def names(self) -> List[str]:
        """Get feature names."""
        flow_names = [f"flow_{name}_mw" for name in self.area_names]
        limit_names = [f"limit_{name}_mw" for name in self.area_names]
        return flow_names + limit_names

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "flows_mw": self.flows_mw.tolist() if isinstance(self.flows_mw, np.ndarray) else self.flows_mw,
            "limits_mw": self.limits_mw.tolist() if isinstance(self.limits_mw, np.ndarray) else self.limits_mw,
            "area_names": self.area_names,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterAreaFlows":
        """Deserialize from dict."""
        return cls(
            flows_mw=np.array(d.get("flows_mw", []), dtype=np.float32),
            limits_mw=np.array(d.get("limits_mw", []), dtype=np.float32),
            area_names=d.get("area_names", []),
        )

    def set_values(self, **kwargs) -> None:
        """Update feature values."""
        if "flows_mw" in kwargs:
            self.flows_mw = np.asarray(kwargs["flows_mw"], dtype=np.float32)
        if "limits_mw" in kwargs:
            self.limits_mw = np.asarray(kwargs["limits_mw"], dtype=np.float32)
        if "area_names" in kwargs:
            self.area_names = kwargs["area_names"]

    def reset(self, **overrides) -> "InterAreaFlows":
        """Reset to zero flows."""
        if "flows_mw" in overrides:
            self.flows_mw = np.asarray(overrides["flows_mw"], dtype=np.float32)
        else:
            self.flows_mw = np.zeros_like(self.flows_mw)
        return self


@dataclass(slots=True)
class SystemImbalance(Feature):
    """System power imbalance (generation - load - losses).

    Tracks real-time power balance for system stability.

    Attributes:
        imbalance_mw: Net power imbalance (positive = excess generation)
        ace_mw: Area Control Error in MW
        frequency_error_hz: Frequency deviation from nominal

    Example:
        >>> imb = SystemImbalance(imbalance_mw=10.0, ace_mw=5.0)
        >>> imb.vector()
        array([10., 5., 0.], dtype=float32)
    """

    visibility: List[str] = field(default_factory=lambda: ["system", "upper_level"])

    imbalance_mw: float = 0.0
    ace_mw: float = 0.0
    frequency_error_hz: float = 0.0

    def vector(self) -> np.ndarray:
        """Get imbalance vector [imbalance, ACE, freq_error]."""
        return np.array(
            [self.imbalance_mw, self.ace_mw, self.frequency_error_hz],
            dtype=np.float32
        )

    def names(self) -> List[str]:
        """Get feature names."""
        return ["imbalance_mw", "ace_mw", "frequency_error_hz"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "imbalance_mw": self.imbalance_mw,
            "ace_mw": self.ace_mw,
            "frequency_error_hz": self.frequency_error_hz,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemImbalance":
        """Deserialize from dict."""
        return cls(
            imbalance_mw=d.get("imbalance_mw", 0.0),
            ace_mw=d.get("ace_mw", 0.0),
            frequency_error_hz=d.get("frequency_error_hz", 0.0),
        )

    def set_values(self, **kwargs) -> None:
        """Update feature values."""
        if "imbalance_mw" in kwargs:
            self.imbalance_mw = kwargs["imbalance_mw"]
        if "ace_mw" in kwargs:
            self.ace_mw = kwargs["ace_mw"]
        if "frequency_error_hz" in kwargs:
            self.frequency_error_hz = kwargs["frequency_error_hz"]

    def reset(self, **overrides) -> "SystemImbalance":
        """Reset to zero imbalance."""
        self.imbalance_mw = overrides.get("imbalance_mw", 0.0)
        self.ace_mw = overrides.get("ace_mw", 0.0)
        self.frequency_error_hz = overrides.get("frequency_error_hz", 0.0)
        return self

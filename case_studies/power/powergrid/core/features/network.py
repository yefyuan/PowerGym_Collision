"""Grid-level network feature providers for GridState.

These features capture system-wide observables like bus voltages,
line flows, and aggregate network metrics.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from heron.core.feature import Feature
from heron.utils.array_utils import as_f32, cat_f32


@dataclass(slots=True)
class BusVoltages(Feature):
    """Bus voltage magnitudes and angles across the network.

    Captures voltage state at all buses, which is essential for
    monitoring system stability and voltage regulation.
    """
    vm_pu: Optional[np.ndarray] = None  # Voltage magnitudes (p.u.)
    va_deg: Optional[np.ndarray] = None  # Voltage angles (degrees)
    bus_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.vm_pu is not None:
            self.vm_pu = as_f32(self.vm_pu).ravel()
        if self.va_deg is not None:
            self.va_deg = as_f32(self.va_deg).ravel()

        # Validate consistency
        sizes = []
        if self.vm_pu is not None:
            sizes.append(self.vm_pu.size)
        if self.va_deg is not None:
            sizes.append(self.va_deg.size)
        if self.bus_names:
            sizes.append(len(self.bus_names))

        if sizes and len(set(sizes)) > 1:
            raise ValueError(f"Inconsistent bus voltage array sizes: {sizes}")

    def vector(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self.vm_pu is not None:
            parts.append(self.vm_pu)
        if self.va_deg is not None:
            parts.append(self.va_deg)
        return cat_f32(parts)

    def names(self) -> List[str]:
        out: List[str] = []
        n_buses = len(self.bus_names) if self.bus_names else (
            self.vm_pu.size if self.vm_pu is not None else
            self.va_deg.size if self.va_deg is not None else 0
        )

        if self.vm_pu is not None:
            if self.bus_names:
                out += [f"vm_pu_{name}" for name in self.bus_names]
            else:
                out += [f"vm_pu_{i}" for i in range(n_buses)]

        if self.va_deg is not None:
            if self.bus_names:
                out += [f"va_deg_{name}" for name in self.bus_names]
            else:
                out += [f"va_deg_{i}" for i in range(n_buses)]

        return out

    def clamp_(self) -> None:
        if self.vm_pu is not None:
            self.vm_pu = np.clip(self.vm_pu, 0.0, 2.0)

    def to_dict(self) -> Dict:
        return {
            "vm_pu": self.vm_pu.tolist() if self.vm_pu is not None else None,
            "va_deg": self.va_deg.tolist() if self.va_deg is not None else None,
            "bus_names": self.bus_names,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BusVoltages":
        return cls(
            vm_pu=as_f32(d["vm_pu"]) if d.get("vm_pu") is not None else None,
            va_deg=as_f32(d["va_deg"]) if d.get("va_deg") is not None else None,
            bus_names=d.get("bus_names", []),
        )

    def set_values(self, **kwargs) -> None:
        """Update bus voltage fields.

        Args:
            **kwargs: Field names and values to update
        """
        allowed_keys = {"vm_pu", "va_deg", "bus_names"}

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"BusVoltages.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__post_init__()
        self.clamp_()


@dataclass(slots=True)
class LineFlows(Feature):
    """Power flows and loading on transmission lines.

    Monitors line utilization and power transfer, critical for
    identifying congestion and ensuring thermal limits.
    """
    p_from_mw: Optional[np.ndarray] = None  # Active power from "from" bus (MW)
    q_from_mvar: Optional[np.ndarray] = None  # Reactive power from "from" bus (MVAr)
    loading_percent: Optional[np.ndarray] = None  # Line loading (%)
    line_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.p_from_mw is not None:
            self.p_from_mw = as_f32(self.p_from_mw).ravel()
        if self.q_from_mvar is not None:
            self.q_from_mvar = as_f32(self.q_from_mvar).ravel()
        if self.loading_percent is not None:
            self.loading_percent = as_f32(self.loading_percent).ravel()

        # Validate consistency
        sizes = []
        if self.p_from_mw is not None:
            sizes.append(self.p_from_mw.size)
        if self.q_from_mvar is not None:
            sizes.append(self.q_from_mvar.size)
        if self.loading_percent is not None:
            sizes.append(self.loading_percent.size)
        if self.line_names:
            sizes.append(len(self.line_names))

        if sizes and len(set(sizes)) > 1:
            raise ValueError(f"Inconsistent line flow array sizes: {sizes}")

    def vector(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self.p_from_mw is not None:
            parts.append(self.p_from_mw)
        if self.q_from_mvar is not None:
            parts.append(self.q_from_mvar)
        if self.loading_percent is not None:
            parts.append(self.loading_percent)
        return cat_f32(parts)

    def names(self) -> List[str]:
        out: List[str] = []
        n_lines = len(self.line_names) if self.line_names else (
            self.p_from_mw.size if self.p_from_mw is not None else
            self.q_from_mvar.size if self.q_from_mvar is not None else
            self.loading_percent.size if self.loading_percent is not None else 0
        )

        if self.p_from_mw is not None:
            if self.line_names:
                out += [f"p_from_mw_{name}" for name in self.line_names]
            else:
                out += [f"p_from_mw_{i}" for i in range(n_lines)]

        if self.q_from_mvar is not None:
            if self.line_names:
                out += [f"q_from_mvar_{name}" for name in self.line_names]
            else:
                out += [f"q_from_mvar_{i}" for i in range(n_lines)]

        if self.loading_percent is not None:
            if self.line_names:
                out += [f"loading_percent_{name}" for name in self.line_names]
            else:
                out += [f"loading_percent_{i}" for i in range(n_lines)]

        return out

    def clamp_(self) -> None:
        if self.loading_percent is not None:
            self.loading_percent = np.clip(self.loading_percent, 0.0, 200.0)

    def to_dict(self) -> Dict:
        return {
            "p_from_mw": self.p_from_mw.tolist() if self.p_from_mw is not None else None,
            "q_from_mvar": self.q_from_mvar.tolist() if self.q_from_mvar is not None else None,
            "loading_percent": self.loading_percent.tolist() if self.loading_percent is not None else None,
            "line_names": self.line_names,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LineFlows":
        return cls(
            p_from_mw=as_f32(d["p_from_mw"]) if d.get("p_from_mw") is not None else None,
            q_from_mvar=as_f32(d["q_from_mvar"]) if d.get("q_from_mvar") is not None else None,
            loading_percent=as_f32(d["loading_percent"]) if d.get("loading_percent") is not None else None,
            line_names=d.get("line_names", []),
        )

    def set_values(self, **kwargs) -> None:
        """Update line flow fields.

        Args:
            **kwargs: Field names and values to update
        """
        allowed_keys = {"p_from_mw", "q_from_mvar", "loading_percent", "line_names"}

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"LineFlows.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__post_init__()
        self.clamp_()


@dataclass(slots=True)
class NetworkMetrics(Feature):
    """Aggregate network-level metrics.

    Provides system-wide statistics like total generation, load,
    and losses for high-level monitoring and control.
    """
    total_gen_mw: float = 0.0
    total_load_mw: float = 0.0
    total_loss_mw: float = 0.0
    total_gen_mvar: float = 0.0
    total_load_mvar: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([
            self.total_gen_mw,
            self.total_load_mw,
            self.total_loss_mw,
            self.total_gen_mvar,
            self.total_load_mvar,
        ], dtype=np.float32)

    def names(self) -> List[str]:
        return [
            "total_gen_mw",
            "total_load_mw",
            "total_loss_mw",
            "total_gen_mvar",
            "total_load_mvar",
        ]

    def clamp_(self) -> None:
        self.total_gen_mw = max(0.0, self.total_gen_mw)
        self.total_load_mw = max(0.0, self.total_load_mw)
        self.total_loss_mw = max(0.0, self.total_loss_mw)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "NetworkMetrics":
        return cls(**d)

    def set_values(self, **kwargs) -> None:
        """Update network metrics fields.

        Args:
            **kwargs: Field names and values to update
        """
        allowed_keys = {
            "total_gen_mw",
            "total_load_mw",
            "total_loss_mw",
            "total_gen_mvar",
            "total_load_mvar",
        }

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"NetworkMetrics.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.clamp_()

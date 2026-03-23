from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency
from heron.core.feature import Feature
from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class ElectricalBasePh(Feature):
    """
    Phase-aware electrical fundamentals at a connection point.

    BALANCED_1PH:
      - Scalars only: P_MW, Q_MVAr, Vm_pu, Va_rad.

    THREE_PHASE:
      - Per-phase arrays only: *_ph with shape (3,) in spec order.
      - Neutral telemetry allowed only if spec.has_neutral is True.
    """
    visibility: List[str] = field(default_factory=list)

    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # Balanced scalars
    P_MW: Optional[float] = None
    Q_MVAr: Optional[float] = None
    Vm_pu: Optional[float] = None
    Va_rad: Optional[float] = None

    # Three-phase arrays
    P_MW_ph: Optional[np.ndarray] = None
    Q_MVAr_ph: Optional[np.ndarray] = None
    Vm_pu_ph: Optional[np.ndarray] = None
    Va_rad_ph: Optional[np.ndarray] = None

    # Neutral telemetry (needs has_neutral=True)
    I_neutral_A: Optional[float] = None
    Vn_earth_V: Optional[float] = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def __post_init__(self):
        check_phase_model_consistency(self.phase_model, self.phase_spec)
        self._validate_inputs()
        self._ensure_shapes_()

    # ------------------------------------------------------------
    # Validation / shapes
    # ------------------------------------------------------------

    def _validate_inputs(self) -> None:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            # 1) Forbid any per-phase arrays
            bad = []
            for name in ("P_MW_ph", "Q_MVAr_ph", "Vm_pu_ph", "Va_rad_ph"):
                if getattr(self, name) is not None:
                    bad.append(name)
            if bad:
                raise ValueError(
                    "Balanced model forbids per-phase fields: " + ", ".join(bad)
                )

            # 2) Neutral telemetry not meaningful in balanced
            if self.I_neutral_A is not None or self.Vn_earth_V is not None:
                raise ValueError("Neutral telemetry not allowed in BALANCED_1PH.")

            # 3) Require at least one scalar present
            if all(
                getattr(self, k) is None
                for k in ("P_MW", "Q_MVAr", "Vm_pu", "Va_rad")
            ):
                raise ValueError(
                    "BALANCED_1PH requires at least one of "
                    "P_MW, Q_MVAr, Vm_pu, or Va_rad."
                )
            return

        # THREE_PHASE
        bad = [
            k for k, v in {
                "P_MW": self.P_MW,
                "Q_MVAr": self.Q_MVAr,
                "Vm_pu": self.Vm_pu,
                "Va_rad": self.Va_rad,
            }.items() if v is not None
        ]
        if bad:
            raise ValueError("THREE_PHASE forbids scalar fields: " + ", ".join(bad))

        if all(x is None for x in
            (self.P_MW_ph, self.Q_MVAr_ph, self.Vm_pu_ph, self.Va_rad_ph)):
            raise ValueError(
                "THREE_PHASE requires at least one per-phase array: "
                "P_MW_ph, Q_MVAr_ph, Vm_pu_ph, or Va_rad_ph."
            )

        # Neutral only if spec.has_neutral
        if self.phase_spec is not None and not self.phase_spec.has_neutral:
            if self.I_neutral_A is not None or self.Vn_earth_V is not None:
                raise ValueError(
                    "Neutral telemetry requires has_neutral=True in PhaseSpec."
                )

    def _ensure_shapes_(self) -> None:
        """Normalize internal shapes for balanced vs three-phase models.

        BALANCED_1PH:
            - Scalars (P_MW, Q_MVAr, Vm_pu, Va_rad) must be scalar-like or None.

        THREE_PHASE:
            - Per-phase arrays (*_ph) must be 1D with length == phase_spec.nph().
        """
        # BALANCED_1PH
        if self.phase_model == PhaseModel.BALANCED_1PH:
            for name in ("P_MW", "Q_MVAr", "Vm_pu", "Va_rad"):
                val = getattr(self, name)
                if val is None:
                    continue
                # Accept anything scalar-like, but normalize to a Python float
                arr = np.asarray(val, dtype=np.float32)
                if arr.size != 1:
                    raise ValueError(
                        f"{name} must be scalar-like for BALANCED_1PH, "
                        f"got shape {arr.shape}"
                    )
                setattr(self, name, float(arr.item()))
            # Per-phase arrays are already forbidden in _validate_inputs
            return

        # THREE_PHASE
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.phase_spec is None:
                raise ValueError("THREE_PHASE requires a PhaseSpec for shape checks.")
            nph = self.phase_spec.nph

            def _check_ph_array(field_name: str) -> None:
                arr = getattr(self, field_name)
                if arr is None:
                    return
                a = np.asarray(arr, dtype=np.float32).ravel()
                if a.shape[0] != nph:
                    raise ValueError(
                        f"{field_name} must have shape ({nph},), got {a.shape}"
                    )
                setattr(self, field_name, a)

            for nm in ("P_MW_ph", "Q_MVAr_ph", "Vm_pu_ph", "Va_rad_ph"):
                _check_ph_array(nm)

            return

        # Unknown model
        raise ValueError(f"Unsupported phase model: {self.phase_model}")

    # ------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------

    def reset(self, **overrides: Any) -> "ElectricalBasePh":
        """
        Reset electrical telemetry to neutral values, optionally applying
        overrides afterward.

        Default behavior (no overrides):

          BALANCED_1PH:
            - P_MW, Q_MVAr, Va_rad -> 0.0 (if not None)
            - Vm_pu                -> 1.0 (if not None)

          THREE_PHASE:
            - P_MW_ph, Q_MVAr_ph, Va_rad_ph -> zeros_like (if not None)
            - Vm_pu_ph                      -> ones_like  (if not None)
            - I_neutral_A, Vn_earth_V       -> 0.0        (if not None)

        Overrides:
            Any keyword arguments are merged on top of the neutral reset and
            forwarded to `set_values(**...)`, e.g.:

                electrical.reset(P_MW=5.0, Q_MVAr=1.0)
                electrical.reset(P_MW_ph=[1.0, 1.0, 1.0])
        """
        updates: Dict[str, Any] = {}

        # Neutral reset, only for fields that are currently active
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.P_MW is not None:
                updates["P_MW"] = 0.0
            if self.Q_MVAr is not None:
                updates["Q_MVAr"] = 0.0
            if self.Va_rad is not None:
                updates["Va_rad"] = 0.0
            if self.Vm_pu is not None:
                updates["Vm_pu"] = 1.0

        elif self.phase_model == PhaseModel.THREE_PHASE:
            for name in ("P_MW_ph", "Q_MVAr_ph", "Va_rad_ph", "Vm_pu_ph"):
                arr = getattr(self, name)
                if arr is None:
                    continue

                if name == "Vm_pu_ph":
                    updates[name] = np.ones_like(arr, dtype=np.float32)
                else:
                    updates[name] = np.zeros_like(arr, dtype=np.float32)

            if self.I_neutral_A is not None:
                updates["I_neutral_A"] = 0.0
            if self.Vn_earth_V is not None:
                updates["Vn_earth_V"] = 0.0

        else:
            raise ValueError(f"Unsupported phase model in reset(): {self.phase_model}")

        # User overrides on top of neutral reset
        if overrides:
            updates.update(overrides)

        if updates:
            self.set_values(**updates)

        return self

    def set_values(self, **kwargs: Any) -> None:
        """
        Update one or more electrical fields and re-validate.

        Example:
            electrical.set_values(P_MW=5.0, Q_MVAr=1.0)

            electrical.set_values(
                P_MW_ph=[1.0, 1.0, 1.0],
                Vm_pu_ph=[1.0, 0.99, 1.01],
            )

        This will:
            - assign the given attributes
            - check phase_model/phase_spec consistency
            - run _validate_inputs() and _ensure_shapes_()
        """
        # Allowed public fields that describe the electrical state
        allowed_keys = {
            "phase_model",
            "phase_spec",
            "P_MW",
            "Q_MVAr",
            "Vm_pu",
            "Va_rad",
            "P_MW_ph",
            "Q_MVAr_ph",
            "Vm_pu_ph",
            "Va_rad_ph",
            "I_neutral_A",
            "Vn_earth_V",
        }

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"ElectricalBasePh.set_values got unknown fields: {sorted(unknown)}"
            )

        # Assign raw values
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Re-check consistency and normalize
        check_phase_model_consistency(self.phase_model, self.phase_spec)
        self._validate_inputs()
        self._ensure_shapes_()

    def vector(self) -> np.ndarray:
        parts = []
        # BALANCED_1PH
        if self.phase_model == PhaseModel.BALANCED_1PH:
            for v in (self.P_MW, self.Q_MVAr, self.Vm_pu, self.Va_rad):
                if v is not None:
                    parts.append(np.array([v], np.float32))
            return cat_f32(parts)

        # THREE_PHASE
        if self.phase_model == PhaseModel.THREE_PHASE:
            for arr in (self.P_MW_ph, self.Q_MVAr_ph, self.Vm_pu_ph, self.Va_rad_ph):
                if arr is not None:
                    parts.append(arr.ravel())
            return cat_f32(parts)

        raise ValueError(f"Unsupported phase model: {self.phase_model}")

    def names(self) -> List[str]:
        out = []
        # BALANCED_1PH
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.P_MW is not None:
                out.append("P_MW")
            if self.Q_MVAr is not None:
                out.append("Q_MVAr")
            if self.Vm_pu is not None:
                out.append("Vm_pu")
            if self.Va_rad is not None:
                out.append("Va_rad")
            return out

        # THREE_PHASE
        if self.phase_model == PhaseModel.THREE_PHASE:
            phases = self.phase_spec.phases
            def per(prefix: str) -> List[str]:
                return [f"{prefix}_{ph}" for ph in phases]
            if self.P_MW_ph is not None:
                out += per("P_MW")
            if self.Q_MVAr_ph is not None:
                out += per("Q_MVAr")
            if self.Vm_pu_ph is not None:
                out += per("Vm_pu")
            if self.Va_rad_ph is not None:
                out += per("Va_rad")
            return out

        raise ValueError(f"Unsupported phase model: {self.phase_model}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElectricalBasePh":
        """
        Construct ElectricalBasePh from a serialized dict.

        Expected keys (all optional except at least one electrical value):
          - phase_model: str | PhaseModel
          - phase_spec: dict | PhaseSpec
          - P_MW, Q_MVAr, Vm_pu, Va_rad
          - P_MW_ph, Q_MVAr_ph, Vm_pu_ph, Va_rad_ph
          - I_neutral_A, Vn_earth_V
        """
        flat = cls._flatten_dict(d)
        return cls(**flat)

    def to_dict(self) -> Dict[str, Any]:
        """Simple explicit serialization; avoids surprises from asdict()."""
        ps_dict = {
            "phases": self.phase_spec.phases,
            "has_neutral": self.phase_spec.has_neutral,
            "earth_bond": self.phase_spec.earth_bond,
        }

        return {
            "phase_model": self.phase_model.value,
            "phase_spec": ps_dict,
            "P_MW": self.P_MW,
            "Q_MVAr": self.Q_MVAr,
            "Vm_pu": self.Vm_pu,
            "Va_rad": self.Va_rad,
            "P_MW_ph": self.P_MW_ph,
            "Q_MVAr_ph": self.Q_MVAr_ph,
            "Vm_pu_ph": self.Vm_pu_ph,
            "Va_rad_ph": self.Va_rad_ph,
            "I_neutral_A": self.I_neutral_A,
            "Vn_earth_V": self.Vn_earth_V,
        }

    # ------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------

    @staticmethod
    def _flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a raw dict into ctor kwargs:
          - phase_model: string → PhaseModel
          - phase_spec: dict → PhaseSpec
          - leave numerical fields as-is
        """
        out: Dict[str, Any] = dict(d)

        # phase_model
        pm = out.get("phase_model", PhaseModel.BALANCED_1PH)
        if not isinstance(pm, PhaseModel):
            out["phase_model"] = PhaseModel(pm)

        # phase_spec
        psd = out.get("phase_spec", None)
        if psd is None or isinstance(psd, PhaseSpec):
            # leave as is
            pass
        else:
            out["phase_spec"] = PhaseSpec(
                phases=psd.get("phases", ""),
                has_neutral=psd.get("has_neutral", False),
                earth_bond=psd.get("earth_bond", False),
            )

        return out

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            return (
                f"ElectricalBasePh(phase_model=BALANCED_1PH, phases='', "
                f"visibility={self.visibility}, "
                f"P_MW={self.P_MW}, Q_MVAr={self.Q_MVAr}, "
                f"Vm_pu={self.Vm_pu}, Va_rad={self.Va_rad})"
            )

        return (
            "ElectricalBasePh("
            f"phase_model=THREE_PHASE, phases={self.phase_spec.phases}, "
            f"visibility={self.visibility}, "
            f"P_MW_ph={self.P_MW_ph}, "
            f"Q_MVAr_ph={self.Q_MVAr_ph}, "
            f"Vm_pu_ph={self.Vm_pu_ph}, "
            f"Va_rad_ph={self.Va_rad_ph}, "
            f"I_neutral_A={self.I_neutral_A}, Vn_earth_V={self.Vn_earth_V})"
        )

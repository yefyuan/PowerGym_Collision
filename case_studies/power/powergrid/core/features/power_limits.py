from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.core.feature import Feature
from powergrid.utils.phase import PhaseModel, PhaseSpec
from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class PowerLimits(Feature):
    """
    Generator capability / constraints.

    Fields
    ------
    s_rated_MVA:
        Rated apparent power (nameplate).
    derate_frac:
        Optional fraction in [0, 1] that uniformly derates capability
        (e.g., due to environment, weather, or thermal limits).

    p_min_MW, p_max_MW:
        Optional static active-power bounds.
    q_min_MVAr, q_max_MVAr:
        Optional static reactive-power bounds.

    pf_min_abs:
        Optional minimum absolute power factor |pf| in (0, 1].
        Applies symmetrically to both leading and lagging operation.

    Notes
    -----
    The feasible set is the intersection of:
        * P range:   p_min_MW <= P <= p_max_MW        (if provided)
        * Q range:   q_min_MVAr <= Q <= q_max_MVAr    (if provided)
        * S circle:  P^2 + Q^2 <= S_avail^2          (if s_rated_MVA provided)
        * PF wedge:  |Q| <= |P| * tan(phi_min)       (if pf_min_abs provided)
    """
    visibility: List[str] = field(default_factory=list)

    # Context
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)

    # Apparent-power (S) capability
    s_rated_MVA: Optional[float] = None
    derate_frac: float = 1.0

    # Static P/Q bounds
    p_min_MW: Optional[float] = None
    p_max_MW: Optional[float] = None
    q_min_MVAr: Optional[float] = None
    q_max_MVAr: Optional[float] = None

    # Symmetric power-factor constraint
    pf_min_abs: Optional[float] = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()
        self.clip_()

    # ------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------

    def _validate(self) -> None:
        # Sanity on power factor
        if self.pf_min_abs is not None:
            pf = float(self.pf_min_abs)
            if not (0.0 < pf <= 1.0):
                raise ValueError("pf_min_abs must be in (0, 1].")

    def _S_avail(self) -> Optional[float]:
        """
        Effective apparent-power capability after applying derate_frac.
        """
        if self.s_rated_MVA is None:
            return None
        s = float(self.s_rated_MVA) * float(np.clip(self.derate_frac, 0.0, 1.0))
        return max(0.0, s)

    def _tan_phi(self) -> Optional[float]:
        """
        Return tan(phi_min) from pf_min_abs, or None if no PF constraint.

        pf_min_abs = cos(phi_min)  =>  tan(phi_min) = sqrt(1/pf^2 - 1).
        """
        if self.pf_min_abs is None:
            return None
        pf = float(self.pf_min_abs)
        if not (0.0 < pf <= 1.0):
            raise ValueError("pf_min_abs must be in (0, 1].")
        return np.sqrt(max(1.0 / (pf * pf) - 1.0, 0.0))

    # ------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------

    def reset(self, *, derate_frac: float = 1.0) -> None:
        """
        Reset dynamic derating while preserving capability configuration.

        By default:
            - derate_frac is set back to 1.0 (no derating).

        You can override the reset target via the keyword argument:

            limits.reset(derate_frac=0.8)

        All other fields (s_rated_MVA, P/Q bounds, pf_min_abs) are left unchanged.
        """
        self.derate_frac = float(derate_frac)
        # Keep everything consistent with normal invariants
        self._validate()
        self.clip_()

    def set_values(self, **kwargs: Any) -> None:
        """
        Update one or more limit fields and re-validate/clip.

        Example:
            limits.set_values(p_min_MW=10.0, p_max_MW=50.0)
            limits.set_values(derate_frac=0.8)
        """
        allowed_keys = {
            "visibility",
            "s_rated_MVA",
            "derate_frac",
            "p_min_MW",
            "p_max_MW",
            "q_min_MVAr",
            "q_max_MVAr",
            "pf_min_abs",
        }

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"PowerLimits.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._validate()
        self.clip_()

    def vector(self) -> np.ndarray:
        """
        Return a flat numeric representation of the limits.
        """
        parts: List[np.ndarray] = []

        def add(x) -> None:
            if isinstance(x, float) or isinstance(x, int):
                parts.append(np.array([x], dtype=np.float32))
            elif isinstance(x, np.ndarray):
                parts.append(x.astype(np.float32))

        # S capability
        add(self.s_rated_MVA)

        # derate_frac is always present (clamped to [0, 1])
        add(np.clip(self.derate_frac, 0.0, 1.0))

        # static bounds
        add(self.p_min_MW)
        add(self.p_max_MW)
        add(self.q_min_MVAr)
        add(self.q_max_MVAr)

        # PF constraint
        add(self.pf_min_abs)

        return cat_f32(parts)

    def names(self) -> List[str]:
        """
        Return feature names aligned with vector().
        """
        out: List[str] = []

        if self.s_rated_MVA is not None:
            out.append("s_rated_MVA")

        # derate_frac is always exported
        out.append("derate_frac")

        if self.p_min_MW is not None:
            out.append("p_min_MW")
        if self.p_max_MW is not None:
            out.append("p_max_MW")
        if self.q_min_MVAr is not None:
            out.append("q_min_MVAr")
        if self.q_max_MVAr is not None:
            out.append("q_max_MVAr")
        if self.pf_min_abs is not None:
            out.append("pf_min_abs")

        return out

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    def clip_(self) -> None:
        """
        Normalize fields into a consistent, safe range:

          - derate_frac is clamped to [0, 1].
          - P/Q bounds are swapped if min > max.
        """
        # Normalize derate in [0, 1]
        self.derate_frac = float(np.clip(self.derate_frac, 0.0, 1.0))

        # Fix swapped P bounds if provided
        if (
            self.p_min_MW is not None
            and self.p_max_MW is not None
            and self.p_min_MW > self.p_max_MW
        ):
            self.p_min_MW, self.p_max_MW = self.p_max_MW, self.p_min_MW

        # Fix swapped Q bounds if provided
        if (
            self.q_min_MVAr is not None
            and self.q_max_MVAr is not None
            and self.q_min_MVAr > self.q_max_MVAr
        ):
            self.q_min_MVAr, self.q_max_MVAr = self.q_max_MVAr, self.q_min_MVAr

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Explicit serialization; avoids surprises from dataclasses.asdict()."""
        return {
            "s_rated_MVA": self.s_rated_MVA,
            "derate_frac": self.derate_frac,
            "p_min_MW": self.p_min_MW,
            "p_max_MW": self.p_max_MW,
            "q_min_MVAr": self.q_min_MVAr,
            "q_max_MVAr": self.q_max_MVAr,
            "pf_min_abs": self.pf_min_abs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PowerLimits":
        """
        Construct PowerLimits from a serialized dict.

        Expected keys (all optional):
          - s_rated_MVA, derate_frac
          - p_min_MW, p_max_MW
          - q_min_MVAr, q_max_MVAr
          - pf_min_abs
        """
        return cls(
            s_rated_MVA=d.get("s_rated_MVA"),
            derate_frac=d.get("derate_frac", 1.0),
            p_min_MW=d.get("p_min_MW"),
            p_max_MW=d.get("p_max_MW"),
            q_min_MVAr=d.get("q_min_MVAr"),
            q_max_MVAr=d.get("q_max_MVAr"),
            pf_min_abs=d.get("pf_min_abs"),
        )

    # ------------------------------------------------------------
    # Constraint logic
    # ------------------------------------------------------------

    def effective_q_bounds(self, P_MW: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute effective (q_min, q_max) at a given P by intersecting:

          - Static q_min/q_max (if set)
          - S circle: P^2 + Q^2 <= S_avail^2
          - PF wedge: |Q| <= |P| * tan(phi_min)

        Returns (qmin_eff, qmax_eff). Either may be None if unconstrained.
        """
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        # Static Q bounds
        if self.q_min_MVAr is not None:
            lower_bounds.append(float(self.q_min_MVAr))
        if self.q_max_MVAr is not None:
            upper_bounds.append(float(self.q_max_MVAr))

        # S circle
        S = self._S_avail()
        if S is not None:
            rad2 = max(S * S - float(P_MW) * float(P_MW), 0.0)
            q_cap = np.sqrt(rad2)
            lower_bounds.append(-q_cap)
            upper_bounds.append(+q_cap)

        # PF wedge
        tphi = self._tan_phi()
        if tphi is not None:
            bound = abs(float(P_MW)) * tphi
            lower_bounds.append(-bound)
            upper_bounds.append(+bound)

        # Intersection = tightest lower/upper
        qmin_eff = max(lower_bounds) if lower_bounds else None
        qmax_eff = min(upper_bounds) if upper_bounds else None

        # If intersection is empty numerically, collapse to (0, 0)
        if (qmin_eff is not None and qmax_eff is not None) and qmin_eff > qmax_eff:
            qmin_eff, qmax_eff = 0.0, 0.0

        return qmin_eff, qmax_eff

    def feasible(self, P_MW: float, Q_MVAr: float) -> Dict[str, float]:
        """
        Return violation magnitudes (>= 0). Zero means feasible.

        Keys:
            - 'p_violation'
            - 'q_violation'
            - 's_excess'
            - 'pf_violation'
        """
        P = P_MW if P_MW is not None else 0.0
        Q = Q_MVAr if Q_MVAr is not None else 0.0

        # P violations
        p_violation = 0.0
        if self.p_min_MW is not None:
            p_violation += max(0.0, self.p_min_MW - P)
        if self.p_max_MW is not None:
            p_violation += max(0.0, P - self.p_max_MW)

        # Q violations
        q_violation = 0.0
        if self.q_min_MVAr is not None:
            q_violation += max(0.0, self.q_min_MVAr - Q)
        if self.q_max_MVAr is not None:
            q_violation += max(0.0, Q - self.q_max_MVAr)

        # Apparent power excess
        s_excess = 0.0
        S = self._S_avail()
        if S is not None:
            s_excess = max(0.0, np.hypot(P, Q) - S)

        # PF violation
        pf_violation = 0.0
        if self.pf_min_abs is not None and (abs(P) > 1e-9 or abs(Q) > 1e-9):
            pf = abs(P) / max(np.hypot(P, Q), 1e-9)
            pf_violation = max(0.0, float(self.pf_min_abs) - pf)

        return {
            "p_violation": p_violation,
            "q_violation": q_violation,
            "s_excess": s_excess,
            "pf_violation": pf_violation,
        }

    def project_pq(
            self,
            P_MW: Optional[float] = None,
            Q_MVAr: Optional[float] = None,
        ) -> Tuple[Optional[float], Optional[float]]:
        """Project (P, Q) into the feasible set using a simple, deterministic strategy.

        1) Clip P to [p_min_MW, p_max_MW] if given.
        2) Compute effective Q bounds at this P (S circle + PF wedge + static Q bounds).
        3) Clip Q to [qmin_eff, qmax_eff] if those bounds are present.

        Args:
            P_MW: Active power in MW (None to skip P clipping, returns None)
            Q_MVAr: Reactive power in MVAr (None to skip Q clipping, returns None)

        Returns:
            Tuple of (clipped_P_MW, clipped_Q_MVAr). Returns None for any input that was None.
        """
        P = P_MW if P_MW is not None else 0.0
        Q = Q_MVAr if Q_MVAr is not None else 0.0

        # 1) Clip P
        if self.p_min_MW is not None:
            P = max(P, self.p_min_MW)
        if self.p_max_MW is not None:
            P = min(P, self.p_max_MW)

        # 2) Effective Q bounds at this P
        qmin_eff, qmax_eff = self.effective_q_bounds(P)

        # 3) Clip Q into feasible range
        if qmin_eff is not None:
            Q = max(Q, qmin_eff)
        if qmax_eff is not None:
            Q = min(Q, qmax_eff)

        P_MW = P if P_MW is not None else None
        Q_MVAr = Q if Q_MVAr is not None else None

        return P_MW, Q_MVAr

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "PowerLimits("
            f"visibility={self.visibility}, "
            f"s_rated_MVA={self.s_rated_MVA}, "
            f"derate_frac={self.derate_frac}, "
            f"p_min_MW={self.p_min_MW}, "
            f"p_max_MW={self.p_max_MW}, "
            f"q_min_MVAr={self.q_min_MVAr}, "
            f"q_max_MVAr={self.q_max_MVAr}, "
            f"pf_min_abs={self.pf_min_abs})"
        )
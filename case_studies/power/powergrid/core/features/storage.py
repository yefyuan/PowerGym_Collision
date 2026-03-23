from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from heron.core.feature import Feature
from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class StorageBlock(Feature):
    """
    Energy storage state, static limits, and simple degradation accounting.

    Core SOC & power fields
    -----------------------
    soc:
        State of charge in [0, 1] (fraction of nominal capacity).

    soc_min, soc_max:
        Optional SOC bounds in [0, 1]. Typically soc_min >= 0, soc_max <= 1.

    e_capacity_MWh:
        Nameplate energy capacity in MWh.

    p_ch_max_MW, p_dsc_max_MW:
        Maximum charging and discharging power (MW), both >= 0.

        Convention (at device level):
          - P > 0: charging  (import from grid)
          - P < 0: discharging (export to grid)
        p_dsc_max_MW is the *magnitude* of allowed discharge power.

    ch_eff, dsc_eff:
        Charging and discharging efficiencies in (0, 1].

    Degradation / cycling fields
    ----------------------------
    e_throughput_MWh:
        Cumulative absolute energy throughput (sum of |ΔE|), in MWh.

    equiv_full_cycles:
        Equivalent full cycles, defined as:
            e_throughput_MWh / e_capacity_MWh
        if e_capacity_MWh is known, else left as provided.

    degr_cost_per_MWh:
        Optional degradation cost coefficient per MWh throughput.

    degr_cost_per_cycle:
        Optional degradation cost coefficient per equivalent full cycle.

    degr_cost_cum:
        Cumulative degradation cost from all past throughput updates.

    Usage for degradation (typical pattern)
    ---------------------------------------
    On each step, if you know the net *absolute* energy moved through the battery:

        delta_e_MWh = abs(P_MW) * dt_h
        step_degr_cost = storage_block.accumulate_throughput(delta_e_MWh)

    This will:
        - increment e_throughput_MWh by |delta_e_MWh|
        - recompute equiv_full_cycles (if capacity is known)
        - compute the incremental degradation cost for this step
        - add it into degr_cost_cum
        - return the incremental cost so you can add it into env.cost
    """

    # Visibility: owner can see their own SOC for reward computation
    visibility: List[str] = field(default_factory=lambda: ["owner"])

    # SOC state and bounds (fractions)
    soc: Optional[float] = None
    soc_min: Optional[float] = None
    soc_max: Optional[float] = None

    # Energy capacity
    e_capacity_MWh: Optional[float] = None

    # Power limits (non-negative magnitudes)
    p_ch_max_MW: Optional[float] = None
    p_dsc_max_MW: Optional[float] = None

    # Efficiencies
    ch_eff: float = 1.0
    dsc_eff: float = 1.0

    # Degradation / cycling
    e_throughput_MWh: float = 0.0         # cumulative |ΔE|
    degr_cost_per_MWh: float = 0.0        # cost per MWh throughput
    degr_cost_per_cycle: float = 0.0      # cost per equivalent full cycle
    degr_cost_cum: float = 0.0            # cumulative degradation cost

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate_inputs()
        self.clip_()

    # ------------------------------------------------------------
    # Validation / clamp
    # ------------------------------------------------------------

    def _validate_inputs(self) -> None:
        # SOC bounds
        for v, nm in (
            (self.soc_min, "soc_min"),
            (self.soc_max, "soc_max"),
            (self.soc, "soc"),
        ):
            if v is None:
                raise ValueError(f"{nm} is None.")

            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{nm} must be in [0, 1].")

        # soc_min <= soc_max if both set
        if self.soc_min > self.soc_max:
            raise ValueError("soc_min cannot be greater than soc_max.")

        # Capacity non-negative
        if self.e_capacity_MWh is None:
            raise ValueError("e_capacity_MWh is None.")

        if self.e_capacity_MWh <= 0.0:
            raise ValueError("e_capacity_MWh must be > 0.")

        # Power limits non-negative
        for v, nm in (
            (self.p_ch_max_MW, "p_ch_max_MW"),
            (self.p_dsc_max_MW, "p_dsc_max_MW"),
        ):
            if v is None:
                raise ValueError(f"{nm} is None.")

            if v < 0.0:
                raise ValueError(f"{nm} must be >= 0.")

        # Efficiencies in (0, 1]
        for v, nm in ((self.ch_eff, "ch_eff"), (self.dsc_eff, "dsc_eff")):
            if not (0.0 < v <= 1.0):
                raise ValueError(f"{nm} must be in (0, 1].")

        # Degradation / cycling fields: non-negative
        for v, nm in (
            (self.e_throughput_MWh, "e_throughput_MWh"),
            (self.degr_cost_per_MWh, "degr_cost_per_MWh"),
            (self.degr_cost_per_cycle, "degr_cost_per_cycle"),
            (self.degr_cost_cum, "degr_cost_cum"),
        ):
            if v < 0.0:
                raise ValueError(f"{nm} must be >= 0.")

    def clip_(self) -> None:
        """
        Clamp numeric fields into valid ranges:
          - soc, soc_min, soc_max in [0, 1]
          - ch_eff, dsc_eff in (0, 1] (soft-clamped to [eps,1])
          - degradation-related scalars >= 0
        """
        # Clamp SOC-related fields
        self.soc_min = np.clip(self.soc_min, 0.0, 1.0)
        self.soc_max = np.clip(self.soc_max, 0.0, 1.0)

        lo = self.soc_min if self.soc_min is not None else 0.0
        hi = self.soc_max if self.soc_max is not None else 1.0
        self.soc = np.clip(self.soc, lo, hi)

        # Efficiencies: clamp softly into (0, 1]
        eps = 1e-6
        self.ch_eff = np.clip(self.ch_eff, eps, 1.0)
        self.dsc_eff = np.clip(self.dsc_eff, eps, 1.0)

        # Degradation-related non-negativity
        self.e_throughput_MWh = max(0.0, self.e_throughput_MWh)
        self.equiv_full_cycles = self.e_throughput_MWh / self.e_capacity_MWh
        self.equiv_full_cycles = max(0.0, self.equiv_full_cycles)
        self.degr_cost_per_MWh = max(0.0, self.degr_cost_per_MWh)
        self.degr_cost_per_cycle = max(0.0, self.degr_cost_per_cycle)
        self.degr_cost_cum = max(0.0, self.degr_cost_cum)

    # ------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------

    def reset(
        self,
        *,
        soc: Optional[float] = None,
        random_init: bool = False,
        seed: Optional[int] = None,
        reset_degradation: bool = True,
    ) -> "StorageBlock":
        """
        Reset SOC and optionally degradation accounting.

        Parameters
        ----------
        soc : float | None
            If provided, set SOC directly (then clamped to [soc_min, soc_max]).

        random_init : bool
            If True and `soc` is None, sample SOC uniformly in:
                [soc_min, soc_max] if both are set,
                [0.0, 1.0] otherwise.

        seed : int | None
            Optional seed for random SOC initialization.

        reset_degradation : bool
            If True, reset throughput, cycle count, and cumulative degradation cost.

        Notes
        -----
        If soc is None and random_init is False, SOC is left unchanged (but will
        still be clamped into valid bounds by the usual validation/clipping).
        """
        updates: Dict[str, Any] = {}

        # --- SOC handling ---
        if soc is not None:
            # Explicit override
            updates["soc"] = soc

        elif random_init:
            # Random SOC from current bounds (or [0,1] if not set)
            rng = np.random.default_rng(seed)
            updates["soc"] = rng.uniform(self.soc_min , self.soc_max)

        # else: leave soc as-is; clip_() will keep it in range

        # --- Degradation reset ---
        if reset_degradation:
            updates["e_throughput_MWh"] = 0.0
            updates["equiv_full_cycles"] = 0.0
            updates["degr_cost_cum"] = 0.0

        if updates:
            self.set_values(**updates)

        return self

    def set_values(self, **kwargs: Any) -> None:
        """
        Update one or more storage fields and re-validate.

        Example:
            storage.set_values(soc=0.6)
            storage.set_values(
                soc_min=0.1,
                soc_max=0.9,
                p_ch_max_MW=5.0,
                degr_cost_per_MWh=2.0,
            )
        """
        allowed_keys = {
            "visibility",
            "soc",
            "soc_min",
            "soc_max",
            "e_capacity_MWh",
            "p_ch_max_MW",
            "p_dsc_max_MW",
            "ch_eff",
            "dsc_eff",
            "e_throughput_MWh",
            "equiv_full_cycles",
            "degr_cost_per_MWh",
            "degr_cost_per_cycle",
            "degr_cost_cum",
        }

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"StorageBlock.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._validate_inputs()
        self.clip_()

    # ------------------------------------------------------------
    # Degradation / cycling helper
    # ------------------------------------------------------------

    def accumulate_throughput(self, delta_e_MWh: float) -> float:
        """
        Record additional energy throughput and update cycles & degradation cost.

        Args:
            delta_e_MWh:
                Net energy moved this step (MWh). The *magnitude* matters
                for degradation, so abs(delta_e_MWh) is used internally.

        Returns:
            step_degr_cost: float
                Incremental degradation cost contributed by this step.
        """
        delta_e = abs(delta_e_MWh)

        prev_throughput = self.e_throughput_MWh
        prev_cycles = self.equiv_full_cycles

        # Update throughput
        self.e_throughput_MWh = prev_throughput + delta_e

        # Recompute equivalent full cycles if capacity is known
        if self.e_capacity_MWh and self.e_capacity_MWh > 0.0:
            self.equiv_full_cycles = self.e_throughput_MWh / self.e_capacity_MWh

        # Compute incremental cost
        #   cost = c_MWh * ΔE + c_cycle * Δcycles
        delta_cycles = max(0.0, self.equiv_full_cycles - prev_cycles)
        step_cost = (
            self.degr_cost_per_MWh * delta_e
            + self.degr_cost_per_cycle * delta_cycles
        )

        # Accumulate
        self.degr_cost_cum += step_cost
        self.clip_()  # keep everything non-negative / sane

        return step_cost

    def soc_violation(self) -> float:
        """
        Compute SOC limit violations.

        Returns: total violation magnitude = soc_below + soc_above

        All values are in SOC fraction units (0–1).
        """
        below = max(0.0, self.soc_min - self.soc)
        above = max(0.0, self.soc - self.soc_max)

        return below + above

    # ------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------

    def vector(self) -> np.ndarray:
        """
        Flatten to a numeric feature vector.

        Order (when present):
            [soc, soc_min, soc_max,
             e_capacity_MWh,
             p_ch_max_MW, p_dsc_max_MW,
             ch_eff, dsc_eff,
             e_throughput_MWh, equiv_full_cycles,
             degr_cost_per_MWh, degr_cost_per_cycle,
             degr_cost_cum]
        """
        parts: List[np.ndarray] = []

        def add(x: Optional[float]) -> None:
            if x is not None:
                parts.append(np.array([x], dtype=np.float32))

        add(self.soc)
        # add(self.soc_min)
        # add(self.soc_max)
        # add(self.e_capacity_MWh)
        # add(self.p_ch_max_MW)
        # add(self.p_dsc_max_MW)
        # add(self.ch_eff)
        # add(self.dsc_eff)
        add(self.e_throughput_MWh)
        add(self.equiv_full_cycles)
        # add(self.degr_cost_per_MWh)
        # add(self.degr_cost_per_cycle)
        # add(self.degr_cost_cum)

        return cat_f32(parts)

    def names(self) -> List[str]:
        """
        Return feature names aligned with vector().
        """
        out: List[str] = []

        if self.soc is not None:
            out.append("soc")
        # if self.soc_min is not None:
        #     out.append("soc_min")
        # if self.soc_max is not None:
        #     out.append("soc_max")
        # if self.e_capacity_MWh is not None:
        #     out.append("e_capacity_MWh")
        # if self.p_ch_max_MW is not None:
        #     out.append("p_ch_max_MW")
        # if self.p_dsc_max_MW is not None:
        #     out.append("p_dsc_max_MW")
        # if self.ch_eff is not None:
        #     out.append("ch_eff")
        # if self.dsc_eff is not None:
        #     out.append("dsc_eff")
        if self.e_throughput_MWh is not None:
            out.append("e_throughput_MWh")
        if self.equiv_full_cycles is not None:
            out.append("equiv_full_cycles")
        # if self.degr_cost_per_MWh is not None:
        #     out.append("degr_cost_per_MWh")
        # if self.degr_cost_per_cycle is not None:
        #     out.append("degr_cost_per_cycle")
        # if self.degr_cost_cum is not None:
        #     out.append("degr_cost_cum")

        return out

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Explicit serialization; avoids surprises from dataclasses.asdict()."""
        return {
            "soc": self.soc,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "e_capacity_MWh": self.e_capacity_MWh,
            "p_ch_max_MW": self.p_ch_max_MW,
            "p_dsc_max_MW": self.p_dsc_max_MW,
            "ch_eff": self.ch_eff,
            "dsc_eff": self.dsc_eff,
            "e_throughput_MWh": self.e_throughput_MWh,
            "equiv_full_cycles": self.equiv_full_cycles,
            "degr_cost_per_MWh": self.degr_cost_per_MWh,
            "degr_cost_per_cycle": self.degr_cost_per_cycle,
            "degr_cost_cum": self.degr_cost_cum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StorageBlock":
        """
        Construct StorageBlock from a serialized dict.

        Expected keys (all optional):
          - soc, soc_min, soc_max
          - e_capacity_MWh
          - p_ch_max_MW, p_dsc_max_MW
          - ch_eff, dsc_eff
          - e_throughput_MWh
          - degr_cost_per_MWh, degr_cost_per_cycle, degr_cost_cum
        """
        return cls(
            soc=d.get("soc"),
            soc_min=d.get("soc_min"),
            soc_max=d.get("soc_max"),
            e_capacity_MWh=d.get("e_capacity_MWh"),
            p_ch_max_MW=d.get("p_ch_max_MW"),
            p_dsc_max_MW=d.get("p_dsc_max_MW"),
            ch_eff=d.get("ch_eff", 1.0),
            dsc_eff=d.get("dsc_eff", 1.0),
            e_throughput_MWh=d.get("e_throughput_MWh", 0.0),
            degr_cost_per_MWh=d.get("degr_cost_per_MWh", 0.0),
            degr_cost_per_cycle=d.get("degr_cost_per_cycle", 0.0),
            degr_cost_cum=d.get("degr_cost_cum", 0.0),
        )

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "StorageBlock("
            f"visibility={self.visibility}, "
            f"soc={self.soc}, "
            f"soc_min={self.soc_min}, soc_max={self.soc_max}, "
            f"e_capacity_MWh={self.e_capacity_MWh}, "
            f"p_ch_max_MW={self.p_ch_max_MW}, "
            f"p_dsc_max_MW={self.p_dsc_max_MW}, "
            f"ch_eff={self.ch_eff}, dsc_eff={self.dsc_eff}, "
            f"e_throughput_MWh={self.e_throughput_MWh}, "
            f"degr_cost_per_MWh={self.degr_cost_per_MWh}, "
            f"degr_cost_per_cycle={self.degr_cost_per_cycle}, "
            f"degr_cost_cum={self.degr_cost_cum})"
        )

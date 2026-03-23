from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from heron.core.feature import Feature
from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class StatusBlock(Feature):
    """
    Status / lifecycle block with optional categorical state.

    Fields
    ------
    in_service, out_service:
        Optional boolean flags for service status.

    state:
        Optional categorical state token (string), drawn from `states_vocab`.
    states_vocab:
        Optional ordered list of allowed state tokens.

    t_in_state_s, t_to_next_s:
        Optional timing telemetry (seconds). Must be non-negative.

    progress_frac:
        Optional normalized progress indicator in [0, 1].

    emit_state_one_hot, emit_state_index:
        Control how `state` is exported into the feature vector:
        - one-hot over `states_vocab`
        - integer index of `state` within `states_vocab`

    lock_schema:
        If True, the set of exported dimensions is fixed after construction,
        even if some values become None later.
    """

    # Visibility tags, e.g. ["public", "owner"]
    visibility: List[str] = field(default_factory=list)

    # Service flags
    in_service: Optional[bool] = None
    out_service: Optional[bool] = None

    # Categorical state
    state: Optional[str] = None
    states_vocab: Optional[List[str]] = None

    # Timing / progress
    t_in_state_s: Optional[float] = None
    t_to_next_s: Optional[float] = None
    progress_frac: Optional[float] = None  # [0..1]

    # Export configuration
    emit_state_one_hot: bool = True
    emit_state_index: bool = False

    # Schema behavior
    lock_schema: bool = True

    # Internal export flags (decided once when schema is locked)
    _export_in_service: bool = field(default=False, init=False, repr=False)
    _export_out_service: bool = field(default=False, init=False, repr=False)
    _export_state_oh: bool = field(default=False, init=False, repr=False)
    _export_state_idx: bool = field(default=False, init=False, repr=False)
    _export_t_in: bool = field(default=False, init=False, repr=False)
    _export_t_to: bool = field(default=False, init=False, repr=False)
    _export_prog: bool = field(default=False, init=False, repr=False)

    # Cached vocabulary index
    _vocab_index: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    # Track if schema has been primed (to prevent expansion after lock)
    _schema_primed: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate_()
        self._prime_schema_()
        self.clip_()

    def _validate_(self) -> None:
        # Service flags consistency
        if self.in_service is True and self.out_service is True:
            raise ValueError(
                "StatusBlock: `in_service` and `out_service` cannot both be True."
            )

        if self.in_service is False and self.out_service is False:
            raise ValueError(
                "StatusBlock: `in_service` and `out_service` cannot both be False."
            )

        # Vocabulary checks
        if self.states_vocab is not None:
            if not isinstance(self.states_vocab, list) or not self.states_vocab:
                raise ValueError("states_vocab must be a non-empty list of strings.")
            if len(set(self.states_vocab)) != len(self.states_vocab):
                raise ValueError("states_vocab contains duplicates.")
            if not all(isinstance(s, str) and s for s in self.states_vocab):
                raise ValueError("states_vocab must contain non-empty strings.")

        # State must be in vocab if both are provided
        if self.state is not None and self.states_vocab is not None:
            if self.state not in self.states_vocab:
                raise ValueError(
                    f"state '{self.state}' not in states_vocab {self.states_vocab}."
                )

        # Timing non-negativity
        for v, nm in (
            (self.t_in_state_s, "t_in_state_s"),
            (self.t_to_next_s, "t_to_next_s"),
        ):
            if v is not None and float(v) < 0.0:
                raise ValueError(f"{nm} must be >= 0.")

        # Progress range
        if self.progress_frac is not None:
            p = float(self.progress_frac)
            if not (0.0 <= p <= 1.0):
                raise ValueError("progress_frac must be in [0, 1].")

        # If state is present, require at least one state representation
        if self.state is not None and self.states_vocab is not None:
            if not (self.emit_state_one_hot or self.emit_state_index):
                raise ValueError(
                    "state is present; enable emit_state_one_hot or emit_state_index."
                )

        # Disallow exporting both representations at once
        if self.emit_state_one_hot and self.emit_state_index:
            raise ValueError(
                "StatusBlock: `emit_state_one_hot` and `emit_state_index` "
                "cannot both be True."
            )

        # Cache vocab index for O(1) lookup later
        if self.states_vocab:
            self._vocab_index = {s: i for i, s in enumerate(self.states_vocab)}

    # ------------------------------------------------------------
    # Schema priming
    # ------------------------------------------------------------

    def _prime_schema_(self) -> None:
        """
        Decide once which slots to export to keep shapes/names stable
        when lock_schema=True.
        """
        if not self.lock_schema:
            return  # dynamic schema allowed (legacy behavior)

        # If schema already primed with lock, don't expand it
        if self._schema_primed:
            return

        # Service flags
        self._export_in_service = (
            self.in_service is not None or self._export_in_service
        )
        self._export_out_service = (
            self.out_service is not None or self._export_out_service
        )

        # State representations
        idx_present = self.state is not None and self.states_vocab is not None
        self._export_state_oh = (
            (self.emit_state_one_hot and idx_present) or self._export_state_oh
        )
        self._export_state_idx = (
            (self.emit_state_index and idx_present) or self._export_state_idx
        )

        # Timing / progress
        self._export_t_in = self.t_in_state_s is not None or self._export_t_in
        self._export_t_to = self.t_to_next_s is not None or self._export_t_to
        self._export_prog = self.progress_frac is not None or self._export_prog

        # Mark schema as primed
        self._schema_primed = True

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _state_index(self) -> Optional[int]:
        if self.state is None or self.states_vocab is None:
            return None
        # O(1) lookup; validation ensures key exists
        return self._vocab_index[self.state]

    @staticmethod
    def one_hot(idx: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        if 0 <= idx < n:
            out[idx] = 1.0
        return out

    def _should_export(self, value: Optional[float], flag: bool) -> bool:
        """
        Decide whether to export a scalar field, given current schema rules.
        """
        if self.lock_schema:
            return flag
        return value is not None

    # ------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------

    def reset(
        self,
        *,
        state: Optional[str] = None,
        in_service: Optional[bool] = None,
        out_service: Optional[bool] = None,
        t_in_state_s: float = 0.0,
        t_to_next_s: Optional[float] = None,
        progress_frac: float = 0.0,
    ) -> "StatusBlock":
        """
        Reset dynamic status telemetry, optionally updating core status.

        By default (no arguments):
            - keeps `state`, `in_service`, `out_service`, `t_to_next_s` as-is
            - sets t_in_state_s = 0.0
            - sets progress_frac = 0.0

        Optional overrides:
            - state:        new state token (must be in states_vocab if vocab is set)
            - in_service:   new in_service flag
            - out_service:  new out_service flag
            - t_in_state_s: new time-in-state (defaults to 0.0)
            - t_to_next_s:  new time-to-next (if provided)
            - progress_frac:new progress value (defaults to 0.0)
        """
        updates: Dict[str, Any] = {}

        # Optional status overrides
        if state is not None:
            updates["state"] = state
        if in_service is not None:
            updates["in_service"] = in_service
        if out_service is not None:
            updates["out_service"] = out_service

        # Telemetry resets / overrides
        updates["t_in_state_s"] = t_in_state_s
        if t_to_next_s is not None:
            updates["t_to_next_s"] = t_to_next_s
        updates["progress_frac"] = progress_frac

        # Delegate to the central mutation path (validate + schema + clamp)
        return self.set_values(**updates)

    def set_values(self, **kwargs: Any) -> None:
        """
        Update one or more status fields and re-validate.

        Example:
            status.set_values(state="online", t_in_state_s=0.0)
            status.set_values(in_service=True, out_service=None)

        This will:
            - assign the given attributes
            - re-run validation
            - re-prime schema (if lock_schema=True)
            - clamp numeric fields
        """
        allowed_keys = {
            "visibility",
            "in_service",
            "out_service",
            "state",
            "states_vocab",
            "t_in_state_s",
            "t_to_next_s",
            "progress_frac",
            "emit_state_one_hot",
            "emit_state_index",
            "lock_schema",
        }

        unknown = set(kwargs.keys()) - allowed_keys
        if unknown:
            raise AttributeError(
                f"StatusBlock.set_values got unknown fields: {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Rebuild internal invariants
        self._validate_()
        self._prime_schema_()
        self.clip_()

    def vector(self) -> np.ndarray:
        """
        Return a flat numeric representation of the status block.
        """
        parts: List[np.ndarray] = []

        def add(x) -> None:
            if isinstance(x, float) or isinstance(x, int):
                parts.append(np.array([x], dtype=np.float32))
            elif isinstance(x, np.ndarray):
                parts.append(x.astype(np.float32))

        # Service flags
        if (
            (self.lock_schema and self._export_in_service)
            or (not self.lock_schema and self.in_service is not None)
        ):
            add(1.0 if self.in_service else 0.0)

        if (
            (self.lock_schema and self._export_out_service)
            or (not self.lock_schema and self.out_service is not None)
        ):
            add(1.0 if self.out_service else 0.0)

        # Categorical state
        idx = self._state_index()
        if idx is not None:
            if (
                (self.lock_schema and self._export_state_oh)
                or (not self.lock_schema and self.emit_state_one_hot)
            ):
                add(self.one_hot(idx, len(self.states_vocab)))

            if (
                (self.lock_schema and self._export_state_idx)
                or (not self.lock_schema and self.emit_state_index)
            ):
                add(idx)
        else:
            # If schema says we must export state but current state is missing,
            # emit zeros to keep shape stable.
            if (
                self.lock_schema
                and (self._export_state_oh or self._export_state_idx)
                and self.states_vocab
            ):
                if self._export_state_oh:
                    add(np.zeros(len(self.states_vocab), np.float32))
                if self._export_state_idx:
                    add(np.zeros(1, np.float32))

        # Timing / progress
        if self._should_export(self.t_in_state_s, self._export_t_in):
            add(0.0 if self.t_in_state_s is None else self.t_in_state_s)

        if self._should_export(self.t_to_next_s, self._export_t_to):
            add(0.0 if self.t_to_next_s is None else self.t_to_next_s)

        if self._should_export(self.progress_frac, self._export_prog):
            add(0.0 if self.progress_frac is None else self.progress_frac)

        return cat_f32(parts)

    def names(self) -> List[str]:
        """
        Return names aligned with vector().
        """
        out: List[str] = []

        # Service flags
        if (
            (self.lock_schema and self._export_in_service)
            or (not self.lock_schema and self.in_service is not None)
        ):
            out.append("in_service")

        if (
            (self.lock_schema and self._export_out_service)
            or (not self.lock_schema and self.out_service is not None)
        ):
            out.append("out_service")

        # State names
        state_idx_present = self._state_index() is not None
        if (
            (self.lock_schema and (self._export_state_oh or self._export_state_idx))
            or (not self.lock_schema and state_idx_present)
        ):
            if (
                (self.lock_schema and self._export_state_oh)
                or (not self.lock_schema and self.emit_state_one_hot)
            ):
                out += [f"state_{tok}" for tok in (self.states_vocab or [])]

            if (
                (self.lock_schema and self._export_state_idx)
                or (not self.lock_schema and self.emit_state_index)
            ):
                out.append("state_idx")

        # Timing / progress
        if (
            (self.lock_schema and self._export_t_in)
            or (not self.lock_schema and self.t_in_state_s is not None)
        ):
            out.append("t_in_state_s")

        if (
            (self.lock_schema and self._export_t_to)
            or (not self.lock_schema and self.t_to_next_s is not None)
        ):
            out.append("t_to_next_s")

        if (
            (self.lock_schema and self._export_prog)
            or (not self.lock_schema and self.progress_frac is not None)
        ):
            out.append("progress_frac")

        return out

    def clip_(self) -> None:
        """
        Clamp numeric fields into valid ranges:
          - t_in_state_s, t_to_next_s >= 0
          - progress_frac in [0, 1]
        """
        if self.t_in_state_s is not None:
            self.t_in_state_s = max(0.0, float(self.t_in_state_s))

        if self.t_to_next_s is not None:
            self.t_to_next_s = max(0.0, float(self.t_to_next_s))

        if self.progress_frac is not None:
            self.progress_frac = float(
                np.clip(self.progress_frac, 0.0, 1.0)
            )

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Explicit serialization; avoids surprises from dataclasses.asdict()."""
        return {
            "in_service": self.in_service,
            "out_service": self.out_service,
            "state": self.state,
            "states_vocab": self.states_vocab,
            "t_in_state_s": self.t_in_state_s,
            "t_to_next_s": self.t_to_next_s,
            "progress_frac": self.progress_frac,
            "emit_state_one_hot": self.emit_state_one_hot,
            "emit_state_index": self.emit_state_index,
            "lock_schema": self.lock_schema,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StatusBlock":
        """
        Construct StatusBlock from a serialized dict.

        Expected keys (all optional):
          - in_service, out_service
          - state, states_vocab
          - t_in_state_s, t_to_next_s, progress_frac
          - emit_state_one_hot, emit_state_index, lock_schema
        """
        return cls(
            in_service=d.get("in_service"),
            out_service=d.get("out_service"),
            state=d.get("state"),
            states_vocab=d.get("states_vocab"),
            t_in_state_s=d.get("t_in_state_s"),
            t_to_next_s=d.get("t_to_next_s"),
            progress_frac=d.get("progress_frac"),
            emit_state_one_hot=d.get("emit_state_one_hot", True),
            emit_state_index=d.get("emit_state_index", False),
            lock_schema=d.get("lock_schema", True),
        )

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "StatusBlock("
            f"visibility={self.visibility}, "
            f"in_service={self.in_service}, "
            f"state={self.state}, "
            f"states_vocab={self.states_vocab}, "
            f"t_in_state_s={self.t_in_state_s}, "
            f"t_to_next_s={self.t_to_next_s}, "
            f"progress_frac={self.progress_frac}, "
            f"emit_state_one_hot={self.emit_state_one_hot}, "
            f"emit_state_index={self.emit_state_index}, "
            f"lock_schema={self.lock_schema})"
        )
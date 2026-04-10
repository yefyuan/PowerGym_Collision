"""Action abstraction for agent control.

This module provides a flexible action representation supporting both
continuous and discrete action spaces, as well as mixed actions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict

from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class Action:
    """Mixed continuous/discrete action representation.

    Supports continuous actions (e.g., power setpoints), discrete actions
    (e.g., on/off switches), or mixed continuous+discrete actions.

    Attributes:
        c: Continuous action values in physical units, shape (dim_c,)
        d: Discrete action indices, shape (dim_d,), each d[i] in {0..ncats_i-1}
        dim_c: Dimension of continuous action space
        dim_d: Number of discrete action heads
        range: Tuple of (lower_bounds, upper_bounds) for continuous actions
        ncats: Number of categories for each discrete head

    Example:
        Create a continuous action with bounds::

            import numpy as np
            from heron.core import Action

            action = Action()
            action.set_specs(
                dim_c=2,
                range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
            )
            action.sample()  # Random action within bounds
            print(action.c)  # e.g., [5.2, 0.3]

        Create a discrete action::

            action = Action()
            action.set_specs(dim_d=1, ncats=[5])  # 5 choices
            action.sample()
            print(action.d)  # e.g., [2]

        Create a mixed continuous+discrete action::

            action = Action()
            action.set_specs(
                dim_c=2,
                dim_d=1,
                ncats=[3],
                range=(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
            )

    Note:
        Use `scale()` / `unscale()` for continuous normalization to [-1, 1].
    """

    c: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    dim_c: int = 0
    dim_d: int = 0

    range: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ncats: Sequence[int] = field(default_factory=list)

    _space: Optional[gym.Space] = None

    def is_valid(self) -> bool:
        """Check whether this action has been configured with actual dimensions."""
        return self.dim_c > 0 or self.dim_d > 0

    @property
    def space(self) -> "gym.Space":
        """Lazily-constructed Gymnasium action space for this Action."""
        if self._space is None:
            self._space = self._to_gym_space()
        return self._space

    def _to_gym_space(self) -> "gym.Space":
        """Construct Gymnasium action space for this Action."""
        def _build_continuous_space() -> Box:
            lb, ub = self.range
            return Box(low=lb, high=ub, dtype=np.float32)

        def _build_discrete_space():
            if self.dim_d == 1:
                return Discrete(self.ncats[0])
            return MultiDiscrete(self.ncats)

        if self.dim_c and self.dim_d:
            return SpaceDict({
                "c": _build_continuous_space(),
                "d": _build_discrete_space(),
            })

        if self.dim_c:
            return _build_continuous_space()

        if self.dim_d:
            return _build_discrete_space()

        # Empty action space - return a single no-op action
        return Discrete(1)

    def _validate_and_prepare(self) -> None:
        """Validate action specs and initialize buffers."""
        # shape init
        if self.dim_c and self.c.size == 0:
            self.c = np.zeros(self.dim_c, dtype=np.float32)
        if self.dim_d and self.d.size == 0:
            self.d = np.zeros(self.dim_d, dtype=np.int32)
        if self.dim_d == 0:
            self.d = np.array([], dtype=np.int32)
        if self.dim_d == 0 and self.ncats != []:
            raise ValueError("ncats must be empty when dim_d == 0.")
        if self.dim_d > 0:
            if len(self.ncats) != self.dim_d:
                raise ValueError("len(ncats) must equal dim_d.")

        # range validation
        if self.range is not None:
            lb, ub = self.range
            if lb.shape != ub.shape:
                raise ValueError("range must be a tuple of (lb, ub) with identical shapes.")
            if lb.ndim != 1 or (self.dim_c and lb.shape[0] != self.dim_c):
                raise ValueError("range arrays must be 1D with length == dim_c.")
            if not np.all(lb <= ub):
                raise ValueError("range lower bounds must be <= upper bounds.")

    def set_specs(
        self,
        dim_c: int = 0,
        dim_d: int = 0,
        ncats: Optional[Sequence[int]] = None,
        range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Set action space specifications.

        Args:
            dim_c: Dimension of continuous action space
            dim_d: Number of discrete action heads
            ncats: Number of categories per discrete head
            range: Tuple of (lower, upper) bounds for continuous actions
        """
        self.dim_c, self.dim_d = int(dim_c), int(dim_d)
        if ncats is None:
            ncats = []
        if isinstance(ncats, int):
            ncats = [ncats] * self.dim_d
        self.ncats = list(ncats)
        self.c = np.zeros(self.dim_c, dtype=np.float32)
        self.d = np.array([], dtype=np.int32)
        if self.dim_d > 0:
            self.d = np.zeros(self.dim_d, dtype=np.int32)
        if range is None:
            low = np.full(self.dim_c, -np.inf, dtype=np.float32)
            high = np.full(self.dim_c, np.inf, dtype=np.float32)
            self.range = np.asarray([low, high], dtype=np.float32)
        else:
            self.range = np.asarray([
                    np.asarray(range[0], dtype=np.float32),
                    np.asarray(range[1], dtype=np.float32),
            ], dtype=np.float32)
        self._validate_and_prepare()
        self._space = None

    def sample(self, seed: Optional[int] = None) -> "Action":
        """Sample random action from the action space.

        Continuous part is sampled according to `range`.
        Discrete part is sampled from Discrete/MultiDiscrete space.

        Args:
            seed: Optional random seed

        Returns:
            self for chaining
        """
        if seed is not None:
            self.space.seed(seed)

        # Sample from Gym space
        action = self.space.sample()

        # Decode Gym action into (c, d)
        if isinstance(action, dict):
            self.c[...] = action["c"]
            self.d[...] = action["d"]
        else:
            if self.dim_c:
                self.c[...] = action
            if self.dim_d:
                self.d[...] = action

        return self

    def reset(
        self,
        action: Any = None,
        *,
        c: Optional[Sequence[float]] = None,
        d: Optional[Sequence[int]] = None,
    ) -> "Action":
        """Reset action to neutral value or user-provided values.

        Priority:
            1) If `action` is given, delegate to `set_values(action)`.
            2) Else if `c`/`d` are given, delegate to `set_values({"c": c, "d": d})`.
            3) Else:
               - continuous `c` is set to midpoint of [lb, ub] where finite, 0.0 otherwise
               - discrete `d` is set to 0 for all heads.
        """
        if action is not None:
            self.set_values(action)
            return self

        if c is not None or d is not None:
            payload: Dict[str, Any] = {}
            if c is not None:
                payload["c"] = c
            if d is not None:
                payload["d"] = d
            self.set_values(payload)
            return self

        # Neutral reset based on specs
        self._validate_and_prepare()

        # Continuous part: neutral based on range
        if self.dim_c:
            if self.range is not None:
                lb, ub = self.range
                neutral = np.zeros_like(lb, dtype=np.float32)
                finite = np.isfinite(lb) & np.isfinite(ub)

                if np.any(finite):
                    neutral[finite] = 0.5 * (lb[finite] + ub[finite])

                self.c[...] = neutral
            else:
                self.c[...] = 0.0

        # Discrete part: neutral = 0 in each head
        if self.dim_d:
            self.d[...] = 0

        return self.clip()

    def set_values(
        self,
        action: Any = None,
        *,
        c: Optional[Sequence[float]] = None,
        d: Optional[Sequence[int]] = None,
    ) -> "Action":
        """Set action from various formats.

        Supported formats:
            - Action object (copies c and d values)
            - dict with optional keys "c", "d"
            - scalar int for pure discrete (dim_c == 0, dim_d > 0)
            - 1D array-like [c..., d...] of length dim_c + dim_d
            - keyword args: c=..., d=...
        """
        # Case 0: Another Action object - copy its values
        if isinstance(action, Action):
            if self.dim_c and action.dim_c:
                self.c[...] = action.c[:self.dim_c]
            if self.dim_d and action.dim_d:
                self.d[...] = action.d[:self.dim_d]
            return self.clip()

        # Normalize kwargs into action object
        if action is None and (c is not None or d is not None):
            payload: Dict[str, Any] = {}
            if c is not None:
                payload["c"] = c
            if d is not None:
                payload["d"] = d
            action = payload

        if action is None:
            return self.clip()

        # Case 1: dict
        if isinstance(action, dict):
            if self.dim_c and "c" in action and action["c"] is not None:
                c_arr = np.asarray(action["c"], dtype=np.float32)
                if c_arr.size != self.dim_c:
                    raise ValueError(
                        f"continuous part length {c_arr.size} != dim_c {self.dim_c}"
                    )
                self.c[...] = c_arr

            if self.dim_d and "d" in action and action["d"] is not None:
                d_raw = action["d"]
                if np.isscalar(d_raw):
                    d_arr = np.array([int(d_raw)], dtype=np.int32)
                else:
                    d_arr = np.asarray(d_raw, dtype=np.int32)

                if d_arr.size != self.dim_d:
                    raise ValueError(
                        f"discrete part length {d_arr.size} != dim_d {self.dim_d}"
                    )
                self.d[...] = d_arr

            return self.clip()

        # Case 2: scalar for pure discrete
        if np.isscalar(action):
            if self.dim_d == 0:
                raise ValueError("set_values: scalar action only valid when dim_d > 0.")
            if self.dim_c != 0:
                raise ValueError(
                    "set_values: scalar action ambiguous when dim_c > 0; "
                    "use dict or d=... instead."
                )
            self.d[...] = int(action)
            return self.clip()

        # Case 3: flat vector [c..., d...]
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                "set_values expects a 1D array-like for non-dict, non-scalar inputs."
            )

        expected = self.dim_c + self.dim_d
        if arr.size != expected:
            raise ValueError(
                f"Action vector length {arr.size} != expected {expected} "
                f"(dim_c={self.dim_c}, dim_d={self.dim_d})"
            )

        if self.dim_c:
            self.c[...] = arr[: self.dim_c]

        if self.dim_d:
            self.d[...] = arr[self.dim_c :].astype(np.int32)

        return self.clip()

    def clip(self) -> "Action":
        """Clip `c` to `range` and `d_i` to [0, ncats_i-1] in-place."""
        if self.dim_c:
            lb, ub = self.range
            np.clip(self.c, lb, ub, out=self.c)

        if self.dim_d:
            for i, K in enumerate(self.ncats):
                if self.d[i] < 0:
                    self.d[i] = 0
                elif self.d[i] >= K:
                    self.d[i] = K - 1
        return self

    def scale(self) -> np.ndarray:
        """Return normalized [-1, 1] copy of `c`. Zero-span axes -> 0."""
        if self.range is None or self.c.size == 0:
            return self.c.astype(np.float32, copy=True)
        lb, ub = self.range
        span = ub - lb
        x = np.zeros_like(self.c, dtype=np.float32)
        mask = span > 0
        if np.any(mask):
            x[mask] = 2.0 * (self.c[mask] - lb[mask]) / span[mask] - 1.0
        return x

    def unscale(self, x: Sequence[float]) -> np.ndarray:
        """Set `c` from normalized [-1, 1] vector (physical units)."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != self.dim_c:
            raise ValueError("normalized vector length must equal dim_c")
        if self.range is None:
            self.c = x.copy()
            return self.c
        lb, ub = self.range
        span = ub - lb
        self.c = np.empty_like(lb, dtype=np.float32)
        mask = span > 0
        if np.any(mask):
            self.c[mask] = lb[mask] + 0.5 * (x[mask] + 1.0) * span[mask]
        if np.any(~mask):
            self.c[~mask] = lb[~mask]
        return self.c

    def vector(self) -> np.ndarray:
        """Flatten to `[c..., d...]` (float32) for logging/export."""
        if self.dim_d:
            parts = [self.c.astype(np.float32), self.d.astype(np.float32)]
            return cat_f32(parts)
        return self.c.astype(np.float32, copy=True)

    def scalar(self, index: int = 0) -> float:
        """Get scalar value from continuous action at given index."""
        if index < len(self.c):
            return float(self.c[index])
        return 0.0

    def as_array(self) -> np.ndarray:
        """Get continuous action as numpy array."""
        return self.c

    def copy(self) -> "Action":
        """Create a deep copy of this action."""
        return Action(
            c=self.c.copy(),
            d=self.d.copy(),
            dim_c=self.dim_c,
            dim_d=self.dim_d,
            range=self.range.copy() if self.range is not None else None,
            ncats=list(self.ncats),
            _space=self._space,
        )

    @classmethod
    def from_vector(
        cls,
        vec: Sequence[float],
        dim_c: int,
        dim_d: int,
        ncats: Optional[Sequence[int]] = None,
        range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "Action":
        """Create an Action from a flat vector `[c..., d...]`."""
        vec = np.asarray(vec, dtype=np.float32)
        expected = dim_c + dim_d
        if vec.size != expected:
            raise ValueError(f"vector length {vec.size} != expected {expected}")
        c = vec[:dim_c].astype(np.float32)
        d = vec[dim_c:].astype(np.int32) if dim_d else np.array([], dtype=np.int32)
        if ncats is None:
            ncats = []
        return cls(
            c=c, d=d, dim_c=dim_c, dim_d=dim_d, ncats=ncats, range=range,
        )

    @classmethod
    def from_gym_space(cls, space: gym.Space) -> "Action":
        """Create an Action from a Gymnasium space.

        Derives action specs (dim_c, dim_d, ncats, range) from the space.

        Args:
            space: Gymnasium action space (Box, Discrete, MultiDiscrete, or Dict)

        Returns:
            Action instance configured for the given space

        Raises:
            TypeError: If space type is not supported
        """
        action = cls()

        if isinstance(space, Box):
            action.set_specs(
                dim_c=int(np.prod(space.shape)),
                dim_d=0,
                ncats=[],
                range=(space.low.ravel().astype(np.float32),
                       space.high.ravel().astype(np.float32)),
            )
        elif isinstance(space, Discrete):
            action.set_specs(
                dim_c=0,
                dim_d=1,
                ncats=[int(space.n)],
            )
        elif isinstance(space, MultiDiscrete):
            action.set_specs(
                dim_c=0,
                dim_d=len(space.nvec),
                ncats=[int(n) for n in space.nvec],
            )
        elif isinstance(space, SpaceDict):
            # Mixed continuous/discrete
            dim_c = 0
            dim_d = 0
            ncats = []
            action_range = None

            if "c" in space.spaces and isinstance(space.spaces["c"], Box):
                c_space = space.spaces["c"]
                dim_c = int(np.prod(c_space.shape))
                action_range = (c_space.low.ravel().astype(np.float32),
                                c_space.high.ravel().astype(np.float32))

            if "d" in space.spaces:
                d_space = space.spaces["d"]
                if isinstance(d_space, Discrete):
                    dim_d = 1
                    ncats = [int(d_space.n)]
                elif isinstance(d_space, MultiDiscrete):
                    dim_d = len(d_space.nvec)
                    ncats = [int(n) for n in d_space.nvec]

            action.set_specs(
                dim_c=dim_c,
                dim_d=dim_d,
                ncats=ncats,
                range=action_range,
            )
        else:
            raise TypeError(f"Unsupported action space type: {type(space).__name__}")

        return action

    def __repr__(self) -> str:
        """Return concise string representation."""
        parts = [f"Action(dim_c={self.dim_c}"]
        if self.dim_c > 0:
            c_str = np.array2string(self.c, precision=3, separator=', ')
            parts.append(f"c={c_str}")
        if self.dim_d > 0:
            parts.append(f"dim_d={self.dim_d}, d={self.d.tolist()}, ncats={self.ncats}")
        return ", ".join(parts) + ")"

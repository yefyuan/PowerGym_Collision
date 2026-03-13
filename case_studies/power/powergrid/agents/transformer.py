"""Transformer agent following grid_agent style.

This module implements an on-load tap changer (OLTC) transformer agent with:
- Direct constructor parameters (no config dataclass)
- Discrete action space for tap position
- Explicit set_state() parameters
"""

from typing import Any, Dict, List, Optional

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID
from powergrid.core.features.tap_changer import TapChangerPh
from powergrid.utils.cost import tap_change_cost
from powergrid.utils.safety import loading_over_pct
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


class Transformer(DeviceAgent):
    """On-load tap changer (OLTC) transformer following grid_agent style.

    Controls transformer tap position for voltage regulation.
    Discrete action selects tap index in [tap_min, tap_max].

    Action: Discrete d[0] in {0, 1, ..., tap_max - tap_min}
            (maps to tap position tap_min + d[0])

    Example:
        >>> trafo = Transformer(
        ...     agent_id="trafo_1",
        ...     tap_min=-5,
        ...     tap_max=5,
        ...     sn_mva=100.0,
        ...     tap_change_cost=1.0,
        ... )
    """

    def __init__(
        self,
        agent_id: AgentID,
        # Tap limits
        tap_min: int = -5,
        tap_max: int = 5,
        # Capacity
        sn_mva: Optional[float] = None,
        # Cost parameters
        tap_change_cost: float = 0.0,
        # Time step
        dt_h: float = 1.0,
        # Phase model
        phase_model: str = "balanced_1ph",
        phase_spec: Optional[Dict[str, Any]] = None,
        # Hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        """Initialize transformer agent.

        Args:
            agent_id: Unique identifier
            tap_min: Minimum tap position
            tap_max: Maximum tap position
            sn_mva: Rated apparent power (MVA)
            tap_change_cost: Cost per tap step moved
            dt_h: Time step in hours
            phase_model: Phase model type
            phase_spec: Phase specification dict
            upstream_id: Parent agent ID
            env_id: Environment ID
            schedule_config: Timing configuration
            policy: Agent policy
            protocol: Coordination protocol
        """
        # Store parameters
        self._tap_min = tap_min
        self._tap_max = tap_max
        self._sn_mva = sn_mva
        self._tap_change_cost = tap_change_cost
        self._dt_h = dt_h

        # Phase model
        self.phase_model = PhaseModel(phase_model)
        self.phase_spec = PhaseSpec().from_dict(phase_spec or {})
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        # Track last tap for cost calculation
        self._last_tap_position = 0

        # Create features
        features = [
            TapChangerPh(
                tap_position=0,
                tap_min=tap_min,
                tap_max=tap_max,
            ),
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            protocol=protocol,
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize discrete action space for tap selection.

        Action: d[0] in {0, 1, ..., tap_max - tap_min}

        Returns:
            Initialized Action object
        """
        action = Action()

        num_positions = self._tap_max - self._tap_min + 1

        action.set_specs(
            dim_c=0,
            dim_d=1,
            ncats=num_positions,
        )

        # Initialize at neutral tap (index for tap=0)
        neutral_index = -self._tap_min
        action.set_values(d=np.array([neutral_index], dtype=np.int32))

        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action from Action object or numpy array.

        Args:
            action: Action object or numpy array
        """
        if isinstance(action, Action):
            if action.d.size > 0:
                self.action.set_values(d=action.d)
        elif isinstance(action, np.ndarray):
            self.action.set_values(d=action)
        elif isinstance(action, int):
            self.action.set_values(d=np.array([action], dtype=np.int32))
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(
        self,
        tap_position: Optional[int] = None,
        loading_percent: float = 0.0,
        **kwargs
    ) -> None:
        """Update state from action or explicit parameters.

        Args:
            tap_position: Tap position. If None, computed from action.
            loading_percent: Transformer loading percentage (for safety).
            **kwargs: Additional state parameters
        """
        # 1. Compute tap position from action if not provided
        if tap_position is None:
            tap_index = int(self.action.d[0]) if self.action.d.size > 0 else 0
            tap_position = self._tap_min + tap_index

        # 2. Update tap changer feature
        self.tap_changer.set_values(tap_position=tap_position)

        # 3. Update cost and safety
        self._update_cost_safety_metrics(tap_position, loading_percent)

    def _update_cost_safety_metrics(self, tap_position: int, loading_percent: float) -> None:
        """Update cost and safety metrics.

        Args:
            tap_position: Current tap position
            loading_percent: Transformer loading percentage
        """
        # Cost: tap change operations
        delta = abs(tap_position - self._last_tap_position)
        cost = tap_change_cost(delta, self._tap_change_cost)
        self._last_tap_position = tap_position

        # Safety: loading violations
        safety = loading_over_pct(loading_percent)

        self.metrics.set_values(cost=cost, safety=safety)

    @property
    def tap_changer(self) -> TapChangerPh:
        """Get tap changer feature."""
        for f in self.state.features.values():
            if isinstance(f, TapChangerPh):
                return f

    @property
    def tap_position(self) -> int:
        """Get current tap position."""
        return self.tap_changer.tap_position

    def __repr__(self) -> str:
        return (f"Transformer(id={self.agent_id}, S={self._sn_mva}MVA, "
                f"tap∈[{self._tap_min},{self._tap_max}])")

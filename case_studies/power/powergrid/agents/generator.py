"""Dispatchable Generator agent following grid_agent style.

This module implements a generator agent with:
- Direct constructor parameters (no config dataclass)
- Normalized [-1, 1] action space
- Explicit set_state() parameters
- Unit commitment logic
"""

from typing import Any, Dict, List, Optional

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.features.metrics import CostSafetyMetrics
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import ScheduleConfig
from heron.utils.typing import AgentID
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.power_limits import PowerLimits
from powergrid.core.features.status import StatusBlock
from powergrid.utils.cost import cost_from_curve
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


class Generator(DeviceAgent):
    """Dispatchable Generator following grid_agent style.

    Controls a generator with:
    - Active power control (P)
    - Optional reactive power control (Q)
    - Unit commitment (on/off with startup/shutdown times)
    - Fuel cost curve

    Action: 1D continuous [P_norm] in [-1, 1] (denormalized to [p_min, p_max])
            Optional 2D for PQ control: [P_norm, Q_norm]
            Optional discrete for UC: d[0] in {0=off, 1=on}

    Example:
        >>> gen = Generator(
        ...     agent_id="gen_1",
        ...     bus="bus_1",
        ...     p_min_MW=1.0,
        ...     p_max_MW=10.0,
        ...     cost_curve_coefs=(0.01, 1.0, 0.0),
        ... )
    """

    def __init__(
        self,
        agent_id: AgentID,
        bus: str,
        # Power limits
        p_min_MW: float = 0.0,
        p_max_MW: float = 10.0,
        q_min_MVAr: Optional[float] = None,
        q_max_MVAr: Optional[float] = None,
        s_rated_MVA: Optional[float] = None,
        pf_min_abs: Optional[float] = None,
        # Unit commitment
        startup_time_hr: Optional[int] = None,
        shutdown_time_hr: Optional[int] = None,
        startup_cost: float = 0.0,
        shutdown_cost: float = 0.0,
        # Economic parameters
        cost_curve_coefs: tuple = (0.0, 0.0, 0.0),
        dt_h: float = 1.0,
        # Phase model
        phase_model: str = "balanced_1ph",
        phase_spec: Optional[Dict[str, Any]] = None,
        # Generator type
        gen_type: str = "fossil",
        source: Optional[str] = None,
        # Hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        """Initialize generator agent.

        Args:
            agent_id: Unique identifier
            bus: Bus name where generator is connected
            p_min_MW: Minimum active power (MW)
            p_max_MW: Maximum active power (MW)
            q_min_MVAr: Minimum reactive power (MVAr), if Q-control enabled
            q_max_MVAr: Maximum reactive power (MVAr), if Q-control enabled
            s_rated_MVA: Rated apparent power (MVA)
            pf_min_abs: Minimum power factor (absolute)
            startup_time_hr: Hours required for startup transition
            shutdown_time_hr: Hours required for shutdown transition
            startup_cost: Cost per startup event
            shutdown_cost: Cost per shutdown event
            cost_curve_coefs: Fuel cost coefficients (a, b, c) for a*P^2 + b*P + c
            dt_h: Time step in hours
            phase_model: Phase model type
            phase_spec: Phase specification dict
            gen_type: Generator type ("fossil", "renewable", etc.)
            source: Renewable source type ("solar", "wind", or None)
            upstream_id: Parent agent ID
            env_id: Environment ID
            schedule_config: Timing configuration
            policy: Agent policy
            protocol: Coordination protocol
        """
        # Store parameters
        self._bus = bus
        self._p_min_MW = p_min_MW
        self._p_max_MW = p_max_MW
        self._q_min_MVAr = q_min_MVAr
        self._q_max_MVAr = q_max_MVAr
        self._dt_h = dt_h
        self._cost_curve_coefs = cost_curve_coefs
        self._startup_cost = startup_cost
        self._shutdown_cost = shutdown_cost
        self._startup_time_hr = startup_time_hr
        self._shutdown_time_hr = shutdown_time_hr
        self._gen_type = gen_type
        self._source = source

        # Phase model
        self.phase_model = PhaseModel(phase_model)
        self.phase_spec = PhaseSpec().from_dict(phase_spec or {})
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        # UC cost tracking
        self._uc_cost_step = 0.0

        # Check if Q-control is enabled
        self._has_q_control = (
            (q_min_MVAr is not None and q_max_MVAr is not None) or
            (pf_min_abs is not None and s_rated_MVA is not None)
        )

        # Check if UC is enabled
        self._has_uc = startup_time_hr is not None or shutdown_time_hr is not None

        # Create features
        features = [
            ElectricalBasePh(P_MW=p_min_MW, Q_MVAr=0.0),
            PowerLimits(
                p_min_MW=p_min_MW,
                p_max_MW=p_max_MW,
                q_min_MVAr=q_min_MVAr,
                q_max_MVAr=q_max_MVAr,
                s_rated_MVA=s_rated_MVA,
                pf_min_abs=pf_min_abs,
            ),
            StatusBlock(
                state="online",
                in_service=True,
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
        """Initialize normalized [-1, 1] action space.

        Action dimensions:
        - c[0]: P_norm in [-1, 1] (denormalized to [p_min, p_max])
        - c[1]: Q_norm in [-1, 1] (if Q-control enabled)
        - d[0]: UC command {0=off, 1=on} (if UC enabled)

        Returns:
            Initialized Action object
        """
        action = Action()

        # Continuous dimensions
        dim_c = 1  # Always have P control
        if self._has_q_control:
            dim_c = 2

        # Discrete dimensions for UC
        dim_d = 1 if self._has_uc else 0
        ncats = 2 if self._has_uc else 0

        action.set_specs(
            dim_c=dim_c,
            dim_d=dim_d,
            ncats=ncats,
            range=(np.array([-1.0] * dim_c), np.array([1.0] * dim_c)),
        )

        # Initialize with zeros (P=mid-range, Q=0 if applicable)
        action.set_values(c=np.zeros(dim_c, dtype=np.float32))
        if self._has_uc:
            action.set_values(d=np.array([1], dtype=np.int32))  # Default to "on"

        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action from Action object or numpy array.

        Args:
            action: Action object or numpy array
        """
        if isinstance(action, Action):
            if action.c.size > 0:
                self.action.set_values(c=action.c)
            if action.d.size > 0:
                self.action.set_values(d=action.d)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(
        self,
        P: Optional[float] = None,
        Q: Optional[float] = None,
        uc_cmd: Optional[int] = None,
        **kwargs
    ) -> None:
        """Update state from action or explicit parameters.

        Args:
            P: Active power setpoint (MW). If None, computed from action.
            Q: Reactive power setpoint (MVAr). If None, computed from action.
            uc_cmd: UC command {0=off, 1=on}. If None, from action.d[0].
            **kwargs: Additional state parameters
        """
        # 1. Update UC status
        if uc_cmd is None and self._has_uc and self.action.d.size > 0:
            uc_cmd = int(self.action.d[0])
        self._update_uc_status(uc_cmd)

        # 2. Update power outputs
        if P is None:
            # Denormalize P from [-1, 1] to [p_min, p_max]
            p_norm = self.action.c[0] if self.action.c.size >= 1 else 0.0
            P = self._p_min_MW + (p_norm + 1) / 2 * (self._p_max_MW - self._p_min_MW)

        if Q is None and self._has_q_control:
            # Denormalize Q from [-1, 1] to [q_min, q_max]
            q_norm = self.action.c[1] if self.action.c.size >= 2 else 0.0
            if self._q_min_MVAr is not None and self._q_max_MVAr is not None:
                Q = self._q_min_MVAr + (q_norm + 1) / 2 * (self._q_max_MVAr - self._q_min_MVAr)
            else:
                Q = 0.0
        elif Q is None:
            Q = 0.0

        # If offline, set power to zero
        if self.status.state != "online":
            P = 0.0
            Q = 0.0

        # Project to limits
        P_eff, Q_eff = self.limits.project_pq(P, Q)

        # Update electrical feature
        self.electrical.set_values(P_MW=P_eff, Q_MVAr=Q_eff)

        # 3. Update cost and safety
        self._update_cost_safety_metrics(P_eff, Q_eff)

    def _update_uc_status(self, uc_cmd: Optional[int]) -> None:
        """Update unit commitment status.

        Args:
            uc_cmd: UC command {0=off, 1=on}
        """
        self._uc_cost_step = 0.0

        if not self._has_uc or uc_cmd is None:
            return

        status = self.status
        state = status.state
        t_in_state_s = status.t_in_state_s or 0.0
        t_to_next_s = status.t_to_next_s
        progress_frac = status.progress_frac
        dt = self._dt_h

        # Update time in state
        t_in_state_s += dt

        t_start = self._startup_time_hr or 0
        t_stop = self._shutdown_time_hr or 0

        # Handle UC commands when not in transition
        if t_to_next_s is None:
            # Request OFF from ONLINE
            if state == "online" and uc_cmd == 0:
                if t_stop <= 0:
                    state = "offline"
                    t_in_state_s = 0.0
                    self._uc_cost_step = self._shutdown_cost
                else:
                    state = "shutdown"
                    t_in_state_s = 0.0
                    t_to_next_s = t_stop * dt
                    progress_frac = 0.0

            # Request ON from OFFLINE
            elif state == "offline" and uc_cmd == 1:
                if t_start <= 0:
                    state = "online"
                    t_in_state_s = 0.0
                    self._uc_cost_step = self._startup_cost
                else:
                    state = "startup"
                    t_in_state_s = 0.0
                    t_to_next_s = t_start * dt
                    progress_frac = 0.0

        # Progress transitional states
        if t_to_next_s is not None and state in ("startup", "shutdown"):
            t_to_next_s = max(0.0, t_to_next_s - dt)

            total = t_start if state == "startup" else t_stop
            if total > 0:
                denom = max(total * dt, 1e-9)
                progress_frac = 1.0 - t_to_next_s / denom

            # Finish transition
            if t_to_next_s == 0.0:
                if state == "startup":
                    state = "online"
                    self._uc_cost_step = self._startup_cost
                elif state == "shutdown":
                    state = "offline"
                    self._uc_cost_step = self._shutdown_cost
                t_in_state_s = 0.0
                t_to_next_s = None
                progress_frac = None

        # Update status feature
        self.state.update_feature(
            StatusBlock.feature_name,
            state=state,
            t_in_state_s=t_in_state_s,
            t_to_next_s=t_to_next_s,
            progress_frac=progress_frac,
        )

    def _update_cost_safety_metrics(self, P: float, Q: float) -> None:
        """Update cost and safety metrics.

        Args:
            P: Active power (MW)
            Q: Reactive power (MVAr)
        """
        on = 1.0 if self.status.state == "online" else 0.0
        dt = self._dt_h

        # Fuel cost
        fuel_cost = cost_from_curve(P, self._cost_curve_coefs)
        cost = fuel_cost * on * dt + self._uc_cost_step

        # Safety violations
        violations = self.limits.feasible(P, Q)
        safety = np.sum(list(violations.values())) * on * dt

        self.metrics.set_values(cost=cost, safety=safety)

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward based on cost and safety.

        Reward = -cost - safety

        Args:
            local_state: State dict with feature vectors

        Returns:
            Reward value (higher is better)
        """
        cost = 0.0
        safety = 0.0

        if "CostSafetyMetrics" in local_state:
            metrics_vec = local_state["CostSafetyMetrics"]
            cost = float(metrics_vec[0])
            safety = float(metrics_vec[1])

        return -cost - safety

    @property
    def electrical(self) -> ElectricalBasePh:
        """Get electrical feature."""
        for f in self.state.features.values():
            if isinstance(f, ElectricalBasePh):
                return f

    @property
    def status(self) -> StatusBlock:
        """Get status feature."""
        for f in self.state.features.values():
            if isinstance(f, StatusBlock):
                return f

    @property
    def limits(self) -> PowerLimits:
        """Get power limits feature."""
        for f in self.state.features.values():
            if isinstance(f, PowerLimits):
                return f

    @property
    def bus(self) -> str:
        """Get bus name."""
        return self._bus

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent_id

    @property
    def source(self) -> Optional[str]:
        """Get renewable source type."""
        return self._source

    def __repr__(self) -> str:
        return (f"Generator(id={self.agent_id}, bus={self._bus}, "
                f"P∈[{self._p_min_MW},{self._p_max_MW}]MW)")

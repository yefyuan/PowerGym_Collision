"""Energy Storage System agent following grid_agent style.

This module implements an ESS agent with:
- Direct constructor parameters (no config dataclass)
- Normalized [-1, 1] action space
- Explicit set_state() parameters
- SOC dynamics and degradation tracking
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.features.metrics import CostSafetyMetrics
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.power_limits import PowerLimits
from powergrid.core.features.storage import StorageBlock
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


class ESS(DeviceAgent):
    """Energy Storage System following grid_agent style.

    Controls a battery storage unit with:
    - Active power control (charge/discharge)
    - SOC tracking and constraints
    - Optional reactive power control
    - Degradation cost tracking

    Action: 1D continuous [P_norm] in [-1, 1]
            - Positive = charging
            - Negative = discharging
            Optional 2D for PQ control: [P_norm, Q_norm]

    Example:
        >>> ess = ESS(
        ...     agent_id="ess_1",
        ...     bus="bus_1",
        ...     p_min_MW=-5.0,
        ...     p_max_MW=5.0,
        ...     capacity_MWh=10.0,
        ...     init_soc=0.5,
        ... )
    """

    def __init__(
        self,
        agent_id: AgentID,
        bus: str,
        # Power limits
        p_min_MW: float = -5.0,  # Discharge limit (negative)
        p_max_MW: float = 5.0,   # Charge limit (positive)
        q_min_MVAr: Optional[float] = None,
        q_max_MVAr: Optional[float] = None,
        s_rated_MVA: Optional[float] = None,
        # Storage parameters
        capacity_MWh: float = 10.0,
        init_soc: float = 0.5,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        ch_eff: float = 0.95,
        dsc_eff: float = 0.95,
        # Degradation parameters
        degr_cost_per_MWh: float = 0.0,
        degr_cost_per_cycle: float = 0.0,
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
        """Initialize ESS agent.

        Args:
            agent_id: Unique identifier
            bus: Bus name where ESS is connected
            p_min_MW: Minimum power (negative = discharge limit)
            p_max_MW: Maximum power (positive = charge limit)
            q_min_MVAr: Minimum reactive power (if Q-control enabled)
            q_max_MVAr: Maximum reactive power (if Q-control enabled)
            s_rated_MVA: Rated apparent power
            capacity_MWh: Energy capacity in MWh
            init_soc: Initial state of charge [0, 1]
            soc_min: Minimum SOC limit
            soc_max: Maximum SOC limit
            ch_eff: Charging efficiency
            dsc_eff: Discharging efficiency
            degr_cost_per_MWh: Degradation cost per MWh throughput
            degr_cost_per_cycle: Degradation cost per equivalent full cycle
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
        self._bus = bus
        self._p_min_MW = p_min_MW
        self._p_max_MW = p_max_MW
        self._q_min_MVAr = q_min_MVAr
        self._q_max_MVAr = q_max_MVAr
        self._dt_h = dt_h

        # Phase model
        self.phase_model = PhaseModel(phase_model)
        self.phase_spec = PhaseSpec().from_dict(phase_spec or {})
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        # Check if Q-control is enabled
        self._has_q_control = q_min_MVAr is not None and q_max_MVAr is not None

        # Create features
        features = [
            ElectricalBasePh(P_MW=0.0, Q_MVAr=0.0),
            StorageBlock(
                e_capacity_MWh=capacity_MWh,
                soc=init_soc,
                soc_min=soc_min,
                soc_max=soc_max,
                p_ch_max_MW=p_max_MW,  # Charge limit (positive)
                p_dsc_max_MW=abs(p_min_MW),  # Discharge limit (positive magnitude)
                ch_eff=ch_eff,
                dsc_eff=dsc_eff,
                degr_cost_per_MWh=degr_cost_per_MWh,
                degr_cost_per_cycle=degr_cost_per_cycle,
            ),
        ]

        # Add PowerLimits if Q-control enabled
        if self._has_q_control:
            features.append(
                PowerLimits(
                    p_min_MW=p_min_MW,
                    p_max_MW=p_max_MW,
                    q_min_MVAr=q_min_MVAr,
                    q_max_MVAr=q_max_MVAr,
                    s_rated_MVA=s_rated_MVA,
                )
            )

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
        - c[0]: P_norm in [-1, 1] (positive=charge, negative=discharge)
        - c[1]: Q_norm in [-1, 1] (if Q-control enabled)

        Returns:
            Initialized Action object
        """
        action = Action()

        dim_c = 2 if self._has_q_control else 1

        action.set_specs(
            dim_c=dim_c,
            dim_d=0,
            ncats=0,
            range=(np.array([-1.0] * dim_c), np.array([1.0] * dim_c)),
        )

        action.set_values(c=np.zeros(dim_c, dtype=np.float32))

        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action from Action object or numpy array.

        Args:
            action: Action object or numpy array
        """
        if isinstance(action, Action):
            if action.c.size > 0:
                self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(
        self,
        P: Optional[float] = None,
        Q: Optional[float] = None,
        **kwargs
    ) -> None:
        """Update state from action or explicit parameters.

        Args:
            P: Active power setpoint (MW). If None, computed from action.
               Positive = charging, Negative = discharging.
            Q: Reactive power setpoint (MVAr). If None, computed from action.
            **kwargs: Additional state parameters
        """
        # 1. Compute power from action if not provided
        if P is None:
            P = self._denormalize_power(self.action.c[0] if self.action.c.size >= 1 else 0.0)

        if Q is None and self._has_q_control:
            q_norm = self.action.c[1] if self.action.c.size >= 2 else 0.0
            Q = self._q_min_MVAr + (q_norm + 1) / 2 * (self._q_max_MVAr - self._q_min_MVAr)
        elif Q is None:
            Q = 0.0

        # 2. Clip power to feasible range (SOC-aware)
        P_eff = self._feasible_power(P)

        # 3. Project PQ if limits available
        if self.limits is not None:
            P_eff, Q_eff = self.limits.project_pq(P_eff, Q)
        else:
            Q_eff = Q

        # 4. Update electrical feature
        self.electrical.set_values(P_MW=P_eff, Q_MVAr=Q_eff)

        # 5. Update SOC dynamics
        self._update_soc(P_eff)

        # 6. Update cost and safety
        self._update_cost_safety_metrics(P_eff, Q_eff)

    def _denormalize_power(self, p_norm: float) -> float:
        """Denormalize power from [-1, 1] to [p_min, p_max].

        For ESS, considers SOC-based limits:
        - If p_norm > 0: map to [0, max_charge] based on SOC headroom
        - If p_norm < 0: map to [max_discharge, 0] based on SOC headroom

        Args:
            p_norm: Normalized power in [-1, 1]

        Returns:
            Denormalized power in MW
        """
        storage = self.storage
        dt = self._dt_h

        # Charging headroom
        if storage.soc < storage.soc_max:
            e_room_ch = (storage.soc_max - storage.soc) * storage.e_capacity_MWh
            max_charge = min(self._p_max_MW, e_room_ch / max(storage.ch_eff * dt, 1e-9))
        else:
            max_charge = 0.0

        # Discharging headroom
        if storage.soc > storage.soc_min:
            e_room_dsc = (storage.soc - storage.soc_min) * storage.e_capacity_MWh
            max_discharge = min(abs(self._p_min_MW), e_room_dsc * storage.dsc_eff / max(dt, 1e-9))
        else:
            max_discharge = 0.0

        # Denormalize
        if p_norm > 0:
            return p_norm * max_charge
        else:
            return p_norm * max_discharge

    def _feasible_power(self, P_req: float) -> float:
        """Clip power to feasible range based on SOC.

        Args:
            P_req: Requested power (MW)

        Returns:
            Clipped power respecting SOC and power constraints
        """
        storage = self.storage
        dt = self._dt_h

        # Charging headroom
        if storage.soc < storage.soc_max:
            e_room_ch = (storage.soc_max - storage.soc) * storage.e_capacity_MWh
            p_max_soc = e_room_ch / max(storage.ch_eff * dt, 1e-9)
        else:
            p_max_soc = 0.0

        # Discharging headroom
        if storage.soc > storage.soc_min:
            e_room_dsc = (storage.soc - storage.soc_min) * storage.e_capacity_MWh
            p_min_soc = -e_room_dsc * storage.dsc_eff / max(dt, 1e-9)
        else:
            p_min_soc = 0.0

        p_min = max(p_min_soc, self._p_min_MW)
        p_max = min(p_max_soc, self._p_max_MW)

        return np.clip(P_req, p_min, p_max)

    def _update_soc(self, P_MW: float) -> None:
        """Update SOC based on power.

        Args:
            P_MW: Active power (positive=charging, negative=discharging)
        """
        storage = self.storage
        dt = self._dt_h

        # Energy change
        if P_MW >= 0.0:
            delta_e = P_MW * storage.ch_eff * dt
        else:
            delta_e = P_MW / max(storage.dsc_eff, 1e-9) * dt

        # Update SOC
        delta_soc = delta_e / storage.e_capacity_MWh
        new_soc = np.clip(
            storage.soc + delta_soc,
            storage.soc_min,
            storage.soc_max,
        )
        storage.set_values(soc=new_soc)

    def _update_cost_safety_metrics(self, P: float, Q: float) -> None:
        """Update cost and safety metrics.

        Args:
            P: Active power (MW)
            Q: Reactive power (MVAr)
        """
        storage = self.storage
        dt = self._dt_h

        # Energy throughput
        if P >= 0.0:
            delta_e = P * storage.ch_eff * dt
        else:
            delta_e = abs(P) / max(storage.dsc_eff, 1e-9) * dt

        # Update throughput tracking
        e_throughput = storage.e_throughput_MWh + abs(delta_e)

        # Degradation cost
        degr_cost_inc = storage.degr_cost_per_MWh * abs(delta_e)
        eq_cycles = abs(delta_e) / storage.e_capacity_MWh
        degr_cost_inc += storage.degr_cost_per_cycle * eq_cycles

        degr_cost_cum = storage.degr_cost_cum + degr_cost_inc

        storage.set_values(
            e_throughput_MWh=e_throughput,
            degr_cost_cum=degr_cost_cum
        )

        cost = degr_cost_inc

        # Safety: SOC violations
        safety = storage.soc_violation()
        if self.limits is not None:
            violations = self.limits.feasible(P, Q)
            safety += np.sum(list(violations.values())) * dt

        self.metrics.set_values(cost=cost, safety=safety)

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward based on SOC, cost, and safety.

        Reward = SOC - cost - safety
        (maximize SOC while minimizing cost and safety violations)

        Args:
            local_state: State dict with feature vectors

        Returns:
            Reward value (higher is better)
        """
        soc_reward = 0.0
        cost = 0.0
        safety = 0.0

        # Extract SOC from StorageBlock
        if "StorageBlock" in local_state:
            storage_vec = local_state["StorageBlock"]
            soc_reward = float(storage_vec[0])

        # Extract cost/safety
        if "CostSafetyMetrics" in local_state:
            metrics_vec = local_state["CostSafetyMetrics"]
            cost = float(metrics_vec[0])
            safety = float(metrics_vec[1])

        return soc_reward - cost - safety

    @property
    def electrical(self) -> ElectricalBasePh:
        """Get electrical feature."""
        for f in self.state.features.values():
            if isinstance(f, ElectricalBasePh):
                return f

    @property
    def storage(self) -> StorageBlock:
        """Get storage feature."""
        for f in self.state.features.values():
            if isinstance(f, StorageBlock):
                return f

    @property
    def limits(self) -> Optional[PowerLimits]:
        """Get power limits feature (if Q-control enabled)."""
        for f in self.state.features.values():
            if isinstance(f, PowerLimits):
                return f
        return None

    @property
    def bus(self) -> str:
        """Get bus name."""
        return self._bus

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent_id

    @property
    def soc(self) -> float:
        """Get current SOC."""
        return self.storage.soc

    @property
    def capacity(self) -> float:
        """Get energy capacity in MWh."""
        return self.storage.e_capacity_MWh

    def __repr__(self) -> str:
        cap = self.storage.e_capacity_MWh if self.storage else 0.0
        return (f"ESS(id={self.agent_id}, bus={self._bus}, "
                f"capacity={cap}MWh, P∈[{self._p_min_MW},{self._p_max_MW}]MW)")

"""Collision-aware PowerGrid coordinator agent.

Extends PowerGridAgent with collision detection capabilities to track
safety violations across the microgrid network.
"""

from typing import Any, Dict, Optional

import numpy as np

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

from powergrid.agents.power_grid_agent import PowerGridAgent
from collision_case.collision_features import CollisionMetrics


class CollisionGridAgent(PowerGridAgent):
    """PowerGrid coordinator with collision detection.

    Extends PowerGridAgent to track safety violations:
    - Voltage violations (over/under)
    - Line overloading
    - Device over-rating
    - Power factor violations
    - SOC violations

    The collision metrics are computed during set_state() after power flow results
    are available from the environment.
    """

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: Optional[Dict[AgentID, Any]] = None,
        protocol: Optional[Protocol] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        # Collision-specific params
        voltage_upper_limit: float = 1.05,
        voltage_lower_limit: float = 0.95,
        line_loading_limit: float = 100.0,
        penalty_weight: float = 10.0,
    ):
        """Initialize collision-aware grid coordinator.

        Args:
            agent_id: Unique identifier
            subordinates: Dict of device agents
            protocol: Coordination protocol
            upstream_id: Parent agent ID
            env_id: Environment ID
            schedule_config: Timing configuration
            policy: Agent policy
            voltage_upper_limit: Upper voltage limit (p.u.)
            voltage_lower_limit: Lower voltage limit (p.u.)
            line_loading_limit: Line loading limit (%)
            penalty_weight: Penalty multiplier for safety violations
        """
        # Initialize parent
        super().__init__(
            agent_id=agent_id,
            subordinates=subordinates,
            protocol=protocol,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
        )

        # Add collision metrics feature to state features dict
        collision_metrics = CollisionMetrics()
        self.state.features[CollisionMetrics.feature_name] = collision_metrics

        # Store collision parameters
        self._voltage_upper = voltage_upper_limit
        self._voltage_lower = voltage_lower_limit
        self._line_loading_limit = line_loading_limit
        self._penalty_weight = penalty_weight

        # Track device-level violations
        self._device_violations: Dict[str, float] = {}

    def set_state(self, **kwargs) -> None:
        """Update state and compute collision metrics.

        Expects kwargs to contain power flow results from environment:
        - bus_voltages: Dict[bus_name, voltage_pu]
        - line_loadings: Dict[line_name, loading_percent]
        - converged: bool indicating power flow convergence

        Also processes device-level safety metrics from subordinates.
        """
        # Call parent to update network features
        super().set_state(**kwargs)

        # Compute collision metrics from power flow results (if available)
        if "bus_voltages" in kwargs or "line_loadings" in kwargs:
            self._compute_collision_metrics(kwargs)
            # Aggregate device-level violations
            self._aggregate_device_violations()

    def _compute_collision_metrics(self, power_flow_data: Dict) -> None:
        """Compute collision metrics from power flow results.

        Args:
            power_flow_data: Dict containing:
                - bus_voltages: Dict[bus_name, voltage_pu]
                - line_loadings: Dict[line_name, loading_percent]
                - converged: bool
        """
        # Get collision metrics feature
        collision = self._get_collision_feature()
        if collision is None:
            return

        # Reset metrics
        collision.reset()

        converged = power_flow_data.get("converged", False)
        if not converged:
            # If power flow didn't converge, apply large penalty
            collision.safety_total = 20.0
            collision.collision_flag = 1.0
            return

        # Extract voltage violations
        bus_voltages = power_flow_data.get("bus_voltages", {})
        for bus_name, voltage_pu in bus_voltages.items():
            # Only process buses belonging to this microgrid
            if self.agent_id in bus_name or bus_name.startswith(f"{self.agent_id}_"):
                if voltage_pu > self._voltage_upper:
                    collision.overvoltage += voltage_pu - self._voltage_upper
                elif voltage_pu < self._voltage_lower:
                    collision.undervoltage += self._voltage_lower - voltage_pu

        # Extract line overloading violations
        line_loadings = power_flow_data.get("line_loadings", {})
        for line_name, loading_pct in line_loadings.items():
            # Only process lines belonging to this microgrid
            if self.agent_id in line_name or line_name.startswith(f"{self.agent_id}_"):
                if loading_pct > self._line_loading_limit:
                    collision.overloading += (loading_pct - self._line_loading_limit) / 100.0

        # Compute total
        collision.compute_total()

        # Update state
        self.state.update_feature(
            CollisionMetrics.feature_name,
            **collision.to_dict()
        )

    def _aggregate_device_violations(self) -> None:
        """Aggregate safety violations from subordinate devices.

        Collects device over-rating, PF violations, and SOC violations
        from subordinate agents and adds to collision metrics.
        """
        collision = self._get_collision_feature()
        if collision is None:
            return

        device_safety = 0.0
        pf_violations = 0.0
        soc_violations = 0.0

        # Collect from subordinates
        for dev_id, device in self.subordinates.items():
            if hasattr(device, 'metrics'):
                # Get safety metric from device
                safety = device.metrics.safety
                device_safety += safety

            # Check for specific violation types
            if hasattr(device, 'limits'):
                # Power factor violations
                P = device.electrical.P_MW if hasattr(device, 'electrical') else 0.0
                Q = device.electrical.Q_MVAr if hasattr(device, 'electrical') else 0.0
                violations = device.limits.feasible(P, Q)
                
                if 'pf_violation' in violations:
                    pf_violations += violations['pf_violation']

            # SOC violations for storage
            if hasattr(device, 'storage'):
                soc = device.storage.soc
                soc_min = device.storage.soc_min
                soc_max = device.storage.soc_max
                
                if soc > soc_max:
                    soc_violations += soc - soc_max
                elif soc < soc_min:
                    soc_violations += soc_min - soc

        # Update collision metrics
        collision.device_overrating = device_safety
        collision.pf_violations = pf_violations
        collision.soc_violations = soc_violations
        collision.compute_total()

        self.state.update_feature(
            CollisionMetrics.feature_name,
            **collision.to_dict()
        )

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward with collision penalties.

        Reward = -cost - penalty * safety_total

        Args:
            local_state: State dict with feature vectors

        Returns:
            Reward value (higher is better)
        """
        # Get base reward from parent (sum of device costs)
        base_reward = super().compute_local_reward(local_state)

        # Add collision penalty
        if "CollisionMetrics" in local_state:
            metrics_vec = local_state["CollisionMetrics"]
            # Extract safety_total (index 6 in the feature vector)
            safety_total = float(metrics_vec[6]) if len(metrics_vec) > 6 else 0.0
            collision_penalty = self._penalty_weight * safety_total
            return base_reward - collision_penalty

        return base_reward

    def _get_collision_feature(self) -> Optional[CollisionMetrics]:
        """Get collision metrics feature.

        Returns:
            CollisionMetrics feature or None
        """
        for f in self.state.features.values():
            if isinstance(f, CollisionMetrics):
                return f
        return None

    def get_collision_stats(self) -> Dict[str, float]:
        """Get current collision statistics.

        Returns:
            Dict with collision metrics
        """
        collision = self._get_collision_feature()
        if collision is None:
            return {}

        return {
            "overvoltage": collision.overvoltage,
            "undervoltage": collision.undervoltage,
            "overloading": collision.overloading,
            "device_overrating": collision.device_overrating,
            "pf_violations": collision.pf_violations,
            "soc_violations": collision.soc_violations,
            "safety_total": collision.safety_total,
            "collision_flag": collision.collision_flag,
        }

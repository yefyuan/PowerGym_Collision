"""Grid System Agent for power grid simulations.

This module implements a system-level agent that coordinates multiple
PowerGridAgent (coordinator) instances following Heron's agent hierarchy.
"""

from typing import Any, Dict as DictType, List, Optional

from heron.agents.system_agent import SystemAgent
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol, NoProtocol
from heron.scheduling.tick_config import ScheduleConfig
from heron.utils.typing import AgentID
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.core.features.system import (
    SystemFrequency,
    AggregateGeneration,
    AggregateLoad,
)


class GridSystemAgent(SystemAgent):
    """System-level agent managing multiple power grid coordinators.

    Follows Heron's agent hierarchy:
    - GridSystemAgent (SystemAgent, level 3)
        -> PowerGridAgent (CoordinatorAgent, level 2)
            -> DeviceAgent (FieldAgent, level 1)

    Example:
        >>> grids = {
        ...     "grid_1": PowerGridAgent(agent_id="grid_1", subordinates=devices_1),
        ...     "grid_2": PowerGridAgent(agent_id="grid_2", subordinates=devices_2),
        ... }
        >>> system = GridSystemAgent(
        ...     agent_id="system",
        ...     subordinates=grids,
        ... )
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        subordinates: Optional[DictType[AgentID, PowerGridAgent]] = None,
        features: Optional[List[Feature]] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        """Initialize grid system agent.

        Args:
            agent_id: Unique identifier (defaults to "grid_system")
            subordinates: PowerGridAgent instances to manage
            features: Additional system-level features
            env_id: Environment ID
            schedule_config: Timing configuration for event-driven scheduling
            policy: System-level policy
            protocol: System-level coordination protocol
        """
        # Add default power-grid features
        default_features = [
            SystemFrequency(frequency_hz=60.0, nominal_hz=60.0),
            AggregateGeneration(total_mw=0.0),
            AggregateLoad(total_mw=0.0),
        ]
        all_features = (features or []) + default_features

        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates or {},
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            protocol=protocol or NoProtocol(),
        )

    # ============================================
    # Domain-Specific: Cost/Safety Aggregation
    # ============================================

    @property
    def cost(self) -> float:
        """Aggregate cost from all coordinators."""
        return sum(grid.cost for grid in self.coordinators.values())

    @property
    def safety(self) -> float:
        """Aggregate safety penalty from all coordinators."""
        return sum(grid.safety for grid in self.coordinators.values())

    def get_reward(self) -> DictType[str, float]:
        """Get system-wide reward metrics.

        Returns:
            Dict with cost, safety, and total reward
        """
        return {
            "cost": self.cost,
            "safety": self.safety,
            "total": -(self.cost + self.safety),
        }

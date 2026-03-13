"""Power Grid Coordinator Agent.

This module implements a coordinator agent for power grid simulations with:
- Direct constructor with pre-initialized device agents
- Cost/safety aggregation from subordinates

NOTE: Agents do NOT have direct access to env objects (e.g., PandaPower network).
      The env owns the network, runs simulations, and pushes results through
      global_state via the proxy.
"""

from typing import Dict as DictType, List, Optional

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import ScheduleConfig
from heron.utils.typing import AgentID
from powergrid.agents.device_agent import DeviceAgent


class PowerGridAgent(CoordinatorAgent):
    """Power grid coordinator managing device agents.

    Follows Heron's agent hierarchy:
    - GridSystemAgent (SystemAgent, level 3)
        -> PowerGridAgent (CoordinatorAgent, level 2)
            -> DeviceAgent (FieldAgent, level 1)

    Example:
        >>> devices = {
        ...     "gen_1": Generator(agent_id="gen_1", ...),
        ...     "ess_1": ESS(agent_id="ess_1", ...),
        ... }
        >>> grid = PowerGridAgent(
        ...     agent_id="grid_1",
        ...     subordinates=devices,
        ... )
    """

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: DictType[AgentID, DeviceAgent],
        features: Optional[List[Feature]] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        """Initialize power grid coordinator.

        Args:
            agent_id: Unique identifier
            subordinates: Pre-initialized device agents (REQUIRED)
            features: Coordinator-level features
            upstream_id: Parent agent ID
            env_id: Environment ID
            schedule_config: Timing configuration
            policy: Coordinator policy
            protocol: Coordination protocol
        """
        if not subordinates:
            raise ValueError(
                "PowerGridAgent requires subordinates. "
                "Create device agents externally and pass as subordinates dict."
            )

        # For caching cost/safety (can be overridden externally)
        self._cost: Optional[float] = None
        self._safety: Optional[float] = None

        super().__init__(
            agent_id=agent_id,
            features=features or [],
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            protocol=protocol,
        )

    # ============================================
    # Domain-Specific: Cost/Safety Aggregation
    # ============================================

    @property
    def devices(self) -> DictType[AgentID, DeviceAgent]:
        """Alias for subordinates in power grid domain."""
        return self.subordinates

    @property
    def cost(self) -> float:
        """Aggregate cost from all devices.

        Returns cached value if set externally, otherwise computes from devices.
        """
        if self._cost is not None:
            return self._cost
        return sum(device.cost for device in self.devices.values())

    @cost.setter
    def cost(self, value: float) -> None:
        self._cost = value

    @property
    def safety(self) -> float:
        """Aggregate safety penalty from all devices.

        Returns cached value if set externally, otherwise computes from devices.
        """
        if self._safety is not None:
            return self._safety
        return sum(device.safety for device in self.devices.values())

    @safety.setter
    def safety(self, value: float) -> None:
        self._safety = value

    def get_reward(self) -> DictType[str, float]:
        """Get cost and safety metrics.

        Returns:
            Dict with cost and safety values
        """
        return {"cost": self.cost, "safety": self.safety}

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute coordinator reward as sum of subordinate device rewards.

        The coordinator's reward is the aggregated reward from all its devices,
        since the coordinator's policy is trained to optimize total device performance.

        Args:
            local_state: Local state dict from proxy, includes 'subordinate_rewards'

        Returns:
            Sum of all subordinate device rewards
        """
        subordinate_rewards = local_state.get("subordinate_rewards", {})
        return sum(subordinate_rewards.values())

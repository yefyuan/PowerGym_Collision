"""Device agent base class for power grid devices.

This module follows the grid_agent style with:
- Simple constructors accepting parameters directly
- Normalized action ranges [-1, 1]
- Explicit set_state() parameters
- apply_action() that calls set_state()
"""

from abc import abstractmethod
from typing import Any, List, Optional

from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID
from powergrid.core.features.metrics import CostSafetyMetrics


class DeviceAgent(FieldAgent):
    """Base class for power grid device agents.

    Follows the grid_agent pattern:
    - init_action(): Define normalized [-1, 1] action space
    - set_action(): Store action values
    - set_state(): Denormalize action and update features
    - apply_action(): Calls set_state()
    - compute_local_reward(): Compute reward from local_state dict

    Subclasses must implement:
    - init_action(): Define action space dimensions
    - set_action(): Handle action input
    - set_state(): Update state from action (with optional explicit params)
    """

    def __init__(
        self,
        agent_id: AgentID,
        features: List[Feature],
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        """Initialize device agent.

        Args:
            agent_id: Unique identifier for this agent
            features: List of feature providers (pre-initialized)
            upstream_id: Parent agent ID in hierarchy
            env_id: Environment ID
            schedule_config: Timing configuration
            policy: Agent policy
            protocol: Coordination protocol
        """
        # Ensure CostSafetyMetrics is included
        has_cost_safety = any(isinstance(f, CostSafetyMetrics) for f in features)
        if not has_cost_safety:
            features = list(features) + [CostSafetyMetrics()]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            protocol=protocol,
            policy=policy,
        )

    @abstractmethod
    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize action space.

        Subclasses define:
        - Continuous action dimensions (dim_c) with normalized [-1, 1] range
        - Discrete action dimensions (dim_d) and categories (ncats)

        Args:
            features: Optional features for action space setup

        Returns:
            Initialized Action object with specs and default values
        """
        pass

    @abstractmethod
    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action from Action object or compatible format.

        Args:
            action: Action object or numpy array with action values
        """
        pass

    def set_state(self, **kwargs) -> None:
        """Update state based on current action.

        Subclasses should override to:
        1. Denormalize action from [-1, 1] to physical ranges
        2. Update feature values
        3. Update cost/safety metrics

        Args:
            **kwargs: Optional explicit state parameters
        """
        pass

    def apply_action(self) -> None:
        """Apply the current action to update state.

        Calls set_state() to update features from action.
        """
        self.set_state()

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward for this agent from local state.

        Default implementation: reward = -cost - safety
        (minimize operating cost and safety violations)

        Args:
            local_state: State dict from proxy.get_local_state() with structure:
                {"FeatureName": np.array([...]), ...}

                For DeviceAgent, this includes:
                - "CostSafetyMetrics": np.array([cost, safety])
                - Plus any device-specific features

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
    def metrics(self) -> CostSafetyMetrics:
        """Get cost/safety metrics feature."""
        for f in self.state.features.values():
            if isinstance(f, CostSafetyMetrics):
                return f

"""Vertical protocols for hierarchical coordination.

Vertical protocols handle Parent -> Subordinate coordination.
Each agent owns its own vertical protocol to coordinate its subordinates.
"""

from typing import Any, Dict, Optional

import numpy as np

from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    NoCommunication,
)
from heron.utils.typing import AgentID


class VectorDecompositionActionProtocol(ActionProtocol):
    """Decomposes coordinator's joint action vector into per-subordinate actions.

    This is the default action protocol for vertical coordination, handling the
    common case where a coordinator computes a joint action vector and needs to
    distribute it to subordinates.

    Decomposition Strategy:
    - If coordinator action is already a dict: use it directly
    - If coordinator action is a vector: split based on subordinate action dimensions
    - If coordinator action is None: return None for all subordinates

    Example:
        Coordinator computes joint action [0.5, 0.3, 0.2, 0.1]
        Subordinate 1 has action_dim=2 → receives [0.5, 0.3]
        Subordinate 2 has action_dim=2 → receives [0.2, 0.1]
    """

    def __init__(self):
        self._subordinate_action_dims: Dict[AgentID, int] = {}

    def register_subordinates(self, subordinates: Dict[AgentID, Any]) -> None:
        """Register subordinate action dimensions at setup time.

        Args:
            subordinates: Dict mapping subordinate_id -> agent object
        """
        self._subordinate_action_dims = {
            sub_id: sub.action.dim_c + sub.action.dim_d
            for sub_id, sub in subordinates.items()
            if hasattr(sub, 'action') and sub.action is not None
        }

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Decompose coordinator action into subordinate actions.

        Args:
            coordinator_action: Joint action from coordinator (vector, dict, or Action object)
            info_for_subordinates: Information for subordinate agents
            coordination_messages: Messages from communication protocol
            context: Additional context (should contain "subordinates" dict)

        Returns:
            Dict mapping subordinate_id -> action
        """
        if coordinator_action is None or info_for_subordinates is None:
            return {sub_id: None for sub_id in (info_for_subordinates or {})}

        # If already per-subordinate dict, use directly
        if isinstance(coordinator_action, dict):
            return coordinator_action

        # Use stored action dimensions (registered at setup time)
        if not self._subordinate_action_dims:
            # No dimension info - distribute same action to all
            return {sub_id: coordinator_action for sub_id in info_for_subordinates}

        # Extract action vector
        if hasattr(coordinator_action, 'as_array'):
            action_vector = coordinator_action.as_array()
        elif isinstance(coordinator_action, np.ndarray):
            action_vector = coordinator_action
        else:
            # Scalar - broadcast to all subordinates
            return {sub_id: np.array([coordinator_action]) for sub_id in info_for_subordinates}

        # Decompose vector by subordinate action dimensions
        actions = {}
        offset = 0

        for sub_id in info_for_subordinates.keys():
            action_dim = self._subordinate_action_dims.get(sub_id)
            if action_dim is not None:
                if offset + action_dim <= len(action_vector):
                    actions[sub_id] = action_vector[offset:offset + action_dim]
                    offset += action_dim
                else:
                    # Not enough elements - give None
                    actions[sub_id] = None
            else:
                # No dimension info - give portion assuming equal split
                sub_ids_list = list(info_for_subordinates.keys())
                portion_size = max(1, len(action_vector) // len(sub_ids_list))
                if offset + portion_size <= len(action_vector):
                    actions[sub_id] = action_vector[offset:offset + portion_size]
                    offset += portion_size
                else:
                    actions[sub_id] = None

        return actions


class BroadcastActionProtocol(ActionProtocol):
    """Broadcasts coordinator's action to all subordinates unchanged.

    Used when the coordinator's action (e.g., pricing signal) should be
    received by every subordinate identically, rather than being
    split/decomposed across them.

    Example:
        Coordinator sets price = 0.25
        All subordinates receive 0.25
    """

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        if coordinator_action is None or info_for_subordinates is None:
            return {sub_id: None for sub_id in (info_for_subordinates or {})}
        return {sub_id: coordinator_action for sub_id in info_for_subordinates}


class VerticalProtocol(Protocol):
    """Default vertical coordination protocol for hierarchical control.

    Default behavior:
    - Communication: No communication (subordinates act based on actions only)
    - Action: Vector decomposition (split coordinator's joint action)

    This provides sensible default behavior for hierarchical coordination where
    the coordinator computes a joint action vector and distributes portions to
    each subordinate based on their action dimensions.

    Example:
        Use with a coordinator for centralized control::

            from heron.agents import CoordinatorAgent
            from heron.protocols import VerticalProtocol

            coordinator = CoordinatorAgent(
                agent_id="grid_operator",
                protocol=VerticalProtocol()
            )

            # Coordinator policy outputs joint action
            # Protocol automatically decomposes and distributes to subordinates
    """

    def __init__(
        self,
        communication_protocol: Optional[CommunicationProtocol] = None,
        action_protocol: Optional[ActionProtocol] = None
    ):
        """Initialize vertical protocol.

        Args:
            communication_protocol: Protocol for message computation (default: NoCommunication)
            action_protocol: Protocol for action coordination (default: VectorDecompositionActionProtocol)
        """
        super().__init__(
            communication_protocol=communication_protocol or NoCommunication(),
            action_protocol=action_protocol or VectorDecompositionActionProtocol()
        )

    def register_subordinates(self, subordinates: Dict[AgentID, Any]) -> None:
        """Register subordinates for action decomposition.

        Args:
            subordinates: Dict mapping subordinate_id -> agent object
        """
        self.action_protocol.register_subordinates(subordinates)

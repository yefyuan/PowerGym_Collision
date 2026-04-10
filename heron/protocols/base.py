"""Base protocol classes for coordination.

This module defines the core protocol abstractions:
- CommunicationProtocol: Defines WHAT to communicate
- ActionProtocol: Defines HOW to coordinate actions
- Protocol: Combines communication and action coordination
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from heron.utils.typing import AgentID


class CommunicationProtocol(ABC):
    neighbors: Set[AgentID]  # Neighboring agents that are reachable

    @abstractmethod
    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_infos: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        pass

    def add_neighbor(self, agent: AgentID) -> None:
        """Add a neighbor agent."""
        self.neighbors.add(agent)

    def init_neighbors(self, neighbors: List[AgentID]) -> None:
        """Initialize neighbor set."""
        self.neighbors = set(neighbors)


class ActionProtocol(ABC):
    @abstractmethod
    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Compute coordinated actions for subordinates.

        Pure function - decomposes coordinator action or computes subordinate
        actions based on coordination strategy.

        Args:
            coordinator_action: Action computed by coordinator policy (if any)
            info_for_subordinates: Information for subordinate agents
            coordination_messages: Messages computed by communication protocol

        Returns:
            Dict mapping subordinate_id -> action (or None for decentralized)
        """
        pass

    def register_subordinates(self, _subordinates: Dict[AgentID, Any]) -> None:
        """Register subordinates. Override in subclasses that need action dims."""
        pass


class Protocol(ABC):
    """Base protocol combining communication and action coordination.

    A protocol consists of:
    1. CommunicationProtocol: Defines coordination signals/messages
    2. ActionProtocol: Defines action coordination strategy

    Protocols can be:
    - Vertical (agent-owned): Parent coordinates subordinates
    - Horizontal (env-owned): Peers coordinate with each other
    """

    def __init__(
        self,
        communication_protocol: Optional[CommunicationProtocol] = None,
        action_protocol: Optional[ActionProtocol] = None
    ):
        self.communication_protocol = communication_protocol or NoCommunication()
        self.action_protocol = action_protocol or NoActionCoordination()

    def no_op(self) -> bool:
        """Check if this is a no-operation protocol."""
        return self.no_action() and self.no_communication()
    
    def no_action(self) -> bool:
        """Check if this protocol has no action coordination."""
        return isinstance(self.action_protocol, NoActionCoordination)
    
    def no_communication(self) -> bool:
        """Check if this protocol has no communication."""
        return isinstance(self.communication_protocol, NoCommunication)

    def register_subordinates(self, _subordinates: Dict[AgentID, Any]) -> None:
        """Register subordinates for action coordination. Override in subclasses."""
        pass

    def coordinate(
        self,
        coordinator_state: Any,
        coordinator_action: Optional[Any] = None,
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Dict[str, Any]], Dict[AgentID, Any]]:
        """Execute full coordination cycle.

        This is the main entry point that orchestrates:
        1. Communication: Compute and deliver messages
        2. Action: Compute and apply coordinated actions

        Args:
            coordinator_state: State of coordinating agent
            coordinator_action: Action from coordinator policy (if any)
            info_for_subordinates: Information for subordinate agents
            context: Additional context (subordinates dict, timestamp, etc.)

        Returns:
            Tuple of (messages, actions)
        """
        context = context or {}

        # Step 1: Communication coordination
        messages = self.communication_protocol.compute_coordination_messages(
            sender_state=coordinator_state,
            receiver_infos=info_for_subordinates,
            context=context
        )

        # Step 2: Action coordination
        actions = self.action_protocol.compute_action_coordination(
            coordinator_action=coordinator_action,
            info_for_subordinates=info_for_subordinates,
            coordination_messages=messages,
            context=context
        )
        return messages, actions


# =============================================================================
# NO-OP PROTOCOL COMPONENTS
# =============================================================================

class NoCommunication(CommunicationProtocol):
    """No message passing."""

    def __init__(self):
        self.neighbors = set()

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_infos: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        return {r_id: {} for r_id in receiver_infos}


class NoActionCoordination(ActionProtocol):
    """No action coordination."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        return {sub_id: None for sub_id in (info_for_subordinates or {})}


class NoProtocol(Protocol):
    """No coordination protocol."""

    def __init__(self):
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=NoActionCoordination()
        )

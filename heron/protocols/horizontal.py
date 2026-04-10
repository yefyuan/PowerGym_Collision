"""Horizontal protocols for peer-to-peer coordination.

Horizontal protocols handle Peer <-> Peer coordination.
The environment owns and runs horizontal protocols, as they require
global view of all agents.
"""

from typing import Any, Dict, List, Optional, Set

from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    NoActionCoordination,
)
from heron.utils.typing import AgentID


class StateShareCommunicationProtocol(CommunicationProtocol):
    """Shares agent states with neighbors for peer-to-peer coordination.

    This is the default communication protocol for horizontal coordination,
    enabling agents to observe their neighbors' states for decentralized
    decision making.

    Communication Strategy:
    - Each agent receives state information from its neighbors
    - Useful for consensus, cooperative control, distributed optimization

    Attributes:
        state_fields: List of state field names to share (if None, share all)
        topology: Optional adjacency structure (if None, fully connected)
    """

    def __init__(
        self,
        state_fields: Optional[List[str]] = None,
        topology: Optional[Dict[AgentID, List[AgentID]]] = None
    ):
        """Initialize state sharing protocol.

        Args:
            state_fields: State fields to share (None = share all)
            topology: Adjacency dict mapping agent_id -> list of neighbor ids
                     If None, uses fully connected topology
        """
        self.state_fields = state_fields
        self.topology = topology
        self.neighbors: Set[AgentID] = set()

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Compute state sharing messages for each agent.

        Args:
            sender_state: Not used (horizontal protocols don't have single sender)
            receiver_states: Dict of all agent states
            context: Optional context with topology info

        Returns:
            Dict mapping agent_id -> message containing neighbor states
        """
        # Get topology from context or use stored topology or default to fully connected
        context = context or {}
        topology = context.get("topology", self.topology)

        if topology is None:
            # Fully connected - each agent sees all others
            topology = {
                agent_id: [other_id for other_id in receiver_states if other_id != agent_id]
                for agent_id in receiver_states
            }

        messages = {}
        for agent_id in receiver_states:
            # Collect neighbor states
            neighbor_ids = topology.get(agent_id, [])
            neighbor_states = {}

            for neighbor_id in neighbor_ids:
                if neighbor_id in receiver_states:
                    state = receiver_states[neighbor_id]

                    # Extract requested fields or all fields
                    if self.state_fields:
                        # Filter to requested fields
                        if hasattr(state, 'local'):
                            filtered_state = {
                                k: v for k, v in state.local.items()
                                if k in self.state_fields
                            }
                        elif hasattr(state, 'get'):
                            filtered_state = {
                                k: state.get(k) for k in self.state_fields
                                if k in state
                            }
                        else:
                            filtered_state = state
                    else:
                        # Share all state
                        filtered_state = state

                    neighbor_states[neighbor_id] = filtered_state

            messages[agent_id] = {
                "type": "neighbor_states",
                "neighbors": neighbor_states
            }

        return messages


class HorizontalProtocol(Protocol):
    """Default horizontal coordination protocol for peer-to-peer coordination.

    Default behavior:
    - Communication: State sharing (agents observe neighbor states)
    - Action: No action coordination (agents act independently based on shared info)

    This provides sensible default behavior for peer-to-peer coordination where
    agents share state information with neighbors but make independent decisions.
    Useful for distributed control, consensus, and cooperative behaviors.

    Example:
        Use with an environment for peer-to-peer state sharing::

            from heron.protocols import HorizontalProtocol

            # Create protocol with specific state fields to share
            protocol = HorizontalProtocol(
                state_fields=["power", "price"],
                topology={"agent_1": ["agent_2"], "agent_2": ["agent_1"]}
            )

            # Agents receive neighbor states and act independently
    """

    def __init__(
        self,
        communication_protocol: Optional[CommunicationProtocol] = None,
        action_protocol: Optional[ActionProtocol] = None,
        state_fields: Optional[List[str]] = None,
        topology: Optional[Dict[AgentID, List[AgentID]]] = None
    ):
        """Initialize horizontal protocol.

        Args:
            communication_protocol: Protocol for message computation (default: StateShareCommunicationProtocol)
            action_protocol: Protocol for action coordination (default: NoActionCoordination)
            state_fields: State fields to share between peers
            topology: Network topology for neighbor relationships
        """
        super().__init__(
            communication_protocol=communication_protocol or StateShareCommunicationProtocol(
                state_fields=state_fields,
                topology=topology
            ),
            action_protocol=action_protocol or NoActionCoordination()
        )

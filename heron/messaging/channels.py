"""Channel management for agent communication.

This module provides channel naming conventions and registry for organizing
message-based communication between agents.
"""

from typing import Dict, List, Optional


class ChannelRegistry:
    """Registry for domain-specific channel types.

    Allows domains to document their custom channel types.

    Example:
        # In domain initialization
        ChannelRegistry.register("simulation", "Simulation results from environment")

        # When creating a channel
        channel = ChannelManager.custom_channel("simulation", env_id, agent_id)
    """
    _registered_types: Dict[str, str] = {}

    @classmethod
    def register(cls, channel_type: str, description: str = "") -> None:
        """Register a domain-specific channel type.

        Args:
            channel_type: Unique identifier for the channel type
            description: Description of the channel's purpose
        """
        cls._registered_types[channel_type] = description

    @classmethod
    def is_registered(cls, channel_type: str) -> bool:
        """Check if a channel type is registered."""
        return channel_type in cls._registered_types

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all registered channel types."""
        return cls._registered_types.copy()


class ChannelManager:
    """Channel name management for agent communication.

    This class generates channel names following a consistent naming convention.
    It's independent of the broker implementation.

    Core channel naming convention:
    - env_{env_id}__action__{upstream_id}_to_{node_id}
    - env_{env_id}__info__{node_id}_to_{upstream_id}
    - env_{env_id}__broadcast__{agent_id}
    - env_{env_id}__state_updates
    - env_{env_id}__results__{agent_id}

    Custom channels (domain-specific):
    - env_{env_id}__{channel_type}__{agent_id}

    Domains can use custom_channel() for domain-specific communication patterns.
    """

    @staticmethod
    def action_channel(upstream_id: str, node_id: str, env_id: str = "default") -> str:
        """Generate action channel name for parent->child communication.

        Args:
            upstream_id: Parent agent ID
            node_id: Child agent ID
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__action__{upstream_id}_to_{node_id}"

    @staticmethod
    def info_channel(node_id: str, upstream_id: str, env_id: str = "default") -> str:
        """Generate info channel name for child->parent communication.

        Args:
            node_id: Child agent ID
            upstream_id: Parent agent ID
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__info__{node_id}_to_{upstream_id}"

    @staticmethod
    def observation_channel(node_id: str, upstream_id: str, env_id: str = "default") -> str:
        """Generate observation channel for child->parent observation delivery.

        Used in fully async event-driven mode (Option B with async_observations=True)
        where subordinates push observations to coordinators instead of coordinators
        pulling them via direct method calls.

        Args:
            node_id: Child agent ID (observation sender)
            upstream_id: Parent agent ID (observation receiver)
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__observation__{node_id}_to_{upstream_id}"

    @staticmethod
    def broadcast_channel(agent_id: str, env_id: str = "default") -> str:
        """Generate broadcast channel name for agent broadcasts.

        Args:
            agent_id: Broadcasting agent ID
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__broadcast__{agent_id}"

    @staticmethod
    def state_update_channel(env_id: str = "default") -> str:
        """Generate state update channel for agent->environment state updates.

        Args:
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__state_updates"

    @staticmethod
    def result_channel(env_id: str, agent_id: str) -> str:
        """Generate result channel for environment->agent results.

        Args:
            env_id: Environment ID
            agent_id: Agent ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__results__{agent_id}"

    @staticmethod
    def custom_channel(channel_type: str, env_id: str, agent_id: str) -> str:
        """Generate a custom channel for domain-specific communication.

        This is a generic method for domains to create their own channel types.
        The channel_type should be registered via ChannelRegistry for documentation.

        Args:
            channel_type: Domain-specific channel type identifier
            env_id: Environment ID
            agent_id: Agent ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__{channel_type}__{agent_id}"

    @staticmethod
    def agent_channels(
        agent_id: str,
        upstream_id: Optional[str],
        subordinate_ids: List[str],
        env_id: str = "default",
        async_observations: bool = False,
    ) -> Dict[str, List[str]]:
        """Get all channels for an agent (subscribe and publish).

        Args:
            agent_id: The agent's ID
            upstream_id: Parent agent ID (if any)
            subordinate_ids: List of subordinate agent IDs
            env_id: Environment identifier
            async_observations: If True, include observation channels for
                fully async mode where subordinates push observations

        Returns:
            Dict with 'subscribe' and 'publish' channel lists
        """
        subscribe_channels = []
        publish_channels = []

        # Subscribe to actions from parent
        if upstream_id:
            subscribe_channels.append(
                ChannelManager.action_channel(upstream_id, agent_id, env_id)
            )

        # Subscribe to info from subordinates
        for sub_id in subordinate_ids:
            subscribe_channels.append(
                ChannelManager.info_channel(sub_id, agent_id, env_id)
            )

        # Subscribe to observations from subordinates (async mode)
        if async_observations:
            for sub_id in subordinate_ids:
                subscribe_channels.append(
                    ChannelManager.observation_channel(sub_id, agent_id, env_id)
                )

        # Publish info to parent
        if upstream_id:
            publish_channels.append(
                ChannelManager.info_channel(agent_id, upstream_id, env_id)
            )

        # Publish observations to parent (async mode)
        if async_observations and upstream_id:
            publish_channels.append(
                ChannelManager.observation_channel(agent_id, upstream_id, env_id)
            )

        # Publish actions to subordinates
        for sub_id in subordinate_ids:
            publish_channels.append(
                ChannelManager.action_channel(agent_id, sub_id, env_id)
            )

        return {
            'subscribe': subscribe_channels,
            'publish': publish_channels
        }

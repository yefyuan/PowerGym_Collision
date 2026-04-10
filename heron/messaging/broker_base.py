"""Abstract message broker interface.

This module defines the MessageBroker abstract base class that all broker
implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from heron.agents.base import Agent

from heron.messaging.messages import Message
from heron.utils.typing import AgentID


class MessageBroker(ABC):
    """Abstract message broker interface.

    This interface defines the contract for any message broker implementation.
    Implementations can use Kafka, RabbitMQ, Redis, in-memory queues, etc.

    The broker provides a pub/sub model where agents publish messages to channels
    and consume messages from channels based on recipient filtering.
    """

    @staticmethod
    def init(config: Optional[Dict[str, Any]] = None) -> "MessageBroker":
        """Initialize the message broker with optional configuration.

        Args:
            config: Optional dictionary of configuration parameters
        """
        from heron.messaging.in_memory_broker import InMemoryBroker
        if not config or "type" not in config:
            return InMemoryBroker()  # Default to in-memory broker for simplicity

        if config["type"] == "in_memory":
            return InMemoryBroker()
        else:
            raise ValueError(f"Unsupported message broker type: {config['type']}")

    def attach(self, agents: Dict[AgentID, "Agent"]) -> None:
        """Attach the broker to a list of agents.

        This allows the broker to manage subscriptions and message routing for these agents.

        Args:
            agents: Dict of Agent instances to attach to the broker
        """
        for agent in agents.values():
            agent.set_message_broker(self)

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def create_channel(self, channel_name: str) -> None:
        """Create a new message channel.

        Channels are named communication pathways. Depending on the implementation,
        these might map to Kafka topics, Redis pub/sub channels, RabbitMQ queues, etc.

        Args:
            channel_name: Unique identifier for the channel
        """
        pass

    @abstractmethod
    def publish(self, channel: str, message: Message) -> None:
        """Publish a message to a channel.

        Args:
            channel: Channel name to publish to
            message: Message to publish
        """
        pass

    @abstractmethod
    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[Message]:
        """Consume messages from a channel.

        Args:
            channel: Channel name to consume from
            recipient_id: Filter messages for this recipient
            env_id: Filter messages for this environment
            clear: If True, remove consumed messages from channel (default: True)

        Returns:
            List of messages matching the filters
        """
        pass

    @abstractmethod
    def subscribe(self, channel: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to a channel with a callback.

        When messages are published to the channel, the callback will be invoked.
        This is for asynchronous/reactive message handling.

        Args:
            channel: Channel name to subscribe to
            callback: Function to call when messages arrive
        """
        pass

    @abstractmethod
    def clear_environment(self, env_id: str) -> None:
        """Clear all messages for a specific environment.

        Useful for resetting environments in vectorized rollouts.

        Args:
            env_id: Environment identifier
        """
        pass

    @abstractmethod
    def get_environment_channels(self, env_id: str) -> List[str]:
        """Get all channels associated with an environment.

        Args:
            env_id: Environment identifier

        Returns:
            List of channel names for this environment
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the broker and clean up resources.

        Should be called when done using the broker to properly release
        connections, threads, etc.
        """
        pass

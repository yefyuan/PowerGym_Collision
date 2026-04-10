"""In-memory message broker implementation.

This implementation stores all messages in memory using Python dictionaries.
It's fast, simple, and perfect for testing, development, and single-process simulations.
"""

import logging
from collections import defaultdict
from threading import Lock
from typing import Callable, Dict, List, Optional

from heron.messaging.broker_base import MessageBroker
from heron.messaging.messages import Message

logger = logging.getLogger(__name__)


class InMemoryBroker(MessageBroker):
    """In-memory message broker for testing and single-process environments.

    This implementation stores all messages in memory using Python dictionaries.
    It's thread-safe, fast, and perfect for:
    - Unit testing
    - Development
    - Single-process simulations
    - Vectorized environments with shared broker

    Not suitable for:
    - Distributed systems (multi-process/multi-machine)
    - Persistence requirements
    - Production deployments requiring fault tolerance

    Attributes:
        channels: Dict mapping channel names to message lists
        subscribers: Dict mapping channel names to callback lists
        lock: Thread lock for thread-safe operations
    """

    def __init__(self):
        """Initialize in-memory broker."""
        self.channels: Dict[str, List[Message]] = defaultdict(list)
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = defaultdict(list)
        self.lock = Lock()

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset the broker state.

        Clears all messages from all channels while preserving the channel structure
        and subscriber connections. Attached agents remain connected to the broker.

        Args:
            seed: Optional random seed (unused, provided for interface compatibility)
            **kwargs: Additional keyword arguments (unused)
        """
        with self.lock:
            # Clear all messages from all channels while keeping the channel structure
            for channel in self.channels:
                self.channels[channel] = []
            # Note: We intentionally keep subscribers intact so agents remain connected

    def create_channel(self, channel_name: str) -> None:
        """Create a channel.

        For in-memory broker, channels are auto-created on first use,
        so this is essentially a no-op. Provided for interface compatibility.

        Args:
            channel_name: Channel name
        """
        with self.lock:
            if channel_name not in self.channels:
                self.channels[channel_name] = []
                self.subscribers[channel_name] = []

    def publish(self, channel: str, message: Message) -> None:
        """Publish message to channel and notify subscribers.

        Args:
            channel: Channel name
            message: Message to publish
        """
        with self.lock:
            # Auto-create channel if needed
            if channel not in self.channels:
                self.channels[channel] = []

            # Append message
            self.channels[channel].append(message)

            # Notify subscribers (outside lock to avoid deadlocks)
            callbacks = self.subscribers.get(channel, []).copy()

        # Call subscribers outside lock
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                # Don't let subscriber errors crash the broker
                logger.warning(f"Error in subscriber callback for channel {channel}: {e}")

    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[Message]:
        """Consume messages matching recipient and environment.

        Args:
            channel: Channel name
            recipient_id: Filter for this recipient
            env_id: Filter for this environment
            clear: If True, remove consumed messages

        Returns:
            List of matching messages
        """
        with self.lock:
            if channel not in self.channels:
                return []

            # Filter messages for this recipient and environment
            messages = [
                msg for msg in self.channels[channel]
                if msg.recipient_id == recipient_id and msg.env_id == env_id
            ]

            if clear and messages:
                # Remove consumed messages
                self.channels[channel] = [
                    msg for msg in self.channels[channel]
                    if not (msg.recipient_id == recipient_id and msg.env_id == env_id)
                ]

            return messages

    def subscribe(self, channel: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to channel with callback.

        The callback will be invoked whenever a message is published to the channel.

        Args:
            channel: Channel name
            callback: Function to call on new messages
        """
        with self.lock:
            self.subscribers[channel].append(callback)

    def clear_environment(self, env_id: str) -> None:
        """Clear all messages for an environment.

        This removes all messages from all channels that belong to the specified
        environment. Useful for resetting vectorized environments.

        Args:
            env_id: Environment identifier
        """
        with self.lock:
            for channel in list(self.channels.keys()):
                if channel.startswith(f"env_{env_id}__"):
                    self.channels[channel] = []

    def get_environment_channels(self, env_id: str) -> List[str]:
        """Get all channels for an environment.

        Args:
            env_id: Environment identifier

        Returns:
            List of channel names
        """
        with self.lock:
            return [
                channel for channel in self.channels.keys()
                if channel.startswith(f"env_{env_id}__")
            ]

    def close(self) -> None:
        """Close broker and clear all data.

        For in-memory broker, this simply clears all messages and subscribers.
        """
        with self.lock:
            self.channels.clear()
            self.subscribers.clear()

    def __repr__(self) -> str:
        """String representation."""
        num_channels = len(self.channels)
        num_messages = sum(len(msgs) for msgs in self.channels.values())
        return f"InMemoryBroker(channels={num_channels}, messages={num_messages})"

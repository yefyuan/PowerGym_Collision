"""Message types and structures for agent communication.

This module defines the core message format and type system used across
all broker implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class MessageType(Enum):
    """Generic message types for agent communication.

    Domains can register additional message types via MessageTypeRegistry.
    """
    ACTION = "action"
    INFO = "info"
    BROADCAST = "broadcast"
    STATE_UPDATE = "state_update"
    RESULT = "result"  # Generic result message
    CUSTOM = "custom"  # For domain-specific message types


class MessageTypeRegistry:
    """Registry for domain-specific message types.

    Allows domains to register custom message type identifiers that can be
    used in the payload's 'custom_type' field when MessageType.CUSTOM is used.

    Example:
        # In domain initialization
        MessageTypeRegistry.register("simulation_result")
        MessageTypeRegistry.register("sensor_update")

        # When creating a message
        msg = Message(
            ...,
            message_type=MessageType.CUSTOM,
            payload={"custom_type": "simulation_result", "data": {...}}
        )
    """
    _registered_types: Dict[str, str] = {}

    @classmethod
    def register(cls, type_name: str, description: str = "") -> None:
        """Register a domain-specific message type.

        Args:
            type_name: Unique identifier for the custom type
            description: Optional description of the message type
        """
        cls._registered_types[type_name] = description

    @classmethod
    def is_registered(cls, type_name: str) -> bool:
        """Check if a custom type is registered."""
        return type_name in cls._registered_types

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all registered custom types."""
        return cls._registered_types.copy()


@dataclass
class Message:
    """Generic message structure for agent communication.

    This message format is implementation-agnostic and works with any broker backend.

    Attributes:
        env_id: Environment/rollout identifier for multi-environment isolation
        sender_id: Sender agent ID
        recipient_id: Recipient agent ID (or "broadcast" for broadcasts)
        timestamp: Message timestamp
        message_type: Type of message (action, info, etc.)
        payload: Arbitrary message data as dict
    """
    env_id: str
    sender_id: str
    recipient_id: str
    timestamp: float
    message_type: MessageType
    payload: Dict[str, Any]

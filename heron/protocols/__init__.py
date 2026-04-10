"""Coordination protocols for HERON multi-agent systems.

This module provides protocol implementations for agent coordination:

Base Protocols:
- Protocol: Abstract base for all protocols
- NoProtocol: No-op protocol (agents act independently)
- CommunicationProtocol: Abstract base for communication protocols
- ActionProtocol: Abstract base for action coordination protocols

Vertical Protocols (hierarchical coordination):
- VerticalProtocol: Default vertical protocol for parent->child coordination

Horizontal Protocols (peer coordination):
- HorizontalProtocol: Default horizontal protocol for peer-to-peer coordination
"""

from heron.protocols.base import (
    Protocol,
    NoProtocol,
    CommunicationProtocol,
    ActionProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.protocols.vertical import VerticalProtocol
from heron.protocols.horizontal import HorizontalProtocol

__all__ = [
    # Base
    "Protocol",
    "NoProtocol",
    "CommunicationProtocol",
    "ActionProtocol",
    "NoCommunication",
    "NoActionCoordination",
    # Vertical
    "VerticalProtocol",
    # Horizontal
    "HorizontalProtocol",
]

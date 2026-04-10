"""Agent classes for HERON hierarchical multi-agent systems.

This module provides the agent hierarchy:
- Agent: Abstract base class
- FieldAgent: Field-level (L1) agents managing individual units
- CoordinatorAgent: Coordinator-level (L2) agents managing groups
- SystemAgent: System-level (L3) agents for system-wide coordination
- Proxy: Proxy for distributed communication
"""

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.agents.proxy_agent import Proxy
from heron.agents.constants import (
    # Agent Hierarchy Levels
    PROXY_LEVEL,
    FIELD_LEVEL,
    COORDINATOR_LEVEL,
    SYSTEM_LEVEL,
    # Special Agent IDs
    PROXY_AGENT_ID,
    SYSTEM_AGENT_ID,
)

__all__ = [
    "Agent",
    "FieldAgent",
    "CoordinatorAgent",
    "SystemAgent",
    "Proxy",
    # Constants
    "PROXY_LEVEL",
    "FIELD_LEVEL",
    "COORDINATOR_LEVEL",
    "SYSTEM_LEVEL",
    "PROXY_AGENT_ID",
    "SYSTEM_AGENT_ID",
]

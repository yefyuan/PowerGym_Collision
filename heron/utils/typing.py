"""Type aliases for the HERON framework."""

from typing import Any, Dict


# Agent ID type
AgentID = str
MultiAgentDict = Dict[AgentID, Any]


def float_if_not_none(x: Any) -> Any:
    """Convert to float if not None."""
    return None if x is None else float(x)

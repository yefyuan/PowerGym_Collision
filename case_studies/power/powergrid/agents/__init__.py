"""Power grid agent module.

This module provides power-grid specific agent implementations
built on the HERON agent framework, following the grid_agent style:
- Direct constructor parameters (no config dataclasses)
- Normalized [-1, 1] action spaces
- Explicit set_state() parameters
- apply_action() that calls set_state()
"""

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.features.metrics import CostSafetyMetrics
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.agents.transformer import Transformer
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.agents.grid_system_agent import GridSystemAgent

# Import Proxy from heron (no custom implementation needed)
from heron.agents.proxy_agent import Proxy, PROXY_LEVEL

# Power grid uses "power_flow" as the channel type for power flow results
POWER_FLOW_CHANNEL_TYPE = "power_flow"

__all__ = [
    # Device agents
    "DeviceAgent",
    "CostSafetyMetrics",
    "Generator",
    "ESS",
    "Transformer",
    # Coordinator agents
    "PowerGridAgent",
    "GridSystemAgent",
    # From heron
    "Proxy",
    "PROXY_LEVEL",
    # Constants
    "POWER_FLOW_CHANNEL_TYPE",
]

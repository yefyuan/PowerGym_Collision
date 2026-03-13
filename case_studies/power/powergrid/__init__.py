"""PowerGrid: Multi-agent power system simulation using HERON.

This module provides power system components following Heron's agent hierarchy:
- Agents: Generator, ESS (Energy Storage), Transformer, DeviceAgent
- Grid Coordination: PowerGridAgent, GridSystemAgent
- Environments: HierarchicalMicrogridEnv, EnvState
- Features: Electrical, Storage, Network state providers

Example:
    Create agents manually and pass to environment::

        from powergrid import Generator, ESS, PowerGridAgent, HierarchicalMicrogridEnv
        from heron.agents.system_agent import SystemAgent

        # Create device agents
        devices = {
            "gen_1": Generator(agent_id="gen_1", ...),
            "ess_1": ESS(agent_id="ess_1", ...),
        }

        # Create coordinator
        grid = PowerGridAgent(agent_id="grid_1", subordinates=devices)

        # Create system agent
        system = SystemAgent(agent_id="system", subordinates={"grid_1": grid})

        # Create environment
        env = HierarchicalMicrogridEnv(system_agent=system, dataset_path="...")
"""

__version__ = "0.1.0"

# Device Agents
from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.agents.transformer import Transformer
from powergrid.core.features.metrics import CostSafetyMetrics

# Grid Agents
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.agents.grid_system_agent import GridSystemAgent
from heron.agents.proxy_agent import Proxy, PROXY_LEVEL
from powergrid.agents import POWER_FLOW_CHANNEL_TYPE

# Environments
from powergrid.envs.common import EnvState
from powergrid.envs.hierarchical_microgrid_env import HierarchicalMicrogridEnv

# Features
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.storage import StorageBlock
from powergrid.core.features.network import BusVoltages, LineFlows, NetworkMetrics
from powergrid.core.features.status import StatusBlock
from powergrid.core.features.power_limits import PowerLimits

# Utilities
from powergrid.utils.phase import PhaseModel, PhaseSpec
from powergrid.utils.safety import SafetySpec, total_safety
from powergrid.utils.cost import quadratic_cost, energy_cost

__all__ = [
    # Version
    "__version__",
    # Device Agents
    "DeviceAgent",
    "CostSafetyMetrics",
    "Generator",
    "ESS",
    "Transformer",
    # Grid Agents
    "PowerGridAgent",
    "GridSystemAgent",
    "Proxy",
    "PROXY_LEVEL",
    "POWER_FLOW_CHANNEL_TYPE",
    # Environments
    "EnvState",
    "HierarchicalMicrogridEnv",
    # Features
    "ElectricalBasePh",
    "StorageBlock",
    "BusVoltages",
    "LineFlows",
    "NetworkMetrics",
    "StatusBlock",
    "PowerLimits",
    # Utilities
    "PhaseModel",
    "PhaseSpec",
    "SafetySpec",
    "total_safety",
    "quadratic_cost",
    "energy_cost",
]

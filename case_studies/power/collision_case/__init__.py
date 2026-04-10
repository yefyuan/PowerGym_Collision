"""Collision Detection Case Study for Multi-Agent Microgrids.

This case study recreates the collision detection experiments from the original platform,
with added support for asynchronous updates using HERON's scheduling capabilities.

Key Features:
- 3 microgrids (MG1, MG2, MG3) connected to IEEE 34-bus main grid
- Each microgrid based on IEEE 13-bus topology
- Collision detection: overvoltage, undervoltage, line overloading
- Shared vs independent reward schemes
- Asynchronous update capabilities with configurable delays
- Comprehensive logging and analysis tools

Example:
    >>> from collision_case import CollisionEnv, create_collision_system
    >>> system = create_collision_system()
    >>> env = CollisionEnv(system_agent=system, dataset_path="data.pkl")
    >>> obs, info = env.reset()
"""

from collision_case.collision_env import CollisionEnv
from collision_case.collision_features import CollisionMetrics
from collision_case.collision_rllib_env import CollisionRLlibMultiAgentEnv
from collision_case.system_builder import create_collision_system

__all__ = [
    "CollisionEnv",
    "CollisionMetrics",
    "CollisionRLlibMultiAgentEnv",
    "create_collision_system",
]

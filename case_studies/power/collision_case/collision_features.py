"""Collision detection features for multi-agent microgrids.

This module provides features for tracking safety violations (collisions) including:
- Overvoltage violations
- Undervoltage violations  
- Line overloading violations
- Device over-rating violations
"""

from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature


@dataclass(slots=True)
class CollisionMetrics(Feature):
    """Collision detection metrics for a microgrid.

    Tracks various safety violations that constitute "collisions":
    - Overvoltage: Sum of (voltage - 1.05) for all buses exceeding upper limit
    - Undervoltage: Sum of (0.95 - voltage) for all buses below lower limit
    - Overloading: Sum of (loading% - 100)/100 for all lines exceeding capacity
    - Device over-rating: Apparent power exceeding device ratings
    
    Visibility: coordinator and upper levels can observe these metrics
    """

    visibility: ClassVar[Sequence[str]] = ["owner", "upper_level", "system"]

    # Voltage violations
    overvoltage: float = 0.0  # Sum of overvoltage violations
    undervoltage: float = 0.0  # Sum of undervoltage violations
    
    # Line violations
    overloading: float = 0.0  # Sum of line overloading violations
    
    # Device violations
    device_overrating: float = 0.0  # Sum of device over-rating violations
    
    # Power factor violations
    pf_violations: float = 0.0  # Sum of power factor violations
    
    # SOC violations (for energy storage)
    soc_violations: float = 0.0  # Sum of SOC limit violations
    
    # Total safety metric (sum of all violations)
    safety_total: float = 0.0
    
    # Binary collision flag (1 if any violation, 0 otherwise)
    collision_flag: float = 0.0

    def compute_total(self) -> None:
        """Compute total safety metric from individual components."""
        self.safety_total = (
            self.overvoltage
            + self.undervoltage
            + self.overloading
            + self.device_overrating
            + self.pf_violations
            + self.soc_violations
        )
        self.collision_flag = 1.0 if self.safety_total > 0.0 else 0.0

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.overvoltage = 0.0
        self.undervoltage = 0.0
        self.overloading = 0.0
        self.device_overrating = 0.0
        self.pf_violations = 0.0
        self.soc_violations = 0.0
        self.safety_total = 0.0
        self.collision_flag = 0.0


@dataclass(slots=True)
class SharedRewardConfig(Feature):
    """Configuration for shared vs independent reward schemes.

    Visibility: system level configuration
    """

    visibility: ClassVar[Sequence[str]] = ["system"]

    # Whether to share rewards across all agents
    share_reward: bool = True
    
    # Penalty multiplier for safety violations
    penalty: float = 10.0
    
    # Whether currently in training mode
    train: bool = True


@dataclass(slots=True)
class AsyncUpdateConfig(Feature):
    """Configuration for asynchronous update experiments.

    Supports testing collision impact under:
    - Different update frequencies per agent
    - Random delays in updates
    - Stale observations

    Visibility: system level configuration
    """

    visibility: ClassVar[Sequence[str]] = ["system"]

    # Enable asynchronous updates
    enable_async: bool = False
    
    # Update delay range in seconds (min, max)
    delay_range_s: tuple = (0.0, 0.0)
    
    # Observation staleness in timesteps
    obs_staleness: int = 0
    
    # Random seed for delay generation
    seed: int = 42

"""Tick configuration for event-driven scheduling.

This module provides ScheduleConfig dataclass for centralized tick timing
configuration with support for randomization (jitter) in testing mode.

Default tick config constants for each agent type:

- ``DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG``       — 1.0s tick interval
- ``DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG`` — 60.0s tick interval
- ``DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG``      — 300.0s tick interval
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class JitterType(Enum):
    """Types of jitter distribution for timing variability."""

    NONE = "none"  # No jitter (deterministic)
    UNIFORM = "uniform"  # Uniform distribution: base +/- jitter_range
    GAUSSIAN = "gaussian"  # Gaussian distribution: base as mean, jitter_range as std


@dataclass
class ScheduleConfig:
    """Configuration for agent timing in event-driven mode.

    Encapsulates tick interval and delay parameters with optional jitter
    for realistic distributed system simulation during testing.

    Attributes:
        tick_interval: Base time between agent steps (seconds)
        obs_delay: Base observation latency (seconds)
        act_delay: Base action effect delay (seconds)
        msg_delay: Base message delivery delay (seconds)
        reward_delay: Base delay for reward aggregation (seconds). Used by parent
            agents to wait for subordinate rewards before computing their own.
        jitter_type: Type of randomization to apply (NONE, UNIFORM, GAUSSIAN)
        jitter_ratio: Jitter magnitude as fraction of base value (e.g., 0.1 = +/- 10%)
        min_delay: Minimum allowed delay after jitter (clamps negative values)
        rng: Optional RNG instance for reproducibility

    Example:
        # Deterministic config (for training)
        config = ScheduleConfig(tick_interval=1.0, obs_delay=0.1)

        # With jitter (for testing)
        config = ScheduleConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.1,
            act_delay=0.2,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,  # 10% std deviation
            seed=42,
        )
    """

    # Base timing parameters
    tick_interval: float = 1.0
    obs_delay: float = 0.0
    act_delay: float = 0.0
    msg_delay: float = 0.0
    reward_delay: float = 0.0

    # Jitter configuration (only active in testing/event-driven mode)
    jitter_type: JitterType = JitterType.NONE
    jitter_ratio: float = 0.1  # 10% by default
    min_delay: float = 0.0  # Floor for delays (clamp negatives)

    # RNG for reproducibility (None = use global numpy RNG)
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate config values."""
        if self.tick_interval <= 0:
            raise ValueError("tick_interval must be positive")
        if self.jitter_ratio < 0:
            raise ValueError("jitter_ratio must be non-negative")
        if self.min_delay < 0:
            raise ValueError("min_delay must be non-negative")

    def _apply_jitter(self, base_value: float, is_interval: bool = False) -> float:
        """Apply jitter to a base timing value.

        Args:
            base_value: The base timing value
            is_interval: If True, use minimum floor of 0.001 (for tick_interval)

        Returns:
            Jittered value, clamped to appropriate minimum
        """
        if self.jitter_type == JitterType.NONE or base_value == 0.0:
            return base_value

        if self.rng is None:
            self.rng = np.random.default_rng()
        rng = self.rng
        jitter_magnitude = base_value * self.jitter_ratio

        if self.jitter_type == JitterType.UNIFORM:
            jitter = rng.uniform(-jitter_magnitude, jitter_magnitude)
        elif self.jitter_type == JitterType.GAUSSIAN:
            jitter = rng.normal(0, jitter_magnitude)
        else:
            jitter = 0.0

        result = base_value + jitter

        # Apply floor: 0.001 for intervals, min_delay for delays
        floor = 0.001 if is_interval else self.min_delay
        return max(result, floor)

    def get_tick_interval(self) -> float:
        """Get (possibly jittered) tick interval.

        Returns:
            Tick interval, always >= 0.001
        """
        return self._apply_jitter(self.tick_interval, is_interval=True)

    def get_obs_delay(self) -> float:
        """Get (possibly jittered) observation delay.

        Returns:
            Observation delay, clamped to min_delay
        """
        return self._apply_jitter(self.obs_delay)

    def get_act_delay(self) -> float:
        """Get (possibly jittered) action delay.

        Returns:
            Action delay, clamped to min_delay
        """
        return self._apply_jitter(self.act_delay)

    def get_msg_delay(self) -> float:
        """Get (possibly jittered) message delay.

        Returns:
            Message delay, clamped to min_delay
        """
        return self._apply_jitter(self.msg_delay)

    def get_reward_delay(self) -> float:
        """Get (possibly jittered) reward aggregation delay.

        Returns:
            Reward delay, clamped to min_delay
        """
        return self._apply_jitter(self.reward_delay)

    def seed(self, seed: int) -> None:
        """Set RNG seed for reproducibility.

        Args:
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)

    @classmethod
    def deterministic(
        cls,
        tick_interval: float = 1.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
        reward_delay: float = 0.0,
    ) -> "ScheduleConfig":
        """Create deterministic config (no jitter) - for training.

        Args:
            tick_interval: Time between agent steps
            obs_delay: Observation latency
            act_delay: Action effect delay
            msg_delay: Message delivery delay
            reward_delay: Reward aggregation delay

        Returns:
            ScheduleConfig with jitter disabled
        """
        return cls(
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
            reward_delay=reward_delay,
            jitter_type=JitterType.NONE,
        )

    @classmethod
    def with_jitter(
        cls,
        tick_interval: float = 1.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
        reward_delay: float = 0.0,
        jitter_type: JitterType = JitterType.GAUSSIAN,
        jitter_ratio: float = 0.1,
        min_delay: float = 0.0,
        seed: Optional[int] = None,
    ) -> "ScheduleConfig":
        """Create config with jitter enabled - for testing.

        Args:
            tick_interval: Base time between agent steps
            obs_delay: Base observation latency
            act_delay: Base action effect delay
            msg_delay: Base message delivery delay
            reward_delay: Base reward aggregation delay
            jitter_type: Distribution type for jitter
            jitter_ratio: Jitter magnitude as fraction of base
            min_delay: Minimum allowed delay after jitter
            seed: Optional RNG seed

        Returns:
            ScheduleConfig with jitter enabled
        """
        rng = np.random.default_rng(seed) if seed is not None else None
        return cls(
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
            reward_delay=reward_delay,
            jitter_type=jitter_type,
            jitter_ratio=jitter_ratio,
            min_delay=min_delay,
            rng=rng,
        )


# =========================================================================
# Default tick configs per agent type
# =========================================================================
# The tick_interval values mirror the agent hierarchy's natural cadence:
#   Field (1s) < Coordinator (60s) < System (300s)

DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG = ScheduleConfig.deterministic(tick_interval=1.0)
DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG = ScheduleConfig.deterministic(tick_interval=60.0)
DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG = ScheduleConfig.deterministic(tick_interval=300.0)

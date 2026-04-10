"""Event-driven scheduling for HERON.

This module provides discrete-event simulation capabilities with:
- Priority-queue based event scheduling
- Configurable latency modeling
- Heterogeneous agent tick rates
- Schedule configuration with optional jitter for testing
- Event analysis and episode result tracking
"""

from heron.scheduling.event import Event, EventType, EVENT_TYPE_FROM_STRING
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.schedule_config import (
    DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG,
    DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG,
    DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG,
    JitterType,
    ScheduleConfig,
)
from heron.scheduling.analysis import EpisodeAnalyzer, EpisodeStats, EventAnalysis

__all__ = [
    "DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG",
    "DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG",
    "DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG",
    "Event",
    "EventType",
    "EventScheduler",
    "JitterType",
    "ScheduleConfig",
    "EVENT_TYPE_FROM_STRING",
    "EpisodeAnalyzer",
    "EpisodeStats",
    "EventAnalysis",
]

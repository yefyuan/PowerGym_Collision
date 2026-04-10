"""Event definitions for discrete-event scheduling.

Events represent scheduled actions in the simulation timeline.
The EventScheduler processes events in timestamp order.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Types of events in the simulation.

    AGENT_TICK: Regular agent step (observe -> act cycle)
    ACTION_EFFECT: Delayed action taking effect on environment
    MESSAGE_DELIVERY: Delayed message arriving at recipient
    OBSERVATION_READY: Delayed observation becoming available
    ENV_UPDATE: Environment state update
    CUSTOM: Domain-specific events
    """
    AGENT_TICK = "agent_tick"
    ACTION_EFFECT = "action_effect"
    MESSAGE_DELIVERY = "message_delivery"
    OBSERVATION_READY = "observation_ready"
    ENV_UPDATE = "env_update"
    SIMULATION = "simulation"
    CUSTOM = "custom"


@dataclass(order=True)
class Event:
    """A scheduled event in the simulation.

    Events are ordered by (timestamp, priority, sequence) for deterministic
    processing. Lower priority values are processed first at same timestamp.

    Attributes:
        timestamp: When this event should be processed (simulation time)
        priority: Tie-breaker for same-timestamp events (lower = first)
        sequence: Auto-assigned sequence number for stable sorting
        event_type: Type of event (not used in ordering)
        agent_id: Target agent ID (not used in ordering)
        payload: Event-specific data (not used in ordering)
    """
    timestamp: float
    priority: int = field(default=0, compare=True)
    sequence: int = field(default=0, compare=True)

    # Fields excluded from comparison (not used in ordering)
    event_type: EventType = field(default=EventType.AGENT_TICK, compare=False)
    agent_id: Optional[str] = field(default=None, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return (
            f"Event(t={self.timestamp:.3f}, type={self.event_type.value}, "
            f"agent={self.agent_id}, prio={self.priority})"
        )

EVENT_TYPE_FROM_STRING = {e.value: e for e in EventType}
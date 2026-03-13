"""Event analysis and episode result tracking for event-driven execution.

This module provides tools for analyzing events during simulation and
collecting episode-level statistics, with focus on state and action tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from heron.scheduling.event import Event, EventType


@dataclass
class EventAnalysis:
    """Analysis result for a single event.

    Focuses on tracking agent observations, states, and action results.

    Attributes:
        event_type: Type of the event
        timestamp: When the event occurred
        agent_id: Agent involved in the event
        message_type: Type of message (e.g., 'get_obs_response', 'set_state')
        data_summary: Summary of observation/state/action data
        metadata: Additional analysis metadata
    """
    event_type: EventType
    timestamp: float
    agent_id: Optional[str] = None
    message_type: Optional[str] = None
    data_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        msg_info = f", msg={self.message_type}" if self.message_type else ""
        return (
            f"EventAnalysis(type={self.event_type.value}, "
            f"t={self.timestamp:.3f}, agent={self.agent_id}{msg_info})"
        )


class EpisodeAnalyzer:
    """Analyzes events during event-driven simulation.

    Focuses on tracking agent observations, global/local states,
    and action execution results.

    Tracks:
    - Observation requests and responses (get_obs_response)
    - Global state requests and updates (get_global_state_response, set_state)
    - Local state requests and updates (get_local_state_response, set_state)
    - Action execution results (set_tick_result)

    Example:
        analyzer = EpisodeAnalyzer()
        result = env.run_event_driven(analyzer, t_end=100.0)
        print(f"Observations: {result.observation_count}")
        print(f"State updates: {result.state_update_count}")
    """

    def __init__(self, verbose: bool = False, track_data: bool = False):
        """Initialize the event analyzer.

        Args:
            verbose: If True, print event details during analysis
            track_data: If True, store full data in analysis (can be memory intensive)
        """
        self.verbose = verbose
        self.track_data = track_data

        # Track counts of specific message types
        self.observation_count = 0
        self.global_state_count = 0
        self.local_state_count = 0
        self.state_update_count = 0
        self.action_result_count = 0

        # Track reward history per agent: {agent_id: [(timestamp, reward), ...]}
        self.reward_history: Dict[str, List[tuple]] = {}

    def parse_event(self, event: Event) -> EventAnalysis:
        """Parse and analyze a single event.

        Extracts information about observations, states, and action results
        from MESSAGE_DELIVERY events.

        Args:
            event: The event to analyze

        Returns:
            EventAnalysis object containing analysis results
        """
        message_type = None
        data_summary = {}

        # Focus on MESSAGE_DELIVERY events that contain state/obs/action data
        if event.event_type == EventType.MESSAGE_DELIVERY:
            payload = event.payload
            message_content = payload.get("message", {})

            # Check for observation responses
            if "get_obs_response" in message_content:
                message_type = "get_obs_response"
                self.observation_count += 1
                obs_data = message_content["get_obs_response"].get("body", {})
                data_summary = self._summarize_observation(obs_data)

            # Check for global state responses
            elif "get_global_state_response" in message_content:
                message_type = "get_global_state_response"
                self.global_state_count += 1
                state_data = message_content["get_global_state_response"].get("body", {})
                data_summary = self._summarize_state(state_data, "global")

            # Check for local state responses
            elif "get_local_state_response" in message_content:
                message_type = "get_local_state_response"
                self.local_state_count += 1
                state_data = message_content["get_local_state_response"].get("body", {})
                data_summary = self._summarize_state(state_data, "local")

            # Check for state update completion
            elif "set_state_completion" in message_content:
                message_type = "set_state_completion"
                self.state_update_count += 1
                data_summary = {"status": message_content.get("set_state_completion")}

            # Check for tick result messages
            elif "set_tick_result" in message_content:
                message_type = "set_tick_result"
                self.action_result_count += 1
                result_data = message_content.get("body", {})
                data_summary = self._summarize_action_result(result_data)

                # Track reward history per agent
                sender_id = payload.get("sender")
                if sender_id and isinstance(result_data, dict) and "reward" in result_data:
                    if sender_id not in self.reward_history:
                        self.reward_history[sender_id] = []
                    self.reward_history[sender_id].append((event.timestamp, result_data["reward"]))

        # Create analysis
        analysis = EventAnalysis(
            event_type=event.event_type,
            timestamp=event.timestamp,
            agent_id=event.agent_id,
            message_type=message_type,
            data_summary=data_summary,
        )

        if self.verbose and message_type:
            print(f"[EpisodeAnalyzer] {analysis}")
            if data_summary:
                print(f"  Data: {data_summary}")

        return analysis

    def _summarize_observation(self, obs_data: Any) -> Dict[str, Any]:
        """Summarize observation data.

        Args:
            obs_data: Observation data from message

        Returns:
            Summary dict
        """
        if not self.track_data:
            if isinstance(obs_data, dict):
                return {"keys": list(obs_data.keys()), "num_keys": len(obs_data)}
            else:
                return {"type": type(obs_data).__name__}
        else:
            return {"data": obs_data}

    def _summarize_state(self, state_data: Any, state_type: str) -> Dict[str, Any]:
        """Summarize state data.

        Args:
            state_data: State data from message
            state_type: Type of state ('global' or 'local')

        Returns:
            Summary dict
        """
        summary = {"state_type": state_type}

        if not self.track_data:
            if isinstance(state_data, dict):
                summary["keys"] = list(state_data.keys())
                summary["num_keys"] = len(state_data)
            else:
                summary["type"] = type(state_data).__name__
        else:
            summary["data"] = state_data

        return summary

    def _summarize_action_result(self, result_data: Any) -> Dict[str, Any]:
        """Summarize action execution result.

        Args:
            result_data: Action result data

        Returns:
            Summary dict
        """
        if not self.track_data:
            if isinstance(result_data, dict):
                return {"keys": list(result_data.keys()), "num_keys": len(result_data)}
            else:
                return {"type": type(result_data).__name__}
        else:
            return {"data": result_data}

    def reset(self) -> None:
        """Reset analyzer state between episodes."""
        self.observation_count = 0
        self.global_state_count = 0
        self.local_state_count = 0
        self.state_update_count = 0
        self.action_result_count = 0
        self.reward_history = {}

    def get_summary(self) -> Dict[str, int]:
        """Get summary of tracked message types.

        Returns:
            Dictionary with counts of each message type
        """
        return {
            "observations": self.observation_count,
            "global_states": self.global_state_count,
            "local_states": self.local_state_count,
            "state_updates": self.state_update_count,
            "action_results": self.action_result_count,
        }

    def get_reward_history(self, agent_id: Optional[str] = None) -> Dict[str, List[tuple]]:
        """Get reward history for agents.

        Args:
            agent_id: If provided, return only that agent's history.
                     If None, return all agents' histories.

        Returns:
            Dict mapping agent_id to list of (timestamp, reward) tuples
        """
        if agent_id:
            return {agent_id: self.reward_history.get(agent_id, [])}
        return self.reward_history


class EpisodeStats:
    """Tracks results from an event-driven episode.

    Accumulates event analyses and provides summary statistics
    focusing on observations, states, and actions.

    Attributes:
        event_analyses: List of all event analyses
        start_time: Episode start time
        end_time: Episode end time
    """

    def __init__(self):
        """Initialize empty episode result."""
        self.event_analyses: List[EventAnalysis] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def add_event_analysis(self, analysis: EventAnalysis) -> None:
        """Add an event analysis to the episode results.

        Args:
            analysis: EventAnalysis to add
        """
        self.event_analyses.append(analysis)

        # Track time range
        if self.start_time is None or analysis.timestamp < self.start_time:
            self.start_time = analysis.timestamp
        if self.end_time is None or analysis.timestamp > self.end_time:
            self.end_time = analysis.timestamp

    @property
    def num_events(self) -> int:
        """Total number of events processed."""
        return len(self.event_analyses)

    @property
    def duration(self) -> float:
        """Episode duration in simulation time."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def observation_count(self) -> int:
        """Number of observation events."""
        return sum(
            1 for a in self.event_analyses
            if a.message_type == "get_obs_response"
        )

    @property
    def global_state_count(self) -> int:
        """Number of global state events."""
        return sum(
            1 for a in self.event_analyses
            if a.message_type == "get_global_state_response"
        )

    @property
    def local_state_count(self) -> int:
        """Number of local state events."""
        return sum(
            1 for a in self.event_analyses
            if a.message_type == "get_local_state_response"
        )

    @property
    def state_update_count(self) -> int:
        """Number of state update events."""
        return sum(
            1 for a in self.event_analyses
            if a.message_type == "set_state_completion"
        )

    @property
    def action_result_count(self) -> int:
        """Number of action result events."""
        return sum(
            1 for a in self.event_analyses
            if a.message_type == "set_tick_result"
        )

    def get_event_counts(self) -> Dict[EventType, int]:
        """Get count of events by type.

        Returns:
            Dictionary mapping event types to counts
        """
        counts: Dict[EventType, int] = {}
        for analysis in self.event_analyses:
            counts[analysis.event_type] = counts.get(analysis.event_type, 0) + 1
        return counts

    def get_message_type_counts(self) -> Dict[Optional[str], int]:
        """Get count of events by message type.

        Returns:
            Dictionary mapping message types to counts
        """
        counts: Dict[Optional[str], int] = {}
        for analysis in self.event_analyses:
            if analysis.message_type:
                counts[analysis.message_type] = counts.get(analysis.message_type, 0) + 1
        return counts

    def get_agent_event_counts(self) -> Dict[Optional[str], int]:
        """Get count of events by agent.

        Returns:
            Dictionary mapping agent IDs to event counts
        """
        counts: Dict[Optional[str], int] = {}
        for analysis in self.event_analyses:
            counts[analysis.agent_id] = counts.get(analysis.agent_id, 0) + 1
        return counts

    def get_agent_observations(self, agent_id: str) -> List[EventAnalysis]:
        """Get all observation events for a specific agent.

        Args:
            agent_id: Agent ID to filter by

        Returns:
            List of observation event analyses
        """
        return [
            a for a in self.event_analyses
            if a.agent_id == agent_id and a.message_type == "get_obs_response"
        ]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the episode.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "num_events": self.num_events,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "observations": self.observation_count,
            "global_states": self.global_state_count,
            "local_states": self.local_state_count,
            "state_updates": self.state_update_count,
            "action_results": self.action_result_count,
            "event_counts": {
                event_type.value: count
                for event_type, count in self.get_event_counts().items()
            },
            "message_type_counts": self.get_message_type_counts(),
            "agent_event_counts": self.get_agent_event_counts(),
        }

    def __repr__(self) -> str:
        return (
            f"EpisodeStats(num_events={self.num_events}, "
            f"duration={self.duration:.3f}, "
            f"obs={self.observation_count}, "
            f"actions={self.action_result_count})"
        )

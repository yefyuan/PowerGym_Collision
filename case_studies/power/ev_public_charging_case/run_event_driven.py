"""Event-driven deployment for the EV public charging case study.

Workflow:
1. Train pricing policies in synchronous CTDE mode
2. Attach trained policies to station coordinator agents
3. Configure ScheduleConfig with jitter for realistic async timing
4. Inject test events (e.g., regulation spikes, EV arrivals, market shocks)
5. Run event-driven simulation via env.run_event_driven()

Usage:
    python -m case_studies.power.ev_public_charging_case.run_event_driven

Test Event Examples:
    - Regulation spikes: Sudden frequency deviations requiring load adjustment
    - Market shocks: Rapid LMP price changes
    - Demand surges: Unexpected EV arrival clusters
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.scheduling.analysis import EpisodeAnalyzer
from heron.scheduling.schedule_config import JitterType, ScheduleConfig

from case_studies.power.ev_public_charging_case.policies import PricingPolicy
from case_studies.power.ev_public_charging_case.train_rllib import create_charging_env, train_simple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Test event types for scenario injection."""
    REGULATION_SPIKE = "regulation_spike"      # Sudden frequency deviation


@dataclass
class TestEvent:
    """Represents a test event to inject during simulation.

    Attributes:
        time_s: Time to trigger event (seconds into simulation)
        event_type: Type of event
        severity: Event magnitude (0.0-1.0 scale)
        duration_s: How long event lasts (seconds)
        description: Human-readable description
        callback: Optional custom function for complex events
            Signature: callback(env, time_s, severity) -> None
        target_agents: Optional list of agent IDs affected (None = all)
    """
    time_s: float
    event_type: EventType
    severity: float = 0.5
    duration_s: float = 300.0
    description: str = ""
    callback: Optional[Callable[[Any, float, float], None]] = None
    target_agents: Optional[List[str]] = None

    def __post_init__(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"severity must be in [0.0, 1.0], got {self.severity}")
        if self.duration_s < 0:
            raise ValueError(f"duration_s must be non-negative, got {self.duration_s}")


def configure_schedule_configs(env, seed: int = 42) -> None:
    """Configure ScheduleConfig with jitter for each agent level.

    System (L3): tick every 300s (one simulation step)
    Coordinator (L2): tick every 300s, with obs/act/msg delays
    Field (L1): tick every 300s, with smaller delays

    Args:
        env: ChargingEnv with registered agents
        seed: Base seed for jitter RNG
    """
    system_config = ScheduleConfig.with_jitter(
        tick_interval=300.0,
        obs_delay=1.0,
        act_delay=2.0,
        msg_delay=1.0,
        reward_delay=5.0,  # wait for coordinator rewards (which wait for field agent rewards)
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=seed,
    )
    coordinator_config = ScheduleConfig.with_jitter(
        tick_interval=300.0,
        obs_delay=1.0,
        act_delay=2.0,
        msg_delay=1.0,
        reward_delay=4.0,  # wait for field agent reward round-trips (~3 msg hops)
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=seed + 1,
    )
    field_config = ScheduleConfig.with_jitter(
        tick_interval=300.0,
        obs_delay=0.5,
        act_delay=1.0,
        msg_delay=0.5,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=seed + 2,
    )

    for agent_id, agent in env.registered_agents.items():
        if isinstance(agent, SystemAgent):
            agent.schedule_config = system_config
        elif isinstance(agent, CoordinatorAgent):
            agent.schedule_config = coordinator_config
        elif isinstance(agent, FieldAgent):
            agent.schedule_config = field_config

    # Update scheduler's cached tick configs (cached during attach())
    for agent_id, agent in env.registered_agents.items():
        if hasattr(agent, 'schedule_config') and agent.schedule_config is not None:
            env.scheduler._agent_schedule_configs[agent_id] = agent.schedule_config


def apply_test_event(env, event: TestEvent, elapsed_s: float) -> None:
    """Apply a test event to the environment.

    Args:
        env: ChargingEnv instance
        event: TestEvent to apply
        elapsed_s: Current elapsed time in simulation
    """
    if event.event_type == EventType.REGULATION_SPIKE:
        # Inject sudden frequency deviation (regulation signal spike)
        logger.info(f"[t={elapsed_s:.0f}s] REGULATION_SPIKE: severity={event.severity:.2f} ({event.description})")
        spike_magnitude = -0.5 + event.severity  # Range: [-0.5, +0.5]

        if hasattr(env, 'reg_scenario'):
            # Override regulation signal for this cycle
            env.reg_scenario.current_signal = spike_magnitude
            logger.info(f"  → Regulation signal set to {spike_magnitude:.3f}")



def deploy_event_driven(
    env,
    policies: Dict[str, PricingPolicy],
    t_end: float = 3600.0,
    test_events: Optional[List[TestEvent]] = None,
) -> None:
    """Deploy trained policies in event-driven mode with optional test events.

    Args:
        env: ChargingEnv instance
        policies: Dict mapping station_id -> trained PricingPolicy
        t_end: Simulation end time in seconds
        test_events: Optional list of TestEvent objects to inject
    """
    # Attach trained policies to coordinator agents
    logger.info("Attaching trained policies to station coordinators...")
    env.set_agent_policies(policies)

    # Configure tick timing with jitter
    logger.info("Configuring ScheduleConfigs with Gaussian jitter...")

    # Log test events if any
    if test_events:
        logger.info(f"Scheduled {len(test_events)} test events:")
        for evt in sorted(test_events, key=lambda e: e.time_s):
            logger.info(f"  t={evt.time_s:.0f}s: {evt.event_type.value} (severity={evt.severity:.2f}) - {evt.description}")

    # Run event-driven simulation
    logger.info(f"Running event-driven simulation for {t_end}s...")
    episode_analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(episode_analyzer=episode_analyzer, t_end=t_end)

    # Print summary
    summary = episode.summary()
    logger.info("Event-driven simulation complete.")
    logger.info(f"  Total events: {summary.get('num_events', 0)}")
    logger.info(f"  Duration: {summary.get('duration', 0):.1f}s")
    logger.info(f"  Observations: {summary.get('observations', 0)}")
    logger.info(f"  State updates: {summary.get('state_updates', 0)}")
    logger.info(f"  Action results: {summary.get('action_results', 0)}")
    logger.info(f"  Event types: {summary.get('event_counts', {})}")
    logger.info(f"  Message types: {summary.get('message_type_counts', {})}")

    reward_history = episode_analyzer.get_reward_history()
    if reward_history:
        for agent_id, rewards in reward_history.items():
            if rewards:
                total_r = sum(r for _, r in rewards)
                logger.info(f"  Agent {agent_id}: total_reward={total_r:.2f}, steps={len(rewards)}")

    return episode


def main():
    """Full pipeline: train CTDE -> deploy event-driven with test events."""
    logger.info("=" * 80)
    logger.info("Phase 1: CTDE Training (synchronous)")
    logger.info("=" * 80)
    env, policies, returns = train_simple(num_episodes=50, seed=42)
    env.close()

    logger.info(f"\nTraining returns (last 5): {[round(r, 2) for r in returns[-5:]]}")

    logger.info("\n" + "=" * 80)
    logger.info("Phase 2: Event-Driven Deployment with Test Events")
    logger.info("=" * 80)

    # Create fresh env for event-driven deployment
    deploy_env = create_charging_env()

    # Define test scenario with various events
    test_events = [
        # Regulation event: positive frequency spike (over-generation scenario)
        TestEvent(
            time_s=300.0,
            event_type=EventType.REGULATION_SPIKE,
            severity=0.8,  # Strong positive spike
            duration_s=600.0,
            description="Positive frequency spike (over-generation) - demand increase needed"
        ),
    ]

    deploy_event_driven(
        deploy_env,
        policies,
        t_end=3600.0,
        test_events=test_events
    )
    deploy_env.close()

    logger.info("\nDone.")


if __name__ == "__main__":
    main()




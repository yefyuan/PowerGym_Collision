"""ScheduleConfig & EventScheduler -- configuring agent timing.

HERON's event-driven mode simulates realistic distributed systems where
agents tick at different rates, observations arrive with latency, and
actions take effect after a delay. ScheduleConfig controls all of this.

This script demonstrates:
1. ScheduleConfig basics -- tick_interval, obs_delay, act_delay, msg_delay
2. Deterministic vs jittered configs -- training vs testing modes
3. JitterType distributions -- NONE, UNIFORM, GAUSSIAN
4. Default configs -- field (1s), coordinator (60s), system (300s)
5. EventScheduler basics -- event queue, handler dispatch, run_until

Domain: Standalone demonstrations (no env needed).

Usage:
    cd "examples/6. event_driven_simulation"
    python schedule_config_and_scheduling.py
"""

import numpy as np

from heron.scheduling import (
    ScheduleConfig,
    JitterType,
    DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG,
    DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG,
    DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG,
    EventScheduler,
    Event,
    EventType,
    EpisodeAnalyzer,
)


# ---------------------------------------------------------------------------
# 1. ScheduleConfig basics
# ---------------------------------------------------------------------------

def demo_schedule_config_basics():
    """Show ScheduleConfig parameters and their meaning."""
    print("=" * 60)
    print("Part 1: ScheduleConfig Basics")
    print("=" * 60)
    print("""
  ScheduleConfig controls timing for each agent in event-driven mode:

    tick_interval  -- time between agent steps (how often agent acts)
    obs_delay      -- latency for observations to arrive
    act_delay      -- delay before action takes effect
    msg_delay      -- delay for message delivery (protocols)
    reward_delay   -- delay for reward aggregation (coordinators)
""")

    # Create a config for a sensor that ticks every 0.5s
    config = ScheduleConfig(
        tick_interval=0.5,
        obs_delay=0.05,    # 50ms observation latency
        act_delay=0.1,     # 100ms action delay
        msg_delay=0.02,    # 20ms message delay
    )

    print(f"  Sensor config:")
    print(f"    tick_interval = {config.tick_interval}s  (acts every 0.5s)")
    print(f"    obs_delay     = {config.obs_delay}s  (50ms observation latency)")
    print(f"    act_delay     = {config.act_delay}s  (100ms action delay)")
    print(f"    msg_delay     = {config.msg_delay}s  (20ms message delay)")
    print(f"    jitter_type   = {config.jitter_type.value}  (deterministic)")

    # Deterministic: every call returns the same value
    print(f"\n  Deterministic timing (10 calls to get_tick_interval):")
    intervals = [config.get_tick_interval() for _ in range(10)]
    print(f"    {intervals[:5]} ... (all identical)")


# ---------------------------------------------------------------------------
# 2. Deterministic vs jittered configs
# ---------------------------------------------------------------------------

def demo_deterministic_vs_jitter():
    """Compare ScheduleConfig.deterministic() vs ScheduleConfig.with_jitter()."""
    print("\n" + "=" * 60)
    print("Part 2: Deterministic vs Jittered Configs")
    print("=" * 60)
    print("""
  Training mode: deterministic (no jitter)
    -> Consistent timing for reproducible gradient steps

  Testing mode: jittered (realistic)
    -> Models network delays, clock drift, sensor noise
""")

    # Deterministic (for training)
    det_config = ScheduleConfig.deterministic(
        tick_interval=1.0,
        obs_delay=0.1,
        act_delay=0.2,
    )

    # Jittered (for testing)
    jit_config = ScheduleConfig.with_jitter(
        tick_interval=1.0,
        obs_delay=0.1,
        act_delay=0.2,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,   # 10% standard deviation
        seed=42,
    )

    print(f"  {'Call':<6} {'Deterministic':>15} {'Jittered (10%)':>15}")
    print(f"  {'-' * 38}")
    for i in range(8):
        det_val = det_config.get_tick_interval()
        jit_val = jit_config.get_tick_interval()
        print(f"  {i + 1:<6} {det_val:>15.4f} {jit_val:>15.4f}")

    print(f"\n  Deterministic: always exactly 1.0000")
    print(f"  Jittered: varies around 1.0 with ~10% std deviation")


# ---------------------------------------------------------------------------
# 3. JitterType distributions
# ---------------------------------------------------------------------------

def demo_jitter_types():
    """Compare UNIFORM and GAUSSIAN jitter distributions."""
    print("\n" + "=" * 60)
    print("Part 3: JitterType Distributions")
    print("=" * 60)

    base = 1.0
    ratio = 0.2  # 20% jitter for visible differences
    n_samples = 1000

    for jtype in [JitterType.UNIFORM, JitterType.GAUSSIAN]:
        config = ScheduleConfig.with_jitter(
            tick_interval=base,
            jitter_type=jtype,
            jitter_ratio=ratio,
            seed=42,
        )
        samples = [config.get_tick_interval() for _ in range(n_samples)]
        arr = np.array(samples)

        print(f"\n  {jtype.value.upper()} (base={base}, ratio={ratio}):")
        print(f"    mean   = {arr.mean():.4f}")
        print(f"    std    = {arr.std():.4f}")
        print(f"    min    = {arr.min():.4f}")
        print(f"    max    = {arr.max():.4f}")
        print(f"    range  = [{base - base * ratio:.2f}, {base + base * ratio:.2f}]"
              f"  (theoretical for UNIFORM)")

    print("""
  UNIFORM: base +/- jitter_range, flat distribution
    Good for: bounded uncertainty (network jitter)

  GAUSSIAN: base as mean, jitter_range as std dev
    Good for: natural variation (sensor noise, clock drift)
""")


# ---------------------------------------------------------------------------
# 4. Default configs
# ---------------------------------------------------------------------------

def demo_default_configs():
    """Show the default tick configs per agent type."""
    print("=" * 60)
    print("Part 4: Default Tick Configs")
    print("=" * 60)
    print("""
  HERON provides default configs matching the agent hierarchy:

    Field agent       -- fast tick (1s)     -- direct sensor/actuator
    Coordinator agent -- medium tick (60s)  -- manages subordinates
    System agent      -- slow tick (300s)   -- global orchestration
""")

    defaults = [
        ("Field Agent", DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG),
        ("Coordinator Agent", DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG),
        ("System Agent", DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG),
    ]

    print(f"  {'Agent Type':<22} {'tick_interval':>14} {'jitter':>10}")
    print(f"  {'-' * 48}")
    for name, config in defaults:
        print(f"  {name:<22} {config.tick_interval:>12.1f}s {config.jitter_type.value:>10}")

    print("""
  Override per agent by passing schedule_config to agent construction:

    ScheduleConfig.with_jitter(
        tick_interval=0.5,    # faster than default 1s
        obs_delay=0.05,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=42,
    )
""")


# ---------------------------------------------------------------------------
# 5. EventScheduler basics
# ---------------------------------------------------------------------------

def demo_scheduler_basics():
    """Show EventScheduler event queue and processing."""
    print("=" * 60)
    print("Part 5: EventScheduler Basics")
    print("=" * 60)
    print("""
  The EventScheduler is a priority queue that processes events
  in timestamp order. Each event has:
    - timestamp: when to process
    - event_type: AGENT_TICK, ACTION_EFFECT, MESSAGE_DELIVERY, etc.
    - agent_id: target agent
    - priority: tie-breaker (lower = first)
""")

    scheduler = EventScheduler(start_time=0.0)

    # Manually schedule some events to show ordering
    events = [
        Event(timestamp=1.0, event_type=EventType.AGENT_TICK,
              agent_id="sensor_a", payload={"step": 1}),
        Event(timestamp=0.5, event_type=EventType.OBSERVATION_READY,
              agent_id="sensor_a"),
        Event(timestamp=1.0, event_type=EventType.ACTION_EFFECT,
              agent_id="sensor_a", priority=0),
        Event(timestamp=1.0, event_type=EventType.MESSAGE_DELIVERY,
              agent_id="sensor_b", priority=2),
        Event(timestamp=0.2, event_type=EventType.ENV_UPDATE,
              agent_id="system"),
    ]

    for e in events:
        scheduler.schedule(e)

    print(f"  Scheduled {len(events)} events. Processing order:")
    print(f"  {'#':<4} {'Time':>6} {'Type':<22} {'Agent':<12} {'Priority':>8}")
    print(f"  {'-' * 56}")

    idx = 1
    while scheduler.event_queue:
        event = scheduler.pop()
        scheduler.current_time = event.timestamp
        print(f"  {idx:<4} {event.timestamp:>6.1f} {event.event_type.value:<22} "
              f"{event.agent_id or '':12} {event.priority:>8}")
        idx += 1

    print("""
  Events at t=1.0 are ordered by priority:
    ACTION_EFFECT (prio=0) before MESSAGE_DELIVERY (prio=2)

  In the full system, the scheduler is attached to agents via
  scheduler.attach(agents), which registers handlers and schedules
  the first system tick. The system tick cascades down the hierarchy.
""")


# ---------------------------------------------------------------------------
# 6. Event types explained
# ---------------------------------------------------------------------------

def demo_event_types():
    """Explain each EventType and when it fires."""
    print("=" * 60)
    print("Part 6: Event Types")
    print("=" * 60)
    print("""
  EventType          When it fires                    Priority
  ─────────────────────────────────────────────────────────────
  AGENT_TICK         Agent's scheduled step time       default
  OBSERVATION_READY  After obs_delay from tick         default
  ACTION_EFFECT      After act_delay from action       0 (high)
  SIMULATION         After physics simulation          1 (mid)
  MESSAGE_DELIVERY   After msg_delay from send         2 (low)
  ENV_UPDATE         Environment state change          default
  CUSTOM             Domain-specific events            default

  Typical event cascade for one agent tick:

    t=0.0  AGENT_TICK(sensor_a)
           -> agent observes, computes action
    t=0.05 OBSERVATION_READY(sensor_a)    [obs_delay=0.05]
           -> observation arrives at agent
    t=0.1  ACTION_EFFECT(sensor_a)        [act_delay=0.1]
           -> action takes effect in environment
    t=0.1  SIMULATION(system)             [physics runs]
           -> environment state updated
    t=0.12 MESSAGE_DELIVERY(coordinator)  [msg_delay=0.02]
           -> state shared with coordinator
    t=1.0  AGENT_TICK(sensor_a)           [next tick]

  With jitter, each delay varies per invocation,
  modeling realistic distributed system behavior.
""")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    demo_schedule_config_basics()
    demo_deterministic_vs_jitter()
    demo_jitter_types()
    demo_default_configs()
    demo_scheduler_basics()
    demo_event_types()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  ScheduleConfig:
    Controls tick_interval, obs/act/msg/reward delays
    .deterministic() for training (no randomness)
    .with_jitter() for testing (realistic timing)

  JitterType:
    NONE     -- deterministic (training)
    UNIFORM  -- bounded random (network jitter)
    GAUSSIAN -- unbounded random (sensor noise)

  EventScheduler:
    Priority queue processing events by (timestamp, priority)
    Handlers registered per agent per event type
    run_until(t_end) drives the simulation

  See dual_mode_execution.py for the full train/eval workflow.
""")
    print("Done.")


if __name__ == "__main__":
    main()

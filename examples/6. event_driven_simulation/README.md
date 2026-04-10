# Level 6: Event-Driven Simulation

**From instant step-based training to realistic event-driven evaluation.**

## What You'll Learn

| Concept | Where |
|---------|-------|
| ScheduleConfig basics (intervals, delays) | `schedule_config_and_scheduling.py` Part 1 |
| Deterministic vs jittered configs | `schedule_config_and_scheduling.py` Part 2 |
| JitterType distributions | `schedule_config_and_scheduling.py` Part 3 |
| Default agent tick configs | `schedule_config_and_scheduling.py` Part 4 |
| EventScheduler priority queue | `schedule_config_and_scheduling.py` Part 5 |
| Event types and cascades | `schedule_config_and_scheduling.py` Part 6 |
| Step-based execution (training) | `dual_mode_execution.py` Part 1 |
| Event-driven execution (eval) | `dual_mode_execution.py` Part 2 |
| EpisodeStats analysis | `dual_mode_execution.py` Part 3 |
| Mode comparison | `dual_mode_execution.py` Part 4 |

## Prerequisites

- **Level 3**: Building Environments (agents, features, SimpleEnv)
- **Level 5**: Training Algorithms (Policy ABC, decorators)

## Architecture

```
Dual-Mode Execution:

  Training Mode (Step-Based)          Evaluation Mode (Event-Driven)
  ==============================      ================================

  env.step(actions)                   env.run_event_driven(analyzer, t_end)
    |                                   |
    v                                   v
  Instant execution:                  EventScheduler:
    observe -> act -> simulate          schedule events with delays
    (no timing, no delays)              process by (timestamp, priority)
                                        jitter on every delay
                                        |
                                        v
                                      Event cascade per tick:
                                        AGENT_TICK -> obs_delay ->
                                        OBSERVATION_READY -> act_delay ->
                                        ACTION_EFFECT -> msg_delay ->
                                        MESSAGE_DELIVERY -> next tick
```

## ScheduleConfig

```
ScheduleConfig fields:
  tick_interval   how often agent acts (seconds)
  obs_delay       observation latency
  act_delay       action effect delay
  msg_delay       message delivery delay
  reward_delay    reward aggregation delay

Constructors:
  ScheduleConfig.deterministic(...)     no jitter (training)
  ScheduleConfig.with_jitter(...)       realistic timing (testing)

JitterType:
  NONE       deterministic
  UNIFORM    bounded random (+/- ratio)
  GAUSSIAN   normal distribution (std = ratio * base)

Defaults (per agent type):
  Field agent        tick_interval = 1s
  Coordinator agent  tick_interval = 60s
  System agent       tick_interval = 300s
```

## Event Types

```
EventType           Priority  When
──────────────────────────────────────────
AGENT_TICK          default   Scheduled agent step
OBSERVATION_READY   default   After obs_delay
ACTION_EFFECT       0 (high)  After act_delay
SIMULATION          1 (mid)   Physics update
MESSAGE_DELIVERY    2 (low)   After msg_delay
ENV_UPDATE          default   State change
CUSTOM              default   Domain-specific
```

## File Structure

```
6. event_driven_simulation/
├── README.md
├── schedule_config_and_scheduling.py   # ScheduleConfig + EventScheduler basics
└── dual_mode_execution.py          # Step-based vs event-driven
```

## Running

```bash
cd "examples/6. event_driven_simulation"

# ScheduleConfig and scheduling fundamentals
python schedule_config_and_scheduling.py

# Dual-mode execution comparison
python dual_mode_execution.py
```

## Key API

### ScheduleConfig

```python
from heron.scheduling import ScheduleConfig, JitterType

# Training: deterministic
config = ScheduleConfig.deterministic(tick_interval=1.0, obs_delay=0.1)

# Testing: jittered
config = ScheduleConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,
    act_delay=0.2,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=42,
)
```

### Event-Driven Execution

```python
from heron.scheduling import EpisodeAnalyzer

# Assign policies
env.set_agent_policies({"room_a": my_policy, "room_b": my_policy})

# Run simulation
analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
result = env.run_event_driven(analyzer, t_end=100.0)

# Analyze results
summary = result.summary()
print(f"Events: {summary['num_events']}")
print(f"Duration: {summary['duration']:.1f}s")
rewards = analyzer.get_reward_history()
```

## Next Steps

- **Level 7**: Advanced Patterns -- mixed action spaces, messaging, heterogeneous agents (coming soon)

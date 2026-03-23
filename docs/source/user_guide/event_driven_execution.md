# Event-Driven Execution

This guide explains HERON's event-driven execution mode for testing trained policies under realistic timing constraints.

## Two Execution Modes

HERON supports two execution modes that share the same agent code:

| | Synchronous (Option A) | Event-Driven (Option B) |
|---|---|---|
| **Use Case** | Training | Testing / Deployment |
| **Agent Timing** | All agents step together | Heterogeneous tick rates |
| **Communication** | Direct method calls | Message broker with delays |
| **Observation** | Instantaneous | Configurable latency |
| **Actions** | Immediate effect | Delayed effect |

## Why Event-Driven Testing?

Policies trained in synchronous mode assume:
- Perfect synchronization between agents
- Zero communication latency
- Instantaneous observations and actions

Real-world systems have:
- Agents running at different frequencies
- Network delays in communication
- Sensor and actuator latency
- Timing variability (jitter)

Event-driven mode tests how your trained policy handles these realistic constraints.

## Quick Start

```python
from heron.scheduling import ScheduleConfig, JitterType

# 1. Train in synchronous mode (Option A)
env = MyEnv()
# ... RL training loop ...

# 2. Test in event-driven mode (Option B)
env.setup_event_driven()

# 3. Configure timing
config = ScheduleConfig.with_jitter(
    tick_interval=1.0,    # 1 second tick rate
    obs_delay=0.1,        # 100ms observation delay
    act_delay=0.2,        # 200ms action delay
    msg_delay=0.05,       # 50ms message delay
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,     # 10% variability
)

# 4. Run simulation
env.run_event_driven(t_end=3600.0)
```

## Timing Parameters

### tick_interval

Time between agent's observe/act cycles.

```
    |----tick_interval----|----tick_interval----|
    ^                     ^                     ^
  tick 0                tick 1               tick 2
```

**Typical values:**
- Field agents (sensors): 0.1 - 1.0 seconds
- Coordinators: 1.0 - 60 seconds
- System agents: 60+ seconds

### obs_delay

Time from when state changes to when agent observes it.

```
State change         Observation ready
    |----obs_delay------|
    ^                   ^
 t=0.0               t=0.1
```

Models sensor latency, data processing time, or network transmission.

### act_delay

Time from when agent computes action to when it takes effect.

```
Agent computes       Action takes effect
    |----act_delay------|
    ^                   ^
 t=0.0               t=0.2
```

Models actuator response time, command transmission, or processing.

### msg_delay

Time for messages between agents to be delivered.

```
Coordinator sends    Subordinate receives
    |----msg_delay------|
    ^                   ^
 t=0.0               t=0.05
```

Models network latency in distributed systems.

## Configuring Agent Timing

### Per-Agent Configuration

```python
from heron.agents import FieldAgent
from heron.scheduling import ScheduleConfig, JitterType

# Fast sensor with low latency
sensor = FieldAgent(
    agent_id="sensor_1",
    tick_interval=0.1,    # 10 Hz
    obs_delay=0.01,       # 10ms
    act_delay=0.02,       # 20ms
)

# Or use ScheduleConfig for full control
config = ScheduleConfig.with_jitter(
    tick_interval=0.1,
    obs_delay=0.01,
    jitter_ratio=0.05,
)
sensor = FieldAgent(agent_id="sensor_1", schedule_config=config)
```

### Hierarchical Timing Patterns

Different agent levels typically have different timing requirements:

```python
# Fast field agents (sensors/actuators)
field_config = ScheduleConfig(tick_interval=0.1, obs_delay=0.01)

# Medium-speed coordinators
coord_config = ScheduleConfig(tick_interval=1.0, obs_delay=0.1, msg_delay=0.05)

# Slow system-level agents
system_config = ScheduleConfig(tick_interval=60.0, obs_delay=1.0)
```

## Adding Timing Variability (Jitter)

Real systems have timing variability. Use jitter to test robustness:

```python
from heron.scheduling import ScheduleConfig, JitterType

# Gaussian jitter (most realistic)
config = ScheduleConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,     # std = 10% of base value
    seed=42,              # For reproducibility
)

# Uniform jitter
config = ScheduleConfig.with_jitter(
    tick_interval=1.0,
    jitter_type=JitterType.UNIFORM,
    jitter_ratio=0.1,     # +/- 10% of base value
)
```

| Jitter Type | Distribution | Use Case |
|-------------|--------------|----------|
| `NONE` | Deterministic | Training, debugging |
| `UNIFORM` | Uniform +/- range | Bounded uncertainty |
| `GAUSSIAN` | Normal distribution | Natural variability |

## Event Types

The scheduler processes these event types:

| Event | Handler Called When | Typical Action |
|-------|---------------------|----------------|
| `AGENT_TICK` | Agent's tick time | Call `agent.tick()` |
| `ACTION_EFFECT` | Action delay elapsed | Apply action to environment |
| `MESSAGE_DELIVERY` | Message delay elapsed | Deliver via message broker |
| `OBSERVATION_READY` | Observation delay elapsed | Make observation available |

## Setting Up Event Handlers

### Using Default Handlers

```python
env.setup_event_driven()
env.setup_default_handlers(
    global_state_fn=lambda: {"time": env.simulation_time},
    on_action_effect=lambda agent_id, action: env.apply_action(agent_id, action),
)
```

### Custom Handlers

```python
from heron.scheduling import EventType

def my_tick_handler(event, scheduler):
    agent = env.get_heron_agent(event.agent_id)
    if agent:
        # Custom tick logic
        obs = agent.observe()
        action = agent.policy.forward(obs)
        scheduler.schedule_action_effect(
            agent_id=event.agent_id,
            action=action,
            delay=agent.act_delay,
        )

env.scheduler.set_handler(EventType.AGENT_TICK, my_tick_handler)
```

## Running Simulations

```python
# Run until time limit
events = env.run_event_driven(t_end=3600.0)  # 1 hour

# Run with event limit
events = env.run_event_driven(t_end=3600.0, max_events=10000)

# Step-by-step execution
scheduler = env.scheduler
while scheduler.current_time < 3600.0:
    scheduler.process_next()
    # ... check conditions, log, etc.
```

## Best Practices

1. **Start simple**: Test with deterministic timing first, then add jitter
2. **Match reality**: Use timing parameters from your real system
3. **Test edge cases**: High jitter, slow networks, fast agents
4. **Compare modes**: Same policy should perform similarly in both modes
5. **Use seeds**: Set RNG seeds for reproducible testing

## Example: Complete Workflow

```python
import numpy as np
from heron.scheduling import ScheduleConfig, JitterType

# 1. Define environment with agents
env = MyHeronEnv()

# 2. Train policy (synchronous mode)
for episode in range(1000):
    obs, _ = env.reset()
    while not done:
        actions = policy.compute_actions(obs)
        obs, rewards, done, _, _ = env.step(actions)
        policy.learn(obs, rewards)

# 3. Evaluate in event-driven mode
env.setup_event_driven()
env.setup_default_handlers(
    global_state_fn=lambda: env.get_state(),
    on_action_effect=lambda aid, act: env.apply_action(aid, act),
)

# Configure realistic timing
for agent_id, agent in env.registered_agents.items():
    config = ScheduleConfig.with_jitter(
        tick_interval=agent.tick_interval,
        obs_delay=0.1,
        act_delay=0.2,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=42,
    )
    env.scheduler.register_agent(agent_id, schedule_config=config)

# Run evaluation
env.run_event_driven(t_end=3600.0)
print(f"Policy performed with realistic timing!")
```

## See Also

- [heron/scheduling/README.md](../../../heron/scheduling/README.md) - Module documentation
- [examples/07_event_driven_mode.py](../../../case_studies/power/examples/07_event_driven_mode.py) - Power grid example
- [Key Concepts](../key_concepts.md) - Core HERON concepts

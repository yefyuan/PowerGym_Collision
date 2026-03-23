# Centralized vs Distributed Execution

HERON supports two execution modes that enable different trade-offs between training efficiency and deployment realism.

## Overview

| Aspect | Centralized Mode | Distributed Mode |
|--------|------------------|------------------|
| **Observability** | Full global state | Local + messages |
| **Communication** | Direct function calls | Message broker |
| **Training Speed** | Fast | Slower |
| **Deployment** | Simulation only | Production-ready |

## Centralized Mode

In centralized mode, all agents have access to global state. This is ideal for:

- Fast algorithm development
- Hyperparameter tuning
- Baseline comparisons

```python
from powergrid.envs import MultiAgentMicrogrids

env = MultiAgentMicrogrids({
    "centralized": True,  # Enable centralized mode
    "max_episode_steps": 24,
})

obs, info = env.reset()
# obs contains full observability for each agent
```

### How It Works

1. Environment maintains global state
2. Each agent's `observe()` returns full state vector
3. Actions are applied directly without message passing

```
┌─────────────────────────────────────┐
│         Global Environment          │
│  ┌─────┐  ┌─────┐  ┌─────┐         │
│  │ A1  │  │ A2  │  │ A3  │         │
│  └──┬──┘  └──┬──┘  └──┬──┘         │
│     │        │        │             │
│     └────────┴────────┘             │
│              │                      │
│     Full State Access               │
└─────────────────────────────────────┘
```

## Distributed Mode

In distributed mode, agents observe only local information plus messages from other agents. This is ideal for:

- Realistic deployment scenarios
- Testing communication protocols
- Distributed system simulation

```python
from powergrid.envs import MultiAgentMicrogrids
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

env = MultiAgentMicrogrids({
    "centralized": False,  # Distributed mode
    "message_broker": broker,
    "max_episode_steps": 24,
})

obs, info = env.reset()
# obs contains only local observations + received messages
```

### How It Works

1. Each agent maintains local state only
2. Agents communicate via message broker
3. Coordination happens through protocols

```
┌─────────────────────────────────────┐
│           Message Broker            │
│  ┌───────────────────────────┐     │
│  │    pub/sub channels       │     │
│  └───────────────────────────┘     │
│         ▲         ▲         ▲      │
│         │         │         │      │
│    ┌────┴───┐ ┌───┴────┐ ┌──┴───┐  │
│    │   A1   │ │   A2   │ │  A3  │  │
│    │ local  │ │ local  │ │local │  │
│    └────────┘ └────────┘ └──────┘  │
└─────────────────────────────────────┘
```

## Switching Modes

The mode can be switched with a single configuration change:

```python
# Training (centralized for speed)
train_config = {"centralized": True, "train": True}

# Evaluation (distributed for realism)
eval_config = {"centralized": False, "train": False}
```

## Proxy Agent for Distributed Mode

For distributed execution, use `Proxy` to handle message-based coordination:

```python
from heron.agents import Proxy
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

# Create proxy that wraps actual agent
proxy = Proxy(
    agent_id="mg1_proxy",
    actual_agent=grid_agent,
    broker=broker,
    upstream_id="system_operator"
)

# Proxy handles message passing automatically
await proxy.step_distributed()
```

## Protocol Behavior by Mode

### Vertical Protocols

| Protocol | Centralized | Distributed |
|----------|-------------|-------------|
| SetpointProtocol | Direct assignment | Message with setpoint |
| PriceSignalProtocol | Direct price access | Price broadcast message |

### Horizontal Protocols

| Protocol | Centralized | Distributed |
|----------|-------------|-------------|
| P2PTradingProtocol | Instant matching | Bid/ask messages |
| ConsensusProtocol | Single iteration | Multiple message rounds |

## Best Practices

1. **Train in centralized mode** for faster iteration
2. **Validate in distributed mode** before deployment
3. **Use shared rewards** in centralized mode for cooperative tasks
4. **Test communication overhead** in distributed mode

## Example: Mode Comparison

```python
import time

# Centralized training
env_centralized = MultiAgentMicrogrids({"centralized": True})
start = time.time()
for _ in range(1000):
    obs, _ = env_centralized.reset()
    for _ in range(24):
        actions = {a: env_centralized.action_spaces[a].sample() for a in env_centralized.agents}
        env_centralized.step(actions)
print(f"Centralized: {time.time() - start:.2f}s")

# Distributed evaluation
env_distributed = MultiAgentMicrogrids({"centralized": False})
start = time.time()
for _ in range(100):  # Fewer iterations due to overhead
    obs, _ = env_distributed.reset()
    for _ in range(24):
        actions = {a: env_distributed.action_spaces[a].sample() for a in env_distributed.agents}
        env_distributed.step(actions)
print(f"Distributed: {time.time() - start:.2f}s")
```

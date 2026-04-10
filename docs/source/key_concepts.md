# Key Concepts

HERON is built around several core abstractions that enable hierarchical multi-agent systems.

## Agent Hierarchy

Agents are organized in a tree structure with configurable depth:

```
Level 3: SystemAgent       (global coordination)
         │
Level 2: CoordinatorAgent  (manages subordinates)
         │
Level 1: FieldAgent        (local sensing/actuation)
```

Each level has distinct responsibilities:

| Agent Type | Level | Responsibility |
|------------|-------|----------------|
| `FieldAgent` | 1 | Local sensing, actuation, state management |
| `CoordinatorAgent` | 2+ | Manages subordinates, aggregates observations |
| `SystemAgent` | Top | Global objectives, system-wide coordination |

```python
from heron.agents import FieldAgent, CoordinatorAgent, SystemAgent

# Build hierarchy bottom-up
field = FieldAgent(agent_id="sensor_1")
coordinator = CoordinatorAgent(
    agent_id="zone_1",
    subordinates={"sensor_1": field}
)
system = SystemAgent(
    agent_id="operator",
    subordinates={"zone_1": coordinator}
)
```

## Features and Visibility

Features are composable state components with visibility control:

```python
from heron.core.feature import Feature

class TemperatureFeature(Feature):
    def __init__(self, value: float):
        # Visibility tags control who can see this feature
        super().__init__(visibility=["owner", "coordinator"])
        self.value = value

    def vector(self) -> np.ndarray:
        return np.array([self.value], dtype=np.float32)

    def dim(self) -> int:
        return 1
```

**Default visibility tags:**

| Tag | Who Can Observe |
|-----|-----------------|
| `owner` | Only the owning agent |
| `coordinator` | Owner + its coordinator |
| `system` | System-level agents |
| `global` | All agents |

Custom tags can be defined for domain-specific needs (e.g., `neighbor`, `region_a`).

## State and Observation

**State** encapsulates an agent's internal features:

```python
from heron.core.state import FieldAgentState

state = FieldAgentState()
state.add_feature("temperature", temp_feature)
state.add_feature("humidity", humidity_feature)

# Get state vector (all features)
full_vector = state.vector()

# Get only coordinator-visible features
coord_vector = state.vector(visibility_tags=["coordinator"])
```

**Observation** is what an agent perceives:

```python
from heron.core.observation import Observation

obs = Observation(
    local={"state": state.vector()},      # Local features
    global_={"price": price_array},        # Global info (centralized mode)
    messages=[msg1, msg2],                 # Received messages (distributed mode)
    timestamp=0.0
)
```

## Actions

Actions support continuous and discrete components:

```python
from heron.core.action import Action

action = Action()
action.set_specs(
    dim_c=2,  # 2 continuous dimensions
    range=(np.array([-1, -1]), np.array([1, 1])),
    dim_d=1,  # 1 discrete dimension
    n_options=[3]  # 3 options for discrete action
)

action.sample()  # Random action
action.c[:] = [0.5, -0.3]  # Set continuous values
action.d[:] = [1]  # Set discrete values
```

## Coordination Protocols

Protocols define how agents coordinate:

### Vertical Protocols (Top-Down)

```python
from heron.protocols.vertical import SetpointProtocol, PriceSignalProtocol

# Direct control
setpoint = SetpointProtocol()

# Market-based coordination
price = PriceSignalProtocol(initial_price=50.0)
```

### Horizontal Protocols (Peer-to-Peer)

```python
from heron.protocols.horizontal import P2PTradingProtocol, ConsensusProtocol

# Resource trading
trading = P2PTradingProtocol()

# Agreement protocol
consensus = ConsensusProtocol(max_iterations=10, tolerance=0.01)
```

## Execution Modes

HERON supports two execution modes:

### Centralized Mode

- All agents have full observability
- Direct function calls between agents
- Fast training and development

```python
env = MyEnv(config={"centralized": True})
```

### Distributed Mode

- Agents observe only local state + messages
- Communication via message broker
- Realistic deployment scenarios

```python
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()
env = MyEnv(config={"centralized": False, "message_broker": broker})
```

## Message Broker

For distributed execution, agents communicate via a message broker:

```python
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

# Publish
await broker.publish("control", {"setpoint": 1.0})

# Subscribe
broker.subscribe("control", callback=handle_message)

# Consume
msg = await broker.consume("status")
```

The `MessageBroker` interface can be extended for production systems (Kafka, Redis, etc.).

## Environment Interface

HERON environments implement PettingZoo's `ParallelEnv`:

```python
from pettingzoo import ParallelEnv

class MyEnv(ParallelEnv):
    def reset(self, seed=None, options=None):
        # Returns (observations, infos)
        pass

    def step(self, actions):
        # Returns (observations, rewards, terminateds, truncateds, infos)
        pass
```

## Timing Parameters

Agents can be configured with timing parameters for event-driven execution:

```python
from heron.agents import FieldAgent
from heron.scheduling import ScheduleConfig, JitterType

# Simple timing parameters
sensor = FieldAgent(
    agent_id="sensor_1",
    tick_interval=1.0,    # Time between ticks (seconds)
    obs_delay=0.1,        # Observation latency
    act_delay=0.2,        # Action effect delay
    msg_delay=0.05,       # Message delivery delay
)

# Full control with ScheduleConfig (includes jitter for testing)
config = ScheduleConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,     # 10% variability
    seed=42,
)
sensor = FieldAgent(agent_id="sensor_1", schedule_config=config)
```

**Timing parameter reference:**

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `tick_interval` | Time between agent ticks | 0.1s - 60s |
| `obs_delay` | Latency before observation is available | 0 - 1s |
| `act_delay` | Delay before action takes effect | 0 - 1s |
| `msg_delay` | Message delivery latency | 0 - 1s |

**Hierarchical timing patterns:**

| Agent Level | tick_interval | Rationale |
|-------------|---------------|-----------|
| FieldAgent | 0.1 - 1.0s | Fast local sensing/actuation |
| CoordinatorAgent | 1.0 - 60s | Aggregate and coordinate |
| SystemAgent | 60s+ | High-level decisions |

See the [Event-Driven Execution Guide](user_guide/event_driven_execution.md) for details.

## Summary

| Concept | Purpose |
|---------|---------|
| **Agent Hierarchy** | Multi-level control structure |
| **Features** | Composable state with visibility |
| **Protocols** | Coordination patterns |
| **Execution Modes** | Training vs deployment |
| **Message Broker** | Distributed communication |
| **Timing Parameters** | Realistic delays for testing |

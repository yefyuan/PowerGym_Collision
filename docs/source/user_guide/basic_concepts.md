# Basic Concepts

HERON provides a domain-agnostic framework for hierarchical multi-agent reinforcement learning. This page introduces the core concepts.

## Agent Hierarchy

Agents are organized in a tree structure where each level has distinct responsibilities:

| Level | Role | Description |
|-------|------|-------------|
| Leaf | **Field Agent** | Local sensing and actuation |
| Mid | **Coordinator Agent** | Manages child agents in a subtree |
| Root | **System Agent** | Global coordination |

```python
from heron.agents import FieldAgent, CoordinatorAgent, SystemAgent

# Create hierarchy
field_agent = FieldAgent(agent_id="sensor_1", level=1)
coordinator = CoordinatorAgent(
    agent_id="zone_controller",
    level=2,
    subordinates={"sensor_1": field_agent}
)
system_agent = SystemAgent(
    agent_id="system_operator",
    level=3,
    subordinates={"zone_controller": coordinator}
)
```

## Features and Visibility

Features use string-based visibility tags to control information sharing:

```python
from heron.core.feature import Feature
import numpy as np

class TemperatureFeature(Feature):
    def __init__(self, value: float = 20.0):
        super().__init__(visibility=["owner", "coordinator"])
        self.value = value

    def vector(self) -> np.ndarray:
        return np.array([self.value], dtype=np.float32)

    def dim(self) -> int:
        return 1
```

### Default Visibility Tags

| Tag | Who Can See |
|-----|-------------|
| `owner` | Only the owning agent |
| `coordinator` | Owner and its coordinator |
| `system` | System-level agents only |
| `global` | Everyone |

## State and Observation

State encapsulates an agent's internal features, while Observation is what an agent perceives:

```python
from heron.core.state import FieldAgentState
from heron.core.observation import Observation

# Create state with features
state = FieldAgentState()
state.add_feature("temperature", TemperatureFeature(25.0))

# Generate observation
obs = Observation(
    local={"state": state.vector()},
    timestamp=0.0
)
```

## Actions

Actions support both continuous and discrete components:

```python
from heron.core.action import Action
import numpy as np

action = Action()
lb = np.array([-1.0, -1.0], dtype=np.float32)
ub = np.array([1.0, 1.0], dtype=np.float32)
action.set_specs(dim_c=2, range=(lb, ub))  # 2 continuous dims

# Sample or set action
action.sample()  # Random action
action.c[:] = [0.5, -0.3]  # Set specific values
```

## Protocols

Protocols define how agents coordinate:

### Vertical Protocols (Top-down)

```python
from heron.protocols.vertical import SetpointProtocol, PriceSignalProtocol

# Direct setpoint control
setpoint = SetpointProtocol()

# Price-based coordination
price = PriceSignalProtocol(initial_price=50.0)
```

### Horizontal Protocols (Peer-to-peer)

```python
from heron.protocols.horizontal import P2PTradingProtocol, ConsensusProtocol

# Peer trading
p2p = P2PTradingProtocol()

# Consensus algorithm
consensus = ConsensusProtocol(max_iterations=10, tolerance=0.01)
```

## Message Broker

For distributed execution, agents communicate through a MessageBroker:

```python
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

# Agents subscribe to channels
broker.subscribe("control_signals", callback=handle_signal)

# Publish messages
broker.publish("control_signals", {"setpoint": 1.0})
```

## Environment Interface

HERON environments implement the PettingZoo `ParallelEnv` interface:

```python
from pettingzoo import ParallelEnv

class MyEnv(ParallelEnv):
    def reset(self, seed=None, options=None):
        # Return (observations, infos)
        pass

    def step(self, actions):
        # Return (observations, rewards, terminateds, truncateds, infos)
        pass
```

## Next Steps

- Learn about [Centralized vs Distributed](centralized_vs_distributed) execution modes
- Explore [Examples](../case_studies/power/examples/index) for hands-on tutorials
- Check the [API Reference](../api/index) for detailed documentation

# Example 6: Distributed Mode with Proxy Agent

This example demonstrates distributed execution mode where agents communicate via message brokers instead of direct function calls.

## What You'll Learn

- Setting up distributed execution mode
- Using Proxy for message-based coordination
- Implementing realistic communication patterns

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Message Broker                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │           pub/sub channels                         │  │
│  └───────────────────────────────────────────────────┘  │
│         ▲              ▲              ▲                  │
│         │              │              │                  │
│    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐           │
│    │ Proxy 1 │    │ Proxy 2 │    │ Proxy 3 │           │
│    │  (MG1)  │    │  (MG2)  │    │  (MG3)  │           │
│    └────┬────┘    └────┬────┘    └────┬────┘           │
│         │              │              │                  │
│    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐           │
│    │ GridAgt │    │ GridAgt │    │ GridAgt │           │
│    └─────────┘    └─────────┘    └─────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Code

```python
from heron.messaging.memory import InMemoryBroker
from heron.agents import Proxy
from powergrid.envs import MultiAgentMicrogrids

# Create message broker
broker = InMemoryBroker()

# Configure distributed mode
env_config = {
    "centralized": False,  # Enable distributed mode
    "message_broker": broker,
    "max_episode_steps": 24,
}

env = MultiAgentMicrogrids(env_config)
```

## Proxy Agent Setup

Proxy wraps actual agents for message-based communication:

```python
from heron.agents import Proxy
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

# Create proxy for each grid agent
proxy = Proxy(
    agent_id="mg1_proxy",
    actual_agent=grid_agent,
    broker=broker,
    upstream_id="system_operator",
)

# Distributed step (async)
await proxy.step_distributed()
```

## Message Flow

```
1. System Operator broadcasts price signal
   └── Message: {type: "price", value: 55.0}

2. Proxies receive and forward to actual agents
   └── Each GridAgent computes response

3. GridAgents send actions back through proxies
   └── Message: {type: "action", p_gen: 1.5, p_ess: -0.3}

4. Horizontal protocol messages (P2P trading)
   └── MG1 → MG2: {type: "offer", quantity: 0.5, price: 52.0}
   └── MG2 → MG1: {type: "accept", quantity: 0.5}
```

## Running the Example

```bash
cd case_studies/power
python examples/06_distributed_mode_with_proxy.py
```

## Key Concepts

### InMemoryBroker

For local simulation without external message systems:

```python
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

# Subscribe to channels
broker.subscribe("control", callback=handle_control)

# Publish messages
broker.publish("control", {"setpoint": 1.0})
```

### Extending to Production

The MessageBroker interface can be extended for production systems:

```python
from heron.messaging.base import MessageBroker

class KafkaBroker(MessageBroker):
    """Kafka-based message broker for production."""

    def __init__(self, bootstrap_servers: list[str]):
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)

    async def publish(self, channel: str, message: dict):
        self.producer.send(channel, json.dumps(message).encode())

    async def consume(self, channel: str) -> dict:
        for msg in self.consumer:
            if msg.topic == channel:
                return json.loads(msg.value.decode())
```

## Centralized vs Distributed Comparison

```python
# Centralized (fast, for training)
env_central = MultiAgentMicrogrids({"centralized": True})
# Observation: Full state vector
# Communication: Direct function calls

# Distributed (realistic, for deployment)
env_dist = MultiAgentMicrogrids({"centralized": False, "message_broker": broker})
# Observation: Local state + received messages
# Communication: Async message passing
```

## Performance Considerations

| Aspect | Centralized | Distributed |
|--------|-------------|-------------|
| Step time | ~10ms | ~50ms |
| Observability | Full | Local + messages |
| Scalability | Limited | High |
| Realism | Low | High |

## Next Steps

- Return to [User Guide](../../../user_guide/centralized_vs_distributed) for mode comparison
- See [API Reference](../../../api/messaging/index) for MessageBroker details

# Extending Agents

This guide covers how to create custom agents for your domain.

## Agent Base Class

All agents inherit from the base `Agent` class:

```python
from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.core.action import Action

class Agent(ABC):
    def __init__(self, agent_id: str, level: int = 1):
        self.agent_id = agent_id
        self.level = level

    @abstractmethod
    def observe(self) -> Observation:
        """Generate observation from current state."""
        pass

    @abstractmethod
    def act(self, observation: Observation, **kwargs) -> Action:
        """Process observation and return action."""
        pass

    @abstractmethod
    def reset(self, seed: int = None):
        """Reset agent state."""
        pass
```

## Creating a Custom Field Agent

Field agents are leaf-level agents with local sensing and actuation:

```python
from heron.agents import FieldAgent
from heron.core.state import FieldAgentState
from heron.core.action import Action
from heron.core.observation import Observation
import numpy as np

class TemperatureSensor(FieldAgent):
    """Custom field agent for temperature monitoring."""

    def __init__(self, agent_id: str, location: str):
        super().__init__(agent_id, level=1)
        self.location = location

        # Initialize state
        self.state = FieldAgentState()
        self.temp_feature = TemperatureFeature(20.0)
        self.state.add_feature("temperature", self.temp_feature)

        # Initialize action (setpoint control)
        self.action = Action()
        self.action.set_specs(
            dim_c=1,
            range=(np.array([15.0]), np.array([30.0]))
        )

    def observe(self) -> Observation:
        return Observation(
            local={"state": self.state.vector()},
            timestamp=self.current_time
        )

    def act(self, observation: Observation, **kwargs) -> Action:
        # Implement control logic
        current_temp = observation.local["state"][0]
        setpoint = kwargs.get("setpoint", 22.0)

        # Simple proportional control
        error = setpoint - current_temp
        self.action.c[0] = np.clip(current_temp + 0.5 * error, 15.0, 30.0)
        return self.action

    def reset(self, seed=None):
        self.temp_feature.value = 20.0
        self.action.reset()
```

## Creating a Custom Coordinator Agent

Coordinator agents manage subordinate agents:

```python
from heron.agents import CoordinatorAgent
from heron.protocols.vertical import SetpointProtocol

class ZoneController(CoordinatorAgent):
    """Coordinator for multiple temperature sensors."""

    def __init__(self, agent_id: str, sensors: dict):
        super().__init__(
            agent_id=agent_id,
            level=2,
            subordinates=sensors
        )
        self.protocol = SetpointProtocol()
        self.target_temperature = 22.0

    def observe(self) -> Observation:
        # Aggregate observations from subordinates
        local_obs = {}
        for sub_id, sub_agent in self.subordinates.items():
            sub_obs = sub_agent.observe()
            local_obs[sub_id] = sub_obs.local["state"]

        # Compute zone average
        temps = [obs[0] for obs in local_obs.values()]
        zone_avg = np.mean(temps)

        return Observation(
            local={"zone_avg": np.array([zone_avg])},
            subordinate_obs=local_obs
        )

    def act(self, observation: Observation, **kwargs) -> Action:
        # Generate setpoints for subordinates
        zone_avg = observation.local["zone_avg"][0]
        error = self.target_temperature - zone_avg

        # Distribute setpoints
        setpoints = {}
        for sub_id in self.subordinates:
            setpoints[sub_id] = self.target_temperature + 0.3 * error

        # Execute protocol
        self.protocol.execute(self.subordinates, setpoints)
        return self.action

    def reset(self, seed=None):
        for sub in self.subordinates.values():
            sub.reset(seed)
```

## Adding Custom Features

Create domain-specific features by extending `Feature`:

```python
from heron.core.feature import Feature
import numpy as np

class HumidityFeature(Feature):
    """Humidity measurement feature."""

    def __init__(self, initial_humidity: float = 50.0):
        super().__init__(visibility=["owner", "coordinator"])
        self.humidity = initial_humidity

    def vector(self) -> np.ndarray:
        return np.array([self.humidity], dtype=np.float32)

    def dim(self) -> int:
        return 1

    def update(self, new_humidity: float):
        self.humidity = np.clip(new_humidity, 0.0, 100.0)
```

## Message-Based Agents (Distributed Mode)

For distributed execution, use `Proxy`:

```python
from heron.agents import Proxy
from heron.messaging.memory import InMemoryBroker

broker = InMemoryBroker()

# Create actual agent
sensor = TemperatureSensor("sensor_1", "room_a")

# Wrap with proxy for message-based communication
proxy = Proxy(
    agent_id="sensor_1_proxy",
    actual_agent=sensor,
    broker=broker,
    upstream_id="zone_controller"
)

# Distributed step
await proxy.step_distributed()
```

## Best Practices

1. **Keep agents focused**: Each agent should have a single responsibility
2. **Use features for state**: Compose state from Features
3. **Define clear visibility**: Control information sharing with visibility tags
4. **Implement reset properly**: Ensure deterministic behavior with seeds
5. **Document action spaces**: Clearly define action dimensions and ranges

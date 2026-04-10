# Getting Started

This guide helps you get up and running with HERON quickly.

> **New to HERON?** Check out the [Glossary](glossary.md) for definitions of key terms.

## Installation

```bash
# Clone the repository
git clone https://github.com/Criss-Wang/PowerGym.git
cd PowerGym

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install HERON
pip install -e .
```

For case study support, see [Installation](user_guide/installation.md).

## Your First HERON Agent

Create a simple hierarchical agent system:

```python
from heron.agents import FieldAgent, CoordinatorAgent
from heron.core.state import FieldAgentState
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.action import Action
import numpy as np

# 1. Define a feature
class SensorFeature(Feature):
    def __init__(self, value: float = 0.0):
        super().__init__(visibility=["owner", "coordinator"])
        self.value = value

    def vector(self) -> np.ndarray:
        return np.array([self.value], dtype=np.float32)

    def dim(self) -> int:
        return 1

# 2. Create a field agent
class MySensor(FieldAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = FieldAgentState()
        self.sensor = SensorFeature(0.0)
        self.state.add_feature("sensor", self.sensor)

        self.action = Action()
        self.action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))

    def observe(self) -> Observation:
        return Observation(local={"state": self.state.vector()})

    def act(self, observation: Observation, **kwargs) -> Action:
        self.action.sample()
        return self.action

    def reset(self, seed=None):
        self.sensor.value = 0.0

# 3. Create agents
sensor1 = MySensor("sensor_1")
sensor2 = MySensor("sensor_2")

# 4. Create coordinator
coordinator = CoordinatorAgent(
    agent_id="controller",
    subordinates={"sensor_1": sensor1, "sensor_2": sensor2}
)

# 5. Run
obs = sensor1.observe()
action = sensor1.act(obs)
print(f"Observation: {obs.local}")
print(f"Action: {action.c}")
```

## Your First Environment

Create a PettingZoo-compatible environment:

```python
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

class SimpleEnv(ParallelEnv):
    metadata = {"name": "simple_env_v0"}

    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()

        # Define spaces
        self.observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, (2,), np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Box(-1, 1, (1,), np.float32)
            for a in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        obs = {a: np.zeros(2, dtype=np.float32) for a in self.agents}
        return obs, {}

    def step(self, actions):
        obs = {a: np.random.randn(2).astype(np.float32) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        terminateds = {a: False for a in self.agents}
        terminateds["__all__"] = False
        truncateds = {a: False for a in self.agents}
        truncateds["__all__"] = False
        return obs, rewards, terminateds, truncateds, {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

# Run environment
env = SimpleEnv()
obs, info = env.reset()

for _ in range(10):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
```

## Runnable Example

For a complete, runnable example without domain-specific dependencies, see:

```bash
python examples/00_hello_world.py
```

This demonstrates:
- Creating custom features with visibility control
- Building field agents and coordinators
- Running a simple hierarchical control simulation

## Next Steps

- [Key Concepts](key_concepts.md) - Understand HERON's core abstractions
- [Glossary](glossary.md) - Definitions of HERON terminology
- [Event-Driven Execution](user_guide/event_driven_execution.md) - Testing with realistic timing
- [User Guide](user_guide/index.rst) - Detailed usage documentation
- [Case Studies](case_studies/index.rst) - See HERON in action with PowerGrid
- [API Reference](api/index.rst) - Full API documentation

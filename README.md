# HERON: Hierarchical Environments for Realistic Observability in Networks

A **domain-agnostic Multi-Agent Reinforcement Learning (MARL) framework** for hierarchical control systems with realistic observability constraints.

---

## Why HERON?

| Challenge | HERON Solution |
|-----------|----------------|
| Flat agent structures don't scale | **3-level hierarchy**: Field → Coordinator → System |
| Full observability is unrealistic | **Declarative visibility** controls what each agent sees |
| Training ≠ deployment timing | **Dual-mode execution**: synchronous training, event-driven testing |
| Domain-specific code everywhere | **Pluggable protocols** for any coordination pattern |

---

## Architecture

```
                    ┌─────────────────┐
         L3        │   SystemAgent    │        System-wide coordination
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
 L2 │ Coordinator A │ │ Coordinator B │ │ Coordinator C │   Regional coordination
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
      ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
      ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
 L1 ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐   Local devices
    │F1 │ │F2 │ │F3 │ │F4 │ │F5 │ │F6 │ │F7 │ │F8 │ │F9 │
    └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
```

### Project Structure

```
heron/                          # Core framework
├── agents/                     # Agent hierarchy (Field, Coordinator, System, Proxy)
├── core/                       # Action, Observation, State, Feature, Policy
├── protocols/                  # Vertical & horizontal coordination protocols
├── envs/                       # Environment base classes & framework adapters
├── messaging/                  # Message broker (InMemory, extensible)
└── scheduling/                 # Event-driven simulation (ScheduleConfig, EventScheduler)

case_studies/power/             # Power grid case study (PandaPower integration)
examples/                       # Framework-level tutorials
docs/                           # Design docs & API reference
```

---

## Installation

```bash
git clone https://github.com/Criss-Wang/PowerGym.git
cd PowerGym
python3 -m venv .venv && source .venv/bin/activate

# Choose your installation
pip install -e .                    # Core framework only
pip install -e ".[powergrid]"       # + Power grid case study
pip install -e ".[multi_agent]"     # + RLlib, PettingZoo
pip install -e ".[all]"             # Everything (includes dev tools)
```

**Requirements:** Python >= 3.10

---

## Quick Start

### 1. Define Features with Visibility Control

Features are the building blocks of agent state. Each feature declares who can observe it.

```python
from dataclasses import dataclass
from typing import ClassVar, Sequence
from heron.core import Feature

@dataclass(slots=True)
class BatterySOC(Feature):
    visibility: ClassVar[Sequence[str]] = ["owner", "upper_level"]
    soc: float = 0.5          # State of charge (0-1)
    capacity: float = 100.0   # kWh

@dataclass(slots=True)
class PowerOutput(Feature):
    visibility: ClassVar[Sequence[str]] = ["public"]  # Everyone can see
    p_kw: float = 0.0
    q_kvar: float = 0.0
```

Visibility options: `"public"`, `"owner"`, `"upper_level"`, `"system"`

### 2. Create Agents

Extend `FieldAgent` and implement four required methods:

```python
from heron.agents import FieldAgent
from heron.core import Action, Feature
from typing import List, Any
import numpy as np

class Battery(FieldAgent):
    def set_action(self, action: Any) -> None:
        """Store the action received from policy or coordinator."""
        self.action.c = np.array(action, dtype=np.float32)

    def set_state(self, **kwargs) -> None:
        """Update features from simulation results."""
        self.state.update_features(**kwargs)

    def apply_action(self):
        """Apply action effects to internal state."""
        charge_kw = self.action.c[0]
        soc_feature = self.state.features["BatterySOC"]
        soc_feature.set_values(soc=soc_feature.soc + charge_kw * 0.01)

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward from local state dict."""
        soc = local_state["BatterySOC"]["soc"]
        return -abs(soc - 0.5)  # Penalize deviation from 50%

    def init_action(self, features: List[Feature] = []) -> Action:
        """Define the action space."""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
        return action
```

### 3. Build a Hierarchy

```python
from heron.agents import CoordinatorAgent, SystemAgent
from heron.protocols import VerticalProtocol

# Create field agents with features
batteries = [
    Battery(
        agent_id=f"bat_{i}",
        features=[BatterySOC(), PowerOutput()],
    )
    for i in range(3)
]

# Coordinator manages field agents via a protocol
coordinator = CoordinatorAgent(
    agent_id="microgrid_0",
    subordinates={b.agent_id: b for b in batteries},
    protocol=VerticalProtocol(),  # Decomposes joint action → per-device actions
)
```

### 4. Run Training or Testing

```python
# Synchronous mode (RL training) — all agents step together
obs, info = env.reset()
obs, rewards, terminated, truncated, infos = env.step(actions)

# Event-driven mode (realistic testing) — heterogeneous timing
from heron.scheduling import ScheduleConfig, JitterType, EpisodeAnalyzer

coordinator.schedule_config = ScheduleConfig.with_jitter(
    tick_interval=5.0,       # Ticks every 5 seconds
    obs_delay=0.1,           # 100ms observation latency
    act_delay=0.2,           # 200ms actuation delay
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,        # 10% timing jitter
)
result = env.run_event_driven(episode_analyzer=EpisodeAnalyzer(), t_end=3600.0)
```

---

## Key Concepts

### Execution Modes

HERON supports two execution modes for the **same** agents and environments:

| Mode | Use Case | Timing | API |
|------|----------|--------|-----|
| **Synchronous** | RL training | All agents step together | `env.step(actions)` |
| **Event-Driven** | Realistic testing | Heterogeneous tick rates + delays | `env.run_event_driven(...)` |

RL training assumes all agents observe and act simultaneously. Real systems don't work that way — a field sensor may tick every 100ms while a coordinator ticks every 5s, observations are delayed, and actuators have response times. Event-driven mode tests trained policies under these realistic conditions without retraining.

### Protocols

Protocols define **how agents coordinate**. They combine two orthogonal concerns:

- **CommunicationProtocol** — *what* to share (messages, signals, prices)
- **ActionProtocol** — *how* to distribute actions (decomposition, proportional, auction)

```python
from heron.protocols import Protocol

class PriceGuidedProtocol(Protocol):
    def __init__(self):
        super().__init__(
            communication_protocol=PriceSignalCommunication(),
            action_protocol=ProportionalAllocation(),
        )
```

Built-in protocols: `VerticalProtocol` (joint action decomposition), `HorizontalProtocol` (peer-to-peer), `NoProtocol`.

### Environment Pattern

Create environments by implementing three abstract methods that bridge HERON agents to your physics simulator:

```python
from heron.envs import HeronEnv

class MyEnv(HeronEnv):
    def global_state_to_env_state(self, global_state):
        """Convert HERON state dict → your simulator's input format."""
        ...

    def run_simulation(self, env_state):
        """Run one step of your physics simulation."""
        ...

    def env_state_to_global_state(self, env_state):
        """Convert simulator output → HERON state dict."""
        ...
```

### Visibility System

Features declare visibility rules as a class variable. The `Proxy` automatically filters observations based on who's requesting:

| Rule | Who Can See |
|------|-------------|
| `"public"` | All agents |
| `"owner"` | The agent that owns this feature |
| `"upper_level"` | The agent's direct parent (coordinator) |
| `"system"` | The system agent (L3) |

---

## Tutorials

### Framework Tutorials (`examples/`)

| Notebook | Topic |
|----------|-------|
| [action_passing_tutorial.ipynb](examples/action_passing_tutorial.ipynb) | Action decomposition via protocols |
| [ctde_event_driven_tutorial.ipynb](examples/ctde_event_driven_tutorial.ipynb) | Full CTDE training + event-driven evaluation |

### Power Grid Tutorials (`case_studies/power/tutorials/`)

A step-by-step guide to building a complete multi-agent system from scratch:

| # | Notebook | Topic | Time |
|---|----------|-------|------|
| 01 | [Features & State](case_studies/power/tutorials/01_features_and_state.ipynb) | `Feature`, declarative visibility, state classes | 15 min |
| 02 | [Building Agents](case_studies/power/tutorials/02_building_agents.ipynb) | `FieldAgent`, `CoordinatorAgent`, `SystemAgent` hierarchy | 20 min |
| 03 | [Building Environment](case_studies/power/tutorials/03_building_environment.ipynb) | `HeronEnv`, `Proxy`, state conversion pattern | 15 min |
| 04 | [Training with RLlib](case_studies/power/tutorials/04_training_with_rllib.ipynb) | CTDE training with MAPPO, `VerticalProtocol` | 10 min |
| 05 | [Event-Driven Testing](case_studies/power/tutorials/05_event_driven_testing.ipynb) | `ScheduleConfig`, `EventScheduler`, jitter, dual-mode | 15 min |
| 06 | [Custom Protocols](case_studies/power/tutorials/06_custom_protocols.ipynb) | Composing `CommunicationProtocol` + `ActionProtocol` | 15 min |

**Total: ~90 minutes** from zero to a fully trained and realistically tested multi-agent system.

---

## Case Study: Power Grid

A complete multi-agent microgrid control case study with PandaPower integration.

```bash
pip install -e ".[powergrid]"
```

Includes IEEE 13/34/123-bus test networks, device models (Generator, ESS, Transformer), MISOCP optimization, and MAPPO training examples.

Full documentation: [case_studies/power/docs/README.md](case_studies/power/docs/README.md)

---

## Creating Your Own Project

HERON provides a project scaffolding tool:

```bash
make new-project NAME=my_project DOMAIN=my_domain
cd my_project && pip install -e ".[heron,dev]"
```

This generates:

```
my_project/
├── my_domain/
│   ├── agents/      # Your agents extending FieldAgent, CoordinatorAgent
│   ├── envs/        # Your environments extending HeronEnv
│   └── utils/
├── examples/
├── tests/
└── pyproject.toml   # Depends on heron-marl
```

To extend HERON for a new domain, you typically need to:

1. **Define features** — Subclass `Feature` with domain-specific state variables
2. **Create agents** — Extend `FieldAgent` (devices) and optionally `CoordinatorAgent` (groups)
3. **Build an environment** — Implement `HeronEnv` with your simulator
4. **Choose/create protocols** — Use built-in `VerticalProtocol` or compose your own

---

## Development

```bash
# Run tests
pytest tests/ -v                              # Core framework
pytest case_studies/power/tests/ -v           # Power grid case study

# Code quality
black heron/ && ruff check heron/             # Format & lint
mypy heron/                                   # Type check
```

---

## License

MIT License — see [LICENSE.txt](LICENSE.txt)

## Citation

If you use HERON in your research, please cite: TBD

## Contact

- **Issues**: [GitHub Issues](https://github.com/Criss-Wang/PowerGym/issues)
- **Email**: zhenlin.wang.criss@gmail.com

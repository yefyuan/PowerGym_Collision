# HERON Tutorials: Build a Case Study from Scratch

Learn HERON by building a complete multi-agent RL system for power grid control.

## What Makes HERON Different?

| Challenge | Traditional Approach | HERON Solution |
|-----------|---------------------|----------------|
| Observation filtering | Manual code per agent | `visibility: ClassVar = ['owner', 'upper_level']` |
| Coordination protocols | Hardcoded logic | Pluggable `VerticalProtocol`, `HorizontalProtocol` |
| Realistic testing | Not supported | **Dual execution modes** (sync + event-driven) |
| Agent hierarchy | Manual implementation | Built-in `FieldAgent` → `CoordinatorAgent` → `SystemAgent` |

## Prerequisites

```bash
pip install ray[rllib] pettingzoo gymnasium numpy
```

## Tutorial Overview

### Core Tutorials (Build a Complete System)

| Notebook | Topic | Time | What You'll Learn |
|----------|-------|------|-------------------|
| [01_features_and_state](01_features_and_state.ipynb) | Features & Visibility | 15 min | `Feature`, declarative visibility, three state classes |
| [02_building_agents](02_building_agents.ipynb) | Agent Hierarchy | 20 min | `FieldAgent`, `CoordinatorAgent`, `SystemAgent`, bottom-up construction |
| [03_building_environment](03_building_environment.ipynb) | Environment | 15 min | `HeronEnv`, `Proxy`, state conversion pattern |
| [04_training_with_rllib](04_training_with_rllib.ipynb) | CTDE Training | 10 min | Centralized training, `VerticalProtocol` action decomposition, event-driven evaluation |
| [05_event_driven_testing](05_event_driven_testing.ipynb) | Dual Mode | 15 min | `ScheduleConfig`, `EventScheduler`, jitter, CPS-calibrated timing |

### Advanced Tutorial (Customization & Extension)

| Notebook | Topic | Time | What You'll Learn |
|----------|-------|------|-------------------|
| [06_custom_protocols](06_custom_protocols.ipynb) | Protocols | 15 min | Protocol architecture, custom `ActionProtocol` and `CommunicationProtocol`, composing protocols |

**Total time:** ~90 minutes (reading + coding)

## Quick Start

If you just want to see the end result:

```bash
# Run the CTDE training example
jupyter notebook 04_training_with_rllib.ipynb
```

## What You'll Build

```
SimpleMultiMicrogridEnv (Environment)
├── SystemAgent (L3)
│   ├── mg_0 (CoordinatorAgent)
│   │   ├── mg_0_bat (FieldAgent - Battery)
│   │   └── mg_0_gen (FieldAgent - Generator)
│   ├── mg_1 (CoordinatorAgent)
│   │   ├── mg_1_bat (FieldAgent - Battery)
│   │   └── mg_1_gen (FieldAgent - Generator)
│   └── mg_2 (CoordinatorAgent)
│       ├── mg_2_bat (FieldAgent - Battery)
│       └── mg_2_gen (FieldAgent - Generator)
└── Proxy (state management, visibility filtering)
```

## Key HERON Contributions (Demonstrated in Tutorials)

### 1. Declarative Visibility (Tutorial 01)
No manual filtering—features declare who can see them:
```python
@dataclass(slots=True)
class BatterySOC(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level']  # Automatic filtering
    soc: float = 0.5
```

### 2. Agent-Centric Architecture (Tutorial 02)
Agents are first-class citizens with state, timing, and hierarchy:
```python
class SimpleBattery(FieldAgent):
    def init_action(self, features) -> Action: ...   # Define action space
    def set_action(self, action): ...                 # Store action
    def set_state(self, **kwargs): ...                # Update features
    def apply_action(self): ...                       # Call set_state()
    def compute_local_reward(self, local_state): ...  # Reward from local state
```

### 3. HeronEnv Pattern (Tutorial 03)
Three abstract methods bridge HERON agents to your physics simulation:
```python
class MyEnv(HeronEnv):
    def global_state_to_env_state(self, global_state) -> EnvState: ...
    def run_simulation(self, env_state) -> EnvState: ...
    def env_state_to_global_state(self, env_state) -> Dict: ...
```

### 4. CTDE Training (Tutorial 04)
Coordinators own policies; `VerticalProtocol` decomposes joint actions into per-device actions:
```python
# Coordinator policy computes joint action
joint_action = policy.forward(aggregated_obs)

# Protocol distributes to devices
_, device_actions = coordinator.protocol.coordinate(
    coordinator_state=..., coordinator_action=joint_action, ...
)
```

### 5. Dual Execution Modes (Tutorial 05 — Key Differentiator)
Train fast, test realistically—**this cannot be achieved by wrapping PettingZoo**:
```python
# Training: synchronous (fast)
env.step(actions)

# Testing: event-driven (realistic timing, delays, jitter)
env.run_event_driven(episode_analyzer=analyzer, t_end=300.0)
```

### 6. Pluggable Protocols (Tutorial 06)
Compose `CommunicationProtocol` + `ActionProtocol` independently:
```python
class PriceGuidedProtocol(Protocol):
    def __init__(self):
        super().__init__(
            communication_protocol=PriceSignalCommunication(),  # WHAT to share
            action_protocol=ProportionalAction(weights=...),    # HOW to distribute
        )
```

## Comparison with Production Code

| Tutorial Code | Production Code (`powergrid/`) |
|---------------|-------------------------------|
| 3 microgrids, 2 devices each | 3 microgrids + DSO, heterogeneous devices |
| 2-3 features per agent | 14+ features |
| Simplified physics | PandaPower integration |
| VerticalProtocol + custom examples | 4 swappable protocols |
| Sync + event-driven | Full dual-mode with CPS timing |

## Next Steps

After completing the core tutorials (01-05):

1. **Explore custom protocols**: Tutorial 06 shows how to build and compose protocols
2. **Explore the full case study**: `powergrid/` directory
3. **Run production training**: `python -m powergrid.scripts.mappo_training --test`
4. **Test with realistic timing**: Use Tutorial 05 patterns for event-driven validation
5. **Add your own domain**: Use these patterns for traffic, robotics, etc.

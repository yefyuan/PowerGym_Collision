# HMARL-CBF Examples

Two end-to-end examples demonstrating Hierarchical Multi-Agent Reinforcement Learning (HMARL) with optional Control Barrier Function (CBF) safety constraints, built on the HERON framework with RLlib.

Both cases share the **same agents and features** -- the only differences are the policy structure (IPPO vs MAPPO) and whether CBF safety is enabled.

## Quick Start

```bash
cd examples/hmarl-cbf

# Case 1: IPPO with independent fleet policies (no CBF)
python case1.py

# Case 2: MAPPO with shared policy + CBF safety filter
python case2.py
```

Each script trains for 10 PPO iterations (~30s) and runs a final event-driven evaluation.

## File Structure

```
hmarl-cbf/
├── features.py     # Feature providers (observations) for all agents
├── agents.py       # Agent class definitions (TransportDrone + TransportCoordinator)
├── env_physics.py  # Simulation / physics step functions (with and without CBF)
├── case1.py        # Case 1 entry point (IPPO, no CBF)
├── case2.py        # Case 2 entry point (MAPPO + CBF)
└── README.md
```

| File | Responsibility |
|------|---------------|
| `features.py` | Defines `Feature` subclasses (`DronePositionFeature`, `FleetSafetyFeature`) that represent the observable state of each agent. Features are the bridge between raw simulation state and what agents "see". |
| `agents.py` | Defines `TransportDrone` (field agent) and `TransportCoordinator` (coordinator agent). Each agent declares its action space, how actions modify state, and how local rewards are computed. |
| `env_physics.py` | Contains simulation functions that advance the world between agent steps: wind drag, state aggregation, and (in Case 2) the CBF safety filter. |
| `case*.py` | Entry points that wire agents, features, and physics together into an RLlib training config, then train and evaluate. |

## Agent Hierarchy

Both cases share the same 2-level hierarchy with identical agents:

```
SystemAgent (root)
├── fleet_0 (TransportCoordinator)    <-- RL policy lives here
│   ├── drone_0_0 (TransportDrone)    <-- receives sub-action from coordinator
│   ├── drone_0_1 (TransportDrone)
│   └── drone_0_2 (TransportDrone)
└── fleet_1 (TransportCoordinator)
    ├── drone_1_0 (TransportDrone)
    ├── drone_1_1 (TransportDrone)
    └── drone_1_2 (TransportDrone)
```

- **TransportCoordinator** owns the RL policy. Its action is a 3D vector (one velocity command per drone).
- **TransportDrone** receives a 1D sub-action from its coordinator via `VerticalProtocol`.
- Rewards flow **bottom-up**: each drone computes a local reward; the coordinator sums them.

## Execution Flow

The following sequence repeats each environment step:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RLlib samples actions from coordinator policies          │
│    (IPPO: independent policies  /  MAPPO: shared policy)    │
├─────────────────────────────────────────────────────────────┤
│ 2. HeronEnvRunner.step()                                    │
│    ├── Coordinator action is split into per-drone sub-actions│
│    │   via VerticalProtocol                                 │
│    ├── Each drone calls apply_action() to update its state  │
│    └── env calls simulation(agent_states) for physics       │
├─────────────────────────────────────────────────────────────┤
│ 3. Simulation function (env_physics.py)                     │
│    ├── Applies wind drag (both cases)                       │
│    ├── [Case 2 only] CBF safety filter corrects positions   │
│    └── Aggregates drone states → coordinator features       │
├─────────────────────────────────────────────────────────────┤
│ 4. Observations are built from updated features             │
│    ├── DronePositionFeature → drone state                   │
│    └── FleetSafetyFeature → fed to coordinator's policy     │
├─────────────────────────────────────────────────────────────┤
│ 5. Rewards are computed bottom-up                           │
│    ├── Each drone: compute_local_reward(own features)       │
│    └── Coordinator: sum of subordinate rewards              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Features (observations)              Actions
═══════════════════════              ═══════
DronePositionFeature (x, y)          Coordinator: Box(-1, 1, (3,))
FleetSafetyFeature (sep, progress)         │
        │                                  ▼
        ▼                          ┌──────────────┐
  ┌──────────────┐   sub-action    │  Transport   │
  │  Transport   │ ◄───────────── │  Coordinator  │
  │    Drone     │     (1D)       │  (RL Policy)  │
  └──────────────┘                └──────────────┘
        │                                ▲
        │  local reward                  │ aggregated reward
        └────────────────────────────────┘

  Between steps:
  ┌──────────────────┐
  │  simulation()    │ ◄── env_physics.py
  │  - wind drag     │
  │  - CBF filter    │     (Case 2 only)
  │  - fleet agg.    │
  └──────────────────┘
```

## Case Comparison

Both cases use `TransportDrone`, `TransportCoordinator`, `DronePositionFeature`, and `FleetSafetyFeature`. The differences are:

|  | Case 1 (IPPO) | Case 2 (MAPPO + CBF) |
|--|---------------|----------------------|
| **Policy structure** | Independent policy per fleet | Single shared policy |
| **Safety filter** | None -- collisions possible | CBF enforces minimum separation |
| **Simulation** | `case1_simulation` (wind drag only) | `case2_simulation` (wind drag + CBF) |
| **Reward** | Same: `x - 0.5*x^2` | Same: `x - 0.5*x^2` |
| **Key takeaway** | How IPPO handles heterogeneous fleet behavior | How CBF provides safety guarantees without reward shaping |

## Extending These Examples

To create your own case:

1. **Define features** in `features.py` -- subclass `Feature` with your observable state fields.
2. **Define agents** in `agents.py` -- subclass `FieldAgent` / `CoordinatorAgent` with action specs, state transitions, and reward functions.
3. **Define physics** in `env_physics.py` -- write a `simulation(agent_states) -> agent_states` function.
4. **Wire it up** in a new `case*.py` -- configure RLlib with your agents, features, and simulation.

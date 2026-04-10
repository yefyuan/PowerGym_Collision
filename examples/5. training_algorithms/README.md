# Level 5: Training Algorithms

**From manual training loops to full RLlib integration.**

## What You'll Learn

| Concept | Where |
|---------|-------|
| Policy ABC with decorators | `policy_and_training.py` Part 1 |
| `@obs_to_vector` / `@vector_to_action` | `policy_and_training.py` Part 1 |
| Manual REINFORCE training | `policy_and_training.py` Part 2-3 |
| IPPO vs MAPPO comparison | `policy_and_training.py` Part 4 |
| `RLlibBasedHeronEnv` setup | `rllib_integration.py` Part 1 |
| IPPO / MAPPO RLlib config | `rllib_integration.py` Part 2-3 |
| `HeronEnvRunner` + event-driven eval | `rllib_integration.py` Part 4 |
| Live RLlib training loop | `rllib_integration.py` Part 5 |

## Prerequisites

- **Level 3**: Building Environments (agents, features, SimpleEnv)
- **Level 4**: Protocols & Coordination (for understanding coordinator actions)

For `rllib_integration.py` Part 5 (live training):
```bash
pip install "ray[rllib]" torch
```

## Architecture

```
Training Stack:

  Manual (no framework)          RLlib Integration
  ========================       ============================

  Policy ABC                     RLlibBasedHeronEnv
    forward(obs) -> action         wraps HeronEnv as MultiAgentEnv
    @obs_to_vector                 flattens Observation -> numpy
    @vector_to_action              exposes action_space to RLlib

  Your training loop             PPOConfig / MAPPO
    collect transitions            .multi_agent(policies={...})
    compute returns                .env_runners(HeronEnvRunner)
    policy.update(...)             algo.train()

                                 HeronEnvRunner
                                   event-driven evaluation
                                   RLlibModuleBridge -> Policy
```

## IPPO vs MAPPO

```
IPPO (Independent PPO)           MAPPO (Shared PPO)
========================          ========================

  room_a -> policy_A              room_a -> shared_policy
  room_b -> policy_B              room_b -> shared_policy

  + Specialization                + 2x sample efficiency
  + Heterogeneous agents          + Consistent behavior
  - More parameters               - Can't specialize
```

**Rule of thumb**: Use MAPPO for homogeneous agents (same type, same dynamics).
Use IPPO when agents have different roles or contexts.

## File Structure

```
5. training_algorithms/
├── README.md
├── policy_and_training.py     # Policy ABC + manual REINFORCE
└── rllib_integration.py       # RLlib adapter + live training
```

## Running

```bash
cd "examples/5. training_algorithms"

# Manual training (no extra deps)
python policy_and_training.py

# RLlib integration (requires ray[rllib] + torch)
python rllib_integration.py
```

## Key API

### Policy ABC

```python
from heron.core.policies import Policy, obs_to_vector, vector_to_action

class MyPolicy(Policy):
    observation_mode = "local"  # "local", "full", or "global"

    @obs_to_vector       # Observation -> np.ndarray
    @vector_to_action    # np.ndarray -> Action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        return my_network(obs_vec)
```

### RLlib Adapter

```python
from heron.adaptors.rllib import RLlibBasedHeronEnv
from heron.adaptors.rllib_runner import HeronEnvRunner

config = (
    PPOConfig()
    .environment(env=RLlibBasedHeronEnv, env_config={
        "agents": [...],
        "coordinators": [...],
        "simulation": my_sim_func,
        "max_steps": 50,
    })
    .env_runners(env_runner_cls=HeronEnvRunner)
    .evaluation(
        evaluation_config=HeronEnvRunner.evaluation_config(t_end=100.0),
    )
)
```

## Next Steps

- **Level 6**: Event-Driven Simulation -- ScheduleConfig, EventScheduler, dual-mode execution

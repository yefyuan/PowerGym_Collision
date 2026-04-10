# Example 5: MAPPO Training

This example demonstrates production-ready training of Multi-Agent PPO (MAPPO) on cooperative multi-agent microgrids with RLlib.

## What You'll Learn

- Setting up RLlib for multi-agent training
- MAPPO vs IPPO for cooperative tasks
- Shared rewards for cooperation
- Experiment tracking and checkpointing

## Architecture

```
RLlib (PPO Algorithm)
└── ParallelPettingZooEnv (Wrapper)
    └── MultiAgentMicrogrids (PowerGrid)
        ├── GridAgent MG1
        ├── GridAgent MG2
        └── GridAgent MG3
```

## Quick Start

```bash
# Install dependencies
pip install "ray[rllib]==2.9.0"

# Quick test run
cd case_studies/power
python examples/05_mappo_training.py --test

# Full training
python examples/05_mappo_training.py --iterations 100 --num-workers 4
```

## Code

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids

def env_creator(env_config):
    env = MultiAgentMicrogrids(env_config)
    return ParallelPettingZooEnv(env)

# Configure PPO
config = (
    PPOConfig()
    .environment(env="multi_agent_microgrids", env_config=env_config)
    .training(lr=5e-5, train_batch_size=4000)
    .multi_agent(
        policies={"shared_policy": (None, obs_space, act_space, {})},
        policy_mapping_fn=lambda agent_id, *args: "shared_policy",
    )
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']:.2f}")
```

## MAPPO vs IPPO

| Aspect | MAPPO (Shared Policy) | IPPO (Independent Policies) |
|--------|----------------------|----------------------------|
| Policy | Single shared network | Separate network per agent |
| Learning | Faster (shared params) | Slower (more params) |
| Cooperation | Better (implicit sharing) | Harder to coordinate |
| Best for | Homogeneous agents | Heterogeneous agents |

```bash
# MAPPO (default)
python examples/05_mappo_training.py --iterations 100

# IPPO
python examples/05_mappo_training.py --iterations 100 --independent-policies
```

## Shared Rewards

Shared rewards encourage cooperation:

```python
# Without shared rewards
rewards = {"MG1": -cost_mg1, "MG2": -cost_mg2, "MG3": -cost_mg3}

# With shared rewards (cooperation)
total_cost = cost_mg1 + cost_mg2 + cost_mg3
shared_reward = -total_cost / 3
rewards = {"MG1": shared_reward, "MG2": shared_reward, "MG3": shared_reward}
```

```bash
# Enable shared rewards
python examples/05_mappo_training.py --share-reward

# Independent rewards
python examples/05_mappo_training.py --no-share-reward
```

## Command-Line Options

```bash
python examples/05_mappo_training.py \
    --iterations 200 \        # Training iterations
    --lr 5e-5 \               # Learning rate
    --hidden-dim 256 \        # Network hidden size
    --num-workers 8 \         # Parallel workers
    --share-reward \          # Cooperative rewards
    --checkpoint-freq 10 \    # Save frequency
    --wandb \                 # Enable W&B logging
    --wandb-project my-exp    # W&B project name
```

## Experiment Tracking

Enable Weights & Biases logging:

```bash
pip install wandb
wandb login

python examples/05_mappo_training.py --wandb --wandb-project powergrid-coop
```

## Checkpointing

```bash
# Save every 10 iterations
python examples/05_mappo_training.py --checkpoint-freq 10 --checkpoint-dir ./checkpoints

# Resume from checkpoint
python examples/05_mappo_training.py --resume ./checkpoints/mappo_shared_mg3_*/checkpoint_000050
```

## Expected Output

```
================================================================
Cooperative Multi-Agent Microgrid Training with RLlib
================================================================
Experiment:        mappo_shared_mg3_20240115_143022
Policy type:       MAPPO (Shared Policy)
Shared reward:     True (encourages cooperation)
Iterations:        100
================================================================

 Iter |     Reward |       Cost | Episodes |     Steps |     Time
----------------------------------------------------------------------
    1 |    -450.25 |     450.25 |       12 |     12000 |    15.2s
    2 |    -380.10 |     380.10 |       14 |     26000 |    28.5s
  ...
  100 |    -120.50 |     120.50 |       18 |   1800000 |  2450.0s

✓ Training complete!
  Best reward achieved: -115.30
```

## Next Steps

- Try [Distributed Mode](distributed_mode) for realistic deployment
- Explore different [Coordination Protocols](../../../api/protocols/index)

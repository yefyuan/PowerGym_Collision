# PowerGrid Environments

The PowerGrid case study provides PettingZoo-compatible environments.

## MultiAgentMicrogrids

The main environment for multi-agent microgrid control.

```python
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.setups.loader import load_setup

# Load configuration
config = load_setup("ieee34_ieee13")
config.update({
    "centralized": True,
    "max_episode_steps": 96,
    "train": True,
})

env = MultiAgentMicrogrids(config)
```

### Configuration Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `centralized` | bool | Execution mode (True=centralized, False=distributed) |
| `max_episode_steps` | int | Steps per episode |
| `train` | bool | Training mode (affects data sampling) |
| `penalty` | float | Safety violation penalty coefficient |
| `share_reward` | bool | Use shared rewards across agents |

### Observation Space

Each agent receives:

```python
observation = {
    "local_state": np.array([...]),    # Device states
    "network_state": np.array([...]),  # Bus voltages, flows
    "time_features": np.array([...]),  # Hour of day, etc.
}
```

### Action Space

```python
action_space = spaces.Dict({
    "continuous": spaces.Box(low=-1, high=1, shape=(n_continuous,)),
    "discrete": spaces.MultiDiscrete([n_options_1, n_options_2, ...])
})
```

## NetworkedGridEnv

Base class for custom power grid environments.

```python
from powergrid.envs.networked_grid_env import NetworkedGridEnv

class MyPowerEnv(NetworkedGridEnv):
    def _build_net(self):
        """Build network and create agents."""
        net = IEEE13Bus("MG1")

        mg_agent = PowerGridAgent(
            net=net,
            grid_config={...},
            protocol=SetpointProtocol(),
        )

        self.possible_agents = ["MG1"]
        self.agent_dict = {"MG1": mg_agent}
        self.net = net

        return net

    def _reward_and_safety(self):
        """Compute rewards and safety violations."""
        rewards = {aid: -agent.cost for aid, agent in self.agent_dict.items()}
        safety = {aid: agent.safety for aid, agent in self.agent_dict.items()}
        return rewards, safety
```

## Environment Setups

Setups provide pre-configured environments with data.

### Loading a Setup

```python
from powergrid.setups.loader import load_setup, get_available_setups

# List available setups
print(get_available_setups())  # ['ieee34_ieee13']

# Load configuration
config = load_setup("ieee34_ieee13")
```

### Setup Structure

```
powergrid/setups/ieee34_ieee13/
├── config.yml    # Environment configuration
└── data.pkl      # Time series data
```

### config.yml Example

```yaml
grids:
  - name: MG1
    network: ieee34
    base_power: 1.0
    load_scale: 1.0
    devices:
      - type: Generator
        name: gen1
        device_state_config:
          bus: "Bus 800"
          p_max_MW: 2.0
          p_min_MW: 0.5

  - name: MG2
    network: ieee13
    devices:
      - type: ESS
        name: ess1
        device_state_config:
          bus: "Bus 633"
          e_capacity_MWh: 5.0

centralized: true
max_episode_steps: 96
penalty: 10.0
```

### Data Format

```python
# data.pkl contains:
{
    "load": np.array([...]),     # Load multipliers (T,)
    "price": np.array([...]),    # Electricity prices (T,)
    "solar": np.array([...]),    # Solar generation (T,)
    "wind": np.array([...]),     # Wind generation (T,)
}
```

## RLlib Integration

Use with RLlib for training:

```python
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

def env_creator(config):
    env = MultiAgentMicrogrids(config)
    return ParallelPettingZooEnv(env)

# Register with RLlib
from ray.tune.registry import register_env
register_env("multi_agent_microgrids", env_creator)
```

## Stable-Baselines3 Integration

Use with SB3 via wrapper:

```python
from stable_baselines3 import PPO
from supersuit import pettingzoo_env_to_vec_env_v1

env = MultiAgentMicrogrids(config)
vec_env = pettingzoo_env_to_vec_env_v1(env)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
```

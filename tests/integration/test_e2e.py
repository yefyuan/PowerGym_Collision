import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

# HERON imports
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.agents.constants import FIELD_LEVEL, COORDINATOR_LEVEL, SYSTEM_LEVEL
from heron.core.observation import Observation
from heron.core.feature import Feature
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import HeronEnv
from heron.scheduling import EventScheduler, EventType, ScheduleConfig, JitterType
from heron.scheduling.analysis import EpisodeAnalyzer


@dataclass(slots=True)
class BatteryChargeFeature(Feature):
    """Battery state of charge feature - auto-registered via FeatureMeta."""
    visibility: ClassVar[Sequence[str]] = ["public"]

    soc: float = 0.5
    capacity: float = 100.0

    def set_values(self, **kwargs: Any) -> None:
        if "soc" in kwargs:
            self.soc = np.clip(kwargs["soc"], 0.0, 1.0)
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


class BatteryAgent(FieldAgent):
    """Battery field agent - Level 1 in the hierarchy.
    """

    @property
    def soc(self) -> float:
        return self.state.features["BatteryChargeFeature"].soc

    @property
    def capacity(self) -> float:
        return self.state.features["BatteryChargeFeature"].capacity

    def init_action(self, features: List[Feature] = []):
        """Initialize action (charge/discharge rate)."""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward for this agent (reward = SOC).

        Args:
            local_state: State dict from proxy.get_local_state() with structure:
                {"BatteryChargeFeature": np.array([soc, capacity])}  # Feature vectors, not dicts!

        Returns:
            Reward value (SOC in this case)
        """
        # Extract SOC from the feature vector (first element is SOC)
        if "BatteryChargeFeature" in local_state:
            feature_vec = local_state["BatteryChargeFeature"]
            return float(feature_vec[0])  # SOC is first element
        return 0.0

    def set_action(self, action: Any) -> None:
        """Set action from Action object or compatible format."""
        if isinstance(action, Action):
            # Extract continuous action vector from Action object
            self.action.set_values(c=action.c)
        else:
            # Direct value (numpy array or dict)
            self.action.set_values(action)

    def set_state(self) -> None:
        """Define state: battery SOC and capacity."""
        new_soc = self.soc + self.action.c[0] * 0.01
        self.state.features["BatteryChargeFeature"].set_values(soc=new_soc)

    def apply_action(self):
        self.set_state()

    
class ZoneCoordinator(CoordinatorAgent):
    pass


class GridSystemAgent(SystemAgent):
    pass

class EnvState:
    def __init__(self, battery_soc: float = 0.5):
        self.battery_soc = battery_soc

class EnergyManagementEnv(HeronEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        env_state.battery_soc = np.clip(env_state.battery_soc, 0.0, 1.0)
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert env_state to global_state format.

        Constructs global state PURELY from env_state, not from agent internal states.
        This maintains proper separation of concerns between environment and agents.

        Args:
            env_state: Environment state after simulation

        Returns:
            Dict with structure: {"agent_states": {agent_id: state_dict, ...}}
        """
        # Construct global state purely from env_state
        agent_states = {}

        # Map agent level to state type
        level_to_state_type = {
            FIELD_LEVEL: "FieldAgentState",
            COORDINATOR_LEVEL: "CoordinatorAgentState",
            SYSTEM_LEVEL: "SystemAgentState"
        }

        # Create state dicts for battery agents based on simulation results
        for agent_id, agent in self.registered_agents.items():
            # Only create states for field-level battery agents
            if hasattr(agent, 'level') and agent.level == FIELD_LEVEL and 'battery' in agent_id:
                agent_states[agent_id] = {
                    "_owner_id": agent_id,
                    "_owner_level": agent.level,
                    "_state_type": level_to_state_type[agent.level],
                    "features": {
                        "BatteryChargeFeature": {
                            "soc": env_state.battery_soc,
                            "capacity": 100.0
                        }
                    }
                }

        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        # Access the nested agent_states dict
        agent_states = global_state.get("agent_states", {})

        # Extract SOC from any battery agent's state dict
        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", {})
            if 'battery' in agent_id and "BatteryChargeFeature" in features:
                battery_feature = features["BatteryChargeFeature"]
                return EnvState(battery_soc=battery_feature.get("soc", 0.5))

        # Fallback to default SOC if no battery state found
        return EnvState(battery_soc=0.5)
    


# Neural Policy for CTDE Demo
# TODO: Use RLlib or SB3 for more efficient implementations
class SimpleMLP:
    """Simple MLP for value function approximation."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def update(self, x, target, lr=0.01):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        d_out = (out - target) * (1 - out**2)
        self.W2 -= lr * np.outer(h, d_out)
        self.b2 -= lr * d_out
        d_h = d_out @ self.W2.T
        d_h[h <= 0] = 0
        self.W1 -= lr * np.outer(x, d_h)
        self.b1 -= lr * d_h


class ActorMLP(SimpleMLP):
    """Actor network with tanh output for bounded actions."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        # Override W2 initialization with smaller weights for actor
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        # Initialize bias to encourage charging (positive actions)
        # tanh(0.5) ≈ 0.46, so initial actions will be around +0.46 (charging)
        self.b2 = np.ones(output_dim) * 0.5

    def update(self, x, action_taken, advantage, lr=0.01):
        """Update actor using policy gradient.

        Args:
            x: Observation vector
            action_taken: Action that was taken
            advantage: Advantage estimate (return - baseline)
            lr: Learning rate
        """
        # Forward pass
        h = np.maximum(0, x @ self.W1 + self.b1)
        current_action = np.tanh(h @ self.W2 + self.b2)

        # Policy gradient: push toward good actions
        # If advantage > 0: move toward action_taken
        # If advantage < 0: move away from action_taken
        error = current_action - action_taken
        grad_scale = advantage * (1 - current_action**2)  # tanh derivative

        # Backprop through actor
        d_W2 = np.outer(h, grad_scale * error)
        d_b2 = grad_scale * error
        d_h = (grad_scale * error) @ self.W2.T
        d_h[h <= 0] = 0  # ReLU derivative
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        # Update weights
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2.flatten()
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1.flatten()


class NeuralPolicy(Policy):
    """Neural network policy for battery control.

    Uses decorators for clean observation/action conversion.
    Methods work directly with numpy arrays!

    Architecture:
        obs (2D local) -> hidden (32) -> action (1D, tanh activation)

    The policy learns to maximize SOC by outputting positive actions (charge).
    Uses LOCAL-ONLY observations via observation_mode for decentralized execution.
    """
    observation_mode = "local"  # Use only local observations (agent's own state)

    def __init__(self, obs_dim, action_dim=1, hidden_dim=32, seed=42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)
        self.hidden_dim = hidden_dim

        # Actor network for policy (outputs actions in [-1, 1])
        self.actor = ActorMLP(obs_dim, hidden_dim, action_dim, seed)

        # Critic network for value estimation
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)

        # Exploration noise (decreases over training)
        # Start with lower noise since we have a good initial bias
        self.noise_scale = 0.15

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute action with exploration noise.

        Decorators auto-convert: observation → obs_vec (local-only) → action_vec → Action
        """
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_vec, -1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute action without exploration noise."""
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        """Estimate value of current state."""
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs, action_taken, advantage, lr=0.01):
        """Update policy using policy gradient with advantage."""
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs, target, lr=0.01):
        """Update critic to better estimate values."""
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate=0.995, min_noise=0.05):
        """Decay exploration noise over training."""
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)


# CTDE Training Loop

def train_ctde(env: HeronEnv, num_episodes=100, steps_per_episode=50, gamma=0.99, lr=0.01):
    """Train policies using CTDE with policy gradient.
    
    Uses the agent hierarchy: env.step() -> GridSystemAgent -> ZoneCoordinator -> BatteryAgent
    Each agent computes its own reward (SOC) and the policy learns to maximize it.
    """
    # Get agent IDs from registered agents that have action spaces
    agent_ids = [aid for aid, agent in env.registered_agents.items() if agent.action_space is not None]
    obs, _ = env.reset(seed=0)
    # Get obs_dim from local state only (agent's own features)
    # With visibility filtering, global_info may include other agents' public features
    # For training, we use only local observations (each agent's own state)
    first_obs = obs[agent_ids[0]]
    local_vec = list(first_obs.local.values())[0] if first_obs.local else np.array([])
    obs_dim = local_vec.shape[0] if hasattr(local_vec, 'shape') else 0
    print(f"Training with obs_dim={obs_dim} (local state only)")

    policies = {aid: NeuralPolicy(obs_dim=obs_dim, seed=42 + i) for i, aid in enumerate(agent_ids)}
    returns_history, soc_history = [], []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        trajectories = {aid: {"obs": [], "actions": [], "rewards": []} for aid in agent_ids}
        episode_return, soc_values = 0.0, []

        for step in range(steps_per_episode):
            actions = {}
            # Only process agents that have policies (field agents with action spaces)
            for aid in agent_ids:
                obs_value = obs[aid]
                # Extract local state only for training (ignore global_info)
                if isinstance(obs_value, Observation):
                    # Get local state vector (first feature in local dict)
                    local_features = list(obs_value.local.values())
                    obs_vec = local_features[0] if local_features else np.array([])
                    observation = Observation(timestamp=step, local=obs_value.local)
                else:
                    # obs_value is a numpy array from step() - use only first 2 elements (local state)
                    obs_vec = obs_value[:2] if len(obs_value) > 2 else obs_value
                    observation = Observation(timestamp=step, local={"obs": obs_vec})

                action = policies[aid].forward(observation)
                actions[aid] = action  # Pass Action object, not dict
                # Store the vector form for training
                trajectories[aid]["obs"].append(obs_vec)
                trajectories[aid]["actions"].append(action.c.copy())

            obs, rewards, terminated, _, info = env.step(actions)

            # Only process rewards and SOC for agents we're training
            for aid in agent_ids:
                if aid in rewards:
                    trajectories[aid]["rewards"].append(rewards[aid])
                    episode_return += rewards[aid]
                # Get SOC from observations (first element of vector is soc)
                if aid in obs:
                    obs_value = obs[aid]
                    # Handle both Observation objects and numpy arrays
                    if isinstance(obs_value, Observation):
                        soc_values.append(obs_value.vector()[0])
                    else:
                        soc_values.append(obs_value[0])  # Assume numpy array
            
            # Check if all agents are terminated (or if __all__ key exists and is True)
            if terminated.get("__all__", False) or all(terminated.get(aid, False) for aid in agent_ids):
                break

        # Update policies using advantage estimation
        for aid, traj in trajectories.items():
            if not traj["rewards"]:
                continue
            
            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns = np.array(returns)
            
            # Use advantage = return - baseline (critic estimate)
            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policies[aid].get_value(Observation(timestamp=t, local={"obs": obs_t}))
                advantage = returns[t] - baseline
                policies[aid].update(obs_t, traj["actions"][t], advantage, lr=lr)
                policies[aid].update_critic(obs_t, returns[t], lr=lr)
            policies[aid].decay_noise()

        returns_history.append(episode_return)
        soc_history.append(np.mean(soc_values))
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1:3d}: return={episode_return:.1f}, avg_SOC={np.mean(soc_values):.1%}")

    return policies, returns_history, soc_history

battery_agent_1 = BatteryAgent(
    agent_id="battery_1",
    features=[BatteryChargeFeature(soc=0.5, capacity=100.0)]
)
battery_agent_2 = BatteryAgent(
    agent_id="battery_2",
    features=[BatteryChargeFeature(soc=0.5, capacity=100.0)]
)
zone_coordinator = ZoneCoordinator(agent_id="zone_1", subordinates={"battery_1": battery_agent_1, "battery_2": battery_agent_2})
grid_system_agent = GridSystemAgent(agent_id="system_agent", subordinates={"zone_1": zone_coordinator})

# Configure environment with custom settings
scheduler_config = {
    "start_time": 0.0,
    "time_step": 1.0,
}

message_broker_config = {
    "buffer_size": 1000,
    "max_queue_size": 100,
}

simulation_wait_interval = 0.01  # 10ms wait between simulation steps

env = EnergyManagementEnv(
    system_agent=grid_system_agent,
    scheduler_config=scheduler_config,
    message_broker_config=message_broker_config,
    simulation_wait_interval=simulation_wait_interval,
)

# Run training
print("CTDE Training with Policy Gradient + Critic Baseline")
print("Reward = SOC (policy learns to charge batteries)")
print("Observations: [soc, capacity]\n")

policies, returns, avg_socs = train_ctde(env, num_episodes=100, steps_per_episode=50, lr=0.02)

print(f"\nResults:")
print(f"  Initial avg SOC: {np.mean(avg_socs[:10]):.1%}")
print(f"  Final avg SOC:   {np.mean(avg_socs[-10:]):.1%}")

# IMPORTANT: After training, attach policies to field agents for deployment
print(f"\nAttaching trained policies to field agents for deployment...")
env.set_agent_policies(policies)

# Update jitter configs for agents, make sure the intervals/delays are small enough (since we only run until t_end=300.0)
field_schedule_config = ScheduleConfig.with_jitter(
    tick_interval=5.0,    # Field agents tick every 5 seconds
    obs_delay=0.1,        # 100ms observation delay
    act_delay=0.2,        # 200ms action delay
    msg_delay=0.1,        # 100ms message delay
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,     # 10% jitter
    seed=42
)
coordinator_schedule_config = ScheduleConfig.with_jitter(
    tick_interval=10.0,   # Coordinators tick every 10 seconds
    obs_delay=0.2,
    act_delay=0.3,
    msg_delay=0.15,
    reward_delay=0.6,     # Wait for field agent reward round-trips
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=43
)
system_schedule_config = ScheduleConfig.with_jitter(
    tick_interval=15.0,   # System agent ticks every 15 seconds
    obs_delay=0.3,
    act_delay=0.5,
    msg_delay=0.2,
    reward_delay=1.0,     # Wait for coordinator rewards (which wait for field agents)
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=44
)

# Apply configs to agents
battery_agent_1.schedule_config = field_schedule_config
battery_agent_2.schedule_config = field_schedule_config
zone_coordinator.schedule_config = coordinator_schedule_config
grid_system_agent.schedule_config = system_schedule_config

# Update scheduler's tick config cache (it caches during attach())
for agent_id, agent in env.registered_agents.items():
    if hasattr(agent, 'schedule_config'):
        env.scheduler._agent_schedule_configs[agent_id] = agent.schedule_config

# Reset environment to apply new tick configs
env.reset(seed=200)

episode_analyzer = EpisodeAnalyzer(verbose=True, track_data=True)
episode = env.run_event_driven(episode_analyzer=episode_analyzer, t_end=300.0)
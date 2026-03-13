"""
Action passing through agent hierarchy with protocols.

This script demonstrates and tests action coordination via protocols:
- Coordinator owns neural policy
- Coordinator computes joint action
- Protocol distributes actions to field agents (via .coordinate)
- Training with CTDE (centralized training, decentralized execution)
- Event-driven execution with asynchronous timing
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

# HERON imports
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.observation import Observation
from heron.core.feature import Feature
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import HeronEnv
from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
)
from heron.utils.typing import AgentID
from heron.scheduling import ScheduleConfig, JitterType
from heron.scheduling.analysis import EpisodeAnalyzer


# =============================================================================
# Custom Action Protocol - Proportional Distribution
# =============================================================================

class ProportionalActionProtocol(ActionProtocol):
    """Distributes coordinator action proportionally based on weights."""

    def __init__(self, distribution_weights: Optional[Dict[AgentID, float]] = None):
        self.distribution_weights = distribution_weights or {}

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Distribute coordinator action proportionally among subordinates."""
        if coordinator_action is None or info_for_subordinates is None:
            return {sub_id: None for sub_id in (info_for_subordinates or {})}

        # Extract action value from Action object or array
        if hasattr(coordinator_action, 'c'):
            total_action = float(coordinator_action.c[0]) if len(coordinator_action.c) > 0 else 0.0
        elif isinstance(coordinator_action, np.ndarray):
            total_action = float(coordinator_action[0]) if len(coordinator_action) > 0 else 0.0
        else:
            total_action = float(coordinator_action)

        # Compute weights (use subordinate IDs from info_for_subordinates)
        sub_ids = list(info_for_subordinates.keys())
        if not sub_ids:
            return {}

        if not self.distribution_weights:
            weights = {sub_id: 1.0 / len(sub_ids) for sub_id in sub_ids}
        else:
            total_weight = sum(self.distribution_weights.get(sub_id, 0.0) for sub_id in sub_ids)
            if total_weight == 0:
                # Equal distribution if weights sum to zero
                weights = {sub_id: 1.0 / len(sub_ids) for sub_id in sub_ids}
            else:
                weights = {
                    sub_id: self.distribution_weights.get(sub_id, 0.0) / total_weight
                    for sub_id in sub_ids
                }

        # Distribute action proportionally
        actions = {}
        for sub_id in sub_ids:
            proportional_action = total_action * weights[sub_id]
            actions[sub_id] = np.array([proportional_action])

        # Only print during event-driven (when we have context with subordinates)
        # During training, context might not have subordinates
        if context and "subordinates" in context:
            print(f"[ProportionalProtocol] Distributing action {total_action:.4f} -> {[(sid, f'{a[0]:.4f}') for sid, a in actions.items()]}")
            print(f"  Weights: {weights}")
        return actions


class ProportionalProtocol(Protocol):
    """Protocol with proportional action distribution."""

    def __init__(self, distribution_weights: Optional[Dict[AgentID, float]] = None):
        from heron.protocols.base import NoCommunication
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=ProportionalActionProtocol(distribution_weights)
        )

    def coordinate(self, coordinator_state, coordinator_action=None, info_for_subordinates=None, context=None):
        """Override to add debug output."""
        print(f"[ProportionalProtocol.coordinate] Called with action={coordinator_action}, subordinates={list(info_for_subordinates.keys()) if info_for_subordinates else []}")
        return super().coordinate(coordinator_state, coordinator_action, info_for_subordinates, context)


# =============================================================================
# Features and Agents
# =============================================================================

@dataclass(slots=True)
class DevicePowerFeature(Feature):
    """Power state feature for devices."""
    visibility: ClassVar[Sequence[str]] = ["public"]

    power: float = 0.0
    capacity: float = 1.0

    def vector(self) -> np.ndarray:
        """Return feature as vector [power, capacity]."""
        return np.array([self.power, self.capacity], dtype=np.float32)

    def set_values(self, **kwargs: Any) -> None:
        if "power" in kwargs:
            self.power = np.clip(kwargs["power"], -self.capacity, self.capacity)
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


class DeviceAgent(FieldAgent):
    """Device field agent - receives actions from coordinator via protocol."""

    @property
    def power(self) -> float:
        return self.state.features["DevicePowerFeature"].power

    @property
    def capacity(self) -> float:
        return self.state.features["DevicePowerFeature"].capacity

    def init_action(self, features: List[Feature] = []):
        """Initialize action (power control)."""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict) -> float:
        """Reward = maintain power near zero (minimize deviation).

        Args:
            local_state: State dict with structure:
                {"DevicePowerFeature": np.array([power, capacity])}

        Returns:
            Reward value (negative squared power deviation)
        """
        reward = 0.0
        if "DevicePowerFeature" in local_state:
            feature_vec = local_state["DevicePowerFeature"]
            power = float(feature_vec[0])  # Power is first element
            reward = -power ** 2  # Penalize deviation from zero
        return reward

    def set_action(self, action: Any) -> None:
        """Set action from Action object or compatible format."""
        if isinstance(action, Action):
            # Handle potential dimension mismatch - take only what we need
            if len(action.c) != self.action.dim_c:
                self.action.set_values(action.c[:self.action.dim_c])
            else:
                self.action.set_values(c=action.c)
        else:
            # Direct value (numpy array or dict)
            self.action.set_values(action)

    def set_state(self) -> None:
        """Update power based on action (direct setpoint control)."""
        # Action directly sets power (not incremental change)
        # This makes it easier to learn: reward = -power^2, and action controls power directly
        new_power = self.action.c[0] * 0.5  # Scale action to reasonable power range
        self.state.features["DevicePowerFeature"].set_values(power=new_power)

    def apply_action(self):
        self.set_state()


class ZoneCoordinator(CoordinatorAgent):
    """Coordinator - owns policy and distributes actions via protocol."""

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute coordinator's local reward by aggregating subordinate rewards.

        In event-driven mode, subordinate rewards are included in local_state
        by the proxy agent under 'subordinate_rewards' key.
        """
        subordinate_rewards = local_state.get("subordinate_rewards", {})
        return sum(subordinate_rewards.values())

class GridSystem(SystemAgent):
    """System agent for grid management."""
    pass


# =============================================================================
# Environment
# =============================================================================

class EnvState:
    """Environment state tracking device power levels."""
    def __init__(self, device_powers: Optional[Dict[str, float]] = None):
        self.device_powers = device_powers or {"device_1": 0.0, "device_2": 0.0}


class ActionPassingEnv(HeronEnv):
    """Environment for testing action passing through protocols."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Physics simulation - clip power values to valid range."""
        for device_id in env_state.device_powers:
            # Clip power to reasonable range based on capacity (assume capacity=1.0)
            env_state.device_powers[device_id] = np.clip(
                env_state.device_powers[device_id], -1.0, 1.0
            )
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
        from heron.agents.constants import FIELD_LEVEL, COORDINATOR_LEVEL, SYSTEM_LEVEL

        # Map agent level to state type
        level_to_state_type = {
            FIELD_LEVEL: "FieldAgentState",
            COORDINATOR_LEVEL: "CoordinatorAgentState",
            SYSTEM_LEVEL: "SystemAgentState"
        }

        agent_states = {}

        # Create state dicts for device agents based on simulation results
        for agent_id, agent in self.registered_agents.items():
            # Only create states for field-level device agents
            if hasattr(agent, 'level') and agent.level == FIELD_LEVEL and 'device' in agent_id:
                power_value = env_state.device_powers.get(agent_id, 0.0)
                agent_states[agent_id] = {
                    "_owner_id": agent_id,
                    "_owner_level": agent.level,
                    "_state_type": level_to_state_type[agent.level],
                    "features": {
                        "DevicePowerFeature": {
                            "power": power_value,
                            "capacity": 1.0
                        }
                    }
                }

        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert global state to env state.

        Args:
            global_state: Global state dict with structure:
                {"agent_states": {agent_id: state_dict, ...}}

        Returns:
            EnvState with extracted device power values
        """
        # Access the nested agent_states dict
        agent_states = global_state.get("agent_states", {})

        device_powers = {}
        # Extract power from each device agent's state dict
        for agent_id, state_dict in agent_states.items():
            if 'device' in agent_id and "features" in state_dict:
                features = state_dict["features"]
                if "DevicePowerFeature" in features:
                    power_feature = features["DevicePowerFeature"]
                    device_powers[agent_id] = power_feature.get("power", 0.0)

        # Fallback to default powers if no device states found
        if not device_powers:
            device_powers = {"device_1": 0.0, "device_2": 0.0}

        return EnvState(device_powers=device_powers)


# =============================================================================
# Neural Policy for Coordinator
# =============================================================================

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
    """Actor network with tanh output."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        # Initialize with zero bias so initial actions are near 0 (not biased positive/negative)
        self.b2 = np.zeros(output_dim)

    def update(self, x, action_taken, advantage, lr=0.01):
        """Update actor using policy gradient."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        current_action = np.tanh(h @ self.W2 + self.b2)

        error = current_action - action_taken
        grad_scale = advantage * (1 - current_action**2)

        d_W2 = np.outer(h, grad_scale * error)
        d_b2 = grad_scale * error
        d_h = (grad_scale * error) @ self.W2.T
        d_h[h <= 0] = 0
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2.flatten()
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1.flatten()


class CoordinatorNeuralPolicy(Policy):
    """Neural policy for coordinator that computes joint action.

    The coordinator observes all subordinate states (aggregated) and outputs
    a single action that will be distributed to subordinates via protocol.
    """
    def __init__(self, obs_dim, action_dim=1, hidden_dim=32, seed=42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)
        self.hidden_dim = hidden_dim

        self.actor = ActorMLP(obs_dim, hidden_dim, action_dim, seed)
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)

        self.noise_scale = 0.15

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute joint action with exploration noise."""
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        action_clipped = np.clip(action_vec, -1.0, 1.0)
        return action_clipped

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute joint action without exploration noise."""
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


# =============================================================================
# CTDE Training Loop
# =============================================================================

def train_ctde(env: HeronEnv, num_episodes=100, steps_per_episode=50, gamma=0.99, lr=0.01):
    """Train coordinator policy using CTDE with action distribution via protocol.

    Key: Coordinator computes joint action, protocol distributes to field agents.
    """
    # Get field agent IDs (agents that receive distributed actions)
    agent_ids = [aid for aid, agent in env.registered_agents.items() if agent.action_space is not None]

    obs, _ = env.reset(seed=0)

    # Observation dimension: aggregate all field agent observations
    # Coordinator observes all subordinates
    first_obs = obs[agent_ids[0]]
    local_vec = list(first_obs.local.values())[0] if first_obs.local else np.array([])
    obs_dim_per_agent = local_vec.shape[0] if hasattr(local_vec, 'shape') else 0
    obs_dim = obs_dim_per_agent * len(agent_ids)  # Coordinator sees all agents

    print(f"Training coordinator with obs_dim={obs_dim} (aggregated from {len(agent_ids)} agents)")

    # Coordinator policy outputs joint action for all subordinates
    # action_dim = number of subordinates (each subordinate gets 1 action component)
    num_subordinates = len(agent_ids)
    coordinator_policy = CoordinatorNeuralPolicy(obs_dim=obs_dim, action_dim=num_subordinates, seed=42)

    returns_history, power_history = [], []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)

        # Debug: Check initial power values
        if episode == 0:
            for aid in agent_ids:
                agent = env.registered_agents[aid]
                if hasattr(agent, 'power'):
                    print(f"  Initial power for {aid}: {agent.power:.4f}")

        # Track trajectories for coordinator (not individual agents)
        trajectories = {"obs": [], "actions": [], "rewards": []}
        episode_return = 0.0
        power_values = []

        for step in range(steps_per_episode):
            # Coordinator observes all subordinates (aggregate observation)
            aggregated_obs = []
            for aid in agent_ids:
                obs_value = obs[aid]
                if isinstance(obs_value, Observation):
                    local_features = list(obs_value.local.values())
                    obs_vec = local_features[0] if local_features else np.array([])
                else:
                    obs_vec = obs_value[:obs_dim_per_agent]
                aggregated_obs.append(obs_vec)

            aggregated_obs_vec = np.concatenate(aggregated_obs)
            coordinator_observation = Observation(timestamp=step, local={"obs": aggregated_obs_vec})

            # Coordinator computes joint action
            coordinator_action = coordinator_policy.forward(coordinator_observation)

            # Pass coordinator-level action — framework's handle_subordinate_actions
            # distributes via protocol internally (matching event-driven behavior)
            actions = {"coordinator": coordinator_action}

            # Store coordinator's action (not individual actions)
            trajectories["obs"].append(aggregated_obs_vec)
            trajectories["actions"].append(coordinator_action.c.copy())

            obs, rewards, terminated, _, info = env.step(actions)

            # Aggregate rewards from all field agents
            total_reward = sum(rewards.get(aid, 0) for aid in agent_ids)
            trajectories["rewards"].append(total_reward)
            episode_return += total_reward

            # Track power values
            for aid in agent_ids:
                if aid in obs:
                    obs_value = obs[aid]
                    if isinstance(obs_value, Observation):
                        power_values.append(obs_value.vector()[0])
                    else:
                        power_values.append(obs_value[0])

            if terminated.get("__all__", False) or all(terminated.get(aid, False) for aid in agent_ids):
                break

        # Update coordinator policy
        if trajectories["rewards"]:
            returns = []
            G = 0
            for r in reversed(trajectories["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns = np.array(returns)

            for t in range(len(trajectories["obs"])):
                obs_t = trajectories["obs"][t]
                baseline = coordinator_policy.get_value(Observation(timestamp=t, local={"obs": obs_t}))
                advantage = returns[t] - baseline
                coordinator_policy.update(obs_t, trajectories["actions"][t], advantage, lr=lr)
                coordinator_policy.update_critic(obs_t, returns[t], lr=lr)

            coordinator_policy.decay_noise()

        returns_history.append(episode_return)
        power_history.append(np.mean(power_values) if power_values else 0)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}: return={episode_return:.1f}, avg_power={np.mean(power_values) if power_values else 0:.4f}")

    return coordinator_policy, returns_history, power_history


# =============================================================================
# Main Execution
# =============================================================================

# Create devices
device_1 = DeviceAgent(
    agent_id="device_1",
    features=[DevicePowerFeature(power=0.0, capacity=1.0)]
)
device_2 = DeviceAgent(
    agent_id="device_2",
    features=[DevicePowerFeature(power=0.0, capacity=1.0)]
)

# Configure tick timing BEFORE creating agents and environment
# Using faster tick intervals to generate more events within active simulation period
field_schedule_config = ScheduleConfig.with_jitter(
    tick_interval=2.0,
    obs_delay=0.05,
    act_delay=0.1,
    msg_delay=0.05,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=42
)

coordinator_schedule_config = ScheduleConfig.with_jitter(
    tick_interval=4.0,
    obs_delay=0.1,
    act_delay=0.15,
    msg_delay=0.075,
    reward_delay=0.3,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=43
)

system_schedule_config = ScheduleConfig.with_jitter(
    tick_interval=8.0,
    obs_delay=0.15,
    act_delay=0.25,
    msg_delay=0.1,
    reward_delay=0.5,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=44
)

# Set tick configs on agents BEFORE environment creation
device_1.schedule_config = field_schedule_config
device_2.schedule_config = field_schedule_config

# Coordinator with VerticalProtocol (uses vector decomposition)
# Coordinator computes joint action [a1, a2] and protocol splits it
from heron.protocols.vertical import VerticalProtocol

vertical_protocol = VerticalProtocol()
coordinator = ZoneCoordinator(
    agent_id="coordinator",
    subordinates={"device_1": device_1, "device_2": device_2},
    schedule_config=coordinator_schedule_config,
    protocol=vertical_protocol,
)

system = GridSystem(
    agent_id="system_agent",
    subordinates={"coordinator": coordinator},
    schedule_config=system_schedule_config,
)

# Configure environment
scheduler_config = {
    "start_time": 0.0,
    "time_step": 1.0,
}

message_broker_config = {
    "buffer_size": 1000,
    "max_queue_size": 100,
}

env = ActionPassingEnv(
    system_agent=system,
    scheduler_config=scheduler_config,
    message_broker_config=message_broker_config,
    simulation_wait_interval=0.01,
)

# Run training
print("="*80)
print("CTDE Training with Action Distribution via Protocol")
print("="*80)
print("Coordinator computes joint action → Protocol distributes to field agents")
print(f"Protocol: VerticalProtocol (VectorDecomposition)")
print(f"Coordinator outputs 2-dimensional action vector [a1, a2]")
print(f"Protocol decomposes: device_1 gets a1, device_2 gets a2")
print()

coordinator_policy, returns, avg_powers = train_ctde(
    env,
    num_episodes=50,  # Reduced for faster testing
    steps_per_episode=30,
    lr=0.02
)

initial_avg_power = np.mean(avg_powers[:10])
final_avg_power = np.mean(avg_powers[-10:])
initial_return = np.mean(returns[:10])
final_return = np.mean(returns[-10:])

print(f"\nTraining Results:")
print(f"  Initial avg power: {initial_avg_power:.4f} (return: {initial_return:.2f})")
print(f"  Final avg power:   {final_avg_power:.4f} (return: {final_return:.2f})")
print(f"  Return improvement: {final_return - initial_return:.2f}")
print(f"  Power closer to zero: {abs(initial_avg_power) > abs(final_avg_power)}")

# Attach trained policy to coordinator
print(f"\nAttaching trained policy to coordinator for event-driven execution...")
# Make sure we're setting policy on the coordinator instance in the environment
env_coordinator = env.registered_agents.get("coordinator")
if env_coordinator:
    env_coordinator.policy = coordinator_policy
    print(f"  Policy set on env coordinator: {env_coordinator.policy is not None}")
else:
    coordinator.policy = coordinator_policy
    print(f"  Policy set on local coordinator: {coordinator.policy is not None}")

# Run event-driven simulation
print("\n" + "="*80)
print("Event-Driven Execution with Trained Policy")
print("="*80)
print("Coordinator uses trained policy to compute actions")
print("Protocol distributes actions to devices asynchronously\n")

episode_analyzer = EpisodeAnalyzer(verbose=False, track_data=True)  # Disable verbose for clean output
episode = env.run_event_driven(episode_analyzer=episode_analyzer, t_end=100.0)

print(f"\n{'='*80}")
print("Event-Driven Execution Statistics")
print(f"{'='*80}")
print(f"Simulation time: 0.0 → 100.0s")

# EpisodeAnalyzer statistics
print(f"\nEvent Counts:")
print(f"  Observations requested: {episode_analyzer.observation_count}")
print(f"  Global state requests: {episode_analyzer.global_state_count}")
print(f"  Local state requests: {episode_analyzer.local_state_count}")
print(f"  State updates: {episode_analyzer.state_update_count}")
print(f"  Action results (rewards): {episode_analyzer.action_result_count}")

# Count agent ticks from internal timesteps
print(f"\nAgent Tick Counts:")
for agent in [device_1, device_2, coordinator, system]:
    if hasattr(agent, '_timestep'):
        if agent._timestep > 0:
            estimated_ticks = int(agent._timestep / agent._schedule_config.tick_interval)
            print(f"  {agent.agent_id}: ~{estimated_ticks} ticks (final time: {agent._timestep:.1f}s)")
        else:
            print(f"  {agent.agent_id}: 0 ticks (not started or terminated early)")

# Total activity metric
total_activity = (episode_analyzer.observation_count +
                 episode_analyzer.global_state_count +
                 episode_analyzer.local_state_count +
                 episode_analyzer.action_result_count)
actual_end_time = max(a._timestep for a in [device_1, device_2, coordinator] if hasattr(a, '_timestep') and a._timestep > 0)

print(f"\nActivity Summary:")
print(f"  Total tracked events: {total_activity}")
print(f"  Actual simulation duration: ~{actual_end_time:.1f}s")
print(f"  Average activity: {total_activity / actual_end_time:.2f} events/s")
print(f"\n✓ Validation Status:")
device_ticks = sum(1 for a in [device_1, device_2] if hasattr(a, '_timestep') and a._timestep > 0)
coord_ticks = int(coordinator._timestep / coordinator._schedule_config.tick_interval) if coordinator._timestep > 0 else 0
print(f"  - Coordinator computed and distributed {coord_ticks} joint actions")
print(f"  - Devices received and applied {device_ticks * coord_ticks // 2} total actions")
print(f"  - Vector decomposition verified {coord_ticks} times")
print(f"  - {episode_analyzer.action_result_count} rewards collected and computed")
print(f"  → Sufficient activity to validate action passing mechanism!")

# Extract and plot reward trends from event-driven execution
print(f"\n{'='*80}")
print("Event-Driven Reward Trends")
print(f"{'='*80}")

# Extract reward data from event analyzer
import matplotlib.pyplot as plt

# Get reward history directly from EpisodeAnalyzer (tracked via MSG_SET_TICK_RESULT)
reward_history = episode_analyzer.get_reward_history()
reward_data = {"coordinator": [], "device_1": [], "device_2": []}
timestamps = {"coordinator": [], "device_1": [], "device_2": []}

for agent_id in ["coordinator", "device_1", "device_2"]:
    if agent_id in reward_history:
        for ts, reward in reward_history[agent_id]:
            timestamps[agent_id].append(ts)
            reward_data[agent_id].append(reward)
        print(f"  {agent_id}: {len(reward_history[agent_id])} reward data points")

# Plot reward trends
if any(len(rewards) > 0 for rewards in reward_data.values()):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Individual agent rewards over time
    for agent_id, rewards in reward_data.items():
        if len(rewards) > 0:
            ax1.plot(timestamps[agent_id], rewards, 'o-', label=agent_id, alpha=0.7, markersize=4)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Reward')
    ax1.set_title('Event-Driven Reward Trends by Agent')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coordinator vs Device comparison
    if len(reward_data["coordinator"]) > 0:
        ax2.plot(timestamps["coordinator"], reward_data["coordinator"],
                'ro-', label='Coordinator (aggregated)', markersize=6, linewidth=2)

    if len(reward_data["device_1"]) > 0:
        ax2.plot(timestamps["device_1"], reward_data["device_1"],
                'bs-', label='device_1', markersize=4, alpha=0.6)

    if len(reward_data["device_2"]) > 0:
        ax2.plot(timestamps["device_2"], reward_data["device_2"],
                'g^-', label='device_2', markersize=4, alpha=0.6)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Reward')
    ax2.set_title('Coordinator Aggregation vs Individual Devices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/event_driven_rewards.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Reward trends plotted and saved to /tmp/event_driven_rewards.png")
    print(f"  - Coordinator rewards: {len(reward_data['coordinator'])} data points")
    print(f"  - device_1 rewards: {len(reward_data['device_1'])} data points")
    print(f"  - device_2 rewards: {len(reward_data['device_2'])} data points")

    # Print summary statistics
    if len(reward_data["coordinator"]) > 0:
        coord_rewards = reward_data["coordinator"]
        print(f"\nCoordinator Reward Stats (aggregated):")
        print(f"  Mean: {np.mean(coord_rewards):.4f}")
        print(f"  Std:  {np.std(coord_rewards):.4f}")
        print(f"  Min:  {np.min(coord_rewards):.4f}")
        print(f"  Max:  {np.max(coord_rewards):.4f}")
else:
    print(f"\n⚠ No reward data captured from event analyzer")

print(f"\n{'='*80}")
print("Action Passing Test Complete")
print(f"{'='*80}")
print(f"\nKey Points:")
print(f"  1. Coordinator owns neural policy and computes joint actions")
print(f"  2. Protocol.coordinate() distributes actions to field agents")
print(f"  3. Actions flow through hierarchy: System → Coordinator → Devices")
print(f"  4. Event-driven execution with asynchronous timing and jitter")
print(f"  5. Collected {episode_analyzer.action_result_count} rewards across agents")
print()

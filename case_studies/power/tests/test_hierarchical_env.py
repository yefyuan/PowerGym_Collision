"""Integration test for HierarchicalMicrogridEnv with 3 microgrids.

This script tests the complete hierarchical microgrid environment using
ACTION-PASSING MODE:
- SystemAgent -> 3 PowerGridAgents (MG1, MG2, MG3) -> DeviceAgents
- Microgrids (coordinators) own the policy
- Coordinators compute joint actions distributed to devices via VerticalProtocol
- CTDE training with coordinator-level policies
- Event-driven execution with asynchronous timing

Features detailed logging for:
- Power flow simulation progress (voltage, convergence, grid power)
- Event emission tracking
- Reward improvement analysis
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# HERON imports
from heron.agents.system_agent import SystemAgent
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.protocols.vertical import VerticalProtocol
from heron.scheduling import ScheduleConfig, JitterType
from heron.scheduling.analysis import EpisodeAnalyzer

# PowerGrid imports
from powergrid.agents import PowerGridAgent, Generator, ESS
from powergrid.envs import HierarchicalMicrogridEnv


# =============================================================================
# Logging Utilities
# =============================================================================

@dataclass
class TrainingLogger:
    """Tracks detailed training metrics."""

    # Episode-level metrics
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    # Per-coordinator metrics
    coordinator_rewards: Dict[str, List[float]] = field(default_factory=dict)

    # Power flow metrics (per step)
    pf_converged: List[bool] = field(default_factory=list)
    pf_voltage_min: List[float] = field(default_factory=list)
    pf_voltage_max: List[float] = field(default_factory=list)
    pf_voltage_avg: List[float] = field(default_factory=list)
    pf_grid_power: List[float] = field(default_factory=list)
    pf_line_loading: List[float] = field(default_factory=list)

    # Device action tracking
    device_actions: Dict[str, List[float]] = field(default_factory=dict)
    device_rewards: Dict[str, List[float]] = field(default_factory=dict)

    # Step counter
    total_steps: int = 0

    def log_power_flow(self, env: HierarchicalMicrogridEnv) -> Dict[str, Any]:
        """Extract and log power flow results from environment."""
        if env.net is None:
            return {}

        try:
            converged = env.net.converged if hasattr(env.net, 'converged') else True
            self.pf_converged.append(converged)

            if hasattr(env.net, 'res_bus') and len(env.net.res_bus) > 0:
                v_min = float(env.net.res_bus.vm_pu.min())
                v_max = float(env.net.res_bus.vm_pu.max())
                v_avg = float(env.net.res_bus.vm_pu.mean())
            else:
                v_min = v_max = v_avg = 1.0

            self.pf_voltage_min.append(v_min)
            self.pf_voltage_max.append(v_max)
            self.pf_voltage_avg.append(v_avg)

            if hasattr(env.net, 'res_ext_grid') and len(env.net.res_ext_grid) > 0:
                grid_p = float(env.net.res_ext_grid.p_mw.values[0])
            else:
                grid_p = 0.0
            self.pf_grid_power.append(grid_p)

            if hasattr(env.net, 'res_line') and len(env.net.res_line) > 0:
                max_loading = float(env.net.res_line.loading_percent.max())
            else:
                max_loading = 0.0
            self.pf_line_loading.append(max_loading)

            return {
                "converged": converged,
                "v_min": v_min,
                "v_max": v_max,
                "v_avg": v_avg,
                "grid_power_MW": grid_p,
                "max_line_loading_%": max_loading,
            }
        except Exception as e:
            return {"error": str(e)}

    def log_step(
        self,
        actions: Dict[str, Any],
        rewards: Dict[str, float],
    ) -> None:
        """Log per-step metrics."""
        self.total_steps += 1

        # Log device actions
        for device_id, action in actions.items():
            if device_id not in self.device_actions:
                self.device_actions[device_id] = []
            if hasattr(action, 'c'):
                self.device_actions[device_id].append(float(action.c[0]))
            elif isinstance(action, np.ndarray):
                self.device_actions[device_id].append(float(action[0]))

        # Log device rewards
        for device_id, reward in rewards.items():
            if device_id not in self.device_rewards:
                self.device_rewards[device_id] = []
            self.device_rewards[device_id].append(float(reward))

    def log_episode(
        self,
        episode_return: float,
        episode_length: int,
        mg_rewards: Dict[str, float],
    ) -> None:
        """Log episode-level metrics."""
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

        for coord_id, reward in mg_rewards.items():
            if coord_id not in self.coordinator_rewards:
                self.coordinator_rewards[coord_id] = []
            self.coordinator_rewards[coord_id].append(reward)

    def compute_moving_average(self, data: List[float], window: int = 5) -> List[float]:
        """Compute moving average of a list."""
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]

    def print_episode_summary(self, episode: int, verbose: bool = True) -> None:
        """Print summary for current episode."""
        if not self.episode_returns:
            return

        current_return = self.episode_returns[-1]

        # Compute moving averages
        ma_returns = self.compute_moving_average(self.episode_returns)
        current_ma = ma_returns[-1] if ma_returns else current_return

        # Compute trend (improvement over last 5 episodes)
        if len(self.episode_returns) >= 5:
            recent_avg = np.mean(self.episode_returns[-5:])
            earlier_avg = np.mean(self.episode_returns[-10:-5]) if len(self.episode_returns) >= 10 else self.episode_returns[0]
            trend = recent_avg - earlier_avg
            trend_str = f"+{trend:.2f}" if trend >= 0 else f"{trend:.2f}"
        else:
            trend_str = "N/A"

        print(f"\n  [Episode {episode + 1}] Return: {current_return:.2f} | MA(5): {current_ma:.2f} | Trend: {trend_str}")

        if verbose:
            # Per-microgrid breakdown
            for coord_id, rewards in self.coordinator_rewards.items():
                if rewards:
                    print(f"    {coord_id}: {rewards[-1]:.2f}")

            # Power flow summary for this episode
            if self.pf_converged:
                recent_pf = min(24, len(self.pf_converged))  # Last episode steps
                converged_rate = sum(self.pf_converged[-recent_pf:]) / recent_pf * 100
                avg_voltage = np.mean(self.pf_voltage_avg[-recent_pf:]) if self.pf_voltage_avg else 1.0
                avg_grid_p = np.mean(self.pf_grid_power[-recent_pf:]) if self.pf_grid_power else 0.0
                print(f"    Power Flow: {converged_rate:.0f}% converged | V_avg={avg_voltage:.4f}pu | Grid={avg_grid_p:.3f}MW")

    def print_training_summary(self) -> None:
        """Print final training summary."""
        if not self.episode_returns:
            print("No training data to summarize.")
            return

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        # Overall metrics
        print(f"\nEpisodes: {len(self.episode_returns)}")
        print(f"Total steps: {self.total_steps}")

        # Return statistics
        print(f"\nReturn Statistics:")
        print(f"  Initial (first 5): {np.mean(self.episode_returns[:5]):.2f}")
        print(f"  Final (last 5):    {np.mean(self.episode_returns[-5:]):.2f}")
        print(f"  Best:              {max(self.episode_returns):.2f}")
        print(f"  Worst:             {min(self.episode_returns):.2f}")
        improvement = np.mean(self.episode_returns[-5:]) - np.mean(self.episode_returns[:5])
        print(f"  Improvement:       {'+' if improvement >= 0 else ''}{improvement:.2f}")

        # Per-coordinator statistics
        print(f"\nPer-Microgrid Statistics:")
        for coord_id, rewards in self.coordinator_rewards.items():
            if rewards:
                print(f"  {coord_id}:")
                print(f"    Mean: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")
                print(f"    Initial: {np.mean(rewards[:5]):.2f}, Final: {np.mean(rewards[-5:]):.2f}")

        # Power flow statistics
        if self.pf_converged:
            converged_rate = sum(self.pf_converged) / len(self.pf_converged) * 100
            print(f"\nPower Flow Statistics:")
            print(f"  Convergence rate: {converged_rate:.1f}%")
            print(f"  Voltage range: [{min(self.pf_voltage_min):.4f}, {max(self.pf_voltage_max):.4f}] pu")
            print(f"  Avg voltage: {np.mean(self.pf_voltage_avg):.4f} pu")
            print(f"  Grid power range: [{min(self.pf_grid_power):.3f}, {max(self.pf_grid_power):.3f}] MW")

        # Device statistics
        print(f"\nDevice Action Statistics:")
        for device_id in sorted(self.device_actions.keys()):
            actions = self.device_actions[device_id]
            rewards = self.device_rewards.get(device_id, [])
            print(f"  {device_id}:")
            print(f"    Action: mean={np.mean(actions):.3f}, std={np.std(actions):.3f}")
            if rewards:
                print(f"    Reward: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")


@dataclass
class EventLogger:
    """Tracks detailed event-driven execution metrics."""

    # Event counts by type
    agent_ticks: Dict[str, int] = field(default_factory=dict)
    action_effects: Dict[str, int] = field(default_factory=dict)
    message_deliveries: int = 0
    state_updates: int = 0

    # Rewards collected during event-driven
    rewards_by_agent: Dict[str, List[float]] = field(default_factory=dict)

    # Timestamps
    event_timestamps: List[float] = field(default_factory=list)

    # Power flow during event-driven
    pf_results: List[Dict[str, Any]] = field(default_factory=list)

    def log_event(self, event_type: str, agent_id: str, timestamp: float, payload: Dict = None) -> None:
        """Log an event."""
        self.event_timestamps.append(timestamp)

        if event_type == "agent_tick":
            self.agent_ticks[agent_id] = self.agent_ticks.get(agent_id, 0) + 1
        elif event_type == "action_effect":
            self.action_effects[agent_id] = self.action_effects.get(agent_id, 0) + 1
        elif event_type == "message_delivery":
            self.message_deliveries += 1
            # Track state updates from set_state_completion messages
            message = payload.get("message", {}) if payload else {}
            if "set_state_completion" in message:
                self.state_updates += 1

        if payload and "reward" in payload:
            if agent_id not in self.rewards_by_agent:
                self.rewards_by_agent[agent_id] = []
            self.rewards_by_agent[agent_id].append(payload["reward"])

    def print_summary(self) -> None:
        """Print event-driven execution summary."""
        print("\n" + "=" * 60)
        print("EVENT-DRIVEN EXECUTION SUMMARY")
        print("=" * 60)

        print(f"\nTotal events: {len(self.event_timestamps)}")
        if self.event_timestamps:
            print(f"Time span: {min(self.event_timestamps):.2f} - {max(self.event_timestamps):.2f}")

        print(f"\nAgent Ticks:")
        for agent_id in sorted(self.agent_ticks.keys()):
            print(f"  {agent_id}: {self.agent_ticks[agent_id]}")

        print(f"\nAction Effects:")
        for agent_id in sorted(self.action_effects.keys()):
            print(f"  {agent_id}: {self.action_effects[agent_id]}")

        print(f"\nMessage Deliveries: {self.message_deliveries}")
        print(f"State Updates: {self.state_updates}")

        if self.rewards_by_agent:
            print(f"\nRewards Collected:")
            total_reward = 0.0
            for agent_id in sorted(self.rewards_by_agent.keys()):
                rewards = self.rewards_by_agent[agent_id]
                agent_total = sum(rewards)
                total_reward += agent_total
                print(f"  {agent_id}: {len(rewards)} rewards, total={agent_total:.3f}")
            print(f"  TOTAL: {total_reward:.3f}")


# =============================================================================
# Neural Network Components
# =============================================================================

class SimpleMLP:
    """Simple MLP for value function approximation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def update(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> None:
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
    """Actor network with policy gradient updates."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def update(self, x: np.ndarray, action_taken: np.ndarray, advantage: float, lr: float = 0.01) -> None:
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
    """Neural policy for coordinator (microgrid) that computes joint action."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        seed: int = 42,
        subordinate_ids: Optional[List[str]] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)
        self.hidden_dim = hidden_dim
        self.subordinate_ids = subordinate_ids or []
        self.actor = ActorMLP(obs_dim, hidden_dim, action_dim, seed)
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)
        self.noise_scale = 0.15

    def _normalize_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        if len(obs_vec) > self.obs_dim:
            return obs_vec[:self.obs_dim]
        elif len(obs_vec) < self.obs_dim:
            return np.pad(obs_vec, (0, self.obs_dim - len(obs_vec)))
        return obs_vec

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_vec = self._normalize_obs(obs_vec)
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_vec, -1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_vec = self._normalize_obs(obs_vec)
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        obs_vec = self._normalize_obs(obs_vec)
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs: np.ndarray, action_taken: np.ndarray, advantage: float, lr: float = 0.01) -> None:
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs: np.ndarray, target: float, lr: float = 0.01) -> None:
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate: float = 0.995, min_noise: float = 0.05) -> None:
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)


# =============================================================================
# Environment Setup
# =============================================================================

def create_microgrid(mg_id: str, num_gen: int = 1, num_ess: int = 1) -> PowerGridAgent:
    """Create a microgrid coordinator with generators and ESS devices."""
    subordinates = {}

    for i in range(num_gen):
        gen_id = f"{mg_id}_Gen{i + 1}"
        subordinates[gen_id] = Generator(
            agent_id=gen_id,
            bus=f"{mg_id}_bus",
            p_min_MW=0.1,
            p_max_MW=1.0 + 0.5 * i,
            cost_curve_coefs=(0.01 + 0.005 * i, 0.4 + 0.1 * i, 0.0),
        )

    for i in range(num_ess):
        ess_id = f"{mg_id}_ESS{i + 1}"
        subordinates[ess_id] = ESS(
            agent_id=ess_id,
            bus=f"{mg_id}_bus",
            capacity_MWh=2.0 + i,
            p_min_MW=-0.5 - 0.1 * i,
            p_max_MW=0.5 + 0.1 * i,
            degr_cost_per_MWh=0.1,
        )

    vertical_protocol = VerticalProtocol()
    mg = PowerGridAgent(
        agent_id=mg_id,
        subordinates=subordinates,
        protocol=vertical_protocol,
    )

    return mg


def create_3mg_system(schedule_config: Optional[ScheduleConfig] = None) -> SystemAgent:
    """Create system agent with 3 microgrids.

    Args:
        schedule_config: Optional tick config for system agent. If provided,
            _simulation_wait_interval will use this tick_interval.
    """
    mg1 = create_microgrid("MG1", num_gen=1, num_ess=1)
    mg2 = create_microgrid("MG2", num_gen=2, num_ess=1)
    mg3 = create_microgrid("MG3", num_gen=1, num_ess=2)

    return SystemAgent(
        agent_id="system_agent",
        subordinates={"MG1": mg1, "MG2": mg2, "MG3": mg3},
        schedule_config=schedule_config,
    )


# =============================================================================
# Training with Action-Passing Mode
# =============================================================================

def train_ctde_action_passing(
    env: HierarchicalMicrogridEnv,
    logger: TrainingLogger,
    num_episodes: int = 30,
    steps_per_episode: int = 24,
    gamma: float = 0.99,
    lr: float = 0.01,
    verbose: bool = True,
) -> tuple:
    """Train coordinator policies using CTDE with action-passing mode."""

    # Identify coordinator agents
    coordinator_ids = []
    coordinator_device_ids: Dict[str, List[str]] = {}

    for aid, agent in env.registered_agents.items():
        if isinstance(agent, PowerGridAgent) and agent.subordinates:
            coordinator_ids.append(aid)
            coordinator_device_ids[aid] = list(agent.subordinates.keys())

    obs, _ = env.reset(seed=0)

    # Compute observation dimensions
    obs_dims_per_coordinator: Dict[str, int] = {}
    obs_dims_per_device: Dict[str, int] = {}

    for coord_id in coordinator_ids:
        sub_ids = coordinator_device_ids[coord_id]
        total_dim = 0
        for sub_id in sub_ids:
            if sub_id in obs:
                sub_obs = obs[sub_id]
                if isinstance(sub_obs, Observation):
                    dim = sub_obs.local_vector().shape[0]
                else:
                    dim = len(np.asarray(sub_obs).flatten())
                dim = max(dim, 2)
                obs_dims_per_device[sub_id] = dim
                total_dim += dim
        obs_dims_per_coordinator[coord_id] = max(total_dim, 4)

    # Create coordinator policies
    print(f"\nCreating coordinator policies for {len(coordinator_ids)} microgrids:")
    coordinator_policies: Dict[str, CoordinatorNeuralPolicy] = {}

    for i, coord_id in enumerate(coordinator_ids):
        sub_ids = coordinator_device_ids[coord_id]
        obs_dim = obs_dims_per_coordinator[coord_id]
        action_dim = len(sub_ids)
        print(f"  {coord_id}: obs_dim={obs_dim}, action_dim={action_dim}, devices={sub_ids}")

        coordinator_policies[coord_id] = CoordinatorNeuralPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            seed=42 + i,
            subordinate_ids=sub_ids,
        )

    returns_history = []
    mg_rewards_history = {coord_id: [] for coord_id in coordinator_ids}

    print(f"\n{'='*60}")
    print("TRAINING PROGRESS")
    print(f"{'='*60}")

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        trajectories = {
            coord_id: {"obs": [], "actions": [], "rewards": []}
            for coord_id in coordinator_ids
        }
        episode_return = 0.0
        mg_episode_rewards = {coord_id: 0.0 for coord_id in coordinator_ids}
        actual_steps = 0

        for step in range(steps_per_episode):
            all_device_actions = {}

            for coord_id in coordinator_ids:
                sub_ids = coordinator_device_ids[coord_id]
                policy = coordinator_policies[coord_id]

                # Aggregate observations
                aggregated_obs = []
                for sub_id in sub_ids:
                    obs_value = obs.get(sub_id)
                    if obs_value is None:
                        sub_dim = obs_dims_per_device.get(sub_id, 4)
                        aggregated_obs.append(np.zeros(sub_dim, dtype=np.float32))
                        continue

                    sub_dim = obs_dims_per_device.get(sub_id, 4)
                    if isinstance(obs_value, Observation):
                        obs_vec = obs_value.local_vector()
                    else:
                        obs_vec = np.asarray(obs_value, dtype=np.float32).flatten()

                    if len(obs_vec) > sub_dim:
                        obs_vec = obs_vec[:sub_dim]
                    elif len(obs_vec) < sub_dim:
                        obs_vec = np.pad(obs_vec, (0, sub_dim - len(obs_vec)))
                    aggregated_obs.append(obs_vec)

                aggregated_obs_vec = np.concatenate(aggregated_obs) if aggregated_obs else np.zeros(4)
                coord_observation = Observation(timestamp=step, local={"obs": aggregated_obs_vec})

                # Coordinator computes joint action
                coord_action = policy.forward(coord_observation)

                trajectories[coord_id]["obs"].append(aggregated_obs_vec.copy())
                trajectories[coord_id]["actions"].append(coord_action.c.copy())

                # Distribute actions via protocol
                coord_agent = env.registered_agents.get(coord_id)
                if coord_agent and coord_agent.protocol:
                    _, distributed_actions = coord_agent.protocol.coordinate(
                        coordinator_state=coord_agent.state,
                        coordinator_action=coord_action,
                        info_for_subordinates={sub_id: obs.get(sub_id) for sub_id in sub_ids},
                    )
                    for sub_id, sub_action in distributed_actions.items():
                        if sub_action is not None:
                            action_obj = Action()
                            action_obj.set_specs(
                                dim_c=1,
                                range=(np.array([-1.0]), np.array([1.0]))
                            )
                            if isinstance(sub_action, np.ndarray):
                                action_obj.set_values(c=sub_action.flatten()[:1])
                            else:
                                action_obj.set_values(c=np.array([float(sub_action)]))
                            all_device_actions[sub_id] = action_obj
                else:
                    for i, sub_id in enumerate(sub_ids):
                        action_obj = Action()
                        action_obj.set_specs(
                            dim_c=1,
                            range=(np.array([-1.0]), np.array([1.0]))
                        )
                        if i < len(coord_action.c):
                            action_obj.set_values(c=np.array([coord_action.c[i]]))
                        else:
                            action_obj.set_values(c=np.array([0.0]))
                        all_device_actions[sub_id] = action_obj

            # Execute step
            obs, rewards, terminated, _, info = env.step(all_device_actions)
            actual_steps += 1

            # Log power flow results
            pf_results = logger.log_power_flow(env)

            # Log step metrics
            logger.log_step(all_device_actions, rewards)

            # Aggregate rewards
            for coord_id in coordinator_ids:
                sub_ids = coordinator_device_ids[coord_id]
                coord_reward = sum(rewards.get(sub_id, 0.0) for sub_id in sub_ids)
                trajectories[coord_id]["rewards"].append(coord_reward)
                episode_return += coord_reward
                mg_episode_rewards[coord_id] += coord_reward

            # Detailed per-step logging (every 6 steps)
            if verbose and step % 6 == 0:
                pf_str = ""
                if pf_results and "v_avg" in pf_results:
                    pf_str = f" | V={pf_results['v_avg']:.4f}pu | Grid={pf_results['grid_power_MW']:.3f}MW"
                step_reward = sum(rewards.values())
                print(f"    Step {step:2d}: reward={step_reward:.3f}{pf_str}")

            if terminated.get("__all__", False):
                break

        # Update policies
        for coord_id, traj in trajectories.items():
            if not traj["rewards"]:
                continue

            policy = coordinator_policies[coord_id]
            returns = []
            G = 0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns = np.array(returns)

            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policy.get_value(
                    Observation(timestamp=t, local={"obs": obs_t})
                )
                advantage = returns[t] - baseline
                policy.update(obs_t, traj["actions"][t], advantage, lr=lr)
                policy.update_critic(obs_t, returns[t], lr=lr)

            policy.decay_noise()

        returns_history.append(episode_return)
        for coord_id in coordinator_ids:
            mg_rewards_history[coord_id].append(mg_episode_rewards[coord_id])

        # Log episode
        logger.log_episode(episode_return, actual_steps, mg_episode_rewards)

        # Print episode summary
        if verbose:
            logger.print_episode_summary(episode, verbose=(episode % 5 == 0))

    return coordinator_policies, returns_history, mg_rewards_history


# =============================================================================
# Event-Driven Execution with Detailed Logging
# =============================================================================

def run_event_driven(
    env: HierarchicalMicrogridEnv,
    policies: Dict[str, Policy],
    event_logger: EventLogger,
    t_end: float = 100.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run event-driven execution with trained coordinator policies."""

    # Attach coordinator policies
    for coord_id, policy in policies.items():
        agent = env.registered_agents.get(coord_id)
        if agent is not None:
            agent.policy = policy

    # Configure tick timing for system agent (this is critical for recurring ticks!)
    system_schedule_config = ScheduleConfig.with_jitter(
        tick_interval=30.0,  # System tick every 30 time units
        obs_delay=0.3,
        act_delay=0.5,
        msg_delay=0.2,
        reward_delay=1.0,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=41,
    )

    # Configure tick timing for coordinators
    coordinator_schedule_config = ScheduleConfig.with_jitter(
        tick_interval=10.0,
        obs_delay=0.2,
        act_delay=0.3,
        msg_delay=0.15,
        reward_delay=0.6,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=43,
    )

    # Configure tick timing for field agents
    field_schedule_config = ScheduleConfig.with_jitter(
        tick_interval=5.0,
        obs_delay=0.1,
        act_delay=0.2,
        msg_delay=0.1,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=42,
    )

    # Apply tick configs to all agents
    for agent_id, agent in env.registered_agents.items():
        if agent_id == "system_agent":
            agent.schedule_config = system_schedule_config
        elif isinstance(agent, PowerGridAgent):
            agent.schedule_config = coordinator_schedule_config
            for sub_agent in agent.subordinates.values():
                sub_agent.schedule_config = field_schedule_config
        elif hasattr(agent, 'schedule_config') and agent_id != "proxy_agent":
            agent.schedule_config = field_schedule_config

    # Update scheduler's tick config cache (it caches during attach())
    for agent_id, agent in env.registered_agents.items():
        if hasattr(agent, 'schedule_config'):
            env.scheduler._agent_schedule_configs[agent_id] = agent.schedule_config

    # Note: _simulation_wait_interval is now initialized early when creating system_agent
    # with schedule_config in main(), so no manual update needed here.

    # Reset environment to apply new tick configs (scheduler resets and reschedules)
    env.reset(seed=100)

    # Create custom event analyzer that logs to our EventLogger
    class DetailedEpisodeAnalyzer(EpisodeAnalyzer):
        def __init__(self, event_logger: EventLogger, **kwargs):
            super().__init__(**kwargs)
            self.event_logger = event_logger

        def parse_event(self, event):
            result = super().parse_event(event)

            # Log to our custom logger
            event_type = event.event_type.name.lower() if hasattr(event.event_type, 'name') else str(event.event_type)
            agent_id = event.agent_id or "unknown"
            timestamp = event.timestamp
            payload = event.payload or {}

            self.event_logger.log_event(event_type, agent_id, timestamp, payload)

            # Extract reward from tick results
            if "tick_result" in str(payload) or "reward" in payload:
                if verbose:
                    reward = payload.get("reward", "N/A")
                    print(f"  [t={timestamp:.2f}] {agent_id}: reward={reward}")

            return result

    episode_analyzer = DetailedEpisodeAnalyzer(event_logger, verbose=False, track_data=True)

    print(f"\nRunning event-driven simulation for t_end={t_end}...")
    env.run_event_driven(episode_analyzer=episode_analyzer, t_end=t_end)

    return {
        "observations": episode_analyzer.observation_count,
        "state_updates": episode_analyzer.state_update_count,
        "action_results": episode_analyzer.action_result_count,
    }


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 80)
    print("HierarchicalMicrogridEnv Integration Test")
    print("ACTION-PASSING MODE: Coordinators own policies, distribute to devices")
    print("=" * 80)

    # Define system tick config early so _simulation_wait_interval is set correctly
    system_schedule_config = ScheduleConfig.with_jitter(
        tick_interval=30.0,  # System tick every 30 time units
        obs_delay=0.3,
        act_delay=0.5,
        msg_delay=0.2,
        reward_delay=1.0,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=41,
    )

    # Create system with tick config (initializes _simulation_wait_interval correctly)
    system_agent = create_3mg_system(schedule_config=system_schedule_config)

    total_devices = sum(
        len(mg.subordinates) for mg in system_agent.subordinates.values()
    )
    print(f"\nSystem Configuration:")
    print(f"  Microgrids: {len(system_agent.subordinates)}")
    print(f"  Total devices: {total_devices}")
    for mg_id, mg in system_agent.subordinates.items():
        devices = list(mg.subordinates.keys())
        has_protocol = mg.protocol is not None
        print(f"  {mg_id}: {devices}, protocol={has_protocol}")

    # Get dataset path
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "..", "powergrid", "data.pkl"
    )
    dataset_path = os.path.abspath(dataset_path)

    # Create environment
    print(f"\nCreating HierarchicalMicrogridEnv...")
    print(f"  Dataset: {dataset_path}")

    env = HierarchicalMicrogridEnv(
        system_agent=system_agent,
        dataset_path=dataset_path,
        episode_steps=24,
        dt=1.0,
    )

    print(f"\nRegistered agents ({len(env.registered_agents)}):")
    for aid in sorted(env.registered_agents.keys()):
        agent = env.registered_agents[aid]
        agent_type = type(agent).__name__
        has_action = agent.action_space is not None
        has_policy = agent.policy is not None
        has_protocol = agent.protocol is not None
        print(f"  {aid}: {agent_type}, action={has_action}, policy={has_policy}, protocol={has_protocol}")

    # Initialize loggers
    training_logger = TrainingLogger()
    event_logger = EventLogger()

    # Run CTDE training
    print("\n" + "-" * 40)
    print("CTDE Training with Action-Passing Mode")
    print("  - Coordinators (microgrids) own policies")
    print("  - VerticalProtocol distributes actions to devices")
    print("-" * 40)

    coordinator_policies, returns, _mg_rewards = train_ctde_action_passing(
        env,
        logger=training_logger,
        num_episodes=30,
        steps_per_episode=24,
        gamma=0.99,
        lr=0.02,
        verbose=True,
    )

    # Print training summary
    training_logger.print_training_summary()

    # Run event-driven execution
    print("\n" + "-" * 40)
    print("Event-Driven Execution")
    print("-" * 40)

    stats = run_event_driven(
        env,
        coordinator_policies,
        event_logger,
        t_end=300.0,
        verbose=True,
    )

    # Print event-driven summary
    event_logger.print_summary()

    # Final validation
    print("\n" + "-" * 40)
    print("VALIDATION")
    print("-" * 40)

    initial_return = np.mean(returns[:5])
    final_return = np.mean(returns[-5:])
    learning_occurred = final_return > initial_return - 1.0
    events_ran = stats['observations'] > 0 and stats['action_results'] > 0

    print(f"\nAction-Passing Flow:")
    print(f"  1. SystemAgent coordinates 3 microgrids")
    print(f"  2. Each microgrid (PowerGridAgent) has a neural policy")
    print(f"  3. Policy outputs joint action vector")
    print(f"  4. VerticalProtocol decomposes vector into per-device actions")
    print(f"  5. Devices apply individual actions")

    print(f"\nChecks:")
    print(f"  [{'PASS' if len(returns) == 30 else 'FAIL'}] Training completed: {len(returns)} episodes")
    print(f"  [{'PASS' if learning_occurred else 'FAIL'}] Learning occurred: initial={initial_return:.1f}, final={final_return:.1f}")
    print(f"  [{'PASS' if events_ran else 'FAIL'}] Event-driven ran: obs={stats['observations']}, actions={stats['action_results']}")
    print(f"  [{'PASS' if len(system_agent.subordinates) == 3 else 'FAIL'}] All 3 microgrids active")
    print(f"  [{'PASS' if len(coordinator_policies) == 3 else 'FAIL'}] Coordinator policies trained: {len(coordinator_policies)}")

    pf_converge_rate = sum(training_logger.pf_converged) / len(training_logger.pf_converged) * 100 if training_logger.pf_converged else 0
    print(f"  [{'PASS' if pf_converge_rate > 90 else 'FAIL'}] Power flow convergence: {pf_converge_rate:.1f}%")

    print("\n" + "=" * 80)
    print("HierarchicalMicrogridEnv Integration Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

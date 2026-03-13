"""QMIX integration test with HERON action-passing environment.

Custom PyTorch QMIX implementation (QMIX was removed from RLlib >=2.10).
Uses the RLlibBasedHeronEnv with discrete action spaces.

Run::

    python tests/integration/test_qmix_action_passing.py
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.feature import Feature
from heron.core.action import Action
from heron.envs.base import HeronEnv
from heron.protocols.vertical import VerticalProtocol
from heron.adaptors.rllib import RLlibBasedHeronEnv


# =============================================================================
# Environment components (shared with test_rllib_action_passing.py)
# =============================================================================

@dataclass(slots=True)
class DevicePowerFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ["public"]
    power: float = 0.0
    capacity: float = 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.power, self.capacity], dtype=np.float32)

    def set_values(self, **kwargs: Any) -> None:
        if "power" in kwargs:
            self.power = np.clip(kwargs["power"], -self.capacity, self.capacity)
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


N_DISCRETE_ACTIONS = 11


class DeviceAgent(FieldAgent):
    """Device agent with native discrete action space for QMIX."""

    @property
    def power(self) -> float:
        return self.state.features["DevicePowerFeature"].power

    @property
    def capacity(self) -> float:
        return self.state.features["DevicePowerFeature"].capacity

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_d=1, ncats=[N_DISCRETE_ACTIONS])
        return action

    def compute_local_reward(self, local_state: dict) -> float:
        if "DevicePowerFeature" in local_state:
            power = float(local_state["DevicePowerFeature"][0])
            return -power ** 2
        return 0.0

    def set_action(self, action: Any) -> None:
        self.action.set_values(action)

    def apply_action(self):
        # Map discrete index to continuous power: index 5 (midpoint) → 0.0
        idx = int(self.action.d[0])
        frac = (idx + 0.5) / N_DISCRETE_ACTIONS
        power = (-1.0 + 2.0 * frac) * 0.5
        self.state.features["DevicePowerFeature"].set_values(power=power)

    def set_state(self) -> None:
        pass


class ZoneCoordinator(CoordinatorAgent):
    def compute_local_reward(self, local_state: dict) -> float:
        return sum(local_state.get("subordinate_rewards", {}).values())


class GridSystem(SystemAgent):
    pass


class EnvState:
    def __init__(self, device_powers: Optional[Dict[str, float]] = None):
        self.device_powers = device_powers or {"device_1": 0.0, "device_2": 0.0}


class ActionPassingEnv(HeronEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        for did in env_state.device_powers:
            env_state.device_powers[did] = np.clip(
                env_state.device_powers[did], -1.0, 1.0,
            )
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        from heron.agents.constants import FIELD_LEVEL
        agent_states = {}
        for aid, ag in self.registered_agents.items():
            if hasattr(ag, "level") and ag.level == FIELD_LEVEL and "device" in aid:
                agent_states[aid] = {
                    "_owner_id": aid,
                    "_owner_level": ag.level,
                    "_state_type": "FieldAgentState",
                    "features": {
                        "DevicePowerFeature": {
                            "power": env_state.device_powers.get(aid, 0.0),
                            "capacity": 1.0,
                        }
                    },
                }
        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        agent_states = global_state.get("agent_states", {})
        device_powers = {}
        for aid, sd in agent_states.items():
            if "device" in aid and "features" in sd:
                feat = sd["features"].get("DevicePowerFeature", {})
                device_powers[aid] = feat.get("power", 0.0)
        return EnvState(
            device_powers=device_powers or {"device_1": 0.0, "device_2": 0.0}
        )


ACTION_PASSING_ENV_CONFIG = {
    "agents": [
        {"agent_id": "device_1", "agent_cls": DeviceAgent,
         "features": [DevicePowerFeature(power=0.0, capacity=1.0)],
         "coordinator": "coordinator"},
        {"agent_id": "device_2", "agent_cls": DeviceAgent,
         "features": [DevicePowerFeature(power=0.0, capacity=1.0)],
         "coordinator": "coordinator"},
    ],
    "coordinators": [
        {"coordinator_id": "coordinator", "agent_cls": ZoneCoordinator,
         "protocol": VerticalProtocol()},
    ],
    "env_class": ActionPassingEnv,
    "env_kwargs": {
        "scheduler_config": {"start_time": 0.0, "time_step": 1.0},
        "message_broker_config": {"buffer_size": 1000, "max_queue_size": 100},
        "simulation_wait_interval": 0.01,
    },
}


# =============================================================================
# QMIX Components
# =============================================================================

class AgentQNetwork(nn.Module):
    """Per-agent Q-network: obs → Q-values for each discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMIXMixer(nn.Module):
    """Monotonic mixing network conditioned on global state.

    Takes per-agent Q-values and global state, produces Q_tot
    with the monotonicity constraint (positive weights).
    """

    def __init__(self, n_agents: int, state_dim: int, mixing_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents

        # Hypernetworks generate mixing weights from global state
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        )

        self.mixing_dim = mixing_dim

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            agent_qs: (batch, n_agents) — per-agent Q-values
            state: (batch, state_dim) — global state

        Returns:
            q_tot: (batch, 1)
        """
        batch_size = agent_qs.size(0)

        # First layer: monotonic (abs weights)
        w1 = torch.abs(self.hyper_w1(state)).view(
            batch_size, self.n_agents, self.mixing_dim
        )
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(
            batch_size, self.mixing_dim, 1
        )
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class ReplayBuffer:
    """Simple replay buffer for multi-agent transitions."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# QMIX Trainer
# =============================================================================

class QMIXTrainer:
    """QMIX training loop over a multi-agent environment."""

    def __init__(
        self,
        env: RLlibBasedHeronEnv,
        n_actions: int = 11,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 2000,
        batch_size: int = 32,
        target_update_freq: int = 200,
        buffer_capacity: int = 5000,
        hidden_dim: int = 64,
    ):
        self.env = env
        self.agent_ids = sorted(env.get_agent_ids())
        self.n_agents = len(self.agent_ids)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Determine dimensions from the env
        sample_obs, _ = env.reset(seed=0)
        self.obs_dim = sample_obs[self.agent_ids[0]].shape[0]
        self.state_dim = self.obs_dim * self.n_agents  # global state = concat obs

        # Per-agent Q-networks
        self.q_nets = nn.ModuleList([
            AgentQNetwork(self.obs_dim, n_actions, hidden_dim)
            for _ in range(self.n_agents)
        ])
        self.target_q_nets = nn.ModuleList([
            AgentQNetwork(self.obs_dim, n_actions, hidden_dim)
            for _ in range(self.n_agents)
        ])

        # Mixing networks
        self.mixer = QMIXMixer(self.n_agents, self.state_dim)
        self.target_mixer = QMIXMixer(self.n_agents, self.state_dim)

        # Sync target networks
        self._hard_update_targets()

        # Optimizer over all parameters
        params = list(self.q_nets.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

    def _hard_update_targets(self):
        for i in range(self.n_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _get_epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.epsilon_decay)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _obs_to_state(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all agent observations into a global state vector."""
        return np.concatenate([obs_dict[aid] for aid in self.agent_ids])

    def _select_actions(
        self, obs_dict: Dict[str, np.ndarray], epsilon: float
    ) -> Dict[str, int]:
        actions = {}
        for i, aid in enumerate(self.agent_ids):
            if random.random() < epsilon:
                actions[aid] = random.randint(0, self.n_actions - 1)
            else:
                obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_nets[i](obs_t)
                actions[aid] = q_values.argmax(dim=-1).item()
        return actions

    def _train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        # Unpack: each element is (obs_list, state, actions, rewards, next_obs_list, next_state, done)
        obs_batch = torch.FloatTensor(np.array([t[0] for t in batch]))  # (B, n_agents, obs_dim)
        state_batch = torch.FloatTensor(np.array([t[1] for t in batch]))  # (B, state_dim)
        action_batch = torch.LongTensor(np.array([t[2] for t in batch]))  # (B, n_agents)
        reward_batch = torch.FloatTensor(np.array([t[3] for t in batch]))  # (B,)
        next_obs_batch = torch.FloatTensor(np.array([t[4] for t in batch]))  # (B, n_agents, obs_dim)
        next_state_batch = torch.FloatTensor(np.array([t[5] for t in batch]))  # (B, state_dim)
        done_batch = torch.FloatTensor(np.array([t[6] for t in batch]))  # (B,)

        # Compute per-agent Q-values for chosen actions
        agent_qs = []
        for i in range(self.n_agents):
            q_all = self.q_nets[i](obs_batch[:, i])  # (B, n_actions)
            q_chosen = q_all.gather(1, action_batch[:, i].unsqueeze(1)).squeeze(1)  # (B,)
            agent_qs.append(q_chosen)
        agent_qs = torch.stack(agent_qs, dim=1)  # (B, n_agents)

        # Q_tot via mixer
        q_tot = self.mixer(agent_qs, state_batch)  # (B,)

        # Target Q-values (max over actions per agent)
        with torch.no_grad():
            target_agent_qs = []
            for i in range(self.n_agents):
                target_q_all = self.target_q_nets[i](next_obs_batch[:, i])  # (B, n_actions)
                target_q_max = target_q_all.max(dim=-1).values  # (B,)
                target_agent_qs.append(target_q_max)
            target_agent_qs = torch.stack(target_agent_qs, dim=1)  # (B, n_agents)

            target_q_tot = self.target_mixer(target_agent_qs, next_state_batch)
            y = reward_batch + self.gamma * (1 - done_batch) * target_q_tot

        loss = F.mse_loss(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_nets.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes: int = 200, max_steps: int = 30) -> List[float]:
        episode_rewards = []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            state = self._obs_to_state(obs)
            ep_reward = 0.0

            for step in range(max_steps):
                epsilon = self._get_epsilon()
                actions = self._select_actions(obs, epsilon)
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                next_state = self._obs_to_state(next_obs)

                # Team reward (sum over agents)
                team_reward = sum(rewards[aid] for aid in self.agent_ids)
                done = terms.get("__all__", False) or truncs.get("__all__", False)

                # Store transition (obs as list of per-agent arrays)
                obs_list = [obs[aid] for aid in self.agent_ids]
                next_obs_list = [next_obs[aid] for aid in self.agent_ids]
                action_list = [actions[aid] for aid in self.agent_ids]

                self.buffer.push((
                    obs_list, state, action_list, team_reward,
                    next_obs_list, next_state, float(done),
                ))

                self._train_step()
                self.total_steps += 1

                if self.total_steps % self.target_update_freq == 0:
                    self._hard_update_targets()

                ep_reward += team_reward
                obs = next_obs
                state = next_state

                if done:
                    break

            episode_rewards.append(ep_reward)

            if (ep + 1) % 50 == 0:
                avg = np.mean(episode_rewards[-50:])
                print(
                    f"  Episode {ep + 1}/{num_episodes}: "
                    f"avg_reward(50) = {avg:.3f}, "
                    f"epsilon = {self._get_epsilon():.3f}"
                )

        return episode_rewards


# =============================================================================
# Main
# =============================================================================

def run_qmix(num_episodes: int = 600, max_steps: int = 30):
    print("\n" + "=" * 60)
    print("QMIX: Value Decomposition (Custom PyTorch)")
    print("=" * 60)
    print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"  Discrete actions: {N_DISCRETE_ACTIONS} (native)")
    print(f"  Theoretical optimal reward: 0  (action=midpoint every step)")
    print(f"  Random baseline reward:    -5  (uniform random actions)")

    env = RLlibBasedHeronEnv({
        **ACTION_PASSING_ENV_CONFIG,
        "max_steps": max_steps,
    })

    agent_ids = sorted(env.get_agent_ids())
    print(f"  Agents: {agent_ids}")
    print(f"  Obs space: {env._obs_spaces[agent_ids[0]]}")
    print(f"  Act space: {env._act_spaces[agent_ids[0]]}")

    trainer = QMIXTrainer(
        env=env,
        n_actions=N_DISCRETE_ACTIONS,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=4000,
        batch_size=64,
        target_update_freq=200,
        buffer_capacity=10000,
    )

    rewards = trainer.train(num_episodes=num_episodes, max_steps=max_steps)

    # Summary
    first_50 = np.mean(rewards[:50])
    last_50 = np.mean(rewards[-50:])
    print(f"\n  First 50 episodes avg reward: {first_50:.3f}")
    print(f"  Last 50 episodes avg reward:  {last_50:.3f}")
    print(f"  Improvement: {last_50 - first_50:+.3f}")

    return {"algorithm": "QMIX", "rewards": rewards}


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    result = run_qmix()

    print("\n" + "=" * 60)
    print("QMIX Test Complete")
    print("=" * 60)

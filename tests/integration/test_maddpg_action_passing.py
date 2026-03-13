"""MADDPG integration test with HERON action-passing environment.

Custom PyTorch MADDPG implementation (MADDPG was removed from RLlib >=2.10).
Uses the RLlibBasedHeronEnv with continuous action spaces.

Run::

    python tests/integration/test_maddpg_action_passing.py
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


class DeviceAgent(FieldAgent):
    @property
    def power(self) -> float:
        return self.state.features["DevicePowerFeature"].power

    @property
    def capacity(self) -> float:
        return self.state.features["DevicePowerFeature"].capacity

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict) -> float:
        if "DevicePowerFeature" in local_state:
            power = float(local_state["DevicePowerFeature"][0])
            return -power ** 2
        return 0.0

    def set_action(self, action: Any) -> None:
        if isinstance(action, Action):
            if len(action.c) != self.action.dim_c:
                self.action.set_values(action.c[: self.action.dim_c])
            else:
                self.action.set_values(c=action.c)
        else:
            self.action.set_values(action)

    def set_state(self) -> None:
        new_power = self.action.c[0] * 0.5
        self.state.features["DevicePowerFeature"].set_values(power=new_power)

    def apply_action(self):
        self.set_state()


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
# MADDPG Components
# =============================================================================

class Actor(nn.Module):
    """Deterministic actor: obs → continuous action (tanh-scaled)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Centralized critic: (all_obs, all_actions) → Q-value.

    Takes the concatenation of all agents' observations and all agents'
    actions as input, producing a scalar Q-value.
    """

    def __init__(self, total_obs_dim: int, total_act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, all_obs: torch.Tensor, all_actions: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([all_obs, all_actions], dim=-1))


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            len(self.state)
        )
        self.state += dx
        return self.state.astype(np.float32)


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
# MADDPG Trainer
# =============================================================================

class MADDPGTrainer:
    """MADDPG training loop over a multi-agent environment.

    Each agent has its own actor (decentralized) and critic (centralized,
    takes all agents' obs + actions).
    """

    def __init__(
        self,
        env: RLlibBasedHeronEnv,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        noise_sigma: float = 0.2,
        hidden_dim: int = 64,
    ):
        self.env = env
        self.agent_ids = sorted(env.get_agent_ids())
        self.n_agents = len(self.agent_ids)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Determine dimensions
        sample_obs, _ = env.reset(seed=0)
        self.obs_dims = {
            aid: sample_obs[aid].shape[0] for aid in self.agent_ids
        }
        self.act_dims = {}
        for aid in self.agent_ids:
            space = env._act_spaces[aid]
            self.act_dims[aid] = int(np.prod(space.shape))

        self.total_obs_dim = sum(self.obs_dims.values())
        self.total_act_dim = sum(self.act_dims.values())

        # Per-agent actors and critics
        self.actors = nn.ModuleList()
        self.target_actors = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        self.actor_optimizers = []
        self.critic_optimizers = []

        for aid in self.agent_ids:
            obs_d = self.obs_dims[aid]
            act_d = self.act_dims[aid]

            actor = Actor(obs_d, act_d, hidden_dim)
            target_actor = Actor(obs_d, act_d, hidden_dim)
            target_actor.load_state_dict(actor.state_dict())

            critic = Critic(self.total_obs_dim, self.total_act_dim, hidden_dim)
            target_critic = Critic(self.total_obs_dim, self.total_act_dim, hidden_dim)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(
                torch.optim.Adam(actor.parameters(), lr=lr_actor)
            )
            self.critic_optimizers.append(
                torch.optim.Adam(critic.parameters(), lr=lr_critic)
            )

        # Exploration noise (one per agent)
        self.noises = {
            aid: OUNoise(self.act_dims[aid], sigma=noise_sigma)
            for aid in self.agent_ids
        }

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Action space bounds
        self.act_lows = {}
        self.act_highs = {}
        for aid in self.agent_ids:
            space = env._act_spaces[aid]
            self.act_lows[aid] = space.low.flatten()
            self.act_highs[aid] = space.high.flatten()

    def _select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        add_noise: bool = True,
        noise_scale: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        actions = {}
        for i, aid in enumerate(self.agent_ids):
            obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)
            with torch.no_grad():
                action = self.actors[i](obs_t).squeeze(0).numpy()
            if add_noise:
                action = action + noise_scale * self.noises[aid].sample()
            # Clip to action bounds
            action = np.clip(action, self.act_lows[aid], self.act_highs[aid])
            actions[aid] = action.astype(np.float32)
        return actions

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def _train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        # Each transition: (obs_list, actions_list, rewards_list, next_obs_list, done)
        obs_batch = [
            torch.FloatTensor(np.array([t[0][i] for t in batch]))
            for i in range(self.n_agents)
        ]  # list of (B, obs_dim_i)
        act_batch = [
            torch.FloatTensor(np.array([t[1][i] for t in batch]))
            for i in range(self.n_agents)
        ]  # list of (B, act_dim_i)
        rew_batch = [
            torch.FloatTensor(np.array([t[2][i] for t in batch]))
            for i in range(self.n_agents)
        ]  # list of (B,)
        next_obs_batch = [
            torch.FloatTensor(np.array([t[3][i] for t in batch]))
            for i in range(self.n_agents)
        ]
        done_batch = torch.FloatTensor(np.array([t[4] for t in batch]))  # (B,)

        # Concatenated obs and actions for critic input
        all_obs = torch.cat(obs_batch, dim=-1)  # (B, total_obs_dim)
        all_acts = torch.cat(act_batch, dim=-1)  # (B, total_act_dim)
        all_next_obs = torch.cat(next_obs_batch, dim=-1)

        # Target actions for next states
        with torch.no_grad():
            target_next_acts = torch.cat(
                [self.target_actors[i](next_obs_batch[i]) for i in range(self.n_agents)],
                dim=-1,
            )

        total_loss = 0.0

        for i in range(self.n_agents):
            # ---- Update critic ----
            with torch.no_grad():
                target_q = self.target_critics[i](all_next_obs, target_next_acts)
                y = rew_batch[i].unsqueeze(1) + self.gamma * (
                    1 - done_batch.unsqueeze(1)
                ) * target_q

            current_q = self.critics[i](all_obs, all_acts)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()

            # ---- Update actor ----
            # Replace agent i's action with current policy output
            current_acts_for_grad = []
            for j in range(self.n_agents):
                if j == i:
                    current_acts_for_grad.append(self.actors[j](obs_batch[j]))
                else:
                    current_acts_for_grad.append(act_batch[j].detach())
            all_acts_for_grad = torch.cat(current_acts_for_grad, dim=-1)

            actor_loss = -self.critics[i](all_obs.detach(), all_acts_for_grad).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            total_loss += critic_loss.item()

        # Soft-update all target networks
        for i in range(self.n_agents):
            self._soft_update(self.target_actors[i], self.actors[i])
            self._soft_update(self.target_critics[i], self.critics[i])

        return total_loss / self.n_agents

    def train(
        self,
        num_episodes: int = 300,
        max_steps: int = 30,
        warmup_steps: int = 500,
    ) -> List[float]:
        episode_rewards = []

        # Warmup: collect random transitions before training
        warmup_count = 0
        while warmup_count < warmup_steps:
            obs, _ = self.env.reset()
            for _ in range(max_steps):
                actions = {}
                for aid in self.agent_ids:
                    space = self.env._act_spaces[aid]
                    actions[aid] = space.sample().astype(np.float32)
                next_obs, rewards, terms, truncs, _ = self.env.step(actions)
                done = terms.get("__all__", False) or truncs.get("__all__", False)
                obs_list = [obs[aid] for aid in self.agent_ids]
                act_list = [actions[aid] for aid in self.agent_ids]
                rew_list = [rewards[aid] for aid in self.agent_ids]
                next_obs_list = [next_obs[aid] for aid in self.agent_ids]
                self.buffer.push((
                    obs_list, act_list, rew_list, next_obs_list, float(done),
                ))
                warmup_count += 1
                obs = next_obs
                if done:
                    break
        print(f"  Warmup complete: {len(self.buffer)} transitions collected")

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            for noise in self.noises.values():
                noise.reset()

            ep_reward = 0.0
            # Decay noise over training
            noise_scale = max(0.05, 1.0 - ep / num_episodes)

            for step in range(max_steps):
                actions = self._select_actions(obs, add_noise=True, noise_scale=noise_scale)
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                done = terms.get("__all__", False) or truncs.get("__all__", False)

                obs_list = [obs[aid] for aid in self.agent_ids]
                act_list = [actions[aid] for aid in self.agent_ids]
                rew_list = [rewards[aid] for aid in self.agent_ids]
                next_obs_list = [next_obs[aid] for aid in self.agent_ids]

                self.buffer.push((
                    obs_list, act_list, rew_list, next_obs_list, float(done),
                ))

                self._train_step()

                team_reward = sum(rewards[aid] for aid in self.agent_ids)
                ep_reward += team_reward
                obs = next_obs

                if done:
                    break

            episode_rewards.append(ep_reward)

            if (ep + 1) % 50 == 0:
                avg = np.mean(episode_rewards[-50:])
                print(
                    f"  Episode {ep + 1}/{num_episodes}: "
                    f"avg_reward(50) = {avg:.3f}, "
                    f"noise_scale = {noise_scale:.3f}"
                )

        return episode_rewards


# =============================================================================
# Main
# =============================================================================

def run_maddpg(num_episodes: int = 500, max_steps: int = 30):
    print("\n" + "=" * 60)
    print("MADDPG: Centralized Critic + Deterministic Policy (Custom PyTorch)")
    print("=" * 60)
    print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"  Theoretical optimal reward: 0  (action=0 every step)")
    print(f"  Random baseline reward:    -5  (uniform random actions)")

    env = RLlibBasedHeronEnv({
        **ACTION_PASSING_ENV_CONFIG,
        "max_steps": max_steps,
    })

    agent_ids = sorted(env.get_agent_ids())
    print(f"  Agents: {agent_ids}")
    print(f"  Obs space: {env._obs_spaces[agent_ids[0]]}")
    print(f"  Act space: {env._act_spaces[agent_ids[0]]}")

    trainer = MADDPGTrainer(
        env=env,
        lr_actor=3e-4,
        lr_critic=3e-3,
        gamma=0.99,
        tau=0.01,
        batch_size=64,
        buffer_capacity=10000,
        noise_sigma=0.15,
    )

    rewards = trainer.train(
        num_episodes=num_episodes, max_steps=max_steps, warmup_steps=500,
    )

    # Summary
    first_50 = np.mean(rewards[:50])
    last_50 = np.mean(rewards[-50:])
    print(f"\n  First 50 episodes avg reward: {first_50:.3f}")
    print(f"  Last 50 episodes avg reward:  {last_50:.3f}")
    print(f"  Improvement: {last_50 - first_50:+.3f}")

    return {"algorithm": "MADDPG", "rewards": rewards}


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    result = run_maddpg()

    print("\n" + "=" * 60)
    print("MADDPG Test Complete")
    print("=" * 60)

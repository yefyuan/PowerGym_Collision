"""Policy Interface & Manual Training -- learning without RLlib.

HERON's Policy class is a lightweight ABC that bridges observations to
actions.  Combined with the @obs_to_vector / @vector_to_action decorators,
you can write a policy that works in BOTH synchronous (training) and
event-driven (testing) modes with zero glue code.

This script demonstrates:
1. Policy subclass with decorators -- automatic obs/action conversion
2. Manual REINFORCE training loop -- pure numpy, no RL framework needed
3. IPPO vs MAPPO -- independent vs shared policy comparison

Domain: Two thermostats controlling room temperature (from Level 3).
  - Each thermostat adjusts heating power [-1, 1].
  - Reward: negative distance to setpoint (target=22C).
  - IPPO: each thermostat has its own policy.
  - MAPPO: both share a single policy.

Usage:
    cd "examples/5. training_algorithms"
    python policy_and_training.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.simple import SimpleEnv


# ---------------------------------------------------------------------------
# 1. Domain: thermostats (reused from Level 3)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RoomTempFeature(Feature):
    """Room temperature (public so coordinator can observe it)."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    temp: float = 20.0
    target: float = 22.0


class Thermostat(FieldAgent):
    """Thermostat that adjusts heating power."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        if "temp" in kwargs:
            self.state.features["RoomTempFeature"].set_values(temp=kwargs["temp"])

    def apply_action(self) -> None:
        feat = self.state.features["RoomTempFeature"]
        new_temp = feat.temp + self.action.c[0] * 2.0
        feat.set_values(temp=float(np.clip(new_temp, 10.0, 35.0)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "RoomTempFeature" not in local_state:
            return 0.0
        vec = local_state["RoomTempFeature"]
        temp, target = float(vec[0]), float(vec[1])
        return -abs(temp - target)


def thermostat_simulation(agent_states: dict) -> dict:
    """Natural cooling toward 15C ambient + cross-room leakage."""
    rooms = {}
    for aid, features in agent_states.items():
        if "RoomTempFeature" in features:
            rooms[aid] = features["RoomTempFeature"]["temp"]

    ambient, cooling_rate, leakage_rate = 15.0, 0.1, 0.05
    for aid in rooms:
        rooms[aid] += cooling_rate * (ambient - rooms[aid])

    if len(rooms) >= 2:
        avg_temp = np.mean(list(rooms.values()))
        for aid in rooms:
            rooms[aid] += leakage_rate * (avg_temp - rooms[aid])

    for aid, temp in rooms.items():
        agent_states[aid]["RoomTempFeature"]["temp"] = float(np.clip(temp, 10.0, 35.0))
    return agent_states


# ---------------------------------------------------------------------------
# 2. Neural policy using HERON's Policy interface
# ---------------------------------------------------------------------------

class ThermostatPolicy(Policy):
    """Simple actor-critic policy using HERON's Policy ABC.

    Key features:
    - observation_mode = "local" for decentralized execution
    - @obs_to_vector extracts local features as numpy vector
    - @vector_to_action wraps output numpy into Action object
    - Works in both sync (training) and event-driven (testing) modes
    """
    observation_mode = "local"

    def __init__(self, obs_dim: int, action_dim: int = 1, seed: int = 0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)

        # Simple linear actor + critic (no hidden layers for this toy domain)
        rng = np.random.default_rng(seed)
        self.actor_w = rng.normal(0, 0.1, (obs_dim, action_dim))
        self.actor_b = np.zeros(action_dim)
        self.critic_w = rng.normal(0, 0.1, (obs_dim, 1))
        self.critic_b = np.zeros(1)

        self.noise_scale = 0.3

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute action with exploration noise.

        Thanks to @obs_to_vector, this receives a numpy vector (not Observation).
        Thanks to @vector_to_action, the returned numpy is wrapped as Action.
        """
        action_mean = np.tanh(obs_vec @ self.actor_w + self.actor_b)
        noise = np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_mean + noise, -1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute action without noise (used in event-driven evaluation mode)."""
        return np.tanh(obs_vec @ self.actor_w + self.actor_b)

    def get_value(self, obs_vec: np.ndarray) -> float:
        """Estimate state value (critic)."""
        return float((obs_vec @ self.critic_w + self.critic_b)[0])

    def update(self, obs: np.ndarray, action: np.ndarray,
               advantage: float, lr: float = 0.01) -> None:
        """Policy gradient update (REINFORCE with baseline)."""
        advantage = np.clip(advantage, -5.0, 5.0)
        current = np.tanh(obs @ self.actor_w + self.actor_b)
        grad = (current - action) * (1 - current ** 2)
        self.actor_w -= lr * np.outer(obs, grad * advantage)
        self.actor_b -= lr * (grad * advantage).flatten()

    def update_critic(self, obs: np.ndarray, target: float, lr: float = 0.01) -> None:
        """Critic update: minimize (V(s) - target)^2."""
        pred = float((obs @ self.critic_w + self.critic_b)[0])
        error = np.clip(pred - target, -10.0, 10.0)
        self.critic_w -= lr * obs.reshape(-1, 1) * error
        self.critic_b -= lr * error


# ---------------------------------------------------------------------------
# 3. Training loop (manual REINFORCE)
# ---------------------------------------------------------------------------

def build_env():
    """Build the two-room thermostat environment."""
    thermo_a = Thermostat(agent_id="room_a", features=[RoomTempFeature()])
    thermo_b = Thermostat(agent_id="room_b", features=[RoomTempFeature()])
    coordinator = CoordinatorAgent(
        agent_id="building",
        subordinates={"room_a": thermo_a, "room_b": thermo_b},
    )
    return SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=thermostat_simulation,
        env_id="thermostat_training",
    )


def train(policies: dict, label: str,
          num_episodes: int = 200, steps_per_ep: int = 30,
          gamma: float = 0.99, lr: float = 0.02) -> list:
    """Train policies using REINFORCE with baseline.

    Args:
        policies: {agent_id: Policy} or {"shared": Policy} for MAPPO
        label: Name for printing
        num_episodes: Training episodes
        steps_per_ep: Steps per episode
        gamma: Discount factor
        lr: Learning rate

    Returns:
        List of per-episode returns
    """
    env = build_env()
    agent_ids = ["room_a", "room_b"]
    returns_history = []

    # Determine if shared policy (MAPPO) or independent (IPPO)
    is_shared = "shared" in policies

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)

        trajectories = {aid: {"obs": [], "actions": [], "rewards": []} for aid in agent_ids}

        for step in range(steps_per_ep):
            actions = {}
            for aid in agent_ids:
                policy = policies["shared"] if is_shared else policies[aid]
                action = policy.forward(obs[aid])
                actions[aid] = action

                # Store trajectory
                obs_value = obs[aid]
                if hasattr(obs_value, "vector"):
                    obs_vec = obs_value.vector()
                else:
                    obs_vec = np.asarray(obs_value, dtype=np.float32)
                trajectories[aid]["obs"].append(obs_vec[:policy.obs_dim])
                trajectories[aid]["actions"].append(action.c.copy())

            obs, rewards, terminated, _, _ = env.step(actions)

            for aid in agent_ids:
                trajectories[aid]["rewards"].append(rewards.get(aid, 0.0))

            if terminated.get("__all__", False):
                break

        # Compute returns and update policies
        episode_return = 0.0
        for aid in agent_ids:
            traj = trajectories[aid]
            if not traj["rewards"]:
                continue

            # Discounted returns
            returns = []
            G = 0.0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)

            episode_return += sum(traj["rewards"])

            # Normalize returns for stable gradients
            returns_arr = np.array(returns)
            std = returns_arr.std()
            if std > 1e-8:
                returns_arr = (returns_arr - returns_arr.mean()) / std

            policy = policies["shared"] if is_shared else policies[aid]
            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policy.get_value(obs_t)
                advantage = returns_arr[t] - baseline
                policy.update(obs_t, traj["actions"][t], advantage, lr=lr)
                policy.update_critic(obs_t, returns_arr[t], lr=lr)

        returns_history.append(episode_return)

        if is_shared:
            # Decay noise once per episode for shared policy
            policies["shared"].noise_scale = max(0.05, policies["shared"].noise_scale * 0.995)
        else:
            for aid in agent_ids:
                policies[aid].noise_scale = max(0.05, policies[aid].noise_scale * 0.995)

        if (episode + 1) % 50 == 0:
            recent = np.mean(returns_history[-20:])
            print(f"    [{label}] Episode {episode + 1:3d}: avg_return(20)={recent:.1f}")

    return returns_history


# ---------------------------------------------------------------------------
# 4. Main: IPPO vs MAPPO comparison
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Part 1: HERON Policy Interface")
    print("=" * 60)
    print("""
  Policy subclass with decorators:

    class MyPolicy(Policy):
        observation_mode = "local"  # or "full", "global"

        @obs_to_vector       # Observation -> np.ndarray
        @vector_to_action    # np.ndarray -> Action
        def forward(self, obs_vec):
            return my_network(obs_vec)

  The decorators handle format conversion automatically.
  Your forward() receives numpy and returns numpy.
""")

    # Quick demo of decorator behavior
    env = build_env()
    obs, _ = env.reset(seed=0)
    policy = ThermostatPolicy(obs_dim=2, seed=0)

    action = policy.forward(obs["room_a"])
    print(f"  Input:  Observation object (local={list(obs['room_a'].local.keys())})")
    print(f"  Output: Action object (c={action.c}, dim_c={action.dim_c})")
    print(f"  Decorators handled the conversion automatically.\n")

    # ------------------------------------------------------------------
    # IPPO: independent policies
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 2: IPPO (Independent PPO)")
    print("=" * 60)
    print("  Each agent has its own policy. No parameter sharing.\n")

    ippo_policies = {
        "room_a": ThermostatPolicy(obs_dim=2, seed=10),
        "room_b": ThermostatPolicy(obs_dim=2, seed=20),
    }
    ippo_returns = train(ippo_policies, "IPPO", num_episodes=200, lr=0.005)

    # ------------------------------------------------------------------
    # MAPPO: shared policy
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Part 3: MAPPO (Multi-Agent PPO with shared policy)")
    print("=" * 60)
    print("  Both agents share the same policy. Parameters are shared.\n")

    mappo_policies = {
        "shared": ThermostatPolicy(obs_dim=2, seed=30),
    }
    mappo_returns = train(mappo_policies, "MAPPO", num_episodes=200, lr=0.005)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Part 4: IPPO vs MAPPO Comparison")
    print("=" * 60)

    def window_avg(data, window=20):
        return [np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))]

    ippo_smooth = window_avg(ippo_returns)
    mappo_smooth = window_avg(mappo_returns)

    print(f"\n  {'Metric':<30} {'IPPO':>10} {'MAPPO':>10}")
    print(f"  {'-' * 50}")
    print(f"  {'First 20 episodes (avg):':<30} {np.mean(ippo_returns[:20]):>10.1f} {np.mean(mappo_returns[:20]):>10.1f}")
    print(f"  {'Last 20 episodes (avg):':<30} {np.mean(ippo_returns[-20:]):>10.1f} {np.mean(mappo_returns[-20:]):>10.1f}")
    print(f"  {'Best 20-ep window:':<30} {max(ippo_smooth):>10.1f} {max(mappo_smooth):>10.1f}")
    print(f"  {'Improvement:':<30} {np.mean(ippo_returns[-20:]) - np.mean(ippo_returns[:20]):>10.1f} {np.mean(mappo_returns[-20:]) - np.mean(mappo_returns[:20]):>10.1f}")

    print("""
  IPPO: Each agent learns independently.
    + Better for heterogeneous tasks (different reward structures)
    + Scales to many agents without coordination overhead

  MAPPO: Agents share a single policy.
    + 2x sample efficiency (both agents' data trains one network)
    + Better for homogeneous tasks (same dynamics, same reward)
    - Can struggle when agents need different strategies
""")
    print("Done.")


if __name__ == "__main__":
    main()

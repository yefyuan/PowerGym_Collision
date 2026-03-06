"""
Multi-station EV Charging Environment - Training Script

Demonstrates:
1. HERON-based multi-agent env with proper CTDE pipeline
2. Policy gradient training with PricingPolicy (actor-critic)
3. Optional Ray RLlib integration via flat dict config
4. Optional event-driven deployment after training
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from heron.core.observation import Observation

from case_studies.power.ev_public_charging_case.agents import ChargingSlot, StationCoordinator
from case_studies.power.ev_public_charging_case.envs.charging_env import ChargingEnv
from case_studies.power.ev_public_charging_case.policies import PricingPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Environment Factory
# ============================================================================
def create_charging_env(config: Dict[str, Any] = None) -> ChargingEnv:
    """Create a ChargingEnv with N stations, each having M charger slots.

    Args:
        config: Dict with keys: num_stations, num_chargers, arrival_rate, dt, episode_length

    Returns:
        Fully initialized ChargingEnv (HERON HeronEnv)
    """
    config = config or {}
    num_stations = config.get("num_stations", 2)
    num_chargers = config.get("num_chargers", 5)
    arrival_rate = config.get("arrival_rate", 10.0)
    dt = config.get("dt", 300.0)
    episode_length = config.get("episode_length", 86400.0)

    coordinators: List[StationCoordinator] = []
    for i in range(num_stations):
        s_id = f"station_{i}"
        slots = {
            f"{s_id}_slot_{j}": ChargingSlot(agent_id=f"{s_id}_slot_{j}", p_max_kw=150.0)
            for j in range(num_chargers)
        }
        coordinators.append(StationCoordinator(agent_id=s_id, subordinates=slots))

    return ChargingEnv(
        coordinator_agents=coordinators,
        arrival_rate=arrival_rate,
        dt=dt,
        episode_length=episode_length,
    )


# ============================================================================
# CTDE Training with PricingPolicy
# ============================================================================
def train_simple(
    num_episodes: int = 50,
    steps_per_episode: int = 288,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 0.01,
) -> Tuple[ChargingEnv, Dict[str, PricingPolicy], List[float]]:
    """Train pricing policies using HERON's CTDE pipeline.

    Each step flows through:
        system_agent.execute(actions, proxy)
        -> layer_actions -> act (coordinators + slots)
        -> run_simulation (EV arrivals, charging, departures)
        -> state update via proxy
        -> compute rewards/observations

    Args:
        num_episodes: Number of training episodes
        steps_per_episode: Steps per episode (288 * 300s = 86400s = 1 day)
        seed: Random seed
        gamma: Discount factor for returns
        lr: Learning rate for policy gradient

    Returns:
        Tuple of (env, policies, returns_history)
    """
    np.random.seed(seed)
    env = create_charging_env()

    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]
    logger.info(f"Station agents: {station_ids}")

    policies = {
        sid: PricingPolicy(obs_dim=8, action_dim=1, hidden_dim=32, seed=seed + i)
        for i, sid in enumerate(station_ids)
    }

    returns_history: List[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        trajectories = {sid: {"obs": [], "actions": [], "rewards": []} for sid in station_ids}
        episode_reward = {sid: 0.0 for sid in station_ids}

        for step in range(steps_per_episode):
            actions = {}
            for sid in station_ids:
                obs_value = obs[sid]
                if isinstance(obs_value, Observation):
                    observation = Observation(timestamp=step, local=obs_value.local)
                elif isinstance(obs_value, np.ndarray):
                    observation = Observation(timestamp=step, local={"obs": obs_value[:8]})
                else:
                    observation = Observation(timestamp=step, local={"obs": np.zeros(8, dtype=np.float32)})

                action = policies[sid].forward(observation)
                actions[sid] = action

                obs_vec = policies[sid].extract_obs_vector(observation, 8)
                reg_signal = float(obs_vec[5])
                headroom_up = float(obs_vec[6])
                headroom_down = float(obs_vec[7])
                if episode == 0 and step % 20 == 0:
                    logger.info(
                        f"[{sid}] t_step={step:03d} reg={reg_signal:+.3f} "
                        f"headroom_up={headroom_up:.3f} headroom_down={headroom_down:.3f}"
                    )
                trajectories[sid]["obs"].append(obs_vec)
                trajectories[sid]["actions"].append(action.c.copy())

            obs, rewards, terminated, truncated, infos = env.step(actions)

            for sid in station_ids:
                r = rewards.get(sid, 0.0)
                trajectories[sid]["rewards"].append(r)
                episode_reward[sid] += r

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        # Policy gradient update with advantage estimation
        for sid, traj in trajectories.items():
            if not traj["rewards"]:
                continue
            returns = []
            G = 0.0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns_arr = np.array(returns)

            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policies[sid].get_value(
                    Observation(timestamp=t, local={"obs": obs_t})
                )
                advantage = returns_arr[t] - baseline
                policies[sid].update(obs_t, traj["actions"][t], advantage, lr)
                policies[sid].update_critic(obs_t, returns_arr[t], lr)
            policies[sid].decay_noise()

        total = sum(episode_reward.values())
        returns_history.append(total)
        if (episode + 1) % 10 == 0:
            logger.info(
                f"Episode {episode+1:3d} | "
                f"Total reward: {total:8.2f} | "
                f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
            )

    logger.info("Training completed.")
    return env, policies, returns_history


# ============================================================================
# Ray RLlib Training (optional)
# ============================================================================
def train_rllib(num_iterations: int = 50):
    """Train with Ray RLlib PPO via the HERON RLlibBasedHeronEnv.

    The RLlibBasedHeronEnv wraps the HERON HeronEnv and handles:
    - Observation vectorization (Observation -> np.ndarray)
    - __all__ injection in terminated/truncated
    - max_steps truncation
    - Agent ID filtering (only agents with action_space)

    Args:
        num_iterations: Number of PPO training iterations
    """
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from heron.adaptors.rllib import RLlibBasedHeronEnv
    except ImportError:
        logger.error("Ray RLlib not installed. Run: pip install 'ray[rllib]'")
        return

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    steps_per_episode = 288  # 288 * 300s = 86400s = 1 day

    num_stations = 2
    num_chargers = 5

    config = (
        PPOConfig()
        .environment(
            env=RLlibBasedHeronEnv,
            env_config={
                "agents": [
                    {"agent_id": f"station_{i}_slot_{j}",
                     "agent_cls": ChargingSlot,
                     "p_max_kw": 150.0,
                     "coordinator": f"station_{i}"}
                    for i in range(num_stations)
                    for j in range(num_chargers)
                ],
                "coordinators": [
                    {"coordinator_id": f"station_{i}",
                     "agent_cls": StationCoordinator}
                    for i in range(num_stations)
                ],
                "env_class": ChargingEnv,
                "env_kwargs": {
                    "arrival_rate": 10.0,
                    "dt": 300.0,
                    "episode_length": 86400.0,
                },
                "max_steps": steps_per_episode,
            },
        )
        .framework("torch")
        .training(
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=10.0,
            train_batch_size=4000,
        )
        .env_runners(
            num_env_runners=2,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
        )
        .resources(num_gpus=0, num_cpus_for_main_process=2)
        .multi_agent(
            policies={"station_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kw: "station_policy",
        )
    )

    logger.info("Building PPO algorithm...")
    algo = config.build_algo()

    try:
        for i in range(num_iterations):
            result = algo.train()
            reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
            episodes = result.get("num_episodes_done", 0)
            logger.info(f"Iter {i:3d} | Reward mean: {reward_mean:7.2f} | Episodes: {episodes}")

            if i % 10 == 0 and i > 0:
                checkpoint = algo.save()
                logger.info(f"Checkpoint: {checkpoint.checkpoint.path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        algo.stop()
        ray.shutdown()


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--rllib":
        train_rllib(num_iterations=50)
    elif len(sys.argv) > 1 and sys.argv[1] == "--event-driven":
        from case_studies.power.ev_public_charging_case.run_event_driven import main as run_ed
        run_ed()
    else:
        train_simple(num_episodes=50)

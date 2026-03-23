"""Validate HERON-RLlib integration with MAPPO and IPPO.

Uses the same action-passing scenario as ``test_action_passing.py``
(two devices coordinated by one coordinator with VerticalProtocol),
but trains with RLlib algorithms through the ``RLlibBasedHeronEnv``.

See also:
  - ``test_qmix_action_passing.py``   — QMIX (custom PyTorch)
  - ``test_maddpg_action_passing.py``  — MADDPG (custom PyTorch)

Run::

    python tests/integration/test_rllib_action_passing.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import ray

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.feature import Feature
from heron.core.action import Action
from heron.envs.base import HeronEnv
from heron.protocols.vertical import VerticalProtocol
from heron.adaptors.rllib import RLlibBasedHeronEnv


# =============================================================================
# Environment components (mirrors test_action_passing.py)
# =============================================================================

@dataclass(slots=True)
class DevicePowerFeature(Feature):
    """Power state feature for devices."""
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
    """Field agent that directly controls a power device."""

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
    """Minimal multi-agent env for testing action passing."""

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


# =============================================================================
# Shared env_config (passed to RLlibBasedHeronEnv)
# =============================================================================

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
# Helpers
# =============================================================================

def _get_mean_reward(result) -> float:
    """Extract mean episode reward across Ray/RLlib API versions."""
    for accessor in [
        lambda r: r["env_runners"]["episode_return_mean"],
        lambda r: r["env_runners"]["episode_reward_mean"],
        lambda r: r["episode_reward_mean"],
        lambda r: r["sampler_results"]["episode_reward_mean"],
    ]:
        try:
            val = accessor(result)
            if val is not None:
                return float(val)
        except (KeyError, TypeError):
            continue
    return float("nan")


# =============================================================================
# Sanity check
# =============================================================================

def sanity_check():
    """Verify that the adapter wraps, resets, and steps correctly."""
    print("\n--- Sanity Check ---")
    adapter = RLlibBasedHeronEnv({
        **ACTION_PASSING_ENV_CONFIG,
        "max_steps": 30,
    })
    agent_ids = adapter.get_agent_ids()
    print(f"  Agent IDs:  {sorted(agent_ids)}")
    print(f"  Obs space:  {adapter.observation_space}")
    print(f"  Act space:  {adapter.action_space}")

    obs, info = adapter.reset(seed=42)
    assert set(obs.keys()) == agent_ids, "Agent IDs mismatch after reset"
    for aid, o in obs.items():
        assert adapter._obs_spaces[aid].contains(o), f"{aid} obs shape mismatch"

    actions = {aid: adapter.action_space[aid].sample() for aid in agent_ids}
    obs2, rew, term, trunc, info2 = adapter.step(actions)
    assert set(obs2.keys()) == agent_ids, "Agent IDs mismatch after step"
    assert "__all__" in term and "__all__" in trunc
    print("  Reset / step shapes OK")

    print("--- Sanity Check PASSED ---\n")


# =============================================================================
# Algorithm runners
# =============================================================================

def run_mappo(num_iters: int = 5, max_steps: int = 30) -> dict:
    """MAPPO: PPO with a single shared policy across all agents."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    print("\n" + "=" * 60)
    print("MAPPO: PPO + shared policy")
    print("=" * 60)

    config = (
        PPOConfig()
        .environment(
            env=RLlibBasedHeronEnv,
            env_config={
                **ACTION_PASSING_ENV_CONFIG,
                "max_steps": max_steps,
            },
        )
        .multi_agent(
            policies={"shared": PolicySpec()},
            policy_mapping_fn=lambda aid, *a, **kw: "shared",
        )
        .training(
            train_batch_size=200,
            minibatch_size=64,
            num_epochs=5,
            lr=5e-4,
        )
        .env_runners(num_env_runners=0)
    )

    algo = config.build()
    rewards = []
    for i in range(num_iters):
        result = algo.train()
        r = _get_mean_reward(result)
        rewards.append(r)
        print(f"  Iter {i + 1}/{num_iters}: mean_reward = {r:.3f}")
    algo.stop()
    return {"algorithm": "MAPPO", "rewards": rewards}


def run_ippo(num_iters: int = 5, max_steps: int = 30) -> dict:
    """IPPO: PPO with independent per-agent policies."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    print("\n" + "=" * 60)
    print("IPPO: PPO + per-agent policies")
    print("=" * 60)

    config = (
        PPOConfig()
        .environment(
            env=RLlibBasedHeronEnv,
            env_config={
                **ACTION_PASSING_ENV_CONFIG,
                "max_steps": max_steps,
            },
        )
        .multi_agent(
            policies={
                "device_1_policy": PolicySpec(),
                "device_2_policy": PolicySpec(),
            },
            policy_mapping_fn=lambda aid, *a, **kw: f"{aid}_policy",
        )
        .training(
            train_batch_size=200,
            minibatch_size=64,
            num_epochs=5,
            lr=5e-4,
        )
        .env_runners(num_env_runners=0)
    )

    algo = config.build()
    rewards = []
    for i in range(num_iters):
        result = algo.train()
        r = _get_mean_reward(result)
        rewards.append(r)
        print(f"  Iter {i + 1}/{num_iters}: mean_reward = {r:.3f}")
    algo.stop()
    return {"algorithm": "IPPO", "rewards": rewards}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_cpus=4)

    print("=" * 60)
    print("HERON RLlib Integration Test — Action Passing")
    print("=" * 60)
    print("Agents: device_1, device_2  (field agents)")
    print("Protocol: VerticalProtocol")
    print("Algorithms: MAPPO, IPPO  (see also test_qmix_*, test_maddpg_*)")

    sanity_check()

    all_results = []
    all_results.append(run_mappo())
    all_results.append(run_ippo())

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for res in all_results:
        algo = res["algorithm"]
        rews = res["rewards"]
        if rews:
            print(f"  {algo:8s}: final_reward = {rews[-1]:.3f}")
        else:
            print(f"  {algo:8s}: NO DATA")

    print()
    print("Additional algorithms (separate scripts):")
    print("  QMIX:   python tests/integration/test_qmix_action_passing.py")
    print("  MADDPG: python tests/integration/test_maddpg_action_passing.py")
    print()

    ray.shutdown()

# heron/envs/adapters/pettingzoo.py
"""
PettingZoo adapter for HERON environments.

Compatibility: Ray 2.40+ / 3.0 with PettingZoo 1.24+

This adapter provides a composition-based wrapper around HeronBaseEnv to expose
the PettingZoo ParallelEnv interface. It is designed to work seamlessly with
Ray RLlib's ParallelPettingZooEnv wrapper for multi-agent training.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np

from heron.envs.base import HeronBaseEnv, AgentID
from heron.messaging.base import MessageBroker

# PettingZoo is an optional dependency
try:
    from pettingzoo import ParallelEnv  # type: ignore

    PETTINGZOO_AVAILABLE = True
    _PZ_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    ParallelEnv = None  # type: ignore
    PETTINGZOO_AVAILABLE = False
    _PZ_IMPORT_ERROR = e

# Ray 2.40+ / 3.0 compatibility check (optional)
try:
    from ray import __version__ as ray_version
    RAY_AVAILABLE = True
    RAY_MAJOR_VERSION = int(ray_version.split('.')[0])
except Exception:  # pragma: no cover
    RAY_AVAILABLE = False
    RAY_MAJOR_VERSION = 0


class PettingZooParallelEnv(ParallelEnv):  # type: ignore[misc]
    """
    HERON -> PettingZoo ParallelEnv adapter (composition-based).

    **Compatibility:** Ray 2.40+ / 3.0 with PettingZoo 1.24+

    Key design choices:
    - Composition (self.core) instead of multi-inheritance to avoid MRO problems
    - HERON core lifecycle kept separate from PettingZoo lifecycle
    - Agent space management via dict-based `observation_spaces` and `action_spaces`
    - Method-based space accessors (`observation_space(agent)`, `action_space(agent)`)
      for Ray 2.40+ / 3.0 RLlib wrapper compatibility

    Ray 2.40+ / 3.0 Changes Addressed:
    - Removed duplicate space method definitions
    - Added dynamic space initialization with proper error handling
    - Ensured reset() returns (Dict[obs], Dict[info]) per Parallel API
    - Ensured step() returns (obs, rewards, terminations, truncations, infos)
    - Agent list management compatible with Ray's `ParallelPettingZooEnv` wrapper

    This adapter is intentionally minimal and test-driven:
    - Provides `_set_agent_ids()` and `_init_spaces()` aliases (backward compat)
    - Delegates HERON core methods via __getattr__
    - Implements PettingZoo Parallel API: reset() and step()
    """

    metadata = {
        "name": "heron_pettingzoo_parallel",
        "render_modes": ["human", "ansi", "rgb_array"],
    }

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        if ParallelEnv is None:  # pragma: no cover
            raise ImportError(
                "PettingZoo is required for PettingZooParallelEnv. "
                "Install with: pip install pettingzoo"
            ) from _PZ_IMPORT_ERROR

        # Init PettingZoo base
        ParallelEnv.__init__(self)

        # Create & init HERON core (mixin-style core; must call _init_heron_core)
        self.core = HeronBaseEnv()
        self.core._init_heron_core(env_id=env_id, message_broker=message_broker)

        # PettingZoo required fields
        self.possible_agents: List[AgentID] = []
        self.agents: List[AgentID] = []

        # PettingZoo expects these dicts (can be filled later)
        self.observation_spaces: Dict[AgentID, gym.Space] = {}
        self.action_spaces: Dict[AgentID, gym.Space] = {}

    # ----------------------------
    # Delegation to HERON core
    # ----------------------------
    def __getattr__(self, name: str):
        """Delegate missing attributes/methods to HERON core."""
        if name == "core":
            raise AttributeError
        return getattr(self.core, name)

    # ----------------------------
    # Compatibility aliases (tests expect these)
    # ----------------------------
    def _set_agent_ids(self, agent_ids: List[AgentID]) -> None:
        self.set_agent_ids(agent_ids)

    def _init_spaces(
        self,
        observation_spaces: Optional[Dict[AgentID, gym.Space]] = None,
        action_spaces: Optional[Dict[AgentID, gym.Space]] = None,
    ) -> None:
        self.init_spaces(observation_spaces=observation_spaces, action_spaces=action_spaces)

    # ----------------------------
    # Helper: HERON Observation.local -> np.ndarray
    # ----------------------------
    def _to_np_obs(self, local: Any) -> np.ndarray:
        """
        Enhanced conversion of HERON Observation.local -> numpy array.
        Supports both FieldAgent (simple vector) and CoordinatorAgent (nested dict).
        """
        if isinstance(local, np.ndarray):
            return local

        if isinstance(local, dict):
            if "coordinator_state" in local:
                return np.array(local["coordinator_state"], dtype=np.float32).flatten()

            if len(local) == 1 and "value" in local:
                return np.asarray([local["value"]], dtype=np.float32)

            vals = []
            for v in local.values():
                if isinstance(v, (int, float, np.number)):
                    vals.append(v)
                elif isinstance(v, np.ndarray):
                    vals.extend(v.flatten().tolist())
            return np.asarray(vals, dtype=np.float32)

        try:
            return np.asarray(local, dtype=np.float32).flatten()
        except Exception:
            return np.asarray([local], dtype=object)

    # ----------------------------
    # Public adapter helpers
    # ----------------------------
    def set_agent_ids(self, agent_ids: List[AgentID]) -> None:
        """Define the agent population for this PettingZoo env."""
        self.possible_agents = list(agent_ids)
        self.agents = list(agent_ids)

    def init_spaces(
        self,
        observation_spaces: Optional[Dict[AgentID, gym.Space]] = None,
        action_spaces: Optional[Dict[AgentID, gym.Space]] = None,
    ) -> None:
        """
        Initialize/update per-agent observation/action spaces.

        If None, tries to infer from registered HERON agents.
        """
        if observation_spaces is None:
            observation_spaces = self.core.get_agent_observation_spaces()
        if action_spaces is None:
            action_spaces = self.core.get_agent_action_spaces()

        self.observation_spaces = dict(observation_spaces)
        self.action_spaces = dict(action_spaces)

    # PettingZoo preferred space accessors (Ray 2.40+ / 3.0 compatible)
    def observation_space(self, agent: AgentID) -> gym.Space:
        """Get observation space for a specific agent.

        Ray 2.40+ / 3.0 expects this method for dynamic space lookup.
        Compatible with both property-based and method-based access patterns.
        """
        if agent not in self.observation_spaces:
            self.init_spaces()

        if agent not in self.observation_spaces:
            raise KeyError(f"Observation space for agent '{agent}' not found.")
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.Space:
        """Get action space for a specific agent.

        Ray 2.40+ / 3.0 expects this method for dynamic space lookup.
        Compatible with both property-based and method-based access patterns.
        """
        if agent not in self.action_spaces:
            self.init_spaces()

        if agent not in self.action_spaces:
            raise KeyError(f"Action space for agent '{agent}' not found after sync. "
                           f"Check if the agent is registered and has an action_space attribute.")
        return self.action_spaces[agent]


    # ----------------------------
    # PettingZoo Parallel API
    # ----------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, dict]]:
        """
        Reset the environment and return initial observations.

        Ray 2.40+ / 3.0 compatible reset signature.

        Args:
            seed: Optional random seed for reproducibility
            options: Optional reset options dict

        Returns:
          observations: dict[agent_id] -> np.ndarray
          infos: dict[agent_id] -> dict (empty by default)
        """
        # PettingZoo convention: reset re-activates current agents
        self.agents = list(self.possible_agents)

        # HERON: reset agent timesteps and wire message broker info
        self.core.reset_agents()
        self.core.configure_agents_for_distributed()

        observations = self.core.get_observations()
        obs_dict = {aid: self._to_np_obs(observations[aid].local) for aid in observations}
        infos = {aid: {} for aid in self.agents}
        return obs_dict, infos

    def step(
        self,
        actions: Dict[AgentID, np.ndarray],
    ) -> Tuple[
        Dict[AgentID, np.ndarray],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, dict],
    ]:
        """
        Execute one environment step for all active agents.

        Ray 2.40+ / 3.0 compatible step signature (Gymnasium-style).

        Args:
            actions: dict[agent_id] -> action (np.ndarray or gym action)

        Returns:
          observations: dict[agent_id] -> np.ndarray
          rewards: dict[agent_id] -> float
          terminations: dict[agent_id] -> bool (episode ended)
          truncations: dict[agent_id] -> bool (time limit reached)
          infos: dict[agent_id] -> dict (metadata)
        """
        # HERON: apply actions to agents
        self.core.apply_actions(actions)

        # HERON: collect new observations
        observations = self.core.get_observations()
        obs_dict = {aid: self._to_np_obs(observations[aid].local) for aid in observations}

        # Minimal default RL signals (tests only check contract; env author can override)
        rewards = {aid: 0.0 for aid in self.agents}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {aid: {} for aid in self.agents}

        return obs_dict, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        self.core.close_heron()

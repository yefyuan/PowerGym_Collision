"""RLlib adapter for HERON multi-agent environments.

Provides ``RLlibBasedHeronEnv`` — a thin wrapper that converts a HERON
``HeronEnv`` into an RLlib-compatible ``MultiAgentEnv`` so that
HERON environments can be plugged directly into RLlib training
pipelines (PPO / MAPPO / IPPO, QMIX, etc.).

Usage — pass agent/coordinator specs directly in ``env_config``::

    config = (
        PPOConfig()
        .environment(
            env=RLlibBasedHeronEnv,
            env_config={
                "agents": [
                    {"agent_id": "drone_0", "agent_cls": Drone,
                     "features": [DroneFeature()], "coordinator": "fleet",
                     "schedule_config": ScheduleConfig(...)},   # optional per-agent
                ],
                "coordinators": [
                    {"coordinator_id": "fleet", "agent_cls": FleetManager,
                     "schedule_config": ScheduleConfig(...)},   # optional per-coordinator
                ],
                "system": {
                    "schedule_config": ScheduleConfig(...),      # optional for SystemAgent
                },
                "simulation": my_sim_func,
                "max_steps": 100,
            },
        )
    )
    algo = config.build()
    algo.train()
"""

from typing import Any, Dict, Optional, Set

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from heron.envs.base import HeronEnv
from heron.envs.builder import EnvBuilder


def _build_heron_env(config: Dict[str, Any]) -> HeronEnv:
    """Build a HERON env from a flat config dict.

    Reads ``agents``, ``coordinators``, and either ``simulation`` or
    ``env_class`` keys and constructs an ``EnvBuilder`` internally.

    An optional ``"system"`` dict with a ``"schedule_config"`` key sets the
    auto-created SystemAgent's tick config.  Individual agent/coordinator
    specs can carry their own ``schedule_config`` kwarg independently.
    """
    if "agents" not in config:
        raise ValueError(
            "env_config must contain 'agents' (list of agent specs)."
        )

    builder = EnvBuilder(config.get("env_id", "default_env"))


    # Add agents and coordinators
    _agent_known_keys = {"agent_id", "agent_cls", "features", "schedule_config", "coordinator"}
    for agent_cfg in config["agents"]:
        extra = {k: v for k, v in agent_cfg.items() if k not in _agent_known_keys}
        builder.add_agent(
            agent_id=agent_cfg["agent_id"],
            agent_cls=agent_cfg["agent_cls"],
            features=agent_cfg.get("features"),
            schedule_config=agent_cfg.get("schedule_config"),
            coordinator=agent_cfg.get("coordinator"),
            **extra,
        )

    _coord_known_keys = {"coordinator_id", "agent_cls", "features", "protocol", "schedule_config", "subordinates"}
    for coord_cfg in config.get("coordinators", []):
        coord_extra = {k: v for k, v in coord_cfg.items() if k not in _coord_known_keys}
        builder.add_coordinator(
            coordinator_id=coord_cfg["coordinator_id"],
            agent_cls=coord_cfg.get("agent_cls"),
            features=coord_cfg.get("features"),
            protocol=coord_cfg.get("protocol"),
            schedule_config=coord_cfg.get("schedule_config"),
            subordinates=coord_cfg.get("subordinates"),
            **coord_extra,
        )
    # Add system agent
    system_cfg = config.get("system", {})
    builder.add_system_agent(schedule_config=system_cfg.get("schedule_config"))

    if "env_class" in config:
        builder.env_class(config["env_class"], **config.get("env_kwargs", {}))
    elif "simulation" in config:
        builder.simulation(config["simulation"])

    return builder.build()


class RLlibBasedHeronEnv(MultiAgentEnv):
    """Wraps a HERON ``HeronEnv`` for RLlib training.

    The adapter exposes HERON agents (those whose ``action_space`` is not
    ``None``) as RLlib agents.  HERON ``Observation`` objects are flattened
    to float32 numpy vectors and RLlib actions are forwarded directly to the
    underlying env.

    HERON's ``Action`` class natively supports continuous (``Box``), discrete
    (``Discrete`` / ``MultiDiscrete``), and mixed action spaces — define them
    in your agent's ``init_action()`` and they will be exposed to RLlib
    automatically.

    Parameters (passed via *config* dict)
    --------------------------------------
    agents : list[dict]
        Agent specs: ``{"agent_id", "agent_cls", "features"?, "coordinator"?, **kwargs}``.
    coordinators : list[dict]
        Coordinator specs: ``{"coordinator_id", "agent_cls"?, "features"?,
        "protocol"?, "subordinates"?, **kwargs}``.
    simulation : Callable
        Simulation function for ``SimpleEnv`` auto-bridge.
        Mutually exclusive with *env_class*.
    env_class : type, optional
        Custom ``HeronEnv`` subclass. Use instead of *simulation* when
        you need a full custom env (e.g. with ``run_simulation``).
    env_kwargs : dict, optional
        Extra kwargs forwarded to *env_class* constructor.
    env_id : str, optional
        Environment identifier (default ``"default_env"``).
    max_steps : int, optional
        Episode truncation length (default 50).
    agent_ids : list[str], optional
        Subset of agent IDs to expose.  Defaults to every registered
        agent whose ``action_space is not None``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}

        self.max_steps: int = config.get("max_steps", 50)
        self._step_count: int = 0

        # ---- build the underlying HERON env ----
        self.heron_env: HeronEnv = _build_heron_env(config)

        # One reset to materialise observation shapes and agent spaces.
        # SystemAgent.reset() returns vectorized np.ndarray observations
        # consistent with the format from step() / proxy.get_step_results().
        init_obs, _ = self.heron_env.reset(seed=0)

        # ---- determine exposed agents ----
        if "agent_ids" in config:
            self._agent_ids: Set[str] = set(config["agent_ids"])
        else:
            self._agent_ids = {
                aid
                for aid, ag in self.heron_env.registered_agents.items()
                if ag.action_space is not None
            }

        # ---- per-agent spaces ----
        self._obs_spaces: Dict[str, gym.Space] = {}
        self._act_spaces: Dict[str, gym.Space] = {}

        for aid in sorted(self._agent_ids):
            ag = self.heron_env.registered_agents[aid]
            obs_vec = np.asarray(init_obs[aid], dtype=np.float32).flatten()

            self._obs_spaces[aid] = Box(
                -np.inf, np.inf, shape=obs_vec.shape, dtype=np.float32,
            )
            self._act_spaces[aid] = ag.action_space

        self.possible_agents = sorted(self._agent_ids)
        self.agents = list(self.possible_agents)

        # Dict-based spaces: the new API stack introspects .spaces to
        # map agent IDs → per-policy spaces automatically.
        self.observation_space = DictSpace(
            {aid: self._obs_spaces[aid] for aid in self.possible_agents}
        )
        self.action_space = DictSpace(
            {aid: self._act_spaces[aid] for aid in self.possible_agents}
        )

    # ------------------------------------------------------------------ #
    #  RLlib interface                                                      #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        raw, _ = self.heron_env.reset(seed=seed)
        obs = {}
        for aid in self._agent_ids:
            if aid in raw:
                obs[aid] = np.asarray(raw[aid], dtype=np.float32).flatten()
            else:
                obs[aid] = np.zeros(self._obs_spaces[aid].shape, dtype=np.float32)
        return obs, {aid: {} for aid in obs}

    def step(self, action_dict: Dict[str, Any]):
        self._step_count += 1

        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = (
            self.heron_env.step(action_dict)
        )

        # --- activity-aware filtering (agents self-check via is_active_at) ---
        agents = self.heron_env.registered_agents
        step = self._step_count

        active_now = {
            aid for aid in self._agent_ids
            if aid in agents and agents[aid].is_active_at(step)
        }
        all_flags = {aid: aid in active_now for aid in self._agent_ids}

        hit_limit = self._step_count >= self.max_steps
        episode_done = raw_term.get("__all__", False) or hit_limit

        # RLlib requires obs for all agents on every step (including final).
        # Use raw obs when available, fall back to zeros for inactive agents.
        obs = {}
        for aid in self._agent_ids:
            if aid in raw_obs:
                obs[aid] = np.asarray(raw_obs[aid], dtype=np.float32).flatten()
            else:
                obs[aid] = np.zeros(
                    self._obs_spaces[aid].shape,
                    dtype=np.float32,
                )
        rew = {aid: float(raw_rew.get(aid, 0.0)) for aid in self._agent_ids}

        term = {aid: bool(raw_term.get(aid, False)) for aid in self._agent_ids}
        term["__all__"] = raw_term.get("__all__", False)

        trunc = {aid: hit_limit for aid in self._agent_ids}
        trunc["__all__"] = hit_limit

        info = {aid: raw_info.get(aid, {}) for aid in self._agent_ids}
        for aid in info:
            info[aid]["is_active"] = all_flags

        return obs, rew, term, trunc, info

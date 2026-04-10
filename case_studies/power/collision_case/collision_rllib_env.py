"""RLlib MultiAgentEnv for collision experiments (HERON + PowerGym agents only).

Exposes one policy per microgrid (MG1, MG2, MG3). Each RL action is a single
normalized vector that ``VerticalProtocol`` splits across ESS / DG / PV / WT
field agents under that ``CollisionGridAgent``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from heron.core.action import Action

from collision_case.collision_env import CollisionEnv
from collision_case.system_builder import create_collision_system

RL_AGENT_IDS = ("MG1", "MG2", "MG3")

# Finite bounds so Gymnasium's checker agrees and RL nets never see NaN/Inf from the env.
_OBS_CLIP = 1e6


def _as_rl_obs(raw) -> np.ndarray:
    v = np.asarray(raw, dtype=np.float32).ravel()
    v = np.nan_to_num(v, nan=0.0, posinf=np.float32(_OBS_CLIP), neginf=np.float32(-_OBS_CLIP))
    np.clip(v, np.float32(-_OBS_CLIP), np.float32(_OBS_CLIP), out=v)
    return np.ascontiguousarray(v)


def _joint_action_dim(mg_agent) -> int:
    return sum(sub.action.dim_c + sub.action.dim_d for sub in mg_agent.subordinates.values())


def _wrap_joint(vec: np.ndarray, dim: int) -> Action:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] != dim:
        raise ValueError(f"Expected action dim {dim}, got {v.shape}")
    a = Action()
    lo = np.full(dim, -1.0, dtype=np.float32)
    hi = np.full(dim, 1.0, dtype=np.float32)
    a.set_specs(dim_c=dim, dim_d=0, ncats=0, range=(lo, hi))
    a.set_values(c=v)
    return a


class CollisionRLlibMultiAgentEnv(MultiAgentEnv):
    """Three RL agents (MG1–MG3); underlying env uses full HERON hierarchy."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = dict(config or {})

        if "dataset_path" not in cfg:
            raise ValueError("env_config must include 'dataset_path'")

        self._episode_steps: int = int(cfg.get("episode_steps", 24))
        self._step_count: int = 0

        system = create_collision_system(
            share_reward=cfg.get("share_reward", True),
            penalty=float(cfg.get("penalty", 10.0)),
            enable_async=cfg.get("enable_async", False),
            field_tick_s=float(cfg.get("field_tick_s", 5.0)),
            coord_tick_s=float(cfg.get("coord_tick_s", 10.0)),
            system_tick_s=float(cfg.get("system_tick_s", 30.0)),
            jitter_ratio=float(cfg.get("jitter_ratio", 0.1)),
            train=True,
        )

        self._env = CollisionEnv(
            system_agent=system,
            dataset_path=str(cfg["dataset_path"]),
            episode_steps=self._episode_steps,
            dt=float(cfg.get("dt", 1.0)),
            share_reward=cfg.get("share_reward", True),
            penalty=float(cfg.get("penalty", 10.0)),
            log_path=cfg.get("log_path"),
        )

        self._mg_agents = {aid: self._env.registered_agents[aid] for aid in RL_AGENT_IDS}
        self._joint_dims = {aid: _joint_action_dim(self._mg_agents[aid]) for aid in RL_AGENT_IDS}

        init_obs, _ = self._env.reset(seed=0)
        self._obs_spaces: Dict[str, Box] = {}
        self._act_spaces: Dict[str, Box] = {}
        for aid in RL_AGENT_IDS:
            vec = _as_rl_obs(init_obs[aid])
            # Scalar bounds + shape: Gymnasium fills with dtype=np.float32 (no float64 warning).
            self._obs_spaces[aid] = Box(
                low=-_OBS_CLIP,
                high=_OBS_CLIP,
                shape=vec.shape,
                dtype=np.float32,
            )
            d = self._joint_dims[aid]
            self._act_spaces[aid] = Box(
                low=-1.0,
                high=1.0,
                shape=(d,),
                dtype=np.float32,
            )

        self.possible_agents = list(RL_AGENT_IDS)
        self.agents = list(RL_AGENT_IDS)

        self.observation_space = DictSpace(dict(self._obs_spaces))
        self.action_space = DictSpace(dict(self._act_spaces))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._step_count = 0
        raw, _ = self._env.reset(seed=seed)
        obs = {aid: _as_rl_obs(raw[aid]) for aid in RL_AGENT_IDS}
        return obs, {aid: {} for aid in RL_AGENT_IDS}

    def step(self, action_dict: Dict[str, Any]):
        self._step_count += 1

        full_actions: Dict[str, Any] = {}
        for aid in RL_AGENT_IDS:
            if aid not in action_dict:
                raise ValueError(f"Missing action for {aid}")
            full_actions[aid] = _wrap_joint(action_dict[aid], self._joint_dims[aid])

        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = self._env.step(full_actions)

        obs = {aid: _as_rl_obs(raw_obs[aid]) for aid in RL_AGENT_IDS}
        rew = {aid: float(raw_rew.get(aid, 0.0)) for aid in RL_AGENT_IDS}

        hit_limit = self._step_count >= self._episode_steps

        term = {aid: bool(raw_term.get(aid, False)) for aid in RL_AGENT_IDS}
        term["__all__"] = bool(raw_term.get("__all__", False))

        trunc = {aid: hit_limit for aid in RL_AGENT_IDS}
        trunc["__all__"] = hit_limit or bool(raw_trunc.get("__all__", False))

        info = {aid: dict(raw_info.get(aid, {})) for aid in RL_AGENT_IDS}

        return obs, rew, term, trunc, info

    def close(self):
        self._env.close()

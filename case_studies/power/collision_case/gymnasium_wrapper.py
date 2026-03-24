"""Gymnasium wrapper for CollisionEnv.

Wraps CollisionEnv to make it compatible with RLlib's requirements.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from collision_case.collision_env import CollisionEnv


class GymnasiumCollisionEnv(gym.Env):
    """Gymnasium-compatible wrapper for CollisionEnv.
    
    Wraps the HERON-based CollisionEnv to provide the standard
    Gymnasium interface required by RLlib.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize wrapper.
        
        Args:
            config: Environment configuration dict
        """
        from collision_case import create_collision_system
        
        # Create system agent
        system = create_collision_system(
            share_reward=config.get("share_reward", True),
            penalty=config.get("penalty", 10.0),
            enable_async=config.get("enable_async", False),
            field_tick_s=config.get("field_tick_s", 5.0),
            coord_tick_s=config.get("coord_tick_s", 10.0),
            system_tick_s=config.get("system_tick_s", 30.0),
            jitter_ratio=config.get("jitter_ratio", 0.1),
            train=True,
        )
        
        # Create CollisionEnv
        self.collision_env = CollisionEnv(
            system_agent=system,
            dataset_path=config["dataset_path"],
            episode_steps=config.get("episode_steps", 24),
            dt=config.get("dt", 1.0),
            share_reward=config.get("share_reward", True),
            penalty=config.get("penalty", 10.0),
            log_path=config.get("log_path"),
        )
        
        # Store config
        self.config = config
        
        # Build observation and action spaces after first reset
        self.observation_space = None
        self.action_space = None
        self._spaces_initialized = False
        
    def _init_spaces(self, obs: Dict):
        """Initialize spaces from first observation.
        
        Args:
            obs: First observation dict
        """
        if self._spaces_initialized:
            return
            
        # Build observation space
        obs_spaces = {}
        for agent_id, agent_obs in obs.items():
            if isinstance(agent_obs, np.ndarray):
                obs_spaces[agent_id] = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=agent_obs.shape, dtype=np.float32
                )
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # Build action space (from system agent)
        act_spaces = {}
        for agent_id, agent in self.collision_env._system_agent.get_all_agents().items():
            if agent_id != "system_agent" and hasattr(agent, 'action'):
                action = agent.action
                if hasattr(action, 'dim_c') and action.dim_c > 0:
                    # Continuous action space
                    low, high = action.get_range()
                    act_spaces[agent_id] = spaces.Box(
                        low=low, high=high, dtype=np.float32
                    )
        
        self.action_space = spaces.Dict(act_spaces)
        self._spaces_initialized = True
        
    def reset(self, *, seed=None, options=None):
        """Reset environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observations, info)
        """
        obs, info = self.collision_env.reset(seed=seed)
        
        # Initialize spaces on first reset
        if not self._spaces_initialized:
            self._init_spaces(obs)
        
        return obs, info
    
    def step(self, action: Dict):
        """Take environment step.
        
        Args:
            action: Dict of actions per agent
            
        Returns:
            Tuple of (obs, rewards, terminated, truncated, infos)
        """
        obs, rewards, terminated, truncated, infos = self.collision_env.step(action)
        return obs, rewards, terminated, truncated, infos
    
    def render(self):
        """Render environment (no-op)."""
        pass
    
    def close(self):
        """Close environment."""
        if hasattr(self.collision_env, 'close'):
            self.collision_env.close()

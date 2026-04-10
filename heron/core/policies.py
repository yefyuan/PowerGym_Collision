"""Policy interfaces for agent decision-making.

This module provides abstract policy interfaces and common implementations
for agent control in multi-agent systems.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Callable
from functools import wraps

import numpy as np

from heron.core.action import Action
from heron.core.observation import Observation


def obs_to_vector(method: Callable) -> Callable:
    """Decorator that converts observation to vector before calling method.

    The decorated method receives obs_vec (np.ndarray) instead of observation.
    Requires self to have obs_dim and extract_obs_vector() method.
    """
    @wraps(method)
    def wrapper(self, observation, *args, **kwargs):
        obs_vec = self.extract_obs_vector(observation, self.obs_dim)
        return method(self, obs_vec, *args, **kwargs)
    return wrapper


def vector_to_action(method: Callable) -> Callable:
    """Decorator that converts returned vector to Action object.

    The decorated method returns np.ndarray, which gets converted to Action.
    Requires self to have action_dim, action_range, and vec_to_action() method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        action_vec = method(self, *args, **kwargs)
        return self.vec_to_action(action_vec, self.action_dim, self.action_range)
    return wrapper


class Policy(ABC):
    """Abstract policy interface for agent decision-making.

    Policies can be:
    - Learned (RL algorithms)
    - Rule-based (heuristics, classical control)
    - Optimization-based (MPC, optimal control)

    For vector-based policies (most RL algorithms), subclasses can override
    _compute_action_vector() instead of forward() to get automatic observation
    extraction and action conversion.

    Attributes:
        observation_mode: Controls which observation components to use:
            - "full": Use both local and global (default, for centralized training)
            - "local": Use only local observations (for decentralized policies)
            - "global": Use only global information
    """
    observation_mode: str = "full"  # Default to full observations

    @abstractmethod
    def forward(self, observation: Observation) -> Optional[Action]:
        """Compute action from observation.

        Args:
            observation: Agent observation

        Returns:
            Action object, or None if no action is computed
        """
        pass

    def reset(self) -> None:
        """Reset policy state (e.g., hidden states for RNNs)."""
        pass

    # ============================================
    # Helper Methods for Policy Implementations
    # ============================================

    def extract_obs_vector(self, observation: Any, obs_dim: int) -> np.ndarray:
        """Extract observation vector from various formats.

        This is a common helper for policies that work with vector observations.
        Handles multiple formats for compatibility between training and deployment modes.

        Uses self.observation_mode to determine which observation components to extract:
        - "full": local + global (default)
        - "local": local only (for decentralized policies)
        - "global": global only

        Args:
            observation: Observation in various formats (Observation object, dict, or array)
            obs_dim: Expected observation dimension

        Returns:
            Numpy array of shape (obs_dim,)
        """
        if isinstance(observation, Observation):
            # Use appropriate vectorization based on observation_mode
            if self.observation_mode == "local":
                return observation.local_vector()
            elif self.observation_mode == "global":
                return observation.global_vector()
            else:  # "full" or any other value
                return observation.vector()
        elif isinstance(observation, dict):
            # Event-driven mode: observation from proxy.get_observation()
            # Structure: {"local": {"FeatureName": array([...])}, "global_info": ...}
            if "local" in observation and observation["local"]:
                local_features = observation["local"]
                # Get first feature vector
                first_feature_vec = list(local_features.values())[0]

                # Convert dict to array if needed (for when features are returned as dicts)
                if isinstance(first_feature_vec, dict):
                    # Try to extract numeric values from feature dict
                    values = [v for v in first_feature_vec.values() if isinstance(v, (int, float, np.number))]
                    if values:
                        return np.array(values, dtype=np.float32)[:obs_dim]
                elif isinstance(first_feature_vec, np.ndarray) and first_feature_vec.size > 0:
                    return first_feature_vec[:obs_dim] if len(first_feature_vec) > obs_dim else first_feature_vec
                elif isinstance(first_feature_vec, (list, tuple)):
                    arr = np.array(first_feature_vec, dtype=np.float32)
                    return arr[:obs_dim] if len(arr) > obs_dim else arr

            # Local is empty or invalid - return zeros as safe fallback
            return np.zeros(obs_dim, dtype=np.float32)
        elif isinstance(observation, np.ndarray) and observation.size > 0:
            # Ensure array matches expected dimension
            return observation[:obs_dim] if len(observation) > obs_dim else observation
        else:
            # Fallback to zeros
            return np.zeros(obs_dim, dtype=np.float32)

    def vec_to_action(self, action_vec: np.ndarray, action_dim: int,
                     action_range: tuple = (-1.0, 1.0)) -> Action:
        """Convert action vector to Action object.

        Args:
            action_vec: Action values as numpy array
            action_dim: Dimension of continuous action
            action_range: Tuple of (min, max) bounds for actions

        Returns:
            Action object with specified values and specs
        """
        action = Action()
        action.set_specs(
            dim_c=action_dim,
            range=(np.full(action_dim, action_range[0]), np.full(action_dim, action_range[1]))
        )
        action.set_values(action_vec)
        return action
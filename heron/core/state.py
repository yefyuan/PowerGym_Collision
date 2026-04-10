"""State abstractions for agent state management.

This module provides generic state containers that compose Features
and support visibility-based observation filtering.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from heron.core.feature import Feature
from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class State(ABC):
    owner_id: str
    owner_level: int

    # Features stored as dict for O(1) lookup by name
    features: Dict[str, Feature] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        """Concatenate all feature vectors into an array."""
        feature_vectors: List[np.ndarray] = []
        for feature in self.features.values():
            feature_vectors.append(feature.vector())

        return cat_f32(feature_vectors)

    def reset(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        for feature in self.features.values():
            feature.reset()

        if overrides is not None:
            self.update(overrides)

    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Apply batch updates to features. O(m) where m = len(updates).

        Args:
            updates: Mapping of feature names to field updates:
                {
                    "FeatureA": {"field1": 5.0, "field2": 1.0},
                    "FeatureB": {"state": "active"},
                    ...
                }
        """
        for feature_name, values in updates.items():
            if values is not None and feature_name in self.features:
                self.features[feature_name].set_values(**values)

    def update_feature(self, feature_name: str, **values: Any) -> None:
        """Update a feature by name. O(1) lookup."""
        if feature_name in self.features:
            self.features[feature_name].set_values(**values)

    def observed_by(self, requestor_id: str, requestor_level: int) -> Dict[str, np.ndarray]:
        observable_feature_dict = {}

        for feature in self.features.values():
            if feature.is_observable_by(
                requestor_id, requestor_level, self.owner_id, self.owner_level
            ):
                observable_feature_dict[feature.feature_name] = cat_f32([feature.vector()])

        self.validate_observation_dict(observable_feature_dict)
        return observable_feature_dict

    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        feature_dict: Dict[str, Any] = {}

        for feature in self.features.values():
            feature_dict[feature.feature_name] = feature.to_dict()

        if include_metadata:
            return {
                "_owner_id": self.owner_id,
                "_owner_level": self.owner_level,
                "_state_type": self.__class__.__name__,
                "features": feature_dict
            }

        return feature_dict

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "State":
        from heron.core.feature import get_feature_class

        # Extract metadata if present
        if "_owner_id" in state_dict and "_owner_level" in state_dict:
            owner_id = state_dict["_owner_id"]
            owner_level = state_dict["_owner_level"]
            state_type = state_dict.get("_state_type")
            features_dict = state_dict.get("features", {})
        else:
            # Fallback: no metadata, assume features are at top level
            owner_id = state_dict.get("owner_id", "unknown")
            owner_level = state_dict.get("owner_level", 1)
            state_type = None
            features_dict = {k: v for k, v in state_dict.items() if not k.startswith("_") and k not in ["owner_id", "owner_level"]}

        # Determine which State class to instantiate
        if state_type and cls == State:
            # If called on base State class, use the type from metadata
            # Use globals() for late binding (classes defined later in file)
            import sys
            current_module = sys.modules[__name__]
            state_class = getattr(current_module, state_type, None)
            if state_class is None:
                raise ValueError(f"Unknown state type: {state_type}")
        else:
            # Called on concrete subclass, use it directly
            state_class = cls

        # Create State instance
        state = state_class(owner_id=owner_id, owner_level=owner_level)

        # Reconstruct features using registry
        for feature_name, feature_data in features_dict.items():
            if feature_name.startswith("_"):
                continue  # Skip internal metadata fields

            try:
                feature_class = get_feature_class(feature_name)
                feature_obj = feature_class.from_dict(feature_data)
                state.features[feature_name] = feature_obj
            except ValueError as e:
                # Feature not registered - skip with warning
                print(f"Warning: Skipping unregistered feature '{feature_name}': {e}")

        return state

    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> None:
        """Validate the collected feature vectors for consistency.

        Override in subclasses to add custom validation.
        """
        pass


@dataclass(slots=True)
class FieldAgentState(State):
    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> None:
        """Validate that all observation vectors are 1D."""
        for vector in obs_dict.values():
            if not (isinstance(vector, np.ndarray) and vector.ndim == 1):
                raise NotImplementedError(
                    "Only 1D vector observations supported. "
                    "Got: " + str(type(vector))
                )


@dataclass(slots=True)
class CoordinatorAgentState(State):
    pass


@dataclass(slots=True)
class SystemAgentState(State):
    pass



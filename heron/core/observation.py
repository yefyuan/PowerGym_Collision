from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class Observation:
    local: Dict[str, Any] = field(default_factory=dict)
    global_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def vector(self) -> np.ndarray:
        """Flatten both local and global info into a single float32 array."""
        parts: list = []

        # Flatten local state
        self._flatten_dict_to_list(self.local, parts)

        # Flatten global info
        self._flatten_dict_to_list(self.global_info, parts)

        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def local_vector(self) -> np.ndarray:
        """Flatten only the local observation dict into a float32 array."""
        parts: list = []
        self._flatten_dict_to_list(self.local, parts)

        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def global_vector(self) -> np.ndarray:
        """Flatten only the global info dict into a float32 array."""
        parts: list = []
        self._flatten_dict_to_list(self.global_info, parts)

        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def __array__(self, dtype=None) -> np.ndarray:
        vec = self.vector()
        if dtype is not None:
            return vec.astype(dtype)
        return vec

    @property
    def shape(self) -> tuple:
        """Return shape of vectorized observation (array-like compatibility)."""
        return self.vector().shape

    @property
    def dtype(self):
        """Return dtype of vectorized observation (array-like compatibility)."""
        return np.float32

    def __len__(self) -> int:
        """Return length of vectorized observation (array-like compatibility)."""
        return len(self.vector())

    def __getitem__(self, key):
        """Support indexing like a numpy array (array-like compatibility)."""
        return self.vector()[key]

    def _flatten_dict_to_list(self, d: Dict, parts: list) -> None:
        """Recursively flatten a dictionary into a list of arrays.

        Args:
            d: Dictionary to flatten
            parts: List to append arrays to (modified in place)
        """
        for key in sorted(d.keys()):
            val = d[key]
            if isinstance(val, (int, float)):
                parts.append(np.array([val], dtype=np.float32))
            elif isinstance(val, np.ndarray):
                parts.append(val.ravel().astype(np.float32))
            elif isinstance(val, dict):
                # Recursively flatten nested dicts
                self._flatten_dict_to_list(val, parts)

    # =========================================================================
    # Serialization Methods (for async message passing)
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation to dictionary for message passing.

        Used in fully async event-driven mode (Option B with async_observations=True)
        where observations are sent via message broker instead of direct method calls.

        **Tricky Part - Nested Observation Serialization**:
        Subordinate observations in coordinator's local dict may themselves be
        Observation objects. These are recursively serialized. When deserializing,
        the receiver must know the structure to reconstruct nested Observations.

        **Follow-up Work Needed**:
        - Add schema/type hints to payload so receiver knows which fields are
          nested Observations vs plain dicts
        - Consider using a more robust serialization format (e.g., pickle, msgpack)
          for complex observation structures

        Returns:
            Serialized observation as dictionary
        """
        return {
            "timestamp": self.timestamp,
            "local": self._serialize_nested(self.local),
            "global_info": self._serialize_nested(self.global_info),
        }

    def _serialize_nested(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize nested structures for message passing.

        Args:
            data: Dictionary to serialize

        Returns:
            Serialized dictionary with numpy arrays converted to lists
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                result[k] = {"__type__": "ndarray", "data": v.tolist(), "dtype": str(v.dtype)}
            elif isinstance(v, Observation):
                result[k] = {"__type__": "Observation", "data": v.to_dict()}
            elif isinstance(v, dict):
                result[k] = self._serialize_nested(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        """Deserialize observation from dictionary.

        Used to reconstruct Observation from message broker payload.

        **Tricky Part - Type Reconstruction**:
        The serialized dict uses "__type__" markers to indicate special types
        (ndarray, Observation). Without these markers, arrays would remain as
        lists and nested Observations would remain as dicts.

        **Follow-up Work Needed**:
        - Handle missing "__type__" markers gracefully (backward compatibility)
        - Add validation for expected observation structure

        Args:
            d: Serialized observation dictionary

        Returns:
            Reconstructed Observation object
        """
        return cls(
            timestamp=d.get("timestamp", 0.0),
            local=cls._deserialize_nested(d.get("local", {})),
            global_info=cls._deserialize_nested(d.get("global_info", {})),
        )

    @classmethod
    def _deserialize_nested(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively deserialize nested structures from message payload.

        Args:
            data: Serialized dictionary

        Returns:
            Deserialized dictionary with types reconstructed
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                if v.get("__type__") == "ndarray":
                    result[k] = np.array(v["data"], dtype=v.get("dtype", "float32"))
                elif v.get("__type__") == "Observation":
                    result[k] = cls.from_dict(v["data"])
                else:
                    result[k] = cls._deserialize_nested(v)
            else:
                result[k] = v
        return result

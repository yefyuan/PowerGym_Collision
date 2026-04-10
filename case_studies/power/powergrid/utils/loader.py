"""Dataset loader for power grid environments.

Provides functionality to load power grid datasets for simulation.
"""

from typing import Any, Dict, Optional
import os


def load_dataset(path: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset from file.

    Args:
        path: Path to dataset file (pickle or h5 format).
              If None, returns empty dataset structure.

    Returns:
        Dataset dictionary with train/test splits containing:
        - load: Load profiles by area
        - solar: Solar generation profiles
        - wind: Wind generation profiles
        - price: Electricity prices
    """
    if path is None:
        return _create_empty_dataset()

    if not os.path.exists(path):
        print(f"Warning: Dataset file not found: {path}. Using empty dataset.")
        return _create_empty_dataset()

    # Try to load based on file extension
    if path.endswith('.pkl') or path.endswith('.pickle'):
        return _load_pickle(path)
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        return _load_hdf5(path)
    else:
        print(f"Warning: Unknown file format for {path}. Using empty dataset.")
        return _create_empty_dataset()


def _create_empty_dataset() -> Dict[str, Any]:
    """Create empty dataset structure for testing."""
    import numpy as np

    # Create minimal dataset with 24*30 hours of data (30 days)
    num_hours = 24 * 30

    empty_split = {
        "load": {
            "AVA": np.ones(num_hours),
            "BANC": np.ones(num_hours),
            "BANCMID": np.ones(num_hours),
            "AZPS": np.ones(num_hours),
        },
        "solar": {
            "NP15": np.zeros(num_hours),
        },
        "wind": {
            "NP15": np.zeros(num_hours),
        },
        "price": {
            "0096WD_7_N001": np.ones(num_hours) * 50.0,  # $50/MWh default
        },
    }

    return {
        "train": empty_split,
        "test": empty_split,
    }


def _load_pickle(path: str) -> Dict[str, Any]:
    """Load dataset from pickle file."""
    import pickle

    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load pickle file {path}: {e}")
        return _create_empty_dataset()


def _load_hdf5(path: str) -> Dict[str, Any]:
    """Load dataset from HDF5 file."""
    try:
        import h5py
        import numpy as np

        data = {"train": {}, "test": {}}

        with h5py.File(path, 'r') as f:
            for split in ["train", "test"]:
                if split in f:
                    split_group = f[split]
                    data[split] = {}
                    for category in ["load", "solar", "wind", "price"]:
                        if category in split_group:
                            cat_group = split_group[category]
                            data[split][category] = {
                                key: np.array(cat_group[key])
                                for key in cat_group.keys()
                            }

        return data
    except ImportError:
        print("Warning: h5py not available. Using empty dataset.")
        return _create_empty_dataset()
    except Exception as e:
        print(f"Warning: Failed to load HDF5 file {path}: {e}")
        return _create_empty_dataset()

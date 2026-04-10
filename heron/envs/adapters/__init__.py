# heron/envs/adapters/__init__.py

from __future__ import annotations

try:
    from .pettingzoo import PettingZooParallelEnv, PETTINGZOO_AVAILABLE
except Exception:  # pragma: no cover
    PettingZooParallelEnv = None  # type: ignore
    PETTINGZOO_AVAILABLE = False


__all__ = [
    "PettingZooParallelEnv",
    "PETTINGZOO_AVAILABLE",
]

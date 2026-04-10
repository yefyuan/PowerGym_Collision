"""Tests for EnvBuilder, including callable factory support."""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np
import pytest

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.builder import EnvBuilder
from heron.protocols.vertical import VerticalProtocol


# ── Minimal test fixtures ────────────────────────────────────────

@dataclass(slots=True)
class DummyFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "value" in kwargs:
            self.value = float(kwargs["value"])


class DummyAgent(FieldAgent):
    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(c=np.zeros(1, dtype=np.float32))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, np.ndarray):
            self.action.set_values(c=action.flatten()[:1])

    def apply_action(self) -> None:
        pass

    def compute_local_reward(self, local_state: dict) -> float:
        return 0.0


class CustomCoordinator(CoordinatorAgent):
    """Coordinator that hardcodes its own protocol (like FleetManager)."""

    def __init__(self, agent_id, subordinates, features=None, **kwargs):
        super().__init__(
            agent_id=agent_id,
            features=features or [],
            subordinates=subordinates,
            protocol=VerticalProtocol(),
            **kwargs,
        )


def _dummy_sim(agent_states: dict) -> dict:
    return agent_states


# ── Tests ────────────────────────────────────────────────────────

def test_builder_callable_returns_env():
    """EnvBuilder() should produce the same result as .build()."""
    builder = (
        EnvBuilder("test_callable")
        .add_agents("agent", DummyAgent, count=2, features=[DummyFeature()])
        .simulation(_dummy_sim)
    )
    env = builder()
    assert env is not None
    obs, info = env.reset()
    assert isinstance(obs, dict)


def test_builder_callable_with_config_arg():
    """Callable accepts a config dict (ignored) for RLlib compatibility."""
    builder = (
        EnvBuilder("test_cfg")
        .add_agents("agent", DummyAgent, count=2, features=[DummyFeature()])
        .simulation(_dummy_sim)
    )
    env = builder({"some_key": 123})
    assert env is not None
    obs, _ = env.reset()
    assert isinstance(obs, dict)


def test_builder_callable_produces_independent_envs():
    """Each call should produce a new, independent environment."""
    builder = (
        EnvBuilder("test_independent")
        .add_agents("agent", DummyAgent, count=2, features=[DummyFeature()])
        .simulation(_dummy_sim)
    )
    env1 = builder()
    env2 = builder()
    assert env1 is not env2

    env1.reset()
    env2.reset()


def test_builder_with_custom_coordinator_no_protocol_conflict():
    """Builder should not pass protocol= when None, avoiding TypeError
    with coordinator subclasses that hardcode their own protocol."""
    builder = (
        EnvBuilder("test_custom_coord")
        .add_agent("a0", DummyAgent, features=[DummyFeature()], coordinator="coord")
        .add_agent("a1", DummyAgent, features=[DummyFeature()], coordinator="coord")
        .add_coordinator("coord", agent_cls=CustomCoordinator)
        .simulation(_dummy_sim)
    )
    # This would raise TypeError before the protocol fix
    env = builder()
    assert env is not None
    obs, _ = env.reset()
    assert isinstance(obs, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

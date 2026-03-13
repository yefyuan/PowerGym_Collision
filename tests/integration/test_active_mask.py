"""Integration tests for is_active logic and action_mask in CTDE.

Tests heterogeneous tick rates, action masking, and is_active flags
for centralized critic awareness.
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.feature import Feature
from heron.core.action import Action
from heron.envs.base import HeronEnv
from heron.protocols.vertical import VerticalProtocol
from heron.scheduling.tick_config import ScheduleConfig
from heron.utils.typing import AgentID


# =============================================================================
# Test Agents and Environment
# =============================================================================

@dataclass(slots=True)
class CounterFeature(Feature):
    """Simple feature that tracks how many times apply_action was called."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0
    apply_count: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.value, self.apply_count], dtype=np.float32)

    def set_values(self, **kwargs: Any) -> None:
        if "value" in kwargs:
            self.value = kwargs["value"]
        if "apply_count" in kwargs:
            self.apply_count = kwargs["apply_count"]


class CountingAgent(FieldAgent):
    """Field agent that counts apply_action calls for testing."""

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def set_action(self, action: Any) -> None:
        if isinstance(action, np.ndarray):
            self.action.set_values(c=action.flatten()[:1])

    def apply_action(self) -> None:
        feat = self.state.features["CounterFeature"]
        feat.set_values(
            value=float(self.action.c[0]),
            apply_count=feat.apply_count + 1.0,
        )

    def compute_local_reward(self, local_state: dict) -> float:
        if "CounterFeature" in local_state:
            return -float(local_state["CounterFeature"][0]) ** 2
        return 0.0


class MaskedAgent(CountingAgent):
    """Agent that provides an action mask (3 discrete actions, mask 2nd)."""

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_d=1, ncats=[3])
        return action

    def set_action(self, action: Any) -> None:
        if isinstance(action, (int, np.integer)):
            self.action.set_values(d=np.array([int(action)]))
        elif isinstance(action, np.ndarray):
            self.action.set_values(d=action.flatten()[:1])

    def get_action_mask(self) -> Optional[np.ndarray]:
        # Mask out action index 1
        return np.array([True, False, True], dtype=bool)

    def apply_action(self) -> None:
        feat = self.state.features["CounterFeature"]
        feat.set_values(apply_count=feat.apply_count + 1.0)


class SimpleCoordinator(CoordinatorAgent):
    pass


class SimpleEnv(HeronEnv):
    def run_simulation(self, env_state, *args, **kwargs):
        return env_state

    def env_state_to_global_state(self, env_state) -> Dict:
        return {"agent_states": env_state.get("agent_states", {})}

    def global_state_to_env_state(self, global_state) -> Any:
        return global_state


# =============================================================================
# Helper to build env
# =============================================================================

def _build_env(
    fast_tick: float = 1.0,
    slow_tick: float = 3.0,
    use_masked_agent: bool = False,
) -> HeronEnv:
    """Build a 2-field-agent env with heterogeneous tick intervals."""
    fast_agent_cls = MaskedAgent if use_masked_agent else CountingAgent
    fast_agent = fast_agent_cls(
        agent_id="fast",
        features=[CounterFeature()],
        schedule_config=ScheduleConfig.deterministic(tick_interval=fast_tick),
    )
    slow_agent = CountingAgent(
        agent_id="slow",
        features=[CounterFeature()],
        schedule_config=ScheduleConfig.deterministic(tick_interval=slow_tick),
    )
    coordinator = SimpleCoordinator(
        agent_id="coord",
        subordinates={"fast": fast_agent, "slow": slow_agent},
        protocol=VerticalProtocol(),
    )
    system = SystemAgent(
        subordinates={"coord": coordinator},
    )
    env = SimpleEnv(system_agent=system)
    return env


def _build_and_reset(fast_tick=1.0, slow_tick=3.0, use_masked_agent=False, seed=0):
    """Build env and reset. Agents derive activity from their own tick_interval."""
    env = _build_env(fast_tick=fast_tick, slow_tick=slow_tick, use_masked_agent=use_masked_agent)
    env.reset(seed=seed)
    return env


# =============================================================================
# Tests
# =============================================================================

class TestIsActive:
    """Test that is_active correctly skips inactive agents."""

    def test_homogeneous_all_active(self):
        """All agents have same tick_interval -> all active every step."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=1.0)

        actions = {"fast": np.array([0.5]), "slow": np.array([-0.3])}
        obs, rew, term, trunc, info = env.step(actions)

        # Both agents should be in results
        assert "fast" in obs
        assert "slow" in obs
        assert "fast" in rew
        assert "slow" in rew

    def test_heterogeneous_step1_slow_inactive(self):
        """Step 1: slow agent (period=3) should NOT have apply_action called."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        actions = {"fast": np.array([0.5]), "slow": np.array([-0.3])}
        env.step(actions)

        # Slow agent is inactive at step 1 — apply_action should not have run
        slow = env.registered_agents["slow"]
        assert slow.state.features["CounterFeature"].apply_count == 0.0
        # Fast agent is active — apply_action should have run
        fast = env.registered_agents["fast"]
        assert fast.state.features["CounterFeature"].apply_count == 1.0

    def test_heterogeneous_step3_both_active(self):
        """Step 3: both agents are active (step % 3 == 0 and step % 1 == 0)."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        actions = {"fast": np.array([0.1])}
        for _ in range(2):
            env.step(actions)

        # Step 3: both should be active
        actions_both = {"fast": np.array([0.5]), "slow": np.array([-0.3])}
        obs, rew, term, trunc, info = env.step(actions_both)

        assert "fast" in rew
        assert "slow" in rew

    def test_inactive_agent_apply_action_not_called(self):
        """Inactive agents should NOT have apply_action() called."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        # Step 1: only fast is active
        actions = {"fast": np.array([0.5])}
        env.step(actions)

        # Check apply_count for slow agent — should still be 0
        slow_agent = env.registered_agents["slow"]
        assert slow_agent.state.features["CounterFeature"].apply_count == 0.0

        # Fast agent should have apply_count = 1
        fast_agent = env.registered_agents["fast"]
        assert fast_agent.state.features["CounterFeature"].apply_count == 1.0

    def test_active_agent_apply_action_called(self):
        """Active agents should have apply_action() called."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        # Run 3 steps so slow becomes active at step 3
        for i in range(3):
            env.step({"fast": np.array([0.1]), "slow": np.array([0.2])})

        fast_agent = env.registered_agents["fast"]
        slow_agent = env.registered_agents["slow"]

        # Fast: active all 3 steps
        assert fast_agent.state.features["CounterFeature"].apply_count == 3.0
        # Slow: active only at step 3
        assert slow_agent.state.features["CounterFeature"].apply_count == 1.0

    def test_agent_is_active_at(self):
        """Agent.is_active_at() should reflect activity period correctly."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        fast = env.registered_agents["fast"]
        slow = env.registered_agents["slow"]

        # fast: period=1 → active every step
        assert fast.is_active_at(1)
        assert fast.is_active_at(2)
        assert fast.is_active_at(3)

        # slow: period=3 → active at steps 3, 6, 9, ...
        assert not slow.is_active_at(1)
        assert not slow.is_active_at(2)
        assert slow.is_active_at(3)
        assert not slow.is_active_at(4)
        assert slow.is_active_at(6)


class TestIsActiveFlags:
    """Test is_active querying via Agent.is_active_at()."""

    def test_activity_flags_at_step1(self):
        """At step 1, fast is active and slow is not."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        fast = env.registered_agents["fast"]
        slow = env.registered_agents["slow"]
        assert fast.is_active_at(1) is True
        assert slow.is_active_at(1) is False

    def test_activity_flags_at_step3(self):
        """At step 3, both agents should be active."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        fast = env.registered_agents["fast"]
        slow = env.registered_agents["slow"]
        assert fast.is_active_at(3) is True
        assert slow.is_active_at(3) is True

    def test_homogeneous_always_active(self):
        """Homogeneous tick rates -> period=1 -> all agents always active."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=1.0)

        fast = env.registered_agents["fast"]
        slow = env.registered_agents["slow"]
        for step in range(1, 10):
            assert fast.is_active_at(step)
            assert slow.is_active_at(step)


class TestActionMask:
    """Test action_mask in info dict."""

    def test_action_mask_in_info(self):
        """Agent with get_action_mask() should have mask in info."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=1.0, use_masked_agent=True)

        actions = {"fast": 0, "slow": np.array([0.1])}
        _, _, _, _, info = env.step(actions)

        assert "action_mask" in info["fast"]
        expected_mask = np.array([True, False, True], dtype=bool)
        np.testing.assert_array_equal(info["fast"]["action_mask"], expected_mask)

    def test_no_action_mask_for_unmasked_agent(self):
        """Agent without get_action_mask() override should not have mask."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=1.0, use_masked_agent=True)

        actions = {"fast": 0, "slow": np.array([0.1])}
        _, _, _, _, info = env.step(actions)

        # slow is a CountingAgent (no mask)
        assert "action_mask" not in info.get("slow", {})


class TestBackwardCompatibility:
    """Ensure homogeneous tick rates produce identical behavior to before."""

    def test_same_obs_reward_structure(self):
        """With same tick intervals, step results should include all field agents."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=1.0)

        for step in range(5):
            actions = {"fast": np.array([0.1 * step]), "slow": np.array([-0.1 * step])}
            obs, rew, term, trunc, info = env.step(actions)

            # HeronEnv returns all agents (including coord/system); field agents must be present
            assert "fast" in obs
            assert "slow" in obs
            assert "fast" in rew
            assert "slow" in rew

    def test_agent_timestep_tracks(self):
        """Agent timestep should increment correctly on each step."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)

        fast = env.registered_agents["fast"]
        assert fast._timestep == 0.0
        env.step({"fast": np.array([0.1])})
        assert fast._timestep == 1.0
        env.step({"fast": np.array([0.1])})
        assert fast._timestep == 2.0

    def test_reset_clears_agent_timestep(self):
        """Reset should reset agent timestep to 0."""
        env = _build_and_reset(fast_tick=1.0, slow_tick=3.0)
        env.step({"fast": np.array([0.1])})

        env.reset(seed=1)
        fast = env.registered_agents["fast"]
        assert fast._timestep == 0.0

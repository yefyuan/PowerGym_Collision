"""Simplified environment with automatic state bridge.

``SimpleEnv`` eliminates the need to implement ``env_state_to_global_state()``
and ``global_state_to_env_state()`` by auto-generating them from agent
registrations.

The user's simulation function receives a flat dict::

    {agent_id: {feature_name: {field: value, ...}, ...}, ...}

and must return the same structure with updated values.

Example::

    def my_simulation(agent_states: dict) -> dict:
        for aid, features in agent_states.items():
            if "BatteryFeature" in features:
                features["BatteryFeature"]["soc"] += 0.01
        return agent_states

    env = SimpleEnv(
        coordinator_agents=[...],
        simulation_func=my_simulation,
    )
"""

from typing import Any, Callable, Dict, List, Optional

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.agents.constants import (
    FIELD_LEVEL,
    COORDINATOR_LEVEL,
    SYSTEM_LEVEL,
)
from heron.envs.base import HeronEnv

_LEVEL_TO_STATE_TYPE = {
    FIELD_LEVEL: "FieldAgentState",
    COORDINATOR_LEVEL: "CoordinatorAgentState",
    SYSTEM_LEVEL: "SystemAgentState",
}


class SimpleEnv(HeronEnv):
    """Multi-agent environment with automatic state bridge.

    Eliminates the ``env_state_to_global_state`` / ``global_state_to_env_state``
    boilerplate by auto-converting between HERON's internal global-state dict
    and a flat ``{agent_id: {feature_name: {field: val}}}`` dict that the
    user's simulation function can read and write directly.

    Parameters
    ----------
    coordinator_agents : list[CoordinatorAgent], optional
        Coordinator agents (BaseEnv auto-creates a SystemAgent).
    system_agent : SystemAgent, optional
        Explicit system agent (mutually exclusive with *coordinator_agents*).
    simulation_func : callable, optional
        ``(agent_states: dict) -> dict``.  Receives and returns a flat dict
        of ``{agent_id: {feature_name: {field: val}}}``.
    """

    def __init__(
        self,
        coordinator_agents: Optional[List[CoordinatorAgent]] = None,
        system_agent: Optional[SystemAgent] = None,
        simulation_func: Optional[Callable] = None,
        env_id: str = "simple_env",
        **kwargs: Any,
    ) -> None:
        self._user_simulation_func = simulation_func
        super().__init__(
            coordinator_agents=coordinator_agents,
            system_agent=system_agent,
            env_id=env_id,
            **kwargs,
        )

    # ------------------------------------------------------------------
    #  Abstract method implementations (auto-bridge)
    # ------------------------------------------------------------------

    def run_simulation(self, env_state: Any, *args: Any, **kwargs: Any) -> Any:
        if self._user_simulation_func is None:
            raise NotImplementedError("No simulation function provided to SimpleEnv.")
        return self._user_simulation_func(env_state)

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> Dict[str, Dict]:
        """Convert HERON global state to a flat ``{aid: {feature: {k: v}}}`` dict."""
        agent_states_raw = global_state.get("agent_states", {})
        flat: Dict[str, Dict] = {}
        for agent_id, state_dict in agent_states_raw.items():
            features = state_dict.get("features", state_dict)
            flat[agent_id] = {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in features.items()
                if not k.startswith("_")
            }
        return flat

    def env_state_to_global_state(self, env_state: Dict[str, Dict]) -> Dict[str, Any]:
        """Pack flat agent states back into HERON global state format."""
        agent_states: Dict[str, Any] = {}
        for agent_id, features_dict in env_state.items():
            agent = self.registered_agents.get(agent_id)
            if agent is None:
                continue
            level = getattr(agent, "level", FIELD_LEVEL)
            agent_states[agent_id] = {
                "_owner_id": agent_id,
                "_owner_level": level,
                "_state_type": _LEVEL_TO_STATE_TYPE.get(level, "FieldAgentState"),
                "features": features_dict,
            }
        return {"agent_states": agent_states}

"""Fluent factory for constructing HERON environments.

``EnvBuilder`` eliminates manual agent-hierarchy wiring by providing a
chainable API that auto-creates coordinators, resolves subordinate
assignments via glob patterns, and builds a ready-to-train environment.

Example::

    from heron.envs.builder import EnvBuilder

    env = (
        EnvBuilder("my_env")
        .add_agents("battery", BatteryAgent, count=3, features=[BatteryFeature()])
        .add_coordinator("zone", subordinates=["battery_*"])
        .simulation(my_sim_func)
        .build()
    )

With ``RLlibBasedHeronEnv``, pass agent specs as plain dicts — the adapter
builds the ``EnvBuilder`` internally::

    config = PPOConfig().environment(
        env=RLlibBasedHeronEnv,
        env_config={
            "agents": [{"agent_id": "b0", "agent_cls": BatteryAgent,
                        "features": [BatteryFeature()]}],
            "simulation": my_sim_func,
            "max_steps": 100,
        },
    )

If no coordinator is specified, a default one is auto-created wrapping all
field agents.
"""

import copy
import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.agents.system_agent import SystemAgent
from heron.core.feature import Feature
from heron.scheduling.tick_config import ScheduleConfig
from heron.envs.base import HeronEnv
from heron.protocols.base import Protocol
from heron.envs.simple import SimpleEnv


@dataclass
class _AgentSpec:
    agent_cls: Type[FieldAgent]
    agent_id: str
    features: List[Feature] = field(default_factory=list)
    coordinator_id: Optional[str] = None
    schedule_config: Optional[ScheduleConfig] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _CoordinatorSpec:
    agent_id: str
    agent_cls: Type[CoordinatorAgent] = CoordinatorAgent
    features: List[Feature] = field(default_factory=list)
    protocol: Optional[Protocol] = None
    subordinate_patterns: List[str] = field(default_factory=list)
    subordinate_ids: List[str] = field(default_factory=list)
    schedule_config: Optional[ScheduleConfig] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class _SystemSpec:
    features: List[Feature] = field(default_factory=list)
    schedule_config: Optional[ScheduleConfig] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class EnvBuilder:
    """Fluent factory for constructing HERON environments.

    Parameters
    ----------
    env_id : str
        Environment identifier (default ``"default_env"``).
    """

    def __init__(self, env_id: str = "default_env") -> None:
        self._env_id = env_id
        self._agent_specs: List[_AgentSpec] = []
        self._coordinator_specs: List[_CoordinatorSpec] = []
        self._system_spec: Optional[_SystemSpec] = None
        self._simulation_func: Optional[Callable] = None
        self._env_cls: Optional[Type[HeronEnv]] = None
        self._env_kwargs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    #  Agent registration
    # ------------------------------------------------------------------

    def add_agents(
        self,
        prefix: str,
        agent_cls: Type[FieldAgent],
        count: int = 1,
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        coordinator: Optional[str] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Register *count* field agents with auto-generated IDs.

        IDs follow the pattern ``{prefix}_0``, ``{prefix}_1``, ...
        When *count* is 1 the ID is just *prefix*.
        """
        for i in range(count):
            agent_id = f"{prefix}_{i}" if count > 1 else prefix
            self._agent_specs.append(_AgentSpec(
                agent_cls=agent_cls,
                agent_id=agent_id,
                features=list(features or []),
                coordinator_id=coordinator,
                schedule_config=schedule_config,
                kwargs=dict(kwargs),
            ))
        return self

    def add_agent(
        self,
        agent_id: str,
        agent_cls: Type[FieldAgent],
        features: Optional[List[Feature]] = None,
        coordinator: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Register a single named field agent."""
        self._agent_specs.append(_AgentSpec(
            agent_cls=agent_cls,
            agent_id=agent_id,
            features=list(features or []),
            coordinator_id=coordinator,
            schedule_config=schedule_config,
            kwargs=dict(kwargs),
        ))
        return self

    # ------------------------------------------------------------------
    #  Coordinator registration
    # ------------------------------------------------------------------

    def add_coordinator(
        self,
        coordinator_id: str,
        agent_cls: Type[CoordinatorAgent] = CoordinatorAgent,
        features: Optional[List[Feature]] = None,
        protocol: Optional[Protocol] = None,
        subordinates: Optional[List[str]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Register a coordinator agent.

        *subordinates* can contain exact agent IDs or glob patterns
        (e.g. ``"battery_*"``).
        """
        patterns: List[str] = []
        explicit: List[str] = []
        for s in (subordinates or []):
            if "*" in s or "?" in s:
                patterns.append(s)
            else:
                explicit.append(s)

        self._coordinator_specs.append(_CoordinatorSpec(
            agent_id=coordinator_id,
            agent_cls=agent_cls,
            features=list(features or []),
            protocol=protocol,
            subordinate_patterns=patterns,
            subordinate_ids=explicit,
            schedule_config=schedule_config,
            kwargs=dict(kwargs),
        ))
        return self
    
    def add_system_agent(
        self,
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Configure the SystemAgent (auto-created if not specified)."""
        if features or schedule_config or kwargs:
            self._system_spec = _SystemSpec(
                features=list(features or []),
                schedule_config=schedule_config,
                kwargs=dict(kwargs),
            )
        return self

    # ------------------------------------------------------------------
    #  Environment configuration
    # ------------------------------------------------------------------

    def simulation(self, func: Callable) -> "EnvBuilder":
        """Set the simulation function (``SimpleEnv`` auto-bridge)."""
        self._simulation_func = func
        return self

    def env_class(self, cls: Type[HeronEnv], **kwargs: Any) -> "EnvBuilder":
        """Use a specific ``HeronEnv`` subclass instead of ``SimpleEnv``."""
        self._env_cls = cls
        self._env_kwargs = kwargs
        return self

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------

    def build(self) -> HeronEnv:
        """Construct and return the configured environment."""
        agents = self._instantiate_agents()
        coordinators = self._resolve_coordinators(agents)
        if system := self._resolve_system_agent(coordinators):
            return self._build_env(system=system)
        return self._build_env(coordinators=coordinators)

    def __call__(self, config: Any = None) -> HeronEnv:
        """Build the environment (callable shorthand for ``build()``).

        Kept for backward compatibility with code that uses the builder
        as a callable factory.
        """
        return self.build()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _instantiate_agents(self) -> Dict[str, FieldAgent]:
        agents: Dict[str, FieldAgent] = {}
        for spec in self._agent_specs:
            ctor_kwargs = dict(agent_id=spec.agent_id, **spec.kwargs)
            if spec.features:
                ctor_kwargs["features"] = [copy.deepcopy(f) for f in spec.features]
            if spec.schedule_config is not None:
                ctor_kwargs["schedule_config"] = spec.schedule_config
            agents[spec.agent_id] = spec.agent_cls(**ctor_kwargs)
        return agents

    def _resolve_coordinators(
        self, agents: Dict[str, FieldAgent],
    ) -> List[CoordinatorAgent]:
        coordinators: List[CoordinatorAgent] = []
        assigned: set = set()

        for cspec in self._coordinator_specs:
            sub_ids: List[str] = list(cspec.subordinate_ids)

            # Resolve glob patterns
            for pattern in cspec.subordinate_patterns:
                for aid in agents:
                    if fnmatch.fnmatch(aid, pattern) and aid not in sub_ids:
                        sub_ids.append(aid)

            # Include agents that declared this coordinator
            for aspec in self._agent_specs:
                if aspec.coordinator_id == cspec.agent_id and aspec.agent_id not in sub_ids:
                    sub_ids.append(aspec.agent_id)

            subordinates = {aid: agents[aid] for aid in sub_ids if aid in agents}
            assigned.update(sub_ids)

            features = [copy.deepcopy(f) for f in cspec.features]
            ctor_kwargs = dict(
                agent_id=cspec.agent_id,
                features=features,
                subordinates=subordinates,
                **cspec.kwargs,
            )
            if cspec.schedule_config is not None:
                ctor_kwargs["schedule_config"] = cspec.schedule_config
            if cspec.protocol is not None:
                ctor_kwargs["protocol"] = cspec.protocol
            coordinators.append(cspec.agent_cls(**ctor_kwargs))

        # Auto-create coordinator for unassigned agents
        unassigned = [aid for aid in agents if aid not in assigned]
        if unassigned:
            coordinators.append(CoordinatorAgent(
                agent_id="auto_coordinator",
                subordinates={aid: agents[aid] for aid in unassigned},
            ))

        return coordinators
    
    def _resolve_system_agent(
        self, coordinators: List[CoordinatorAgent],
    ) -> SystemAgent | None:
        if not self._system_spec:
            return None
        features = list(self._system_spec.features)
        schedule_config = self._system_spec.schedule_config
        kwargs = dict(self._system_spec.kwargs)

        system_agent = SystemAgent(
            features=features,
            schedule_config=schedule_config,
            subordinates={c.agent_id: c for c in coordinators},
            **kwargs,
        )
        return system_agent

    def _build_env(
        self, 
        system: Optional[SystemAgent] = None, 
        coordinators: Optional[List[CoordinatorAgent]] = None
    ) -> HeronEnv:
        if system and coordinators:
            raise ValueError("Cannot build env with both SystemAgent and coordinators (not supported yet).")
        
        if self._env_cls is not None:
            return self._env_cls(
                env_id=self._env_id,
                system_agent=system, 
                coordinator_agents=coordinators,
                **self._env_kwargs,
            )
    
        if self._simulation_func is not None:
            return SimpleEnv(
                env_id=self._env_id,
                system_agent=system,
                coordinator_agents=coordinators,
                simulation_func=self._simulation_func,
            )

        raise ValueError(
            "EnvBuilder requires either a simulation function (.simulation()) "
            "or a custom env class (.env_class()) to build an environment."
        )


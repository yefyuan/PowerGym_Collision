from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import uuid

import gymnasium as gym

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.policies import Policy
from heron.messaging import MessageBroker, ChannelManager, Message, MessageType
from heron.utils.typing import AgentID, MultiAgentDict
from heron.scheduling import EventScheduler, Event, EpisodeAnalyzer, EpisodeStats
from heron.scheduling.tick_config import ScheduleConfig
from heron.agents.system_agent import SystemAgent
from heron.agents.proxy_agent import Proxy
from heron.agents.constants import SYSTEM_AGENT_ID, PROXY_AGENT_ID


class BaseEnv:
    def __init__(
        self,
        env_id: Optional[str] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        message_broker_config: Optional[Dict[str, Any]] = None,
        # agents
        system_agent: Optional[SystemAgent] = None,
        coordinator_agents: Optional[List[CoordinatorAgent]] = None,
        # simulation-related params
        simulation_wait_interval: Optional[float] = None,
        # timing
        system_agent_schedule_config: Optional[ScheduleConfig] = None,
    ) -> None:
        # environment attributes
        self.env_id = env_id or f"env_{uuid.uuid4().hex[:8]}"
        self.simulation_wait_interval = simulation_wait_interval

        # agent-specific fields
        self.registered_agents: Dict[AgentID, Agent] = {}
        self._register_agents(system_agent, coordinator_agents, system_agent_schedule_config)

        # initialize proxy agent (singleton) for state access and action dispatch
        self.proxy = Proxy(agent_id=PROXY_AGENT_ID)
        self._register_agent(self.proxy)

        # setup message broker (before proxy attach - proxy needs it for channels)
        self.message_broker = MessageBroker.init(message_broker_config)
        self.message_broker.attach(self.registered_agents)

        # attach message broker to proxy agent for communication
        self.proxy.set_message_broker(self.message_broker)
        # establish direction link between registered agents and proxy for state access
        self.proxy.attach(self.registered_agents)

        # setup scheduler (before initialization - agents need it)
        self.scheduler = EventScheduler.init(scheduler_config)
        self.scheduler.attach(self.registered_agents)

    # ============================================
    # Agent Management Methods
    # ============================================
    def _register_agents(
        self,
        system_agent: Optional[SystemAgent],
        coordinator_agents: Optional[List[CoordinatorAgent]],
        system_agent_schedule_config: Optional[ScheduleConfig] = None,
    ) -> None:
        """Internal method to register agents during initialization."""
        # register system agent (singleton) & its subordinates
        if system_agent and coordinator_agents:
            raise ValueError("Cannot provide both SystemAgent and List[CoordinatorAgent]. Provide one or the other.")
        self._system_agent = None
        if system_agent:
            self._system_agent = system_agent
        else:
            print("No system agent provided, using default system agent")
            sys_kwargs = {
                "agent_id": SYSTEM_AGENT_ID,
                "subordinates": {agent.agent_id: agent for agent in coordinator_agents},
            }
            if system_agent_schedule_config is not None:
                sys_kwargs["schedule_config"] = system_agent_schedule_config
            self._system_agent = SystemAgent(**sys_kwargs)
        self._system_agent.set_simulation(
            self.run_simulation,
            self.env_state_to_global_state,
            self.global_state_to_env_state,
            self.simulation_wait_interval,
            self.pre_step,
        )
        self._register_agent(self._system_agent)
        

    def get_agent(self, agent_id: AgentID) -> Optional[Agent]:
        """Get a registered agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent instance or None if not found
        """
        return self.registered_agents.get(agent_id)

    def _register_agent(self, agent: Agent) -> None:
        """Register an agent with the environment.

        Args:
            agent: Agent to register
        """
        agent.env_id = self.env_id
        self.registered_agents[agent.agent_id] = agent
        for subordinate in agent.subordinates.values():
            self._register_agent(subordinate)

    # ===========================================
    # Environment Interaction Methods
    # ==========================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        jitter_seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset all registered agents.

        Args:
            seed: Random seed for agent/env reset.
            jitter_seed: If provided, enables jitter on all agents with
                this seed (for reproducible event-driven evaluation).
                Tick configs should be set at agent construction time
                (e.g. via ``schedule_config`` in env_config).
            **kwargs: Additional reset parameters
        """
        # Re-seed jitter RNG per episode for reproducible event-driven eval
        if jitter_seed is not None:
            for agent in self.registered_agents.values():
                agent.enable_jitter(seed=jitter_seed)

        # Sync tick configs in case agents were reconfigured after construction
        self.scheduler.sync_schedule_configs(self.registered_agents)
        # reset scheduler and clear messages before resetting agents to ensure a clean slate
        self.scheduler.reset(start_time=0.0)  # Always reset to time 0
        self.clear_broker_environment()

        # reset agents (system agent will reset subordinates)
        self.proxy.reset(seed=seed)
        obs = self._system_agent.reset(seed=seed, proxy=self.proxy)
        self.proxy.init_global_state()  # Cache initial state in proxy after reset
        return obs
    
    def step(self, actions: Dict[AgentID, Any]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one environment step.

        The system_agent in the environment is responsible for entire 
        simulation step

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
            - observations: Dict mapping agent IDs to observation arrays
            - rewards: Dict mapping agent IDs to reward floats
            - terminated: Dict with agent IDs and "__all__" key
            - truncated: Dict with agent IDs and "__all__" key
            - infos: Dict mapping agent IDs to info dicts
        """
        self._system_agent.execute(actions, self.proxy)
        return self.proxy.get_step_results()
    
    def run_event_driven(
        self,
        episode_analyzer: EpisodeAnalyzer,
        t_end: float,
        max_events: Optional[int] = None,
    ) -> EpisodeStats:
        """Run event-driven simulation until time limit.

        Args:
            episode_analyzer: EpisodeAnalyzer to parse events during simulation
            t_end: Stop when simulation time exceeds this
            max_events: Optional maximum number of events to process

        Returns:
            EpisodeStats containing all event analyses from the simulation
        """
        result = EpisodeStats()
        for event in self.scheduler.run_until(t_end=t_end, max_events=max_events):
            result.add_event_analysis(episode_analyzer.parse_event(event))
        return result

    # ============================================
    # Simulation-related Methods
    # ============================================
    def pre_step(self) -> None:
        """Hook called at the start of each step before agent actions.

        Override this method in subclasses to perform environment-specific
        setup at the beginning of each step (e.g., updating profiles, loading
        time-series data for current timestep).

        Default implementation is a no-op.
        """
        pass

    @abstractmethod
    def run_simulation(self, env_state: Any, *args, **kwargs) -> Any:
        """ Custom simulation logic post-system_agent.act and before system_agent.update_from_environment().

        In the long run, this can be eventually turned into a static SimulatorAgent.
        """
        pass

    @abstractmethod
    def env_state_to_global_state(self, env_state: Any) -> Dict[str, Any]:
        """Convert custom environment state to HERON global state dict format.

        This method is called after run_simulation() to convert the updated
        environment state back into the global state dict structure that will
        be stored in proxy.state_cache["global"].

        Args:
            env_state: Custom environment state after simulation

        Returns:
            Dict that will be merged into proxy.state_cache["global"] via .update()
            Typically includes "agent_states" dict with updated agent state dicts.

        Example:
            return {
                "agent_states": {
                    "agent_1": {"FeatureName": {"field": value}},
                    ...
                }
            }
        """
        pass

    @abstractmethod
    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> Any:
        """Convert HERON global state dict to custom environment state format.

        This method is called before run_simulation() to extract relevant info
        from proxy.state_cache["global"] and convert it to the custom format
        your simulation function expects.

        Args:
            global_state: Dict from proxy.state_cache["global"] with structure:
                {
                    "agent_states": {agent_id: state_dict, ...},
                    ... other global fields ...
                }
                Note: state_dict is the dict representation from State.to_dict()

        Returns:
            Custom environment state object for your simulation

        Example:
            agent_states = global_state.get("agent_states", {})
            battery_soc = agent_states["battery_1"]["BatteryFeature"]["soc"]
            return CustomEnvState(battery_soc=battery_soc)
        """
        pass


    # ============================================
    # Utility Methods
    # ============================================
    def get_all_policies(self) -> Dict[AgentID, Policy]:
        return {
            agent_id: agent.policy
            for agent_id, agent in self.registered_agents.items()
            if agent.policy
        }
    
    def set_agent_policies(self, policies: Dict[AgentID, Policy]) -> None:
        for agent_id, policy in policies.items():
            agent = self.registered_agents.get(agent_id)
            if agent:
                agent.policy = policy

    def clear_broker_environment(self) -> None:
        """Clear all messages for this environment from the broker.

        Useful for resetting the environment.
        """
        if self.message_broker is not None:
            self.message_broker.clear_environment(self.env_id)

    def close_core(self) -> None:
        """Clean up core resources. [Both Modes]"""
        if self.message_broker is not None:
            self.message_broker.close()
    

class HeronEnv(BaseEnv):
    def close(self) -> None:
        """Clean up environment resources."""
        self.close_core()

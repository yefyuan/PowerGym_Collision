

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.core.feature import Feature
from heron.protocols.base import Protocol
from heron.messaging import ChannelManager, Message, MessageBroker, MessageType
from heron.utils.typing import AgentID
from heron.scheduling.scheduler import Event, EventScheduler
from heron.agents.constants import (
    PROXY_LEVEL,
    PROXY_AGENT_ID,
    DEFAULT_HISTORY_LENGTH,
    MSG_GET_INFO,
    MSG_SET_STATE,
    MSG_SET_TICK_RESULT,
    MSG_SET_STATE_COMPLETION,
    INFO_TYPE_OBS,
    INFO_TYPE_GLOBAL_STATE,
    INFO_TYPE_LOCAL_STATE,
    STATE_TYPE_GLOBAL,
    STATE_TYPE_LOCAL,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
)


from heron.core.state import State

class Proxy(Agent):

    def __init__(
        self,
        agent_id: AgentID = PROXY_AGENT_ID,
        env_id: Optional[str] = None,
        registered_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        message_broker: Optional[MessageBroker] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            level=PROXY_LEVEL,
            upstream_id=None,
            subordinates={},
            env_id=env_id,
        )

        self.history_length = history_length
        self.registered_agents: List[AgentID] = registered_agents or []
        self.visibility_rules: Dict[AgentID, List[str]] = visibility_rules or {}
        self.state_cache: Dict[str, Any] = {}
        self._agent_levels: Dict[AgentID, int] = {}  # Track agent hierarchy levels for visibility
        self._agent_upstreams: Dict[AgentID, Optional[AgentID]] = {}  # Track upstream (parent) for each agent
        self._tick_results: Dict[AgentID, Dict[str, Any]] = {}  # Store tick results per agent

        if message_broker:
            self.set_message_broker(message_broker)

    # ============================================
    # Initialization and State/Action Management Overrides
    # Note that Proxy does not maintain its own state or action in the traditional sense, so these methods are either no-ops or can be used to set up any necessary internal structures for
    # managing the proxy's functionality (e.g. state cache, visibility rules, etc.)
    # ============================================
    def init_state(self, features: List[Feature] = []) -> None:
        pass

    def init_action(self, features: List[Feature] = []) -> None:
        pass

    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    def attach(self, agents:  Dict[AgentID, Agent]) -> None:
        """Attach proxy to the environment by registering all agents.

        This should be called during environment initialization after all agents are created but before any agent initialization or reset.

        Args:
            agents: Dict of agent_id to Agent objects to register with the proxy
        """
        for agent_id, agent in agents.items():
            if agent_id == self.agent_id:
                continue  # Skip registering the proxy itself
            self._register_agent(agent)

        # Initialize global state after all agents are registered
        self.init_global_state()

        # Individual agent post-proxy-attach handling with initialized global state
        for agent_id, agent in agents.items():
            if agent_id == self.agent_id:
                continue  # Skip the proxy itself
            agent.post_proxy_attach(proxy=self)

        # Setup communication channels after all agents are registered
        self._setup_channels()

    def _setup_channels(self) -> None:
        if self._message_broker is None:
            raise ValueError("Message broker is required to setup channels in Proxy")

        # Create proxy->agent channels for distributing state
        for agent_id in self.registered_agents:
            agent_channel = ChannelManager.info_channel(
                self.agent_id, agent_id, self.env_id
            )
            self._message_broker.create_channel(agent_channel)


    def _register_agent(self, agent: Agent) -> None:
        """Register a new agent that can request state.

        Args:
            agent: Agent to register
        """
        agent_id = agent.agent_id
        agent_state = agent.state
        if agent_id == self.agent_id:
            print("Proxy agent doesn't register itself.")
            return

        if agent_id not in self.registered_agents:
            self.registered_agents.append(agent_id)

        # Track upstream (parent) for hierarchy-aware operations
        self._agent_upstreams[agent_id] = agent.upstream_id

        # Always update state if provided (supports both init and reset)
        if agent_state is not None:
            # Track agent level for visibility checks
            self._agent_levels[agent_id] = agent_state.owner_level
            # Store State object directly (no serialization!)
            self.set_local_state(agent_id, agent_state)

    def init_global_state(self) -> None:
        """Initialize global state by compiling all registered agent states.

        Should be called after all agents are registered and their initial states are set.
        """
        if "agents" not in self.state_cache or not self.state_cache["agents"]:
            print("Warning: No agent states to compile into global state")
            return

        # Compile all agent states into global state
        if "global" not in self.state_cache:
            self.state_cache["global"] = {}

        # Aggregate relevant global information from all agents
        # (Override this method in subclasses for custom aggregation logic)
        for agent_id, agent_state in self.state_cache["agents"].items():
            # Store agent states as part of global state for now
            # Subclasses can implement more sophisticated aggregation
            if "agent_states" not in self.state_cache["global"]:
                self.state_cache["global"]["agent_states"] = {}
            self.state_cache["global"]["agent_states"][agent_id] = agent_state


    # ============================================
    # Core Agent Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # Note that Proxy does not follow the standard observe-decide-act loop, so execute and tick are empty logic.
    # ============================================
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset proxy agent state.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        # Proxy doesn't need parent reset - it manages its own state cache
        self.state_cache = {}

    def execute(self, actions: Dict[AgentID, Any], proxy: Optional["Proxy"] = None) -> None:
        pass

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        pass

    # ============================================
    # Custom Handlers for Event-Driven Execution (see heron/scheduling/scheduler.py for more details on event handling)
    # ============================================
    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        recipient_id = event.agent_id
        if recipient_id != self.agent_id:
            raise ValueError(f"Event {event} sent to {event.agent_id} is handled in proxy_agent!")
        sender_id = event.payload.get("sender")
        message_content = event.payload.get("message", {})

        assert self._message_broker
        if MSG_GET_INFO in message_content:
            info_type = message_content[MSG_GET_INFO]
            protocol = message_content.get(MSG_KEY_PROTOCOL)
            info = self._handle_get_info_request(sender_id, info_type, protocol)

            # Serialize Observation objects before sending via message
            # In async/distributed systems, objects must be serialized for message passing
            info_type_key = "get_" + info_type + "_response" # e.g. obs -> get_obs_response

            # Special handling for observation requests: send both obs and local_state
            if isinstance(info, Observation):
                # When requesting observation, proxy gives BOTH obs & state (design principle)
                # Agent asks for obs, but proxy provides both for state syncing
                local_state = self.get_local_state(sender_id, protocol)
                info_data = {
                    "obs": info.to_dict(),
                    "local_state": local_state
                }
            else:
                # For other info types (global_state, local_state), send as-is
                info_data = info

            scheduler.schedule_message_delivery(
                sender_id=recipient_id, # same as self.agent_id
                recipient_id=sender_id,
                message={
                    info_type_key: {
                        MSG_KEY_BODY: info_data
                    }
                },
                delay=self._schedule_config.msg_delay,
            )
        elif MSG_SET_STATE in message_content:
            from heron.core.state import State

            state_type = message_content[MSG_SET_STATE]
            if state_type == STATE_TYPE_GLOBAL:
                # Clear previous tick results — new simulation cycle starts fresh
                self._tick_results = {}

                # Global state is dict of {agent_id: state_dict}
                global_state_payload = message_content.get(MSG_KEY_BODY, {})
                agent_states = global_state_payload.get("agent_states", {})
                for agent_id, state_dict in agent_states.items():
                    # Deserialize dict → State object
                    state_obj = State.from_dict(state_dict)
                    self.set_local_state(agent_id, state_obj)
            elif state_type == STATE_TYPE_LOCAL:
                state_dict = message_content.get(MSG_KEY_BODY, {})
                # Deserialize dict → State object
                state_obj = State.from_dict(state_dict)
                self.set_local_state(state_obj.owner_id, state_obj)
            else:
                raise NotImplementedError(f"Unknown state type {state_type} in {MSG_SET_STATE} message")
            scheduler.schedule_message_delivery(
                sender_id=recipient_id, # same as self.agent_id
                recipient_id=sender_id,
                message={MSG_SET_STATE_COMPLETION: "success"},
                delay=self._schedule_config.msg_delay,
            )
        elif MSG_SET_TICK_RESULT in message_content:
            result_type = message_content[MSG_SET_TICK_RESULT]
            tick_result = message_content.get(MSG_KEY_BODY, {})
            # Store tick result per-agent for retrieval by parent agents
            self._tick_results[sender_id] = tick_result
        else:
            raise NotImplementedError(f"Unknown message content {message_content} in message_delivery to proxy_agent")

    def _handle_get_info_request(self, sender_id: AgentID, request_type: str, protocol: Optional[Protocol] = None):
        if request_type == INFO_TYPE_OBS:
            return self.get_observation(sender_id, protocol)
        elif request_type == INFO_TYPE_GLOBAL_STATE:
            return self.get_global_states(sender_id, protocol, for_simulation=True)
        elif request_type == INFO_TYPE_LOCAL_STATE:
            return self.get_local_state(sender_id, protocol)
        else:
            raise NotImplementedError("not yet complete")
    
    # ============================================
    # Core Logic Methods for Proxy Functionality
    # These methods define the core logic of how the proxy agent computes observations, manages state, and handles requests from other agents. They are designed to be overridden in subclasses to implement specific proxy behaviors (e.g. different state aggregation methods, visibility rules, reward computation logic, etc.)
    # ============================================
    def get_observation(self, sender_id: AgentID, protocol: Optional[Protocol] = None) -> Observation:
        """
        Compute observation for particular sender_id.
        A protocol may be given to specify the format of the observation, but the content is determined by the proxy agent's logic.

        if sender_id is SYSTEM_AGENT_ID -> Full global obs + agent-specific local obs (id-based)
        otherwise -> partial global obs + agent-specific local obs (id-based)

        Returns:
            Observation object that can be automatically converted to np.ndarray via __array__()
        """
        from heron.agents.system_agent import SYSTEM_AGENT_ID

        # Get global and local components
        # Note: exclude subordinate_rewards from observation (only used for reward computation)
        global_state = self.get_global_states(sender_id, protocol)
        local_state = self.get_local_state(sender_id, protocol, include_subordinate_rewards=False)

        # Return Observation object
        return Observation(
            local=local_state,
            global_info=global_state,
            timestamp=self._timestep
        )

    def get_global_states(
        self,
        sender_id: AgentID,
        protocol: Optional[Protocol] = None,
        for_simulation: bool = False,
    ) -> Dict:
        """Get global state information from all agents.

        Args:
            sender_id: ID of agent requesting global state
            protocol: Optional protocol for formatting
            for_simulation: If True, returns all agents' full state dicts
                (via ``to_dict(include_metadata=True)``) for the simulation
                pipeline.  If False (default), returns visibility-filtered
                numpy arrays for the observation pipeline.

        Returns:
            Dict containing global state information.
            Also includes "env_context" if available.
        """
        global_filtered = {}

        if for_simulation:
            for agent_id, state_obj in self.state_cache.get("agents", {}).items():
                global_filtered[agent_id] = state_obj.to_dict(include_metadata=True)
        else:
            requestor_level = self._agent_levels.get(sender_id, 1)
            for agent_id, state_obj in self.state_cache.get("agents", {}).items():
                if agent_id == sender_id:
                    continue  # Don't include own state in global (it's in local)
                observable = state_obj.observed_by(sender_id, requestor_level)
                if observable:
                    global_filtered[agent_id] = observable

        # Include env_context if available (external data like price, solar, wind profiles)
        global_cache = self.state_cache.get("global", {})
        if "env_context" in global_cache:
            global_filtered["env_context"] = global_cache["env_context"]

        return global_filtered

    def get_local_state(
        self,
        sender_id: AgentID,
        protocol: Optional[Protocol] = None,
        include_subordinate_rewards: bool = True
    ) -> Dict:
        """Get local state information for the requesting agent with visibility filtering.

        NEW: Applies feature-level visibility filtering via state.observed_by()
        Also includes subordinate rewards for parent agents (coordinators/system agents).

        Args:
            sender_id: ID of agent requesting local state
            protocol: Optional protocol for formatting
            include_subordinate_rewards: Whether to include subordinate rewards (default True).
                Set to False when building observations for action computation.

        Returns:
            Dict containing agent-specific local state (filtered by visibility rules)
            For parent agents, includes 'subordinate_rewards' with rewards from children
        """
        agents_cache = self.state_cache.get("agents", {})
        state_obj = agents_cache.get(sender_id)

        if state_obj is None:
            return {}

        # Apply visibility filtering using observed_by()
        requestor_level = self._agent_levels.get(sender_id, 1)
        local_state = state_obj.observed_by(sender_id, requestor_level)

        # Include subordinate rewards for parent agents (only for reward computation, not observations)
        if include_subordinate_rewards:
            subordinate_rewards = self._get_subordinate_rewards(sender_id)
            if subordinate_rewards:
                local_state["subordinate_rewards"] = subordinate_rewards

        return local_state

    def _get_subordinate_rewards(self, parent_id: AgentID) -> Dict[AgentID, float]:
        """Get rewards from agents whose upstream is the given parent.

        Args:
            parent_id: ID of the parent agent

        Returns:
            Dict mapping subordinate agent IDs to their rewards
        """
        subordinate_rewards = {}
        for agent_id, upstream_id in self._agent_upstreams.items():
            if upstream_id == parent_id and agent_id in self._tick_results:
                tick_result = self._tick_results[agent_id]
                if "reward" in tick_result:
                    subordinate_rewards[agent_id] = tick_result["reward"]
        return subordinate_rewards

    def set_global_state(self, global_state: Dict) -> None:
        """Update the global state in cache.

        Also propagates agent_states back to state_cache["agents"] so that
        subsequent observe() / compute_rewards() calls see simulation results.

        Args:
            global_state: Global state dictionary to cache
        """
        from heron.core.state import State

        if "global" not in self.state_cache:
            self.state_cache["global"] = {}
        self.state_cache["global"].update(global_state)

        # Propagate simulation results back to per-agent state cache
        agent_states = global_state.get("agent_states", {})
        for agent_id, state_dict in agent_states.items():
            if isinstance(state_dict, dict):
                state_obj = State.from_dict(state_dict)
                self.set_local_state(agent_id, state_obj)

    def set_local_state(self, agent_id: str, state: "State") -> None:
        """Update local state for agents in cache.

        NEW: Stores State objects directly for visibility filtering.

        Args:
            agent_id: ID of the agent owning this state
            state: State object (FieldAgentState, CoordinatorAgentState, etc.)
        """
        if "agents" not in self.state_cache:
            self.state_cache["agents"] = {}

        # Store State object directly!
        self.state_cache["agents"][agent_id] = state

    def get_serialized_agent_states(self) -> Dict[AgentID, Dict[str, Any]]:
        """Get serialized (dict) versions of all agent states.

        Used for message passing and global state construction where
        State objects need to be converted to serializable dicts.

        Returns:
            Dict mapping agent_id to serialized state dict with metadata
        """
        agents_cache = self.state_cache.get("agents", {})
        serialized = {}
        for agent_id, state_obj in agents_cache.items():
            if isinstance(state_obj, State):
                serialized[agent_id] = state_obj.to_dict(include_metadata=True)
            else:
                # Fallback if state is already a dict
                serialized[agent_id] = state_obj if isinstance(state_obj, dict) else {}
        return serialized

    def set_step_result(self, obs: Dict[AgentID, Observation], rewards, terminateds, truncateds, infos):
        """Cache the step results from environment execution.

        Args:
            obs: Observations dict mapping agent IDs to Observation objects
            rewards: Rewards dict mapping agent IDs to reward values
            terminateds: Terminated flags dict
            truncateds: Truncated flags dict
            infos: Info dicts mapping agent IDs to info dicts
        """
        self._step_results = {
            "obs": obs,
            "rewards": rewards,
            "terminateds": terminateds,
            "truncateds": truncateds,
            "infos": infos,
        }

    def get_step_results(self) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, Dict]]:
        """Retrieve cached step results.

        Automatically converts Observation objects to np.ndarray for RL algorithms.

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
            - observations: Dict[AgentID, np.ndarray] - vectorized observations for RL
        """
        if not hasattr(self, "_step_results") or self._step_results is None:
            raise RuntimeError("No step results available. Call set_step_result() first.")

        results = self._step_results
        obs: Dict[AgentID, Observation] = results["obs"]

        # Convert all Observation objects to np.ndarray for RL algorithms
        obs_vectorized = {
            agent_id: observation.vector()
            for agent_id, observation in obs.items()
        }

        return (
            obs_vectorized,
            results["rewards"],
            results["terminateds"],
            results["truncateds"],
            results["infos"],
        )
    
    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        num_registered = len(self.registered_agents)
        has_broker = self._message_broker is not None
        return f"Proxy(id={self.agent_id}, registered_agents={num_registered}, broker={has_broker})"

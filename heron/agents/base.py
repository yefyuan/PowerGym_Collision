

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from heron.agents.proxy_agent import ProxyAgent

from heron.core.feature import FeatureProvider
from heron.messaging import MessageBroker, ChannelManager, Message as BrokerMessage, MessageType
from heron.utils.typing import AgentID
from heron.scheduling.tick_config import TickConfig, JitterType
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.event import Event, EventType, EVENT_TYPE_FROM_STRING
from heron.core.policies import Policy
from heron.core.state import State
from heron.core.action import Action
from heron.protocols.base import Protocol
from heron.agents.constants import PROXY_AGENT_ID, FIELD_LEVEL, EMPTY_REWARD


class Agent(ABC):
    # class-level handler function mapping
    _event_handler_funcs: Dict[EventType, Callable[[Event, "EventScheduler"], None]] = {}

    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own copy of the handlers dict and register handlers."""
        super().__init_subclass__(**kwargs)
        # Create a new dict for this subclass, inheriting parent handlers
        cls._event_handler_funcs = cls._event_handler_funcs.copy()

        # Register any handlers marked with _handler_event_type
        for name in dir(cls):
            # Skip special attributes (dunder methods)
            if name.startswith('__'):
                continue
            try:
                attr = getattr(cls, name)
                if callable(attr) and hasattr(attr, '_handler_event_type'):
                    cls._event_handler_funcs[attr._handler_event_type] = attr
            except AttributeError:
                # Skip attributes that can't be accessed
                pass

    def __init__(
        self,
        agent_id: AgentID,
        level: int = 1,
        features: Optional[List[FeatureProvider]] = None,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = policy
        self.protocol = protocol
        features = features or []
        self.action = self.init_action(features=features)
        self.state = self.init_state(features=features)

        # Execution state
        self._timestep: float = 0.0

        # Message broker reference (set by environment in distributed mode)
        self._message_broker: Optional[MessageBroker] = None

        # Timing configuration (via TickConfig)
        self._tick_config = tick_config or TickConfig.deterministic()

        # Hierarchy structure (used by coordinators)
        self.env_id = env_id
        self.upstream_id = upstream_id
        self.subordinates = self._build_subordinates(subordinates)

    def _build_subordinates(self, subordinates: Optional[Dict[AgentID, "Agent"]] = None,) -> Dict[AgentID, "Agent"]:
        if not subordinates:
            return {}
        for _, agent in subordinates.items():
            agent.upstream_id = self.agent_id
            agent.env_id = self.env_id
        # Register subordinates with protocol for action decomposition
        if self.protocol:
            self.protocol.register_subordinates(subordinates)
        return subordinates

    @abstractmethod
    def init_state(self, features: List[FeatureProvider] = []) -> State:
        """Initialize the agent's State object from the given features.

        Args:
            features: Feature providers to include in the state
        """
        pass

    @abstractmethod
    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """Initialize the agent's Action object.

        Args:
            features: Feature providers (unused by default, available for custom logic)
        """
        pass

    @abstractmethod
    def set_state(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    def post_proxy_attach(self, proxy: "ProxyAgent") -> None:
        """Hook for any additional setup after proxy attachment and global state initialization."""
        pass

    # ============================================
    # Core Lifecycle Methods (Both Modes)
    # - reset: resetting fields and states, potentially returning current states
    # - execute: Synchronous Execution (for Training phase)
    # - tick: Event-Driven Execution (for Testing phase)
    # ============================================
    def reset(self, *, seed: Optional[int] = None, proxy: Optional["ProxyAgent"] = None, **kwargs) -> Any:
        self._timestep = 0.0
        self.action.reset(**kwargs)  # Reset action to initial values, with optional overrides
        self.state.reset(**kwargs)  # Reset state to initial values, with optional overrides

        # Cache initial state in proxy
        if not proxy:
            raise ValueError("Agent requires a proxy agent to reset states")
        if self.state is None:
            raise ValueError("Agent state is not initialized, cannot reset. Please call initialize() first.")
        proxy.set_local_state(self.agent_id, self.state)  # Pass State object directly

        for subordinate in self.subordinates.values():
            subordinate.reset(seed=seed, proxy=proxy, **kwargs)

    def execute(self, actions: Dict[AgentID, Any], proxy: Optional["ProxyAgent"] = None) -> None:
        """Execute actions in CTDE mode. [Training Mode]

        Default implementation:
        1. Sync state from proxy (state syncing as by-product of global state updates)
        2. Act with given actions

        Subclasses can override for custom behavior but should maintain state syncing.
        """
        if not proxy:
            raise ValueError("Agent requires a proxy agent to execute")

        # Sync state first (by-product of proxy having updated global state)
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        self.sync_state_from_observed(local_state)

        self.act(actions, proxy)

    @abstractmethod
    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        self._timestep = current_time
        self._check_for_upstream_action()  # Check for any new upstream action at the start of the tick


    # ============================================
    # Observation related functions
    # ============================================
    def observe(self, global_state: Optional[Dict[str, Any]] = None, proxy: Optional["ProxyAgent"] = None, *args, **kwargs) -> Dict[AgentID, Any]:
        """Collect observations for this agent and all subordinates.

        Returns a flat dict mapping each agent ID to its Observation vector,
        recursively including all subordinates.

        Args:
            global_state: Optional global state dict (unused in default impl)
            proxy: ProxyAgent used to retrieve observations

        Returns:
            Dict mapping agent IDs to observation arrays
        """
        if not proxy:
            raise ValueError("Agent requires a proxy agent to observe states")
        obs = {
            self.agent_id: proxy.get_observation(self.agent_id, self.protocol), # local observation
        }
        for subordinate in self.subordinates.values():
            obs.update(subordinate.observe(proxy=proxy))
        return obs

    # ============================================
    # Reward related functions
    # ============================================
    def compute_rewards(self, proxy: "ProxyAgent") -> Dict[AgentID, float]:
        """Compute rewards for this agent and all subordinates.

        Retrieves local state from proxy and delegates to compute_local_reward().

        Args:
            proxy: ProxyAgent for state retrieval

        Returns:
            Dict mapping agent IDs to scalar rewards
        """
        if not proxy:
            raise ValueError("Agent requires a proxy agent to compute rewards")

        # Local Reward computation steps:
        # 1. get local states from proxy
        # 2. collect reward params (e.g. safety, cost)
        # 3. calculate reward
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        local_reward = self.compute_local_reward(local_state)
        rewards = {
            self.agent_id: local_reward,
        }
        for subordinate in self.subordinates.values():
            rewards.update(subordinate.compute_rewards(proxy))
        return rewards
    
    def compute_local_reward(self, local_state: dict) -> float:
        # Default implementation returns empty reward. Override in subclasses for custom reward logic.
        return EMPTY_REWARD

    # ============================================
    # Info related functions
    # ============================================
    def get_info(self, proxy: "ProxyAgent") -> Dict[AgentID, Dict]:
        """Collect info dicts for this agent and all subordinates.

        Retrieves local state from proxy and delegates to get_local_info().

        Args:
            proxy: ProxyAgent for state retrieval

        Returns:
            Dict mapping agent IDs to info dicts
        """
        if not proxy:
            raise ValueError("Agent requires a proxy agent to get infos")
        # Local info derivation
        # May use proxy to retrieve local states via proxy.get_local_state(self.agent_id)
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        local_info = self.get_local_info(local_state)
        infos = {
            self.agent_id: local_info,
        }
        for subordinate in self.subordinates.values():
            infos.update(subordinate.get_info(proxy))
        return infos
    
    def get_action_mask(self) -> Optional[np.ndarray]:
        """Return action mask for this agent, or ``None`` if no masking.

        Override in subclasses to constrain valid actions based on current state.

        For ``Discrete(n)``: return boolean array of shape ``(n,)``
            (``True`` = action allowed).
        For ``MultiDiscrete([n1, n2, ...])``: return concatenated boolean arrays.
        For ``Box`` (continuous): return ``None`` (continuous actions don't use masks).

        Returns:
            ``np.ndarray`` mask or ``None``
        """
        return None

    def get_local_info(self, local_state: dict) -> Dict[AgentID, Any]:
        info: Dict[str, Any] = {}
        mask = self.get_action_mask()
        if mask is not None:
            info["action_mask"] = mask
        return info
    
    # ============================================
    # terminateds related functions
    # ============================================
    def get_terminateds(self, proxy: "ProxyAgent") -> Dict[AgentID, bool]:
        """Collect termination flags for this agent and all subordinates.

        Args:
            proxy: ProxyAgent for state retrieval

        Returns:
            Dict mapping agent IDs to termination booleans
        """
        if not proxy:
            raise ValueError("Agent requires a proxy agent to derive termination state")
        # May need to use env fields to decide
        # TODO: pass in the env field elegantly
        # e.g. done = (self._t % self.max_episode_steps) == 0
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        terminated = self.is_terminated(local_state)
        terminateds = {
            self.agent_id: terminated,
        }
        for subordinate in self.subordinates.values():
            terminateds.update(subordinate.get_terminateds(proxy))
        return terminateds
    
    def is_terminated(self, local_state: dict) -> bool:
        return False
    
    # ============================================
    # truncateds related functions
    # ============================================
    def get_truncateds(self, proxy: "ProxyAgent") -> Dict[AgentID, bool]:
        """Collect truncation flags for this agent and all subordinates.

        Args:
            proxy: ProxyAgent for state retrieval

        Returns:
            Dict mapping agent IDs to truncation booleans
        """
        if not proxy:
            raise ValueError("Agent requires a proxy agent to derive truncated states")
        # May need to use env fields to decide
        # TODO: pass in the env field elegantly
        # e.g. done = (self._t % self.max_episode_steps) == 0
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        truncated = self.is_truncated(local_state)
        truncateds = {
            self.agent_id: truncated
        }
        for subordinate in self.subordinates.values():
            truncateds.update(subordinate.get_truncateds(proxy))
        return truncateds
    
    def is_truncated(self, local_state: dict) -> bool:
        return False

    # ============================================
    # Action taking related functions
    # ============================================
    def is_active_at(self, step: int) -> bool:
        """Whether this agent should apply its action at the given env step.

        Each env step = 1 second.  An agent with ``tick_interval=3.0`` acts
        every 3 steps.  Agents with ``tick_interval <= 1.0`` act every step.

        Args:
            step: 1-indexed env step number.
        """
        period = max(1, round(self._tick_config.tick_interval))
        return step % period == 0

    def act(self, actions: Dict[str, Any], proxy: Optional["ProxyAgent"] = None) -> None:
        self._timestep += 1

        # run self action & store local state updates in proxy
        self.handle_self_action(actions['self'], proxy)

        # run subordinate actions & store local state updates in proxy
        self.handle_subordinate_actions(actions['subordinates'], proxy)

    def layer_actions(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        """
        Format:
        {
            "self": action1,
            "subordinates: {
                "sub1": {
                    "self": subaction1,
                    "subordinates": {
                        ....
                    }
                },
                ....
            }
        }
        """
        return {
            "self": actions.get(self.agent_id),
            "subordinates": {
                subordinate_id: subordinate.layer_actions(actions)
                for subordinate_id, subordinate in self.subordinates.items()
            }
        }

    def handle_self_action(self, action: Any, proxy: Optional["ProxyAgent"] = None):
        """Handle this agent's own action.

        If the agent is inactive at the current timestep (determined by
        ``is_active_at``), ``set_action`` and ``apply_action`` are skipped.
        The agent's state is still synced to the proxy so that simulation
        results remain visible.
        """
        if self.is_active_at(int(self._timestep)):
            if action is not None:
                self.set_action(action)
            elif self.policy:
                local_obs = proxy.get_observation(self.agent_id)
                self.set_action(self.policy.forward(observation=local_obs))
            else:
                if self.level == 1:
                    logging.debug(f"No action built for ({self}) because there's no upstream action and no action policy")

            self.apply_action()

        if not self.state:
            raise ValueError("We cannot find appropriate agent state, double check your state update logic")
        proxy.set_local_state(self.agent_id, self.state)

    def handle_subordinate_actions(self, actions: Dict[AgentID, Any], proxy: Optional["ProxyAgent"] = None):
        # Use protocol to produce subordinate actions (mirrors event-driven coordinate()).
        # This allows parent-controlled action decomposition (e.g., broadcast, vector split)
        # in training mode, not just event-driven mode.
        if self._should_send_subordinate_actions():
            _, sub_actions = self.protocol.coordinate(
                coordinator_state=self.state,
                coordinator_action=self.action,
                info_for_subordinates={sub_id: None for sub_id in self.subordinates},
            )
            for sub_id, sub_action in sub_actions.items():
                if sub_action is None:
                    continue
                if sub_id not in actions:
                    actions[sub_id] = {'self': sub_action, 'subordinates': {}}
                elif actions[sub_id].get('self') is None:
                    # Upstream actions > protocol-coordinated actions > no action
                    actions[sub_id]['self'] = sub_action

        for subordinate_id, subordinate in self.subordinates.items():
            if subordinate_id in actions:
                subordinate.execute(actions[subordinate_id], proxy)
            else:
                print(f"{subordinate} not executed in current execution cycle")

    def apply_action(self):
        """Update self.state in agent based on self.action"""
        pass

    def sync_state_from_observed(self, observed_state: Dict[str, Any]) -> None:
        """Synchronize internal state features from observed state data received from proxy.

        This updates the agent's internal state to reflect any external changes (e.g., from simulation)
        that were applied in the proxy. The observed_state format depends on whether it contains
        numpy arrays (from observed_by()) or field-level dicts (from full state retrieval).

        Args:
            observed_state: State data from proxy, either:
                - Dict[str, np.ndarray]: Feature vectors from observed_by()
                - Dict[str, Dict[str, Any]]: Feature field dicts for direct update
        """
        if not observed_state:
            return

        for feature_name, feature_data in observed_state.items():
            # O(1) lookup by feature name
            if feature_name not in self.state.features:
                continue
            feature = self.state.features[feature_name]
            if isinstance(feature_data, dict):
                # Direct field-level update
                feature.set_values(**feature_data)
            elif isinstance(feature_data, np.ndarray):
                # Vector format - reconstruct field values using feature.names()
                field_names = feature.names()
                if len(field_names) == len(feature_data):
                    updates = {name: float(val) for name, val in zip(field_names, feature_data)}
                    feature.set_values(**updates)

    # ============================================
    # Event-Driven Execution via scheduler
    # ============================================
    
    # Event tick related methods. Note: these methods only define the logic of what to do when receiving certain events
    # (e.g. message delivery), but not when to trigger these events. The latter is determined by the scheduler and 
    # the tick configuration (e.g. tick interval, message delay, etc.) that defines the timing of event scheduling.
    @property
    def tick_config(self) -> TickConfig:
        """Get the tick configuration for this agent."""
        return self._tick_config

    @tick_config.setter
    def tick_config(self, config: TickConfig) -> None:
        """Set the tick configuration for this agent."""
        self._tick_config = config


    def enable_jitter(
        self,
        jitter_type: Optional[JitterType] = None,
        jitter_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Enable jitter for testing mode.

        Converts current tick_config to use jitter with same base values.

        Args:
            jitter_type: Distribution type for jitter (default: GAUSSIAN)
            jitter_ratio: Jitter magnitude as fraction of base
            seed: Optional RNG seed for reproducibility
        """
        if jitter_type is None:
            jitter_type = JitterType.GAUSSIAN

        self._tick_config = TickConfig.with_jitter(
            tick_interval=self._tick_config.tick_interval,
            obs_delay=self._tick_config.obs_delay,
            act_delay=self._tick_config.act_delay,
            msg_delay=self._tick_config.msg_delay,
            jitter_type=jitter_type,
            jitter_ratio=jitter_ratio,
            seed=seed,
        )

    def disable_jitter(self) -> None:
        """Disable jitter (switch to deterministic mode)."""
        self._tick_config = TickConfig.deterministic(
            tick_interval=self._tick_config.tick_interval,
            obs_delay=self._tick_config.obs_delay,
            act_delay=self._tick_config.act_delay,
            msg_delay=self._tick_config.msg_delay,
        )

    def _check_for_upstream_action(self) -> None:
        """Check message broker for action from upstream parent."""
        if not self.upstream_id or not self._message_broker:
            self._upstream_action = None
            return

        actions = self.receive_upstream_actions(
            sender_id=self.upstream_id,
            clear=True,
        )
        self._upstream_action = actions[-1] if actions else None

    # Action-related functions for event-driven execution (testing mode)
    def compute_action(self, obs: Any, scheduler: EventScheduler):
        # Priority: upstream action > policy > no action
        # Action construction: 
        #   - Option A: Get from message_broker
        #   - Option B: Get from policy forward function (for coordinators with policies)
        # Action decomposition: decomposition defined by protocol (for coordinators with protocols)
        if self._upstream_action is not None:
            action = self._upstream_action
            self._upstream_action = None  # Clear after use
        elif self.policy:
            action = self.policy.forward(observation=obs)
        else:
            if self.level == FIELD_LEVEL and not self.upstream_id:
                raise ValueError(f"Warning: {self} has no policy and no upstream action")
            logging.debug(f"{self} skipping action: no upstream action received and no local policy (upstream={self.upstream_id})")
            return

        # Coordinate subordinate actions if needed
        if self._should_send_subordinate_actions():
            self.coordinate(obs, action)

        # Set self action
        self.set_action(action)
        if self.action:
            # if action is not None, apply it to update state and sync with proxy
            scheduler.schedule_action_effect(
                agent_id=self.agent_id,
                delay=self._tick_config.act_delay,
            )

    # ============================================
    # Handler-related utils
    # ============================================
    class handler:
        """Decorator for registering event handlers.

        This is a descriptor that works both as @Agent.handler (from outside/subclasses)
        and as @handler (from inside the Agent class definition).
        """
        def __init__(self, event_type_str: str):
            self.event_type_str = event_type_str
            event_type = EVENT_TYPE_FROM_STRING.get(event_type_str)
            if not event_type:
                raise KeyError(f"Event type '{event_type_str}' not found. Choose from: {list(EVENT_TYPE_FROM_STRING.keys())}")
            self.event_type = event_type

        def __call__(self, func: Callable):
            """Called when used as @handler("event_type")."""
            # Store event type on the function itself for later registration
            func._handler_event_type = self.event_type
            return func

    def get_handlers(self) -> Dict[EventType, Callable[[Event, "EventScheduler"], None]]:
        """Return handlers bound to this agent instance.

        Handlers are stored as unbound methods at the class level, but need to be
        bound to the specific agent instance when retrieved.
        """
        bound_handlers = {}
        for event_type, func in self._event_handler_funcs.items():
            # Bind the unbound method to this agent instance
            bound_handlers[event_type] = lambda e, s, f=func: f(self, e, s)
        return bound_handlers
    
    # ============================================
    # Additional subordinate controls (Protocol Usage)
    # ============================================
    def _should_send_subordinate_actions(self) -> bool:
        """Check if agent should coordinate subordinate actions."""
        if not self.action or not self.action.is_valid():
            return False
        if not self.subordinates or not self.protocol:
            return False
        return not self.protocol.no_action()
    
    def coordinate(self, obs: Any, action: Any) -> None:
        """Coordinate subordinate actions based on protocol."""
        if not self.protocol or not self.subordinates:
            return

        messages, actions = self.protocol.coordinate(
            coordinator_state=self.state,
            coordinator_action=action,
            info_for_subordinates={sub_id: obs for sub_id in self.subordinates},
        )
        # Send coordinated actions to subordinates via message broker
        for sub_id, sub_action in actions.items():
            self.send_subordinate_action(sub_id, sub_action)

        # Send coordination messages to subordinates via message broker
        # Note: the message passed is not used yet
        for sub_id, message in messages.items():
            self.send_info(
                broker=self._message_broker,
                recipient_id=sub_id,
                info=message,
            )
    
    # ============================================
    # Messaging via message broker
    #
    # Key utils:
    # - publish
    # - consume
    # - send_action
    # - send_info
    # - receive_actions
    # - receive_info
    # ============================================
    def set_message_broker(self, broker: MessageBroker) -> None:
        """Set the message broker for this agent. [Both Modes]

        Called by the environment to configure distributed messaging.

        Args:
            broker: MessageBroker instance
        """
        self._message_broker = broker

    @property
    def message_broker(self) -> Optional[MessageBroker]:
        """Get the message broker for this agent. [Both Modes]"""
        return self._message_broker


    def receive_upstream_actions(
        self,
        sender_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Any]:
        """Receive action messages from the message broker.

        Convenience method for receiving action messages from upstream.

        Args:
            sender_id: Optional sender ID (uses upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of actions received
        """
        if self._message_broker is None:
            return []

        return self.receive_actions(
            broker=self._message_broker,
            upstream_id=sender_id,
            clear=clear,
        )

    def send_subordinate_action(
        self,
        recipient_id: str,
        action: Any,
    ) -> None:
        """Send an action to a subordinate agent.

        Convenience method for sending actions to subordinates.

        Args:
            recipient_id: ID of the subordinate agent
            action: Action to send
        """
        if self._message_broker is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has no message broker configured."
            )
        if action is None:
            print(f"Warning: No action to send to subordinate {recipient_id}")
            return

        self.send_action(
            broker=self._message_broker,
            recipient_id=recipient_id,
            action=action,
        )

    def _publish(
        self,
        broker: MessageBroker,
        channel: str,
        payload: Dict[str, Any],
        recipient_id: str = "broadcast",
        message_type: str = "INFO",
    ) -> None:
        """Publish a message to a channel.

        Args:
            broker: MessageBroker instance
            channel: Channel name to publish to
            payload: Message payload
            recipient_id: Recipient agent ID (default: broadcast)
            message_type: Type of message (ACTION, INFO, BROADCAST, etc.)
        """
        msg = BrokerMessage(
            env_id=self.env_id or "default",
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            timestamp=self._timestep,
            message_type=MessageType[message_type],
            payload=payload,
        )
        broker.publish(channel, msg)

    def _consume(
        self,
        broker: MessageBroker,
        channel: str,
        clear: bool = True,
    ) -> List[BrokerMessage]:
        """Consume messages from a channel.

        Args:
            broker: MessageBroker instance
            channel: Channel name to consume from
            clear: If True, remove consumed messages

        Returns:
            List of messages for this agent
        """
        return broker.consume(
            channel=channel,
            recipient_id=self.agent_id,
            env_id=self.env_id or "default",
            clear=clear,
        )

    def send_action(
        self,
        broker: MessageBroker,
        recipient_id: str,
        action: Any,
    ) -> None:
        """Send an action to a subordinate.

        Args:
            broker: MessageBroker instance
            recipient_id: ID of the recipient agent
            action: Action to send
        """
        channel = ChannelManager.action_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self._publish(
            broker=broker,
            channel=channel,
            payload={"action": action},
            recipient_id=recipient_id,
            message_type="ACTION",
        )

    def send_info(
        self,
        broker: MessageBroker,
        recipient_id: str,
        info: Dict[str, Any],
    ) -> None:
        """Send info to an upstream agent.

        Args:
            broker: MessageBroker instance
            recipient_id: ID of the recipient agent (typically upstream)
            info: Information payload
        """
        channel = ChannelManager.info_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self._publish(
            broker=broker,
            channel=channel,
            payload=info,
            recipient_id=recipient_id,
            message_type="INFO",
        )

    def receive_actions(
        self,
        broker: MessageBroker,
        upstream_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Any]:
        """Receive actions from upstream.

        Args:
            broker: MessageBroker instance
            upstream_id: ID of the upstream agent (uses self.upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of actions received
        """
        if upstream_id is None:
            upstream_id = self.upstream_id
        if upstream_id is None:
            return []

        channel = ChannelManager.action_channel(
            upstream_id, self.agent_id, self.env_id or "default"
        )
        messages = self._consume(broker, channel, clear=clear)
        return [msg.payload.get("action") for msg in messages if "action" in msg.payload]

    def receive_info(
        self,
        broker: MessageBroker,
        subordinate_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Receive info from subordinates.

        Args:
            broker: MessageBroker instance
            subordinate_ids: IDs of subordinate agents (uses self.subordinates if not provided)

        Returns:
            Dict mapping subordinate IDs to their info payloads
        """
        if subordinate_ids is None:
            subordinate_ids = list(self.subordinates.keys())

        result = {}
        for sub_id in subordinate_ids:
            channel = ChannelManager.info_channel(
                sub_id, self.agent_id, self.env_id or "default"
            )
            messages = self._consume(broker, channel)
            if messages:
                result[sub_id] = [msg.payload for msg in messages]

        return result
    
    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"

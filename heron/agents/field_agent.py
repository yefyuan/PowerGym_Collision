

from typing import Any, Dict, Optional, List

import numpy as np
from gymnasium.spaces import Box, Space

from heron.agents.base import Agent
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import FieldAgentState, State
from heron.core.action import Action
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG, ScheduleConfig
from heron.scheduling.scheduler import EventScheduler, Event
from heron.agents.proxy_agent import Proxy
from heron.agents.constants import (
    FIELD_LEVEL,
    PROXY_AGENT_ID,
    MSG_GET_INFO,
    MSG_SET_STATE,
    STATE_TYPE_LOCAL,
    MSG_SET_STATE_COMPLETION,
    MSG_SET_TICK_RESULT,
    INFO_TYPE_OBS,
    INFO_TYPE_LOCAL_STATE,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
)


class FieldAgent(Agent):
    def __init__(
        self,
        agent_id: AgentID,
        features: Optional[List[Feature]] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        schedule_config: Optional[ScheduleConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):

        self.protocol = protocol
        self.policy = policy

        super().__init__(
            agent_id=agent_id,
            level=FIELD_LEVEL,
            features=features,
            upstream_id=upstream_id,
            subordinates=None, # L1 agent has no subordinate
            env_id=env_id,
            schedule_config=schedule_config or DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG,
            policy=policy,
            protocol=protocol,
        )

    # ============================================
    # Functions requiring overriding:
    # 
    # [Must]
    # 1. compute_local_reward
    #   - applying specific reward computation mechanism
    #   - may use additional info via proxy (for additional info) + protocol (for visibility and info control)
    # 2. set_action
    #   - set self.action
    # 3. set_state
    #   - update self.state
    # 4. apply_action
    #   - given self.action, update self.state
    # 
    # [Optional]
    # 1. init_state & init_action
    #   - by default, state is initialized with features from parent coordinator, and action is
    #     initialized as empty Action. Override if you want to add additional features or custom initialization logic.
    # 2. get_local_info
    # 3. is_terminated
    # 4. is_truncated
    # 5. post_proxy_attach
    # ============================================

    def init_state(self, features: List[Feature] = []) -> State:
        """Initialize a FieldAgentState from the provided features."""
        return FieldAgentState(
            owner_id=self.agent_id,
            owner_level=FIELD_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize an empty Action (override to define custom action structure)."""
        return Action()

    def set_state(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "No implementation of set_state found, you need to define how to update state for this agent"
        )

    def set_action(self, action: Any, *args, **kwargs) -> None:
        raise NotImplementedError(
            "No implementation of set_action found, you need to define how to set action for this agent"
        )

    def post_proxy_attach(self, proxy: "Proxy") -> None:
        """Hook for any additional setup after proxy attachment and global state initialization."""
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(proxy)

    def compute_local_reward(self, local_state: dict) -> float:
        raise NotImplementedError(
            "No implementation of compute_local_reward found, you need to define reward computation mechanism for this agent"
        )

    def apply_action(self):
        raise NotImplementedError(
            "No implementation of apply_action found, you need to define action effect mechanism for this agent"
        )

    def get_action_space(self) -> Space:
        """Get action space based on agent action. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        return self.action.space

    def get_observation_space(self, proxy: Optional[Proxy] = None) -> Space:
        """Get observation space based on agent state. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        if hasattr(self, "observation_space") and self.observation_space is not None:
            return self.observation_space
        if not proxy:
            raise ValueError("[get_observation_space]: We need proxy agent to retrieve observations for field agent")

        sample_obs = self.observe(proxy=proxy)
        obs_vector = sample_obs[self.agent_id]
        if len(obs_vector.shape) == 1:  # Vector observation
            return Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_vector.shape,
                dtype=np.float32,
            )
        else:
            raise NotImplementedError(
                "Extend _get_observation_space to handle images, discrete, "
                "or structured observations."
            )
    
    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    def reset(self, *, seed: Optional[int] = None, proxy: Optional[Proxy] = None, **kwargs) -> Any:
        super().reset(seed=seed, proxy=proxy, **kwargs)

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(proxy)
            
    # execute() inherited from base class - uses default implementation

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """
        Action phase - equivalent to `self.act`

        Note:
        - FieldAgent ticks only upon CoordinatorAgent.tick (see heron/agents/coordinator_agent.py)
        - Upstream actions checked in base.Agent.tick() via _check_for_upstream_action()

        Always requests obs from proxy first for state sync. If an upstream
        action was received (stored in self._upstream_action by super().tick()),
        it is applied after state sync in the get_obs_response handler.
        """
        super().tick(scheduler, current_time)  # Update internal timestep and check for upstream actions

        # Always request obs from proxy first for state sync.
        # Upstream action (if any) will be applied after sync in get_obs_response handler.
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: self.protocol},
            delay=self._schedule_config.msg_delay,
        )
    
    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.tick(scheduler, event.timestamp)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.apply_action()
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_SET_STATE: STATE_TYPE_LOCAL, "body": self.state.to_dict(include_metadata=True)},
            delay=self._schedule_config.msg_delay,
        )

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Deliver message via message broker."""
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})

        # Publish message via broker
        if "get_obs_response" in message_content:
            assert isinstance(message_content, dict)
            response_data = message_content["get_obs_response"]
            body = response_data[MSG_KEY_BODY]

            # Proxy sends both obs and local_state (design principle: agent asks for obs, proxy gives both)
            obs_dict = body["obs"]
            local_state = body["local_state"]

            # Deserialize observation dict back to Observation object
            obs = Observation.from_dict(obs_dict)

            # Sync state first (proxy gives both obs & state)
            self.sync_state_from_observed(local_state)

            self.compute_action(obs, scheduler)
        elif "get_local_state_response" in message_content:
            response_data = message_content["get_local_state_response"]
            local_state = response_data[MSG_KEY_BODY]

            # Sync internal state with what's stored in proxy (may have been modified by simulation)
            self.sync_state_from_observed(local_state)

            tick_result = {
                "reward": self.compute_local_reward(local_state),
                "terminated": self.is_terminated(local_state),
                "truncated": self.is_truncated(local_state),
                "info": self.get_local_info(local_state)
            }

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_TICK_RESULT: INFO_TYPE_LOCAL_STATE, MSG_KEY_BODY: tick_result},
            )
        elif MSG_SET_STATE_COMPLETION in message_content:
            if message_content[MSG_SET_STATE_COMPLETION] != "success":
                raise ValueError(f"State update failed in proxy, cannot proceed")

            # We don't need reward delays for field agents
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
            )
        else:
            raise NotImplementedError
    

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================
    def __repr__(self) -> str:
        return f"FieldAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"


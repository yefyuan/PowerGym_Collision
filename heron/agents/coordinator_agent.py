

from typing import Any, Dict, List, Optional

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import CoordinatorAgentState, State
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG, ScheduleConfig
from heron.scheduling.scheduler import EventScheduler, Event
from heron.agents.constants import (
    COORDINATOR_LEVEL,
    PROXY_AGENT_ID,
    MSG_GET_INFO,
    MSG_SET_STATE_COMPLETION,
    MSG_SET_TICK_RESULT,
    INFO_TYPE_OBS,
    INFO_TYPE_LOCAL_STATE,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
)


class CoordinatorAgent(Agent):
    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        features: Optional[List[Feature]] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        schedule_config: Optional[ScheduleConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):

        super().__init__(
            agent_id=agent_id,
            level=COORDINATOR_LEVEL,
            features=features,
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            schedule_config=schedule_config or DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG,
            policy=policy,
            protocol=protocol,
        )

    def init_state(self, features: List[Feature] = []) -> State:
        """Initialize a CoordinatorAgentState from the provided features."""
        return CoordinatorAgentState(
            owner_id=self.agent_id,
            owner_level=COORDINATOR_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize an empty Action (coordinators delegate actions to subordinates)."""
        return Action()

    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Store action for protocol-based distribution to subordinates."""
        if isinstance(action, Action):
            self.action = action

    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    # execute() inherited from base class - uses default implementation

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """
        Action phase - equivalent to `self.act`

        Note: 
        - CoordinatorAgent ticks only upon SystemAgent.tick (see heron/agents/system_agent.py)
        - Upstream actions checked in SystemAgent.tick() via _check_for_upstream_action()
        """
        super().tick(scheduler, current_time)  # Update internal timestep and check for upstream actions

        # Schedule subordinate ticks -> initiate action process
        for subordinate_id in self.subordinates:
            scheduler.schedule_agent_tick(subordinate_id)
        
        # Always request obs from proxy first for state sync.
        # Upstream action (if any) will be applied after sync in get_obs_response handler
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
        """Coordinator actions don't update local state (they coordinate subordinates).

        No-op handler to handle action_effect events scheduled by compute_action.
        """
        pass

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

            # Compute action - policy decides which parts of observation to use
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
            for subordinate_id in self.subordinates:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=subordinate_id,
                    message={MSG_SET_STATE_COMPLETION: message_content[MSG_SET_STATE_COMPLETION]},
                )
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
                delay=self._schedule_config.reward_delay,
            )
        else:
            raise NotImplementedError
    

    # ============================================
    # Convenience Property
    # ============================================
    @property
    def field_agents(self) -> Dict[AgentID, FieldAgent]:
        """Alias for subordinates - more descriptive for CoordinatorAgent context."""
        return self.subordinates

    def __repr__(self) -> str:
        num_fields = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"CoordinatorAgent(id={self.agent_id}, field_agents={num_fields}, protocol={protocol_name})"

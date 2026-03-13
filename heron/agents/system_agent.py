import logging
from typing import List, Any, Callable, Dict, Optional, Tuple
from enum import Enum

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import State, SystemAgentState
from heron.utils.typing import AgentID, MultiAgentDict
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.scheduler import Event, EventScheduler
from heron.scheduling.tick_config import DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG, ScheduleConfig
from gymnasium.spaces import Box, Space
from heron.agents.constants import (
    SYSTEM_LEVEL,
    SYSTEM_AGENT_ID,
    PROXY_AGENT_ID,
    MSG_GET_INFO,
    MSG_SET_STATE,
    MSG_SET_STATE_COMPLETION,
    MSG_SET_TICK_RESULT,
    INFO_TYPE_OBS,
    INFO_TYPE_GLOBAL_STATE,
    INFO_TYPE_LOCAL_STATE,
    STATE_TYPE_GLOBAL,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
)


class SystemAgent(Agent):
    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        features: Optional[List[Feature]] = None,
        # hierarchy params
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
            agent_id=agent_id or SYSTEM_AGENT_ID,
            level=SYSTEM_LEVEL,
            features=features,
            subordinates=subordinates,
            upstream_id=None,  # System agent has no upstream
            env_id=env_id,
            schedule_config=schedule_config or DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG,
            policy=policy,
            protocol=protocol
        )

    def init_state(self, features: List[Feature] = []) -> State:
        """Initialize a SystemAgentState from the provided features."""
        return SystemAgentState(
            owner_id=self.agent_id,
            owner_level=SYSTEM_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize an empty Action (system agent manages simulation, not direct actions)."""
        return Action()

    
    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    def reset(self, *, seed: Optional[int] = None, proxy: Optional[Proxy] = None, **kwargs) -> Any:
        """Reset system agent and all coordinators. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed, proxy=proxy, **kwargs)

        return self.observe(proxy=proxy), {}
    
    def execute(self, actions: Dict[AgentID, Any], proxy: Optional[Proxy] = None) -> None:
        """Execute actions with hierarchical coordination and simulation. [Training Mode]"""
        if not proxy:
            raise ValueError("We still require a valid proxy agent so far")

        # Run pre-step hook (e.g., update profiles for current timestep)
        if self._pre_step_func is not None:
            self._pre_step_func()

        # Sync state first (by-product of proxy having updated global state)
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        self.sync_state_from_observed(local_state)

        # Layer actions for hierarchical structure and act
        actions = self.layer_actions(actions)
        self.act(actions, proxy)
        
        # get latest global state in dict format for simulation pipeline
        global_state = proxy.get_global_states(self.agent_id, self.protocol, for_simulation=True)
        # run external environment simulation step (upon action -> agent state update)
        updated_global_state = self.simulate(global_state)
        
        # broadcast updated global state via proxy
        proxy.set_global_state(updated_global_state)

        # get current step statistics
        obs = self.observe(proxy=proxy)
        rewards = self.compute_rewards(proxy)
        infos = self.get_info(proxy)
        terminateds = self.get_terminateds(proxy)
        truncateds = self.get_truncateds(proxy)

        # set step results in proxy agent
        proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)


    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """
        Action phase - equivalent to `self.act`
        - Initiate tick for subordinates
        - Initiate self
        - Schedule for simulation

        Note on entire event-driven flow:
        - Schedule Actions
        - Schedule Simulation (tick handles until this part)
        (following steps are handled by individual event handlers)
        - Actions take effect (some may fail to take effect before next step)
        - Run simulation
        - Update states
        - Compute rewards
        - Schedule next tick
        """
        # Run pre-step hook (e.g., update profiles for current timestep)
        if self._pre_step_func is not None:
            self._pre_step_func()

        super().tick(scheduler, current_time)  # Update internal timestep and check for upstream actions

        # Schedule subordinate ticks
        for subordinate_id in self.subordinates:
            scheduler.schedule_agent_tick(subordinate_id)
        
        if self.policy:
            # Compute & execute self action
            # Ask proxy_agent for global state to compute local action
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: self.protocol},
                delay=self._schedule_config.msg_delay,
            )
        else:
            logging.debug(f"{self} doesn't act itself, because there's no action policy")
        
        # schedule simulation
        scheduler.schedule_simulation(self.agent_id, self._simulation_wait_interval)


    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.tick(scheduler, event.timestamp)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """System agent actions don't update local state (they manage simulation).

        No-op handler to handle action_effect events scheduled by compute_action.
        """
        pass

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Deliver message via message broker."""
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})

        # 4 cases:
        # a. getting observation -> computing actions
        # b. getting global state -> starting simulation
        # c. getting local state -> compute rewards
        # d. receiving "state set completion" message from proxy -> schedule reward computation
        assert isinstance(message_content, dict)
        if "get_obs_response" in message_content:
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
        elif "get_global_state_response" in message_content:
            # The response structure is {'get_global_state_response': {'body': {...}}}
            response_data = message_content["get_global_state_response"]
            global_state = response_data[MSG_KEY_BODY] # a dict of {agent_id: state}
            updated_global_state = self.simulate(global_state)
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_STATE: STATE_TYPE_GLOBAL, MSG_KEY_BODY: updated_global_state},
                delay=self._schedule_config.msg_delay,
            )
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

            # Schedule next tick for continuous operation if not terminated/truncated
            if not tick_result["terminated"] and not tick_result["truncated"]:
                scheduler.schedule_agent_tick(self.agent_id)

        elif MSG_SET_STATE_COMPLETION in message_content:
            if message_content[MSG_SET_STATE_COMPLETION] != "success":
                raise ValueError(f"State update failed in proxy, cannot proceed with reward computation")

            # Initiate state syncing + reward computation after state update by propagating the set state completion message.
            # Each coordinator handles triggering its own subordinate (field agent) reward computation
            # internally (see CoordinatorAgent.message_delivery_handler get_local_state_response branch).
            for subordinate_id in self.subordinates:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=subordinate_id,
                    message={MSG_SET_STATE_COMPLETION: message_content[MSG_SET_STATE_COMPLETION]},
                )

            # System agent waits for coordinators (which wait for field agents) before
            # computing its own reward with aggregated subordinate rewards.
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
                delay=self._schedule_config.reward_delay,
            )
        else:
            raise NotImplementedError

    @Agent.handler("simulation")
    def simulation_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """
        Simulation cycle:
        1. Ask for proxy for global states
        2. Proxy returns the global states via message delivery
        3. Run simulation
        4. Gather updated global state and broadcast via proxy
        """
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_GET_INFO: INFO_TYPE_GLOBAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
            delay=self._schedule_config.msg_delay,
        )
    

    # ============================================
    # Simulation related functions - SystemAgent-specific
    # ============================================
    def set_simulation(
        self,
        simulation_func: Callable,
        env_state_to_global_state: Callable,
        global_state_to_env_state: Callable,
        wait_interval: Optional[float] = None,
        pre_step_func: Optional[Callable] = None,
    ):
        """
        simulation_func: simulation function passed from environment
        wait_interval: waiting time between action kick-off and simulation starts.
            If None, derives from current schedule_config.tick_interval at runtime.
        pre_step_func: optional hook called at the start of each step (before actions)
        """
        self._simulation_func = simulation_func
        self._env_state_to_global_state = env_state_to_global_state
        self._global_state_to_env_state = global_state_to_env_state
        self._explicit_wait_interval = wait_interval  # None means derive from schedule_config
        self._pre_step_func = pre_step_func

    @property
    def _simulation_wait_interval(self) -> float:
        if self._explicit_wait_interval is not None:
            return self._explicit_wait_interval
        return self.schedule_config.tick_interval

    def simulate(self, global_state: Dict[AgentID, Any]) -> Any:
        # proxy.get_global_states() returns a flat {agent_id: state_dict} dict,
        # but global_state_to_env_state() expects {"agent_states": {agent_id: ...}}.
        # Wrap if needed so both execute() and event-driven paths work correctly.
        if "agent_states" not in global_state:
            global_state = {"agent_states": global_state}
        env_state = self._global_state_to_env_state(global_state)
        updated_env_state = self._simulation_func(env_state)
        updated_global_state = self._env_state_to_global_state(updated_env_state)
        return updated_global_state


    # ============================================
    # Utility Methods
    # ============================================
    @property
    def coordinators(self) -> Dict[AgentID, CoordinatorAgent]:
        """Alias for subordinates - more descriptive for SystemAgent context."""
        return self.subordinates

    def __repr__(self) -> str:
        num_coords = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords}, protocol={protocol_name})"

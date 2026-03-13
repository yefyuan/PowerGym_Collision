# HERON Execution Flows - Complete Reference

This document provides a comprehensive description of the two execution modes in the HERON framework:
- **Mode A**: CTDE (Centralized Training with Decentralized Execution) - Synchronous training mode
- **Mode B**: Event-Driven Testing - Asynchronous testing mode with realistic timing

---

## **Table of Contents**

1. [Key Components](#key-components)
2. [Agent Hierarchy & Action Passing](#agent-hierarchy--action-passing)
3. [Message Broker Architecture](#message-broker-architecture)
4. [CTDE Training Mode](#ctde-training-mode)
5. [Event-Driven Testing Mode](#event-driven-testing-mode)
6. [Protocol-Based Coordination](#protocol-based-coordination)
7. [Observation & Action Data Flow](#observation--action-data-flow)
8. [Proxy State Management](#proxyagent-state-management)
9. [Handler Registration & Event Processing](#handler-registration--event-processing)
10. [Complete Execution Examples](#complete-execution-examples)

---

## **Key Components**

### **Agent Hierarchy (3 Levels)**
```
SystemAgent (L3) - Root orchestrator
  └─> CoordinatorAgent (L2) - Middle coordination layer
      └─> FieldAgent (L1) - Leaf execution agents
```

### **Core Classes**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| **BaseEnv** | Environment base class | `step()`, `run_event_driven()`, `reset()` |
| **SystemAgent** | Top-level orchestrator | `execute()`, `tick()`, `simulate()` |
| **CoordinatorAgent** | Mid-level coordinator | `execute()`, `tick()`, `coordinate()` |
| **FieldAgent** | Leaf execution agent | `execute()`, `tick()`, `apply_action()` |
| **Proxy** | State distribution hub | `get_observation()`, `set_local_state()` |
| **Action** | Action representation | `vector()`, `set_values()`, `clip()` |
| **Observation** | Observation with array interface | `vector()`, `__array__()`, `shape` |
| **State** | Agent state container (features as Dict) | `observed_by()`, `to_dict()`, `from_dict()` |
| **Feature** | Feature definition | `vector()`, `is_observable_by()` |
| **EventScheduler** | Discrete event simulation | `run_until()`, `schedule_agent_tick()` |
| **MessageBroker** | Inter-agent messaging | `publish()`, `consume()` |
| **Protocol** | Coordination logic (communication + action) | `coordinate()` |

---

## **Agent Hierarchy & Action Passing**

### **Hierarchical Structure**

```
┌─────────────────────────────────────────────────────────────────┐
│ SystemAgent (Level 3, SYSTEM_LEVEL=3)                           │
│ - Root of hierarchy (upstream_id=None)                          │
│ - Manages simulation/physics via set_simulation()               │
│ - Orchestrates all subordinates                                 │
│ - Owns pre_step hook for timestep-specific setup                │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ - execute() -> Orchestrate hierarchical execution               │
│ - simulate() -> Run environment physics                         │
│ - observe() -> Aggregate observations from all levels           │
│ - compute_rewards() -> Aggregate rewards                        │
│ Default tick_interval: 300.0s                                   │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ subordinates (coordinators)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoordinatorAgent (Level 2, COORDINATOR_LEVEL=2)                 │
│ - Has upstream_id pointing to system                            │
│ - Has subordinates (field agents)                               │
│ - Coordinates action decomposition                              │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ - receive_upstream_actions() -> Get action from system          │
│ - coordinate() -> Decompose actions for field agents            │
│ - send_subordinate_action() -> Pass actions to fields           │
│ Default tick_interval: 60.0s                                    │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ subordinates (field agents)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ FieldAgent (Level 1, FIELD_LEVEL=1)                             │
│ - Leaf nodes (subordinates=None)                                │
│ - Has upstream_id pointing to coordinator                       │
│ - Executes atomic actions                                       │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ - receive_upstream_actions() -> Get action from coordinator     │
│ - set_action() -> Apply received or computed action             │
│ - apply_action() -> Update state based on action                │
│ - compute_local_reward() -> Calculate local reward              │
│ Default tick_interval: 1.0s                                     │
└─────────────────────────────────────────────────────────────────┘
```

### **Action Passing Protocol**

**Priority Rules:**
1. **Upstream action** from parent (highest priority)
2. **Policy-computed action** if agent has policy
3. **No action** (neutral)

```python
# In Agent.compute_action() (event-driven mode):
def compute_action(self, obs, scheduler):
    # Priority 1: Check cached upstream action
    if self._upstream_action is not None:
        action = self._upstream_action
        self._upstream_action = None  # Clear after use
    # Priority 2: Compute via policy if available
    elif self.policy:
        action = self.policy.forward(observation=obs)
    else:
        if self.level == FIELD_LEVEL and not self.upstream_id:
            raise ValueError(f"Warning: {self} has no policy and no upstream action")

    # Coordinate subordinate actions if needed
    if self._should_send_subordinate_actions():
        self.coordinate(obs, action)

    # Set self action and schedule effect
    self.set_action(action)
    if self.action:
        scheduler.schedule_action_effect(
            agent_id=self.agent_id,
            delay=self._schedule_config.act_delay,
        )
```

### **Action Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│ SystemAgent computes subordinate actions via protocol           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ protocol.coordinate(                                        │ │
│ │     coordinator_state=self.state,                           │ │
│ │     coordinator_action=action,                              │ │
│ │     info_for_subordinates={coord_id: obs for coords}        │ │
│ │ )                                                           │ │
│ │ -> Returns (messages, subordinate_actions)                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ For each coordinator:                                           │
│     send_subordinate_action(coord_id, coord_action)             │
│     └─> broker.publish(action_channel, Message(action))         │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoordinatorAgent.tick()                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ # base.tick() calls _check_for_upstream_action()            │ │
│ │ # which caches upstream action in self._upstream_action     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ # ALWAYS request obs from proxy first (for state sync)          │
│ schedule_message_delivery(                                      │
│     sender=self, recipient=proxy,                               │
│     message={MSG_GET_INFO: INFO_TYPE_OBS, protocol}             │
│ )                                                               │
│                                                                 │
│ # In message_delivery_handler on get_obs_response:              │
│ #   parse obs + local_state from proxy response                 │
│ #   sync_state_from_observed(local_state)                       │
│ #   compute_action(obs, scheduler)                              │
│ #     -> uses self._upstream_action if present (priority)       │
│ #     -> else uses policy to select action                      │
│ #   protocol.coordinate() -> field_actions                      │
│ #   send_subordinate_action(field_id, field_action)             │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ FieldAgent.tick()                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ # base.tick() calls _check_for_upstream_action()            │ │
│ │ # which caches upstream action in self._upstream_action     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ # ALWAYS request obs from proxy first (for state sync)          │
│ schedule_message_delivery(                                      │
│     sender=self, recipient=proxy,                               │
│     message={MSG_GET_INFO: INFO_TYPE_OBS, protocol}             │
│ )                                                               │
│                                                                 │
│ # In message_delivery_handler on get_obs_response:              │
│ #   parse obs + local_state from proxy response                 │
│ #   sync_state_from_observed(local_state)                       │
│ #   compute_action(obs, scheduler)                              │
│ #     -> uses self._upstream_action if present (priority)       │
│ #     -> else uses policy to select action                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Message Broker Architecture**

### **Message Structure**

```python
@dataclass
class Message:
    env_id: str                # Environment identifier
    sender_id: str             # Who sent the message
    recipient_id: str          # Who receives the message
    timestamp: float           # When message was created
    message_type: MessageType  # ACTION, INFO, BROADCAST, etc.
    payload: Dict[str, Any]    # Message content
```

### **Message Types**

```python
class MessageType(Enum):
    ACTION = "action"          # Parent -> Child action commands
    INFO = "info"              # Information/observation requests
    BROADCAST = "broadcast"    # Multi-recipient messages
```

### **Channel Naming Convention**

```python
class ChannelManager:
    @staticmethod
    def action_channel(sender_id: str, recipient_id: str, env_id: str) -> str:
        return f"env:{env_id}/action/{sender_id}->{recipient_id}"

    @staticmethod
    def info_channel(sender_id: str, recipient_id: str, env_id: str) -> str:
        return f"env:{env_id}/info/{sender_id}->{recipient_id}"
```

### **Broker Operations**

```python
class MessageBroker(ABC):
    @abstractmethod
    def publish(self, channel: str, message: Message) -> None:
        """Send message to channel."""

    @abstractmethod
    def consume(self, channel: str, recipient_id: str, env_id: str, clear: bool = True) -> List[Message]:
        """Receive messages from channel (non-blocking)."""

    @abstractmethod
    def create_channel(self, channel_name: str) -> None:
        """Create a new channel."""

    @abstractmethod
    def clear_environment(self, env_id: str) -> None:
        """Clear all channels for an environment (used on reset)."""
```

### **Message Constants (from heron/agents/constants.py)**

```python
# Message types for proxy-agent communication
MSG_GET_INFO = "get_info"
MSG_SET_STATE = "set_state"
MSG_SET_TICK_RESULT = "set_tick_result"
MSG_SET_STATE_COMPLETION = "set_state_completion"

# Info request types
INFO_TYPE_OBS = "obs"
INFO_TYPE_GLOBAL_STATE = "global_state"
INFO_TYPE_LOCAL_STATE = "local_state"

# State types
STATE_TYPE_GLOBAL = "global"
STATE_TYPE_LOCAL = "local"

# Message content keys
MSG_KEY_BODY = "body"
MSG_KEY_PROTOCOL = "protocol"
```

---

## **CTDE Training Mode**

### **1. Initialization Flow**

```
env = CustomEnv(system_agent=..., coordinator_agents=[...])
│
├─> BaseEnv.__init__()
│   │
│   ├─> _register_agents(system_agent, coordinator_agents)
│   │   ├─> If system_agent provided: use it directly
│   │   ├─> Elif coordinator_agents: create default SystemAgent wrapping them
│   │   ├─> system_agent.set_simulation(
│   │   │       run_simulation,
│   │   │       env_state_to_global_state,
│   │   │       global_state_to_env_state,
│   │   │       simulation_wait_interval,
│   │   │       pre_step,
│   │   │   )
│   │   └─> _register_agent(system_agent)  [recursive for all subordinates]
│   │       └─> Sets env_id on each agent, registers in self.registered_agents
│   │
│   ├─> Create Proxy(agent_id=PROXY_AGENT_ID)
│   │   └─> _register_agent(proxy)
│   │
│   ├─> Setup MessageBroker
│   │   ├─> MessageBroker.init(config) -> InMemoryBroker (default)
│   │   ├─> message_broker.attach(registered_agents)
│   │   └─> proxy.set_message_broker(message_broker)
│   │
│   ├─> proxy.attach(registered_agents)
│   │   ├─> For each agent (except proxy):
│   │   │   └─> _register_agent(agent)
│   │   │       ├─> Track agent_id, level, upstream_id
│   │   │       └─> set_local_state(agent_id, agent.state)  # State object!
│   │   ├─> init_global_state()  # Compile all agent states
│   │   ├─> For each agent: agent.post_proxy_attach(proxy)
│   │   │   └─> FieldAgent: Compute action_space, observation_space
│   │   └─> _setup_channels()  # Create proxy->agent info channels
│   │
│   └─> Setup EventScheduler
│       ├─> EventScheduler.init(config) -> EventScheduler
│       └─> scheduler.attach(registered_agents)
│           ├─> Register tick configs per agent
│           ├─> Register event handlers per agent
│           └─> Schedule initial AGENT_TICK for system_agent only
```

**Note:** Agent state/action are initialized in `Agent.__init__()`:
```python
class Agent.__init__():
    self.action = self.init_action(features=features)  # Action()
    self.state = self.init_state(features=features)    # State with features as Dict
```

### **2. Reset Flow**

```
obs, info = env.reset(seed=42)
│
├─> scheduler.reset(start_time=0.0) -> Clear event queue, re-schedule system tick
├─> clear_broker_environment() -> Clear all messages for this env_id
├─> proxy.reset(seed) -> state_cache = {} (cleared)
│
├─> system_agent.reset(seed, proxy)
│   ├─> Agent.reset() [base class]:
│   │   ├─> self._timestep = 0.0
│   │   ├─> self.action.reset()  # Reset action to neutral
│   │   ├─> self.state.reset()   # Reset state features to defaults
│   │   ├─> proxy.set_local_state(self.agent_id, self.state)  # State object!
│   │   └─> For each subordinate: subordinate.reset(seed, proxy)
│   │       └─> [Recursive - coordinators then field agents]
│   │
│   └─> SystemAgent.reset() override:
│       └─> return self.observe(proxy=proxy), {}
│
├─> proxy.init_global_state()  # Re-compile global state after all resets
│
└─> return (observations, {})
```

### **3. Step Execution Flow**

```
obs, rewards, terminated, truncated, info = env.step(actions)
│
├─> system_agent.execute(actions, proxy)
│   │
│   ├─> PRE-STEP: Run pre_step_func() hook (if configured)
│   │   └─> e.g., update load profiles for current timestep
│   │
│   ├─> STATE SYNC: Sync internal state from proxy
│   │   ├─> local_state = proxy.get_local_state(self.agent_id, self.protocol)
│   │   └─> self.sync_state_from_observed(local_state)
│   │
│   ├─> PHASE 1: Action Application
│   │   │
│   │   ├─> actions = self.layer_actions(actions)
│   │   │   └─> {
│   │   │         "self": actions.get("system_agent"),
│   │   │         "subordinates": {
│   │   │           "coord_1": {
│   │   │             "self": actions.get("coord_1"),
│   │   │             "subordinates": {
│   │   │               "field_1": {"self": actions["field_1"], "subordinates": {}},
│   │   │               "field_2": {"self": actions["field_2"], "subordinates": {}}
│   │   │             }
│   │   │           }
│   │   │         }
│   │   │       }
│   │   │
│   │   └─> self.act(actions, proxy)
│   │       ├─> self._timestep += 1
│   │       │
│   │       ├─> handle_self_action(actions['self'], proxy)
│   │       │   ├─> if action provided:
│   │       │   │   └─> set_action(action)
│   │       │   ├─> elif has policy:
│   │       │   │   ├─> obs = proxy.get_observation(agent_id)
│   │       │   │   └─> set_action(policy.forward(observation=obs))
│   │       │   ├─> apply_action() -> Updates self.state
│   │       │   └─> proxy.set_local_state(self.agent_id, self.state)
│   │       │
│   │       └─> handle_subordinate_actions(actions['subordinates'], proxy)
│   │           └─> For each subordinate:
│   │               └─> subordinate.execute(layered_actions[sub_id], proxy)
│   │                   ├─> Coordinator: sync state, then act (recursively)
│   │                   └─> FieldAgent: sync state, then act
│   │
│   ├─> PHASE 2: Simulation
│   │   │
│   │   ├─> global_state = proxy.get_global_states(system_agent_id, protocol)
│   │   │   └─> Returns visibility-filtered feature vectors for all agents
│   │   │
│   │   ├─> updated_global_state = self.simulate(global_state)
│   │   │   ├─> env_state = global_state_to_env_state(global_state)
│   │   │   ├─> updated_env_state = run_simulation(env_state)
│   │   │   └─> return env_state_to_global_state(updated_env_state)
│   │   │
│   │   └─> proxy.set_global_state(updated_global_state)
│   │
│   ├─> PHASE 3: Observation Collection
│   │   │
│   │   └─> obs = self.observe(proxy=proxy)
│   │       └─> For each agent in hierarchy:
│   │           ├─> observation = proxy.get_observation(agent_id, protocol)
│   │           │   ├─> local = state.observed_by(agent_id, level)
│   │           │   ├─> global_info = {filtered states from other agents}
│   │           │   └─> return Observation(local, global_info, timestamp)
│   │           │
│   │           └─> Returns: Dict[AgentID, Observation]
│   │
│   ├─> PHASE 4: Reward & Status Computation
│   │   │
│   │   ├─> rewards = self.compute_rewards(proxy)
│   │   │   └─> For each agent:
│   │   │       ├─> local_state = proxy.get_local_state(agent_id, protocol)
│   │   │       └─> reward = compute_local_reward(local_state)
│   │   │
│   │   ├─> infos = self.get_info(proxy)
│   │   ├─> terminateds = self.get_terminateds(proxy)
│   │   └─> truncateds = self.get_truncateds(proxy)
│   │
│   └─> PHASE 5: Cache Results
│       └─> proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)
│
└─> return proxy.get_step_results()
    ├─> obs_vectorized = {aid: obs.vector() for aid, obs in obs.items()}
    └─> return (obs_vectorized, rewards, terminateds, truncateds, infos)
```

### **Key Features:**

- **Synchronous execution**: All agents act, then simulate, then observe
- **Centralized training**: Full observability via proxy
- **State sync**: Each agent syncs from proxy before acting (to pick up simulation changes)
- **Type safety**: Observations are Observation objects, auto-vectorized for RL
- **State management**: Proxy maintains State objects with Dict-based features
- **Action passing**: Hierarchical action distribution via `layer_actions()`

---

## **Event-Driven Testing Mode**

### **1. Event Types**

| Event Type | When Scheduled | Handler | Purpose |
|------------|----------------|---------|---------|
| `AGENT_TICK` | At tick_interval | `agent_tick_handler()` | Trigger agent's tick() method |
| `MESSAGE_DELIVERY` | After msg_delay | `message_delivery_handler()` | Deliver async messages |
| `ACTION_EFFECT` | After act_delay | `action_effect_handler()` | Apply delayed action effects |
| `SIMULATION` | After wait_interval | `simulation_handler()` | Run physics simulation |

### **2. Event Priority**

```python
class EventType(Enum):
    AGENT_TICK = "agent_tick"              # Default priority
    ACTION_EFFECT = "action_effect"        # Priority: 0 (highest - state changes first)
    SIMULATION = "simulation"              # Priority: 1 (physics after actions)
    MESSAGE_DELIVERY = "message_delivery"  # Priority: 2 (communication last)
    OBSERVATION_READY = "observation_ready"
    ENV_UPDATE = "env_update"
    CUSTOM = "custom"
```

### **3. ScheduleConfig - Timing Parameters**

```python
@dataclass
class ScheduleConfig:
    tick_interval: float = 1.0   # How often agent ticks
    obs_delay: float = 0.0       # Observation latency
    act_delay: float = 0.0       # Action execution delay
    msg_delay: float = 0.0       # Message delivery latency

    jitter_type: JitterType      # NONE, GAUSSIAN, or UNIFORM
    jitter_ratio: float = 0.1    # Jitter magnitude (0.1 = 10%)
    rng: Optional[np.random.Generator]  # For reproducibility

    @classmethod
    def deterministic(cls, tick_interval=1.0, obs_delay=0.0, act_delay=0.0, msg_delay=0.0):
        return cls(tick_interval, obs_delay, act_delay, msg_delay, JitterType.NONE, 0.0)

    @classmethod
    def with_jitter(cls, tick_interval=1.0, jitter_ratio=0.1, jitter_type=JitterType.GAUSSIAN, ...):
        return cls(tick_interval, ..., jitter_type, jitter_ratio)
```

### **4. Event-Driven Execution Flow**

```
result = env.run_event_driven(episode_analyzer, t_end=100.0)
│
└─> for event in scheduler.run_until(t_end=100.0):
    │
    ├─> EventScheduler.process_next()
    │   ├─> event = heapq.heappop(event_queue)  # By (timestamp, priority, sequence)
    │   ├─> current_time = event.timestamp
    │   ├─> handler = get_handler(event.event_type, event.agent_id)
    │   └─> handler(event, scheduler)
    │
    └─> result.add_event_analysis(episode_analyzer.parse_event(event))
```

### **5. Detailed Event Sequence**

#### **Timeline Example** (System -> Coordinator -> 2 Field Agents):

```
t=0.0: AGENT_TICK(system_agent)
├─> SystemAgent.tick()
│   │
│   ├─> pre_step_func() if configured
│   │
│   ├─> super().tick() -> Update timestep, check upstream actions
│   │
│   ├─> Schedule subordinate ticks:
│   │   └─> schedule_agent_tick(coordinator_1)
│   │       └─> timestamp = 0.0 + coordinator.tick_interval
│   │
│   ├─> Action passing to subordinates:
│   │   ├─> protocol.coordinate() -> subordinate_actions
│   │   └─> For each coordinator:
│   │       └─> send_subordinate_action(coord_id, action)
│   │           └─> broker.publish(action_channel, Message(action))
│   │
│   ├─> If has policy: Request observation
│   │   └─> schedule_message_delivery(
│   │           sender=system_agent, recipient=proxy,
│   │           message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: protocol},
│   │           delay=msg_delay
│   │       )
│   │
│   └─> Schedule simulation:
│       └─> schedule_simulation(system_agent, wait_interval)
```

```
t=msg_delay: MESSAGE_DELIVERY(proxy) - observation request
└─> Proxy.message_delivery_handler()
    ├─> info_type = "obs"
    ├─> obs = get_observation(system_agent, protocol)
    │   ├─> local = state.observed_by(system_id, 3)
    │   ├─> global_info = {all agents' observed_by() filtered states}
    │   └─> return Observation(local, global_info, timestamp)
    │
    ├─> Also compute local_state for state syncing:
    │   local_state = get_local_state(system_agent, protocol)
    │
    └─> schedule_message_delivery(
            sender=proxy, recipient=system_agent,
            message={"get_obs_response": {
                "body": {
                    "obs": obs.to_dict(),       # Serialized observation
                    "local_state": local_state   # Feature vectors for state sync
                }
            }},
            delay=msg_delay
        )
```

```
t=2*msg_delay: MESSAGE_DELIVERY(system_agent) - observation response
└─> SystemAgent.message_delivery_handler()
    ├─> Extract obs_dict and local_state from body
    ├─> obs = Observation.from_dict(obs_dict)
    │
    ├─> Sync state: self.sync_state_from_observed(local_state)
    │
    └─> compute_action(obs, scheduler)
        ├─> action = policy.forward(observation=obs)
        │   └─> obs.__array__() auto-converts to np.ndarray
        ├─> If protocol: coordinate() -> send subordinate actions
        ├─> set_action(action)
        └─> schedule_action_effect(agent_id, delay=act_delay)
```

```
t=2*msg_delay+act_delay: ACTION_EFFECT(system_agent)
└─> SystemAgent.action_effect_handler()
    └─> pass  # System agent actions are no-op (manages simulation, not state)
```

```
t=coordinator_tick_interval: AGENT_TICK(coordinator_1)
└─> CoordinatorAgent.tick()
    │
    ├─> super().tick() -> _check_for_upstream_action()
    │   └─> broker.consume(action_channel) -> self._upstream_action  # Cached!
    │
    ├─> Schedule subordinate ticks:
    │   └─> For each field: schedule_agent_tick(field_id)
    │
    └─> Always request obs from proxy (for state sync):
        └─> schedule_message_delivery(proxy, {get_info: "obs"})

t=coordinator_tick+2×msg_delay: MESSAGE_DELIVERY(coordinator_1) - obs response
└─> sync_state_from_observed(local_state)
    └─> compute_action(obs, scheduler)
        ├─> Uses cached upstream action if present (priority)
        ├─> Else uses policy.forward(obs)
        ├─> coordinate() -> field_actions via protocol
        └─> send_subordinate_action(field_id, action) for each
```

```
t=field_tick_interval: AGENT_TICK(field_1)
└─> FieldAgent.tick()
    │
    ├─> super().tick() -> _check_for_upstream_action()
    │   └─> broker.consume(action_channel) -> self._upstream_action  # Cached!
    │
    └─> Always request obs from proxy (for state sync):
        └─> schedule_message_delivery(proxy, {get_info: "obs"})
```

```
t=field_tick+act_delay: ACTION_EFFECT(field_1)
└─> FieldAgent.action_effect_handler()
    ├─> apply_action()  # Updates self.state
    └─> schedule_message_delivery(
            message={"set_state": "local", "body": self.state.to_dict(include_metadata=True)},
            delay=msg_delay
        )
```

```
t=field_tick+act_delay+msg_delay: MESSAGE_DELIVERY(proxy) - state update
└─> Proxy.message_delivery_handler()
    ├─> state = State.from_dict(message['body'])
    ├─> set_local_state(agent_id, state)
    └─> schedule_message_delivery(
            message={MSG_SET_STATE_COMPLETION: "success"},
            delay=msg_delay
        )
```

```
t=state_completion_time: MESSAGE_DELIVERY(parent) - state completion
└─> Parent.message_delivery_handler()  # System or Coordinator
    ├─> State update confirmed
    │
    ├─> Initiate reward computation for subordinates:
    │   For each subordinate_id:
    │       schedule_message_delivery(
    │           sender=subordinate_id, recipient=proxy,
    │           message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: protocol}
    │       )
    │
    └─> Request own local state for reward computation:
        schedule_message_delivery(
            sender=self.agent_id, recipient=proxy,
            message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: protocol}
        )
```

```
t=wait_interval: SIMULATION(system_agent)
└─> SystemAgent.simulation_handler()
    └─> schedule_message_delivery(
            message={MSG_GET_INFO: INFO_TYPE_GLOBAL_STATE, MSG_KEY_PROTOCOL: protocol}
        )
```

```
t=wait_interval+msg_delay: MESSAGE_DELIVERY(proxy) - global state request
└─> Proxy returns global_state
```

```
t=wait_interval+2*msg_delay: MESSAGE_DELIVERY(system_agent) - global state response
└─> SystemAgent.message_delivery_handler()
    ├─> global_state = message['body']
    ├─> updated_global_state = simulate(global_state)
    │   ├─> env_state = global_state_to_env_state(global_state)
    │   ├─> updated_env_state = run_simulation(env_state)
    │   └─> return env_state_to_global_state(updated_env_state)
    │
    └─> schedule_message_delivery(
            message={MSG_SET_STATE: STATE_TYPE_GLOBAL, MSG_KEY_BODY: updated_global_state}
        )
```

```
t=reward_time: MESSAGE_DELIVERY(agents) - local state response for reward
└─> Agent.message_delivery_handler()
    ├─> local_state from response body
    │
    ├─> Sync state: self.sync_state_from_observed(local_state)
    │
    ├─> tick_result = {
    │       "reward": compute_local_reward(local_state),
    │       "terminated": is_terminated(local_state),
    │       "truncated": is_truncated(local_state),
    │       "info": get_local_info(local_state)
    │   }
    │
    └─> schedule_message_delivery(
            message={MSG_SET_TICK_RESULT: INFO_TYPE_LOCAL_STATE, MSG_KEY_BODY: tick_result}
        )

    # SystemAgent additionally: if not terminated/truncated, schedule next tick
```

### **6. Result Collection via EpisodeAnalyzer**

```
env.run_event_driven(episode_analyzer, t_end)
│
└─> For each event:
    └─> episode_analyzer.parse_event(event)
        │
        ├─> If event.event_type == MESSAGE_DELIVERY:
        │   ├─> message = event.payload["message"]
        │   │
        │   ├─> "get_obs_response" -> observation_count++
        │   ├─> "get_global_state_response" -> global_state_count++
        │   ├─> "get_local_state_response" -> local_state_count++
        │   ├─> "set_state_completion" -> state_update_count++
        │   └─> "set_tick_result" -> action_result_count++
        │       └─> Track reward_history per agent: {agent_id: [(t, reward), ...]}
        │
        └─> Returns EventAnalysis for this event

EpisodeStats accumulates all EventAnalysis objects
├─> result.summary() -> event counts, message type counts, agent counts
├─> result.get_event_counts() -> {EventType: count}
├─> result.get_message_type_counts() -> {message_type: count}
└─> result.get_agent_event_counts() -> {agent_id: count}
```

---

## **Protocol-Based Coordination**

### **Protocol Architecture**

```python
# Two-layer protocol design:
Protocol (ABC)
├── communication_protocol: CommunicationProtocol  # WHAT to communicate
└── action_protocol: ActionProtocol                  # HOW to coordinate actions

# Protocol hierarchy:
CommunicationProtocol (ABC)
├── NoCommunication              # No message passing (default)
└── [Custom implementations]

ActionProtocol (ABC)
├── NoActionCoordination         # No action coordination (default)
├── VectorDecompositionActionProtocol  # Split joint action vector
└── [Custom implementations]

Protocol (ABC)
├── NoProtocol                   # NoCommunication + NoActionCoordination
└── VerticalProtocol             # NoCommunication + VectorDecompositionActionProtocol
```

### **Protocol Base**

```python
class Protocol(ABC):
    def __init__(self, communication_protocol, action_protocol):
        self.communication_protocol = communication_protocol or NoCommunication()
        self.action_protocol = action_protocol or NoActionCoordination()

    def no_op(self) -> bool:
        """Check if this is a no-operation protocol."""
        return self.no_action() and self.no_communication()

    def no_action(self) -> bool:
        return isinstance(self.action_protocol, NoActionCoordination)

    def no_communication(self) -> bool:
        return isinstance(self.communication_protocol, NoCommunication)

    def coordinate(
        self,
        coordinator_state: Any,
        coordinator_action: Optional[Any] = None,
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Dict], Dict[AgentID, Any]]:
        """Execute full coordination cycle.
        Returns: (messages, actions)
        """
        # Step 1: Communication coordination
        messages = self.communication_protocol.compute_coordination_messages(
            sender_state=coordinator_state,
            receiver_infos=info_for_subordinates,
            context=context
        )
        # Step 2: Action coordination
        actions = self.action_protocol.compute_action_coordination(
            coordinator_action=coordinator_action,
            info_for_subordinates=info_for_subordinates,
            coordination_messages=messages,
            context=context
        )
        return messages, actions
```

### **VerticalProtocol (Default for Hierarchical)**

```python
class VerticalProtocol(Protocol):
    """Default: NoCommunication + VectorDecompositionActionProtocol."""

    def __init__(self, communication_protocol=None, action_protocol=None):
        super().__init__(
            communication_protocol=communication_protocol or NoCommunication(),
            action_protocol=action_protocol or VectorDecompositionActionProtocol()
        )

    def register_subordinates(self, subordinates):
        """Register subordinates for action decomposition."""
        self.action_protocol.register_subordinates(subordinates)
```

**VectorDecompositionActionProtocol decomposition strategy:**
- If coordinator action is already a dict: use it directly
- If coordinator action is a vector: split based on subordinate action dimensions
- If coordinator action is None: return None for all subordinates

### **Protocol Usage in Agent**

```python
# In Agent.coordinate() (called from compute_action):
def coordinate(self, obs, action):
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

    # Send coordination messages (not used yet in default protocols)
    for sub_id, message in messages.items():
        self.send_info(broker=self._message_broker, recipient_id=sub_id, info=message)
```

---

## **Observation & Action Data Flow**

### **Observation Flow (Both Modes)**

```
Agent needs observation
│
├─> CTDE Mode: Direct call
│   └─> obs = proxy.get_observation(agent_id, protocol)
│       ├─> global_state = get_global_states(agent_id, protocol)
│       │   └─> For each agent: state.observed_by(requestor_id, level)
│       │   └─> Includes "env_context" from global cache if available
│       ├─> local_state = get_local_state(agent_id, protocol, include_subordinate_rewards=False)
│       │   └─> state.observed_by(agent_id, level)
│       └─> return Observation(local, global_info, timestamp)
│
├─> Event-Driven Mode: Via message
│   ├─> Request: schedule_message_delivery({MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: protocol})
│   └─> Response: {"get_obs_response": {"body": {"obs": obs_dict, "local_state": local_state}}}
│       ├─> obs = Observation.from_dict(response['body']['obs'])
│       └─> self.sync_state_from_observed(response['body']['local_state'])
│
│   KEY DESIGN PRINCIPLE: When agent asks for obs, proxy gives BOTH obs AND local_state
│   This enables state syncing alongside observation delivery.
│
└─> Proxy.get_observation(sender_id, protocol)
    ├─> global_state = get_global_states(sender_id, protocol)
    │   └─> For each agent: state.observed_by(sender_id, requestor_level)
    ├─> local_state = get_local_state(sender_id, protocol, include_subordinate_rewards=False)
    └─> return Observation(local=local_state, global_info=global_state, timestamp=self._timestep)
```

### **Observation Array Interface**

```python
obs = Observation(local={"voltage": np.array([1.02])}, global_info={"freq": 60.0})

# Automatic conversion for policies:
policy.forward(observation=obs)  # __array__() converts to np.ndarray

# Array-like properties:
obs.shape    # -> (2,)
obs.dtype    # -> float32
len(obs)     # -> 2
obs[0]       # -> 1.02
obs[:1]      # -> array([1.02])

# Explicit vectorization:
vec = obs.vector()  # -> array([1.02, 60.0], dtype=float32)
```

### **Action Flow**

```
CTDE Mode:
  agent.act(actions, proxy)
  ├─> layer_actions(actions) -> Hierarchical structure
  ├─> handle_self_action(action)
  │   ├─> set_action(action) -> self.action.set_values(action)
  │   └─> apply_action() -> Updates self.state immediately
  ├─> handle_subordinate_actions(subordinate_actions)
  │   └─> For each: subordinate.execute(action, proxy)
  └─> proxy.set_local_state(self.state)

Event-Driven Mode:
  agent.tick(scheduler)
  ├─> _check_for_upstream_action() via broker (caches in self._upstream_action)
  ├─> Both FieldAgent and CoordinatorAgent:
  │   └─> Always request obs from proxy first (for state sync)
  │       └─> schedule_message_delivery(proxy, {get_info: "obs"})
  └─> No action and no upstream: warning printed

  [Later, on obs response MESSAGE_DELIVERY:]
  agent.message_delivery_handler() on "get_obs_response"
  ├─> sync_state_from_observed(local_state)
  └─> compute_action(obs, scheduler)
      ├─> Uses cached upstream action if present (priority)
      ├─> Else uses policy.forward(obs)
      ├─> CoordinatorAgent: coordinate() -> send_subordinate_action()
      └─> FieldAgent: set_action + schedule_action_effect(delay)

  [Later, at ACTION_EFFECT event (FieldAgent only):]
  FieldAgent.action_effect_handler()
  ├─> apply_action() -> Updates self.state after delay
  └─> schedule message to update proxy with serialized state
```

**Key Differences:**
- **CTDE**: Actions apply immediately, state updated in proxy synchronously
- **Event-Driven**: Both FieldAgent and CoordinatorAgent always request obs first for state sync, then apply actions via compute_action
- **Upstream action priority**: Cached upstream actions always take priority over policy in compute_action()
- **CoordinatorAgent**: Additionally runs protocol.coordinate() to decompose actions for subordinates

---

## **Proxy State Management**

### **State Cache Structure**

```python
proxy.state_cache = {
    # Per-agent states (State objects with Dict-based features!)
    "agents": {
        "system_agent": SystemAgentState(
            owner_id="system_agent",
            owner_level=3,
            features={"SystemFeature": Feature(...)}  # Dict, not List!
        ),
        "coordinator_1": CoordinatorAgentState(...),
        "field_1": FieldAgentState(...),
        "field_2": FieldAgentState(...),
    },

    # Global state (environment-wide data)
    "global": {
        "agent_states": {agent_id: state_obj, ...},  # Compiled references
        "env_context": {...}  # External data (price, solar, wind profiles)
    }
}

# Agent levels for visibility checks
proxy._agent_levels = {
    "field_1": 1,
    "field_2": 1,
    "coordinator_1": 2,
    "system_agent": 3
}

# Agent upstream tracking
proxy._agent_upstreams = {
    "field_1": "coordinator_1",
    "field_2": "coordinator_1",
    "coordinator_1": "system_agent",
    "system_agent": None
}

# Tick results per agent (for reward tracking in event-driven mode)
proxy._tick_results = {
    "field_1": {"reward": 0.5, "terminated": False, "truncated": False, "info": {}},
    ...
}
```

### **State Operations**

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `set_local_state(aid, state)` | State object | None | Stores object directly |
| `get_local_state(aid, protocol, include_sub_rewards)` | agent_id, protocol, bool | Dict[str, np.ndarray] | Returns filtered vectors + optional subordinate rewards |
| `get_global_states(aid, protocol)` | agent_id, protocol | Dict[aid, Dict] | Returns all filtered states + env_context |
| `get_observation(aid, protocol)` | agent_id, protocol | Observation | Combines local + global (excludes subordinate_rewards) |
| `set_global_state(dict)` | global_dict | None | Updates global state cache via .update() |
| `get_serialized_agent_states()` | None | Dict[aid, Dict] | Serializes all states for message passing |

### **Visibility Filtering**

```python
def get_local_state(self, sender_id, protocol, include_subordinate_rewards=True):
    """Get agent's state filtered by visibility rules."""
    state_obj = self.state_cache["agents"][sender_id]
    requestor_level = self._agent_levels.get(sender_id, 1)
    local_state = state_obj.observed_by(sender_id, requestor_level)

    # Include subordinate rewards for parent agents (reward computation only)
    if include_subordinate_rewards:
        subordinate_rewards = self._get_subordinate_rewards(sender_id)
        if subordinate_rewards:
            local_state["subordinate_rewards"] = subordinate_rewards

    return local_state

def get_global_states(self, sender_id, protocol):
    """Get all agents' states filtered by visibility."""
    global_filtered = {}
    requestor_level = self._agent_levels.get(sender_id, 1)

    for agent_id, state_obj in self.state_cache.get("agents", {}).items():
        if agent_id == sender_id:
            continue  # Exclude self
        observable = state_obj.observed_by(sender_id, requestor_level)
        if observable:
            global_filtered[agent_id] = observable

    # Include env_context if available
    global_cache = self.state_cache.get("global", {})
    if "env_context" in global_cache:
        global_filtered["env_context"] = global_cache["env_context"]

    return global_filtered
```

---

## **Handler Registration & Event Processing**

### **Handler Registration**

```python
class Agent:
    _event_handler_funcs: Dict[EventType, Callable] = {}

    def __init_subclass__(cls, **kwargs):
        """Each subclass gets its own copy and auto-registers @handler methods."""
        super().__init_subclass__(**kwargs)
        cls._event_handler_funcs = cls._event_handler_funcs.copy()
        for name in dir(cls):
            if name.startswith('__'):
                continue
            try:
                attr = getattr(cls, name)
                if callable(attr) and hasattr(attr, '_handler_event_type'):
                    cls._event_handler_funcs[attr._handler_event_type] = attr
            except AttributeError:
                pass

    class handler:
        """Decorator for registering event handlers."""
        def __init__(self, event_type_str: str):
            self.event_type = EVENT_TYPE_FROM_STRING[event_type_str]

        def __call__(self, func):
            func._handler_event_type = self.event_type
            return func

    def get_handlers(self) -> Dict[EventType, Callable]:
        """Return handlers bound to this agent instance."""
        bound_handlers = {}
        for event_type, func in self._event_handler_funcs.items():
            bound_handlers[event_type] = lambda e, s, f=func: f(self, e, s)
        return bound_handlers
```

### **Handler Implementation Summary**

| Agent | agent_tick | action_effect | message_delivery | simulation |
|-------|-----------|--------------|-----------------|-----------|
| **SystemAgent** | tick() | No-op | obs/global_state/local_state/completion handlers | Request global state |
| **CoordinatorAgent** | tick() | No-op | obs/local_state/completion handlers | N/A |
| **FieldAgent** | tick() | apply_action() + send state | obs/local_state/completion handlers | N/A |
| **Proxy** | N/A | N/A | get_info/set_state/set_tick_result handlers | N/A |

### **Event Processing Loop**

```python
class EventScheduler:
    def run_until(self, t_end: float, max_events: Optional[int] = None) -> Iterable[Event]:
        count = 0
        while self.event_queue:
            if self.peek().timestamp > t_end:
                break
            if max_events is not None and count >= max_events:
                break
            if event := self.process_next():
                count += 1
                yield event

    def process_next(self) -> Event:
        event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp
        handler = self.get_handler(event.event_type, event.agent_id)
        handler(event, self)
        return event
```

---

## **Complete Execution Examples**

### **CTDE Training Step**

```python
# Step 0: Reset
obs, info = env.reset()  # All agents initialized, proxy has states

# Step 1: RL algorithm provides actions
actions = {
    "field_1": Action(c=[0.3]),
    "field_2": Action(c=[-0.2]),
}

# Step 2: Environment executes
obs, rewards, terminated, truncated, info = env.step(actions)

# Internal execution:
# 1. SystemAgent.execute(actions, proxy)
#    a. pre_step() hook
#    b. Sync state from proxy
#    c. layer_actions() -> hierarchical structure
#    d. act() -> handle_self_action() + handle_subordinate_actions()
#       - Each agent: sync state, set_action(), apply_action(), proxy.set_local_state()
#    e. simulate() -> Physics update via proxy.get_global_states()
#    f. proxy.set_global_state() -> Broadcast updated state
#    g. observe() -> Collect observations (with visibility filtering)
#    h. compute_rewards() -> Compute from filtered states
# 2. proxy.set_step_result() -> Cache results
# 3. proxy.get_step_results() -> Vectorize and return

# Step 3: RL algorithm trains
# obs: Dict[AgentID, np.ndarray] ready for neural network input
```

### **Event-Driven Episode**

```python
# Setup
obs, info = env.reset()
episode_analyzer = EpisodeAnalyzer(verbose=False, track_data=False)

# Run simulation
episode_stats = env.run_event_driven(episode_analyzer, t_end=100.0)

# Internal execution:
# - Scheduler starts with AGENT_TICK(system_agent) at t=0
# - System agent cascades ticks to subordinates
# - Actions passed via broker from parents to children
# - FieldAgents apply actions with realistic delays
# - States updated via messages through proxy
# - State completion triggers reward computation cascade from parent
# - EpisodeAnalyzer extracts results from event stream

# Retrieve results
print(episode_stats.summary())
print(f"Events: {episode_stats.num_events}")
print(f"Observations: {episode_stats.observation_count}")
print(f"Actions: {episode_stats.action_result_count}")
print(episode_analyzer.get_reward_history())
```

---

## **Comparison: CTDE vs Event-Driven**

| Aspect | CTDE Training | Event-Driven Testing |
|--------|---------------|----------------------|
| **Execution** | Synchronous (step-based) | Asynchronous (event-based) |
| **Timing** | Instantaneous | Realistic delays (msg_delay, act_delay) |
| **Action Application** | Immediate in `act()` | Delayed via ACTION_EFFECT |
| **Action Passing** | Via `layer_actions()` dict | Via broker messages |
| **Upstream Action Handling** | Integrated in handle_self_action | FieldAgent: direct set; Coordinator: compute_action |
| **State Updates** | Synchronous to proxy | Asynchronous via messages |
| **State Syncing** | sync_state_from_observed in execute() | sync_state_from_observed in message handler |
| **Observation** | Direct proxy.get_observation() | Message-based (obs + local_state bundled) |
| **Reward Computation** | Direct via compute_rewards() | Parent-initiated cascade after state completion |
| **Result Retrieval** | `proxy.get_step_results()` | EpisodeAnalyzer from event stream |
| **Hierarchy Coordination** | Sequential function calls | Cascading tick events |
| **Use Case** | RL training with full observability | Deployment testing with latency |

---

## **Critical Design Patterns**

### **1. Two-Phase State Updates**

**CTDE Mode:**
```python
# Phase 1: Act
agent.act(actions, proxy)
└─> apply_action() -> Updates state
└─> proxy.set_local_state(state) -> Synchronous

# Phase 2: Simulate
simulate(global_state) -> Physics update
proxy.set_global_state(updated_global_state)
```

**Event-Driven Mode:**
```python
# Phase 1: Action computed
compute_action(obs) -> Set action
schedule_action_effect(delay)

# Phase 2: Action applied (after delay)
action_effect_handler()
└─> apply_action() -> Updates state
└─> schedule message to update proxy -> Asynchronous
```

### **2. State Sync via sync_state_from_observed()**

```python
def sync_state_from_observed(self, observed_state):
    """Reconcile internal state with proxy state (e.g., after simulation)."""
    for feature_name, feature_data in observed_state.items():
        if feature_name not in self.state.features:  # Dict lookup: O(1)
            continue
        feature = self.state.features[feature_name]
        if isinstance(feature_data, dict):
            feature.set_values(**feature_data)    # Direct field update
        elif isinstance(feature_data, np.ndarray):
            field_names = feature.names()          # Reconstruct from vector
            updates = {name: float(val) for name, val in zip(field_names, feature_data)}
            feature.set_values(**updates)
```

### **3. Reward Computation Cascade (Event-Driven)**

```
State completion message received by parent agent
│
├─> Parent initiates reward computation for EACH subordinate:
│   └─> schedule message: {MSG_GET_INFO: INFO_TYPE_LOCAL_STATE}
│       └─> Proxy responds with filtered local state
│           └─> Agent computes tick_result (reward, terminated, truncated, info)
│               └─> schedule message: {MSG_SET_TICK_RESULT: ...}
│
└─> Parent also requests own local state for self-reward computation
```

### **4. Visibility-Based State Access**

```python
# All state access goes through observed_by()
local_state = state.observed_by(requestor_id, requestor_level)

# Features filtered by visibility rules:
# - "public": All agents can see
# - "owner": Only owning agent can see
# - "upper_level": Only one level above can see
# - "system": Only system-level (L3) can see
```

### **5. Message-Action Duality**

```python
# CTDE: Direct action passing via layer_actions
subordinate.execute(layered_action, proxy)

# Event-Driven: Message-based action passing via broker
send_subordinate_action(subordinate_id, action)
└─> broker.publish(action_channel, Message(action))
```

---

## **Summary**

### **CTDE Training Mode**
- Proper hierarchical action distribution via `layer_actions()`
- State syncing via `sync_state_from_observed()` before each action
- Synchronized state management via proxy with State objects (Dict-based features)
- Observations auto-vectorized for RL algorithms
- Visibility filtering via `state.observed_by()`
- Full observability for centralized training

### **Event-Driven Testing Mode**
- Asynchronous execution with realistic timing (configurable delays + jitter)
- Hierarchical tick cascade (system -> coordinators -> field agents)
- Action passing via message broker with priority rules
- Observation responses bundle both obs and local_state for state syncing
- Delayed action effects (FieldAgent only)
- Parent-initiated reward computation cascade after state completion
- Results collected via EpisodeAnalyzer from event stream
- Tests deployment scenario with communication delays

**Both modes share the same agent hierarchy, state management infrastructure, protocol system, and action priority rules, ensuring consistency between training and testing.**

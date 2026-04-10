# HERON Complete Data Flow Documentation

This document provides a comprehensive analysis of how data flows through the HERON framework in both CTDE Training and Event-Driven Testing modes.

---

## **Table of Contents**

1. [Core Data Types](#core-data-types)
2. [State Data Flow](#state-data-flow)
3. [Action Data Flow](#action-data-flow)
4. [Action Passing (Hierarchical Coordination)](#action-passing-hierarchical-coordination)
5. [Observation Data Flow](#observation-data-flow)
6. [Feature Data Flow](#feature-data-flow)
7. [Message Passing Architecture](#message-passing-architecture)
8. [CTDE Training Mode - Complete Flow](#ctde-training-mode---complete-flow)
9. [Event-Driven Testing Mode - Complete Flow](#event-driven-testing-mode---complete-flow)
10. [Proxy State Cache Architecture](#proxyagent-state-cache-architecture)
11. [Protocol-Based Coordination](#protocol-based-coordination)

---

## **Core Data Types**

### **Type Transformation Rules**

| Component | Internal Representation | Storage in Proxy | Message Passing |
|-----------|------------------------|------------------|-----------------|
| **State** | State object (features as Dict[str, Feature]) | **State object** | Dict with metadata via `to_dict()` |
| **Action** | Action object | Not stored in proxy | Action object (direct) or Dict (messages) |
| **Observation** | Observation object | Not stored (computed on-demand) | Dict via `obs.to_dict()` |
| **Feature** | Feature object | Part of State.features Dict | Dict via `feature.to_dict()` |
| **Message** | Event payload Dict | Broker queues | Serialized payload |

### **Key Principles**

**1. State.features is a Dict[str, Feature], keyed by feature_name:**
- O(1) lookup by feature name: `state.features["BatteryChargeFeature"]`
- Features maintain class identity (Feature instances)
- Direct access to `state.observed_by()` for visibility filtering

**2. Proxy stores State objects directly** - This enables feature-level visibility filtering:
- State objects maintained with full type information
- Serialization needed only at message boundaries (State <-> Dict)

**3. Observation responses bundle obs + local_state:**
- When agent requests observation, proxy sends both
- Enables state syncing alongside observation delivery

---

## **State Data Flow**

### **State Object Structure**

```python
# Agent creates State with features as Dict
state = FieldAgentState(
    owner_id="battery_1",
    owner_level=1,
    features={
        "BatteryChargeFeature": BatteryChargeFeature(soc=0.5, capacity=100.0)
    }
)

# State object attributes:
state.owner_id = "battery_1"
state.owner_level = 1
state.features = {"BatteryChargeFeature": Feature}  # Dict keyed by name!
```

### **State Operations**

#### **1. Init State (Initialization)**

```
Agent.__init__()
├─> self.state = self.init_state(features=features)
│   ├─> FieldAgentState(
│   │       owner_id=self.agent_id,
│   │       owner_level=FIELD_LEVEL,
│   │       features={f.feature_name: f for f in features}  # List -> Dict conversion!
│   │   )
│   └─> Returns State object

Example:
    # Constructor receives features as List[Feature]
    features = [BatteryChargeFeature(soc=0.5, capacity=100.0)]
    # init_state converts to Dict:
    state.features = {"BatteryChargeFeature": BatteryChargeFeature(soc=0.5, capacity=100.0)}
```

**Data Outflow:**
- `self.state` -> State object with features Dict

---

#### **2. Set State (Storage in Proxy)**

**CTDE Mode:**
```
Agent updates state -> Send to proxy

agent.apply_action()
├─> Updates self.state.features["BatteryChargeFeature"].soc = new_value
└─> State object modified in-place

proxy.set_local_state(agent_id, state)
├─> Input: agent_id="battery_1", state=FieldAgentState(...)
├─> Storage: state_cache["agents"]["battery_1"] = state  # State object!
└─> Stored as STATE OBJECT with full type information!
```

**Event-Driven Mode:**
```
Agent updates state -> Serialize -> Send message -> Proxy stores

action_effect_handler()
├─> apply_action() -> Updates self.state
├─> Serialize: self.state.to_dict(include_metadata=True)
│   └─> Returns: {
│           "_owner_id": "battery_1",
│           "_owner_level": 1,
│           "_state_type": "FieldAgentState",
│           "features": {"BatteryChargeFeature": {"soc": 0.53, "capacity": 100.0}}
│       }
├─> Send message: {"set_state": "local", "body": serialized_dict}
└─> Proxy receives message

proxy.message_delivery_handler()
├─> Extract: state_dict from body
├─> Reconstruct: state = State.from_dict(state_dict)
└─> Call: set_local_state(state.owner_id, state)
    └─> Storage: state_cache["agents"]["battery_1"] = state  # State object!
```

**Data Transformations:**
```
State object -> .to_dict() -> Message -> State.from_dict() -> State object
                          ↓
            proxy.state_cache["agents"][agent_id]
```

---

#### **3. Get State (Retrieval from Proxy)**

**CTDE Mode:**
```
Agent requests own state from proxy

proxy.get_local_state(sender_id, protocol, include_subordinate_rewards=True)
├─> Input: sender_id="battery_1"
├─> Retrieval: state_obj = state_cache["agents"]["battery_1"]  # State object!
├─> Apply visibility filtering: state_obj.observed_by(sender_id, requestor_level)
├─> Include subordinate rewards (if parent agent):
│   └─> subordinate_rewards = _get_subordinate_rewards(agent_id)
└─> Return: {"BatteryChargeFeature": np.array([0.53, 100.0])}  # Feature vectors!
```

**Event-Driven Mode:**
```
Agent sends message requesting state -> Proxy responds

Agent:
├─> schedule_message_delivery(
│       message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol}
│   )
└─> Sent to proxy

Proxy.message_delivery_handler():
├─> Receives request
├─> local_state = self.get_local_state(sender_id, protocol)
│   └─> Returns filtered feature vectors + optional subordinate rewards
├─> Send response: {"get_local_state_response": {"body": local_state}}
└─> Agent receives dict in message
```

**Data Outflow:**
```
state_cache["agents"][agent_id] -> observed_by() -> Filtered vectors -> Agent
```

**Usage Example (With Visibility Filtering):**
```python
# Agent receives feature vectors from proxy (after visibility filtering)
local_state = proxy.get_local_state(self.agent_id, self.protocol)
# local_state = {"BatteryChargeFeature": np.array([0.53, 100.0])}  # Numpy arrays!

# Extract values from feature vector
feature_vec = local_state["BatteryChargeFeature"]  # np.array([0.53, 100.0])
soc = feature_vec[0]  # First element is SOC
reward = soc  # Use filtered state, not self.state!
```

---

#### **4. Sync State (Reconciliation)**

```python
def sync_state_from_observed(self, observed_state):
    """Reconcile internal state from proxy data (e.g., after simulation)."""
    for feature_name, feature_data in observed_state.items():
        if feature_name not in self.state.features:  # Dict lookup O(1)
            continue
        feature = self.state.features[feature_name]
        if isinstance(feature_data, dict):
            feature.set_values(**feature_data)           # Direct field update
        elif isinstance(feature_data, np.ndarray):
            field_names = feature.names()
            updates = {name: float(val) for name, val in zip(field_names, feature_data)}
            feature.set_values(**updates)                # Reconstruct from vector
```

**When is sync used?**
- CTDE: In `execute()` before acting (picks up simulation changes)
- Event-Driven: In `message_delivery_handler()` when receiving obs or local_state response

---

### **State Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (State Objects)                                      │
├─────────────────────────────────────────────────────────────────┤
│ init_state(features) -> self.state = FieldAgentState(           │
│     owner_id="battery_1",                                       │
│     features={"BatteryChargeFeature": BatteryChargeFeature(...)}│
│ )                                                               │
│                                                                 │
│ apply_action() -> self.state.features["BatteryChargeFeature"]   │
│                   .soc = new_value                              │
│                                                                 │
│ Send to proxy: proxy.set_local_state(aid, state)  # State obj!  │
└─────────────────────────────────────────────────────────────────┘
                            ↓ (No conversion needed!)
┌─────────────────────────────────────────────────────────────────┐
│ Proxy Layer (State Object Storage)                              │
├─────────────────────────────────────────────────────────────────┤
│ state_cache["agents"]["battery_1"] = FieldAgentState(           │
│     features={"BatteryChargeFeature": BatteryChargeFeature(...)}│
│ )                                                               │
│                                                                 │
│ Stores FULL State objects with Feature instances!        │
└─────────────────────────────────────────────────────────────────┘
                            ↓ get_local_state() + visibility filtering
┌─────────────────────────────────────────────────────────────────┐
│ Visibility Filtering Layer (state.observed_by())                │
├─────────────────────────────────────────────────────────────────┤
│ state_obj.observed_by(requestor_id, requestor_level)            │
│     ↓ Feature-level filtering                                   │
│ Returns: {"BatteryChargeFeature": np.array([0.53, 100.0])}      │
│          └─> Only visible features, as numpy arrays!            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (Filtered Vector Usage + State Sync)                │
├─────────────────────────────────────────────────────────────────┤
│ # For reward computation:                                       │
│ local_state = proxy.get_local_state(aid, protocol)              │
│ feature_vec = local_state["BatteryChargeFeature"]  # array      │
│ soc = feature_vec[0]  # Extract from vector                     │
│                                                                 │
│ # For state syncing:                                            │
│ sync_state_from_observed(local_state) -> Reconcile features     │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Action Data Flow**

### **Action Object Structure**

```python
action = Action()
action.set_specs(
    dim_c=1,  # 1 continuous action
    range=(np.array([-1.0]), np.array([1.0]))
)
action.set_values(np.array([0.3]))

# Action object attributes:
action.c = np.array([0.3])      # Continuous actions
action.d = np.array([])         # Discrete actions (empty)
action.dim_c = 1
action.dim_d = 0
action.range = ([-1.0], [1.0])
```

### **Action Operations**

#### **1. Init Action (Initialization)**

```
Agent.__init__()
├─> self.action = self.init_action(features=features)
│   └─> Return Action()  # Empty action by default
│
└─> Subclasses may override to set specs based on features

Example (default):
    self.action = Action()  # No specs set yet
    # Specs configured later via set_action() or policy output
```

**Data Outflow:**
- `self.action` -> Action object (possibly unconfigured)

---

#### **2. Set Action (Policy Decision or Upstream)**

**CTDE Mode - From External Actions Dict:**
```
env.step(actions)
├─> actions = {"battery_1": action_object, ...}
├─> SystemAgent.execute(actions, proxy)
    ├─> layered = layer_actions(actions)
    │   └─> {"self": actions.get("system_agent"),
    │        "subordinates": {"battery_1": {...}, ...}}
    │
    └─> handle_self_action(layered["self"], proxy)
        ├─> set_action(action)
        │   └─> self.action.set_values(action)
        │       ├─> If action is Action object: Copy c and d values
        │       └─> If action is dict: Extract "c" and "d" keys
        └─> Action object updated!
```

**CTDE Mode - From Policy:**
```
handle_self_action(None, proxy)  # No external action
├─> obs = proxy.get_observation(sender_id=self.agent_id)
│   └─> Returns Observation object
├─> action = policy.forward(observation=obs)
│   ├─> obs.__array__() converts to np.ndarray automatically
│   └─> Returns Action object
└─> set_action(action)
    └─> self.action.set_values(action)
```

**Event-Driven Mode (FieldAgent and CoordinatorAgent):**
```
agent.tick()
├─> _check_for_upstream_action()
│   └─> self._upstream_action = broker.consume(action_channel)  # Cached
└─> Always request obs from proxy (for state sync)
    └─> schedule_message_delivery(proxy, {get_info: "obs"})

[Later] message_delivery_handler() - "get_obs_response"
├─> body = message["get_obs_response"]["body"]
├─> obs_dict = body["obs"]
├─> local_state = body["local_state"]
├─> obs = Observation.from_dict(obs_dict)
├─> sync_state_from_observed(local_state)  # Sync first!
└─> compute_action(obs, scheduler)
    ├─> If self._upstream_action: use it (priority!), clear after use
    ├─> Elif has policy: action = policy.forward(observation=obs)
    ├─> If protocol: coordinate() -> send subordinate actions
    ├─> set_action(action)
    └─> schedule_action_effect(delay=act_delay)
```

**Data Transformations:**
```
Policy -> Action object -> set_values() -> self.action updated
                                        ↓
                        No storage in proxy!
                        Actions exist only in agents
```

---

#### **3. Apply Action (State Update)**

```
Agent.apply_action()
└─> set_state()
    ├─> Read: current_soc = self.state.features["BatteryChargeFeature"].soc
    ├─> Read: action_value = self.action.c[0]
    ├─> Compute: new_soc = current_soc + action_value * 0.01
    └─> Update: self.state.features["BatteryChargeFeature"].set_values(soc=new_soc)
        └─> State object modified in-place!
```

**Data Flow:**
```
self.action (Action object) -> Extract values -> Update self.state (State object)
```

---

### **Action Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Policy Layer                                                    │
├─────────────────────────────────────────────────────────────────┤
│ policy.forward(observation) -> Returns Action object            │
│     action = Action()                                           │
│     action.set_specs(dim_c=1, range=([-1,1]))                   │
│     action.set_values(np.array([0.3]))                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Training Loop / Environment Layer                               │
├─────────────────────────────────────────────────────────────────┤
│ actions[agent_id] = action  # Action object                     │
│ env.step(actions)                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (Execution)                                         │
├─────────────────────────────────────────────────────────────────┤
│ set_action(action) -> self.action.set_values(action)            │
│                                                                 │
│ apply_action() -> Read self.action.c[0]                         │
│     -> Update self.state.features["FeatureName"].field = value  │
└─────────────────────────────────────────────────────────────────┘
```

**Important: Actions are NEVER stored in proxy!** They exist only within agents during the act() phase.

---

## **Action Passing (Hierarchical Coordination)**

### **Overview**

Action passing enables hierarchical coordination where parent agents can send actions to their subordinates. This is essential for centralized control strategies where a coordinator or system agent makes decisions for lower-level agents.

### **Action Passing Modes**

#### **1. Self-Action (Policy-Driven)**
```
Agent computes its own action via policy:

agent.tick() or agent.act()
├─> obs = proxy.get_observation(self.agent_id)
├─> action = self.policy.forward(obs)
└─> self.set_action(action)
```

#### **2. Upstream Action (Parent-Driven)**
```
Parent sends action to subordinate:

Parent (Coordinator/System):
├─> Compute subordinate actions via protocol.coordinate()
├─> self.send_subordinate_action(sub_id, action)
│   └─> broker.publish(action_channel, Message(action))
└─> Actions queued for subordinates

Subordinate receives:
├─> _check_for_upstream_action() [at start of tick]
│   └─> self._upstream_action = broker.consume(action_channel)[-1]
│
├─> Both FieldAgent and CoordinatorAgent always request obs from proxy first
│   └─> schedule_message_delivery(proxy, {get_info: "obs"})
│
└─> On obs response: compute_action(obs, scheduler)
    ├─> If self._upstream_action: use it (priority!)
    ├─> Elif has policy: policy.forward(obs)
    ├─> CoordinatorAgent: coordinate() -> decompose for subordinates
    └─> FieldAgent: set_action + schedule_action_effect
```

### **Action Passing Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│ SystemAgent (Level 3)                                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Receive global observation                                   │
│ 2. Compute system-level action via policy                       │
│ 3. Protocol.coordinate() decomposes into subordinate actions    │
│ 4. send_subordinate_action() to each coordinator                │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoordinatorAgent (Level 2)                                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. _check_for_upstream_action() -> cache from broker            │
│ 2. Always request obs from proxy (for state sync)               │
│ 3. On obs response: compute_action(obs, scheduler)              │
│    - Uses cached upstream action if present (priority)          │
│    - Else uses own policy                                       │
│ 4. Protocol.coordinate() decomposes into field actions          │
│ 5. send_subordinate_action() to each field agent                │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ FieldAgent (Level 1)                                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. _check_for_upstream_action() -> cache from broker            │
│ 2. Always request obs from proxy (for state sync)               │
│ 3. On obs response: compute_action(obs, scheduler)              │
│    - Uses cached upstream action if present (priority)          │
│    - Else uses own policy                                       │
│ 4. set_action(action) + schedule_action_effect(delay)           │
│ 5. [ACTION_EFFECT event]: apply_action() -> Update state        │
└─────────────────────────────────────────────────────────────────┘
```

### **Message Broker Integration**

```python
# Channel naming convention
action_channel = ChannelManager.action_channel(
    upstream_id="coordinator_1",
    node_id="field_1",
    env_id="default"
)
# Result: "env_default__action__coordinator_1_to_field_1"

# Sending action (Parent)
def send_subordinate_action(self, recipient_id, action):
    self.send_action(
        broker=self._message_broker,
        recipient_id=recipient_id,
        action=action,
    )

# Inside send_action:
def send_action(self, broker, recipient_id, action):
    channel = ChannelManager.action_channel(
        self.agent_id, recipient_id, self.env_id or "default"
    )
    self._publish(
        broker=broker, channel=channel,
        payload={"action": action},
        recipient_id=recipient_id,
        message_type="ACTION",
    )

# Receiving action (Subordinate) - at tick start
def _check_for_upstream_action(self):
    if not self.upstream_id or not self._message_broker:
        self._upstream_action = None
        return
    actions = self.receive_upstream_actions(sender_id=self.upstream_id, clear=True)
    self._upstream_action = actions[-1] if actions else None
```

### **CTDE Mode Action Passing**

```
env.step(actions)
│
└─> SystemAgent.execute(actions, proxy)
    │
    ├─> actions = self.layer_actions(actions)
    │   # Hierarchical structure:
    │   # {
    │   #   "self": actions.get("system_agent"),
    │   #   "subordinates": {
    │   #     "coord_1": {
    │   #       "self": actions.get("coord_1"),
    │   #       "subordinates": {"field_1": {...}, "field_2": {...}}
    │   #     }
    │   #   }
    │   # }
    │
    ├─> self.act(actions, proxy)
    │   ├─> handle_self_action(actions["self"], proxy)
    │   └─> handle_subordinate_actions(actions["subordinates"], proxy)
    │       └─> For each coordinator:
    │           └─> coordinator.execute(coord_actions, proxy)
    │               ├─> Sync state from proxy
    │               └─> coordinator.act(coord_actions, proxy)
    │                   └─> For each field:
    │                       └─> field.execute(field_actions, proxy)
    │                           ├─> Sync state from proxy
    │                           ├─> handle_self_action(action, proxy)
    │                           │   ├─> set_action(action)
    │                           │   ├─> apply_action()
    │                           │   └─> proxy.set_local_state(state)
    │                           └─> Done (no subordinates)
```

### **Event-Driven Mode Action Passing**

```
t=0.0: AGENT_TICK(system_agent)
├─> SystemAgent.tick()
│   ├─> Schedule subordinate ticks
│   ├─> If has policy:
│   │   └─> Request obs -> [later] compute_action(obs, scheduler)
│   │       ├─> coordinate() -> subordinate actions via protocol
│   │       └─> send_subordinate_action(coord_id, action)
│   └─> schedule_simulation()

t=coord_tick: AGENT_TICK(coordinator_1)
├─> _check_for_upstream_action() -> consume from broker (caches in self._upstream_action)
├─> Schedule subordinate ticks
└─> Always request obs from proxy (for state sync):
    └─> schedule_message_delivery(proxy, {get_info: "obs"})

t=coord_tick+2×msg_delay: MESSAGE_DELIVERY(coordinator_1) - obs response
├─> sync_state_from_observed(local_state)  # State sync first!
└─> compute_action(obs, scheduler)
    ├─> If self._upstream_action: use it (priority!)
    ├─> Elif has policy: policy.forward(obs)
    ├─> coordinate() -> field actions via protocol
    └─> send_subordinate_action(field_id, action) for each

t=field_tick: AGENT_TICK(field_1)
├─> _check_for_upstream_action() -> consume from broker (caches in self._upstream_action)
├─> Always request obs from proxy (for state sync)
│   └─> schedule_message_delivery(proxy, {get_info: "obs"})

t=field_tick+2×msg_delay: MESSAGE_DELIVERY(field_1) - obs response
├─> sync_state_from_observed(local_state)  # State sync first!
└─> compute_action(obs, scheduler)
    ├─> If self._upstream_action: use it (priority!)
    ├─> Elif has policy: policy.forward(obs)
    ├─> set_action(action)
    └─> schedule_action_effect(delay=act_delay)

t=field_tick+2×msg_delay+act_delay: ACTION_EFFECT(field_1)
├─> apply_action() -> Update self.state
└─> schedule message to proxy with serialized state
```

---

## **Observation Data Flow**

### **Observation Object Structure**

```python
obs = Observation(
    local={"BatteryChargeFeature": np.array([0.53, 100.0])},
    global_info={"agent_states": {...}},
    timestamp=0.0
)

# Observation attributes:
obs.local = dict        # Agent-specific state (filtered vectors)
obs.global_info = dict  # Global/shared state
obs.timestamp = float   # Current time
```

### **Observation Operations**

#### **1. Build Observation (Construction from Proxy State)**

```
proxy.get_observation(sender_id, protocol)
├─> global_state = get_global_states(sender_id, protocol)
│   ├─> For each agent in state_cache["agents"] (excluding sender):
│   │   ├─> state_obj = state_cache["agents"][agent_id]
│   │   ├─> filtered = state_obj.observed_by(sender_id, requestor_level)
│   │   └─> Add to global_state if visible
│   │
│   ├─> Include "env_context" from global cache if available
│   └─> Returns: Dict of filtered feature vectors + env_context
│
├─> local_state = get_local_state(sender_id, protocol, include_subordinate_rewards=False)
│   ├─> state_obj = state_cache["agents"][sender_id]
│   ├─> filtered = state_obj.observed_by(sender_id, requestor_level)
│   └─> Returns: Dict of filtered feature vectors (NO subordinate rewards for obs)
│
└─> return Observation(
        local=local_state,        # Filtered vectors!
        global_info=global_state, # Filtered vectors + env_context!
        timestamp=self._timestep
    )
```

**Key Insight:** Observation contains **filtered numpy arrays**, not raw State objects!

**Subordinate Rewards:** The `include_subordinate_rewards=False` parameter ensures observations
used for action computation do not include subordinate reward data (which is only relevant for
reward computation via `get_local_state()` with default `include_subordinate_rewards=True`).

---

#### **2. Observation Delivery in Event-Driven Mode**

```
Agent requests obs -> Proxy bundles obs + local_state together

Proxy.message_delivery_handler() on get_info request:
├─> obs = get_observation(sender_id, protocol)  # Observation object
├─> local_state = get_local_state(sender_id, protocol)  # Feature vectors
│
├─> Bundle both for response:
│   info_data = {
│       "obs": obs.to_dict(),         # Serialized Observation
│       "local_state": local_state     # Feature vectors for state sync
│   }
│
└─> schedule_message_delivery(
        message={"get_obs_response": {"body": info_data}},
        delay=msg_delay
    )

Agent.message_delivery_handler() on get_obs_response:
├─> body = message["get_obs_response"]["body"]
├─> obs = Observation.from_dict(body["obs"])        # Reconstruct Observation
├─> self.sync_state_from_observed(body["local_state"])  # Sync state!
└─> compute_action(obs, scheduler)
```

**Design Principle:** When agent asks for obs, proxy gives BOTH obs and local_state.
This enables state syncing alongside observation delivery.

---

#### **3. Vectorize Observation (For RL Algorithms)**

```
Observation -> numpy array conversion

obs.vector()
├─> Flatten obs.local dict
│   └─> For each feature_name, feature_vec in local.items():
│       └─> Append feature_vec (already np.array)
│
├─> Flatten obs.global_info dict
│   └─> [Similar recursive flattening]
│
└─> Concatenate all parts
    └─> Returns: np.array([0.53, 100.0, ...], dtype=float32)

# Automatic conversion:
policy.forward(observation=obs)
└─> obs.__array__() called automatically
    └─> Returns obs.vector()
```

**Data Transformations:**
```
Observation(local={...}, global_info={...})
    ↓ .vector() or __array__()
np.ndarray([0.53, 100.0, ...])  # For neural networks
```

---

#### **4. Serialize/Deserialize Observation (Message Passing)**

**Serialization (Sending):**
```
obs = Observation(local={...}, global_info={...}, timestamp=0.0)

obs.to_dict()
└─> Returns: {
        "timestamp": 0.0,
        "local": {"BatteryChargeFeature": [0.53, 100.0]},  # Arrays -> lists
        "global_info": {"agent_states": {...}}
    }
```

**Deserialization (Receiving):**
```
obs_dict = message["get_obs_response"]["body"]["obs"]

obs = Observation.from_dict(obs_dict)
└─> Reconstructs Observation object
    └─> obs.local = {k: np.array(v) for k, v in obs_dict["local"].items()}
    └─> obs.global_info = obs_dict["global_info"]
    └─> obs.timestamp = obs_dict["timestamp"]
```

---

### **Observation Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Proxy State Cache (State Objects with Dict features)            │
├─────────────────────────────────────────────────────────────────┤
│ state_cache["agents"]["battery_1"] = FieldAgentState(           │
│     owner_id="battery_1",                                       │
│     features={"BatteryChargeFeature": BatteryChargeFeature(...)}│
│ )                                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓ get_observation() + observed_by()
┌─────────────────────────────────────────────────────────────────┐
│ Observation Construction (Visibility Filtered)                  │
├─────────────────────────────────────────────────────────────────┤
│ Observation(                                                    │
│     local={"BatteryChargeFeature": np.array([0.53, 100.0])},   │
│     global_info={...filtered states... + env_context},          │
│     timestamp=0.0                                               │
│ )                                                               │
│                                                                 │
│ Event-Driven: Also bundled with local_state for sync            │
└─────────────────────────────────────────────────────────────────┘
                            ↓ __array__()
┌─────────────────────────────────────────────────────────────────┐
│ Policy / RL Algorithm                                           │
├─────────────────────────────────────────────────────────────────┤
│ obs_vec = np.array([0.53, 100.0, ...])                          │
│ action = policy.forward(observation=obs)  # Auto-vectorized     │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Feature Data Flow**

### **Feature Object Structure**

```python
class BatteryChargeFeature(Feature):
    visibility = ["public"]  # Class-level metadata

    soc: float = 0.5
    capacity: float = 100.0

# Feature object attributes (auto-set by FeatureMeta metaclass):
feature.feature_name = "BatteryChargeFeature"  # From _class_feature_name
feature.visibility = ["public"]
feature.soc = 0.5
feature.capacity = 100.0
```

### **Feature Operations**

#### **1. Feature Creation & Registration**

```
Agent.__init__() -> init_state(features=features_list)
├─> For each feature in features_list:
│   └─> Feature object created
│   └─> Auto-registered in _FEATURE_REGISTRY via FeatureMeta
│
├─> self.state = FieldAgentState(
│       owner_id="battery_1",
│       owner_level=1,
│       features={f.feature_name: f for f in features_list}  # List -> Dict!
│   )
└─> State contains Dict of Feature objects keyed by name
```

**Data Outflow:**
```
Feature object -> Stored in state.features dict by name
```

**Feature access:**
```python
# O(1) access by name (Dict, not List!)
feature = state.features["BatteryChargeFeature"]
feature.soc  # 0.5
feature.capacity  # 100.0
```

---

#### **2. Feature Serialization (State -> Dict)**

```
state.to_dict()
├─> For each feature_name, feature in self.features.items():
│   ├─> feature_dict = feature.to_dict()
│   │   └─> Returns: {"soc": 0.5, "capacity": 100.0}
│   └─> result["features"][feature_name] = feature_dict
│
└─> Returns: {
        "_owner_id": "battery_1",
        "_owner_level": 1,
        "_state_type": "FieldAgentState",
        "features": {
            "BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}
        }
    }
```

---

#### **3. Feature Update**

```python
# Direct field-level update via Dict access:
state.features["BatteryChargeFeature"].set_values(soc=0.8)

# Batch update via State.update():
state.update({
    "BatteryChargeFeature": {"soc": 0.8}
})

# Reset features to defaults:
state.reset()  # or state.reset(overrides={"BatteryChargeFeature": {"soc": 0.5}})
```

---

#### **4. Feature Vectorization**

```
Get numeric representation of feature

feature.vector()
└─> Returns: np.array([feature.soc, feature.capacity], dtype=float32)
    └─> Example: np.array([0.5, 100.0])

state.vector()
├─> For each feature_name, feature in features.items():
│   └─> vectors.append(feature.vector())
├─> Concatenate all feature vectors
└─> Returns: np.array([0.5, 100.0, ...], dtype=float32)
```

---

#### **5. Feature Visibility (Observation Filtering)**

```
state.observed_by(requestor_id, requestor_level)
├─> For each feature_name, feature in state.features.items():
│   ├─> Check: feature.is_observable_by(requestor_id, requestor_level, owner_id, owner_level)
│   │   ├─> If visibility = ["public"]: Always True
│   │   ├─> If visibility = ["owner"]: True if requestor_id == owner_id
│   │   ├─> If visibility = ["upper_level"]: True if requestor_level == owner_level + 1
│   │   └─> If visibility = ["system"]: True if requestor_level >= 3
│   │
│   └─> If observable: Include feature.vector() in result
│
└─> Returns: {
        "BatteryChargeFeature": np.array([0.5, 100.0])  # Only visible features!
    }
```

**Visibility Enforcement Active** in `proxy.get_local_state()` and `proxy.get_global_states()`.

#### **6. Complete Visibility Filtering Example**

```
Field Agent (battery_1, level=1) requests observation:

proxy.get_observation("battery_1", protocol)
│
├─> LOCAL STATE (own state):
│   state_obj = state_cache["agents"]["battery_1"]  # FieldAgentState
│   filtered = state_obj.observed_by("battery_1", requestor_level=1)
│   │
│   │   Feature: BatteryChargeFeature (visibility=["public"])
│   │   ├─> is_observable_by("battery_1", 1, "battery_1", 1)
│   │   ├─> Check "public": True
│   │   └─> Include: array([0.503, 100.0])
│   │
│   Returns: {"BatteryChargeFeature": array([0.503, 100.0])}
│
├─> GLOBAL STATE (other agents):
│   For battery_2:
│       state_obj = state_cache["agents"]["battery_2"]
│       filtered = state_obj.observed_by("battery_1", requestor_level=1)
│       │
│       │   Feature: BatteryChargeFeature (visibility=["public"])
│       │   ├─> is_observable_by("battery_1", 1, "battery_2", 1)
│       │   ├─> Check "public": True
│       │   └─> Include: array([0.498, 100.0])
│       │
│       Returns: {"BatteryChargeFeature": array([0.498, 100.0])}
│
│   For coordinator_1:
│       Feature: CoordinatorPrivateFeature (visibility=["owner"])
│       ├─> is_observable_by("battery_1", 1, "coordinator_1", 2)
│       ├─> Check "owner": battery_1 != coordinator_1 -> False
│       └─> NOT included (filtered out!)
│
│   Include env_context from global cache if available
│
└─> Return: Observation(
        local={"BatteryChargeFeature": array([0.503, 100.0])},
        global_info={
            "battery_2": {"BatteryChargeFeature": array([0.498, 100.0])},
            "env_context": {...}  # If available
            # coordinator_1 private features NOT visible!
        },
        timestamp=1.0
    )
```

---

### **Feature Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Feature Definition (Class)                                      │
├─────────────────────────────────────────────────────────────────┤
│ class BatteryChargeFeature(Feature):                    │
│     visibility = ["public"]                                     │
│     soc: float = 0.5                                            │
│     capacity: float = 100.0                                     │
│                                                                 │
│ # Auto-registered via FeatureMeta metaclass!                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Instantiation
┌─────────────────────────────────────────────────────────────────┐
│ Feature Object Creation                                         │
├─────────────────────────────────────────────────────────────────┤
│ feature = BatteryChargeFeature(soc=0.5, capacity=100.0)         │
│ feature.feature_name = "BatteryChargeFeature"  # Auto-set       │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Added to State (as Dict entry)
┌─────────────────────────────────────────────────────────────────┐
│ State Composition                                               │
├─────────────────────────────────────────────────────────────────┤
│ state.features = {                                              │
│     "BatteryChargeFeature": feature,  # Dict keyed by name!     │
│     "AnotherFeature": feature2,                                 │
│ }                                                               │
│                                                                 │
│ state.vector() -> Concatenates all feature.vector() outputs     │
│ state.features["BatteryChargeFeature"].soc -> O(1) access       │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Proxy retrieval
┌─────────────────────────────────────────────────────────────────┐
│ Visibility Filtering (observed_by)                              │
├─────────────────────────────────────────────────────────────────┤
│ {                                                               │
│     "BatteryChargeFeature": np.array([0.5, 100.0])  # Visible!  │
│     # PrivateFeature filtered out if not authorized             │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Message Passing Architecture**

### **Message System Components**

#### **Message Structure (imported as BrokerMessage in base.py)**

```python
@dataclass
class Message:
    env_id: str                # Environment identifier
    sender_id: str             # Who sent the message
    recipient_id: str          # Who receives the message
    timestamp: float           # When message was created
    message_type: MessageType  # ACTION, INFO, BROADCAST
    payload: Dict[str, Any]    # Message content
```

#### **Message Types**

```python
class MessageType(Enum):
    ACTION = "action"          # Parent -> Child action commands
    INFO = "info"              # Information/observation requests
    BROADCAST = "broadcast"    # Multi-recipient messages
    STATE_UPDATE = "state_update"  # State update messages
    RESULT = "result"          # Generic result message
    CUSTOM = "custom"          # For domain-specific message types
```

### **Message Broker**

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
        """Clear all channels for an environment."""
```

### **Channel Management**

```python
class ChannelManager:
    @staticmethod
    def action_channel(upstream_id: str, node_id: str, env_id: str = "default") -> str:
        return f"env_{env_id}__action__{upstream_id}_to_{node_id}"

    @staticmethod
    def info_channel(node_id: str, upstream_id: str, env_id: str = "default") -> str:
        return f"env_{env_id}__info__{node_id}_to_{upstream_id}"
```

### **Message Flow Patterns**

#### **1. Observation Request/Response (Event-Driven)**

```
Agent -> Proxy (Request via scheduler):
├─> schedule_message_delivery(
│       sender_id=agent_id,
│       recipient_id=PROXY_AGENT_ID,
│       message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: protocol},
│       delay=msg_delay
│   )

Proxy -> Agent (Response with BOTH obs and local_state):
├─> obs = get_observation(sender_id, protocol)
├─> local_state = get_local_state(sender_id, protocol)
├─> schedule_message_delivery(
│       message={"get_obs_response": {
│           "body": {
│               "obs": obs.to_dict(),
│               "local_state": local_state
│           }
│       }}
│   )
```

#### **2. Action Passing (via Broker)**

```
Coordinator -> FieldAgent (via message broker):
├─> channel = action_channel(coord_id, field_id, env_id)
├─> message = Message(
│       env_id=env_id,
│       sender_id=coord_id,
│       recipient_id=field_id,
│       message_type=MessageType.ACTION,
│       payload={"action": action}
│   )
└─> broker.publish(channel, message)

FieldAgent receives (at tick start):
├─> _check_for_upstream_action()
│   ├─> actions = receive_upstream_actions(sender_id=upstream_id, clear=True)
│   │   └─> receive_actions(broker, upstream_id, clear=True)
│   │       ├─> channel = ChannelManager.action_channel(upstream_id, self.agent_id, env_id)
│   │       ├─> messages = broker.consume(channel, self.agent_id, env_id, clear=True)
│   │       └─> return [msg.payload["action"] for msg in messages]
│   └─> self._upstream_action = actions[-1] if actions else None
```

#### **3. State Update (Event-Driven)**

```
Agent -> Proxy (after apply_action):
├─> schedule_message_delivery(
│       message={"set_state": "local", "body": state.to_dict(include_metadata=True)}
│   )

Proxy receives and updates:
├─> state_dict = message["body"]
├─> state = State.from_dict(state_dict)
├─> set_local_state(state.owner_id, state)  # State object!
└─> schedule_message_delivery(
        message={MSG_SET_STATE_COMPLETION: "success"}
    )
```

### **Event-Driven Message Scheduling**

```python
def schedule_message_delivery(
    self,
    sender_id: str,
    recipient_id: str,
    message: Dict,
    delay: Optional[float] = None,
):
    """Schedule a message to be delivered after delay."""
    if delay is None:
        delay = self.get_msg_delay(sender_id)

    self.schedule(Event(
        timestamp=self.current_time + delay,
        event_type=EventType.MESSAGE_DELIVERY,
        agent_id=recipient_id,
        priority=2,  # Communication-level (after state changes)
        payload={"sender": sender_id, "message": message}
    ))
```

---

## **CTDE Training Mode - Complete Flow**

### **Phase 0: Initialization**

```
env = CustomEnv(system_agent=grid_system_agent)
│
├─> BaseEnv.__init__()
│   │
│   ├─> _register_agents(system_agent, coordinator_agents)
│   │   ├─> system_agent.set_simulation(run_simulation, ...)
│   │   └─> _register_agent(system_agent) [recursive for all]
│   │
│   ├─> Create Proxy(), register it
│   │
│   ├─> Setup MessageBroker, attach to agents
│   │
│   ├─> proxy.attach(registered_agents)
│   │   ├─> For each agent:
│   │   │   └─> _register_agent(agent)
│   │   │       ├─> Track agent_id, level, upstream_id
│   │   │       └─> set_local_state(agent_id, agent.state)  # State object!
│   │   ├─> init_global_state()
│   │   ├─> For each agent: post_proxy_attach(proxy)
│   │   │   └─> FieldAgent: Compute action_space, observation_space
│   │   └─> _setup_channels()
│   │
│   └─> Setup EventScheduler, attach to agents
```

**Data State After Init:**
```python
proxy.state_cache = {
    "agents": {
        "battery_1": FieldAgentState(features={"BatteryChargeFeature": ...}),
        "battery_2": FieldAgentState(features={"BatteryChargeFeature": ...}),
        "coordinator_1": CoordinatorAgentState(features={...}),
        "system_agent": SystemAgentState(features={...})
    }
}
```

---

### **Phase 1: Reset**

```
obs, info = env.reset(seed=42)
│
├─> scheduler.reset(start_time=0.0)
├─> clear_broker_environment()
├─> proxy.reset() -> state_cache = {}
│
├─> system_agent.reset(seed, proxy)
│   ├─> Agent.reset():
│   │   ├─> self._timestep = 0.0
│   │   ├─> self.action.reset()
│   │   ├─> self.state.reset()
│   │   ├─> proxy.set_local_state(self.agent_id, self.state)
│   │   └─> For each subordinate: subordinate.reset(seed, proxy)
│   │
│   └─> return self.observe(proxy=proxy), {}
│
├─> proxy.init_global_state()
│
└─> return (observations, {})
```

---

### **Phase 2: Step Execution**

```
obs, rewards, terminated, truncated, info = env.step(actions)
│
└─> SystemAgent.execute(actions, proxy)
```

#### **Step 2.1: Pre-Step + State Sync + Action Application**

```
PHASE 0: Pre-step hook
├─> pre_step_func() if configured

PHASE 0.5: State Sync
├─> local_state = proxy.get_local_state(sender_id=system_agent_id, protocol=protocol)
├─> sync_state_from_observed(local_state)

PHASE 1: Actions -> State Updates

actions = self.layer_actions(actions)
self.act(actions, proxy)
│
├─> self._timestep += 1
│
├─> handle_self_action(actions['self'], proxy)
│   ├─> set_action(action) or policy.forward()
│   ├─> apply_action()
│   └─> proxy.set_local_state(self.agent_id, self.state)
│
└─> handle_subordinate_actions(actions['subordinates'], proxy)
    └─> For each subordinate:
        └─> subordinate.execute(layered_actions, proxy)
            ├─> Sync state from proxy
            └─> subordinate.act(actions, proxy)
                └─> [Recursive for entire hierarchy]
```

---

#### **Step 2.2: Simulation**

```
PHASE 2: Physics Simulation

global_state = proxy.get_global_states(sender_id=system_agent_id, protocol=protocol)
│
DATA RETRIEVAL:
├─> For each agent in state_cache["agents"]:
│   └─> filtered = state.observed_by(system_agent_id, system_level)
├─> Include env_context if available
└─> Returns: Dict of all filtered states

updated_global_state = simulate(global_state)
│
├─> env_state = global_state_to_env_state(global_state)
├─> updated_env_state = run_simulation(env_state)
└─> return env_state_to_global_state(updated_env_state)

proxy.set_global_state(updated_global_state)
│
└─> state_cache["global"].update(updated_global_state)
```

---

#### **Step 2.3: Observation Collection**

```
PHASE 3: Collect Observations

obs = self.observe(proxy=proxy)
│
└─> For each agent in hierarchy:
    │
    └─> observation = proxy.get_observation(sender_id=agent_id, protocol=protocol)
        │
        ├─> local_state = get_local_state(sender_id=agent_id, protocol=protocol, include_subordinate_rewards=False)
        │   └─> state.observed_by(sender_id, requestor_level)
        │   └─> Returns: {"BatteryChargeFeature": np.array([0.503, 100.0])}
        │
        ├─> global_state = get_global_states(agent_id, agent_level)
        │   └─> {other agents' filtered states + env_context}
        │
        └─> return Observation(
                local=local_state,
                global_info=global_state,
                timestamp=timestep
            )

Returns: {
    "battery_1": Observation(local={...}, global_info={...}),
    "battery_2": Observation(...),
}
```

---

#### **Step 2.4: Reward Computation**

```
PHASE 4: Compute Rewards

rewards = self.compute_rewards(proxy)
│
└─> For each agent:
    │
    ├─> local_state = proxy.get_local_state(sender_id=agent_id, protocol=protocol)
    │   │                                   ↑ include_subordinate_rewards=True (default)
    │   └─> Returns: {"BatteryChargeFeature": np.array([0.503, 100.0]),
    │                  "subordinate_rewards": {...} (if parent agent)}
    │
    └─> reward = compute_local_reward(local_state)
        │
        DATA USAGE:
        feature_vec = local_state["BatteryChargeFeature"]
        soc = feature_vec[0]  # Extract from numpy array!
        return soc  # 0.503

Returns: {
    "battery_1": 0.503,
    "battery_2": 0.498,
}
```

---

#### **Step 2.5: Result Caching & Vectorization**

```
PHASE 5: Cache Results

proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)
│
STORAGE:
_step_results = {
    "obs": {"battery_1": Observation(...), ...},  # Observation objects!
    "rewards": {"battery_1": 0.503, ...},
    "terminateds": {...},
    "truncateds": {...},
    "infos": {...}
}

Return to RL algorithm:

proxy.get_step_results()
│
DATA VECTORIZATION:
obs_vectorized = {
    agent_id: observation.vector()
    for agent_id, observation in obs.items()
}
│
└─> observation.vector()
    ├─> Flatten local: {"BatteryChargeFeature": np.array([0.503, 100.0])}
    │   └─> [0.503, 100.0]
    ├─> Flatten global_info
    └─> Concatenate: np.array([0.503, 100.0, ...], dtype=float32)

Returns: (
    obs_vectorized: {"battery_1": np.ndarray([0.503, 100.0, ...]), ...},
    rewards,
    terminateds,
    truncateds,
    infos
)
```

---

## **Event-Driven Testing Mode - Complete Flow**

### **Timeline Execution with Data Transformations**

#### **t=0.0: AGENT_TICK(system_agent)**

```
SystemAgent.tick(scheduler, current_time=0.0)
│
├─> pre_step_func() if configured
│
├─> super().tick() -> self._timestep = 0.0, _check_for_upstream_action()
│
├─> Schedule subordinate ticks:
│   └─> scheduler.schedule_agent_tick("coordinator_1")
│
├─> If has policy: Request observation
│   └─> schedule_message_delivery(
│           sender=system_agent, recipient=proxy,
│           message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: protocol},
│           delay=msg_delay
│       )
│
└─> Schedule simulation:
    └─> scheduler.schedule_simulation(system_agent, wait_interval)
```

---

#### **t=msg_delay: MESSAGE_DELIVERY(proxy) - Observation Request**

```
Proxy.message_delivery_handler()
│
├─> Receive: {MSG_GET_INFO: "obs", MSG_KEY_PROTOCOL: protocol}
│
├─> obs = get_observation(system_agent, protocol)
│   ├─> global = get_global_states() -> visibility-filtered
│   ├─> local = get_local_state(include_subordinate_rewards=False)
│   └─> return Observation(local, global, timestamp)
│
├─> local_state = get_local_state(system_agent, protocol)  # For state sync
│
├─> Bundle BOTH obs and local_state:
│   info_data = {
│       "obs": obs.to_dict(),
│       "local_state": local_state
│   }
│
└─> schedule_message_delivery(
        message={"get_obs_response": {"body": info_data}},
        delay=msg_delay
    )
```

---

#### **t=2*msg_delay: MESSAGE_DELIVERY(system_agent) - Observation Response**

```
SystemAgent.message_delivery_handler()
│
├─> body = message["get_obs_response"]["body"]
├─> obs_dict = body["obs"]
├─> local_state = body["local_state"]
│
├─> obs = Observation.from_dict(obs_dict)
│
├─> Sync state: self.sync_state_from_observed(local_state)
│
└─> compute_action(obs, scheduler)
    ├─> action = policy.forward(observation=obs)
    │   └─> obs.__array__() auto-converts
    ├─> If protocol: coordinate() -> send subordinate actions
    ├─> set_action(action)
    └─> schedule_action_effect(delay=act_delay)
```

---

#### **t=coordinator_tick: AGENT_TICK(coordinator_1)**

```
CoordinatorAgent.tick(scheduler, current_time)
│
├─> super().tick() -> _check_for_upstream_action()
│   └─> self._upstream_action = broker.consume(action_channel)[-1]  # Cached!
│
├─> Schedule subordinate ticks
│
└─> Always request obs from proxy (for state sync):
    └─> schedule_message_delivery(
            sender=self.agent_id, recipient=proxy,
            message={get_info: "obs", protocol: protocol},
            delay=msg_delay
        )
```

#### **t=coord_tick+2×msg_delay: MESSAGE_DELIVERY(coordinator_1) - obs response**

```
CoordinatorAgent.message_delivery_handler() on "get_obs_response"
│
├─> Parse obs + local_state from response body
├─> obs = Observation.from_dict(obs_dict)
├─> sync_state_from_observed(local_state)  # State sync first!
│
└─> compute_action(obs, scheduler)
    ├─> If self._upstream_action: use it (priority!)
    │   └─> self._upstream_action = None  # Clear after use
    ├─> Elif has policy: action = policy.forward(obs)
    ├─> coordinate() -> field actions via protocol
    └─> send_subordinate_action(field_id, action) for each
```

---

#### **t=field_tick: AGENT_TICK(field_1)**

```
FieldAgent.tick(scheduler, current_time)
│
├─> super().tick() -> _check_for_upstream_action()
│   └─> self._upstream_action = broker.consume(action_channel)[-1]  # Cached!
│
└─> Always request obs from proxy (for state sync):
    └─> schedule_message_delivery(
            sender=self.agent_id, recipient=proxy,
            message={get_info: "obs", protocol: protocol},
            delay=msg_delay
        )
```

#### **t=field_tick+2×msg_delay: MESSAGE_DELIVERY(field_1) - obs response**

```
FieldAgent.message_delivery_handler() on "get_obs_response"
│
├─> Parse obs + local_state from response body
├─> obs = Observation.from_dict(obs_dict)
├─> sync_state_from_observed(local_state)  # State sync first!
│
└─> compute_action(obs, scheduler)
    ├─> If self._upstream_action: use it (priority!)
    │   └─> self._upstream_action = None  # Clear after use
    ├─> Elif has policy: action = policy.forward(obs)
    ├─> set_action(action)
    └─> schedule_action_effect(delay=act_delay)
```

---

#### **t=field_tick+2×msg_delay+act_delay: ACTION_EFFECT(field_1)**

```
FieldAgent.action_effect_handler()
│
├─> apply_action()
│   └─> Read self.action, update self.state
│
└─> schedule_message_delivery(
        message={"set_state": "local", "body": self.state.to_dict(include_metadata=True)},
        delay=msg_delay
    )
```

---

#### **Reward Computation Cascade**

```
Proxy sends MSG_SET_STATE_COMPLETION back to the SENDER (not parent).
SystemAgent receives it first and propagates down the hierarchy.

1. SystemAgent.message_delivery_handler() on MSG_SET_STATE_COMPLETION:
   │
   ├─> Propagate completion to each coordinator (subordinate):
   │   └─> For each subordinate_id:
   │       └─> schedule_message_delivery(
   │               sender=self.agent_id, recipient=subordinate_id,
   │               message={MSG_SET_STATE_COMPLETION: "success"}
   │           )
   │
   └─> Request own local state (with reward_delay to wait for subordinates):
       └─> schedule_message_delivery(
               sender=self.agent_id, recipient=proxy,
               message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: protocol},
               delay=self._schedule_config.reward_delay
           )

2. CoordinatorAgent.message_delivery_handler() on MSG_SET_STATE_COMPLETION:
   │
   ├─> Propagate completion to each field agent (subordinate):
   │   └─> For each subordinate_id:
   │       └─> schedule_message_delivery(
   │               sender=self.agent_id, recipient=subordinate_id,
   │               message={MSG_SET_STATE_COMPLETION: "success"}
   │           )
   │
   └─> Request own local state (with reward_delay to wait for field agents):
       └─> schedule_message_delivery(
               sender=self.agent_id, recipient=proxy,
               message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: protocol},
               delay=self._schedule_config.reward_delay
           )

3. FieldAgent.message_delivery_handler() on MSG_SET_STATE_COMPLETION:
   │
   └─> Request own local state immediately (no reward_delay):
       └─> schedule_message_delivery(
               sender=self.agent_id, recipient=proxy,
               message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: protocol}
           )

4. All agents on "get_local_state_response":
   │
   ├─> local_state = response body
   ├─> sync_state_from_observed(local_state)  # Sync first!
   │
   ├─> tick_result = {
   │       "reward": compute_local_reward(local_state),
   │       "terminated": is_terminated(local_state),
   │       "truncated": is_truncated(local_state),
   │       "info": get_local_info(local_state)
   │   }
   │
   └─> schedule_message_delivery(
           sender=self.agent_id, recipient=proxy,
           message={MSG_SET_TICK_RESULT: INFO_TYPE_LOCAL_STATE, MSG_KEY_BODY: tick_result}
       )

# Proxy stores tick_results per agent for retrieval by parent agents
# SystemAgent: if not terminated/truncated, schedule next system tick
```

---

## **Proxy State Cache Architecture**

### **Cache Structure**

```python
proxy.state_cache = {
    # Per-agent states (State objects with Dict features!)
    "agents": {
        "battery_1": FieldAgentState(
            owner_id="battery_1",
            owner_level=1,
            features={"BatteryChargeFeature": BatteryChargeFeature(soc=0.503, capacity=100.0)}
        ),
        "battery_2": FieldAgentState(...),
        "coordinator_1": CoordinatorAgentState(...),
        "system_agent": SystemAgentState(...)
    },

    # Global state (environment-wide data)
    "global": {
        "agent_states": {agent_id: state_obj, ...},
        "env_context": {...}  # External data (price, solar, wind profiles)
    }
}

# Agent levels tracked for visibility checks:
proxy._agent_levels = {
    "battery_1": 1,
    "battery_2": 1,
    "coordinator_1": 2,
    "system_agent": 3
}

# Agent upstream tracking:
proxy._agent_upstreams = {
    "battery_1": "coordinator_1",
    "battery_2": "coordinator_1",
    "coordinator_1": "system_agent",
    "system_agent": None
}

# Tick results per agent (event-driven mode):
proxy._tick_results = {
    "battery_1": {"reward": 0.503, "terminated": False, "truncated": False, "info": {}},
    ...
}
```

### **State Cache Operations**

#### **SET Operations**

| Method | Input Data | Transformation | Storage Location |
|--------|-----------|----------------|------------------|
| `set_local_state(aid, state)` | `agent_id: str`, `state: State` | None (stores object!) | `state_cache["agents"][aid]` |
| `set_global_state(dict)` | `global_dict: Dict` | None | `state_cache["global"].update(...)` |

#### **GET Operations**

| Method | Retrieval Source | Filtering | Return Type |
|--------|-----------------|-----------|-------------|
| `get_local_state(aid, protocol, incl_sub_rewards)` | `state_cache["agents"][aid]` | `state.observed_by()` + optional subordinate rewards | `Dict[str, np.ndarray]` |
| `get_global_states(aid, protocol)` | `state_cache["agents"]` (all except self) | `state.observed_by()` + env_context | `Dict[str, Dict[str, np.ndarray]]` |
| `get_observation(aid, protocol)` | Both local + global | Feature visibility (no subordinate rewards) | `Observation` |
| `get_serialized_agent_states()` | `state_cache["agents"]` | `to_dict(include_metadata=True)` | `Dict[aid, Dict]` |

---

## **Protocol-Based Coordination**

### **Protocol Architecture**

```python
class Protocol(ABC):
    """Two-layer design: communication + action coordination."""

    def __init__(self, communication_protocol, action_protocol):
        self.communication_protocol = communication_protocol or NoCommunication()
        self.action_protocol = action_protocol or NoActionCoordination()

    def coordinate(
        self,
        coordinator_state: Any,
        coordinator_action: Optional[Any] = None,
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """
        Returns:
            messages: Dict of coordination messages per subordinate
            actions: Dict of actions per subordinate
        """
        messages = self.communication_protocol.compute_coordination_messages(
            sender_state=coordinator_state,
            receiver_infos=info_for_subordinates,
            context=context
        )
        actions = self.action_protocol.compute_action_coordination(
            coordinator_action=coordinator_action,
            info_for_subordinates=info_for_subordinates,
            coordination_messages=messages,
            context=context
        )
        return messages, actions
```

### **Protocol Implementations**

#### **1. NoProtocol (Default)**
- Communication: NoCommunication
- Action: NoActionCoordination
- Each agent acts independently

#### **2. VerticalProtocol (Hierarchical)**
- Communication: NoCommunication (default)
- Action: VectorDecompositionActionProtocol (splits joint action vector)
- Parent coordinates subordinates via action decomposition

```python
class VerticalProtocol(Protocol):
    def __init__(self, communication_protocol=None, action_protocol=None):
        super().__init__(
            communication_protocol=communication_protocol or NoCommunication(),
            action_protocol=action_protocol or VectorDecompositionActionProtocol()
        )

    def register_subordinates(self, subordinates):
        self.action_protocol.register_subordinates(subordinates)
```

**VectorDecompositionActionProtocol:**
- If coordinator action is a dict: use directly
- If coordinator action is a vector: split by subordinate action dimensions
- If coordinator action is None: return None for all subordinates

### **Protocol Usage in Execution**

```
Agent.compute_action(obs, scheduler) [event-driven mode]
│
├─> action = policy.forward(obs) or upstream_action
│
├─> If _should_send_subordinate_actions():
│   └─> self.coordinate(obs, action)
│       │
│       ├─> messages, sub_actions = protocol.coordinate(
│       │       coordinator_state=self.state,
│       │       coordinator_action=action,
│       │       info_for_subordinates={sub_id: obs for sub_id in subordinates},
│       │   )
│       │
│       ├─> Send actions to subordinates via broker:
│       │   for sub_id, sub_action in sub_actions.items():
│       │       self.send_subordinate_action(sub_id, sub_action)
│       │
│       └─> Send coordination messages (not used in default protocols):
│           for sub_id, message in messages.items():
│               self.send_info(broker, sub_id, message)
│
└─> set_action(action)
```

---

## **Key Design Patterns**

### **1. Object-Dict Duality**

**Agent Layer:**
- Works with rich objects (State, Action, Observation)
- Has methods: `.vector()`, `.set_values()`, `.update()`
- State.features is Dict[str, Feature] for O(1) access

**Proxy Layer:**
- Stores State objects directly
- Returns filtered vectors via `observed_by()`
- Serialization only at message boundaries

**Boundary:**
- **Agent -> Proxy (CTDE)**: Pass State object directly
- **Agent -> Proxy (Event-Driven)**: Call `.to_dict()` -> Message -> `from_dict()`
- **Proxy -> Agent**: Return filtered feature vectors + optional subordinate rewards

---

### **2. Serialization Points**

| Crossing Point | Data Type | Transformation |
|----------------|-----------|----------------|
| Agent -> Proxy (CTDE) | State object | None (direct reference) |
| Agent -> Message | State object | `.to_dict(include_metadata=True)` |
| Agent -> Message | Observation object | `.to_dict()` |
| Message -> Agent | Observation dict | `Observation.from_dict()` |
| Proxy -> Agent | State object | `state.observed_by()` -> filtered vectors |
| Proxy -> Agent (obs response) | Observation + local_state | Bundled together |
| Observation -> Policy | Observation object | `.__array__()` auto-converts |
| Agent -> Broker (Action) | Action object | Direct (in payload dict) |
| Broker -> Agent (Action) | Action from payload | `msg.payload["action"]` |

---

### **3. Data Consistency Rules**

**DO:**
- Use Action objects throughout (no dict conversion in CTDE)
- Pass State objects to proxy in CTDE mode
- Convert State -> Dict only when sending messages (with `include_metadata=True`)
- Use `local_state` parameter (numpy array dict) in reward/info methods
- Access features via Dict key: `state.features["FeatureName"]`
- Access feature vector elements: `local_state["FeatureName"][0]`
- Use upstream actions when available (priority over policy)
- Call `sync_state_from_observed()` after receiving state from proxy

**DON'T:**
- Send State objects in messages (must serialize to dict first)
- Access features by list index: `state.features[0]` (it's a Dict!)
- Access dict fields like `local_state["Feature"]["soc"]` (it's a numpy array!)
- Access `self.state` in `compute_local_reward()` (use filtered parameter)
- Ignore upstream actions (they have priority!)
- Skip state syncing after simulation (data will be stale)

---

### **4. Type Consistency Summary**

- **State**: Object (agent) -> **Object (proxy)** -> **Filtered vectors (via observed_by())**
- **State.features**: Dict[str, Feature] (keyed by feature_name, O(1) lookup)
- **Action**: Object (throughout, passed via broker in event-driven mode)
- **Observation**: Object -> Dict (messages) -> Object (reconstruction)
- **Feature**: Object (in State Dict) -> Stays in State object -> Vector (in observations)
- **Messages**: Always serialized (dicts with constants from heron/agents/constants.py)

---

## **Complete Example Trace**

### **Single Agent, Single Step - All Data Transformations**

```
=== INITIALIZATION ===
agent.__init__(features=[BatteryChargeFeature(soc=0.5, capacity=100.0)]):
    CREATE: FieldAgentState(
        owner_id="battery_1", owner_level=1,
        features={"BatteryChargeFeature": BatteryChargeFeature(soc=0.5, capacity=100.0)}
    )
    TYPE: State object with Dict features

proxy._register_agent(agent):
    STORE: state_cache["agents"]["battery_1"] = agent.state
    TYPE: State object (direct reference!)

=== STEP EXECUTION ===
env.step({"battery_1": Action(c=[0.3])}):

1. State Sync (in execute):
    local_state = proxy.get_local_state("battery_1", protocol)
    sync_state_from_observed(local_state)
    ACTION: Reconcile internal state with proxy

2. Action Application:
    INPUT: Action(c=[0.3])
    TYPE: Action object

    set_action(action):
        PROCESS: self.action.set_values(action)
        UPDATE: self.action.c = [0.3]

    apply_action():
        READ: self.action.c[0] = 0.3
        READ: self.state.features["BatteryChargeFeature"].soc = 0.5
        COMPUTE: new_soc = 0.5 + 0.3*0.01 = 0.503
        UPDATE: self.state.features["BatteryChargeFeature"].soc = 0.503

    proxy.set_local_state("battery_1", state):
        STORE: state_cache["agents"]["battery_1"] = state
        TYPE: State object (same reference, updated!)

3. Observation Collection:
    proxy.get_observation("battery_1", protocol):
        FILTER: state.observed_by("battery_1", 1)
        CONSTRUCT: Observation(
            local={"BatteryChargeFeature": np.array([0.503, 100.0])},
            global_info={...filtered states... + env_context}
        )
        TYPE: Observation object

4. Reward Computation:
    proxy.get_local_state("battery_1", protocol):
        FILTER: state.observed_by("battery_1", 1)
        RETURN: {"BatteryChargeFeature": np.array([0.503, 100.0]),
                 "subordinate_rewards": {...} if parent}
        TYPE: Dict[str, np.ndarray]

    compute_local_reward(local_state):
        ACCESS: local_state["BatteryChargeFeature"][0]
        RETURN: 0.503
        TYPE: float

5. Vectorization for RL:
    proxy.get_step_results():
        INPUT: obs = {"battery_1": Observation(...)}
        TRANSFORM: observation.vector() for each
        OUTPUT: {"battery_1": np.array([0.503, 100.0, ...])}
        TYPE: Dict[str, np.ndarray]

=== RETURN TO RL ===
    obs: Dict[str, np.ndarray]     <- Numpy arrays
    rewards: Dict[str, float]      <- Scalars
    terminated: Dict[str, bool]
    truncated: Dict[str, bool]
    info: Dict[str, Dict]
```

---

## **Critical Takeaways**

### **1. State.features is Dict[str, Feature]**
- Keyed by feature_name for O(1) lookup
- Converted from List at init_state: `{f.feature_name: f for f in features}`
- Access pattern: `state.features["BatteryChargeFeature"].soc`

### **2. Proxy Storage is State Objects**
- State objects stored directly in `state_cache["agents"]`
- Enables feature-level visibility filtering via `state.observed_by()`
- Maintains full type information (Feature instances)
- Serialization needed only for message passing

### **3. Observation Response Bundles Obs + Local State**
- Proxy sends both obs and local_state when agent requests observation
- Enables state syncing via `sync_state_from_observed()`
- Design principle: "agent asks for obs, proxy gives both"

### **4. Action Passing Uses Obs-First Pattern**
- Both FieldAgent and CoordinatorAgent always request obs from proxy first (for state sync)
- Upstream actions (cached via `_check_for_upstream_action()`) have priority over policy in `compute_action()`
- CoordinatorAgent additionally runs `protocol.coordinate()` to decompose actions for subordinates

### **5. Reward Computation Is Parent-Initiated (Event-Driven)**
- State completion triggers cascade from parent agent
- Parent requests local states for all subordinates
- Each agent computes own tick_result (reward, terminated, truncated, info)

### **6. State Syncing via sync_state_from_observed()**
- Reconciles agent internal state with proxy state
- Handles both Dict and numpy array formats
- Called in CTDE (execute) and event-driven (message handler)

### **7. Type Consistency**
- **State**: Object (agent) -> Object (proxy) -> Filtered vectors (retrieval)
- **Action**: Object (throughout, no proxy storage)
- **Observation**: Object -> Dict (messages) -> Object (reconstruction)
- **Feature**: Object (in State Dict) -> Vector (in observations)

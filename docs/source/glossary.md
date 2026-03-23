# Glossary

This glossary defines key terms used throughout the HERON framework documentation.

## Agent Hierarchy

### Agent
A decision-making entity that observes its environment and takes actions. In HERON, agents are organized hierarchically.

### FieldAgent (L1)
The lowest level in the agent hierarchy. Field agents:
- Manage individual units (sensors, actuators, devices)
- Have local state and actions
- Report to a CoordinatorAgent
- Example: A temperature sensor, a generator controller

### CoordinatorAgent (L2)
Mid-level agents that manage groups of field agents. Coordinators:
- Aggregate observations from subordinates
- Compute and distribute actions via protocols
- Can have their own state and policies
- Example: A zone controller, a microgrid operator

### SystemAgent (L3)
Top-level agent for global coordination. System agents:
- Manage multiple coordinators
- Make system-wide decisions
- Interface with external systems
- Example: A grid operator, a building manager

### Proxy
Special agent for state distribution in distributed mode. The proxy:
- Caches environment state
- Filters state based on visibility permissions
- Enables delayed observations

### Subordinate
An agent managed by a higher-level coordinator. Field agents are subordinates of coordinators.

### Upstream
The coordinator or system agent that manages a subordinate. A field agent's upstream is its coordinator.

## State and Observation

### Feature
A composable unit of agent state. Features define:
- What data the agent tracks
- Visibility permissions (who can observe it)
- Serialization for RL algorithms
- See: `Feature` base class

### Feature
Abstract base class for defining features. Subclasses must implement:
- `vector()`: Convert to numpy array
- `names()`: Human-readable dimension names
- `to_dict()` / `from_dict()`: Serialization
- `set_values()`: Update feature values

### Visibility
Controls which agents can observe a feature. Standard tags:

| Tag | Who Can Observe |
|-----|-----------------|
| `owner` | Only the owning agent |
| `coordinator` | Owner + its coordinator |
| `system` | System-level agents |
| `global` | All agents |

Custom tags can be defined for domain-specific needs.

### State
The internal state of an agent, composed of features. Types:
- `FieldAgentState`: For field agents
- `CoordinatorAgentState`: For coordinators

### Observation
What an agent perceives at a given time. Contains:
- `local`: Agent's local state/observations
- `global_info`: System-wide information (centralized mode)
- `messages`: Received messages (distributed mode)
- `timestamp`: When the observation was made

## Actions

### Action
A decision made by an agent. HERON actions support:
- Continuous components (`dim_c`): Real-valued actions
- Discrete components (`dim_d`): Categorical choices
- Bounds specification via `range` parameter

### Joint Action
Combined actions for all subordinates, computed by a coordinator.

### Upstream Action
Action received from a coordinator, directing a subordinate's behavior.

## Protocols

### Protocol
Defines how agents coordinate. Composed of:
- `CommunicationProtocol`: What messages to exchange
- `ActionProtocol`: How to compute coordinated actions

### Vertical Protocol
Coordination between levels (coordinator -> subordinates):
- **SetpointProtocol**: Direct action assignment
- **PriceSignalProtocol**: Indirect coordination via prices

### Horizontal Protocol
Peer-to-peer coordination (environment-mediated):
- **PeerToPeerTradingProtocol**: Resource trading marketplace
- **ConsensusProtocol**: Distributed agreement via gossip

### Setpoint
A direct action command from coordinator to subordinate.

### Price Signal
An indirect coordination signal (e.g., energy price) that influences subordinate behavior without direct control.

## Execution Modes

### Synchronous Mode (Option A)
Training mode where all agents step together:
- Direct method calls between agents
- Instantaneous communication
- Used for RL training (CTDE pattern)

### Event-Driven Mode (Option B)
Testing mode with realistic timing:
- Agents tick at different rates
- Configurable delays for observations, actions, messages
- Uses `EventScheduler` for priority-queue execution

### CTDE (Centralized Training, Decentralized Execution)
Common MARL pattern where:
- **Training**: Agents have full observability
- **Execution**: Agents use only local observations

## Timing Parameters

### tick_interval
Time between an agent's observe/act cycles. Measured in simulation time (seconds).

### obs_delay
Latency between state change and agent observing it. Models sensor/network delay.

### act_delay
Latency between action computation and effect. Models actuator response time.

### msg_delay
Latency for message delivery between agents. Models network transmission time.

### Jitter
Random variability added to timing parameters for realistic testing. Types:
- `NONE`: Deterministic (training)
- `UNIFORM`: Bounded randomness
- `GAUSSIAN`: Natural variability

## Messaging

### Message Broker
System for agent communication in distributed mode. Provides:
- Publish/subscribe channels
- Message persistence
- Environment isolation

### InMemoryBroker
Default message broker implementation for development/testing. Fast but not distributed.

### Channel
Named topic for message routing. Types:
- Action channels: Coordinator -> Subordinate
- Info channels: Subordinate -> Coordinator
- Broadcast channels: One-to-many

## Environment

### HeronBaseEnv
Mixin providing HERON functionality to any environment. Handles:
- Agent registration and management
- Event-driven scheduling
- Message broker integration

### BaseEnv
Single-agent Gymnasium-compatible environment.

### HeronEnv
Multi-agent environment base class. Adapters available for:
- PettingZoo (`PettingZooParallelEnv`)
- RLlib (`RLlibMultiAgentEnv`)

## Event-Driven Scheduling

### EventScheduler
Priority-queue scheduler for event-driven execution. Processes events in timestamp order.

### Event
A scheduled occurrence in the simulation. Types:
- `AGENT_TICK`: Regular agent step
- `ACTION_EFFECT`: Delayed action takes effect
- `MESSAGE_DELIVERY`: Message arrives at recipient

### ScheduleConfig
Configuration object for agent timing. Supports deterministic or jittered timing.

## See Also

- [Key Concepts](key_concepts.md) - Conceptual overview with code examples
- [Getting Started](getting_started.md) - Quick start guide
- [Event-Driven Execution](user_guide/event_driven_execution.md) - Timing and scheduling guide

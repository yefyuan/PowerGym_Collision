# Architecture Overview

HERON provides a layered architecture with clear separation of concerns between the domain-agnostic framework and domain-specific case studies.

## Project Structure

```
heron/                          # Domain-agnostic MARL framework
├── agents/                     # Hierarchical agent abstractions
│   ├── base.py                 # Agent base class with level property
│   ├── field_agent.py          # Leaf-level agents (local sensing/actuation)
│   ├── coordinator_agent.py    # Mid-level agents (manages subordinate agents)
│   ├── system_agent.py         # Top-level agent (global coordination)
│   ├── proxy.py          # Singleton state mediation hub
│   └── constants.py            # SYSTEM_AGENT_ID, PROXY_AGENT_ID constants
│
├── core/                       # Core abstractions
│   ├── action.py               # Action with continuous/discrete support
│   ├── observation.py          # Observation with local/global/messages
│   ├── state.py                # State with Feature composition
│   ├── feature.py              # Feature with visibility tags + registry
│   └── policies.py             # Policy abstractions (random, rule-based)
│
├── protocols/                  # Coordination protocols
│   ├── base.py                 # Protocol, CommunicationProtocol, ActionProtocol
│   ├── vertical.py             # VectorDecompositionActionProtocol, BroadcastActionProtocol
│   └── horizontal.py           # StateShareCommunicationProtocol
│
├── messaging/                  # Message broker system
│   ├── broker_base.py          # MessageBroker abstract interface
│   ├── in_memory_broker.py     # InMemoryBroker implementation
│   ├── channels.py             # ChannelManager for routing
│   └── messages.py             # Message dataclass, MessageType enum
│
├── scheduling/                 # Event-driven scheduling
│   ├── scheduler.py            # EventScheduler (heap-based priority queue)
│   ├── event.py                # Event dataclass, EventType enum
│   ├── schedule_config.py          # ScheduleConfig (intervals, delays, jitter)
│   └── analysis.py             # EpisodeAnalyzer, EpisodeStats
│
├── envs/                       # Base environment interfaces
│   └── base.py                 # BaseEnv, HeronEnv (extends BaseEnv)
│
├── adaptors/                   # RL framework adaptors
│   ├── epymarl.py              # EPyMARL integration
│   └── rllib.py                # RLlib integration
│
└── utils/                      # Common utilities
    ├── typing.py               # Type definitions (AgentID, MultiAgentDict)
    └── array_utils.py          # Array manipulation utilities
```

## Design Principles

### 1. Hierarchical Agents

Multi-level hierarchy with configurable depth:

```
Level 3: SystemAgent (global coordination)
         └── Level 2: CoordinatorAgent (zone management)
                      └── Level 1: FieldAgent (local control)
```

Each level has distinct responsibilities:
- **FieldAgent**: Local sensing, actuation, state management
- **CoordinatorAgent**: Manages subordinate field agents, aggregates observations
- **SystemAgent**: Global objectives, system constraints, simulation orchestration

### 2. Feature-Based State

Composable `Feature` using metaclass auto-registration:

```python
class Feature(metaclass=FeatureMeta):
    """Base class for feature providers.

    Subclasses should:
    1. Use @dataclass(slots=True) decorator
    2. Define visibility as ClassVar[Sequence[str]]
    """
    visibility: ClassVar[Sequence[str]]

    def vector(self) -> np.ndarray:
        """Return all field values as a flat float32 numpy array."""
        ...

    def names(self) -> list[str]:
        """Return the names of all dataclass fields."""
        ...
```

Visibility levels: `"public"`, `"owner"`, `"upper_level"`, `"system"`

### 3. Protocol-Driven Coordination

Protocols are composed of two components:
- **CommunicationProtocol**: Defines WHAT to communicate
- **ActionProtocol**: Defines HOW to coordinate actions

```python
class Protocol(ABC):
    def __init__(
        self,
        communication_protocol: Optional[CommunicationProtocol] = None,
        action_protocol: Optional[ActionProtocol] = None
    ):
        ...

    def coordinate(
        self,
        coordinator_state: Any,
        coordinator_action: Optional[Any] = None,
        info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Dict[str, Any]], Dict[AgentID, Any]]:
        """Execute full coordination cycle (communication + action)."""
        ...
```

Built-in protocols:
- **Vertical**: `VectorDecompositionActionProtocol`, `BroadcastActionProtocol`
- **Horizontal**: `StateShareCommunicationProtocol`

### 4. Message Broker Abstraction

Synchronous pub/sub communication via `MessageBroker`:

```python
class MessageBroker(ABC):
    def publish(self, channel: str, message: Message) -> None:
        ...

    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[Message]:
        ...

    def create_channel(self, channel_name: str) -> None:
        ...
```

## Data Flow

### Centralized Mode (CTDE Training)

```
┌──────────────────────────────────────────────────────────────┐
│                      Proxy                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  State Cache (per-agent features + visibility filter)  │  │
│  └────────────────────────────────────────────────────────┘  │
│         ▲              ▲              ▲              ▲        │
│    set_state      get_obs       set_state      get_obs       │
│         │              │              │              │        │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│    │ System  │    │  Coord  │    │ Field 1 │    │ Field 2 │ │
│    │ Agent   │    │  Agent  │    │         │    │         │ │
│    └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│         │              │              ▲              ▲        │
│         └──────────────┘              │              │        │
│         action decomposition    upstream actions     │        │
│         via Protocol.coordinate()                    │        │
└──────────────────────────────────────────────────────────────┘
```

### Event-Driven Mode (Testing)

```
┌──────────────────────────────────────────────────────────────┐
│                    EventScheduler                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Priority Queue: (timestamp, priority, sequence)       │  │
│  │  Events: AGENT_TICK, ACTION_EFFECT, MESSAGE_DELIVERY,  │  │
│  │          SIMULATION, OBSERVATION_READY, ENV_UPDATE      │  │
│  └────────────────────────────────────────────────────────┘  │
│         │                                                    │
│    ┌────┴────────────────────────────────────────────┐       │
│    │         Proxy (singleton)                   │       │
│    │    State cache + visibility-filtered responses   │       │
│    └────┬──────────┬──────────┬──────────┬──────────┘       │
│         │          │          │          │                    │
│    ┌────┴───┐ ┌────┴───┐ ┌───┴────┐ ┌───┴────┐             │
│    │ System │ │ Coord  │ │Field 1 │ │Field 2 │             │
│    │ Agent  │ │ Agent  │ │        │ │        │             │
│    └────────┘ └────────┘ └────────┘ └────────┘             │
│    Messages routed via scheduler.schedule_message_delivery() │
└──────────────────────────────────────────────────────────────┘
```

## Extension Points

| Component | How to Extend |
|-----------|---------------|
| Agents | Subclass `FieldAgent`, `CoordinatorAgent`, or `SystemAgent` |
| Features | Subclass `Feature` (auto-registered via `FeatureMeta`) |
| Protocols | Implement `CommunicationProtocol` and/or `ActionProtocol` |
| Brokers | Implement `MessageBroker` interface |
| Environments | Subclass `HeronEnv` (extends `BaseEnv`) |

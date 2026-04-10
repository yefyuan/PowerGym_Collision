# Protocols & Coordination

How agents coordinate actions and share information in HERON.

Protocols are composable: pair any **CommunicationProtocol** with any **ActionProtocol**.

| Direction | What it does | Built-in protocols |
|-----------|-------------|-------------------|
| **Vertical** (parent -> child) | Coordinator decomposes action for subordinates | `VectorDecomposition`, `Broadcast` |
| **Horizontal** (peer <-> peer) | Agents share state with neighbors | `StateShare` with topology control |
| **Custom** | Your own logic | Subclass `ActionProtocol` / `CommunicationProtocol` |

## Quick Start

```bash
cd "examples/4. protocols_and_coordination"

python vertical_action_decomposition.py    # ~5 sec
python horizontal_state_sharing.py         # ~5 sec
python custom_protocol.py                  # ~5 sec
```

## File Structure

```
4. protocols_and_coordination/
├── vertical_action_decomposition.py   # VectorDecomposition and Broadcast
├── horizontal_state_sharing.py        # StateShare with topologies
├── custom_protocol.py                 # Build your own protocol
└── README.md
```

## What You'll Learn

### vertical_action_decomposition.py

| Concept | What's shown |
|---------|-------------|
| `VectorDecompositionActionProtocol` | Split coordinator's joint action by subordinate dims |
| `BroadcastActionProtocol` | Send same action (e.g., pricing signal) to all subordinates |
| `protocol.coordinate()` | Direct call to see decomposition mechanics |
| Coordinator action flow | Pass joint action via `env.step({"coordinator": action})` |

**Key takeaway**: Vertical protocols let a coordinator compute one joint action that the protocol automatically distributes. `VectorDecomposition` splits by dimension; `Broadcast` sends the same value to all.

### horizontal_state_sharing.py

| Concept | What's shown |
|---------|-------------|
| `StateShareCommunicationProtocol` | Share agent states with neighbors |
| Fully connected topology | Default: every agent sees all others |
| Ring topology | Each agent sees only adjacent neighbors |
| Star topology | One hub sees all; leaves see only the hub |
| `state_fields` filtering | Share only selected fields (hide internals) |
| `HorizontalProtocol` | Composed protocol: StateShare + NoActionCoordination |

**Key takeaway**: Horizontal protocols enable decentralized decision-making. Agents share state through configurable topologies but act independently.

### custom_protocol.py

| Concept | What's shown |
|---------|-------------|
| Custom `ActionProtocol` | `WeightedAllocationActionProtocol` -- proportional resource distribution |
| Custom `CommunicationProtocol` | `ThresholdAlertProtocol` -- send alerts when levels are low |
| Protocol composition | Combine custom communication + action into a single `Protocol` |
| Running through `env.step()` | Custom protocol works seamlessly with HERON's step loop |

**Key takeaway**: Build domain-specific coordination by subclassing `ActionProtocol` and/or `CommunicationProtocol`. The `Protocol` base class orchestrates both.

## Protocol Architecture

```
Protocol = CommunicationProtocol + ActionProtocol

Protocol.coordinate(state, action, subordinates)
  ├── 1. communication_protocol.compute_coordination_messages()
  │      → messages: {agent_id: {type, data, ...}}
  └── 2. action_protocol.compute_action_coordination()
         → actions: {agent_id: action_value}
```

### Built-in Components

```
CommunicationProtocol          ActionProtocol
├── NoCommunication            ├── NoActionCoordination
└── StateShareProtocol         ├── VectorDecomposition
    (topology, field filter)   └── BroadcastAction
```

### Pre-composed Protocols

| Protocol | Communication | Action | Use case |
|----------|--------------|--------|----------|
| `VerticalProtocol()` | NoCommunication | VectorDecomposition | Hierarchical control |
| `VerticalProtocol(action_protocol=Broadcast())` | NoCommunication | Broadcast | Global signals |
| `HorizontalProtocol()` | StateShare | NoActionCoordination | Peer coordination |
| `HorizontalProtocol(topology=ring)` | StateShare (ring) | NoActionCoordination | Local consensus |

## Vertical vs Horizontal

```
Vertical (coordinator-owned):          Horizontal (peer-to-peer):

   Coordinator                           Agent_A ←→ Agent_B
   ├── action=[a1,a2,a3]                    ↕           ↕
   │                                     Agent_C ←→ Agent_D
   ├──→ Sub_1 gets a1
   ├──→ Sub_2 gets a2                   Each agent shares state with
   └──→ Sub_3 gets a3                   neighbors and acts independently.

   Protocol decomposes the              Protocol computes messages;
   coordinator's joint action.          actions are None (decentralized).
```

## Next Steps

Now that you understand protocols, learn how to train policies:

- **Level 5: Training Algorithms** -- Policy ABC, IPPO vs MAPPO, RLlib integration.

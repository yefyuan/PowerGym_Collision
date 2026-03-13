# PowerGrid Architecture

The PowerGrid case study extends HERON for power systems control.

## Directory Structure

```
case_studies/power/
├── powergrid/                  # Python package
│   ├── agents/                 # Power-specific agents
│   │   ├── power_grid_agent.py # Coordinator with PandaPower integration
│   │   ├── device_agent.py     # Base for power device agents
│   │   ├── generator.py        # Dispatchable generator device
│   │   ├── storage.py          # Energy storage system (ESS)
│   │   ├── transformer.py      # Transformer with tap changer
│   │   └── proxy.py      # Extends heron.agents.proxy
│   │
│   ├── core/                   # Extensions to heron.core
│   │   ├── features/           # Power-specific features
│   │   │   ├── electrical.py   # P, Q, voltage features
│   │   │   ├── network.py      # Bus voltages, line flows
│   │   │   ├── storage.py      # SOC, energy capacity
│   │   │   └── ...
│   │   │
│   │   └── state/              # Power-specific state
│   │       └── state.py        # Device and grid state classes
│   │
│   ├── networks/               # IEEE/CIGRE test networks
│   │   ├── ieee13.py           # IEEE 13-bus feeder
│   │   ├── ieee34.py           # IEEE 34-bus feeder
│   │   ├── ieee123.py          # IEEE 123-bus feeder
│   │   └── cigre_lv.py         # CIGRE low-voltage network
│   │
│   ├── envs/                   # Power environments
│   │   ├── networked_grid_env.py      # Base networked environment
│   │   └── multi_agent_microgrids.py  # Multi-microgrid environment
│   │
│   ├── setups/                 # Environment setups
│   │   ├── loader.py           # Setup loading utilities
│   │   └── ieee34_ieee13/      # Example setup
│   │       ├── config.yml
│   │       └── data.pkl
│   │
│   ├── optimization/           # Power system optimization
│   │   └── misocp.py           # Mixed-integer SOCP solver
│   │
│   └── utils/                  # Power-specific utilities
│       ├── cost.py             # Cost functions
│       ├── safety.py           # Safety penalties
│       └── phase.py            # Phase utilities
│
├── examples/                   # Example scripts
└── tests/                      # Power grid tests
```

## Agent Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                  System Operator                         │
│                  (Optional top level)                    │
└─────────────────────────┬───────────────────────────────┘
                          │ Price Signals
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  PowerGridAgent │ │  PowerGridAgent │ │  PowerGridAgent │
│      (MG1)      │ │      (MG2)      │ │      (MG3)      │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │  Generator  │ │ │ │    ESS      │ │ │ │  Generator  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │    ESS      │ │ │ │   Solar     │ │ │ │    ESS      │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Data Flow

### During `step()`

```
1. Environment receives actions from all agents
   └── actions = {"MG1": array([...]), "MG2": array([...]), ...}

2. Each PowerGridAgent processes its action
   └── Distributes setpoints to devices via protocol
   └── Updates device states

3. Power flow calculation (PandaPower)
   └── Each microgrid runs power flow
   └── Updates electrical state (voltages, flows)

4. Reward and safety computation
   └── Cost = generation cost + trading cost
   └── Safety = voltage violations + thermal limits

5. Observations generated for each agent
   └── Based on centralized/distributed mode
```

## Integration with HERON

| HERON Component | PowerGrid Extension |
|-----------------|---------------------|
| `FieldAgent` | `DeviceAgent`, `Generator`, `ESS`, `Transformer` |
| `CoordinatorAgent` | `PowerGridAgent` |
| `Feature` | `ElectricalFeature`, `StorageFeature`, `ThermalFeature` |
| `Protocol` | `SetpointProtocol`, `PriceSignalProtocol` |
| `ParallelEnv` | `NetworkedGridEnv`, `MultiAgentMicrogrids` |

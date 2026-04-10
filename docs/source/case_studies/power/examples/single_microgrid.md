# Example 1: Single Microgrid

This example demonstrates basic usage with a single microgrid containing multiple devices controlled by a centralized GridAgent.

## What You'll Learn

- Creating device agents (Generator, ESS)
- Building a GridAgent to coordinate devices
- Using SetpointProtocol for direct control
- Running a simulation loop

## Architecture

```
GridAgent (Microgrid Controller)
├── Generator (Dispatchable power source)
└── ESS (Energy Storage System)
```

## Code

```python
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.networks.ieee13 import IEEE13Bus
from heron.protocols.vertical import SetpointProtocol

# Create IEEE 13-bus network
net = IEEE13Bus("MG1")

# Create GridAgent with devices
mg_agent = PowerGridAgent(
    net=net,
    grid_config={
        "name": "MG1",
        "base_power": 1.0,
        "devices": [
            {
                "type": "Generator",
                "name": "gen1",
                "device_state_config": {
                    "bus": "Bus 633",
                    "p_max_MW": 2.0,
                    "p_min_MW": 0.5,
                },
            },
            {
                "type": "ESS",
                "name": "ess1",
                "device_state_config": {
                    "bus": "Bus 634",
                    "e_capacity_MWh": 5.0,
                    "p_max_MW": 1.0,
                },
            },
        ],
    },
    protocol=SetpointProtocol(),
)
```

## Running the Example

```bash
cd case_studies/power
python examples/01_single_microgrid_basic.py
```

## Expected Output

```
[1] Creating environment...
    Possible agents: ['MG1']
    Action spaces: {'MG1': Dict(...)}

[2] Resetting environment...
    Initial observation shape for MG1: (32,)

[3] Running 24-hour simulation with random actions...
    Hour   Reward       Safety       Done
    ---------------------------------------------
    1       -10.25        0.00       False
    ...
    24      -15.30        0.50       True

[4] Simulation Summary:
    Total reward: -285.50
    Total safety violations: 2.30
```

## Key Concepts

### Device Configuration

Devices are configured via `grid_config["devices"]`:

```python
{
    "type": "Generator",  # Device class name
    "name": "gen1",       # Unique identifier
    "device_state_config": {
        "bus": "Bus 633",        # Connection bus
        "p_max_MW": 2.0,         # Max active power
        "p_min_MW": 0.5,         # Min active power
        "cost_curve_coefs": [0.02, 10.0, 0.0],  # Cost function
    },
}
```

### SetpointProtocol

The `SetpointProtocol` provides direct control over device setpoints:

```python
from heron.protocols.vertical import SetpointProtocol

protocol = SetpointProtocol()
# Actions are directly applied as setpoints to devices
```

## Next Steps

- Try [Multi-Microgrid P2P](multi_microgrid_p2p) for peer-to-peer trading
- Explore [MAPPO Training](mappo_training) for RL-based control

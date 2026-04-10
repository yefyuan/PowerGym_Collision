# Power Devices

The PowerGrid case study includes several device types.

## Generator

Dispatchable power source with cost curves and operational constraints.

```python
from powergrid.agents.generator import Generator

generator = Generator(
    agent_id="gen1",
    config={
        "bus": "Bus 633",
        "p_max_MW": 2.0,
        "p_min_MW": 0.5,
        "q_max_MVAr": 1.0,
        "q_min_MVAr": -1.0,
        "s_rated_MVA": 2.5,
        "startup_time_hr": 1.0,
        "shutdown_time_hr": 1.0,
        "cost_curve_coefs": [0.02, 10.0, 0.0],  # a*P^2 + b*P + c
    }
)
```

### Generator Features

| Feature | Description | Visibility |
|---------|-------------|------------|
| `electrical` | P, Q output | owner, coordinator |
| `power_limits` | P_min, P_max, Q_min, Q_max | owner, coordinator |
| `cost` | Operating cost | owner |

### Generator Actions

| Action | Type | Range |
|--------|------|-------|
| P setpoint | Continuous | [P_min, P_max] |
| Q setpoint | Continuous | [Q_min, Q_max] |
| On/Off | Discrete | {0, 1} |

## Energy Storage System (ESS)

Battery storage with state of charge management.

```python
from powergrid.agents.storage import ESS

ess = ESS(
    agent_id="ess1",
    config={
        "bus": "Bus 634",
        "e_capacity_MWh": 5.0,
        "soc_max": 0.9,
        "soc_min": 0.1,
        "p_max_MW": 1.0,       # Max discharge
        "p_min_MW": -1.0,      # Max charge (negative)
        "q_max_MVAr": 0.5,
        "q_min_MVAr": -0.5,
        "s_rated_MVA": 1.2,
        "init_soc": 0.5,
        "ch_eff": 0.95,        # Charging efficiency
        "dsc_eff": 0.95,       # Discharging efficiency
    }
)
```

### ESS Features

| Feature | Description | Visibility |
|---------|-------------|------------|
| `electrical` | P, Q output | owner, coordinator |
| `storage` | SOC, energy capacity | owner, coordinator |
| `power_limits` | Charge/discharge limits | owner |

### ESS Dynamics

```python
# SOC update equation
if P > 0:  # Discharging
    delta_E = P * dt / dsc_eff
else:      # Charging
    delta_E = P * dt * ch_eff

SOC_new = SOC - delta_E / E_capacity
```

## Transformer

Transformer with tap changer for voltage regulation.

```python
from powergrid.agents.transformer import Transformer

transformer = Transformer(
    agent_id="trafo1",
    config={
        "from_bus": "Bus 650",
        "to_bus": "Bus 632",
        "s_rated_MVA": 5.0,
        "tap_min": 0.9,
        "tap_max": 1.1,
        "tap_step": 0.0125,
        "init_tap": 1.0,
    }
)
```

### Transformer Actions

| Action | Type | Description |
|--------|------|-------------|
| Tap position | Discrete | Tap ratio adjustment |

## Custom Devices

Create custom devices by extending `DeviceAgent`:

```python
from powergrid.agents.device_agent import DeviceAgent

class WindTurbine(DeviceAgent):
    device_type = "wind"

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        # Initialize wind-specific features
        self.rated_power = config.get("rated_power_MW", 2.0)
        self.cut_in_speed = config.get("cut_in_speed", 3.0)
        self.rated_speed = config.get("rated_speed", 12.0)
        self.cut_out_speed = config.get("cut_out_speed", 25.0)

    def compute_power(self, wind_speed: float) -> float:
        """Compute power output based on wind speed."""
        if wind_speed < self.cut_in_speed:
            return 0.0
        elif wind_speed < self.rated_speed:
            # Cubic region
            return self.rated_power * (wind_speed / self.rated_speed) ** 3
        elif wind_speed < self.cut_out_speed:
            return self.rated_power
        else:
            return 0.0  # Cut out
```

## Device Registration

Devices are automatically created from `grid_config`:

```python
grid_config = {
    "name": "MG1",
    "devices": [
        {"type": "Generator", "name": "gen1", "device_state_config": {...}},
        {"type": "ESS", "name": "ess1", "device_state_config": {...}},
        {"type": "WindTurbine", "name": "wind1", "device_state_config": {...}},
    ]
}
```

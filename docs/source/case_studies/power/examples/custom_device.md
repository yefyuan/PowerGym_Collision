# Example 4: Custom Device

This example demonstrates how to create custom device agents for the power grid case study.

## What You'll Learn

- Extending `DeviceAgent` for custom devices
- Creating custom features with Feature
- Integrating custom devices into environments

## Creating a Custom Device

### Step 1: Define Custom Features

```python
from heron.core.feature import Feature
import numpy as np

class SolarFeature(Feature):
    """Feature for solar panel state."""

    def __init__(self, panel_area: float, efficiency: float):
        super().__init__(visibility=["owner", "coordinator"])
        self.panel_area = panel_area  # m^2
        self.efficiency = efficiency
        self.irradiance = 0.0  # W/m^2

    def vector(self) -> np.ndarray:
        power_output = self.panel_area * self.efficiency * self.irradiance / 1e6  # MW
        return np.array([power_output, self.irradiance], dtype=np.float32)

    def dim(self) -> int:
        return 2
```

### Step 2: Create Custom Device Agent

```python
from powergrid.agents.device_agent import DeviceAgent
from heron.core.state import FieldAgentState
from heron.core.action import Action
import numpy as np

class SolarPanel(DeviceAgent):
    """Solar panel device agent."""

    device_type = "solar"

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)

        # Create state with custom features
        self.state = FieldAgentState()
        self.solar_feature = SolarFeature(
            panel_area=config.get("panel_area", 100.0),
            efficiency=config.get("efficiency", 0.2),
        )
        self.state.add_feature("solar", self.solar_feature)

        # Solar panels have no controllable actions
        self.action = Action()
        self.action.set_specs(dim_c=0, range=(np.array([]), np.array([])))

    def update_irradiance(self, irradiance: float):
        """Update solar irradiance from external data."""
        self.solar_feature.irradiance = irradiance

    @property
    def power_output(self) -> float:
        """Get current power output in MW."""
        return self.solar_feature.vector()[0]
```

### Step 3: Integrate into Environment

```python
from powergrid.envs.networked_grid_env import NetworkedGridEnv

class SolarMicrogridEnv(NetworkedGridEnv):
    def _build_net(self):
        # ... create network and grid agent ...

        # Add custom solar device
        solar = SolarPanel(
            agent_id="solar_1",
            config={
                "bus": "Bus 634",
                "panel_area": 500.0,
                "efficiency": 0.22,
            }
        )

        # Register with grid agent
        mg_agent.add_device(solar)

        return net

    def _apply_external_data(self, timestep: int):
        """Apply time-varying data."""
        # Update solar irradiance from dataset
        irradiance = self.dataset["irradiance"][timestep]
        self.agent_dict["MG1"].devices["solar_1"].update_irradiance(irradiance)
```

## Running the Example

```bash
cd case_studies/power
python examples/04_custom_device.py
```

## Key Concepts

### Device State Configuration

Devices are configured via `device_state_config`:

```python
{
    "type": "SolarPanel",  # Custom device class
    "name": "solar_1",
    "device_state_config": {
        "bus": "Bus 634",
        "panel_area": 500.0,
        "efficiency": 0.22,
    },
}
```

### Feature Visibility

Control who can observe device features:

```python
# Only owner (device itself) can see
SolarFeature(visibility=["owner"])

# Owner and coordinator can see
SolarFeature(visibility=["owner", "coordinator"])

# Everyone can see
SolarFeature(visibility=["global"])
```

## Next Steps

- Try [MAPPO Training](mappo_training) to train policies for custom devices
- See [Distributed Mode](distributed_mode) for message-based coordination

# Example 3: Price Coordination

This example demonstrates hierarchical coordination using price signals from a system operator to multiple microgrids.

## What You'll Learn

- Using PriceSignalProtocol for market-based coordination
- Implementing hierarchical control with System → Coordinator → Field agents
- Price-responsive device operation

## Architecture

```
        ┌─────────────────┐
        │ System Operator │  (Price Signal)
        └────────┬────────┘
                 │ $price
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐       ┌─────────┐
   │   MG1   │       │   MG2   │
   │ respond │       │ respond │
   │ to price│       │ to price│
   └─────────┘       └─────────┘
```

## Code

```python
from heron.protocols.vertical import PriceSignalProtocol

# Create price signal protocol
protocol = PriceSignalProtocol(initial_price=50.0)

# Price signals are broadcast to all subordinate agents
# Agents respond by adjusting their dispatch based on price
```

## Price-Responsive Behavior

Devices respond to price signals:

```python
# High price → Reduce consumption, increase generation
# Low price  → Increase consumption, decrease generation

def respond_to_price(price: float, device_state):
    if device_state.is_generator:
        # Generators increase output when price is high
        target_p = device_state.p_max if price > 60 else device_state.p_min
    elif device_state.is_storage:
        # Storage charges when price is low, discharges when high
        if price < 40:
            target_p = -device_state.p_max  # Charge
        elif price > 70:
            target_p = device_state.p_max   # Discharge
        else:
            target_p = 0
    return target_p
```

## Running the Example

```bash
cd case_studies/power
python examples/03_price_coordination.py
```

## Key Concepts

### PriceSignalProtocol

```python
from heron.protocols.vertical import PriceSignalProtocol

protocol = PriceSignalProtocol(
    initial_price=50.0,  # Starting price $/MWh
)

# System operator sets price based on system conditions
# - High demand → High price
# - Low demand → Low price
```

### Market-Based Coordination

Price signals provide implicit coordination without direct control:

| Approach | Control Type | Scalability |
|----------|--------------|-------------|
| Setpoint | Direct | Limited |
| Price Signal | Indirect | High |

## Next Steps

- Try [Custom Device](custom_device) to create domain-specific devices
- See [MAPPO Training](mappo_training) for learning-based price response

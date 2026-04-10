# Test Networks

The PowerGrid case study includes standard IEEE test systems implemented with PandaPower.

## IEEE 13-Bus

Small radial distribution feeder for basic testing.

```python
from powergrid.networks.ieee13 import IEEE13Bus

net = IEEE13Bus(name="MG1")
```

### Specifications

| Property | Value |
|----------|-------|
| Buses | 13 |
| Voltage | 4.16 kV |
| Topology | Radial |
| Load | ~3.5 MW |

### Diagram

```
    650 (Substation)
     |
    632
   / | \
 633 645 671
  |       |
 634     692
          |
         675
```

## IEEE 34-Bus

Larger radial feeder with voltage regulators.

```python
from powergrid.networks.ieee34 import IEEE34Bus

net = IEEE34Bus(name="MG2")
```

### Specifications

| Property | Value |
|----------|-------|
| Buses | 34 |
| Voltage | 24.9 kV / 4.16 kV |
| Topology | Radial with laterals |
| Load | ~1.8 MW |
| Regulators | 2 |

## IEEE 123-Bus

Large-scale test feeder for comprehensive studies.

```python
from powergrid.networks.ieee123 import IEEE123Bus

net = IEEE123Bus(name="MG3")
```

### Specifications

| Property | Value |
|----------|-------|
| Buses | 123 |
| Voltage | 4.16 kV |
| Topology | Complex radial |
| Load | ~3.5 MW |
| Switches | 11 |
| Regulators | 4 |

## CIGRE Low-Voltage

European low-voltage benchmark network.

```python
from powergrid.networks.cigre_lv import CIGRE_LV

net = CIGRE_LV(name="LV1")
```

### Specifications

| Property | Value |
|----------|-------|
| Buses | 44 |
| Voltage | 0.4 kV |
| Topology | Radial residential |
| Load | ~200 kW |

## Using Networks

### Create Network for Environment

```python
from powergrid.envs.networked_grid_env import NetworkedGridEnv
from powergrid.networks.ieee13 import IEEE13Bus

class MyEnv(NetworkedGridEnv):
    def _build_net(self):
        net = IEEE13Bus("MG1")
        # Add devices, configure agents
        return net
```

### Access Network Properties

```python
# PandaPower network object
net = IEEE13Bus("MG1")

# Access buses
print(net.bus)

# Access lines
print(net.line)

# Access loads
print(net.load)

# Run power flow
import pandapower as pp
pp.runpp(net)

# Access results
print(net.res_bus)  # Voltage results
print(net.res_line)  # Line flow results
```

### Multi-Network Setup

```python
from powergrid.networks.ieee34 import IEEE34Bus
from powergrid.networks.ieee13 import IEEE13Bus

# Create multiple networks for multi-agent environment
networks = {
    "MG1": IEEE34Bus("MG1"),
    "MG2": IEEE13Bus("MG2"),
    "MG3": IEEE13Bus("MG3"),
}
```

## Custom Networks

Create custom networks using PandaPower:

```python
import pandapower as pp

def create_custom_network(name: str):
    net = pp.create_empty_network(name=name)

    # Add buses
    b0 = pp.create_bus(net, vn_kv=20.0, name="Substation")
    b1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=20.0, name="Bus 2")

    # Add external grid
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)

    # Add lines
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1,
        length_km=1.0, r_ohm_per_km=0.1,
        x_ohm_per_km=0.1, c_nf_per_km=0,
        max_i_ka=1.0
    )

    # Add loads
    pp.create_load(net, bus=b1, p_mw=0.5, q_mvar=0.1)

    return net
```

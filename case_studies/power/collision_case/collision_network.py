"""Network builder for collision detection using IEEE 34 + IEEE 13 topology.

Recreates the exact network structure from the original collision experiment.
"""

import pandapower as pp
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus


def create_collision_network():
    """Create IEEE 34 + 3x IEEE 13 network for collision experiments.
    
    Network topology:
    - Main grid: IEEE 34-bus (DSO)
    - Microgrid 1: IEEE 13-bus connected at Bus 822
    - Microgrid 2: IEEE 13-bus connected at Bus 848  
    - Microgrid 3: IEEE 13-bus connected at Bus 856
    
    Returns:
        Pandapower network with full topology
    """
    # Create main IEEE 34-bus grid
    net = IEEE34Bus('DSO')
    
    # Create and merge IEEE 13-bus microgrids
    # Microgrid 1
    mg1_net = IEEE13Bus('MG1')
    mg1_net.ext_grid.in_service = False  # Disable local grid
    net, mg1_index = pp.merge_nets(net, mg1_net, validate=False, 
                                    return_net2_reindex_lookup=True)
    
    # Connect MG1 to DSO at Bus 822
    dso_bus_822 = pp.get_element_index(net, 'bus', 'DSO Bus 822')
    mg1_bus_650 = mg1_index['bus'][pp.get_element_index(mg1_net, 'bus', 'MG1 Bus 650')]
    pp.fuse_buses(net, mg1_bus_650, dso_bus_822)
    
    # Microgrid 2
    mg2_net = IEEE13Bus('MG2')
    mg2_net.ext_grid.in_service = False
    net, mg2_index = pp.merge_nets(net, mg2_net, validate=False,
                                    return_net2_reindex_lookup=True)
    
    # Connect MG2 to DSO at Bus 848
    dso_bus_848 = pp.get_element_index(net, 'bus', 'DSO Bus 848')
    mg2_bus_650 = mg2_index['bus'][pp.get_element_index(mg2_net, 'bus', 'MG2 Bus 650')]
    pp.fuse_buses(net, mg2_bus_650, dso_bus_848)
    
    # Microgrid 3
    mg3_net = IEEE13Bus('MG3')
    mg3_net.ext_grid.in_service = False
    net, mg3_index = pp.merge_nets(net, mg3_net, validate=False,
                                    return_net2_reindex_lookup=True)
    
    # Connect MG3 to DSO at Bus 856
    dso_bus_856 = pp.get_element_index(net, 'bus', 'DSO Bus 856')
    mg3_bus_650 = mg3_index['bus'][pp.get_element_index(mg3_net, 'bus', 'MG3 Bus 650')]
    pp.fuse_buses(net, mg3_bus_650, dso_bus_856)
    
    # Scale loads to 20% to match original experiment
    load_scale = 0.2
    for mg_prefix in ['MG1', 'MG2', 'MG3']:
        mg_loads = net.load[net.load.name.str.contains(mg_prefix)]
        net.load.loc[mg_loads.index, 'scaling'] = load_scale
    
    # Run initial power flow to verify network
    try:
        pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
        print(f"✓ Network created successfully: {len(net.bus)} buses, {len(net.line)} lines")
    except Exception as e:
        print(f"⚠ Warning: Initial power flow failed: {e}")
    
    return net


def get_microgrid_devices(net, mg_id):
    """Get device information for a specific microgrid.
    
    Args:
        net: Pandapower network
        mg_id: Microgrid ID (e.g., 'MG1')
        
    Returns:
        Dict with device bus indices and names
    """
    devices = {
        'storage': [],
        'sgen': [],
        'buses': [],
        'lines': []
    }
    
    # Find all buses for this microgrid
    mg_buses = net.bus[net.bus.name.str.contains(f"{mg_id} Bus")]
    devices['buses'] = mg_buses.index.tolist()
    
    # Find all lines for this microgrid  
    mg_lines = net.line[net.line.name.str.contains(f"{mg_id} Line")]
    devices['lines'] = mg_lines.index.tolist()
    
    # Storage devices (to be added dynamically)
    # Sgen devices (to be added dynamically)
    
    return devices


if __name__ == '__main__':
    """Test network creation."""
    net = create_collision_network()
    
    print("\n=== Network Statistics ===")
    print(f"Total buses: {len(net.bus)}")
    print(f"Total lines: {len(net.line)}")
    print(f"Total loads: {len(net.load)}")
    print(f"Total transformers: {len(net.trafo)}")
    
    print("\n=== Microgrid Devices ===")
    for mg_id in ['MG1', 'MG2', 'MG3']:
        devices = get_microgrid_devices(net, mg_id)
        print(f"{mg_id}: {len(devices['buses'])} buses, {len(devices['lines'])} lines")
    
    print("\n=== Power Flow Results ===")
    print(f"Voltage range: {net.res_bus.vm_pu.min():.3f} - {net.res_bus.vm_pu.max():.3f} p.u.")
    print(f"Max line loading: {net.res_line.loading_percent.max():.1f}%")

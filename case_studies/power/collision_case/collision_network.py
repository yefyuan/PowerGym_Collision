"""IEEE 34-bus (DSO) + 3× IEEE 13-bus interconnection (Collision experiment topology)."""

import numpy as np
import pandapower as pp

from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus


def create_collision_ieee_topology(load_scale: float = 0.2) -> pp.pandapowerNet:
    """Merge IEEE34 (DSO) with three IEEE13 subnets at buses 822, 848, 856.

    Matches the topology used in the original Collision ``ieee34_ieee13`` case.
    Applies initial ``load_scale`` to local loads (time-varying factors applied in env).
    """
    net = IEEE34Bus("DSO")
    # Keep IEEE34 defaults (slack 1.05 and max-boost regulators) to match the
    # original Collision case dynamics. Voltage violations are meant to occur,
    # and learning should reduce their frequency under penalties.
    dso_loads = net.load[net.load.name.str.contains("DSO")]
    net.load.loc[dso_loads.index, "scaling"] = load_scale

    def _merge_mg(subnet, bus_main: str, mg_id: str):
        nonlocal net
        subnet.ext_grid.in_service = False
        sub_root = pp.get_element_index(subnet, "bus", f"{mg_id} Bus 650")
        net, lookup = pp.merge_nets(net, subnet, validate=False, return_net2_reindex_lookup=True)
        dso_bus = pp.get_element_index(net, "bus", f"DSO {bus_main}")
        sub_here = lookup["bus"][sub_root]
        pp.fuse_buses(net, sub_here, dso_bus)
        return net

    for mg_id, bus_point in [
        ("MG1", "Bus 822"),
        ("MG2", "Bus 848"),
        ("MG3", "Bus 856"),
    ]:
        sn = IEEE13Bus(mg_id)
        mg_loads = sn.load[sn.load.name.str.contains(mg_id)]
        sn.load.loc[mg_loads.index, "scaling"] = load_scale
        net = _merge_mg(sn, bus_point, mg_id)

    return net


def _circle_q_bounds(max_p_mw: float, sn_mva: float) -> tuple[float, float]:
    lim = float(np.sqrt(max(0.0, sn_mva**2 - max_p_mw**2)))
    return -lim, lim


def attach_collision_devices(
    net: pp.pandapowerNet,
    mg_id: str,
    dg_max_p_mw: float,
) -> None:
    """Add ESS / DG / PV / WT at IEEE13 bus locations (names = agent_id strings)."""
    bus_645 = pp.get_element_index(net, "bus", f"{mg_id} Bus 645")
    bus_675 = pp.get_element_index(net, "bus", f"{mg_id} Bus 675")
    bus_652 = pp.get_element_index(net, "bus", f"{mg_id} Bus 652")
    q_ess_lo, q_ess_hi = _circle_q_bounds(0.5, 1.0)
    q_dg_lo, q_dg_hi = _circle_q_bounds(dg_max_p_mw, 1.0)
    q_r_lo, q_r_hi = _circle_q_bounds(0.1, 1.0)

    pp.create_storage(
        net,
        bus=bus_645,
        p_mw=0.0,
        max_e_mwh=2.0,
        soc_percent=50.0,
        min_e_mwh=0.2,
        name=f"{mg_id}_ESS1",
        sn_mva=1.0,
        max_p_mw=0.5,
        min_p_mw=-0.5,
        min_q_mvar=q_ess_lo,
        max_q_mvar=q_ess_hi,
        q_mvar=0.0,
        controllable=True,
    )
    pp.create_sgen(
        net,
        bus=bus_675,
        p_mw=0.3,
        q_mvar=0.0,
        sn_mva=1.0,
        name=f"{mg_id}_DG1",
        max_p_mw=dg_max_p_mw,
        min_p_mw=0.0,
        min_q_mvar=q_dg_lo,
        max_q_mvar=q_dg_hi,
        type="DG",
        controllable=True,
    )
    pp.create_sgen(
        net,
        bus=bus_652,
        p_mw=0.05,
        q_mvar=0.0,
        sn_mva=1.0,
        name=f"{mg_id}_PV1",
        max_p_mw=0.1,
        min_p_mw=0.0,
        min_q_mvar=q_r_lo,
        max_q_mvar=q_r_hi,
        type="PV",
        controllable=True,
    )
    pp.create_sgen(
        net,
        bus=bus_645,
        p_mw=0.05,
        q_mvar=0.0,
        sn_mva=1.0,
        name=f"{mg_id}_WT1",
        max_p_mw=0.1,
        min_p_mw=0.0,
        min_q_mvar=q_r_lo,
        max_q_mvar=q_r_hi,
        type="WT",
        controllable=True,
    )


def create_collision_network():
    """Backward-compatible name: full IEEE topology without per-MG devices."""
    return create_collision_ieee_topology(load_scale=0.2)


def get_microgrid_devices(net, mg_id: str):
    """Return indices/names for devices belonging to one microgrid prefix."""
    mask = lambda df: df.name.str.startswith(f"{mg_id}_")
    return {
        "storage": net.storage[mask(net.storage)],
        "sgen": net.sgen[mask(net.sgen)],
        "load": net.load[net.load.name.str.contains(mg_id)],
    }

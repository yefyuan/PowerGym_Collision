"""Horizontal State Sharing -- how peers share information.

Horizontal protocols handle Peer <-> Peer coordination.
Unlike vertical protocols (action decomposition), horizontal protocols
enable agents to observe neighbor states for decentralized decisions.

This script demonstrates:
1. StateShareCommunicationProtocol -- the communication layer
2. Topology control -- fully connected, ring, star, custom
3. State field filtering -- share only selected features
4. HorizontalProtocol -- the composed protocol

Domain: Temperature sensors in a building.
  - Sensors share readings with neighbors to detect anomalies.
  - Different topologies model different network architectures.

Usage:
    cd "examples/4. protocols_and_coordination"
    python horizontal_state_sharing.py
"""

from typing import Any, Dict

from heron.protocols.horizontal import (
    HorizontalProtocol,
    StateShareCommunicationProtocol,
)


# ---------------------------------------------------------------------------
# 1. Simulated agent states (standalone, no env needed)
# ---------------------------------------------------------------------------

def make_sensor_states(n: int = 4) -> Dict[str, Dict[str, Any]]:
    """Create mock sensor states for protocol demonstrations.

    Returns:
        Dict mapping sensor_id -> state dict with temperature/humidity fields
    """
    return {
        f"sensor_{i}": {
            "temperature": 20.0 + i * 2.5,     # 20, 22.5, 25, 27.5
            "humidity": 40.0 + i * 5.0,         # 40, 45, 50, 55
            "calibration": 0.1 * (i + 1),       # 0.1, 0.2, 0.3, 0.4 (internal)
        }
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# 2. Topology demonstrations
# ---------------------------------------------------------------------------

def demo_fully_connected():
    """Default: every agent sees every other agent."""
    print("=" * 60)
    print("Demo 1: Fully Connected Topology (default)")
    print("=" * 60)
    print("  Every sensor sees all other sensors\n")

    states = make_sensor_states(4)
    protocol = StateShareCommunicationProtocol()  # No topology = fully connected

    messages = protocol.compute_coordination_messages(
        sender_state=None,          # Not used in horizontal protocols
        receiver_states=states,
    )

    for agent_id in sorted(messages):
        neighbors = list(messages[agent_id]["neighbors"].keys())
        print(f"  {agent_id} sees: {neighbors}")

    # Show what sensor_0 actually receives
    print(f"\n  sensor_0's neighbor data:")
    for neighbor_id, state in messages["sensor_0"]["neighbors"].items():
        print(f"    {neighbor_id}: temp={state['temperature']}, humidity={state['humidity']}")


def demo_ring_topology():
    """Ring: each agent sees only its immediate neighbors."""
    print("\n" + "=" * 60)
    print("Demo 2: Ring Topology")
    print("=" * 60)
    print("  Each sensor sees only adjacent sensors (0-1-2-3-0)\n")

    states = make_sensor_states(4)
    ids = sorted(states.keys())

    # Build ring: sensor_i sees sensor_{i-1} and sensor_{i+1}
    ring = {}
    for i, aid in enumerate(ids):
        left = ids[(i - 1) % len(ids)]
        right = ids[(i + 1) % len(ids)]
        ring[aid] = [left, right]

    protocol = StateShareCommunicationProtocol(topology=ring)
    messages = protocol.compute_coordination_messages(
        sender_state=None,
        receiver_states=states,
    )

    for agent_id in sorted(messages):
        neighbors = list(messages[agent_id]["neighbors"].keys())
        print(f"  {agent_id} sees: {neighbors}")

    print(f"\n  sensor_1's neighbor data (ring):")
    for neighbor_id, state in messages["sensor_1"]["neighbors"].items():
        print(f"    {neighbor_id}: temp={state['temperature']}")


def demo_star_topology():
    """Star: one hub sees all, others see only the hub."""
    print("\n" + "=" * 60)
    print("Demo 3: Star Topology")
    print("=" * 60)
    print("  sensor_0 is the hub; others see only the hub\n")

    states = make_sensor_states(4)
    ids = sorted(states.keys())
    hub = ids[0]
    leaves = ids[1:]

    star = {hub: leaves}  # Hub sees all leaves
    for leaf in leaves:
        star[leaf] = [hub]  # Each leaf sees only the hub

    protocol = StateShareCommunicationProtocol(topology=star)
    messages = protocol.compute_coordination_messages(
        sender_state=None,
        receiver_states=states,
    )

    for agent_id in sorted(messages):
        neighbors = list(messages[agent_id]["neighbors"].keys())
        print(f"  {agent_id} sees: {neighbors}")


def demo_state_field_filtering():
    """Share only selected fields (e.g., temperature but not calibration)."""
    print("\n" + "=" * 60)
    print("Demo 4: State Field Filtering")
    print("=" * 60)
    print("  Share only 'temperature', hide 'calibration' and 'humidity'\n")

    states = make_sensor_states(4)

    # Without filtering -- shares all fields
    protocol_all = StateShareCommunicationProtocol()
    messages_all = protocol_all.compute_coordination_messages(
        sender_state=None,
        receiver_states=states,
    )

    # With filtering -- shares only temperature
    protocol_filtered = StateShareCommunicationProtocol(
        state_fields=["temperature"]
    )
    messages_filtered = protocol_filtered.compute_coordination_messages(
        sender_state=None,
        receiver_states=states,
    )

    print("  Without filtering (sensor_0 sees sensor_1):")
    s0_sees_s1_all = messages_all["sensor_0"]["neighbors"]["sensor_1"]
    print(f"    {s0_sees_s1_all}")

    print(f"\n  With state_fields=['temperature'] (sensor_0 sees sensor_1):")
    s0_sees_s1_filtered = messages_filtered["sensor_0"]["neighbors"]["sensor_1"]
    print(f"    {s0_sees_s1_filtered}")


def demo_horizontal_protocol():
    """HorizontalProtocol combines communication + no-action coordination."""
    print("\n" + "=" * 60)
    print("Demo 5: HorizontalProtocol (composed)")
    print("=" * 60)
    print("  Communication: state sharing with ring topology")
    print("  Action: NoActionCoordination (agents decide independently)\n")

    states = make_sensor_states(4)
    ids = sorted(states.keys())
    ring = {
        ids[i]: [ids[(i - 1) % len(ids)], ids[(i + 1) % len(ids)]]
        for i in range(len(ids))
    }

    protocol = HorizontalProtocol(
        state_fields=["temperature"],
        topology=ring,
    )

    # Show the two components working together.
    # Note: We call sub-protocols directly because StateShareCommunicationProtocol
    # uses parameter name `receiver_states` while the base class ABC declares
    # `receiver_infos`. Protocol.coordinate() passes the keyword `receiver_infos`,
    # causing a TypeError. This is a framework bug to fix in horizontal.py.

    # 1. Communication: compute neighbor state messages
    messages = protocol.communication_protocol.compute_coordination_messages(
        sender_state=None,
        receiver_states=states,
    )

    print("  Messages (neighbor states):")
    for agent_id in sorted(messages):
        neighbor_temps = {
            nid: nstate.get("temperature", "?")
            for nid, nstate in messages[agent_id].get("neighbors", {}).items()
        }
        print(f"    {agent_id} receives: {neighbor_temps}")

    # 2. Action: no coordination (all None)
    actions = protocol.action_protocol.compute_action_coordination(
        coordinator_action=None,
        info_for_subordinates=states,
    )

    print(f"\n  Actions (all None -- agents act independently):")
    for agent_id in sorted(actions):
        print(f"    {agent_id} -> {actions[agent_id]}")

    print("\n  Key: Horizontal protocols share state but don't decompose actions.")
    print("  Each agent uses shared neighbor info to make its own decision.")


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    demo_fully_connected()
    demo_ring_topology()
    demo_star_topology()
    demo_state_field_filtering()
    demo_horizontal_protocol()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  StateShareCommunicationProtocol:
    Each agent receives neighbor states based on topology
    Fully connected (default), ring, star, or custom adjacency

  State field filtering:
    state_fields=["temperature"]  ->  hide internal fields

  HorizontalProtocol = StateShare + NoActionCoordination:
    Agents share info but act independently
    Useful for consensus, anomaly detection, distributed control

  Vertical vs Horizontal:
    Vertical: parent decomposes action -> subordinates execute
    Horizontal: peers share state -> each decides independently
""")
    print("Done.")


if __name__ == "__main__":
    main()

"""Vertical Action Decomposition -- how coordinators distribute actions.

Vertical protocols handle the Parent -> Subordinate action flow.
A coordinator computes a joint action, and the protocol decomposes it
into per-subordinate actions.

This script demonstrates:
1. VectorDecompositionActionProtocol -- splits joint action by subordinate dims
2. BroadcastActionProtocol -- sends same action to all subordinates
3. Direct protocol.coordinate() calls to show the mechanics

Domain: Power grid with a grid operator and three generators.
  - The grid operator (coordinator) computes a joint action vector.
  - The protocol distributes power setpoints to generators.
  - Each generator maps its action [-1, 1] to power output [0, capacity].

Usage:
    cd "examples/4. protocols_and_coordination"
    python vertical_action_decomposition.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.simple import SimpleEnv
from heron.protocols.vertical import (
    VerticalProtocol,
    VectorDecompositionActionProtocol,
    BroadcastActionProtocol,
)


# ---------------------------------------------------------------------------
# 1. Features and agents
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GeneratorFeature(Feature):
    """Generator power output and capacity."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    power: float = 0.0          # current output (MW)
    capacity: float = 100.0     # max capacity (MW)


class Generator(FieldAgent):
    """Generator that receives power setpoints from a grid operator."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        if "power" in kwargs:
            self.state.features["GeneratorFeature"].set_values(power=kwargs["power"])

    def apply_action(self) -> None:
        feat = self.state.features["GeneratorFeature"]
        # Map action [-1, 1] to power [0, capacity]
        power = (self.action.c[0] + 1.0) / 2.0 * feat.capacity
        feat.set_values(power=float(np.clip(power, 0.0, feat.capacity)))

    def compute_local_reward(self, local_state: dict) -> float:
        if "GeneratorFeature" not in local_state:
            return 0.0
        vec = local_state["GeneratorFeature"]
        power, capacity = float(vec[0]), float(vec[1])
        # Target: 70% utilization
        target = 0.7 * capacity
        return -abs(power - target) / capacity


# ---------------------------------------------------------------------------
# 2. Simulation function
# ---------------------------------------------------------------------------

def grid_simulation(agent_states: dict) -> dict:
    """Simple grid physics: generators lose 5% per step (efficiency loss)."""
    for aid, features in agent_states.items():
        if "GeneratorFeature" in features:
            power = features["GeneratorFeature"]["power"]
            features["GeneratorFeature"]["power"] = power * 0.95
    return agent_states


# ---------------------------------------------------------------------------
# 3. Protocol mechanics (direct calls)
# ---------------------------------------------------------------------------

def demo_protocol_mechanics():
    """Show protocol.coordinate() calls directly to visualize decomposition."""
    print("=" * 60)
    print("Part 1: Protocol Mechanics (direct coordinate() calls)")
    print("=" * 60)

    # Create agents to register their action dimensions
    gen_a = Generator(agent_id="gen_a", features=[GeneratorFeature()])
    gen_b = Generator(agent_id="gen_b", features=[GeneratorFeature()])
    gen_c = Generator(agent_id="gen_c", features=[GeneratorFeature()])
    subordinates = {"gen_a": gen_a, "gen_b": gen_b, "gen_c": gen_c}

    # --- VectorDecomposition ---
    print("\n  VectorDecompositionActionProtocol:")
    print("  Coordinator outputs [g1, g2, g3] -> each gets its slice")
    vec_protocol = VerticalProtocol()
    vec_protocol.register_subordinates(subordinates)

    joint_action = np.array([0.7, -0.3, 0.5])
    messages, actions = vec_protocol.coordinate(
        coordinator_state=None,
        coordinator_action=joint_action,
        info_for_subordinates={aid: None for aid in subordinates},
    )

    print(f"\n    Input:  joint_action = {joint_action}")
    print(f"    Output:")
    for sub_id, action in actions.items():
        print(f"      {sub_id} -> {action}")

    # --- Broadcast ---
    print("\n  BroadcastActionProtocol:")
    print("  Coordinator outputs single signal -> all get the same value")
    bc_protocol = VerticalProtocol(action_protocol=BroadcastActionProtocol())

    signal = np.array([0.5])
    messages, actions = bc_protocol.coordinate(
        coordinator_state=None,
        coordinator_action=signal,
        info_for_subordinates={aid: None for aid in subordinates},
    )

    print(f"\n    Input:  signal = {signal}")
    print(f"    Output:")
    for sub_id, action in actions.items():
        print(f"      {sub_id} -> {action}")


# ---------------------------------------------------------------------------
# 4. Running through env.step()
# ---------------------------------------------------------------------------

def demo_vector_decomposition_env():
    """VectorDecomposition through env.step(): joint action split by dims."""
    print("\n" + "=" * 60)
    print("Part 2: VectorDecomposition through env.step()")
    print("=" * 60)
    print("Coordinator sends joint action -> protocol splits -> generators apply\n")

    # Note: env.reset() resets features to dataclass defaults (capacity=100.0)
    gen_a = Generator(agent_id="gen_a", features=[GeneratorFeature()])
    gen_b = Generator(agent_id="gen_b", features=[GeneratorFeature()])
    gen_c = Generator(agent_id="gen_c", features=[GeneratorFeature()])

    coordinator = CoordinatorAgent(
        agent_id="grid_op",
        subordinates={"gen_a": gen_a, "gen_b": gen_b, "gen_c": gen_c},
        protocol=VerticalProtocol(),  # default: VectorDecomposition
    )

    env = SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=grid_simulation,
        env_id="vector_decomp_demo",
    )

    obs, _ = env.reset(seed=0)

    # Build a joint action: [gen_a_setpoint, gen_b_setpoint, gen_c_setpoint]
    joint_action = Action()
    joint_action.set_specs(dim_c=3, range=(np.array([-1.0] * 3), np.array([1.0] * 3)))

    print("  Sending joint_action=[0.4, 0.6, 0.8] for 5 steps:")
    for step in range(5):
        joint_action.set_values(c=[0.4, 0.6, 0.8])
        obs, rewards, _, _, _ = env.step({"grid_op": joint_action})

        if step == 0 or (step + 1) % 2 == 0:
            print(f"\n    Step {step + 1}:")
            for aid in ["gen_a", "gen_b", "gen_c"]:
                if aid in obs:
                    vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
                    print(f"      {aid}: power={vec[0]:6.1f}MW (cap={vec[1]:.0f}), reward={rewards.get(aid, 0):.3f}")

    print("\n  Result: Each generator got its own slice of the 3-dim action vector")


def demo_broadcast_env():
    """Broadcast through env.step(): same action sent to all."""
    print("\n" + "=" * 60)
    print("Part 3: BroadcastActionProtocol through env.step()")
    print("=" * 60)
    print("Coordinator sends pricing signal -> all generators receive it\n")

    gen_a = Generator(agent_id="gen_a", features=[GeneratorFeature()])
    gen_b = Generator(agent_id="gen_b", features=[GeneratorFeature()])
    gen_c = Generator(agent_id="gen_c", features=[GeneratorFeature()])

    broadcast_protocol = VerticalProtocol(
        action_protocol=BroadcastActionProtocol()
    )

    coordinator = CoordinatorAgent(
        agent_id="grid_op",
        subordinates={"gen_a": gen_a, "gen_b": gen_b, "gen_c": gen_c},
        protocol=broadcast_protocol,
    )

    env = SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=grid_simulation,
        env_id="broadcast_demo",
    )

    obs, _ = env.reset(seed=0)

    # Single-dim action (same signal to all generators)
    signal_action = Action()
    signal_action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))

    print("  Sending signal_action=[0.5] (same to all) for 5 steps:")
    for step in range(5):
        signal_action.set_values(c=[0.5])
        obs, rewards, _, _, _ = env.step({"grid_op": signal_action})

        if step == 0 or (step + 1) % 2 == 0:
            print(f"\n    Step {step + 1}:")
            for aid in ["gen_a", "gen_b", "gen_c"]:
                if aid in obs:
                    vec = obs[aid].vector() if hasattr(obs[aid], "vector") else obs[aid]
                    print(f"      {aid}: power={vec[0]:6.1f}MW (cap={vec[1]:.0f})")

    print("\n  Result: All generators received the SAME action [0.5]")
    print("  Contrast with VectorDecomposition where each gets a different slice")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    demo_protocol_mechanics()
    demo_vector_decomposition_env()
    demo_broadcast_env()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  VectorDecomposition (default VerticalProtocol):
    Coordinator outputs [a1, a2, ..., aN]
    Protocol splits by subordinate action dimensions
    Each subordinate gets its own slice

  Broadcast:
    Coordinator outputs single action / signal
    Every subordinate receives the same value
    Useful for pricing signals, global commands

  Both are pluggable via:
    VerticalProtocol(action_protocol=BroadcastActionProtocol())
""")
    print("Done.")


if __name__ == "__main__":
    main()

"""Mixed Action Spaces -- continuous, discrete, and hybrid actions.

HERON's Action class supports three action space types:
  - Continuous: power setpoints, velocity targets (Box)
  - Discrete: on/off switches, tap positions (Discrete/MultiDiscrete)
  - Mixed: continuous + discrete in one action (Dict space)

This script demonstrates:
1. Continuous actions -- bounded float values
2. Discrete actions -- categorical choices
3. Mixed actions -- continuous + discrete combined
4. Gym space conversion -- Action <-> gymnasium.Space
5. Heterogeneous agents -- different action types under one coordinator
6. Scale/unscale -- normalized [-1,1] <-> physical units

Domain: Power grid with generators (mixed) and transformers (discrete).

Usage:
    cd "examples/7. advanced_patterns"
    python mixed_action_spaces.py
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.simple import SimpleEnv


# ---------------------------------------------------------------------------
# 1. Continuous actions
# ---------------------------------------------------------------------------

def demo_continuous():
    """Continuous actions with bounds."""
    print("=" * 60)
    print("Part 1: Continuous Actions")
    print("=" * 60)
    print("""
  Most common: bounded float values for setpoints/controls.

    action = Action()
    action.set_specs(
        dim_c=2,
        range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
    )
""")

    action = Action()
    action.set_specs(
        dim_c=2,
        range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
    )

    print(f"  Specs: dim_c={action.dim_c}, dim_d={action.dim_d}")
    print(f"  Range: [{action.range[0]} , {action.range[1]}]")
    print(f"  Gym space: {action.space}")

    # Set values
    action.set_values(c=[5.0, 0.3])
    print(f"\n  set_values(c=[5.0, 0.3])")
    print(f"  c = {action.c}")

    # Values are clipped to range
    action.set_values(c=[15.0, -2.0])
    print(f"\n  set_values(c=[15.0, -2.0])  (out of range)")
    print(f"  c = {action.c}  (clipped to bounds)")

    # Sample random action
    action.sample(seed=42)
    print(f"\n  action.sample(seed=42)")
    print(f"  c = {action.c}")


# ---------------------------------------------------------------------------
# 2. Discrete actions
# ---------------------------------------------------------------------------

def demo_discrete():
    """Discrete actions for categorical choices."""
    print("\n" + "=" * 60)
    print("Part 2: Discrete Actions")
    print("=" * 60)
    print("""
  For switches, modes, or categorical selections.

    # Single discrete head with 3 choices (0, 1, 2)
    action.set_specs(dim_d=1, ncats=[3])

    # Multiple discrete heads
    action.set_specs(dim_d=2, ncats=[3, 5])
""")

    # Single discrete
    action = Action()
    action.set_specs(dim_d=1, ncats=[3])
    print(f"  Single discrete: dim_d={action.dim_d}, ncats={action.ncats}")
    print(f"  Gym space: {action.space}")

    action.set_values(d=[2])
    print(f"  set_values(d=[2]) -> d = {action.d}")

    # Scalar shorthand for pure discrete
    action.set_values(1)
    print(f"  set_values(1)     -> d = {action.d}")

    # Multiple discrete heads
    print()
    action_multi = Action()
    action_multi.set_specs(dim_d=2, ncats=[3, 5])
    print(f"  Multi-discrete: dim_d={action_multi.dim_d}, ncats={action_multi.ncats}")
    print(f"  Gym space: {action_multi.space}")

    action_multi.set_values(d=[1, 4])
    print(f"  set_values(d=[1, 4]) -> d = {action_multi.d}")

    # Out-of-range clipping
    action_multi.set_values(d=[5, 10])
    print(f"  set_values(d=[5, 10]) -> d = {action_multi.d}  (clipped to [0, ncats-1])")


# ---------------------------------------------------------------------------
# 3. Mixed actions
# ---------------------------------------------------------------------------

def demo_mixed():
    """Mixed continuous + discrete actions."""
    print("\n" + "=" * 60)
    print("Part 3: Mixed Actions (Continuous + Discrete)")
    print("=" * 60)
    print("""
  Combine continuous and discrete in one action:

    action.set_specs(
        dim_c=2,                               # 2 continuous dims
        dim_d=1, ncats=[3],                    # 1 discrete head, 3 choices
        range=(np.array([0.0, 0.0]),
               np.array([100.0, 50.0]))
    )
""")

    action = Action()
    action.set_specs(
        dim_c=2,
        dim_d=1,
        ncats=[3],
        range=(np.array([0.0, 0.0]), np.array([100.0, 50.0])),
    )

    print(f"  Specs: dim_c={action.dim_c}, dim_d={action.dim_d}, ncats={action.ncats}")
    print(f"  Gym space: {action.space}")
    print(f"  Space type: {type(action.space).__name__} (Dict with 'c' and 'd' keys)")

    # Set via dict
    action.set_values({"c": [75.0, 30.0], "d": [2]})
    print(f"\n  set_values({{'c': [75, 30], 'd': [2]}})")
    print(f"  c = {action.c}, d = {action.d}")

    # Set via keyword args
    action.set_values(c=[50.0, 25.0], d=[1])
    print(f"\n  set_values(c=[50, 25], d=[1])")
    print(f"  c = {action.c}, d = {action.d}")

    # Set via flat vector [c..., d...]
    action.set_values(np.array([80.0, 40.0, 0.0]))
    print(f"\n  set_values(np.array([80, 40, 0]))  (flat: [c0, c1, d0])")
    print(f"  c = {action.c}, d = {action.d}")

    # Vector export
    print(f"\n  action.vector() = {action.vector()}  (flat: [c..., d...])")


# ---------------------------------------------------------------------------
# 4. Gym space conversion
# ---------------------------------------------------------------------------

def demo_gym_conversion():
    """Convert between Action and Gymnasium spaces."""
    print("\n" + "=" * 60)
    print("Part 4: Gym Space Conversion")
    print("=" * 60)
    print("""
  Action.from_gym_space() creates an Action from any Gymnasium space.
  action.space returns the corresponding Gymnasium space.
""")

    from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict

    spaces = {
        "Box(2)": Box(low=-1.0, high=1.0, shape=(2,)),
        "Discrete(5)": Discrete(5),
        "MultiDiscrete([3,4])": MultiDiscrete([3, 4]),
        "Dict(c+d)": SpaceDict({
            "c": Box(low=0.0, high=10.0, shape=(2,)),
            "d": Discrete(3),
        }),
    }

    print(f"  {'Gym Space':<25} {'dim_c':>6} {'dim_d':>6} {'ncats':>10}")
    print(f"  {'-' * 49}")
    for name, space in spaces.items():
        action = Action.from_gym_space(space)
        print(f"  {name:<25} {action.dim_c:>6} {action.dim_d:>6} {str(action.ncats):>10}")


# ---------------------------------------------------------------------------
# 5. Scale/unscale for RL normalization
# ---------------------------------------------------------------------------

def demo_scale_unscale():
    """Normalize continuous actions to [-1, 1] and back."""
    print("\n" + "=" * 60)
    print("Part 5: Scale / Unscale (RL Normalization)")
    print("=" * 60)
    print("""
  RL policies typically output in [-1, 1].
  scale() converts physical -> normalized.
  unscale() converts normalized -> physical.
""")

    action = Action()
    action.set_specs(
        dim_c=2,
        range=(np.array([0.0, 100.0]), np.array([500.0, 200.0])),
    )

    # Physical -> normalized
    action.set_values(c=[250.0, 150.0])
    normalized = action.scale()
    print(f"  Physical:   c = {action.c}")
    print(f"  Normalized: scale() = {normalized}  (midpoint -> 0.0)")

    action.set_values(c=[0.0, 100.0])
    normalized = action.scale()
    print(f"\n  Physical:   c = {action.c}")
    print(f"  Normalized: scale() = {normalized}  (lower bound -> -1.0)")

    # Normalized -> physical
    action.unscale([0.5, -0.5])
    print(f"\n  Normalized: [0.5, -0.5]")
    print(f"  Physical:   unscale() = {action.c}")


# ---------------------------------------------------------------------------
# 6. Heterogeneous agents with different action types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GeneratorFeature(Feature):
    """Generator with power output and status."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    power: float = 0.0
    is_on: float = 1.0   # 1.0 = on, 0.0 = off


@dataclass(slots=True)
class TransformerFeature(Feature):
    """Transformer with tap position."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    tap_pos: float = 5.0
    voltage: float = 1.0


class Generator(FieldAgent):
    """Generator with mixed action: continuous power + discrete on/off."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,                                      # power setpoint [0, 100] MW
            dim_d=1, ncats=[2],                           # on/off switch
            range=(np.array([0.0]), np.array([100.0])),
        )
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        feat = self.state.features["GeneratorFeature"]
        is_on = float(self.action.d[0])
        power = self.action.c[0] * is_on
        feat.set_values(power=power, is_on=is_on)

    def compute_local_reward(self, local_state: dict) -> float:
        if "GeneratorFeature" not in local_state:
            return 0.0
        vec = local_state["GeneratorFeature"]
        power = float(vec[0])
        return -abs(power - 50.0) / 100.0  # target 50MW


class Transformer(FieldAgent):
    """Transformer with pure discrete action: tap position selection."""

    def __init__(self, *args, num_taps: int = 11, **kwargs):
        self._num_taps = num_taps
        super().__init__(*args, **kwargs)

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_d=1, ncats=[self._num_taps])  # tap positions 0-10
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        feat = self.state.features["TransformerFeature"]
        tap = float(self.action.d[0])
        voltage = 0.95 + tap * 0.01  # tap 0=0.95pu, tap 5=1.00pu, tap 10=1.05pu
        feat.set_values(tap_pos=tap, voltage=voltage)

    def compute_local_reward(self, local_state: dict) -> float:
        if "TransformerFeature" not in local_state:
            return 0.0
        vec = local_state["TransformerFeature"]
        voltage = float(vec[1])
        return -abs(voltage - 1.0) * 10.0  # target 1.0 pu


def grid_simulation(agent_states: dict) -> dict:
    """Simple grid physics: power affects voltage."""
    total_power = 0.0
    for aid, features in agent_states.items():
        if "GeneratorFeature" in features:
            total_power += features["GeneratorFeature"]["power"]
    # Higher generation -> higher voltage
    voltage_effect = (total_power - 50.0) / 200.0
    for aid, features in agent_states.items():
        if "TransformerFeature" in features:
            v = features["TransformerFeature"]["voltage"] + voltage_effect * 0.02
            features["TransformerFeature"]["voltage"] = float(np.clip(v, 0.9, 1.1))
    return agent_states


def demo_heterogeneous_agents():
    """Different agent types with different action spaces under one coordinator."""
    print("\n" + "=" * 60)
    print("Part 6: Heterogeneous Agents")
    print("=" * 60)
    print("""
  Different agent types can coexist under one coordinator:
    - Generator: mixed action (continuous power + discrete on/off)
    - Transformer: pure discrete (tap position)

  The framework handles different action spaces automatically.
""")

    gen = Generator(agent_id="gen_1", features=[GeneratorFeature()])
    trafo = Transformer(agent_id="trafo_1", features=[TransformerFeature()], num_taps=11)

    coordinator = CoordinatorAgent(
        agent_id="grid_op",
        subordinates={"gen_1": gen, "trafo_1": trafo},
    )

    env = SimpleEnv(
        coordinator_agents=[coordinator],
        simulation_func=grid_simulation,
        env_id="mixed_action_demo",
    )

    obs, _ = env.reset(seed=0)

    # Show action spaces
    print(f"  Agent action spaces:")
    for aid in ["gen_1", "trafo_1"]:
        agent = env.registered_agents[aid]
        a = agent.action
        print(f"    {aid}: dim_c={a.dim_c}, dim_d={a.dim_d}, ncats={a.ncats}, space={a.space}")

    # Step with different action types
    print(f"\n  Running 5 steps:")
    print(f"  {'Step':<6} {'gen power':>10} {'gen on':>8} {'trafo tap':>10} {'voltage':>10}")
    print(f"  {'-' * 46}")

    for step in range(5):
        gen_action = Action()
        gen_action.set_specs(
            dim_c=1, dim_d=1, ncats=[2],
            range=(np.array([0.0]), np.array([100.0])),
        )
        gen_action.set_values(c=[30.0 + step * 10.0], d=[1])  # increasing power, on

        trafo_action = Action()
        trafo_action.set_specs(dim_d=1, ncats=[11])
        trafo_action.set_values(d=[5 + step % 3])  # varying tap

        obs, rewards, _, _, _ = env.step({"gen_1": gen_action, "trafo_1": trafo_action})

        gen_obs = obs["gen_1"].vector() if hasattr(obs["gen_1"], "vector") else np.asarray(obs["gen_1"])
        trafo_obs = obs["trafo_1"].vector() if hasattr(obs["trafo_1"], "vector") else np.asarray(obs["trafo_1"])
        print(f"  {step + 1:<6} {gen_obs[0]:>10.1f} {gen_obs[1]:>8.0f} {trafo_obs[0]:>10.0f} {trafo_obs[1]:>10.4f}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    demo_continuous()
    demo_discrete()
    demo_mixed()
    demo_gym_conversion()
    demo_scale_unscale()
    demo_heterogeneous_agents()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  Action Space Types:

    Continuous:  set_specs(dim_c=2, range=(...))    -> Box
    Discrete:    set_specs(dim_d=1, ncats=[5])      -> Discrete
    Multi-disc:  set_specs(dim_d=2, ncats=[3, 5])   -> MultiDiscrete
    Mixed:       set_specs(dim_c=2, dim_d=1, ...)   -> Dict(c=Box, d=Discrete)

  Key Methods:

    set_values(c=..., d=...)     set action values (dict, array, scalar)
    sample(seed=42)              random action from space
    scale() / unscale(vec)       normalize [-1,1] <-> physical units
    vector()                     flatten to [c..., d...] array
    Action.from_gym_space(space) create from Gymnasium space

  Heterogeneous agents:
    Different agents can have different action types
    Framework routes actions correctly per agent
""")
    print("Done.")


if __name__ == "__main__":
    main()

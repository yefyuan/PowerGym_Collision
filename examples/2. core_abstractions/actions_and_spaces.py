"""Actions & Spaces -- HERON's mixed continuous/discrete action model.

HERON's Action class unifies continuous, discrete, and mixed action spaces
behind a single interface.  This matters for MARL because different agents
in the same environment often need different action types (e.g. a generator
outputs continuous power while a switch is on/off).

This script walks through:
1. Continuous-only actions (e.g. power setpoints)
2. Discrete-only actions (e.g. on/off switch)
3. Mixed continuous + discrete actions
4. Scale / unscale for policy normalization
5. Gymnasium space interop

Usage:
    cd "examples/2. core_abstractions"
    python actions_and_spaces.py
"""

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict

from heron.core.action import Action


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main():
    # ------------------------------------------------------------------
    # 1. Continuous-only action
    # ------------------------------------------------------------------
    section("1. Continuous Action (2D power setpoint in [0, 100] kW)")

    action = Action()
    action.set_specs(
        dim_c=2,
        range=(np.array([0.0, 0.0]), np.array([100.0, 100.0])),
    )

    print(f"dim_c       : {action.dim_c}")
    print(f"dim_d       : {action.dim_d}")
    print(f"range       : [{action.range[0]} , {action.range[1]}]")
    print(f"Gym space   : {action.space}")

    action.sample(seed=0)
    print(f"Sampled     : c = {action.c}")

    action.set_values(c=[25.0, 75.0])
    print(f"After set   : c = {action.c}")

    action.reset()
    print(f"After reset : c = {action.c}  (midpoint of range)")

    # ------------------------------------------------------------------
    # 2. Discrete-only action
    # ------------------------------------------------------------------
    section("2. Discrete Action (switch with 3 modes: off / low / high)")

    action_d = Action()
    action_d.set_specs(dim_d=1, ncats=[3])

    print(f"dim_c       : {action_d.dim_c}")
    print(f"dim_d       : {action_d.dim_d}")
    print(f"ncats       : {action_d.ncats}")
    print(f"Gym space   : {action_d.space}")

    action_d.sample(seed=1)
    print(f"Sampled     : d = {action_d.d}")

    # Scalar set for pure discrete
    action_d.set_values(2)
    print(f"Set to 2    : d = {action_d.d}")

    # ------------------------------------------------------------------
    # 3. Mixed continuous + discrete action
    # ------------------------------------------------------------------
    section("3. Mixed Action (2D continuous + 1 discrete head with 4 categories)")

    action_m = Action()
    action_m.set_specs(
        dim_c=2,
        dim_d=1,
        ncats=[4],
        range=(np.array([0.0, -1.0]), np.array([10.0, 1.0])),
    )

    print(f"dim_c       : {action_m.dim_c}")
    print(f"dim_d       : {action_m.dim_d}")
    print(f"ncats       : {action_m.ncats}")
    print(f"Gym space   : {action_m.space}")

    action_m.sample(seed=2)
    print(f"Sampled     : c = {action_m.c}, d = {action_m.d}")

    # Dict-style set_values
    action_m.set_values({"c": [5.0, 0.0], "d": [3]})
    print(f"Dict set    : c = {action_m.c}, d = {action_m.d}")

    # Flat vector set_values: [c0, c1, d0]
    action_m.set_values(np.array([1.0, -0.5, 2.0]))
    print(f"Vector set  : c = {action_m.c}, d = {action_m.d}")

    # ------------------------------------------------------------------
    # 4. Scale / unscale (policy normalization)
    # ------------------------------------------------------------------
    section("4. Scale / Unscale (normalize continuous part to [-1, 1])")

    action.set_values(c=[25.0, 75.0])
    normalized = action.scale()
    print(f"Physical    : {action.c}")
    print(f"Normalized  : {normalized}  ([-1,1] range)")

    # Unscale maps [-1,1] back to physical range
    action.unscale(np.array([0.0, 0.0]))
    print(f"Unscale 0.0 : {action.c}  (midpoint = 50.0)")

    # ------------------------------------------------------------------
    # 5. Gymnasium space interop
    # ------------------------------------------------------------------
    section("5. Gymnasium Space Interop")

    # Create Action from a Gym Box
    box_action = Action.from_gym_space(Box(-1.0, 1.0, shape=(3,)))
    print(f"From Box    : dim_c={box_action.dim_c}, space={box_action.space}")

    # Create Action from a Gym Discrete
    disc_action = Action.from_gym_space(Discrete(5))
    print(f"From Disc   : dim_d={disc_action.dim_d}, ncats={disc_action.ncats}, space={disc_action.space}")

    # Create Action from a Gym Dict (mixed)
    dict_space = SpaceDict({
        "c": Box(0.0, 10.0, shape=(2,)),
        "d": MultiDiscrete([3, 5]),
    })
    mixed_action = Action.from_gym_space(dict_space)
    print(f"From Dict   : dim_c={mixed_action.dim_c}, dim_d={mixed_action.dim_d}, ncats={mixed_action.ncats}")

    # ------------------------------------------------------------------
    # 6. Utility methods
    # ------------------------------------------------------------------
    section("6. Utility Methods")

    action.set_values(c=[30.0, 60.0])
    print(f"vector()    : {action.vector()}")
    print(f"scalar(0)   : {action.scalar(0)}")
    print(f"scalar(1)   : {action.scalar(1)}")
    print(f"is_valid()  : {action.is_valid()}")

    clone = action.copy()
    clone.set_values(c=[0.0, 0.0])
    print(f"Original    : {action.c}  (unchanged after copy mutation)")
    print(f"Clone       : {clone.c}")

    print("\nDone.")


if __name__ == "__main__":
    main()

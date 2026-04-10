"""Agent definitions shared across HMARL-CBF case studies."""

from typing import Any, Dict, List

import numpy as np
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.protocols.vertical import VerticalProtocol

from features import FleetSafetyFeature

NUM_DRONES_PER_FLEET = 3


class TransportDrone(FieldAgent):
    """Drone whose velocity is set by the coordinator."""

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def set_state(self, **kwargs) -> None:
        if "x_pos" in kwargs:
            self.state.features["DronePositionFeature"].set_values(x_pos=kwargs["x_pos"])
        if "y_pos" in kwargs:
            self.state.features["DronePositionFeature"].set_values(y_pos=kwargs["y_pos"])

    def apply_action(self) -> None:
        pf = self.state.features["DronePositionFeature"]
        # Action drives x-axis movement (toward delivery goal at x=1.0)
        new_x = pf.x_pos + self.action.c[0] * 0.02
        pf.set_values(x_pos=new_x)

    def compute_local_reward(self, local_state: dict) -> float:
        pf = local_state.get("DronePositionFeature")
        if pf is None:
            return 0.0
        x_pos = float(pf[0])
        # Reward: progress toward delivery goal (x=1.0), with diminishing returns.
        return x_pos - 0.5 * x_pos ** 2


class TransportCoordinator(CoordinatorAgent):
    """Coordinator that assigns velocity targets to its drones.

    Action: N-D vector in [-1, 1] -- one velocity command per subordinate drone.
    VerticalProtocol splits this into per-drone 1D actions.
    """

    def __init__(self, agent_id, subordinates, features=None, **kwargs):
        default_features = [FleetSafetyFeature()]
        all_features = (features or []) + default_features

        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates,
            protocol=VerticalProtocol(),
            **kwargs,
        )

        self.observation_space = Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = self.action.space

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=NUM_DRONES_PER_FLEET,
            range=(np.full(NUM_DRONES_PER_FLEET, -1.0), np.full(NUM_DRONES_PER_FLEET, 1.0)),
        )
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def compute_rewards(self, proxy) -> Dict[str, float]:
        sub_rewards: Dict[str, float] = {}
        for sub in self.subordinates.values():
            sub_rewards.update(sub.compute_rewards(proxy))
        coordinator_reward = sum(sub_rewards.values())
        rewards = {self.agent_id: coordinator_reward}
        rewards.update(sub_rewards)
        return rewards

    def compute_local_reward(self, local_state: dict) -> float:
        sf = local_state.get("FleetSafetyFeature")
        if sf is None:
            return 0.0
        return float(sf[1])  # payload_progress

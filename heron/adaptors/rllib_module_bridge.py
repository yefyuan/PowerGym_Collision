from __future__ import annotations

from typing import Optional

import gymnasium as gym

from ray.rllib.core.rl_module.rl_module import RLModule

from heron.core.action import Action
from heron.core.observation import Observation
from heron.core.policies import Policy


class RLlibModuleBridge(Policy):
    """Wraps an RLlib RLModule (new API stack) as a HERON ``Policy``.

    Parameters
    ----------
    rl_module : ray.rllib.core.rl_module.RLModule
        An RLModule obtained via ``algo.get_module(module_id)``.
    agent_id : str
        The HERON agent ID this policy is attached to.
    action_space : gymnasium.Space, optional
        Action space for building HERON Actions. If *None*, falls back to
        ``rl_module.config.action_space``.
    """

    observation_mode: str = "full"

    def __init__(
        self,
        rl_module: RLModule,
        agent_id: str,
        action_space: gym.Space | None = None,
    ) -> None:
        self._module = rl_module
        self._agent_id = agent_id
        space = action_space or rl_module.config.action_space
        self._action_template = Action.from_gym_space(space)

    def forward(self, observation: Observation) -> Optional[Action]:
        import torch

        obs_vec = observation.vector()
        obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).float()

        self._module.eval()
        with torch.no_grad():
            output = self._module.forward_inference({"obs": obs_tensor})

        # Extract action from distribution inputs
        if "actions" in output:
            raw = output["actions"].cpu().numpy()[0]
        elif "action_dist_inputs" in output:
            dist_inputs = output["action_dist_inputs"].cpu().numpy()[0]
            # For continuous: use mean (first half of dist inputs)
            raw = dist_inputs[:len(dist_inputs) // 2]
        else:
            raise ValueError(f"Unexpected RLModule output keys: {output.keys()}")

        action = self._action_template.copy()
        action.set_values(raw)
        return action

    def reset(self) -> None:
        pass

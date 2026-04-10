"""Smoke test: HERON collision env exposed to RLlib as three MG policies."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collision_case.collision_rllib_env import CollisionRLlibMultiAgentEnv


def main() -> None:
    print("Building CollisionRLlibMultiAgentEnv (missing pickle → empty fallback)...")
    env = CollisionRLlibMultiAgentEnv(
        {
            "dataset_path": str(Path("/nonexistent/collision_data.pkl")),
            "episode_steps": 2,
            "share_reward": True,
            "penalty": 10.0,
            "log_path": None,
        }
    )
    obs, _ = env.reset(seed=0)
    print("obs keys", obs.keys(), "shapes", {k: v.shape for k, v in obs.items()})
    actions = {aid: np.zeros(env.action_space[aid].shape, dtype=np.float32) for aid in env.possible_agents}
    obs2, rew, term, trunc, _ = env.step(actions)
    print("step ok rew", rew, "trunc __all__", trunc.get("__all__"))
    env.close()
    print("OK")


if __name__ == "__main__":
    main()

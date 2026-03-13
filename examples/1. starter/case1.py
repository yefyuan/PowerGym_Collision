"""Case 1: IPPO -- independent policies for multi-drone payload transport.

Hierarchy:
    SystemAgent
    ├── fleet_0 (TransportCoordinator) -- independent policy
    │   ├── drone_0_0 (TransportDrone) -- receives 1D velocity command
    │   ├── drone_0_1 (TransportDrone)
    │   └── drone_0_2 (TransportDrone)
    └── fleet_1 (TransportCoordinator) -- independent policy
        ├── drone_1_0 (TransportDrone)
        ├── drone_1_1 (TransportDrone)
        └── drone_1_2 (TransportDrone)

Domain: Multi-drone payload transport across a warehouse floor.
    Each fleet coordinator assigns velocity targets to its drones.
    Drones must make progress toward x=1.0 against wind drag.
    No safety filter -- collisions are possible.

Training: IPPO -- each fleet coordinator has an independent policy.
Evaluation: event-driven with jittered tick configs.

Usage:
    cd examples/hmarl-cbf
    python case1.py
"""

from agents import TransportCoordinator, TransportDrone, NUM_DRONES_PER_FLEET
from env_physics import case1_simulation
from features import DronePositionFeature

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from heron.adaptors.rllib import RLlibBasedHeronEnv
from heron.adaptors.rllib_runner import HeronEnvRunner
from heron.scheduling import ScheduleConfig, JitterType

def main():

    print("=" * 60)
    print("Case 1: IPPO -- 2 fleet coordinators (independent policies), 3 drones each")
    print("  No safety filter -- collisions possible")
    print("=" * 60)

    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    try:
        fleet_ids = ["fleet_0", "fleet_1"]

        config = (
            PPOConfig()
            .environment(
                env=RLlibBasedHeronEnv,
                env_config={
                    "env_id": "ippo_transport",
                    "agents": [
                        {
                            "agent_id": f"drone_{f}_{d}", 
                            "agent_cls": TransportDrone,
                            "features": [DronePositionFeature(y_pos=0.2 + 0.3 * d)],
                            # Optional per-agent tick config with jitter (overrides DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG)
                            "coordinator": f"fleet_{f}"
                        }
                        for f in range(2) for d in range(NUM_DRONES_PER_FLEET)
                    ],
                    "coordinators": [
                        {
                            "coordinator_id": f"fleet_{f}", 
                            "agent_cls": TransportCoordinator,
                            # Optional per-coordinator tick config with jitter (overrides DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG)
                        }
                        for f in range(2)
                    ],
                    "simulation": case1_simulation,
                    "max_steps": 50,
                    "agent_ids": fleet_ids,
                },
            )
            .multi_agent(
                policies={
                    "fleet_0_policy": PolicySpec(),
                    "fleet_1_policy": PolicySpec(),
                },
                policy_mapping_fn=lambda agent_id, *a, **kw: f"{agent_id}_policy",
            )
            .env_runners(
                env_runner_cls=HeronEnvRunner,
                num_env_runners=1,
                num_envs_per_env_runner=1,
            )
            .evaluation(
                evaluation_interval=5,
                evaluation_num_env_runners=0,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_config=HeronEnvRunner.evaluation_config(t_end=50.0),
            )
            .training(
                lr=5e-4,
                gamma=0.99,
                train_batch_size=400,
                minibatch_size=64,
                num_epochs=3,
            )
            .framework("torch")
        )

        algo = config.build()
        print("\nTraining for 10 iterations...")
        for i in range(10):
            result = algo.train()
            reward = result["env_runners"]["episode_return_mean"]
            if (i + 1) % 2 == 0:
                print(f"  Iter {i + 1}/10: reward={reward:.3f}")

        print("\nRunning event-driven evaluation...")
        eval_result = algo.evaluate()
        print(f"  Evaluation result: {eval_result}")

        algo.stop()
        print("\nCase 1 PASSED.")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

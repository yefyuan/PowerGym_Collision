"""Training script for collision detection experiments using RLlib.

Reproduces the original collision experiments with:
- Shared vs independent reward schemes
- 3 microgrids with PPO training
- Collision metrics logging
- Support for asynchronous updates (event-driven mode)

Usage:
    # Shared reward training
    python -m collision_case.train_collision --share-reward --penalty 10 --log-path results/shared_reward.csv

    # Independent reward training
    python -m collision_case.train_collision --no-share-reward --penalty 10 --log-path results/independent_reward.csv

    # Asynchronous update experiment
    python -m collision_case.train_collision --share-reward --enable-async --log-path results/async_shared.csv
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from collision_case import create_collision_system, CollisionEnv
from collision_case.gymnasium_wrapper import GymnasiumCollisionEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default paths
CURRENT_DIR = Path(__file__).parent
DEFAULT_DATASET_PATH = CURRENT_DIR.parent / "powergrid" / "data.pkl"
DEFAULT_LOG_DIR = CURRENT_DIR / "results"


def build_env_config(args):
    """Build environment config from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dict with environment configuration
    """
    env_config = {
        "dataset_path": str(args.dataset_path),
        "episode_steps": args.episode_steps,
        "dt": args.dt,
        "share_reward": args.share_reward,
        "penalty": args.penalty,
        "log_path": str(args.log_path) if args.log_path else None,
        "enable_async": args.enable_async,
        "field_tick_s": args.field_tick_s,
        "coord_tick_s": args.coord_tick_s,
        "system_tick_s": args.system_tick_s,
        "jitter_ratio": args.jitter_ratio,
    }

    return env_config


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """Map device agents to shared policies by type.

    Maps:
    - All ESS devices -> "ess_policy"
    - All DG devices -> "gen_policy"
    - All PV devices -> "pv_policy"
    - All WT devices -> "wt_policy"

    Args:
        agent_id: Agent identifier
        episode: Episode object (unused)
        worker: Worker object (unused)
        **kwargs: Additional arguments

    Returns:
        Policy ID string
    """
    if "ESS" in agent_id:
        return "ess_policy"
    elif "DG" in agent_id:
        return "gen_policy"
    elif "PV" in agent_id:
        return "pv_policy"
    elif "WT" in agent_id:
        return "wt_policy"
    else:
        # Default policy for unknown agents
        return "default_policy"


def train_collision_detection(args):
    """Train collision detection using PPO.

    Args:
        args: Parsed command line arguments

    Returns:
        Training result dict
    """
    import ray
    from ray.tune.registry import register_env

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    # Build environment config
    env_config = build_env_config(args)

    # Register environment with Gymnasium wrapper
    register_env("collision_env", lambda config: GymnasiumCollisionEnv(config))

    # Define policies (one per device type)
    policies = {
        "ess_policy",
        "gen_policy",
        "pv_policy",
        "wt_policy",
    }

    # Configure PPO algorithm
    config = (
        PPOConfig()
        .environment(env="collision_env", env_config=env_config)
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_clip_param=10.0,
            train_batch_size=args.train_batch_size,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_cpus_per_env_runner=1,
        )
        .resources(
            num_gpus=args.num_gpus,
            num_cpus_for_main_process=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    # Add evaluation config if async mode is enabled
    if args.enable_async and args.eval_interval > 0:
        from heron.adaptors.rllib_runner import HeronEnvRunner

        config = config.evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_num_env_runners=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config=HeronEnvRunner.evaluation_config(t_end=args.eval_t_end),
        )

    # Build algorithm
    logger.info("Building PPO algorithm for collision detection...")
    logger.info(f"  Share reward: {args.share_reward}")
    logger.info(f"  Penalty: {args.penalty}")
    logger.info(f"  Enable async: {args.enable_async}")
    logger.info(f"  Log path: {args.log_path}")

    algo = config.build()

    # Training loop
    reward_history = []
    safety_history = []

    try:
        for iteration in range(args.num_iterations):
            result = algo.train()

            # Extract metrics
            reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0.0) or 0.0
            reward_history.append(reward_mean)

            # Log progress
            if (iteration + 1) % args.log_every == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{args.num_iterations} | "
                    f"Reward: {reward_mean:.3f} | "
                    f"Episodes: {result.get('env_runners', {}).get('episodes_this_iter', 0)}"
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info(f"Training complete. {len(reward_history)} iterations.")

    # Print summary
    if len(reward_history) >= 10:
        logger.info(f"  First 10 avg reward: {np.mean(reward_history[:10]):.3f}")
        logger.info(f"  Last 10 avg reward: {np.mean(reward_history[-10:]):.3f}")

    # Save checkpoint if requested
    if args.checkpoint_path:
        checkpoint_path = algo.save(args.checkpoint_path)
        logger.info(f"Saved checkpoint to: {checkpoint_path}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    return {
        "reward_history": reward_history,
        "safety_history": safety_history,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train collision detection with RLlib PPO")

    # Environment args
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--episode-steps", type=int, default=24, help="Episode length in timesteps"
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Time step duration (hours)")

    # Experiment args
    parser.add_argument(
        "--share-reward",
        action="store_true",
        default=True,
        help="Share rewards across microgrids",
    )
    parser.add_argument(
        "--no-share-reward",
        action="store_false",
        dest="share_reward",
        help="Independent rewards per microgrid",
    )
    parser.add_argument(
        "--penalty", type=float, default=10.0, help="Safety penalty multiplier"
    )
    parser.add_argument("--log-path", type=Path, default=None, help="CSV log file path")

    # Async experiment args
    parser.add_argument(
        "--enable-async",
        action="store_true",
        default=False,
        help="Enable asynchronous updates (event-driven mode)",
    )
    parser.add_argument(
        "--field-tick-s", type=float, default=5.0, help="Field agent tick interval (s)"
    )
    parser.add_argument(
        "--coord-tick-s", type=float, default=10.0, help="Coordinator tick interval (s)"
    )
    parser.add_argument(
        "--system-tick-s", type=float, default=30.0, help="System tick interval (s)"
    )
    parser.add_argument(
        "--jitter-ratio", type=float, default=0.1, help="Timing jitter ratio"
    )

    # Training args
    parser.add_argument(
        "--num-iterations", type=int, default=300, help="Number of training iterations"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=8760, help="Training batch size"
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of SGD epochs")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size")

    # Resource args
    parser.add_argument(
        "--num-env-runners", type=int, default=2, help="Number of env runners"
    )
    parser.add_argument("--num-cpus", type=int, default=4, help="Total CPUs for Ray")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs")

    # Evaluation args
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Evaluation interval (0=disable, N=eval every N iters)",
    )
    parser.add_argument(
        "--eval-t-end", type=float, default=100.0, help="Event-driven eval duration (s)"
    )

    # Logging args
    parser.add_argument("--log-every", type=int, default=10, help="Log every N iterations")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=None, help="Checkpoint save directory"
    )

    args = parser.parse_args()

    # Create log directory if needed
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Run training
    train_collision_detection(args)


if __name__ == "__main__":
    main()

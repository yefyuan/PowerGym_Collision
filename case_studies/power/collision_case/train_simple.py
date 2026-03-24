"""Simple training script matching original Collision platform structure."""

from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus
from collision_case.collision_network import create_collision_network

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.tune.registry import get_trainable_cls, register_env

# Import the simple multi-agent env we'll create
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from simple_ma_env import SimpleMultiAgentMicrogrids

parser = add_rllib_example_script_args(
    default_iters=5,
    default_timesteps=300000,
    default_reward=0.0,
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    assert args.num_agents > 0, "Must set --num-agents > 0"
    
    # Environment config matching original
    env_config = {
        "train": True,
        "penalty": 10,
        "share_reward": True,
        "log_path": "collision_test.csv",
    }
    
    register_env("env", lambda _: SimpleMultiAgentMicrogrids(env_config))
    
    # Policies matching original (one per microgrid)
    policies = {"MG{}".format(i+1) for i in range(args.num_agents)}
    rl_module_specs = {
        "MG1": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(fcnet_hiddens=[128, 128]),
            catalog_class=PPOCatalog),
        "MG2": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]),
            catalog_class=PPOCatalog),
        "MG3": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(fcnet_hiddens=[64, 128]),
            catalog_class=PPOCatalog),
    }
    
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("env")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .training(train_batch_size=8760)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs),
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .env_runners(num_env_runners=0)
    )
    
    run_rllib_example_script_experiment(base_config, args)

"""Simplified multi-agent environment matching original Collision platform.

Directly uses IEEE networks and pandapower, bypassing HERON complexity.
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pickle
import os
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from collision_case.collision_network import create_collision_network


class SimpleMultiAgentMicrogrids(MultiAgentEnv):
    """Multi-agent microgrid environment matching original structure."""
    
    def __init__(self, env_config):
        super().__init__()
        self.env_config = env_config
        self.train = env_config.get('train', True)
        self.penalty = env_config.get('penalty', 10)
        self.share_reward = env_config.get('share_reward', True)
        self.log_path = env_config.get('log_path', None)
        
        # Build network
        self.net = create_collision_network()
        
        # Agent names
        self.possible_agents = ['MG1', 'MG2', 'MG3']
        self.agents = self.possible_agents.copy()
        
        # Episode settings
        self.max_episode_steps = 24
        self._t = 0
        self._episode = 0
        
        # Load dataset
        dataset_path = os.path.join(os.path.dirname(__file__), '../powergrid/data.pkl')
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.dataset = dataset['train'] if self.train else dataset['test']
        self.data_size = len(self.dataset['price']['0096WD_7_N001'])
        self.total_days = self.data_size // self.max_episode_steps
        
        # Define action and observation spaces
        self._action_dim = 4  # ESS_P, DG_P, PV_Q, WT_Q per microgrid
        self._obs_dim = 20  # Simplified observation
        
        self.action_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Initialize log
        if self.log_path:
            with open(self.log_path, 'w') as f:
                f.write("timestep,episode,reward_sum,safety_sum")
                for mg in self.possible_agents:
                    f.write(f",{mg}_overvoltage,{mg}_undervoltage,{mg}_overloading,{mg}_safety_sum")
                f.write("\n")
    
    def reset(self, *, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Select random day for training
        if self.train:
            self._day = np.random.randint(self.total_days - 1)
            self._t = self._day * self.max_episode_steps
        else:
            self._t = 0
            self._day = 0
        
        # Run power flow
        try:
            pp.runpp(self.net)
        except:
            pass
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action_dict):
        """Execute one step."""
        # Apply actions (simplified - just update device setpoints)
        for mg_id, action in action_dict.items():
            mg_idx = self.possible_agents.index(mg_id)
            
            # Scale actions to device limits
            ess_p = action[0] * 0.5  # ESS: -0.5 to 0.5 MW
            dg_p = (action[1] + 1) / 2 * 0.6  # DG: 0 to 0.6 MW
            
            # Update storage
            storage_name = f"{mg_id}_ESS1"
            storage_idx = self.net.storage[self.net.storage.name == storage_name]
            if len(storage_idx) > 0:
                idx = storage_idx.index[0]
                self.net.storage.at[idx, 'p_mw'] = ess_p
                
            # Update generator
            for gen_suffix in ['_DG1', '_PV1', '_WT1']:
                gen_name = f"{mg_id}{gen_suffix}"
                gen_idx = self.net.sgen[self.net.sgen.name == gen_name]
                if len(gen_idx) > 0:
                    idx = gen_idx.index[0]
                    if 'DG' in gen_suffix:
                        self.net.sgen.at[idx, 'p_mw'] = dg_p
        
        # Run power flow
        try:
            pp.runpp(self.net)
            converged = True
        except:
            converged = False
        
        # Compute rewards and safety
        rewards, safety = self._compute_rewards_and_safety(converged)
        
        # Apply reward sharing if enabled
        if self.share_reward:
            avg_reward = np.mean(list(rewards.values()))
            rewards = {agent: avg_reward for agent in self.possible_agents}
        
        # Log metrics
        if self.log_path:
            self._log_step(rewards, safety)
        
        # Increment timestep
        self._t += 1
        done = (self._t % self.max_episode_steps == 0)
        
        if done:
            self._episode += 1
        
        terminateds = {"__all__": done}
        truncateds = {"__all__": False}
        
        obs = self._get_obs()
        infos = {agent: {} for agent in self.possible_agents}
        
        return obs, rewards, terminateds, truncateds, infos
    
    def _get_obs(self):
        """Get observations for all agents."""
        obs_dict = {}
        
        for agent in self.possible_agents:
            # Simplified observation: random for now
            # In full implementation, would extract bus voltages, line loadings, etc.
            obs_dict[agent] = np.random.randn(self._obs_dim).astype(np.float32) * 0.1
        
        return obs_dict
    
    def _compute_rewards_and_safety(self, converged):
        """Compute rewards and safety metrics."""
        rewards = {}
        safety = {}
        
        if not converged:
            # Non-convergence penalty
            for agent in self.possible_agents:
                rewards[agent] = -200.0
                safety[agent] = 20.0
            return rewards, safety
        
        # Extract safety violations per microgrid
        for agent in self.possible_agents:
            # Get buses for this microgrid
            mg_buses = self.net.bus[self.net.bus.name.str.contains(agent)]
            mg_bus_indices = mg_buses.index.tolist()
            
            # Voltage violations
            if len(mg_bus_indices) > 0:
                vm_pu = self.net.res_bus.loc[mg_bus_indices, 'vm_pu'].values
                overvoltage = np.maximum(vm_pu - 1.05, 0).sum()
                undervoltage = np.maximum(0.95 - vm_pu, 0).sum()
            else:
                overvoltage = 0.0
                undervoltage = 0.0
            
            # Line overloading
            mg_lines = self.net.line[self.net.line.name.str.contains(agent)]
            mg_line_indices = mg_lines.index.tolist()
            
            if len(mg_line_indices) > 0:
                loading_pct = self.net.res_line.loc[mg_line_indices, 'loading_percent'].values
                overloading = np.maximum(loading_pct - 100, 0).sum() * 0.01
            else:
                overloading = 0.0
            
            # Total safety metric
            agent_safety = overvoltage + undervoltage + overloading
            safety[agent] = agent_safety
            
            # Reward = -cost - penalty * safety
            # Simplified cost (would compute from generator costs in full version)
            cost = 100.0  # Placeholder
            rewards[agent] = -cost - self.penalty * agent_safety
        
        return rewards, safety
    
    def _log_step(self, rewards, safety):
        """Log step metrics to CSV."""
        reward_sum = sum(rewards.values())
        safety_sum = sum(safety.values())
        
        # Extract detailed safety per agent
        with open(self.log_path, 'a') as f:
            f.write(f"{self._t},{self._episode},{reward_sum},{safety_sum}")
            
            for agent in self.possible_agents:
                # Get detailed metrics
                mg_buses = self.net.bus[self.net.bus.name.str.contains(agent)]
                mg_bus_indices = mg_buses.index.tolist()
                
                if len(mg_bus_indices) > 0 and len(self.net.res_bus) > 0:
                    vm_pu = self.net.res_bus.loc[mg_bus_indices, 'vm_pu'].values
                    overvoltage = np.maximum(vm_pu - 1.05, 0).sum()
                    undervoltage = np.maximum(0.95 - vm_pu, 0).sum()
                else:
                    overvoltage = 0.0
                    undervoltage = 0.0
                
                mg_lines = self.net.line[self.net.line.name.str.contains(agent)]
                mg_line_indices = mg_lines.index.tolist()
                
                if len(mg_line_indices) > 0 and len(self.net.res_line) > 0:
                    loading_pct = self.net.res_line.loc[mg_line_indices, 'loading_percent'].values
                    overloading = np.maximum(loading_pct - 100, 0).sum() * 0.01
                else:
                    overloading = 0.0
                
                agent_safety = safety.get(agent, 0.0)
                f.write(f",{overvoltage},{undervoltage},{overloading},{agent_safety}")
            
            f.write("\n")

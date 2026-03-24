"""Collision detection environment for networked microgrids.

Recreates the multi-agent collision experiment from the original platform
with IEEE 34-bus main grid and 3x IEEE 13-bus microgrids.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandapower as pp

from heron.agents.system_agent import SystemAgent, SYSTEM_AGENT_ID
from powergrid.envs.hierarchical_microgrid_env import HierarchicalMicrogridEnv
from powergrid.envs.common import EnvState
from collision_case.collision_features import CollisionMetrics, SharedRewardConfig
from collision_case.collision_network import create_collision_network


class CollisionEnv(HierarchicalMicrogridEnv):
    """Environment for collision detection experiments.

    Extends HierarchicalMicrogridEnv with:
    - Collision metrics tracking and logging
    - Shared vs independent reward schemes
    - Network topology: IEEE 34 main grid + 3x IEEE 13 microgrids
    - CSV logging of collision statistics per timestep

    Config options:
        share_reward: bool - Share rewards across all microgrids
        penalty: float - Penalty multiplier for safety violations
        log_path: str - Path to CSV log file (optional)
    """

    def __init__(
        self,
        system_agent: SystemAgent,
        dataset_path: str,
        episode_steps: int = 24,
        dt: float = 1.0,
        # Collision-specific config
        share_reward: bool = True,
        penalty: float = 10.0,
        log_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize collision detection environment.

        Args:
            system_agent: Pre-initialized SystemAgent with 3 microgrids
            dataset_path: Path to dataset file
            episode_steps: Episode length (default: 24 hours)
            dt: Time step duration in hours
            share_reward: Whether to share rewards across agents
            penalty: Safety penalty multiplier
            log_path: Path to CSV log file (optional)
            **kwargs: Additional arguments for HierarchicalMicrogridEnv
        """
        super().__init__(
            system_agent=system_agent,
            dataset_path=dataset_path,
            episode_steps=episode_steps,
            dt=dt,
            **kwargs,
        )

        # Collision config
        self._share_reward = share_reward
        self._penalty = penalty
        self._log_path = log_path

        # Episode counter for logging
        self._episode_counter = 0
        
        # Initialize log file
        if self._log_path:
            self._init_log_file()

    def _create_network_from_agents(self, global_state: Dict) -> pp.pandapowerNet:
        """Create IEEE 34 + 3x IEEE 13 network topology.
        
        Overrides parent to use real IEEE network topology instead of simplified.
        
        Args:
            global_state: Global state dict (unused, for signature compatibility)
            
        Returns:
            Pandapower network with full IEEE topology
        """
        print("Creating IEEE 34 + 3x IEEE 13 network topology...")
        net = create_collision_network()
        
        # Add storage and sgen devices based on registered agents
        for mg_id in self._get_microgrid_ids():
            mg_agent = self.registered_agents.get(mg_id)
            if not mg_agent:
                continue
                
            # Add devices for this microgrid
            for dev_id, dev_agent in mg_agent.subordinates.items():
                from powergrid.agents import ESS, Generator
                
                if isinstance(dev_agent, ESS):
                    # Determine bus for ESS (Bus 645 by default in IEEE13)
                    bus_name = f"{mg_id} Bus 645"
                    bus_idx = pp.get_element_index(net, 'bus', bus_name)
                    
                    capacity = getattr(dev_agent, 'capacity', 2.0)
                    pp.create_storage(
                        net, bus=bus_idx, p_mw=0.0, max_e_mwh=capacity,
                        soc_percent=50.0, name=dev_id, controllable=True
                    )
                    
                elif isinstance(dev_agent, Generator):
                    # Determine bus based on device type
                    if 'DG' in dev_id:
                        bus_name = f"{mg_id} Bus 675"
                    elif 'PV' in dev_id:
                        bus_name = f"{mg_id} Bus 652"
                    elif 'WT' in dev_id:
                        bus_name = f"{mg_id} Bus 645"
                    else:
                        bus_name = f"{mg_id} Bus 632"  # Default
                    
                    bus_idx = pp.get_element_index(net, 'bus', bus_name)
                    max_p = getattr(dev_agent, 'max_p', 1.0)
                    
                    pp.create_sgen(
                        net, bus=bus_idx, p_mw=max_p * 0.5, q_mvar=0.0,
                        name=dev_id, controllable=True
                    )
        
        print(f"✓ Network created: {len(net.bus)} buses, {len(net.line)} lines, "
              f"{len(net.storage)} storage, {len(net.sgen)} sgen")
        
        return net

    def _init_log_file(self) -> None:
        """Initialize CSV log file with headers."""
        if self._log_path is None:
            return

        try:
            with open(self._log_path, "w") as f:
                # Base columns
                f.write("timestep,episode,reward_sum,safety_sum")
                
                # Add columns for each microgrid
                for mg_id in self._get_microgrid_ids():
                    f.write(
                        f",{mg_id}_overvoltage,{mg_id}_undervoltage,"
                        f"{mg_id}_overloading,{mg_id}_safety_sum"
                    )
                f.write("\n")
        except Exception as e:
            print(f"Warning: Failed to initialize log file: {e}")

    def _get_microgrid_ids(self) -> list:
        """Get list of microgrid coordinator IDs.

        Returns:
            List of microgrid agent IDs (e.g., ['MG1', 'MG2', 'MG3'])
        """
        mg_ids = []
        for agent_id in self.registered_agents.keys():
            # Skip system agent and device agents
            if agent_id != SYSTEM_AGENT_ID and "MG" in agent_id and "_" not in agent_id:
                mg_ids.append(agent_id)
        return sorted(mg_ids)

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert simulation results to global state with collision metrics.

        Extracts bus voltages and line loadings from power flow results
        and passes them to microgrid coordinators for collision computation.

        Args:
            env_state: EnvState with power flow results

        Returns:
            Updated global state dict
        """
        # Get base global state from parent
        global_state = super().env_state_to_global_state(env_state)

        # Extract detailed power flow results for collision detection
        if not env_state.power_flow_results.get("converged", False):
            # Non-convergence handled in collision metrics
            bus_voltages = {}
            line_loadings = {}
        else:
            # Extract per-bus voltages
            bus_voltages = {}
            for idx, bus_name in enumerate(self.net.bus.name):
                if idx < len(self.net.res_bus):
                    bus_voltages[bus_name] = float(self.net.res_bus.iloc[idx].vm_pu)

            # Extract per-line loadings
            line_loadings = {}
            for idx, line_name in enumerate(self.net.line.name):
                if idx < len(self.net.res_line):
                    line_loadings[line_name] = float(
                        self.net.res_line.iloc[idx].loading_percent
                    )

        # Add power flow details to agent states for collision computation
        agent_states = global_state.get("agent_states", {})
        for agent_id, agent_state in agent_states.items():
            # Add power flow data for coordinator agents
            if "Coordinator" in agent_state.get("_state_type", ""):
                agent_state["bus_voltages"] = bus_voltages
                agent_state["line_loadings"] = line_loadings
                agent_state["converged"] = env_state.power_flow_results.get(
                    "converged", False
                )

        return {"agent_states": agent_states}

    def step(self, actions: Dict[str, Any]):
        """Execute one environment step with collision logging.

        Args:
            actions: Dict of {agent_id: action} for each agent

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        # Execute base step
        obs, rewards, terminated, truncated, infos = super().step(actions)

        # Apply reward sharing if enabled
        if self._share_reward:
            rewards = self._apply_reward_sharing(rewards)

        # Log collision metrics
        if self._log_path:
            self._log_collision_metrics(rewards, infos)

        # Increment episode counter on episode end
        if terminated.get("__all__", False):
            self._episode_counter += 1

        return obs, rewards, terminated, truncated, infos

    def _apply_reward_sharing(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Share rewards equally across all microgrid agents.

        Args:
            rewards: Dict of {agent_id: reward}

        Returns:
            Dict with shared rewards
        """
        # Get microgrid IDs (exclude system agent and devices)
        mg_ids = self._get_microgrid_ids()
        
        if len(mg_ids) == 0:
            return rewards

        # Compute average reward across microgrids
        mg_rewards = [rewards.get(mg_id, 0.0) for mg_id in mg_ids]
        avg_reward = float(np.mean(mg_rewards))

        # Assign shared reward to all microgrids
        shared_rewards = {}
        for agent_id in rewards.keys():
            if agent_id in mg_ids:
                shared_rewards[agent_id] = avg_reward
            else:
                shared_rewards[agent_id] = rewards[agent_id]

        return shared_rewards

    def _log_collision_metrics(
        self, rewards: Dict[str, float], infos: Dict[str, Any]
    ) -> None:
        """Log collision metrics to CSV file.

        Args:
            rewards: Dict of {agent_id: reward}
            infos: Dict of {agent_id: info_dict}
        """
        if self._log_path is None:
            return

        try:
            # Compute aggregated metrics
            reward_sum = sum(rewards.values())
            safety_sum = 0.0

            # Collect per-microgrid stats
            mg_stats = {}
            mg_ids = self._get_microgrid_ids()
            
            for mg_id in mg_ids:
                # Get collision stats from microgrid agent
                mg_agent = self.registered_agents.get(mg_id)
                if mg_agent and hasattr(mg_agent, "get_collision_stats"):
                    stats = mg_agent.get_collision_stats()
                    mg_stats[mg_id] = stats
                    safety_sum += stats.get("safety_total", 0.0)
                else:
                    mg_stats[mg_id] = {
                        "overvoltage": 0.0,
                        "undervoltage": 0.0,
                        "overloading": 0.0,
                        "safety_total": 0.0,
                    }

            # Write log entry
            with open(self._log_path, "a") as f:
                f.write(f"{self._timestep},{self._episode_counter},{reward_sum},{safety_sum}")
                
                # Add per-microgrid metrics
                for mg_id in mg_ids:
                    stats = mg_stats[mg_id]
                    f.write(
                        f",{stats['overvoltage']},{stats['undervoltage']},"
                        f"{stats['overloading']},{stats['safety_total']}"
                    )
                f.write("\n")

        except Exception as e:
            print(f"Warning: Failed to log collision metrics: {e}")

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """Reset environment and episode counter.

        Args:
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Tuple of (observations, info)
        """
        # Reset episode counter on first reset
        if not hasattr(self, "_episode_counter"):
            self._episode_counter = 0

        return super().reset(seed=seed, **kwargs)

    def get_collision_summary(self) -> Dict[str, Any]:
        """Get collision statistics summary for current episode.

        Returns:
            Dict with aggregated collision metrics
        """
        mg_ids = self._get_microgrid_ids()
        
        summary = {
            "total_safety": 0.0,
            "total_collisions": 0,
            "microgrids": {},
        }

        for mg_id in mg_ids:
            mg_agent = self.registered_agents.get(mg_id)
            if mg_agent and hasattr(mg_agent, "get_collision_stats"):
                stats = mg_agent.get_collision_stats()
                summary["microgrids"][mg_id] = stats
                summary["total_safety"] += stats.get("safety_total", 0.0)
                if stats.get("collision_flag", 0.0) > 0:
                    summary["total_collisions"] += 1

        return summary

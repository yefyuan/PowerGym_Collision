"""Hierarchical multi-agent power grid environment.

This module implements a hierarchical microgrid environment following the grid_agent style:
- SystemAgent -> PowerGridAgents -> DeviceAgents (Generator, ESS, Transformer)

Each device is its own field agent, and microgrids are coordinator agents.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandapower as pp

from heron.envs.base import HeronEnv
from heron.agents.base import AgentID, Agent
from heron.agents.system_agent import SystemAgent, SYSTEM_AGENT_ID
from powergrid.envs.common import EnvState
from powergrid.agents import (
    PowerGridAgent,
    Generator,
    ESS,
    Transformer,
)
from powergrid.utils.loader import load_dataset


class HierarchicalMicrogridEnv(HeronEnv):
    """Hierarchical multi-agent environment for networked microgrids.

    This environment supports CTDE (Centralized Training with Decentralized Execution):
    - Training: Agents share a collective reward to encourage cooperation
    - Execution: Agents can operate with limited communication
    """

    def __init__(
        self,
        system_agent: SystemAgent,  # REQUIRED: Always pass pre-initialized system agent
        dataset_path: str,
        episode_steps: int = 24,
        dt: float = 1.0,
        **kwargs,
    ):
        """Initialize hierarchical microgrid environment.

        Args:
            system_agent: Pre-initialized SystemAgent with agent hierarchy
            dataset_path: Path to dataset file
            episode_steps: Episode length in time steps (default: 24)
            dt: Time step duration in hours (default: 1.0)
            **kwargs: Additional arguments for HeronEnv

        Note:
            Rewards are computed by individual agents via compute_local_reward().
            Safety penalties and reward sharing should be configured at the agent level.
        """
        self.episode_steps = episode_steps
        self.dt = dt
        self.num_microgrids = len(system_agent.subordinates)

        # Load dataset
        self._dataset = load_dataset(dataset_path)
        self._total_days = 0

        # Initialize episode state
        self._episode = 0
        self._timestep = 0
        self._train = True

        # Current profiles (updated each timestep, accessible to agents via observations)
        self._current_profiles: Dict[str, Dict[str, float]] = {}

        # Call parent init (registers agents)
        super().__init__(
            system_agent=system_agent,
            **kwargs,
        )

        # Network will be created during first reset()
        self.net = None

    def _read_data(self, load_area: str, renew_area: str) -> Dict[str, Any]:
        """Read data from dataset with train/test split support.

        Args:
            load_area: Load area identifier (e.g., 'AVA', 'BANC', 'BANCMID')
            renew_area: Renewable energy area identifier (e.g., 'NP15')

        Returns:
            Dict with load, solar, wind, price data
        """
        split = "train" if self._train else "test"
        data = self._dataset[split]

        return {
            "load": data["load"][load_area],
            "solar": data["solar"][renew_area],
            "wind": data["wind"][renew_area],
            "price": data["price"]["0096WD_7_N001"],
        }

    def _create_network_from_agents(self, global_state: Dict) -> pp.pandapowerNet:
        """Create Pandapower network from registered agents.

        Called during first reset after agents are registered.
        Uses agent structure to build network topology.

        Args:
            global_state: Global state dict (unused, for signature compatibility)

        Returns:
            Pandapower network matching agent structure
        """
        net = pp.create_empty_network(name="Hierarchical Microgrid Network")

        # Create main grid bus (slack)
        slack_bus = pp.create_bus(net, vn_kv=12.47, name="Grid")
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0)

        # Build network from registered microgrid coordinators
        microgrids: Dict[AgentID, Agent] = {}
        for agent_id, agent in self.registered_agents.items():
            if isinstance(agent, PowerGridAgent):
                microgrids[agent_id] = agent

        # Build network from microgrids
        for mg_id, mg_agent in microgrids.items():
            # Create microgrid bus
            mg_bus = pp.create_bus(net, vn_kv=0.48, name=f"{mg_id}_Bus")

            # Connect to grid via transformer
            pp.create_transformer_from_parameters(
                net,
                hv_bus=slack_bus,
                lv_bus=mg_bus,
                sn_mva=2.5,
                vn_hv_kv=12.47,
                vn_lv_kv=0.48,
                vkr_percent=1.0,
                vk_percent=5.0,
                pfe_kw=0.1,
                i0_percent=0.1,
                name=f"{mg_id}_Trafo",
            )

            # Create devices based on subordinate agent types
            for dev_id, dev_agent in mg_agent.subordinates.items():
                if isinstance(dev_agent, ESS):
                    # Storage device
                    capacity = dev_agent.capacity if hasattr(dev_agent, "capacity") else 1.0
                    pp.create_storage(
                        net,
                        bus=mg_bus,
                        p_mw=0.0,
                        max_e_mwh=capacity,
                        name=dev_id,
                        controllable=True,
                    )
                elif isinstance(dev_agent, Generator):
                    # Generator device
                    max_p = dev_agent.max_p if hasattr(dev_agent, "max_p") else 1.0
                    pp.create_sgen(
                        net,
                        bus=mg_bus,
                        p_mw=max_p * 0.5,
                        q_mvar=0.0,
                        name=dev_id,
                        controllable=True,
                    )
                elif isinstance(dev_agent, Transformer):
                    # Transformer is already created as connection
                    pass

            # Add load for this microgrid
            pp.create_load(
                net, bus=mg_bus, p_mw=0.2, q_mvar=0.05, name=f"{mg_id}_Load"
            )

        # Calculate total days from price data
        for agent in microgrids.values():
            if hasattr(agent, "dataset") and agent.dataset is not None:
                self._total_days = len(agent.dataset["price"]) // self.episode_steps
                break

        return net

    def _update_profiles(self, timestep: int) -> None:
        """Update profiles for current timestep using real data.

        Stores profile data in env (self._current_profiles) and pushes to
        proxy agent so agents can access via global state observations.
        Does NOT modify agent state directly - env should not control
        agent state changes.

        Args:
            timestep: Current timestep index
        """
        # Read profile data from dataset
        split = "train" if self._train else "test"
        data = self._dataset.get(split, {})

        if data is None or len(data) == 0:
            return

        # Compute hour index
        price_data = data.get("price", {}).get("0096WD_7_N001", [])
        if len(price_data) == 0:
            return
        hour = timestep % len(price_data)

        # Store current profiles in env (accessible via get_current_profiles)
        self._current_profiles = {
            "price": float(price_data[hour]) if hour < len(price_data) else 0.0,
            "timestep": timestep,
            "hour": hour,
        }

        # Add area-specific data if available
        for area_key in ["load", "solar", "wind"]:
            area_data = data.get(area_key, {})
            for area_name, values in area_data.items():
                if hour < len(values):
                    self._current_profiles[f"{area_key}_{area_name}"] = float(values[hour])

        # Push profiles to proxy for agents to access via global state
        if self.proxy is not None:
            self.proxy.set_global_state({"env_context": self._current_profiles})

    def pre_step(self) -> None:
        """Update profiles at the start of each step.

        Called by SystemAgent at the beginning of execute() and tick()
        before agent actions are processed.
        """
        self._update_profiles(self._timestep)

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert global state from proxy to custom env state for simulation.

        Extracts device setpoints from agent states.

        Args:
            global_state: Dict from proxy with structure:
                {"agent_states": {agent_id: state_dict, ...}}

        Returns:
            EnvState with device setpoints for power flow
        """
        env_state = EnvState()

        agent_states = global_state.get("agent_states", {})

        # Extract device setpoints from features
        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", {})

            # Extract power setpoints from ElectricalBasePh
            if "ElectricalBasePh" in features:
                elec_feature = features["ElectricalBasePh"]
                env_state.set_device_setpoint(
                    agent_id,
                    P=elec_feature.get("P_MW", 0.0),
                    Q=elec_feature.get("Q_MVAr", 0.0),
                    in_service=elec_feature.get("in_service", True),
                )

        return env_state

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run Pandapower AC power flow simulation.

        Args:
            env_state: EnvState with device setpoints

        Returns:
            Updated EnvState with power flow results
        """
        device_setpoints = env_state.device_setpoints

        # Update Pandapower network with device setpoints
        for device_id, setpoint in device_setpoints.items():
            P = setpoint.get("P", 0.0)
            Q = setpoint.get("Q", 0.0)
            in_service = setpoint.get("in_service", True)

            # Update storage devices
            storage_idx = self.net.storage[self.net.storage.name == device_id]
            if len(storage_idx) > 0:
                idx = storage_idx.index[0]
                self.net.storage.at[idx, "p_mw"] = P
                self.net.storage.at[idx, "q_mvar"] = Q
                self.net.storage.at[idx, "in_service"] = in_service
                continue

            # Update generator devices
            sgen_idx = self.net.sgen[self.net.sgen.name == device_id]
            if len(sgen_idx) > 0:
                idx = sgen_idx.index[0]
                self.net.sgen.at[idx, "p_mw"] = P
                self.net.sgen.at[idx, "q_mvar"] = Q
                self.net.sgen.at[idx, "in_service"] = in_service

        # Run AC power flow
        try:
            pp.runpp(self.net, algorithm="nr", calculate_voltage_angles=True)
            converged = True
        except Exception:
            converged = False

        # Extract power flow results
        if converged:
            results = {
                "converged": True,
                "voltage_min": float(self.net.res_bus.vm_pu.min()),
                "voltage_max": float(self.net.res_bus.vm_pu.max()),
                "voltage_avg": float(self.net.res_bus.vm_pu.mean()),
                "max_line_loading": (
                    float(self.net.res_line.loading_percent.max() / 100.0)
                    if len(self.net.res_line) > 0
                    else 0.0
                ),
                "grid_power": (
                    float(self.net.res_ext_grid.p_mw.values[0])
                    if len(self.net.res_ext_grid) > 0
                    else 0.0
                ),
                "overvoltage": float(
                    np.maximum(self.net.res_bus.vm_pu.values - 1.05, 0).sum()
                ),
                "undervoltage": float(
                    np.maximum(0.95 - self.net.res_bus.vm_pu.values, 0).sum()
                ),
            }
        else:
            results = {
                "converged": False,
                "voltage_min": 1.0,
                "voltage_max": 1.0,
                "voltage_avg": 1.0,
                "max_line_loading": 0.0,
                "grid_power": 0.0,
                "overvoltage": 0.0,
                "undervoltage": 0.0,
            }

        env_state.update_power_flow_results(results)

        # Increment timestep for next step (pre_step will update profiles)
        self._timestep += 1

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert simulation results back to global state format.

        Updates agent states with power flow results.

        Args:
            env_state: EnvState with power flow results

        Returns:
            Updated global state dict (serialized for message passing)
        """
        power_flow_results = env_state.power_flow_results

        # Get serialized agent states (for message passing, not observations)
        agent_states = self.proxy.get_serialized_agent_states()

        # Update microgrid coordinator features with power flow results
        for agent_id, agent_state in agent_states.items():
            # Check if this is a microgrid coordinator (by checking state type)
            state_type = agent_state.get("_state_type", "")

            if "Coordinator" in state_type:
                # Update network features with power flow results if present
                features = agent_state.get("features", {})
                if "NetworkMetrics" in features:
                    features["NetworkMetrics"]["voltage_avg"] = power_flow_results.get(
                        "voltage_avg", 1.0
                    )
                    features["NetworkMetrics"]["converged"] = power_flow_results.get(
                        "converged", False
                    )

        return {"agent_states": agent_states}

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """Reset environment.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional arguments

        Returns:
            Tuple of (observations, info)
        """
        if seed is not None:
            np.random.seed(seed)

        self._episode += 1
        self._timestep = 0

        # Select random day for training
        if self._train and self._total_days > 0:
            day = np.random.randint(0, self._total_days - 1)
            self._timestep = day * self.episode_steps

        # Reset agents and proxy (clears proxy state, resets agent hierarchy)
        super().reset(seed=seed, **kwargs)

        # Push profiles now that proxy is ready (must be after proxy.reset())
        self._update_profiles(self._timestep)

        # Create network on first reset
        if self.net is None:
            global_state = self.proxy.get_global_states(
                sender_id=SYSTEM_AGENT_ID, protocol=None
            )
            self.net = self._create_network_from_agents(global_state)

        # Re-collect obs with profiles populated so dims match step()
        return self._system_agent.reset(proxy=self.proxy)

    def set_train_mode(self, train: bool = True) -> None:
        """Set training/evaluation mode.

        Args:
            train: True for training, False for evaluation
        """
        self._train = train

    def get_current_profiles(self) -> Dict[str, Any]:
        """Get current timestep profiles (price, solar, wind, load).

        Returns profile data that was updated in _update_profiles().
        Agents can access this through their observation pipeline.

        Returns:
            Dict with current profile values
        """
        return self._current_profiles.copy()

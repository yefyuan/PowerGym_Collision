"""Collision detection environment for networked microgrids.

Recreates the multi-agent collision experiment from the original platform
with IEEE 34-bus main grid and 3x IEEE 13-bus microgrids.
"""

from typing import Any, Dict, Optional

import os
import uuid
import fcntl
from pathlib import Path
import logging

import numpy as np

from heron.agents.system_agent import SystemAgent, SYSTEM_AGENT_ID
from powergrid.envs.hierarchical_microgrid_env import HierarchicalMicrogridEnv
from powergrid.envs.common import EnvState
import pandapower as pp

from collision_case.collision_features import CollisionMetrics, SharedRewardConfig
from collision_case.collision_network import (
    attach_collision_devices,
    create_collision_ieee_topology,
)

# Silence extremely noisy pandapower warnings inside Ray worker processes.
logging.getLogger("pandapower").setLevel(logging.ERROR)
logging.getLogger("pandapower.auxiliary").setLevel(logging.ERROR)


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
        # Stable identifier for this env instance (helps when many RLlib env runners
        # append to the same CSV).
        self._env_instance = uuid.uuid4().hex[:8]

        # If multiple envs run in parallel, writing to a single shared CSV makes
        # analysis ambiguous (each env has its own timestep counter) and can corrupt
        # the file with concurrent appends. Support per-env log file naming via:
        # - directory path (write one file per env into that dir)
        # - format placeholders: {pid}, {env_instance}
        if self._log_path:
            lp = Path(self._log_path).expanduser()
            if lp.exists() and lp.is_dir():
                lp = lp / f"collision_{self._env_instance}_{os.getpid()}.csv"
            else:
                # Allow templated filenames, e.g. "sync_shared_{pid}_{env_instance}.csv"
                try:
                    rendered = str(self._log_path).format(
                        pid=os.getpid(), env_instance=self._env_instance
                    )
                    lp = Path(rendered).expanduser()
                except Exception:
                    # If formatting fails, fall back to given path.
                    lp = Path(self._log_path).expanduser()
            self._log_path = str(lp)
        
        # Initialize log file
        if self._log_path:
            self._init_log_file()

        self._load_scale = 0.2
        self._normalize_collision_dataset()

    def _normalize_collision_dataset(self) -> None:
        """Support flat ``data2024.pkl`` and alias ``LMP`` → price key parent expects."""
        if "train" not in self._dataset:
            flat = self._dataset
            self._dataset = {"train": flat, "test": flat}
        for split in ("train", "test"):
            price = self._dataset[split]["price"]
            if "0096WD_7_N001" not in price and "LMP" in price:
                price["0096WD_7_N001"] = price["LMP"]
        sp = "train" if self._train else "test"
        banc = self._dataset[sp]["load"]["BANC"]
        self._data_horizon = len(banc)
        self._total_days = max(1, self._data_horizon // self.episode_steps)

    def _create_network_from_agents(self, global_state: Dict) -> pp.pandapowerNet:
        """IEEE 34 + 3× IEEE 13 + DER placement (Collision topology)."""
        net = create_collision_ieee_topology(self._load_scale)
        dg_max = {"MG1": 0.66, "MG2": 0.60, "MG3": 0.50}
        for mg_id in sorted(self._get_microgrid_ids()):
            attach_collision_devices(net, mg_id, dg_max.get(mg_id, 0.5))
        try:
            pp.runpp(net)
        except Exception:
            pass
        return net

    def _update_profiles(self, timestep: int) -> None:
        """Like parent, but tolerate missing ``0096WD_7_N001`` before alias."""
        split = "train" if self._train else "test"
        data = self._dataset.get(split, {})
        if not data:
            return
        price_tbl = data.get("price", {})
        price_series = price_tbl.get("0096WD_7_N001")
        if price_series is None:
            price_series = price_tbl.get("LMP", [])
        if len(price_series) == 0:
            return
        hour = int(timestep % len(price_series))
        self._current_profiles = {
            "price": float(price_series[hour]) if hour < len(price_series) else 0.0,
            "timestep": timestep,
            "hour": hour,
        }
        for area_key in ["load", "solar", "wind"]:
            for area_name, values in data.get(area_key, {}).items():
                if hour < len(values):
                    self._current_profiles[f"{area_key}_{area_name}"] = float(values[hour])
        if self.proxy is not None:
            self.proxy.set_global_state({"env_context": self._current_profiles})

    def pre_step(self) -> None:
        self._update_profiles(self._timestep)
        if self.net is not None:
            self._apply_ieee_load_scaling()

    def _apply_ieee_load_scaling(self) -> None:
        """Time-varying load scaling (AVA / BANCMID / AZPS / BANC), ×0.2 like Collision."""
        split = "train" if self._train else "test"
        data = self._dataset[split]
        ti = int(self._timestep % self._data_horizon)
        ls = self._load_scale

        def _scale_block(prefix: str, series):
            idx = pp.get_element_index(self.net, "load", prefix, False)
            self.net.load.loc[idx, "scaling"] = float(series[ti])
            self.net.load.loc[idx, "scaling"] *= ls

        _scale_block("DSO", data["load"]["BANC"])
        _scale_block("MG1", data["load"]["AVA"])
        _scale_block("MG2", data["load"]["BANCMID"])
        _scale_block("MG3", data["load"]["AZPS"])

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Apply stochastic renewable **P** from NP15 profiles (Q still from agents)."""
        env_state = super().global_state_to_env_state(global_state)
        sun = float(self._current_profiles.get("solar_NP15", 0.0))
        wind = float(self._current_profiles.get("wind_NP15", 0.0))
        for mg in self._get_microgrid_ids():
            for dev, factor in ((f"{mg}_PV1", sun), (f"{mg}_WT1", wind)):
                if dev in env_state.device_setpoints:
                    env_state.device_setpoints[dev]["P"] = 0.1 * factor
        return env_state

    def _init_log_file(self) -> None:
        """Initialize CSV log file with headers."""
        if self._log_path is None:
            return

        try:
            Path(self._log_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            # In RLlib we may have many env instances in parallel; avoid clobbering an
            # existing log file (multiple envs starting at different times).
            try:
                with open(self._log_path, "r") as rf:
                    if rf.read(1):
                        return
            except FileNotFoundError:
                pass

            with open(self._log_path, "w") as f:
                # Base columns
                f.write("timestep,episode,env_instance,pid,reward_sum,safety_sum")
                
                # Add columns for each microgrid
                for mg_id in self._get_microgrid_ids():
                    f.write(
                        f",{mg_id}_overvoltage,{mg_id}_undervoltage,"
                        f"{mg_id}_overloading,{mg_id}_safety_sum"
                    )
                # Debug: reactive power setpoints per microgrid (helps verify Q-control path).
                for mg_id in self._get_microgrid_ids():
                    f.write(f",{mg_id}_q_setpoint_sum")
                f.write("\n")
        except Exception as e:
            if not getattr(self, "_log_failed", False):
                self._log_failed = True
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

        # Add collision metrics directly into coordinator feature dicts so they survive
        # state serialization/sync. Top-level ad-hoc keys are not persisted by State.
        agent_states = global_state.get("agent_states", {})
        converged = bool(env_state.power_flow_results.get("converged", False))

        for mg_id in self._get_microgrid_ids():
            agent_state = agent_states.get(mg_id)
            if not agent_state:
                continue
            features = agent_state.setdefault("features", {})
            collision = self._compute_collision_from_pf(
                mg_id=mg_id,
                bus_voltages=bus_voltages,
                line_loadings=line_loadings,
                converged=converged,
                voltage_upper=1.05,
                voltage_lower=0.95,
                line_loading_limit=100.0,
            )
            # Keep any device-level terms produced by agent-side logic (if present),
            # but always overwrite PF-driven terms from this step.
            prev = features.get("CollisionMetrics", {})
            device_overrating = float(prev.get("device_overrating", 0.0))
            pf_violations = float(prev.get("pf_violations", 0.0))
            soc_violations = float(prev.get("soc_violations", 0.0))
            safety_total = (
                collision["overvoltage"]
                + collision["undervoltage"]
                + collision["overloading"]
                + device_overrating
                + pf_violations
                + soc_violations
            )
            features["CollisionMetrics"] = {
                "overvoltage": collision["overvoltage"],
                "undervoltage": collision["undervoltage"],
                "overloading": collision["overloading"],
                "device_overrating": device_overrating,
                "pf_violations": pf_violations,
                "soc_violations": soc_violations,
                "safety_total": safety_total,
                "collision_flag": 1.0 if safety_total > 0.0 else 0.0,
            }

        return {"agent_states": agent_states}

    def _compute_collision_from_pf(
        self,
        mg_id: str,
        bus_voltages: Dict[str, float],
        line_loadings: Dict[str, float],
        converged: bool,
        voltage_upper: float,
        voltage_lower: float,
        line_loading_limit: float,
    ) -> Dict[str, float]:
        """Compute MG-level collision terms from power-flow outputs."""
        if not converged:
            return {"overvoltage": 0.0, "undervoltage": 0.0, "overloading": 20.0}

        def _belongs(name: str) -> bool:
            return (mg_id in name) or name.startswith(f"{mg_id}_")

        over_v = 0.0
        under_v = 0.0
        over_l = 0.0

        for bus_name, vm_pu in bus_voltages.items():
            if not _belongs(bus_name):
                continue
            if vm_pu > voltage_upper:
                over_v += float(vm_pu - voltage_upper)
            elif vm_pu < voltage_lower:
                under_v += float(voltage_lower - vm_pu)

        for line_name, loading_pct in line_loadings.items():
            if not _belongs(line_name):
                continue
            if loading_pct > line_loading_limit:
                over_l += float((loading_pct - line_loading_limit) / 100.0)

        return {"overvoltage": over_v, "undervoltage": under_v, "overloading": over_l}

    def step(self, actions: Dict[str, Any]):
        """Execute one environment step with collision logging.

        Args:
            actions: Dict of {agent_id: action} for each agent

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        # Execute base step (agent-local rewards computed inside HERON hierarchy).
        obs, rewards, terminated, truncated, infos = super().step(actions)

        # Apply collision penalty at the environment level.
        # NOTE: Coordinator agents' `compute_local_reward()` expects CollisionMetrics
        # to be present in their local_state, but in this PowerGym+HERON wiring the
        # per-step PF violation terms are computed here in the env. Without this
        # penalty, policies do not learn to reduce overvoltage/overloading and
        # collision frequency can stay at 100%.
        try:
            mg_ids = self._get_microgrid_ids()
            state_cache = self.proxy.get_serialized_agent_states() if self.proxy else {}
            for mg_id in mg_ids:
                cm = (
                    state_cache.get(mg_id, {})
                    .get("features", {})
                    .get("CollisionMetrics", {})
                )
                safety_total = float(cm.get("safety_total", 0.0))
                if safety_total != 0.0:
                    rewards[mg_id] = float(rewards.get(mg_id, 0.0)) - float(self._penalty) * safety_total
        except Exception:
            pass

        # Apply reward sharing if enabled (share after penalties).
        if self._share_reward:
            rewards = self._apply_reward_sharing(rewards)

        # Log collision metrics
        if self._log_path:
            self._log_collision_metrics(rewards, infos)

        # Always attach collision metrics to info dicts so RLlib callbacks can
        # aggregate/print them (works even when CSV logging is disabled).
        try:
            mg_ids = self._get_microgrid_ids()
            state_cache = self.proxy.get_serialized_agent_states() if self.proxy else {}
            mg_stats: Dict[str, Dict[str, float]] = {}
            safety_sum = 0.0
            for mg_id in mg_ids:
                cm = (
                    state_cache.get(mg_id, {})
                    .get("features", {})
                    .get("CollisionMetrics", {})
                )
                stats = {
                    "overvoltage": float(cm.get("overvoltage", 0.0)),
                    "undervoltage": float(cm.get("undervoltage", 0.0)),
                    "overloading": float(cm.get("overloading", 0.0)),
                    "safety_total": float(cm.get("safety_total", 0.0)),
                }
                mg_stats[mg_id] = stats
                safety_sum += stats["safety_total"]

            payload = {
                "safety_sum": float(safety_sum),
                "collision_flag": float(safety_sum != 0.0),
                "microgrids": mg_stats,
            }
            for mg_id in mg_ids:
                if mg_id in infos:
                    infos[mg_id]["collision"] = payload
        except Exception:
            # Never break training because of logging/metrics.
            pass

        # Increment episode counter on episode boundary (wrapper truncates at episode_steps).
        # `self._timestep` is already incremented inside the simulation step.
        if self.episode_steps > 0 and (self._timestep % int(self.episode_steps) == 0):
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
            Path(self._log_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            # Compute aggregated metrics
            reward_sum = sum(rewards.values())
            safety_sum = 0.0

            # Collect per-microgrid stats from proxy state (source of truth for this step)
            mg_stats = {}
            mg_ids = self._get_microgrid_ids()
            state_cache = self.proxy.get_serialized_agent_states() if self.proxy else {}

            for mg_id in mg_ids:
                cm = (
                    state_cache.get(mg_id, {})
                    .get("features", {})
                    .get("CollisionMetrics", {})
                )
                stats = {
                    "overvoltage": float(cm.get("overvoltage", 0.0)),
                    "undervoltage": float(cm.get("undervoltage", 0.0)),
                    "overloading": float(cm.get("overloading", 0.0)),
                    "safety_total": float(cm.get("safety_total", 0.0)),
                }
                mg_stats[mg_id] = stats
                safety_sum += stats["safety_total"]

            # Write log entry
            with open(self._log_path, "a") as f:
                # Serialize concurrent writers (multiple env runners) to prevent
                # interleaved CSV lines.
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(
                    f"{self._timestep},{self._episode_counter},{self._env_instance},{os.getpid()},"
                    f"{reward_sum},{safety_sum}"
                )
                
                # Add per-microgrid metrics
                for mg_id in mg_ids:
                    stats = mg_stats[mg_id]
                    f.write(
                        f",{stats['overvoltage']},{stats['undervoltage']},"
                        f"{stats['overloading']},{stats['safety_total']}"
                    )
                # Add Q setpoint sums (from pandapower net)
                for mg_id in mg_ids:
                    q_sum = 0.0
                    if self.net is not None:
                        if hasattr(self.net, "sgen") and len(self.net.sgen) > 0:
                            sg = self.net.sgen[self.net.sgen.name.astype(str).str.startswith(f"{mg_id}_")]
                            if len(sg) > 0:
                                q_sum += float(sg.q_mvar.astype(float).sum())
                        if hasattr(self.net, "storage") and len(self.net.storage) > 0:
                            st = self.net.storage[self.net.storage.name.astype(str).str.startswith(f"{mg_id}_")]
                            if len(st) > 0:
                                q_sum += float(st.q_mvar.astype(float).sum())
                    f.write(f",{q_sum}")
                f.write("\n")
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            if not getattr(self, "_log_failed", False):
                self._log_failed = True
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

        state_cache = self.proxy.get_serialized_agent_states() if self.proxy else {}
        for mg_id in mg_ids:
            cm = (
                state_cache.get(mg_id, {})
                .get("features", {})
                .get("CollisionMetrics", {})
            )
            stats = {
                "overvoltage": float(cm.get("overvoltage", 0.0)),
                "undervoltage": float(cm.get("undervoltage", 0.0)),
                "overloading": float(cm.get("overloading", 0.0)),
                "device_overrating": float(cm.get("device_overrating", 0.0)),
                "pf_violations": float(cm.get("pf_violations", 0.0)),
                "soc_violations": float(cm.get("soc_violations", 0.0)),
                "safety_total": float(cm.get("safety_total", 0.0)),
                "collision_flag": float(cm.get("collision_flag", 0.0)),
            }
            summary["microgrids"][mg_id] = stats
            summary["total_safety"] += stats["safety_total"]
            if stats["collision_flag"] > 0:
                summary["total_collisions"] += 1

        return summary

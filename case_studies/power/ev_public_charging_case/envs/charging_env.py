"""Multi-station EV charging environment using HERON HeronEnv.

Follows the same pattern as powergrid/envs/hierarchical_microgrid_env.py:
- Extends HeronEnv (which extends BaseEnv)
- Implements the 3 abstract simulation methods
- Receives coordinator_agents, BaseEnv auto-creates SystemAgent
- CTDE training via system_agent.execute() → layer_actions → act → simulate
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.envs.base import HeronEnv
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.utils.typing import AgentID, MultiAgentDict

from case_studies.power.ev_public_charging_case.envs.common import EnvState, SlotState
from case_studies.power.ev_public_charging_case.envs.market_scenario import MarketScenario
from case_studies.power.ev_public_charging_case.envs.regulation_scenario import RegulationScenario


class ChargingEnv(HeronEnv):
    """Multi-station EV public charging environment."""

    def __init__(
        self,
        coordinator_agents: List[CoordinatorAgent],
        arrival_rate: float = 10.0,
        dt: float = 300.0,
        episode_length: float = 86400.0,
        env_id: str = "ev_charging_env",
        # Regulation scenario params (Route A: metrics only)
        reg_freq: float = 4.0,
        reg_alpha: float = 0.2,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.dt = float(dt)
        self.episode_length = float(episode_length)
        self._arrival_rate = float(arrival_rate)

        self._time_s = 0.0

        # RNG for reproducibility (affects arrivals assignment + SOC sampling)
        self._rng = np.random.default_rng(seed)

        # External scenarios
        self.scenario = MarketScenario(self._arrival_rate, 3600.0)
        self.reg_scenario = RegulationScenario(reg_freq=reg_freq, alpha=reg_alpha, seed=seed or 0)

        # Build slot → station mapping from coordinator subordinates
        self._slot_to_station: Dict[str, str] = {}
        for coord in coordinator_agents:
            for slot_id in coord.subordinates:
                self._slot_to_station[str(slot_id)] = str(coord.agent_id)

        super().__init__(
            coordinator_agents=coordinator_agents,
            env_id=env_id,
            **kwargs,
        )

    # ============================================
    # Lifecycle overrides
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset environment state for a new episode."""

        self.scenario = MarketScenario(self._arrival_rate, 3600.0)
        # reset regulation scenario clock too
        self.reg_scenario = RegulationScenario(
            reg_freq=self.reg_scenario.reg_freq,
            alpha=self.reg_scenario.alpha,
            seed=(seed or 0),
        )

        self._time_s = 0.0
        return super().reset(seed=seed, **kwargs)

    def step(self, actions: Dict[AgentID, Any]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one step and add __all__ + episode truncation."""
        obs, rewards, terminated, truncated, infos = super().step(actions)

        any_terminated = any(v for k, v in terminated.items() if k != "__all__")
        terminated["__all__"] = any_terminated

        time_up = self._time_s >= self.episode_length
        truncated["__all__"] = time_up

        # Route A: attach regulation metrics to system info (if present)
        # We store them under infos["__all__"] as a convenient place.
        if "__all__" not in infos:
            infos["__all__"] = {}
        infos["__all__"].update(getattr(self, "_latest_reg_metrics", {}))

        return obs, rewards, terminated, truncated, infos

    # ============================================
    # Abstract simulation methods (required by BaseEnv)
    # ============================================

    def pre_step(self) -> None:
        """Advance market scenario clock (called at start of each step)."""
        # Market update is done inside run_simulation
        return

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> EnvState:
        """Extract simulation inputs from proxy global state."""
        agent_states = global_state.get("agent_states", {})
        env_state = EnvState(
            slot_to_station=dict(self._slot_to_station),
            dt=self.dt,
            time_s=self._time_s,
        )

        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", state_dict)

            # Coordinator agent → extract pricing
            if "ChargingStationFeature" in features:
                csf = features["ChargingStationFeature"]
                env_state.station_prices[str(agent_id)] = float(csf.get("charging_price", 0.25))

            # Charging slot → extract charger + EV slot state
            if "ChargerFeature" in features and "EVSlotFeature" in features:
                cf = features["ChargerFeature"]
                ef = features["EVSlotFeature"]
                env_state.slot_states[str(agent_id)] = SlotState(
                    p_kw=float(cf.get("p_kw", 0.0)),
                    p_max_kw=float(cf.get("p_max_kw", 150.0)),
                    open_or_not=int(cf.get("open_or_not", 1)),
                    occupied=int(ef.get("occupied", 0)),
                    soc=float(ef.get("soc", 0.0)),
                    soc_target=float(ef.get("soc_target", 0.8)),
                    arrival_time=float(ef.get("arrival_time", 0.0)),
                    max_wait_time=float(ef.get("max_wait_time", 3600.0)),
                    price_sensitivity=float(ef.get("price_sensitivity", 0.5)),
                )

        return env_state

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run one step of EV charging simulation."""
        # 1) Advance market scenario
        scenario_data = self.scenario.step(self.dt)
        self._time_s = float(scenario_data["t"])
        env_state.lmp = float(scenario_data["lmp"])
        env_state.time_s = float(scenario_data["t"])
        env_state.new_arrivals = int(scenario_data["arrivals"])

        # 1b) Advance regulation scenario (Route A: metrics only)
        reg_data = self.reg_scenario.step(self.dt)
        env_state.reg_signal = float(reg_data["reg_signal"])

        # 2) EV arrivals — assign to random empty slots
        empty_slots = [
            sid for sid, ss in env_state.slot_states.items()
            if ss.occupied == 0 and ss.open_or_not == 1
        ]
        num_to_assign = min(env_state.new_arrivals, len(empty_slots))
        if num_to_assign > 0:
            chosen = self._rng.choice(empty_slots, size=num_to_assign, replace=False)
            for slot_id in chosen:
                ss = env_state.slot_states[slot_id]
                ss.occupied = 1
                ss.soc = float(self._rng.uniform(0.1, 0.3))
                ss.soc_target = float(self._rng.uniform(0.7, 0.95))
                ss.price_sensitivity = float(self._rng.uniform(0.2, 0.8))
                ss.arrival_time = env_state.time_s

        # 3) Charging physics — compute p_kw from price + occupancy, update SOC
        for slot_id, ss in env_state.slot_states.items():
            ss.revenue = 0.0
            if ss.occupied == 0 or ss.open_or_not == 0:
                ss.p_kw = 0.0
                continue

            station_id = env_state.slot_to_station.get(slot_id)
            price = float(env_state.station_prices.get(station_id, 0.25))

            # Price-responsive charging: charge at max if price is viable for EV
            if price < ss.price_sensitivity * 1.2:
                ss.p_kw = ss.p_max_kw
            else:
                ss.p_kw = 0.0

            energy_kwh = ss.p_kw * self.dt / 3600.0
            if energy_kwh > 0 and ss.soc < ss.soc_target:
                battery_kwh = 75.0
                delta_soc = energy_kwh / battery_kwh
                ss.soc = min(1.0, ss.soc + delta_soc)
                ss.revenue = (price - env_state.lmp) * energy_kwh

        # 4) EV departures — slots where SOC >= target or max wait exceeded
        for slot_id, ss in env_state.slot_states.items():
            if ss.occupied == 0:
                continue
            time_connected = env_state.time_s - ss.arrival_time
            if ss.soc >= ss.soc_target or time_connected > ss.max_wait_time:
                ss.occupied = 0
                ss.soc = 0.0
                ss.p_kw = 0.0

        # 5) Aggregate station power/capacity (open chargers only)
        station_power: Dict[str, float] = {}
        station_capacity: Dict[str, float] = {}
        for slot_id, ss in env_state.slot_states.items():
            st = env_state.slot_to_station.get(slot_id)
            if st is None:
                continue
            station_power.setdefault(st, 0.0)
            station_capacity.setdefault(st, 0.0)

            if ss.open_or_not == 1:
                station_capacity[st] += float(ss.p_max_kw)
                # actual power only if occupied/open (we already set p_kw=0 if not)
                station_power[st] += float(ss.p_kw)

        env_state.station_power = station_power
        env_state.station_capacity = station_capacity

        # 6) Route A metrics: compute target/error/violation at station level
        # Target: request a fraction of open capacity
        alpha = float(self.reg_scenario.alpha)
        errors = []
        violation_seconds = 0.0
        max_abs_error = 0.0

        for st, cap in station_capacity.items():
            p_act = station_power.get(st, 0.0)
            # regulation target around 0 baseline (metrics-only)
            p_tgt = alpha * env_state.reg_signal * cap

            err = p_act - p_tgt
            errors.append(err)
            max_abs_error = max(max_abs_error, abs(err))

            # violation: requesting nonzero when cap is zero, or request exceeds cap (rare here)
            if cap <= 1e-9 and abs(p_tgt) > 1e-6:
                violation_seconds += self.dt

        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

        self._latest_reg_metrics = {
            "reg_signal": float(env_state.reg_signal),
            "reg_rmse": rmse,
            "reg_max_abs_error": float(max_abs_error),
            "reg_violation_seconds": float(violation_seconds),
        }

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict[str, Any]:
        """Convert simulation results back to proxy global state format."""
        agent_states: Dict[str, Any] = {}

        # Update slot agent states (FieldAgent, level 1)
        for slot_id, ss in env_state.slot_states.items():
            agent_states[slot_id] = {
                "_owner_id": slot_id,
                "_owner_level": 1,
                "_state_type": "FieldAgentState",
                "features": {
                    "ChargerFeature": {
                        "p_kw": ss.p_kw,
                        "p_max_kw": ss.p_max_kw,
                        "open_or_not": ss.open_or_not,
                    },
                    "EVSlotFeature": {
                        "occupied": ss.occupied,
                        "soc": ss.soc,
                        "soc_target": ss.soc_target,
                        "arrival_time": ss.arrival_time,
                        "max_wait_time": ss.max_wait_time,
                        "price_sensitivity": ss.price_sensitivity,
                    },
                },
            }

        # Update coordinator agent states (level 2)
        for station_id, price in env_state.station_prices.items():
            # Count open chargers for this station (occupied==0 means open/available)
            station_slots = [sid for sid, st in env_state.slot_to_station.items() if st == station_id]
            open_count = sum(
                1 for sid in station_slots
                if sid in env_state.slot_states and env_state.slot_states[sid].occupied == 0 and env_state.slot_states[sid].open_or_not == 1
            )

            # RegulationFeature write-back (so StationCoordinator obs last 3 dims are non-zero)
            cap = float(env_state.station_capacity.get(station_id, 0.0))
            p_act = float(env_state.station_power.get(station_id, 0.0))
            if cap > 1e-9:
                headroom_up = (cap - p_act) / cap
                headroom_down = p_act / cap
            else:
                headroom_up = 0.0
                headroom_down = 0.0

            agent_states[station_id] = {
                "_owner_id": station_id,
                "_owner_level": 2,
                "_state_type": "CoordinatorAgentState",
                "features": {
                    "ChargingStationFeature": {
                        "charging_price": float(price),
                        "open_chargers": int(open_count),
                    },
                    "MarketFeature": {
                        "lmp": float(env_state.lmp),
                        "t_day_s": float(env_state.time_s),
                    },
                    "RegulationFeature": {
                        "reg_signal": float(env_state.reg_signal),
                        "headroom_up": float(headroom_up),
                        "headroom_down": float(headroom_down),
                    },
                },
            }

        return {"agent_states": agent_states}
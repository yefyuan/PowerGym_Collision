"""Environment physics / simulation functions for HMARL-CBF case studies.

case1_simulation -- wind drag + fleet progress tracking (no safety filter)
case2_simulation -- wind drag + CBF safety filter + fleet progress tracking
"""

from typing import Dict

import numpy as np

# CBF safety parameters (used in case2 only)
CBF_SAFETY_RADIUS = 0.1   # minimum allowed separation between drones
CBF_REPULSION_GAIN = 0.5  # strength of the CBF repulsion correction


def _collect_drone_positions(agent_states: dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Apply wind drag and group drone positions by fleet.

    Returns:
        {fleet_id: {drone_id: {"x_pos": float, "y_pos": float}}}
    """
    fleet_drones: Dict[str, Dict[str, Dict[str, float]]] = {}
    for aid, features in agent_states.items():
        if "DronePositionFeature" not in features:
            continue

        # Wind drag: drones drift backward slightly
        x = features["DronePositionFeature"]["x_pos"]
        x -= 0.005
        features["DronePositionFeature"]["x_pos"] = float(np.clip(x, 0.0, 1.0))

        # Group by fleet: drone_0_1 -> fleet_0
        parts = aid.split("_")
        if len(parts) >= 2:
            fleet_id = f"fleet_{parts[1]}"
            fleet_drones.setdefault(fleet_id, {})[aid] = {
                "x_pos": features["DronePositionFeature"]["x_pos"],
                "y_pos": features["DronePositionFeature"]["y_pos"],
            }

    return fleet_drones


def _update_fleet_features(agent_states: dict, fleet_drones: dict) -> None:
    """Aggregate drone states into fleet-level features."""
    for fleet_id, positions in fleet_drones.items():
        if fleet_id not in agent_states or "FleetSafetyFeature" not in agent_states[fleet_id]:
            continue

        coords = np.array([[p["x_pos"], p["y_pos"]] for p in positions.values()])
        n = len(coords)

        # Mean pairwise separation
        if n >= 2:
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(np.linalg.norm(coords[i] - coords[j]))
            mean_sep = float(np.mean(dists))
        else:
            mean_sep = 1.0

        # Payload progress: mean x-position (goal is x=1.0)
        mean_x = float(np.mean(coords[:, 0]))

        agent_states[fleet_id]["FleetSafetyFeature"]["mean_separation"] = mean_sep
        agent_states[fleet_id]["FleetSafetyFeature"]["payload_progress"] = mean_x


def cbf_safety_filter(positions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Control Barrier Function filter enforcing minimum drone separation.

    For each pair of drones, if their Euclidean distance falls below
    CBF_SAFETY_RADIUS, apply a repulsive correction proportional to the
    barrier violation: dx_i += gamma * (h_safe - d) * (p_i - p_j) / d.

    This guarantees forward invariance of the safe set {d(i,j) >= r}
    without requiring the policy to learn collision avoidance.
    """
    drone_ids = list(positions.keys())
    n = len(drone_ids)

    for i in range(n):
        for j in range(i + 1, n):
            di, dj = drone_ids[i], drone_ids[j]
            pi = np.array([positions[di]["x_pos"], positions[di]["y_pos"]])
            pj = np.array([positions[dj]["x_pos"], positions[dj]["y_pos"]])

            dist = np.linalg.norm(pi - pj)
            if dist < CBF_SAFETY_RADIUS and dist > 1e-6:
                direction = (pi - pj) / dist
                correction = CBF_REPULSION_GAIN * (CBF_SAFETY_RADIUS - dist)

                pi_new = pi + correction * direction
                pj_new = pj - correction * direction

                positions[di]["x_pos"] = float(np.clip(pi_new[0], 0.0, 1.0))
                positions[di]["y_pos"] = float(np.clip(pi_new[1], 0.0, 1.0))
                positions[dj]["x_pos"] = float(np.clip(pj_new[0], 0.0, 1.0))
                positions[dj]["y_pos"] = float(np.clip(pj_new[1], 0.0, 1.0))

    return positions


# ── Case 1: wind drag only (no CBF) ─────────────────────────────

def case1_simulation(agent_states: dict) -> dict:
    """Apply wind drag and update fleet features. No safety filter."""
    fleet_drones = _collect_drone_positions(agent_states)
    _update_fleet_features(agent_states, fleet_drones)
    return agent_states


# ── Case 2: wind drag + CBF safety filter ────────────────────────

def case2_simulation(agent_states: dict) -> dict:
    """Apply wind drag, CBF safety filter, and update fleet features."""
    fleet_drones = _collect_drone_positions(agent_states)

    # Apply CBF safety filter per fleet
    for fleet_id, positions in fleet_drones.items():
        corrected = cbf_safety_filter(positions)
        for did, pos in corrected.items():
            agent_states[did]["DronePositionFeature"]["x_pos"] = pos["x_pos"]
            agent_states[did]["DronePositionFeature"]["y_pos"] = pos["y_pos"]

    _update_fleet_features(agent_states, fleet_drones)
    return agent_states

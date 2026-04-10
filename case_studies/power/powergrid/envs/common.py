"""Common utilities and classes shared across power grid environments."""

from typing import Any, Dict


class EnvState:
    """Custom environment state for power flow simulation.

    Stores device setpoints and power flow results.
    Used for both flat and hierarchical environments.
    """

    def __init__(self):
        self.device_setpoints: Dict[str, Dict[str, float]] = {}  # {device_id: {P, Q, ...}}
        self.power_flow_results: Dict[str, Any] = {}
        self.converged: bool = False

    def set_device_setpoint(
        self,
        device_id: str,
        P: float = 0.0,
        Q: float = 0.0,
        in_service: bool = True,
    ) -> None:
        """Set power setpoint for a device.

        Args:
            device_id: Device identifier
            P: Active power (MW)
            Q: Reactive power (MVAr)
            in_service: Whether device is in service
        """
        self.device_setpoints[device_id] = {
            "P": P,
            "Q": Q,
            "in_service": in_service,
        }

    def update_setpoints(self, setpoints: Dict[str, Dict[str, float]]) -> None:
        """Batch update device setpoints.

        Args:
            setpoints: Dict mapping device_id to {"P": ..., "Q": ...}
        """
        self.device_setpoints.update(setpoints)

    def update_power_flow_results(self, results: Dict[str, Any]) -> None:
        """Update power flow results.

        Args:
            results: Dict with voltage, line loading, convergence info
        """
        self.power_flow_results = results
        self.converged = results.get("converged", False)

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "device_setpoints": self.device_setpoints,
            "power_flow_results": self.power_flow_results,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EnvState":
        """Create from dict."""
        state = cls()
        state.device_setpoints = data.get("device_setpoints", {})
        state.power_flow_results = data.get("power_flow_results", {})
        state.converged = data.get("converged", False)
        return state

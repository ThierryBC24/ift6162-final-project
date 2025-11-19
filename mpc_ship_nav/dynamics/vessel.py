from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..charts.environment import ChartEnvironment


@dataclass
class VesselState:
    lat: float
    lon: float
    psi: float  # heading [rad]
    v: float  # speed [m/s]
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class VesselParams:
    max_yaw_rate: float  # [rad/s]
    length: float = 50.0
    width: float = 10.0


class Vessel:
    def __init__(self, state: VesselState, params: VesselParams):
        self.state = state
        self.params = params

    def step(
        self, u: float, dt: float, chart_env: Optional["ChartEnvironment"] = None
    ) -> None:
        """Update vessel state using kinematic model."""
        u = np.clip(u, -self.params.max_yaw_rate, self.params.max_yaw_rate)

        self.state.psi = self.state.psi + u
        self.state.psi = np.arctan2(np.sin(self.state.psi), np.cos(self.state.psi))

        self.state.lon += self.state.v * np.cos(self.state.psi) * dt
        self.state.lat += self.state.v * np.sin(self.state.psi) * dt

        if chart_env is not None:
            self.state.x, self.state.y = chart_env.to_local(
                self.state.lat, self.state.lon
            )

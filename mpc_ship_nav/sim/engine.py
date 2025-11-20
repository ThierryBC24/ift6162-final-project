from dataclasses import dataclass, field, replace
from typing import List, Protocol, Optional
import numpy as np

from ..charts.environment import ChartEnvironment
from ..dynamics.traffic import Scenario
from ..dynamics.vessel import Vessel, VesselState


class Controller(Protocol):
    """Interface that any local controller must implement."""

    def compute_control(
        self,
        t: float,
        own_ship: Vessel,
        other_vessels: List[Vessel],
        env: ChartEnvironment,
    ) -> float:
        """Return control input (e.g. yaw rate [rad/s]) at time t."""
        ...


@dataclass
class SimConfig:
    dt: float = 1.0           # time step [s]
    t_final: float = 600.0    # total simulation time [s]
    log_interval: int = 1     # log every N steps


@dataclass
class SimLog:
    times: List[float] = field(default_factory=list)
    own_states: List[VesselState] = field(default_factory=list)
    traffic_states: List[List[VesselState]] = field(default_factory=list)
    controls: List[float] = field(default_factory=list)


class Simulator:
    def __init__(
        self,
        env: ChartEnvironment,
        scenario: Scenario,
        controller: Controller,
        config: SimConfig,
    ) -> None:
        self.env = env
        self.scenario = scenario
        self.controller = controller
        self.cfg = config

        self.own_ship: Vessel = scenario.own_ship
        self.other_vessels: List[Vessel] = list(scenario.other_vessels)

        # Make sure x,y are set in local frame at t=0
        self._sync_local_coords()

    def _sync_local_coords(self) -> None:
        """Ensure all vessels have x,y in local coordinates."""
        for v in [self.own_ship] + self.other_vessels:
            if v.state.x is None or v.state.y is None:
                x, y = self.env.to_local(v.state.lat, v.state.lon)
                v.state.x = x
                v.state.y = y

    def run(self) -> SimLog:
        """Run the forward simulation and return logs."""
        log = SimLog()
        dt = self.cfg.dt
        n_steps = int(self.cfg.t_final / dt)

        for k in range(n_steps + 1):
            t = k * dt

            # --- logging (store copies of states) ---
            if k % self.cfg.log_interval == 0:
                log.times.append(t)

                # Copy own ship state
                log.own_states.append(replace(self.own_ship.state))

                # Copy every traffic vessel state
                log.traffic_states.append(
                    [replace(v.state) for v in self.other_vessels]
                )

            # --- compute control for own ship ---
            u = self.controller.compute_control(
                t, self.own_ship, self.other_vessels, self.env
            )
            log.controls.append(u)

            # --- advance own ship ---
            self.own_ship.step(u, dt, chart_env=self.env)

            # --- advance other vessels (simple kinematics, no control) ---
            for v in self.other_vessels:
                v.step(0.0, dt, chart_env=self.env)

        return log

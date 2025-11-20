from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.traffic import TrafficGenerator
from mpc_ship_nav.sim.engine import Simulator, SimConfig, SimLog


class DummyHeadingController:
    """Very simple controller: hold current heading (no collision avoidance)."""

    def compute_control(self, t, own_ship, other_vessels, env):
        return 0.0  # no yaw rate change

def plot_trajectories(
    env: ChartEnvironment,
    log: SimLog,
    ax: Optional[plt.Axes] = None,
):
    """Plot own ship + traffic trajectories on the chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Base map (land)
    env.plot_base_map(ax)

    # Own ship trajectory
    own_x = [s.x for s in log.own_states if s.x is not None]
    own_y = [s.y for s in log.own_states if s.y is not None]
    ax.plot(own_x, own_y, "-o", ms=2, label="own ship")

    # Traffic trajectories
    if log.traffic_states:
        n_traffic = len(log.traffic_states[0])
        for idx in range(n_traffic):
            xs = []
            ys = []
            for step_states in log.traffic_states:
                s = step_states[idx]
                if s.x is not None and s.y is not None:
                    xs.append(s.x)
                    ys.append(s.y)
            ax.plot(xs, ys, "--", label=f"traffic {idx+1}")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")
    ax.legend()
    ax.set_title("Ship trajectories on chart")

    return ax

# examples/run_scenario_head_on.py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.traffic import TrafficGenerator
from mpc_ship_nav.sim.engine import Simulator, SimConfig, Controller
from mpc_ship_nav.sim.visualize import plot_trajectories
from mpc_ship_nav.charts.planner import GlobalPlanner, PlannerConfig

class WaypointController:
    def __init__(self, waypoints, gain=0.8):
        self.waypoints = waypoints
        self.i = 0
        self.gain = gain

    def compute_control(self, t, own_ship, other_vessels, env):
        x, y = own_ship.state.x, own_ship.state.y

        if self.i >= len(self.waypoints):
            return 0.0

        wx, wy = self.waypoints[self.i]
        dx, dy = wx - x, wy - y
        target_heading = np.arctan2(dy, dx)

        heading_error = target_heading - own_ship.state.psi

        if np.hypot(dx, dy) < 50:
            self.i += 1

        return self.gain * heading_error


def main():
    DATA_DIR = "../../data/GSHHS_h_L1.shp"

    # 1) Region Config (roughly centered around scenario)
    cfg = RegionConfig(
        lat_min=43.2,
        lat_max=43.8,
        lon_min=15.6,
        lon_max=16.8,
        grid_resolution=300.0,
        coastal_buffer=500.0,
        coastline_file=DATA_DIR,
    )
    env = ChartEnvironment(cfg)

    # 2) Build a COLREG encounter scenario from TrafficGenerator
    scenario = TrafficGenerator.head_on_encounter(
        lat_center=43.45491,
        lon_center=15.97686,
        own_speed=5.0,
        other_speed=5.0,
    )

    # 3) Simulation config
    sim_cfg = SimConfig(dt=1.0, t_final=800.0)
    env.set_origin(
        scenario.own_ship.state.lat,
        scenario.own_ship.state.lon,
    )

    planner = GlobalPlanner(env, PlannerConfig(use_theta_star=True))

    start_xy = env.to_local(scenario.own_ship.state.lat,
                            scenario.own_ship.state.lon)

    goal_lat = scenario.own_ship.state.lat + 0.05
    goal_lon = scenario.own_ship.state.lon + 0.04
    goal_xy = env.to_local(goal_lat, goal_lon)

    waypoints = planner.plan(start_xy, goal_xy)

    # 4) Simulator + controller
    controller = WaypointController(waypoints)
    sim = Simulator(env, scenario, controller, sim_cfg)
    log = sim.run()

    # 5) Plot trajectories
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_trajectories(env, log, ax=ax)
    
    # Plot waypoints along the route as markers-
    wx, wy = zip(*waypoints)
    ax.scatter(wx, wy, c="orange", s=15, zorder=4, label="waypoints")
    plt.show()


if __name__ == "__main__":
    main()

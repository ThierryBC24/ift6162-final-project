import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.traffic import TrafficGenerator
from mpc_ship_nav.sim.engine import Simulator, SimConfig, Controller
from mpc_ship_nav.sim.visualize import plot_trajectories, animate_trajectories
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
    
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometers (rough but good enough)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def auto_bounds(lat1, lon1, lat2, lon2):
    """
    Paper-faithful auto-bounds:
    - small padding for coastal (~<150 km)
    - medium padding for regional (150–300 km)
    - reject trips that are too long for the paper's scale (>300 km)
    """

    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    print(f"[auto_bounds] trip distance ~{dist_km:.1f} km")

    # Case 1: local coastal navigation (< 150 km)
    if dist_km < 150.0:
        pad = 0.3      # degrees
        max_pad = 0.5  # sanity cap
        print("[auto_bounds] Using small coastal bounding box")

    # Case 2: regional navigation (150–300 km)
    elif dist_km < 300.0:
        pad = 0.5
        max_pad = 1.5
        print("[auto_bounds] Using medium regional bounding box")

    # Case 3: beyond paper-scale scope
    else:
        raise RuntimeError(
            f"[auto_bounds] Trip distance {dist_km:.1f} km exceeds the "
            "intended paper-scale navigation range. Split the route into "
            "smaller legs or use a different planner."
        )

    lat_min = min(lat1, lat2) - pad
    lat_max = max(lat1, lat2) + pad
    lon_min = min(lon1, lon2) - pad
    lon_max = max(lon1, lon2) + pad

    # # Extra guard: if something goes weird and the box is too large
    # if (lat_max - lat_min) > (2 * max_pad) or (lon_max - lon_min) > (2 * max_pad):
    #     raise RuntimeError("[auto_bounds] Computed bounds exceed safe limit.")

    print("Computed auto-bounds:")
    print(" lat:", lat_min, "→", lat_max)
    print(" lon:", lon_min, "→", lon_max)

    return lat_min, lat_max, lon_min, lon_max

FILE_PATH = Path(__file__).resolve()
def main():
    DATA_DIR =  FILE_PATH.parent.parent.parent / "data" / "GSHHS_h_L1.shp"

    lat_start, lon_start = 43.45491, 15.97686
    lat_goal,  lon_goal  = 43.09396, 17.16086
    lat_min, lat_max, lon_min, lon_max = auto_bounds(
        lat_start, lon_start, lat_goal, lon_goal
    )

    # 1) Region Config (roughly centered around scenario)
    cfg = RegionConfig(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
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
    sim_cfg = SimConfig(dt=1.0, t_final=20000, log_interval=100)
    env.set_origin(
        scenario.own_ship.state.lat,
        scenario.own_ship.state.lon,
    )

    planner = GlobalPlanner(env, PlannerConfig(use_theta_star=True))

    start_xy = env.to_local(scenario.own_ship.state.lat,
                            scenario.own_ship.state.lon)

    goal_lat = 43.09396
    goal_lon = 17.16086
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

    # Animated playback
    x1, y1 = env.to_local(lat_min, lon_min)
    x2, y2 = env.to_local(lat_max, lon_max)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    bounds = (x_min, x_max, y_min, y_max)
    animate_trajectories(env, log, waypoints, fps=10, save_path= FILE_PATH.parent / "toy_scenario.mp4", bounds=bounds)

if __name__ == "__main__":
    main()

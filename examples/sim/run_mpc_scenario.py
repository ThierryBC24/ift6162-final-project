import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from mpc_ship_nav.sim.engine import Simulator, SimConfig
from mpc_ship_nav.mpc.mpc_controller import SimplifiedMPCController
from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.charts.planner import GlobalPlanner, PlannerConfig
from mpc_ship_nav.sim.visualize import plot_trajectories, animate_trajectories

from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams
from mpc_ship_nav.dynamics.traffic import GenerateTraffic, Scenario

def spawn_random_background_vessels(
    env: ChartEnvironment,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_vessels: int,
    speed_range=(3.0, 8.0),
    max_yaw_rate_deg: float = 20.0,
) -> List[Vessel]:
    vessels: List[Vessel] = []

    # Precompute local bounds to sample in (x,y)
    x1, y1 = env.to_local(lat_min, lon_min)
    x2, y2 = env.to_local(lat_max, lon_max)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    for _ in range(n_vessels):
        for _ in range(200):  # max tries per vessel
            x = float(np.random.uniform(x_min, x_max))
            y = float(np.random.uniform(y_min, y_max))

            if not env.is_navigable(x, y):
                continue

            lat, lon = env.to_geo(x, y)
            psi = float(np.random.uniform(-np.pi, np.pi))
            v = float(np.random.uniform(*speed_range))

            state = VesselState(lat=lat, lon=lon, psi=psi, v=v, x=x, y=y)
            params = VesselParams(max_yaw_rate=np.radians(max_yaw_rate_deg))
            vessels.append(Vessel(state, params))
            break  # next vessel

    return vessels


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

    print("Computed auto-bounds:")
    print(" lat:", lat_min, "→", lat_max)
    print(" lon:", lon_min, "→", lon_max)

    return lat_min, lat_max, lon_min, lon_max

from pathlib import Path
FILE_PATH = Path(__file__).resolve()
def main():
    DATA_DIR = FILE_PATH.parent.parent.parent / "data" / "GSHHS_h_L1.shp"

    # Start / goal taken from the original script
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

    # 2) Build scenario
    # Own ship at the start position, roughly heading East with modest speed.
    own_state = VesselState(
        lat=lat_start,
        lon=lon_start,
        psi=0.0,      # East
        v=5.0,        # [m/s]
    )
    own_params = VesselParams(max_yaw_rate=np.radians(20))
    own_ship = Vessel(own_state, own_params)

    # Traffic generator configured around the own ship
    traffic_gen = GenerateTraffic(
        own_ship=own_ship,
        distance=1.0,          # mean distance to targets [nm]
        max_target_speed=8.0,  # [m/s] max target speed
        epsilon=0.8,           # some variability
        env=env,
    )

    other_vessels = []
    # Encounter types: 1=overtaking, 2=overtaken, 3=head-on, 4=crossing, 5=random
    # Mix several encounters to stress-test the MPC:
    encounter_types = [3, 4, 1, 5]  # head-on, crossing, overtaking, random
    for enc_type in encounter_types:
        enc, target_state = traffic_gen.generate_enc(enc_type=enc_type, epsilon=0.5)
        if enc == -1 or target_state is None:
            # could not find a valid target within max_iter → skip
            continue
        target_params = VesselParams(max_yaw_rate=np.radians(20))
        target_ship = Vessel(target_state, target_params)
        other_vessels.append(target_ship)

    # add background traffic everywhere in region
    background_vessels = spawn_random_background_vessels(
        env=env,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        n_vessels=60,  # tweak as you like
    )
    other_vessels.extend(background_vessels)

    scenario = Scenario(own_ship=own_ship, other_vessels=other_vessels)

    # 3) Simulation config
    sim_cfg = SimConfig(dt=1.0, t_final=20000, log_interval=100)
    env.set_origin(
        scenario.own_ship.state.lat,
        scenario.own_ship.state.lon,
    )

    # 4) Plan global route (Theta* over chart)
    planner = GlobalPlanner(env, PlannerConfig(use_theta_star=True))

    start_xy = env.to_local(
        scenario.own_ship.state.lat,
        scenario.own_ship.state.lon,
    )

    goal_xy = env.to_local(lat_goal, lon_goal)
    waypoints = planner.plan(start_xy, goal_xy)

    # 5) Simulator + MPC controller
    controller = SimplifiedMPCController(
        dt=sim_cfg.dt,
        waypoints_xy=waypoints,
    )
    sim = Simulator(env, scenario, controller, sim_cfg)
    log = sim.run()

    # 6) Plot trajectories (static)
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_trajectories(env, log, ax=ax)

    # Plot waypoints along the route as markers
    wx, wy = zip(*waypoints)
    ax.scatter(wx, wy, c="orange", s=15, zorder=4, label="waypoints")
    ax.legend()
    plt.show()

    # 7) Animated playback
    x1, y1 = env.to_local(lat_min, lon_min)
    x2, y2 = env.to_local(lat_max, lon_max)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    bounds = (x_min, x_max, y_min, y_max)
    animate_trajectories(
        env,
        log,
        fps=10,
        save_path=FILE_PATH.parent / "toy_scenario.mp4",
        bounds=bounds,
    )


if __name__ == "__main__":
    main()

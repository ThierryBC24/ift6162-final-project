import argparse
from pathlib import Path
import numpy as np

from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.charts.planner import GlobalPlanner, PlannerConfig
from mpc_ship_nav.dynamics.traffic import GenerateTraffic
from mpc_ship_nav.dynamics.vessel import Vessel, VesselParams
from mpc_ship_nav.mpc.mpc_controller import SimplifiedMPCController
from mpc_ship_nav.sim.engine import SimConfig
from mpc_ship_nav.sim.visualize import animate_trajectories, plot_trajectories
import matplotlib.pyplot as plt

from scenario_utils import (
    auto_bounds,
    make_vessel_at_xy,
    run_scenario,
    spawn_random_background_vessels,
)


def main():
    parser = argparse.ArgumentParser(description="Multi-waypoint navigation scenario with traffic")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step (seconds)")
    parser.add_argument("--t-final", type=float, default=40000.0, help="Simulation duration (seconds)")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--n-background", type=int, default=300, help="Number of background vessels")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    parser.add_argument("--no-animate", action="store_true", help="Skip animation")
    args = parser.parse_args()
    
    file_path = Path(__file__).resolve()
    DATA_DIR = file_path.parent.parent.parent / "data" / "GSHHS_h_L1.shp"

    # Define the waypoints as provided
    waypoints = [
        (43.489619, 16.431957),  # Waypoint 1
        (43.08694, 16.21595),    # Waypoint 2
        (43.121905, 17.214908),   # Waypoint 3
        (43.489619, 16.431957),  # Waypoint 4
    ]

    # Auto-bounds based on the first and last waypoints
    lat_min, lat_max, lon_min, lon_max = auto_bounds(
        waypoints[0][0], waypoints[0][1], waypoints[2][0], waypoints[2][1]
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
    # Own ship at the first waypoint, heading East with modest speed.
    own_ship = make_vessel_at_xy(
        env,
        x=0.0,  # Will be set after origin is set
        y=0.0,
        psi=0.0,  # East
        v=5.0,    # [m/s]
        max_yaw_rate_deg=20.0,
    )
    # Set initial position from waypoint
    own_ship.state.lat = waypoints[0][0]
    own_ship.state.lon = waypoints[0][1]

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

    # Add background traffic everywhere in region
    background_vessels = spawn_random_background_vessels(
        env=env,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        n_vessels=args.n_background,
    )
    other_vessels.extend(background_vessels)

    # 3) Set origin and update own ship coordinates
    env.set_origin(own_ship.state.lat, own_ship.state.lon)
    # Update own ship local coordinates after origin is set
    x, y = env.to_local(own_ship.state.lat, own_ship.state.lon)
    own_ship.state.x = x
    own_ship.state.y = y

    # 4) Plan global route through multiple waypoints
    planner = GlobalPlanner(env, PlannerConfig(use_theta_star=True))

    # Use plan_multi to get the full route through multiple waypoints
    waypoints_xy = [env.to_local(lat, lon) for lat, lon in waypoints]
    route = planner.plan_multi(waypoints_xy)

    # 5) Setup controller and simulation config
    sim_cfg = SimConfig(dt=args.dt, t_final=args.t_final, log_interval=args.log_interval)
    controller = SimplifiedMPCController(
        dt=sim_cfg.dt,
        waypoints_xy=route,  # Pass the full route to the controller
    )

    # 6) Run simulation
    log, _ = run_scenario(env, own_ship, other_vessels, controller, sim_cfg)

    # 7) Plot trajectories (static)
    if not args.no_plot:
        # Calculate bounds for plotting
        x1, y1 = env.to_local(lat_min, lon_min)
        x2, y2 = env.to_local(lat_max, lon_max)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        bounds = (x_min, x_max, y_min, y_max)
        
        # Plot with custom bounds
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_trajectories(env, log, ax=ax, bounds=bounds)
        
        # Plot waypoints along the route as markers
        wx, wy = zip(*route)
        ax.scatter(wx, wy, c="orange", s=15, zorder=4, label="waypoints", marker="*")
        ax.legend()
        plt.show()

    # 8) Animated playback
    if not args.no_animate:
        x1, y1 = env.to_local(lat_min, lon_min)
        x2, y2 = env.to_local(lat_max, lon_max)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        bounds = (x_min, x_max, y_min, y_max)
        
        output_path = file_path.parent / "multipoint_scenario.gif"
        animate_trajectories(
            env,
            log,
            route,
            fps=10,
            save_path=output_path,
            bounds=bounds,
        )
        print(f"Saved animation: {output_path}")


if __name__ == "__main__":
    main()

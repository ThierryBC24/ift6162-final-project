"""
Test COLREG Rule 15: Crossing encounter with traffic on port.
Own ship is stand-on vessel, traffic should give way.
"""

import math
from pathlib import Path
from typing import List, Tuple

from scenario_utils import (
    build_open_water_env,
    check_collisions,
    create_standard_parser,
    make_vessel_at_xy,
    run_scenario,
    setup_standard_scenario,
    plot_results,
    save_animation,
)


def main():
    parser = create_standard_parser()
    args = parser.parse_args()

    env = build_open_water_env()
    route: List[Tuple[float, float]] = [(0.0, 0.0), (10_000.0, 0.0), (20_000.0, 0.0)]
    own_ship, controller, sim_cfg = setup_standard_scenario(
        env, route=route, dt=args.dt, t_final=args.t_final, log_interval=args.log_interval
    )

    traffic_max_yaw_rate = 0.0 if args.no_traffic_colreg else 20.0
    # Traffic on port side: positioned to cross at exactly 90 degrees later in the path
    # Own ship: at (0,0) heading East (0°), speed 8 m/s, moving along y=0
    # COLREG Rule 15: Traffic on port side must give way, should turn starboard (right)
    cross_x = 8000.0  # Where they will cross (later in the path)
    traffic_y = 6000.0  # Far above own ship (port side, positive y) to start outside COLREG
    traffic_heading = -math.pi / 2  # South (-90°), perpendicular to East (0°)
    
    traffic_ship = make_vessel_at_xy(
        env, x=cross_x, y=traffic_y, psi=traffic_heading, v=6.0, max_yaw_rate_deg=traffic_max_yaw_rate
    )

    log, _ = run_scenario(env, own_ship, [traffic_ship], controller, sim_cfg)

    collision_occurred, collision_time, min_dist = check_collisions(log)
    if collision_occurred:
        print(f"ERROR: Collision at t={collision_time:.1f}s, min_d={min_dist:.1f}m")
        return 1
    else:
        print(f"OK: No collision, min_d={min_dist:.1f}m")

    if not args.no_plot:
        plot_results(env, log, route)

    if args.animate:
        out_path = Path(__file__).resolve().parent / "test_colreg_crossing_port.gif"
        save_animation(env, log, route, out_path)

    return 0


if __name__ == "__main__":
    exit(main())


"""
Test COLREG Rule 13: Overtaken encounter.
Faster traffic vessel overtakes own ship.
Traffic must keep clear.
"""

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
    # Overtaken: faster traffic vessel overtakes own ship
    # Traffic behind, faster, same direction
    traffic_ship = make_vessel_at_xy(
        env, x=-6000.0, y=0.0, psi=0.0, v=12.0, max_yaw_rate_deg=traffic_max_yaw_rate
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
        out_path = Path(__file__).resolve().parent / "test_colreg_overtaken.gif"
        save_animation(env, log, route, out_path)

    return 0


if __name__ == "__main__":
    exit(main())


"""
COLREG Benchmark Scenario Suite

Runs each COLREG scenario individually and collects benchmark metrics
for comparing different controllers.

Metrics collected:
- Path deviation (lateral deviation, path length increase)
- Minimum distance to traffic
- Collision avoidance success
- Time to complete route
- Maximum lateral deviation
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.colreg import D_COLREG, D_COLLISION
from mpc_ship_nav.dynamics.traffic import Scenario
from mpc_ship_nav.dynamics.vessel import Vessel, VesselParams, VesselState
from mpc_ship_nav.mpc.mpc_controller import SimplifiedMPCController
from mpc_ship_nav.sim.engine import SimConfig, Simulator
from mpc_ship_nav.sim.visualize import plot_trajectories

from scenario_utils import (
    build_open_water_env,
    check_collisions,
    make_vessel_at_xy,
    run_scenario,
    setup_standard_scenario,
)


@dataclass
class ScenarioResult:
    """Results from a single COLREG scenario."""
    name: str
    log: Any
    route: List[Tuple[float, float]]
    collision_occurred: bool
    collision_time: float | None
    min_distance: float
    final_time: float
    path_length: float
    straight_line_distance: float
    max_lateral_deviation: float
    avg_speed: float
    success: bool


def compute_path_metrics(
    log,
    route: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """
    Compute path metrics from simulation log.
    
    Returns:
        (path_length, straight_line_distance, max_lateral_deviation, avg_speed)
    """
    own_states = log.own_states
    if len(own_states) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # Path length: sum of distances between consecutive states
    path_length = 0.0
    speeds = []
    times = log.times
    for i in range(1, len(own_states)):
        # Account for log_interval - multiply by time difference
        if i < len(times):
            dt_actual = times[i] - times[i-1] if i > 0 else 1.0
        else:
            dt_actual = 1.0
        
        dx = own_states[i].x - own_states[i-1].x
        dy = own_states[i].y - own_states[i-1].y
        segment_dist = math.hypot(dx, dy)
        # If states are logged at intervals, estimate actual distance traveled
        # Use average speed to estimate distance
        if segment_dist < 1e-6 and own_states[i].v > 0:
            # States are very close, estimate from speed
            path_length += own_states[i].v * dt_actual
        else:
            path_length += segment_dist
        speeds.append(own_states[i].v)
    
    # Straight-line distance: from start to final position (or final waypoint if reached)
    start = own_states[0]
    final_state = own_states[-1]
    # Use final waypoint if ship reached it, otherwise use final position
    final_wp = route[-1]
    dist_to_final_wp = math.hypot(final_wp[0] - final_state.x, final_wp[1] - final_state.y)
    # If close to final waypoint, use waypoint; otherwise use final position
    if dist_to_final_wp < 1000.0:  # Within 1km of final waypoint
        straight_line_distance = math.hypot(final_wp[0] - start.x, final_wp[1] - start.y)
    else:
        straight_line_distance = math.hypot(final_state.x - start.x, final_state.y - start.y)
    
    # Maximum lateral deviation: maximum distance from straight-line path
    max_lateral_deviation = 0.0
    if len(route) >= 2:
        # For simplicity, measure deviation from the main route segment
        route_start = route[0]
        route_end = route[-1]
        route_dx = route_end[0] - route_start[0]
        route_dy = route_end[1] - route_start[1]
        route_len = math.hypot(route_dx, route_dy)
        
        if route_len > 1e-6:
            for state in own_states:
                # Project point onto route line
                dx = state.x - route_start[0]
                dy = state.y - route_start[1]
                t = (dx * route_dx + dy * route_dy) / (route_len * route_len)
                t = max(0.0, min(1.0, t))  # Clamp to segment
                proj_x = route_start[0] + t * route_dx
                proj_y = route_start[1] + t * route_dy
                deviation = math.hypot(state.x - proj_x, state.y - proj_y)
                max_lateral_deviation = max(max_lateral_deviation, deviation)
    
    avg_speed = np.mean(speeds) if speeds else 0.0
    
    return path_length, straight_line_distance, max_lateral_deviation, avg_speed


def run_colreg_scenario(
    scenario_name: str,
    traffic_x: float,
    traffic_y: float,
    traffic_heading: float,
    traffic_speed: float,
    env: ChartEnvironment,
    dt: float,
    t_final: float,
    log_interval: int,
    traffic_max_yaw_rate: float,
    fix_traffic_positions: List[Tuple[float, float, float]] | None = None,
) -> ScenarioResult:
    """Run a single COLREG scenario and return results."""
    route: List[Tuple[float, float]] = [(0.0, 0.0), (10_000.0, 0.0), (20_000.0, 0.0)]
    own_ship, controller, sim_cfg = setup_standard_scenario(
        env, route=route, dt=dt, t_final=t_final, log_interval=log_interval
    )
    
    traffic_ship = make_vessel_at_xy(
        env,
        x=traffic_x,
        y=traffic_y,
        psi=traffic_heading,
        v=traffic_speed,
        max_yaw_rate_deg=traffic_max_yaw_rate,
    )
    
    log, _ = run_scenario(
        env, own_ship, [traffic_ship], controller, sim_cfg,
        fix_traffic_positions=fix_traffic_positions
    )
    
    collision_occurred, collision_time, min_dist = check_collisions(log)
    final_time = log.times[-1] if log.times else 0.0
    
    path_length, straight_line_dist, max_lateral_dev, avg_speed = compute_path_metrics(log, route)
    
    success = not collision_occurred and min_dist > D_COLLISION
    
    return ScenarioResult(
        name=scenario_name,
        log=log,
        route=route,
        collision_occurred=collision_occurred,
        collision_time=collision_time,
        min_distance=min_dist,
        final_time=final_time,
        path_length=path_length,
        straight_line_distance=straight_line_dist,
        max_lateral_deviation=max_lateral_dev,
        avg_speed=avg_speed,
        success=success,
    )


def print_benchmark_report(results: List[ScenarioResult]) -> None:
    """Print a formatted benchmark report."""
    print("\n" + "="*80)
    print("COLREG BENCHMARK REPORT")
    print("="*80)
    
    print(f"\n{'Scenario':<30} {'Status':<10} {'Min Dist (m)':<15} {'Path Eff':<12} {'Max Dev (m)':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in results:
        status = "PASS" if result.success else "FAIL"
        path_efficiency = (result.straight_line_distance / result.path_length * 100) if result.path_length > 0 else 0.0
        print(
            f"{result.name:<30} {status:<10} {result.min_distance:>12.1f}     "
            f"{path_efficiency:>6.1f}%      {result.max_lateral_deviation:>12.1f}     "
            f"{result.final_time:>8.1f}"
        )
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    successful = [r for r in results if r.success]
    print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    if successful:
        avg_min_dist = np.mean([r.min_distance for r in successful])
        avg_path_eff = np.mean([r.straight_line_distance / r.path_length * 100 for r in successful if r.path_length > 0])
        avg_max_dev = np.mean([r.max_lateral_deviation for r in successful])
        avg_time = np.mean([r.final_time for r in successful])
        
        print(f"Average minimum distance: {avg_min_dist:.1f} m")
        print(f"Average path efficiency: {avg_path_eff:.1f}%")
        print(f"Average max lateral deviation: {avg_max_dev:.1f} m")
        print(f"Average completion time: {avg_time:.1f} s")
    
    print("="*80 + "\n")


def plot_all_scenarios(
    env: ChartEnvironment,
    results: List[ScenarioResult],
    output_path: Path | None = None,
) -> None:
    """Plot all scenarios in a combined figure."""
    n_scenarios = len(results)
    cols = 3
    rows = (n_scenarios + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_scenarios == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Plot trajectories
        plot_trajectories(env, result.log, ax=ax)
        
        # Plot route waypoints
        wx, wy = zip(*result.route)
        ax.scatter(wx, wy, c="orange", s=30, zorder=4, label="waypoints", marker="*")
        
        # Add title with key metrics
        status = "[PASS]" if result.success else "[FAIL]"
        title = f"{status} {result.name}\n"
        title += f"Min dist: {result.min_distance:.0f}m | "
        title += f"Path eff: {result.straight_line_distance/result.path_length*100:.1f}% | "
        title += f"Time: {result.final_time:.0f}s"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_scenarios, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined plot: {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COLREG Benchmark Suite - Runs individual COLREG scenarios and collects metrics"
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Time step (seconds)")
    parser.add_argument("--t-final", type=float, default=10000.0, help="Max simulation duration (seconds)")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval (steps)")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    parser.add_argument("--no-traffic-colreg", action="store_true", help="Disable COLREG for traffic ships")
    parser.add_argument("--save-plot", type=str, help="Save combined plot to file (e.g., 'benchmark.png')")
    args = parser.parse_args()
    
    # Setup environment
    env = build_open_water_env()
    traffic_max_yaw_rate = 0.0 if args.no_traffic_colreg else 20.0
    
    # Define all COLREG scenarios
    scenarios = [
        {
            "name": "Head-on (Rule 14)",
            "traffic_x": 6000.0,
            "traffic_y": 0.0,
            "traffic_heading": math.pi,  # West
            "traffic_speed": 8.0,
            "fix_positions": [(6000.0, 0.0, math.pi)],
        },
        {
            "name": "Crossing Starboard (Rule 15)",
            "traffic_x": 8000.0,
            "traffic_y": -6000.0,
            "traffic_heading": math.pi / 2,  # North
            "traffic_speed": 6.0,
            "fix_positions": None,
        },
        {
            "name": "Crossing Port (Rule 15)",
            "traffic_x": 8000.0,
            "traffic_y": 6000.0,
            "traffic_heading": -math.pi / 2,  # South
            "traffic_speed": 6.0,
            "fix_positions": None,
        },
        {
            "name": "Overtaking (Rule 13)",
            "traffic_x": 2000.0,
            "traffic_y": 0.0,
            "traffic_heading": 0.0,  # East
            "traffic_speed": 2.0,
            "fix_positions": None,
        },
        {
            "name": "Overtaken (Rule 13)",
            "traffic_x": -6000.0,
            "traffic_y": 0.0,
            "traffic_heading": 0.0,  # East
            "traffic_speed": 12.0,
            "fix_positions": None,
        },
    ]
    
    # Run all scenarios
    print("Running COLREG Benchmark Suite...")
    print(f"Total scenarios: {len(scenarios)}\n")
    
    results: List[ScenarioResult] = []
    
    for i, scenario_config in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] Running: {scenario_config['name']}...")
        
        result = run_colreg_scenario(
            scenario_name=scenario_config["name"],
            traffic_x=scenario_config["traffic_x"],
            traffic_y=scenario_config["traffic_y"],
            traffic_heading=scenario_config["traffic_heading"],
            traffic_speed=scenario_config["traffic_speed"],
            env=env,
            dt=args.dt,
            t_final=args.t_final,
            log_interval=args.log_interval,
            traffic_max_yaw_rate=traffic_max_yaw_rate,
            fix_traffic_positions=scenario_config.get("fix_positions"),
        )
        
        results.append(result)
        
        status = "PASS" if result.success else "FAIL"
        print(f"  [{status}] - Min distance: {result.min_distance:.1f}m, Time: {result.final_time:.1f}s")
    
    # Print benchmark report
    print_benchmark_report(results)
    
    # Plot all scenarios
    if not args.no_plot:
        output_path = Path(args.save_plot) if args.save_plot else None
        plot_all_scenarios(env, results, output_path)
    
    # Exit code based on success
    all_successful = all(r.success for r in results)
    exit(0 if all_successful else 1)


if __name__ == "__main__":
    main()

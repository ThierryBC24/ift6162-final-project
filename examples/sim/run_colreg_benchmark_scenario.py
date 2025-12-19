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
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.colreg import D_COLLISION
from mpc_ship_nav.dynamics.traffic import Scenario
from mpc_ship_nav.dynamics.vessel import Vessel
from mpc_ship_nav.sim.engine import SimConfig, Simulator
from mpc_ship_nav.sim.visualize import plot_trajectories

from scenario_utils import (
    build_open_water_env,
    check_collisions,
    make_vessel_at_xy,
    setup_standard_scenario,
)


class BaselineWaypointController:
    """
    Simple baseline controller: follows waypoints using proportional heading control.
    No collision avoidance - just basic waypoint following.
    This serves as a baseline to compare against MPC.
    """
    def __init__(self, waypoints_xy: np.ndarray, dt: float, gain: float = 0.8):
        self.waypoints_xy = np.asarray(waypoints_xy, dtype=float)
        self.dt = dt
        self.gain = gain  # Proportional gain for heading control
        self.idx = 0
        self.transition_radius = 926.0  # Same as collision radius
        self._finished = False  # Flag to track if we've reached the final waypoint
        
        # Create a route-like object for early termination detection
        # This mimics the WaypointRoute interface used by the simulator
        class SimpleRoute:
            def __init__(self, controller):
                self.controller = controller
            
            def is_finished(self) -> bool:
                """Check if we've reached the final waypoint."""
                # The controller sets _finished=True when within transition_radius of final waypoint
                return self.controller._finished
        
        self.route = SimpleRoute(self)
        
    def _wrap_angle(self, a: float) -> float:
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi
    
    def compute_control(
        self,
        t: float,
        own_ship: Vessel,
        other_vessels: List[Vessel],
        env: ChartEnvironment,
    ) -> Tuple[Tuple[float, int], Tuple[np.ndarray, np.ndarray]]:
        """Compute control: simple waypoint following without collision avoidance."""
        own = own_ship.state
        
        # Ensure local coordinates
        if own.x is None or own.y is None:
            own.x, own.y = env.to_local(own.lat, own.lon)
        
        # Get current waypoint
        n = len(self.waypoints_xy)
        if n == 0:
            return ((0.0, 0), (np.array([]), np.array([])))
        
        if self.idx >= n:
            self.idx = n - 1
        
        wp_x, wp_y = self.waypoints_xy[self.idx]
        dx = own.x - wp_x
        dy = own.y - wp_y
        dist = math.hypot(dx, dy)
        
        # Advance to next waypoint if close enough
        if dist <= self.transition_radius and self.idx < n - 1:
            self.idx += 1
            wp_x, wp_y = self.waypoints_xy[self.idx]
            dx = own.x - wp_x
            dy = own.y - wp_y
        
        # Check if we've reached the final waypoint
        if self.idx >= n - 1:
            final_wp = self.waypoints_xy[-1]
            final_dist = math.hypot(own.x - final_wp[0], own.y - final_wp[1])
            if final_dist <= self.transition_radius:
                self._finished = True
        
        # Calculate target heading
        theta_target = math.atan2(wp_y - own.y, wp_x - own.x)
        
        # Proportional heading control
        heading_error = self._wrap_angle(theta_target - own.psi)
        u = self.gain * heading_error
        
        # Clip to reasonable yaw rate (same as MPC max)
        max_yaw_rate = math.radians(20.0)
        u = max(-max_yaw_rate, min(max_yaw_rate, u))
        
        # Return in same format as MPC controller
        return ((float(u), 0), (np.array([]), np.array([])))


@dataclass
class ScenarioResult:
    """Results from a single COLREG scenario."""
    name: str
    controller_type: str  # "MPC" or "Baseline"
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
    
    Path efficiency = (straight_line_distance / path_length) * 100%
    - 100% = perfectly straight path (no detours)
    - < 100% = path with detours/turns (lower = more deviation from straight line)
    
    Returns:
        (path_length, straight_line_distance, max_lateral_deviation, avg_speed)
    """
    own_states = log.own_states
    if len(own_states) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # Path length: sum of distances traveled between consecutive logged states
    path_length = 0.0
    speeds = []
    times = log.times
    
    for i in range(1, len(own_states)):
        if i < len(times):
            dt_actual = times[i] - times[i-1]
        else:
            dt_actual = 5.0
        
        # Calculate distance using average speed over the interval
        # This accounts for the actual path traveled, including any oscillations or turns
        avg_speed = (own_states[i-1].v + own_states[i].v) / 2.0
        distance_from_speed = avg_speed * dt_actual
        
        # Also calculate direct distance between logged points
        dx = own_states[i].x - own_states[i-1].x
        dy = own_states[i].y - own_states[i-1].y
        direct_distance = math.hypot(dx, dy)
        
        # Use the larger of the two to avoid underestimating
        # If the ship oscillated or turned between logged points, distance_from_speed will be larger
        # If the ship went straight, direct_distance will be accurate
        path_length += max(distance_from_speed, direct_distance)
        
        speeds.append(own_states[i].v)
    
    start = own_states[0]
    final_state = own_states[-1]
    straight_line_distance = math.hypot(final_state.x - start.x, final_state.y - start.y)
    
    if straight_line_distance > path_length + 1e-3:
        print(f"Warning: straight_line_distance ({straight_line_distance:.2f}) > path_length ({path_length:.2f}), capping to path_length")
        straight_line_distance = path_length
    
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
    controller_type: str = "MPC",
    fix_traffic_positions: List[Tuple[float, float, float]] | None = None,
) -> ScenarioResult:
    """Run a single COLREG scenario and return results."""
    route: List[Tuple[float, float]] = [(0.0, 0.0), (10_000.0, 0.0), (10_000.0, 0.0)]
    
    own_ship = make_vessel_at_xy(
        env, x=0.0, y=0.0, psi=0.0, v=8.0, max_yaw_rate_deg=20.0
    )
    
    if controller_type == "Baseline":
        controller = BaselineWaypointController(
            waypoints_xy=np.array(route),
            dt=dt,
            gain=0.8
        )
        sim_cfg = SimConfig(dt=dt, t_final=t_final, log_interval=log_interval)
    else:
        _, controller, sim_cfg = setup_standard_scenario(
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
    
    scenario = Scenario(own_ship=own_ship, other_vessels=[traffic_ship])
    sim = Simulator(env, scenario, controller, sim_cfg)
    
    if fix_traffic_positions is not None:
        for i, (x, y, psi) in enumerate(fix_traffic_positions):
            if i < len(sim.other_vessels):
                v = sim.other_vessels[i]
                v.state.x = x
                v.state.y = y
                v.state.psi = psi
                lat, lon = env.to_geo(x, y)
                v.state.lat = lat
                v.state.lon = lon
    
    log = sim.run()
    
    collision_occurred, collision_time, min_dist = check_collisions(log)
    final_time = log.times[-1] if log.times else 0.0
    path_length, straight_line_dist, max_lateral_dev, avg_speed = compute_path_metrics(log, route)
    success = not collision_occurred and min_dist > D_COLLISION
    
    return ScenarioResult(
        name=scenario_name,
        controller_type=controller_type,
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
    """Print a formatted benchmark report with MPC vs Baseline comparison."""
    print("\n" + "="*100)
    print("COLREG BENCHMARK REPORT - MPC vs Baseline Comparison")
    print("="*100)
    
    # Separate results by controller type
    mpc_results = {r.name: r for r in results if r.controller_type == "MPC"}
    baseline_results = {r.name: r for r in results if r.controller_type == "Baseline"}
    
    # Print comparison table
    print(f"\n{'Scenario':<30} {'Controller':<12} {'Status':<10} {'Min Dist (m)':<15} {'Path Eff':<12} {'Max Dev (m)':<15} {'Time (s)':<10}")
    print("-" * 100)
    
    # Get all unique scenario names
    all_scenarios = set(mpc_results.keys()) | set(baseline_results.keys())
    
    for scenario_name in sorted(all_scenarios):
        # Print MPC result
        if scenario_name in mpc_results:
            r = mpc_results[scenario_name]
            status = "PASS" if r.success else "FAIL"
            path_efficiency = (r.straight_line_distance / r.path_length * 100) if r.path_length > 0 else 0.0
            print(
                f"{r.name:<30} {'MPC':<12} {status:<10} {r.min_distance:>12.1f}     "
                f"{path_efficiency:>6.1f}%      {r.max_lateral_deviation:>12.1f}     "
                f"{r.final_time:>8.1f}"
            )
        
        # Print Baseline result
        if scenario_name in baseline_results:
            r = baseline_results[scenario_name]
            status = "PASS" if r.success else "FAIL"
            path_efficiency = (r.straight_line_distance / r.path_length * 100) if r.path_length > 0 else 0.0
            print(
                f"{r.name:<30} {'Baseline':<12} {status:<10} {r.min_distance:>12.1f}     "
                f"{path_efficiency:>6.1f}%      {r.max_lateral_deviation:>12.1f}     "
                f"{r.final_time:>8.1f}"
            )
    
    # Summary statistics by controller type
    print("\n" + "-" * 100)
    print("SUMMARY STATISTICS BY CONTROLLER")
    print("-" * 100)
    
    for ctrl_type in ["MPC", "Baseline"]:
        ctrl_results = [r for r in results if r.controller_type == ctrl_type]
        if not ctrl_results:
            continue
            
        successful = [r for r in ctrl_results if r.success]
        print(f"\n{ctrl_type}:")
        print(f"  Success rate: {len(successful)}/{len(ctrl_results)} ({len(successful)/len(ctrl_results)*100:.1f}%)")
        
        if successful:
            avg_min_dist = np.mean([r.min_distance for r in successful])
            avg_path_eff = np.mean([r.straight_line_distance / r.path_length * 100 for r in successful if r.path_length > 0])
            avg_max_dev = np.mean([r.max_lateral_deviation for r in successful])
            avg_time = np.mean([r.final_time for r in successful])
            
            print(f"  Average minimum distance: {avg_min_dist:.1f} m")
            print(f"  Average path efficiency: {avg_path_eff:.1f}%")
            print(f"  Average max lateral deviation: {avg_max_dev:.1f} m")
            print(f"  Average completion time: {avg_time:.1f} s")
    
    # Comparison summary
    if mpc_results and baseline_results:
        print("\n" + "-" * 100)
        print("MPC vs BASELINE COMPARISON")
        print("-" * 100)
        
        mpc_successful = [r for r in mpc_results.values() if r.success]
        baseline_successful = [r for r in baseline_results.values() if r.success]
        
        print(f"Success rate (no collision): MPC {len(mpc_successful)}/{len(mpc_results)} vs Baseline {len(baseline_successful)}/{len(baseline_results)}")
        
        if mpc_successful and baseline_successful:
            mpc_avg_min_dist = np.mean([r.min_distance for r in mpc_successful])
            baseline_avg_min_dist = np.mean([r.min_distance for r in baseline_successful])
            improvement = ((mpc_avg_min_dist - baseline_avg_min_dist) / baseline_avg_min_dist * 100) if baseline_avg_min_dist > 0 else 0.0
            print(f"Min distance: MPC {mpc_avg_min_dist:.1f}m vs Baseline {baseline_avg_min_dist:.1f}m ({improvement:+.1f}%)")
            
            mpc_avg_path_eff = np.mean([r.straight_line_distance / r.path_length * 100 for r in mpc_successful if r.path_length > 0])
            baseline_avg_path_eff = np.mean([r.straight_line_distance / r.path_length * 100 for r in baseline_successful if r.path_length > 0])
            improvement = ((mpc_avg_path_eff - baseline_avg_path_eff) / baseline_avg_path_eff * 100) if baseline_avg_path_eff > 0 else 0.0
            print(f"Path efficiency: MPC {mpc_avg_path_eff:.1f}% vs Baseline {baseline_avg_path_eff:.1f}% ({improvement:+.1f}%)")
    
    print("="*100 + "\n")


def plot_all_scenarios(
    env: ChartEnvironment,
    results: List[ScenarioResult],
    output_path: Path | None = None,
) -> None:
    """Plot all scenarios in two separate figures: one for MPC, one for Baseline."""
    # Separate results by controller type
    mpc_results = [r for r in results if r.controller_type == "MPC"]
    baseline_results = [r for r in results if r.controller_type == "Baseline"]
    
    # Sort by scenario name for consistent ordering
    mpc_results.sort(key=lambda x: x.name)
    baseline_results.sort(key=lambda x: x.name)
    
    # Create two separate figures
    figures = []
    
    # Plot MPC scenarios
    if mpc_results:
        n_mpc = len(mpc_results)
        cols = 3
        rows = (n_mpc + cols - 1) // cols
        
        fig_mpc, axes_mpc = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_mpc == 1:
            axes_mpc = [axes_mpc]
        else:
            axes_mpc = axes_mpc.flatten()
        
        for idx, result in enumerate(mpc_results):
            ax = axes_mpc[idx]
            plot_trajectories(env, result.log, ax=ax)
            wx, wy = zip(*result.route)
            ax.scatter(wx, wy, c="orange", s=30, zorder=4, label="waypoints", marker="*")
            status = "[PASS]" if result.success else "[FAIL]"
            path_eff = (result.straight_line_distance / result.path_length * 100) if result.path_length > 0 else 0.0
            title = f"{status} {result.name}\n"
            title += f"Min dist: {result.min_distance:.0f}m | "
            title += f"Path eff: {path_eff:.1f}% | "
            title += f"Time: {result.final_time:.0f}s"
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_mpc, len(axes_mpc)):
            axes_mpc[idx].axis('off')
        
        fig_mpc.suptitle("MPC Controller - All Scenarios", fontsize=14, fontweight='bold')
        fig_mpc.tight_layout()
        figures.append(("MPC", fig_mpc))
    
    # Plot Baseline scenarios
    if baseline_results:
        n_baseline = len(baseline_results)
        cols = 3
        rows = (n_baseline + cols - 1) // cols
        
        fig_baseline, axes_baseline = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_baseline == 1:
            axes_baseline = [axes_baseline]
        else:
            axes_baseline = axes_baseline.flatten()
        
        for idx, result in enumerate(baseline_results):
            ax = axes_baseline[idx]
            plot_trajectories(env, result.log, ax=ax)
            wx, wy = zip(*result.route)
            ax.scatter(wx, wy, c="orange", s=30, zorder=4, label="waypoints", marker="*")
            status = "[PASS]" if result.success else "[FAIL]"
            path_eff = (result.straight_line_distance / result.path_length * 100) if result.path_length > 0 else 0.0
            title = f"{status} {result.name}\n"
            title += f"Min dist: {result.min_distance:.0f}m | "
            title += f"Path eff: {path_eff:.1f}% | "
            title += f"Time: {result.final_time:.0f}s"
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_baseline, len(axes_baseline)):
            axes_baseline[idx].axis('off')
        
        fig_baseline.suptitle("Baseline Controller - All Scenarios", fontsize=14, fontweight='bold')
        fig_baseline.tight_layout()
        figures.append(("Baseline", fig_baseline))
    
    # Save or show figures
    if output_path:
        # Save each figure separately
        base_path = Path(output_path)
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        for ctrl_type, fig in figures:
            save_path = parent / f"{stem}_{ctrl_type.lower()}{suffix}"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {ctrl_type} plot: {save_path}")
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
    
    # Run all scenarios with both MPC and Baseline
    print("Running COLREG Benchmark Suite...")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Controllers: MPC + Baseline\n")
    
    results: List[ScenarioResult] = []
    
    for i, scenario_config in enumerate(scenarios, 1):
        scenario_name = scenario_config["name"]
        print(f"[{i}/{len(scenarios)}] Running: {scenario_name}...")
        
        # Run with MPC
        print(f"  Running with MPC...")
        mpc_result = run_colreg_scenario(
            scenario_name=scenario_name,
            traffic_x=scenario_config["traffic_x"],
            traffic_y=scenario_config["traffic_y"],
            traffic_heading=scenario_config["traffic_heading"],
            traffic_speed=scenario_config["traffic_speed"],
            env=env,
            dt=args.dt,
            t_final=args.t_final,
            log_interval=args.log_interval,
            traffic_max_yaw_rate=traffic_max_yaw_rate,
            controller_type="MPC",
            fix_traffic_positions=scenario_config.get("fix_positions"),
        )
        results.append(mpc_result)
        status = "PASS" if mpc_result.success else "FAIL"
        print(f"    [{status}] - Min distance: {mpc_result.min_distance:.1f}m, Time: {mpc_result.final_time:.1f}s")
        
        # Run with Baseline
        print(f"  Running with Baseline...")
        baseline_result = run_colreg_scenario(
            scenario_name=scenario_name,
            traffic_x=scenario_config["traffic_x"],
            traffic_y=scenario_config["traffic_y"],
            traffic_heading=scenario_config["traffic_heading"],
            traffic_speed=scenario_config["traffic_speed"],
            env=env,
            dt=args.dt,
            t_final=args.t_final,
            log_interval=args.log_interval,
            traffic_max_yaw_rate=traffic_max_yaw_rate,
            controller_type="Baseline",
            fix_traffic_positions=scenario_config.get("fix_positions"),
        )
        results.append(baseline_result)
        status = "PASS" if baseline_result.success else "FAIL"
        print(f"    [{status}] - Min distance: {baseline_result.min_distance:.1f}m, Time: {baseline_result.final_time:.1f}s")
        print()
    
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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState
from mpc_ship_nav.sim.engine import Controller


@dataclass
class MPCConfig:
    """Configuration for the simplified MPC controller."""
    dt: float
    horizon: int = 16                  # H (paper: Section 2.6.3)
    n_candidates: int = 45             # M (paper: Section 2.6.3)
    max_delta_heading_deg: float = 20  # Δθ_max per step (paper: ±20°)
    collision_radius_nm: float = 0.5   # d_collision
    colreg_radius_nm: float = 3.0      # d_COLREG

    @property
    def max_delta_heading(self) -> float:
        """Maximum heading increment per step [rad]."""
        return math.radians(self.max_delta_heading_deg)

    @property
    def max_yaw_rate(self) -> float:
        """Corresponding yaw-rate limit [rad/s]. u * dt = Δθ."""
        return self.max_delta_heading / self.dt

    @property
    def collision_radius(self) -> float:
        """Collision radius [m]."""
        return self.collision_radius_nm * 1852.0

    @property
    def colreg_radius(self) -> float:
        """COLREG zone radius [m]."""
        return self.colreg_radius_nm * 1852.0


class WaypointRoute:
    """Waypoint manager in LOCAL (x,y) coordinates."""
    def __init__(self, waypoints_xy: np.ndarray, transition_radius: float):
        self.waypoints_xy = waypoints_xy  # List of waypoints in (x, y)
        self.transition_radius = transition_radius  # When within this radius, move to next waypoint
        self.idx = 0  # Index of the current waypoint

    def current_waypoint(self, own: VesselState) -> Tuple[float, float]:
        """Return the current waypoint and advance when inside transition radius."""
        n = len(self.waypoints_xy)
        if n == 0:
            # If no waypoints, return current position
            return own.x, own.y

        # Clamp index to ensure it doesn't go out of bounds
        if self.idx >= n:
            self.idx = n - 1

        wp_x, wp_y = self.waypoints_xy[self.idx]
        dx = own.x - wp_x
        dy = own.y - wp_y
        dist = math.hypot(dx, dy)

        if dist <= self.transition_radius and self.idx < n - 1:
            # If close enough, advance to next waypoint
            self.idx += 1
            wp_x, wp_y = self.waypoints_xy[self.idx]

        return float(wp_x), float(wp_y)

    def is_finished(self) -> bool:
        return self.idx >= len(self.waypoints_xy) - 1

class StaticControlSeqGenrator:
    """Generates a set of static trajectories based max yaw rate, horizon and number of trajectories."""
    
    def __init__(self, max_yaw_rate: float=np.radians(20), horizon: int=20, num_trajectories: int=45, decay_factor: float=0.95):
        """
        Args:
            max_yaw_rate (float, optional): the maximum yaw rate in radians per second. Defaults to np.radians(20).
            horizon (int, optional): the number of time steps in the trajectory horizon. Defaults to 20.
            num_trajectories (int, optional): the number of trajectories to generate. Defaults to 45.
            decay_factor (float, optional): the factor by which the yaw rate decays at each time step. Defaults to 0.95.
        """
        self.max_yaw_rate = max_yaw_rate
        self.horizon = horizon
        self.num_trajectories = num_trajectories
        self.decay_factor = decay_factor
        self.control_sequences = self.generate_controls()
        
        
    def generate_controls(self) -> np.ndarray[np.ndarray[np.float64]]:
        initial_angles = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, self.num_trajectories)
        # Initialize the control matrix
        controls = np.zeros((self.num_trajectories, self.horizon), dtype=np.float64)
        print("Generating trajectories with control shape:")
        print(controls.shape)

        # 2. Compute the sequence for each trajectory
        for i, start_angle in enumerate(initial_angles):
            dtheta = start_angle

            for h in range(self.horizon):
                # Store current control
                controls[i, h] = dtheta

                # Apply decay for the next step (smoothing)
                # This corresponds to: dtheta = dtheta * .95
                dtheta *= self.decay_factor
        return controls 


class SimplifiedMPCController(Controller):
    """
    Simplified fan-based MPC controller as in Sec. 2.6.3:
    - Generate M constant-turn trajectories over H steps.
    - Discard those that violate d_collision to static/dynamic obstacles.
    - Select trajectory based on heading/endpoint relative to next waypoint.
    - Apply only first control input (receding horizon).
    """

    def __init__(self, dt: float, waypoints_xy: np.ndarray, vis_scale: int = 100) -> None:
        self.cfg = MPCConfig(dt=dt)
        self.route = WaypointRoute(
            waypoints_xy=np.asarray(waypoints_xy, dtype=float),
            transition_radius=self.cfg.collision_radius,
        )
        self.control_sequences = StaticControlSeqGenrator(self.cfg.max_yaw_rate, self.cfg.horizon, self.cfg.n_candidates).control_sequences
        self.u_candidates = self.control_sequences[:, 0]
        self.vis_scale = vis_scale  # for trajectory visualization
    def compute_control(
        self,
        t: float,
        own_ship: Vessel,
        other_vessels: List[Vessel],
        env: ChartEnvironment,
    ) -> Tuple[Tuple[float, int], Tuple[np.ndarray, np.ndarray]]:
        own = own_ship.state  # VesselState

        # Ensure we have local coordinates
        if own.x is None or own.y is None:
            own.x, own.y = env.to_local(own.lat, own.lon)

        # 1) Get the current waypoint in local coords, possibly advance
        wp_x, wp_y = self.route.current_waypoint(own)

        # If we are at the last waypoint, just go straight
        if self.route.is_finished():
            # Return 0.0 yaw rate, index 0, and empty debug info to satisfy unpacking in engine.py
            return ((0.0, 0), (np.array([]), np.array([])))

        # 2) Calculate bearing to waypoint (target heading)
        dx = wp_x - own.x
        dy = wp_y - own.y
        theta_target = math.atan2(dy, dx)

        # 3) Collect dynamic obstacles within COLREG zone
        dyn_states: List[VesselState] = []
        for v in other_vessels:
            s = v.state
            if s.x is None or s.y is None:
                s.x, s.y = env.to_local(s.lat, s.lon)

            ddx = s.x - own.x
            ddy = s.y - own.y
            if ddx * ddx + ddy * ddy <= self.cfg.colreg_radius ** 2:
                dyn_states.append(s)

        # 4) Precompute predicted dynamic trajectories (constant velocity)
        dyn_trajs = self._predict_dynamic(dyn_states)

        # 5) Generate candidate yaw-rates and simulate own trajectories
        # Paper: M candidate trajectories with yaw rates in [-max_yaw_rate, max_yaw_rate]
        u_candidates = np.linspace(
            -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate, self.cfg.n_candidates
        )
        feasible_mask, own_trajs, own_trajs_vis = self._simulate_and_filter(
            own, dyn_trajs
        )

        if not np.any(feasible_mask):
            # fallback: choose u closest to 0 (maintain current heading as safest option)
            # This matches the paper's approach when no feasible trajectory exists
            idx = np.argmin(np.abs(u_candidates))
            return ((float(u_candidates[idx]), idx), (feasible_mask, own_trajs_vis))

        # 6) Select the best feasible candidate (with COLREG awareness)
        idx = self._select_best(
            own,
            (wp_x, wp_y),
            theta_target,
            u_candidates,
            own_trajs,
            feasible_mask,
            dyn_states=dyn_states,  # Pass dynamic states for COLREG detection
        )

        return (float(u_candidates[idx]), idx), (feasible_mask, own_trajs_vis)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


    # ------------------------------------------------------------------
    # MPC internals
    # ------------------------------------------------------------------

    def _simulate_and_filter(
        self,
        own: VesselState,
        dyn_trajs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate own-ship trajectories for each yaw-rate candidate and
        return:
            feasible_mask: shape (M,), bool
            own_trajs:     shape (M, H, 2) (points over horizon)
        """
        H = self.cfg.horizon
        dt = self.cfg.dt
        v = own.v
        M = self.cfg.n_candidates
        
        sequence_by_traj = self.control_sequences 
        own_trajs_vis = np.zeros((M, H, 2), dtype=float)
        own_trajs = np.zeros((M, H, 2), dtype=float)
        feasible = np.ones(M, dtype=bool)

        for m in range(M):
            px = own.x
            py = own.y
            psi = own.psi
            px_vis = own.x
            py_vis = own.y
            psi_vis = own.psi
            sequence = sequence_by_traj[m]
            for h in range(H):
                u = sequence[h]
                psi_vis = self._wrap_angle(psi_vis + u * dt)
                px_vis += v * math.cos(psi_vis) * dt * self.vis_scale
                py_vis += v * math.sin(psi_vis) * dt * self.vis_scale
                own_trajs_vis[m, h, 0] = px_vis
                own_trajs_vis[m, h, 1] = py_vis
                
            for h in range(H):
                u = sequence[h]
                psi = self._wrap_angle(psi + u * dt)
                px += v * math.cos(psi) * dt
                py += v * math.sin(psi) * dt
                own_trajs[m, h, 0] = px
                own_trajs[m, h, 1] = py

                # --- dynamic collision check ---
                for traj_other in dyn_trajs:
                    if h >= traj_other.shape[0]:
                        continue
                    ox, oy = traj_other[h]
                    dist = math.hypot(px - ox, py - oy)
                    if dist < self.cfg.collision_radius:
                        feasible[m] = False
                        break

                if not feasible[m]:
                    break

                # --- static collision check (if enabled) ---
                # if static_xy.size > 0:
                #     dx = static_xy[:, 0] - px
                #     dy = static_xy[:, 1] - py
                #     if np.any(dx * dx + dy * dy < self.cfg.collision_radius ** 2):
                #         feasible[m] = False
                #         break

        return feasible, own_trajs, own_trajs_vis

    def _predict_dynamic(self, dyn_states: List[VesselState]) -> List[np.ndarray]:
        """Predict constant-velocity paths for dynamic obstacles over the horizon."""
        H = self.cfg.horizon
        dt = self.cfg.dt

        dyn_trajs: List[np.ndarray] = []

        for s in dyn_states:
            px = s.x
            py = s.y
            psi = s.psi
            v = s.v

            traj = np.zeros((H, 2), dtype=float)
            for h in range(H):
                px += v * math.cos(psi) * dt
                py += v * math.sin(psi) * dt
                traj[h, 0] = px
                traj[h, 1] = py

            dyn_trajs.append(traj)

        return dyn_trajs

    def _select_best(
        self,
        own: VesselState,
        waypoint_xy: Tuple[float, float],
        theta_target: float,
        u_candidates: np.ndarray,
        own_trajs: np.ndarray,        # shape (M, H, 2)
        feasible_mask: np.ndarray,    # shape (M,)
        dyn_states: List[VesselState] = None,
    ) -> int:
        """
        COLREG-aware selection.
        Base objective (as in your description):
        - if aligned: minimize heading error after 1 step
        - else: minimize distance from endpoint to waypoint

        COLREG overlay:
        - if a plausible head-on or starboard-crossing give-way situation exists,
            bias toward starboard (right) turns.
        """
        H = self.cfg.horizon
        dt = self.cfg.dt

        idxs = np.where(feasible_mask)[0]
        if idxs.size == 0:
            # Shouldn't happen if feasible_mask ensured, but be safe.
            return int(np.argmin(np.isfinite(feasible_mask)))


        # -----------------------------
        # Helpers
        # -----------------------------
        def ang_diff(a, b):
            return abs(self._wrap_angle(a - b))

        def is_starboard_turn(u: float) -> bool:
            # Your comment implies: u < 0 => starboard turn
            return u < 0.0

        def is_port_turn(u: float) -> bool:
            return u > 0.0

        # -----------------------------
        # Determine COLREG preference
        # -----------------------------
        prefer_starboard = False

        # COLREG detection parameters
        d_alert = self.cfg.colreg_radius
        fwd_sector = math.radians(20.0)
        starboard_sector = math.radians(112.5)
        same_course_tolerance = math.radians(10.0)  # For overtaking detection

        # Track closest COLREG threat for distance-based bias scaling
        closest_threat_dist = float('inf')
        traffic_passing_factor = 1.0  # Factor to reduce bias when traffic is passing/moving away
        
        if dyn_states:
            for other in dyn_states:
                dx = other.x - own.x
                dy = other.y - own.y
                dist = math.hypot(dx, dy)

                # Too far -> ignore (prevents always-on starboard bias)
                if dist > d_alert:
                    continue

                bearing_math = math.atan2(dy, dx)
                rel_bearing = self._wrap_angle(bearing_math - own.psi)
                heading_diff = ang_diff(own.psi, other.psi)
                
                # Only apply COLREG bias if traffic is ahead or on the side (not astern)
                # If traffic is astern (behind), it has passed and we don't need COLREG bias
                # Astern is typically defined as rel_bearing > 112.5° or < -112.5°
                astern_sector = math.radians(112.5)
                if abs(rel_bearing) > astern_sector:
                    # Traffic is astern (behind), has passed - no COLREG bias needed
                    continue

                # Check if traffic is moving away (passing) to reduce bias
                # Relative velocity vector
                vx_own = own.v * math.cos(own.psi)
                vy_own = own.v * math.sin(own.psi)
                vx_other = other.v * math.cos(other.psi)
                vy_other = other.v * math.sin(other.psi)
                
                # Position vector from own to other (for closing check)
                # If relative velocity is moving away from other, reduce bias
                dvx = vx_own - vx_other
                dvy = vy_own - vy_other
                
                # Normalize position vector
                if dist > 1e-6:
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    # Project relative velocity onto position vector
                    # If positive, we're moving away from traffic
                    closing_rate = -(dvx * dx_norm + dvy * dy_norm)
                    
                    # If closing rate is negative (moving away), reduce bias
                    if closing_rate < 0:
                        # Traffic is moving away, reduce bias significantly
                        traffic_passing_factor = min(traffic_passing_factor, 0.2)
                    elif closing_rate < 0.5:  # Slow closing or moving away
                        traffic_passing_factor = min(traffic_passing_factor, 0.5)

                # --- Overtaking (Rule 13): own ship is faster and other is ahead
                # Own ship must keep clear, should pass on starboard (right) side
                if own.v > other.v and heading_diff < same_course_tolerance:
                    # Other vessel is ahead (in forward sector, not astern)
                    if abs(rel_bearing) <= fwd_sector:
                        prefer_starboard = True
                        closest_threat_dist = min(closest_threat_dist, dist)
                        break

                # --- Head-on (Rule 14): other near dead ahead and reciprocal courses
                # Reciprocal ~ pi (180°). Use > 150° as tolerant threshold.
                if abs(rel_bearing) <= fwd_sector and heading_diff >= math.radians(150.0):
                    prefer_starboard = True
                    closest_threat_dist = min(closest_threat_dist, dist)
                    break

                # --- Crossing from starboard (Rule 15): other on your starboard bow sector
                # According to COLREG code: "Starboard side = negative (clockwise) relative bearing"
                # Sector: starboard bow is rel_bearing in [-112.5°, 0)
                # Exclude "dead ahead" (already handled) and "abaft the beam" (overtaking-ish)
                if (-starboard_sector < rel_bearing < -math.radians(1.0)):
                    prefer_starboard = True
                    closest_threat_dist = min(closest_threat_dist, dist)
                    break

        # -----------------------------
        # Alignment rule (your original)
        # -----------------------------
        angle_err_now = self._wrap_angle(theta_target - own.psi)
        aligned = abs(angle_err_now) <= math.radians(90.0)

        # Weights (tune)
        w_smooth = 0.05          # penalize large immediate heading change a bit
        w_starboard = 3.0        # how strongly we bias toward starboard when needed (increased for COLREG compliance)
        
        # Distance-based scaling for COLREG bias (stronger when closer, weaker when farther)
        # Scale from 1.0 at collision_radius to 0.3 at colreg_radius
        # Also apply traffic_passing_factor to reduce bias when traffic is passing/moving away
        if prefer_starboard and closest_threat_dist < float('inf'):
            colreg_scale = 1.0 - 0.7 * max(0.0, (closest_threat_dist - self.cfg.collision_radius) / 
                                          (self.cfg.colreg_radius - self.cfg.collision_radius))
            colreg_scale = max(0.3, min(1.0, colreg_scale))  # Clamp to [0.3, 1.0]
            colreg_scale *= traffic_passing_factor  # Reduce further if traffic is passing
        else:
            colreg_scale = 1.0

        best_idx = int(idxs[0])
        best_score = float("inf")

        if aligned:
            for m in idxs:
                u = float(u_candidates[m])

                # heading after first step
                psi1 = self._wrap_angle(own.psi + u * dt)

                # primary: align to target
                score = ang_diff(theta_target, psi1)

                # mild smoothness preference
                score += w_smooth * ang_diff(psi1, own.psi)

                # COLREG bias: penalize port when we prefer starboard
                # Scale bias based on distance to threat (stronger when closer)
                if prefer_starboard:
                    if is_port_turn(u):
                        score += w_starboard * 1.5 * colreg_scale  # Penalty for port turn (COLREG violation)
                    elif is_starboard_turn(u):
                        score -= w_starboard * 0.5 * colreg_scale  # Bonus for starboard turn (COLREG compliant)

                if score < best_score:
                    best_score = score
                    best_idx = int(m)

            return best_idx

        else:
            wp_x, wp_y = waypoint_xy

            # Precompute path lengths only once for feasible candidates (optional but faster)
            # path_len[m] = sum ||p[h]-p[h-1]||
            path_len = {}
            for m in idxs:
                traj = own_trajs[m]  # (H, 2)
                diffs = traj[1:] - traj[:-1]
                path_len[m] = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

            for m in idxs:
                u = float(u_candidates[m])
                end_x, end_y = own_trajs[m, H - 1]

                # primary: endpoint distance to waypoint
                score = math.hypot(end_x - wp_x, end_y - wp_y)

                # detour penalty: discourage overly long trajectories
                straight = math.hypot(wp_x - own.x, wp_y - own.y)
                # avoid division by zero
                if straight > 1e-6:
                    detour_ratio = path_len[m] / straight
                    if detour_ratio > 1.3:
                        score += 0.2 * (detour_ratio - 1.3)  # soft penalty

                # COLREG bias
                # Scale bias based on distance to threat and waypoint distance
                # Make it proportional to waypoint distance so it doesn't override waypoint following too much
                if prefer_starboard:
                    # Scale by both threat distance and waypoint distance for balanced behavior
                    # Reduce bias when far from waypoint to prioritize waypoint following
                    waypoint_scale = min(1.0, straight / 4000.0)  # Normalize to waypoint distance
                    combined_scale = colreg_scale * waypoint_scale
                    
                    if is_port_turn(u):
                        score += w_starboard * 15.0 * combined_scale  # Penalty for port turn (COLREG violation)
                    elif is_starboard_turn(u):
                        score -= w_starboard * 5.0 * combined_scale  # Bonus for starboard turn (COLREG compliant)

                if score < best_score:
                    best_score = score
                    best_idx = int(m)

            return best_idx

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

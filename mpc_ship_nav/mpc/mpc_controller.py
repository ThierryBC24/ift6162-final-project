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
    horizon: int = 16                  # H
    n_candidates: int = 45             # M
    max_delta_heading_deg: float = 20  # Δθ_max per step
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


class SimplifiedMPCController(Controller):
    """
    Simplified fan-based MPC controller as in Sec. 2.6.3:
    - Generate M constant-turn trajectories over H steps.
    - Discard those that violate d_collision to static/dynamic obstacles.
    - Select trajectory based on heading/endpoint relative to next waypoint.
    - Apply only first control input (receding horizon).
    """

    def __init__(self, dt: float, waypoints_xy: np.ndarray) -> None:
        self.cfg = MPCConfig(dt=dt)
        self.route = WaypointRoute(
            waypoints_xy=np.asarray(waypoints_xy, dtype=float),
            transition_radius=self.cfg.collision_radius,
        )

    def compute_control(
        self,
        t: float,
        own_ship: Vessel,
        other_vessels: List[Vessel],
        env: ChartEnvironment,
    ) -> float:
        own = own_ship.state  # VesselState

        # Ensure we have local coordinates
        if own.x is None or own.y is None:
            own.x, own.y = env.to_local(own.lat, own.lon)

        # 1) Get the current waypoint in local coords, possibly advance
        wp_x, wp_y = self.route.current_waypoint(own)

        # If we are at the last waypoint, just go straight
        if self.route.is_finished():
            return 0.0

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
        u_candidates = np.linspace(
            -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate, self.cfg.n_candidates
        )
        feasible_mask, own_trajs = self._simulate_and_filter(
            own, u_candidates, dyn_trajs, static_xy=np.empty((0, 2))  # Empty static obstacles array
        )

        if not np.any(feasible_mask):
            # fallback: simple proportional heading correction toward waypoint
            angle_err = self._wrap_angle(theta_target - own.psi)
            u_fallback = np.clip(angle_err / self.cfg.dt,
                                -self.cfg.max_yaw_rate,
                                self.cfg.max_yaw_rate)
            return float(u_fallback)

        # 6) Select the best feasible candidate
        idx = self._select_best(
            own,
            (wp_x, wp_y),
            theta_target,
            u_candidates,
            own_trajs,
            feasible_mask,
        )

        return float(u_candidates[idx])

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


    # ------------------------------------------------------------------
    # MPC internals
    # ------------------------------------------------------------------

    def _simulate_and_filter(
        self,
        own: VesselState,
        u_candidates: np.ndarray,
        dyn_trajs: List[np.ndarray],
        static_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate own-ship trajectories for each yaw-rate candidate and
        return:
            feasible_mask: shape (M,), bool
            own_trajs:     shape (M, H, 2) (points over horizon)
        """
        H = self.cfg.horizon
        dt = self.cfg.dt
        v = own.v

        M = len(u_candidates)

        own_trajs = np.zeros((M, H, 2), dtype=float)
        feasible = np.ones(M, dtype=bool)

        for m in range(M):
            u = u_candidates[m]
            px = own.x
            py = own.y
            psi = own.psi

            for h in range(H):
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

        return feasible, own_trajs

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
        own_trajs: np.ndarray,
        feasible_mask: np.ndarray,
    ) -> int:
        """
        Selection rule from Sec. 2.6.3:
        - if own heading is within ±90° of target bearing: select trajectory whose
          initial heading aligns best with theta_target;
        - else: select trajectory whose endpoint is closest to waypoint.
        """
        H = self.cfg.horizon

        # alignment check
        angle_err_now = self._wrap_angle(theta_target - own.psi)
        aligned = abs(angle_err_now) <= math.radians(90.0)

        idxs = np.where(feasible_mask)[0]
        if aligned:
            best_score = float("inf")
            best_idx = int(idxs[0])
            for m in idxs:
                # heading after first step: psi_1 = psi_0 + u * dt
                psi1 = self._wrap_angle(own.psi + u_candidates[m] * self.cfg.dt)
                score = abs(self._wrap_angle(theta_target - psi1))
                if score < best_score:
                    best_score = score
                    best_idx = int(m)
            return best_idx
        else:
            wp_x, wp_y = waypoint_xy
            best_score = float("inf")
            best_idx = int(idxs[0])
            for m in idxs:
                end_x, end_y = own_trajs[m, H - 1]
                score = math.hypot(end_x - wp_x, end_y - wp_y)
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

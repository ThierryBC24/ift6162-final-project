from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..dynamics.vessel import VesselState


@dataclass
class MPCConfig:
    dt: float
    horizon: int = 16               # H
    n_trajectories: int = 45        # M
    max_delta_heading_deg: float = 20.0  # Δθ_max per step
    collision_radius: float = 0.5 * 1852  # 0.5 nm
    colreg_radius: float = 3.0 * 1852    # 3 nm

    @property
    def max_delta_heading(self) -> float:
        """Δθ_max in radians (per step)."""
        return np.deg2rad(self.max_delta_heading_deg)

    @property
    def max_yaw_rate(self) -> float:
        """Corresponding yaw rate [rad/s] so that u*dt = Δθ."""
        return self.max_delta_heading / self.dt


class SimplifiedMPC:
    """
    Simplified fan-based MPC using the same kinematics as Vessel.step:
    psi_{k+1} = psi_k + u * dt
    x_{k+1}   = x_k + v * cos(psi_{k+1}) * dt
    y_{k+1}   = y_k + v * sin(psi_{k+1}) * dt
    """

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_control(
        self,
        own: VesselState,
        waypoint_xy: Tuple[float, float],
        dynamic_obstacles: List[VesselState],
        static_obstacles_xy: np.ndarray,
    ) -> float:
        """
        Returns optimal yaw rate command [rad/s] for the next step.
        """
        cfg = self.cfg

        # 1) Generate candidate trajectories (fan in yaw-rate)
        (u_candidates,
         traj_x, traj_y, traj_psi) = self._generate_fan(own)

        # 2) Predict motion of dynamic obstacles (constant-heading, constant-speed)
        dyn_x, dyn_y = self._predict_dynamic(dynamic_obstacles)

        # 3) Feasibility check (static + dynamic) within collision_radius
        feasible = self._check_feasibility(
            traj_x, traj_y,
            static_obstacles_xy,
            dyn_x, dyn_y,
        )

        # If nothing is feasible, choose u closest to 0
        if not np.any(feasible):
            idx = np.argmin(np.abs(u_candidates))
            return float(u_candidates[idx])

        # 4) Select "best" feasible trajectory (waypoint-based rule)
        wp_x, wp_y = waypoint_xy
        best_idx = self._select_best(
            own, (wp_x, wp_y),
            u_candidates,
            traj_x, traj_y, traj_psi,
            feasible,
        )

        return float(u_candidates[best_idx])

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_fan(self, own: VesselState):
        cfg = self.cfg
        H = cfg.horizon
        M = cfg.n_trajectories

        # yaw-rate candidates in [-max_yaw_rate, max_yaw_rate]
        u_candidates = np.linspace(-cfg.max_yaw_rate, cfg.max_yaw_rate, M)

        traj_x = np.zeros((M, H + 1))
        traj_y = np.zeros((M, H + 1))
        traj_psi = np.zeros((M, H + 1))

        traj_x[:, 0] = own.x
        traj_y[:, 0] = own.y
        traj_psi[:, 0] = own.psi

        for h in range(H):
            traj_psi[:, h + 1] = self._wrap(traj_psi[:, h] + u_candidates * cfg.dt)

            traj_x[:, h + 1] = traj_x[:, h] + own.v * np.cos(traj_psi[:, h + 1]) * cfg.dt
            traj_y[:, h + 1] = traj_y[:, h] + own.v * np.sin(traj_psi[:, h + 1]) * cfg.dt

        return u_candidates, traj_x, traj_y, traj_psi

    # ------------------------------------------------------------------
    # Dynamic obstacles prediction
    # ------------------------------------------------------------------

    def _predict_dynamic(self, dyn: List[VesselState]):
        cfg = self.cfg
        H = cfg.horizon
        N = len(dyn)

        if N == 0:
            return np.zeros((0, H + 1)), np.zeros((0, H + 1))

        dyn_x = np.zeros((N, H + 1))
        dyn_y = np.zeros((N, H + 1))

        for i, s in enumerate(dyn):
            dyn_x[i, 0] = s.x
            dyn_y[i, 0] = s.y
            for h in range(H):
                dyn_x[i, h + 1] = dyn_x[i, h] + s.v * np.cos(s.psi) * cfg.dt
                dyn_y[i, h + 1] = dyn_y[i, h] + s.v * np.sin(s.psi) * cfg.dt

        return dyn_x, dyn_y

    # ------------------------------------------------------------------
    # Feasibility (Eq. 14)
    # ------------------------------------------------------------------

    def _check_feasibility(
        self,
        traj_x: np.ndarray,
        traj_y: np.ndarray,
        static_xy: np.ndarray,
        dyn_x: np.ndarray,
        dyn_y: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        M, H_plus_1 = traj_x.shape
        feasible = np.ones(M, dtype=bool)

        # --- Static obstacles
        if static_xy is not None and len(static_xy) > 0:
            sx = static_xy[:, 0][None, None, :]
            sy = static_xy[:, 1][None, None, :]

            px = traj_x[:, :, None]
            py = traj_y[:, :, None]

            d2 = (px - sx) ** 2 + (py - sy) ** 2
            too_close = np.any(d2 < cfg.collision_radius**2, axis=(1, 2))
            feasible &= ~too_close

        # --- Dynamic obstacles
        if dyn_x.size > 0:
            px = traj_x[:, :, None]
            py = traj_y[:, :, None]

            dx = np.swapaxes(dyn_x[None, :, :], 1, 2)
            dy = np.swapaxes(dyn_y[None, :, :], 1, 2)

            d2 = (px - dx) ** 2 + (py - dy) ** 2
            too_close = np.any(d2 < cfg.collision_radius**2, axis=(1, 2))
            feasible &= ~too_close

        return feasible

    # ------------------------------------------------------------------
    # Selection rule (Eqs. 15–17)
    # ------------------------------------------------------------------

    def _select_best(
        self,
        own: VesselState,
        waypoint_xy: Tuple[float, float],
        u_candidates: np.ndarray,
        traj_x: np.ndarray,
        traj_y: np.ndarray,
        traj_psi: np.ndarray,
        feasible: np.ndarray,
    ) -> int:
        wp_x, wp_y = waypoint_xy

        theta_target = np.arctan2(wp_y - own.y, wp_x - own.x)
        aligned = np.abs(self._angdiff(own.psi, theta_target)) <= np.deg2rad(90.0)

        idxs = np.where(feasible)[0]

        if aligned:
            # choose heading at first step closest to target bearing
            psi1 = traj_psi[idxs, 1]
            err = np.abs(self._angdiff(psi1, theta_target))
            return idxs[np.argmin(err)]
        else:
            # choose final position closest to waypoint
            xf = traj_x[idxs, -1]
            yf = traj_y[idxs, -1]
            d2 = (xf - wp_x) ** 2 + (yf - wp_y) ** 2
            return idxs[np.argmin(d2)]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _angdiff(self, a, b):
        return self._wrap(a - b)

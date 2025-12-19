"""
Unit tests for MPC controller components.
"""

import math
import numpy as np
import pytest

from mpc_ship_nav.mpc.mpc_controller import (
    MPCConfig,
    WaypointRoute,
    SimplifiedMPCController,
)
from mpc_ship_nav.dynamics.vessel import VesselState, VesselParams, Vessel


class TestMPCConfig:
    """Tests for MPCConfig dataclass."""

    def test_max_delta_heading(self):
        """Test max_delta_heading property conversion."""
        cfg = MPCConfig(dt=1.0, max_delta_heading_deg=20.0)
        assert abs(cfg.max_delta_heading - math.radians(20.0)) < 1e-6

    def test_max_yaw_rate(self):
        """Test max_yaw_rate calculation."""
        cfg = MPCConfig(dt=2.0, max_delta_heading_deg=20.0)
        expected = math.radians(20.0) / 2.0
        assert abs(cfg.max_yaw_rate - expected) < 1e-6

    def test_collision_radius(self):
        """Test collision radius conversion from nautical miles to meters."""
        cfg = MPCConfig(dt=1.0, collision_radius_nm=0.5)
        expected = 0.5 * 1852.0  # 926.0 m
        assert abs(cfg.collision_radius - expected) < 1e-6

    def test_colreg_radius(self):
        """Test COLREG radius conversion from nautical miles to meters."""
        cfg = MPCConfig(dt=1.0, colreg_radius_nm=3.0)
        expected = 3.0 * 1852.0  # 5556.0 m
        assert abs(cfg.colreg_radius - expected) < 1e-6


class TestWaypointRoute:
    """Tests for WaypointRoute class."""

    def test_current_waypoint_initial(self):
        """Test initial waypoint is returned correctly."""
        waypoints = np.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
        route = WaypointRoute(waypoints, transition_radius=100.0)
        
        # Start away from first waypoint to avoid immediate transition
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=500.0, y=0.0)
        wp_x, wp_y = route.current_waypoint(state)
        
        # Should return first waypoint since we're > 100m away
        assert wp_x == 0.0
        assert wp_y == 0.0
        assert route.idx == 0

    def test_waypoint_transition(self):
        """Test waypoint advances when within transition radius."""
        waypoints = np.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
        route = WaypointRoute(waypoints, transition_radius=100.0)
        
        # Start away from first waypoint
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=500.0, y=0.0)
        route.current_waypoint(state)
        assert route.idx == 0
        
        # Move close to first waypoint (within transition radius)
        state.x = 50.0
        state.y = 0.0
        wp_x, wp_y = route.current_waypoint(state)
        
        # Should advance to second waypoint
        assert route.idx == 1
        assert wp_x == 1000.0
        assert wp_y == 0.0

    def test_waypoint_no_transition_at_last(self):
        """Test that waypoint doesn't advance beyond last waypoint."""
        waypoints = np.array([[0.0, 0.0], [1000.0, 0.0]])
        route = WaypointRoute(waypoints, transition_radius=100.0)
        
        # Move to last waypoint
        route.idx = 1
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=1000.0, y=0.0)
        wp_x, wp_y = route.current_waypoint(state)
        
        # Should stay at last waypoint
        assert route.idx == 1
        assert wp_x == 1000.0
        assert wp_y == 0.0

    def test_is_finished(self):
        """Test is_finished method."""
        waypoints = np.array([[0.0, 0.0], [1000.0, 0.0]])
        route = WaypointRoute(waypoints, transition_radius=100.0)
        
        assert not route.is_finished()  # At first waypoint
        
        route.idx = 1
        assert route.is_finished()  # At last waypoint

    def test_empty_waypoints(self):
        """Test behavior with empty waypoint list."""
        waypoints = np.array([]).reshape(0, 2)
        route = WaypointRoute(waypoints, transition_radius=100.0)
        
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=100.0, y=200.0)
        wp_x, wp_y = route.current_waypoint(state)
        
        # Should return current position
        assert wp_x == 100.0
        assert wp_y == 200.0


class TestAngleWrapping:
    """Tests for angle wrapping utility."""

    def test_wrap_angle_normal_range(self):
        """Test angles already in [-π, π] range."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [1000, 0]]))
        
        assert abs(controller._wrap_angle(0.0)) < 1e-6
        assert abs(controller._wrap_angle(math.pi / 2) - math.pi / 2) < 1e-6
        assert abs(controller._wrap_angle(-math.pi / 2) + math.pi / 2) < 1e-6

    def test_wrap_angle_positive_overflow(self):
        """Test wrapping angles > π."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [1000, 0]]))
        
        # 2π should wrap to 0
        assert abs(controller._wrap_angle(2 * math.pi)) < 1e-6
        
        # 3π/2 should wrap to -π/2
        assert abs(controller._wrap_angle(3 * math.pi / 2) + math.pi / 2) < 1e-6
        
        # π + ε should wrap to -π + ε
        assert abs(controller._wrap_angle(math.pi + 0.1) + math.pi - 0.1) < 1e-6

    def test_wrap_angle_negative_overflow(self):
        """Test wrapping angles < -π."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [1000, 0]]))
        
        # -2π should wrap to 0
        assert abs(controller._wrap_angle(-2 * math.pi)) < 1e-6
        
        # -3π/2 should wrap to π/2
        assert abs(controller._wrap_angle(-3 * math.pi / 2) - math.pi / 2) < 1e-6
        
        # -π - ε should wrap to π - ε
        assert abs(controller._wrap_angle(-math.pi - 0.1) - math.pi + 0.1) < 1e-6


class TestSelectBest:
    """Tests for _select_best trajectory selection method."""

    def test_aligned_case_minimize_heading_error(self):
        """Test Equation (16): minimize heading error when aligned."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [1000, 0]]))
        
        # Own ship heading East (0°), waypoint also East
        own = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        waypoint_xy = (1000.0, 0.0)  # East
        theta_target = 0.0  # East
        
        # Create candidate trajectories with different yaw rates
        n_candidates = 5
        u_candidates = np.linspace(-0.1, 0.1, n_candidates)  # Small yaw rates
        
        # Simulate trajectories
        H = controller.cfg.horizon
        own_trajs = np.zeros((n_candidates, H, 2))
        for m, u in enumerate(u_candidates):
            x, y = 0.0, 0.0
            psi = 0.0
            for h in range(H):
                psi = controller._wrap_angle(psi + u * controller.cfg.dt)
                x += 8.0 * math.cos(psi) * controller.cfg.dt
                y += 8.0 * math.sin(psi) * controller.cfg.dt
                own_trajs[m, h] = [x, y]
        
        feasible_mask = np.ones(n_candidates, dtype=bool)
        
        best_idx = controller._select_best(
            own, waypoint_xy, theta_target, u_candidates, own_trajs, feasible_mask
        )
        
        # Should select trajectory with smallest heading error after 1 step
        # u=0 should be best (no turn, heading stays at 0°)
        assert best_idx == 2  # Middle index (u=0)

    def test_unaligned_case_minimize_distance(self):
        """Test Equation (17): minimize endpoint distance when not aligned."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [0, 1000]]))
        
        # Own ship heading East (0°), waypoint North (90°)
        own = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        waypoint_xy = (0.0, 1000.0)  # North
        theta_target = math.pi / 2  # North
        
        # Create candidate trajectories
        n_candidates = 5
        u_candidates = np.linspace(-0.2, 0.2, n_candidates)
        
        # Simulate trajectories
        H = controller.cfg.horizon
        own_trajs = np.zeros((n_candidates, H, 2))
        for m, u in enumerate(u_candidates):
            x, y = 0.0, 0.0
            psi = 0.0
            for h in range(H):
                psi = controller._wrap_angle(psi + u * controller.cfg.dt)
                x += 8.0 * math.cos(psi) * controller.cfg.dt
                y += 8.0 * math.sin(psi) * controller.cfg.dt
                own_trajs[m, h] = [x, y]
        
        feasible_mask = np.ones(n_candidates, dtype=bool)
        
        best_idx = controller._select_best(
            own, waypoint_xy, theta_target, u_candidates, own_trajs, feasible_mask
        )
        
        # Should select trajectory that ends closest to waypoint
        # Positive yaw rate (turning left toward North) should be better
        assert best_idx >= 2  # Should prefer positive yaw rates

    def test_alignment_threshold(self):
        """Test that 90° threshold correctly determines alignment."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [1000, 0]]))
        
        # Test at exactly 90° (should be considered aligned)
        own = VesselState(lat=0.0, lon=0.0, psi=math.pi / 2, v=8.0, x=0.0, y=0.0)
        waypoint_xy = (1000.0, 0.0)
        theta_target = 0.0
        
        n_candidates = 3
        u_candidates = np.array([-0.1, 0.0, 0.1])
        H = controller.cfg.horizon
        own_trajs = np.zeros((n_candidates, H, 2))
        feasible_mask = np.ones(n_candidates, dtype=bool)
        
        best_idx = controller._select_best(
            own, waypoint_xy, theta_target, u_candidates, own_trajs, feasible_mask
        )
        
        # Should still return a valid index
        assert 0 <= best_idx < n_candidates

    def test_no_feasible_trajectories(self):
        """Test behavior when no trajectories are feasible."""
        controller = SimplifiedMPCController(dt=1.0, waypoints_xy=np.array([[0, 0], [1000, 0]]))
        
        own = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        waypoint_xy = (1000.0, 0.0)
        theta_target = 0.0
        
        n_candidates = 5
        u_candidates = np.linspace(-0.1, 0.1, n_candidates)
        H = controller.cfg.horizon
        own_trajs = np.zeros((n_candidates, H, 2))
        feasible_mask = np.zeros(n_candidates, dtype=bool)  # All infeasible
        
        best_idx = controller._select_best(
            own, waypoint_xy, theta_target, u_candidates, own_trajs, feasible_mask
        )
        
        # Should return a valid index (fallback behavior)
        assert 0 <= best_idx < n_candidates


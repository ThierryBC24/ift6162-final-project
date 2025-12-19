"""
Unit tests for vessel dynamics.
"""

import math
import numpy as np
import pytest

from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams


class TestVesselState:
    """Tests for VesselState dataclass."""

    def test_copy(self):
        """Test VesselState copy method."""
        state = VesselState(
            lat=45.0, lon=-73.0, psi=math.pi / 4, v=10.0, x=100.0, y=200.0
        )
        copied = state.copy()
        
        assert copied.lat == state.lat
        assert copied.lon == state.lon
        assert copied.psi == state.psi
        assert copied.v == state.v
        assert copied.x == state.x
        assert copied.y == state.y
        
        copied.x = 999.0
        assert state.x == 100.0


class TestVesselDynamics:
    """Tests for Vessel step method and kinematics."""

    def test_step_straight_ahead(self):
        """Test vessel moving straight ahead (no yaw rate)."""
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=10.0, x=0.0, y=0.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        dt = 1.0
        vessel.step(u=0.0, dt=dt)
        
        expected_x = 10.0
        expected_y = 0.0
        
        assert abs(vessel.state.x - expected_x) < 1e-6
        assert abs(vessel.state.y - expected_y) < 1e-6
        assert abs(vessel.state.psi - 0.0) < 1e-6

    def test_step_turn_starboard(self):
        """Test vessel turning starboard (negative yaw rate)."""
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=10.0, x=0.0, y=0.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        dt = 1.0
        u = -math.radians(10.0)
        vessel.step(u=u, dt=dt)
        
        assert vessel.state.psi < 0.0
        assert abs(vessel.state.psi - (-math.radians(10.0))) < 1e-3

    def test_step_turn_port(self):
        """Test vessel turning port (positive yaw rate)."""
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=10.0, x=0.0, y=0.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        dt = 1.0
        u = math.radians(10.0)
        vessel.step(u=u, dt=dt)
        
        assert vessel.state.psi > 0.0
        assert abs(vessel.state.psi - math.radians(10.0)) < 1e-3

    def test_step_yaw_rate_clipping(self):
        """Test that yaw rate is clipped to max_yaw_rate."""
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=10.0, x=0.0, y=0.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        dt = 1.0
        u_excessive = math.radians(50.0)
        vessel.step(u=u_excessive, dt=dt)
        
        expected_psi = math.radians(20.0)
        assert abs(vessel.state.psi - expected_psi) < 1e-3

    def test_step_heading_wrapping(self):
        """Test that heading wraps correctly around ±π."""
        state = VesselState(lat=0.0, lon=0.0, psi=math.pi - 0.1, v=10.0, x=0.0, y=0.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        initial_psi = state.psi
        dt = 1.0
        u = math.radians(15.0)
        vessel.step(u=u, dt=dt)
        
        expected_psi = initial_psi + u * dt
        expected_psi = np.arctan2(np.sin(expected_psi), np.cos(expected_psi))
        
        assert -math.pi <= vessel.state.psi <= math.pi, \
            f"Heading {vessel.state.psi} should be in [-π, π]"
        assert abs(vessel.state.psi - expected_psi) < 1e-6, \
            f"Expected wrapped heading {expected_psi:.6f}, got {vessel.state.psi:.6f}"

    def test_step_multiple_steps(self):
        """Test vessel over multiple time steps."""
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=10.0, x=0.0, y=0.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        dt = 1.0
        u = math.radians(5.0)
        
        for _ in range(5):
            vessel.step(u=u, dt=dt)
        
        assert vessel.state.x > 0.0
        assert vessel.state.y > 0.0
        assert vessel.state.psi > 0.0

    def test_get_position(self):
        """Test get_position method."""
        state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=10.0, x=123.0, y=456.0)
        params = VesselParams(max_yaw_rate=math.radians(20.0))
        vessel = Vessel(state, params)
        
        x, y = vessel.get_position()
        
        assert x == 123.0
        assert y == 456.0


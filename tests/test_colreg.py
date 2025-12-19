"""
Unit tests for COLREG compliance logic.
"""

import math
import numpy as np
import pytest

from mpc_ship_nav.dynamics.colreg import COLREGLogic, D_COLLISION, D_COLREG
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams


class TestCOLREGLogic:
    """Tests for COLREG compliance logic."""

    def test_distance_calculation(self):
        """Test distance calculation between vessels."""
        logic = COLREGLogic()
        
        state1 = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        state2 = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=1000.0, y=0.0)
        
        vessel1 = Vessel(state1, VesselParams(max_yaw_rate=math.radians(20.0)))
        vessel2 = Vessel(state2, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        distance = logic._distance(vessel1, vessel2)
        assert abs(distance - 1000.0) < 1e-6

    def test_risk_of_collision_approaching(self):
        """Test collision risk detection for approaching vessels."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=math.pi, v=8.0, x=2000.0, y=0.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        risk = logic._risk_of_collision(target, own_ship)
        assert risk is True

    def test_risk_of_collision_diverging(self):
        """Test collision risk for diverging vessels."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=0.0, v=8.0, x=2000.0, y=0.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        risk = logic._risk_of_collision(target, own_ship)
        assert risk is False

    def test_classify_encounter_headon(self):
        """Test head-on encounter classification."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=math.pi, v=8.0, x=2000.0, y=0.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        encounter = logic._classify_encounter(target, own_ship)
        assert encounter == "head_on"

    def test_classify_encounter_crossing_starboard(self):
        """Test crossing encounter from starboard side."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=math.pi / 2, v=8.0, x=2000.0, y=-2000.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        encounter = logic._classify_encounter(target, own_ship)
        assert encounter == "crossing"

    def test_classify_encounter_overtaking(self):
        """Test overtaking encounter classification."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=5.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=0.0, v=10.0, x=-1000.0, y=0.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        encounter = logic._classify_encounter(target, own_ship)
        assert encounter == "overtaking"

    def test_target_must_give_way_crossing_starboard(self):
        """Test COLREG crossing give-way logic - target on starboard must give way."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=math.pi / 2, v=8.0, x=0.0, y=-1000.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        encounter = logic._classify_encounter(target, own_ship)
        assert encounter == "crossing"
        
        must_give_way = logic._target_must_give_way(encounter, target, own_ship)
        
        assert must_give_way == True

    def test_target_must_give_way_crossing_port(self):
        """Test COLREG crossing give-way logic - target on port does NOT give way."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=math.pi / 2, v=8.0, x=0.0, y=1000.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        encounter = logic._classify_encounter(target, own_ship)
        assert encounter == "crossing"
        
        must_give_way = logic._target_must_give_way(encounter, target, own_ship)
        
        assert must_give_way == False

    def test_compute_target_control_no_risk(self):
        """Test that target maintains course when no collision risk."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=0.0, v=8.0, x=10000.0, y=0.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        u = logic.compute_target_control(target, own_ship)
        
        assert abs(u) < 1e-6

    def test_compute_target_control_emergency_avoidance(self):
        """Test emergency collision avoidance (< 0.5 nm)."""
        logic = COLREGLogic()
        
        own_state = VesselState(lat=0.0, lon=0.0, psi=0.0, v=8.0, x=0.0, y=0.0)
        own_ship = Vessel(own_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        target_state = VesselState(
            lat=0.0, lon=0.0, psi=math.pi, v=8.0, x=500.0, y=0.0
        )
        target = Vessel(target_state, VesselParams(max_yaw_rate=math.radians(20.0)))
        
        u = logic.compute_target_control(target, own_ship)
        
        assert abs(u) > 1e-6

    def test_wrap_angle(self):
        """Test angle wrapping utility."""
        logic = COLREGLogic()
        
        assert abs(logic._wrap_angle(0.0)) < 1e-6
        assert abs(logic._wrap_angle(math.pi / 2) - math.pi / 2) < 1e-6
        assert abs(logic._wrap_angle(2 * math.pi)) < 1e-6
        assert abs(logic._wrap_angle(3 * math.pi / 2) + math.pi / 2) < 1e-6
        assert abs(logic._wrap_angle(-2 * math.pi)) < 1e-6
        assert abs(logic._wrap_angle(-3 * math.pi / 2) - math.pi / 2) < 1e-6


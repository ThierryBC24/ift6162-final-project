"""
Unit tests for StaticControlSeqGenerator.
"""

import math
import numpy as np
import pytest

from mpc_ship_nav.mpc.mpc_controller import StaticControlSeqGenrator


class TestStaticControlSeqGenerator:
    """Tests for StaticControlSeqGenerator class."""

    def test_generate_controls_shape(self):
        """Test that control sequences have correct shape."""
        generator = StaticControlSeqGenrator(
            max_yaw_rate=math.radians(20.0),
            horizon=16,
            num_trajectories=45,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        assert controls.shape == (45, 16)

    def test_generate_controls_range(self):
        """Test that initial yaw rates span the correct range."""
        max_yaw_rate = math.radians(20.0)
        generator = StaticControlSeqGenrator(
            max_yaw_rate=max_yaw_rate,
            horizon=16,
            num_trajectories=45,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        
        # First column should span [-max_yaw_rate, max_yaw_rate]
        first_column = controls[:, 0]
        assert abs(first_column[0] - (-max_yaw_rate)) < 1e-6
        assert abs(first_column[-1] - max_yaw_rate) < 1e-6

    def test_decay_factor(self):
        """Test that yaw rate decays by decay_factor each step."""
        decay_factor = 0.95
        generator = StaticControlSeqGenrator(
            max_yaw_rate=math.radians(20.0),
            horizon=16,
            num_trajectories=5,
            decay_factor=decay_factor
        )
        
        controls = generator.control_sequences
        
        # For each trajectory, check decay
        for traj_idx in range(5):
            for step in range(15):  # Check up to second-to-last step
                current = controls[traj_idx, step]
                next_val = controls[traj_idx, step + 1]
                
                if abs(current) > 1e-6:  # Avoid division by zero
                    actual_decay = next_val / current
                    assert abs(actual_decay - decay_factor) < 1e-6

    def test_zero_decay_factor(self):
        """Test behavior with zero decay factor."""
        generator = StaticControlSeqGenrator(
            max_yaw_rate=math.radians(20.0),
            horizon=5,
            num_trajectories=3,
            decay_factor=0.0
        )
        
        controls = generator.control_sequences
        
        # After first step, all should be zero
        for traj_idx in range(3):
            for step in range(1, 5):
                assert abs(controls[traj_idx, step]) < 1e-6

    def test_single_trajectory(self):
        """Test with single trajectory."""
        generator = StaticControlSeqGenrator(
            max_yaw_rate=math.radians(20.0),
            horizon=10,
            num_trajectories=1,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        assert controls.shape == (1, 10)
        # With num_trajectories=1, np.linspace(-max, max, 1) returns [-max]
        # So the first value should be -max_yaw_rate
        assert abs(controls[0, 0] - (-math.radians(20.0))) < 1e-6

    def test_horizon_one(self):
        """Test with horizon of 1."""
        generator = StaticControlSeqGenrator(
            max_yaw_rate=math.radians(20.0),
            horizon=1,
            num_trajectories=5,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        assert controls.shape == (5, 1)
        
        # Should still span the range
        first_column = controls[:, 0]
        max_yaw = math.radians(20.0)
        assert abs(first_column[0] - (-max_yaw)) < 1e-6
        assert abs(first_column[-1] - max_yaw) < 1e-6


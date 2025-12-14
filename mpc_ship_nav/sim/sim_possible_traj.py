import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.sim.engine import SimLog, SimConfig
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams
from tqdm import tqdm
import math

class StaticTrajectoryGenrator:
    """Generates a set of static trajectories based max yaw rate, horizon and number of trajectories."""
    
    def __init__(self, max_yaw_rate: float=np.radians(20), horizon: int=20, num_trajectories: int=45, decay_factor: float=0.95, scale: int=300):
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
        self.scale = scale  # control applied every 1000 steps
        self.trajectories = self.generate_trajectories()
        
        
    def generate_trajectories(self) -> List[List]:
        initial_angles = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, self.num_trajectories)
        # Initialize the control matrix
        controls = np.zeros((self.num_trajectories, self.horizon * self.scale))
        print("Generating trajectories with control shape:")
        print(controls.shape)

        # 2. Compute the sequence for each trajectory
        for i, start_angle in enumerate(initial_angles):
            dtheta = start_angle

            for h in range(self.horizon):
                # Store current control
                controls[i, h*self.scale] = (dtheta)

                # Apply decay for the next step (smoothing)
                # This corresponds to: dtheta = dtheta * .95
                dtheta *= self.decay_factor
                

        return controls 

class SimulateHypotheticalTraj:
    """Class to simulate possible trajectories for analysis purposes."""
    
    def __init__(self, env: ChartEnvironment, simLog: SimLog, horizon: int=20, max_yaw_rate: float=np.radians(20), number_of_trajectories: int=45, decay_factor: float=0.95, scale: int=300, dump_zone:int = 300, sim_config: SimConfig=SimConfig()):
        self.env = env
        self.own_ship_states = simLog.own_states
        self.traffic_states = simLog.traffic_states
        self.vessel_params = VesselParams(max_yaw_rate=max_yaw_rate)
        self.plot_dt = simLog.times[1] - simLog.times[0]
        self.trajectories_generator = StaticTrajectoryGenrator(max_yaw_rate=max_yaw_rate, horizon=horizon, num_trajectories=number_of_trajectories, decay_factor=decay_factor, scale=scale)
        self.sim_config = sim_config
        self.dump_zone = dump_zone
        
    def _get_env_state_at(self, time_index: int):
        """Get the environment state at a specific time index."""
        own_ship_state = self.own_ship_states[time_index]
        traffic_states = self.traffic_states[time_index] if self.traffic_states else []
        return own_ship_state, traffic_states
    
    def simulate_trajectories_at(self, time_index: int) -> List[Tuple[List[float], List[float]]]:
        """Simulate all possible trajectories from a specific time index.
        
        Args:
            time_index (int): The time index to simulate from.
        
        Returns:
            List[Tuple[List[float], List[float]]]: A list of trajectories, each represented as a tuple of x and y positions.
        """
        own_ship_state, _ = self._get_env_state_at(time_index)
        ship_params = self.vessel_params
        dt = self.sim_config.dt
        trajectories = []
        for control_sequence in self.trajectories_generator.trajectories:
            vessel = Vessel(own_ship_state.copy(), ship_params)
            x, y = vessel.get_position()

            trajectory_positions_x = [x]
            trajectory_positions_y = [y]
            for i, u in enumerate(control_sequence):
                vessel.step(u, dt, self.env)
                # print(f"Step {i}, control: {u}")
                if i%self.trajectories_generator.scale == self.trajectories_generator.scale - 1:
                    new_x, new_y = vessel.get_position()
                    trajectory_positions_x.append(new_x)
                    trajectory_positions_y.append(new_y)
            trajectories.append((trajectory_positions_x, trajectory_positions_y))
        return trajectories

    def simulate_all_trajectories(self) -> List[List[Tuple[List[float], List[float]]]]:
        """Simulate all possible trajectories for each time step in the log.
        
        Returns:
            List[List[Tuple[List[float], List[float]]]]: let res be the output of the function, res[t] is the list of hypothetical trajectories at time index t.
        """
        all_trajectories = []
        for time_index in tqdm(range(len(self.own_ship_states)), desc="Simulating hypothetical trajectories"):
            trajectories_at_time = self.simulate_trajectories_at(time_index)
            all_trajectories.append(trajectories_at_time)
        self.trajectories = all_trajectories
        return all_trajectories
    
    def color_trajectory_by_risk(self, trajectory: Tuple[List[float], List[float]], traffic_states: List[VesselState]) -> str:
        """Color a trajectory based on its risk level with respect to traffic vessels.
        
        Args:
            trajectory (Tuple[List[float], List[float]]): The trajectory to evaluate.
            traffic_states (List[VesselState]): The states of the traffic vessels at the starting time.
            dump_zone: The area considered as a dump zone.
        
        Returns:
            str: The color representing the risk level ('green', 'yellow', 'red').
        """
        traj_x, traj_y = trajectory
        horizon = len(traj_x)
        traffic_states = [ship.copy() for ship in traffic_states]
        for t in range(horizon):
            my_x, my_y = traj_x[t], traj_y[t]
            
            if not self.env.is_navigable(my_x, my_y):
                return 'red'  # Collision with land or coastline
            
            for ship in traffic_states:
                dist = math.sqrt((my_x - ship.x) ** 2 + (my_y - ship.y) ** 2)
                
                if dist < self.dump_zone:
                    return 'red'  # Collision with another vessel
                
                ship.x += ship.v * math.cos(0) * self.plot_dt
                ship.y += ship.v * math.sin(0) * self.plot_dt
        return 'green'  # Safe trajectory
    
    def color_all_trajectories_by_risk(self) -> List[List[str]]:
        """Color all simulated trajectories based on their risk levels.
        
        Returns:
            List[List[str]]: A list where each element corresponds to a time index and contains a list of colors for each trajectory.
        """
        all_colors = []
        try:
            trajectories = self.trajectories
        except AttributeError:
            trajectories = self.simulate_all_trajectories()
        for time_index in tqdm(range(len(self.own_ship_states)), desc="Coloring trajectories by risk"):
            _, traffic_states = self._get_env_state_at(time_index)
            trajectories_at_time = trajectories[time_index]
            colors_at_time = []
            for trajectory in trajectories_at_time:
                color = self.color_trajectory_by_risk(trajectory, traffic_states)
                colors_at_time.append(color)
            all_colors.append(colors_at_time)
        self.trajectory_colors = all_colors
        return all_colors
                
                 
    
    
    
            
    
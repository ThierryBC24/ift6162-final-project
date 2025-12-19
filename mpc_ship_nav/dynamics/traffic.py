from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import math
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams
from mpc_ship_nav.charts.environment import ChartEnvironment


R_EARTH = 3440.065  # Earth's radius in nautical miles


class GenerateTraffic:
    def __init__(self, own_ship: Vessel, distance: float, max_target_speed: float, epsilon: float = 1, env: ChartEnvironment = None):
        self.own_ship = own_ship # Vessel object representing the own ship
        self.mean_distance = distance # Distance from own ship to target ship in nautical miles
        self.max_target_speed = max_target_speed # Maximum speed of the target ship in m/s
        self.epsilon = epsilon # Small value that defines the range of variation when generating different encounter scenarios
        self.env = env # ChartEnvironment for navigability
        
    
    def _find_target_position(self, rel_bearing: float, distance: float) -> Tuple[float, float]:
        """Calculates the position of a target ship given a relative bearing and distance from the observer.
        Args:
            rel_bearing (float): Relative bearing from own ship to target ship in radians.
            distance (float): Distance from own ship to target ship in nautical miles.
            
        Returns:
            Tuple[float, float]: Target ship latitude and longitude in degrees.
        """
        lat1 = math.radians(self.own_ship.state.lat)
        lon1 = math.radians(self.own_ship.state.lon)
        bearing = (self.own_ship.state.psi + rel_bearing) % (2 * math.pi)

        angular_distance = distance / R_EARTH

        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance) +
            math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
        )

        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2)
        )

        lat2_deg = math.degrees(lat2)
        lon2_deg = math.degrees(lon2)

        return lat2_deg, lon2_deg
        
    def _find_target_velocity(self, t_lat: float, t_lon: float, rel_speed: float) -> Tuple[float, float]:
        """Calculates the target ship's speed and heading based on the own ship's velocity, relative bearing, and relative speed.

        Args:
            t_lat (float): Target ship latitude in degrees.
            t_lon (float): Target ship longitude in degrees.
            rel_speed (float): Relative speed between own ship and target ship in m/s.

        Returns:
            Tuple[float, float]: Target ship speed and heading in radians.
        """
        own_vx = self.own_ship.state.v * math.cos(self.own_ship.state.psi)
        own_vy = self.own_ship.state.v * math.sin(self.own_ship.state.psi)

        rel_bearing = math.atan2(
            t_lat - self.own_ship.state.lat,
            t_lon - self.own_ship.state.lon
        )

        rel_vx = rel_speed * math.cos(rel_bearing)
        rel_vy = rel_speed * math.sin(rel_bearing)

        target_vx = own_vx + rel_vx
        target_vy = own_vy + rel_vy

        target_speed = math.sqrt(target_vx**2 + target_vy**2)
        target_heading = math.atan2(target_vy, target_vx)

        return target_speed, target_heading
    
    def _pick_relative_bearing(self, mean_rel_bearing: float, rel_b_half_bound: float, distribution=np.random.uniform) -> float:
        """Picks a random relative bearing around a mean value within specified bounds.
        Args:
            mean_rel_bearing (float): Mean relative bearing in radians.
            rel_b_half_bound (float): Half the length of the bounds around the mean in radians.
            distribution (callable, optional): Function to sample the relative bearing. Defaults to np.random.uniform.
        Returns:
            float: Sampled relative bearing in radians.
        """
        lower_bound = mean_rel_bearing - rel_b_half_bound
        upper_bound = mean_rel_bearing + rel_b_half_bound
        rel_b = distribution(lower_bound, upper_bound)
        return rel_b % (2 * np.pi)
    
    def generate_target_ship(self, relative_bearing: float, relative_speed: float, mean_distance: float=None) -> VesselState:   
        """Generates a target ship state based on the own ship's state, relative bearing, distance, and relative speed.
        Args:
            mean_relative_bearing (float): Relative bearing from own ship to target ship in radians.
            mean_relative_speed (float): Relative speed between own ship and target ship in m/s.
            mean_distance (float, optional): Distance from own ship to target ship in nautical miles. Defaults to None, in which case the instance's mean_distance is used.
        Returns:
            VesselState: The state of the generated target ship.
        """
        m_d = self.mean_distance if mean_distance is None else mean_distance
            
        t_lat, t_lon = self._find_target_position(relative_bearing, m_d)
        t_speed, t_psi = self._find_target_velocity(t_lat, t_lon, relative_speed)

        target_state = VesselState(
            lat=t_lat,
            lon=t_lon,
            psi=t_psi,
            v=t_speed
        )

        return target_state
    
    def generate_overtaking_enc(self, epsilon = None) -> VesselState:
        """Generates a target ship state for an overtaking encounter scenario.
        Returns:
            VesselState: The state of the generated target ship.
        """
        eps = self.epsilon if epsilon is None else epsilon
        rel_bearing = self._pick_relative_bearing(0, (67.5 * np.pi / 180)*eps)  # Overtaking sector
        mean_speed = self.own_ship.state.v / 2
        rel_speed = np.random.uniform(mean_speed-mean_speed*eps, mean_speed+mean_speed*eps)
        target_state = self.generate_target_ship(rel_bearing, rel_speed)
        return target_state
    
    def generate_overtaken_enc(self, epsilon = None) -> VesselState:
        """Generates a target ship state for an overtaken encounter scenario.
        Returns:
            VesselState: The state of the generated target ship.
        """
        eps = self.epsilon if epsilon is None else epsilon
        rel_bearing = self._pick_relative_bearing(180, (67.5 * np.pi / 180)*eps)  # Overtaking sector
        mean_speed = self.own_ship.state.v / 2
        rel_speed = np.random.uniform(0.1, mean_speed+mean_speed*eps)
        target_state = self.generate_target_ship(rel_bearing, rel_speed)
        return target_state
    
    def generate_head_on_enc(self, epsilon = None) -> VesselState:
        """Generates a target ship state for a head-on encounter scenario.
        Returns:
            VesselState: The state of the generated target ship.
        """
        eps = self.epsilon if epsilon is None else epsilon
        rel_bearing = self._pick_relative_bearing(0, (80 * np.pi / 180)*eps)  # Head-on sector
        mean_speed = self.own_ship.state.v * 2
        rel_speed = np.random.uniform(mean_speed-mean_speed*eps, mean_speed+mean_speed*eps)
        target_state = self.generate_target_ship(rel_bearing, rel_speed)
        return target_state
    
    def generate_crossing_enc(self, epsilon = None) -> VesselState:
        """Generates a target ship state for a crossing encounter scenario.
        Returns:
            VesselState: The state of the generated target ship.
        """
        eps = self.epsilon if epsilon is None else epsilon
        rel_bearing = self._pick_relative_bearing(90, (67.5 * np.pi / 180)*eps)  # Crossing sector
        side = np.random.choice([0, 1])
        mean_bearing = 112.5 / 2
        rel_bearing = self._pick_relative_bearing(mean_bearing, mean_bearing*eps) + side * np.pi
        mean_speed = self.own_ship.state.v / 2
        rel_speed = np.random.uniform(0.1, mean_speed+mean_speed*eps)
        target_state = self.generate_target_ship(rel_bearing, rel_speed)
        return target_state
    
    def generate_random(self, eps=None) -> VesselState:
        """Generates a target ship state with random relative bearing and speed.
        
        Args:
            epsilon (float, optional): Variation parameter. Defaults to None.
        Returns:
            VesselState: The state of the generated target ship.
        """
        rel_bearing = np.random.uniform(0, 2 * np.pi)
        max_rel_speed = self.max_target_speed - self.own_ship.state.v 
        min_rel_speed = -self.own_ship.state.v
        rel_speed = np.random.uniform(min_rel_speed, max_rel_speed)
        target_state = self.generate_target_ship(rel_bearing, rel_speed)
        return target_state
    
    
    def generate_enc(self, enc_type: int = None, epsilon=None, max_iter: int = 100) -> Tuple[int, VesselState]:
        """Generates a target ship state for a specified encounter scenario.

        Args:
            enc_type (int): Encounter type
                1: Overtaking
                2: Overtaken
                3: Head-on
                4: Crossing
                5: Random
            epsilon (float, optional): Variation parameter.
        Returns:
            Tuple[int, VesselState]:
                int: The encounter type generated. If -1 then no valid encounter
                     was generated within max_iter.
                VesselState: The state of the generated target ship.
        """
        if enc_type is None:
            enc_type = np.random.choice([1, 2, 3, 4, 5])

        call = [
            self.generate_overtaking_enc,
            self.generate_overtaken_enc,
            self.generate_head_on_enc,
            self.generate_crossing_enc,
            self.generate_random,
        ]

        if not (1 <= enc_type <= 5):
            raise ValueError(
                "Invalid encounter type. Must be 1 (Overtaking), 2 (Overtaken), "
                "3 (Head-on), 4 (Crossing), or 5 (Random Scenario). "
                "Note that for type 5, the generated target ship may not correspond "
                "to any specific COLREG encounter type."
            )

        while max_iter > 0:

            target_state = call[enc_type - 1](epsilon)
            speed = target_state.v

            if speed > self.max_target_speed:
                max_iter -= 1
                continue

            if self.env is not None:
                x, y = self.env.to_local(target_state.lat, target_state.lon)
                if not self.env.is_navigable(x, y):   # avoid land/buffer
                    max_iter -= 1
                    continue

            return enc_type, target_state

        return -1, None
        

@dataclass
class Scenario:
    own_ship: Vessel
    other_vessels: List[Vessel]


class TrafficGenerator:
    """Generate standard COLREG encounter scenarios."""

    @staticmethod
    def head_on_encounter(
        own_speed: float = 10.0,
        other_speed: float = 10.0,
        lat_center: float = 43.5,
        lon_center: float = 16.4,
    ) -> Scenario:
        """
        Head-on encounter: vessels approaching on reciprocal courses.
        Own ship heading East (0°), other ship heading West (180°).
        """
        own_state = VesselState(
            lat=lat_center,
            lon=lon_center - 0.02,
            psi=0.0,  # East
            v=own_speed,
        )
        own_params = VesselParams(max_yaw_rate=np.radians(20))
        own_ship = Vessel(own_state, own_params)

        other_state = VesselState(
            lat=lat_center,
            lon=lon_center + 0.02,
            psi=np.pi,  # West
            v=other_speed,
        )
        other_params = VesselParams(max_yaw_rate=np.radians(20))
        other_ship = Vessel(other_state, other_params)

        return Scenario(own_ship, [other_ship])

    @staticmethod
    def crossing_encounter(
        own_speed: float = 10.0,
        other_speed: float = 10.0,
        lat_center: float = 43.5,
        lon_center: float = 16.4,
        other_on_starboard: bool = True,
    ) -> Scenario:
        """
        Crossing encounter: other vessel crosses own ship's path.

        Args:
            other_on_starboard: If True, other ship approaches from starboard
                               (own ship must give way per COLREG Rule 15)
        """
        own_state = VesselState(
            lat=lat_center,
            lon=lon_center,
            psi=np.pi / 2,  # North
            v=own_speed,
        )
        own_params = VesselParams(max_yaw_rate=np.radians(20))
        own_ship = Vessel(own_state, own_params)

        if other_on_starboard:
            other_lat, other_lon = lat_center - 0.015, lon_center + 0.03
            other_psi = np.pi  # West
        else:
            other_lat, other_lon = lat_center - 0.015, lon_center - 0.03
            other_psi = 0.0  # East

        other_state = VesselState(
            lat=other_lat, lon=other_lon, psi=other_psi, v=other_speed
        )
        other_params = VesselParams(max_yaw_rate=np.radians(20))
        other_ship = Vessel(other_state, other_params)

        return Scenario(own_ship, [other_ship])

    @staticmethod
    def overtaking_encounter(
        own_speed: float = 12.0,
        other_speed: float = 8.0,
        lat_center: float = 43.5,
        lon_center: float = 16.4,
    ) -> Scenario:
        """
        Overtaking: own ship (faster) approaches slower vessel from behind.
        Both heading North, own ship must keep clear (COLREG Rule 13).
        """
        other_state = VesselState(
            lat=lat_center + 0.01,
            lon=lon_center,
            psi=np.pi / 2,  # North
            v=other_speed,
        )
        other_params = VesselParams(max_yaw_rate=np.radians(20))
        other_ship = Vessel(other_state, other_params)

        own_state = VesselState(
            lat=lat_center,
            lon=lon_center,
            psi=np.pi / 2,  # North
            v=own_speed,
        )
        own_params = VesselParams(max_yaw_rate=np.radians(20))
        own_ship = Vessel(own_state, own_params)

        return Scenario(own_ship, [other_ship])
    
    @staticmethod
    def overtaken_encounter(
        own_speed: float = 8.0,
        other_speed: float = 12.0,
        lat_center: float = 43.5,
        lon_center: float = 16.4,
    ) -> Scenario:
        """
        Overtaken: own ship (slower) is approached from behind by faster vessel.
        Both heading North, other ship must keep clear (COLREG Rule 13).
        """
        other_state = VesselState(
            lat=lat_center - 0.01,
            lon=lon_center,
            psi=np.pi / 2,  # North
            v=other_speed,
        )
        other_params = VesselParams(max_yaw_rate=np.radians(20))
        other_ship = Vessel(other_state, other_params)

        own_state = VesselState(
            lat=lat_center,
            lon=lon_center,
            psi=np.pi / 2,  # North
            v=own_speed,
        )
        own_params = VesselParams(max_yaw_rate=np.radians(20))
        own_ship = Vessel(own_state, own_params)

        return Scenario(own_ship, [other_ship])
from dataclasses import dataclass
from typing import List
import numpy as np
from .vessel import Vessel, VesselState, VesselParams


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

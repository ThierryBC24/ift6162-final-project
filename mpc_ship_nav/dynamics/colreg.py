from dataclasses import dataclass
from typing import List
import numpy as np
from .vessel import Vessel, VesselState, VesselParams

D_COLLISION = 0.5 # nautical miles [nm]
D_COLREG = 3.0 # nautical miles [nm]

@dataclass
class EncouterType:
    UNDEFINED: int = 0
    OVERTAKING: int = 1
    OVERTAKEN: int = 2
    HEAD_ON: int = 3
    CROSSING: int = 4
    

def _compute_relative_bearing(own_ship: VesselState, target_ship: VesselState) -> float:
    """Compute the relative bearing from own ship to target ship

    Args:
        own_ship (VesselState): the controlled ship
        target_ship (VesselState): the encountered ship
    Returns:
        float: relative bearing in radians [0, 2pi)
    """
    delta_lon = target_ship.lon - own_ship.lon
    y = np.sin(delta_lon) * np.cos(target_ship.lat)
    x = (np.cos(own_ship.lat) * np.sin(target_ship.lat) -
         np.sin(own_ship.lat) * np.cos(target_ship.lat) * np.cos(delta_lon))
    true_bearing = np.arctan2(y, x) % (2 * np.pi)
    relative_bearing = (true_bearing - own_ship.psi) % (2 * np.pi)
    return relative_bearing

def _compute_relative_speed(own_ship: VesselState, target_ship: VesselState) -> float:
    """Compute the relative speed between two vessels

    Args:
        own_ship (VesselState): the controlled ship
        target_ship (VesselState): the encountered ship

    Returns:
        float: relative speed in knots
    """
    velocity_own = np.array([own_ship.v * np.cos(own_ship.psi), own_ship.v * np.sin(own_ship.psi)])
    velocity_target = np.array([target_ship.v * np.cos(target_ship.psi), target_ship.v * np.sin(target_ship.psi)])
    relative_velocity = velocity_target - velocity_own
    relative_speed = np.linalg.norm(relative_velocity)
    return relative_speed

def classify_encounter(own_ship: VesselState, target_ship: VesselState) -> int:
    """Classify the encounter type between two vessels based on their relative bearing

    Args:
        own_ship (VesselState): the controlled ship
        target_ship (VesselState): the encountered ship

    Returns:
        int: encounter type as defined in EncouterType
    """
    rel_bearing = _compute_relative_bearing(own_ship, target_ship)
    rel_speed = _compute_relative_speed(own_ship, target_ship)
    if (292.5 < rel_bearing or rel_bearing < 67.5) and rel_speed < own_ship.v:
        return EncouterType.OVERTAKING
    elif 112.5 < rel_bearing < 247.5 and rel_speed > 0:
        return EncouterType.OVERTAKEN
    elif rel_speed > own_ship.v:
        return EncouterType.HEAD_ON
    elif 247.5 <= rel_bearing  or rel_bearing <= 112.5 and rel_speed > 0:
        return EncouterType.CROSSING
    else:
        return EncouterType.UNDEFINED
    
        

def _compute_distance(own_ship: Vessel, target_ship: Vessel) -> float:
    """Compute the distance between two vessels in nautical miles

    Args:
        own_ship (Vessel): the controlled ship
        target_ship (Vessel): the encountered ship

    Returns:
        float: distance in nautical miles
    """
    d_lat = (target_ship.state.lat - own_ship.state.lat) * 60.0  # nm
    d_lon = (target_ship.state.lon - own_ship.state.lon) * 60.0  # nm
    return np.sqrt(d_lat**2 + d_lon**2)

def in_collision_zone(own_ship: Vessel, target_ship: Vessel, safety_radius: float = D_COLLISION) -> bool:
    """Check if a ship entered the collision zone

    Args:
        own_ship (Vessel): the controlled ship
        target_ship (Vessel): the encountered ship
        safety_radius (float, optional): object. Defaults to 0.05.

    Returns:
        bool: whether the target ship is in the collision zone
    """
    distance = _compute_distance(own_ship, target_ship)
    return distance <= safety_radius

def in_colreg_zone(own_ship: Vessel, target_ship: Vessel, colreg_radius: float = D_COLREG) -> bool:
    """Check if a ship entered the COLREG zone

    Args:
        own_ship (Vessel): the controlled ship
        target_ship (Vessel): the encountered ship
        colreg_radius (float, optional): object. Defaults to 3.0.

    Returns:
        bool: whether the target ship is in the COLREG zone
    """
    distance = _compute_distance(own_ship, target_ship)
    return distance <= colreg_radius
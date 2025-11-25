"""
COLREG compliance logic for target ships.

Target ships follow these rules explicitly, while the own ship (MPC-controlled)
avoids collisions implicitly through its optimization constraints.
"""

import numpy as np
from .vessel import Vessel

# Paper values (jmse-13-01246): dCollision=0.5 nm, dCOLREG=3.0 nm
NM = 1852.0
D_COLLISION = 0.5 * NM  # 926 m
D_COLREG = 3.0 * NM     # 5556 m


class COLREGLogic:
    """
    COLREG compliance logic for target ships (Rule 13/14/15).

    The target ship will generate a yaw rate command when it must give way;
    otherwise it maintains course (returns 0.0).
    """

    def __init__(self, collision_threshold: float = D_COLLISION):
        """
        Args:
            collision_threshold: Distance threshold for collision risk [m]
        """
        self.collision_threshold = collision_threshold

    def compute_target_control(self, target: Vessel, own_ship: Vessel) -> float:
        """
        Compute control for a target ship based on COLREG rules.

        Args:
            target: The target ship to control
            own_ship: The own ship (autonomous vessel)

        Returns:
            u: yaw rate command [rad/s]
        """
        if not self._risk_of_collision(target, own_ship):
            return 0.0

        encounter = self._classify_encounter(target, own_ship)

        if self._target_must_give_way(encounter, target, own_ship):
            sign = self._turn_direction(encounter, target, own_ship)
            return sign * target.params.max_yaw_rate

        return 0.0

    def _distance(self, ship1: Vessel, ship2: Vessel) -> float:
        """Euclidean distance between two vessels in meters."""
        dx = ship2.state.x - ship1.state.x
        dy = ship2.state.y - ship1.state.y
        return np.hypot(dx, dy)

    def _risk_of_collision(self, target: Vessel, own_ship: Vessel) -> bool:
        """
        Determine if there is a risk of collision.

        Simple distance check plus whether the relative velocity points toward
        the other vessel (dot product < 0).
        """
        distance = self._distance(target, own_ship)
        if distance > self.collision_threshold:
            return False

        dx = own_ship.state.x - target.state.x
        dy = own_ship.state.y - target.state.y

        vx_t = target.state.v * np.cos(target.state.psi)
        vy_t = target.state.v * np.sin(target.state.psi)
        vx_o = own_ship.state.v * np.cos(own_ship.state.psi)
        vy_o = own_ship.state.v * np.sin(own_ship.state.psi)

        dvx = vx_o - vx_t
        dvy = vy_o - vy_t

        dot_product = dvx * dx + dvy * dy
        return dot_product < 0

    def _classify_encounter(self, target: Vessel, own_ship: Vessel) -> str:
        """
        Classify the type of encounter according to COLREG.

        Returns:
            'head_on', 'crossing', 'overtaking', or 'none'
        """
        rel_bearing = self._relative_bearing(target, own_ship)

        heading_diff = abs(own_ship.state.psi - target.state.psi)
        heading_diff = min(heading_diff, 2 * np.pi - heading_diff)

        # Head-on: nearly reciprocal courses and own ship in target's forward cone
        if heading_diff > np.radians(170) and abs(rel_bearing) < np.radians(20):
            return "head_on"

        # Overtaking: target is faster and own ship is in target's stern sector
        rel_bearing_own = self._relative_bearing(own_ship, target)
        if target.state.v > own_ship.state.v and abs(rel_bearing_own) > np.radians(112.5):
            return "overtaking"

        # Crossing: intersecting paths with appreciable heading difference
        if np.radians(5) < heading_diff < np.radians(170):
            return "crossing"

        return "none"

    def _target_must_give_way(self, encounter: str, target: Vessel, own_ship: Vessel) -> bool:
        """Return True if the target ship must maneuver."""
        if encounter == "head_on":
            return True

        if encounter == "crossing":
            rel_bearing = self._relative_bearing(target, own_ship)
            # Starboard side = negative (clockwise) relative bearing in our convention
            return -np.radians(112.5) < rel_bearing <= 1e-6

        if encounter == "overtaking":
            rel_bearing_own = self._relative_bearing(own_ship, target)
            return target.state.v > own_ship.state.v and abs(rel_bearing_own) > np.radians(112.5)

        return False

    def _turn_direction(self, encounter: str, target: Vessel, own_ship: Vessel) -> int:
        """
        Determine turn direction for avoiding action.

        Returns +1 for port (left) turn, -1 for starboard (right) turn.
        """
        if encounter == "head_on":
            return -1  # Rule 14: turn starboard

        if encounter == "overtaking":
            return -1  # Rule 13: overtaking vessel keeps to starboard (in this demo)

        # crossing or fallback: turn away from own ship based on relative bearing
        rel_bearing = self._relative_bearing(target, own_ship)
        if rel_bearing >= 0:
            return -1  # own on port side -> turn starboard
        return +1      # own on starboard side -> turn port

    def _relative_bearing(self, observer: Vessel, target: Vessel) -> float:
        """
        Relative bearing from observer to target in radians, normalized to [-π, π].
        0 = dead ahead of observer; +π/2 = starboard; -π/2 = port; ±π = astern.
        """
        dx = target.state.x - observer.state.x
        dy = target.state.y - observer.state.y

        bearing_absolute = np.arctan2(dy, dx)
        relative = bearing_absolute - observer.state.psi
        return np.arctan2(np.sin(relative), np.cos(relative))

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import heapq
import math
import numpy as np
from .environment import ChartEnvironment


Coord = Tuple[float, float]
GridIndex = Tuple[int, int]


@dataclass
class PlannerConfig:
    use_theta_star: bool = True
    # factor to resample waypoints (max distance between points)
    max_waypoint_spacing: float = 500.0  # meters


class GlobalPlanner:
    """
    Grid-based planner over ChartEnvironment, using A* or Theta*.

    Conceptually corresponds to X2_route_planner.m + Section 2.2 (Theta* planner).
    """

    def __init__(self, env: ChartEnvironment, cfg: Optional[PlannerConfig] = None):
        self.env = env
        self.cfg = cfg or PlannerConfig()

        # Use region + grid_resolution from env.region_cfg
        self.res = env.region_cfg.grid_resolution

        # Build grid metadata (indices <-> world coordinates)
        self._init_grid()

    # ---------- public API ----------

    def plan(self, start_xy: Coord, goal_xy: Coord) -> List[Coord]:
        """
        Compute a waypoint route from start to goal, avoiding land/coastal buffer.

        Returns a list of (x, y) coordinates.
        """
        start_ij = self._xy_to_ij(start_xy)
        goal_ij = self._xy_to_ij(goal_xy)

        path_ij = self._theta_star(start_ij, goal_ij) if self.cfg.use_theta_star else self._a_star(start_ij, goal_ij)

        if not path_ij:
            raise RuntimeError("GlobalPlanner: no path found between start and goal.")

        # Convert grid indices back to (x, y) coordinates
        path_xy = [self._ij_to_xy(ij) for ij in path_ij]
        path_xy = self._resample_path(path_xy, self.cfg.max_waypoint_spacing)

        return path_xy

    # ---------- grid definition ----------

    def _init_grid(self) -> None:
        cfg = self.env.region_cfg
        # For now, derive grid in LOCAL coordinates by transforming region corners.
        lat_min, lat_max = cfg.lat_min, cfg.lat_max
        lon_min, lon_max = cfg.lon_min, cfg.lon_max

        x_min, y_min = self.env.to_local(lat_min, lon_min)
        x_max, y_max = self.env.to_local(lat_max, lon_max)

        # Ensure correct ordering
        self.x_min, self.x_max = min(x_min, x_max), max(x_min, x_max)
        self.y_min, self.y_max = min(y_min, y_max), max(y_min, y_max)

        self.nx = int(math.ceil((self.x_max - self.x_min) / self.res))
        self.ny = int(math.ceil((self.y_max - self.y_min) / self.res))

    def _xy_to_ij(self, xy: Coord) -> GridIndex:
        x, y = xy
        i = int((x - self.x_min) / self.res)
        j = int((y - self.y_min) / self.res)
        return i, j

    def _ij_to_xy(self, ij: GridIndex) -> Coord:
        i, j = ij
        x = self.x_min + (i + 0.5) * self.res
        y = self.y_min + (j + 0.5) * self.res
        return x, y

    def _valid_ij(self, ij: GridIndex) -> bool:
        i, j = ij
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return False
        x, y = self._ij_to_xy(ij)
        return self.env.is_navigable(x, y)

    # ---------- A* / Theta* ----------

    def _neighbors(self, ij: GridIndex):
        i, j = ij
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                nb = (i + di, j + dj)
                if self._valid_ij(nb):
                    yield nb

    def _dist(self, a: GridIndex, b: GridIndex) -> float:
        """
        Actual cost between two neighboring grid cells (Euclidean in world coords).
        """
        ax, ay = self._ij_to_xy(a)
        bx, by = self._ij_to_xy(b)
        return math.hypot(bx - ax, by - ay)
    
    def _line_of_sight(self, a: GridIndex, b: GridIndex) -> bool:
        """
        True if the straight line between grid cells a and b lies entirely
        in navigable space.

        """
        ax, ay = self._ij_to_xy(a)
        bx, by = self._ij_to_xy(b)

        # If the segment intersects land → no line of sight
        if self.env.line_intersects_land(ax, ay, bx, by):
            return False

        return True


    def _a_star(self, start: GridIndex, goal: GridIndex) -> List[GridIndex]:
        """
        Simple 8-connected A* on the occupancy grid.
        """
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {start: None}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nb in self._neighbors(current):
                tentative_g = g_score[current] + self._dist(current, nb)
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self._dist(nb, goal)
                    heapq.heappush(open_set, (f, nb))

        return []

    def _theta_star(self, start: GridIndex, goal: GridIndex) -> List[GridIndex]:
        """
        Theta* search:
        - Like A*, but allows the parent of a node to "see" neighbors directly
          via line-of-sight, which shortens and smooths paths.
        """
        open_set = []
        heapq.heappush(open_set, (0.0, start))

        # parent: store predecessor in the path; parent[start] = start
        parent = {start: start}
        g_score = {start: 0.0}

        while open_set:
            _, s = heapq.heappop(open_set)

            if s == goal:
                return self._reconstruct_path(parent, s)

            for s_next in self._neighbors(s):
                if s not in parent:
                    # This shouldn't happen, but guard anyway
                    parent[s] = s

                p = parent[s]  # parent of current node

                # Case 1: parent has direct line-of-sight to neighbor
                if self._line_of_sight(p, s_next):
                    # Try to update via parent
                    tentative_g = g_score[p] + self._dist(p, s_next)
                    if tentative_g < g_score.get(s_next, float("inf")):
                        parent[s_next] = p
                        g_score[s_next] = tentative_g
                        f = tentative_g + self._dist(s_next, goal)
                        heapq.heappush(open_set, (f, s_next))
                else:
                    # Case 2: fall back to standard A* update via s
                    tentative_g = g_score[s] + self._dist(s, s_next)
                    if tentative_g < g_score.get(s_next, float("inf")):
                        parent[s_next] = s
                        g_score[s_next] = tentative_g
                        f = tentative_g + self._dist(s_next, goal)
                        heapq.heappush(open_set, (f, s_next))

        # no path
        return []

    def _reconstruct_path(self, parent, current: GridIndex) -> List[GridIndex]:
        path = [current]
        while parent[current] != current:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    # ---------- post-processing ----------

    @staticmethod
    def _resample_path(path_xy: List[Coord], max_spacing: float) -> List[Coord]:
        """
        Insert intermediate waypoints so that consecutive points are
        at most max_spacing apart. This helps the route following / MPC layer.
        """
        if len(path_xy) < 2:
            return path_xy

        out = [path_xy[0]]
        for p0, p1 in zip(path_xy, path_xy[1:]):
            x0, y0 = p0
            x1, y1 = p1
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= max_spacing:
                out.append(p1)
                continue
            n_extra = int(seg_len // max_spacing)
            for k in range(1, n_extra + 1):
                alpha = k / (n_extra + 1)
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
                out.append((x, y))
            out.append(p1)
        return out

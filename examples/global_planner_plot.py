"""
Global planner demo (paper-scale, Adriatic region).

- You specify only (lat_start, lon_start) and (lat_goal, lon_goal).
- auto_bounds() builds a small bounding box appropriate for the paper's scale.
- We use GSHHG high-resolution coastline, Theta* global planner, and plot the path.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt

from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.charts.planner import GlobalPlanner, PlannerConfig


# ============================================================
# 1. USER INPUT: start / goal in LAT/LON
#    (This example is coastal Adriatic, near Split / islands)
# ============================================================
lat_start, lon_start = 43.45491, 15.97686
lat_goal,  lon_goal  = 43.09396, 17.16086


# ============================================================
# 2. Helpers: distance + paper-scale auto_bounds
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometers (rough but good enough)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def auto_bounds(lat1, lon1, lat2, lon2):
    """
    Paper-faithful auto-bounds:
    - small padding for coastal (~<150 km)
    - medium padding for regional (150–300 km)
    - reject trips that are too long for the paper's scale (>300 km)
    """

    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    print(f"[auto_bounds] trip distance ~{dist_km:.1f} km")

    # Case 1: local coastal navigation (< 150 km)
    if dist_km < 150.0:
        pad = 0.3      # degrees
        max_pad = 0.5  # sanity cap
        print("[auto_bounds] Using small coastal bounding box")

    # Case 2: regional navigation (150–300 km)
    elif dist_km < 300.0:
        pad = 0.5
        max_pad = 1.5
        print("[auto_bounds] Using medium regional bounding box")

    # Case 3: beyond paper-scale scope
    else:
        raise RuntimeError(
            f"[auto_bounds] Trip distance {dist_km:.1f} km exceeds the "
            "intended paper-scale navigation range. Split the route into "
            "smaller legs or use a different planner."
        )

    lat_min = min(lat1, lat2) - pad
    lat_max = max(lat1, lat2) + pad
    lon_min = min(lon1, lon2) - pad
    lon_max = max(lon1, lon2) + pad

    # # Extra guard: if something goes weird and the box is too large
    # if (lat_max - lat_min) > (2 * max_pad) or (lon_max - lon_min) > (2 * max_pad):
    #     raise RuntimeError("[auto_bounds] Computed bounds exceed safe limit.")

    print("Computed auto-bounds:")
    print(" lat:", lat_min, "→", lat_max)
    print(" lon:", lon_min, "→", lon_max)

    return lat_min, lat_max, lon_min, lon_max


# ============================================================
# 3. Build RegionConfig using auto_bounds
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COASTLINE_FILE = DATA_DIR / "GSHHS_h_L1.shp"

lat_min, lat_max, lon_min, lon_max = auto_bounds(
    lat_start, lon_start, lat_goal, lon_goal
)

cfg = RegionConfig(
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    grid_resolution=300.0,
    coastal_buffer=500.0,
    coastline_file=str(COASTLINE_FILE),
)


# ============================================================
# 4. Build environment + planner, compute route
# ============================================================
env = ChartEnvironment(cfg)
planner = GlobalPlanner(env, PlannerConfig(use_theta_star=True))

start_xy = env.to_local(lat_start, lon_start)
goal_xy  = env.to_local(lat_goal,  lon_goal)

print("start_xy:", start_xy)
print("goal_xy :", goal_xy)
print("start is_navigable:", env.is_navigable(*start_xy))
print("goal  is_navigable:", env.is_navigable(*goal_xy))
print("dist to shore start:", env.distance_to_shore(*start_xy))
print("dist to shore goal :", env.distance_to_shore(*goal_xy))

route = planner.plan(start_xy, goal_xy)
xs, ys = zip(*route)

print(f"Route found with {len(route)} waypoints.")


# ============================================================
# 5. Plot result
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

env.plot_base_map(ax)

# Plot main route polyline
ax.plot(xs, ys, "-", color="blue", linewidth=1.5, label="planned route")

# Plot waypoints along the route as markers-
ax.scatter(xs, ys, s=15, color="blue", zorder=3, label="waypoints")

# Label a subset of waypoints (start, goal, and every 50th)
for k, (x, y) in enumerate(route):
    if k == 0:
        label = "WP0 (start)"
    elif k == len(route) - 1:
        label = f"WP{len(route)-1} (goal)"
    elif k % 50 != 0:
        continue
    else:
        label = f"WP{k}"

    ax.text(
        x, y, label,
        fontsize=7,
        ha="left",
        va="bottom",
        color="black",
        zorder=4,
    )

# Highlight start and goal explicitly
ax.scatter([start_xy[0]], [start_xy[1]], c="green", s=40, zorder=4, label="start")
ax.scatter([goal_xy[0]],  [goal_xy[1]],  c="red",   s=40, zorder=4, label="goal")

ax.set_aspect("equal", "box")
ax.set_xlabel("x (m, local UTM)")
ax.set_ylabel("y (m, local UTM)")
ax.set_title("Theta* global route (paper-scale coastal example)")
ax.legend()
plt.tight_layout()
plt.show()

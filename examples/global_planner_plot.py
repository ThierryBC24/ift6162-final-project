import matplotlib.pyplot as plt
from pathlib import Path
from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.charts.planner import GlobalPlanner

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

cfg = RegionConfig(
    lat_min=43.0, lat_max=43.5,
    lon_min=16.0, lon_max=16.5,
    grid_resolution=500.0,
    coastal_buffer=100.0,
    coastline_file=str(DATA_DIR / "GSHHS_h_L1.shp"),
)

env = ChartEnvironment(cfg)
planner = GlobalPlanner(env)

lat_start, lon_start = 43.05, 16.25
lat_goal,  lon_goal  = 43.45, 16.45

start_xy = env.to_local(lat_start, lon_start)
goal_xy  = env.to_local(lat_goal,  lon_goal)
print("start_xy:", start_xy)
print("goal_xy:", goal_xy)
print("start is_navigable:", env.is_navigable(*start_xy))
print("goal  is_navigable:", env.is_navigable(*goal_xy))
print("dist to shore start:", env.distance_to_shore(*start_xy))
print("dist to shore goal :", env.distance_to_shore(*goal_xy))


route = planner.plan(start_xy, goal_xy)
xs, ys = zip(*route)

fig, ax = plt.subplots()
env.plot_base_map(ax)
ax.plot(xs, ys, "-o", markersize=2, label="planned route")
ax.scatter([start_xy[0]], [start_xy[1]], c="green", label="start")
ax.scatter([goal_xy[0]],  [goal_xy[1]],  c="red",   label="goal")
ax.set_aspect("equal", adjustable="box")
ax.legend()
plt.show()

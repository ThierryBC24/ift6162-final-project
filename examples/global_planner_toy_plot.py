import matplotlib.pyplot as plt

from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.charts.planner import GlobalPlanner

cfg = RegionConfig(
    lat_min=43.0, lat_max=43.5,
    lon_min=16.0, lon_max=16.5,
    grid_resolution=500.0,
    coastal_buffer=2000.0,
)

env = ChartEnvironment(cfg)
planner = GlobalPlanner(env)

start_xy = env.to_local(43.05, 16.05)
goal_xy  = env.to_local(43.45, 16.45)

route = planner.plan(start_xy, goal_xy)
xs, ys = zip(*route)

fig, ax = plt.subplots()

# Plot land
if not env.land_geometry.is_empty:
    try:
        x_land, y_land = env.land_geometry.exterior.xy
        ax.plot(x_land, y_land, color="black", linewidth=2, label="land boundary")
    except:
        pass

# Plot route
ax.plot(xs, ys, marker=".", linestyle="-", label="planned route")
ax.scatter([start_xy[0]], [start_xy[1]], c="green", label="start")
ax.scatter([goal_xy[0]],  [goal_xy[1]],  c="red",   label="goal")

ax.set_aspect("equal", adjustable="box")
ax.legend()
plt.show()

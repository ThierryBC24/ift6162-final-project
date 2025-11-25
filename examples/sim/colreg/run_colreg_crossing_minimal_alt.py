"""
Alternate crossing COLREG demo (blank map) with GIF output.

Own/traffic swapped vs base crossing demo; each takes the other's start and heading.
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from mpc_ship_nav.dynamics import COLREGLogic, D_COLLISION
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams
from mpc_ship_nav.sim.engine import SimConfig, SimLog


class HoldCourseController:
    def compute_control(self, t, own_ship, other_vessels, env=None):
        return 0.0


def main():
    # Own starts where traffic was, heading West, farther out
    own_state = VesselState(lat=0.0, lon=0.0, psi=np.pi, v=5.0, x=4000.0, y=0.0)
    own_params = VesselParams(max_yaw_rate=np.radians(20))
    own_ship = Vessel(own_state, own_params)

    # Traffic starts where own was, with own's heading (North)
    traf_state = VesselState(lat=0.0, lon=0.0, psi=np.pi / 2, v=5.0, x=2000.0, y=-1000.0)
    traf_params = VesselParams(max_yaw_rate=np.radians(20))
    traf_ship = Vessel(traf_state, traf_params)

    colreg = COLREGLogic(collision_threshold=D_COLLISION)
    controller = HoldCourseController()
    cfg = SimConfig(dt=1.0, t_final=1200.0, log_interval=1)

    log = SimLog()
    n_steps = int(cfg.t_final / cfg.dt)
    for k in range(n_steps + 1):
        t = k * cfg.dt
        if k % cfg.log_interval == 0:
            log.times.append(t)
            log.own_states.append(own_ship.state.__class__(**own_ship.state.__dict__))
            log.traffic_states.append([traf_ship.state.__class__(**traf_ship.state.__dict__)])

        u_own = controller.compute_control(t, own_ship, [traf_ship], None)
        u_traf = colreg.compute_target_control(traf_ship, own_ship)

        own_ship.step(u_own, cfg.dt)
        traf_ship.step(u_traf, cfg.dt)

    own_x = [s.x for s in log.own_states]
    own_y = [s.y for s in log.own_states]
    traf_x = [step[0].x for step in log.traffic_states]
    traf_y = [step[0].y for step in log.traffic_states]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Crossing COLREG demo (alt, blank map)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(-1000, 3000)
    ax.set_xlim(min(traf_x + own_x) - 100, max(traf_x + own_x) + 100)
    ax.grid(True)

    own_line, = ax.plot([], [], "-b", label="own (holds course)")
    traf_line, = ax.plot([], [], "--r", label="traffic (COLREG)")
    own_head, = ax.plot([], [], "bo", markersize=5)
    traf_head, = ax.plot([], [], "ro", markersize=5)
    ax.legend()

    def init():
        own_line.set_data([], [])
        traf_line.set_data([], [])
        own_head.set_data([], [])
        traf_head.set_data([], [])
        return own_line, traf_line, own_head, traf_head

    def update(frame):
        own_line.set_data(own_x[: frame + 1], own_y[: frame + 1])
        traf_line.set_data(traf_x[: frame + 1], traf_y[: frame + 1])
        own_head.set_data([own_x[frame]], [own_y[frame]])
        traf_head.set_data([traf_x[frame]], [traf_y[frame]])
        return own_line, traf_line, own_head, traf_head

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(own_x),
        init_func=init,
        blit=True,
        interval=50,
    )

    anim.save("colreg_crossing_minimal_alt.gif", fps=20)
    print("Saved colreg_crossing_minimal_alt.gif")
    plt.close(fig)


if __name__ == "__main__":
    main()

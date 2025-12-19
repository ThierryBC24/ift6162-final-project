import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch 
from typing import Optional, List, Tuple
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.sim.engine import SimLog
from mpc_ship_nav.sim.sim_possible_traj import SimulateTraj
from mpc_ship_nav.charts.planner import Coord 


class DummyHeadingController:
    """Very simple controller: hold current heading (no collision avoidance)."""

    def compute_control(self, t, own_ship, other_vessels, env):
        return 0.0  # no yaw rate change

def plot_trajectories(
    env: ChartEnvironment,
    log: SimLog,
    ax: Optional[plt.Axes] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    pad: float = 500.0,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    env.plot_base_map(ax)

    own_x = [s.x for s in log.own_states if s.x is not None]
    own_y = [s.y for s in log.own_states if s.y is not None]
    ax.plot(own_x, own_y, "-o", ms=2, label="own ship")

    all_x = list(own_x)
    all_y = list(own_y)

    if log.traffic_states:
        n_traffic = len(log.traffic_states[0])
        for idx in range(n_traffic):
            xs, ys = [], []
            for step_states in log.traffic_states:
                s = step_states[idx]
                if s.x is not None and s.y is not None:
                    xs.append(s.x); ys.append(s.y)
            ax.plot(xs, ys, "--", label=f"traffic {idx+1}")
            all_x.extend(xs); all_y.extend(ys)

    # --- bounds handling ---
    if bounds is None:
        # Prefer region corners if present (keeps map + traj consistent)
        cfg = getattr(env, "cfg", None) or getattr(env, "config", None)
        if cfg is not None and all(hasattr(cfg, k) for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            x1, y1 = env.to_local(cfg.lat_min, cfg.lon_min)
            x2, y2 = env.to_local(cfg.lat_max, cfg.lon_max)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
        else:
            x_min, x_max = min(all_x) - pad, max(all_x) + pad
            y_min, y_max = min(all_y) - pad, max(all_y) + pad
    else:
        x_min, x_max, y_min, y_max = bounds

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")
    ax.legend()
    ax.set_title("Ship trajectories")
    return ax



def animate_trajectories(
        env : ChartEnvironment, 
        log: SimLog, 
        waypoints: List[Coord],
        fps: int = 10, 
        save_path: str | None = None, 
        bounds: tuple[float, float, float, float] | None = None
):
    """
    Create an animation of ship trajectories on the chart.

    Parameters
    ----------
    env : ChartEnvironment
        Environment with land_geometry and to_geo/to_local helpers.
    log : SimLog
        Simulation log with:
          - times: list[float]
          - own_states: list[VesselState]  (fields x,y)
          - traffic_states: list[list[VesselState]]  (for each time step)
    fps : int
        Frames per second for the animation (controls playback speed).
    save_path : str or None
        If given, path to save the animation (e.g. 'sim.mp4').
        If None, just display interactively.
    """
    # --- Precompute positions ---
    waypoints_x, waypoints_y = zip(*waypoints)
    min_x, max_x = min(waypoints_x), max(waypoints_x)
    min_y, max_y = min(waypoints_y), max(waypoints_y)
    padding = 10000.0  # meters
    bounds = (min_x - padding, max_x + padding, min_y - padding, max_y + padding)
    
    
    own_x = np.array([s.x for s in log.own_states])
    own_y = np.array([s.y for s in log.own_states])
    own_psi = np.array([s.psi for s in log.own_states])
    n_frames = len(log.own_states)
    print(f"Animating {n_frames} frames at {fps} fps...")

    # own_hypothetical_trajectories = SimulateHypotheticalTraj(env, log, dump_zone=300, scale=100)
    # hypothetical_trajectories = own_hypothetical_trajectories.simulate_all_trajectories()
    # hypothetical_colors =  own_hypothetical_trajectories.color_all_trajectories_by_risk()
    # num_trajectories = len(hypothetical_trajectories[0])
    mpc_trajs = SimulateTraj(log)
    
    # Handle possible 0-traffic case
    n_traffic = len(log.traffic_states[0]) if log.traffic_states else 0
    traffic_x = []
    traffic_y = []
    for v_idx in range(n_traffic):
        traffic_x.append(np.array([step[v_idx].x for step in log.traffic_states]))
        traffic_y.append(np.array([step[v_idx].y for step in log.traffic_states]))

    # --- Figure and static background (land) ---
    fig, ax = plt.subplots(figsize=(20, 20))
    env.plot_base_map(ax)

    ax.set_title("Ship trajectories")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")

    # --- View window ---
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
    else:
        all_x = [*own_x]
        all_y = [*own_y]
        for tx in traffic_x:
            all_x.extend(tx)
        for ty in traffic_y:
            all_y.extend(ty)

        pad = 500.0  # fallback when no explicit bounds
        x_min, x_max = min(all_x) - pad, max(all_x) + pad
        y_min, y_max = min(all_y) - pad, max(all_y) + pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # --- Animation ---
    waypoints_line, = ax.plot(waypoints_x, waypoints_y, "-", color="grey", label="waypoints", zorder=5)
    
    own_line, = ax.plot([], [], "b-", label="own ship", zorder=3, linewidth=0.5)
    own_head, = ax.plot([], [], "bo", markersize=1, zorder=4, linewidth=0.5)
    # Arrow showing own-ship heading
    own_arrow = FancyArrowPatch(
        (0, 0), (0, 0),
        arrowstyle="->",
        color="blue",
        mutation_scale=10,
        zorder=4,
    )
    ax.add_patch(own_arrow)


    traffic_lines : List[plt.Line2D] = []
    traffic_heads : List[plt.Line2D] = []
    traffic_arrows : List[FancyArrowPatch] = []
    for i in range(n_traffic):
        line, = ax.plot([], [], "--", color="orange", label=f"traffic {i+1}", zorder=2)
        head, = ax.plot([], [], "bo", color="orange", markersize=1, zorder=3)
        traffic_arrow = FancyArrowPatch(
            (50, 0), (0, 0),
            arrowstyle="->",
            color="orange",
            mutation_scale=10,
            zorder=4,
        )
        ax.add_patch(traffic_arrow)
        traffic_arrows.append(traffic_arrow)
        
        traffic_lines.append(line)
        traffic_heads.append(head)

    hypothetical_lines : List[plt.Line2D] = []
    for traj_idx in range(mpc_trajs.n_candidate):
        line, = ax.plot([], [], "-", color='red',zorder=5, alpha=0.3)
        hypothetical_lines.append(line)
    ax.legend(loc="upper right")

    # --- Init and update functions for FuncAnimation ---
    def init():
        waypoints_line.set_data(waypoints_x, waypoints_y)
        own_line.set_data([], [])
        own_head.set_data([], [])
        for line, head in zip(traffic_lines, traffic_heads):
            line.set_data([], [])
            head.set_data([], [])
        for line in hypothetical_lines:
            line.set_data([], [])
        return [own_line, own_head, own_arrow, *traffic_lines, *traffic_heads, *traffic_arrows, *hypothetical_lines]


    def update(frame):
        # own ship up to this frame
        own_line.set_data(own_x[: frame + 1], own_y[: frame + 1])
        own_head.set_data([own_x[frame]], [own_y[frame]])

        # update heading arrow
        arrow_len = 2000.0  # meters
        x = own_x[frame]
        y = own_y[frame]
        psi = own_psi[frame]

        x2 = x + arrow_len * np.cos(psi)
        y2 = y + arrow_len * np.sin(psi)
        
        own_arrow.set_positions((x, y), (x2, y2))

        for i in range(n_traffic):
            # update heading arrow for traffic vessel
            tx = traffic_x[i]
            ty = traffic_y[i]

            # approximate heading using last two positions
            if frame == 0:
                psi_t = 0.0
            else:
                dx = tx[frame] - tx[frame - 1]
                dy = ty[frame] - ty[frame - 1]
                psi_t = np.arctan2(dy, dx)

            x_t = tx[frame]
            y_t = ty[frame]

            x2_t = x_t + arrow_len * np.cos(psi_t)
            y2_t = y_t + arrow_len * np.sin(psi_t)

            traffic_arrows[i].set_positions((x_t, y_t), (x2_t, y2_t))


        # each traffic vessel
        for i in range(n_traffic):
            tx = traffic_x[i]
            ty = traffic_y[i]
            traffic_lines[i].set_data(tx[max(0, frame - 5): frame + 1], ty[max(0, frame - 5): frame + 1])
            traffic_heads[i].set_data([tx[frame]], [ty[frame]])
        
        trjectories = mpc_trajs.get_traj_per_snapshot(frame)
        colors, u_armin = mpc_trajs.get_colors_per_snapshot(frame)
        for traj_idx in range(mpc_trajs.n_candidate):
            traj_x = trjectories[traj_idx][:, 0]
            traj_y = trjectories[traj_idx][:, 1]
            if traj_idx == u_armin:
                color = "blue"
            else:
                color = "green" if colors[traj_idx] else "red"
            hypothetical_lines[traj_idx].set_data(traj_x, traj_y)
            hypothetical_lines[traj_idx].set_color(color)


        return [own_line, own_head, own_arrow, *traffic_lines, *traffic_heads, *traffic_arrows, *hypothetical_lines]

    interval_ms = 1000.0 / fps
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if save_path is not None:
        # requires ffmpeg installed if saving as mp4
        anim.save(save_path, dpi=150,fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim
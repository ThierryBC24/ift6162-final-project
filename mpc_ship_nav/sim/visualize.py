import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch 
from typing import Optional, List
from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.traffic import TrafficGenerator
from mpc_ship_nav.sim.engine import Simulator, SimConfig, SimLog
from mpc_ship_nav.sim.sim_possible_traj import SimulateHypotheticalTraj


class DummyHeadingController:
    """Very simple controller: hold current heading (no collision avoidance)."""

    def compute_control(self, t, own_ship, other_vessels, env):
        return 0.0  # no yaw rate change

def plot_trajectories(
    env: ChartEnvironment,
    log: SimLog,
    ax: Optional[plt.Axes] = None,
):
    """Plot own ship + traffic trajectories on the chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Base map (land)
    env.plot_base_map(ax)

    # Own ship trajectory
    own_x = [s.x for s in log.own_states if s.x is not None]
    own_y = [s.y for s in log.own_states if s.y is not None]
    ax.plot(own_x, own_y, "-o", ms=2, label="own ship")

    # Traffic trajectories
    if log.traffic_states:
        n_traffic = len(log.traffic_states[0])
        for idx in range(n_traffic):
            xs = []
            ys = []
            for step_states in log.traffic_states:
                s = step_states[idx]
                if s.x is not None and s.y is not None:
                    xs.append(s.x)
                    ys.append(s.y)
            ax.plot(xs, ys, "--", label=f"traffic {idx+1}")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")
    ax.legend()
    ax.set_title("Ship trajectories")

    return ax


def animate_trajectories(
        env : ChartEnvironment, 
        log: SimLog, 
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
    own_x = np.array([s.x for s in log.own_states])
    own_y = np.array([s.y for s in log.own_states])
    own_psi = np.array([s.psi for s in log.own_states])
    n_frames = len(log.own_states)
    print(f"Animating {n_frames} frames at {fps} fps...")

    own_hypothetical_trajectories = SimulateHypotheticalTraj(env, log)
    hypothetical_trajectories = own_hypothetical_trajectories.simulate_all_trajectories()
    hypothetical_colors =  own_hypothetical_trajectories.color_all_trajectories_by_risk()
    num_trajectories = len(hypothetical_trajectories[0])
    
    # Handle possible 0-traffic case
    n_traffic = len(log.traffic_states[0]) if log.traffic_states else 0
    traffic_x = []
    traffic_y = []
    for v_idx in range(n_traffic):
        traffic_x.append(np.array([step[v_idx].x for step in log.traffic_states]))
        traffic_y.append(np.array([step[v_idx].y for step in log.traffic_states]))

    # --- Figure and static background (land) ---
    fig, ax = plt.subplots(figsize=(7, 3))
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
        line, = ax.plot([], [], "--", color="orange", label="traffic 1" if i == 0 else None, zorder=2)
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
    for traj_idx in range(num_trajectories):
        line, = ax.plot([], [], "-", color='red',zorder=5, alpha=0.3)
        hypothetical_lines.append(line)
    ax.legend(loc="upper right")

    # --- Init and update functions for FuncAnimation ---
    def init():
        own_line.set_data([], [])
        own_head.set_data([], [])
        for line, head in zip(traffic_lines, traffic_heads):
            line.set_data([], [])
            head.set_data([], [])
        for line in hypothetical_lines:
            line.set_data([], [])
        return [own_line, own_head, own_arrow, *traffic_lines, *traffic_heads, *hypothetical_lines]


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
            traffic_lines[i].set_data(tx[max(0, frame - 20): frame + 1], ty[max(0, frame - 20): frame + 1])
            traffic_heads[i].set_data([tx[frame]], [ty[frame]])
        
        trjectories = hypothetical_trajectories[frame]
        for traj_idx in range(num_trajectories):
            traj_x, traj_y = trjectories[traj_idx]
            color = hypothetical_colors[frame][traj_idx]
            hypothetical_lines[traj_idx].set_data(traj_x, traj_y)
            hypothetical_lines[traj_idx].set_color(color)


        return [own_line, own_head, own_arrow, *traffic_lines, *traffic_heads, *hypothetical_lines]

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
        anim.save(save_path, fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim
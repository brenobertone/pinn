import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

from pinn.problems_definitions import Problem


def animate_problem(problem: Problem, steps: int = 200, hash_id: str = "default") -> None:
    model = problem.net

    x_plot = np.linspace(problem.x_bounds[0], problem.x_bounds[1], 100)
    y_plot = np.linspace(problem.y_bounds[0], problem.y_bounds[1], 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    t_vals = np.linspace(problem.t_bounds[0], problem.t_bounds[1], steps)

    u_vals_list = []
    for t in t_vals:
        T = np.full_like(X, t)
        xyt_plot = torch.tensor(
            np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T, dtype=torch.float32
        )
        with torch.no_grad():
            u = model(xyt_plot).cpu().numpy().reshape(X.shape)
        u_vals_list.append(u)
    u_vals = np.stack(u_vals_list)
    u_min, u_max = np.min(u_vals), np.max(u_vals)

    fig = plt.figure(figsize=(12, 6))
    ax3d: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    contour = ax2d.contourf(X, Y, u_vals[0], levels=50, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax2d)  # create colorbar once

    def update(frame: int):
        ax3d.clear()
        ax2d.clear()

        # 3D Surface
        ax3d.plot_surface(X, Y, u_vals[frame], cmap="viridis")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("u")
        ax3d.set_zlim(u_min, u_max)
        ax3d.view_init(elev=30, azim=-135)
        ax3d.set_title(f"{problem.name} - 3D - t = {t_vals[frame]:.2f}")

        # 2D Contour
        contour = ax2d.contourf(X, Y, u_vals[frame], levels=50, cmap="viridis")
        ax2d.set_xlabel("x")
        ax2d.set_ylabel("y")
        ax2d.set_title(f"{problem.name} - Contour - t = {t_vals[frame]:.2f}")

        # Optional: maintain axis orientation
        if problem.x_orientation == "crescent":
            ax3d.set_xlim(problem.x_bounds[0], problem.x_bounds[1])
            ax2d.set_xlim(problem.x_bounds[0], problem.x_bounds[1])
        else:
            ax3d.set_xlim(problem.x_bounds[1], problem.x_bounds[0])
            ax2d.set_xlim(problem.x_bounds[1], problem.x_bounds[0])

        if problem.y_orientation == "crescent":
            ax3d.set_ylim(problem.y_bounds[0], problem.y_bounds[1])
            ax2d.set_ylim(problem.y_bounds[0], problem.y_bounds[1])
        else:
            ax3d.set_ylim(problem.y_bounds[1], problem.y_bounds[0])
            ax2d.set_ylim(problem.y_bounds[1], problem.y_bounds[0])

        return ax3d, ax2d

    anim = FuncAnimation(fig, update, frames=len(t_vals), blit=False)

    save_path = Path("videos")
    save_path.mkdir(exist_ok=True, parents=True)
    anim.save(save_path / f"animate_problem_{hash_id}.mp4", writer=FFMpegWriter(fps=15))

    plt.close(fig)

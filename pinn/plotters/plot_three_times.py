import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

from pinn.problems_definitions import Problem


def plot_three_times(problem: Problem, steps: int = 50) -> plt.Figure:
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

    times_idx = [0, len(t_vals) // 2, len(t_vals) - 1]
    time_labels = ["initial", "middle", "final"]

    fig = plt.figure(figsize=(18, 10))  # 3 columns, 2 rows
    for i, (idx, label) in enumerate(zip(times_idx, time_labels)):
        t = t_vals[idx]
        title = f"t = {t:.2f}"

        # --- 3D Surface Plot ---
        ax3d: Axes3D = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax3d.plot_surface(X, Y, u_vals[idx], cmap="viridis")
        ax3d.set_title(f"{problem.name} - {label}")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("u")
        ax3d.set_zlim(u_min, u_max)
        ax3d.view_init(elev=30, azim=-135)

        if problem.x_orientation == "crescent":
            ax3d.set_xlim(problem.x_bounds[0], problem.x_bounds[1])
        else:
            ax3d.set_xlim(problem.x_bounds[1], problem.x_bounds[0])

        if problem.y_orientation == "crescent":
            ax3d.set_ylim(problem.y_bounds[0], problem.y_bounds[1])
        else:
            ax3d.set_ylim(problem.y_bounds[1], problem.y_bounds[0])

        # --- Contour Plot ---
        ax2d = fig.add_subplot(2, 3, i + 4)
        contour = ax2d.contourf(X, Y, u_vals[idx], levels=50, cmap="viridis")
        plt.colorbar(contour, ax=ax2d)
        ax2d.set_title(f"{problem.name} - Contour - {label}")
        ax2d.set_xlabel("x")
        ax2d.set_ylabel("y")

        if problem.x_orientation == "crescent":
            ax2d.set_xlim(problem.x_bounds[0], problem.x_bounds[1])
        else:
            ax2d.set_xlim(problem.x_bounds[1], problem.x_bounds[0])

        if problem.y_orientation == "crescent":
            ax2d.set_ylim(problem.y_bounds[0], problem.y_bounds[1])
        else:
            ax2d.set_ylim(problem.y_bounds[1], problem.y_bounds[0])

    plt.tight_layout()
    return fig

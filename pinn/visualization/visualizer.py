from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation

from pinn.core.problems import Problem1D, Problem2D
from pinn.experiments.tracker import ExperimentTracker


class Visualizer:
    def __init__(self, experiment_ids: list[str], results_dir: str = "results"):
        self.tracker = ExperimentTracker(results_dir)
        self.experiment_ids = experiment_ids
        self.results_dir = Path(results_dir)

    def plot_losses(self, save_path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_id in self.experiment_ids:
            plot_path = self.results_dir / f"plot_{exp_id}.png"
            if not plot_path.exists():
                print(f"Plot not found for {exp_id}")
                continue

        df = self.tracker.load_experiments()
        df_filtered = df[df["exp_id"].isin(self.experiment_ids)]

        for _, row in df_filtered.iterrows():
            label = f"{row['problem_name']} - {row['residual_method']}"
            ax.scatter(
                row["exp_id"],
                row["final_loss"],
                label=label,
                s=100,
                alpha=0.7,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Experiment ID")
        ax.set_ylabel("Final Loss (log scale)")
        ax.set_title("Experiment Comparison")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def comparison_table(self) -> pd.DataFrame:
        df = self.tracker.load_experiments()
        df_filtered = df[df["exp_id"].isin(self.experiment_ids)]

        columns = [
            "exp_id",
            "problem_name",
            "residual_method",
            "network_layers",
            "final_loss",
            "training_time",
        ]
        return df_filtered[columns].sort_values("final_loss")

    def animate_solution_1d(
        self, exp_id: str, problem: Problem1D, steps: int = 200
    ) -> None:
        df = self.tracker.load_experiments()
        record = df[df["exp_id"] == exp_id].iloc[0]

        from pinn.core.architectures import NetworkConfig

        net_config = NetworkConfig(
            record["network_layers"],
            record["network_activation"],
            n_inputs=2,
            n_outputs=1,
        )
        model = self.tracker.load_model(exp_id, net_config)

        x_plot = np.linspace(problem.x_bounds[0], problem.x_bounds[1], 200)
        t_vals = np.linspace(problem.t_bounds[0], problem.t_bounds[1], steps)

        u_vals_list = []
        for t in t_vals:
            T = np.full_like(x_plot, t)
            xt_plot = torch.tensor(
                np.vstack([x_plot, T]).T, dtype=torch.float32
            )
            with torch.no_grad():
                u = model(xt_plot).cpu().numpy().flatten()
            u_vals_list.append(u)

        u_vals = np.array(u_vals_list)
        u_min, u_max = np.min(u_vals), np.max(u_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        (line,) = ax.plot(x_plot, u_vals[0])
        ax.set_xlim(problem.x_bounds)
        ax.set_ylim(u_min - 0.1, u_max + 0.1)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True)

        def update(frame):
            line.set_ydata(u_vals[frame])
            ax.set_title(f"{problem.name} - t = {t_vals[frame]:.3f}")
            return (line,)

        anim = FuncAnimation(fig, update, frames=len(t_vals), blit=True)

        save_path = self.results_dir / "videos"
        save_path.mkdir(exist_ok=True, parents=True)
        anim.save(
            save_path / f"animate_1d_{exp_id}.mp4",
            writer=FFMpegWriter(fps=15),
        )
        plt.close(fig)
        print(f"Saved animation to {save_path / f'animate_1d_{exp_id}.mp4'}")

    def animate_solution_2d(
        self, exp_id: str, problem: Problem2D, steps: int = 200
    ) -> None:
        df = self.tracker.load_experiments()
        record = df[df["exp_id"] == exp_id].iloc[0]

        from pinn.core.architectures import NetworkConfig

        net_config = NetworkConfig(
            record["network_layers"],
            record["network_activation"],
            n_inputs=3,
            n_outputs=1,
        )
        model = self.tracker.load_model(exp_id, net_config)

        x_plot = np.linspace(problem.x_bounds[0], problem.x_bounds[1], 100)
        y_plot = np.linspace(problem.y_bounds[0], problem.y_bounds[1], 100)
        X, Y = np.meshgrid(x_plot, y_plot)
        t_vals = np.linspace(problem.t_bounds[0], problem.t_bounds[1], steps)

        u_vals_list = []
        for t in t_vals:
            T = np.full_like(X, t)
            xyt_plot = torch.tensor(
                np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T,
                dtype=torch.float32,
            )
            with torch.no_grad():
                u = model(xyt_plot).cpu().numpy().reshape(X.shape)
            u_vals_list.append(u)

        u_vals = np.stack(u_vals_list)
        u_min, u_max = np.min(u_vals), np.max(u_vals)

        fig = plt.figure(figsize=(12, 6))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        def update(frame: int):
            ax3d.clear()
            ax2d.clear()

            ax3d.plot_surface(
                X, Y, u_vals[frame], cmap="viridis", vmin=u_min, vmax=u_max
            )
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("u")
            ax3d.set_zlim(u_min, u_max)
            ax3d.view_init(elev=30, azim=-135)
            ax3d.set_title(f"{problem.name} - 3D - t = {t_vals[frame]:.2f}")

            ax2d.contourf(
                X,
                Y,
                u_vals[frame],
                levels=50,
                cmap="viridis",
                vmin=u_min,
                vmax=u_max,
            )
            ax2d.set_xlabel("x")
            ax2d.set_ylabel("y")
            ax2d.set_title(
                f"{problem.name} - Contour - t = {t_vals[frame]:.2f}"
            )

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

        save_path = self.results_dir / "videos"
        save_path.mkdir(exist_ok=True, parents=True)
        anim.save(
            save_path / f"animate_2d_{exp_id}.mp4",
            writer=FFMpegWriter(fps=15),
        )
        plt.close(fig)
        print(f"Saved animation to {save_path / f'animate_2d_{exp_id}.mp4'}")

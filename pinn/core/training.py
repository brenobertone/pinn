import time
from typing import Callable, Literal

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.figure import Figure

from pinn.core.problems import Problem, Problem1D, Problem2D
from pinn.core.residuals import (
    residual_1d_autograd,
    residual_1d_mm2,
    residual_2d_autograd,
    residual_2d_mm2,
    residual_2d_mm3,
    residual_2d_uno,
)

matplotlib.use("Agg")

PADDING = 2

device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class Config:
    def __init__(
        self,
        epsilon: float,
        n_points: int,
        epochs: int,
        residual_method: Literal["autograd", "mm2", "mm3", "uno"] = "autograd",
        optimizer: Literal["adam", "adamw", "rmsprop"] = "adamw",
        learning_rate: float = 1e-3,
    ):
        self.epsilon = epsilon
        self.n_points = n_points
        self.epochs = epochs
        self.residual_method = residual_method
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def get_residual_fn(self, problem: Problem) -> Callable:
        if isinstance(problem, Problem1D):
            residual_map = {
                "autograd": residual_1d_autograd,
                "mm2": residual_1d_mm2,
            }
            if self.residual_method not in residual_map:
                raise ValueError(
                    f"Residual method {self.residual_method} not supported for 1D problems"
                )
            fn = residual_map[self.residual_method]
        elif isinstance(problem, Problem2D):
            residual_map = {
                "autograd": residual_2d_autograd,
                "mm2": residual_2d_mm2,
                "mm3": residual_2d_mm3,
                "uno": residual_2d_uno,
            }
            fn = residual_map[self.residual_method]
        else:
            raise ValueError(f"Unknown problem type: {type(problem)}")

        return lambda model, xyt: fn(model, problem, xyt, self.epsilon)


def uniform_mesh_1d(
    n_points: int, x_bounds: tuple[float, float], t_bounds: tuple[float, float], device
) -> torch.Tensor:
    Nx = Nt = round(n_points ** (1 / 2))
    x_min, x_max = x_bounds
    t_min, t_max = t_bounds

    Nx_pad, Nt_pad = Nx + 2 * PADDING, Nt + 2 * PADDING

    x = torch.linspace(x_min, x_max, Nx_pad, device=device)
    t = torch.linspace(t_min, t_max, Nt_pad, device=device)

    X, T = torch.meshgrid(x, t, indexing="ij")
    return torch.stack([X, T], dim=-1)


def uniform_mesh_2d(
    n_points: int, bounds: list[tuple[float, float]], device
) -> torch.Tensor:
    Nx = Ny = Nt = round(n_points ** (1 / 3))
    (x_min, x_max), (y_min, y_max), (t_min, t_max) = bounds

    Nx_pad, Ny_pad, Nt_pad = (
        Nx + 2 * PADDING,
        Ny + 2 * PADDING,
        Nt + 2 * PADDING,
    )

    x = torch.linspace(x_min, x_max, Nx_pad, device=device)
    y = torch.linspace(y_min, y_max, Ny_pad, device=device)
    t = torch.linspace(t_min, t_max, Nt_pad, device=device)

    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    return torch.stack([X, Y, T], dim=-1)


def train(
    problem: Problem, model: nn.Module, config: Config
) -> tuple[nn.Module, Figure, dict]:
    print(f"Training {problem.name}...")

    model.to(device)

    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    optimizer = optimizer_map[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    if isinstance(problem, Problem1D):
        coords_f = uniform_mesh_1d(
            config.n_points,
            problem.x_bounds,
            problem.t_bounds,
            device,
        )
        Nx = Nt = round(config.n_points ** (1 / 2))
        x_mask = slice(PADDING, PADDING + Nx)
        t_mask = slice(PADDING, PADDING + Nt)
        coords_inner = coords_f[x_mask, t_mask, :].reshape(-1, 2)

        x_ic = coords_inner[:, 0:1]
        t_ic = torch.zeros_like(x_ic, device=device)
        coords_ic = torch.cat([x_ic, t_ic], dim=1)
        u0 = problem.initial_condition(x_ic)

    elif isinstance(problem, Problem2D):
        coords_f = uniform_mesh_2d(
            config.n_points,
            [
                problem.x_bounds,
                problem.y_bounds,
                problem.t_bounds,
            ],
            device,
        )
        Nx = Ny = Nt = round(config.n_points ** (1 / 3))
        x_mask = slice(PADDING, PADDING + Nx)
        y_mask = slice(PADDING, PADDING + Ny)
        t_mask = slice(PADDING, PADDING + Nt)
        coords_inner = coords_f[x_mask, y_mask, t_mask, :].reshape(-1, 3)

        x_ic = coords_inner[:, 0:1]
        y_ic = coords_inner[:, 1:2]
        t_ic = torch.zeros_like(x_ic, device=device)
        coords_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
        u0 = problem.initial_condition(x_ic, y_ic)
    else:
        raise ValueError(f"Unknown problem type: {type(problem)}")

    residual_fn = config.get_residual_fn(problem)

    loss_history = []
    epochs_measured = []
    start_training = time.time()

    for epoch in range(config.epochs):
        f = residual_fn(model, coords_f)
        loss_f = torch.mean(f**2)

        u_pred_ic = model(coords_ic)
        loss_ic = torch.mean((u_pred_ic - u0) ** 2)

        loss = loss_f + loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5000 == 0:
            elapsed = time.time() - start_training
            print(
                f"Problem {problem.name}:"
                f"Epoch {epoch}: Loss = {loss.item():.5e}, "
                f"loss_f = {loss_f.item():.5e}, "
                f"loss_ic = {loss_ic.item():.5e}, "
                f"elapsed time = {elapsed:.2f}s"
            )
        loss_history.append(loss.item())
        epochs_measured.append(epoch)

    total_time = time.time() - start_training
    print(f"Total training time: {total_time:.2f} seconds")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs_measured, loss_history, label="Total Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(
        f"{problem.name}'s Training Loss History."
        + f" Elapsed time: {total_time:.2e} seconds"
    )
    plt.legend()
    plt.grid(True)

    metrics = {
        "final_loss": loss_history[-1],
        "final_loss_f": loss_f.item(),
        "final_loss_ic": loss_ic.item(),
        "training_time": total_time,
    }

    return model.to("cpu"), fig, metrics

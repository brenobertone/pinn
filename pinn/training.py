import time
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .problems_definitions import PINN, Problem

matplotlib.use("Agg")

PADDING = 10


class Config:
    def __init__(
        self,
        epsilon: float,
        n_points: int,
        epochs: int,
        residual: Callable,
    ):
        self.epsilon = epsilon
        self.n_points = n_points
        self.epochs = epochs
        self.residual = lambda model, problem, xyt: residual(
            model, problem, xyt, self.epsilon
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uniform_mesh(
    n_points: int, 
    bounds: list[tuple[float, float]],
) -> torch.Tensor:
    """
    Generate a structured uniform mesh in [x,y,t].
    n_points: (Nx, Ny, Nt)
    bounds: [(x_min, x_max), (y_min, y_max), (t_min, t_max)]
    """
    Nx = Ny = Nt = round(n_points**(1/3))
    (x_min, x_max), (y_min, y_max), (t_min, t_max) = bounds

    Nx_pad, Ny_pad, Nt_pad = Nx + 2 * PADDING, Ny + 2 * PADDING, Nt + 2 * PADDING

    x = torch.linspace(x_min, x_max, Nx_pad, device=device)
    y = torch.linspace(y_min, y_max, Ny_pad, device=device)
    t = torch.linspace(t_min, t_max, Nt_pad, device=device)

    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    return torch.stack([X, Y, T], dim=-1)  # shape (Nx,Ny,Nt,3)


def train(problem: Problem, config: Config) -> tuple[PINN, Figure]:
    print(f"Training {problem.name}...")

    model = problem.net
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=500,
        threshold=1e-5,
        cooldown=50,
        min_lr=1e-6,
    )

    xyt_f = uniform_mesh(
        config.n_points,
        [
            (problem.x_bounds[0], problem.x_bounds[1]),
            (problem.y_bounds[0], problem.y_bounds[1]),
            (problem.t_bounds[0], problem.t_bounds[1])
        ]
    )

    Nx = Ny = Nt = round(config.n_points**(1/3))
    x_mask = slice(PADDING, PADDING + Nx)
    y_mask = slice(PADDING, PADDING + Ny)
    t_mask = slice(PADDING, PADDING + Nt)

    xyt_inner = xyt_f[x_mask, y_mask, t_mask, :].reshape(-1, 3)

    x_ic = xyt_inner.reshape(-1, 3)[:, 0:1]
    y_ic = xyt_inner.reshape(-1, 3)[:, 1:2]
    t_ic = torch.zeros_like(x_ic, device=device)
    xyt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)

    u0 = problem.initial_condition(x_ic, y_ic)

    loss_history = []
    epochs_measured = []
    start_training = time.time()
    for epoch in range(config.epochs):

        f = config.residual(model, problem, xyt_f)
        loss_f = torch.mean(f**2)

        u_pred_ic = model(xyt_ic)
        loss_ic = torch.mean((u_pred_ic - u0) ** 2)

        loss = loss_f + loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss_f.item())

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

    return model.to("cpu"), fig

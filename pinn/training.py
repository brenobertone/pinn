import time
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
import numpy as np

from .problems_definitions import PINN, Problem

matplotlib.use("Agg")


class Config:
    def __init__(
        self,
        epsilon: float,
        delta: float,
        n_internal: int,
        n_initial_condition: int,
        epochs: int,
        residual: Callable,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.n_internal = n_internal
        self.n_initial_condition = n_initial_condition
        self.epochs = epochs
        self.residual = lambda model, problem, xyt: residual(
            model, problem, xyt, self.epsilon, self.delta
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latin_hypercube(n: int, d: int) -> np.ndarray:
    """Generate an n x d Latin Hypercube sample in [0,1]^d."""
    rng = np.random.default_rng()
    cut = np.linspace(0, 1, n+1)

    u = rng.uniform(size=(n, d))
    a = cut[:n]
    b = cut[1:n+1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)

    for j in range(d):
        order = rng.permutation(n)
        H[:, j] = rdpoints[order, j]

    return H

def scale_samples(samples: np.ndarray, lower: list[float], upper: list[float]) -> np.ndarray:
    lower, upper = np.array(lower), np.array(upper)
    return lower + samples * (upper - lower)



def train(problem: Problem, config: Config) -> tuple[PINN, Figure]:
    print(f"Training {problem.name}...")

    model = problem.net
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    # Internal residual points (x,y,t)
    sample_f = latin_hypercube(config.n_internal, d=3)
    xyt_f_np = scale_samples(
        sample_f,
        [problem.x_bounds[0], problem.y_bounds[0], problem.t_bounds[0]],
        [problem.x_bounds[1], problem.y_bounds[1], problem.t_bounds[1]],
    )
    xyt_f = torch.tensor(xyt_f_np, dtype=torch.float32, device=device)

    # Initial condition (x,y)
    sample_ic = latin_hypercube(config.n_initial_condition, d=2)
    xy_ic_np = scale_samples(
        sample_ic,
        [problem.x_bounds[0], problem.y_bounds[0]],
        [problem.x_bounds[1], problem.y_bounds[1]],
    )
    xy_ic = torch.tensor(xy_ic_np, dtype=torch.float32, device=device)

    x_ic = torch.tensor(xy_ic_np[:, 0:1], dtype=torch.float32, device=device)
    y_ic = torch.tensor(xy_ic_np[:, 1:2], dtype=torch.float32, device=device)
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

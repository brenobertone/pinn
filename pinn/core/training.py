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
        sampling_method: Literal["uniform", "latin_hypercube"] = "uniform",
        snr_compute_interval: int = 250,
        snr_batches: int = 100,
        rba_enabled: bool = False,
        rba_eta: float = 0.1,
    ):
        self.epsilon = epsilon
        self.n_points = n_points
        self.epochs = epochs
        self.residual_method = residual_method
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.sampling_method = sampling_method
        self.snr_compute_interval = snr_compute_interval
        self.snr_batches = snr_batches
        self.rba_enabled = rba_enabled
        self.rba_eta = rba_eta

        if (
            self.sampling_method == "latin_hypercube"
            and self.residual_method != "autograd"
        ):
            raise ValueError(
                "Latin Hypercube Sampling is only supported with the 'autograd' residual method."
            )

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


def lhs_samples(
    n_samples: int,
    d_dimensions: int,
    bounds: list[tuple[float, float]],
    device,
) -> torch.Tensor:
    """Native PyTorch implementation of Latin Hypercube Sampling."""
    # Create the intervals
    probs = torch.rand(n_samples, d_dimensions, device=device)
    # Generate a random permutation for each dimension
    perms = torch.stack(
        [torch.randperm(n_samples, device=device) for _ in range(d_dimensions)],
        dim=1,
    )
    # Combine to get LHS in [0, 1]^d
    samples = (perms.float() + probs) / n_samples

    # Scale to bounds
    for d in range(d_dimensions):
        min_val, max_val = bounds[d]
        samples[:, d] = samples[:, d] * (max_val - min_val) + min_val

    return samples


def uniform_mesh_1d(
    n_points: int,
    x_bounds: tuple[float, float],
    t_bounds: tuple[float, float],
    device,
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
    problem: Problem, model: nn.Module, config: Config, exp_id: str = None
) -> tuple[nn.Module, Figure, dict]:
    print(f"Training {problem.name}...")

    model.to(device)
    # try:
    #     model = torch.compile(model)
    #     print("Model compiled successfully using torch.compile")
    # except Exception as e:
    #     print(f"Warning: torch.compile failed ({e}). Proceeding without compilation.")

    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    optimizer = optimizer_map[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    if isinstance(problem, Problem1D):
        if config.sampling_method == "uniform":
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
        else:
            coords_f = lhs_samples(
                config.n_points,
                2,
                [problem.x_bounds, problem.t_bounds],
                device,
            )
            n_ic = round(config.n_points ** (1 / 2))
            x_ic = lhs_samples(n_ic, 1, [problem.x_bounds], device)
            t_ic = torch.zeros_like(x_ic, device=device)
            coords_ic = torch.cat([x_ic, t_ic], dim=1)
            u0 = problem.initial_condition(x_ic)

    elif isinstance(problem, Problem2D):
        if config.sampling_method == "uniform":
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
            coords_f = lhs_samples(
                config.n_points,
                3,
                [problem.x_bounds, problem.y_bounds, problem.t_bounds],
                device,
            )
            n_ic = round(config.n_points ** (2 / 3))
            spatial_lhs = lhs_samples(
                n_ic, 2, [problem.x_bounds, problem.y_bounds], device
            )
            x_ic = spatial_lhs[:, 0:1]
            y_ic = spatial_lhs[:, 1:2]
            t_ic = torch.zeros_like(x_ic, device=device)
            coords_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
            u0 = problem.initial_condition(x_ic, y_ic)
    else:
        raise ValueError(f"Unknown problem type: {type(problem)}")

    residual_fn = config.get_residual_fn(problem)

    loss_history = []
    loss_f_history = []
    loss_ic_history = []
    epochs_measured = []
    
    snr_history = []
    snr_epochs = []

    if config.rba_enabled:
        # Use flattened coords_f to determine the number of weights
        n_colloc = coords_f.view(-1, coords_f.shape[-1]).shape[0]
        lambda_weights = torch.ones(n_colloc, device=device)
        gamma = 1.0 - config.rba_eta
    
    start_training = time.time()

    for epoch in range(config.epochs):
        if config.snr_compute_interval > 0 and epoch % config.snr_compute_interval == 0:
            batch_grads = []
            coords_f_chunks = torch.tensor_split(coords_f, config.snr_batches)
            coords_ic_chunks = torch.tensor_split(coords_ic, config.snr_batches)
            u0_chunks = torch.tensor_split(u0, config.snr_batches)
            
            for b_coords_f, b_coords_ic, b_u0 in zip(coords_f_chunks, coords_ic_chunks, u0_chunks):
                b_loss = 0.0
                if b_coords_f.numel() > 0:
                    f_val = residual_fn(model, b_coords_f)
                    b_loss = b_loss + torch.mean(f_val**2)
                if b_coords_ic.numel() > 0:
                    u_pred_ic = model(b_coords_ic)
                    b_loss = b_loss + torch.mean((u_pred_ic - b_u0) ** 2)
                
                if isinstance(b_loss, torch.Tensor):
                    optimizer.zero_grad(set_to_none=True)
                    b_loss.backward()
                    
                    grads = []
                    for p in model.parameters():
                        if p.grad is not None:
                            grads.append(p.grad.view(-1))
                        else:
                            grads.append(torch.zeros_like(p).view(-1))
                    if grads:
                        batch_grads.append(torch.cat(grads))
            
            if batch_grads:
                G = torch.stack(batch_grads)
                mu = G.mean(dim=0)
                sigma = G.std(dim=0, unbiased=False)
                snr = torch.norm(mu, p=2) / (torch.norm(sigma, p=2) + 1e-8)
                snr_history.append(snr.item())
                snr_epochs.append(epoch)
            
            optimizer.zero_grad(set_to_none=True)

        f = residual_fn(model, coords_f)
        if config.rba_enabled:
            with torch.no_grad():
                abs_f = torch.abs(f).view(-1)
                max_abs_f = torch.max(abs_f)
                lambda_weights = gamma * lambda_weights + config.rba_eta * (
                    abs_f / (max_abs_f + 1e-8)
                )
            loss_f = torch.mean((lambda_weights.view_as(f) * f) ** 2)
        else:
            loss_f = torch.mean(f**2)

        u_pred_ic = model(coords_ic)
        loss_ic = torch.mean((u_pred_ic - u0) ** 2)

        loss = loss_f + loss_ic

        optimizer.zero_grad(set_to_none=True)
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

        if isinstance(problem, Problem1D) and epoch in [config.epochs // 10, 2 * (config.epochs // 3), config.epochs - 1]:
            import os
            os.makedirs("results/heatmaps", exist_ok=True)
            
            x_min, x_max = problem.x_bounds
            t_min, t_max = problem.t_bounds
            
            n_grid = 200
            x_line = torch.linspace(x_min, x_max, n_grid, device=device)
            t_line = torch.linspace(t_min, t_max, n_grid, device=device)
            X, T = torch.meshgrid(x_line, t_line, indexing="ij")
            
            grid_coords = torch.stack([X.flatten(), T.flatten()], dim=-1)
            grid_coords.requires_grad_(True)
            
            f_grid = residual_fn(model, grid_coords)
            
            with torch.no_grad():
                u_grid = model(grid_coords)
                
            U = u_grid.reshape(n_grid, n_grid).detach().cpu().numpy()
            F = f_grid.reshape(n_grid, n_grid).detach().cpu().numpy()
            
            # Divide by the maximum absolute residual on collocation points
            f_colloc_max = torch.max(torch.abs(f)).item()
            import numpy as np
            F = np.abs(F) / (f_colloc_max + 1e-8)
            
            X_np = X.detach().cpu().numpy()
            T_np = T.detach().cpu().numpy()
            
            X_norm = (X_np - x_min) / (x_max - x_min)
            T_norm = (T_np - t_min) / (t_max - t_min)
            
            fig_hm, axes_hm = plt.subplots(1, 2, figsize=(12, 5))
            
            c1 = axes_hm[0].pcolormesh(T_norm, X_norm, U, cmap="viridis", shading="auto")
            axes_hm[0].set_title(f"Solution at Epoch {epoch}")
            axes_hm[0].set_xlabel("Normalized t")
            axes_hm[0].set_ylabel("Normalized x")
            fig_hm.colorbar(c1, ax=axes_hm[0])
            
            c2 = axes_hm[1].pcolormesh(T_norm, X_norm, F, cmap="viridis", shading="auto", vmin=0, vmax=1)
            axes_hm[1].set_title(f"Residual at Epoch {epoch}")
            axes_hm[1].set_xlabel("Normalized t")
            axes_hm[1].set_ylabel("Normalized x")
            fig_hm.colorbar(c2, ax=axes_hm[1])
            
            fig_hm.tight_layout()
            
            exp_suffix = f"_{exp_id}" if exp_id else ""
            fig_hm.savefig(f"results/heatmaps/{problem.name}{exp_suffix}_epoch_{epoch}.png")
            plt.close(fig_hm)

        loss_history.append(loss.detach())
        loss_f_history.append(loss_f.detach())
        loss_ic_history.append(loss_ic.detach())
        epochs_measured.append(epoch)

    total_time = time.time() - start_training
    
    # Convert history tensors to CPU numbers at the end
    loss_history = [l.item() for l in loss_history]
    loss_f_history = [l.item() for l in loss_f_history]
    loss_ic_history = [l.item() for l in loss_ic_history]

    print(f"Total training time: {total_time:.2f} seconds")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs_measured, loss_history, label="Total Loss", linewidth=2)
    axes[0].plot(epochs_measured, loss_f_history, label="PDE Residual (loss_f)", alpha=0.7)
    axes[0].plot(epochs_measured, loss_ic_history, label="Initial Condition (loss_ic)", alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch (log scale)")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title(
        f"{problem.name}'s Training Loss History.\n"
        + f"Elapsed time: {total_time:.2e} seconds"
    )
    axes[0].legend()
    axes[0].grid(True)

    if snr_history:
        axes[1].plot(snr_epochs, snr_history, label="Gradient SNR", color="purple")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Epoch (log scale)")
        axes[1].set_ylabel("SNR (log scale)")
        axes[1].set_title("Gradient Signal-to-Noise Ratio (SNR)")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis("off")
        axes[1].set_title("SNR Tracking Disabled or No Data")

    plt.tight_layout()

    metrics = {
        "final_loss": loss_history[-1],
        "final_loss_f": loss_f.item(),
        "final_loss_ic": loss_ic.item(),
        "training_time": total_time,
        "loss_history": loss_history,
        "loss_f_history": loss_f_history,
        "loss_ic_history": loss_ic_history,
        "epochs_measured": epochs_measured,
        "snr_history": snr_history,
        "snr_epochs": snr_epochs,
    }

    return model.to("cpu"), fig, metrics

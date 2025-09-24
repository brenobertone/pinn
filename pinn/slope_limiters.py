import torch
from torch import Tensor
import torch.nn.functional as F

from .problems_definitions import PINN, Problem

PADDING = 10

def shift(xyt, dx=0, dy=0, dt=0):
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    return torch.cat([x + dx, y + dy, t + dt], dim=1)


def mm2(x, y):
    return (
        (torch.sign(x) + torch.sign(y))
        * torch.minimum(torch.abs(x), torch.abs(y))
        / 2
    )


def mm3(x, y, z):
    return (
        0.125
        * (
            (torch.sign(x) + torch.sign(y))
            * (torch.sign(y) + torch.sign(z))
            * (torch.sign(x) + torch.sign(z))
        )
        * torch.minimum(
            torch.minimum(torch.abs(x), torch.abs(y)), torch.abs(z)
        )
    )


def uno(
    f_minus2: torch.Tensor,
    f_minus1: torch.Tensor,
    f_0: torch.Tensor,
    f_plus1: torch.Tensor,
    f_plus2: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    Delta_u_p = (f_plus1 - f_0) / delta
    Delta_u_m = (f_0 - f_minus1) / delta

    Delta_2u_j = (f_plus1 - 2 * f_0 + f_minus1) / (2 * delta)
    Delta_2u_jp1 = (f_plus2 - 2 * f_plus1 + f_0) / (2 * delta)
    Delta_2u_jm1 = (f_0 - 2 * f_minus1 + f_minus2) / (2 * delta)

    delta_1_2 = 0.5 * mm2(Delta_2u_jp1, Delta_2u_j)
    delta_2_2 = 0.5 * mm2(Delta_2u_j, Delta_2u_jm1)

    return mm2(Delta_u_p - delta_1_2, Delta_u_m + delta_2_2)


def advection_residual_autograd(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    xyt = xyt.reshape(-1, 3)
    xyt.requires_grad_(True)
    u = model(xyt)

    grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[
        0
    ]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    f1_u_x = torch.autograd.grad(
        problem.f1(u), xyt, torch.ones_like(u), create_graph=True
    )[0][:, 0:1]
    f2_u_y = torch.autograd.grad(
        problem.f2(u), xyt, torch.ones_like(u), create_graph=True
    )[0][:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, xyt, torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, xyt, torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2]

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)


def diff_ops_mm2(u: torch.Tensor, deltas: list[float], Ns: list[int], PADDING: int = 1):
    dx, dy, dt = deltas
    Nx, Ny, Nt = Ns
    Nx -= 2 * PADDING
    Ny -= 2 * PADDING
    Nt -= 2 * PADDING

    shape_inner = (Nx, Ny, Nt)
    u_x  = torch.zeros(shape_inner, device=u.device)
    u_y  = torch.zeros(shape_inner, device=u.device)
    u_t  = torch.zeros(shape_inner, device=u.device)
    u_xx = torch.zeros(shape_inner, device=u.device)
    u_yy = torch.zeros(shape_inner, device=u.device)

    # interior slices
    xs = slice(PADDING, Nx + PADDING)
    ys = slice(PADDING, Ny + PADDING)
    ts = slice(PADDING, Nt + PADDING)

    # first differences
    u_x_forward  = (u[xs.start+1:xs.stop+1, ys, ts] - u[xs, ys, ts]) / dx
    u_x_backward = (u[xs, ys, ts] - u[xs.start-1:xs.stop-1, ys, ts]) / dx
    u_y_forward  = (u[xs, ys.start+1:ys.stop+1, ts] - u[xs, ys, ts]) / dy
    u_y_backward = (u[xs, ys, ts] - u[xs, ys.start-1:ys.stop-1, ts]) / dy
    u_t_forward  = (u[xs, ys, ts.start+1:ts.stop+1] - u[xs, ys, ts]) / dt
    u_t_backward = (u[xs, ys, ts] - u[xs, ys, ts.start-1:ts.stop-1]) / dt

    # slope-limited derivatives (minmod 2)
    u_x[:] = mm2(u_x_forward, u_x_backward)
    u_y[:] = mm2(u_y_forward, u_y_backward)
    u_t[:] = mm2(u_t_forward, u_t_backward)

    # second derivatives (centered, no limiter)
    u_xx[:] = (u[xs.start+1:xs.stop+1, ys, ts] - 2*u[xs, ys, ts] + u[xs.start-1:xs.stop-1, ys, ts]) / dx**2
    u_yy[:] = (u[xs, ys.start+1:ys.stop+1, ts] - 2*u[xs, ys, ts] + u[xs, ys.start-1:ys.stop-1, ts]) / dy**2

    return u_x, u_y, u_t, u_xx, u_yy


def advection_residual_mm2(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:

    dx = (xyt[1, 0, 0, 0] - xyt[0, 0, 0, 0]).item()
    dy = (xyt[0, 1, 0, 1] - xyt[0, 0, 0, 1]).item()
    dt = (xyt[0, 0, 1, 2] - xyt[0, 0, 0, 2]).item()
    deltas = [dx, dy, dt]

    Nx, Ny, Nt, _ = xyt.shape
    Ns = [Nx, Ny, Nt]
    u = model(xyt.reshape(-1, 3))
    u = u.view(Nx, Ny, Nt)

    _, _, u_t, u_xx, u_yy = diff_ops_mm2(u, deltas, Ns)
    f1_u_x, _, _, _, _ = diff_ops_mm2(problem.f1(u), deltas, Ns)
    _, f2_u_y, _, _, _ = diff_ops_mm2(problem.f2(u), deltas, Ns)

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)


def diff_ops_mm3(u: torch.Tensor, deltas: list[float], Ns: list[int], PADDING: int = 1):
    dx, dy, dt = deltas
    Nx, Ny, Nt = Ns
    Nx -= 2 * PADDING
    Ny -= 2 * PADDING
    Nt -= 2 * PADDING

    shape_inner = (Nx, Ny, Nt)
    u_x  = torch.zeros(shape_inner, device=u.device)
    u_y  = torch.zeros(shape_inner, device=u.device)
    u_t  = torch.zeros(shape_inner, device=u.device)
    u_xx = torch.zeros(shape_inner, device=u.device)
    u_yy = torch.zeros(shape_inner, device=u.device)

    # interior slices
    xs = slice(PADDING, Nx + PADDING)
    ys = slice(PADDING, Ny + PADDING)
    ts = slice(PADDING, Nt + PADDING)

    # forward / backward / centered differences
    u_x_forward  = (u[xs.start+1:xs.stop+1, ys, ts] - u[xs, ys, ts]) / dx
    u_x_backward = (u[xs, ys, ts] - u[xs.start-1:xs.stop-1, ys, ts]) / dx
    u_x_centered = (u[xs.start+1:xs.stop+1, ys, ts] - u[xs.start-1:xs.stop-1, ys, ts]) / (2*dx)

    u_y_forward  = (u[xs, ys.start+1:ys.stop+1, ts] - u[xs, ys, ts]) / dy
    u_y_backward = (u[xs, ys, ts] - u[xs, ys.start-1:ys.stop-1, ts]) / dy
    u_y_centered = (u[xs, ys.start+1:ys.stop+1, ts] - u[xs, ys.start-1:ys.stop-1, ts]) / (2*dy)

    u_t_forward  = (u[xs, ys, ts.start+1:ts.stop+1] - u[xs, ys, ts]) / dt
    u_t_backward = (u[xs, ys, ts] - u[xs, ys, ts.start-1:ts.stop-1]) / dt
    u_t_centered = (u[xs, ys, ts.start+1:ts.stop+1] - u[xs, ys, ts.start-1:ts.stop-1]) / (2*dt)

    # slope-limited derivatives (minmod 3)
    u_x[:] = mm3(u_x_forward, u_x_backward, u_x_centered)
    u_y[:] = mm3(u_y_forward, u_y_backward, u_y_centered)
    u_t[:] = mm3(u_t_forward, u_t_backward, u_t_centered)

    # second derivatives (centered, no limiter)
    u_xx[:] = (u[xs.start+1:xs.stop+1, ys, ts] - 2*u[xs, ys, ts] + u[xs.start-1:xs.stop-1, ys, ts]) / dx**2
    u_yy[:] = (u[xs, ys.start+1:ys.stop+1, ts] - 2*u[xs, ys, ts] + u[xs, ys.start-1:ys.stop-1, ts]) / dy**2

    return u_x, u_y, u_t, u_xx, u_yy


def advection_residual_mm3(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:

    dx = (xyt[1, 0, 0, 0] - xyt[0, 0, 0, 0]).item()
    dy = (xyt[0, 1, 0, 1] - xyt[0, 0, 0, 1]).item()
    dt = (xyt[0, 0, 1, 2] - xyt[0, 0, 0, 2]).item()
    deltas = [dx, dy, dt]

    Nx, Ny, Nt, _ = xyt.shape
    Ns = [Nx, Ny, Nt]
    u = model(xyt.reshape(-1, 3))
    u = u.view(Nx, Ny, Nt)

    _, _, u_t, u_xx, u_yy = diff_ops_mm3(u, deltas, Ns)
    f1_u_x, _, _, _, _ = diff_ops_mm3(problem.f1(u), deltas, Ns)
    _, f2_u_y, _, _, _ = diff_ops_mm3(problem.f2(u), deltas, Ns)

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)


def diff_ops_uno(u: torch.Tensor, deltas: list[float], Ns: list[int]):
    """
    Compute UNO slope-limited derivatives u_x, u_y, u_t
    and UNO-limited second derivatives u_xx, u_yy.
    Needs PADDING >= 2 for 5-point stencil.
    """
    dx, dy, dt = deltas
    Nx, Ny, Nt = Ns
    Nx -= 2 * PADDING
    Ny -= 2 * PADDING
    Nt -= 2 * PADDING

    shape_inner = (Nx, Ny, Nt)
    u_x = torch.zeros(shape_inner, device=u.device)
    u_y = torch.zeros(shape_inner, device=u.device)
    u_t = torch.zeros(shape_inner, device=u.device)
    u_xx = torch.zeros(shape_inner, device=u.device)
    u_yy = torch.zeros(shape_inner, device=u.device)

    xs = slice(PADDING, Nx + PADDING)
    ys = slice(PADDING, Ny + PADDING)
    ts = slice(PADDING, Nt + PADDING)

    # ---- X-DERIVATIVE ----
    f_m2 = u[xs.start-2:xs.stop-2, ys, ts]
    f_m1 = u[xs.start-1:xs.stop-1, ys, ts]
    f_0  = u[xs, ys, ts]
    f_p1 = u[xs.start+1:xs.stop+1, ys, ts]
    f_p2 = u[xs.start+2:xs.stop+2, ys, ts]

    Δu_p = (f_p1 - f_0) / dx
    Δu_m = (f_0 - f_m1) / dx
    Δ2u_j   = (f_p1 - 2*f_0 + f_m1) / (2*dx)
    Δ2u_jp1 = (f_p2 - 2*f_p1 + f_0) / (2*dx)
    Δ2u_jm1 = (f_0 - 2*f_m1 + f_m2) / (2*dx)

    δ1_2 = 0.5 * mm2(Δ2u_jp1, Δ2u_j)
    δ2_2 = 0.5 * mm2(Δ2u_j, Δ2u_jm1)

    u_x[:] = mm2(Δu_p - δ1_2, Δu_m + δ2_2)

    # UNO second derivative
    Δ2_central = (f_p1 - 2*f_0 + f_m1) / dx**2
    Δ2_neighbors = 0.5 * ((f_p2 - 2*f_p1 + f_0) / dx**2 +
                          (f_0 - 2*f_m1 + f_m2) / dx**2)
    u_xx[:] = mm2(Δ2_central, Δ2_neighbors)

    # ---- Y-DERIVATIVE ----
    f_m2 = u[xs, ys.start-2:ys.stop-2, ts]
    f_m1 = u[xs, ys.start-1:ys.stop-1, ts]
    f_0  = u[xs, ys, ts]
    f_p1 = u[xs, ys.start+1:ys.stop+1, ts]
    f_p2 = u[xs, ys.start+2:ys.stop+2, ts]

    Δu_p = (f_p1 - f_0) / dy
    Δu_m = (f_0 - f_m1) / dy
    Δ2u_j   = (f_p1 - 2*f_0 + f_m1) / (2*dy)
    Δ2u_jp1 = (f_p2 - 2*f_p1 + f_0) / (2*dy)
    Δ2u_jm1 = (f_0 - 2*f_m1 + f_m2) / (2*dy)

    δ1_2 = 0.5 * mm2(Δ2u_jp1, Δ2u_j)
    δ2_2 = 0.5 * mm2(Δ2u_j, Δ2u_jm1)

    u_y[:] = mm2(Δu_p - δ1_2, Δu_m + δ2_2)

    # UNO second derivative
    Δ2_central = (f_p1 - 2*f_0 + f_m1) / dy**2
    Δ2_neighbors = 0.5 * ((f_p2 - 2*f_p1 + f_0) / dy**2 +
                          (f_0 - 2*f_m1 + f_m2) / dy**2)
    u_yy[:] = mm2(Δ2_central, Δ2_neighbors)

    # ---- T-DERIVATIVE (no u_tt returned, only slope) ----
    f_m2 = u[xs, ys, ts.start-2:ts.stop-2]
    f_m1 = u[xs, ys, ts.start-1:ts.stop-1]
    f_0  = u[xs, ys, ts]
    f_p1 = u[xs, ys, ts.start+1:ts.stop+1]
    f_p2 = u[xs, ys, ts.start+2:ts.stop+2]

    Δu_p = (f_p1 - f_0) / dt
    Δu_m = (f_0 - f_m1) / dt
    Δ2u_j   = (f_p1 - 2*f_0 + f_m1) / (2*dt)
    Δ2u_jp1 = (f_p2 - 2*f_p1 + f_0) / (2*dt)
    Δ2u_jm1 = (f_0 - 2*f_m1 + f_m2) / (2*dt)

    δ1_2 = 0.5 * mm2(Δ2u_jp1, Δ2u_j)
    δ2_2 = 0.5 * mm2(Δ2u_j, Δ2u_jm1)

    u_t[:] = mm2(Δu_p - δ1_2, Δu_m + δ2_2)

    return u_x, u_y, u_t, u_xx, u_yy

def advection_residual_uno(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    dx = (xyt[1, 0, 0, 0] - xyt[0, 0, 0, 0]).item()
    dy = (xyt[0, 1, 0, 1] - xyt[0, 0, 0, 1]).item()
    dt = (xyt[0, 0, 1, 2] - xyt[0, 0, 0, 2]).item()
    deltas = [dx, dy, dt]

    Nx, Ny, Nt, _ = xyt.shape
    Ns = [Nx, Ny, Nt]
    u = model(xyt.reshape(-1, 3))
    u = u.view(Nx, Ny, Nt)

    _, _, u_t, u_xx, u_yy = diff_ops_mm3(u, deltas, Ns)
    f1_u_x, _, _, _, _ = diff_ops_mm3(problem.f1(u), deltas, Ns)
    _, f2_u_y, _, _, _ = diff_ops_mm3(problem.f2(u), deltas, Ns)

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)
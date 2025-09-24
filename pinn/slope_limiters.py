import torch

from .problems_definitions import PINN, Problem

PADDING = 2


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

    xs = slice(PADDING, Nx + PADDING)
    ys = slice(PADDING, Ny + PADDING)
    ts = slice(PADDING, Nt + PADDING)

    u_x_forward  = (u[xs.start+1:xs.stop+1, ys, ts] - u[xs, ys, ts]) / dx
    u_x_backward = (u[xs, ys, ts] - u[xs.start-1:xs.stop-1, ys, ts]) / dx
    u_y_forward  = (u[xs, ys.start+1:ys.stop+1, ts] - u[xs, ys, ts]) / dy
    u_y_backward = (u[xs, ys, ts] - u[xs, ys.start-1:ys.stop-1, ts]) / dy
    u_t_forward  = (u[xs, ys, ts.start+1:ts.stop+1] - u[xs, ys, ts]) / dt
    u_t_backward = (u[xs, ys, ts] - u[xs, ys, ts.start-1:ts.stop-1]) / dt

    u_x[:] = mm2(u_x_forward, u_x_backward)
    u_y[:] = mm2(u_y_forward, u_y_backward)
    u_t[:] = mm2(u_t_forward, u_t_backward)

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

    xs = slice(PADDING, Nx + PADDING)
    ys = slice(PADDING, Ny + PADDING)
    ts = slice(PADDING, Nt + PADDING)

    u_x_forward  = (u[xs.start+1:xs.stop+1, ys, ts] - u[xs, ys, ts]) / dx
    u_x_backward = (u[xs, ys, ts] - u[xs.start-1:xs.stop-1, ys, ts]) / dx
    u_x_centered = (u[xs.start+1:xs.stop+1, ys, ts] - u[xs.start-1:xs.stop-1, ys, ts]) / (2*dx)

    u_y_forward  = (u[xs, ys.start+1:ys.stop+1, ts] - u[xs, ys, ts]) / dy
    u_y_backward = (u[xs, ys, ts] - u[xs, ys.start-1:ys.stop-1, ts]) / dy
    u_y_centered = (u[xs, ys.start+1:ys.stop+1, ts] - u[xs, ys.start-1:ys.stop-1, ts]) / (2*dy)

    u_t_forward  = (u[xs, ys, ts.start+1:ts.stop+1] - u[xs, ys, ts]) / dt
    u_t_backward = (u[xs, ys, ts] - u[xs, ys, ts.start-1:ts.stop-1]) / dt
    u_t_centered = (u[xs, ys, ts.start+1:ts.stop+1] - u[xs, ys, ts.start-1:ts.stop-1]) / (2*dt)

    u_x[:] = mm3(u_x_forward, u_x_backward, u_x_centered)
    u_y[:] = mm3(u_y_forward, u_y_backward, u_y_centered)
    u_t[:] = mm3(u_t_forward, u_t_backward, u_t_centered)

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


def diff_ops_uno(u: torch.Tensor, deltas: list[float], Ns: list[int], PADDING: int = 2):
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

    xs = slice(PADDING, Nx + PADDING)
    ys = slice(PADDING, Ny + PADDING)
    ts = slice(PADDING, Nt + PADDING)

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

    u_xx[:] = (f_p1 - 2*f_0 + f_m1) / dx**2

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

    u_yy[:] = (f_p1 - 2*f_0 + f_m1) / dy**2

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
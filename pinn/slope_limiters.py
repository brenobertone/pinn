import torch

from .problems_definitions import PINN, Problem


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
    delta: float,
) -> torch.Tensor:
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


def advection_residual_mm2(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
    delta: float,
) -> torch.Tensor:
    u = model(xyt)

    t_left = shift(xyt, dt=-delta)
    t_right = shift(xyt, dt=delta)
    u_left_t = model(t_left)
    u_right_t = model(t_right)
    slope_t_left = (u - u_left_t) / delta
    slope_t_right = (u_right_t - u) / delta
    u_t = mm2(slope_t_left, slope_t_right)

    x_left = shift(xyt, dx=-delta)
    x_right = shift(xyt, dx=delta)
    u_left_x = model(x_left)
    u_right_x = model(x_right)
    slope_x_left = (u - u_left_x) / delta
    slope_x_right = (u_right_x - u) / delta
    mm2(slope_x_left, slope_x_right)

    y_left = shift(xyt, dy=-delta)
    y_right = shift(xyt, dy=delta)
    u_left_y = model(y_left)
    u_right_y = model(y_right)
    slope_y_left = (u - u_left_y) / delta
    slope_y_right = (u_right_y - u) / delta
    mm2(slope_y_left, slope_y_right)

    f1u_left = problem.f1(model(x_left))
    f1u_right = problem.f1(model(x_right))
    slope_f1_x_left = (problem.f1(u) - f1u_left) / delta
    slope_f1_x_right = (f1u_right - problem.f1(u)) / delta
    f1_u_x = mm2(slope_f1_x_left, slope_f1_x_right)

    f2u_left = problem.f2(model(y_left))
    f2u_right = problem.f2(model(y_right))
    slope_f2_y_left = (problem.f2(u) - f2u_left) / delta
    slope_f2_y_right = (f2u_right - problem.f2(u)) / delta
    f2_u_y = mm2(slope_f2_y_left, slope_f2_y_right)

    u_xx = (u_right_x - 2 * u + u_left_x) / (delta**2)

    u_yy = (u_right_y - 2 * u + u_left_y) / (delta**2)

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)


def advection_residual_mm3(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
    delta: float,
) -> torch.Tensor:
    u = model(xyt)

    # Create shifted points
    t_minus = shift(xyt, dt=-delta)
    t_plus = shift(xyt, dt=delta)
    u_t_minus = model(t_minus)
    u_t_plus = model(t_plus)
    u_t = mm3(
        (u - u_t_minus) / delta,
        (u_t_plus - u) / delta,
        (u_t_plus - u_t_minus) / (2 * delta),
    )

    x_minus = shift(xyt, dx=-delta)
    x_plus = shift(xyt, dx=delta)
    u_x_minus = model(x_minus)
    u_x_plus = model(x_plus)

    y_minus = shift(xyt, dy=-delta)
    y_plus = shift(xyt, dy=delta)
    u_y_minus = model(y_minus)
    u_y_plus = model(y_plus)

    # problem.f1(u)_x with mm3
    f1u_minus = problem.f1(model(x_minus))
    f1u_plus = problem.f1(model(x_plus))
    f1_u_x = mm3(
        (problem.f1(u) - f1u_minus) / delta,
        (f1u_plus - problem.f1(u)) / delta,
        (f1u_plus - f1u_minus) / (2 * delta),
    )

    # problem.f2(u)_y with mm3
    f2u_minus = problem.f2(model(y_minus))
    f2u_plus = problem.f2(model(y_plus))
    f2_u_y = mm3(
        (problem.f2(u) - f2u_minus) / delta,
        (f2u_plus - problem.f2(u)) / delta,
        (f2u_plus - f2u_minus) / (2 * delta),
    )

    # u_xx and u_yy (second order centered)
    u_xx = (u_x_plus - 2 * u + u_x_minus) / delta**2
    u_yy = (u_y_plus - 2 * u + u_y_minus) / delta**2

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)


def advection_residual_uno(
    model: PINN,
    problem: Problem,
    xyt: torch.Tensor,
    epsilon: float,
    delta: float,
) -> torch.Tensor:
    u0 = model(xyt)

    u_tm2 = model(shift(xyt, dt=-2 * delta))
    u_tm1 = model(shift(xyt, dt=-delta))
    u_tp1 = model(shift(xyt, dt=delta))
    u_tp2 = model(shift(xyt, dt=2 * delta))
    u_t = uno(u_tm2, u_tm1, u0, u_tp1, u_tp2, delta)

    u_xm2 = model(shift(xyt, dx=-2 * delta))
    u_xm1 = model(shift(xyt, dx=-delta))
    u_xp1 = model(shift(xyt, dx=delta))
    u_xp2 = model(shift(xyt, dx=2 * delta))
    uno(u_xm2, u_xm1, u0, u_xp1, u_xp2, delta)

    u_ym2 = model(shift(xyt, dy=-2 * delta))
    u_ym1 = model(shift(xyt, dy=-delta))
    u_yp1 = model(shift(xyt, dy=delta))
    u_yp2 = model(shift(xyt, dy=2 * delta))
    uno(u_ym2, u_ym1, u0, u_yp1, u_yp2, delta)

    f1_u_x = uno(
        problem.f1(u_xm2),
        problem.f1(u_xm1),
        problem.f1(u0),
        problem.f1(u_xp1),
        problem.f1(u_xp2),
        delta,
    )
    f2_u_y = uno(
        problem.f2(u_ym2),
        problem.f2(u_ym1),
        problem.f2(u0),
        problem.f2(u_yp1),
        problem.f2(u_yp2),
        delta,
    )

    u_xx = (u_xp1 - 2 * u0 + u_xm1) / delta**2
    u_yy = (u_yp1 - 2 * u0 + u_ym1) / delta**2

    return u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)

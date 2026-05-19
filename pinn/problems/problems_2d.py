import numpy as np
import torch

from pinn.core.problems import Problem2D


class PeriodicSine2D(Problem2D):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 1.0)
    name = "PeriodicSine2D"
    x_orientation = "decrescent"
    y_orientation = "decrescent"

    @staticmethod
    def f1(u):
        return u

    @staticmethod
    def f2(u):
        return u

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return (torch.sin(np.pi * x) ** 2) * (torch.sin(np.pi * y) ** 2)


class Rarefaction1D(Problem2D):
    x_bounds = (-6.0, 6.0)
    y_bounds = (-1.5, 1.5)
    t_bounds = (0.0, 2.5)
    name = "Rarefaction1D"

    @staticmethod
    def f1(u):
        return u**2 / 2

    @staticmethod
    def f2(u):
        return u**2 / 2

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        u = torch.zeros_like(x)
        u[x > 0] = 1.0
        u[x < 0] = -1.0
        return u


class Shock1D(Problem2D):
    x_bounds = (-6.0, 6.0)
    y_bounds = (-1.5, 1.5)
    t_bounds = (0.0, 2.5)
    name = "Shock1D"

    @staticmethod
    def f1(u):
        return u**2 / 2

    @staticmethod
    def f2(u):
        return u**2 / 2

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        u = torch.zeros_like(x)
        u[x > 0] = -1.0
        u[x < 0] = 0
        return u


class Pulse(Problem2D):
    x_bounds = (-3.0, 3.0)
    y_bounds = (-3, 3)
    t_bounds = (0.0, 2.5)
    name = "Pulse"

    @staticmethod
    def f1(u):
        return u**2 / 2

    @staticmethod
    def f2(u):
        return u**2 / 2

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        in_square = (0 < x) & (x < 1) & (0 < y) & (y < 1)
        u = torch.where(
            in_square,
            torch.tensor(1.0, device=x.device),
            torch.tensor(0.0, device=x.device),
        )
        return u


class RiemannOblique(Problem2D):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 0.5)
    name = "RiemannOblique"
    x_orientation = "decrescent"
    y_orientation = "decrescent"

    @staticmethod
    def f1(u):
        return u**2 / 2

    @staticmethod
    def f2(u):
        return u**2 / 2

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        u = torch.zeros_like(x)
        u[(x > 0.5) & (y > 0.5)] = -1.0
        u[(x < 0.5) & (y > 0.5)] = -0.2
        u[(x < 0.5) & (y < 0.5)] = 0.5
        u[(x > 0.5) & (y < 0.5)] = 0.8
        return u


class Riemann2D(Problem2D):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 1.0 / 12.0)
    name = "Riemann2D"

    @staticmethod
    def f1(u):
        return u**2 / 2

    @staticmethod
    def f2(u):
        return u**2 / 2

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        u = torch.ones_like(x)
        u[(x < 0.25) & (y < 0.25)] = 2.0
        u[(x > 0.25) & (y > 0.25)] = 3.0
        return u


class BuckleyLeverett(Problem2D):
    x_bounds = (-1.5, 1.5)
    y_bounds = (-1.5, 1.5)
    t_bounds = (0.0, 0.5)
    name = "BuckleyLeverett"

    µ_w_µ_0 = 1
    C_g = 5

    @staticmethod
    def f1(Sw: torch.Tensor) -> torch.Tensor:
        return Sw**2 / (Sw**2 + BuckleyLeverett.µ_w_µ_0 * (1 - Sw) ** 2)

    @staticmethod
    def f2(Sw: torch.Tensor) -> torch.Tensor:
        return BuckleyLeverett.f1(Sw) * (
            1 - BuckleyLeverett.C_g * (1 - Sw) ** 2
        )

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        u = torch.zeros_like(x)
        mask = x**2 + y**2 < 0.5
        u[mask] = 1
        return u


class NonLinearNonConvexFlow(Problem2D):
    x_bounds = (-2.0, 2.0)
    y_bounds = (-2.0, 2.0)
    t_bounds = (0.0, 1.0)
    name = "NonLinearNonConvexFlow"

    @staticmethod
    def f1(u):
        return torch.sin(u)

    @staticmethod
    def f2(u):
        return torch.cos(u)

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        u = torch.full_like(x, 0.25 * torch.pi)
        mask = x**2 + y**2 < 1.0
        u[mask] = 3.5 * torch.pi
        return u

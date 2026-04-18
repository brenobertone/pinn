import torch

from pinn.core.problems import Problem1D


class Burgers1D(Problem1D):
    x_bounds = (-1.0, 1.0)
    t_bounds = (0.0, 1.0)
    name = "Burgers1D"

    @staticmethod
    def f(u):
        return u**2 / 2

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.sin(torch.pi * x)


class Shock1DPure(Problem1D):
    x_bounds = (-5.0, 5.0)
    t_bounds = (0.0, 2.0)
    name = "Shock1DPure"

    @staticmethod
    def f(u):
        return u**2 / 2

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(x)
        u[x < 0] = 1.0
        u[x > 0] = -0.5
        return u


class Rarefaction1DPure(Problem1D):
    x_bounds = (-5.0, 5.0)
    t_bounds = (0.0, 2.0)
    name = "Rarefaction1DPure"

    @staticmethod
    def f(u):
        return u**2 / 2

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(x)
        u[x < 0] = -1.0
        u[x > 0] = 1.0
        return u


class LinearAdvection1D(Problem1D):
    x_bounds = (0.0, 1.0)
    t_bounds = (0.0, 1.0)
    name = "LinearAdvection1D"

    @staticmethod
    def f(u):
        return u

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * x)

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


class AdvectionTanh1D(Problem1D):
    x_bounds = (-5.0, 5.0)
    t_bounds = (0.0, 2.0)
    name = "AdvectionTanh1D"
    benchmark_source = "exact"

    @staticmethod
    def f(u):
        return u  # linear advection

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        # u(x,0) = sum_{j=1}^{k} w2_j tanh(A_j * x + w1_j)
        # Example: k=3 with varying coefficients
        w2 = [1.0, -0.5, 0.8]
        A = [0.5, 1.0, 0.3]
        w1 = [0.0, 2.0, -1.5]

        u = torch.zeros_like(x)
        for w2j, Aj, w1j in zip(w2, A, w1):
            u += w2j * torch.tanh(Aj * x + w1j)
        return u

    def benchmark_solution(self, x, t):
        c = 1
        return self.initial_condition(x - c*t)
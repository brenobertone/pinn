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

    def benchmark_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Inviscid Burgers with u0(x) = -sin(pi*x)
        # Solve u + sin(pi*(x - u*t)) = 0
        # Use bisection to find the correct physical branch:
        # For x > 0, u is in [-1, 0]. For x < 0, u is in [0, 1].
        low = torch.where(x > 0, -torch.ones_like(x), torch.zeros_like(x))
        high = torch.where(x > 0, torch.zeros_like(x), torch.ones_like(x))

        for _ in range(25):
            mid = (low + high) / 2
            val = mid + torch.sin(torch.pi * (x - mid * t))
            mask = val > 0
            # If val > 0, we need to go more negative (lower the bound)
            high = torch.where(mask, mid, high)
            low = torch.where(mask, low, mid)
        return (low + high) / 2


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

    def benchmark_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = 0.25
        return torch.where(x < s * t, 1.0, -0.5)


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

    def benchmark_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_safe = torch.maximum(t, torch.tensor(1e-8, device=t.device))
        u = x / t_safe
        u = torch.where(x < -t, -1.0, u)
        u = torch.where(x > t, 1.0, u)
        return u


class LinearAdvection1D(Problem1D):
    x_bounds = (0.0, 1.0)
    t_bounds = (0.0, 1.0)
    name = "LinearAdvection1D"
    c = 0.5  # wave speed

    @staticmethod
    def f(u):
        c = 1/2
        return c * u

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * x)

    def benchmark_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * (x - self.c * t))


class AdvectionTanh1D(Problem1D):
    x_bounds = (-5.0, 5.0)
    t_bounds = (0.0, 2.0)
    name = "AdvectionTanh1D"
    benchmark_source = "exact"
    c = 0.5  # wave speed

    @staticmethod
    def f(u):
        c = 1/2
        return c * u  # linear advection

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        # u(x,0) = sum_{j=1}^{k} w2_j tanh(A_j * x + b1_j)
        # Example: k=3 with varying coefficients
        w2 = [1.0, -0.5, 0.8]
        A = [0.5, 1.0, 0.3]
        b1 = [0.0, 2.0, -1.5]

        u = torch.zeros_like(x)
        for w2_j, A_j, b1_j in zip(w2, A, b1):
            u += w2_j * torch.tanh(A_j * x + b1_j)
        return u

    def benchmark_solution(self, x, t):
        return self.initial_condition(x - self.c*t)
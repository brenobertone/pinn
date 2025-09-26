from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, n_inputs=3, n_outputs=1):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, n_outputs),
        )

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)


class Problem(ABC):
    name: str
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    t_bounds: tuple[float, float]
    f1: Callable
    f2: Callable

    net: PINN
    x_orientation: str = "crescent"
    y_orientation: str = "crescent"

    def __init_subclass__(cls):
        super().__init_subclass__()
        required_attrs = [
            "x_bounds",
            "y_bounds",
            "t_bounds",
            "name",
            "net",
            "f1",
            "f2",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise TypeError(
                    f"Class '{cls.__name__}' is missing attribute: '{attr}'"
                )

    @abstractmethod
    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor: ...

    def boundary_condition(self, xyt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Boundary condition not implemented")


class PeriodicSine2D(Problem):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 4*1.0)
    name = "PeriodicSine2D"
    net = PINN(n_inputs=3, n_outputs=1)
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


class Rarefaction1D(Problem):
    x_bounds = (-6.0, 6.0)
    y_bounds = (-1.5, 1.5)
    t_bounds = (0.0, 4*2.5)
    name = "Rarefaction1D"
    net = PINN(n_inputs=3, n_outputs=1)

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


class Shock1D(Problem):
    x_bounds = (-6.0, 6.0)
    y_bounds = (-1.5, 1.5)
    t_bounds = (0.0, 4*2.5)
    name = "Rarefaction1D"
    net = PINN(n_inputs=3, n_outputs=1)

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


class RiemannOblique(Problem):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 4*0.5)
    name = "RiemannOblique"
    net = PINN(n_inputs=3, n_outputs=1)
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


class Riemann2D(Problem):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 4*1.0 / 12.0)
    name = "Riemann2D"
    net = PINN(n_inputs=3, n_outputs=1)

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


class BuckleyLeverett(Problem):
    x_bounds = (-1.5, 1.5)
    y_bounds = (-1.5, 1.5)
    t_bounds = (0.0, 4*0.5)
    name = "BuckleyLeverett"
    net = PINN(n_inputs=3, n_outputs=1)

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


class NonLinearNonConvexFlow(Problem):
    x_bounds = (-2.0, 2.0)
    y_bounds = (-2.0, 2.0)
    t_bounds = (0.0, 4*1.0)
    name = "NonLinearNonConvexFlow"
    net = PINN(n_inputs=3, n_outputs=1)

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

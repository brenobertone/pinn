import inspect
from abc import ABC, abstractmethod
from typing import Callable

import torch


class Problem(ABC):
    name: str
    spatial_dims: int
    t_bounds: tuple[float, float]

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Only validate concrete classes (not intermediate abstract classes)
        if not inspect.isabstract(cls):
            required_attrs = ["name", "spatial_dims", "t_bounds"]
            for attr in required_attrs:
                if not hasattr(cls, attr):
                    raise TypeError(
                        f"Class '{cls.__name__}' is missing attribute: '{attr}'"
                    )


class Problem1D(Problem):
    spatial_dims = 1
    x_bounds: tuple[float, float]
    f: Callable
    x_orientation: str = "crescent"

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Only validate concrete classes
        if not inspect.isabstract(cls):
            required_attrs = ["x_bounds", "f"]
            for attr in required_attrs:
                if not hasattr(cls, attr):
                    raise TypeError(
                        f"Class '{cls.__name__}' is missing attribute: '{attr}'"
                    )

    @abstractmethod
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor: ...

    def boundary_condition(self, xt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Boundary condition not implemented")


class Problem2D(Problem):
    spatial_dims = 2
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    f1: Callable
    f2: Callable
    x_orientation: str = "crescent"
    y_orientation: str = "crescent"

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Only validate concrete classes
        if not inspect.isabstract(cls):
            required_attrs = ["x_bounds", "y_bounds", "f1", "f2"]
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

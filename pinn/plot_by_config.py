import sys
from itertools import product
from pathlib import Path
from typing import Any, Tuple

import torch

from matplotlib import pyplot as plt

from plotters.plot_three_times import plot_three_times
from plotters.animate import animate_problem
from run_training import get_config_hash

from pinn.problems_definitions import (BuckleyLeverett, NonLinearNonConvexFlow,
                                       PeriodicSine2D, Problem, Rarefaction1D,
                                       Riemann2D, RiemannOblique, Shock1D, Pulse)
from pinn.slope_limiters import (advection_residual_autograd,
                                 advection_residual_mm2,
                                 advection_residual_mm3,
                                 advection_residual_uno)
from pinn.training import Config


def get_config_description(problem: Problem, config: Config) -> str:
    return (
        f"ε={config.epsilon} "
        f"res={config.residual.__closure__[0].cell_contents.__name__}"  # type: ignore
        f"n_int={config.n_points} "
    )


def load_results(
    problem: Problem, config: Config, results_dir: Path = Path("results")
) -> Tuple[Problem, Any]:
    hash_id = get_config_hash(problem, config)
    model_path = results_dir / f"model_{hash_id}.pth"
    plot_path = results_dir / f"plot_{hash_id}.png"

    model_state = torch.load(model_path)
    problem.net.load_state_dict(model_state)

    training_history = plt.imread(plot_path)
    return problem, training_history


if __name__ == "__main__":
    problems = [
        PeriodicSine2D,
        Rarefaction1D,
        RiemannOblique,
        Riemann2D,
        BuckleyLeverett,
        NonLinearNonConvexFlow,
        Shock1D,
        Pulse,
    ]

    epsilons = [0.0025]
    n_points = [1024000]
    epochs = [15000]
    residuals = [
        advection_residual_autograd,
        advection_residual_mm2,
        advection_residual_mm3,
        advection_residual_uno,
    ]

    configs = [
        Config(
            epsilon=e,
            n_points=n,
            epochs=ep,
            residual=r,
        )
        for e, n, ep, r in product(epsilons, n_points, epochs, residuals)
    ]

    for config in configs:
        for problem in problems:
            print(f"Plotting {problem.name} with {config}")
            try:
                trained_problem, _ = load_results(problem, config)
            except:
                continue
            hash_id = get_config_hash(problem, config)
            animate_problem(trained_problem, hash_id=hash_id)
            plot_three_times(trained_problem, hash_id=hash_id)
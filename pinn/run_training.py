import hashlib
import json
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from threading import Lock

import matplotlib
import torch

from pinn.problems_definitions import (BuckleyLeverett, NonLinearNonConvexFlow,
                                       PeriodicSine2D, Problem, Rarefaction1D,
                                       Riemann2D, RiemannOblique)
from pinn.slope_limiters import (advection_residual_autograd,
                                 advection_residual_mm2,
                                 advection_residual_mm3,
                                 advection_residual_uno)
from pinn.training import Config, train


def get_config_hash(problem: Problem, config: Config) -> str:
    config_dict = {
        "epsilon": config.epsilon,
        "n_points": config.n_points,
        "epochs": config.epochs,
        "residual": config.residual if isinstance(config.residual, str) else config.residual.__closure__[0].cell_contents.__name__,  # type: ignore
        "problem": problem.name,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


if __name__ == "__main__":
    matplotlib.use("Agg")

    problems = [
        PeriodicSine2D,
        Rarefaction1D,
        RiemannOblique,
        Riemann2D,
        BuckleyLeverett,
        NonLinearNonConvexFlow,
    ]

    epsilons = [0.0025]
    n_points = [128]  # 000, 256000, 512000]
    epochs = [10]
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

    execution_times: dict[tuple[int, int], list[float]] = defaultdict(list)
    lock = Lock()

    def run_training(problem: Problem, config: Config) -> None:
        config_hash = get_config_hash(problem, config)
        config_path = Path(f"results/config_{config_hash}.json")

        if config_path.exists():
            print(f"Skipping {config_path.name}, already processed.")
            return

        start = time.time()
        model, plot = train(problem, config)
        elapsed = time.time() - start

        key = (config.n_points, config.epochs)
        with lock:
            execution_times[key].append(elapsed)
            mean = sum(execution_times[key]) / len(execution_times[key])
            print(
                f"Group {key} | New time: {elapsed:.2f}s | Mean: {mean:.2f}s"
            )

        Path("results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"results/model_{config_hash}.pth")
        plot.savefig(f"results/plot_{config_hash}.png")

        with config_path.open("w") as f:
            json.dump(
                {
                    "epsilon": config.epsilon,
                    "n_points": config.n_points,
                    "epochs": config.epochs,
                    "residual": config.residual.__closure__[0].cell_contents.__name__,  # type: ignore
                    "problem": problem.name,
                },
                f,
                indent=2,
            )

    combinations = list(product(configs, problems))

    for config, problem in combinations:
        run_training(problem(), config)  # type: ignore

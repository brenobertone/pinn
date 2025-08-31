import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import matplotlib
import torch

import time
from collections import defaultdict
from threading import Lock

from pinn.problems_definitions import (BuckleyLeverett, NonLinearNonConvexFlow,
                                       PeriodicSine2D, Problem, Rarefaction1D,
                                       Riemann2D, RiemannOblique)
from pinn.slope_limiters import (advection_residual_autograd,
                                 advection_residual_mm2,
                                 advection_residual_mm3,
                                 advection_residual_uno)
from pinn.training import Config, train
import hashlib


def get_config_hash(problem: Problem, config: Config) -> str:
    config_dict = {
        "epsilon": config.epsilon,
        "delta": config.delta,
        "n_internal": config.n_internal,
        "n_initial_condition": config.n_initial_condition,
        "epochs": config.epochs,
        "residual": config.residual if isinstance(config.residual, str) else config.residual.__closure__[0].cell_contents.__name__,
        "problem": problem.name,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()



if __name__ == "__main__":
    matplotlib.use("Agg")

    problems = [
        # PeriodicSine2D,
        # Rarefaction1D,
        RiemannOblique,
        Riemann2D,
        BuckleyLeverett,
        NonLinearNonConvexFlow,
    ]

    epsilons = [0.0025, 0.0005]
    deltas = [1e-3, 1e-4]
    n_internals = [1000000]
    n_ics = [100000]
    epochs = [30000]
    residuals = [
        advection_residual_autograd,
        advection_residual_mm2,
        advection_residual_mm3,
        advection_residual_uno,
    ]

    configs = [
        Config(
            epsilon=e,
            delta=d,
            n_internal=ni,
            n_initial_condition=nic,
            epochs=ep,
            residual=r,
        )
        for e, d, ni, nic, ep, r in product(
            epsilons, deltas, n_internals, n_ics, epochs, residuals
        )
    ]

    execution_times: dict[tuple[int, int, int], list[float]] = defaultdict(list)
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

        key = (config.n_internal, config.n_initial_condition, config.epochs)
        with lock:
            execution_times[key].append(elapsed)
            mean = sum(execution_times[key]) / len(execution_times[key])
            print(f"Group {key} | New time: {elapsed:.2f}s | Mean: {mean:.2f}s")

        Path("results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"results/model_{config_hash}.pth")
        plot.savefig(f"results/plot_{config_hash}.png")

        with config_path.open("w") as f:
            json.dump({
                "epsilon": config.epsilon,
                "delta": config.delta,
                "n_internal": config.n_internal,
                "n_initial_condition": config.n_initial_condition,
                "epochs": config.epochs,
                "residual": config.residual.__closure__[0].cell_contents.__name__,
                "problem": problem.name,
            }, f, indent=2)


    combinations = list(product(problems, configs))

    for problem, config in combinations:
        run_training(problem(), config)

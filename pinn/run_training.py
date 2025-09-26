import hashlib
import json
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
import torch
import torch.multiprocessing as mp
from threading import Lock
import matplotlib

from pinn.problems_definitions import (
    BuckleyLeverett, NonLinearNonConvexFlow, PeriodicSine2D,
    Problem, Rarefaction1D, Riemann2D, RiemannOblique, Shock1D, Pulse
)
from pinn.slope_limiters import (
    advection_residual_autograd, advection_residual_mm2,
    advection_residual_mm3, advection_residual_uno
)
from pinn.training import Config, train

execution_times: dict[tuple[int, int], list[float]] = defaultdict(list)
lock = Lock()

def get_config_hash(problem: Problem, config: Config) -> str:
    config_dict = {
        "epsilon": config.epsilon,
        "n_points": config.n_points,
        "epochs": config.epochs,
        "residual": config.residual if isinstance(config.residual, str)
                    else config.residual_fn.__name__,
        "problem": problem.name,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def run_training_parallel(problem_class, config: Config, device_id: int):
    # Assign device
    global device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    problem = problem_class()
    problem.net.to(device)
    
    config_hash = get_config_hash(problem, config)
    config_path = Path(f"results/config_{config_hash}.json")

    if config_path.exists():
        print(f"Skipping {config_path.name}, already processed.")
        return

    start = time.time()
    model, plot = train(problem, config, device)
    elapsed = time.time() - start

    key = (config.n_points, config.epochs)
    with lock:
        execution_times[key].append(elapsed)
        mean = sum(execution_times[key]) / len(execution_times[key])
        print(f"Group {key} | New time: {elapsed:.2f}s | Mean: {mean:.2f}s")

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"results/model_{config_hash}.pth")
    plot.savefig(f"results/plot_{config_hash}.png")

    with config_path.open("w") as f:
        json.dump(
            {
                "epsilon": config.epsilon,
                "n_points": config.n_points,
                "epochs": config.epochs,
                "residual": config.residual_fn.__name__,
                "problem": problem.name,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    matplotlib.use("Agg")

    problems = [
        PeriodicSine2D, Rarefaction1D, RiemannOblique, Riemann2D,
        BuckleyLeverett, NonLinearNonConvexFlow, Shock1D, Pulse
    ]

    epsilons = [0.0025]
    n_points = [128000, 256000, 512000, 1024000]
    epochs = [15000]
    residuals = [
        advection_residual_autograd,
        advection_residual_mm2,
        advection_residual_mm3,
        advection_residual_uno,
    ]

    configs = [
        Config(epsilon=e, n_points=n, epochs=ep, residual=r)
        for e, n, ep, r in product(epsilons, n_points, epochs, residuals)
    ]

    combinations = list(product(configs, problems))

    processes = []
    for i, (config, problem_class) in enumerate(combinations):
        device_id = i % 2  # alternate between GPU 0 and 1
        p = mp.Process(target=run_training_parallel, args=(problem_class, config, device_id))
        p.start()
        processes.append(p)

        # Run only 2 processes at a time
        if len(processes) == 2:
            for proc in processes:
                proc.join()
            processes = []

    # Join any remaining processes
    for proc in processes:
        proc.join()

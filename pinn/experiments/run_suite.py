from itertools import product

import matplotlib

from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.experiments.tracker import ExperimentTracker
from pinn.problems.problems_1d import (
    Burgers1D,
    LinearAdvection1D,
    Rarefaction1DPure,
    Shock1DPure,
)
from pinn.problems.problems_2d import (
    BuckleyLeverett,
    NonLinearNonConvexFlow,
    PeriodicSine2D,
    Pulse,
    Rarefaction1D,
    Riemann2D,
    RiemannOblique,
    Shock1D,
)

matplotlib.use("Agg")


if __name__ == "__main__":
    tracker = ExperimentTracker()

    problems_2d = [
        PeriodicSine2D,
        Rarefaction1D,
        RiemannOblique,
        Riemann2D,
        BuckleyLeverett,
        NonLinearNonConvexFlow,
        Shock1D,
        Pulse,
    ]

    problems_1d = [
        Burgers1D,
        Shock1DPure,
        Rarefaction1DPure,
        LinearAdvection1D,
    ]

    architectures = [
        NetworkConfig([20, 20, 20, 20, 20], "relu"),  # Original
        NetworkConfig([50, 50, 50], "tanh"),  # Wider/shallower
        NetworkConfig([20] * 10, "relu"),  # Deeper
    ]

    epsilons = [0.0025]
    n_points_2d = [512000]
    n_points_1d = [10000]
    epochs = [5000]
    residuals = ["autograd", "mm2", "mm3", "uno"]
    optimizers = ["adamw"]

    for problem_cls in problems_2d:
        for arch, e, n, ep, r, opt in product(
            architectures, epsilons, n_points_2d, epochs, residuals, optimizers
        ):
            problem = problem_cls()
            model = NetworkConfig(
                arch.layers, arch.activation, n_inputs=3, n_outputs=1
            ).build()
            config = Config(
                epsilon=e,
                n_points=n,
                epochs=ep,
                residual_method=r,
                optimizer=opt,
            )

            model, fig, metrics = train(problem, model, config)
            tracker.log_run(problem, config, arch, model, metrics, fig)

    for problem_cls in problems_1d:
        for arch, e, n, ep, r, opt in product(
            architectures,
            epsilons,
            n_points_1d,
            epochs,
            ["autograd", "mm2"],
            optimizers,
        ):
            problem = problem_cls()
            model = NetworkConfig(
                arch.layers, arch.activation, n_inputs=2, n_outputs=1
            ).build()
            config = Config(
                epsilon=e,
                n_points=n,
                epochs=ep,
                residual_method=r,
                optimizer=opt,
            )

            model, fig, metrics = train(problem, model, config)
            tracker.log_run(problem, config, arch, model, metrics, fig)

    print("\n=== Training Complete ===")
    df = tracker.load_experiments()
    print(f"\nTotal experiments: {len(df)}")
    print("\nBest results by problem:")
    for prob in df["problem_name"].unique():
        best = tracker.get_best(problem=prob)
        print(f"{prob}: loss={best['final_loss']:.5e}, time={best['training_time']:.2f}s")

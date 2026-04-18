"""
Quick test of all 1D problems with minimal training.
Fast execution for testing the refactored system.
"""

from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.experiments.tracker import ExperimentTracker
from pinn.problems.problems_1d import (
    Burgers1D,
    LinearAdvection1D,
    Rarefaction1DPure,
    Shock1DPure,
)


def main():
    print("=" * 60)
    print("Quick Test: All 1D Problems")
    print("=" * 60)

    tracker = ExperimentTracker()

    # All 1D problems
    problems = [
        Burgers1D(),
        Shock1DPure(),
        Rarefaction1DPure(),
        LinearAdvection1D(),
    ]

    # Small architecture for fast training
    arch = NetworkConfig([20, 20, 20], "tanh", n_inputs=2, n_outputs=1)

    # Fast config
    config = Config(
        epsilon=0.001,
        n_points=2500,  # 50x50 grid (small)
        epochs=1000,  # Few epochs
        residual_method="autograd",  # Fastest method
        optimizer="adamw",
        learning_rate=1e-3,
    )

    print(f"\nConfiguration:")
    print(f"  Architecture: {arch.layers}")
    print(f"  Points: {config.n_points} ({int(config.n_points**0.5)}x{int(config.n_points**0.5)} grid)")
    print(f"  Epochs: {config.epochs}")
    print(f"  Residual: {config.residual_method}")
    print(f"  Optimizer: {config.optimizer}")
    print()

    results = []
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Training: {problem.name}")
        print(f"{'='*60}")

        model = arch.build()
        model, fig, metrics = train(problem, model, config)
        exp_id = tracker.log_run(problem, config, arch, model, metrics, fig)

        results.append({
            "problem": problem.name,
            "exp_id": exp_id,
            "loss": metrics["final_loss"],
            "time": metrics["training_time"],
        })

        print(f"✓ Completed: {problem.name}")
        print(f"  Loss: {metrics['final_loss']:.5e}")
        print(f"  Time: {metrics['training_time']:.2f}s")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Problem':<25} {'Loss':>12} {'Time':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['problem']:<25} {r['loss']:>12.5e} {r['time']:>7.2f}s")

    print(f"\n✓ All experiments saved to results/")
    print(f"✓ View with: poetry run python -c \"from pinn.experiments.tracker import ExperimentTracker; print(ExperimentTracker().load_experiments())\"")


if __name__ == "__main__":
    main()

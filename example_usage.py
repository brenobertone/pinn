"""
Example usage of the refactored PINN framework.

This script demonstrates:
1. Training a problem with different architectures
2. Comparing results programmatically
3. Generating visualizations
"""

from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.experiments.tracker import ExperimentTracker
from pinn.problems.problems_1d import Burgers1D
from pinn.problems.problems_2d import Riemann2D
from pinn.visualization.visualizer import Visualizer


def main():
    tracker = ExperimentTracker()

    # Example 1: Train 2D problem with different architectures
    print("=" * 60)
    print("Example 1: Testing different architectures on Riemann2D")
    print("=" * 60)

    problem_2d = Riemann2D()

    architectures = [
        NetworkConfig([20, 20, 20, 20, 20], "relu"),  # Original
        NetworkConfig([50, 50, 50], "tanh"),  # Wider
        NetworkConfig([15, 15, 15, 15, 15, 15, 15], "relu"),  # Deeper
    ]

    exp_ids = []
    for arch in architectures:
        print(f"\nTraining with architecture: {arch}")
        model = arch.build()
        config = Config(
            epsilon=0.0025,
            n_points=125000,
            epochs=1000,
            residual_method="mm2",
            optimizer="adamw",
        )

        model, fig, metrics = train(problem_2d, model, config)
        exp_id = tracker.log_run(problem_2d, config, arch, model, metrics, fig)
        exp_ids.append(exp_id)

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    viz = Visualizer(exp_ids)
    comparison = viz.comparison_table()
    print(comparison.to_string())

    # Example 2: Test different slope limiters
    print("\n" + "=" * 60)
    print("Example 2: Testing slope limiters on Burgers1D")
    print("=" * 60)

    problem_1d = Burgers1D()
    arch = NetworkConfig([20, 20, 20], "relu")

    exp_ids_1d = []
    for method in ["autograd", "mm2"]:
        print(f"\nTraining with residual method: {method}")
        model = NetworkConfig(
            arch.layers, arch.activation, n_inputs=2, n_outputs=1
        ).build()
        config = Config(
            epsilon=0.001,
            n_points=10000,
            epochs=1000,
            residual_method=method,
        )

        model, fig, metrics = train(problem_1d, model, config)
        exp_id = tracker.log_run(problem_1d, config, arch, model, metrics, fig)
        exp_ids_1d.append(exp_id)

    # Example 3: Get best performing configuration
    print("\n" + "=" * 60)
    print("Best configurations by problem")
    print("=" * 60)

    df = tracker.load_experiments()
    for problem in df["problem_name"].unique():
        best = tracker.get_best(problem=problem)
        print(f"\n{problem}:")
        print(f"  Experiment ID: {best['exp_id']}")
        print(f"  Architecture: {best['network_layers']}")
        print(f"  Residual: {best['residual_method']}")
        print(f"  Final Loss: {best['final_loss']:.5e}")
        print(f"  Training Time: {best['training_time']:.2f}s")

    # Example 4: Generate animations for best results
    print("\n" + "=" * 60)
    print("Generating animations")
    print("=" * 60)

    if exp_ids:
        best_2d = tracker.get_best(problem="Riemann2D")
        viz_2d = Visualizer([best_2d["exp_id"]])
        print(f"Animating 2D solution for {best_2d['exp_id']}...")
        viz_2d.animate_solution_2d(best_2d["exp_id"], problem_2d, steps=50)

    if exp_ids_1d:
        best_1d = tracker.get_best(problem="Burgers1D")
        viz_1d = Visualizer([best_1d["exp_id"]])
        print(f"Animating 1D solution for {best_1d['exp_id']}...")
        viz_1d.animate_solution_1d(best_1d["exp_id"], problem_1d, steps=50)

    print("\n" + "=" * 60)
    print("Done! Check results/ directory for outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()

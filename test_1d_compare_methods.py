"""
Compare autograd vs mm2 slope limiter on 1D problems.
Quick test with minimal points.
"""

from itertools import product

from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.experiments.tracker import ExperimentTracker
from pinn.problems.problems_1d import Burgers1D, Shock1DPure


def main():
    print("=" * 60)
    print("Compare: autograd vs mm2 on 1D problems")
    print("=" * 60)

    tracker = ExperimentTracker()

    problems = [
        Burgers1D(),
        Shock1DPure(),
    ]

    methods = ["autograd", "mm2"]

    # Small architecture
    arch = NetworkConfig([20, 20, 20], "relu", n_inputs=2, n_outputs=1)

    results = []
    for problem, method in product(problems, methods):
        print(f"\n{'='*60}")
        print(f"{problem.name} with {method}")
        print(f"{'='*60}")

        config = Config(
            epsilon=0.001,
            n_points=2500,  # 50x50
            epochs=100,
            residual_method=method,
            optimizer="adamw",
        )

        model = arch.build()
        model, fig, metrics = train(problem, model, config)
        exp_id = tracker.log_run(problem, config, arch, model, metrics, fig)

        results.append({
            "problem": problem.name,
            "method": method,
            "exp_id": exp_id,
            "loss": metrics["final_loss"],
            "time": metrics["training_time"],
        })

        print(f"✓ Loss: {metrics['final_loss']:.5e}, Time: {metrics['training_time']:.2f}s")

    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"{'Problem':<20} {'Method':<10} {'Loss':>12} {'Time':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['problem']:<20} {r['method']:<10} {r['loss']:>12.5e} {r['time']:>7.2f}s")

    print(f"\n{'='*60}")
    print("Winner by Problem (lowest loss)")
    print(f"{'='*60}")
    for problem in [Burgers1D().name, Shock1DPure().name]:
        prob_results = [r for r in results if r["problem"] == problem]
        best = min(prob_results, key=lambda x: x["loss"])
        print(f"{problem:<20} → {best['method']} (loss={best['loss']:.5e})")


if __name__ == "__main__":
    main()

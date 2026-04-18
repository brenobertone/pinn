#!/usr/bin/env python3
"""
Interactive PINN training script.
Prompts for all configuration options with defaults.
"""

import sys

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


def get_input(prompt, default=None, type_fn=str):
    """Get user input with default value."""
    if default is not None:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    value = input(full_prompt).strip()
    if not value and default is not None:
        return default

    try:
        return type_fn(value)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default


def get_multi_choice(prompt, options, default_indices=None):
    """Get multiple choice selection."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")

    if default_indices:
        default_str = ",".join(str(i+1) for i in default_indices)
        print(f"\nEnter numbers separated by commas [default: {default_str}]")
        print("Or enter 'all' to select all options")
    else:
        print("\nEnter numbers separated by commas, or 'all'")

    value = input("> ").strip().lower()

    if value == "all":
        return list(range(len(options)))

    if not value and default_indices is not None:
        return default_indices

    try:
        indices = [int(x.strip()) - 1 for x in value.split(",")]
        return [i for i in indices if 0 <= i < len(options)]
    except:
        if default_indices is not None:
            print(f"Invalid input. Using default.")
            return default_indices
        return []


def main():
    print("=" * 70)
    print("PINN Interactive Training")
    print("=" * 70)

    # Select problem dimension
    print("\n1D or 2D problems?")
    print("  1. 1D problems")
    print("  2. 2D problems")
    dim_choice = get_input("Select", default="1", type_fn=int)

    # Select problems
    if dim_choice == 1:
        problems_1d = [
            ("Burgers1D", Burgers1D),
            ("Shock1DPure", Shock1DPure),
            ("Rarefaction1DPure", Rarefaction1DPure),
            ("LinearAdvection1D", LinearAdvection1D),
        ]
        problem_names = [p[0] for p in problems_1d]
        selected = get_multi_choice(
            "Select 1D problems to train:",
            problem_names,
            default_indices=[0]
        )
        problems = [problems_1d[i][1]() for i in selected]
        n_inputs = 2
        default_points = 10000
        points_desc = "points (sqrt gives grid resolution)"
    else:
        problems_2d = [
            ("PeriodicSine2D", PeriodicSine2D),
            ("Riemann2D", Riemann2D),
            ("RiemannOblique", RiemannOblique),
            ("Shock1D", Shock1D),
            ("Rarefaction1D", Rarefaction1D),
            ("Pulse", Pulse),
            ("BuckleyLeverett", BuckleyLeverett),
            ("NonLinearNonConvexFlow", NonLinearNonConvexFlow),
        ]
        problem_names = [p[0] for p in problems_2d]
        selected = get_multi_choice(
            "Select 2D problems to train:",
            problem_names,
            default_indices=[1]
        )
        problems = [problems_2d[i][1]() for i in selected]
        n_inputs = 3
        default_points = 125000
        points_desc = "points (cube root gives grid resolution)"

    if not problems:
        print("No problems selected. Exiting.")
        return

    print(f"\nSelected: {', '.join(p.name for p in problems)}")

    # Architecture config
    print("\n" + "=" * 70)
    print("Network Architecture")
    print("=" * 70)

    arch_preset = get_input(
        "\nUse preset architecture?\n"
        "  1. Small (3 layers × 20 neurons)\n"
        "  2. Default (5 layers × 20 neurons)\n"
        "  3. Wide (3 layers × 50 neurons)\n"
        "  4. Deep (10 layers × 20 neurons)\n"
        "  5. Custom\n"
        "Select",
        default="2",
        type_fn=int
    )

    presets = {
        1: [20, 20, 20],
        2: [20, 20, 20, 20, 20],
        3: [50, 50, 50],
        4: [20] * 10,
    }

    if arch_preset in presets:
        layers = presets[arch_preset]
    else:
        layers_input = get_input(
            "Enter layer sizes (comma-separated)",
            default="20,20,20,20,20"
        )
        layers = [int(x.strip()) for x in layers_input.split(",")]

    activation = get_input(
        "Activation function (relu/tanh/sigmoid)",
        default="relu"
    ).lower()

    print(f"\nArchitecture: {layers} with {activation} activation")

    # Training config
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)

    epsilon = get_input(
        f"\nViscosity coefficient epsilon",
        default=0.0025,
        type_fn=float
    )

    n_points = get_input(
        f"Number of {points_desc}",
        default=default_points,
        type_fn=int
    )

    epochs = get_input(
        "Number of epochs",
        default=5000,
        type_fn=int
    )

    # Residual method
    if dim_choice == 1:
        print("\nResidual computation method:")
        print("  1. autograd (pure autodiff)")
        print("  2. mm2 (MinMod2 slope limiter)")
        method_choice = get_input("Select", default="1", type_fn=int)
        residual_method = "autograd" if method_choice == 1 else "mm2"
    else:
        print("\nResidual computation method:")
        print("  1. autograd (pure autodiff)")
        print("  2. mm2 (MinMod2 slope limiter)")
        print("  3. mm3 (MinMod3 slope limiter)")
        print("  4. uno (UNO scheme)")
        method_choice = get_input("Select", default="1", type_fn=int)
        methods = ["autograd", "mm2", "mm3", "uno"]
        residual_method = methods[method_choice - 1] if 1 <= method_choice <= 4 else "autograd"

    optimizer = get_input(
        "Optimizer (adam/adamw/rmsprop)",
        default="adamw"
    ).lower()

    learning_rate = get_input(
        "Learning rate",
        default=1e-3,
        type_fn=float
    )

    # Summary
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Problems: {', '.join(p.name for p in problems)}")
    print(f"Architecture: {layers} ({activation})")
    print(f"Epsilon: {epsilon}")
    print(f"Points: {n_points}")
    print(f"Epochs: {epochs}")
    print(f"Residual: {residual_method}")
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70)

    proceed = get_input("\nProceed with training? (y/n)", default="y").lower()
    if proceed != "y":
        print("Cancelled.")
        return

    # Create config
    arch = NetworkConfig(layers, activation, n_inputs=n_inputs, n_outputs=1)
    config = Config(
        epsilon=epsilon,
        n_points=n_points,
        epochs=epochs,
        residual_method=residual_method,
        optimizer=optimizer,
        learning_rate=learning_rate,
    )

    tracker = ExperimentTracker()
    results = []

    # Train
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] Training {problem.name}...")
        print("-" * 70)

        model = arch.build()
        model, fig, metrics = train(problem, model, config)
        exp_id = tracker.log_run(problem, config, arch, model, metrics, fig)

        results.append({
            "problem": problem.name,
            "exp_id": exp_id,
            "loss": metrics["final_loss"],
            "time": metrics["training_time"],
        })

        print(f"✓ {problem.name} complete")
        print(f"  Experiment ID: {exp_id}")
        print(f"  Final Loss: {metrics['final_loss']:.5e}")
        print(f"  Time: {metrics['training_time']:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("Training Complete - Summary")
    print("=" * 70)
    print(f"{'Problem':<30} {'Loss':>12} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['problem']:<30} {r['loss']:>12.5e} {r['time']:>7.2f}s")

    print(f"\n✓ All results saved to results/")
    print(f"✓ Models: results/model_<exp_id>.pth")
    print(f"✓ Plots: results/plot_<exp_id>.png")
    print(f"✓ Database: results/experiments.jsonl")

    # Ask about visualization
    print("\n" + "=" * 70)
    visualize = get_input("Generate animations now? (y/n)", default="n").lower()
    if visualize == "y":
        print("\nRun: python visualize_interactive.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

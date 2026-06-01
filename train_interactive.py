#!/usr/bin/env python3
"""
Interactive PINN training script.
Prompts for all configuration options with defaults.
"""

import itertools
import os
import shutil
import sys
from datetime import datetime

from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.experiments.tracker import ExperimentTracker
from pinn.problems.problems_1d import (
    AdvectionTanh1D,
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


def get_input_list(prompt, default=None, type_fn=str):
    """Get user input as a list of values."""
    if default is not None:
        if isinstance(default, list):
            default_str = ",".join(map(str, default))
        else:
            default_str = str(default)
        full_prompt = f"{prompt} [{default_str}]: "
    else:
        full_prompt = f"{prompt}: "

    value = input(full_prompt).strip()
    if not value and default is not None:
        return default if isinstance(default, list) else [default]

    try:
        return [type_fn(x.strip()) for x in value.split(",")]
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default if isinstance(default, list) else [default]


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
    dim_choice = get_input("Select", default=1, type_fn=int)

    # Select problems
    if dim_choice == 1:
        problems_1d = [
            ("Burgers1D", Burgers1D),
            ("Shock1DPure", Shock1DPure),
            ("Rarefaction1DPure", Rarefaction1DPure),
            ("LinearAdvection1D", LinearAdvection1D),
            ("AdvectionTanh1D", AdvectionTanh1D),
        ]
        problem_names = [p[0] for p in problems_1d]
        selected = get_multi_choice(
            "Select 1D problems to train:",
            problem_names,
            default_indices=[0]
        )
        problems = [problems_1d[i][1]() for i in selected]
        n_inputs = 2
        default_points = 5000
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
        "  1. Small (3 layers × 32 neurons)\n"
        "  2. Default (5 layers × 20 neurons)\n"
        "  3. Wide (3 layers × 50 neurons)\n"
        "  4. Deep (10 layers × 20 neurons)\n"
        "  5. Custom\n"
        "Select",
        default="1",
        type_fn=int
    )

    presets = {
        1: [32, 32, 32],
        2: [20, 20, 20, 20, 20],
        3: [50, 50, 50],
        4: [20] * 10,
    }

    if arch_preset in presets:
        layers = presets[arch_preset]
    else:
        layers_input = get_input(
            "Enter layer sizes (comma-separated)",
            default="32,32,32"
        )
        layers = [int(x.strip()) for x in layers_input.split(",")]

    activation = get_input(
        "Activation function (relu/tanh/sigmoid)",
        default="tanh"
    ).lower()

    # Characteristic transform (only for 1D problems)
    use_characteristic = False
    if dim_choice == 1:
        char_choice = get_input(
            "\nUse characteristic transform (x - c*t) for advection? (y/n)",
            default="n"
        ).lower()
        use_characteristic = (char_choice == "y")

    if use_characteristic:
        print(f"\nArchitecture: {layers} with {activation} activation + characteristic transform")
        print("  (wave speed c will be read from each problem definition)")
    else:
        print(f"\nArchitecture: {layers} with {activation} activation")

    # Training config
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)

    epsilons = get_input_list(
        f"\nViscosity coefficients epsilon (comma-separated)",
        default=0.0,
        type_fn=float
    )

    points_list = get_input_list(
        f"Number of {points_desc} (comma-separated)",
        default=default_points,
        type_fn=int
    )

    epochs_list = get_input_list(
        "Number of epochs (comma-separated)",
        default=50000,
        type_fn=int
    )

    # Residual methods selection
    if dim_choice == 1:
        methods_options = ["autograd", "mm2"]
    else:
        methods_options = ["autograd", "mm2", "mm3", "uno"]

    selected_methods_indices = get_multi_choice(
        "Select residual computation methods:",
        methods_options,
        default_indices=[0]
    )
    residual_methods = [methods_options[i] for i in selected_methods_indices]

    # Sampling method
    if "autograd" in residual_methods:
        print("\nSampling strategy (only applies to 'autograd' method):")
        print("  1. uniform (standard mesh)")
        print("  2. latin_hypercube (LHS)")
        sampling_choice = get_input("Select", default="2", type_fn=int)
        sampling_method = "uniform" if sampling_choice == 1 else "latin_hypercube"
    else:
        sampling_method = "uniform"

    optimizers = get_input_list(
        "Optimizers (adam/adamw/rmsprop, comma-separated)",
        default="adam",
        type_fn=str
    )
    optimizers = [opt.lower() for opt in optimizers]

    learning_rates = get_input_list(
        "Learning rates (comma-separated)",
        default=1e-3,
        type_fn=float
    )

    # RBA configuration
    rba_selection = get_multi_choice(
        "Residual-Based Attention (RBA) modes:",
        ["Disabled", "Enabled"],
        default_indices=[0]
    )

    rba_configs = []
    if 0 in rba_selection:
        rba_configs.append((False, 0.0))
    if 1 in rba_selection:
        rba_etas = get_input_list(
            "RBA update rates (eta) for enabled mode (comma-separated)",
            default=0.001,
            type_fn=float
        )
        for eta in rba_etas:
            rba_configs.append((True, eta))

    # Generate Cartesian product of all parameters
    combinations = list(itertools.product(
        epsilons,
        points_list,
        epochs_list,
        residual_methods,
        optimizers,
        learning_rates,
        rba_configs
    ))

    # Summary
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Problems: {', '.join(p.name for p in problems)}")
    print(f"Total Combinations per Problem: {len(combinations)}")
    print(f"Total Experiments: {len(problems) * len(combinations)}")
    print("-" * 70)
    print(f"Epsilons: {epsilons}")
    print(f"Points: {points_list}")
    print(f"Epochs: {epochs_list}")
    print(f"Residual Methods: {residual_methods}")
    print(f"Optimizers: {optimizers}")
    print(f"Learning Rates: {learning_rates}")
    print(f"RBA Configs: {rba_configs}")
    arch_desc = f"{layers} ({activation})"
    if use_characteristic:
        arch_desc += " + characteristic"
    print(f"Architecture: {arch_desc}")
    print(f"Sampling: {sampling_method} (if autograd)")
    print("=" * 70)

    proceed = get_input("\nProceed with training? (y/n)", default="y").lower()
    if proceed != "y":
        print("Cancelled.")
        return

    tracker = ExperimentTracker()
    results = []

    # Train
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    total_runs = len(problems) * len(combinations)
    run_idx = 0

    for problem in problems:
        for combo in combinations:
            run_idx += 1
            eps, pts, eps_count, res_method, opt, lr, (rba_on, rba_val) = combo
            
            print(f"\n[{run_idx}/{total_runs}] Problem: {problem.name}")
            print(f"  Params: eps={eps}, points={pts}, epochs={eps_count}, res={res_method}, opt={opt}, lr={lr}, rba={rba_on}({rba_val})")
            print("-" * 70)

            # Enforce uniform sampling for non-autograd methods
            current_sampling = sampling_method if res_method == "autograd" else "uniform"

            config = Config(
                epsilon=eps,
                n_points=pts,
                epochs=eps_count,
                residual_method=res_method,
                optimizer=opt,
                learning_rate=lr,
                sampling_method=current_sampling,
                rba_enabled=rba_on,
                rba_eta=rba_val,
            )

            # Get wave speed from problem if using characteristic transform
            characteristic_c = getattr(problem, 'c', 1.0) if use_characteristic else 1.0

            # Create architecture config for this problem
            arch = NetworkConfig(
                layers,
                activation,
                n_inputs=n_inputs,
                n_outputs=1,
                use_characteristic=use_characteristic,
                characteristic_c=characteristic_c,
            )

            model = arch.build()
            
            # Pre-generate ID to use for intermediate artifacts like heatmaps
            exp_id = tracker.generate_id(problem, config, arch)
            
            model, fig, metrics = train(problem, model, config, exp_id=exp_id)
            tracker.log_run(problem, config, arch, model, metrics, fig)

            results.append({
                "problem": problem.name,
                "exp_id": exp_id,
                "loss": metrics["final_loss"],
                "time": metrics["training_time"],
                "params": f"eps={eps}, res={res_method}, rba={rba_on}({rba_val})"
            })

            print(f"✓ {problem.name} complete (ID: {exp_id})")

            # Zip and download results between cases
            zip_name = f"results_checkpoint_{run_idx}"
            zip_path = f"{zip_name}.zip"
            shutil.make_archive(zip_name, 'zip', 'results')
            
            if 'google.colab' in sys.modules:
                try:
                    from google.colab import files
                    files.download(zip_path)
                except Exception as e:
                    print(f"  ✗ Colab auto-download failed: {e}")

    # Final Summary Table
    print("\n" + "=" * 70)
    print("Training Complete - Summary")
    print("=" * 70)
    print(f"{'Problem':<20} {'Params':<40} {'Loss':>12}")
    print("-" * 70)
    for r in results:
        param_summary = r['params'][:37] + "..." if len(r['params']) > 40 else r['params']
        print(f"{r['problem']:<20} {param_summary:<40} {r['loss']:>12.5e}")

    print(f"\n✓ All results saved to results/")
    print(f"✓ Models: results/model_<exp_id>.pth")
    print(f"✓ Plots: results/plot_<exp_id>.png")
    print(f"✓ Database: results/experiments.jsonl")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


#!/usr/bin/env python3
"""
Interactive visualization script.
Generates animations from tracked experiments.
"""

import sys

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
from pinn.visualization.visualizer import Visualizer


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


def main():
    print("=" * 70)
    print("PINN Interactive Visualization")
    print("=" * 70)

    tracker = ExperimentTracker()
    df = tracker.load_experiments()

    if df.empty:
        print("\nNo experiments found.")
        print("Run: python train_interactive.py first")
        return

    print(f"\nFound {len(df)} experiments")

    # Problem instances
    problem_map_1d = {
        "Burgers1D": Burgers1D(),
        "Shock1DPure": Shock1DPure(),
        "Rarefaction1DPure": Rarefaction1DPure(),
        "LinearAdvection1D": LinearAdvection1D(),
    }

    problem_map_2d = {
        "PeriodicSine2D": PeriodicSine2D(),
        "Riemann2D": Riemann2D(),
        "RiemannOblique": RiemannOblique(),
        "Shock1D": Shock1D(),
        "Rarefaction1D": Rarefaction1D(),
        "Pulse": Pulse(),
        "BuckleyLeverett": BuckleyLeverett(),
        "NonLinearNonConvexFlow": NonLinearNonConvexFlow(),
    }

    # Show available problems
    print("\n" + "=" * 70)
    print("Available Problems")
    print("=" * 70)

    df_1d = df[df["problem_spatial_dims"] == 1]
    df_2d = df[df["problem_spatial_dims"] == 2]

    if not df_1d.empty:
        print("\n1D Problems:")
        for name in df_1d["problem_name"].unique():
            count = len(df_1d[df_1d["problem_name"] == name])
            best = df_1d[df_1d["problem_name"] == name]["final_loss"].min()
            print(f"  - {name}: {count} experiments, best loss={best:.5e}")

    if not df_2d.empty:
        print("\n2D Problems:")
        for name in df_2d["problem_name"].unique():
            count = len(df_2d[df_2d["problem_name"] == name])
            best = df_2d[df_2d["problem_name"] == name]["final_loss"].min()
            print(f"  - {name}: {count} experiments, best loss={best:.5e}")

    # Select dimension
    print("\n" + "=" * 70)
    print("Visualize 1D or 2D problems?")
    print("  1. 1D problems")
    print("  2. 2D problems")
    dim_choice = get_input("Select", default="1", type_fn=int)

    if dim_choice == 1:
        df_selected = df_1d
        problem_map = problem_map_1d
        animate_fn = lambda viz, exp_id, prob: viz.animate_solution_1d(
            exp_id, prob, steps=get_input("Number of frames", default=100, type_fn=int)
        )
    else:
        df_selected = df_2d
        problem_map = problem_map_2d
        animate_fn = lambda viz, exp_id, prob: viz.animate_solution_2d(
            exp_id, prob, steps=get_input("Number of frames", default=100, type_fn=int)
        )

    if df_selected.empty:
        print(f"\nNo {'1D' if dim_choice == 1 else '2D'} experiments found.")
        return

    # Select problems to visualize
    available_problems = df_selected["problem_name"].unique().tolist()
    print(f"\nAvailable problems: {', '.join(available_problems)}")

    visualize_all = get_input(
        "\nVisualize all problems? (y/n)",
        default="y"
    ).lower()

    if visualize_all == "y":
        selected_problems = available_problems
    else:
        print("\nEnter problem names (comma-separated):")
        selection = get_input("> ", default=available_problems[0])
        selected_problems = [p.strip() for p in selection.split(",")]

    # Filter to only best experiments
    use_best = get_input(
        "\nUse only best experiment per problem? (y/n)",
        default="y"
    ).lower()

    # Generate animations
    print("\n" + "=" * 70)
    print("Generating Animations")
    print("=" * 70)

    for problem_name in selected_problems:
        if problem_name not in problem_map:
            print(f"\n✗ {problem_name}: Not available")
            continue

        try:
            if use_best == "y":
                best = tracker.get_best(problem=problem_name)
                exp_ids = [best["exp_id"]]
                print(f"\n{problem_name}:")
                print(f"  Using best: {best['exp_id']}")
                print(f"  Loss: {best['final_loss']:.5e}")
                print(f"  Method: {best['residual_method']}")
            else:
                exp_ids = df_selected[
                    df_selected["problem_name"] == problem_name
                ]["exp_id"].tolist()
                print(f"\n{problem_name}: {len(exp_ids)} experiments")

            problem = problem_map[problem_name]
            viz = Visualizer(exp_ids)

            for exp_id in exp_ids:
                animate_fn(viz, exp_id, problem)
                print(f"  ✓ Saved: results/videos/animate_{'1d' if dim_choice == 1 else '2d'}_{exp_id}.mp4")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print("Check results/videos/ for animations")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Interactive visualization script.
Generates animations from tracked experiments.
"""

import sys
from pathlib import Path

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
    problem_map = {
        "Burgers1D": Burgers1D(),
        "Shock1DPure": Shock1DPure(),
        "Rarefaction1DPure": Rarefaction1DPure(),
        "LinearAdvection1D": LinearAdvection1D(),
        "AdvectionTanh1D": AdvectionTanh1D(),
        "PeriodicSine2D": PeriodicSine2D(),
        "Riemann2D": Riemann2D(),
        "RiemannOblique": RiemannOblique(),
        "Shock1D": Shock1D(),
        "Rarefaction1D": Rarefaction1D(),
        "Pulse": Pulse(),
        "BuckleyLeverett": BuckleyLeverett(),
        "NonLinearNonConvexFlow": NonLinearNonConvexFlow(),
    }

    print("\nSelect mode:")
    print("  1. Create all videos")
    print("  2. Create only missing videos")
    mode_choice = get_input("Select", default="2", type_fn=int)

    num_frames = get_input("Number of frames", default=100, type_fn=int)
    fps = get_input("Frames per second (FPS)", default=15, type_fn=int)

    videos_dir = Path("results/videos")
    videos_dir.mkdir(exist_ok=True, parents=True)

    viz = Visualizer(df["exp_id"].tolist())

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    print("\n" + "=" * 70)
    print("Processing Experiments")
    print("=" * 70)

    for _, record in df.iterrows():
        exp_id = record["exp_id"]
        dims = record["problem_spatial_dims"]
        problem_name = record["problem_name"]
        
        video_filename = f"animate_{dims}d_{exp_id}.mp4"
        video_path = videos_dir / video_filename

        if mode_choice == 2 and video_path.exists():
            skipped_count += 1
            continue

        if problem_name not in problem_map:
            print(f"  ✗ {exp_id}: Problem '{problem_name}' not found in map")
            failed_count += 1
            continue

        problem = problem_map[problem_name]
        print(f"  → Generating: {video_filename} ({problem_name})")
        
        try:
            if dims == 1:
                viz.animate_solution_1d(exp_id, problem, steps=num_frames, fps=fps)
            else:
                viz.animate_solution_2d(exp_id, problem, steps=num_frames, fps=fps)
            processed_count += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            failed_count += 1

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Failed:    {failed_count}")
    print("\nCheck results/videos/ for animations")


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

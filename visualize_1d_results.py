"""
Visualize results from 1D experiments.
Generates animations for best performing models.
"""

from pinn.experiments.tracker import ExperimentTracker
from pinn.problems.problems_1d import (
    Burgers1D,
    LinearAdvection1D,
    Rarefaction1DPure,
    Shock1DPure,
)
from pinn.visualization.visualizer import Visualizer


def main():
    tracker = ExperimentTracker()
    df = tracker.load_experiments()

    if df.empty:
        print("No experiments found. Run test_1d_quick.py first.")
        return

    # Filter 1D experiments
    df_1d = df[df["problem_spatial_dims"] == 1]

    if df_1d.empty:
        print("No 1D experiments found.")
        return

    print("=" * 60)
    print("1D Experiment Results")
    print("=" * 60)
    print(f"\nTotal 1D experiments: {len(df_1d)}")
    print(f"\n{df_1d[['problem_name', 'residual_method', 'final_loss', 'training_time']].to_string()}")

    # Problem instances for visualization
    problem_map = {
        "Burgers1D": Burgers1D(),
        "Shock1DPure": Shock1DPure(),
        "Rarefaction1DPure": Rarefaction1DPure(),
        "LinearAdvection1D": LinearAdvection1D(),
    }

    print("\n" + "=" * 60)
    print("Generating Animations")
    print("=" * 60)

    for problem_name in df_1d["problem_name"].unique():
        try:
            best = tracker.get_best(problem=problem_name)
            print(f"\n{problem_name}:")
            print(f"  Best exp_id: {best['exp_id']}")
            print(f"  Loss: {best['final_loss']:.5e}")
            print(f"  Method: {best['residual_method']}")

            problem = problem_map.get(problem_name)
            if problem:
                viz = Visualizer([best["exp_id"]])
                viz.animate_solution_1d(best["exp_id"], problem, steps=100)
                print(f"  ✓ Animation saved to results/videos/animate_1d_{best['exp_id']}.mp4")
        except Exception as e:
            print(f"  ✗ Failed to animate {problem_name}: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("Check results/videos/ for animations")


if __name__ == "__main__":
    main()

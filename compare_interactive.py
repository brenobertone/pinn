#!/usr/bin/env python3
"""
Interactive experiment comparison script.
Query and compare tracked experiments.
"""

import sys

from pinn.experiments.tracker import ExperimentTracker


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
    print("PINN Experiment Comparison")
    print("=" * 70)

    tracker = ExperimentTracker()
    df = tracker.load_experiments()

    if df.empty:
        print("\nNo experiments found.")
        print("Run: python train_interactive.py first")
        return

    print(f"\nTotal experiments: {len(df)}")

    # Show overview
    print("\n" + "=" * 70)
    print("Overview")
    print("=" * 70)
    print(f"\nProblems: {', '.join(df['problem_name'].unique())}")
    print(f"Residual methods: {', '.join(df['residual_method'].unique())}")
    print(f"Optimizers: {', '.join(df['optimizer'].unique())}")

    # What to compare
    print("\n" + "=" * 70)
    print("What would you like to compare?")
    print("=" * 70)
    print("  1. All experiments (full table)")
    print("  2. Best by problem")
    print("  3. Compare residual methods")
    print("  4. Compare architectures")
    print("  5. Filter by problem")
    print("  6. Custom query")

    choice = get_input("Select", default="2", type_fn=int)

    if choice == 1:
        # All experiments
        print("\n" + "=" * 70)
        print("All Experiments")
        print("=" * 70)
        columns = [
            "exp_id",
            "problem_name",
            "residual_method",
            "final_loss",
            "training_time",
        ]
        print(df[columns].to_string(index=False))

    elif choice == 2:
        # Best by problem
        print("\n" + "=" * 70)
        print("Best Experiment Per Problem")
        print("=" * 70)
        print(f"\n{'Problem':<30} {'Method':<10} {'Loss':>12} {'Time':>8} {'Exp ID':<12}")
        print("-" * 80)
        for problem in df["problem_name"].unique():
            try:
                best = tracker.get_best(problem=problem)
                print(
                    f"{problem:<30} {best['residual_method']:<10} "
                    f"{best['final_loss']:>12.5e} {best['training_time']:>7.2f}s "
                    f"{best['exp_id']:<12}"
                )
            except:
                pass

    elif choice == 3:
        # Compare residual methods
        problem = get_input(
            "\nProblem name (or 'all' for aggregate)",
            default="all"
        )

        if problem == "all":
            df_filtered = df
        else:
            df_filtered = df[df["problem_name"] == problem]

        print("\n" + "=" * 70)
        print(f"Residual Method Comparison - {problem}")
        print("=" * 70)
        print(f"\n{'Method':<15} {'Experiments':>12} {'Avg Loss':>12} {'Avg Time':>10}")
        print("-" * 70)
        for method in df_filtered["residual_method"].unique():
            method_df = df_filtered[df_filtered["residual_method"] == method]
            count = len(method_df)
            avg_loss = method_df["final_loss"].mean()
            avg_time = method_df["training_time"].mean()
            print(f"{method:<15} {count:>12} {avg_loss:>12.5e} {avg_time:>9.2f}s")

    elif choice == 4:
        # Compare architectures
        problem = get_input(
            "\nProblem name (or 'all' for aggregate)",
            default="all"
        )

        if problem == "all":
            df_filtered = df
        else:
            df_filtered = df[df["problem_name"] == problem]

        print("\n" + "=" * 70)
        print(f"Architecture Comparison - {problem}")
        print("=" * 70)
        print(f"\n{'Architecture':<40} {'Experiments':>12} {'Avg Loss':>12}")
        print("-" * 70)

        # Group by architecture
        df_filtered["arch_str"] = df_filtered["network_layers"].astype(str)
        for arch_str in df_filtered["arch_str"].unique():
            arch_df = df_filtered[df_filtered["arch_str"] == arch_str]
            count = len(arch_df)
            avg_loss = arch_df["final_loss"].mean()
            # Get actual architecture
            arch = arch_df.iloc[0]["network_layers"]
            print(f"{str(arch):<40} {count:>12} {avg_loss:>12.5e}")

    elif choice == 5:
        # Filter by problem
        available = df["problem_name"].unique().tolist()
        print(f"\nAvailable: {', '.join(available)}")
        problem = get_input("Problem name", default=available[0])

        df_filtered = df[df["problem_name"] == problem]
        print(f"\n{len(df_filtered)} experiments for {problem}")
        print("\n" + df_filtered[[
            "exp_id",
            "residual_method",
            "final_loss",
            "training_time",
        ]].to_string(index=False))

    elif choice == 6:
        # Custom query
        print("\nAvailable columns:")
        print(", ".join(df.columns))

        column = get_input("\nColumn to filter by", default="problem_name")
        value = get_input(f"Value for {column}")

        df_filtered = df[df[column] == value]
        print(f"\nFound {len(df_filtered)} matching experiments")

        if not df_filtered.empty:
            print("\n" + df_filtered[[
                "exp_id",
                "problem_name",
                "residual_method",
                "final_loss",
            ]].to_string(index=False))

    # Export option
    print("\n" + "=" * 70)
    export = get_input("Export results to CSV? (y/n)", default="n").lower()
    if export == "y":
        filename = get_input("Filename", default="experiments_comparison.csv")
        df.to_csv(filename, index=False)
        print(f"✓ Exported to {filename}")

    print("\n" + "=" * 70)
    print("Experiment files:")
    print("  - Database: results/experiments.jsonl")
    print("  - Models: results/model_<exp_id>.pth")
    print("  - Plots: results/plot_<exp_id>.png")
    print("  - Videos: results/videos/animate_*_<exp_id>.mp4")


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

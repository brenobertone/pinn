#!/usr/bin/env python3
"""
Load trained model and visualize weights.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn.core.architectures import NetworkConfig
from pinn.experiments.tracker import ExperimentTracker


def visualize_weights(exp_id: str, show_values: bool = False):
    """Load model and visualize weight matrices."""
    tracker = ExperimentTracker()

    # Load experiment metadata
    df = tracker.load_experiments()
    exp = df[df["exp_id"] == exp_id].iloc[0]

    print(f"Experiment: {exp_id}")
    print(f"Problem: {exp['problem_name']}")
    print(f"Architecture: {exp['network_layers']} ({exp['network_activation']})")
    print(f"Final Loss: {exp['final_loss']:.5e}")
    print()

    # Reconstruct network config
    n_inputs = 2 if exp["problem_spatial_dims"] == 1 else 3
    arch = NetworkConfig(
        exp["network_layers"], exp["network_activation"], n_inputs=n_inputs, n_outputs=1
    )

    # Load model
    model = tracker.load_model(exp_id, arch)
    model.eval()

    # Extract weights and biases
    weights = []
    biases = []
    layer_names = []

    for name, param in model.named_parameters():
        if "weight" in name:
            weights.append(param.detach().cpu().numpy())
            layer_names.append(name.replace(".weight", ""))
        elif "bias" in name:
            biases.append(param.detach().cpu().numpy())

    # Print raw values if requested
    if show_values:
        print("=" * 70)
        print("RAW WEIGHT VALUES")
        print("=" * 70)
        for name, w, b in zip(layer_names, weights, biases):
            print(f"\n{name}:")
            print(f"  Weight matrix shape: {w.shape}")
            print(f"  Weights:\n{w}")
            print(f"  Biases:\n{b}")
            print()

    # Create visualization - heatmaps of weight matrices
    n_layers = len(weights)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 10))

    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for i, (w, b, name) in enumerate(zip(weights, biases, layer_names)):
        # Weight matrix heatmap
        ax_w = axes[0, i]
        im = ax_w.imshow(w, cmap="RdBu_r", aspect="auto", vmin=-abs(w).max(), vmax=abs(w).max())
        ax_w.set_title(f"{name}\nWeight Matrix {w.shape}")
        ax_w.set_xlabel("Input")
        ax_w.set_ylabel("Output")
        plt.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)

        # Add text annotation for small matrices
        if w.shape[0] <= 10 and w.shape[1] <= 10:
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    ax_w.text(col, row, f"{w[row, col]:.2f}",
                             ha="center", va="center", fontsize=8,
                             color="white" if abs(w[row, col]) > abs(w).max()*0.5 else "black")

        # Bias vector heatmap
        ax_b = axes[1, i]
        b_matrix = b.reshape(-1, 1)
        im_b = ax_b.imshow(b_matrix, cmap="RdBu_r", aspect="auto",
                          vmin=-abs(b).max() if abs(b).max() > 0 else -1,
                          vmax=abs(b).max() if abs(b).max() > 0 else 1)
        ax_b.set_title(f"{name}\nBias Vector {b.shape}")
        ax_b.set_xlabel("Bias")
        ax_b.set_ylabel("Neuron")
        ax_b.set_xticks([0])
        plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)

        # Add text for biases
        if len(b) <= 20:
            for row in range(len(b)):
                ax_b.text(0, row, f"{b[row]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(b[row]) > abs(b).max()*0.5 else "black")

    plt.tight_layout()
    output_path = f"results/weights_{exp_id}.png"
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.show()

    # Print weight statistics
    print("\nWeight Statistics:")
    print("-" * 70)
    for name, w, b in zip(layer_names, weights, biases):
        print(f"{name}:")
        print(f"  Weights: shape={w.shape}, μ={w.mean():.4f}, σ={w.std():.4f}")
        print(f"  Biases:  shape={b.shape}, μ={b.mean():.4f}, σ={b.std():.4f}")
        print(f"  Weight range: [{w.min():.4f}, {w.max():.4f}]")
        print(f"  Bias range:   [{b.min():.4f}, {b.max():.4f}]")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Show available experiments
        tracker = ExperimentTracker()
        df = tracker.load_experiments()
        if df.empty:
            print("No experiments found. Run training first.")
            sys.exit(1)

        print("Available experiments:")
        print("-" * 70)
        for _, exp in df.iterrows():
            print(
                f"{exp['exp_id']}: {exp['problem_name']} "
                f"(loss={exp['final_loss']:.5e})"
            )
        print("\nUsage: python inspect_weights.py <exp_id> [--values]")
        print("  --values: Print raw weight matrices to console")
        sys.exit(0)

    exp_id = sys.argv[1]
    show_values = "--values" in sys.argv
    visualize_weights(exp_id, show_values=show_values)

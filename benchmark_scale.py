
import torch
import torch.nn as nn
import time
import json
import traceback
from pinn.core.training import Config, train
from pinn.core.architectures import NetworkConfig
from pinn.problems.problems_1d import Burgers1D
from pinn.problems.problems_2d import Riemann2D

def benchmark_scale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking Scale on {device}")
    
    epochs = 2
    snr_batches = 1000
    
    # 1. Branch Testing (All methods from interactive)
    branches = [
        # 1D
        {"name": "1D_autograd_LHS", "dim": 1, "res": "autograd", "samp": "latin_hypercube", "pts": 10000},
        {"name": "1D_autograd_Uniform", "dim": 1, "res": "autograd", "samp": "uniform", "pts": 10000},
        {"name": "1D_mm2_Uniform", "dim": 1, "res": "mm2", "samp": "uniform", "pts": 10000},
        # 2D
        {"name": "2D_autograd_LHS", "dim": 2, "res": "autograd", "samp": "latin_hypercube", "pts": 50000},
        {"name": "2D_autograd_Uniform", "dim": 2, "res": "autograd", "samp": "uniform", "pts": 50000},
        {"name": "2D_mm2_Uniform", "dim": 2, "res": "mm2", "samp": "uniform", "pts": 50000},
        {"name": "2D_mm3_Uniform", "dim": 2, "res": "mm3", "samp": "uniform", "pts": 50000},
        {"name": "2D_uno_Uniform", "dim": 2, "res": "uno", "samp": "uniform", "pts": 50000},
    ]
    
    results = []

    print("\n--- Testing Execution Branches ---")
    for b in branches:
        prob = Burgers1D() if b["dim"] == 1 else Riemann2D()
        arch = NetworkConfig([32, 32, 32], "tanh", n_inputs=b["dim"]+1).build()
        config = Config(
            epsilon=0.01,
            n_points=b["pts"],
            epochs=epochs,
            residual_method=b["res"],
            sampling_method=b["samp"],
            snr_compute_interval=1,
            snr_batches=snr_batches
        )
        
        print(f"Scenario: {b['name']}...", end=" ", flush=True)
        start = time.time()
        status = "Success"
        error = None
        duration = 0
        
        try:
            train(prob, arch, config)
            duration = time.time() - start
            print(f"Done ({duration:.2f}s)")
        except Exception as e:
            status = "Failed"
            error = str(e)
            print(f"FAILED: {error[:50]}...")
            
        results.append({
            "scenario": b["name"],
            "status": status,
            "error": error,
            "duration": duration,
            "pts": b["pts"],
            "snr_batches": snr_batches
        })

    # 2. Scaling Points (Using Autograd LHS as the scale baseline)
    print("\n--- Testing Scale ---")
    point_scales = [10000, 50000, 100000, 250000]
    for p in point_scales:
        prob = Riemann2D()
        arch = NetworkConfig([32, 32, 32], "tanh", n_inputs=3).build()
        config = Config(
            epsilon=0.01,
            n_points=p,
            epochs=epochs,
            residual_method="autograd",
            sampling_method="latin_hypercube",
            snr_compute_interval=1,
            snr_batches=snr_batches
        )
        
        print(f"Scale: {p} points...", end=" ", flush=True)
        start = time.time()
        status = "Success"
        error = None
        duration = 0
        
        try:
            train(prob, arch, config)
            duration = time.time() - start
            print(f"Done ({duration:.2f}s)")
        except Exception as e:
            status = "Failed"
            error = str(e)
            print(f"FAILED")
            
        results.append({
            "scenario": f"Scale_{p}_pts",
            "status": status,
            "error": error,
            "duration": duration,
            "pts": p,
            "snr_batches": snr_batches
        })

    with open("benchmark_scale_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to benchmark_scale_results.json")

if __name__ == "__main__":
    benchmark_scale()

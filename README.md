# IC-PINN

Physics-Informed Neural Network solver for 1D and 2D hyperbolic conservation laws with flexible architectures and slope limiters.

## Overview

Trains neural networks to solve PDEs:

**2D form:**
```
u_t + ∂f1(u)/∂x + ∂f2(u)/∂y = ε(u_xx + u_yy)
```

**1D form:**
```
u_t + ∂f(u)/∂x = ε(u_xx)
```

Where:
- `u` is solution field
- `f`, `f1`, `f2` are flux functions (problem-specific)
- `ε` is viscosity coefficient

Loss = PDE residual loss + initial condition loss.

## Features

✅ **Flexible architectures** - Separate network from problem definition  
✅ **1D + 2D support** - Burgers, shocks, Riemann problems  
✅ **Experiment tracking** - Automatic logging, comparison, metrics DB  
✅ **Multiple slope limiters** - autograd, MM2, MM3, UNO  
✅ **Programmatic comparison** - Query best configs, compare architectures  
✅ **Unified visualization** - Animations, loss plots, comparison tables

## Quick Start

```python
from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.problems.problems_2d import Riemann2D
from pinn.experiments.tracker import ExperimentTracker

# Define problem
problem = Riemann2D()

# Define architecture (flexible!)
arch = NetworkConfig([50, 50, 50], "tanh")
model = arch.build()

# Configure training
config = Config(
    epsilon=0.0025,
    n_points=125000,
    epochs=5000,
    residual_method="mm2",  # autograd, mm2, mm3, uno
    optimizer="adamw",
)

# Train
model, fig, metrics = train(problem, model, config)

# Log experiment
tracker = ExperimentTracker()
exp_id = tracker.log_run(problem, config, arch, model, metrics, fig)
```

## Comparing Architectures

```python
from pinn.experiments.tracker import ExperimentTracker
from pinn.visualization.visualizer import Visualizer

# Query experiments
tracker = ExperimentTracker()
df = tracker.compare({"problem_name": "Riemann2D"})
print(df.sort_values("final_loss"))

# Get best
best = tracker.get_best(problem="Riemann2D")
print(f"Best loss: {best['final_loss']:.5e}")
print(f"Architecture: {best['network_layers']}")

# Visualize
viz = Visualizer([best["exp_id"]])
viz.animate_solution_2d(best["exp_id"], problem, steps=200)
```

## Problems

**2D Problems** (`pinn/problems/problems_2d.py`):
- PeriodicSine2D, Riemann2D, RiemannOblique
- Shock1D, Rarefaction1D, Pulse
- BuckleyLeverett, NonLinearNonConvexFlow

**1D Problems** (`pinn/problems/problems_1d.py`):
- Burgers1D, LinearAdvection1D
- Shock1DPure, Rarefaction1DPure

## Slope Limiters

| Method | 1D | 2D | Description |
|--------|----|----|-------------|
| `autograd` | ✅ | ✅ | Pure automatic differentiation |
| `mm2` | ✅ | ✅ | MinMod2 limiter (forward/backward) |
| `mm3` | ❌ | ✅ | MinMod3 limiter (forward/backward/centered) |
| `uno` | ❌ | ✅ | UNO scheme (5-point stencil) |

## Batch Experiments

```bash
# Run full test suite
python -m pinn.experiments.run_suite

# Or see example
python example_usage.py
```

Results → `results/`:
- `experiments.jsonl` - metrics database
- `model_{id}.pth` - trained weights
- `plot_{id}.png` - loss curves
- `videos/` - animations

## Project Structure

```
pinn/
├── core/
│   ├── problems.py          # Base Problem1D/Problem2D
│   ├── architectures.py     # NetworkConfig, flexible networks
│   ├── residuals.py         # All slope limiters (1D+2D)
│   └── training.py          # Unified train() function
├── problems/
│   ├── problems_1d.py       # Burgers, shocks, advection
│   └── problems_2d.py       # Riemann, BL, etc.
├── experiments/
│   ├── tracker.py           # ExperimentTracker (logging + queries)
│   └── run_suite.py         # Batch runner
└── visualization/
    └── visualizer.py        # Unified viz (1D + 2D animations)
```

## Migration

Upgrading from old API? See [MIGRATION.md](MIGRATION.md) for guide.

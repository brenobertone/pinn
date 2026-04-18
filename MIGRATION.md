# Migration Guide

## Overview

Major refactor separates architecture from problems, adds 1D support, flexible slope limiters, and experiment tracking.

## Key Changes

### 1. Separate Architecture from Problems

**Before:**
```python
class PeriodicSine2D(Problem):
    net = PINN(n_inputs=3, n_outputs=1)  # Hardcoded in problem
```

**After:**
```python
# Problems only define physics
class PeriodicSine2D(Problem2D):
    # No net attribute
    
# Architecture separate
from pinn.core.architectures import NetworkConfig
arch = NetworkConfig([20, 20, 20, 20, 20], "relu")
model = arch.build()
```

### 2. New Directory Structure

```
pinn/
├── core/
│   ├── problems.py       # Base Problem1D, Problem2D
│   ├── architectures.py  # NetworkConfig, build functions
│   ├── residuals.py      # All residual methods (1D + 2D)
│   └── training.py       # Generic train() function
├── problems/
│   ├── problems_1d.py    # 1D problems (Burgers, Shock, etc.)
│   └── problems_2d.py    # 2D problems (Riemann, etc.)
├── experiments/
│   ├── tracker.py        # ExperimentTracker class
│   └── run_suite.py      # Batch runner
└── visualization/
    └── visualizer.py     # Unified visualization
```

### 3. Training API Changes

**Before:**
```python
from problems_definitions import Riemann2D
from training import Config, train
from slope_limiters import advection_residual_mm2

problem = Riemann2D()
config = Config(
    epsilon=0.0025,
    n_points=125000,
    epochs=5000,
    residual=advection_residual_mm2,
)
model, fig = train(problem, config)
```

**After:**
```python
from pinn.core.architectures import NetworkConfig
from pinn.core.training import Config, train
from pinn.problems.problems_2d import Riemann2D

problem = Riemann2D()
arch = NetworkConfig([20, 20, 20, 20, 20], "relu")
model = arch.build()
config = Config(
    epsilon=0.0025,
    n_points=125000,
    epochs=5000,
    residual_method="mm2",  # String instead of function
    optimizer="adamw",
)
model, fig, metrics = train(problem, model, config)
```

### 4. Experiment Tracking

**New feature:**
```python
from pinn.experiments.tracker import ExperimentTracker

tracker = ExperimentTracker()
exp_id = tracker.log_run(problem, config, arch, model, metrics, fig)

# Query experiments
df = tracker.load_experiments()
best = tracker.get_best(problem="Riemann2D")
```

### 5. Visualization

**Before:** Separate scripts for each plot type

**After:** Unified interface
```python
from pinn.visualization.visualizer import Visualizer

viz = Visualizer([exp_id1, exp_id2])
viz.comparison_table()
viz.animate_solution_2d(exp_id, problem)
```

### 6. 1D Problems

**New support:**
```python
from pinn.problems.problems_1d import Burgers1D
from pinn.core.architectures import NetworkConfig

problem = Burgers1D()
arch = NetworkConfig([20, 20, 20], "relu", n_inputs=2, n_outputs=1)
model = arch.build()
config = Config(
    epsilon=0.001,
    n_points=10000,
    epochs=1000,
    residual_method="autograd",
)
model, fig, metrics = train(problem, model, config)
```

## Backward Compatibility

Old code still works via legacy imports (not yet removed). To migrate:

1. Update imports from `pinn.core.*` and `pinn.problems.*`
2. Separate architecture creation from problem definition
3. Pass `model` to `train()` function
4. Use string `residual_method` instead of function
5. Handle third return value `metrics` from `train()`

## Running Old Scripts

Old `run_training.py` deprecated. Use:
- `example_usage.py` - Simple examples
- `pinn/experiments/run_suite.py` - Full batch runs

## Benefits

1. **Flexibility**: Test any architecture on any problem
2. **Tracking**: Automatic experiment logging + comparison
3. **1D Support**: Burgers, linear advection, shocks
4. **Cleaner**: Separation of concerns
5. **Programmatic**: Query best configs, compare metrics

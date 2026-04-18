# IC-PINN

Physics-Informed Neural Network solver for 2D hyperbolic conservation laws with slope limiters.

## Overview

Trains neural networks to solve PDEs of form:

```
u_t + ∂f1(u)/∂x + ∂f2(u)/∂y = ε(u_xx + u_yy)
```

Where:
- `u(x,y,t)` is solution field
- `f1(u)`, `f2(u)` are flux functions (problem-specific)
- `ε` is viscosity coefficient

Loss = PDE residual loss + initial condition loss.

## Architecture

**Network**: 5-layer MLP (20 neurons/layer, ReLU activation)
- Input: (x, y, t) coordinates
- Output: u(x, y, t) prediction

**Training**: AdamW optimizer (lr=1e-3), uniform 3D mesh sampling

## Slope Limiters

Four residual computation methods:

1. **autograd**: Pure automatic differentiation
2. **mm2**: MinMod2 limiter (forward/backward differences)
3. **mm3**: MinMod3 limiter (forward/backward/centered differences)
4. **uno**: UNO scheme (5-point stencil with second derivatives)

MinMod limiters reduce spurious oscillations at discontinuities.

## Problems

Eight test problems defined in `problems_definitions.py`:

| Problem | Domain | f1(u) | f2(u) | Initial Condition |
|---------|--------|-------|-------|-------------------|
| **PeriodicSine2D** | [0,1]² × [0,1] | u | u | sin²(πx)sin²(πy) |
| **Rarefaction1D** | [-6,6] × [-1.5,1.5] × [0,2.5] | u²/2 | u²/2 | -1 (x<0), 1 (x>0) |
| **Shock1D** | [-6,6] × [-1.5,1.5] × [0,2.5] | u²/2 | u²/2 | 0 (x<0), -1 (x>0) |
| **Pulse** | [-3,3]² × [0,2.5] | u²/2 | u²/2 | 1 in [0,1]², else 0 |
| **RiemannOblique** | [0,1]² × [0,0.5] | u²/2 | u²/2 | 4 quadrants: {-1,-0.2,0.5,0.8} |
| **Riemann2D** | [0,1]² × [0,1/12] | u²/2 | u²/2 | 2 (x<0.25,y<0.25), 3 (x>0.25,y>0.25), else 1 |
| **BuckleyLeverett** | [-1.5,1.5]² × [0,0.5] | Sw²/(Sw²+μ(1-Sw)²) | f1·(1-Cg(1-Sw)²) | 1 (r<√0.5), else 0 |
| **NonLinearNonConvexFlow** | [-2,2]² × [0,1] | sin(u) | cos(u) | 0.25π outside, 3.5π inside unit circle |

## Usage

Train all problems with all limiters:
```bash
python -m pinn.run_training
```

Results saved to `results/`:
- `model_{hash}.pth` - trained weights
- `plot_{hash}.png` - loss curve
- `config_{hash}.json` - hyperparameters

Generate animations:
```python
from pinn.plotters.animate import animate_problem
from pinn.problems_definitions import Riemann2D

problem = Riemann2D()
problem.net.load_state_dict(torch.load("results/model_{hash}.pth"))
animate_problem(problem, steps=200, hash_id="hash")
```

Outputs MP4 to `results/videos/`.

## Configuration

Edit `run_training.py` to modify:
- `epsilons` - viscosity values
- `n_points` - mesh resolution (cube root gives per-dimension points)
- `epochs` - training iterations
- `residuals` - slope limiter methods

Default: ε=0.0025, 512k points (80³ mesh), 5000 epochs

## File Structure

- `problems_definitions.py` - Problem classes and PINN network
- `training.py` - Training loop, mesh generation, Config class
- `slope_limiters.py` - Residual computation methods (autograd, mm2, mm3, uno)
- `run_training.py` - Batch runner for all problems/configs
- `plotters/animate.py` - 3D surface + contour animation generator
- `plotters/plot_three_times.py` - Multi-timestep visualization

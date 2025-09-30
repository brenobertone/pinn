# README — Experimentos PINN / Weak-PINN

Repositório para experimentar PINNs com *viscous regularization* e variantes com *slope limiters* (MM2, MM3, UNO). Implementação em PyTorch; treino baseado em minimizar o residual físico + condição inicial. A teoria está no relatório `relatorio.pdf`.

---

## Estrutura de arquivos
- `problems_definitions.py` — definições das classes `Problem` (ex.: `Riemann2D`, `Shock1D`, ...) e da rede `PINN` usada para aproximar \(u(x,y,t)\). Declara domínio, condição inicial e fluxos `f1`, `f2`. 
- `training.py` — rotina de treino principal: geração de malha (`uniform_mesh`), criação de `Config` (ε, n_points, epochs, residual), loop de treino (RMSprop + scheduler) e funções auxiliares de plot. Função principal: `train(problem, config, device)`.  
- `slope_limiters.py` — implementa diferenças finitas com limitadores de inclinação: `mm2`, `mm3`, `uno` e residuais que usam essas discretizações (alternativa ao `autograd`).  

---

## O que o código faz — algoritmo geral

### 1. Definir problema

```python
class PeriodicSine2D(Problem):
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 4 * 1.0)
    name = "PeriodicSine2D"
    net = PINN(n_inputs=3, n_outputs=1)
    x_orientation = "decrescent"
    y_orientation = "decrescent"

    @staticmethod
    def f1(u):
        return u

    @staticmethod
    def f2(u):
        return u

    def initial_condition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return (torch.sin(np.pi * x) ** 2) * (torch.sin(np.pi * y) ** 2)
```

### 2. Definir parâmetros do treinamento
    - número de pontos N a ser usado
    - learning_rate
    - epochs
    - optimizer
    - scheduler
    - resíduo a ser usado (advection_residual_autograd, advection_residual_mm2, advection_residual_mm3, advection_residual_uno)


### 3. Sampling de pontos

Feito com malha uniforme, selecionando N^(1/3) pontos em cada um dos eixos x, y, t, e uma padding de no mínimo 2 pontos, afim de permitir o cálculo de derivadas usando diferenças finitas

### 4. Iterações de treinamento

Para cada uma das épocas, calculamos o resíduo e 

```python
f = config.residual(model, problem, xyt_f)
loss_f = torch.mean(f**2)

u_pred_ic = model(xyt_ic)
loss_ic = torch.mean((u_pred_ic - u0) ** 2)
```

#### 4.1 Cálculo do resíduo

Entrada:
    model    ← rede PINN
    problem  ← funções f1, f2
    xyt      ← malha de pontos [Nx, Ny, Nt, 3]
    epsilon  ← coef. de difusão

Saída:
    residual ← resíduo da PDE em cada ponto da malha

Passos:

1. Extrair espaçamentos da malha:
    dx, dy, dt ← diferenças em x, y, t

2. Avaliar modelo na malha:
    u ← model(xyt)  # reshape para [Nx, Ny, Nt]

3. Definir operador de diferenças finitas (diff_ops_mm2):
    - Calcular derivadas forward/backward em x, y, t
    - Combinar com mm2 → derivadas de 1ª ordem (u_x, u_y, u_t)
    - Usar diferença central → derivadas de 2ª ordem (u_xx, u_yy)

4. Aplicar operador em:
    - u → obter u_t, u_xx, u_yy
    - f1(u) → obter f1_u_x
    - f2(u) → obter f2_u_y

5. Montar resíduo da equação de advecção-difusão:
    residual ← u_t + f1_u_x + f2_u_y - epsilon * (u_xx + u_yy)

6. Retornar residual


# Exemplo míninmo

```python
from typing import Tuple
import torch
from problems_definitions import PeriodicSine2D
from training import Config, train
from slope_limiters import advection_residual_mm2

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

problem = PeriodicSine2D()
config = Config(
    epsilon=0.0025,
    n_points=125,   # 5^3 => Nx=Ny=Nt=5
    epochs=20000,
    residual=advection_residual_mm2,
)

model, fig = train(problem, config, device)
fig.savefig("loss_history.png")
import sys
from itertools import product
from pathlib import Path
from typing import Any, Tuple

import matplotlib
import torch

matplotlib.use("Qt5Agg")  # Qt backend

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from plotters.plot_three_times import plot_three_times
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QTabWidget,
                             QVBoxLayout, QWidget)
from run_training import get_config_hash

from pinn.problems_definitions import (BuckleyLeverett, NonLinearNonConvexFlow,
                                       PeriodicSine2D, Problem, Rarefaction1D,
                                       Riemann2D, RiemannOblique)
from pinn.slope_limiters import (advection_residual_autograd,
                                 advection_residual_mm2,
                                 advection_residual_mm3,
                                 advection_residual_uno)
from pinn.training import Config


def get_config_description(problem: Problem, config: Config) -> str:
    return (
        f"ε={config.epsilon} "
        f"res={config.residual.__closure__[0].cell_contents.__name__}"  # type: ignore
        f"n_int={config.n_points} "
    )


def load_results(
    problem: Problem, config: Config, results_dir: Path = Path("results")
) -> Tuple[Problem, Any]:
    hash_id = get_config_hash(problem, config)
    model_path = results_dir / f"model_{hash_id}.pth"
    plot_path = results_dir / f"plot_{hash_id}.png"

    model_state = torch.load(model_path)
    problem.net.load_state_dict(model_state)

    training_history = plt.imread(plot_path)
    return problem, training_history


class ProblemWindow(QMainWindow):
    def __init__(self, problem: Problem, configs: list[Config]):
        super().__init__()
        self.setWindowTitle(f"Problem: {problem.__class__.__name__}")

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        for config in configs:
            try:
                trained_problem, _ = load_results(problem, config)
                fig = plot_three_times(trained_problem)

                canvas = FigureCanvas(fig)
                widget = QWidget()
                layout = QVBoxLayout(widget)
                layout.addWidget(canvas)

                desc = QLabel(get_config_description(problem, config))
                desc.setWordWrap(True)  # wrap text if long
                layout.addWidget(desc)

                tabs.addTab(widget, get_config_description(problem, config))

                width, height = fig.get_size_inches() * fig.dpi
                self.resize(int(width), int(height) + 150)

            except FileNotFoundError:
                print(
                    f"Results missing for {get_config_description(problem, config)}"
                )


if __name__ == "__main__":
    problems = [
        # PeriodicSine2D(),
        # Rarefaction1D(),
        # RiemannOblique(),
        # Riemann2D(),
        BuckleyLeverett(),
        # NonLinearNonConvexFlow(),
    ]

    epsilons = [0.0005]
    n_points = [100000]
    epochs = [30000]
    residuals = [
        advection_residual_autograd,
        advection_residual_mm2,
        advection_residual_mm3,
        advection_residual_uno,
    ]

    # epsilons = [0.0025]
    # deltas = [1e-3]
    # n_internals = [50000, 200000, 1000000]
    # n_ics = [15000, 30000, 60000, 100000]
    # epochs = [10000, 30000]
    # residuals = [
    #     advection_residual_autograd,
    #     advection_residual_mm2,
    #     advection_residual_mm3,
    #     advection_residual_uno,
    # ]

    configs = [
        Config(
            epsilon=e,
            n_points=n,
            epochs=ep,
            residual=r,
        )
        for e, n, ep, r in product(epsilons, n_points, epochs, residuals)
    ]

    app = QApplication(sys.argv)

    windows = []
    for problem in problems:
        win = ProblemWindow(problem, configs)
        win.show()
        windows.append(win)

    sys.exit(app.exec_())

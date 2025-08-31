import json
import os
from typing import Any

from run_training import get_config_hash

from pinn.problems_definitions import (BuckleyLeverett, NonLinearNonConvexFlow,
                                       PeriodicSine2D, Rarefaction1D,
                                       Riemann2D, RiemannOblique)

problems = [
    # PeriodicSine2D,
    # Rarefaction1D,
    # RiemannOblique,
    Riemann2D,
    # BuckleyLeverett,
    # NonLinearNonConvexFlow,
]


def show_all_configs(folder_path: str, problem: Any) -> None:
    print(f"\n=== Configs for problem: {problem.__name__} ===\n")
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as f:
            config_data = json.load(f)

        class DummyConfig:
            def __init__(self, data: dict):
                for k, v in data.items():
                    setattr(self, k, v)

        config_obj = DummyConfig(config_data)
        config_hash = get_config_hash(problem, config_obj)

        print(f"File: {file_name}")
        print(f"Hash: {config_hash}")
        print(f"Data:\n{json.dumps(config_data, indent=2)}\n")


if __name__ == "__main__":
    folder = "results"
    for p in problems:
        show_all_configs(folder, p)

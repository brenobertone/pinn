import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pinn.core.architectures import NetworkConfig
from pinn.core.problems import Problem
from pinn.core.training import Config


@dataclass
class ExperimentRecord:
    exp_id: str
    problem_name: str
    problem_spatial_dims: int
    network_layers: list[int]
    network_activation: str
    epsilon: float
    n_points: int
    epochs: int
    residual_method: str
    optimizer: str
    learning_rate: float
    final_loss: float
    final_loss_f: float
    final_loss_ic: float
    training_time: float


class ExperimentTracker:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.db_path = self.results_dir / "experiments.jsonl"

    def _generate_id(
        self, problem: Problem, config: Config, network_config: NetworkConfig
    ) -> str:
        import hashlib

        config_str = json.dumps(
            {
                "problem": problem.name,
                "layers": network_config.layers,
                "activation": network_config.activation,
                "epsilon": config.epsilon,
                "n_points": config.n_points,
                "epochs": config.epochs,
                "residual": config.residual_method,
                "optimizer": config.optimizer,
                "lr": config.learning_rate,
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def log_run(
        self,
        problem: Problem,
        config: Config,
        network_config: NetworkConfig,
        model: torch.nn.Module,
        metrics: dict[str, Any],
        figure=None,
    ) -> str:
        exp_id = self._generate_id(problem, config, network_config)

        record = ExperimentRecord(
            exp_id=exp_id,
            problem_name=problem.name,
            problem_spatial_dims=problem.spatial_dims,
            network_layers=network_config.layers,
            network_activation=network_config.activation,
            epsilon=config.epsilon,
            n_points=config.n_points,
            epochs=config.epochs,
            residual_method=config.residual_method,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            final_loss=metrics["final_loss"],
            final_loss_f=metrics["final_loss_f"],
            final_loss_ic=metrics["final_loss_ic"],
            training_time=metrics["training_time"],
        )

        with self.db_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        torch.save(model.state_dict(), self.results_dir / f"model_{exp_id}.pth")
        if figure:
            figure.savefig(self.results_dir / f"plot_{exp_id}.png")

        with (self.results_dir / f"config_{exp_id}.json").open("w") as f:
            json.dump(asdict(record), f, indent=2)

        print(f"Logged experiment: {exp_id}")
        return exp_id

    def load_experiments(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()

        records = []
        with self.db_path.open("r") as f:
            for line in f:
                records.append(json.loads(line))

        return pd.DataFrame(records)

    def compare(self, filters: dict[str, Any] = None) -> pd.DataFrame:
        df = self.load_experiments()
        if df.empty:
            return df

        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]

        return df

    def get_best(
        self, problem: str = None, metric: str = "final_loss"
    ) -> pd.Series:
        df = self.load_experiments()
        if df.empty:
            raise ValueError("No experiments logged")

        if problem:
            df = df[df["problem_name"] == problem]

        return df.loc[df[metric].idxmin()]

    def load_model(
        self, exp_id: str, network_config: NetworkConfig
    ) -> torch.nn.Module:
        model = network_config.build()
        model_path = self.results_dir / f"model_{exp_id}.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

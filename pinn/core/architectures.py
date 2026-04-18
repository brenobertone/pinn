from typing import Literal

import torch.nn as nn


class NetworkConfig:
    def __init__(
        self,
        layers: list[int],
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        n_inputs: int = 3,
        n_outputs: int = 1,
    ):
        self.layers = layers
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def build(self) -> nn.Module:
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        act_fn = activation_map[self.activation]

        layers_list = []
        layers_list.append(nn.Linear(self.n_inputs, self.layers[0]))
        layers_list.append(act_fn())

        for i in range(len(self.layers) - 1):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers_list.append(act_fn())

        layers_list.append(nn.Linear(self.layers[-1], self.n_outputs))

        return nn.Sequential(*layers_list)

    def __repr__(self) -> str:
        return f"NetworkConfig(layers={self.layers}, activation={self.activation}, n_inputs={self.n_inputs}, n_outputs={self.n_outputs})"


def build_default_pinn(n_inputs: int = 3, n_outputs: int = 1) -> nn.Module:
    """Build the original PINN architecture (5 layers of 20 neurons)"""
    config = NetworkConfig([20, 20, 20, 20, 20], "relu", n_inputs, n_outputs)
    return config.build()

from typing import Literal

import torch
import torch.nn as nn


class CharacteristicTransform(nn.Module):
    """Transform (x,t) → (x - c*t) for advection problems."""
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.register_buffer('c', torch.tensor(c))

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt: [N, 2] tensor with columns [x, t]
        Returns:
            xi: [N, 1] tensor with characteristic coordinate x - c*t
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return x - self.c * t


class TransformedNetwork(nn.Module):
    """Wrapper that applies coordinate transform before network."""
    def __init__(self, base_network: nn.Module, transform: nn.Module):
        super().__init__()
        self.transform = transform
        self.network = base_network

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        transformed = self.transform(coords)
        return self.network(transformed)


class NetworkConfig:
    def __init__(
        self,
        layers: list[int],
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        n_inputs: int = 3,
        n_outputs: int = 1,
        use_characteristic: bool = False,
        characteristic_c: float = 1.0,
    ):
        self.layers = layers
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.use_characteristic = use_characteristic
        self.characteristic_c = characteristic_c

    def build(self) -> nn.Module:
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        act_fn = activation_map[self.activation]

        # For characteristic transform, network sees 1D input (x - c*t)
        # Otherwise uses full n_inputs
        net_inputs = 1 if self.use_characteristic else self.n_inputs

        layers_list = []
        layers_list.append(nn.Linear(net_inputs, self.layers[0]))
        layers_list.append(act_fn())

        for i in range(len(self.layers) - 1):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers_list.append(act_fn())

        layers_list.append(nn.Linear(self.layers[-1], self.n_outputs))

        base_network = nn.Sequential(*layers_list)

        # Wrap with characteristic transform if needed
        if self.use_characteristic:
            if self.n_inputs != 2:
                raise ValueError("Characteristic transform only for 1D problems (n_inputs=2)")
            transform = CharacteristicTransform(c=self.characteristic_c)
            return TransformedNetwork(base_network, transform)

        return base_network

    def __repr__(self) -> str:
        char_str = f", characteristic=True(c={self.characteristic_c})" if self.use_characteristic else ""
        return f"NetworkConfig(layers={self.layers}, activation={self.activation}, n_inputs={self.n_inputs}, n_outputs={self.n_outputs}{char_str})"


def build_default_pinn(n_inputs: int = 3, n_outputs: int = 1) -> nn.Module:
    """Build the original PINN architecture (5 layers of 20 neurons)"""
    config = NetworkConfig([20, 20, 20, 20, 20], "relu", n_inputs, n_outputs)
    return config.build()

from __future__ import annotations

from torch import Tensor, nn


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    if num_layers <= 0:
        raise ValueError("num_layers must be greater than zero")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for layer_index in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0 and layer_index < num_layers - 1:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class ResidueEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = _build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class PairEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = _build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


__all__ = ["PairEncoder", "ResidueEncoder"]

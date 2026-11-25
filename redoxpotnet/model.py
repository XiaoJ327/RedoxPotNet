"""Neural-network architecture used by the released RedoxPotNet encoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class RedoxEncoder(nn.Module):
    """Fixed multilayer perceptron used to produce the RedoxPotNet score."""

    def __init__(
        self,
        input_dim: int = 77,
        hidden_dims: Sequence[int] = (32, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one RedoxPotNet score per sample."""
        return self.net(x)

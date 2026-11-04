"""
Transformer encoder forecaster with learned positional embeddings.
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerForecaster(nn.Module):
    """Project each day to ``units`` dimensions, run two encoder layers, read the last position."""

    def __init__(self, input_size: int, num_features: int, units: int, num_outputs: int | None = None, heads: int = 4) -> None:
        super().__init__()
        num_outputs = num_outputs or num_features
        self.norm = nn.BatchNorm1d(input_size, eps=1e-3)
        self.embed = nn.Linear(num_features, units)
        self.position = nn.Parameter(torch.zeros(1, input_size, units))
        layer = nn.TransformerEncoderLayer(
            d_model=units, nhead=heads, dim_feedforward=2 * units, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(units, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(self.norm(x)) + self.position
        x = self.encoder(x)
        return self.head(x[:, -1])

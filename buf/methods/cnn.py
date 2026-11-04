"""
Dilated 1D convolution forecaster.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNForecaster(nn.Module):
    """Three causal-style convolutions with growing dilation, then a linear head on the last step."""

    def __init__(self, input_size: int, num_features: int, units: int, num_outputs: int | None = None) -> None:
        super().__init__()
        num_outputs = num_outputs or num_features
        self.norm = nn.BatchNorm1d(input_size, eps=1e-3)
        self.convs = nn.Sequential(
            nn.Conv1d(num_features, units, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(units, units, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv1d(units, units, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(),
        )
        self.head = nn.Linear(units, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).transpose(1, 2)
        x = self.convs(x)
        return self.head(x[:, :, -1])

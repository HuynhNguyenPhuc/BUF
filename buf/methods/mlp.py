"""
Feed-forward forecasters: a plain MLP and DLinear (trend and seasonal parts, linear).
"""

from __future__ import annotations

import torch
from torch import nn


class MLPForecaster(nn.Module):
    """Flatten the window, then two hidden layers of ``units``."""

    def __init__(self, input_size: int, num_features: int, units: int, num_outputs: int | None = None) -> None:
        super().__init__()
        num_outputs = num_outputs or num_features
        self.norm = nn.BatchNorm1d(input_size, eps=1e-3)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * num_features, units),
            nn.SiLU(),
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class DLinearForecaster(nn.Module):
    """DLinear: split each feature into a moving-average trend and a remainder, then apply one linear map per part over time."""

    def __init__(self, input_size: int, num_features: int, units: int, num_outputs: int | None = None, kernel: int = 25) -> None:
        super().__init__()
        num_outputs = num_outputs or num_features
        self.kernel = kernel
        self.time_trend = nn.Linear(input_size, 1)
        self.time_seasonal = nn.Linear(input_size, 1)
        self.head = nn.Linear(num_features, num_outputs)

    def trend(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the moving average over time, replicating the edges.

        Args:
            x: Input of shape (batch, time, features).

        Returns:
            trend: Moving average with the same shape as x.
        """
        pad = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, pad, 1)
        back = x[:, -1:].repeat(1, self.kernel - 1 - pad, 1)
        padded = torch.cat([front, x, back], dim=1)
        return padded.unfold(1, self.kernel, 1).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.trend(x)
        seasonal = x - trend
        t = self.time_trend(trend.transpose(1, 2)).squeeze(-1)
        s = self.time_seasonal(seasonal.transpose(1, 2)).squeeze(-1)
        return self.head(t + s)

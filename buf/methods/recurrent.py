"""
PyTorch recurrent forecasters that predict the next day's OHLC prices.
"""

from __future__ import annotations

import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    """Batch norm, two stacked recurrent layers, then a dense head.

    Input is (batch, input_size, num_features). Output is (batch, num_features),
    the predicted next-day values in the same feature order as the input.
    Layer names match the converted checkpoints in ``checkpoints/``.
    """

    def __init__(
        self,
        cell: type[nn.Module],
        input_size: int,
        num_features: int,
        units: int,
        num_outputs: int | None = None,
    ) -> None:
        super().__init__()
        num_outputs = num_outputs or num_features
        # eps matches the Keras checkpoints this model was converted from.
        self.norm = nn.BatchNorm1d(input_size, eps=1e-3)
        self.rnn1 = cell(num_features, units, batch_first=True)
        self.rnn2 = cell(units, units, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(units, units // 2),
            nn.SiLU(),
            nn.Linear(units // 2, units // 2),
            nn.SiLU(),
            nn.Linear(units // 2, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.head(x[:, -1])


class LSTMForecaster(RecurrentForecaster):
    """RecurrentForecaster built from LSTM cells. Used by the demo."""

    def __init__(self, input_size: int, num_features: int, units: int, num_outputs: int | None = None) -> None:
        super().__init__(nn.LSTM, input_size, num_features, units, num_outputs)


class GRUForecaster(RecurrentForecaster):
    """RecurrentForecaster built from GRU cells. No trained checkpoints yet."""

    def __init__(self, input_size: int, num_features: int, units: int, num_outputs: int | None = None) -> None:
        super().__init__(nn.GRU, input_size, num_features, units, num_outputs)

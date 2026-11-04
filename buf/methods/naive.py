"""
Naive baseline: predict a return of 0.
"""

from __future__ import annotations

import numpy as np


class NaiveForecaster:
    """Predict a return of 0, so the forecast price equals the last close."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict a zero return for every window.

        Args:
            X: Flattened windows of shape (samples, features).

        Returns:
            pred: Zeros of shape (samples,).
        """
        return np.zeros(len(X))

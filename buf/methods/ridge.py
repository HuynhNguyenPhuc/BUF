"""
Ridge regression baseline on the flattened window.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeForecaster:
    """Ridge regression on the flattened window."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeForecaster:
        """
        Fit the regression.

        Args:
            X: Flattened training windows of shape (samples, features).
            y: Training targets.

        Returns:
            self: The fitted forecaster.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target of each window.

        Args:
            X: Flattened windows of shape (samples, features).

        Returns:
            pred: Predicted targets.
        """
        return self.model.predict(X)

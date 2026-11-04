"""
LightGBM baseline on the flattened window.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np


class LightGBMForecaster:
    """Gradient-boosted trees on the flattened window, with early stopping on validation."""

    def __init__(self, seed: int = 0, rounds: int = 1000, patience: int = 50) -> None:
        self.params = dict(
            objective='regression', learning_rate=0.03, num_leaves=63, feature_fraction=0.5,
            bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200, seed=seed, verbose=-1,
        )
        self.rounds = rounds
        self.patience = patience
        self.booster: lgb.Booster | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> LightGBMForecaster:
        """
        Train until the validation loss stops improving.

        Args:
            X: Flattened training windows of shape (samples, features).
            y: Training targets.
            X_val: Flattened validation windows.
            y_val: Validation targets.

        Returns:
            self: The fitted forecaster.
        """
        train = lgb.Dataset(X, y)
        valid = lgb.Dataset(X_val, y_val, reference=train)
        self.booster = lgb.train(
            self.params, train, num_boost_round=self.rounds, valid_sets=[valid],
            callbacks=[lgb.early_stopping(self.patience, verbose=False)],
        )
        return self

    @property
    def best_iteration(self) -> int:
        """Number of boosting rounds kept after early stopping."""
        return self.booster.best_iteration

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target of each window with the best round.

        Args:
            X: Flattened windows of shape (samples, features).

        Returns:
            pred: Predicted targets.
        """
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

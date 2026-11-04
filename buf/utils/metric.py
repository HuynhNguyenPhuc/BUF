"""Metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the root mean squared error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        rmse: Root mean squared error.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute percentage error, as a fraction (0.05 means 5%).

    Args:
        y_true: Actual values, all non-zero.
        y_pred: Predicted values.

    Returns:
        mape: Mean absolute percentage error.
    """
    return float(np.mean(np.abs(y_pred - y_true) / np.abs(y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination. It is negative when the model is worse than the mean.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        r2: Coefficient of determination.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def direction_accuracy(true_return: np.ndarray, pred_return: np.ndarray) -> float:
    """
    Compute the share of days where the predicted return has the same sign as the actual one.
    Days with an actual return of exactly zero are left out.

    Args:
        true_return: Actual returns.
        pred_return: Predicted returns.

    Returns:
        direction_accuracy: Share of days with matching sign.
    """
    keep = true_return != 0
    return float(np.mean(np.sign(pred_return[keep]) == np.sign(true_return[keep])))


def median_group_mape(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> float:
    """
    Compute the median of the per-group MAPE, so each ticker counts equally regardless of its size.

    Args:
        y_true: Actual values, all non-zero.
        y_pred: Predicted values.
        groups: Group label of each value, such as its ticker.

    Returns:
        median_group_mape: Median of the per-group MAPE.
    """
    errors = np.abs(y_pred - y_true) / np.abs(y_true)
    _, inverse = np.unique(groups, return_inverse=True)
    per_group = np.bincount(inverse, weights=errors) / np.bincount(inverse)
    return float(np.median(per_group))


def evaluate(
    base_close: np.ndarray,
    true_close: np.ndarray,
    pred_return: np.ndarray,
    true_return: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """
    Compute the standard metric set for one set of predictions.
    Predicted prices are base_close * exp(pred_return).

    Args:
        base_close: Close on the day before each predicted day.
        true_close: Actual close on each predicted day.
        pred_return: Predicted next-day log returns.
        true_return: Actual next-day log returns.
        groups: Group label of each value, such as its ticker.

    Returns:
        metrics: Price RMSE, MAPE, median group MAPE and R2, return RMSE and R2, and direction accuracy.
    """
    pred_close = base_close * np.exp(pred_return)
    return {
        "price_rmse": rmse(true_close, pred_close),
        "price_mape": mape(true_close, pred_close),
        "median_group_mape": median_group_mape(true_close, pred_close, groups),
        "price_r2": r2(true_close, pred_close),
        "return_rmse": rmse(true_return, pred_return),
        "return_r2": r2(true_return, pred_return),
        "direction_accuracy": direction_accuracy(true_return, pred_return),
    }

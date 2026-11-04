"""
ARIMA(1,0,1) baseline on the daily returns of one stock.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(returns: np.ndarray, train_size: int) -> np.ndarray:
    """
    Fit ARIMA(1,0,1) on the first train_size returns, then filter one step ahead over all of them.

    Args:
        returns: Daily log returns of one stock.
        train_size: Number of leading returns used to fit the parameters.

    Returns:
        pred: One-step-ahead prediction of every return.
    """
    fitted = ARIMA(returns[:train_size], order=(1, 0, 1), trend='n').fit()
    filtered = ARIMA(returns, order=(1, 0, 1), trend='n').filter(fitted.params)
    return np.asarray(filtered.predict(start=0, end=len(returns) - 1))

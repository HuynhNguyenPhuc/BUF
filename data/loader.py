"""
Load stock price CSV files and build model input windows.

Files live at ``data/<EXCHANGE>/<TICKER>.csv``. The file name is the ticker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_SIZE = 50
# Column order the checkpoints were trained with.
FEATURES = ["Open", "Close", "High", "Low"]


def list_stocks(data_dir: Path) -> dict[str, Path]:
    """
    Map EXCHANGE:TICKER labels to CSV paths.

    Args:
        data_dir: Folder holding one sub-folder of CSV files per exchange.

    Returns:
        stocks: Label to CSV path, sorted by path.
    """
    return {
        f"{path.parent.name}:{path.stem}": path
        for path in sorted(data_dir.glob("*/*.csv"))
    }


def load_prices(path: Path) -> pd.DataFrame:
    """
    Read a stock CSV and sort its rows by date.

    Args:
        path: Path to a stock CSV file.

    Returns:
        prices: Prices with a datetime Date column, oldest first.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.sort_values("Date").reset_index(drop=True)


def last_window(df: pd.DataFrame, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Take the last days of FEATURES as one model input.

    Args:
        df: Prices sorted by date, as returned by load_prices.
        input_size: Number of days in the window.

    Returns:
        window: Float32 array of shape (1, input_size, len(FEATURES)).
    """
    if len(df) < input_size:
        raise ValueError(f"Need at least {input_size} rows, got {len(df)}.")
    values = df[FEATURES].to_numpy(dtype="float32")
    return values[-input_size:][np.newaxis]

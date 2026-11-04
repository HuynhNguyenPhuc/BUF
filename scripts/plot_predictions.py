"""
Plot the next-day predictions of every method against the actual values for one stock.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "naive": "Naive", "arima(1,0,1)": "ARIMA", "ridge": "Ridge", "lightgbm": "LightGBM", "lstm": "LSTM",
    "gru": "GRU", "mlp": "MLP", "dlinear": "DLinear", "cnn": "CNN", "transformer": "Transformer",
}
INK, MUTE, TRUTH, PRED = "#1e293b", "#475569", "#db2777", "#2563eb"
CMAP = plt.get_cmap("tab10")


def load_window(path: Path, stock: str, days: int) -> pd.DataFrame:
    """
    Read the saved predictions and keep the last days of one stock.

    Args:
        path: Table of predicted returns written by the benchmark.
        stock: Stock label in the form EXCHANGE:TICKER.
        days: Number of trading days to keep.

    Returns:
        window: The last days of the stock, oldest first.
    """
    exchange, ticker = stock.split(":")
    table = pd.read_csv(path, parse_dates=["date"])
    table = table[(table["exchange"] == exchange) & (table["ticker"] == ticker)].sort_values("date")
    return table.tail(days).reset_index(drop=True)


def plot_prices(window: pd.DataFrame, methods: list[str], stock: str, out: Path) -> None:
    """
    Draw the actual close and the close predicted by each method.

    Args:
        window: Output of load_window.
        methods: Column names of the predicted returns.
        stock: Stock label, used as the title.
        out: Path of the image to write.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    ax.plot(window["date"], window["close"], color=INK, lw=2.4, label="Actual", zorder=5)
    for i, m in enumerate(methods):
        pred = window["base_close"] * np.exp(window[m])
        ax.plot(window["date"], pred, color=CMAP(i), lw=1, alpha=0.9, label=LABELS[m.rsplit("_", 1)[0]])
    ax.set_title(f"{stock}: actual and predicted next-day close", fontsize=10, color=INK, loc="left")
    ax.set_ylabel("Close price", fontsize=9, color=MUTE)
    ax.tick_params(labelsize=8, colors=MUTE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)


def plot_returns(window: pd.DataFrame, methods: list[str], stock: str, out: Path) -> None:
    """
    Draw one panel per method with its predicted return against the actual return.

    Args:
        window: Output of load_window.
        methods: Column names of the predicted returns.
        stock: Stock label, used as the title.
        out: Path of the image to write.
    """
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.2), dpi=200, sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), methods):
        ax.plot(window["date"], window["return"] * 100, color="#94a3b8", lw=1, label="Actual")
        ax.plot(window["date"], window[m] * 100, color=PRED, lw=1.4, label="Predicted")
        ax.axhline(0, color=MUTE, lw=0.6)
        ax.set_title(LABELS[m.rsplit("_", 1)[0]], fontsize=9, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=MUTE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Return (%)", fontsize=8, color=MUTE)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"{stock}: next-day return, actual and predicted", fontsize=10, color=INK, x=0.01, ha="left")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)


def main() -> None:
    """
    Write the price and return figures for one stock and seed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions_test.csv.gz"))
    parser.add_argument("--stock", default="HOSE:VNM")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    window = load_window(args.predictions, args.stock, args.days)
    order = ["naive", "arima(1,0,1)", "ridge", "lightgbm", "lstm", "gru", "mlp", "dlinear", "cnn", "transformer"]
    methods = [f"{m}_{args.seed}" for m in order]
    plot_prices(window, methods, args.stock, args.out / "predictions_price.png")
    plot_returns(window, methods, args.stock, args.out / "predictions_return.png")


if __name__ == "__main__":
    main()

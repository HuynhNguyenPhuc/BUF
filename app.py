"""
Gradio demo: predict the next trading day's prices for one stock.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import torch

from data.loader import FEATURES, INPUT_SIZE, last_window, list_stocks, load_prices
from buf.methods.recurrent import LSTMForecaster

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
STOCKS = list_stocks(ROOT / "data")
PLOT_DAYS = 60
NOTE = (
    "Prediction for the next trading day, using the last "
    f"{INPUT_SIZE} days of prices. The checkpoints come from an earlier model trained on an "
    "earlier version of the price data and have not been re-evaluated on the current data. "
    "On that earlier data the model was much less accurate than predicting that tomorrow's "
    "price equals today's price. For demonstration only; do not use it for decisions."
)

_models: dict[str, LSTMForecaster] = {}


def get_model(exchange: str) -> LSTMForecaster:
    """
    Load the LSTM for one exchange on first use, then reuse it.

    Args:
        exchange: Exchange name (HNX, HOSE or UPCOM), matching a file in checkpoints/.

    Returns:
        model: The LSTM forecaster in eval mode.
    """
    if exchange not in _models:
        model = LSTMForecaster(INPUT_SIZE, len(FEATURES), units=64)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{exchange}.pt", map_location="cpu"))
        model.eval()
        _models[exchange] = model
    return _models[exchange]


def predict(label: str) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Predict the next trading day's prices for one stock.

    Args:
        label: Stock label in the form EXCHANGE:TICKER.

    Returns:
        fig: Chart of recent closes with the predicted next close.
        table: Last-day and predicted Open, Close, High and Low.
    """
    exchange = label.split(":")[0]
    df = load_prices(STOCKS[label])
    with torch.no_grad():
        next_ohlc = get_model(exchange)(torch.from_numpy(last_window(df))).numpy()[0]

    last = df.iloc[-1]
    next_date = last["Date"] + pd.offsets.BDay(1)
    table = pd.DataFrame({
        "Feature": FEATURES,
        f"Last day ({last['Date']:%Y-%m-%d})": [float(last[f]) for f in FEATURES],
        f"Predicted ({next_date:%Y-%m-%d})": [float(v) for v in next_ohlc],
    })

    recent = df.tail(PLOT_DAYS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(recent["Date"], recent["Close"], label="Close")
    ax.scatter([next_date], [next_ohlc[FEATURES.index("Close")]], color="red", label="Predicted close")
    ax.set_title(label)
    ax.legend()
    fig.autofmt_xdate()
    return fig, table


with gr.Blocks(title="BUF stock demo") as demo:
    gr.Markdown(NOTE)
    stock = gr.Dropdown(choices=list(STOCKS), value=next(iter(STOCKS)), label="Stock (exchange:ticker)")
    run = gr.Button("Predict")
    plot = gr.Plot(label="Chart")
    table = gr.Dataframe(label="Prices")
    run.click(predict, inputs=stock, outputs=[plot, table])


if __name__ == "__main__":
    demo.launch()

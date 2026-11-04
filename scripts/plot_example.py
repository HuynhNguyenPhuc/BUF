"""
Plot one real input window and its target for the problem definition figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.loader import INPUT_SIZE, load_prices

TICKER = ("HOSE", "VNM")
TARGET_DATE = "2024-06-14"
PALETTE = {"ink": "#1e293b", "mute": "#475569", "blue": "#2563eb", "pink": "#db2777"}


def main() -> None:
    """
    Save a figure of the 50 closes before a chosen day and the close of that day.
    The output path is the first command-line argument.
    """
    out = Path(sys.argv[1])
    exchange, ticker = TICKER
    df = load_prices(Path("data") / exchange / f"{ticker}.csv")
    target = int(np.flatnonzero(df["Date"] == TARGET_DATE)[0])
    window = df.iloc[target - INPUT_SIZE:target]
    day = df.iloc[target]
    prev = window["Close"].iloc[-1]
    ret = np.log(day["Close"] / prev)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    ax.plot(window["Date"], window["Close"], color=PALETTE["blue"], lw=1.8, label="Closes over the 50-day input window")
    ax.scatter([day["Date"]], [day["Close"]], color=PALETTE["pink"], zorder=3, label="Target: close of day t")
    ax.annotate(
        rf"$r=\ln(C_t/C_{{t-1}})={ret:+.2%}$".replace("%", r"\%"),
        xy=(day["Date"], day["Close"]),
        xytext=(-150, 28 if ret >= 0 else -34),
        textcoords="offset points",
        color=PALETTE["pink"],
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": PALETTE["pink"]},
    )
    ax.set_title(f"{exchange}:{ticker}", fontsize=10, color=PALETTE["ink"], loc="left")
    ax.set_ylabel("Close price", fontsize=9, color=PALETTE["mute"])
    ax.tick_params(labelsize=8, colors=PALETTE["mute"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()

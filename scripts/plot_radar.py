"""
Plot the benchmark summary as one radar chart per method, one figure per scope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "gru": "GRU", "lstm": "LSTM", "transformer": "Transformer", "cnn": "CNN", "lightgbm": "LightGBM",
    "mlp": "MLP", "ridge": "Ridge", "dlinear": "DLinear", "arima(1,0,1)": "ARIMA",
}
BASELINES = {"arima(1,0,1)", "ridge", "lightgbm"}
AXES = [
    ("return_rmse", "Return RMSE\nreduction", True),
    ("return_r2", "Return R²", False),
    ("direction_accuracy", "Direction\naccuracy", False),
    ("price_mape", "Price MAPE\nreduction", True),
    ("median_group_mape", "Median per-ticker\nMAPE reduction", True),
]
COLORS = {"neural": "#2563eb", "baseline": "#64748b"}
INK, MUTE = "#1e293b", "#475569"
FLOOR = 0.15


def axis_values(data: pd.DataFrame, naive: pd.Series) -> pd.DataFrame:
    """
    Compute the value of every method on every axis; error metrics become the percentage reduction
    against naive, so that higher is better on every axis.

    Args:
        data: Mean of each metric, indexed by method, without naive.
        naive: Mean of each metric for naive.

    Returns:
        values: Methods as rows and axes as columns.
    """
    out = {}
    for metric, label, relative in AXES:
        out[label] = (1 - data[metric] / naive[metric]) * 100 if relative else data[metric]
    return pd.DataFrame(out)


def plot_scope(summary: pd.DataFrame, scope: str, out: Path) -> None:
    """
    Draw one radar per method. Each axis runs from the worst method (inner ring) to the best method
    (outer ring) of the scope, so only the relative position of the methods is shown; the best raw value
    of every axis is printed at its tip.

    Args:
        summary: Mean and standard deviation of each metric, indexed by scope and method.
        scope: Scope to draw, such as full_test or subset_test.
        out: Path of the image to write.
    """
    means = summary.loc[scope].xs("mean", axis=1, level=1)
    values = axis_values(means.drop("naive"), means.loc["naive"])
    values = values.loc[(-means.drop("naive")["return_rmse"]).sort_values().index[::-1]]
    scaled = FLOOR + (1 - FLOOR) * (values - values.min()) / (values.max() - values.min())
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(angles, angles[0])
    cols = 4
    rows = int(np.ceil(len(values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.9 * rows), dpi=200, subplot_kw={"polar": True}, gridspec_kw={"hspace": 0.45, "wspace": 0.55})
    for ax, method in zip(axes.ravel(), values.index):
        color = COLORS["baseline"] if method in BASELINES else COLORS["neural"]
        for other in values.index:
            ring = np.append(scaled.loc[other], scaled.loc[other].iloc[0])
            ax.plot(closed, ring, color="#cbd5e1", lw=0.6)
        ring = np.append(scaled.loc[method], scaled.loc[method].iloc[0])
        ax.plot(closed, ring, color=color, lw=1.8)
        ax.fill(closed, ring, color=color, alpha=0.25)
        ax.set_ylim(0, 1)
        ax.set_yticks([FLOOR, 0.5, 1])
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels([a[1] for a in AXES], fontsize=7.5, color=MUTE)
        ax.tick_params(axis="x", pad=9)
        ax.grid(color="#e2e8f0", lw=0.5)
        ax.spines["polar"].set_color("#e2e8f0")
        ax.set_title(LABELS[method], fontsize=11, color=INK, pad=14, fontweight="bold")
    for ax in axes.ravel()[len(values):]:
        ax.axis("off")
    best = "; ".join(f"{label.replace(chr(10), ' ')} {values[label].min():.3g} to {values[label].max():.3g}" for _, label, _ in AXES)
    handles = [plt.Line2D([0], [0], color=c, lw=3) for c in COLORS.values()]
    fig.legend(handles, ["Neural network", "Baseline"], loc="lower left", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0, 0.01))
    note = (
        "Each axis is min-max scaled within this scope: the worst method sits on the inner ring, the best on the outer ring.\n"
        "The chart shows relative position only; radars of different scopes are not comparable. Naive is not drawn; grey lines are the other methods.\n"
        "Raw ranges, worst to best: " + best.replace("; ", "\n", 2).replace("; ", " | ")
    )
    fig.text(0.995, 0.012, note, ha="right", va="bottom", fontsize=7.5, color=MUTE, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """
    Read the summary table and write one radar image per scope.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    summary = pd.read_csv(args.results / "results_summary.csv", header=[0, 1], index_col=[0, 1])
    for scope in summary.index.get_level_values(0).unique():
        plot_scope(summary, scope, args.out / f"radar_{scope}.png")


if __name__ == "__main__":
    main()

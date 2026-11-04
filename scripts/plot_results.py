"""
Plot the benchmark summary, one figure per scope: bars for the error metrics and R2, dots for direction accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "gru": "GRU", "lstm": "LSTM", "transformer": "Transformer", "cnn": "CNN", "lightgbm": "LightGBM",
    "mlp": "MLP", "ridge": "Ridge", "dlinear": "DLinear", "arima(1,0,1)": "ARIMA", "naive": "Naive",
}
BASELINES = {"naive", "arima(1,0,1)", "ridge", "lightgbm"}
PANELS = [
    ("return_rmse", "Return RMSE: reduction vs naive (%)"),
    ("return_r2", "Return R² (higher is better)"),
    ("direction_accuracy", "Direction accuracy"),
    ("price_mape", "Price MAPE: reduction vs naive (%)"),
]
RELATIVE = {"return_rmse", "price_mape"}
COLORS = {"neural": "#2563eb", "baseline": "#94a3b8"}
INK, MUTE, REFERENCE = "#1e293b", "#475569", "#db2777"


def majority_sign(predictions: Path) -> dict[str, float]:
    """
    Compute the accuracy of always predicting the more frequent sign of the non-zero returns.

    Args:
        predictions: Table of test predictions written by the benchmark.

    Returns:
        reference: Scope name to majority-sign accuracy.
    """
    t = pd.read_csv(predictions, usecols=["return", "subset"])
    out = {}
    for scope, part in (("full_test", t), ("subset_test", t[t["subset"]])):
        nonzero = part["return"][part["return"] != 0]
        p = (nonzero > 0).mean()
        out[scope] = max(p, 1 - p)
    return out


def plot_scope(summary: pd.DataFrame, scope: str, reference: float, out: Path) -> None:
    """
    Draw one panel per metric for one scope, sorted by return RMSE. Error metrics are shown as the
    percentage reduction relative to the naive forecast; error bars are the standard deviation over seeds.

    Args:
        summary: Mean and standard deviation of each metric, indexed by scope and method.
        scope: Scope to draw, such as full_test or subset_test.
        reference: Majority-sign accuracy of the scope.
        out: Path of the image to write.
    """
    data = summary.loc[scope].sort_values(("return_rmse", "mean"))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 3.6), dpi=200, sharey=True)
    labels = [LABELS[m] for m in data.index]
    colors = [COLORS["baseline"] if m in BASELINES else COLORS["neural"] for m in data.index]
    for ax, (metric, title) in zip(axes, PANELS):
        mean, std = data[(metric, "mean")], data[(metric, "std")].fillna(0)
        if metric in RELATIVE:
            naive = mean["naive"]
            mean, std = (1 - mean / naive) * 100, std / naive * 100
        if metric == "direction_accuracy":
            keep = data.index != "naive"
            ax.errorbar(mean[keep], [labels[i] for i, k in enumerate(keep) if k], xerr=std[keep], fmt="o", color=INK, ms=4, lw=1)
            ax.axvline(reference, color=REFERENCE, lw=1.2, ls="--")
            ax.text(reference, -0.9, f"majority sign {reference:.3f}", color=REFERENCE, fontsize=7, ha="left", va="bottom")
        else:
            ax.barh(labels, mean, xerr=std, color=colors, height=0.68, error_kw={"lw": 1, "ecolor": INK})
            if metric in RELATIVE:
                ax.axvline(0, color=MUTE, lw=1)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=14)
        ax.tick_params(labelsize=8, colors=MUTE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, ["Neural network", "Baseline"], loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.text(0.995, 0.01, "Naive is the zero reference of the first and last panels.", ha="right", fontsize=7, color=MUTE)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)


def main() -> None:
    """
    Read the summary table and write one image per scope.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    summary = pd.read_csv(args.results / "results_summary.csv", header=[0, 1], index_col=[0, 1])
    reference = majority_sign(args.results / "predictions_test.csv.gz")
    for scope in summary.index.get_level_values(0).unique():
        plot_scope(summary, scope, reference[scope], args.out / f"results_{scope}.png")


if __name__ == "__main__":
    main()

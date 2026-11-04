"""
Convert the Keras .h5 weights of the original LSTM into PyTorch checkpoints.

Usage: python -m scripts.convert <h5 folder> <checkpoint folder>
Only h5py is needed; TensorFlow is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import torch

from buf.methods.recurrent import LSTMForecaster

INPUT_SIZE = 50
NUM_FEATURES = 4
UNITS = 64


def read_layers(path: Path) -> list[list]:
    """
    Read the weights of each layer that has any, in model order.

    Args:
        path: Keras .h5 weights file.

    Returns:
        layers: Weight arrays of each layer.
    """
    layers = []
    with h5py.File(path, "r") as f:
        for name in f.attrs["layer_names"]:
            name = name.decode() if isinstance(name, bytes) else name
            group = f[name]
            weight_names = [w.decode() if isinstance(w, bytes) else w
                            for w in group.attrs["weight_names"]]
            if weight_names:
                layers.append([group[w][()] for w in weight_names])
    return layers


def to_state_dict(layers: list[list]) -> dict[str, torch.Tensor]:
    """
    Map Keras weights onto the state dict of LSTMForecaster.
    Keras stores kernels as (in, out) and PyTorch as (out, in), so each is transposed.
    Both use the gate order i, f, c/g, o. Keras keeps one bias per LSTM; PyTorch has
    two, so the second is set to zero.

    Args:
        layers: Weight arrays of each layer, as returned by read_layers.

    Returns:
        state: State dict that LSTMForecaster can load.
    """
    bn, lstm1, lstm2, dense1, dense2, dense3 = layers
    t = torch.from_numpy
    state = {
        "norm.weight": t(bn[0].copy()),
        "norm.bias": t(bn[1].copy()),
        "norm.running_mean": t(bn[2].copy()),
        "norm.running_var": t(bn[3].copy()),
    }
    for prefix, (kernel, recurrent_kernel, bias) in [("rnn1", lstm1), ("rnn2", lstm2)]:
        state[f"{prefix}.weight_ih_l0"] = t(kernel.T.copy())
        state[f"{prefix}.weight_hh_l0"] = t(recurrent_kernel.T.copy())
        state[f"{prefix}.bias_ih_l0"] = t(bias.copy())
        state[f"{prefix}.bias_hh_l0"] = torch.zeros_like(state[f"{prefix}.bias_ih_l0"])
    for index, (kernel, bias) in zip([0, 2, 4], [dense1, dense2, dense3]):
        state[f"head.{index}.weight"] = t(kernel.T.copy())
        state[f"head.{index}.bias"] = t(bias.copy())
    return state


def main() -> None:
    h5_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for h5_path in sorted(h5_dir.glob("*_model.h5")):
        exchange = h5_path.name.removesuffix("_model.h5")
        model = LSTMForecaster(INPUT_SIZE, NUM_FEATURES, UNITS)
        model.load_state_dict(to_state_dict(read_layers(h5_path)))
        torch.save(model.state_dict(), out_dir / f"{exchange}.pt")
        print(f"{exchange}: saved")


if __name__ == "__main__":
    main()

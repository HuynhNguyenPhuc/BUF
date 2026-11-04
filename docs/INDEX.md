# Index: Concepts, Code and Files

This document maps the concepts of the Problem, Methods, Metrics and Results documents to the code, and lists the data, result files, figures and run options. Paths are relative to the repository root. It is the only document that names code, files and flags.

## 1. Concept to code

| Concept | Code |
|---|---|
| Benchmark run | `scripts/benchmark.py` |
| Naive | `buf/methods/naive.py` (`NaiveForecaster`) |
| ARIMA(1,0,1) | `buf/methods/arima.py` |
| Ridge | `buf/methods/ridge.py` (`RidgeForecaster`) |
| LightGBM | `buf/methods/lgbm.py` (`LightGBMForecaster`) |
| LSTM, GRU | `buf/methods/recurrent.py` (`LSTMForecaster`, `GRUForecaster`, base `RecurrentForecaster`) |
| MLP, DLinear | `buf/methods/mlp.py` (`MLPForecaster`, `DLinearForecaster`) |
| CNN | `buf/methods/cnn.py` (`CNNForecaster`) |
| Transformer | `buf/methods/transformer.py` (`TransformerForecaster`) |
| Metrics | `buf/utils/metric.py`: `rmse`, `mape`, `r2`, `direction_accuracy`, `median_group_mape`; `evaluate` returns the set with the keys `price_rmse`, `price_mape`, `median_group_mape`, `price_r2`, `return_rmse`, `return_r2`, `direction_accuracy` |
| Day features | `daily_features` in `scripts/benchmark.py` |
| 50-day samples | `build_samples` in `scripts/benchmark.py` |
| Window length, split dates | `L`, `TRAIN_END`, `TEST_START` in `scripts/benchmark.py` |
| Data loading and filter | `load_frame` in `scripts/benchmark.py` (keeps each duplicated ticker once, in the order HOSE, HNX, UPCOM; series with fewer than `MIN_ROWS` rows are dropped) |
| 90-stock subset | `draw_subset` in `scripts/benchmark.py`; `SUBSET_PER_EXCHANGE = 30`; `numpy.random.default_rng(0)`; only stocks with more than `L + 1` rows are eligible |
| Training settings | `BATCH`, `EVAL_BATCH`, `PATIENCE` in `scripts/benchmark.py`; epochs and boosting rounds from `--epochs` and `--rounds` |
| Demo | `app.py`, with `data/loader.py` and `checkpoints/<EXCHANGE>.pt` (`HNX.pt`, `HOSE.pt`, `UPCOM.pt`) |
| Weights conversion (Keras `.h5` to PyTorch) | `scripts/convert.py` |
| Price refresh | `scripts/fetch_history.py` |

The demo's model is `LSTMForecaster` built with `units=64`; its input is the raw prices in the order open, close, high, low (`FEATURES` in `data/loader.py`). `GRUForecaster` has no demo checkpoint.

## 2. Data

- Files: `data/<EXCHANGE>/<TICKER>.csv`, columns `Date` (`MM/DD/YYYY`), `Open`, `High`, `Low`, `Close`, `Volume`.
- Duplicated tickers, identical files under two exchanges: `BID`, `CII`, `CMS`, `CTP`, `CVN`, `LPB`, `MSN` (HNX and HOSE) and `CMW` (HOSE and UPCOM).
- Price refresh: `python scripts/fetch_history.py --out <dir> [--start --end --pause]`.

## 3. Results files (`results/`)

| File | Contents |
|---|---|
| `results_raw.csv` | Scores of every method, seed and scope |
| `results_summary.csv` | Mean and standard deviation over seeds, by scope and method; read by the plotting scripts |
| `results_raw_long_budget.csv`, `results_summary_long_budget.csv` | Seed 0 rerun with 20 epochs and 3,000 boosting rounds (Results §7) |
| `benchmark.log`, `benchmark_long_budget.log` | Run logs |
| `predictions_test.csv.gz` | Seed 0 test predictions for every full-test sample, written by `--save-predictions`; the `subset` column marks the 90-stock subset rows, and ARIMA is filled only on those rows |
| `with_duplicates/` | Earlier run that counted the duplicated tickers twice (Results §8) |

## 4. Figures (`docs/figures/`)

| Figure | Source |
|---|---|
| `results_full_test.png`, `results_subset_test.png` | `scripts/plot_results.py` (reads `results_summary.csv` and `predictions_test.csv.gz`) |
| `radar_full_test.png`, `radar_subset_test.png` | `scripts/plot_radar.py` (min-max scaling within each scope; naive not drawn) |
| `predictions_price.png`, `predictions_return.png` | `scripts/plot_predictions.py`; the [README](../README.md) example uses `--stock HOSE:VCI --seed 0 --days 60` (defaults: `HOSE:VNM`, seed 0, 60 days) |
| `example.png` | `scripts/plot_example.py <output path>` (sample of HOSE:VNM) |
| `task.svg`, `split.svg`, `metrics.svg`, `pipeline.svg` | Hand-drawn diagrams, not generated |
| `networks.svg` | `scripts/draw_methods.py` (overview of the five architectures; `--networks` sets the output path) |
| `methods/*.svg` (LSTM, GRU, Transformer, LightGBM, CNN, DLinear, MLP, Ridge, ARIMA, Naive) | `scripts/draw_methods.py` (schematic diagrams, one per method; the tree and the validation curve in the LightGBM figure are schematic) |

## 5. Run

```bash
pip install -r requirements.txt
python -m scripts.benchmark --data data --out results     # --quick for a smoke test
python app.py                                             # demo
```

Options of `scripts/benchmark.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--models` | all ten methods | Subset of methods to run |
| `--seeds` | `0,1,2` | Seeds |
| `--epochs` | 6 | Maximum epochs of the neural networks |
| `--rounds` | 1000 | Maximum boosting rounds of LightGBM |
| `--train-cap` | 0 | Maximum number of training samples, the same for every method (0 = all) |
| `--subset-per-exchange` | 30 | Stocks per exchange in the subset |
| `--save-predictions` | off | Write `predictions_test.csv.gz` |
| `--quick` | off | Smoke test: 20 boosting rounds and 3 stocks per exchange |

The full run holds all 2.9 million windows in memory several times over. It ran out of memory on a machine with 7.7 GB of RAM and completed on one with 62 GB. The reported run took about 70 minutes on a shared NVIDIA TITAN RTX.

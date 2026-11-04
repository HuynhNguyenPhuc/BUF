"""
Benchmark next-day forecasting for stocks on HNX, HOSE and UPCOM.

Compares naive, ARIMA, Ridge, LightGBM and neural models (LSTM, GRU, MLP, DLinear, CNN,
Transformer) on the next-day log return of Close, from a window of the last 50 days of
log-return features (see docs/METHODS.md). Data is split by date (TRAIN_END, TEST_START).
Writes results_raw.csv and results_summary.csv to --out.

    python -m scripts.benchmark --data data --out results    # add --quick for a smoke test
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from buf.methods.arima import arima_forecast
from buf.methods.cnn import CNNForecaster
from buf.methods.lgbm import LightGBMForecaster
from buf.methods.mlp import DLinearForecaster, MLPForecaster
from buf.methods.naive import NaiveForecaster
from buf.methods.recurrent import GRUForecaster, LSTMForecaster
from buf.methods.ridge import RidgeForecaster
from buf.methods.transformer import TransformerForecaster
from buf.utils.metric import evaluate
from data.loader import list_stocks, load_prices

warnings.filterwarnings('ignore')

L = 50
NUM_FEATURES = 5
UNITS = 64
MIN_ROWS = 300
EXCHANGE_PRIORITY = ['HOSE', 'HNX', 'UPCOM']
TRAIN_END = pd.Timestamp('2022-11-30')
TEST_START = pd.Timestamp('2023-06-01')
SUBSET_PER_EXCHANGE = 30
BATCH = 2048
EVAL_BATCH = 2048
PATIENCE = 3
ALL_METHODS = [
    'naive', 'arima', 'ridge', 'lightgbm',
    'lstm', 'gru', 'mlp', 'dlinear', 'cnn', 'transformer',
]
T0 = time.time()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='folder with <EXCHANGE>/<TICKER>.csv')
    p.add_argument('--out', required=True)
    p.add_argument('--quick', action='store_true', help='tiny smoke-test run')
    p.add_argument('--models', default=','.join(ALL_METHODS),
                   help=f'comma-separated subset of: {",".join(ALL_METHODS)}')
    p.add_argument('--epochs', type=int, default=6, help='max epochs for neural models')
    p.add_argument('--seeds', default='0,1,2', help='comma-separated seeds')
    p.add_argument('--train-cap', type=int, default=0,
                   help='keep at most this many training samples (0 = all), same sample for every method')
    p.add_argument('--subset-per-exchange', type=int, default=SUBSET_PER_EXCHANGE)
    p.add_argument('--rounds', type=int, default=1000, help='max boosting rounds for LightGBM')
    p.add_argument('--save-predictions', action='store_true',
                   help='also write the predicted returns of every method on the test samples')
    args = p.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(',')]
    unknown = set(args.models.split(',')) - set(ALL_METHODS)
    if unknown:
        p.error(f'unknown methods: {sorted(unknown)}')
    args.models = args.models.split(',')
    return args


def load_frame(data_dir):
    """
    Load every stock CSV with its exchange and ticker, dropping series shorter than MIN_ROWS. A ticker
    whose identical file appears under several exchanges is kept once, under the first exchange of
    EXCHANGE_PRIORITY; the exchange label of such a ticker is therefore not verified.

    Args:
        data_dir: Folder holding one sub-folder of CSV files per exchange.

    Returns:
        d: All rows, with exchange and Ticker columns.
    """
    frames, seen = [], {}
    stocks = sorted(list_stocks(Path(data_dir)).items(), key=lambda kv: (EXCHANGE_PRIORITY.index(kv[0].split(':')[0]), kv[0]))
    for label, path in stocks:
        exchange, ticker = label.split(':')
        prices = load_prices(path)
        assert prices['Date'].is_unique, f'{label} has repeated dates'
        if ticker in seen:
            assert prices.equals(seen[ticker]), f'{ticker} differs between exchanges'
            continue
        seen[ticker] = prices
        frames.append(prices.assign(exchange=exchange, Ticker=ticker))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(['exchange', 'Ticker', 'Date'])
    sizes = d.groupby(['exchange', 'Ticker'])['Date'].transform('size')
    return d[sizes >= MIN_ROWS]


def daily_features(g):
    """
    Compute per-day features relative to the previous close. Row k describes day k+1.

    Args:
        g: Prices of one stock, sorted by date.

    Returns:
        C: Close prices.
        F: Features of shape (days - 1, NUM_FEATURES).
    """
    C, O, H, Lo, V = (g[c].to_numpy(float) for c in ['Close', 'Open', 'High', 'Low', 'Volume'])
    prev = C[:-1]
    F = np.stack([
        np.log(C[1:] / prev),
        np.log(O[1:] / prev),
        np.log(H[1:] / prev),
        np.log(Lo[1:] / prev),
        np.log1p(V[1:]),
    ], axis=1)
    return C, F


def build_samples(d):
    cols = {k: [] for k in ['X', 'ret', 'base', 'close', 'date', 'exch', 'tick', 'hidx']}
    for (exch, tick), g in d.groupby(['exchange', 'Ticker'], sort=False):
        C, F = daily_features(g)
        n = len(C)
        m = n - 1
        if m <= L:
            continue
        W = sliding_window_view(F, (L, NUM_FEATURES))[:, 0][:m - L]
        cols['X'].append(W.astype(np.float32))
        cols['ret'].append(F[L:, 0])
        cols['base'].append(C[L:n - 1])
        cols['close'].append(C[L + 1:])
        cols['date'].append(g['Date'].to_numpy()[L + 1:])
        cols['exch'].append(np.full(m - L, exch))
        cols['tick'].append(np.full(m - L, tick))
        cols['hidx'].append(np.arange(m - L) + L)
    return {k: np.concatenate(v) for k, v in cols.items()}


def draw_subset(frame, per_exchange):
    """
    Draw the stocks of the evaluation subset: per_exchange stocks at random from each exchange, among
    the stocks with enough rows to form at least one sample.

    Args:
        frame: All rows, as returned by load_frame.
        per_exchange: Number of stocks to draw per exchange.

    Returns:
        keys: (exchange, ticker) pairs, HNX first, then HOSE, then UPCOM.
    """
    rng = np.random.default_rng(0)
    sizes = frame.groupby(['exchange', 'Ticker']).size()
    eligible = sizes[sizes > L + 1].index.to_frame(index=False)
    keys = []
    for exch in ['HNX', 'HOSE', 'UPCOM']:
        tickers = sorted(set(eligible.loc[eligible['exchange'] == exch, 'Ticker']))
        pick = rng.choice(tickers, size=min(per_exchange, len(tickers)), replace=False)
        keys += [(exch, t) for t in pick]
    return keys


def metrics(P, mask, S):
    return evaluate(S['base'][mask], S['close'][mask], P[mask], S['ret'][mask],
                    S['exch'][mask] + '|' + S['tick'][mask])


NETWORKS = {
    'lstm': LSTMForecaster,
    'gru': GRUForecaster,
    'mlp': MLPForecaster,
    'dlinear': DLinearForecaster,
    'cnn': CNNForecaster,
    'transformer': TransformerForecaster,
}


def predict(model, X):
    """
    Run the model over X in chunks.

    Args:
        model: Network to evaluate.
        X: Input windows.

    Returns:
        pred: Flat array with one prediction per window.
    """
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[j:j + EVAL_BATCH].to(DEVICE)).reshape(-1).cpu()
                          for j in range(0, len(X), EVAL_BATCH)]).numpy()


def fit_net(name, seed, Xn, y, tr, va, te, ysd, max_epochs):
    """
    Train one network on the train split, keep the best validation epoch, and predict the test split.

    Args:
        name: Key of NETWORKS.
        seed: Random seed.
        Xn: Standardized windows of all samples.
        y: Standardized targets of all samples.
        tr: Mask of training samples.
        va: Mask of validation samples.
        te: Mask of test samples.
        ysd: Standard deviation of the training returns.
        max_epochs: Maximum number of epochs.

    Returns:
        pred: Predicted log returns of the test samples.
    """
    torch.manual_seed(seed)
    model = NETWORKS[name](L, NUM_FEATURES, UNITS, num_outputs=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, ytr = torch.from_numpy(Xn[tr]), torch.from_numpy(y[tr])
    Xva, yva = torch.from_numpy(Xn[va]), torch.from_numpy(y[va])
    best, best_ep, state, bad = np.inf, -1, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)
            loss = ((model(xb).reshape(-1) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = ((torch.from_numpy(predict(model, Xva)) - yva) ** 2).mean().item()
        log(f'{name} seed={seed} epoch={ep} val_mse_std={vl:.4f}')
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    log(f'{name} seed={seed} best_epoch={best_ep} epochs_run={ep + 1} best_val_mse={best:.10f}')
    model.load_state_dict(state)
    return predict(model, torch.from_numpy(Xn[te])) * ysd


def arima_predictions(d, keys):
    """
    Run ARIMA(1,0,1) on the daily log returns of the selected stocks, fitted up to TRAIN_END.

    Args:
        d: All rows, as returned by load_frame.
        keys: (exchange, ticker) pairs to fit.

    Returns:
        out: Predicted log return of every day, per (exchange, ticker).
    """
    out = {}
    for (exch, tick), g in d.groupby(['exchange', 'Ticker'], sort=False):
        if (exch, tick) not in keys:
            continue
        _, F = daily_features(g)
        train_size = int((g['Date'].to_numpy()[1:] <= np.datetime64(TRAIN_END)).sum())
        out[(exch, tick)] = arima_forecast(F[:, 0], train_size)
    return out


def main(argv=None):
    args = parse_args(argv)
    quick = args.quick
    subset_per_exchange = 3 if quick else args.subset_per_exchange
    seeds = [0] if quick else args.seeds
    max_epochs = 1 if quick else args.epochs
    log('device', DEVICE)
    lgb_rounds = 20 if quick else args.rounds
    want = set(args.models)
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    frame = load_frame(args.data)
    log('rows', len(frame), 'series', frame.groupby(['exchange', 'Ticker']).ngroups)
    S = build_samples(frame)
    N = len(S['ret'])
    log('samples', N)

    date = S['date']
    tr = date <= np.datetime64(TRAIN_END)
    va = (date > np.datetime64(TRAIN_END)) & (date < np.datetime64(TEST_START))
    te = date >= np.datetime64(TEST_START)
    if args.train_cap and tr.sum() > args.train_cap:
        keep = np.random.default_rng(0).choice(np.flatnonzero(tr), args.train_cap, replace=False)
        tr = np.zeros(N, dtype=bool)
        tr[keep] = True
    log('train', tr.sum(), 'val', va.sum(), 'test', te.sum())

    mu = S['X'][tr].reshape(-1, NUM_FEATURES).mean(0)
    sd = S['X'][tr].reshape(-1, NUM_FEATURES).std(0) + 1e-8
    Xn = ((S['X'] - mu) / sd).astype(np.float32)
    Xf = Xn.reshape(N, -1)
    ysd = float(S['ret'][tr].std())
    y = (S['ret'] / ysd).astype(np.float32)

    subset_keys = draw_subset(frame, subset_per_exchange)
    subset = np.zeros(N, dtype=bool)
    for exch, t in subset_keys:
        subset |= (S['exch'] == exch) & (S['tick'] == t)
    log('subset tickers', len(subset_keys), 'subset test samples', (subset & te).sum())

    preds = {}
    if 'naive' in want:
        preds[('naive', 0)] = NaiveForecaster().predict(Xf)

    if 'arima' in want:
        ar = arima_predictions(frame, set(subset_keys))
        P = np.full(N, np.nan)
        for exch, t in subset_keys:
            idx = np.where((S['exch'] == exch) & (S['tick'] == t) & te)[0]
            P[idx] = ar[(exch, t)][S['hidx'][idx]]
        preds[('arima(1,0,1)', 0)] = P
        log('arima done')

    if 'ridge' in want:
        ridge = RidgeForecaster().fit(Xf[tr], y[tr])
        P = np.full(N, np.nan)
        P[te] = ridge.predict(Xf[te]) * ysd
        preds[('ridge', 0)] = P
        log('ridge done')

    for seed in seeds:
        if 'lightgbm' in want:
            booster = LightGBMForecaster(seed, lgb_rounds).fit(Xf[tr], y[tr], Xf[va], y[va])
            P = np.full(N, np.nan)
            P[te] = booster.predict(Xf[te]) * ysd
            preds[('lightgbm', seed)] = P
            log('lightgbm done seed', seed, 'best_iter', booster.best_iteration)

        for name in ['lstm', 'gru', 'mlp', 'dlinear', 'cnn', 'transformer']:
            if name not in want:
                continue
            P = np.full(N, np.nan)
            P[te] = fit_net(name, seed, Xn, y, tr, va, te, ysd, max_epochs)
            preds[(name, seed)] = P
            log(name, 'done seed', seed)

    rows = []
    for (name, seed), P in preds.items():
        for scope, mask in [('full_test', te), ('subset_test', te & subset)]:
            if scope == 'full_test' and name.startswith('arima'):
                continue  # ARIMA only fitted on the subset
            rows.append({'method': name, 'seed': seed, 'scope': scope, **metrics(P, mask, S)})
    if args.save_predictions:
        keep = te
        table = pd.DataFrame({
            'exchange': S['exch'][keep], 'ticker': S['tick'][keep], 'date': S['date'][keep],
            'base_close': S['base'][keep], 'close': S['close'][keep], 'return': S['ret'][keep],
            'subset': subset[keep],
        })
        for (name, seed), P in preds.items():
            table[f'{name}_{seed}'] = P[keep]
        table.to_csv(os.path.join(args.out, 'predictions_test.csv.gz'), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, 'results_raw.csv'), index=False)

    num = [c for c in res.columns if c not in ['method', 'seed', 'scope']]
    summ = res.groupby(['scope', 'method'])[num].agg(['mean', 'std'])
    summ.to_csv(os.path.join(args.out, 'results_summary.csv'))
    log('saved results to', args.out)
    print(summ.round(5).to_string())


if __name__ == '__main__':
    main()

"""
Download full daily price history for every ticker in data/ from agentic-vnstock.

Writes one CSV per ticker to --out/<EXCHANGE>/<TICKER>.csv, with the columns the
price loader reads: Date (MM/DD/YYYY), Open, High, Low, Close, Volume.

The ticker list comes from the existing files in data/. Tickers already written to
--out are skipped, so an interrupted run can be resumed. Run it with a Python
environment that has agentic-vnstock installed.

Usage: python scripts/fetch_history.py --out <folder> [--start 2015-01-01] [--pause 0.5]
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd
from agentic_vnstock import AgenticVNStock

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
ATTEMPTS = 3
STOP_AFTER_FAILURES = 20


class NoDataError(Exception):
    """The source answered, but with no rows for this ticker."""


def universe(data_dir: Path) -> list[tuple[str, str]]:
    """
    List the tickers of the existing data folder.

    Args:
        data_dir: Folder holding one sub-folder of CSV files per exchange.

    Returns:
        pairs: Sorted (exchange, ticker) pairs.
    """
    return sorted((p.parent.name, p.stem) for p in data_dir.glob("*/*.csv"))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the loader's columns, sort by date, and reject empty or non-positive prices.

    Args:
        df: Raw history from the source.

    Returns:
        df: Cleaned history with Date formatted as MM/DD/YYYY.
    """
    if not set(COLUMNS).issubset(df.columns):
        raise NoDataError("source returned no rows")
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates("Date").sort_values("Date")
    if df.empty or df[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("empty or missing prices")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("non-positive price")
    df["Date"] = df["Date"].dt.strftime("%m/%d/%Y")
    return df


def fetch(client: AgenticVNStock, ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch one ticker, retrying a few times with growing waits.

    Args:
        client: Client of the price source.
        ticker: Ticker to download.
        start: First date, as YYYY-MM-DD.
        end: Last date, as YYYY-MM-DD.

    Returns:
        df: Cleaned history of the ticker.
    """
    last_error: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return clean(client.get_stock_historical(ticker, start=start, end=end))
        except NoDataError:
            raise
        except Exception as error:  # the source is unofficial and may fail in many ways
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{ticker}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between tickers")
    args = parser.parse_args()

    client = AgenticVNStock()
    failures: list[str] = []
    consecutive = 0
    pairs = universe(DATA_DIR)
    for i, (exchange, ticker) in enumerate(pairs, 1):
        target = args.out / exchange / f"{ticker}.csv"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch(client, ticker, args.start, args.end).to_csv(target, index=False)
            consecutive = 0
        except NoDataError as error:
            failures.append(f"{exchange}/{ticker}: {error}")
            print(f"[{i}/{len(pairs)}] NO DATA {exchange}/{ticker}", flush=True)
        except Exception as error:
            failures.append(f"{exchange}/{ticker}: {error}")
            consecutive += 1
            print(f"[{i}/{len(pairs)}] FAIL {exchange}/{ticker}: {error}", flush=True)
            if consecutive >= STOP_AFTER_FAILURES:
                print("Stopping: too many failures in a row. The source may be blocking requests.", flush=True)
                break
        if i % 50 == 0:
            print(f"[{i}/{len(pairs)}] done", flush=True)
        time.sleep(args.pause)

    (args.out / "_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf8")
    written = sum(1 for _ in args.out.glob("*/*.csv"))
    print(f"finished: {written} files in {args.out}, {len(failures)} tickers without data or errors", flush=True)


if __name__ == "__main__":
    main()

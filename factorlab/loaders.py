"""Real-data loader (optional). Writes the same CSV schema as synth.py.

    pip install yfinance
    python -m factorlab fetch --tickers tickers.txt --start 2012-01-01

Caveats, which are worth saying out loud in an interview:
  * Survivorship bias: a ticker list of today's index members excludes firms
    that were delisted. Real research uses point-in-time constituents.
  * Shares outstanding come from a single snapshot, so `size` is approximate.
  * Yahoo has no point-in-time book value. Using today's book on old dates is
    look-ahead bias, so this loader writes NO fundamentals file and the
    pipeline simply drops the value factor. Plug in a point-in-time source
    (Compustat, SimFin, SEC XBRL filings) as data/fundamentals.csv with
    columns date,sym,bps, where `date` is when the number became public.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_yahoo(tickers: list[str], start: str, end: str | None, out_dir: str | Path = "data") -> None:
    import yfinance as yf  # optional dependency

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False)

    frames = []
    for t in tickers:
        try:
            df = raw[t][["Close", "Volume"]].dropna()
        except KeyError:
            continue
        info = yf.Ticker(t).info
        df = df.rename(columns={"Close": "close", "Volume": "volume"}).reset_index(names="date")
        df["sym"] = t
        df["shares"] = int(info.get("sharesOutstanding") or 0)
        df["sector"] = (info.get("sector") or "UNKNOWN").replace(" ", "_")
        frames.append(df[["date", "sym", "close", "volume", "shares", "sector"]])

    prices = pd.concat(frames, ignore_index=True)
    prices = prices[prices["shares"] > 0]
    prices.to_csv(out_dir / "prices.csv", index=False, date_format="%Y-%m-%d")
    if (out_dir / "fundamentals.csv").exists():
        print("note: an existing fundamentals.csv was left in place; make sure it matches these tickers")
    print(f"wrote {len(prices):,} rows for {prices['sym'].nunique()} tickers to {out_dir}/prices.csv")

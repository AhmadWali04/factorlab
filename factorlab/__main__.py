"""CLI.

  python -m factorlab synth                       # generate synthetic universe into data/
  python -m factorlab run --engine numpy          # or --engine kdb (needs licence)
  python -m factorlab run --engine kdb --neutral --lags 0   # lags 0 = auto rule
  python -m factorlab parity                      # check kdb and numpy agree
  python -m factorlab fetch --tickers tickers.txt # real prices from Yahoo (optional)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(prog="factorlab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth")
    s.add_argument("--stocks", type=int, default=500)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--data", default="data")

    for name in ("run", "parity"):
        r = sub.add_parser(name)
        r.add_argument("--data", default="data")
        r.add_argument("--out", default="out")
        r.add_argument("--lags", type=int, default=6, help="Newey-West lags; 0 = auto rule")
        r.add_argument("--neutral", action="store_true", help="sector-neutralize exposures")
        if name == "run":
            r.add_argument("--engine", choices=["numpy", "kdb"], default="numpy")

    f = sub.add_parser("fetch")
    f.add_argument("--tickers", required=True, help="file with one ticker per line")
    f.add_argument("--start", default="2012-01-01")
    f.add_argument("--end", default=None)
    f.add_argument("--data", default="data")

    a = ap.parse_args()

    if a.cmd == "synth":
        from .synth import generate
        t = generate(a.data, n_stocks=a.stocks, seed=a.seed)
        print(f"wrote synthetic universe to {a.data}/ (planted premia: {t['premia']})")
        return

    if a.cmd == "fetch":
        from .loaders import fetch_yahoo
        tickers = [x.strip() for x in Path(a.tickers).read_text().split() if x.strip()]
        fetch_yahoo(tickers, a.start, a.end, a.data)
        return

    from .engine import get_engine
    from .regression import auto_lags

    def run(engine_name: str, out: str):
        eng = get_engine(engine_name)
        t0 = time.perf_counter()
        lags = a.lags
        res = eng.run(a.data, out, lags if lags > 0 else 6, a.neutral)
        if lags == 0:  # recompute summary with the data-driven lag choice
            from .regression import summarize
            lags = auto_lags(len(res["gammas"]))
            res["summary"] = summarize(res["gammas"], list(res["summary"]["factor"]), lags)
        res["engine"] = engine_name
        return res, lags, time.perf_counter() - t0

    if a.cmd == "run":
        from .report import build
        res, lags, dt = run(a.engine, a.out)
        with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 120):
            print(res["summary"].to_string(index=False))
        path = build(res, a.out, lags, Path(a.data) / "truth.json")
        print(f"\n{a.engine} engine: {dt:.2f}s  |  report: {path}")
        return

    if a.cmd == "parity":
        from .parity import compare
        rn, _, tn = run("numpy", str(Path(a.out) / "numpy"))
        rk, _, tk = run("kdb", str(Path(a.out) / "kdb"))
        worst = compare(rn, rk)
        print(f"numpy {tn:.2f}s, kdb {tk:.2f}s, max abs difference {worst:.2e}")


if __name__ == "__main__":
    main()

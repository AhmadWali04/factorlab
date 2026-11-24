"""Two interchangeable engines that return the same results dict.

  NumpyEngine  pure pandas/numpy reference implementation (runs anywhere)
  KdbEngine    loads q/factorlab.q into an embedded q process via PyKX
               (needs a kdb+ licence: the free personal edition works)

Results: {"summary", "gammas", "quintiles", "ic"} as pandas DataFrames, plus
"panel" (standardized characteristics) from the numpy engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import xsec
from .regression import fama_macbeth, summarize

ALL_FACTORS = ["mom", "btp", "size", "vol", "rev"]
REPO = Path(__file__).resolve().parent.parent


class NumpyEngine:
    name = "numpy"

    def build_panel(self, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
        px = pd.read_csv(data_dir / "prices.csv", parse_dates=["date"])
        px = px.sort_values(["sym", "date"], kind="stable").reset_index(drop=True)
        g = px.groupby("sym", sort=False)
        px["lret"] = np.log(px["close"] / g["close"].shift())
        px["vol"] = (
            px.groupby("sym", sort=False)["lret"]
            .rolling(63, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
            * np.sqrt(252)
        )
        px["month"] = px["date"].dt.to_period("M")
        me = (
            px.groupby(["sym", "month"], sort=True)
            .agg(date=("date", "last"), close=("close", "last"), shares=("shares", "last"),
                 sector=("sector", "last"), vol=("vol", "last"))
            .reset_index()
        )
        c = me.groupby("sym", sort=False)["close"]
        me["rev"] = me["close"] / c.shift(1) - 1
        me["mom"] = c.shift(1) / c.shift(12) - 1
        me["fwd"] = c.shift(-1) / me["close"] - 1
        me["size"] = np.log(me["close"] * me["shares"])

        factors = [f for f in ALL_FACTORS if f != "btp"]
        fpath = data_dir / "fundamentals.csv"
        if fpath.exists():
            fd = pd.read_csv(fpath, parse_dates=["date"]).sort_values("date", kind="stable")
            me = pd.merge_asof(me.sort_values("date", kind="stable"), fd, on="date", by="sym")
            me = me.sort_values(["sym", "month"], kind="stable").reset_index(drop=True)
            me["btp"] = np.log(me["bps"] / me["close"])
            factors = ALL_FACTORS
        panel = me.dropna(subset=factors + ["fwd"]).reset_index(drop=True)
        return panel, factors

    @staticmethod
    def standardize(panel: pd.DataFrame, factors: list[str], neutral: bool) -> pd.DataFrame:
        p = panel.copy()
        for f in factors:
            p[f"z{f}"] = p.groupby("month")[f].transform(lambda s: xsec.prep(s.to_numpy()))
            if neutral:
                p[f"z{f}"] -= p.groupby(["month", "sector"])[f"z{f}"].transform("mean")
                p[f"z{f}"] = p.groupby("month")[f"z{f}"].transform(lambda s: xsec.zscore(s.to_numpy()))
        return p

    def run(self, data_dir="data", out_dir="out", lags=6, neutral=False) -> dict:
        data_dir, out_dir = Path(data_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel, factors = self.build_panel(data_dir)
        panel = self.standardize(panel, factors, neutral)
        zcols = [f"z{f}" for f in factors]

        gammas = fama_macbeth(panel, zcols)
        quint, ic = [], []
        for f in factors:
            z = f"z{f}"
            b = panel.groupby("month")[z].transform(lambda s: xsec.bucket(s.to_numpy()))
            q = (panel.assign(bucket=b).groupby(["month", "bucket"])["fwd"]
                 .agg(ret="mean", n="size").reset_index())
            quint.append(q.assign(factor=f))
            m = panel.groupby("month")[[z, "fwd"]].apply(
                lambda d: xsec.spearman(d[z].to_numpy(), d["fwd"].to_numpy()))
            ic.append(m.rename("ic").reset_index().assign(factor=f))

        res = {
            "summary": summarize(gammas, factors, lags),
            "gammas": gammas,
            "quintiles": pd.concat(quint, ignore_index=True),
            "ic": pd.concat(ic, ignore_index=True),
        }
        for k, v in res.items():
            v.to_csv(out_dir / f"{k}.csv", index=False)
        res["panel"] = panel
        return _normalize(res)


class KdbEngine:
    """Runs q/factorlab.q in-process through PyKX and pulls results into pandas."""

    name = "kdb"

    def __init__(self):
        import pykx as kx  # optional dependency

        if not kx.licensed:
            raise RuntimeError(
                "PyKX is running unlicensed. Get the free kdb+ personal licence "
                "(kx.com) and set QLIC / KDB_LICENSE_B64, then retry."
            )
        self.kx = kx
        kx.q(f'\\l {REPO / "q" / "factorlab.q"}')

    def run(self, data_dir="data", out_dir="out", lags=6, neutral=False) -> dict:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        r = self.kx.q(".fp.run", str(data_dir), str(out_dir), self.kx.LongAtom(lags),
                      self.kx.BooleanAtom(neutral))
        res = {k: r[k].pd() for k in ("summary", "gammas", "quintiles", "ic")}
        return _normalize(res)


def _normalize(res: dict) -> dict:
    """Make both engines' outputs directly comparable (types, sort order)."""
    for k in ("gammas", "quintiles", "ic"):
        df = res[k].copy()
        df["month"] = pd.PeriodIndex(pd.to_datetime(df["month"].astype(str)), freq="M")
        keys = ["factor", "month"] + (["bucket"] if k == "quintiles" else [])
        keys = [c for c in keys if c in df.columns]
        if "factor" in df.columns:
            df["factor"] = df["factor"].astype(str)
        res[k] = df.sort_values(keys).reset_index(drop=True)
    res["summary"]["factor"] = res["summary"]["factor"].astype(str)
    return res


def get_engine(name: str):
    return {"numpy": NumpyEngine, "kdb": KdbEngine}[name]()

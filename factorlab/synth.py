"""Synthetic equity universe with *known* factor premia.

Why synthetic: it lets you prove the pipeline is correct. We plant a monthly
premium on each standardized characteristic, then check that the Fama-MacBeth
regression recovers it. `rev` is planted at zero, so a correct pipeline should
find it insignificant.

Model, for stock i in month m (characteristics measured at the end of m-1):

    log r_im = beta_i * mkt_m + sector_shock_{s(i),m} + sum_k lambda_k z_ik + idio

Daily prices are built by spreading the monthly drift across trading days and
adding daily idiosyncratic noise, so volatility and momentum are computed
from the same daily series the engines read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .xsec import prep

# Planted monthly premia per 1 cross-sectional standard deviation of exposure.
TRUE_PREMIA = {
    "mom": 0.0040,    # +40 bp/month: momentum pays
    "btp": 0.0030,    # +30 bp: value, cheap (high book/price) beats expensive
    "size": -0.0020,  # -20 bp: small beats large
    "vol": -0.0025,   # -25 bp: low-volatility anomaly
    "rev": 0.0000,    # null factor: should come out insignificant
}
FACTORS = list(TRUE_PREMIA)
SECTORS = ["TECH", "FIN", "HLTH", "ENER", "INDU", "CONS", "STAP", "UTIL", "MATL", "COMM"]
REPORT_LAG_DAYS = 45  # book value becomes public ~45 days after quarter end


def generate(
    out_dir: str | Path = "data",
    n_stocks: int = 500,
    start: str = "2009-01-01",
    end: str = "2025-12-31",
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syms = np.array([f"S{i:04d}" for i in range(n_stocks)])
    sector = rng.choice(SECTORS, n_stocks)
    beta = rng.uniform(0.5, 1.5, n_stocks)
    sigma_d = rng.uniform(0.010, 0.030, n_stocks)  # daily idio vol
    shares = rng.integers(20, 2000, n_stocks) * 1_000_000
    price = rng.uniform(20, 200, n_stocks)
    bps = price * np.exp(rng.normal(-0.5, 0.5, n_stocks))

    days = pd.bdate_range(start, end)
    months = days.to_period("M")
    month_keys = months.unique()

    # Book-value reports: (available_date, bps vector). Seed one before start.
    reports = [(days[0] - pd.Timedelta(days=1), bps.copy())]

    me_price: list[np.ndarray] = []          # month-end close history
    daily_lr: list[np.ndarray] = []          # all daily log returns, (N,) each
    price_rows, fund_rows = [], []
    fund_rows.append(pd.DataFrame({"date": reports[0][0], "sym": syms, "bps": bps}))

    for m, key in enumerate(month_keys):
        mdays = days[months == key]
        nd = len(mdays)

        # --- expected return from characteristics at the end of month m-1 ---
        mu = np.zeros(n_stocks)
        if m >= 13:
            form_date = days[months == month_keys[m - 1]][-1]
            p1, p2, p13 = me_price[m - 1], me_price[m - 2], me_price[m - 13]
            avail = [b for (d, b) in reports if d <= form_date][-1]
            lr63 = np.vstack(daily_lr[-63:])
            chars = {
                "mom": p2 / p13 - 1,
                "rev": p1 / p2 - 1,
                "vol": lr63.std(axis=0, ddof=0) * np.sqrt(252),
                "btp": np.log(avail / p1),
                "size": np.log(p1 * shares),
            }
            for f, lam in TRUE_PREMIA.items():
                mu += lam * prep(chars[f])

        mkt = rng.normal(0.007, 0.045)
        sec_shock = {s: rng.normal(0, 0.015) for s in SECTORS}
        # Subtract the convexity term so the premia hold for *simple* returns, which
        # is what the engines measure. Without it, E[exp(x)] = exp(mu + sigma^2/2)
        # hands high-vol stocks an extra ~sigma^2/2 and masks the low-vol premium.
        drift = (beta * mkt + np.array([sec_shock[s] for s in sector]) + mu
                 - 0.5 * sigma_d**2 * nd)

        # --- daily path: drift spread evenly + daily idio noise ---
        eps = rng.standard_normal((nd, n_stocks)) * sigma_d
        lr = drift / nd + eps
        for d in range(nd):
            price = price * np.exp(lr[d])
            daily_lr.append(lr[d])
            price_rows.append((mdays[d], price.copy()))
        me_price.append(price.copy())

        # --- quarterly book update, published with a lag ---
        if key.month in (3, 6, 9, 12):
            bps = bps * np.exp(rng.normal(0.02, 0.05, n_stocks))
            avail_date = mdays[-1] + pd.Timedelta(days=REPORT_LAG_DAYS)
            reports.append((avail_date, bps.copy()))
            fund_rows.append(pd.DataFrame({"date": avail_date, "sym": syms, "bps": bps}))

    # --- write CSVs in the schema both engines read ---
    n_days = len(price_rows)
    prices = pd.DataFrame({
        "date": np.repeat([d for d, _ in price_rows], n_stocks),
        "sym": np.tile(syms, n_days),
        "close": np.concatenate([p for _, p in price_rows]),
        "volume": rng.integers(10_000, 5_000_000, n_days * n_stocks),
        "shares": np.tile(shares, n_days),
        "sector": np.tile(sector, n_days),
    })
    fund = pd.concat(fund_rows, ignore_index=True)
    fund = fund[fund["date"] <= days[-1]]

    prices.to_csv(out_dir / "prices.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    fund.to_csv(out_dir / "fundamentals.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    truth = {"premia": TRUE_PREMIA, "n_stocks": n_stocks, "start": start, "end": end, "seed": seed}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth

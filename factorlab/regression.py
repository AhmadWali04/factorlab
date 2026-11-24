"""Fama-MacBeth cross-sectional regressions with Newey-West inference.

Step 1 (each month t): regress next-month returns on today's standardized
factor exposures across all stocks:

    r_{i,t+1} = g0_t + g1_t * z_mom_it + g2_t * z_btp_it + ... + e_it

Each slope g_k,t is the return that month of a long-short portfolio with unit
exposure to factor k and zero exposure to the others.

Step 2: the premium estimate is the time-series mean of g_k,t. Its standard
error uses Newey-West (Bartlett kernel) because factor returns are
autocorrelated and overlapping signals (12-month momentum) make that worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def xsreg(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS for one cross-section. X is n x k including the intercept column."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, float(r2)


def fama_macbeth(panel: pd.DataFrame, zcols: list[str], y: str = "fwd") -> pd.DataFrame:
    """Return one row per month: const, one gamma per factor, cross-sectional R^2."""
    rows = []
    for month, g in panel.groupby("month", sort=True):
        X = np.column_stack([np.ones(len(g)), g[zcols].to_numpy()])
        b, r2 = xsreg(g[y].to_numpy(), X)
        rows.append([month, *b, r2])
    return pd.DataFrame(rows, columns=["month", "const", *zcols, "r2"])


def newey_west(x: np.ndarray, lags: int) -> dict:
    """Mean of x with Bartlett-kernel HAC standard error. Matches .fp.nw."""
    x = np.asarray(x, dtype=float)
    n, m = len(x), x.mean()
    e = x - m
    gam = [(e[l:] @ e[: n - l]) / n for l in range(lags + 1)]
    w = [1.0] + [2 * (1 - l / (lags + 1)) for l in range(1, lags + 1)]
    se = np.sqrt(np.dot(w, gam) / n)
    return {"mean": m, "se": se, "t": m / se, "pos": float((x > 0).mean())}


def auto_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def summarize(gammas: pd.DataFrame, factors: list[str], lags: int) -> pd.DataFrame:
    out = pd.DataFrame(
        [{"factor": f, **newey_west(gammas[f"z{f}"].to_numpy(), lags)} for f in factors]
    )
    return out


def vif(panel: pd.DataFrame, zcols: list[str]) -> pd.Series:
    """Variance inflation factors from the pooled exposure correlation matrix."""
    c = panel[zcols].corr().to_numpy()
    return pd.Series(np.diag(np.linalg.inv(c)), index=zcols, name="vif")

"""Cross-sectional transforms.

Every function here has an exact twin in q/factorlab.q (same definitions, same
tie-breaking, population standard deviation), so the numpy and kdb+ engines
produce identical numbers. If you change one, change the other.
"""
from __future__ import annotations

import numpy as np


def winsorize(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    """Symmetric rank-based clip: k = floor(p*(n-1)) values replaced in each tail.
    Matches .fp.winsor."""
    s = np.sort(x)
    k = int(np.floor(p * (len(s) - 1)))
    return np.clip(x, s[k], s[len(s) - 1 - k])


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=0)


def prep(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    """Winsorize at p / 1-p, then standardize. Matches .fp.prep."""
    return zscore(winsorize(np.asarray(x, dtype=float), p))


def rank(x: np.ndarray) -> np.ndarray:
    """0-based ordinal rank, ties broken by position (q's `rank`)."""
    return np.argsort(np.argsort(x, kind="stable"), kind="stable")


def bucket(x: np.ndarray, n: int = 5) -> np.ndarray:
    """Quantile bucket 1..n. Matches .fp.bucket."""
    return 1 + np.floor(n * rank(x) / len(x)).astype(int)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rank(x), rank(y))[0, 1])

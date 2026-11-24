"""Number-for-number comparison of the numpy and kdb+ engines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare(a: dict, b: dict, tol: float = 1e-8) -> float:
    worst = 0.0
    for k in ("summary", "gammas", "quintiles", "ic"):
        x, y = a[k], b[k]
        if len(x) != len(y):
            raise AssertionError(f"{k}: row count differs ({len(x)} vs {len(y)})")
        for c in x.select_dtypes("number").columns:
            if c not in y.columns:
                raise AssertionError(f"{k}: column {c} missing from kdb output")
            d = float(np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float))))
            worst = max(worst, d)
            if d > tol:
                raise AssertionError(f"{k}.{c}: max abs diff {d:.3e} > {tol}")
    return worst

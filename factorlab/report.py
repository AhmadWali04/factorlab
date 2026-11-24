"""Charts + a self-contained HTML research note from an engine's results."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .regression import vif

LABELS = {"mom": "Momentum (12-1)", "btp": "Value (book/price)", "size": "Size (log mcap)",
          "vol": "Volatility (63d)", "rev": "Reversal (1m)"}
COLORS = ["#2a6fdb", "#e07a1f", "#2e9e6a", "#c23b4e", "#7a5cc7"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def build(res: dict, out_dir="out", lags=6, truth_path: str | Path | None = None) -> Path:
    out_dir = Path(out_dir)
    s, gam, q, ic = res["summary"], res["gammas"], res["quintiles"], res["ic"]
    factors = list(s["factor"])
    truth = None
    if truth_path and Path(truth_path).exists():
        truth = json.loads(Path(truth_path).read_text())["premia"]

    # 1. cumulative Fama-MacBeth factor returns
    fig, ax = plt.subplots(figsize=(9, 4))
    t = gam["month"].dt.to_timestamp()
    for f, c in zip(factors, COLORS):
        ax.plot(t, gam[f"z{f}"].cumsum() * 100, label=LABELS.get(f, f), color=c, lw=1.6)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("cumulative return, %")
    ax.set_title("Fama-MacBeth factor returns (cumulative sum of monthly gammas)", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _style(ax)
    img_cum = _png(fig)

    # 2. quintile average returns
    avg = q.groupby(["factor", "bucket"])["ret"].mean().unstack() * 100
    fig, axes = plt.subplots(1, len(factors), figsize=(2.2 * len(factors), 2.8), sharey=True)
    for ax, f, c in zip(np.atleast_1d(axes), factors, COLORS):
        ax.bar(avg.columns, avg.loc[f], color=c)
        ax.set_title(LABELS.get(f, f), fontsize=9)
        ax.set_xticks(avg.columns)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("avg next-month return, %")
    fig.suptitle("Quintile portfolios (1 = low exposure, 5 = high)", x=0.01, y=1.06, ha="left", fontsize=10)
    img_q = _png(fig)

    # 3. rolling 12-month IC
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for f, c in zip(factors, COLORS):
        d = ic[ic["factor"] == f]
        ax.plot(d["month"].dt.to_timestamp(), d["ic"].rolling(12).mean(), label=LABELS.get(f, f), color=c, lw=1.4)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Rank IC, 12-month rolling mean", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _style(ax)
    img_ic = _png(fig)

    # tables
    icsum = ic.groupby("factor")["ic"].agg(["mean", "std"])
    spread = (q[q.bucket == 5].set_index(["factor", "month"])["ret"]
              - q[q.bucket == 1].set_index(["factor", "month"])["ret"]).groupby("factor").mean()
    rows = []
    for _, r in s.iterrows():
        f = r["factor"]
        cells = [LABELS.get(f, f), f"{r['mean']*100:+.3f}", f"{r['mean']*1200:+.2f}",
                 f"{r['t']:+.2f}", f"{r['pos']*100:.0f}%", f"{icsum.loc[f, 'mean']:+.4f}",
                 f"{icsum.loc[f, 'mean']/icsum.loc[f, 'std']*np.sqrt(12):+.2f}", f"{spread[f]*100:+.3f}"]
        if truth:
            cells.insert(1, f"{truth[f]*100:+.3f}")
        sig = "sig" if abs(r["t"]) > 2 else ""
        rows.append(f"<tr class='{sig}'>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head = ["Factor", "Premium %/mo", "Annualized %", f"NW t (L={lags})", "% months &gt; 0",
            "Mean IC", "ICIR (ann.)", "Q5&minus;Q1 %/mo"]
    if truth:
        head.insert(1, "Planted %/mo")

    v = vif(res["panel"], [f"z{f}" for f in factors]) if "panel" in res else None
    vif_html = ""
    if v is not None:
        vif_html = "<p class='note'>Variance inflation factors: " + ", ".join(
            f"{LABELS.get(k[1:], k)} {x:.2f}" for k, x in v.items()) + \
            ". Values near 1 mean the exposures are not collinear, so each gamma is cleanly identified.</p>"

    T = len(gam)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>factorlab report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 20px;color:#1d2330}}
h1{{font-size:24px;margin-bottom:4px}} .sub{{color:#667;margin-top:0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th,td{{padding:7px 9px;border-bottom:1px solid #e4e6eb;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f5f6f8;font-weight:600}}
tr.sig td{{font-weight:600}} img{{width:100%;margin:10px 0}} .note{{color:#556;font-size:13px}}
</style></head><body>
<h1>Equity factor research: Fama-MacBeth report</h1>
<p class="sub">{T} monthly cross-sections, mean R&sup2; {gam['r2'].mean():.3f}, engine: {res.get('engine','numpy')}</p>
<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(rows)}</table>
<p class="note">Premium is the time-series mean of the monthly cross-sectional slope on each standardized
(winsorized 1/99, z-scored) exposure, so it is the monthly return to one standard deviation of that
characteristic, holding the others fixed. Bold rows have |t| &gt; 2 using Newey-West standard errors.</p>
{vif_html}
<img src="data:image/png;base64,{img_cum}"><img src="data:image/png;base64,{img_q}"><img src="data:image/png;base64,{img_ic}">
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(html)
    return path

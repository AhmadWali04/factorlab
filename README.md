# factorlab

Equity factor research in **kdb+/q**, with a **Python** layer for orchestration, testing and reporting.

The pipeline goes from daily prices and point-in-time fundamentals to a monthly panel of stock characteristics. It then runs quintile portfolio sorts, rank ICs, and a **Fama-MacBeth cross-sectional regression with Newey-West standard errors** to estimate each factor's premium.

```
prices.csv ─┐                 ┌─ quintile sorts ── Q5−Q1 spreads
            ├─ month-end panel ┼─ rank IC (Spearman) ── ICIR
fund.csv ───┘   (aj as-of join) └─ Fama-MacBeth ── gammas ── Newey-West t-stats
                 winsorize, z-score,
                 sector-neutralize (opt.)
```

## Factors

| Factor | Definition (at month-end *t*) | Planted (synthetic) |
|---|---|---|
| `mom`  | 12-1 momentum: close(t−1) / close(t−12) − 1 | +40 bp/mo |
| `btp`  | value: log(book per share / price), book as publicly known on that date | +30 bp/mo |
| `size` | log market cap | −20 bp/mo |
| `vol`  | 63-day realized volatility of daily log returns, annualized | −25 bp/mo |
| `rev`  | 1-month return (null factor) | 0 |

The target is the next month's simple return. Each month, exposures are winsorized at the 1st and 99th percentiles and z-scored.

## The regression

For each month *t*, regress next-month returns on the standardized exposures across all stocks:

    r_{i,t+1} = γ0_t + γ_mom,t · z_mom,it + γ_btp,t · z_btp,it + … + ε_it

Each slope γ_k,t is that month's return to a portfolio with one standard deviation of exposure to factor *k* and none to the others. The premium estimate is the time-series mean of γ_k. Its t-statistic uses **Newey-West (Bartlett) standard errors**, because factor returns are autocorrelated, so naive t-stats overstate significance. The report also gives variance inflation factors to check for multicollinearity between exposures.

## Why synthetic data first

`factorlab synth` builds a 500-stock universe with known premia, planted through a factor model. The test suite asserts that the regression **recovers every planted premium within 3 standard errors** and that the null factor comes out insignificant. That proves the pipeline is correct before any real data is involved. Current output:

```
factor   planted   estimate   NW t
mom      +0.400%   +0.454%   +10.5
btp      +0.300%   +0.319%    +6.7
size     −0.200%   −0.197%    −4.1
vol      −0.250%   −0.276%    −7.3
rev       0.000%   +0.005%    +0.2
```

The generator plants premia in simple-return space. Planting them in log space instead would give high-volatility stocks an extra +½σ² from convexity (E[eˣ] = e^(μ+σ²/2)), which hides the low-volatility premium. The first version of this project made exactly that mistake, and the regression caught it.

## Two engines, one answer

| Engine | What it is |
|---|---|
| `kdb` | `q/factorlab.q`, run in-process through PyKX (or standalone with `q q/run.q`) |
| `numpy` | a pandas/numpy reference implementation that mirrors every q function |

`python -m factorlab parity` runs both engines and asserts they agree to within 1e-8. The numpy engine runs anywhere, and the kdb engine is the production path.

## Quickstart

```bash
pip install -e ".[dev]"
python -m factorlab synth                     # ~115 MB of synthetic data into data/
python -m factorlab run                       # numpy engine → out/*.csv + out/report.html
python -m pytest                              # recovery, statsmodels cross-checks, parity

# kdb+: get the free personal licence from kx.com, then
pip install pykx
python -m factorlab run --engine kdb
python -m factorlab parity
q q/run.q -data data -out out -lags 6 -neutral 0   # pure q, no Python

# options
python -m factorlab run --neutral             # sector-neutral exposures
python -m factorlab run --lags 0              # Newey-West lag by rule: floor(4(T/100)^(2/9))
```

## Real data

`python -m factorlab fetch --tickers tickers.txt` pulls prices from Yahoo Finance and writes the same schema (`pip install yfinance`). Three caveats apply, documented in `factorlab/loaders.py`:

- **Survivorship bias:** a list of today's tickers leaves out firms that were delisted.
- **Approximate size:** shares outstanding come from a single snapshot.
- **No value factor:** Yahoo has no point-in-time book value, so the pipeline drops `btp` rather than introduce look-ahead bias. To get it back, supply `data/fundamentals.csv` (`date,sym,bps`), where `date` is when the figure became public. Source it from SEC XBRL, SimFin or Compustat.

## Layout

```
q/factorlab.q          q library: panel, aj, winsorize, quintiles, IC, Fama-MacBeth, Newey-West
q/run.q                command-line entry for pure-q runs
factorlab/xsec.py      cross-sectional helpers (twins of the q ones)
factorlab/engine.py    NumpyEngine + KdbEngine (PyKX)
factorlab/regression.py  Fama-MacBeth, Newey-West, VIF
factorlab/synth.py     synthetic universe with planted premia
factorlab/report.py    charts + HTML note
tests/                 recovery, statsmodels cross-checks, kdb/numpy parity
```

## Ideas to extend

- Add industry dummies to the cross-sectional regression instead of demeaning within sectors.
- Use weighted least squares with √(market cap) weights, the Barra convention.
- Estimate a factor covariance matrix from the gammas and build a risk model.
- Feed the factor forecasts into a constrained mean-variance optimizer and backtest it net of costs.
- Store data in a date-partitioned kdb+ database (`.Q.dpft`) instead of loading CSVs.

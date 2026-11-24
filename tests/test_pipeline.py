import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from factorlab import xsec
from factorlab.engine import NumpyEngine
from factorlab.regression import fama_macbeth, newey_west
from factorlab.synth import TRUE_PREMIA, generate


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d, n_stocks=300, start="2010-01-01", end="2023-12-31", seed=11)
    return NumpyEngine().run(d, tmp_path_factory.mktemp("out"), lags=6)


def test_recovers_planted_premia(results):
    s = results["summary"].set_index("factor")
    for f, lam in TRUE_PREMIA.items():
        err = abs(s.loc[f, "mean"] - lam) / s.loc[f, "se"]
        assert err < 3, f"{f}: estimate {s.loc[f, 'mean']:.5f} vs planted {lam} ({err:.1f} se off)"


def test_null_factor_insignificant(results):
    assert abs(results["summary"].set_index("factor").loc["rev", "t"]) < 2.5


def test_no_lookahead_in_panel(results):
    p = results["panel"]
    # fundamentals must have been public on or before the formation date
    assert (p["month"].dt.to_timestamp(how="end") >= p["date"]).all()
    # first usable month needs 12 months of history for momentum
    assert p["month"].min() >= pd.Period("2011-01", "M")


def test_gammas_match_statsmodels(results):
    p = results["panel"]
    zc = [c for c in p.columns if c.startswith("z")]
    g = fama_macbeth(p, zc).set_index("month")
    for m in list(g.index)[:: max(1, len(g) // 10)]:
        d = p[p["month"] == m]
        ols = sm.OLS(d["fwd"], sm.add_constant(d[zc])).fit()
        np.testing.assert_allclose(g.loc[m, ["const", *zc]].to_numpy(float), ols.params.to_numpy(), atol=1e-12)
        assert g.loc[m, "r2"] == pytest.approx(ols.rsquared, abs=1e-12)


@pytest.mark.parametrize("lags", [0, 3, 6, 12])
def test_newey_west_matches_statsmodels(lags):
    x = np.random.default_rng(0).standard_normal(240).cumsum() * 0.001 + 0.002
    ours = newey_west(x, lags)
    ref = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    assert ours["se"] == pytest.approx(float(ref.bse[0]), rel=1e-10)


def test_xsec_helpers():
    x = np.array([5.0, 1.0, 100.0, 3.0, 2.0, 4.0, -50.0, 6.0, 7.0, 8.0])
    w = xsec.winsorize(x, 0.2)  # k = floor(0.2*9) = 1 per tail
    assert w.max() == 8.0 and w.min() == 1.0
    z = xsec.prep(x, 0.2)
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    b = xsec.bucket(np.arange(10.0))
    assert list(b) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_quintile_spread_sign(results):
    q = results["quintiles"]
    avg = q.groupby(["factor", "bucket"])["ret"].mean().unstack()
    assert avg.loc["mom", 5] > avg.loc["mom", 1]
    assert avg.loc["vol", 5] < avg.loc["vol", 1]

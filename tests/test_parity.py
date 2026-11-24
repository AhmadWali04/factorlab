"""kdb+ vs numpy parity. Skipped unless PyKX has a kdb+ licence."""
import pytest

pykx = pytest.importorskip("pykx")
if not pykx.licensed:
    pytest.skip("PyKX unlicensed: install the free kdb+ personal licence to run", allow_module_level=True)

from factorlab.engine import KdbEngine, NumpyEngine
from factorlab.parity import compare
from factorlab.synth import generate


@pytest.mark.parametrize("neutral", [False, True])
def test_engines_agree(tmp_path, neutral):
    generate(tmp_path / "d", n_stocks=150, start="2012-01-01", end="2020-12-31", seed=3)
    a = NumpyEngine().run(tmp_path / "d", tmp_path / "n", 6, neutral)
    b = KdbEngine().run(tmp_path / "d", tmp_path / "k", 6, neutral)
    assert compare(a, b) < 1e-8

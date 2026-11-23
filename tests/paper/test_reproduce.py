"""End-to-end: regenerate the paper from the results archive and compare with paper/expected/ (about 20 seconds; skipped without the data)."""
import os

import pytest

from paper import check, dataio, make_paper

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not os.path.isdir(dataio.R("closed_set")):
        pytest.skip("results archive not unpacked (python scripts/fetch_data.py, or set FLIPS_DATA)")
    out = tmp_path_factory.mktemp("paper_out")
    old = os.environ.get("FLIPS_OUT")
    os.environ["FLIPS_OUT"] = str(out)
    try:
        res = make_paper.run(None, verbose=False)
    finally:
        if old is None:
            os.environ.pop("FLIPS_OUT")
        else:
            os.environ["FLIPS_OUT"] = old
    return str(out), res


def test_every_step_runs(built):
    _, res = built
    assert [r[0] for r in res] == [m for m, _ in make_paper.STEPS]
    assert all(err is None for _, _, err in res)


def test_output_equals_expected(built):
    out, _ = built
    fails, n_text, n_mac, n_json = check.compare(out, check.EXPECTED)
    assert fails == [] and n_text >= 2 and n_mac >= 60 and n_json >= 10

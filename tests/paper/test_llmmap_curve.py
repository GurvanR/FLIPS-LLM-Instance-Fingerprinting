"""The LLMmap-IF baseline curve: the above-mean-split filter is explicit and consistent."""
import json

import pytest

from paper.stats import llmmap_curve


def _entry(kept, all_mean, n=10):
    return {"n_splits": n, "accuracy_kept_splits": kept, "accuracy_kept_mean": sum(kept) / len(kept), "accuracy_all_splits_mean": all_mean}


@pytest.fixture
def data(tmp_path, monkeypatch):
    (tmp_path / "results" / "llmmap").mkdir(parents=True)
    ds = {"1": _entry([0.2, 0.21], 0.12), "2": _entry([0.26, 0.27, 0.28], 0.2)}
    (tmp_path / "results" / "llmmap" / "results.json").write_text(json.dumps({"per_nt": ds}))
    monkeypatch.setenv("FLIPS_DATA", str(tmp_path))
    monkeypatch.setenv("FLIPS_OUT", str(tmp_path / "out"))


def test_table_and_curve(data):
    rows = llmmap_curve.table()
    assert [r[0] for r in rows] == [1, 2]
    assert rows[0][1] == "2/10" and rows[0][2] == 0.205 and rows[0][3] == 0.12      # kept 2 of 10, filtered mean vs all-split mean
    assert llmmap_curve.curve() == {1: 0.205, 2: 0.27}
    assert llmmap_curve.text().splitlines()[0] == llmmap_curve.HEADER


def test_inconsistent_kept_mean_is_rejected(data, tmp_path):
    p = tmp_path / "results" / "llmmap" / "results.json"
    d = json.loads(p.read_text())
    d["per_nt"]["1"]["accuracy_kept_mean"] = 0.9
    p.write_text(json.dumps(d))
    with pytest.raises(AssertionError):
        llmmap_curve.table()


def test_main_writes_the_table(data, tmp_path):
    llmmap_curve.main([])
    assert (tmp_path / "out" / "figures" / "llmmap_splits.txt").read_text().startswith("N_t kept/n")

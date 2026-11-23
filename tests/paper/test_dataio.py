"""Fast tests of paper/dataio.py helpers on tiny inputs (no archive needed)."""
import json
import os

import numpy as np
import pytest

from paper import dataio

NAMES = [
    "meta-llama/Llama-3.1-8B-Instruct_temp-1.0_sp--1",
    "meta-llama/Llama-3.1-8B-Instruct_temp-0.4_sp--1",
    "Qwen/Qwen2-7B-Instruct@@fp8_temp-1.0_sp-3",
    "natong19/Qwen2-7B-Instruct-abliterated_temp-1.0_sp--1",
]


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    (tmp_path / "results" / "meta").mkdir(parents=True)
    (tmp_path / "results" / "meta" / "class_order.json").write_text(json.dumps({"class_names": NAMES}))
    monkeypatch.setenv("FLIPS_DATA", str(tmp_path))
    monkeypatch.setenv("FLIPS_OUT", str(tmp_path / "out"))
    return tmp_path


def test_roots_follow_the_environment(data_root):
    assert dataio.data_root() == str(data_root)
    assert dataio.R("meta", "x.json") == os.path.join(str(data_root), "results", "meta", "x.json")
    assert dataio.fig_dir() == os.path.join(str(data_root), "out", "figures") and os.path.isdir(dataio.fig_dir())
    assert dataio.closed_dir().endswith(os.path.join("out", "closed"))


def test_default_roots_without_environment(monkeypatch):
    monkeypatch.delenv("FLIPS_DATA", raising=False)
    monkeypatch.delenv("FLIPS_OUT", raising=False)
    assert dataio.data_root() == os.path.join(dataio.REPO, "data")
    assert dataio.out_root() == os.path.join(dataio.PKG, "out")


def test_class_name_parsing():
    assert dataio.attrs(NAMES[2]) == ("Qwen/Qwen2-7B-Instruct", "fp8", "1.0", "3")
    assert dataio.attrs(NAMES[0]) == ("meta-llama/Llama-3.1-8B-Instruct", "-", "1.0", "-1")
    with pytest.raises(ValueError):
        dataio.attrs("not-a-class")
    assert dataio.orig_name(NAMES[3]) == "Qwen/Qwen2-7B-Instruct"        # abliterated -> its original model
    assert dataio.orig_name(NAMES[2]) == "Qwen/Qwen2-7B-Instruct"        # quantisation stripped
    assert dataio.var_name(NAMES[3]) == "ablit" and dataio.var_name(NAMES[2]) == "fp8_temp-1.0_sp-3"
    assert dataio.fmt_var("temp-0.4_sp--1") == "temp 0.4\n(no sp)"
    assert dataio.fmt_var("fp8_temp-1.0_sp-3") == "fp8\nsp 2\n(temp 1.0)"      # class digit 3 is system prompt 2 of the paper


def test_attr_matrix_sums_over_classes_sharing_a_value():
    names = NAMES[:3]
    M = np.arange(9).reshape(3, 3)
    vals, A, G = dataio.attr_matrix(M, names, "T")
    assert vals == ["0.4", "1.0"]
    assert A.shape == (2, 2) and A.sum() == M.sum()
    assert A[1, 1] == M[0, 0] + M[0, 2] + M[2, 0] + M[2, 2]


def test_ctx_cells_and_records(data_root):
    tree = data_root / "tree"
    (tree / "cells").mkdir(parents=True)
    cell = {"cells": [{"nt": 32, "nr": 300, "tp": "0-1", "split": 0, "acc": 0.5, "class_correct": [1, 2], "class_batches": [2, 2]},
                      {"nt": 8, "nr": 300, "tp": "0-1", "split": 0, "acc": 0.1}]}
    (tree / "cells" / "0-1_s0.json").write_text(json.dumps(cell))
    ctx = dataio.Ctx(out=str(data_root / "o"), tree=str(tree))
    assert ctx.C == len(NAMES)
    runs = ctx.cells()
    assert len(dataio.records(runs)) == 2 and len(dataio.records(runs, nt=32)) == 1
    assert dataio.acc_by_cell(runs) == {("0-1", 0): 0.5}
    assert dataio.pair_means({("0-1", 0): 0.5, ("0-1", 1): 0.7, ("a", 0): 1.0}) == {"0-1": pytest.approx(0.6), "a": 1.0}
    k, n = dataio.class_counts(runs, C=2)
    assert list(k) == [1, 2] and list(n) == [2, 2]


def test_heatmap_layout_stats():
    S = dataio.nxm_stats(NAMES, [0.1, 0.2, 0.3, 0.4])
    assert S["grand"] == pytest.approx(0.25)
    assert S["vars"][-1].startswith("fp8")            # quantised variations go last
    assert "ablit" in S["vars"]


def test_post_confusion_checks_order_and_n_t(data_root):
    ctx = dataio.Ctx(out=str(data_root / "o"), tree=str(data_root / "tree"))
    cl = {"class_names": NAMES, "confusion": {"32": np.eye(4, dtype=int).tolist()}}
    assert dataio.post_confusion(ctx, cl, 32).trace() == 4
    with pytest.raises(dataio.Skip):
        dataio.post_confusion(ctx, cl, 8)
    with pytest.raises(AssertionError):
        dataio.post_confusion(ctx, dict(cl, class_names=NAMES[::-1]), 32)

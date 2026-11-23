# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Open-set method on the fixture against stored reference outputs: the Unseen detector and the open-set decision of
one fixture cell reproduce ``fixtures/reference/open_set.npz`` (scaled-down constants listed in ``open_set.json``).

Float outputs must agree within 1e-9 (they agree bitwise on the reference machine with 4 XGBoost threads), integer
outputs exactly. About 35 s on 4 threads (grid build included)."""

import json

import numpy as np
import pytest

from audit_llm.Classification import unseen_detector as UD
from audit_llm.Classification.protocol import make_splits, select_rows

from .conftest import FIXTURES

TOL = 1e-9


@pytest.fixture(scope="module")
def open_reference():
    meta = json.loads((FIXTURES / "reference" / "open_set.json").read_text())
    return meta, np.load(FIXTURES / "reference" / "open_set.npz")


@pytest.fixture(scope="module")
def open_run(fixture_grid, open_reference):
    meta, _ = open_reference
    g = fixture_grid
    X, y, _ = select_rows(g.pair(meta["token_pair"]).features, g.feature_names)
    splits = make_splits(y, meta["n_splits"], meta["test_per_class"])
    return UD.evaluate_open_set_cell(X, y, splits, g.class_names, meta["token_pair"], meta["split"], meta["draw"],
                                     batch_sizes=meta["batch_sizes"], **meta["params"])


def close(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    f = np.isfinite(a)
    assert np.array_equal(a[~f], b[~f])
    return float(np.abs(a[f] - b[f]).max()) if f.any() else 0.0


def test_cell_and_classifiers(open_run, open_reference):
    _, ref = open_reference
    fitted, _ = open_run
    assert np.array_equal(fitted.unseen, ref["unseen"])
    assert np.array_equal(fitted.cell.train_rows, ref["train_rows"]) and np.array_equal(fitted.cell.test_rows, ref["test_rows"])
    assert close(fitted.test_log_probs, ref["test_log_probs"]) <= TOL
    assert close(fitted.test_per_llm_log_probs[::4], ref["test_per_llm_log_probs_every4"]) <= TOL


def test_detector_folds(open_run, open_reference):
    _, ref = open_reference
    det = open_run[0].detector
    assert np.array_equal(det.class_fold, ref["class_fold"])
    fold0 = det.folds[0]
    assert fold0.index == 0
    for name in ("train", "known", "unseen"):
        assert np.array_equal(getattr(fold0, f"{name}_rows"), ref[f"fold0_{name}_rows"])
    assert close(fold0.log_probs, ref["fold0_log_probs"]) <= TOL
    assert close(fold0.per_llm_log_probs[::4], ref["fold0_per_llm_log_probs_every4"]) <= TOL


def test_detector_and_decisions(open_run, open_reference):
    meta, ref = open_reference
    _, results = open_run
    for n_t in meta["batch_sizes"]:
        r, k = results[n_t], (lambda q: ref[f"n{n_t}_{q}"])
        S = r["samples"]
        assert np.array_equal(S.is_unseen, k("sample_is_unseen").astype(bool))
        for name in ("source", "fold", "pred"):
            assert np.array_equal(getattr(S, name), k(f"sample_{name}"))
        assert np.array_equal(S.correct, k("sample_correct").astype(bool))
        assert close(S.statistics[::10], k("sample_statistics_every10")) <= TOL
        assert close(r["oof_scores"], k("oof_scores")) <= TOL
        assert abs(r["threshold"] - float(k("threshold"))) <= TOL
        grid = [UD.threshold(r["oof_scores"][S.is_unseen], t) for t in UD.TARGET_RECALL_GRID]
        assert close(grid, k("threshold_grid")) <= TOL
        assert close(r["scores"], k("test_scores")) <= TOL
        assert np.array_equal(r["hierarchical_pred"], k("test_pred"))
        assert np.array_equal(r["flags"], k("flags").astype(bool))
        assert np.array_equal(r["pred"], k("final_pred"))
        if f"n{n_t}_test_statistics" in ref.files:
            assert close(r["statistics"], k("test_statistics")) <= TOL
        assert abs(r["accuracy"] - meta["metrics"][str(n_t)]["accuracy"]) <= 1e-12
        assert abs(r["unseen_recall"] - meta["metrics"][str(n_t)]["unseen_recall"]) <= 1e-12

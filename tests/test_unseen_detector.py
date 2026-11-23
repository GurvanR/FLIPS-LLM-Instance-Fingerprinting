# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Fast unit tests of the Unseen detector's building blocks on synthetic arrays."""

import numpy as np
import pytest

from audit_llm.Classification import unseen_detector as UD


def _names(per_llm):
    return [f"org/llm{l}_temp-{0.1 * (i + 1):.1f}_sp--1" for l, n in enumerate(per_llm) for i in range(n)]


def _log_probs(n, K, rng):
    m = rng.normal(size=(n, K)) * 2
    return m - np.log(np.exp(m).sum(1, keepdims=True))


def test_statistics_shape_and_finite():
    rng = np.random.default_rng(0)
    llm = np.repeat([0, 1, 2], [5, 4, 3])
    L = _log_probs(200, 12, rng)
    batches = rng.integers(0, 200, size=(30, 4))
    Z = UD.sample_statistics(L, batches, llm)
    assert Z.shape == (30, 38) and len(UD.STATISTIC_NAMES) == 38 and len(set(UD.STATISTIC_NAMES)) == 38
    assert np.isfinite(Z).all()
    # chunking does not change the result; the decision is the argmax of the summed log-probabilities
    assert np.array_equal(Z, UD.sample_statistics(L, batches, llm, chunk=7))
    pred = UD.summed_decision(L, batches)
    assert np.array_equal(pred, L[batches].sum(1).argmax(1))
    # gap_2 = best minus second-best summed score
    S = np.sort(L[batches].sum(1), axis=1)
    assert np.allclose(Z[:, UD.STATISTIC_NAMES.index("gap_2")], S[:, -1] - S[:, -2])


def test_fold_assignment_spreads_llms_and_keeps_siblings():
    names = _names([8, 7, 4, 1])
    known = np.arange(len(names))
    fold = UD.assign_class_folds(names, known, split=0, draw=0)
    llm = np.repeat([0, 1, 2, 3], [8, 7, 4, 1])
    assert fold[llm == 3][0] == -1  # only Known instance of its LLM: never held out
    for l in range(3):
        f = fold[llm == l]
        assert len(np.unique(f)) == len(f) and (f >= 0).all()  # one LLM's instances in different folds
    for k in np.flatnonzero(fold >= 0):
        assert np.any((llm == llm[k]) & (fold != fold[k]))  # a sibling stays in training
    assert np.array_equal(fold, UD.assign_class_folds(names, known, split=0, draw=0))


def test_weights_mean_one_and_side_shares():
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.3
    src = rng.integers(0, 12, size=500)
    w = UD.sample_weights(y, src, 0.097)
    assert np.isclose(w.mean(), 1.0)
    assert np.isclose(w[y].sum() / w.sum(), 0.097)
    for side in (True, False):  # equal total weight per source instance within a side
        tot = np.bincount(src[y == side], weights=w[y == side])
        assert np.allclose(tot[tot > 0], tot[tot > 0][0])


def test_threshold_is_quantile():
    q = np.random.default_rng(2).normal(size=101)
    assert UD.threshold(q, 0.5) == pytest.approx(np.median(q))
    assert UD.threshold(q, 0.2) == pytest.approx(np.quantile(q, 0.8))
    assert np.mean(q >= UD.threshold(q, 0.5)) == pytest.approx(51 / 101)


def test_logistic_detector_separates():
    rng = np.random.default_rng(3)
    Z = np.r_[rng.normal(size=(100, 3)), rng.normal(size=(100, 3)) + 2]
    y = np.r_[np.zeros(100, bool), np.ones(100, bool)]
    det = UD.LogisticDetector().fit(Z, y, np.ones(200))
    s = det.score(Z)
    assert s[y].mean() > s[~y].mean()


def test_decision_rule_and_test_samples():
    y_test = np.repeat([0, 3, 1, 2], 5)  # instance 3 Unseen
    rows, labels, unseen = UD.verification_samples(y_test, np.array([0, 1, 2]), 2)
    assert list(labels) == [0, 0, 1, 1, 2, 2, 3, 3] and list(unseen) == [False] * 6 + [True] * 2  # Known first, then Unseen
    scores = np.array([0.1, -1, 2, -3, 0.5, 0, 1, -2])
    flags = scores >= 0.5
    pred = np.where(flags, UD.UNSEEN, labels)
    assert list(pred) == [0, 0, -1, 1, -1, 2, -1, 3]

# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Closed-set method on the fixture against stored reference outputs.

``fixtures/reference/closed_set.npz`` holds the outputs recorded on the fixture with the protocol of
``XP_configs/smoke.yaml``: features, training draws, per-sequence log-probabilities of the global and per-LLM
classifiers, decisions, hierarchical scores and confidences. They must be reproduced to 1e-9
(xgboost 3.2, 4 threads for the global classifier).
"""

import numpy as np
import pytest

from audit_llm.Classification.hierarchical import PER_LLM_STREAM, HierarchicalClassifier
from audit_llm.Classification.protocol import make_splits, pair_index, prepare_cell, sample_batches, select_rows

TOL = 1e-9
N_TRAIN, INNER_VAL, TEST_PER_CLASS = 100, 10, 40


@pytest.mark.parametrize("token_pair", ["ali-yg", "df-bits"])
def test_features_match(fixture_grid, closed_reference, token_pair):
    X, y, flat = select_rows(fixture_grid.pair(token_pair).features, fixture_grid.feature_names)
    ref = closed_reference
    assert np.array_equal(y, ref[f"{token_pair}/y"]) and np.array_equal(flat, ref[f"{token_pair}/flat"])
    assert np.array_equal(np.isnan(X), np.isnan(ref[f"{token_pair}/features"]))
    assert np.nanmax(np.abs(X - ref[f"{token_pair}/features"])) <= TOL


@pytest.mark.parametrize("token_pair,split", [("ali-yg", 0), ("df-bits", 1)])
def test_classifier_outputs_match(fixture_grid, closed_reference, token_pair, split):
    ref, k = closed_reference, f"{token_pair}/s{split}"
    X, y, _ = select_rows(fixture_grid.pair(token_pair).features, fixture_grid.feature_names)
    cell = prepare_cell(X, y, make_splits(y, split + 1, TEST_PER_CLASS), token_pair, split, N_TRAIN, INNER_VAL)
    assert np.array_equal(cell.train_rows, ref[f"{k}/train_rows"]) and np.array_equal(cell.test_rows, ref[f"{k}/test_rows"])
    clf = HierarchicalClassifier(fixture_grid.class_names, n_jobs=4).fit(cell.X_train, cell.y_train, (PER_LLM_STREAM, split, pair_index(token_pair)))
    L, L2 = clf.global_log_probs(cell.X_test), clf.per_llm_log_probs(cell.X_test)
    assert np.abs(L.astype(np.float64) - ref[f"{k}/global_log_probs"]).max() <= TOL
    assert np.abs(L2 - ref[f"{k}/per_llm_log_probs"]).max() <= TOL
    for n_t in (1, 2, 4, 8):
        batches, _ = sample_batches(cell.y_test, n_t)
        d = clf.decide(L, L2, batches)
        assert np.array_equal(d["global_pred"], ref[f"{k}/nt{n_t}/global_pred"])
        assert np.array_equal(d["pred"], ref[f"{k}/nt{n_t}/pred"])
        assert np.abs(d["scores"] - ref[f"{k}/nt{n_t}/scores"]).max() <= TOL
        assert np.abs(d["confidence"] - ref[f"{k}/nt{n_t}/confidence"]).max() <= TOL

# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Log-sum aggregation: argmax of the summed per-query log-probabilities."""

import numpy as np

from audit_llm.Classification.log_sum_aggregation import LOG_FLOOR, floor_log_probs, log_sum_predict, log_sum_scores


def random_log_probs(rng, n=40, c=12):
    z = rng.normal(size=(n, c)) * 2.0
    return (z - np.log(np.exp(z).sum(1, keepdims=True))).astype(np.float32)


def test_scores_are_sums_over_the_rows_of_each_batch():
    rng = np.random.default_rng(0)
    L = random_log_probs(rng)
    batches = np.arange(40).reshape(10, 4)
    S = log_sum_scores(L, batches)
    assert S.shape == (10, 12) and S.dtype == np.float32
    np.testing.assert_allclose(S[3], L[12:16].sum(0), rtol=1e-6)
    assert (log_sum_predict(L, batches) == S.argmax(1)).all()


def test_single_query_is_the_argmax_of_the_posterior():
    rng = np.random.default_rng(1)
    L = random_log_probs(rng)
    assert (log_sum_predict(L, np.arange(40)[:, None]) == L.argmax(1)).all()


def test_floor_bounds_tiny_probabilities():
    L = floor_log_probs(np.array([[-1e4, -0.1, -2.0]]))
    assert L.dtype == np.float32 and L[0, 0] == LOG_FLOOR and L[0, 1] == np.float32(-0.1)


def test_empty_batch_list():
    assert log_sum_scores(np.zeros((5, 3), np.float32), np.zeros((0, 2), int)).shape == (0, 3)

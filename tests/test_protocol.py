# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Protocol: row selection, splits, validation carve, training draws, verification samples."""

import numpy as np

from audit_llm.Classification.protocol import (
    carve_validation, make_splits, nested_draw, pair_index, sample_batches, select_rows, seeded,
)


def test_draws_are_deterministic_prefixes_of_one_permutation():
    y = np.repeat(np.arange(4), 50)
    pool = np.random.default_rng(0).permutation(len(y))[:150]
    small, _ = nested_draw(pool, y, 10)
    large, eff = nested_draw(pool, y, 30)
    for c in range(4):
        assert np.array_equal(small[y[small] == c], large[y[large] == c][:10])
    assert all(v <= 30 for v in eff.values())


def test_validation_rows_are_disjoint_from_the_pool_and_capped_at_a_third():
    y = np.repeat(np.arange(3), [30, 6, 60])
    pool, val = carve_validation(np.arange(len(y)), y, 10, seeded(0, 0, "ali-yg", 7))
    assert not set(pool) & set(val)
    assert np.bincount(y[val]).tolist() == [10, 2, 10]


def test_splits_are_stratified_and_reproducible():
    y = np.repeat(np.arange(5), 100)
    a, b = make_splits(y, 2, 20), make_splits(y, 2, 20)
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
    assert (np.bincount(y[a[0][1]]) == 20).all()
    assert np.array_equal(make_splits(y, 1, 20)[0][1], a[0][1])  # split 0 does not depend on the number of splits


def test_samples_are_consecutive_rows_of_one_class():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 2])
    rows, labels = sample_batches(y, 2)
    assert labels.tolist() == [0, 0, 1, 1]
    assert rows.tolist() == [[0, 1], [2, 5], [3, 4], [6, 7]]
    assert sample_batches(y, 20)[0].shape == (0, 20)


def test_row_selection_drops_undefined_answers_and_fills_run_statistic():
    S, C, F = 6, 2, 57
    X = np.random.default_rng(0).normal(size=(S, C, F))
    X[0, 0, :] = np.nan  # no answer
    X[1, 0, :25] = np.nan  # > 80 % of the features.nist columns undefined
    names = [f"f{i}" for i in range(F)]
    names[22] = "run_ts"
    X[2, 1, 22] = np.nan
    Xs, y, flat = select_rows(X, names)
    assert (y == 0).sum() == 4 and (y == 1).sum() == 6
    assert flat[y == 0].tolist() == [4, 6, 8, 10]  # class-major order, answer index * C + class
    assert Xs[np.flatnonzero(flat == 5)[0], 22] == 0.0


def test_pair_index_follows_the_corpus_order():
    assert pair_index("0-1") == 0 and pair_index("wise-Flag") == 30 and pair_index("not-a-pair") >= 1000

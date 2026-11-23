# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""The mean imputer: a missing value is replaced by the mean of its column over the drawn training rows."""

import numpy as np

from audit_llm.Classification.imputation import TrainingMeanImputer
from audit_llm.Classification.protocol import make_splits, prepare_cell


def test_fill_value_is_the_column_mean_of_the_fitted_rows():
    X_train = np.array([[1.0, 4], [2.0, 5], [6.0, 9]])
    imp = TrainingMeanImputer().fit(X_train)
    np.testing.assert_array_equal(imp.mean_, [3.0, 6.0])


def test_transform_fills_missing_values_with_the_fitted_means():
    X_train = np.array([[1.0, 2, 3], [3.0, 4, np.nan], [np.nan, 6, 7]])
    out = TrainingMeanImputer().fit(X_train).transform(np.array([[np.nan, 1.0, np.nan], [5.0, np.nan, 5.0]]))
    np.testing.assert_array_equal(out, [[2.0, 1.0, 5.0], [5.0, 4.0, 5.0]])


def test_all_missing_training_column_is_filled_with_zero_and_observed_values_are_kept():
    X_train = np.array([[1.0, np.nan], [3.0, np.nan]])
    out = TrainingMeanImputer().fit(X_train).transform(np.array([[np.nan, np.nan], [7.0, 8.0]]))
    np.testing.assert_array_equal(out, [[2.0, 0.0], [7.0, 8.0]])


def test_cell_fill_is_the_mean_of_the_drawn_rows():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 60)
    X = rng.normal(size=(len(y), 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    splits = make_splits(y, 1, test_per_class=10)
    cell = prepare_cell(X, y, splits, "ali-yg", 0, n_train=20, inner_val=5)
    expected = np.nanmean(X[cell.train_rows], axis=0)
    nan_test = np.isnan(X[cell.test_rows])
    np.testing.assert_array_equal(cell.X_test[nan_test], np.broadcast_to(expected, nan_test.shape)[nan_test])
    assert not np.isnan(cell.X_train).any() and not np.isnan(cell.X_val).any()

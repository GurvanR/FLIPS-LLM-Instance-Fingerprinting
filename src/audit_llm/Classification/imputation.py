# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Imputation of the undefined feature values.

Some statistics are undefined on some sequences. A missing value is replaced by the mean of its column over the
drawn training rows (a column with no defined training value gets 0), and the same means fill the training,
validation and test rows.
"""

import warnings

import numpy as np


class TrainingMeanImputer:
    """Column means of the fitted rows (NaN-aware); all-NaN columns are filled with 0."""

    def fit(self, X_train: np.ndarray) -> "TrainingMeanImputer":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> NaN mean -> 0 below
            mean = np.nanmean(X_train, axis=0)
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), self.mean_, X)

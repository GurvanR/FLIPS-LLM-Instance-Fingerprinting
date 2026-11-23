# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Log-sum aggregation of the queries of a verification sample.

A classifier trained on single answers returns, for answer ``x_i``, log-probabilities ``log p(m_k | x_i)`` over the
instances. The ``N_t`` answers of a sample are combined by summing them, ``s_k = sum_i log p(m_k | x_i)`` (the
training classes are balanced, so the prior is uniform), and the sample is attributed to ``argmax_k s_k``. The
log-probabilities are the exact log-softmax of the classifier margins, floored at ``LOG_FLOOR`` (a class the
classifier never saw gets the floor), and are summed in float32. Any ``N_t`` is served without retraining.
"""

import numpy as np

LOG_FLOOR = -80.0


def floor_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """float32 copy of ``log_probs`` floored at ``LOG_FLOOR``."""
    out = np.asarray(log_probs, dtype=np.float32).copy()
    np.maximum(out, LOG_FLOOR, out=out)
    return out


def log_sum_scores(log_probs: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Scores ``(n_batches, C)``: per batch (row of ``batches``, indices into ``log_probs``) the sum of the rows' log-probabilities.

    The sum is taken in the dtype of ``log_probs`` (float32 for the global classifier)."""
    if len(batches) == 0:
        return np.zeros((0, log_probs.shape[1]), dtype=log_probs.dtype)
    return log_probs[batches].sum(axis=1)


def log_sum_predict(log_probs: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Predicted class of every batch: argmax of the log-sum scores (ties -> lowest class index)."""
    return log_sum_scores(log_probs, batches).argmax(axis=1)

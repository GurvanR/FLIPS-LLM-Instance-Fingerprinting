# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""XGBoost classifiers of the method.

Every classifier (the global instance classifier, the per-LLM classifiers, the classifiers of the Unseen detector's
held-out folds) is an ``XGBClassifier`` with the same regularized hyper-parameters (``XGB_PARAMS``) and balanced
sample weights. The number of boosting rounds is frozen on validation sequences: ``GLOBAL_ROUNDS`` = 357 for the
global classifier; a per-LLM classifier gets its rounds from a cross-half early-stopping rule on its own
training sequences (``select_rounds``). Outputs are exact log-probabilities (log-softmax of the margins).
"""

from typing import Optional, Sequence, Tuple

import numpy as np

XGB_PARAMS = dict(learning_rate=0.05, max_depth=4, min_child_weight=1.0, colsample_bytree=0.6, tree_method="hist", max_bin=64)
GLOBAL_ROUNDS = 357
RANDOM_STATE = 42
ROUND_SELECTION_PATIENCE = 60
ROUND_SELECTION_MAX = 1000


def fit_xgb(X: np.ndarray, y: np.ndarray, n_estimators: int, n_jobs: int = 4, eval_set: Optional[Sequence] = None,
            early_stopping_rounds: Optional[int] = None) -> Tuple[object, np.ndarray]:
    """Fit an XGBClassifier on (X, y) with balanced sample weights. Returns (model, classes): model column j = classes[j].

    A two-class fit uses the multi-class softmax objective with 2 classes, like every other fit."""
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_sample_weight

    classes, y_encoded = np.unique(y, return_inverse=True)
    params = dict(XGB_PARAMS, n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=n_jobs)
    if len(classes) == 2:  # keep the softmax family (two margin columns) instead of xgboost's binary objective
        params.update(objective="multi:softprob", num_class=2)
    if early_stopping_rounds:
        params.update(eval_metric="mlogloss", early_stopping_rounds=early_stopping_rounds)
    model = xgb.XGBClassifier(**params)
    encoded_eval = None if eval_set is None else [(Xe, np.searchsorted(classes, ye)) for Xe, ye in eval_set]
    model.fit(X, y_encoded, sample_weight=compute_sample_weight("balanced", y_encoded), eval_set=encoded_eval, verbose=False)
    return model, classes


def log_softmax(margins: np.ndarray) -> np.ndarray:
    m = np.asarray(margins, dtype=np.float64).copy()
    m -= m.max(axis=1, keepdims=True)
    return m - np.log(np.exp(m).sum(axis=1, keepdims=True))


def log_probs(model, X: np.ndarray) -> np.ndarray:
    """Exact log-probabilities (n, n_model_classes), float64, over the classes the model was trained on."""
    return log_softmax(model.predict(X, output_margin=True))


def full_log_probs(model, classes: np.ndarray, X: np.ndarray, n_classes: int, floor: float = -80.0, dtype=np.float32) -> np.ndarray:
    """Log-probabilities over all ``n_classes`` (classes absent from training at ``floor``), floored at ``floor``."""
    out = np.full((len(X), n_classes), floor, dtype=dtype)
    out[:, classes] = log_probs(model, X)
    np.maximum(out, floor, out=out)
    return out


def select_rounds(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, patience: int = ROUND_SELECTION_PATIENCE,
                  max_rounds: int = ROUND_SELECTION_MAX) -> Tuple[int, int, int]:
    """Cross-half early stopping. Each class's rows are permuted with ``rng`` and cut in two halves (first floor(n/2) rows: A);
    a model fitted on A is early-stopped on the mlogloss of B and vice versa. Returns (best rounds A->B, best rounds B->A,
    rounds = ceil of their mean)."""
    half_a, half_b = [], []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        rows = rows[rng.permutation(len(rows))]
        h = len(rows) // 2
        if h < 1:
            raise ValueError(f"class {c} has {len(rows)} training rows: cannot be split in halves")
        half_a.append(rows[:h])
        half_b.append(rows[h:])
    half_a, half_b = np.sort(np.concatenate(half_a)), np.sort(np.concatenate(half_b))
    best = []
    for train, held in ((half_a, half_b), (half_b, half_a)):
        model, _ = fit_xgb(X[train], y[train], max_rounds, 1, eval_set=[(X[held], y[held])], early_stopping_rounds=patience)
        curve = model.evals_result()["validation_0"]["mlogloss"]
        best.append(int(np.argmin(curve)) + 1)
    return best[0], best[1], (best[0] + best[1] + 1) // 2

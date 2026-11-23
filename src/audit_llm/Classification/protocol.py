# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Evaluation protocol: sequences, splits, training draws and verification samples of one (token pair, split) cell.

- **Sequences.** The rows of a token pair are its answers whose ``features.nist`` columns (the first ``N_NIST``: the
  NIST test statistics of the test library and the sequence length) are defined: per class, an answer with more than
  80 % of these columns undefined is dropped, and a class with fewer than 2 remaining answers is dropped; at
  most 500 answers per class are kept, in query order. ``run_ts`` (the runs test statistic) is set to 0 where it is
  undefined on an otherwise defined answer.
- **Splits.** ``StratifiedShuffleSplit(random_state=42)`` with ``test_per_class`` x (number of classes) test rows
  (128 per class in the paper); the same splits for every experiment.
- **Validation rows.** Per class, up to ``inner_val`` training rows (40; at most a third of the class) are kept out of
  the training draw. The paper uses these 40 validation sequences per instance for the temperature-scaling calibration
  of the closed-set confidence, a stored result (docs/reproduction/recompute.md#results-without-a-recompute-path).
- **Training draw.** ``N_r`` rows per class (300 in the paper) are drawn without replacement from the remaining
  training rows (one seeded permutation per class).
- **Verification samples.** A sample of ``N_t`` queries is ``N_t`` consecutive test rows of one class; a class with
  fewer than ``N_t`` test rows gives no sample.
"""

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from audit_llm.Classification.imputation import TrainingMeanImputer

# canonical order of the 31 token pairs of the corpus (it keys the random streams of the per-LLM classifiers)
TOKEN_PAIRS = (
    "0-1 ali-yg awk-ise City-load crit-SA df-bits distance-Menu ensor-buf FILE-occup four-iously g-ants hed-DI HE-mind hou-te "
    "isan-Provider Li-LOCK omic-anges Open-mart Options-abol orb-touch ORY-child Pr-eto Price-CI rel-IM scar-este Sec-prise "
    "Seg-ument ular-icons ump-reek way-ither wise-Flag"
).split()
N_NIST = 29  # columns of features.nist: 28 NIST test statistics and the sequence length
CLASS_SIZE_CAP = 500
UNDEFINED_FILL = {"run_ts": 0.0}
SPLIT_SEED = 42


def pair_index(token_pair: str) -> int:
    """Index of the token pair in ``TOKEN_PAIRS`` (a pair outside the corpus list gets a crc32-derived id)."""
    return TOKEN_PAIRS.index(token_pair) if token_pair in TOKEN_PAIRS else 1000 + zlib.crc32(token_pair.encode()) % 100000


def seeded(seed: int, split: int, token_pair: str, *tags: int) -> np.random.Generator:
    """Random stream of one (seed, split, token pair) cell, separated by integer tags."""
    return np.random.default_rng([seed, split, zlib.crc32(token_pair.encode()), *tags])


def _clean(X: np.ndarray, names: Sequence[str], fill: Dict[str, float]) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan)
    defined = ~np.isnan(X).all(axis=2)
    for name, value in fill.items():
        if name in names:
            col = X[:, :, list(names).index(name)]
            col[np.isnan(col) & defined] = value
    return X


def select_rows(features: np.ndarray, names: Sequence[str], max_undefined_share: float = 0.8, min_rows: int = 2,
                max_per_class: int = CLASS_SIZE_CAP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of a token pair. ``features`` (S, C, F) with the ``features.nist`` columns first.

    Returns X (N, F) float64, y (N,) class index, flat (N,) = answer index * C + class (class-major order)."""
    S, C, F = features.shape
    nist = _clean(np.array(features[:, :, :N_NIST], dtype=np.float64), names[:N_NIST], UNDEFINED_FILL)
    for c in range(C):
        block = nist[:, c, :]
        block[np.isnan(block).sum(axis=1) > max_undefined_share * N_NIST, :] = np.nan
        if (~np.isnan(block).all(axis=1)).sum() < min_rows:
            block[:, :] = np.nan
    flat_nist = nist.reshape(S * C, N_NIST)
    y_all = np.tile(np.arange(C), S)
    flat_all = np.arange(S * C)
    keep = ~np.isnan(flat_nist).all(axis=1)
    y_all, flat_all = y_all[keep], flat_all[keep]
    order = np.concatenate([np.where(y_all == c)[0][:max_per_class] for c in np.unique(y_all)])
    y, flat = y_all[order], flat_all[order]
    full = _clean(np.array(features, dtype=np.float64), names, UNDEFINED_FILL)
    X = full.reshape(S * C, F)[flat]
    return X, y, flat


def make_splits(y: np.ndarray, n_splits: int, test_per_class: int = 128, seed: int = SPLIT_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(train rows, test rows) of each split."""
    sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=seed, test_size=test_per_class * len(np.unique(y)))
    return list(sss.split(np.zeros(len(y)), y))


def carve_validation(train: np.ndarray, y: np.ndarray, n_per_class: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Per class, set aside up to ``n_per_class`` training rows (at most a third of the class). Returns (pool, validation)."""
    pool, val = [], []
    for c in np.unique(y[train]):
        rows = train[y[train] == c]
        k = min(n_per_class, len(rows) // 3)
        perm = rng.permutation(len(rows))
        val.append(rows[perm[:k]])
        pool.append(rows[perm[k:]])
    return np.concatenate(pool), np.concatenate(val)


def nested_draw(pool: np.ndarray, y: np.ndarray, n_per_class: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """``n_per_class`` rows per class without replacement (the whole pool of a smaller class), grouped by ascending
    class: the first ``n_per_class`` rows of one permutation per class, seeded by the pool. Returns (rows, rows per
    class)."""
    rng = np.random.default_rng([zlib.crc32(np.ascontiguousarray(pool, dtype=np.int64).tobytes()), 2828])
    out, effective = [], {}
    for c in np.unique(y[pool]):
        rows = pool[y[pool] == c]
        perm = rng.permutation(len(rows))
        sel = perm[:n_per_class]
        out.append(rows[sel])
        effective[int(c)] = len(sel)
    return np.concatenate(out), effective


def sample_batches(y_eval: np.ndarray, n_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Verification samples: ``n_t`` consecutive evaluation rows of one class. Returns rows (n_samples, n_t) as positions
    into the evaluation rows, and the label of every sample."""
    rows, labels = [], []
    for c in np.unique(y_eval):
        pos = np.flatnonzero(y_eval == c)
        n = len(pos) // n_t
        if n:
            rows.append(pos[: n * n_t].reshape(n, n_t))
            labels.append(np.full(n, c))
    if not rows:
        return np.zeros((0, n_t), int), np.zeros(0, int)
    return np.concatenate(rows), np.concatenate(labels)


@dataclass
class Cell:
    """The data of one (token pair, split) cell, imputed. Row arrays index the token pair's rows (``select_rows``)."""

    token_pair: str
    split: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray
    val_rows: np.ndarray
    pool_rows: np.ndarray
    rows_per_class: Dict[int, int] = field(default_factory=dict)


def prepare_cell(X: np.ndarray, y: np.ndarray, splits, token_pair: str, split: int, n_train: int, inner_val: int = 40,
                 seed: int = 0, classes: Optional[np.ndarray] = None) -> Cell:
    """Training draw and imputation of one cell.

    ``classes``: restrict the training side to these classes (open set: the Known instances); the test rows keep every class.
    The imputer is fitted on the drawn training rows and applied to training, test and validation rows."""
    train, test = splits[split]
    if classes is not None:
        train = train[np.isin(y[train], classes)]
    pool, val = carve_validation(train, y, inner_val, seeded(seed, split, token_pair, 7))
    drawn, effective = nested_draw(pool, y, n_train)
    imputer = TrainingMeanImputer().fit(X[drawn])
    return Cell(token_pair, split, imputer.transform(X[drawn]), y[drawn], imputer.transform(X[test]), y[test],
                imputer.transform(X[val]), y[val], drawn, test, val, pool, effective)


def independent_draw(pool: np.ndarray, y: np.ndarray, n_per_class: int, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[int, int]]:
    """``n_per_class`` rows per class without replacement (the whole pool of a smaller class), grouped by ascending class,
    from one permutation per class drawn with ``rng``. Returns (rows, rows per class)."""
    out, effective = [], {}
    for c in np.unique(y[pool]):
        rows = pool[y[pool] == c]
        sel = rng.permutation(len(rows))[:n_per_class]
        out.append(rows[sel])
        effective[int(c)] = len(sel)
    return np.concatenate(out), effective


def prepare_open_set_cell(X: np.ndarray, y: np.ndarray, splits, token_pair: str, split: int, n_train: int, known: np.ndarray,
                          inner_val: int = 40, seed: int = 0) -> Cell:
    """Training draw and imputation of one open-set cell.

    The training side keeps the Known instances ``known`` only; the test rows keep every instance (Known and Unseen).
    Validation rows are set aside as in the closed set, then ``n_train`` rows per Known instance are drawn without
    replacement with the stream ``seeded(seed, split, token_pair, 1, n_train)`` (``independent_draw``). The imputer is
    fitted on the drawn training rows and applied to training, test and validation rows."""
    train, test = splits[split]
    train = train[np.isin(y[train], known)]
    pool, val = carve_validation(train, y, inner_val, seeded(seed, split, token_pair, 7))
    drawn, effective = independent_draw(pool, y, n_train, seeded(seed, split, token_pair, 1, n_train))
    imputer = TrainingMeanImputer().fit(X[drawn])
    return Cell(token_pair, split, imputer.transform(X[drawn]), y[drawn], imputer.transform(X[test]), y[test],
                imputer.transform(X[val]), y[val], drawn, test, val, pool, effective)

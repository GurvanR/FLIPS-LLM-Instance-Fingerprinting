# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Open set: the Unseen detector and the evaluation of one open-set cell.

**Open-set cell.** ``n_unseen`` instances (23 of 237 in the paper) are drawn at random, independently of the token
pair, and treated as *Unseen*: they are removed from the training side and must be labelled as such at
verification. The other instances are *Known*. From the training sequences of the Known instances we train the
instance classifier (a global classifier and the per-LLM classifiers, ``hierarchical.py``) and build the Unseen
detector; samples of ``N_t`` test sequences of both Known and Unseen instances are then labelled: *Unseen* when the
detector flags them, else the instance chosen by the hierarchical classifier.

**Unseen detector (BuildDetector).** Built from the Known training sequences only; Known instances are held out in
turn to simulate Unseen ones:

1. *Hold out.* The Known instances are split into ``n_folds`` (10) class folds, the instances of one LLM being spread
   over different folds (an LLM with a single Known instance is never held out, so a held-out instance always keeps a
   sibling in training). For each fold ``f`` a global classifier ``c_f`` is trained on the instances of the other
   folds, with ``held_out_per_class`` (40, at most a third) sequences of each of them set aside. For ``c_f`` the
   instances of fold ``f`` are the *held-out instances* (simulated Unseen); the instances of the other folds are Known.
2. *Draw samples.* Samples of ``N_t`` sequences are drawn and scored by ``c_f``: ``known_batches_per_instance`` (10)
   per Known instance from its set-aside sequences, ``unseen_batches_per_instance`` (40) per held-out instance from
   ``unseen_rows_per_instance`` (40) of its sequences. An instance with fewer than ``min_rows_per_instance`` (120)
   training sequences gives no sample.
3. *Describe.* Each sample is described by the 38 statistics of ``sample_statistics``, computed on the output of
   ``c_f`` for the sample (its log-probabilities over the instances), not on the bit sequences.
4. *Fit.* A logistic regression (L2, ``C`` = 0.1, standardised statistics) separates the held-out-instance samples from
   the Known samples that are identified correctly. A Known sample is identified correctly when the hierarchical
   decision of fold ``f`` (``c_f`` selects the LLM, the per-LLM classifier of that LLM trained on the fold's training
   sequences picks the instance) is its instance. Sample weights: every source instance has the same total weight
   within its side, the held-out side carries ``unseen_weight_share`` (9.7 %) of the weight; the weights are computed
   on all the training samples (mean 1) and the misclassified Known samples are then dropped (no renormalisation).
   Its output is the Unseen score ``q`` (the logit of the Unseen probability).
5. *Set the threshold.* ``tau`` is the ``1 - r*`` quantile of the out-of-fold scores of the held-out-instance
   samples (``r*`` = 50 %: the median), so that a share ``r*`` of them is flagged. Out-of-fold: the samples of fold
   ``f`` are scored by a regression fitted on the samples of the other folds whose source instance is not in fold ``f``.

The final regression is fitted on all the samples. At verification, the statistics are computed from the scores of
the final global classifier (trained on all the Known training sequences) and a sample is flagged Unseen when
``q >= tau``. The fold classifiers are trained on fewer sequences than the final global classifier, whose scores are
therefore somewhat sharper than those on which the detector was fitted.

Log-probabilities: the global and fold classifiers give exact log-softmax outputs over the Known (fold) instances,
unfloored, stored in float32; the per-LLM classifiers give log-probabilities floored at -80. Everything else is float64.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from audit_llm.Classification.hierarchical import PER_LLM_STREAM, HierarchicalClassifier, decide_within, llm_blocks
from audit_llm.Classification.protocol import Cell, prepare_open_set_cell, sample_batches
from audit_llm.Classification.xgb import GLOBAL_ROUNDS, ROUND_SELECTION_MAX, ROUND_SELECTION_PATIENCE, fit_xgb, log_probs

UNSEEN = -1  # label of a sample flagged Unseen

# paper constants
N_UNSEEN = 23
N_TRAIN = 300
INNER_VAL = 40
N_FOLDS = 10
HELD_OUT_PER_CLASS = 40
UNSEEN_ROWS_PER_INSTANCE = 40
KNOWN_BATCHES_PER_INSTANCE = 10
UNSEEN_BATCHES_PER_INSTANCE = 40
MIN_ROWS_PER_INSTANCE = 120
UNSEEN_WEIGHT_SHARE = 0.097
DETECTOR_C = 0.1
TARGET_RECALL = 0.5
TARGET_RECALL_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95

# first keys of the random streams (the rest of a key is (split, draw, ...))
UNSEEN_DRAW_STREAM = 4242
FOLD_ASSIGN_STREAM = 12
FOLD_ROWS_STREAM = 13
SAMPLE_STREAM = 32
FOLD_PER_LLM_STREAM = 53

STATISTIC_GROUPS = {
    # decision margins of the summed log-probabilities S_c = sum_i log p(c | x_i); c^ = argmax S
    "decision": ["margin",                    # S_c^ - logsumexp_{c != c^} S_c
                 "margin_within_llm",         # S_c^ - logsumexp over the other instances of c^'s LLM (= margin if none)
                 "llm_margin",                # logsumexp of S over c^'s LLM - logsumexp over the other LLMs
                 "mean_log_prob_decided",     # mean_i log p(c^ | x_i)
                 "vote_share",                # share of the answers whose own argmax is c^
                 "std_log_prob_decided"],     # std_i log p(c^ | x_i)
    # gaps of the summed log-probabilities (sharpness-invariant)
    "gaps": [f"gap_{j}" for j in range(2, 11)]                      # S_(1) - S_(j), sorted scores
            + [f"gap_within_llm_{j}" for j in range(2, 6)]          # S_(1) - j-th score inside c^'s LLM (padded with the gap to the best other LLM)
            + ["llm_mass_gap_12", "llm_mass_gap_13"],               # gaps between the sorted LLM log-masses logsumexp_{c in l} S_c
    # mean posterior P_c = mean_i p(c | x_i), renormalised
    "mean_posterior": [f"log_mean_prob_{j}" for j in range(1, 6)]   # log of the 5 largest P_c
                      + ["log_mean_prob_margin",                    # log P_(1) - log P_(2)
                         "mean_prob_entropy",                       # entropy of P
                         "llm_log_odds"],                           # log-odds of the P mass of c^'s LLM
    # distribution of the per-answer evidence
    "per_answer": ["max_prob_mean", "max_prob_std", "max_prob_q10", "max_prob_q50", "max_prob_q90",   # max_c p(c | x_i)
                   "log_prob_decided_q10", "log_prob_decided_q50", "log_prob_decided_q90",            # log p(c^ | x_i)
                   "llm_vote_share"],                                                                 # share of answers voting for c^'s LLM
}
STATISTIC_NAMES = [name for group in STATISTIC_GROUPS.values() for name in group]
N_STATISTICS = len(STATISTIC_NAMES)  # 38


# ---------------------------------------------------------------- the 38 statistics of a sample
def _lse(S: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(S, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m + np.log(np.exp(S - m).sum(axis=axis, keepdims=True)), axis)


def _sorted_desc(A: np.ndarray) -> np.ndarray:
    return -np.sort(-A, axis=1)


def _statistics_chunk(Q: np.ndarray, llm: np.ndarray) -> np.ndarray:
    """Q (n, N_t, K) float64 log-probabilities, llm (K,) LLM id of every column."""
    n, n_t, K = Q.shape
    ix = np.arange(n)
    S = Q.sum(1)
    pred = S.argmax(1)
    top = S[ix, pred]
    same_llm = llm[None, :] == llm[pred][:, None]
    # decision margins
    others = S.copy()
    others[ix, pred] = -np.inf
    margin = top - _lse(others)
    inside = _lse(np.where(same_llm, others, -np.inf))
    alone = ~np.isfinite(inside)
    margin_within = np.where(alone, margin, top - np.where(alone, 0, inside))
    llm_margin = _lse(np.where(same_llm, S, -np.inf)) - _lse(np.where(same_llm, -np.inf, S))
    q_dec = Q[ix, :, pred]
    decision = np.c_[margin, margin_within, llm_margin, q_dec.mean(1), (Q.argmax(2) == pred[:, None]).mean(1), q_dec.std(1)]
    # gaps
    Ss = _sorted_desc(S)
    gaps = Ss[:, :1] - Ss[:, 1:10]
    pad = Ss[:, 0] - np.where(same_llm, -np.inf, S).max(1)
    Sb = _sorted_desc(np.where(same_llm, S, -np.inf))
    within = np.empty((n, 4))
    for j in range(4):
        v = Sb[:, j + 1] if j + 1 < K else np.full(n, -np.inf)
        within[:, j] = np.where(np.isfinite(v), Ss[:, 0] - v, pad)
    n_llm = llm.max() + 1
    M = np.full((n, n_llm), -np.inf)
    for b in range(n_llm):
        cols = llm == b
        if cols.any():
            M[:, b] = _lse(S[:, cols])
    Ms = _sorted_desc(M)
    mass_gaps = np.c_[Ms[:, 0] - Ms[:, 1], Ms[:, 0] - Ms[:, 2]] if n_llm >= 3 else np.zeros((n, 2))
    # mean posterior
    P = np.exp(Q).mean(1)
    P = P / P.sum(1, keepdims=True)
    lp = np.log(np.maximum(_sorted_desc(P)[:, :5], 1e-300))
    entropy = -(P * np.log(np.maximum(P, 1e-300))).sum(1)
    mass = np.clip((P * same_llm).sum(1), 1e-12, 1 - 1e-12)
    posterior = np.c_[lp, lp[:, 0] - lp[:, 1], entropy, np.log(mass) - np.log1p(-mass)]
    # per-answer evidence
    mx = np.exp(Q.max(2))
    vote_llm = (llm[Q.argmax(2)] == llm[pred][:, None]).mean(1)
    per_answer = np.c_[mx.mean(1), mx.std(1), *np.quantile(mx, [0.1, 0.5, 0.9], axis=1), *np.quantile(q_dec, [0.1, 0.5, 0.9], axis=1), vote_llm]
    return np.c_[decision, gaps, within, mass_gaps, posterior, per_answer]


def sample_statistics(log_probs: np.ndarray, batches: np.ndarray, column_llm: np.ndarray, chunk: int = 256) -> np.ndarray:
    """The 38 statistics (``STATISTIC_NAMES``, grouped as ``STATISTIC_GROUPS``) of every sample.

    Args:
        log_probs: (n_rows, K) per-answer log-probabilities of a classifier over its K instances (unfloored).
        batches: (n_samples, N_t) rows of ``log_probs`` forming each sample.
        column_llm: (K,) LLM id of every column (ids of the full instance list, so absent LLMs count in the LLM mass gaps).
    Returns (n_samples, 38) float64. Needs K >= 10 (gaps up to the 10th score) and at least 3 LLMs with >= 2 columns for
    every statistic to be finite."""
    column_llm = np.asarray(column_llm)
    out = [_statistics_chunk(log_probs[batches[i:i + chunk]].astype(np.float64), column_llm) for i in range(0, len(batches), chunk)]
    return np.concatenate(out) if out else np.zeros((0, N_STATISTICS))


def summed_decision(log_probs: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Column of the largest float64 sum of the log-probabilities of every sample (the decision the statistics refer to)."""
    return log_probs[batches].astype(np.float64).sum(1).argmax(1)


# ---------------------------------------------------------------- folds, samples, weights, regression, threshold
def draw_unseen(n_instances: int, n_unseen: int, draw: int) -> np.ndarray:
    """Unseen instances of open-set draw ``draw`` (sorted), independent of the token pair and split."""
    return np.sort(np.random.default_rng([UNSEEN_DRAW_STREAM, draw]).choice(n_instances, n_unseen, replace=False))


def assign_class_folds(instance_names: Sequence[str], known: np.ndarray, split: int, draw: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold (0..n_folds-1) of every Known instance (positions of ``known``); -1 for an instance that is the only Known one of its LLM.

    LLMs in sorted name order; the instances of one LLM get consecutive folds (mod ``n_folds``) from a random offset, in a
    random order, so they are spread over different folds as long as the LLM has at most ``n_folds`` Known instances."""
    llm_of_known = np.unique(llm_blocks(instance_names)[0][known], return_inverse=True)[1]
    members = [np.flatnonzero(llm_of_known == b) for b in range(llm_of_known.max() + 1)]
    key = [FOLD_ASSIGN_STREAM, split, draw] + ([] if n_folds == N_FOLDS else [n_folds, 0])
    rng = np.random.default_rng(key)
    fold = np.full(len(known), -1)
    for m in members:
        if len(m) < 2:
            continue
        offset = int(rng.integers(n_folds))
        for i, k in enumerate(rng.permutation(m)):
            fold[k] = (offset + i) % n_folds
    for k in np.flatnonzero(fold >= 0):
        if not any(s != k and fold[s] != fold[k] for s in members[llm_of_known[k]]):
            raise ValueError(f"fold {fold[k]}: held-out instance {instance_names[known[k]]} has no sibling left in training")
    return fold


def sample_weights(is_unseen: np.ndarray, source: np.ndarray, unseen_share: float = UNSEEN_WEIGHT_SHARE) -> np.ndarray:
    """Weights of the detector's training samples: every source instance has the same total weight within its side, the
    Unseen side totals ``unseen_share`` of the weight; rescaled to mean 1. ``source``: non-negative instance codes."""
    is_unseen = np.asarray(is_unseen, bool)
    w = np.zeros(len(is_unseen))
    for side, total in ((True, unseen_share), (False, 1 - unseen_share)):
        m = is_unseen == side
        if not m.any():
            continue
        count = np.bincount(source[m])
        w[m] = (total / np.count_nonzero(count)) / count[source[m]]
    return w * len(is_unseen) / w.sum()


class LogisticDetector:
    """L2 logistic regression (lbfgs) on standardised statistics (mean / std of its training samples; std < 1e-9 -> 1). Label 1 = Unseen."""

    def __init__(self, C: float = DETECTOR_C, max_iter: int = 5000):
        self.C, self.max_iter = C, max_iter

    def fit(self, Z: np.ndarray, is_unseen: np.ndarray, weights: np.ndarray) -> "LogisticDetector":
        from sklearn.linear_model import LogisticRegression

        self.mean_ = Z.mean(0)
        sd = Z.std(0)
        self.scale_ = np.where(sd < 1e-9, 1.0, sd)
        self.model_ = LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter)
        self.model_.fit((Z - self.mean_) / self.scale_, np.asarray(is_unseen).astype(np.int64), sample_weight=weights)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        """Unseen score q (logit of the Unseen probability; higher = more Unseen)."""
        return self.model_.decision_function((Z - self.mean_) / self.scale_)


def threshold(held_out_scores: np.ndarray, target_recall: float = TARGET_RECALL) -> float:
    """``1 - target_recall`` quantile (linear) of the out-of-fold scores of the held-out-instance samples."""
    return float(np.quantile(held_out_scores, 1 - target_recall))


@dataclass
class ClassFold:
    """One class fold: its held-out instances and the classifiers trained without them.

    Row arrays are positions into the training draw; ``log_probs`` / ``per_llm_log_probs`` rows are the set-aside
    sequences of the Known instances, then the sequences of the held-out instances; columns are ``classes``."""

    index: int
    held: np.ndarray            # positions (into the Known list) of the held-out instances
    classes: np.ndarray         # instance ids of the fold classifier's columns
    train_rows: np.ndarray
    known_rows: np.ndarray
    known_source: np.ndarray    # Known position of each set-aside row
    unseen_rows: np.ndarray
    unseen_source: np.ndarray
    log_probs: np.ndarray       # float32, unfloored
    per_llm_log_probs: np.ndarray  # float32, floored at -80; 0 in the columns of a one-instance LLM
    per_llm_rounds: Dict[int, int] = field(default_factory=dict)


@dataclass
class DetectorSamples:
    """The detector's training samples at one N_t (all folds, fold order)."""

    statistics: np.ndarray      # (n, 38)
    is_unseen: np.ndarray       # sample of a held-out instance
    source: np.ndarray          # Known position of the sampled instance
    fold: np.ndarray            # fold whose classifier scored the sample
    batches: np.ndarray         # rows into the fold's [set-aside; held-out] rows
    global_pred: np.ndarray     # instance id chosen by the fold's global classifier
    pred: np.ndarray            # hierarchical decision of the fold
    correct: np.ndarray         # Known sample whose hierarchical decision is its instance


class UnseenDetector:
    """BuildDetector on the training sequences of the Known instances (see the module docstring).

    ``fit_folds`` trains the fold classifiers (independent of N_t); ``build(n_t)`` draws the samples, fits the regression
    and sets the threshold for samples of ``n_t`` sequences. XGBoost ``hist`` results depend on the thread count
    (``n_jobs`` of the fold classifiers; the per-LLM classifiers use 1 thread)."""

    def __init__(self, instance_names: Sequence[str], n_folds: int = N_FOLDS, held_out_per_class: int = HELD_OUT_PER_CLASS,
                 unseen_rows_per_instance: int = UNSEEN_ROWS_PER_INSTANCE, known_batches_per_instance: int = KNOWN_BATCHES_PER_INSTANCE,
                 unseen_batches_per_instance: int = UNSEEN_BATCHES_PER_INSTANCE, min_rows_per_instance: int = MIN_ROWS_PER_INSTANCE,
                 unseen_weight_share: float = UNSEEN_WEIGHT_SHARE, C: float = DETECTOR_C, global_rounds: int = GLOBAL_ROUNDS,
                 round_patience: int = ROUND_SELECTION_PATIENCE, round_max: int = ROUND_SELECTION_MAX, n_jobs: int = 4):
        self.instance_names = list(instance_names)
        self.blocks, _ = llm_blocks(self.instance_names)
        self.n_folds, self.held_out_per_class, self.unseen_rows_per_instance = n_folds, held_out_per_class, unseen_rows_per_instance
        self.known_batches_per_instance, self.unseen_batches_per_instance = known_batches_per_instance, unseen_batches_per_instance
        self.min_rows_per_instance, self.unseen_weight_share, self.C = min_rows_per_instance, unseen_weight_share, C
        self.global_rounds, self.round_patience, self.round_max, self.n_jobs = global_rounds, round_patience, round_max, n_jobs

    # ---- hold out
    def fit_folds(self, X_train: np.ndarray, y_train: np.ndarray, split: int, draw: int) -> "UnseenDetector":
        """Class folds of the Known instances (``np.unique(y_train)``) and, per non-empty fold, the fold's global and per-LLM classifiers."""
        self.split, self.draw = split, draw
        self.known = np.unique(y_train)
        position = np.full(len(self.instance_names), -1)
        position[self.known] = np.arange(len(self.known))
        source = position[y_train]
        self.rows_per_instance = np.bincount(source, minlength=len(self.known))
        self.class_fold = assign_class_folds(self.instance_names, self.known, split, draw, self.n_folds)
        self.folds: List[ClassFold] = []
        for f in range(self.n_folds):
            held = np.flatnonzero(self.class_fold == f)
            if len(held) == 0:
                continue
            rng = np.random.default_rng([FOLD_ROWS_STREAM, split, draw, self.n_folds, 0, f])
            train, aside = [], []
            for k in range(len(self.known)):
                if self.class_fold[k] == f:
                    continue
                rows = rng.permutation(np.flatnonzero(source == k))
                n_aside = min(self.held_out_per_class, len(rows) // 3)
                aside.append(rows[:n_aside])
                train.append(rows[n_aside:])
            train, aside = np.concatenate(train), np.concatenate(aside)
            unseen = np.concatenate([rng.permutation(np.flatnonzero(source == k))[:self.unseen_rows_per_instance] for k in held])
            model, classes = fit_xgb(X_train[train], y_train[train], self.global_rounds, self.n_jobs)
            scored = np.r_[aside, unseen]
            L = log_probs(model, X_train[scored]).astype(np.float32)
            per_llm = HierarchicalClassifier(self.instance_names, round_patience=self.round_patience, round_max=self.round_max)
            per_llm.fit_per_llm(X_train[train], y_train[train], (FOLD_PER_LLM_STREAM, split, draw, self.n_folds, 0, f))
            L2 = per_llm.per_llm_log_probs(X_train[scored])[:, classes].astype(np.float32)
            self.folds.append(ClassFold(f, held, classes, train, aside, source[aside], unseen, source[unseen], L, L2,
                                          {g: m["rounds"] for g, m in per_llm.per_llm.items()}))
        return self

    # ---- draw samples, describe, decide
    def samples(self, n_t: int) -> DetectorSamples:
        """The detector's training samples of ``n_t`` sequences, described by the fold classifiers."""
        acc = {k: [] for k in ("statistics", "is_unseen", "source", "fold", "batches", "global_pred", "pred")}
        for fd in self.folds:
            rng = np.random.default_rng([SAMPLE_STREAM, self.split, self.draw, self.n_folds, 0, fd.index])
            offset = len(fd.known_rows)
            batches, is_unseen, src = [], [], []
            for k in np.unique(fd.known_source):
                if self.rows_per_instance[k] < self.min_rows_per_instance:
                    continue
                rows = np.flatnonzero(fd.known_source == k)
                for _ in range(self.known_batches_per_instance):
                    batches.append(rng.choice(rows, n_t, replace=len(rows) < n_t))
                    is_unseen.append(False)
                    src.append(k)
            for k in fd.held:
                if self.rows_per_instance[k] < self.min_rows_per_instance:
                    continue
                rows = offset + np.flatnonzero(fd.unseen_source == k)
                for _ in range(self.unseen_batches_per_instance):
                    batches.append(rng.choice(rows, n_t, replace=len(rows) < n_t))
                    is_unseen.append(True)
                    src.append(k)
            if not batches:
                continue
            batches = np.array(batches)
            L = fd.log_probs.astype(np.float64)
            column_llm = self.blocks[fd.classes]
            global_col = summed_decision(L, batches)
            groups = {int(g): np.flatnonzero(column_llm == g) for g in np.unique(column_llm)}
            hier_col = decide_within(fd.per_llm_log_probs.astype(np.float64), batches, groups, column_llm[global_col])
            for k, v in (("statistics", sample_statistics(L, batches, column_llm)), ("is_unseen", np.array(is_unseen)), ("source", np.array(src)),
                         ("fold", np.full(len(batches), fd.index)), ("batches", batches), ("global_pred", fd.classes[global_col]), ("pred", fd.classes[hier_col])):
                acc[k].append(v)
        out = {k: np.concatenate(v) for k, v in acc.items()}
        correct = ~out["is_unseen"] & (out["pred"] == self.known[out["source"]])
        return DetectorSamples(correct=correct, **out)

    # ---- fit, out-of-fold scores, threshold
    def _fit_regression(self, S: DetectorSamples, train: np.ndarray) -> LogisticDetector:
        w = np.zeros(len(S.is_unseen))
        w[train] = sample_weights(S.is_unseen[train], S.source[train], self.unseen_weight_share)
        used = train & (S.is_unseen | S.correct)
        return LogisticDetector(self.C).fit(S.statistics[used], S.is_unseen[used], w[used])

    def out_of_fold_scores(self, S: DetectorSamples) -> np.ndarray:
        """Score of every sample by a regression fitted without its fold and without the samples of that fold's held-out instances."""
        q = np.full(len(S.is_unseen), np.nan)
        for f in np.unique(S.fold):
            scored = S.fold == f
            train = ~scored & (self.class_fold[S.source] != f)
            q[scored] = self._fit_regression(S, train).score(S.statistics[scored])
        return q

    def build(self, n_t: int, target_recall: float = TARGET_RECALL) -> Dict[str, object]:
        """Samples, out-of-fold scores, threshold ``tau`` (target recall ``target_recall``) and final regression for ``n_t``.

        Returns dict(samples, oof_scores, threshold, detector)."""
        S = self.samples(n_t)
        q = self.out_of_fold_scores(S)
        return dict(samples=S, oof_scores=q, threshold=threshold(q[S.is_unseen], target_recall), detector=self._fit_regression(S, np.ones(len(q), bool)))


# ---------------------------------------------------------------- one open-set cell
def verification_samples(y_test: np.ndarray, known: np.ndarray, n_t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verification samples of ``n_t`` consecutive test rows of one instance: Known instances first, then Unseen (each in
    ascending instance order). Returns (rows, true instance of each sample, is_unseen)."""
    rows, labels = sample_batches(y_test, n_t)
    is_unseen = ~np.isin(labels, known)
    order = np.argsort(is_unseen, kind="stable")
    return rows[order], labels[order], is_unseen[order]


@dataclass
class OpenSetCell:
    """A fitted open-set cell: the training draw, the instance classifier of the Known instances and the Unseen detector's folds."""

    cell: Cell
    unseen: np.ndarray
    known: np.ndarray
    classifier: HierarchicalClassifier
    detector: UnseenDetector
    test_log_probs: np.ndarray          # (n_test, K) float32, global classifier, unfloored, Known columns
    test_per_llm_log_probs: np.ndarray  # (n_test, K) float64, per-LLM classifiers, floored at -80


def fit_open_set_cell(X: np.ndarray, y: np.ndarray, splits, instance_names: Sequence[str], token_pair: str, split: int, draw: int,
                      n_train: int = N_TRAIN, inner_val: int = INNER_VAL, n_unseen: int = N_UNSEEN, seed: int = 0, n_folds: int = N_FOLDS,
                      held_out_per_class: int = HELD_OUT_PER_CLASS, unseen_rows_per_instance: int = UNSEEN_ROWS_PER_INSTANCE,
                      known_batches_per_instance: int = KNOWN_BATCHES_PER_INSTANCE, unseen_batches_per_instance: int = UNSEEN_BATCHES_PER_INSTANCE,
                      min_rows_per_instance: int = MIN_ROWS_PER_INSTANCE, unseen_weight_share: float = UNSEEN_WEIGHT_SHARE, C: float = DETECTOR_C,
                      global_rounds: int = GLOBAL_ROUNDS, round_patience: int = ROUND_SELECTION_PATIENCE, round_max: int = ROUND_SELECTION_MAX,
                      n_jobs: int = 4) -> OpenSetCell:
    """Everything of an open-set cell that does not depend on N_t: Unseen draw, training draw of the Known instances,
    global and per-LLM classifiers (per-LLM round selection keyed by (split, draw, LLM)), fold classifiers of the detector."""
    unseen = draw_unseen(len(instance_names), n_unseen, draw)
    known_all = np.setdiff1d(np.arange(len(instance_names)), unseen)
    cell = prepare_open_set_cell(X, y, splits, token_pair, split, n_train, known_all, inner_val, seed)
    known = np.unique(cell.y_train)
    clf = HierarchicalClassifier(instance_names, global_rounds, n_jobs, round_patience, round_max)
    clf.fit(cell.X_train, cell.y_train, (PER_LLM_STREAM, split, draw))
    L = log_probs(clf.global_model, cell.X_test).astype(np.float32)
    L2 = clf.per_llm_log_probs(cell.X_test)[:, known]
    det = UnseenDetector(instance_names, n_folds, held_out_per_class, unseen_rows_per_instance, known_batches_per_instance, unseen_batches_per_instance,
                         min_rows_per_instance, unseen_weight_share, C, global_rounds, round_patience, round_max, n_jobs)
    det.fit_folds(cell.X_train, cell.y_train, split, draw)
    return OpenSetCell(cell, unseen, known, clf, det, L, L2)


def evaluate_at(fitted: OpenSetCell, n_t: int, target_recall: float = TARGET_RECALL) -> Dict[str, object]:
    """Open-set evaluation of a fitted cell at ``n_t``: build the detector, label the test samples, metrics.

    Returned keys: ``accuracy`` (share of correct labels, Unseen samples being correct when flagged), ``unseen_recall``,
    ``unseen_precision`` (share of Unseen samples among the flagged ones; nan if none flagged), ``known_accuracy_closed``
    (hierarchical decision on the Known samples, no rejection), ``accuracy_never_reject``, ``threshold``, ``scores``,
    ``flags``, ``pred`` (final labels, ``UNSEEN`` = -1), ``hierarchical_pred``, ``global_pred``, ``labels``, ``is_unseen``,
    ``batches`` (rows into the cell's test rows), ``statistics``, and the detector's ``samples`` / ``oof_scores`` / ``detector``."""
    built = fitted.detector.build(n_t, target_recall)
    known = fitted.known
    rows, labels, is_unseen = verification_samples(fitted.cell.y_test, known, n_t)
    column_llm = fitted.detector.blocks[known]
    Z = sample_statistics(fitted.test_log_probs, rows, column_llm)
    global_col = summed_decision(fitted.test_log_probs, rows)
    groups = {int(g): np.flatnonzero(column_llm == g) for g in np.unique(column_llm)}
    hier = known[decide_within(fitted.test_per_llm_log_probs, rows, groups, column_llm[global_col])]
    scores = built["detector"].score(Z)
    flags = scores >= built["threshold"]
    pred = np.where(flags, UNSEEN, hier)
    truth = np.where(is_unseen, UNSEEN, labels)
    n_flag = int(flags.sum())
    return dict(n_t=n_t, accuracy=float((pred == truth).mean()), unseen_recall=float(flags[is_unseen].mean()) if is_unseen.any() else float("nan"),
                unseen_precision=float((flags & is_unseen).sum() / n_flag) if n_flag else float("nan"),
                known_accuracy_closed=float((hier[~is_unseen] == labels[~is_unseen]).mean()),
                accuracy_never_reject=float(((hier == labels) & ~is_unseen).mean()),
                threshold=built["threshold"], scores=scores, flags=flags, pred=pred, hierarchical_pred=hier, global_pred=known[global_col],
                labels=labels, is_unseen=is_unseen, batches=rows, statistics=Z, samples=built["samples"], oof_scores=built["oof_scores"],
                detector=built["detector"])


def evaluate_open_set_cell(X: np.ndarray, y: np.ndarray, splits, instance_names: Sequence[str], token_pair: str, split: int, draw: int,
                           batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64), target_recall: float = TARGET_RECALL,
                           **params) -> Tuple[OpenSetCell, Dict[int, Dict[str, object]]]:
    """One open-set cell (token pair, split, draw) at every ``N_t`` of ``batch_sizes``.

    ``X, y``: rows of the token pair (``protocol.select_rows``); ``splits``: ``protocol.make_splits``; ``params``: the
    keyword arguments of ``fit_open_set_cell`` (paper values by default). Returns (fitted cell, {N_t: ``evaluate_at`` dict})."""
    fitted = fit_open_set_cell(X, y, splits, instance_names, token_pair, split, draw, **params)
    return fitted, {int(n_t): evaluate_at(fitted, int(n_t), target_recall) for n_t in batch_sizes}

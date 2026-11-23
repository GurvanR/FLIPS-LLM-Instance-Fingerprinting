# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Hierarchical instance classifier.

An *LLM* is the base model of an instance, its quantized variants included (``model@@quant`` belongs to ``model``;
an abliterated model is its own LLM). The instance classifier is made of

1. one **global classifier** trained on all the instances (XGBoost, ``GLOBAL_ROUNDS`` frozen rounds), and
2. one **per-LLM classifier** for every LLM with at least 2 instances, trained on the training sequences of that LLM's
   instances only, with the same hyper-parameters and its own number of rounds (``xgb.select_rounds``).

For a verification sample of ``N_t`` answers, the global classifier's log-sum scores pick an instance; its LLM
``l*`` is the selected LLM. The per-LLM classifier of ``l*`` then decides: the predicted instance is the argmax, over
the instances of ``l*``, of the summed per-LLM log-probabilities (an LLM with a single instance gives that instance).

The module also gives the hierarchical score vector ``K^H`` of a sample (log-sum over the answers of
``log p_global(LLM of c) + log p_LLM(c)`` for the instances of ``l*``, global log-sum scores elsewhere) and the
confidence of the decision (its softmax mass under ``K^H``).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from audit_llm.Classification.log_sum_aggregation import LOG_FLOOR, log_sum_scores
from audit_llm.Classification.xgb import GLOBAL_ROUNDS, ROUND_SELECTION_MAX, ROUND_SELECTION_PATIENCE, fit_xgb, full_log_probs, log_probs, select_rounds

PER_LLM_STREAM = 54  # first key of the random stream of a per-LLM classifier's round selection


def llm_name(instance_name: str) -> str:
    """LLM of an instance named ``<model>[@@<quant>]_temp-<t>_sp-<sp>``."""
    return instance_name.split("_temp-")[0].split("@@")[0]


def llm_blocks(instance_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """LLM id of every instance (ids = position in the sorted list of LLM names) and the LLM names."""
    names, blocks = np.unique(np.array([llm_name(n) for n in instance_names]), return_inverse=True)
    return blocks, list(names)


def llm_groups(blocks: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(g): np.flatnonzero(blocks == g) for g in np.unique(blocks)}


def logsumexp(S: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(S, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m + np.log(np.exp(S - m).sum(axis=axis, keepdims=True)), axis)


def batch_sums(M: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """(n_batches, C) float64 sums of the rows of every batch."""
    if len(batches) == 0:
        return np.zeros((0, M.shape[1]))
    return M[batches].astype(np.float64).sum(1)


def llm_log_mass(L: np.ndarray, blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Global-classifier log-probability of every LLM (log-sum-exp over its instances, floored): (n, G) and, per instance column, that of its LLM (n, C)."""
    Lf = np.maximum(np.asarray(L, np.float64), LOG_FLOOR)
    G = int(blocks.max()) + 1
    LG = np.empty((len(Lf), G))
    for g in range(G):
        LG[:, g] = np.maximum(logsumexp(Lf[:, blocks == g]), LOG_FLOOR)
    return LG, LG[:, blocks]


def decide_within(L2: np.ndarray, batches: np.ndarray, groups: Dict[int, np.ndarray], selected: np.ndarray) -> np.ndarray:
    """Per batch, the instance of LLM ``selected[b]`` with the largest summed per-LLM log-probability (one-instance LLM: that instance)."""
    out = np.empty(len(batches), int)
    for g in np.unique(selected):
        idx = np.flatnonzero(selected == g)
        cols = groups[int(g)]
        if len(cols) == 1:
            out[idx] = cols[0]
            continue
        S2 = L2[:, cols][batches[idx]].sum(1)
        if not np.isfinite(S2).all():
            raise ValueError("per-LLM score missing for a routed sample")
        out[idx] = cols[S2.argmax(1)]
    return out


def softmax_mass(S: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Softmax mass of class ``pred`` under the scores ``S`` (n, C)."""
    ix = np.arange(len(S))
    return np.exp(S[ix, pred] - logsumexp(S))


class HierarchicalClassifier:
    """Global + per-LLM XGBoost classifiers over ``n_classes`` instances.

    Args:
        instance_names: names of all the instances (defines the LLM of every class index).
        global_rounds: boosting rounds of the global classifier.
        n_jobs: threads of the global classifier (the per-LLM classifiers use 1 thread). XGBoost ``hist`` results depend
            on the thread count; the default is 4.
    """

    def __init__(self, instance_names: Sequence[str], global_rounds: int = GLOBAL_ROUNDS, n_jobs: int = 4,
                 round_patience: int = ROUND_SELECTION_PATIENCE, round_max: int = ROUND_SELECTION_MAX):
        self.instance_names = list(instance_names)
        self.n_classes = len(self.instance_names)
        self.blocks, self.llm_names = llm_blocks(self.instance_names)
        self.groups = llm_groups(self.blocks)
        self.global_rounds, self.n_jobs = global_rounds, n_jobs
        self.round_patience, self.round_max = round_patience, round_max

    def fit_global(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalClassifier":
        self.global_model, self.global_classes = fit_xgb(X, y, self.global_rounds, self.n_jobs)
        return self

    def fit_per_llm(self, X: np.ndarray, y: np.ndarray, stream_key: Sequence[int]) -> "HierarchicalClassifier":
        """One classifier per LLM with >= 2 instances among ``y``; its round selection uses ``default_rng([*stream_key, LLM id])``."""
        self.per_llm = {}
        for g, cols in self.groups.items():
            cols = cols[np.isin(cols, y)]
            if len(cols) < 2:
                continue
            rows = np.flatnonzero(np.isin(y, cols))
            rng = np.random.default_rng([*stream_key, g])
            r_a, r_b, rounds = select_rounds(X[rows], y[rows], rng, self.round_patience, self.round_max)
            model, classes = fit_xgb(X[rows], y[rows], rounds, 1)
            self.per_llm[g] = dict(model=model, classes=classes, rounds=rounds, rounds_halves=(r_a, r_b), n_rows=len(rows))
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, stream_key: Sequence[int]) -> "HierarchicalClassifier":
        return self.fit_global(X, y).fit_per_llm(X, y, stream_key)

    def global_log_probs(self, X: np.ndarray) -> np.ndarray:
        """(n, C) float32 log-probabilities of the global classifier, floored at -80."""
        return full_log_probs(self.global_model, self.global_classes, X, self.n_classes)

    def per_llm_log_probs(self, X: np.ndarray) -> np.ndarray:
        """(n, C) float64: column c = log-probability of instance c under its LLM's classifier (floored at -80); 0 for one-instance LLMs."""
        out = np.zeros((len(X), self.n_classes))
        for g, m in self.per_llm.items():
            out[:, m["classes"]] = np.maximum(log_probs(m["model"], X), LOG_FLOOR)
        return out

    def decide(self, L: np.ndarray, L2: np.ndarray, batches: np.ndarray) -> Dict[str, np.ndarray]:
        """Decisions for the samples ``batches`` (rows into L / L2). Returns global prediction ``global_pred``, selected LLM
        ``llm``, hierarchical prediction ``pred``, its confidence ``confidence`` and the scores ``scores`` (K^H)."""
        flat = log_sum_scores(L, batches).argmax(1)
        llm = self.blocks[flat]
        pred = decide_within(L2, batches, self.groups, llm)
        _, LGc = llm_log_mass(L, self.blocks)
        A = LGc + np.asarray(L2, np.float64)
        inside = self.blocks[None, :] == llm[:, None]
        KH = np.where(inside, batch_sums(A, batches), batch_sums(L, batches))
        return dict(global_pred=flat, llm=llm, pred=pred, confidence=softmax_mass(KH, pred), scores=KH)

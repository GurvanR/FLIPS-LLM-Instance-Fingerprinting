# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Feature grid: generated answers -> one feature tensor (57 features) per token pair.

An *instance* (class) is a (model, temperature, system prompt) triple of the answers table; its name is
``<model>_temp-<temperature>_sp-<system prompt index>`` and the classes are sorted by (model, temperature, system prompt).
For a token pair, row ``j`` of a class is the ``j``-th answer of that class by query index. Each answer is described by
the 57 features: the columns of ``features.nist`` followed by those of ``features.statistics``.
Answers shorter than ``min_seq_length`` bits (100 in the paper; empty answers included) get NaN in every column.

On disk a grid is a directory with ``meta.json`` and, per token pair, ``<pair>/{features,bits,lens,query_index}.npy``
(memory-mapped when loaded, so the full corpus does not need to fit in memory).
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from audit_llm.features.nist import nist_column_names, nist_columns, nist_feature_names
from audit_llm.features.statistics import STATISTIC_NAMES, sequence_statistics

MIN_SEQ_LENGTH = 100
PAD = 2  # value of the bit tensor past the end of a sequence
CLASS_KEYS = ("Model", "temperature", "system_prompt_idx")


def feature_names() -> List[str]:
    """The 57 feature names, in tensor order."""
    return nist_feature_names() + list(STATISTIC_NAMES)


def class_name(model: str, temperature, system_prompt_idx) -> str:
    return f"{model}_temp-{temperature}_sp-{system_prompt_idx}"


@dataclass
class TokenPairGrid:
    """Grid of one token pair. ``features`` (S, C, 57) float64; ``bits`` (S, C, L) uint8 (0/1, PAD past the end);
    ``lens`` (S, C) bit-sequence lengths (0 = no answer); ``query_index`` (S, C) query index of each answer."""

    features: np.ndarray
    bits: np.ndarray
    lens: np.ndarray
    query_index: np.ndarray


class Grid:
    """All token pairs of a corpus, with the shared class order."""

    def __init__(self, class_names: Sequence[str], token_pairs: Sequence[str], pairs: Optional[Dict[str, TokenPairGrid]] = None,
                 root: Optional[str] = None, min_seq_length: int = MIN_SEQ_LENGTH):
        self.class_names = list(class_names)
        self.token_pairs = list(token_pairs)
        self.feature_names = feature_names()
        self.min_seq_length = min_seq_length
        self.root = root
        self._pairs = dict(pairs or {})

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def pair(self, token_pair: str) -> TokenPairGrid:
        if token_pair not in self._pairs:
            if self.root is None:
                raise KeyError(token_pair)
            d = os.path.join(self.root, token_pair)
            self._pairs[token_pair] = TokenPairGrid(
                features=np.load(os.path.join(d, "features.npy"), mmap_mode="r"),
                bits=np.load(os.path.join(d, "bits.npy"), mmap_mode="r"),
                lens=np.load(os.path.join(d, "lens.npy")),
                query_index=np.load(os.path.join(d, "query_index.npy")),
            )
        return self._pairs[token_pair]

    def select_classes(self, class_names: Sequence[str]) -> "Grid":
        """In-memory grid restricted to ``class_names`` (kept in this grid's class order)."""
        missing = [n for n in class_names if n not in self.class_names]
        if missing:
            raise KeyError(f"instances absent from the grid: {missing[:5]}")
        keep = [i for i, n in enumerate(self.class_names) if n in set(class_names)]
        pairs = {}
        for tp in self.token_pairs:
            g = self.pair(tp)
            pairs[tp] = TokenPairGrid(*(np.asarray(getattr(g, k))[:, keep] for k in ("features", "bits", "lens", "query_index")))
        return Grid([self.class_names[i] for i in keep], self.token_pairs, pairs, min_seq_length=self.min_seq_length)

    def save(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        for tp in self.token_pairs:
            g = self.pair(tp)
            d = os.path.join(root, tp)
            os.makedirs(d, exist_ok=True)
            for key in ("features", "bits", "lens", "query_index"):
                np.save(os.path.join(d, f"{key}.npy"), np.asarray(getattr(g, key)))
        meta = dict(class_names=self.class_names, token_pairs=self.token_pairs, feature_names=self.feature_names,
                    min_seq_length=self.min_seq_length, pad_value=PAD)
        with open(os.path.join(root, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        self.root = root

    @classmethod
    def load(cls, root: str) -> "Grid":
        with open(os.path.join(root, "meta.json")) as f:
            meta = json.load(f)
        if meta["feature_names"] != feature_names():
            raise ValueError(f"{root}: feature columns differ from the 57 features")
        return cls(meta["class_names"], meta["token_pairs"], root=root, min_seq_length=meta["min_seq_length"])


def _bits_tensor(answers: List[str], rows: np.ndarray, classes: np.ndarray, shape) -> tuple:
    S, C = shape
    max_len = max(1, max((len(s) for s in answers), default=1))
    bits = np.full((S, C, max_len), PAD, np.uint8)
    lens = np.zeros((S, C), np.int16)
    for r, s in enumerate(answers):
        n = len(s)
        lens[rows[r], classes[r]] = n
        if n:
            bits[rows[r], classes[r], :n] = np.frombuffer(s.encode("ascii"), np.uint8) - 48
    if bits.max() > PAD:
        raise ValueError("answers must be strings of '0' and '1'")
    return bits, lens


def build_grid(answers_df, token_pairs: Optional[Sequence[str]] = None, min_seq_length: int = MIN_SEQ_LENGTH, n_jobs: int = 1) -> Grid:
    """Compute the feature grid of an answers table.

    Args:
        answers_df: polars (or pandas) DataFrame with columns ``Token_pair``, ``Model``, ``Dataset_Question Index``, ``Answer``,
            ``temperature``, ``system_prompt_idx`` (a missing or null answer counts as an empty sequence).
        token_pairs: pairs to compute (default: all, sorted).
        min_seq_length: answers shorter than this get NaN features.
        n_jobs: worker processes for the NIST tests.
    """
    import polars as pl

    df = answers_df if isinstance(answers_df, pl.DataFrame) else pl.from_pandas(answers_df)
    df = df.select("Token_pair", *CLASS_KEYS, "Dataset_Question Index", "Answer")
    keys = df.select(*CLASS_KEYS).unique().sort(list(CLASS_KEYS)).rows()
    names = [class_name(*k) for k in keys]
    key_index = {k: i for i, k in enumerate(keys)}
    all_pairs = sorted(df["Token_pair"].unique().to_list())
    token_pairs = list(token_pairs) if token_pairs is not None else all_pairs
    missing = [t for t in token_pairs if t not in all_pairs]
    if missing:
        raise KeyError(f"token pairs absent from the answers: {missing}")
    nist_all = nist_column_names()
    nist_keep = [nist_all.index(n) for n in nist_feature_names()]
    S = int(df.group_by("Token_pair", *CLASS_KEYS).len()["len"].max())
    pairs = {}
    for tp in token_pairs:
        sub = df.filter(pl.col("Token_pair") == tp).sort([*CLASS_KEYS, "Dataset_Question Index"])
        classes = np.array([key_index[k] for k in sub.select(*CLASS_KEYS).rows()], dtype=int)
        query = sub["Dataset_Question Index"].to_numpy()
        answers = [s if isinstance(s, str) else "" for s in sub["Answer"].to_list()]
        rows = np.zeros(len(classes), int)
        for c in range(len(names)):
            pos = np.flatnonzero(classes == c)
            if len(np.unique(query[pos])) != len(pos):
                raise ValueError(f"{tp} / {names[c]}: duplicate query index")
            rows[pos] = np.arange(len(pos))
        nist = nist_columns(answers, n_jobs)[:, nist_keep]
        bits, lens = _bits_tensor(answers, rows, classes, (S, len(names)))
        features = np.full((S, len(names), 57), np.nan)
        features[rows, classes, :29] = nist
        features[:, :, 29:] = sequence_statistics(bits, lens)
        features[lens < min_seq_length] = np.nan
        query_index = np.full((S, len(names)), -1, np.int32)
        query_index[rows, classes] = query
        pairs[tp] = TokenPairGrid(features=features, bits=bits, lens=lens, query_index=query_index)
    return Grid(names, token_pairs, pairs, min_seq_length=min_seq_length)

"""nist_split.py - per-pair NIST scores and per-instance recall, the two arrays of the token-pair 0-1 analysis (figures/zero_one_vs_pairs.py).
NIST score of (pair, instance) = results/nist/nist_scores.json `score[pair][instance]` (per NIST test, share of the instance's sequences with p >= 0.05; mean over the 28 tests).
Recall of (pair, instance) at N_t = the hierarchical closed set's class_correct / class_batches, pooled over the splits of a pair (as dataio.class_recall_pairmean, before its mean over pairs)."""
import numpy as np
from ..dataio import R, jl, records

PAIR = "0-1"            # the token pair analysed apart from the others


def arrays(ctx, NT):
    cells = ctx.cells(); nist = jl(R("nist", "nist_scores.json"))
    idx = {n: i for i, n in enumerate(nist["class_names"])}
    ordr = [idx.get(n, -1) for n in ctx.names]
    # NIST per pair x instance
    S = {p: np.array([np.nan if (j < 0 or v[j] is None) else v[j] for j in ordr], float) for p, v in nist["score"].items()}
    # recall per pair x instance (pooled over splits of a pair, as dataio.class_recall_pairmean)
    k, n = {}, {}
    for c in records(cells, NT, ctx.nr):
        t = c["tp"]; k[t] = k.get(t, 0) + np.asarray(c["class_correct"], float); n[t] = n.get(t, 0) + np.asarray(c["class_batches"], float)
    Rc = {t: np.where(n[t] > 0, k[t] / np.maximum(n[t], 1), np.nan) for t in k}
    return S, Rc

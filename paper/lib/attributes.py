"""attributes.py - numbers of the attribute figure of the closed set (appendix; drawn by figures/attribute_errors.py).
The closed-set confusion at N_t (predictions.json, via dataio.post_confusion), errors only, summed over the instances that share a value of each attribute (base LLM, quantisation,
temperature, system prompt): the share of the errors that keep the true value of the attribute, and the same share for a uniformly drawn wrong instance. attr_data(ctx) -> D."""
import numpy as np
from ..dataio import ATTR, Skip, attr_matrix, load_post, post_confusion

TITLE = {"model": "base LLM", "quant": "quantization", "T": "temperature", "SP": "system prompt"}


def attr_data(ctx):
    cl, _ = load_post(ctx)
    if not cl: raise Skip("predictions.json missing")
    M = post_confusion(ctx, cl, ctx.nt); names = ctx.names; C = len(names); Me = M - np.diag(np.diag(M)); ne = Me.sum(1); w = ne / max(ne.sum(), 1)
    D = dict(n_batches=int(M.sum()), n_errors=int(Me.sum()), accuracy=float(np.trace(M) / M.sum()), C=C, attributes={})
    for k in ATTR:
        _, E, G = attr_matrix(Me, names, k); cnt = G.sum(0)
        D["attributes"][k] = dict(keep=float(np.trace(E) / E.sum()), keep_chance=float((w * ((G @ cnt) - 1) / (C - 1)).sum()))
    D["error_rate"] = 1 - D["accuracy"]
    return D

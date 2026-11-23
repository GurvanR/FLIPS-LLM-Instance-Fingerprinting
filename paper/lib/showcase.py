"""showcase.py - numbers of the abliterated showcase: how often the audits of an abliterated instance are attributed to its original LLM.
FLIPS = share of the audits (N_t-query batches) of an abliterated instance that the closed-set instance classifier attributes to ANY non-abliterated instance of the original LLM
(N_r ctx.nr, N_t ctx.nt, 31 pairs x 5 splits pooled; confusion of predictions.json -> dataio.post_confusion). The LLMmap values were measured with the pretrained LLMmap, taken off the
shelf, and are entered here as constants (LLMMAP); its raw outputs are not in the results archive. figures/showcase.py draws the figure from these numbers (closed/numbers.json["showcase"])."""
import numpy as np
from ..dataio import ABLIT_MAP, Skip, load_post, orig_name, post_confusion, r4

SHOW = [("failspy/Phi-3-medium-4k-instruct-abliterated-v3", "Phi-3-medium-4k"), ("failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3", "Smaug-Llama-3-70B"), ("natong19/Qwen2-7B-Instruct-abliterated", "Qwen2-7B")]
LLMMAP = [75.0, 92.5, 95.0]                 # pretrained LLMmap (off the shelf): % of the audits attributed to the original model, same order as SHOW; constants, not recomputed here


def showcase_numbers(ctx):
    cl, _ = load_post(ctx)
    if cl is None: raise Skip("predictions.json missing")
    M = post_confusion(ctx, cl, ctx.nt); names = ctx.names; orig = np.array([orig_name(n) for n in names]); ab = np.array(["abliterated" in n.lower() for n in names])
    fl, fs, n, src = [], [], [], {}
    for full, short in SHOW:
        i = next(k for k, nm in enumerate(names) if nm.startswith(full + "_") or nm == full); o = ABLIT_MAP[full]; row = M[i]
        tgt = np.flatnonzero((orig == o) & ~ab); assert len(tgt) > 0 and row.sum() > 0
        fl.append(100 * row[tgt].sum() / row.sum()); fs.append(100 * row[i] / row.sum()); n.append(int(row.sum()))
        src[short] = dict(instance=names[i], original_llm=o, n_original_instances=int(len(tgt)), batches=int(row.sum()), attributed_to_original=int(row[tgt].sum()), correct_abliterated=int(row[i]),
                          top_predicted=[(names[j], int(row[j])) for j in np.argsort(row)[::-1][:5] if row[j] > 0])
    ctx.num["showcase"] = dict(flips_attributed_to_original_pct={s[1]: r4(v) for s, v in zip(SHOW, fl)}, flips_recall_abliterated_pct={s[1]: r4(v) for s, v in zip(SHOW, fs)}, llmmap_pct={s[1]: v for s, v in zip(SHOW, LLMMAP)},
                               n_batches={s[1]: v for s, v in zip(SHOW, n)}, sources=src,
                               note=f"closed set, N_r {ctx.nr}, N_t {ctx.nt}, 31 pairs x 5 splits pooled (confusion row of the abliterated instance); attributed = predicted any non-abliterated instance of the original LLM (dataio.ABLIT_MAP); LLMmap values = constants measured with the pretrained LLMmap (off the shelf), not recomputed")
    print("[ok] showcase numbers")

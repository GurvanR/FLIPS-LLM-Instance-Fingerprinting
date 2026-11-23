"""closed_heatmaps.py - closed-set per-instance recall heatmaps of the paper, hierarchical closed-set classifier (155 cells = 31 pairs x 5 splits), N_r 300, N_t 32.
  fig07_closed_shortlist  : 14-LLM shortlist (dataio.SHORTLIST), layout LLM x variation, quantisations block, LLM / variation averages.
  fig12_closed_heatmap    : all 237 instances (LLM x variation layout, the abliterated column labelled on one line).
Per-instance recall = share of the instance's N_t-query verification batches identified as that instance, pooled over the 5 splits of a pair, then mean over the 31 pairs
(dataio.class_recall_pairmean). The all-cells pooled recall (sum over the 155 cells) is reported next to it in the json. Row 'LLM Average' / column
'Variation Average' = means over the instances drawn in the figure.
  python -m paper.figures.closed_heatmaps
Writes <out>/figures/fig07_closed_shortlist.{pdf,png,json}, <out>/figures/fig12_closed_heatmap.{pdf,png,json}."""
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from ..dataio import COL2, SHORTLIST, Ctx, class_counts, class_recall_pairmean, draw_heatmap, fig_dir, fmt_var, nxm_stats, orig_name, r4, short_model, style, var_name


def lab(v, two_lines):
    s = "abliterated" if v == "ablit" else fmt_var(v)
    return s if two_lines else s.replace("\n", " ")


def summarise(S, names, rec, pooled, keep):
    """numbers of one heatmap: row / column averages, grand, int4 / fp8 / temperature / sp pooled groups"""
    nm = [names[i] for i in keep]; r = rec[keep]; pl = pooled[keep]; var = [var_name(n) for n in nm]
    def grp(pred): ii = [k for k, v in enumerate(var) if pred(v)]; return dict(recall=r4(np.nanmean(r[ii])), recall_pooled_cells=r4(np.nanmean(pl[ii])), n_instances=len(ii))
    out = dict(n_instances=len(nm), n_llms=len(S["origs"]), grand=r4(S["grand"]), grand_pooled_cells=r4(np.nanmean(pl)),
               llm_average={o: r4(S["row"][o]) for o in S["origs"]}, variation_average={lab(v, False): r4(S["col"][v]) for v in S["vars"]},
               groups=dict(int4_all=grp(lambda v: v.startswith("bitsandbytes_int4")), fp8_all=grp(lambda v: v.startswith("fp8")),
                           unquantised_non_abliterated=grp(lambda v: v.startswith("temp-")), abliterated=grp(lambda v: v == "ablit")))
    rows = sorted([(o, S["row"][o]) for o in S["origs"] if np.isfinite(S["row"][o])], key=lambda t: t[1])
    out["llm_lowest"] = [rows[0][0], r4(rows[0][1])]; out["llm_highest"] = [rows[-1][0], r4(rows[-1][1])]
    out["llm_bottom3"] = [[o, r4(v)] for o, v in rows[:3]]; out["llm_top3"] = [[o, r4(v)] for o, v in rows[::-1][:3]]
    out["cells"] = {f"{o} | {lab(v, False)}": r4(S["mat"][(o, v)]) for o in S["origs"] for v in S["vars"] if np.isfinite(S["mat"][(o, v)])}
    return out


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args([] if argv is None else argv)
    style(); ctx = Ctx(fig_dir(), nt=32)
    cells = ctx.cells(); assert len(cells) == 155, len(cells)
    rec = class_recall_pairmean(cells, ctx.nt, ctx.nr, ctx.C); k, n = class_counts(cells, ctx.nt, ctx.nr, ctx.C); pooled = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    assert np.isfinite(rec).all() and (n > 0).all()
    src = dict(n_cells=len(cells), nt=ctx.nt, nr=ctx.nr, model="hierarchical closed set",
               recall="per instance: pooled over the 5 splits of a pair, mean over the 31 pairs (dataio.class_recall_pairmean); *_pooled_cells = pooled over all 155 cells",
               script="paper/figures/closed_heatmaps.py")
    names = ctx.names
    # ---- all 237 instances
    Sf = nxm_stats(names, rec)
    fig, ax = draw_heatmap(Sf, figsize=(COL2, 4.7))
    ax.set_xticklabels(["LLM Average"] + [lab(v, False) for v in Sf["vars"]], fontsize=5.5, rotation=45, ha="right"); ax.get_xticklabels()[0].set_weight("bold")
    fig.savefig(os.path.join(ctx.out, "fig12_closed_heatmap.pdf")); fig.savefig(os.path.join(ctx.out, "fig12_closed_heatmap.png"), dpi=150); plt.close(fig)
    nf = summarise(Sf, names, rec, pooled, list(range(ctx.C))); nf["source"] = src
    with open(os.path.join(ctx.out, "fig12_closed_heatmap.json"), "w") as f: json.dump(nf, f, indent=1)
    # ---- shortlist
    keep = [i for i, nm in enumerate(names) if orig_name(nm) in SHORTLIST]; nm = [names[i] for i in keep]; Ss = nxm_stats(nm, rec[keep])
    assert len(Ss["origs"]) == len(SHORTLIST), (len(Ss["origs"]), len(SHORTLIST))
    fig, ax = plt.subplots(figsize=(COL2, 3.3)); draw_heatmap(Ss, ax=ax, fs=6.2, cbar_label="Per-instance recall")
    ax.set_xticklabels(["LLM Average"] + [lab(v, False) for v in Ss["vars"]], fontsize=6, rotation=45, ha="right", rotation_mode="anchor"); ax.get_xticklabels()[0].set_weight("bold")
    ax.set_yticklabels([short_model(o) for o in Ss["origs"]] + ["Variation Average"], fontsize=6.5); ax.get_yticklabels()[-1].set_weight("bold")
    ax.tick_params(axis="x", pad=1)
    fig.tight_layout(); fig.savefig(os.path.join(ctx.out, "fig07_closed_shortlist.pdf")); fig.savefig(os.path.join(ctx.out, "fig07_closed_shortlist.png"), dpi=150); plt.close(fig)
    ns_ = summarise(Ss, names, rec, pooled, keep); ns_["shortlist"] = SHORTLIST; ns_["source"] = src
    ns_["full_237_for_reference"] = dict(grand=nf["grand"], variation_average=nf["variation_average"], groups=nf["groups"], llm_lowest=nf["llm_lowest"], llm_highest=nf["llm_highest"],
                                         llm_bottom3=nf["llm_bottom3"], llm_top3=nf["llm_top3"], note="all 237 instances / 25 LLM rows (abliterated checkpoints in their original LLM's row, dataio.ABLIT_MAP), = fig12_closed_heatmap")
    with open(os.path.join(ctx.out, "fig07_closed_shortlist.json"), "w") as f: json.dump(ns_, f, indent=1)
    print(json.dumps({k: v for k, v in ns_.items() if k not in ("cells", "source", "shortlist")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

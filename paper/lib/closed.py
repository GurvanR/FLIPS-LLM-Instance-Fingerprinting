"""closed.py - closed-set figures, tables and numbers of the paper: the classifier comparison, the grouped feature importance, the per-pair accuracy histogram,
the token-pair table and the closed-set calibration numbers.
Every function takes a Ctx (dataio.py), draws from stored results only, records its numbers in ctx.num[<id>], raises dataio.Skip if its inputs
are absent. N_r = ctx.nr (300 training sequences per instance) and N_t = ctx.nt (32) are arguments."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ..dataio import COL1, COL2, CONSTANTS, GRID, RED, Skip, acc_by_cell, fig_dir, load_post, pair_means, r4, sd, tex_table


def _need(cond, msg):
    if not cond: raise Skip(msg)


# ------------------------------------------------------------------------------------------------------------------------------------------
# per-pair accuracy histogram and the token-pair table
# ------------------------------------------------------------------------------------------------------------------------------------------
def _pair_acc(ctx):
    cells = ctx.cells(); _need(cells, "closed-set cells missing"); d = acc_by_cell(cells, ctx.nt, ctx.nr); _need(d, f"no N_t {ctx.nt} / N_r {ctx.nr} cells")
    return pair_means(d)


def fig15(ctx):
    """per-pair accuracy histogram, drawn into <out>/figures"""
    pm = _pair_acc(ctx); v = np.array(list(pm.values())); step = 0.02; lo = np.floor(v.min() / step) * step; hi = np.ceil(v.max() / step) * step + step
    edges = np.round(np.arange(lo, hi + 1e-9, step), 4); fig, ax = plt.subplots(figsize=(COL1, 2.2))
    ax.bar(edges[:-1], np.histogram(v, bins=edges)[0], width=step * 0.9, align="edge", alpha=0.75, edgecolor="black", color="red", lw=0.5)
    ax.set_xlabel(f"Accuracy per token pair ($N_t$={ctx.nt})"); ax.set_ylabel("Number of token pairs"); ax.xaxis.set_major_locator(plt.MaxNLocator(7)); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.grid(True, color="grey", ls="--", lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    o = sorted(pm.items(), key=lambda kv: -kv[1])
    ctx.num["fig15"] = dict(n_pairs=len(pm), mean=r4(v.mean()), std=r4(v.std()), min=r4(v.min()), max=r4(v.max()), top5=[(t, r4(x)) for t, x in o[:5]], bottom5=[(t, r4(x)) for t, x in o[-5:]],
                            note=f"N_r {ctx.nr}, N_t {ctx.nt}, mean over the splits of a pair")
    fig.tight_layout(); ctx.save(fig, "fig15_pair_accuracy_hist", fig_dir())


def table3(ctx):
    """top-5 / bottom-5 token pairs, written to <out>/figures/table3_pairs.tex"""
    pm = _pair_acc(ctx); o = sorted(pm.items(), key=lambda kv: -kv[1]); rows = [("Top-5", t, f"{100 * v:.2f}") for t, v in o[:5]] + [("Bottom-5", t, f"{100 * v:.2f}") for t, v in o[-5:]]
    tex = tex_table([(a, "\\texttt{" + b + "}", c) for a, b, c in rows], ["", "Token pair", f"Acc. at $N_t$={ctx.nt}"], f"Table 3 (tab:accuracy_tokens): N_r {ctx.nr}, N_t {ctx.nt}, 31 token pairs")
    open(os.path.join(fig_dir(), "table3_pairs.tex"), "w").write(tex)
    ctx.produced += ["table3_pairs.tex"]; ctx.num["table3"] = dict(top5=[(t, r4(v)) for t, v in o[:5]], bottom5=[(t, r4(v)) for t, v in o[-5:]]); print("[ok] table3_pairs.tex")


# ------------------------------------------------------------------------------------------------------------------------------------------
# closed-set precision / recall against the temperature-scaled confidence at N_t 32: numbers (calibration.json); figures/closed_pr.py draws the curve
# ------------------------------------------------------------------------------------------------------------------------------------------
def _pr_arrays(e):
    pr = e["pr"]; T = np.asarray(pr["confidence"], float); P = np.asarray(pr["prec"], float); Rc = np.asarray(pr["rec"], float)
    if P.ndim == 1: P, Rc = P[None], Rc[None]
    return T, P.mean(0), P.std(0), Rc.mean(0), Rc.std(0)


def calibration(ctx):
    _, cal = load_post(ctx); _need(cal, "calibration.json missing"); b = str(ctx.nt); _need(b in cal["res"], f"calibration.json has no N_t {ctx.nt}")
    e = cal["res"][b]; num = {b: dict(recall_at_p99=r4(e["recall_at_p99"]), recall_at_p90=r4(e["recall_at_p90"]))}
    fits = cal["fits"][b]                                                       # which data fitted the temperature of each cell: the inner validation split ("val") or leave-one-pair-out ("lopo")
    src = [v["source"] for sp in fits.values() for v in sp.values()]
    ctx.num["fig13"] = dict(by_nt=num, fit_source_cells={k: src.count(k) for k in sorted(set(src))},
                            note="temperature-scaled confidences on the argmax decision; temperature fitted on the inner validation split (leave-one-pair-out where it holds fewer than N_t rows per class); mean over pairs, band std")
    print("[ok] calibration numbers")


# ------------------------------------------------------------------------------------------------------------------------------------------
# grouped feature importance
# ------------------------------------------------------------------------------------------------------------------------------------------
GROUPS = {"NIST tests": "NIST statistic", "sequence length": "length", "dependence on past bits": "past dependence", "opening of the sequence": "opening bits"}
# the groups of constants/feature_labels.json (those of the paper's table of statistics) -> their short names in the legend and in the numbers


def feature_labels():
    """{feature: (label, short group name)} of the 57 features (constants/feature_labels.json)"""
    return {n: (lab, GROUPS[g]) for n, (lab, g) in json.load(open(os.path.join(CONSTANTS, "feature_labels.json"), encoding="utf-8")).items()}


def fig10(ctx):
    fs_ = sorted(glob.glob(ctx.p("importance", "*.json"))); _need(fs_, "importance/*.json missing")
    J = [json.load(open(f)) for f in fs_]; names = J[0]["features"]; ix = {n: i for i, n in enumerate(names)}; I = []
    for j in J:
        v = np.zeros(len(names))
        for n, g in j["gain"].items(): v[ix[n]] = g
        I.append(v / max(v.sum(), 1e-12))
    I = np.array(I); mean = I.mean(0); q25, q75 = np.percentile(I, [25, 75], axis=0); lab = feature_labels(); _need(set(names) == set(lab), "importance features differ from constants/feature_labels.json")
    GC = {"NIST statistic": "#1f77b4", "length": "#7f7f7f", "past dependence": "#ff7f0e", "opening bits": "#9467bd"}; grp = np.array([lab[n][1] for n in names]); groups = [g for g in GC if (grp == g).any()]
    TOP = 25; o = np.argsort(mean)[::-1][:TOP][::-1]
    fig, a1 = plt.subplots(figsize=(COL2 * 0.72, 3.5))                       # the per-group totals are in numbers.json
    a1.barh(range(TOP), mean[o], color=[GC[grp[i]] for i in o], height=0.75); a1.hlines(range(TOP), q25[o], q75[o], color="black", lw=0.6)
    a1.set_yticks(range(TOP)); a1.set_yticklabels([lab[names[i]][0][:42] for i in o], fontsize=5.2); a1.set_xlabel("Mean importance (normalised gain)"); a1.grid(axis="x", **{k: v for k, v in GRID.items() if k != "which"}); a1.set_ylim(-0.6, TOP - 0.4)
    a1.legend(handles=[Patch(color=GC[g], label=g) for g in groups], fontsize=6, loc="lower right", frameon=True)
    tot = {g: I[:, grp == g].sum(1) for g in groups}
    ctx.num["fig10"] = dict(top5=[(lab[names[i]][0], r4(mean[i])) for i in np.argsort(mean)[::-1][:5]], group_share={g: r4(tot[g].mean()) for g in groups}, n_features={g: int((grp == g).sum()) for g in groups},
                            n_cells=len(J), note=f"importance/*.json: 31 pairs x 5 splits = {len(J)} fits; XGBoost gain normalised per model, mean over the {len(J)} fits, IQR bars; features missing from a model's gain dict = 0")
    fig.tight_layout(); ctx.save(fig, "fig10_feature_importance")


# ------------------------------------------------------------------------------------------------------------------------------------------
# classifier comparison (the other classifiers' cells + the XGBoost bar from the XGBoost cells)
# ------------------------------------------------------------------------------------------------------------------------------------------
def fig09(ctx):
    cells = [json.load(open(f)) for f in sorted(glob.glob(ctx.p("classifier_cells", "*_s0.json")))]; _need(cells, "classifier_cells/*_s0.json missing")
    tps = [c["tp"] for c in cells]; xgb = ctx.cells(); _need(xgb, "XGBoost cells missing"); bsl = (ctx.nt, 8); st = {}
    for b in bsl:
        d = acc_by_cell(xgb, b, ctx.nr, tps=set(tps), splits={0}); _need(d, "the XGBoost cells have none of the classifier-comparison pairs at split 0")
        v = list(d.values()); st.setdefault("xgb", {})[b] = (float(np.mean(v)), sd(v), len(v))
    nm = {"xgb": "XGBoost"}
    for k in sorted({k for c in cells for k in c["res"]}):
        nm[k] = next((c["res"][k]["name"] for c in cells if k in c["res"] and "name" in c["res"][k]), k)
        for b in bsl:
            v = [c["res"][k][str(b)] for c in cells if k in c["res"] and str(b) in c["res"][k]]
            if v: st.setdefault(k, {})[b] = (float(np.mean(v)), sd(v), len(v))
    ks = sorted([k for k in st if all(b in st[k] and st[k][b][2] == len(tps) for b in bsl)], key=lambda k: -st[k][ctx.nt][0])      # a classifier missing in some pair is NOT shown as a partial mean
    fig, ax = plt.subplots(figsize=(COL1, 2.5)); x = np.arange(len(ks)); w = 0.38
    for off, b, col, lab in ((-w / 2, 8, "#f4b6b6", "$N_t$=8"), (w / 2, ctx.nt, RED, f"$N_t$={ctx.nt}")):
        bars = ax.bar(x + off, [st[k][b][0] for k in ks], w, yerr=[st[k][b][1] for k in ks], color=col, label=lab, capsize=1.5, error_kw=dict(lw=0.6))
        for bb, k in zip(bars, ks): ax.text(bb.get_x() + bb.get_width() / 2, st[k][b][0] + st[k][b][1] + 0.01, f"{100 * st[k][b][0]:.0f}", ha="center", va="bottom", fontsize=4.8)
    ax.axhline(1 / ctx.C, color="black", ls="--", lw=0.9, label=f"chance (1/{ctx.C})"); ax.set_xticks(x); ax.set_xticklabels([nm[k].replace("Logistic Regression", "Log. Regr.").replace("Gradient Boosting", "HistGB").replace("Random Forest", "Rand. Forest") for k in ks], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0); ax.legend(fontsize=6, loc="upper right"); ax.grid(axis="y", **{k_: v for k_, v in GRID.items() if k_ != "which"}); ax.set_axisbelow(True)
    ctx.num["fig09"] = dict(acc={nm[k]: {b: [r4(st[k][b][0]), r4(st[k][b][1]), st[k][b][2]] for b in bsl} for k in ks}, pairs=tps, missing=[nm[k] for k in st if k not in ks],
                            note=f"N_r {ctx.nr}, {len(tps)} token pairs x split 0, std across pairs; XGBoost bar = xgboost_cells of the same pairs (split 0), the others = classifier_cells")
    fig.tight_layout(); ctx.save(fig, "fig09_classifier_comparison")

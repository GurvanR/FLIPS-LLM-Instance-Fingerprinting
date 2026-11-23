"""open_heatmap.py - per-instance open-set result of the method at the pre-test 50 %-target cut, N_t 32, 155 cells (figA_open_heatmap).
Figure: Known recall per instance = share of its test batches (as a Known instance) that are accepted (not flagged Unseen) AND correctly identified (class decision == source), drawn on the
closed-set heatmap layout (LLM x variation); the pooled Unseen recall / precision at the 50 % target (from tab_openset_op.json) is printed under it.
Sidecar: also the Unseen flagged share per instance = share of its test batches (in the cells where it is Unseen) flagged Unseen (instances never Unseen in the 5 draws have no value).
Aggregation = the closed heatmap's (dataio.class_recall_pairmean): pooled over the cells of a token pair, then mean over the pairs where the instance has batches.
Input: results/open_set/per_instance/test_rows.npz (stored per-batch test scores + class decisions; nothing is refitted), <out>/tab_openset_op.json (run stats/openset_operating_point first).
  python -m paper.figures.open_heatmap [--out DIR]
Writes <out>/figA_open_heatmap.{pdf,png,json} (default <out> = the figures directory of the output root)."""
import argparse, copy, json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..dataio import R, COL2, SP_PAPER, Skip, attrs, canon_names, draw_heatmap, fig_dir, fmt_var, nxm_stats, r4, short_model, style


def pairmean(num, den, pair):
    """num / den per (pair, instance) pooled over the pair's cells, then nanmean over pairs (= dataio.class_recall_pairmean)."""
    P = np.unique(pair); rec = []
    for p in P:
        n, d = num[pair == p].sum(0), den[pair == p].sum(0); rec.append(np.where(d > 0, n / np.maximum(d, 1), np.nan))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(rec), 0)


def groups(names, vals):
    """per-variation averages (mean over instances): temp 0.4/0.6/0.8/1.0 (no sp, unquantised), sp 1-4 (temp 1.0, unquantised), int4, fp8 (all settings), abliterated."""
    out = {}; vals = np.asarray(vals, float)
    def m(ix): a = vals[ix]; a = a[np.isfinite(a)]; return (r4(a.mean()) if len(a) else None, int(len(a)))
    ab = np.array(["abliterated" in n.lower() for n in names]); A = [attrs(n) for n in names]; q = np.array([a[1] for a in A]); T = np.array([a[2] for a in A]); S = np.array([a[3] for a in A])
    for t in ("0.4", "0.6", "0.8", "1.0"): out[f"temp {t} (no sp)"] = m(np.flatnonzero(~ab & (q == "-") & (T == t) & (S == "-1")))
    for sp, lab in SP_PAPER.items(): out[f"sp {lab} (temp 1.0)"] = m(np.flatnonzero(~ab & (q == "-") & (S == sp)))
    out["int4 (all settings)"] = m(np.flatnonzero(q == "bitsandbytes_int4")); out["fp8 (all settings)"] = m(np.flatnonzero(q == "fp8")); out["abliterated"] = m(np.flatnonzero(ab))
    return {k: dict(mean=v[0], n_instances=v[1]) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None, help="output directory (default: <FLIPS_OUT>/figures)"); a = ap.parse_args(argv)
    OUT = a.out or fig_dir(); os.makedirs(OUT, exist_ok=True); DATA = R("open_set", "per_instance")
    style(); T = np.load(os.path.join(DATA, "test_rows.npz")); names_t = [str(x) for x in T["names"]]; names = canon_names(); C = len(names)
    assert set(names_t) == set(names) and C == 237; perm = np.array([names_t.index(n) for n in names])          # canon index -> test_rows index
    cells = [str(c) for c in T["cells"]]; pair_of_cell = np.array([c.split("_s")[0] for c in cells]); ncell = len(cells); assert ncell == 155
    s, y, ok, cell, src, pred = T["s"], T["y"].astype(int), T["ok"].astype(bool), T["cell"].astype(int), T["src"].astype(int), T["pred"].astype(int)
    f50 = s >= T["tau50"][cell]; K = y == 0; U = ~K
    assert np.array_equal(ok[K], (pred == src)[K]) and not ok[U].any()
    cnt = lambda m: np.vstack([np.bincount(src[(cell == i) & m], minlength=len(names_t)) for i in range(ncell)])[:, perm]       # (cells, canon instances)
    nK, kacc = cnt(K), cnt(K & ok & ~f50); nU, uf = cnt(U), cnt(U & f50)
    known_rec = pairmean(kacc, nK, pair_of_cell); unseen_fl = pairmean(uf, nU, pair_of_cell)
    SK = nxm_stats(names, known_rec); SU0 = nxm_stats(names, unseen_fl); SU = copy.deepcopy(SK)
    for k in ("mat", "row", "col"): SU[k] = SU0[k]
    SU["grand"] = SU0["grand"]
    # ---- figure: Known recall; the Unseen recall / precision of the pooled Unseen batches (N_t 32, 50 % target) is printed at the bottom
    tab_path = os.path.join(OUT, "tab_openset_op.json")
    if not os.path.exists(tab_path): raise Skip(f"{tab_path} is missing: run `python -m paper.stats.openset_operating_point` first")
    TAB = json.load(open(tab_path))["rows"]["32"]; rec_u, prec_u = TAB["c50_unseen_recall"]["est"], TAB["c50_unseen_precision"]["est"]
    fig, ax = draw_heatmap(SK, figsize=(COL2, 5.1), cbar_label="Known recall (accepted and correct)")
    fig.text(0.5, 0.012, f"Unseen samples (all instances pooled): recall {100 * rec_u:.1f} %, precision {100 * prec_u:.1f} %", ha="center", va="bottom", fontsize=7.5)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(OUT, "figA_open_heatmap.pdf")); fig.savefig(os.path.join(OUT, "figA_open_heatmap.png"), dpi=170); plt.close(fig)
    # ---- numbers
    nUinst = int(np.isfinite(unseen_fl).sum()); kb = int(nK.sum()); ub = int(nU.sum())
    cols = lambda S: {fmt_var(v).replace("\n", " "): r4(S["col"][v]) for v in S["vars"]}
    num = dict(nt=32, n_cells=ncell, n_pairs=int(len(set(pair_of_cell))), operating_point="pre-test 50 %-target cut tau50 per cell (logit), flag s >= tau",
               aggregation="per instance: pooled over the cells of a token pair, mean over pairs (= dataio.class_recall_pairmean, as the closed heatmap); averages = means over instances",
               known=dict(definition="share of the instance's Known test batches with flag == 0 AND class decision == source instance", grand_mean_over_instances=r4(SK["grand"]),
                          batch_pooled=r4(kacc.sum() / kb), n_instances=int(np.isfinite(known_rec).sum()), n_batches=kb, columns_heatmap=cols(SK), groups=groups(names, known_rec),
                          row_min=sorted([(short_model(o), r4(x)) for o, x in SK["row"].items()], key=lambda t: t[1])[:3], row_max=sorted([(short_model(o), r4(x)) for o, x in SK["row"].items()], key=lambda t: -t[1])[:3]),
               unseen=dict(definition="share of the instance's Unseen test batches with flag == 1", grand_mean_over_instances=r4(SU0["grand"]), batch_pooled=r4(uf.sum() / ub), n_instances=nUinst,
                           n_never_unseen=int(C - nUinst), n_batches=ub, columns_heatmap=cols(SU), groups=groups(names, unseen_fl),
                           n_cells_as_unseen=dict(min=int((nU > 0).sum(0)[np.isfinite(unseen_fl)].min()), max=int((nU > 0).sum(0).max()))),
               encoding="heatmap (figure): Known recall of all 237 instances on the LLM x variation layout of the closed heatmap; `unseen` (sidecar only): Unseen flagged share of the "
                        f"{nUinst} instances that are Unseen in at least one of the 5 draws (the 5 Unseen draws of 23 instances are the same in every pair); row / column / grand averages of `unseen` are over those instances only.",
               sources=dict(rows="results/open_set/per_instance/test_rows.npz: per-batch test scores s, labels y, flag ok (class decision correct), source instance src, class decision pred and the pre-test cut tau50 "
                                 "of every cell (155 cells, N_t 32); the stored counts of the 50 % cut reproduce the curve archive in every cell"),
               values=dict(names=names, known_recall=[r4(v) for v in known_rec], unseen_flagged=[r4(v) for v in unseen_fl], n_known_batches=nK.sum(0).astype(int).tolist(), n_unseen_batches=nU.sum(0).astype(int).tolist(), n_cells_unseen=(nU > 0).sum(0).astype(int).tolist()))
    path = os.path.join(OUT, "figA_open_heatmap.json"); json.dump(num, open(path, "w"), indent=1); print(f"[ok] {path}")
    print(json.dumps({k: {kk: vv for kk, vv in num[k].items() if kk not in ("columns_heatmap",)} for k in ("known", "unseen")}, indent=1))


if __name__ == "__main__":
    main()

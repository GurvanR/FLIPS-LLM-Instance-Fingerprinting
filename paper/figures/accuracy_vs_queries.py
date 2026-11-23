"""accuracy_vs_queries.py - accuracy against the number of queries N_t (fig05_accuracy_vs_nt).
  python -m paper.figures.accuracy_vs_queries [--out DIR]
Curves:
  closed set   hierarchical closed set, 155 cells (31 pairs x 5 splits), mean over pairs, band = population sd across the 31 pair means (results/closed_set/hierarchical/summary.json)
  open set     the method's open set at the 50 % Unseen-recall target (pre-test threshold), pooled 5-draw summary, 155 cells (31 pairs x 5 draws), mean over pairs
               (results/open_set/summary.json), band = population sd across the 31 pair means (percell.csv)
  LLMmap-IF    stored values at N_t 1, 2, 4, 8 (the points of the FLIPS N_t grid up to 8; stats/llmmap_curve.py has 1..8)
Writes <out>/fig05_accuracy_vs_nt.{pdf,png,json} (default <out> = the figures directory of the output root)."""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..dataio import R, RED, GREEN, PURPLE, XL_NT, GRID, data_root, fig_dir, style
from ..common import NTS, COL, save, jdump, llmmap, open_summary_path, open_percell_path, closed_summary_path
from ..common import open_summary, open_level, open_pair_sd, closed_stats


def rel(p): return os.path.relpath(p, data_root())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None, help="output directory (default: <FLIPS_OUT>/figures)"); a = ap.parse_args(argv)
    out = a.out or fig_dir(); os.makedirs(out, exist_ok=True)
    style(); S = open_summary(); C = closed_stats(); LL = llmmap()
    num = dict(sources=dict(closed=rel(closed_summary_path()), open=rel(open_summary_path()), open_sd=rel(open_percell_path()), llmmap=rel(R("llmmap", "results.json")) + " (LLMmap-IF, filtered mean)"),
               cells=dict(closed="hierarchical closed set: 31 pairs x 5 splits = 155 cells", open="open set of the method: 31 pairs x 5 draws = 155 cells"),
               aggregation="mean over the 31 token pairs of the per-pair mean over cells; sd = population sd (ddof 0) across the 31 pair means; ci95 = pair-clustered percentile bootstrap (B 2000, seed 20261031) from summary.json",
               band="sd across token pairs (closed and open 50 % target)", chance=1 / 237, per_nt={})
    for n in NTS:
        d = dict(closed_hier=dict(mean=C[n]["mean"], sd=C[n]["sd"], n_cells=155))
        e, lo, hi = open_level(S, n, "c50_acc"); mu, sd, npairs = open_pair_sd(n, "c50_acc"); assert abs(mu - e) < 1e-9 and npairs == 31, (n, mu, e)
        d["open_c50"] = dict(mean=e, ci95=[lo, hi], sd=sd, n_cells=155)
        for k, m in (("c50_unseen_recall", "c50_rec"), ("c50_unseen_precision", "c50_up"), ("auc_known_vs_unseen", "auc_all")):
            e, lo, hi = open_level(S, n, m); d[k] = dict(est=e, ci95=[lo, hi])
        if n in LL: d["llmmap"] = LL[n]
        num["per_nt"][n] = d
    num["llmmap"] = LL

    fig, ax = plt.subplots(figsize=(COL, 2.8)); x = NTS; P = num["per_nt"]
    ax.axhline(1 / 237, color="black", ls="--", lw=1.0, label="Random chance (1/237)")
    cm = np.array([P[n]["closed_hier"]["mean"] for n in x]); cs = np.array([P[n]["closed_hier"]["sd"] for n in x])
    ax.fill_between(x, cm - cs, cm + cs, color=RED, alpha=.15, lw=0); ax.plot(x, cm, marker="o", ms=3.8, color=RED, lw=1.5, label="FLIPS closed set", zorder=5)
    om = np.array([P[n]["open_c50"]["mean"] for n in x]); osd = np.array([P[n]["open_c50"]["sd"] for n in x])
    ax.fill_between(x, om - osd, om + osd, color=GREEN, alpha=.18, lw=0); ax.plot(x, om, marker="s", ms=3.6, color=GREEN, lw=1.5, label="FLIPS open set, 50 % Unseen-recall target", zorder=5)
    xl = [n for n in sorted(LL) if n in NTS]; ax.plot(xl, [LL[b] for b in xl], marker="o", ms=3.2, color=PURPLE, lw=1.3, label="LLMmap-IF, closed set")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([str(n) for n in x]); ax.minorticks_off(); ax.set_xlim(.85, 75)
    ax.set_xlabel(XL_NT); ax.set_ylabel("Accuracy"); ax.set_ylim(-.02, 1.0); ax.set_yticks(np.arange(0, 1.01, .2)); ax.grid(**GRID)
    h, l = ax.get_legend_handles_labels(); order = [1, 2, 3, 0]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=1, fontsize=6.3, frameon=False, handlelength=2.6, labelspacing=.25)
    fig.tight_layout(); save(fig, "fig05_accuracy_vs_nt", out); plt.close(fig)
    jdump(num, os.path.join(out, "fig05_accuracy_vs_nt.json"))
    for n in NTS:
        p = P[n]; print(f"N_t {n:2d}: closed {100*p['closed_hier']['mean']:.1f}+-{100*p['closed_hier']['sd']:.1f}  c50 {100*p['open_c50']['mean']:.1f}+-{100*p['open_c50']['sd']:.1f}"
                        f"  llmmap {p.get('llmmap', '-')}")


if __name__ == "__main__":
    main()

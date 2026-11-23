"""unseen_pr.py - Unseen precision against Unseen recall of the method (open set; fig06_unseen_pr), N_t 32.
  python -m paper.figures.unseen_pr [--out DIR]
Pooled 5-draw open set (results/open_set/curves.tar.gz), 155 cells = 31 pairs x 5 draws, pooled by pair: the PRE-TEST calibrated sweep (threshold
fixed on the training pool at a nominal Unseen-recall target .05 ... .95; each point placed at the recall and precision realised on the test batches); the 50 % target is marked
(diamond); AUC of P(Unseen) on the binary Known / Unseen task (summary.json auc_all, pooled, pair-bootstrap 95 % CI) annotated. The Unseen prevalence among the test batches is drawn as the
precision of a random flagger. Files: fig06_unseen_pr.{pdf,png,json}."""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..dataio import ORANGE, GREY, GRID, data_root, fig_dir, style
from ..common import COL, TARGETS, I50, save, jdump, load_curves, sweep, up_curve, prevalence, open_curves_path
from ..common import open_summary, open_level

NT = 32
LAB = (0, 3, 13, 18)                         # nominal targets annotated on the curve: .05, .20, .70, .95 (the .50 point is the diamond)


def rel(p): return os.path.relpath(p, data_root())


def auc_text(S, nt):
    e, lo, hi = open_level(S, nt, "auc_all"); return e, lo, hi, f"AUC Known vs Unseen\n{e:.3f} [{lo:.3f}, {hi:.3f}]"


def frame(ax, xmax):
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.0); ax.grid(**GRID); ax.set_ylabel("Unseen precision")


def main_fig(S, out, nt=NT):
    cells = load_curves(nt); Q = sweep(cells); assert np.allclose(Q["recall"].shape[1], len(TARGETS))
    x, y, ndef = up_curve(cells, Q); prev = prevalence(cells)
    e_rec, e_up = open_level(S, nt, "c50_rec"), open_level(S, nt, "c50_up")
    assert abs(x[I50] - e_rec[0]) < 1e-9 and abs(y[I50] - e_up[0]) < 1e-9, (nt, x[I50], e_rec, y[I50], e_up)      # curve pooling == summary.json
    auc, alo, ahi, txt = auc_text(S, nt)
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ax.axhline(prev, color=GREY, lw=.8, ls="--", zorder=1); ax.text(.01, prev - .015, f"Unseen prevalence {100 * prev:.1f} % (random flagging)", transform=ax.get_yaxis_transform(), ha="left", va="top", fontsize=6, color=GREY)
    ax.plot(x, y, color=ORANGE, lw=1.6, marker="o", ms=2.4, zorder=4, label="pre-test threshold")
    for j in LAB:
        if np.isfinite(y[j]): ax.annotate(f"{TARGETS[j]:.2f}".lstrip("0"), (x[j], y[j]), textcoords="offset points", xytext=(3, 3) if j < 13 else (-3, 5), ha="left" if j < 13 else "center", fontsize=5.8, color=ORANGE)
    ax.plot([x[I50]], [y[I50]], "D", color=ORANGE, mec="k", mew=.7, ms=6, zorder=7, label="50 % recall target")
    ax.annotate(f"50 % target: recall {100 * x[I50]:.1f} %,\nprecision {100 * y[I50]:.1f} %", (x[I50], y[I50]), textcoords="offset points", xytext=(6, 8), fontsize=6, color="k")
    ax.text(.98, .97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, bbox=dict(boxstyle="round,pad=.3", fc="white", ec="0.7", lw=.5))
    frame(ax, 1.0); ax.set_xlabel(f"Unseen recall on the test samples ($N_t$={nt})"); ax.legend(loc="upper right", bbox_to_anchor=(1, .8), fontsize=6, frameon=False)
    fig.tight_layout(); name = "fig06_unseen_pr"; save(fig, name, out); plt.close(fig)
    num = dict(nt=nt, cells="open set of the method, 31 pairs x 5 draws = 155 cells", source=rel(open_curves_path()) + f" (nt{nt}/*.json, test.sweep)",
               aggregation="per-cell Unseen precision cu/(cu+kr) (undefined when nothing flagged, left out) and realised recall -> per-pair mean -> mean over pairs; point dropped if < 50 % of pairs define the precision",
               curve=dict(nominal_target=TARGETS, realised_recall=x, unseen_precision=y, share_pairs_defined=ndef),
               c50=dict(nominal_target=.5, realised_recall=open_level(S, nt, "c50_rec"), unseen_precision=open_level(S, nt, "c50_up"), accuracy=open_level(S, nt, "c50_acc")),
               auc_known_vs_unseen=[auc, alo, ahi], unseen_prevalence=prev,
               note="(est, lo, hi) triples: summary.json, pair-clustered percentile bootstrap 95 % CI (B 2000, seed 20261031)")
    jdump(num, os.path.join(out, name + ".json")); print(f"{name}: c50 recall {x[I50]:.4f} precision {y[I50]:.4f}  AUC {auc:.4f} [{alo:.4f}, {ahi:.4f}]  prevalence {prev:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None, help="output directory (default: <FLIPS_OUT>/figures)"); a = ap.parse_args(argv)
    out = a.out or fig_dir(); os.makedirs(out, exist_ok=True)
    style(); main_fig(open_summary(), out)


if __name__ == "__main__":
    main()

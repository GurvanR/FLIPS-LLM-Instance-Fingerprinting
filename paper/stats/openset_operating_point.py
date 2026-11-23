"""openset_operating_point.py - open-set results of the method at its operating point (tab_openset_op.json).
  python -m paper.stats.openset_operating_point [--out DIR]
Rows N_t 1..64, open set of the method (hierarchical class decision + Unseen detector), pooled 5-draw summary: 155 cells = 31 pairs x 5 draws (mean over pairs). Numbers straight from
results/open_set/summary.json (pair-clustered percentile 95 % CI, B 2000, seed 20261031): accuracy, Unseen recall and Unseen precision at the 50 % recall target, AUC of the Unseen probability.
Writes <out>/tab_openset_op.json (default <out> = the figures directory of the output root); read by the numbers step and by figures/open_heatmap.py."""
import argparse, os
from ..dataio import data_root, fig_dir
from ..common import NTS, jdump, open_summary_path
from ..common import open_summary, open_level

METRICS = dict(c50_acc="c50_acc", c50_unseen_recall="c50_rec", c50_unseen_precision="c50_up", auc="auc_all")


def rows(S):
    out = {}
    for n in NTS:
        g = S["table"][str(n)]; d = dict(n_pairs=g["n_pairs"], n_cells=g["n_cells"])
        for k, m in METRICS.items():
            e, lo, hi = open_level(S, n, m); d[k] = dict(est=e, lo=lo, hi=hi)
        out[n] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None, help="output directory (default: <FLIPS_OUT>/figures)"); a = ap.parse_args(argv)
    out = a.out or fig_dir(); os.makedirs(out, exist_ok=True)
    S = open_summary()
    num = dict(source=os.path.relpath(open_summary_path(), data_root()), method="hierarchical class decision + Unseen detector", cells="31 pairs x 5 draws = 155 cells, mean over the pairs",
               units="fractions", ci="pair-clustered percentile bootstrap 95 % (B 2000, seed 20261031)",
               definitions=dict(c50="threshold of the pre-test sweep at a nominal 50 % Unseen-recall target (index 9)", auc="AUC of P(Unseen), all Known test batches vs Unseen test batches"),
               rows=rows(S))
    path = os.path.join(out, "tab_openset_op.json"); jdump(num, path); print(f"[ok] {path}")


if __name__ == "__main__":
    main()

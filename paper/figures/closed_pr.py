"""closed_pr.py - closed-set precision/recall figure of the paper (tempered confidence, N_t 32).
One precision curve and one recall curve against the confidence threshold (temperature-scaled posteriors of calibration.json), mean over the token pairs, band = std across pairs.
Guard: recall at precision 0.99 / 0.9 must equal the values stored in closed/numbers.json (fig13, N_t 32), else the step stops.
  python -m paper.figures.closed_pr
Writes <out>/figures/fig13_closed_pr.{pdf,png}."""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from ..dataio import BLUE, COL1, GRID, Ctx, closed_dir, fig_dir, load_post
from ..lib import closed as FC

NT = 32


def main(argv=None):
    ap = argparse.ArgumentParser(description="closed-set precision/recall figure (fig13_closed_pr.pdf/.png)")
    ap.add_argument("--data", help="data root (default: $FLIPS_DATA)"); ap.add_argument("--out", help="output root (default: $FLIPS_OUT)")
    a = ap.parse_args(argv)
    if a.data: os.environ["FLIPS_DATA"] = a.data
    if a.out: os.environ["FLIPS_OUT"] = a.out
    out = fig_dir(); ctx = Ctx(out, nr=300, nt=NT); _, cal = load_post(ctx); e = cal["res"][str(NT)]
    with open(os.path.join(closed_dir(), "numbers.json")) as f: stored = json.load(f)["fig13"]["by_nt"][str(NT)]
    assert abs(e["recall_at_p99"] - stored["recall_at_p99"]) < 1e-4 and abs(e["recall_at_p90"] - stored["recall_at_p90"]) < 1e-4, "calibration.json does not match closed/numbers.json"
    T, mp, sp, mr, sr = FC._pr_arrays(e)
    with plt.rc_context():                                                       # matplotlib defaults (no shared style); rcParams restored afterwards
        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=(COL1 * 1.15, 2.3))
        ax.plot(T, mp, color=BLUE, ls="--", label="precision"); ax.plot(T, mr, color=BLUE, ls="-", label="recall")
        ax.fill_between(T, np.clip(mp - sp, 0, 1), np.clip(mp + sp, 0, 1), color=BLUE, alpha=0.08, lw=0); ax.fill_between(T, np.clip(mr - sr, 0, 1), np.clip(mr + sr, 0, 1), color=BLUE, alpha=0.08, lw=0)
        for tgt, c in ((0.99, "lightsteelblue"), (0.9, "cornflowerblue")):
            m = mp >= tgt
            if m.any(): i = int(m.argmax()); ax.axvline(T[i], color=c, lw=1.0, label=f"recall at precision {tgt} = {mr[i]:.0%}"); print(tgt, round(float(T[i]), 3), round(float(mr[i]), 4))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_xlabel("Confidence threshold", fontsize=7); ax.set_ylabel("Precision / recall", fontsize=7); ax.tick_params(labelsize=6); ax.grid(**GRID)
        ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9); fig.tight_layout()
        for ext in ("pdf", "png"): fig.savefig(os.path.join(out, f"fig13_closed_pr.{ext}"), dpi=200)
    plt.close(fig)
    print("stored", stored["recall_at_p99"], stored["recall_at_p90"])


if __name__ == "__main__":
    main()

"""detector_scores.py - Unseen probability q of the Unseen detector on the test samples, N_t 32, all token pairs and draws pooled (155 cells = 31 pairs x 5 draws; figA_detector_scores).
Known samples (identified correctly or not, together) against Unseen samples; vertical line: the threshold tau (mean over the 155 cells).
Input: results/open_set/per_instance/test_rows.npz (stored per-sample test scores, labels and the threshold of each cell; nothing is refitted here).
  python -m paper.figures.detector_scores [--out DIR]
Writes <out>/figA_detector_scores.{pdf,png,json} (default <out> = the figures directory of the output root)."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..dataio import R, GRID, fig_dir, r4, style

XL, XR, BW = -7.0, 6.0, 0.25                                       # plotted logit range and bin width; scores outside the range are counted in the edge bins
QS = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95]                             # tick labels: values of q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None, help="output directory (default: <FLIPS_OUT>/figures)"); a = ap.parse_args(argv)
    OUT = a.out or fig_dir(); os.makedirs(OUT, exist_ok=True)
    style(); T = np.load(os.path.join(R("open_set", "per_instance"), "test_rows.npz"))
    s, y, cell, tau = T["s"], T["y"].astype(int), T["cell"].astype(int), T["tau50"]
    assert len(tau) == 155, len(tau)
    flagged = s >= tau[cell]; edges = np.round(np.arange(XL, XR + 1e-9, BW), 4)
    fig, ax = plt.subplots(figsize=(3.3, 2.3)); counts, outside = {}, {}
    for lab, v, col in (("Known samples", s[y == 0], "tab:blue"), ("Unseen samples", s[y == 1], "tab:red")):
        cnt = np.histogram(np.clip(v, XL, XR - 1e-9), edges)[0]; d = cnt / (len(v) * BW)
        counts[lab] = cnt.tolist(); outside[lab] = dict(below=int((v < XL).sum()), above=int((v >= XR).sum()))
        ax.stairs(d, edges, color=col, lw=1.1, fill=True, alpha=0.30); ax.stairs(d, edges, color=col, lw=1.1, label=lab)
    ax.axvline(tau.mean(), color="black", lw=1.0, label=r"threshold $\tau$")
    ax.set_xticks([np.log(q / (1 - q)) for q in QS]); ax.set_xticklabels([f"{q:g}" for q in QS])
    ax.set_xlabel(r"Unseen probability $q$ (logit scale)", fontsize=8); ax.set_ylabel("density", fontsize=8); ax.grid(**GRID); ax.set_xlim(XL, XR); ax.set_ylim(0, None); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figA_detector_scores.pdf")); fig.savefig(os.path.join(OUT, "figA_detector_scores.png"), dpi=200); plt.close(fig)
    num = dict(nt=32, n_cells=len(tau), n_known=int((y == 0).sum()), n_unseen=int((y == 1).sum()),
               unseen_above_tau=r4(flagged[y == 1].mean()), known_above_tau=r4(flagged[y == 0].mean()),
               tau_logit_mean=r4(tau.mean()), tau_q_at_mean=r4(1 / (1 + np.exp(-tau.mean()))),
               plot=dict(x="logit of q (the stored detector score is this logit; q = sigmoid)", range=[XL, XR], bin_width=BW, edges=edges.tolist(),
                         density="per group, counts / (n x bin width), each sample with unit weight, pooled over the 155 cells; scores outside the range are counted in the edge bins"),
               counts=counts, outside_plot_range=outside,
               sources=dict(curves="per-sample test scores s and labels y (0 Known, 1 Unseen) of the method, results/open_set/per_instance/test_rows.npz",
                            tau="threshold of each cell (tau50 in test_rows.npz: median of the out-of-fold q of the pseudo-Unseen samples, target recall 50 %); line = mean over the 155 cells",
                            above_tau="share of the samples with q >= the threshold of their cell, pooled over the 155 cells"))
    path = os.path.join(OUT, "figA_detector_scores.json"); json.dump(num, open(path, "w"), indent=1); print(f"[ok] {path}")
    print(json.dumps({k: v for k, v in num.items() if k not in ("plot", "counts", "outside_plot_range", "sources")}, indent=1))


if __name__ == "__main__":
    main()

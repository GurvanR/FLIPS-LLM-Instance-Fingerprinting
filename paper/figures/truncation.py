"""truncation.py - first-100-bits truncation figure of the paper: hierarchical closed set, full-length sequences vs features of the first 100 bits,
8 token pairs x splits 0-3 = 32 cells, N_t 1-64, drawn from results/closed_set/truncation/summary.json.
Bars = mean over the 8 pairs of the split-mean accuracy, error bars = 95 % pair-bootstrap CI (B 2000, seed 0); label above = paired drop hier 100 bit - hier full (pp, summary.json).
  python -m paper.figures.truncation
Writes <out>/figures/fig18_truncation.{pdf,png,json}."""
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from ..dataio import COL1, GRID, NT, R, RED, TRUNC, XL_NT, fig_dir, r4, style

LIGHT = "#f4b6b6"                                   # second bar colour of the closed-set bar figures


def boot_ci(v, B=2000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def trunc(out):
    d = json.load(open(R(*TRUNC, "summary.json"))); nts = [str(n) for n in NT]; st = {}
    for nt in nts:
        e = d["per_nt"][nt]; pp = e["per_pair"]; ps = sorted(pp)
        for arm in ("hier_full", "hier_100"):
            v = [pp[p][arm] for p in ps]; assert abs(np.mean(v) - e[arm]["acc"]) < 1e-9, (nt, arm)
            st.setdefault(nt, {})[arm] = dict(acc=float(np.mean(v)), ci95=boot_ci(v), std=float(np.std(v)))
        st[nt]["drop_hier"] = dict(mean=e["drop_hier"]["mean"])
        st[nt]["n_pairs"] = len(ps)
    fig, ax = plt.subplots(figsize=(COL1, 2.3)); x = np.arange(len(nts)); w = 0.38
    for off, arm, lab, hatch, fc in ((-w / 2, "hier_full", "full length", None, RED), (w / 2, "hier_100", "first 100 bits", "////", LIGHT)):
        m = np.array([st[n][arm]["acc"] for n in nts]); lo = np.array([st[n][arm]["ci95"][0] for n in nts]); hi = np.array([st[n][arm]["ci95"][1] for n in nts])
        ax.bar(x + off, m, w, yerr=[m - lo, hi - m], color=fc, hatch=hatch, edgecolor=RED if hatch else None, lw=0.0, capsize=1.3, error_kw=dict(lw=0.6), label=lab)
    for xi, n in zip(x, nts):
        top = max(st[n]["hier_full"]["ci95"][1], st[n]["hier_100"]["ci95"][1]); ax.text(xi, top + 0.02, f"{100 * st[n]['drop_hier']['mean']:+.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels(nts); ax.set_xlabel(XL_NT); ax.set_ylabel("Accuracy (closed set)"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", **{k_: v for k_, v in GRID.items() if k_ != "which"}); ax.set_axisbelow(True)
    ax.legend(fontsize=6, loc="upper left", frameon=False, title="instance classifier\n(numbers: 100 bits $-$ full, pp)", title_fontsize=6, alignment="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig18_truncation.pdf")); fig.savefig(os.path.join(out, "fig18_truncation.png"), dpi=150); plt.close(fig)
    rr = lambda z: {k: (r4(v) if isinstance(v, float) else ([r4(q) for q in v] if isinstance(v, list) else v)) for k, v in z.items()}
    num = dict(per_nt={n: {k: (rr(v) if isinstance(v, dict) else v) for k, v in st[n].items()} for n in nts}, pairs=sorted(d["per_nt"]["32"]["per_pair"]), splits=[0, 1, 2, 3], n_cells=d["n_cells"],
               note="hierarchical closed set on full-length vs first-100-bit features; acc = mean over the 8 token pairs of the split-mean (4 splits); ci95 of acc = pair bootstrap B 2000 seed 0 (this script); drop_hier = mean paired difference 100 bit - full, copied from summary.json",
               script="paper/figures/truncation.py")
    with open(os.path.join(out, "fig18_truncation.json"), "w") as f: json.dump(num, f, indent=1)
    return num


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args([] if argv is None else argv); style()
    t = trunc(fig_dir()); print("[ok] fig18_truncation")
    print(json.dumps({n: {k: t["per_nt"][n][k] for k in ("hier_full", "hier_100", "drop_hier")} for n in t["per_nt"]}, indent=0))


if __name__ == "__main__":
    main(sys.argv[1:])

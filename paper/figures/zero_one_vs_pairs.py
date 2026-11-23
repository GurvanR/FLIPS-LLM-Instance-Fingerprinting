"""zero_one_vs_pairs.py - randomness quality (NIST score) of token pair 0-1 against each of the 30 other pairs, and its closed-set recall at N_t 32.
NIST score = results/nist/nist_scores.json `score[pair][instance]` (per NIST test, share of the instance's sequences with p >= 0.05; mean over the 28 tests), the array lib/nist_split.arrays returns.
Per-instance recall at N_t 32 = hierarchical closed set, pooled over the 5 splits of a pair (lib/nist_split.arrays); the text quotes its mean for pair 0-1 and for the other pairs.
Figure: one box per pair (IQR, whiskers 5-95 %, dot = mean over the 237 instances), pairs sorted by mean NIST score, 0-1 highlighted, mean of the 30 others dashed.
  python -m paper.figures.zero_one_vs_pairs
Writes <out>/figures/fig04_zero_one_vs_pairs.{pdf,png,json}."""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..dataio import GREY, GRID, COL1, RED, Ctx, fig_dir, r4, se, style
from ..lib.nist_split import PAIR, arrays

C01, COTH = "#1f77b4", RED            # pair 0-1 blue, the other pairs red
NT = 32                               # N_t of the recall


def analyse(ctx, NT):
    """-> (the sidecar dict, what the figure needs)"""
    S, R = arrays(ctx, NT); pairs = sorted(S); assert len(pairs) == 31 and PAIR in pairs and set(R) == set(pairs)
    M = np.vstack([S[p] for p in pairs]); assert np.isfinite(M).all(), "NaN NIST score"            # (31, 237)
    n = M.shape[1]; others = [p for p in pairs if p != PAIR]; Mo = np.vstack([S[p] for p in others])
    mean = {p: float(S[p].mean()) for p in pairs}; order = sorted(pairs, key=lambda p: -mean[p]); rank01 = order.index(PAIR) + 1
    best = max(others, key=lambda p: mean[p]); s01 = S[PAIR]
    def beat(x, y): return float(((x > y).sum() + 0.5 * (x == y).sum()) / len(x))
    share_vs = {p: beat(s01, S[p]) for p in others}; ties_vs = {p: int((s01 == S[p]).sum()) for p in others}
    d_best = s01 - S[best]; med_inst = np.median(Mo, 0); max_inst = Mo.max(0); mean_oth = Mo.mean(0)
    # instance-level rank of 0-1 among the 31 pairs (1 = highest NIST score)
    rank_inst = 1 + (Mo > s01[None, :]).sum(0)
    from scipy.stats import t as tdist, wilcoxon
    rec = {p: R[p] for p in pairs}                    # recall at N_t (hierarchical, per instance, pooled over the splits of a pair)
    tq = float(tdist.ppf(0.975, n - 1)); se_d = se(d_best)
    num = dict(
        definition="NIST score of (instance, pair) = mean over the 28 NIST tests of the share of the instance's sequences with p >= 0.05 (results/nist/nist_scores.json)",
        n_instances=n, n_pairs=len(pairs),
        nist_mean_pair01=r4(mean[PAIR]), nist_mean_others_mean=r4(np.mean([mean[p] for p in others])),
        nist_mean_per_pair={p: r4(mean[p]) for p in order}, rank_pair01_by_mean=rank01,
        best_other_pair=best, best_other_mean=r4(mean[best]), second_best_other_pair=order[2] if order[0] == PAIR else None, second_best_other_mean=r4(mean[order[2]]) if order[0] == PAIR else None,
        worst_pair=order[-1], worst_pair_mean=r4(mean[order[-1]]),
        paired_diff_pair01_minus_best=dict(mean=r4(d_best.mean()), se_over_instances=r4(se_d), ci95_t=[r4(d_best.mean() - tq * se_d), r4(d_best.mean() + tq * se_d)], sd=r4(d_best.std(ddof=1)),
                                           wilcoxon_p=float(wilcoxon(d_best).pvalue), note="SE over the 237 instances, treated as independent (they are not: 30 base LLMs, shared settings); read as descriptive"),
        paired_diff_pair01_minus_mean_others=dict(mean=r4((s01 - mean_oth).mean()), se_over_instances=r4(se(s01 - mean_oth))),
        share_instances_pair01_beats_best_other=r4(share_vs[best]), ties_with_best_other=ties_vs[best],
        share_instances_pair01_beats_instance_median_of_others=r4(beat(s01, med_inst)),
        share_instances_pair01_beats_all_30_others=r4(float((s01 > max_inst).mean())), share_instances_pair01_rank1_or_tied=r4(float((s01 >= max_inst).mean())),
        median_pair_by_mean=dict(pairs=[order[15], order[16]] if order[0] == PAIR else None, note="with 0-1 first, the 30 others occupy ranks 2-31; their median = ranks 16 and 17"),
        share_instances_pair01_beats_median_pairs={p: r4(share_vs[p]) for p in ([order[15], order[16]] if order[0] == PAIR else [])},
        share_instances_pair01_beats_each_other={p: r4(share_vs[p]) for p in order if p != PAIR}, min_share_vs_any_other=r4(min(share_vs.values())), min_share_pair=min(share_vs, key=share_vs.get),
        rank_of_pair01_per_instance=dict(mean=r4(rank_inst.mean()), median=float(np.median(rank_inst)), share_rank1=r4(float((rank_inst == 1).mean())), share_top3=r4(float((rank_inst <= 3).mean()))),
        recall=dict(nt=NT, model="hierarchical closed set, per-instance recall pooled over the 5 splits of a pair (lib/nist_split.arrays)",
                    recall_mean_pair01=r4(np.nanmean(rec[PAIR])), recall_mean_others=r4(np.nanmean([np.nanmean(rec[p]) for p in others]))))
    num["source"] = dict(nist="results/nist/nist_scores.json", cells="results/closed_set/hierarchical/cells", script="paper/figures/zero_one_vs_pairs.py")
    return num, dict(S=S, order=order, mean=mean, best=best)


def draw(num, S, order, mean, best):
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    x = np.arange(len(order)); data = [S[p] for p in order]
    for xi, p, v in zip(x, order, data):
        c = C01 if p == PAIR else COTH; lo, q1, md, q3, hi = np.percentile(v, [5, 25, 50, 75, 95])
        ax.vlines(xi, lo, hi, color=c, lw=0.6, alpha=0.8); ax.add_patch(plt.Rectangle((xi - 0.32, q1), 0.64, q3 - q1, facecolor=c, alpha=0.35 if p != PAIR else 0.55, edgecolor=c, lw=0.6))
        ax.plot([xi - 0.32, xi + 0.32], [md, md], color=c, lw=0.8); ax.plot(xi, mean[p], marker="D" if p == PAIR else "o", ms=3.2, color=c, mec="white", mew=0.3, zorder=4)
    mo = num["nist_mean_others_mean"]; ax.axhline(mo, color=GREY, ls="--", lw=0.8, zorder=1)
    ib = order.index(best); ax.annotate(f"best other pair: {best} ({mean[best]:.2f})", xy=(ib, mean[best]), xytext=(ib + 1.2, 0.80), fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"), va="center")
    ax.text(order.index(PAIR) + 0.45, 0.93, f"pair 0-1 ({mean[PAIR]:.2f})", fontsize=6.5, color=C01, fontweight="bold", va="center")
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=C01, ms=3.2, label="pair 0-1"), Line2D([], [], marker="o", ls="", color=COTH, ms=3.2, label="other pair"),
                       Line2D([], [], color=GREY, ls="--", lw=0.8, label=f"mean of the 30 others ({mo:.2f})")], fontsize=5.8, loc="lower left", ncol=1, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.set_ylabel("NIST score"); ax.set_ylim(0, 1.0); ax.grid(axis="y", **{k: v for k, v in GRID.items() if k != "which"}); ax.set_axisbelow(True)
    ax.set_title("NIST score of the 237 instances, per token pair", fontsize=7, loc="left", pad=3)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=90, fontsize=5.6); ax.get_xticklabels()[order.index(PAIR)].set_color(C01); ax.get_xticklabels()[order.index(PAIR)].set_fontweight("bold")
    ax.set_xlabel("Token pair (sorted by mean NIST score)", fontsize=8); ax.set_xlim(-0.7, len(order) - 0.3)
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="NIST score of token pair 0-1 against each other pair (figure and JSON) and its recall at N_t 32 (JSON)")
    ap.add_argument("--data", help="data root (default: $FLIPS_DATA)"); ap.add_argument("--out", help="output root (default: $FLIPS_OUT)")
    a = ap.parse_args(argv)
    if a.data: os.environ["FLIPS_DATA"] = a.data
    if a.out: os.environ["FLIPS_OUT"] = a.out
    out = fig_dir(); ctx = Ctx(out, nr=300, nt=NT); stem = "fig04_zero_one_vs_pairs"
    num, aux = analyse(ctx, NT)
    with plt.rc_context():                                                       # matplotlib defaults + the shared style; rcParams restored afterwards
        plt.rcdefaults(); style(); fig = draw(num, **aux)
        fig.savefig(os.path.join(out, stem + ".pdf")); fig.savefig(os.path.join(out, stem + ".png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out, stem + ".json"), "w") as f: json.dump(num, f, indent=1)
    print(json.dumps({k: v for k, v in num.items() if k not in ("nist_mean_per_pair", "share_instances_pair01_beats_each_other")}, indent=1))


if __name__ == "__main__":
    main()

"""corpus.py - the two corpus-description figures of the appendix and their numbers.
  fig16_discard_rate   share of generations that are still empty after the retries (rows discarded), per base LLM  (results/corpus/valid_share.json)
  fig20_seq_length     CCDF of the sequence length (valid sequences, >= 100 bits) per base LLM                     (results/corpus/length_stats.json)
Also writes corpus_numbers.json (discard rate: overall and the three worst base LLMs; mean valid length, share >= 500 bits and share < 100 bits per base LLM) and
discard_macros.tex (the \\rDiscard... macros of the paper text; the share at temperature 1.0 is computed by tone_share()).
Both figures are drawn with matplotlib's default parameters, whatever style the calling process set. Outputs go to <out>/figures."""
import argparse, functools, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..dataio import R, fig_dir, orig_name, r4


def default_style(fn):
    """run fn with matplotlib's default parameters"""
    @functools.wraps(fn)
    def run(*a, **k):
        with plt.style.context("default"): return fn(*a, **k)
    return run


def save(fig, name):
    out = fig_dir()
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(out, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[ok] {name}")


def base_llm(n):
    """base LLM of a class name (an abliterated checkpoint is attributed to the original LLM, dataio.ABLIT_MAP)"""
    return orig_name(n)


# ------------------------------------------------------------------------------------------------------------------------------------------
# discard rate per base LLM
# ------------------------------------------------------------------------------------------------------------------------------------------
@default_style
def fig_fail():
    v = json.load(open(R("corpus", "valid_share.json"))); names = v["class_names"]; e = np.array(v["empty_share"])
    orig = [base_llm(n) for n in names]; labels = sorted(set(orig))
    fr = [float(np.mean([e[i] for i in range(len(names)) if orig[i] == l])) for l in labels]
    fig, ax = plt.subplots(figsize=(10, 6)); x = np.arange(len(labels))
    ax.bar(x, fr, color="#2e86c1"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Failure Rate", fontsize=13); ax.set_xlabel("LLM", fontsize=13); ax.grid(True, ls="--", alpha=.5); ax.set_axisbelow(True); fig.tight_layout()
    o = np.argsort(fr)[::-1]
    num = dict(overall=r4(e.mean()), top3=[(labels[i], r4(fr[i])) for i in o[:3]], n_zero=int(sum(f == 0 for f in fr)))
    save(fig, "fig16_discard_rate")
    return num


# ------------------------------------------------------------------------------------------------------------------------------------------
# sequence-length CCDF per base LLM: share of VALID (>= 100) sequences with length >= l
# ------------------------------------------------------------------------------------------------------------------------------------------
@default_style
def fig_seqlen():
    g = json.load(open(R("corpus", "length_stats.json"))); names = g["class_names"]
    H = np.array(g["hist"], float); orig = [base_llm(n) for n in names]; labels = sorted(set(orig))
    fig, ax = plt.subplots(figsize=(14, 7.5)); L = np.arange(H.shape[1]); out = {}
    cm = plt.get_cmap("tab20")
    for i, l in enumerate(labels):
        h = H[[j for j, o in enumerate(orig) if o == l]].sum(0); hv = h.copy(); hv[:100] = 0
        cc = hv[::-1].cumsum()[::-1] / hv.sum(); xs = np.arange(100, 1001)
        ax.plot(xs, cc[100:1001], color=cm(i % 20), lw=1.2, label=l, ls="-" if i < 20 else "--")
        out[l] = dict(mean_len_valid=r4((hv * L).sum() / hv.sum()), share_ge500=r4(cc[500]), share_lt100=r4(h[:100].sum() / h.sum()))
    ax.axvline(500, color="black", ls=":", lw=1.2); ax.text(503, 0.98, "MaxTokens = 500", rotation=-90, va="top", fontsize=9, color="gray")
    ax.set_xlabel("Sequence Length", fontsize=16); ax.set_ylabel(r"Proportion of Samples $\geq$ Sequence Length", fontsize=14); ax.grid(True, ls="--", alpha=.5)
    ax.legend(title="LLM", fontsize=8, title_fontsize=14, loc="upper right"); fig.tight_layout()
    save(fig, "fig20_seq_length")
    return dict(per_llm=out)


# ------------------------------------------------------------------------------------------------------------------------------------------
# macros of the paper text
# ------------------------------------------------------------------------------------------------------------------------------------------


def tone_share():
    """share of the empty (discarded) mass that lies on instances generated at temperature 1.0 (equal weight per instance; the abliterated checkpoints are generated at 1.0 too)"""
    v = json.load(open(R("corpus", "valid_share.json"))); e = np.array(v["empty_share"])
    t1 = np.array(["_temp-1.0_" in n for n in v["class_names"]])
    return float(e[t1].sum() / e.sum())


def discard_macros(fail_rate):
    """TeX definitions of the discard shares quoted in the corpus analysis (4 decimals), from the figure's own numbers"""
    t = fail_rate["top3"]; f = lambda x: "%.4f" % x
    L = ["% generated by paper/figures/corpus.py: do not edit",
         "\\newcommand{\\rDiscardAll}{%s}       %% share of generations still empty after the retries, over all instances (corpus_numbers.json: fail_rate.overall)" % f(fail_rate["overall"])]
    for k, nm in enumerate(("A", "B", "C")):
        L.append("\\newcommand{\\rDiscardTop%s}{%s}      %% %s, corpus_numbers.json: fail_rate.top3[%d]" % (nm, f(t[k][1]), t[k][0].split("/")[-1], k))
    L.append("\\newcommand{\\rDiscardTone}{%.2f}       %% share of the discarded generations at temperature 1.0 (tone_share(): equal weight per instance)" % tone_share())
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="corpus figures (discard rate, sequence length), corpus_numbers.json and discard_macros.tex")
    ap.add_argument("--data", help="data root (default: $FLIPS_DATA)"); ap.add_argument("--out", help="output root (default: $FLIPS_OUT)")
    a = ap.parse_args(argv)
    if a.data: os.environ["FLIPS_DATA"] = a.data
    if a.out: os.environ["FLIPS_OUT"] = a.out
    num = dict(fail_rate=fig_fail(), seqlen=fig_seqlen()); out = fig_dir()
    p = os.path.join(out, "corpus_numbers.json"); json.dump(num, open(p, "w"), indent=1); print(f"[ok] {p}")
    p = os.path.join(out, "discard_macros.tex"); open(p, "w").write(discard_macros(num["fail_rate"])); print(f"[ok] {p}")


if __name__ == "__main__":
    main()

"""showcase.py - abliterated showcase figure of the paper (fig:showcase).
Bars of closed/numbers.json["showcase"] (N_t 32): share of the audits of each abliterated LLM attributed to its original LLM by LLMmap and by FLIPS. The LLMmap bars are constants
(lib/showcase.LLMMAP), measured with the pretrained LLMmap taken off the shelf; its raw outputs are not in the results archive.
  python -m paper.figures.showcase
Writes <out>/figures/showcase_abliterated.{pdf,png}."""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from ..dataio import COL1, PURPLE, RED, closed_dir, fig_dir, style

NAMES = ["Phi-3-medium-4k", "Smaug-Llama-3-70B", "Qwen2-7B"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="abliterated showcase figure (showcase_abliterated.pdf/.png)")
    ap.add_argument("--data", help="data root (default: $FLIPS_DATA)"); ap.add_argument("--out", help="output root (default: $FLIPS_OUT)")
    a = ap.parse_args(argv)
    if a.data: os.environ["FLIPS_DATA"] = a.data
    if a.out: os.environ["FLIPS_OUT"] = a.out
    with open(os.path.join(closed_dir(), "numbers.json")) as f: S = json.load(f)["showcase"]
    LL = [S["llmmap_pct"][n] for n in NAMES]; FL = [S["flips_attributed_to_original_pct"][n] for n in NAMES]
    with plt.rc_context():                                                       # matplotlib defaults + the shared style; rcParams restored afterwards
        plt.rcdefaults(); style(); fig, ax = plt.subplots(figsize=(COL1, 2.1)); x = np.arange(3); w = 0.38
        b1 = ax.bar(x - w / 2, LL, w, color=PURPLE, label="LLMmap-IPP"); b2 = ax.bar(x + w / 2, FL, w, color=RED, label="FLIPS ($N_t$=32)")
        for b, v in list(zip(b1, LL)) + list(zip(b2, FL)): ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=6); ax.set_ylim(0, 122); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=6)
        ax.set_ylabel("Wrongly predicted\nas original model", fontsize=6.5); ax.set_xlabel("Abliterated LLM", fontsize=6.5)
        ax.legend(fontsize=5.5, loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.2); ax.grid(axis="y", ls="--", alpha=.4); ax.set_axisbelow(True); fig.tight_layout()
        out = fig_dir(); fig.savefig(os.path.join(out, "showcase_abliterated.pdf")); fig.savefig(os.path.join(out, "showcase_abliterated.png"), dpi=150)
    plt.close(fig)
    print(f"[ok] showcase_abliterated -> {out}")


if __name__ == "__main__":
    main()

"""llmmap_curve.py - the LLMmap-IF baseline curve of the accuracy-vs-N_t figure and its numbers (N_t 1..8).

Input: results/llmmap/results.json, the per-split accuracies stored by the LLMmap-IF baseline runs (N_r 40, 10 splits per N_t).
FILTER (explicit): the curve is the *filtered mean*, i.e. the mean over only the splits whose accuracy is above the across-split mean
(`accuracy_kept_splits` holds exactly these kept splits, 4-8 of 10 per N_t; `accuracy_kept_mean` is their mean). The all-split mean is also reported
(`accuracy_all_splits_mean`: 0.12-0.26, against 0.20-0.33 filtered). The filter favours the baseline, so the comparison with FLIPS is conservative.
The baseline was trained with N_r = 40 sequences per instance, FLIPS with N_r = 300."""
import json, os
from ..dataio import R, out_dir

HEADER = "N_t kept/n filtered_mean all_split_mean kept_min kept_max dropped_mean"


def load():
    return json.load(open(R("llmmap", "results.json")))["per_nt"]


def table():
    """rows of (N_t, 'kept/n', filtered mean, all-split mean, min kept, max kept, implied mean of the dropped splits)"""
    d = load(); rows = []
    for k in sorted(d, key=int):
        c = d[k]
        n, kept = c["n_splits"], c["accuracy_kept_splits"]
        assert abs(sum(kept) / len(kept) - c["accuracy_kept_mean"]) < 1e-9, k                 # accuracy_kept_mean is the mean of the kept (above-mean) splits
        dropped = (c["accuracy_all_splits_mean"] * n - sum(kept)) / (n - len(kept))
        rows.append((int(k), f"{len(kept)}/{n}", round(c["accuracy_kept_mean"], 4), round(c["accuracy_all_splits_mean"], 4), round(min(kept), 3), round(max(kept), 3), round(dropped, 4)))
    return rows


def text():
    return "\n".join([HEADER] + [" ".join(str(x) for x in r) for r in table()]) + "\n"


def curve():
    """{N_t: filtered mean accuracy (4 decimals)}, N_t 1..8"""
    return {r[0]: r[2] for r in table()}


def main(argv=None):
    path = os.path.join(out_dir("figures"), "llmmap_splits.txt")
    open(path, "w").write(text()); print(f"[ok] {path}")


if __name__ == "__main__":
    main()

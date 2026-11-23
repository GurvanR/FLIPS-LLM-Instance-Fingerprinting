"""common.py - shared loaders of the open-set and closed-set summary figures.

Sources (all in the results archive, nothing is fitted here):
  open set    results/open_set/{summary.json, percell.csv, curves.tar.gz}: the method's open set (hierarchical class decision + Unseen detector),
              155 cells = 31 token pairs x 5 draws, N_t 1..64. summary.json: mean over pairs of per-pair means, 95 % pair-clustered percentile CIs (B 2000, seed 20261031).
              The paper's open-set numbers are this pooled 5-draw summary.
  closed set  results/closed_set/hierarchical/summary.json: hierarchical closed set, 155 cells = 31 pairs x 5 splits, N_t 1..64 (acc_hier, std_hier = population sd
              across the 31 pair means).
Pooling of the per-cell curves: per-cell value -> per-pair nanmean -> nanmean over pairs; the Unseen precision cu / (cu + kr) is undefined (left out) for a cell that
flags nothing; a curve point needs a defined precision for >= MIN_DEF of the pairs."""
import csv, json, os, re, tarfile, warnings
import numpy as np
from .dataio import R, out_dir, CLOSED, OPEN

NTS = [1, 2, 4, 8, 16, 32, 64]
TARGETS = np.array([round(0.05 * i, 2) for i in range(1, 20)]); I50 = 9           # pre-test nominal Unseen-recall targets .05-.95, TARGETS[9] = .50
MIN_DEF = .5
COL = 3.3                                                                # column width (in)


def open_summary_path(): return R(*OPEN, "summary.json")
def open_percell_path(): return R(*OPEN, "percell.csv")
def open_curves_path(): return R(*OPEN, "curves.tar.gz")
def closed_summary_path(): return R(*CLOSED, "summary.json")


def r4(x):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), 4)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)): return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        return o
    with open(path, "w") as f: json.dump(conv(obj), f, indent=1)


def save(fig, name, out=None):
    out = out or out_dir("figures"); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=200)


def pair_of(c): return c.rsplit("_s", 1)[0]


# ---------------------------------------------------------------- open-set summary / percell
def open_summary():
    S = json.load(open(open_summary_path())); assert sorted(int(k) for k in S["table"]) == NTS, sorted(S["table"]); return S


def open_level(S, nt, metric):
    """(est, lo, hi) of summary.json"""
    d = S["table"][str(nt)][metric]; return d["est"], d["lo"], d["hi"]


def open_pair_sd(nt, metric):
    """population sd (ddof 0, as std_hier of the closed set) across the 31 per-pair means of a percell.csv metric, and the mean of the pair means (= summary est)"""
    d = {}
    for r in csv.DictReader(open(open_percell_path())):
        if int(r["nt"]) == nt:
            v = float(r[metric])
            if np.isfinite(v): d.setdefault(pair_of(r["cell"]), []).append(v)
    pm = np.array([np.mean(v) for v in d.values()], float); return float(pm.mean()), float(pm.std(ddof=0)), len(pm)


# ---------------------------------------------------------------- open-set per-cell curves (read from the tarball in memory)
def load_curves(nt):
    """{cell: dict(test=...)} for the 155 cells (5 draws) at N_t = nt"""
    cells = {}
    with tarfile.open(open_curves_path()) as tf:
        for m in tf.getmembers():
            g = re.fullmatch(r"nt(\d+)/(.+)\.json", m.name)
            if not g or int(g.group(1)) != nt: continue
            j = json.load(tf.extractfile(m)); assert j["nt"] == nt and j["cell"] == g.group(2) and j["cell"] not in cells, (m.name, j["nt"], j["cell"])
            cells[j["cell"]] = dict(test=j["test"])
    assert len(cells) == 155 and len({pair_of(c) for c in cells}) == 31, (nt, len(cells))
    return cells


def pairmeans(cells, x):
    """per-cell vector (or matrix, cells x grid) -> (sorted pairs, pair means)"""
    ks = list(cells); pairs = sorted({pair_of(c) for c in ks})
    pm = np.array([np.nanmean([x[i] for i, c in enumerate(ks) if pair_of(c) == p], axis=0) for p in pairs]); return pairs, pm


def prec(num, den): return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def sweep(cells):
    """per-cell arrays (cells x targets) of the pre-test threshold sweep on the test side"""
    return {q: np.array([v["test"]["sweep"][q] for v in cells.values()], float) for q in ("recall", "kr", "cu")}


def up_curve(cells, Q, x=None):
    """pair-pooled Unseen precision of the sweep (and pair-pooled realised recall unless x is given); NaN where < MIN_DEF of the pairs have a defined precision"""
    with np.errstate(invalid="ignore"):
        pm = pairmeans(cells, prec(Q["cu"], Q["cu"] + Q["kr"]))[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = np.nanmean(pm, 0)
    ndef = np.mean(np.isfinite(pm), 0); y[ndef < MIN_DEF] = np.nan
    xx = np.nanmean(pairmeans(cells, Q["recall"])[1], 0) if x is None else np.asarray(x, float)
    return xx, y, ndef


def prevalence(cells):
    """pair-pooled share of Unseen batches among the test batches (= Unseen precision of a detector that flags at random)"""
    L = [v["test"]["batches"] for v in cells.values()]; p = np.array([x["nU"] / (x["nU"] + x["nK"]) for x in L], float)
    return float(np.mean(pairmeans(cells, p)[1]))


# ---------------------------------------------------------------- closed set
def closed_stats():
    S = json.load(open(closed_summary_path())); assert S["n_cells"] == 155 and S["pairs_complete"] == 31, (S["n_cells"], S["pairs_complete"])
    out = {}
    for n in NTS:
        a = S["per_nt"][str(n)]["all"]; pa = S["per_nt"][str(n)]["per_pair_acc"]; v = np.array([pa[p]["hier"] for p in pa], float)
        assert len(v) == 31 and abs(v.mean() - a["acc_hier"]) < 1e-9 and abs(v.std(ddof=0) - a["std_hier"]) < 1e-9, n
        out[n] = dict(mean=a["acc_hier"], sd=a["std_hier"], n_pairs=a["n_pairs"])
    return out


def llmmap():
    """stored LLMmap-IF accuracy, N_t 1..8 (filtered mean, see stats/llmmap_curve.py)"""
    from .stats import llmmap_curve
    ll = llmmap_curve.curve(); assert ll and sorted(ll) == list(range(1, 9)), ll
    return ll

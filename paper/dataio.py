"""dataio.py - shared data access, naming and drawing helpers of the analysis tier (numpy + scipy + matplotlib only).

Everything the figures and numbers read comes from ONE data root, `$FLIPS_DATA/results/...` (the unpacked `flips_results.zip`, see
data/README.md); everything they write goes to ONE output root, `$FLIPS_OUT` (default `paper/out/`). Defaults: `FLIPS_DATA` = `<repo>/data`.

Results layout used here (`R(...)` joins below `<data>/results`):
  closed_set/hierarchical/        summary.json  cells/<pair>_s<split>.json  importance/<pair>_s<split>.json  predictions.json  calibration.json
  closed_set/classifier_comparison/  xgboost_cells/  classifier_cells/
  closed_set/truncation/summary.json
  open_set/                       summary.json  percell.csv  curves.tar.gz  distance2.json  per_instance/
  nist/nist_scores.json  corpus/{valid_share,length_stats}.json  meta/class_order.json  llmmap/results.json

CLASS ORDER. Every per-class array (cell counts, the confusion matrix, NIST scores, length histograms) follows the class order of `meta/class_order.json`.
The per-attribute error counts are computed from the confusion matrix.
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(PKG)
CONSTANTS = os.path.join(PKG, "constants")


def data_root():
    return os.environ.get("FLIPS_DATA") or os.path.join(REPO, "data")


def out_root():
    return os.environ.get("FLIPS_OUT") or os.path.join(PKG, "out")


def R(*parts):
    """path below <data>/results"""
    return os.path.join(data_root(), "results", *parts)


def out_dir(*parts):
    """<out>/<parts>, created"""
    p = os.path.join(out_root(), *parts); os.makedirs(p, exist_ok=True); return p


CLOSED = ("closed_set", "hierarchical")             # R(*CLOSED, "summary.json"), "cells", "importance", "predictions.json", "calibration.json"
CLF_CMP = ("closed_set", "classifier_comparison")
TRUNC = ("closed_set", "truncation")
OPEN = ("open_set",)

NT = [1, 2, 4, 8, 16, 32, 64]
RED, GREEN, PURPLE, BLUE, GREY, ORANGE = "red", "#4daf4a", "#984ea3", "tab:blue", "#777777", "tab:orange"
GRID = dict(visible=True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
XL_NT = r"Number of queries at verification $(N_t)$"
ABLIT_MAP = {   # abliterated checkpoint -> the original model it was derived from
    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3": "meta-llama/Llama-3.1-8B-Instruct", "failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3": "abacusai/Smaug-Llama-3-70B-Instruct",
    "natong19/Qwen2-7B-Instruct-abliterated": "Qwen/Qwen2-7B-Instruct", "failspy/Phi-3-medium-4k-instruct-abliterated-v3": "microsoft/Phi-3-medium-4k-instruct",
    "dphn/dolphin-2.9.2-Phi-3-Medium-abliterated": "microsoft/Phi-3-medium-128k-instruct"}
RX = re.compile(r"^(?P<model>.+?)(?:@@(?P<quant>.+?))?_temp-(?P<T>[^_]+)_sp-(?P<SP>.+)$")
ATTR = ("model", "quant", "T", "SP")
SP_PAPER = {"0": "1", "3": "2", "6": "3", "7": "4"}                # the paper's numbering of the system prompts (class name digit -> paper index)
QUANT_DISPLAY = {"bitsandbytes_int4": "int4", "fp8": "fp8", "-": "none"}
SHORTLIST = ["CohereForAI/c4ai-command-r-plus", "CohereLabs/aya-23-35B", "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", "Qwen/Qwen2-72B-Instruct", "Qwen/Qwen3-30B-A3B-Instruct-2507",
             "abacusai/Smaug-Llama-3-70B-Instruct", "google/gemma-2-27b-it", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct",
             "microsoft/Phi-3-medium-128k-instruct", "Qwen/Qwen2-7B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
COL1, COL2 = 3.4, 7.1                                # figure widths (in) of one / two columns of the paper page


class Skip(Exception):
    """a figure whose input files are missing / incomplete"""


def style():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.5,
                         "lines.markersize": 4.5, "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                         "axes.spines.top": False, "axes.spines.right": False})


def r4(x): return None if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)) else round(float(x), 4)
def jl(p): return json.load(open(p)) if os.path.exists(p) else None
def se(x): x = np.asarray(x, float); return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
def sd(x): x = np.asarray(x, float); return float(x.std(ddof=0)) if len(x) else float("nan")


# ------------------------------------------------------------------------------------------------------------------------------------------
# class names, attributes
# ------------------------------------------------------------------------------------------------------------------------------------------
def canon_names():
    return list(json.load(open(R("meta", "class_order.json")))["class_names"])


def attrs(n):
    m = RX.match(n)
    if not m: raise ValueError(f"unparsable class name {n}")
    return m["model"], m["quant"] or "-", m["T"], m["SP"]


def tkey(v):
    try: return (0, float(v))
    except ValueError: return (1, v)


def orig_name(n):
    """base LLM of a class for the heatmaps: abliterated -> the original model (ABLIT_MAP), quantisation stripped"""
    if "abliterated" in n.lower(): return ABLIT_MAP[n.split("_temp-")[0]]
    return n.split("_temp-")[0].split("@@")[0]


def var_name(n):
    if "abliterated" in n.lower(): return "ablit"
    m = RX.match(n); q = m["quant"]; v = f"temp-{m['T']}_sp-{m['SP']}"
    return f"{q}_{v}" if q else v


def fmt_var(v):
    if v == "ablit": return "abli-\nterated"
    m = re.search(r"^(?:(?P<q>.+?)_)?temp-(?P<T>[\d.]+)_sp-(?P<SP>-?\d+)$", v)
    if not m: return v
    ql = QUANT_DISPLAY.get(m["q"], m["q"]) + "\n" if m["q"] else ""
    return f"{ql}temp {m['T']}\n(no sp)" if m["SP"] == "-1" else f"{ql}sp {SP_PAPER.get(m['SP'], m['SP'])}\n(temp {m['T']})"


def short_model(n, k=26):
    n = n.rsplit("/", 1)[-1] if len(n) > k else n
    return n if len(n) <= k else n[:k - 3] + "..."


def attr_values(names, k):
    """sorted values of attribute k over the classes"""
    ai = ATTR.index(k); return sorted({attrs(n)[ai] for n in names}, key=tkey)


def attr_matrix(M, names, k):
    """confusion counts (C, C) -> (values, matrix (V, V)) summed over the classes sharing an attribute value (rows true, cols predicted)"""
    ai = ATTR.index(k); vals = attr_values(names, k); ix = {v: i for i, v in enumerate(vals)}
    G = np.zeros((len(names), len(vals))); G[np.arange(len(names)), [ix[attrs(n)[ai]] for n in names]] = 1
    return vals, G.T @ np.asarray(M, float) @ G, G


# ------------------------------------------------------------------------------------------------------------------------------------------
# one build over a closed-set results tree
# ------------------------------------------------------------------------------------------------------------------------------------------
class Ctx:
    """one build over a closed-set results tree: `tree` holds cells/, importance/, predictions.json, calibration.json (default: results/closed_set/hierarchical);
    `cells_sub` names the subfolder of the cell JSONs; out = output dir; num = the numbers dict of the build"""

    def __init__(self, out, tree=None, cells_sub="cells", nr=300, nt=32):
        self.tree = tree or R(*CLOSED); self.out, self.cells_sub = out, cells_sub
        self.nr, self.nt = nr, nt                                     # training sequences per instance / queries per verification sample (N_r 300, N_t 32 in the paper; any value present in the files)
        self.names = canon_names(); self.C = len(self.names); self.num = {}; self.produced = []; self._c = {}
        os.makedirs(out, exist_ok=True)

    def p(self, *a): return os.path.join(self.tree, *a)
    def exists(self, *a): return os.path.exists(self.p(*a))

    def save(self, fig, name, out=None):
        out = out or self.out; fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=150); plt.close(fig)
        self.produced.append(name); print(f"[ok] {name}")

    def cache(self, key, fn):
        if key not in self._c: self._c[key] = fn()
        return self._c[key]

    def cells(self, sub=None):
        """list of cell JSONs under <tree>/<sub or cells_sub>/*.json"""
        pat = self.p(sub or self.cells_sub, "*.json")
        return self.cache(("cells", pat), lambda: [json.load(open(f)) for f in sorted(glob.glob(pat))])


def records(runs, nt=None, nr=None):
    """flat list of per-(pair, split, N_t, N_r) records of cell JSONs"""
    return [c for j in runs for c in j["cells"] if (nt is None or c["nt"] == nt) and (nr is None or c["nr"] == nr)]


def acc_by_cell(runs, nt=32, nr=None, tps=None, splits=None, field="acc"):
    """{(pair, split): value}"""
    d = {}
    for c in records(runs, nt, nr):
        if (tps is None or c["tp"] in tps) and (splits is None or c["split"] in splits): d[(c["tp"], c["split"])] = c[field]
    return d


def pair_means(d):
    """{(pair, split): v} -> {pair: mean over splits}"""
    t = {}
    for (tp, s), v in d.items(): t.setdefault(tp, []).append(v)
    return {k: float(np.mean(v)) for k, v in t.items()}


def class_counts(runs, nt=32, nr=None, C=237):
    """per-class (correct, batches) pooled over all cells of the runs"""
    k = np.zeros(C); n = np.zeros(C)
    for c in records(runs, nt, nr): k += np.asarray(c["class_correct"], float); n += np.asarray(c["class_batches"], float)
    return k, n


def class_recall_pairmean(runs, nt=32, nr=None, C=237):
    """per-class recall: pooled over the splits of one pair, then nanmean over pairs
    (pairs carry unequal batch counts per class, so this differs from the all-cells pooled recall by up to ~0.09 on a class)"""
    k, n = {}, {}
    for c in records(runs, nt, nr):
        t = c["tp"]; k[t] = k.get(t, 0) + np.asarray(c["class_correct"], float); n[t] = n.get(t, 0) + np.asarray(c["class_batches"], float)
    if not k: return np.full(C, np.nan)
    Rm = np.vstack([np.where(n[t] > 0, k[t] / np.maximum(n[t], 1), np.nan) for t in k])
    with np.errstate(all="ignore"): return np.nanmean(Rm, axis=0)


# ------------------------------------------------------------------------------------------------------------------------------------------
# stored predictions / calibration
# ------------------------------------------------------------------------------------------------------------------------------------------
def load_post(ctx):
    """(predictions.json, calibration.json) of the tree"""
    return jl(ctx.p("predictions.json")), jl(ctx.p("calibration.json"))


def post_confusion(ctx, cl, nt):
    """(C, C) confusion counts of the reported N_t, canonical class order, rows true / cols predicted"""
    if str(nt) not in cl["confusion"]: raise Skip(f"predictions.json has no N_t {nt} confusion (has {sorted(cl['confusion'], key=int)})")
    assert cl["class_names"] == ctx.names, "predictions.json class order differs from meta/class_order.json"
    return np.asarray(cl["confusion"][str(nt)], float)


# ------------------------------------------------------------------------------------------------------------------------------------------
# heatmap (LLM x variation layout)
# ------------------------------------------------------------------------------------------------------------------------------------------
def nxm_stats(names, vals):
    orig = [orig_name(n) for n in names]; var = [var_name(n) for n in names]; vals = np.asarray(vals, float)
    origs = sorted(set(orig)); vars_ = sorted(set(var))
    if "ablit" in vars_: vars_.remove("ablit"); vars_.append("ablit")
    mat = {}
    for o in origs:
        for v in vars_:
            idx = next((i for i in range(len(names)) if orig[i] == o and var[i] == v), None); mat[(o, v)] = vals[idx] if idx is not None else np.nan
    def mean(ids): ids = list(ids); a = vals[ids] if ids else np.array([]); return float(np.nanmean(a)) if len(a) and not np.all(np.isnan(a)) else np.nan
    row = {o: mean([i for i in range(len(names)) if orig[i] == o]) for o in origs}; col = {v: mean([i for i in range(len(names)) if var[i] == v]) for v in vars_}
    grand = mean(range(len(names)))
    vars_ = [v for v in vars_ if any(not np.isnan(mat[(o, v)]) for o in origs)]
    qset = {v for v in vars_ if any("@@" in names[i] and var[i] == v for i in range(len(names)))}
    qv = [v for v in vars_ if v in qset]; bv = [v for v in vars_ if v not in qset]
    qcap = {o for o in origs if any(not np.isnan(mat[(o, v)]) for v in qv)}
    origs = [o for o in origs if o not in qcap] + [o for o in origs if o in qcap]
    return dict(origs=origs, vars=bv + qv, quant_vars=qv, qcap=qcap, mat=mat, row=row, col=col, grand=grand)


def draw_heatmap(S, ax=None, figsize=(COL2, 4.6), cbar_label="Per-class Recall", cmap_name="RdYlGn", fs=5.5, cbar=True, title=None, ylabels=True, xrot=None):
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    origs, vars_ = S["origs"], S["vars"]; n_rows, n_cols = len(origs) + 1, len(vars_) + 1; data = np.full((n_rows, n_cols), np.nan)
    for i, o in enumerate(origs):
        data[i, 0] = S["row"][o]
        for j, v in enumerate(vars_): data[i, j + 1] = S["mat"][(o, v)]
    for j, v in enumerate(vars_): data[-1, j + 1] = S["col"][v]
    data[-1, 0] = S["grand"]
    n_q = len(S["quant_vars"]); n_top = sum(1 for o in origs if o not in S["qcap"]); n_base = len(vars_) - n_q
    own = ax is None
    if own: fig, ax = plt.subplots(figsize=figsize)
    else: fig = ax.figure
    cmap = plt.get_cmap(cmap_name); norm = Normalize(0, 1)
    m = ax.pcolormesh(np.ma.masked_invalid(data), cmap=cmap, norm=norm, edgecolors="lightgray", linewidth=0.3); ax.set_xlim(0, n_cols); ax.set_ylim(n_rows, 0)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v): continue
            r, g, b, _ = cmap(norm(v)); lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=fs, color=".15" if lum > 0.408 else "w")
    if cbar:
        cb = fig.colorbar(m, ax=ax, pad=0.02, fraction=0.04); cb.set_label(cbar_label, fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    ax.add_patch(Rectangle((0, n_rows - 1), 1, 1, fill=False, edgecolor="black", linewidth=1.5, zorder=10))
    ax.plot([1, 1], [0, n_rows - 1], color="black", linewidth=1.2, zorder=9, solid_capstyle="butt"); ax.plot([1, n_cols], [n_rows - 1, n_rows - 1], color="black", linewidth=1.2, zorder=9, solid_capstyle="butt")
    ax.plot([0, 1 + n_base, 1 + n_base, n_cols, n_cols, 0, 0], [0, 0, n_top, n_top, n_rows, n_rows, 0], color="black", linewidth=0.5, zorder=8, solid_capstyle="butt")
    if n_q > 0 and n_rows - 1 - n_top > 0:
        ax.plot([1 + n_base, n_cols, n_cols, 1 + n_base, 1 + n_base], [n_top, n_top, n_rows, n_rows, n_top], color="black", linewidth=0.5, zorder=8, solid_capstyle="butt")
    if n_top > 0 and n_q > 0:
        ax.add_patch(Rectangle((1 + n_base, 0), n_q, n_top, facecolor="white", edgecolor="white", linewidth=1.0, zorder=5))
        ax.text(1 + n_base + n_q / 2, n_top * 39 / 40, "Quantizations", ha="center", va="center", fontsize=7, fontweight="bold", zorder=7)
    rot = xrot if xrot is not None else (45 if len(vars_) > 10 else 0); sep = " " if rot else "\n"
    ax.set_yticks(np.arange(n_rows) + 0.5); ax.set_yticklabels(([short_model(o) for o in origs] + ["Variation Average"]) if ylabels else [], fontsize=6)
    ax.set_xticks(np.arange(n_cols) + 0.5); ax.set_xticklabels(["LLM Average"] + [fmt_var(v).replace("\n", sep) for v in vars_], fontsize=5.5 if rot < 80 else 4.8, rotation=rot, ha="right" if 0 < rot < 80 else "center")
    if ylabels: ax.get_yticklabels()[-1].set_weight("bold")
    ax.get_xticklabels()[0].set_weight("bold")
    ax.set_xlabel("Variation", fontsize=8, fontweight="bold"); ax.set_ylabel("LLM" if ylabels else "", fontsize=8, fontweight="bold"); ax.tick_params(length=2)
    if title: ax.set_title(title, fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    return fig, ax


# ------------------------------------------------------------------------------------------------------------------------------------------
# misc helpers
# ------------------------------------------------------------------------------------------------------------------------------------------
def tex_table(rows, header, caption_comment=""):
    """minimal booktabs table (list of rows of strings)"""
    cols = "l" + "r" * (len(header) - 1)
    lines = [f"% {caption_comment}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"] + [" & ".join(map(str, r)) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------------------------------
# output directories shared by the build steps (see make_paper.py for the order)
# ------------------------------------------------------------------------------------------------------------------------------------------
def closed_dir():
    """<out>/closed : numbers.json of the closed-set build (closed_build.py)"""
    return out_dir("closed")


def fig_dir():
    """<out>/figures : every final figure (pdf/png), its JSON sidecar and the TeX fragments the numbers step reads"""
    return out_dir("figures")

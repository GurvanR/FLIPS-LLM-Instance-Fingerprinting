#!/usr/bin/env python3
"""make_paper.py - regenerate the paper's figures, tables and quoted numbers from the results archive (one driver, fixed build order);
Figure 3 (white-box log-probability analysis, data not in the archive) and the LLMmap bars of Figure 1 (constants) are not regenerated (paper/README.md).

    python paper/make_paper.py                       # data: $FLIPS_DATA (default data/), output: $FLIPS_OUT (default paper/out/)
    python paper/make_paper.py --data /path/to/unpacked/archive --out /tmp/paper_out --check
    python paper/make_paper.py --only closed_build,figures.distance2     # a subset (keeps the order; later steps may need earlier outputs)

Needs numpy, scipy and matplotlib only; about 20 seconds on a CPU. Outputs under <out>/:
  closed/          closed-set build (numbers.json, ...)           figures/    every figure (pdf, png), its JSON sidecar, TeX fragments
  numbers.tex      the \\n... macros of the text, each read from a result file (written last)

Build order matters: several steps guard their inputs against the closed-set build's numbers.json or read the sidecars of earlier steps
(attribute numbers before the attribute figure, the operating-point table before the open-set heatmap, every sidecar before numbers.tex).
`paper/check.py` compares the output with `paper/expected/`.
"""
import argparse, importlib, os, sys, time, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [REPO] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]   # run as a script: use the package, not paper/ as import root

# (module below paper/, what it produces) in build order
STEPS = [
    ("closed_build", "closed-set build: numbers.json, pair-accuracy histogram, token-pair table, classifier comparison"),
    ("figures.feature_importance", "feature-importance figure"),
    ("figures.closed_heatmaps", "closed-set heatmaps (shortlist, full)"),
    ("figures.attribute_errors", "attribute figure (how often each attribute is missed) and numbers"),
    ("figures.closed_pr", "closed-set precision / recall against confidence"),
    ("figures.showcase", "abliterated-instance showcase"),
    ("figures.zero_one_vs_pairs", "pair 0-1 against the other pairs (NIST)"),
    ("stats.openset_operating_point", "open-set operating-point table (json)"),
    ("figures.accuracy_vs_queries", "accuracy against N_t (closed, open, LLMmap-IF)"),
    ("figures.unseen_pr", "Unseen precision / recall trade-off"),
    ("figures.detector_scores", "detector score distribution"),
    ("figures.open_heatmap", "open-set per-instance heatmap"),
    ("stats.absorption", "absorption numbers"),
    ("figures.distance2", "open set with attribute distance >= 2"),
    ("figures.truncation", "first-100-bits truncation"),
    ("figures.corpus", "discard rate and generated-length distribution"),
    ("stats.llmmap_curve", "LLMmap-IF baseline table"),
    ("stats.make_numbers", "numbers.tex"),
]


def run(modules, verbose=True):
    """run the given steps in order; returns [(module, seconds, error or None)]"""
    results = []
    for mod, what in STEPS:
        if modules is not None and mod not in modules:
            continue
        t0 = time.time()
        if verbose:
            print(f"== {mod}: {what}", flush=True)
        try:
            importlib.import_module(f"paper.{mod}").main([])
            err = None
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            print(err, file=sys.stderr)
        results.append((mod, time.time() - t0, err))
        if err:
            break
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data root holding results/ (default: $FLIPS_DATA or <repo>/data)")
    ap.add_argument("--out", help="output root (default: $FLIPS_OUT or paper/out)")
    ap.add_argument("--only", help="comma list of steps (default: all; see STEPS)")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--check", action="store_true", help="run paper/check.py on the output afterwards")
    a = ap.parse_args(argv)
    if a.list:
        for mod, what in STEPS:
            print(f"{mod:34s} {what}")
        return 0
    if a.data:
        os.environ["FLIPS_DATA"] = os.path.abspath(a.data)
    if a.out:
        os.environ["FLIPS_OUT"] = os.path.abspath(a.out)
    only = set(a.only.split(",")) if a.only else None
    if only and not only <= {m for m, _ in STEPS}:
        raise SystemExit(f"unknown step(s): {sorted(only - {m for m, _ in STEPS})}")
    res = run(only)
    print("\n".join(f"{m:34s} {'FAILED' if e else 'ok':6s} {s:6.1f}s" for m, s, e in res))
    if any(e for _, _, e in res):
        return 1
    print(f"all {len(res)} steps ok in {sum(s for _, s, _ in res):.0f}s; output: {os.environ.get('FLIPS_OUT') or os.path.join(REPO, 'paper', 'out')}")
    if a.check:
        from paper import check
        return check.main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

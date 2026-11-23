"""closed_build.py - closed-set build of the paper: the figures, tables and numbers drawn from the stored hierarchical closed-set results, and the classifier comparison.

  python -m paper.closed_build [--which main|clf|all] [--only fig15,table3,...] [--nr 300] [--nt 32] [--strict]

  main  results/closed_set/hierarchical  ->  <out>/closed/numbers.json (the numbers of the build, read by the figure scripts as a consistency guard and by the numbers step),
                                             <out>/figures/fig15_pair_accuracy_hist.{pdf,png} and table3_pairs.tex
  clf   results/closed_set/classifier_comparison  ->  <out>/figures/clf/   fig09 + numbers.json
Temperature-scaled calibration, N_r 300, N_t 32 (the paper's configuration). A figure whose inputs are missing is skipped with a reason (exit 0 unless --strict and a
figure FAILED). numbers.json holds the figures of the run only: a partial --only run rewrites it."""
import argparse, json, os, sys, time, traceback
from .dataio import CLF_CMP, Ctx, R, Skip, closed_dir, fig_dir, style
from .lib import closed as FC
from .lib.showcase import showcase_numbers


# id -> (build, function, paper element)
REGISTRY = [("fig09", "clf", FC.fig09, "classifier comparison"),
            ("fig13", "main", FC.calibration, "closed-set precision / recall numbers (temperature-scaled confidence)"),
            ("fig15", "main", FC.fig15, "per-pair accuracy histogram"),
            ("table3", "main", FC.table3, "top / bottom token pairs"),
            ("showcase", "main", showcase_numbers, "abliterated showcase numbers")]


def _ctx(build, nr, nt):
    if build == "main": return Ctx(closed_dir(), nr=nr, nt=nt)
    return Ctx(os.path.join(fig_dir(), "clf"), tree=R(*CLF_CMP), cells_sub="xgboost_cells", nr=nr, nt=nt)


def _run(build, want, nr, nt):
    """draw the registered figures of one build that are in `want`; returns (ctx, {id: status})"""
    ctx = _ctx(build, nr, nt); status = {}
    for fid, b, fn, what in REGISTRY:
        if b != build or fid not in want: continue
        before = len(ctx.produced)
        try: fn(ctx); status[fid] = dict(status="ok", element=what, files=ctx.produced[before:])
        except Skip as e: status[fid] = dict(status="skipped", element=what, reason=str(e)); print(f"[skip] {fid}: {e}")
        except Exception as e: status[fid] = dict(status="FAILED", element=what, reason=repr(e)); print(f"[FAIL] {fid}"); traceback.print_exc()
    num = dict(ctx.num, _build=dict(nr=nr, nt=nt))
    with open(os.path.join(ctx.out, "numbers.json"), "w") as f: json.dump(num, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return ctx, status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=("main", "clf", "all"), default="all", help="main = hierarchical closed-set figures, clf = classifier comparison")
    ap.add_argument("--only", default=None, help="comma list of ids (default all): " + ",".join(r[0] for r in REGISTRY)); ap.add_argument("--nr", type=int, default=300); ap.add_argument("--nt", type=int, default=32)
    ap.add_argument("--strict", action="store_true", help="exit 1 if a figure FAILED (skips are fine)")
    a = ap.parse_args([] if argv is None else argv); style(); t0 = time.time()
    want = a.only.split(",") if a.only else [r[0] for r in REGISTRY]; unknown = [w for w in want if w not in [r[0] for r in REGISTRY]]
    if unknown: ap.error(f"unknown ids {unknown}")
    status = {}
    for build in (("main", "clf") if a.which == "all" else (a.which,)):
        if not any(b == build and fid in want for fid, b, _, _ in REGISTRY): print(f"[{build}] nothing to build for --only {a.only}"); continue
        ctx, st = _run(build, want, a.nr, a.nt); status.update(st)
        print(f"[{build}] -> {ctx.out}")
    n = {k: sum(1 for v in status.values() if v["status"] == k) for k in ("ok", "skipped", "FAILED")}; print(f"\n{n} in {time.time() - t0:.0f}s")
    for k, v in status.items():
        if v["status"] != "ok": print(f"  {v['status']:8s} {k}: {v['reason']}")
    return 1 if a.strict and n["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

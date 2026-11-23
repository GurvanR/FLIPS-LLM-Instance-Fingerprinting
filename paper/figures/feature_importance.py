"""feature_importance.py - feature-importance figure of the paper: grouped XGBoost gain of the per-LLM classifiers of the instance classifier (importance/*.json of the
hierarchical closed-set tree: one gain vector per (token pair, split), 31 pairs x 5 splits), drawn by lib.closed.fig10. Labels and groups of the 57 features: constants/feature_labels.json (the groups of the paper's table of statistics).
  python -m paper.figures.feature_importance
Writes <out>/figures/fig10_feature_importance.{pdf,png,json}."""
import argparse, json, os, sys
from ..dataio import Ctx, fig_dir, style
from ..lib import closed as FC


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args([] if argv is None else argv); style()
    out = fig_dir(); ctx = Ctx(out); FC.fig10(ctx); num = dict(ctx.num["fig10"], script="paper/figures/feature_importance.py")
    with open(os.path.join(out, "fig10_feature_importance.json"), "w") as f: json.dump(num, f, indent=1)
    print(json.dumps({"group_share": num["group_share"], "n_features": num["n_features"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

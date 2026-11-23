#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Gate A: out-of-the-box smoke run (CPU only, no download).

Runs ``XP_configs/smoke.yaml`` on the in-repo fixture ``tests/fixtures/answers.parquet`` through the experiment
driver (feature grid -> hierarchical instance classifier, 2 token pairs x 2 splits) and checks the result:
every cell present, accuracy above chance at every N_t and higher at the largest N_t than with one query.

    python scripts/gates/gate_a.py [--out DIR]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="results directory (default: a temporary directory)")
    a = ap.parse_args()
    from audit_llm.pipeline import main as pipeline_main

    out = Path(a.out or tempfile.mkdtemp(prefix="flips_gate_a_"))
    t0 = time.time()
    pipeline_main(["run", "--config", str(REPO / "XP_configs" / "smoke.yaml"), "--answers", str(REPO / "tests" / "fixtures" / "answers.parquet"),
                   "--out", str(out), "--n-jobs", "4"])
    summary = json.loads((out / "summary.json").read_text())
    n_classes = summary["cells"][0]["n_classes"]
    problems = []
    if len(summary["cells"]) != 4:
        problems.append(f"expected 4 cells, got {len(summary['cells'])}")
    accs = {n: v["mean_accuracy"] for n, v in summary["by_n_t"].items()}
    if any(v is None or v <= 1.0 / n_classes for v in accs.values()):
        problems.append(f"accuracy at chance level: {accs}")
    if accs[max(accs, key=int)] <= accs["1"]:
        problems.append(f"accuracy does not grow with N_t: {accs}")
    print(f"gate A: {len(summary['cells'])} cells, accuracy by N_t {({k: round(v, 3) for k, v in accs.items()})}, {time.time() - t0:.0f} s -> {out}")
    if problems:
        print("gate A FAILED: " + "; ".join(problems))
        return 1
    print("gate A passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

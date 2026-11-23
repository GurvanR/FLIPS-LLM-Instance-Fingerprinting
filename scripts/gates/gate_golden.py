#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Gate golden: numeric regression of the smoke run against its recorded golden summary (CPU only, no download).

Runs ``XP_configs/smoke.yaml`` on the fixture and compares every cell's accuracy at every N_t, and the number of
samples, with ``tests/fixtures/reference/smoke_summary.json``. With the pinned environment (xgboost 3.2, 4 threads
for the global classifier) the run is deterministic and must match exactly; ``--tolerance`` allows a band (in
accuracy points, e.g. 0.02) for other library versions. ``tests/test_closed_set_reference.py`` compares the
per-sequence outputs of the method with stored reference outputs.

    python scripts/gates/gate_golden.py [--tolerance 0.0] [--update]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "fixtures" / "reference" / "smoke_summary.json"
sys.path.insert(0, str(REPO / "src"))


def cell_table(summary: dict) -> dict:
    return {f"{c['token_pair']}_s{c['split']}": dict(accuracy=c["accuracy"], n_test=c["n_test"]) for c in summary["cells"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--update", action="store_true", help="rewrite the golden summary from this run")
    a = ap.parse_args()
    from audit_llm.pipeline import main as pipeline_main

    out = Path(tempfile.mkdtemp(prefix="flips_gate_golden_"))
    pipeline_main(["run", "--config", str(REPO / "XP_configs" / "smoke.yaml"), "--answers", str(REPO / "tests" / "fixtures" / "answers.parquet"),
                   "--out", str(out), "--n-jobs", "4"])
    got = cell_table(json.loads((out / "summary.json").read_text()))
    if a.update:
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        print(f"golden summary written: {GOLDEN}")
        return 0
    want = json.loads(GOLDEN.read_text())
    problems = []
    if set(got) != set(want):
        problems.append(f"cells differ: {sorted(set(got) ^ set(want))}")
    for cell in sorted(set(got) & set(want)):
        if got[cell]["n_test"] != want[cell]["n_test"]:
            problems.append(f"{cell}: n_test {got[cell]['n_test']} != {want[cell]['n_test']}")
        for n_t, acc in want[cell]["accuracy"].items():
            diff = abs(got[cell]["accuracy"].get(n_t, float("nan")) - acc)
            if not diff <= a.tolerance:
                problems.append(f"{cell} N_t={n_t}: {got[cell]['accuracy'].get(n_t)} vs golden {acc}")
    if problems:
        print("gate golden FAILED:\n  " + "\n  ".join(problems))
        return 1
    print(f"gate golden passed: {len(got)} cells, tolerance {a.tolerance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

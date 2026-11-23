# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""End-to-end smoke test of the experiment driver on the fixture (XP_configs/smoke.yaml, one split)."""

import json
from pathlib import Path

from audit_llm.experiment_config_schema import load_experiment_config
from audit_llm.pipeline import run_experiment

REPO = Path(__file__).resolve().parents[1]


def test_closed_set_smoke(fixture_grid, tmp_path):
    cfg = load_experiment_config(REPO / "XP_configs" / "smoke.yaml")
    cfg["protocol"]["splits"] = [0]
    cfg["xp_name"] = "smoke"
    summary = run_experiment(cfg, fixture_grid, str(tmp_path))
    assert set(summary["by_n_t"]) == {"1", "2", "4", "8"} and summary["by_n_t"]["8"]["n_cells"] == 2
    acc = [summary["by_n_t"][n]["mean_accuracy"] for n in ("1", "8")]
    assert 1 / 22 < acc[0] < acc[1] <= 1.0  # better than chance, and more queries help
    cell = json.loads((tmp_path / "cells" / "ali-yg_s0.json").read_text())
    r = cell["by_n_t"]["4"]
    assert len(r["labels"]) == len(r["predictions"]) == r["n_samples"] > 0 and set(r["labels"]) == set(range(22))

# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Experiment config schema: validation, defaults, duplicate keys, shipped configs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from audit_llm.experiment_config_schema import ExperimentConfig, load_experiment_config

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path, text, name="c.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_defaults_are_the_paper_protocol(tmp_path):
    cfg = load_experiment_config(write(tmp_path, "experiment_fun: closed_set\n"))
    assert cfg["protocol"] == dict(splits=[0, 1, 2, 3, 4], test_per_class=128, inner_val=40, n_train=300, seed=0)
    assert cfg["classifier"]["global_rounds"] == 357 and cfg["batch_sizes"] == [1, 2, 4, 8, 16, 32, 64]
    assert cfg["open_set"] is None


def test_open_set_section_defaults(tmp_path):
    cfg = load_experiment_config(write(tmp_path, "experiment_fun: open_set\n"))
    assert cfg["open_set"]["n_unseen"] == 23 and cfg["open_set"]["target_recall"] == 0.5 and len(cfg["open_set"]["cells"]) == 5


def test_duplicate_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate YAML key 'token_pairs'"):
        load_experiment_config(write(tmp_path, "experiment_fun: closed_set\ntoken_pairs: []\ntoken_pairs: [a]\n"))


def test_unknown_keys_and_functions_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_experiment_config(write(tmp_path, "experiment_fun: closed_set\nunknown_key: 1\n"))
    with pytest.raises(ValidationError):
        load_experiment_config(write(tmp_path, "experiment_fun: something_else\n"))
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(dict(experiment_fun="closed_set", open_set={}))


@pytest.mark.parametrize("path", sorted(str(p.relative_to(REPO)) for p in (REPO / "XP_configs").rglob("*.yaml")))
def test_shipped_configs_validate(path):
    cfg = load_experiment_config(REPO / path)
    assert cfg["experiment_fun"] in ("closed_set", "open_set")

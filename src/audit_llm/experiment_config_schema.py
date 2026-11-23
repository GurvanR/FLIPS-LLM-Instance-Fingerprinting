# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Pydantic schema of the experiment YAML files (``XP_configs/closed_set.yaml``, ``open_set.yaml``, ``smoke.yaml``).

Duplicate YAML keys and unknown keys are rejected; omitted keys take the paper's values.

Usage
-----
>>> from audit_llm.experiment_config_schema import load_experiment_config
>>> xp_config = load_experiment_config("XP_configs/closed_set.yaml")   # validated dict
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Custom YAML loader that rejects duplicate keys
# ---------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that raises on duplicate mapping keys."""

    pass


def _unique_key_constructor(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """Construct a mapping, raising ValueError on duplicate keys."""
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"Duplicate YAML key '{key}' found at line {key_node.start_mark.line + 1}"
            )
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_key_constructor,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProtocolConfig(_Strict):
    """Rows of a (token pair, split) cell (see ``audit_llm.Classification.protocol``)."""

    splits: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    test_per_class: int = 128
    inner_val: int = 40
    n_train: int = 300
    seed: int = 0


class ClassifierConfig(_Strict):
    """XGBoost classifiers of the instance classifier (see ``audit_llm.Classification.xgb``)."""

    global_rounds: int = 357
    n_jobs: int = 4
    round_selection_patience: int = 60
    round_selection_max: int = 1000


class OpenSetConfig(_Strict):
    """Open-set protocol and Unseen detector (see ``audit_llm.Classification.unseen_detector``)."""

    cells: List[Tuple[int, int]] = Field(default_factory=lambda: [(0, 0), (1, 1), (2, 2), (0, 3), (3, 4)])  # (split, draw)
    n_unseen: int = 23
    n_folds: int = 10
    held_out_per_class: int = 40
    unseen_rows_per_instance: int = 40
    known_batches_per_instance: int = 10
    unseen_batches_per_instance: int = 40
    min_rows_per_instance: int = 120
    unseen_weight_share: float = 0.097
    regularization_c: float = 0.1
    target_recall: float = 0.5


class ExperimentConfig(_Strict):
    """One experiment: the closed-set instance classifier or the open-set evaluation, over token pairs and splits."""

    experiment_fun: Literal["closed_set", "open_set"]
    xp_name: Optional[str] = None
    description: Optional[str] = None
    token_pairs: List[str] = Field(default_factory=list)  # [] = every token pair of the answers
    scenario: Optional[str] = None  # restrict the instances to a scenario (name in config/scenarios/ or path to its YAML)
    min_seq_length: int = 100
    batch_sizes: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    open_set: Optional[OpenSetConfig] = None
    n_workers: int = 1

    @model_validator(mode="after")
    def _open_set_section(self) -> "ExperimentConfig":
        if self.experiment_fun == "open_set" and self.open_set is None:
            self.open_set = OpenSetConfig()
        if self.experiment_fun == "closed_set" and self.open_set is not None:
            raise ValueError("an 'open_set' section is only valid with experiment_fun: open_set")
        if any(b < 1 for b in self.batch_sizes):
            raise ValueError("batch_sizes must be >= 1")
        return self


def load_experiment_config(path: Union[str, Path]) -> dict:
    """Load and validate an experiment YAML file; returns the validated config as a plain dict (defaults filled)."""
    with open(path) as f:
        raw = yaml.load(f, Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return ExperimentConfig.model_validate(raw).model_dump()

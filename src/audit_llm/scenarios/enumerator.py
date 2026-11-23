# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Scenario enumerator of the analysis layer.

:func:`build_instances` gives *which* model × variation :class:`~audit_llm.scenarios.variation.Instance` objects a
scenario materialises and what their classification labels are. It builds the union of:

- ``base_models`` × ``base_variations`` (temperature × system_prompt_idx);
- ``quantized_base_models`` × ``quantized_variations`` (temperature ×
  system_prompt_idx × quantization);
- ``abliterated_models`` (different weights repos) × ``abliteration_variations``
  (pinned temperature=1.0 / system_prompt_idx=-1).

The generation path (``scripts/Run_Inferences.py``) does its own expansion, one model and one quantization per
run. This module is used by the analysis layer only.

This module imports :class:`Instance`, which pulls in the vLLM backend module.
The package ``__init__`` deliberately stays light; import ``build_instances`` from this module explicitly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL
from audit_llm.scenarios.schema import Scenario, VariationGroup
from audit_llm.scenarios.variation import Instance


def _expand_grid(groups: Tuple[VariationGroup, ...]) -> List[Dict]:
    """Flatten a variation grid (tuple of groups) to a deduped list of cells.

    Each group expands to its cartesian product (:meth:`VariationGroup.cartesian`);
    the grid is the *union* across groups, preserving first-occurrence order.
    """
    cells: List[Dict] = []
    seen = set()
    for group in groups:
        for cell in group.cartesian():
            key = tuple(sorted(cell.items()))
            if key not in seen:
                seen.add(key)
                cells.append(cell)
    return cells


def build_instances(scenario: Scenario) -> List[Instance]:
    """Enumerate every :class:`Instance` a scenario materialises (union, deduped).

    These are the classes of the classification and their labels. The order is: all base instances, then all
    quantized, then all abliterated; cells in grid order, models in scenario-declared order.
    """
    instances: List[Instance] = []

    # Base models × base grid.
    for cell in _expand_grid(scenario.base_variations):
        for base in scenario.base_models:
            instances.append(
                Instance(
                    base_model=base,
                    temperature=cell["temperature"],
                    system_prompt_idx=cell["system_prompt_idx"],
                )
            )

    # Quantized base models × quant grid (× quantization levels carried in the cell).
    for cell in _expand_grid(scenario.quantized_variations):
        quant = cell.get("quantization")
        for base in scenario.quantized_base_models:
            instances.append(
                Instance(
                    base_model=base,
                    temperature=cell["temperature"],
                    system_prompt_idx=cell["system_prompt_idx"],
                    quantization=quant,
                )
            )

    # Abliterated repos × abliteration grid (pinned). The scenario lists the abliterated repos directly;
    # map each back to its base so that the Instance has ``storage_name`` = repo and ``label`` = ``{repo}_temp-<t>_sp-<sp>``.
    for cell in _expand_grid(scenario.abliteration_variations):
        for repo in scenario.abliterated_models:
            instances.append(
                Instance(
                    base_model=_abliterated_repo_to_base(repo),
                    temperature=cell["temperature"],
                    system_prompt_idx=cell["system_prompt_idx"],
                    abliterated=True,
                )
            )

    # Union, not multiset: dedup preserving order (Instance is a frozen dataclass).
    return list(dict.fromkeys(instances))


def _abliterated_repo_to_base(repo: str) -> str:
    """Map an abliterated HF repo id back to its base model."""
    try:
        return ABLITERATED_MODELS_MAP_TO_ORIGINAL[repo]
    except KeyError:
        raise ValueError(
            f"Unknown abliterated repo {repo!r}; declared abliterated repos are "
            f"{sorted(ABLITERATED_MODELS_MAP_TO_ORIGINAL)}."
        ) from None

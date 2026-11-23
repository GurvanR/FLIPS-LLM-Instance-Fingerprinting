# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Typed declarative schema for FLIPS *scenarios*.

A **scenario** selects which model × variation instances the analysis layer
materialises. It is expressed over **four variation axes**:

- ``temperature``       — sampling-request field
- ``system_prompt_idx`` — sampling-request field
- ``quantization``      — model transform: a vLLM engine kwarg (same weights)
- ``abliteration``      — model transform: a *different weights repo*

A variation grid is a list of groups (dicts of axis lists):

- **cartesian within a group** — one ``VariationGroup`` (one dict) expands to the
  cartesian product of its populated axis lists;
- **union across groups** — a grid (the list of groups) is the union of its
  groups' products.

Example, the paper's base grid
``[{temperature:[0.4,0.6,0.8,1.0], system_prompt_idx:[-1]}, {temperature:[1.0], system_prompt_idx:[0,3,6,7]}]``
is two ``VariationGroup`` objects: the first yields 4×1 cells, the second 1×4,
unioned to 8 distinct ``(temperature, system_prompt_idx)`` pairs.

A quantized grid crosses ``quantization`` (e.g. ``quantization: [fp8, bitsandbytes_int4]``); abliterated
models are listed in ``abliterated_models``, and the shipped groups leave the ``abliteration`` axis empty.

This module describes a scenario; ``audit_llm.scenarios.enumerator`` enumerates
its instances (see ``config/scenarios/README.md`` for the request files).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# The four variation axes, in canonical order.
AXES: Tuple[str, ...] = (
    "temperature",
    "system_prompt_idx",
    "quantization",
    "abliteration",
)


@dataclass(frozen=True)
class VariationGroup:
    """One cartesian cell-set over the four variation axes.

    Each axis is a tuple of allowed values; an empty tuple means the axis is not
    part of this group. The group expands (:meth:`cartesian`) to the cartesian
    product of its *populated* axes. A grid (a list of groups) is the union of
    its groups' products.
    """

    temperature: Tuple[float, ...] = ()
    system_prompt_idx: Tuple[int, ...] = ()
    quantization: Tuple[str, ...] = ()
    abliteration: Tuple[str, ...] = ()

    def populated_axes(self) -> Dict[str, Tuple]:
        """Return the ``{axis: values}`` mapping for the non-empty axes only."""
        return {axis: getattr(self, axis) for axis in AXES if getattr(self, axis)}

    def cartesian(self) -> List[Dict]:
        """Expand this group to the cartesian product over its populated axes.

        Returns a list of dicts, one per cell, keyed by the populated axis names.
        An entirely empty group yields a single empty cell ``[{}]``.
        """
        axes = self.populated_axes()
        if not axes:
            return [{}]
        names = list(axes.keys())
        return [dict(zip(names, combo)) for combo in itertools.product(*axes.values())]


@dataclass(frozen=True)
class Scenario:
    """A declarative selection of model × variation instances.

    Three model classes, each with its own variation grid:

    - ``base_models`` crossed over ``base_variations`` (temperature × system_prompt_idx);
    - ``quantized_base_models`` crossed over ``quantized_variations`` (temperature ×
      system_prompt_idx × quantization);
    - ``abliterated_models`` (different weights repos) crossed over
      ``abliteration_variations`` (pinned to temperature=1.0 / system_prompt_idx=-1).

    Construction is normally via :func:`audit_llm.scenarios.loader.load_scenario`,
    which validates every model selection against ``config/models.yaml``.
    """

    name: str
    base_models: Tuple[str, ...]
    base_variations: Tuple[VariationGroup, ...]
    quantized_base_models: Tuple[str, ...] = ()
    quantized_variations: Tuple[VariationGroup, ...] = ()
    abliterated_models: Tuple[str, ...] = ()
    abliteration_variations: Tuple[VariationGroup, ...] = field(default_factory=tuple)

# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Definition consistency of the shipped ``main`` scenario.

``config/scenarios/main.yaml`` enumerates exactly 237 distinct instances with the
200 / 32 / 5 base / quantized / abliterated split, and the load-bearing label forms
(``@@fp8``, ``@@bitsandbytes_int4``, ``temp-<t>_sp-<sp>``) are produced byte-for-byte.

A schema-guard test confirms the loader rejects any abliteration cell outside the
pin (temp=1.0 / sp=-1).
"""

from __future__ import annotations

import pytest
import yaml

from audit_llm.scenarios.enumerator import build_instances
from audit_llm.scenarios.loader import load_scenario


def _split(instances):
    base = [i for i in instances if not i.abliterated and not i.is_quantized]
    quant = [i for i in instances if i.is_quantized]
    ablit = [i for i in instances if i.abliterated]
    return base, quant, ablit


# ---------------------------------------------------------------------------
# main.yaml — definition consistency: 237 = 200 / 32 / 5 + label forms
# ---------------------------------------------------------------------------


def test_main_total_and_split_is_237():
    instances = build_instances(load_scenario("main"))
    base, quant, ablit = _split(instances)
    assert len(instances) == 237 == len({i.label for i in instances})
    assert (len(base), len(quant), len(ablit)) == (200, 32, 5)


def test_main_label_forms_spot_check():
    instances = build_instances(load_scenario("main"))
    labels = {i.label for i in instances}

    # quantization @@ suffix uses the quant KEY, with a temp/sp variation suffix.
    assert "Qwen/Qwen2-7B-Instruct@@fp8_temp-1.0_sp--1" in labels
    assert "Qwen/Qwen2-7B-Instruct@@bitsandbytes_int4_temp-1.0_sp-0" in labels
    # abliterated label: the abliterated repo id with its (pinned) cell.
    assert "natong19/Qwen2-7B-Instruct-abliterated_temp-1.0_sp--1" in labels
    # a base variation label (note sp--1 for system_prompt_idx=-1).
    assert "CohereForAI/c4ai-command-r-plus_temp-1.0_sp--1" in labels


# ---------------------------------------------------------------------------
# Schema guard: abliteration cell outside temp=1.0 / sp=-1 is rejected
# ---------------------------------------------------------------------------


def _write_min_scenario(tmp_path, abliteration_variations):
    """Write a minimal registry + scenario whose abliteration grid is overridable."""
    registry = {
        "base_models": ["Qwen/Qwen2-7B-Instruct"],
        "quantized_variants": [],
        "abliterated_variants": [
            {
                "base": "Qwen/Qwen2-7B-Instruct",
                "abliterated_hf_id": "natong19/Qwen2-7B-Instruct-abliterated",
            }
        ],
    }
    scenario = {
        "name": "ablit_guard",
        "base_models": ["Qwen/Qwen2-7B-Instruct"],
        "base_variations": [{"temperature": [1.0], "system_prompt_idx": [-1]}],
        "abliterated_models": ["natong19/Qwen2-7B-Instruct-abliterated"],
        "abliteration_variations": abliteration_variations,
    }
    registry_path = tmp_path / "models.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    registry_path.write_text(yaml.safe_dump(registry))
    scenario_path.write_text(yaml.safe_dump(scenario))
    return scenario_path, registry_path


def test_abliteration_pin_is_satisfied_when_valid(tmp_path):
    scenario_path, registry_path = _write_min_scenario(
        tmp_path, [{"temperature": [1.0], "system_prompt_idx": [-1]}]
    )
    # The valid pinned cell loads cleanly.
    scenario = load_scenario(scenario_path, registry_path=registry_path)
    assert len(scenario.abliteration_variations) == 1


def test_abliteration_cell_with_bad_temperature_is_rejected(tmp_path):
    scenario_path, registry_path = _write_min_scenario(
        tmp_path, [{"temperature": [0.6], "system_prompt_idx": [-1]}]
    )
    with pytest.raises(ValueError, match="abliteration_variations"):
        load_scenario(scenario_path, registry_path=registry_path)


def test_abliteration_cell_with_bad_system_prompt_is_rejected(tmp_path):
    scenario_path, registry_path = _write_min_scenario(
        tmp_path, [{"temperature": [1.0], "system_prompt_idx": [0]}]
    )
    with pytest.raises(ValueError, match="abliteration_variations"):
        load_scenario(scenario_path, registry_path=registry_path)

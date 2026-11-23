# Scenarios

A scenario (`config/scenarios/<name>.yaml`) is a declarative file that selects which model x variation
**instances** the analysis uses as classes. It is read by `audit_llm.scenarios.load_scenario`, validated
against the model registry `config/models.yaml`, and expanded to instances by
`audit_llm.scenarios.enumerator.build_instances`. The scenario layer selects; the generation pipeline
([`inference.md`](inference.md)) creates the data. The YAML field contract (axes, grids, validation) is in
`config/scenarios/README.md`.

Two scenarios ship: `main` (the 237-instance benchmark) and `example_custom` (a 4-instance template that
runs on the in-repo fixture, see [`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md)).

## `main`: the 237 instances

`config/scenarios/main.yaml` lists 25 base LLMs, 4 quantized base LLMs, 5 abliterated repositories and three
variation grids. A grid is a list of groups; a group is the cartesian product of its axes and the grid is the
union of its groups.

| Group | Instances | Construction |
|-------|-----------|--------------|
| base | 200 | 25 LLMs x 8 cells |
| quantized | 32 | 4 LLMs x 2 levels (`fp8`, `bitsandbytes_int4`) x 4 cells |
| abliterated | 5 | 5 repositories x 1 cell |
| **total** | **237** | |

- Base cells (union of two groups of 4, not a 4 x 4 cross): `temperature` 0.4, 0.6, 0.8, 1.0 with
  `system_prompt_idx` -1; and `temperature` 1.0 with `system_prompt_idx` 0, 3, 6, 7.
- Quantized cells: `temperature` 0.6 and 1.0 with `system_prompt_idx` -1; and `temperature` 1.0 with
  `system_prompt_idx` 0 and 3; each crossed with the two levels ([`quantized-models.md`](quantized-models.md)).
- Abliterated cell: `temperature` 1.0, `system_prompt_idx` -1.

`tests/test_main_scenario_237.py` asserts that the enumerator gives 237 distinct instances
(200 base + 32 quantized + 5 abliterated). Check: `python -m pytest tests/test_main_scenario_237.py -q`.

The LLM lists are those of `config/models.yaml` (`base_models`, 25 entries); see
[`../provenance.md`](../provenance.md) for the list. The cells all exist in `datasets/Bits_Datasets/`
([`dataset.md`](dataset.md)).

## `example_custom`

`config/scenarios/example_custom.yaml` selects 2 registered LLMs (`CohereForAI/c4ai-command-r-plus`,
`google/gemma-2-9b-it`) in 2 cells (`temperature` 0.6 and 1.0, `system_prompt_idx` -1): 4 instances, all
present in `tests/fixtures/answers.parquet`. It is used by `XP_configs/example_custom/example_custom.yaml`.

## Instances and labels

An `Instance` (`scenarios/variation.py`) is a base model, a cell and optionally a quantization level or the
abliterated flag. Its names are exact strings that the analysis code parses:

| Form | Example | Notes |
|------|---------|-------|
| storage name, quantized | `Qwen/Qwen2-7B-Instruct@@fp8` | `@@` is `QUANTIZATION_SEPARATOR`; the suffix is the level key |
| storage name, abliterated | `natong19/Qwen2-7B-Instruct-abliterated` | the abliterated repository id |
| label | `CohereForAI/c4ai-command-r-plus_temp-1.0_sp--1` | storage name plus `temp-<t>_sp-<sp>`, temperature first; `sp--1` is `system_prompt_idx` -1; `1.0` keeps its `.0` |

The label is the name of the instance in the feature grid and in the stored results
(`features.grid.class_name` of the storage name and the cell). `tests/test_variation_resolver.py` and
`tests/test_main_scenario_237.py` pin these strings byte for byte, and `xp_tools/model_filtering.py`
(`full_var_model_name_to_original_model_name`, `full_var_model_name_to_var_name`) parses them. They are
identifiers of stored results: keep them unchanged when you add scenarios.

## Abliterated instances

An abliterated instance uses a different weights repository, so it is selected by repository id
(`abliterated_models`, ids declared as `abliterated_hf_id` in `config/models.yaml`) and is not an axis crossed
with a base model. `config/models.yaml` also declares the base model each repository is attached to
(used to label rows of analysis tables):

| Abliterated repository | Declared base model |
|------------------------|---------------------|
| `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3` | `meta-llama/Llama-3.1-8B-Instruct` |
| `failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3` | `abacusai/Smaug-Llama-3-70B-Instruct` |
| `natong19/Qwen2-7B-Instruct-abliterated` | `Qwen/Qwen2-7B-Instruct` |
| `failspy/Phi-3-medium-4k-instruct-abliterated-v3` | `microsoft/Phi-3-medium-4k-instruct` |
| `dphn/dolphin-2.9.2-Phi-3-Medium-abliterated` | `microsoft/Phi-3-medium-128k-instruct` |

Their requests are in `FLiPS_ICML_uncensored.csv`. The hierarchical classifier treats each abliterated
instance as an LLM of its own (`llm_name` of the instance name), so none shares a per-LLM classifier with a
base model.

**No-crossing rule.** The data holds one cell per abliterated repository. `load_scenario` raises a
`ValueError` for an `abliteration_variations` group whose temperature is not 1.0, whose
`system_prompt_idx` is not -1, or that sets the `quantization` or `abliteration` axis. An instance cannot be
both abliterated and quantized (`Variation` and `Instance` raise).

## Request files

The scenario layer reads and validates. The request CSVs of `datasets/Bits_Datasets/` are inputs shared by
generation and analysis; the `Index` values and (temperature, system prompt) cells they define are what the
stored answers refer to, and a scenario selects cells among them. A scenario is validated against
`config/models.yaml`: a scenario naming an instance that has no answers loads, and the restriction of the grid
to its instances raises `KeyError: instances absent from the grid`.

## Use in an experiment

An experiment config selects a scenario with `scenario: <name or path>`
(`audit_llm.pipeline.scenario_instances`, then `Grid.select_classes`). A config without `scenario:` uses every
instance present in the answers; `XP_configs/closed_set.yaml` and `XP_configs/open_set.yaml` set `scenario: main`
(remove it from a copy when the answers hold more cells than the benchmark). See
[`configuration.md`](configuration.md) and [`experiment.md`](experiment.md).

## Related

[`dataset.md`](dataset.md) - [`quantized-models.md`](quantized-models.md) - [`inference.md`](inference.md) -
[`../provenance.md`](../provenance.md) - [`conventions.md`](conventions.md)

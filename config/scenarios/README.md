# Scenarios — the model × variation selection contract

A **scenario** is a declarative YAML file (`config/scenarios/<name>.yaml`) that
selects which model × variation **instances** the analysis layer materialises.
It is loaded and validated by
[`audit_llm.scenarios.load_scenario`](../../src/audit_llm/scenarios/loader.py)
against the model registry [`config/models.yaml`](../models.yaml).

> **The scenario layer governs model × variation *selection*.** It reads and
> validates scenario files; inference and features are computed elsewhere. See
> **Request files** below.

## The four axes

Every variation is a point in a four-axis space:

| Axis | Kind | Meaning |
|---|---|---|
| `temperature` | sampling-request field | sampling temperature passed in the request |
| `system_prompt_idx` | sampling-request field | which system prompt (index; `-1` = none) |
| `quantization` | **model transform** | a vLLM **engine kwarg** applied to the *same* base weights (e.g. `fp8`, `bitsandbytes_int4`) |
| `abliteration` | **model transform** | a **different weights repo** (the abliterated model is a distinct Hugging Face id, not a kwarg) |

`temperature` and `system_prompt_idx` change *how a model is sampled*.
`quantization` and `abliteration` produce **a transformed copy of a base model**:

- **quantization** keeps the same weights and sets a vLLM engine kwarg; the
  quantized grid crosses it as an axis;
- **abliteration** points at *different weights* (a separate repo), listed in
  `abliterated_models`; the shipped scenarios leave the `abliteration` axis of
  their variation groups empty.

## Cartesian within a group, union across groups

A *variation grid* is a **list of groups** (dicts). Each group lists allowed
values per axis. Semantics:

- **cartesian within a group** — one group expands to the cartesian product of
  its populated axes;
- **union across groups** — the grid is the union of its groups' products.

Example (the paper's base grid: `4×1 ∪ 1×4 = 8` distinct `(temp, sp)` cells):

```yaml
base_variations:
  - temperature: [0.4, 0.6, 0.8, 1.0]
    system_prompt_idx: [-1]
  - temperature: [1.0]
    system_prompt_idx: [0, 3, 6, 7]
```

## YAML keys

```yaml
name: main                       # optional; defaults to the file stem

base_models:                     # HF ids; each must exist in config/models.yaml `base_models`
  - CohereForAI/c4ai-command-r-plus
  # ...
base_variations:                 # grid for base models (temperature × system_prompt_idx)
  - {temperature: [0.4, 0.6, 0.8, 1.0], system_prompt_idx: [-1]}
  - {temperature: [1.0], system_prompt_idx: [0, 3, 6, 7]}

quantized_base_models:           # HF ids; each must be a declared quantized_variants base
  - Qwen/Qwen2-7B-Instruct
quantized_variations:            # grid for quantized models (temperature × system_prompt_idx × quantization)
  - {temperature: [0.6, 1.0], system_prompt_idx: [-1], quantization: [fp8, bitsandbytes_int4]}
  - {temperature: [1.0], system_prompt_idx: [0, 3], quantization: [fp8, bitsandbytes_int4]}

abliterated_models:              # abliterated repo HF ids; each must be a declared abliterated_variants `abliterated_hf_id`
  - failspy/Meta-Llama-3-8B-Instruct-abliterated-v3
abliteration_variations:         # MUST be pinned (see below)
  - {temperature: [1.0], system_prompt_idx: [-1]}
```

## Validation against the registry

`load_scenario` raises a clear `ValueError` (naming the offender and the
registry path) when:

- a `base_models` entry is not in the registry `base_models`;
- a `quantized_base_models` entry has no `quantized_variants` declared, or a
  `quantization` level requested in `quantized_variations` is not a declared
  `(base, quantization)` pair;
- an `abliterated_models` entry is not a declared `abliterated_hf_id`;
- a grid is structurally malformed (not a list of mappings, or an unknown axis key).

### Abliteration is pinned — no crossing

The paper's corpus contains abliterated runs **only** at `temperature=1.0,
system_prompt_idx=-1`. The loader therefore **rejects** any
`abliteration_variations` group whose `temperature ≠ 1.0`, whose
`system_prompt_idx ≠ -1`, or that sets a `quantization` / `abliteration` axis.
Abliterated instances are never crossed beyond that pinned cell.

## Request files

The `datasets/Bits_Datasets/*.csv` files define the generation requests: their `Index` values and
(temperature, system prompt) cells are what the stored answers refer to. They are inputs of the generation and
of the analysis, and a scenario selects model × variation instances among them.

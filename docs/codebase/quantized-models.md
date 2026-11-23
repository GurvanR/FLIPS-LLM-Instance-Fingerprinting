# Quantized instances

The benchmark holds 32 quantized instances: 4 base LLMs, each served with two quantization levels (`fp8`
and `bitsandbytes_int4`) in 4 cells. A quantized instance uses the weights of its base LLM; only an
engine argument of vLLM differs. This page covers how such instances are generated, named, parsed and
selected.

## Supported models

`config/models.yaml` (`quantized_variants`) declares 8 entries, the base models below times the two levels:

| Base model | Levels |
|------------|--------|
| `Qwen/Qwen2-7B-Instruct` | `fp8`, `bitsandbytes_int4` |
| `meta-llama/Meta-Llama-3-8B-Instruct` | `fp8`, `bitsandbytes_int4` |
| `microsoft/Phi-3-mini-4k-instruct` | `fp8`, `bitsandbytes_int4` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `fp8`, `bitsandbytes_int4` |

Cells of the benchmark (`config/scenarios/main.yaml`): `system_prompt_idx` `-1` at temperatures 0.6 and 1.0,
and system prompts 0 and 3 at temperature 1.0. 4 models x 2 levels x 4 cells = 32 instances
([`scenarios.md`](scenarios.md)).

## Generation

`scripts/Run_Inferences.py --quantization <level>` accepts the keys of `QUANTIZATION_VLLM_PARAMS` in
`src/audit_llm/LLM_Classes/vLLM_Classes.py`:

| Level | vLLM engine argument |
|-------|----------------------|
| `no_quantized` | none (bfloat16 weights) |
| `fp8` | `quantization="fp8"` |
| `bitsandbytes_int4` | `quantization="bitsandbytes"` (4-bit) |

```
python scripts/Run_Inferences.py --dataset FLiPS_ICML --model Qwen/Qwen2-7B-Instruct \
    --quantization fp8 --sub_run qwen2_7b_fp8 --gpu 1 --parse_gen
```

(GPU command, see [`../reproduction/generation.md`](../reproduction/generation.md).) The run records the model
as `Qwen/Qwen2-7B-Instruct@@fp8` and keeps the level in `quantization_map` of `run_config.json`. The weights
folder, the chat template and the stop tokens are those of the base model: every lookup drops the
`@@<level>` suffix (`get_base_model_name`). Quantized instances use the cells of `FLiPS_ICML.csv`
([`dataset.md`](dataset.md)).

## Naming

`QUANTIZATION_SEPARATOR = "@@"` in `src/audit_llm/file_io.py` separates the base model from the level:

```
Qwen/Qwen2-7B-Instruct@@fp8_temp-0.6_sp--1
|                      |    |
|                      |    variation suffix temp-<t>_sp-<sp>
|                      quantization level (after @@)
base model (before @@)
```

`get_base_model_name("Qwen/Qwen2-7B-Instruct@@fp8")` returns `Qwen/Qwen2-7B-Instruct` and
`get_quantization_suffix` returns `fp8`. The level `bitsandbytes_int4` contains an underscore; the helpers
split on `@@` first, so it is handled as one level. The `Model` column of the answers tables holds the name
with its suffix, and the per-model answers file is `<org>__<model>@@<level>.parquet` (`/` becomes `__`).

## Where the suffix is used

| Stage | Module | Behaviour |
|-------|--------|-----------|
| scenario | `src/audit_llm/scenarios/` | `quantization` axis of a variation group; `Instance.storage_name` = `<base>@@<level>`; `resolve_quantization` gives the engine argument |
| classifier | `src/audit_llm/Classification/hierarchical.py` | `llm_name` strips the suffix: a quantized instance belongs to the LLM of its base, so the per-LLM classifier of `Qwen/Qwen2-7B-Instruct` decides among its 8 base and 8 quantized instances |
| analysis tables | `src/audit_llm/xp_tools/model_filtering.py` | `full_var_model_name_to_original_model_name` returns the base model (same row as the base instances); `full_var_model_name_to_var_name` prefixes the level to the variation (`fp8_temp-0.6_sp--1`) |

In a scenario, quantized instances come from two sections of the YAML:

```yaml
quantized_base_models: [Qwen/Qwen2-7B-Instruct]
quantized_variations:
  - {temperature: [0.6, 1.0], system_prompt_idx: [-1], quantization: [fp8, bitsandbytes_int4]}
```

The loader checks every (base model, level) pair against `quantized_variants` of `config/models.yaml`.

## Related

[`inference.md`](inference.md) - [`scenarios.md`](scenarios.md) - [`../provenance.md`](../provenance.md)

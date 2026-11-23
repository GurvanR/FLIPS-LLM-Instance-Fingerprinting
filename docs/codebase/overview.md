# Codebase overview

Entry point to the codebase documentation. The repository holds the code and data of the FLIPS paper:
generation of random bit sequences by LLM instances, the 57 statistics of a sequence, the hierarchical instance
classifier and Unseen detector, and the analysis tier that regenerates the paper's computed figures, tables and quoted numbers (except Figure 3 and the LLMmap bars of Figure 1).

## Architecture

```
datasets/ + config/        request tables, prompts, model registry, scenarios
        |
 scripts/Run_Inferences.py     generation (vLLM) and parsing           -> answers parquet files    (tier 4)
        |
 audit_llm.features            57 statistics per answer                -> feature grid            (tier 2)
        |
 audit_llm.pipeline            closed set / open set protocols         -> cells/*.json, summary.json (tiers 2, 3)

 results archive  --->  paper/make_paper.py + paper/check.py           -> figures, tables, numbers (tier 1)
```

Tiers: [`../reproduction.md`](../reproduction.md). The package `audit_llm` lives in `src/audit_llm/`; the
distribution is named `flips`.

## Repository map

| Path | Content |
|---|---|
| `src/audit_llm/` | the package (layout below) |
| `paper/` | analysis tier: one module per figure, `make_paper.py` driver, `check.py`, `expected/` reference outputs ([`paper/README.md`](../../paper/README.md)) |
| `XP_configs/` | experiment configs (`closed_set.yaml`, `open_set.yaml`, `smoke.yaml`, `smoke_open_set.yaml`, `example_custom/`) |
| `scripts/` | `fetch_data.py`, `Run_Inferences.py`, `parsing_generations.py`, `merge_runs.py`, SLURM template, `screen_run.sh` (standalone helper), `gates/`, `gen_manifest.py` |
| `datasets/` | request tables and prompt assembly |
| `config/` | `models.yaml` (model registry), `scenarios/` |
| `data/` | download target, manifests, data licence |
| `Productions/` | `Intersection_vocab/` (token vocabulary shared by the models); generation runs are written here |
| `tests/` | unit, regression, generation and paper tests; `fixtures/` |

## Package layout (`src/audit_llm/`)

| Module | Role |
|---|---|
| `pipeline.py` | experiment driver: `build-grid` and `run` ([`experiment.md`](experiment.md)) |
| `features/` | `nist.py` (NIST test statistics and sequence length), `statistics.py` (complementary statistics and 3 NIST statistics), `grid.py` (answers to feature tensor) ([`features.md`](features.md)) |
| `Classification/` | `protocol.py` (splits, draws, verification samples), `imputation.py`, `xgb.py`, `log_sum_aggregation.py`, `hierarchical.py` (instance classifier), `unseen_detector.py` ([`classification.md`](classification.md)) |
| `experiment_config_schema.py` | pydantic schema of the experiment YAML files ([`configuration.md`](configuration.md), [`configuration-advanced.md`](configuration-advanced.md)) |
| `Analysis_Classes.py` | parse the generations of a run under `Productions/` into per-model answers files |
| `LLM_Classes/` | generation: model loading, chat-template rendering (`chat_rendering.py`, `chat_templates/`), vLLM backend, run configuration, parsing ([`inference.md`](inference.md)) |
| `Bits_Generation/` | prompt assembly (`Bits_Dataset_Making/`), token-pair sets, answer parsing, the vendored NIST test suite (`NIST_Tests_lib/`) ([`dataset.md`](dataset.md)) |
| `Tokens_analysis/` | tokenizer vocabularies, token intersections across models, token sampling |
| `scenarios/` | declarative selection of instances: loader, enumerator, resolver ([`scenarios.md`](scenarios.md)) |
| `models_management/`, `xp_tools/` | model-name conventions, quantization filtering, label formatting ([`quantized-models.md`](quantized-models.md)) |
| `path_utils.py`, `file_io.py`, `system_utils.py`, `model_analysis.py` | paths, storage formats, helpers, per-model answer extraction |

## Guides

| Guide | Topic |
|---|---|
| [`classification.md`](classification.md) | hierarchical classifier, log-sum aggregation, Unseen detector, protocol |
| [`features.md`](features.md) | the 57 statistics and the feature grid |
| [`experiment.md`](experiment.md) | pipeline CLI, cells, output files |
| [`configuration.md`](configuration.md), [`configuration-advanced.md`](configuration-advanced.md) | experiment config files |
| [`dataset.md`](dataset.md) | request tables and prompt assembly |
| [`inference.md`](inference.md) | generation and parsing |
| [`quantized-models.md`](quantized-models.md) | quantized instances |
| [`scenarios.md`](scenarios.md) | instance selection |
| [`testing.md`](testing.md) | tests, gates, CI |
| [`conventions.md`](conventions.md) | naming, paths, licences |
| [`../provenance.md`](../provenance.md) | the benchmark: LLMs, licences, data hosting |

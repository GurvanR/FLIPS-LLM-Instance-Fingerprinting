# Run the pipeline on your own instances

The classification pipeline runs on any set of instances for which answers exist: your own LLMs, cells,
quantization levels. Two independent layers are involved.

## The two layers

1. **Generation produces answers.** Which (model, quantization, temperature, system prompt) combinations have
   answers is decided when you generate: by the request CSV and the runs of `scripts/Run_Inferences.py`
   ([`../codebase/dataset.md`](../codebase/dataset.md), [`generation.md`](generation.md)).
2. **A scenario selects classes.** A scenario (`config/scenarios/<name>.yaml`) lists the instances that become
   classes; an experiment config (`scenario: <name>`) applies it to the answers
   ([`../codebase/scenarios.md`](../codebase/scenarios.md), field contract in `config/scenarios/README.md`).

A scenario can only select instances that have answers. It is validated against the model registry
`config/models.yaml`, not against the data: a missing instance is reported later, when the answers are
restricted to the scenario.

## Run the shipped example (CPU, no download)

`config/scenarios/example_custom.yaml` selects 2 LLMs x 2 temperatures = 4 instances from the in-repo fixture
`tests/fixtures/answers.parquet` (22 instances, 2 token pairs, 200 sequences per instance and pair), and
`XP_configs/example_custom/example_custom.yaml` is the matching experiment config (closed set, token pairs
`ali-yg` and `df-bits`, `N_t` in 1, 2, 4, 8, a protocol scaled to 200 sequences per class):

```
python -m audit_llm.pipeline run --config XP_configs/example_custom/example_custom.yaml \
    --answers tests/fixtures/answers.parquet --out out/example_custom
```

It takes about 15 s on a CPU (about 3 s for the two classification cells, the rest is feature computation and
imports) and prints the mean accuracy per `N_t` over the two cells; its last lines read

```
N_t=1: {'mean_accuracy': 0.8688, 'mean_accuracy_global': 0.8594, 'n_cells': 2}
N_t=2: {'mean_accuracy': 0.9115, 'mean_accuracy_global': 0.9115, 'n_cells': 2}
N_t=4: {'mean_accuracy': 1.0, 'mean_accuracy_global': 1.0, 'n_cells': 2}
N_t=8: {'mean_accuracy': 1.0, 'mean_accuracy_global': 1.0, 'n_cells': 2}
```

(XGBoost `hist` results depend on the thread count, so your digits can differ.) The output directory holds
`summary.json` and `cells/<token pair>_s<split>.json` with the per-sample labels, predictions and confidences.

## Select your own instances

1. Copy `config/scenarios/example_custom.yaml` to `config/scenarios/<name>.yaml`. List `base_models` (ids from
   `config/models.yaml`) and `base_variations`, a list of groups of `temperature` and `system_prompt_idx`
   values. A group is a cartesian product and the grid is the union of its groups. Quantized and abliterated
   instances have their own sections (`quantized_base_models`, `quantized_variations`, `abliterated_models`,
   `abliteration_variations`; see `config/scenarios/main.yaml`).
2. Copy `XP_configs/example_custom/example_custom.yaml`, set `scenario: <name>` (a name under
   `config/scenarios/` or a path to the YAML), and scale `protocol` to your data.
3. Run on your answers. To reuse the features for several scenarios compute the grid once:

   ```
   python -m audit_llm.pipeline build-grid --answers <answers dir> --out <grid dir> --n-jobs 8
   python -m audit_llm.pipeline run --config <your config>.yaml --grid <grid dir> --out <results dir>
   ```

   (Passing `--answers <answers dir>` in place of `--grid` recomputes the features on each call.) The answers table is
   read from a parquet file or a directory of parquet files, with the columns `Token_pair`, `Model`,
   `Dataset_Question Index`, `Answer`, `temperature`, `system_prompt_idx` (what the generation pipeline writes,
   [`../codebase/inference.md`](../codebase/inference.md)).

## Generate your own variations

- **New cells** (other temperatures or system prompts): write a request CSV, register it in
  `scripts/Inference_configs.yaml`, run `Run_Inferences.py`, parse
  ([`../codebase/dataset.md`](../codebase/dataset.md#creating-a-new-dataset)). Instances are identified by
  (model, temperature, system prompt index): vary these. Cells that differ only in `prompt_idx` or
  `frequency_penalty` merge into one instance.
- **New LLM**: add its Hugging Face id to `base_models` of `config/models.yaml` (the scenario loader accepts
  only registered models), make its weights available (`$AUDIT_LLM_MODEL_CACHE/<org>/<model>`, or the Hugging Face cache), and add its end-of-turn
  ids to `src/audit_llm/LLM_Classes/chat_templates/stop_token_ids.json`; without an entry vLLM's default stop
  tokens apply. `tests/generation/test_chat_rendering.py::test_stop_token_ids_cover_every_instance_model` pins
  the 30 benchmark models: adapt it when you extend the file.
- **Quantized instances** of a registered base need an entry of `quantized_variants` in `config/models.yaml` and
  a run with `--quantization` ([`../codebase/quantized-models.md`](../codebase/quantized-models.md)).
- The labels of the instances (`<model>[@@<level>]_temp-<t>_sp-<sp>`) are exact strings
  ([`../codebase/scenarios.md`](../codebase/scenarios.md#instances-and-labels)).

## Pitfalls

| Pitfall | Symptom |
|---------|---------|
| The scenario selects an instance without answers | `KeyError: instances absent from the grid: [...]` (exit 1) |
| A model of the scenario is not in `config/models.yaml` | `ValueError: Unknown base model(s) [...] not in registry` |
| An abliteration group other than temperature 1.0 and no system prompt | `ValueError` from `load_scenario` |
| A `token_pairs` entry that is not in the answers | `KeyError: token pairs absent from the answers` |
| A config without `scenario:` | every instance present in the answers is a class (`closed_set.yaml` and `open_set.yaml` set `scenario: main`) |
| Few sequences per class | lower `protocol.test_per_class`, `inner_val`, `n_train` as the example does; a class with fewer rows than `n_train` contributes what it has |
| Open set on a small scenario | the defaults assume 237 instances (`open_set.n_unseen: 23`, `n_folds: 10`); scale them as `XP_configs/smoke_open_set.yaml` does (3 Unseen instances on 22) |

## Validate before spending compute

```
python -c "from audit_llm.scenarios.loader import load_scenario; print(load_scenario('<name>'))"
python -c "from audit_llm.experiment_config_schema import load_experiment_config as l; print(l('<your config>.yaml'))"
```

The first checks the scenario against the registry and the abliteration pin; the second validates the
experiment config (unknown or duplicated keys are rejected). To list the instances of a scenario that have no
answers, without computing any feature:

```python
import polars as pl
from audit_llm.pipeline import scenario_instances

answers = pl.read_parquet("<answers file>")
present = {f"{m}_temp-{t}_sp-{s}" for m, t, s in answers.select("Model", "temperature", "system_prompt_idx").unique().rows()}
print(sorted(set(scenario_instances("<name>")) - present))      # [] when every instance has answers
```

For the shipped example the list is empty (`<name>` = `example_custom`, answers file
`tests/fixtures/answers.parquet`).

## Related

[`generation.md`](generation.md) - [`recompute.md`](recompute.md) -
[`../codebase/scenarios.md`](../codebase/scenarios.md) - [`../codebase/configuration.md`](../codebase/configuration.md)

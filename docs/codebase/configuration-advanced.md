# Configuration: field reference

Complete reference of the experiment YAML schema, `src/audit_llm/experiment_config_schema.py`. For a short guide with
examples, see [`configuration.md`](configuration.md). "Read by" names the function that consumes the field
(`pipeline.py` unless stated otherwise); the classifier and protocol functions are described in
[`classification.md`](classification.md).

## Loading and validation

`load_experiment_config(path)` reads the YAML with a loader that raises `ValueError` on a duplicate key (message
`Duplicate YAML key '<key>' found at line <n>`), requires a mapping at the top level, validates it with the pydantic model
`ExperimentConfig`, and returns `model_dump()`: a plain dict with every default filled in and the nested sections as dicts.

| condition | result |
|---|---|
| duplicate key, at any level | `ValueError` |
| top level is not a mapping | `ValueError` |
| unknown key, at any level (for example `protocol.n_trian`) | `pydantic.ValidationError`, the key is named |
| `experiment_fun` missing or not `closed_set` / `open_set` | `pydantic.ValidationError` |
| `open_set` section with `experiment_fun: closed_set` | `pydantic.ValidationError` ("an 'open_set' section is only valid with experiment_fun: open_set") |
| a `batch_sizes` value below 1 | `pydantic.ValidationError` ("batch_sizes must be >= 1") |
| a value that cannot be converted to the field's type (for example `n_train: abc`; a numeric string such as `'300'` is converted) | `pydantic.ValidationError` |

With `experiment_fun: open_set` and no `open_set` section, the section takes its defaults. No field has a range check
beyond the ones in the table. `python -m audit_llm.pipeline run` loads the file with this function.

## Top-level fields (`ExperimentConfig`)

| field | type | default | read by |
|---|---|---|---|
| `experiment_fun` | `closed_set` or `open_set` | required | `run_experiment` (cell enumeration), `_run_cell` (which cell function), `_cell_data` (which splits to build), `_summary` (which metrics) |
| `xp_name` | string or none | none | recorded in `summary.json`. `pipeline.main` sets it to the config file name without extension when unset |
| `description` | string or none | none | not read by any function; free text for the reader of the file |
| `token_pairs` | list of strings | `[]` | `run_experiment`: the pairs to evaluate, `[]` = every pair of the grid (a pair absent from the grid raises `KeyError`); with `--answers`, also the pairs for which the grid is computed |
| `scenario` | string or none | none (`closed_set.yaml` and `open_set.yaml` set `main`) | `run_experiment` through `scenario_instances`, see below |
| `min_seq_length` | int | 100 | `features.grid.build_grid`, when the grid is computed during `run --answers`; ignored with `--grid`, whose `<grid dir>/meta.json` holds the value used by `build-grid --min-seq-length` |
| `batch_sizes` | list of ints, each >= 1 | `[1, 2, 4, 8, 16, 32, 64]` | the values of `N_t`: `sample_batches` in `closed_set_cell`, `evaluate_open_set_cell` in `open_set_cell`, and the keys of `by_n_t` in `_summary` |
| `protocol` | `ProtocolConfig` | defaults below | `_cell_data` and the cell functions |
| `classifier` | `ClassifierConfig` | defaults below | `HierarchicalClassifier`; in the open set also `UnseenDetector` |
| `open_set` | `OpenSetConfig` or none | none (open set: defaults) | `open_set_cell`; valid only with `experiment_fun: open_set` |
| `n_workers` | int | 1 | `run_experiment`: a process pool of that size evaluates the cells when it is above 1, otherwise they run one after the other |

## `protocol` (`ProtocolConfig`)

| field | type | default | read by |
|---|---|---|---|
| `splits` | list of ints | `[0, 1, 2, 3, 4]` | closed set only: the cells of `run_experiment`, and the number of splits generated (largest value + 1) by `_cell_data`; ignored in the open set |
| `test_per_class` | int | 128 | `protocol.make_splits`: each split has `test_per_class` x (number of instances) test rows |
| `inner_val` | int | 40 | `protocol.carve_validation`: validation rows set aside per instance (at most a third of its training rows) |
| `n_train` | int | 300 | `N_r`: `protocol.nested_draw` (closed set), `protocol.independent_draw` (open set) |
| `seed` | int | 0 | `protocol.seeded` streams: the validation rows of the cell, and the open-set training draw. The split seed (42) is not configurable |

## `classifier` (`ClassifierConfig`)

| field | type | default | read by |
|---|---|---|---|
| `global_rounds` | int | 357 | boosting rounds of the global classifier, `HierarchicalClassifier` and the fold classifiers of `UnseenDetector` |
| `n_jobs` | int | 4 | threads of the global classifier and of the fold classifiers (per-LLM classifiers use 1) |
| `round_selection_patience` | int | 60 | early-stopping patience of `xgb.select_rounds` (per-LLM classifiers) |
| `round_selection_max` | int | 1000 | maximum number of rounds tried by `xgb.select_rounds` |

## `open_set` (`OpenSetConfig`)

All fields are read by `open_set_cell`, which passes them to `unseen_detector.evaluate_open_set_cell`.

| field | type | default | read by |
|---|---|---|---|
| `cells` | list of `[split, draw]` pairs of ints | `[[0, 0], [1, 1], [2, 2], [0, 3], [3, 4]]` | `run_experiment` (one cell per token pair and pair of values); `draw` selects the Unseen instances and keys the detector's random streams |
| `n_unseen` | int | 23 | `draw_unseen`: instances treated as Unseen, drawn with `default_rng([4242, draw])` |
| `n_folds` | int | 10 | `assign_class_folds`: class folds of the Known instances (a value other than 10 uses a different random stream) |
| `held_out_per_class` | int | 40 | `UnseenDetector.fit_folds`: rows of each Known instance set aside in a fold (at most a third) |
| `unseen_rows_per_instance` | int | 40 | `fit_folds`: rows of each held-out instance used to draw simulated Unseen samples |
| `known_batches_per_instance` | int | 10 | `UnseenDetector.samples`: Known samples per instance |
| `unseen_batches_per_instance` | int | 40 | `samples`: simulated Unseen samples per held-out instance |
| `min_rows_per_instance` | int | 120 | `samples`: an instance with fewer training rows gives no detector sample |
| `unseen_weight_share` | float | 0.097 | `sample_weights`: share of the regression weight carried by the simulated Unseen samples |
| `regularization_c` | float | 0.1 | `LogisticDetector`: `C` of the regularised logistic regression |
| `target_recall` | float | 0.5 | `threshold`: `tau` is the `1 - target_recall` quantile of the out-of-fold scores of the simulated Unseen samples |

`sample_statistics` needs at least 10 instances in the classifier whose output it describes, and at least 3 LLMs with 2 or
more of those instances, for every statistic to be finite; a scenario for the open set has to be large enough for that.

## The `scenario` key

`scenario: <name>` restricts the experiment to the instances of a scenario file. `scenario_instances` takes the value as a
path if a file or directory of that name exists (relative to the working directory, so run from the repository root),
otherwise as `config/scenarios/<name>.yaml`. The file is loaded and validated by `audit_llm.scenarios.loader.load_scenario`
against the model registry `config/models.yaml`, and expanded to instances named
`<model>[@@<quant>]_temp-<temperature>_sp-<system prompt index>`. These names must all be classes of the grid, otherwise
`Grid.select_classes` raises `KeyError`; the grid is then restricted to them, keeping its class order. The restricted
grid is held in memory. In the open set, `n_unseen` instances are drawn among the instances kept. The rules of a scenario file are in
[`scenarios.md`](scenarios.md); a worked example is in [`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md).
`config/scenarios/main.yaml` defines the 237 instances of the benchmark.

## Configuration on disk

```
XP_configs/
  closed_set.yaml  open_set.yaml  smoke.yaml  smoke_open_set.yaml     the shipped experiments
  example_custom/example_custom.yaml                                  template with a scenario
config/
  scenarios/main.yaml  scenarios/example_custom.yaml  scenarios/README.md
  models.yaml                                                         model registry read by the scenario loader
```

Experiment configs live under `XP_configs/` and are run with `python -m audit_llm.pipeline run --config <file>` ([`experiment.md`](experiment.md)). `tests/test_experiment_config_schema.py` validates every
YAML file found under `XP_configs/`. Scenario files live in `config/scenarios/` and are referenced by name from the
`scenario` key.

## Related documentation

[`../README.md`](../README.md) · [`overview.md`](overview.md) · [`configuration.md`](configuration.md) ·
[`experiment.md`](experiment.md) · [`classification.md`](classification.md) · [`scenarios.md`](scenarios.md) ·
[`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md)

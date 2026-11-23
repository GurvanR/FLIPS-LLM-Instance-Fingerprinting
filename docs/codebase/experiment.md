# Experiment pipeline

An experiment evaluates the decision function of the paper on a feature grid: the hierarchical instance classifier in the
closed set, the classifier plus the Unseen detector in the open set. It is driven by `src/audit_llm/pipeline.py`:

```
python -m audit_llm.pipeline build-grid --answers <parquet|dir> --out <grid dir> [--n-jobs N]
python -m audit_llm.pipeline run --config XP_configs/<cfg>.yaml (--grid <grid dir> | --answers <parquet|dir>) --out <dir> [--n-jobs N]
```

`build-grid` computes the 57 statistics of every answer and saves the grid ([`features.md`](features.md)). `run` reads a
YAML config ([`configuration.md`](configuration.md)), splits the experiment into cells, evaluates each cell
([`classification.md`](classification.md)) and writes one JSON file per cell plus a summary. With `--answers`, the
grid is computed in memory and not saved; with `--grid`, it is opened from disk.

## From a config to cells

`run_experiment(cfg, grid, out_dir)` proceeds as follows.

1. If `scenario` is set (`closed_set.yaml` and `open_set.yaml` set `scenario: main`), the grid is restricted to the instances of that scenario (`Grid.select_classes`); an instance
   absent from the grid raises `KeyError`.
2. The token pairs are `token_pairs`, or every token pair of the grid when the list is empty. A pair absent from the grid
   raises `KeyError`.
3. The cells are enumerated:
   - closed set: one cell per (token pair, split) with the splits of `protocol.splits`;
   - open set: one cell per (token pair, split, draw) with the `(split, draw)` couples of `open_set.cells`.
4. Each cell is evaluated, in `n_workers` processes if `n_workers > 1`.
5. When all cells are done, the cell files and `summary.json` are written (an interrupted run leaves no cell file); to keep partial
   progress, split a run by token pair ([`../reproduction/recompute.md`](../reproduction/recompute.md#running-in-parallel-and-in-parts)).

| config | cells | per token pair |
|---|---|---|
| `XP_configs/closed_set.yaml` | 31 pairs x 5 splits = 155 | 5 splits |
| `XP_configs/open_set.yaml` | 31 pairs x 5 (split, draw) = 155 | `[[0, 0], [1, 1], [2, 2], [0, 3], [3, 4]]` |
| `XP_configs/smoke.yaml` | 2 pairs x 2 splits = 4 | on the fixture |
| `XP_configs/smoke_open_set.yaml` | 2 pairs x 2 (split, draw) = 4 | `[[0, 0], [1, 1]]`, on the fixture |

A closed-set cell (`closed_set_cell`) selects the rows of the token pair, makes the split, draws the training rows,
fits the hierarchical classifier, scores the test rows once, and then aggregates the samples at every `N_t` of
`batch_sizes`. An open-set cell (`open_set_cell`) draws the Unseen instances for its `draw`, fits the classifier and the
fold classifiers of the detector once, then builds the detector and labels the test samples at every `N_t`. The
cell code is the same whatever the entry point.

## Parallelism

| setting | where | meaning |
|---|---|---|
| `n_workers` | config | number of cells evaluated in parallel (processes); default 1 |
| `classifier.n_jobs` | config | threads of the global XGBoost classifier (and of the fold classifiers of the detector); per-LLM classifiers use 1 thread. Default 4, the value of the paper |
| `--n-jobs` | `build-grid`, `run` | processes for the NIST tests when the grid is computed (`build-grid`, or `run --answers`); default: the CPU count; unused with `--grid` |
| `FLIPS_NUM_THREADS` | environment | numba threads of the statistics of `features/statistics.py` ([`features.md`](features.md)) |

The cores used by the training stage are about `n_workers` x `classifier.n_jobs`. XGBoost `hist` results depend on the thread
count: changing `classifier.n_jobs` changes the numbers slightly, changing `n_workers` does not (each cell is independent).
With a saved grid and no `scenario`, each worker opens it from disk (memory-mapped). An in-memory grid (`--answers`) and a grid
restricted by `scenario` (`Grid.select_classes` copies every token pair into memory) are sent to the workers with each task.
Both shipped configs set `scenario: main`, so for them the memory use is that of the in-memory case: on the 31 token
pairs (a grid of 3.7 GB on disk) the parent process holds about 12 GB and each worker about 5 GB, so about 45 GB for
7 workers; choose `n_workers` by the memory available (about 5 GB per worker on top of the 12 GB), not by the number of
cores alone.
The config headers of `closed_set.yaml` and `open_set.yaml` give the cost of the full runs (about 80 core-hours for the
closed set, about 640 core-hours for the five open-set draws); see
[`../reproduction/compute-resources.md`](../reproduction/compute-resources.md).

## Output layout

```
<out>/
  cells/<pair>_s<split>.json            closed set
  cells/<pair>_s<split>_d<draw>.json    open set
  summary.json
```

`<pair>` is the token pair as written in the config (for example `ali-yg`). Labels and predictions in the cell files are
indices into the class order of the grid (sorted by model, temperature, system prompt; restricted to the scenario's
instances when `scenario` is set); a saved grid lists the names in `<grid dir>/meta.json`.

### Closed-set cell file

| key | content |
|---|---|
| `token_pair`, `split`, `n_classes` | the cell and the number of instances |
| `n_train_per_class_min` | smallest number of training rows drawn for an instance |
| `n_test` | number of test rows |
| `per_llm_rounds` | boosting rounds selected for each per-LLM classifier, by LLM name |
| `seconds` | wall time of the cell |
| `by_n_t["<N_t>"]` | one entry per `N_t` with at least one sample, below |

An entry of `by_n_t`: `n_samples`; `accuracy` (hierarchical decision); `accuracy_global` (stage 1 instance alone);
`llm_accuracy` (selected LLM is the true LLM); and the per-sample lists `labels`, `predictions`, `global_predictions`,
`confidence` (8 decimals).

### Open-set cell file

Keys `token_pair`, `split`, `draw`, `n_classes`, `n_test`, `seconds` as above, and `unseen`, the names of the Unseen
instances of the draw. An entry of `by_n_t`: `n_samples`; `accuracy` (an Unseen sample is correct when flagged);
`unseen_recall`, `unseen_precision` (`null` when undefined); `known_accuracy_closed` (hierarchical decision on the Known
samples, no rejection); `threshold` (`tau`); `accuracy_global` (always `null`); and per-sample lists `labels` (true
instance index), `is_unseen` (0/1), `predictions` (`-1` = flagged Unseen), `hierarchical_predictions`, `unseen_scores`
(8 decimals).

### `summary.json`

| key | content |
|---|---|
| `experiment_fun`, `xp_name` | the config's `experiment_fun`; `xp_name` (default: the config file name without extension) |
| `by_n_t["<N_t>"]` | `mean_accuracy`, `mean_accuracy_global` (`null` in the open set), in the open set also `mean_unseen_recall`, `mean_unseen_precision`, and `n_cells` |
| `cells` | per cell: the metadata of the cell file without `by_n_t`, and `accuracy`: `{"<N_t>": accuracy}` |

Means are over cells (the mean of the per-cell values); cells with an undefined value are skipped.

Excerpt of the summary of `XP_configs/smoke.yaml` on the fixture (22 instances, chance level 1/22; keys and `N_t`
entries omitted, values rounded; XGBoost 3.2.0 with 4 threads, the per-cell accuracies of
`tests/fixtures/reference/smoke_summary.json`):

```json
{"experiment_fun": "closed_set", "xp_name": "smoke",
 "by_n_t": {"1": {"mean_accuracy": 0.3935, "mean_accuracy_global": 0.3895, "n_cells": 4},
            "8": {"mean_accuracy": 0.6020, "mean_accuracy_global": 0.6336, "n_cells": 4}},
 "cells": [{"token_pair": "ali-yg", "split": 0, "n_classes": 22, "n_train_per_class_min": 98, "n_test": 880,
            "accuracy": {"1": 0.3727, "2": 0.4630, "4": 0.4952, "8": 0.6078}}]}
```

## Commands

| command | what it runs |
|---|---|
| `make smoke` | `XP_configs/smoke.yaml` on `tests/fixtures/answers.parquet`, output in `$(OUT)/smoke` (`OUT` defaults to `out`) |
| `make smoke-open-set` | `XP_configs/smoke_open_set.yaml` on the fixture, output in `$(OUT)/smoke_open_set` |
| `make gate-a` | `scripts/gates/gate_a.py`: the smoke run, then checks of the cell count and that accuracy is above chance at every `N_t` and higher at the largest `N_t` than at `N_t = 1` |
| `make gate-golden` | `scripts/gates/gate_golden.py`: the smoke run compared with `tests/fixtures/reference/smoke_summary.json` (`--tolerance` allows a band for other library versions) |
| `make fetch-generations` | `scripts/fetch_data.py --with-generations`: the release data archive into `Productions/` |
| `make grid [ANSWERS=<dir>]` | `build-grid` of `$(ANSWERS)` (default `Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers`, the fetched generations) into `$(GRID)` (default `out/grid`) |
| `make recompute-closed` | `XP_configs/closed_set.yaml` on `$(GRID)`, output in `$(OUT)/closed_set` |
| `make recompute-open` | `XP_configs/open_set.yaml` on `$(GRID)`, output in `$(OUT)/open_set` |

`JOBS` (default: the CPU count) is the `--n-jobs` of `make smoke`, `make smoke-open-set` and `make grid`. The tiers
that use these targets are described in [`../reproduction.md`](../reproduction.md) and
[`../reproduction/recompute.md`](../reproduction/recompute.md). To run an experiment on your own set of instances,
see [`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md); the example config
`XP_configs/example_custom/example_custom.yaml` runs with:

```
python -m audit_llm.pipeline run --config XP_configs/example_custom/example_custom.yaml --answers tests/fixtures/answers.parquet --out <dir>
```

## Generation runs

A generation run (tier 4) parses its raw generations into per-model answers parquet files under
`Productions/<run name>/Analysis/<scrapping rule>/` (`src/audit_llm/Analysis_Classes.py`: `AuditionsAnalysis(run_path)`
reads the run's `run_config.json`; `model_analysis()` parses the generations of the vLLM models). Pass those parquet
files, or a directory of them, to `--answers` of `build-grid` or `run` of `python -m audit_llm.pipeline`
([`../reproduction/generation.md`](../reproduction/generation.md)).

## Related documentation

[`../README.md`](../README.md) · [`overview.md`](overview.md) · [`classification.md`](classification.md) ·
[`features.md`](features.md) · [`configuration.md`](configuration.md) · [`configuration-advanced.md`](configuration-advanced.md) ·
[`testing.md`](testing.md) · [`../reproduction.md`](../reproduction.md) · [`../reproduction/recompute.md`](../reproduction/recompute.md) ·
[`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md) · [`../reproduction/compute-resources.md`](../reproduction/compute-resources.md)

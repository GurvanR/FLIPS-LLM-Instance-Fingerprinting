# Configuration

An experiment is one YAML file, validated by `src/audit_llm/experiment_config_schema.py` and run by
`python -m audit_llm.pipeline run --config <file>` ([`experiment.md`](experiment.md)). Omitted keys take the values of
the paper's protocol, so a config only lists what it changes. Unknown keys and duplicate keys are rejected.
Every field is listed in [`configuration-advanced.md`](configuration-advanced.md).

## Minimal configs

The shortest valid file is one line. It runs the paper's closed-set protocol (5 splits, `N_r` = 300, `N_t` from 1 to 64)
on every token pair of the grid:

```yaml
experiment_fun: closed_set      # or: open_set
```

A small closed-set config for the in-repo fixture (one token pair, one split):

```yaml
experiment_fun: closed_set
token_pairs: [ali-yg]
batch_sizes: [1, 4]
protocol:
  splits: [0]
  test_per_class: 40
  inner_val: 10
  n_train: 100
```

```
python -m audit_llm.pipeline run --config <file> --answers tests/fixtures/answers.parquet --out <dir>
```

## Shipped configs (`XP_configs/`)

| file | experiment | what it sets |
|---|---|---|
| `closed_set.yaml` | closed set, paper protocol | `scenario: main` (the 237 instances), all token pairs, splits 0-4, `test_per_class` 128, `inner_val` 40, `n_train` 300, 357 global rounds, `n_jobs` 4 |
| `open_set.yaml` | open set, paper protocol | `scenario: main` and the same protocol, plus the `open_set` section: 23 Unseen instances per draw, 10 class folds, `target_recall` 0.5, the five (split, draw) cells `[[0, 0], [1, 1], [2, 2], [0, 3], [3, 4]]` |
| `smoke.yaml` | closed set on the fixture | `token_pairs: [ali-yg, df-bits]`, `batch_sizes: [1, 2, 4, 8]`, splits 0-1, `test_per_class` 40, `inner_val` 10, `n_train` 100 |
| `smoke_open_set.yaml` | open set on the fixture | the same scaling, 3 Unseen instances, `global_rounds` 100, `round_selection_max` 100, `min_rows_per_instance` 60, cells `[[0, 0], [1, 1]]` |
| `XP_configs/example_custom/example_custom.yaml` | closed set on your own instances | `scenario: example_custom` (4 instances of the fixture), one split; a template to copy |

`closed_set.yaml` and `open_set.yaml` state their cost in their header (about 80 core-hours and about 640 core-hours);
the two smoke configs run on the fixture on CPU with no download. Bitwise identical results need XGBoost 3.2 and 4
threads (`classifier.n_jobs`).

## Common keys

| key | default | meaning |
|---|---|---|
| `experiment_fun` | required | `closed_set` or `open_set` |
| `xp_name` | none | name recorded in `summary.json`; the config file name when unset |
| `token_pairs` | `[]` | token pairs to evaluate; `[]` = every token pair of the grid |
| `scenario` | none | restrict the instances to a scenario: a name in `config/scenarios/` or a path to its YAML |
| `batch_sizes` | `[1, 2, 4, 8, 16, 32, 64]` | the values of `N_t` (each at least 1) |
| `protocol.splits` | `[0, 1, 2, 3, 4]` | splits of a closed-set run (an open-set run takes its splits from `open_set.cells`) |
| `protocol.test_per_class` | 128 | test sequences per instance in each split |
| `protocol.inner_val` | 40 | validation sequences per instance kept out of the training draw (the paper calibrates the closed-set confidence on them, a [stored result](../reproduction/recompute.md#results-without-a-recompute-path)) |
| `protocol.n_train` | 300 | `N_r`, training sequences per instance |
| `classifier.global_rounds` | 357 | boosting rounds of the global classifier |
| `classifier.n_jobs` | 4 | threads of the global classifier |
| `open_set.cells` | the five cells above | `[split, draw]` pairs of an open-set run |
| `open_set.n_unseen` | 23 | Unseen instances per draw |
| `open_set.target_recall` | 0.5 | share of simulated Unseen samples flagged by the threshold |
| `n_workers` | 1 | cells evaluated in parallel |

A config for an experiment on a subset of instances, such as `example_custom.yaml`, sets `scenario`; the scenario files
and their rules are described in [`scenarios.md`](scenarios.md) and
[`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md).

## Validation

```
python -c "from audit_llm.experiment_config_schema import load_experiment_config as load; print(load('XP_configs/smoke.yaml')['protocol'])"
```

`load_experiment_config` returns the config as a plain dict with every default filled in, or raises an error that names
the offending key. `tests/test_experiment_config_schema.py` checks the defaults, the rejection of unknown and duplicate
keys, and that every YAML file under `XP_configs/` validates.

## Related documentation

[`../README.md`](../README.md) · [`overview.md`](overview.md) · [`configuration-advanced.md`](configuration-advanced.md) ·
[`experiment.md`](experiment.md) · [`classification.md`](classification.md) · [`scenarios.md`](scenarios.md) ·
[`../reproduction.md`](../reproduction.md) · [`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md)

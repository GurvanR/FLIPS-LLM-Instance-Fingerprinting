# Tiers 2 and 3: recompute the results from the generations

The results archive read by tier 1 holds the outputs of the paper's closed-set and open-set runs. These
two tiers recompute such runs from the generated answers with the code of this repository. They need
the answers (the generations of the release data archive, or tier 4's output), CPU time (hours to days)
and the pinned environment for bitwise identical results. Overview of all tiers:
[`../reproduction.md`](../reproduction.md).

```bash
make fetch-generations      # = python scripts/fetch_data.py --with-generations
```

downloads the release data archive `flips_release_data.tar.gz` of the data record (4.6 GB; about 11 GB of
free disk while it unpacks) and verifies it into `Productions/`: the generations in
`Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/{answers,token_ids}/` (one parquet file per model, 0.7 GB) and
the run data of the LLMmap baseline in `Productions/LLMmap_ICML_run/` (configurations, trained classifiers, logs,
evaluations, provided as they are; tiers 2 and 3 read the generations).

## Status

The modules are regression-tested on the fixture against stored reference outputs
(`tests/test_closed_set_reference.py`, `tests/test_open_set_reference.py`): the features, the training draws, both stages of the
hierarchical classifier, the aggregated log-probabilities, the decisions, and for the open set the folds, the
detector statistics, weights, out-of-fold scores, thresholds and final decisions agree with the reference
outputs with a maximum absolute difference of 0. The full-size runs below are not re-run in the tests.

## Input

`ANSWERS`, by default `Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers`: a directory of parquet files, one row per generated answer, with the columns `Token_pair`, `Model`,
`Dataset_Question Index`, `Answer` (the bit string), `temperature` and `system_prompt_idx`. An *instance* is a
(`Model`, `temperature`, `system_prompt_idx`) triple, named `<model>_temp-<temperature>_sp-<index>`; quantized
variants carry the quantization in `Model` as `<base model>@@<quantization>` (see [`../codebase/quantized-models.md`](../codebase/quantized-models.md)).
`tests/fixtures/answers.parquet` is a small example. The directory must contain only answers files: the
`answers/` directory of the release data archive or of a tier-4 run, not its `token_ids/` sibling. The shipped configs set `scenario: main`, so the classes are the 237 benchmark
instances ([`../codebase/scenarios.md`](../codebase/scenarios.md)); remove `scenario` from a copy of the config to
use every instance present in the answers (for example all 20 request cells of every model).

## Step 1: feature grid

```bash
make grid JOBS=16     # = python -m audit_llm.pipeline build-grid --answers Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers --out out/grid --n-jobs 16
```

Computes the 57 statistics of every answer ([`../codebase/features.md`](../codebase/features.md)) and stores, per
token pair, the feature tensor, the bit sequences, their lengths and the query indices. Answers shorter than
100 bits get NaN in every column. The feature computation takes CPU-hours on the full corpus;
`--n-jobs` parallelises over answers, and `poetry install --extras speed` compiles the statistics of
`features/statistics.py` (`FLIPS_NUM_THREADS` sets the number of numba threads, default 4). The grid is built once and
shared by both protocols. On the generations of the release data archive, two token pairs (237 instances x 500
answers each) take about 6.5 minutes with 12 processes (49 CPU-minutes), so the 31 token pairs take about
13 CPU-hours (39 minutes with 24 processes on a 24-core machine). The saved grid of the 31 token pairs takes 3.7 GB on disk.

## Step 2: closed set (tier 2)

```bash
make recompute-closed       # = python -m audit_llm.pipeline run --config XP_configs/closed_set.yaml --grid out/grid --out out/closed_set
```

`XP_configs/closed_set.yaml` is the paper's protocol: every token pair of the grid (31 in the paper), 5 splits,
so 155 cells; per cell `N_r = 300` training sequences per instance, 128 test sequences per instance, and
verification samples of `N_t` = 1, 2, 4, ..., 64 queries. 40 validation sequences per instance are kept out of the
training draw; the paper uses them for the temperature-scaling calibration of the closed-set confidence, a stored
result ([below](#results-without-a-recompute-path)). The global classifier runs a constant
number of boosting rounds (`classifier.global_rounds: 357`); the per-LLM classifiers select theirs per cell by
early stopping on their training rows. Each cell fits the hierarchical instance classifier
and writes `cells/<pair>_s<split>.json`; `summary.json` aggregates the accuracy per `N_t`.

Cost, measured on a 24-core machine: about 1 800 core-seconds per cell, about 80 core-hours for the 155 cells. With
`n_workers: 6` (6 cells in parallel, 4 threads each: 24 threads) a cell takes about 490 s of wall time and the closed set
about 4 hours; the figures depend on the CPU and on the load. `n_workers` cells run in parallel, each using
`classifier.n_jobs` threads; [below](#running-in-parallel-and-in-parts) for the memory this needs.

## Step 3: open set (tier 3)

```bash
make recompute-open         # = python -m audit_llm.pipeline run --config XP_configs/open_set.yaml --grid out/grid --out out/open_set
```

`XP_configs/open_set.yaml` is the paper's open-set protocol: per token pair, 5 (split, draw) cells; in each draw
23 instances are Unseen, drawn at random, and the classifier and the Unseen detector are built from the Known
instances' training sequences. The detector is trained on simulated Unseen instances: the Known instances
are split in 10 class folds, a classifier is fitted on the other folds, and the held-out fold plays the Unseen
role; the threshold flags half of the simulated Unseen samples. Details:
[`../codebase/classification.md`](../codebase/classification.md).

Cost: dominated by the 10 fold classifiers of every cell, about 32 x 4 core-hours per draw over the 31 token
pairs, about 640 core-hours for the 5 draws.

## Running in parallel and in parts

`n_workers` (cells in parallel) is a key of the config, with no command-line flag and no `make` variable: copy
`XP_configs/closed_set.yaml` (or `open_set.yaml`), set `n_workers` there, and pass the copy with `--config`:

```bash
cp XP_configs/closed_set.yaml my_closed_set.yaml      # edit: n_workers: 6
python -m audit_llm.pipeline run --config my_closed_set.yaml --grid out/grid --out out/closed_set
```

Results do not depend on `n_workers`; the thread count `classifier.n_jobs` has to stay 4 for bitwise identical results
([Determinism](#determinism)). Choose `n_workers` by memory as much as by cores: the shipped configs set `scenario: main`,
which loads the grid into memory, so the parent process takes about 12 GB and each worker about 5 GB (about 45 GB
for 7 workers). `n_workers` x `classifier.n_jobs` should not exceed the number of cores.

The cell files and `summary.json` are written at the end of a run: an interrupted run leaves nothing. To make progress
durable, split the run by token pair. The token pairs of the grid are listed in `out/grid/meta.json`; give each part a
config copy whose `token_pairs` lists its pairs, and its own output directory:

```bash
# in my_closed_set_part1.yaml:  token_pairs: [ali-yg, df-bits, ...]
python -m audit_llm.pipeline run --config my_closed_set_part1.yaml --grid out/grid --out out/closed_set_part1
```

Cells are independent and named by token pair (`cells/<pair>_s<split>.json`), so the `cells/` directories of the parts
together are the cells of the full run (copy or link them into one directory). The `summary.json` of a part covers only the
cells of that part. The open set splits the same way, with the `open_set.cells` of the paper's config kept.

## Results without a recompute path

These results of the paper are in the results archive, which tier 1 reads, but this repository does not
contain the code that computed them, so tiers 2 and 3 do not recompute them:

| result | in the results archive |
|---|---|
| classifier comparison (other classifiers than the hierarchical XGBoost classifier) | `results/closed_set/classifier_comparison/` |
| truncation of the answers to their first 100 bits | `results/closed_set/truncation/` |
| feature importances of the closed-set classifiers | `results/closed_set/hierarchical/importance/` |
| calibration of the closed-set confidence (temperature fitted per cell on the validation sequences that the closed-set step sets aside, of the cell's token pair or, where they hold fewer than N_t sequences per instance, of the other token pairs; the fit source of each cell is stored) | `results/closed_set/hierarchical/calibration.json` |
| distance-2 open-set protocol | `results/open_set/distance2.json` |
| LLMmap baseline | `results/llmmap/` (the run data of the baseline, provided as they are: `Productions/LLMmap_ICML_run/` of the release data archive) |

## Determinism

- XGBoost's histogram method gives bitwise identical results only for the same version and thread count: the
  paper's global classifiers used XGBoost 3.2 with 4 threads (`classifier.n_jobs: 4`).
- scikit-learn 1.8 and numpy 2.4 are the versions of the reference results; `poetry.lock` pins them.
- The splits and training draws are seeded (`protocol.seed`, the split seed, per-cell streams), and the Unseen
  instances of draw `d` come from a generator seeded by `d` (`open_set.yaml`).

## Comparing with the archive

The cell and summary files written here are in the pipeline's format (per-sample labels, predictions and
confidences at every `N_t`; accuracy per `N_t` in `summary.json`, described in
[`../codebase/experiment.md`](../codebase/experiment.md)). The results archive stores the same quantities in the
format read by `paper/`, so a recomputed run is compared with the archive through the mean accuracy per `N_t`
of `summary.json`; `make paper` reads the archive only.

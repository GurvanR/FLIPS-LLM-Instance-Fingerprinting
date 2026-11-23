# Classification

The package `src/audit_llm/Classification/` holds the decision function of the paper: the evaluation protocol
(rows, splits, training draws, validation rows, verification samples), the imputation, the XGBoost
classifiers, the hierarchical instance classifier, the log-sum aggregation of the `N_t` queries of a sample, and, for
the open set, the Unseen detector. It works on the feature grid built by [`features.md`](features.md) and is driven
cell by cell (one token pair, one split, in the open set one draw) by `src/audit_llm/pipeline.py`
([`experiment.md`](experiment.md)).

Vocabulary: an *instance* is a class of the grid (an LLM with its quantization, system prompt and temperature, named
`<model>[@@<quant>]_temp-<t>_sp-<sp>`). An *LLM* is the base model of an instance: `model@@fp8` belongs to `model`,
while an abliterated checkpoint is its own LLM (it has its own repository id). In the 237-instance scenario `main`
this gives 30 LLM groups (25 base models and 5 abliterated checkpoints).

## Data flow

```
answers table (parquet)
  -> features.build_grid             per token pair: features (S, C, 57), bits, lens, query_index
  -> protocol.select_rows            X (N, 57), y (N,), flat (N,): the usable rows of one token pair
  -> protocol.make_splits            (train rows, test rows) of every split
  -> protocol.prepare_cell           validation rows set aside, training draw, imputation   -> Cell
  -> HierarchicalClassifier.fit      stage 1 (global) and stage 2 (per LLM) XGBoost classifiers
  -> log-probabilities of the test rows (stage 1: L, stage 2: L2)
  -> protocol.sample_batches         verification samples: N_t test rows of one instance
  -> HierarchicalClassifier.decide   log-sum aggregation, LLM selection, instance, confidence
  -> pipeline.closed_set_cell        cells/<pair>_s<split>.json
```

The open-set cell (`unseen_detector.evaluate_open_set_cell`) draws the Unseen instances, builds the training side from
the Known instances (`protocol.prepare_open_set_cell`), fits the same hierarchical classifier, builds the Unseen
detector (`UnseenDetector.fit_folds`, then `build(n_t)`), and labels the test samples of Known and Unseen instances
(`evaluate_at`). `pipeline.open_set_cell` writes `cells/<pair>_s<split>_d<draw>.json`.

## Module map

| file | role |
|---|---|
| `protocol.py` | rows, splits, validation rows, training draws, verification samples, `Cell` (`select_rows`, `make_splits`, `carve_validation`, `nested_draw`, `independent_draw`, `prepare_cell`, `prepare_open_set_cell`, `sample_batches`) |
| `imputation.py` | `TrainingMeanImputer` |
| `xgb.py` | the XGBoost fits and exact log-probabilities (`XGB_PARAMS`, `GLOBAL_ROUNDS`, `fit_xgb`, `select_rounds`) |
| `log_sum_aggregation.py` | `log_sum_scores`, `log_sum_predict`, `LOG_FLOOR` |
| `hierarchical.py` | `HierarchicalClassifier`, `llm_name`, `llm_blocks`, `decide_within` |
| `unseen_detector.py` | `UnseenDetector`, `LogisticDetector`, `sample_statistics`, `draw_unseen`, `assign_class_folds`, `sample_weights`, `threshold`, `evaluate_open_set_cell` |

## Protocol (`protocol.py`)

- **Rows of a token pair.** `select_rows` keeps the answers whose `features.nist` columns (NIST test statistics and
  sequence length) are defined. Per instance, an answer with more than 80 % of these columns undefined is dropped
  (answers shorter than `min_seq_length` have no features and are dropped with it), an instance left with fewer than 2 answers is dropped, and at most 500 answers
  per instance are kept, in query order. The run-test statistic `run_ts` is set to 0 where it is undefined on an
  otherwise defined answer.
- **Splits.** `make_splits` calls `StratifiedShuffleSplit(random_state=42)` with `test_per_class` x (number of
  instances) test rows (128 per instance in the paper; stratification gives exactly `test_per_class` rows per
  instance when the instances have the same number of rows). The splits are the same for every experiment, and split
  `k` does not depend on the number of splits requested.
- **Validation rows.** Per instance, `carve_validation` sets aside up to `inner_val` training rows (40; at most a
  third of the instance's training rows) before the draw. The paper uses these 40 validation sequences per instance
  for the temperature-scaling calibration of the closed-set confidence, a stored result
  ([recompute.md](../reproduction/recompute.md#results-without-a-recompute-path)).
- **Training draw.** `N_r = n_train` rows (300) per instance are drawn without replacement from the remaining
  training rows (the pool), the first `N_r` rows of one seeded permutation per instance. An instance whose pool is
  smaller contributes its whole pool; the pipeline reports the smallest per-instance count as `n_train_per_class_min`.
  The classifiers of the cell are fitted on the drawn rows (`cell.X_train`, `cell.y_train`).
  - Closed set: `nested_draw` seeds the permutations with the pool itself.
  - Open set: `prepare_open_set_cell` restricts the training side to the Known instances and uses
    `independent_draw`, whose permutations come from the stream `seeded(seed, split, token_pair, 1, n_train)`.
- **Verification samples.** `sample_batches` builds, per instance, `len(test rows) // N_t` samples of `N_t`
  consecutive test rows of that instance (test rows are in the shuffled order of the split); an instance with fewer
  than `N_t` test rows gives no sample. `N_t` is any integer, the paper uses 1, 2, 4, ..., 64.
- **Random streams.** Every stream is a NumPy `default_rng` keyed by integers: `seeded(seed, split, token_pair, *tags)`
  for the validation rows and the open-set draw, `[54, split, pair index]` (closed set) or `[54, split, draw]` (open set)
  plus the LLM id for the per-LLM round selection, `[4242, draw]` for the Unseen instances. `pair_index` is the position
  of the pair in `TOKEN_PAIRS`, the 31 pairs of the corpus.

## Imputation (`imputation.py`)

A missing value is replaced by the mean of its column over the drawn training rows (`TrainingMeanImputer`: NaN-aware;
a column with no defined training value is filled with 0). `prepare_cell` and `prepare_open_set_cell` fit the means on
the drawn training rows of the cell and apply them to its training, validation and test rows.

## XGBoost classifiers (`xgb.py`)

Every classifier of the method (global, per LLM, and the fold classifiers of the Unseen detector) is an
`XGBClassifier` with the hyper-parameters `XGB_PARAMS` (learning rate 0.05, depth 4, `min_child_weight` 1,
`colsample_bytree` 0.6, `hist` tree method, 64 bins), `random_state=42` and balanced sample weights. A two-instance
group uses the multi-class softmax objective with 2 classes. Outputs are exact log-probabilities (log-softmax of the
margins).

- **Global classifier**: `GLOBAL_ROUNDS = 357` boosting rounds, a constant of the configuration
  (`classifier.global_rounds`), frozen on validation sequences. Its log-probabilities are floored at `-80` (`LOG_FLOOR`)
  and stored in float32; an instance absent from its training data gets the floor.
- **Per-LLM classifiers**: the number of rounds is chosen for each cell by `select_rounds`, a cross-half early
  stopping on the LLM's own training rows: each instance's rows are permuted and cut in two halves, a model fitted on
  one half is early-stopped (patience 60, at most 1000 rounds) on the multi-class log-loss of the other half and
  vice versa, and the final number of rounds is the ceiling of the mean of the two best round counts.
- XGBoost `hist` results depend on the thread count. The global classifier runs with `classifier.n_jobs` threads
  (4 in the paper), the per-LLM classifiers with 1.

## Log-sum aggregation (`log_sum_aggregation.py`)

A classifier trained on single answers gives `log p(m_k | x_i)` over the instances for answer `x_i`. A sample of
`N_t` answers is scored by `s_k = sum_i log p(m_k | x_i)` and attributed to `argmax_k s_k` (ties go to the lowest index).
The prior is uniform because the training classes are balanced. The sum is taken in the dtype of the
log-probabilities (float32 for the global classifier). Any `N_t` is served by the same trained classifier.

## Hierarchical instance classifier (`hierarchical.py`)

1. Stage 1: one global classifier over all instances. For a sample, the log-sum scores of the global classifier pick an
   instance; its LLM `l*` is the selected LLM.
2. Stage 2: one classifier per LLM that has at least 2 instances in the training data, trained on the training rows of
   that LLM's instances. The predicted instance is the argmax, over the instances of `l*`, of the summed per-LLM
   log-probabilities (floored at `-80`); an LLM with one instance returns that instance.

`decide` returns `global_pred` (stage 1 instance), `llm`, `pred` (hierarchical instance), `confidence` and `scores`.
`scores` is the hierarchical vector `K^H`: for the instances of `l*`, the sum over the answers of
`log p_global(LLM of c) + log p_LLM(c)` (the global log-mass of an LLM is the log-sum-exp of its instances'
log-probabilities); for the other instances, the summed global log-probabilities. `confidence` is the softmax mass of
the predicted instance under `K^H`.

## Unseen detector (`unseen_detector.py`)

**Open-set cell.** `n_unseen` instances (23 of 237 in the paper) are drawn by `draw_unseen` with
`default_rng([4242, draw])`, independently of the token pair and the split. They are removed from the training side and
must be labelled Unseen at verification; the others are Known. The instance classifier and the detector are built from
the training rows of the Known instances. A test sample is labelled Unseen (`-1`) when the detector flags it,
otherwise with the instance of the hierarchical classifier.

**Building the detector** (`UnseenDetector`, from the Known training rows; Known instances are held out in turn to
simulate Unseen ones):

1. *Class folds.* `assign_class_folds` splits the Known instances into `n_folds` (10) folds; the instances of one LLM get
   consecutive folds from a random offset, so they are spread over different folds. An LLM with a single Known instance
   is never held out, so a held-out instance always keeps a sibling in training (an error is raised otherwise). For
   each fold `f`, `fit_folds` trains a global classifier and per-LLM classifiers on the instances of the other folds,
   after setting aside `held_out_per_class` (40, at most a third) rows of each of them.
2. *Samples.* Samples of `N_t` rows are scored by the fold's classifiers: `known_batches_per_instance` (10) per Known
   instance from its set-aside rows, `unseen_batches_per_instance` (40) per held-out instance from
   `unseen_rows_per_instance` (40) of its rows. An instance with fewer than `min_rows_per_instance` (120) training rows
   gives no sample.
3. *Statistics.* `sample_statistics` describes each sample by 38 statistics of the classifier's output (the
   log-probabilities of its answers over the instances), not of the bit sequences:

   | group | names |
   |---|---|
   | decision | `margin`, `margin_within_llm`, `llm_margin`, `mean_log_prob_decided`, `vote_share`, `std_log_prob_decided` |
   | gaps | `gap_2` ... `gap_10`, `gap_within_llm_2` ... `gap_within_llm_5`, `llm_mass_gap_12`, `llm_mass_gap_13` |
   | mean posterior | `log_mean_prob_1` ... `log_mean_prob_5`, `log_mean_prob_margin`, `mean_prob_entropy`, `llm_log_odds` |
   | per answer | `max_prob_mean`, `max_prob_std`, `max_prob_q10`, `max_prob_q50`, `max_prob_q90`, `log_prob_decided_q10`, `log_prob_decided_q50`, `log_prob_decided_q90`, `llm_vote_share` |

4. *Regression.* `LogisticDetector` is a regularised logistic regression (scikit-learn `LogisticRegression`, lbfgs,
   `C` = `open_set.regularization_c` = 0.1) on the standardised statistics; label 1 is a held-out-instance sample. The
   Known samples it uses are those the fold's hierarchical decision identifies correctly. `sample_weights` gives every
   source instance the same total weight within its side, the held-out side carrying `unseen_weight_share` (0.097) of the
   weight; weights are computed on all training samples (mean 1) before the misclassified Known samples are dropped.
   The output is the Unseen score `q`, the logit of the Unseen probability.
5. *Threshold.* `tau` is the `1 - target_recall` quantile of the out-of-fold scores of the held-out-instance samples,
   with `target_recall = 0.5` (the median): half of the simulated Unseen samples are flagged. Out of fold means that the
   samples of fold `f` are scored by a regression fitted on the samples of the other folds whose source instance is not
   in fold `f` (`out_of_fold_scores`).

The final regression is fitted on all the samples. At verification, the statistics are computed from the log-probabilities
of the final global classifier (trained on all the Known training rows, outputs over the Known instances, unfloored) and a
sample is flagged when `q >= tau`. The 50 % target applies to the out-of-fold scores that set `tau`; the share of
test Unseen samples that is flagged is a measured output (`unseen_recall`), as are `unseen_precision`, `accuracy`
(an Unseen sample is correct when flagged) and `known_accuracy_closed` (hierarchical decision on the Known samples).
`UnseenDetector.build(n_t, target_recall)` accepts any target; `TARGET_RECALL_GRID` lists 0.05 ... 0.95.

## Tests that pin each module

| module | tests |
|---|---|
| `protocol.py` | `tests/test_protocol.py` (draws are prefixes of one permutation, validation rows disjoint from the pool and capped at a third, stratified reproducible splits, consecutive samples of one class, row selection and `run_ts` fill, `pair_index`); `tests/test_closed_set_reference.py` (train and test rows against the stored reference) |
| `imputation.py` | `tests/test_imputation.py` |
| `xgb.py` | through `tests/test_closed_set_reference.py` (global and per-LLM log-probabilities to 1e-9); `tests/test_experiment_config_schema.py` pins the default of 357 rounds |
| `log_sum_aggregation.py` | `tests/test_log_sum_aggregation.py`: sums over the rows of each batch, single query, floor, empty batch list |
| `hierarchical.py` | `tests/test_hierarchical.py` (LLM of an instance, decision inside the selected LLM, routing); `tests/test_closed_set_reference.py` (decisions, `K^H`, confidences) |
| `unseen_detector.py` | `tests/test_unseen_detector.py` (38 finite statistics, folds, weights, threshold quantile, detector, decision rule); `tests/test_open_set_reference.py` (folds, samples, out-of-fold scores, `tau`, test scores, flags, labels, accuracy, recall) |
| the cell driver | `tests/test_pipeline.py` (closed-set smoke on the fixture) |

The regression tests compare one closed-set and one open-set configuration on the in-repo fixture with the reference outputs in
`tests/fixtures/reference/` (tolerance 1e-9; XGBoost 3.2 with 4 threads for the global classifier).

```
python -m pytest tests/test_protocol.py tests/test_imputation.py tests/test_log_sum_aggregation.py tests/test_hierarchical.py tests/test_unseen_detector.py -q
```

## Related documentation

[`../README.md`](../README.md) · [`overview.md`](overview.md) · [`features.md`](features.md) · [`experiment.md`](experiment.md) ·
[`configuration-advanced.md`](configuration-advanced.md) · [`testing.md`](testing.md) · [`../reproduction.md`](../reproduction.md) ·
[`../reproduction/recompute.md`](../reproduction/recompute.md)

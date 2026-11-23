# Features

The package `src/audit_llm/features/` turns the generated bit sequences into the fingerprint of the paper: 57 statistics
per sequence, stored in a feature grid. It has three modules.

| file | content |
|---|---|
| `nist.py` | 28 of the 31 NIST test statistics, computed with `src/audit_llm/Bits_Generation/NIST_Tests_lib`, and the sequence length |
| `statistics.py` | the 25 complementary statistics and the 3 other NIST statistics, computed directly from the bits (numba-compiled when numba is installed) |
| `grid.py` | `build_grid`, `Grid`, `TokenPairGrid`, `feature_names`: answers table to feature tensors, and their on-disk format |

The 57 features are the 31 test statistics of the NIST suite, the sequence length and 25 complementary statistics
(19 on the dependence on past bits, 6 on the opening of the sequence), as in the paper. The columns are stored in the
order below, which `feature_names()` returns: the columns of `nist.py` (indices 0 to 28), then those of `statistics.py`
(indices 29 to 56). `Grid.load` refuses a grid whose stored feature names differ from it.

## The 57 statistics

### Columns of `nist.py`: indices 0 to 28

Each NIST test of `src/audit_llm/Bits_Generation/NIST_Tests_lib` returns (p-value, pass flag, test statistic); the
feature set keeps the test statistic (suffix `_ts`) of 28 tests, then the sequence length.

| column names | paper group | test |
|---|---|---|
| `block frequency_30_ts`, `block frequency_100_ts` | NIST | block frequency, blocks of 30 and 100 bits |
| `non overlapping_<t>_75_ts` | NIST | non-overlapping template matching, blocks of 75 bits, 8 templates `<t>` in `01 10 001 010 011 100 101 110` |
| `overlapping patterns_<t>_75_ts` | NIST | overlapping template matching, blocks of 75 bits, the same 8 templates |
| `cumulative sums 1s_ts`, `cumulative sums 2s_ts` | NIST | cumulative sums, forward and backward |
| `random excursion_ts`, `monobit_ts`, `run_ts`, `longest one block_ts` | NIST | random excursions, frequency (monobit), runs, longest run of ones in a block |
| `binary matrix rank_ts`, `spectral_ts`, `linear complexity_ts`, `approximate entropy_ts` | NIST | binary matrix rank, discrete Fourier (spectral), linear complexity, approximate entropy (`m` = 10) |
| `seq_length` | sequence length | length of the sequence in bits |

### Columns of `statistics.py`: indices 29 to 56

They are functions of the bits only (no fitted parameter).

| name | paper group | definition |
|---|---|---|
| `longest_run0`, `longest_run1` | past dependence | longest run of 0s, of 1s |
| `switch_rate` | past dependence | share of positions where the bit differs from the preceding one |
| `p1_k0` | NIST (second Monobit statistic) | share of ones, `(ones + 0.5) / (n + 1)` |
| `p1_k1_c0` ... `p1_k3_c7` (14 columns) | past dependence | P(bit = 1 given the preceding `k` bits equal context `c`) for `k` = 1, 2, 3, the context being the preceding `k` bits read as an integer (oldest bit first); add-1/2 smoothing `(ones + 0.5) / (seen + 1)`, so an unseen context gives 0.5 |
| `haz0_l1`, `haz1_l1` | past dependence | probability that a run of 0s (of 1s) stops after one bit; the last run, cut by the end of the sequence, is not counted; add-1/2 smoothing |
| `apen_m3`, `apen_m4` | NIST (approximate entropy) | approximate entropy with `m` = 3 and 4 (overlapping, non-cyclic m-grams) |
| `op_run0_sym`, `op_run0_len` | opening | symbol and length of the first run |
| `op_first8` | opening | the first 8 bits read as an integer |
| `op_w0_ones` | opening | share of ones among the first 16 bits |
| `op_quart0_ones`, `op_quart3_ones` | opening | share of ones in the first and in the last quarter of the sequence |

The 15 `p1_*` columns are `p1_k0`, `p1_k1_c0`, `p1_k1_c1`, `p1_k2_c0` ... `p1_k2_c3`, `p1_k3_c0` ... `p1_k3_c7`.
Per paper group: NIST 28 + 3 = 31, sequence length 1, past dependence 2 + 1 + 14 + 2 = 19, opening 6.

## NaN semantics

- The tests of `nist.py` run only on sequences longer than 10 bits (`MIN_TEST_LENGTH`); a shorter sequence has NaN in
  every `_ts` column and keeps its `seq_length`. A test that raises an exception, or returns a statistic that is not a plain
  number, gives NaN for that column only.
- A statistic of `statistics.py` is NaN for a sequence shorter than 2 bits.
- `build_grid` sets all 57 columns to NaN for an answer shorter than `min_seq_length` bits (100 in the paper; an empty,
  missing or null answer counts as an empty sequence), `seq_length` included.
- `select_rows` in `src/audit_llm/Classification/protocol.py` turns non-finite values into NaN, drops an answer with more
  than 80 % of the columns of `nist.py` undefined, and sets `run_ts` to 0 where it is undefined on an otherwise defined
  answer. A remaining missing value is replaced by the mean of its column over the drawn training rows
  ([`classification.md`](classification.md)).

## The grid

An instance (class) is a (model, temperature, system prompt) triple of the answers table, named
`<model>_temp-<temperature>_sp-<system prompt index>` (for example `<model>_temp-0.2_sp--1`;
a quantized instance has `<model>@@<quant>` in the model column). Classes are sorted by (model, temperature, system prompt index) and share
one order over all token pairs. For a token pair, row `j` of a class is its `j`-th answer by query index.

`build_grid(answers_df, token_pairs=None, min_seq_length=100, n_jobs=1)` needs the columns `Token_pair`, `Model`,
`Dataset_Question Index`, `Answer` (a string of `0` and `1`), `temperature`, `system_prompt_idx`. It raises an error on a
duplicate query index within a (token pair, class), and on an answer that is not a bit string.

| field of `TokenPairGrid` | shape and type | content |
|---|---|---|
| `features` | `(S, C, 57)` float64 | the 57 statistics, NaN where undefined |
| `bits` | `(S, C, L)` uint8 | the bits (0/1); positions past the end of a sequence hold the pad value 2 |
| `lens` | `(S, C)` int16 | length of each sequence in bits (0 = no answer) |
| `query_index` | `(S, C)` int32 | query index of each answer (-1 = no answer) |

`S` is the largest number of answers of a class, `C` the number of classes, `L` the longest answer of the token pair.

### On-disk format (what `build-grid` writes)

```
<grid dir>/
  meta.json            class_names, token_pairs, feature_names, min_seq_length, pad_value
  <token pair>/
    features.npy  bits.npy  lens.npy  query_index.npy
```

`Grid.load(<grid dir>)` reads `<grid dir>/meta.json` and opens `features.npy` and `bits.npy` memory-mapped on first access of a
token pair (`grid.pair("ali-yg")`), so a grid larger than memory can be used; `lens` and `query_index` are loaded.
`Grid.select_classes(names)` returns an in-memory grid restricted to the named instances (used by the `scenario` key of
a config, [`configuration-advanced.md`](configuration-advanced.md)). On the in-repo fixture (22 instances, 200
sequences each, 2 token pairs) a grid takes 4.1 MB per token pair.

```
python -m audit_llm.pipeline build-grid --answers tests/fixtures/answers.parquet --out out/grid --n-jobs 4
```

Options: `--answers` (a parquet file or a directory of parquet files), `--out`, `--token-pairs a,b,...` (default: all
pairs of the table), `--min-seq-length` (default 100), `--n-jobs` (processes for the NIST tests, default: the CPU
count). On the full corpus the NIST tests take about 13 CPU-hours; `make grid` is the same command on the fetched
generations (`ANSWERS=<dir>` for other answers) with the output in `out/grid`. `python -m audit_llm.pipeline run --answers ...` computes the same grid in memory without saving
it ([`experiment.md`](experiment.md)).

## Numba and threads

- `statistics.py` compiles its kernel with numba when numba is installed (`poetry install --extras speed`); otherwise
  the same code runs in plain Python, much slower. The kernel is `parallel=True` over sequences, with no on-disk
  compilation cache.
- `FLIPS_NUM_THREADS` sets the number of numba threads (default 4, capped at the number of threads numba
  detects). It is read once, when `audit_llm.features.statistics` is imported.
- The NIST columns do not use numba: `nist_columns` distributes the bit strings over a `multiprocessing` pool of
  `--n-jobs` processes. The two stages run one after the other.

## Determinism

- The statistics of `statistics.py` are pure functions of the bits, and each sequence is processed independently. On the
  fixture, 1 and 4 numba threads give identical values.
- `build_grid` sorts the answers by class and query index, so the grid does not depend on the order of the rows of the
  answers table.
- Features of the fixture are compared with stored reference values to 1e-9 by `tests/test_closed_set_reference.py`
  (`test_features_match`); `tests/test_features.py` checks hand-computed values of the statistics of `statistics.py` on short
  sequences, the names and order of the 57 columns, the short-sequence NaN rule, and the grid layout on a small table.
- Downstream, XGBoost `hist` results depend on the thread count of the classifier (`classifier.n_jobs`), not on
  `FLIPS_NUM_THREADS` ([`classification.md`](classification.md)).

## Related documentation

[`../README.md`](../README.md) · [`overview.md`](overview.md) · [`classification.md`](classification.md) ·
[`experiment.md`](experiment.md) · [`dataset.md`](dataset.md) · [`testing.md`](testing.md) ·
[`../reproduction/recompute.md`](../reproduction/recompute.md)

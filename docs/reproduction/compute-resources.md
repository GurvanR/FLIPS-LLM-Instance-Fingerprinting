# Compute resources

What each reproduction tier costs. Tier definitions: [`../reproduction.md`](../reproduction.md).

| Tier | Work | Cost | Notes |
|---|---|---|---|
| 0 | `make test` | about 70 s on 12 CPU cores (1-2 min) | 4 threads for the XGBoost fits |
| 0 | `make smoke` | about 35 s wall (20 s for the 4 cells) | 22 instances, 2 token pairs, 2 splits |
| 0 | `make smoke-open-set` | about 1.5 min wall | the closed-set smoke run is about 35 s |
| 0 | `make gate-a`, `make gate-golden` | about 35 s wall each | each runs the smoke configuration |
| 1 | `make paper` | about 20 s, one core, numpy / scipy / matplotlib only | reads 4.4 MB of results (2.2 MB zipped) |
| 2 | `make grid` | about 13 CPU-hours (about 1.7 h with 12 processes, about 40 min with 24) | the 57 statistics for 237 instances x 31 token pairs x 500 answers; parallel over answers (`--n-jobs`); `--extras speed` compiles the statistics of `features/statistics.py` |
| 2 | `make recompute-closed` | about 80 core-hours (about 4 h wall on 24 cores with `n_workers: 6`) | 155 cells (31 token pairs x 5 splits); per cell about 1 800 core-seconds, or about 490 s wall with 6 cells in parallel at 4 threads each on 24 cores |
| 3 | `make recompute-open` | about 640 core-hours | 31 token pairs x 5 (split, draw) cells; per draw about 32 x 4 core-hours, dominated by the 10 held-out-fold classifiers of every cell |
| 4 | `make generate` (all 237 instances) | about 200 GPU-hours | 237 instances x 31 token pairs x 500 answers; vLLM, `bfloat16` weights, GPUs with 80 GB for the large models |

## Reading the cost

- **Wall time** for tiers 2 and 3 is the core-hours divided by the cores in use. The figures above were measured on a 24-core machine; they depend on the CPU and on the load. `n_workers` (cells in
  parallel) times `classifier.n_jobs` (threads per cell) should not exceed the number of cores. Changing the
  thread count of the global classifier changes the bitwise result ([`recompute.md`](recompute.md#determinism)).
- **Disk.** Results archive: 4.4 MB unpacked. Release data archive: 4.6 GB, 6.4 GB unpacked (the generations
  0.7 GB, the LLMmap run 5.6 GB), about 11 GB of free disk while it unpacks. The feature grid takes 3.7 GB for the 31 token pairs.
- **Memory** (tiers 2 and 3). A run with the shipped configs, which set
  `scenario: main`, copies the grid into memory: the parent process holds about 12 GB and each of the `n_workers`
  workers about 5 GB (about 45 GB for 7 workers). A machine with 32 GB can run `n_workers` 1 to 3; set `n_workers` by the
  memory available ([`recompute.md`](recompute.md#running-in-parallel-and-in-parts)).
- **Tier 4** cost scales with the request table: the full `FLiPS_ICML.csv` has 20 request cells per model, of which
  the 237 instances use 8 (4 for quantized instances), so restricting the requests to the cells you need
  ([`generation.md`](generation.md)) is the main saving.
- **Measuring.** `/usr/bin/time -v <command>` reports wall time, CPU time and peak memory of any target; the
  per-cell `seconds` field of the pipeline's cell files gives the time of each cell.

# Reproduction

How to rerun the paper's code, from a seconds-long smoke test to GPU generation. Five tiers; each one
needs more input and compute than the one before it and verifies more of the pipeline.

| Tier | What it does | Input | Cost | Targets |
|---|---|---|---|---|
| 0 | tests, regression tests, smoke runs of both protocols | in-repo fixture (`tests/fixtures/`, 2.4 MB) | about 2 min, CPU | `make test`, `make smoke`, `make smoke-open-set`, `make gate-a`, `make gate-golden` |
| 1 | regenerates the computed figures, tables and numbers listed in [`paper/README.md`](../paper/README.md) and compares them with the reference files | results archive `flips_results.zip` (4.4 MB unpacked) | about 20 s, CPU | `make fetch`, `make paper` |
| 2 | recomputes the closed-set results from the generated answers | release data archive `flips_release_data.tar.gz` (the generations; 4.6 GB with the LLMmap run) | about 80 core-hours | `make fetch-generations`, `make grid`, `make recompute-closed` |
| 3 | recomputes the open-set results | generations | about 640 core-hours | `make recompute-open` |
| 4 | regenerates the answers with vLLM | model weights, GPU | about 200 GPU-hours | `make generate` |

Compute figures for tiers 2 to 4 are in [`reproduction/compute-resources.md`](reproduction/compute-resources.md).

## Verification status

| Tier | Status |
|---|---|
| 0 | executed: all tests pass; smoke runs and both gates pass. |
| 1 | executed: from the archive, `paper/make_paper.py` runs all 18 steps and `paper/check.py` reports every compared file equal to `paper/expected/`. |
| 2, 3 | the code ships and is regression-tested against stored reference outputs on the fixture (per-sequence log-probabilities, decisions, Unseen-detector scores: maximum absolute difference 0). On the generations of the release data archive, the feature grid of two token pairs (237 instances) and their closed-set cells of split 0 run end to end. The full runs are not re-run in the tests. |
| 4 | the code ships and its GPU-free parts are unit-tested (`tests/generation/`). End-to-end generation is not covered by the tests; sampling is unseeded, so a rerun is not bitwise. The items that need a GPU are listed in [`reproduction/generation.md`](reproduction/generation.md#verification-status). |

## 1. Prerequisites and install

- Python 3.11 or newer, [Poetry](https://python-poetry.org/), git. Work from a clone or an unpacked archive of the
  repository: the repository root is the directory that holds `pyproject.toml`, so a plain `pip install` into
  `site-packages` cannot find `datasets/`, `config/` and `XP_configs/`. `pip install -e .` inside the repository works.
- Tier 4 only: a GPU with at least 24 GB of memory (large models need several), and vLLM.

```bash
git clone https://github.com/GurvanR/FLIPS-LLM-Instance-Fingerprinting.git
cd FLIPS-LLM-Instance-Fingerprinting
poetry install                      # CPU stack: tiers 0 to 3
poetry install --extras speed       # optional: numba for the feature statistics (tier 2)
poetry install --extras generation  # tier 4: torch, transformers, jinja2; vLLM is installed separately
cp .env.example .env                # optional: edit, then `set -a; source .env; set +a` (data root, model cache; see the file)
```

Run the commands inside the environment: `poetry run make <target>` or `poetry shell` first. Outside it,
pass an interpreter that has `audit_llm` installed: `make smoke PYTHON=.venv/bin/python`.

`tests/paper/` and tier 1 need only numpy, scipy and matplotlib; the pinned versions of the reference
outputs are in [`paper/README.md`](../paper/README.md).

## 2. Tier 0: tests and smoke runs

```bash
make test           # pytest tests: unit tests, regression tests on the fixture, generation unit tests
make smoke          # closed-set pipeline on the fixture, writes out/smoke/
make smoke-open-set # open-set pipeline on the fixture, writes out/smoke_open_set/
make gate-a         # smoke run plus sanity checks (every cell present, accuracy above chance and increasing with N_t)
make gate-golden    # smoke run compared with tests/fixtures/reference/smoke_summary.json
```

The fixture (`tests/fixtures/answers.parquet`) holds 22 instances on 2 token pairs with 200 sequences each
(variants of three LLMs). The smoke configs (`XP_configs/smoke.yaml`, `smoke_open_set.yaml`) run the paper's
method with the protocol scaled to that size. Both write one JSON file per cell and a `summary.json`; see
[`codebase/experiment.md`](codebase/experiment.md).

The regression tests (`tests/test_closed_set_reference.py`, `tests/test_open_set_reference.py`) compare the package with stored
reference outputs of the method on the fixture: statistics, training draws, log-probabilities of both stages,
decisions, Unseen-detector scores and thresholds. Tests that need the generation extras are skipped when
those are absent; tests marked `requires_data` need the results archive (below) and are skipped without it.

`make gate-golden` is exact for the pinned environment (XGBoost 3.2, 4 threads); for other library versions
run `python scripts/gates/gate_golden.py --tolerance 0.02`.

## 3. Tier 1: analysis from the results archive

```bash
make fetch          # download flips_results.zip, check its hash, unpack into data/results/, check every file
make paper          # regenerate into paper/out/, then compare with paper/expected/
```

- `make fetch` runs `scripts/fetch_data.py` ([`data/README.md`](../data/README.md)): it resolves the Zenodo
  record of the release (`--url` or `$FLIPS_DATA_URL` select another source),
  checks the archive against `data/archives.sha256`, unpacks `results/` and checks each file against
  `data/manifest.sha256`. `make fetch-check` verifies the files on disk without network access.
- `make paper` runs `paper/make_paper.py` (18 steps in a defined order, about 20 s) and then `paper/check.py`.
  To read an archive unpacked elsewhere: `make paper FLIPS_DATA=/path/to/unpacked` (the directory that
  contains `results/`). `make paper-list` prints the steps.
- Outputs, under `paper/out/`: `numbers.tex` and `figures/*.tex` (every macro quoted in the text), `figures/<name>.pdf`
  and `.png` with a JSON sidecar of the plotted data, and TeX fragments for the tables.
  [`paper/README.md`](../paper/README.md) lists which module produces which figure, table and numbers;
  that list is what this tier regenerates. Static illustrations and hand-written tables are not generated; Figure 3 (white-box log-probability analysis) and the LLMmap bars of Figure 1 (constants) are not regenerated.
- The comparison: the macro files are compared macro by macro (a missing, extra or different macro fails), the
  table fragment byte for byte, the JSON sidecars as parsed JSON (floats to 1e-9). The PDF and PNG figures are
  not compared, because their bytes depend on the matplotlib and freetype builds. A successful run ends with
  `RESULT: all equal`.

## 4. Tiers 2 and 3: recompute from the generations

```bash
make fetch-generations                      # release data archive -> Productions/FLiPS_ICML_run/ (and Productions/LLMmap_ICML_run/)
make grid                                   # NIST tests and statistics of every answer -> out/grid
make recompute-closed                       # closed-set protocol of the paper -> out/closed_set
make recompute-open                         # open-set protocol of the paper   -> out/open_set
```

`make grid` reads `ANSWERS`, by default `Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers` (the
generations of the release data archive); set `ANSWERS=<dir>` to use the answers of your own tier-4 run, which
have the same files and columns.
[`reproduction/recompute.md`](reproduction/recompute.md) describes the protocols, the configs, the cost
per cell and the determinism requirements. The pipeline's cell and summary files hold the per-sample
labels, predictions and accuracies per `N_t` in the pipeline's own format ([`codebase/experiment.md`](codebase/experiment.md));
`paper/` reads the format of the results archive, so a recomputed run is compared with the archive through
the accuracies per `N_t` in `summary.json`, not through `make paper`.

## 5. Tier 4: generation

```bash
make generate MODEL=Qwen/Qwen2-7B-Instruct SUB_RUN=qwen2_7b
make generate MODEL=Qwen/Qwen2-7B-Instruct QUANTIZATION=fp8 SUB_RUN=qwen2_7b_fp8
```

One call generates the answers of one LLM over the request table `datasets/Bits_Datasets/FLiPS_ICML.csv`
(abliterated instances use `FLiPS_ICML_uncensored.csv`) and parses them into bit sequences.
Hardware sizing, vLLM installation, SLURM usage and the merge of several runs are in
[`reproduction/generation.md`](reproduction/generation.md). Your own instances: see
[`reproduction/custom-scenario.md`](reproduction/custom-scenario.md).

## 6. Determinism

- Tier 1: deterministic given the archive.
- Tiers 2 and 3: bitwise identical results need XGBoost 3.2 and 4 threads for the global classifiers
  (`classifier.n_jobs`, set in the configs); the histogram method depends on the thread count. The paper's
  results were computed with scikit-learn 1.8; `poetry.lock` pins the environment.
- Tier 4: sampling runs at temperature above 0 with an unseeded sampler and GPU reductions are not
  bit-reproducible, so the generated answers differ between runs.

## 7. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `poetry install` aborts with `GetPassWarning` or `EOFError` in a session without a terminal (ssh command, job, container) | Poetry asks the system keyring for a password; run it with `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install` |
| `ModuleNotFoundError: No module named 'audit_llm'` | the command ran outside the environment: `poetry run make ...` or `PYTHON=<interpreter>` |
| paths such as `datasets/` not found | the package is not installed from a clone; `pip install -e .` inside the clone |
| `fetch_data.py` cannot reach the record | no network access, or the record is unavailable; pass `--url <archive URL>` or set `FLIPS_DATA_URL` |
| `make paper` cannot find `results/` | `FLIPS_DATA` must be the directory that contains `results/` |
| `make gate-golden` reports differences | other library versions than the pinned ones; use `--tolerance` as above |

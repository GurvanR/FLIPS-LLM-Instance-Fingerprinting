# FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data license: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](LICENSES/CC-BY-4.0.txt)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-3776ab.svg)](pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03330-b31b1b.svg)](https://arxiv.org/abs/2606.03330)

Code and data of the paper *FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences*
(Richardeau, Dashyan, Le Merrer, Tredan; [arXiv:2606.03330](https://arxiv.org/abs/2606.03330)).

## Abstract

Literature reveals that a Large Language Model's (LLM) behavior is not only conditioned by its original weights but also its instance-level parameters, such as instructional prompt, sampling configuration or quantization. A model that generates safe outputs under one configuration may produce toxic content under another. However, current LLM identification techniques (such as fingerprinting) focus on intellectual property protection, and their design favors robustness to changes in these instance-level parameters. This poses a critical challenge for AI regulation in which compliance assessments target actual deployed behaviors, not model provenance. In this paper, we introduce instance-level fingerprinting, a regulator-oriented paradigm that distinguishes configurations of the same LLM. Our method, FLIPS, exploits biases in generated binary random sequences. On a benchmark of 237 model instances, it identifies the instance with **77% accuracy from 32 queries** to the target, and with **60% from 8 queries**, against at most 33% for the adapted LLMmap baseline. Recognizing that a target is unseen, also called open-set challenge, is much harder. When 10% of instances are unseen, FLIPS realises a **recall of 41%** on the unseen instances, at a **precision of 23%**, i.e. more than twice better than random. Closed-set instance fingerprinting is thus practical with a few dozen queries, whereas the open-set setting remains an open problem.

## Method in brief

FLIPS queries a target instance with one prompt template that asks for a random binary sequence over a pair of
tokens (for example `0`/`1`, or `car`/`sun` mapped to bits), and exploits the biases of the generated
sequences.

- **Fingerprint.** Each answer is summarised by 57 statistics of its bit sequence: the 31 test statistics of the
  NIST randomness test suite, the sequence length and 25 complementary statistics (dependence on past bits,
  opening bits).
- **Closed set.** A hierarchical instance classifier decides which reference instance the target is. Stage 1
  is an XGBoost classifier over all instances and picks the LLM; stage 2 is one XGBoost classifier per LLM and
  picks the instance among that LLM's instances. The per-answer log-probabilities of the `N_t` queries of a
  verification sample are summed (log-sum aggregation), so the verification budget `N_t` is chosen at
  deployment time without retraining.
- **Open set.** An Unseen detector, a regularised logistic regression on statistics of the classifier's
  output, decides whether the target belongs to none of the reference instances. It is trained on *simulated*
  Unseen instances (reference instances held out in turn) and its threshold flags 50 % of them, so no real
  Unseen instance is needed at extraction time.
- **Benchmark.** 237 instances from 25 open-weight LLMs (temperature, system prompt, `int4` / `fp8`
  quantization, abliteration), 31 token pairs, 500 sequences per instance and pair, `N_r = 300` training
  sequences per instance, `N_t` from 1 to 64.

Main results (means over 31 token pairs, 5 splits; produced by [`paper/`](paper/), see Mode A below):

| queries `N_t` | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| closed-set accuracy | 23.7 % | 60.1 % | 76.6 % | 82.6 % |

Chance is 0.42 % (1 / 237). The adapted LLMmap baseline, whose stored results are plotted in the paper's
accuracy-versus-queries figure, reaches at most 33.4 % (conventions of that curve:
[`paper/README.md`](paper/README.md)). In the open set, with 23 of the 237 instances Unseen, FLIPS flags
41.1 % of the Unseen targets at a precision of 22.9 % at `N_t = 32`.

## A note on naming

The repository and all documentation are branded **FLIPS**, but the importable Python package is
`audit_llm` (its historical name), so you `import audit_llm` even though everything else says FLIPS. The
installable distribution is named `flips`.

## Repository map

| Path | Role |
|---|---|
| `src/audit_llm/` | The installable package (`import audit_llm`): `features/` (the 57 statistics and the feature grid), `Classification/` (hierarchical classifier, Unseen detector, protocol), `pipeline.py` (experiment driver), `LLM_Classes/` and `Bits_Generation/` (generation, parsing, the NIST test suite), `scenarios/` (instance selection). |
| `paper/` | Analysis tier: regenerates the paper's computed figures and tables and the numbers its text quotes through macros from the results archive (except Figure 3 and the LLMmap bars of Figure 1; see `paper/README.md`). |
| `XP_configs/` | Experiment configurations: `closed_set.yaml`, `open_set.yaml` (the paper's protocols), `smoke*.yaml` (fixture-sized), `example_custom/`. |
| `scripts/` | `fetch_data.py` (data download and verification), `Run_Inferences.py` and the SLURM / merge / parsing helpers (generation), `gates/` (smoke gates). |
| `datasets/` | Request tables (`Bits_Datasets/*.csv`) and prompt assembly (`prompt_config_index.yaml`, `system_prompts.json`). |
| `config/` | Model registry (`models.yaml`) and scenarios (`scenarios/`). |
| `data/` | Target of the data download; file hashes in `manifest.sha256`. |
| `tests/` | Unit tests, regression tests, and the in-repo fixture (`tests/fixtures/`). |
| `docs/` | Reproduction guide and codebase documentation ([`docs/README.md`](docs/README.md)). |

## Environment setup

Prerequisites: **Python 3.11 or newer** and **[Poetry](https://python-poetry.org/)**. Work from a clone or an
unpacked archive of the repository: the code locates the repository root as the directory that holds
`pyproject.toml`.

```bash
git clone https://github.com/GurvanR/FLIPS-LLM-Instance-Fingerprinting.git
cd FLIPS-LLM-Instance-Fingerprinting
poetry install                      # CPU stack: everything except generation (all of Mode A)
poetry install --extras speed       # optional: numba, compiles the feature statistics
poetry install --extras generation  # optional: GPU generation stack for Mode B (vLLM is installed separately)
cp .env.example .env                # optional: data root, output root, model cache (see the file)
```

The plain `poetry install` is all you need for Mode A. The commands below run inside the environment
(`poetry run make ...`, or `poetry shell` first).

## Reproduction at a glance

Five tiers, from seconds to GPU-days. Tiers 0 to 3 are **Mode A** (CPU, from existing generations); tier 4 is
**Mode B** (GPU, regenerates the answers). Each tier is described, with its inputs, outputs and what it
verifies, in [`docs/reproduction.md`](docs/reproduction.md).

| Tier | Mode | What | Input | Cost | Command |
|---|---|---|---|---|---|
| 0 | A (a) | tests, smoke runs, regression tests against stored reference outputs on a fixture | in-repo fixture (2.4 MB) | about 2 min, CPU | `make test`, `make smoke` |
| 1 | A (b) | regenerate the computed figures, tables and numbers listed in [`paper/README.md`](paper/README.md) | results archive (4.4 MB) | about 20 s, CPU | `make fetch`, `make paper` |
| 2 | A (c) | recompute the closed-set results | generations (release data archive) | about 80 core-hours | `make fetch-generations`, `make grid`, `make recompute-closed` |
| 3 | A (c) | recompute the open-set results | generations | about 640 core-hours | `make recompute-open` |
| 4 | B | regenerate the answers | GPU, vLLM, model weights | about 200 GPU-hours | `make generate` |

`make help` lists every target by tier.

## Mode A — analysis quickstart (no GPU)

Mode A consumes existing generations or results and runs on CPU.

### (a) Out-of-box — zero download

```bash
make test           # unit tests and regression tests on the in-repo fixture (about 1-2 min)
make smoke          # closed-set pipeline on the fixture: 22 instances, 2 token pairs (under a minute)
make smoke-open-set # open-set pipeline on the fixture
make gate-a         # smoke run plus sanity checks on its accuracies
make gate-golden    # smoke run compared with the stored reference summary
```

`make smoke` writes `out/smoke/summary.json` (accuracy per `N_t`, per cell) and one JSON file per cell under
`out/smoke/cells/`. The fixture is small, so these runs check that the pipeline works, not the paper's numbers.

### (b) Data-gated — the paper's figures and numbers

```bash
make fetch          # download and verify the results archive into data/results/
make paper          # regenerate into paper/out/, then compare with paper/expected/
```

This is the entry point for checking the paper's figures and numbers: it reads the cached results of the full
runs (tiers 2 to 4) and compares its output with the reference files in [`paper/expected/`](paper/expected/)
(the numbers macro by macro, the figure data as parsed JSON).

### (c) Recompute the results from the generations

```bash
make fetch-generations   # download and verify the release data archive into Productions/
make grid                # feature grid of the 237 instances
make recompute-closed    # closed-set protocol of the paper
make recompute-open      # open-set protocol of the paper
```

These runs need the compute stated in the table above; see
[`docs/reproduction/recompute.md`](docs/reproduction/recompute.md) and
[`docs/reproduction/compute-resources.md`](docs/reproduction/compute-resources.md). Some appendix results are
in the results archive only, without the code that computed them: the classifier comparison, the truncation to
the first 100 bits, the distance-2 open-set protocol, the feature importances, the calibration of the
closed-set confidence, and the LLMmap baseline
([`docs/reproduction/recompute.md`](docs/reproduction/recompute.md#results-without-a-recompute-path)).

## Mode B — inference from scratch (GPU required)

Mode B regenerates the answers, then feeds Mode A (c). It reproduces the *methodology*, not the exact paper
numbers (see the non-determinism caveats below). It needs a GPU, `poetry install --extras generation`, a
matching vLLM build and the model weights.

```bash
make generate MODEL=<hf id> [QUANTIZATION=fp8|bitsandbytes_int4] SUB_RUN=<name> [GPUS=4]
```

One call generates one instance family. For the full setup and cluster dispatch, see
[`docs/reproduction/generation.md`](docs/reproduction/generation.md).

## Run your own scenario

The same pipeline runs on any set of instances you generate: inference produces the answers, a small
scenario YAML selects which (model, temperature, system prompt, quantization) cells become classes. A runnable
example on the in-repo fixture (about 15 s on a CPU, no download):

```bash
python -m audit_llm.pipeline run --config XP_configs/example_custom/example_custom.yaml \
    --answers tests/fixtures/answers.parquet --out out/example_custom
```

Copy [`config/scenarios/example_custom.yaml`](config/scenarios/example_custom.yaml) and
[`XP_configs/example_custom/`](XP_configs/example_custom/), edit the model list and variation grid, and run. The
full walkthrough, including generating new instances, is in
[`docs/reproduction/custom-scenario.md`](docs/reproduction/custom-scenario.md).

## Experiment → command

| Experiment | What | From stored results (Mode A (b)) | Recompute (Mode A (c)) |
|---|---|---|---|
| Closed set | instance identification among the 237 instances, accuracy versus `N_t` | `make paper` | `make recompute-closed` ([`XP_configs/closed_set.yaml`](XP_configs/closed_set.yaml)) |
| Open set | Unseen detection with 10 % of the instances held out | `make paper` | `make recompute-open` ([`XP_configs/open_set.yaml`](XP_configs/open_set.yaml)) |
| LLMmap baseline | adapted LLMmap, stored results | `make paper` | not recomputed; run data in the release data archive |
| Appendix analyses | classifier comparison, truncation, distance-2 protocol, feature importances, calibration | `make paper` | stored results only (see Mode A (c)) |

## Figure / table → command

Every computed figure, table and number of the paper is regenerated by `make paper` into `paper/out/`, then
compared with [`paper/expected/`](paper/expected/). `make paper-list` prints the build steps.
[`paper/README.md`](paper/README.md) maps each module to the figures and tables it produces, and lists the two
computed elements that are not regenerated (Figure 3 and the LLMmap bars of Figure 1).

## Data

| Archive | Content | Size |
|---|---|---|
| `flips_results.zip` | cached results read by `paper/` (closed set, open set, NIST scores, corpus statistics, stored baseline results) | 2.2 MB zipped, 4.4 MB unpacked |
| `flips_release_data.tar.gz` | the generated answers and token ids of the 237 instances (`Productions/FLiPS_ICML_run/`, input of Mode A (c)) and the run data of the LLMmap baseline (configurations, trained classifiers, logs, evaluations), provided as they are (`Productions/LLMmap_ICML_run/`) | 4.6 GB zipped, 6.4 GB unpacked |

Both are files of the Zenodo record of the release
([doi:10.5281/zenodo.24461715](https://doi.org/10.5281/zenodo.24461715)). `python scripts/fetch_data.py`
downloads the results archive (`make fetch`); `python scripts/fetch_data.py --with-generations` also downloads
the release data archive into `Productions/` (`make fetch-generations`). Each archive's hash is checked, then
every unpacked file against [`data/manifest.sha256`](data/manifest.sha256). Details, flags and the data root
convention (`$FLIPS_DATA`): [`data/README.md`](data/README.md). Model weights are not redistributed; they come
from the model providers under their own licenses.

## Non-determinism caveats

- Mode A (b) outputs are deterministic given the archive; the bytes of the PDF and PNG figures depend on the
  matplotlib / freetype build, so `paper/check.py` compares the numbers and the plotted data (JSON sidecars),
  not the figure files.
- Mode A (c) is bitwise reproducible only with XGBoost 3.2 and 4 threads for the global classifiers
  (`classifier.n_jobs` in the configs): XGBoost's histogram method depends on the thread count.
- Mode B samples at temperature above 0 with an unseeded sampler; GPU reductions are not bit-reproducible
  either. A rerun reproduces the distribution of the answers, not the answers themselves.

## Citation

If you use FLIPS, please cite the paper. The preprint is on arXiv:
[arXiv:2606.03330](https://arxiv.org/abs/2606.03330). Metadata also lives in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{richardeau2026flipsinstancefingerprintingllmspseudorandom,
      title={FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences},
      author={Gurvan Richardeau and Gohar Dashyan and Erwan Le Merrer and Gilles Tredan},
      year={2026},
      eprint={2606.03330},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03330},
}
```

## License

- **Code:** MIT, see [`LICENSE`](LICENSE).
- **Data:** CC-BY-4.0, see [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt). Released model *outputs*
  additionally inherit each model's own terms (Cohere Command-R outputs are CC-BY-NC, Gemma outputs follow
  the Gemma Terms of Use); see [`data/LICENSE`](data/LICENSE) and [`docs/provenance.md`](docs/provenance.md).

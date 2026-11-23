# Conventions

Project-wide conventions: naming, paths, configuration, determinism and licensing.

## Instance names

An *instance* is an LLM with its configuration. Three string forms are parsed byte for byte by
`xp_tools.model_filtering` and by the scenario layer; change one only together with all its consumers.

| Form | Example | Rule |
|---|---|---|
| `<model>@@<quantization>` | `Qwen/Qwen2-7B-Instruct@@fp8` | `@@` is `QUANTIZATION_SEPARATOR` (`audit_llm.file_io`); the suffix is the quantization *key* (`fp8`, `bitsandbytes_int4`), not the vLLM engine value |
| `<abliterated repo>` | `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3` | the Hugging Face id of the *abliterated* weights, in place of the model |
| `temp-<t>_sp-<idx>` | `temp-1.0_sp--1` | `sp--1` is "no system prompt"; the trailing `.0` of the temperature is kept |

The classifier's class names, and the instance labels of the scenario layer, are `<model>_temp-<temperature>_sp-<idx>`
(`features.grid.class_name`).
`tests/test_variation_resolver.py` pins the label forms; `scenarios.resolver` and `scenarios.variation` are
their single source.

## Paths and secrets

- **Paths come from the environment.** `.env.example` lists
  the variables the code reads (`FLIPS_DATA`, `FLIPS_OUT`, `FLIPS_DATA_URL`, `FLIPS_NUM_THREADS`, `AUDIT_LLM_MODEL_CACHE`, `HF_HOME`).
- **`config/models.yaml` is the single source of truth for the model universe** (25 base LLMs, quantized
  aliases, abliterated repositories). Local weight folders are set with `AUDIT_LLM_MODEL_CACHE`, never in the file.
- **One data root.** The analysis tier reads `$FLIPS_DATA/results/...` and writes below `$FLIPS_OUT`
  (default `paper/out/`). Pipeline runs write below the directory given to `--out`; the Makefile uses `out/`.
- **Repository root discovery.** The code finds the repository root by walking up to the first directory that
  holds `pyproject.toml`; work from a clone or an unpacked archive of the repository (or `pip install -e .` inside it).

## Run record

`run_config.json` is the record of a generation run: the models used, their resolved paths, the sampling
defaults and an environment snapshot (Python, vLLM, torch, CUDA versions, GPU type). It is written under the run
root and kept current as models complete ([`inference.md`](inference.md)).

## Determinism

- Splits, training draws and the Unseen instances of each draw are seeded; the seeds live in the configs
  (`protocol.seed`, `open_set.cells`) and in `src/audit_llm/Classification/protocol.py`.
- XGBoost's histogram method depends on the thread count: the global classifiers use `classifier.n_jobs: 4`
  in the shipped configs, and bitwise results need XGBoost 3.2 ([`../reproduction/recompute.md`](../reproduction/recompute.md)).
- Generation is not seeded end to end (sampling at temperature above 0); generated answers differ between runs.

## Experiment configs

Experiment YAML files are validated by `experiment_config_schema.py` (pydantic) before anything runs. Unknown
keys are rejected, so a typo fails at load time. Field reference: [`configuration-advanced.md`](configuration-advanced.md).

## Style

Line length 120 (`black`, `isort`, `flake8` in `setup.cfg` and `pyproject.toml`). Lint is advisory in CI.

## Third-party components

`src/audit_llm/Bits_Generation/NIST_Tests_lib/` is a vendored Python implementation of the NIST SP 800-22
statistical test suite (see the README in that directory). `src/audit_llm/features/nist.py` calls its test functions; the files
are upstream code, except the matrix size (8x8, `Matrix.py`) and the linear-complexity block size (20, `Complexity.py`),
set for short sequences.

## Licensing and REUSE

- `REUSE.toml` declares the SPDX metadata for the whole tree with `[[annotations]]` glob blocks, so binary and
  JSON files need no inline header. `reuse lint` must report the repository compliant (CI runs it). Every file
  needs both a copyright text and a licence identifier.
- Buckets: first-party code, configs, CI and docs are **MIT**; datasets, derived results, the test fixture and
  the analysis tier's reference outputs are **CC-BY-4.0**; the lockfile is **CC0-1.0**. The CC-BY-4.0 tag
  does not override the stricter per-model output terms (Cohere CC-BY-NC, Gemma Terms of Use): see
  `data/LICENSE` and [`../provenance.md`](../provenance.md).
- When several globs match a path, the last matching block wins, so the data block comes after the code block.
- Files with an inline SPDX header keep it; the header takes precedence over the glob.
